"""
CubicLinearPoseTrajectoryInterpolator
=====================================

接口与 `PoseTrajectoryInterpolator` 完全兼容（schedule_waypoint /
drive_to_waypoint / trim / __call__），可在 RTDE 控制器中直接替换 import 使用。

设计目标
--------
策略模型异步推理场景下，每个控制周期通过 `schedule_waypoint` 单点插入新路点。
朴素线性插值在新旧 chunk 衔接处会出现速度跳变（导致机械臂抖动）。
本类的核心是：
    * 位置 (xyz) 用三次插值（CubicSpline 或 PCHIP），并以 **当前实际速度**
      作为新插值器的左端一阶导边界条件 → C1 连续，速度无跳变。
    * 旋转 (rotvec) 仍用 Slerp，旋转向量本身不适合做三次插值。
    * 每次 schedule_waypoint 都会在 trim **之前**采样 curr_time 处的速度，
      避免 trim 之后丢失"当前正在执行的速度"信息（这是它与
      CubicPoseTrajectoryInterpolator 的关键区别，后者的 schedule_waypoint
      在 trim 之后才取速度，无法跨 chunk 保速度连续）。

用法示例
--------
    from diffusion_policy.common.cubic_linear_pose_trajectory_interpolator \\
        import CubicLinearPoseTrajectoryInterpolator as PoseTrajectoryInterpolator

    # 与原 PoseTrajectoryInterpolator 用法完全一致
    interp = PoseTrajectoryInterpolator(times=[t0], poses=[pose0])
    interp = interp.schedule_waypoint(
        pose=target_pose, time=target_time,
        max_pos_speed=0.25, max_rot_speed=0.16,
        curr_time=curr_time, last_waypoint_time=last_waypoint_time)
    pose_command = interp(t_now)

切换插值方式：
    interp = CubicLinearPoseTrajectoryInterpolator(
        times, poses, interp_kind='pchip')   # 不过冲、单调保形

from diffusion_policy.common.cubic_linear_pose_trajectory_interpolator \
    import CubicLinearPoseTrajectoryInterpolator as PoseTrajectoryInterpolator

pose_interp = PoseTrajectoryInterpolator(
    times=[curr_t], poses=[curr_pose], interp_kind='pchip')


场景 1：首次构造（机械臂静止、刚启动）
pose_interp = CubicLinearPoseTrajectoryInterpolator(
    times=[curr_t],
    poses=[curr_pose],
    pos_velocity_boundary=None,   # ← 推荐 None
)

场景 2：每次重建插值器（schedule_waypoint / drive_to_waypoint / trim 内部）
left_vel = old_interp.get_position_velocity(curr_time)
new_interp = CubicLinearPoseTrajectoryInterpolator(
    times=new_times,
    poses=new_poses,
    pos_velocity_boundary=left_vel,   # ← 来自旧插值器
)
"""

from typing import Union, Optional
import numbers
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()


def pose_distance(start_pose, end_pose):
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist


class CubicLinearPoseTrajectoryInterpolator:
    """
    位置三次 + 旋转 Slerp 的位姿轨迹插值器，专为 RTDE 单点插入场景设计。

    与 `PoseTrajectoryInterpolator` 接口完全兼容：
        - times / poses 属性
        - trim(start_t, end_t)
        - drive_to_waypoint(...)
        - schedule_waypoint(...)
        - __call__(t)

    额外保证：每次重建插值器时，左端一阶导（速度）会被显式约束为「当前正在
    执行的速度」，从而消除新旧 chunk 衔接处的速度跳变。
    """

    # ---------- 构造 ----------
    def __init__(self,
                 times: np.ndarray,
                 poses: np.ndarray,
                 pos_velocity_boundary: Optional[np.ndarray] = None,
                 interp_kind: str = 'cubic'):
        """
        Args:
            times: (N,) 时间戳，单调非递减。
            poses: (N, 6) 位姿 [x, y, z, rx, ry, rz]，旋转向量。
            pos_velocity_boundary: (3,) 左端位置速度，用于保证 C1 连续。
                None 表示自然边界（首次构造时用）。
            interp_kind: 'cubic' (CubicSpline，最平滑) 或 'pchip'
                (PCHIP，单调保形不过冲)。
        """
        assert len(times) >= 1
        assert len(poses) == len(times)
        assert interp_kind in ('cubic', 'pchip')
        if not isinstance(times, np.ndarray):
            times = np.array(times, dtype=np.float64)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses, dtype=np.float64)

        self.interp_kind = interp_kind
        self._pos_velocity_boundary = pos_velocity_boundary

        if len(times) == 1:
            # 单点退化：返回常值
            self.single_step = True
            self._times = times
            self._poses = poses
            return

        self.single_step = False
        assert np.all(times[1:] >= times[:-1])

        # 处理重复时间戳（Slerp 要求严格递增）
        times, poses = self._dedup(times, poses)

        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._poses = poses
            return

        pos = poses[:, :3]
        rot = st.Rotation.from_rotvec(poses[:, 3:])

        if interp_kind == 'pchip':
            # PCHIP 不直接支持显式边界条件；速度连续性由当前点位置自身保证，
            # 这里左边界速度仅在 cubic 模式下生效。
            self.pos_interp = si.PchipInterpolator(times, pos, axis=0)
        else:
            bc_type = self._resolve_bc_type(times, pos, pos_velocity_boundary)
            self.pos_interp = si.CubicSpline(
                times, pos, bc_type=bc_type, axis=0)

        self.rot_interp = st.Slerp(times, rot)

    @staticmethod
    def _dedup(times, poses):
        """去掉重复（或几乎重复）的时间戳，保留第一次出现的。"""
        keep = np.concatenate([[True], np.diff(times) > 1e-9])
        return times[keep], poses[keep]

    @staticmethod
    def _resolve_bc_type(times, pos, left_vel):
        """根据点数和左端速度，选 CubicSpline 的边界条件。"""
        n = len(times)
        if left_vel is None:
            return 'not-a-knot' if n >= 3 else 'natural'

        if n >= 3:
            # 左端一阶导 = 当前速度；右端 not-a-knot 自动
            return ((1, np.asarray(left_vel, dtype=np.float64)),
                    'not-a-knot')
        else:
            # n == 2：not-a-knot 不可用，右端用有限差分速度
            dt = max(times[-1] - times[-2], 1e-8)
            right_vel = (pos[-1] - pos[-2]) / dt
            return ((1, np.asarray(left_vel, dtype=np.float64)),
                    (1, right_vel))

    # ---------- 属性 ----------
    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        return self.pos_interp.x

    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        n = len(self.times)
        out = np.zeros((n, 6))
        out[:, :3] = self.pos_interp(self.times)
        out[:, 3:] = self.rot_interp(self.times).as_rotvec()
        return out

    # ---------- 速度查询 ----------
    def get_position_velocity(self, t: float) -> np.ndarray:
        """返回时刻 t 的位置速度向量 (3,) = d(xyz)/dt。"""
        if self.single_step:
            return np.zeros(3)
        t_clip = float(np.clip(t, self.times[0], self.times[-1]))
        if self.interp_kind == 'pchip':
            return self.pos_interp.derivative()(t_clip)
        return self.pos_interp(t_clip, 1)

    # ---------- 轨迹操作 ----------
    def trim(self, start_t: float, end_t: float
             ) -> "CubicLinearPoseTrajectoryInterpolator":
        """
        裁剪到 [start_t, end_t]。
        在裁剪 **之前** 先记录 start_t 处的速度，作为新插值器的左端速度边界，
        保证 C1 连续。
        """
        assert start_t <= end_t
        # 裁剪前的速度
        left_vel = self.get_position_velocity(start_t)

        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        all_times = np.unique(all_times)
        all_poses = self(all_times)

        return CubicLinearPoseTrajectoryInterpolator(
            times=all_times,
            poses=all_poses,
            pos_velocity_boundary=left_vel,
            interp_kind=self.interp_kind)

    def drive_to_waypoint(self,
                          pose, time, curr_time,
                          max_pos_speed=np.inf,
                          max_rot_speed=np.inf
                          ) -> "CubicLinearPoseTrajectoryInterpolator":
        """
        从 curr_time 平滑驱动到 (pose, time)，受速度限制。
        左端速度用 curr_time 处的当前速度，避免速度跳变。
        """
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        time = max(time, curr_time)

        # 关键：在任何 trim 操作之前先取当前速度
        curr_vel = self.get_position_velocity(curr_time)

        curr_pose = self(curr_time)
        pos_dist, rot_dist = pose_distance(curr_pose, pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        return CubicLinearPoseTrajectoryInterpolator(
            times=times,
            poses=poses,
            pos_velocity_boundary=curr_vel,
            interp_kind=self.interp_kind)

    def schedule_waypoint(self,
                          pose, time,
                          max_pos_speed=np.inf,
                          max_rot_speed=np.inf,
                          curr_time=None,
                          last_waypoint_time=None
                          ) -> "CubicLinearPoseTrajectoryInterpolator":
        """
        在未来时刻 `time` 调度一个新路点 `pose`，受速度限制。
        与 PoseTrajectoryInterpolator.schedule_waypoint 行为一致，但额外
        保证从 curr_time 起的速度连续 → 新旧 chunk 衔接处无跳变。
        """
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        # 关键修复：在任何 trim/schedule 之前，先采样 curr_time 处的当前速度
        # 作为新插值器的左端边界条件。这是与 CubicPoseTrajectoryInterpolator
        # 的核心区别：后者在 trim 之后才取速度，会丢失跨 chunk 的速度信息。
        if curr_time is not None:
            ref_t_for_vel = max(curr_time, self.times[0])
            ref_t_for_vel = min(ref_t_for_vel, self.times[-1])
            left_vel = self.get_position_velocity(ref_t_for_vel)
        else:
            left_vel = self.get_position_velocity(self.times[0])

        if curr_time is not None:
            if time <= curr_time:
                # 新路点在过去，忽略
                return self
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)

        assert start_time <= end_time
        assert end_time <= time

        trimmed_interp = self.trim(start_time, end_time)

        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration

        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        return CubicLinearPoseTrajectoryInterpolator(
            times=times,
            poses=poses,
            pos_velocity_boundary=left_vel,
            interp_kind=self.interp_kind)

    # ---------- 求值 ----------
    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        if self.single_step:
            pose = np.zeros((len(t), 6))
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            pose = np.zeros((len(t), 6))
            pose[:, :3] = self.pos_interp(t)
            pose[:, 3:] = self.rot_interp(t).as_rotvec()

        if is_single:
            pose = pose[0]
        return pose
