'''
from diffusion_policy.common.cubic_pose_trajectory_interpolator import CubicPoseTrajectoryInterpolator


# 初始化（第一个 chunk）
interp = CubicPoseTrajectoryInterpolator(times=chunk_times, poses=chunk_poses)


# 控制循环中查询当前目标姿态
target_pose = interp(current_time)


# 当新 chunk 到来时，平滑过渡
interp = interp.update_chunk(
    new_times=new_chunk_times,
    new_poses=new_chunk_poses,
    curr_time=current_time,
    blend_duration=0.1  # 融合窗口，建议 2~5 个控制周期
)

'''


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


def _smoothstep(x):
    """Hermite smoothstep: 3x^2 - 2x^3, maps [0,1] -> [0,1] with zero derivatives at endpoints."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class CubicPoseTrajectoryInterpolator:
    """
    三次样条姿态轨迹插值器，专为异步策略推理场景设计。

    核心问题：策略模型每推理一次输出 N 步 action chunk，但每执行 K 步（K < N）
    就会重新推理，新旧 chunk 衔接处存在轨迹跳变。

    解决方案：
    - 位置(xyz)：使用 CubicSpline，在 chunk 切换时通过速度边界条件保证 C1 连续
    - 旋转(rotvec)：使用 Slerp，从当前旋转状态起始
    - 可选融合窗口：在过渡区间用 smoothstep 混合新旧轨迹

    接口与 PoseTrajectoryInterpolator 兼容，可直接替换使用。
    """

    def __init__(self, times: np.ndarray, poses: np.ndarray,
                 pos_velocity_boundary: Optional[np.ndarray] = None):
        """
        Args:
            times: (N,) 时间戳序列，单调非递减
            poses: (N, 6) 姿态序列，每行 [x, y, z, rx, ry, rz] (旋转向量)
            pos_velocity_boundary: (3,) 左边界处的位置速度向量，用于保证 chunk
                切换时的速度连续性。None 表示使用自然边界条件。
        """
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times, dtype=np.float64)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses, dtype=np.float64)

        self._pos_velocity_boundary = pos_velocity_boundary

        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            pos = poses[:, :3]
            rot = st.Rotation.from_rotvec(poses[:, 3:])

            bc_type = self._resolve_bc_type(times, pos, pos_velocity_boundary)
            self.pos_interp = si.CubicSpline(
                times, pos, bc_type=bc_type, axis=0)

            self.rot_interp = st.Slerp(times, rot)

    @staticmethod
    def _resolve_bc_type(times, pos, left_vel):
        """根据数据点数量和是否指定左边界速度，选择合适的边界条件。"""
        n = len(times)
        if left_vel is None:
            if n >= 3:
                return 'not-a-knot'
            else:
                return 'natural'

        if n >= 3:
            # 左边界：一阶导数（速度），右边界：not-a-knot（自动）
            return ((1, left_vel), 'not-a-knot')
        else:
            # n == 2 时 not-a-knot 不可用，用有限差分估计右边界速度
            dt = max(times[-1] - times[-2], 1e-8)
            right_vel = (pos[-1] - pos[-2]) / dt
            return ((1, left_vel), (1, right_vel))

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
        poses = np.zeros((n, 6))
        poses[:, :3] = self.pos_interp(self.times)
        poses[:, 3:] = self.rot_interp(self.times).as_rotvec()
        return poses

    def get_position_velocity(self, t: float) -> np.ndarray:
        """返回时刻 t 的位置速度向量 (3,)，即 d(xyz)/dt。"""
        if self.single_step:
            return np.zeros(3)
        t = np.clip(t, self.times[0], self.times[-1])
        return self.pos_interp(t, 1)

    def trim(self, start_t: float, end_t: float
             ) -> "CubicPoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        all_times = np.unique(all_times)
        all_poses = self(all_times)
        vel = self.get_position_velocity(all_times[0])
        return CubicPoseTrajectoryInterpolator(
            times=all_times, poses=all_poses, pos_velocity_boundary=vel)

    def update_chunk(self,
                     new_times: np.ndarray,
                     new_poses: np.ndarray,
                     curr_time: float,
                     blend_duration: float = 0.0
                     ) -> "CubicPoseTrajectoryInterpolator":
        """
        平滑地从当前轨迹过渡到新的 action chunk，消除跳变。

        机制：
        1. 速度边界匹配：在 curr_time 获取当前轨迹的位置速度，作为新三次样条
           的左边界条件，保证位置和速度连续（C1 连续）。
        2. 可选融合（blend_duration > 0）：在过渡窗口内，用 smoothstep 权重
           混合新旧轨迹的位置和旋转，进一步减少瞬态过冲。

        Args:
            new_times: (M,) 新 chunk 的时间戳
            new_poses: (M, 6) 新 chunk 的姿态
            curr_time: 当前执行时刻
            blend_duration: 融合窗口时长（秒）。0 = 仅靠速度边界匹配。
                建议值：2~5 个控制周期的时间，例如 0.1s。

        Returns:
            新的 CubicPoseTrajectoryInterpolator 实例，轨迹平滑无跳变。
        """
        if not isinstance(new_times, np.ndarray):
            new_times = np.array(new_times, dtype=np.float64)
        if not isinstance(new_poses, np.ndarray):
            new_poses = np.array(new_poses, dtype=np.float64)

        curr_pose = self(curr_time)
        curr_vel = self.get_position_velocity(curr_time)

        # 只保留未来的路点
        mask = new_times > curr_time
        if not np.any(mask):
            return self

        future_times = new_times[mask]
        future_poses = new_poses[mask].copy()

        # 在融合窗口内混合新旧轨迹
        if blend_duration > 0:
            blend_end = curr_time + blend_duration
            old_end_time = self.times[-1]
            for i in range(len(future_times)):
                t = future_times[i]
                if t >= blend_end or t > old_end_time:
                    break

                alpha = _smoothstep((t - curr_time) / blend_duration)

                old_pose = self(t)

                # 位置线性混合
                future_poses[i, :3] = (
                    (1.0 - alpha) * old_pose[:3] + alpha * future_poses[i, :3])

                # 旋转 Slerp 混合
                old_rot = st.Rotation.from_rotvec(old_pose[3:])
                new_rot = st.Rotation.from_rotvec(future_poses[i, 3:])
                rots = st.Rotation.concatenate([old_rot, new_rot])
                blended = st.Slerp([0.0, 1.0], rots)([alpha])[0]
                future_poses[i, 3:] = blended.as_rotvec()

        # 以当前姿态为起点，拼接未来路点
        all_times = np.concatenate([[curr_time], future_times])
        all_poses = np.concatenate([[curr_pose], future_poses], axis=0)

        return CubicPoseTrajectoryInterpolator(
            times=all_times, poses=all_poses, pos_velocity_boundary=curr_vel)

    def drive_to_waypoint(self,
                          pose, time, curr_time,
                          max_pos_speed=np.inf,
                          max_rot_speed=np.inf
                          ) -> "CubicPoseTrajectoryInterpolator":
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        time = max(time, curr_time)

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
        vel = self.get_position_velocity(curr_time)

        return CubicPoseTrajectoryInterpolator(
            times, poses, pos_velocity_boundary=vel)

    def schedule_waypoint(self,
                          pose, time,
                          max_pos_speed=np.inf,
                          max_rot_speed=np.inf,
                          curr_time=None,
                          last_waypoint_time=None
                          ) -> "CubicPoseTrajectoryInterpolator":
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
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
        vel = trimmed_interp.get_position_velocity(start_time)

        return CubicPoseTrajectoryInterpolator(
            times, poses, pos_velocity_boundary=vel)

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
