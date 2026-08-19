"""UMI6 单臂 (右臂) 6D 旋转 policy — 参照 libero_policy.py

UMI6 数据格式 (单臂, 左臂零填充):
- State: 20D [right_pos(3) + right_rot_6d(6) + right_gripper(1) + left_zeros(10)]
- Action: 20D 绝对 EE pose (训练时 DeltaActions 转为 chunk-wise delta)
- 图像: cam_right (右腕), cam_head (头部)

pi0 期望的图像:
- base_0_rgb: 头部相机 <- cam_head
- left_wrist_0_rgb: 零填充 (无左腕相机, mask=False)
- right_wrist_0_rgb: 右腕相机 <- cam_right
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_umi6_example() -> dict:
    """创建随机输入样例，用于测试。"""
    return {
        "cam_right": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "cam_head": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "state": np.random.rand(20).astype(np.float32),
        "prompt": "pick up the cup",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UMI6Inputs(transforms.DataTransformFn):
    """UMI6 -> pi0.5 输入 transform (单臂, 右腕相机 + 头部相机)

    映射:
    - cam_head  -> base_0_rgb (头部相机)
    - cam_right -> right_wrist_0_rgb
    - left_wrist_0_rgb: 零填充 (mask=False)
    - state: 20D (右臂10D + 左臂零10D)
    - actions: 20D (绝对 pose, DeltaActions 转为 chunk-wise delta)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        right_image = _parse_image(data["cam_right"])
        head_image = _parse_image(data["cam_head"])

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": np.zeros_like(right_image),
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UMI6Outputs(transforms.DataTransformFn):
    """pi0.5 输出 -> UMI6 动作格式 (单臂, 只取右臂 10D)

    AbsoluteActions 已将 chunk-wise delta 转回绝对 pose, 这里截取前 10 维
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :10])}


# ============================================================
# UMI6 Right-Only (只用右腕相机, 无头部相机)
# ============================================================


@dataclasses.dataclass(frozen=True)
class UMI6RightOnlyInputs(transforms.DataTransformFn):
    """UMI6 -> pi0.5 输入 transform (单臂, 只有右腕相机)

    映射:
    - cam_right -> right_wrist_0_rgb
    - base_0_rgb: 零填充 (mask=False, 无头部相机)
    - left_wrist_0_rgb: 零填充 (mask=False)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        right_image = _parse_image(data["cam_right"])

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": np.zeros_like(right_image),
                "left_wrist_0_rgb": np.zeros_like(right_image),
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


# ============================================================
# UMI6 Head-Only (只用头部相机, 无腕部相机)
# ============================================================


@dataclasses.dataclass(frozen=True)
class UMI6HeadOnlyInputs(transforms.DataTransformFn):
    """UMI6 -> pi0.5 输入 transform (单臂, 只有头部相机)

    映射:
    - cam_head -> base_0_rgb (头部相机)
    - right_wrist_0_rgb: 零填充 (mask=False)
    - left_wrist_0_rgb: 零填充 (mask=False)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        head_image = _parse_image(data["cam_head"])

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": np.zeros_like(head_image),
                "right_wrist_0_rgb": np.zeros_like(head_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


# ============================================================
# UMI5 (原版 Cheng Chi UMI, 单臂) 支持
# ============================================================


def make_umi5_example() -> dict:
    """创建 UMI5 随机输入样例。"""
    return {
        "gopro_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(20).astype(np.float32),
        "prompt": "pick up the cup",
    }


@dataclasses.dataclass(frozen=True)
class UMI5Inputs(transforms.DataTransformFn):
    """UMI5 -> pi0.5 输入 transform (单臂单相机)

    UMI5 特点:
    - 只有 1 路 GoPro -> base_0_rgb, 腕部相机零填充
    - state/action: 20D (右臂10D + 左臂零)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["gopro_image"])

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UMI5Outputs(transforms.DataTransformFn):
    """pi0.5 输出 -> UMI5 动作格式

    截取前 20 维; 部署时再取 [:, :10] 只用右臂
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :20])}
