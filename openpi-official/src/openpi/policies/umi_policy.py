"""Policy transforms for the UMI bimanual real-robot interface.

Each arm uses ``xyz + rotation_6d + gripper`` (10 values). The right arm is
stored first, followed by the left arm. Dataset actions are absolute TCP poses;
the model predicts body-frame pose deltas and absolute gripper targets.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

_ARM_DIM = 10
_BIMANUAL_DIM = 20
_CAMERAS = {
    "base_0_rgb": "cam_head",
    "left_wrist_0_rgb": "cam_left_top",
    "right_wrist_0_rgb": "cam_right_top",
}
_HEAD_MASK_KEY = "fixed_head_mask"
_VISION_TOKEN_GRID = 16
_MIN_KEEP_HEAD_TOKENS = 64


@dataclasses.dataclass(frozen=True)
class UMIRepackTransform(transforms.DataTransformFn):
    """Repack task487 while preserving its optional fixed-head mask video."""

    head_feature: str = "observation.images.head_main"
    use_head_mask: bool = True
    use_head_camera: bool = True
    load_images: bool = True

    def __call__(self, data: dict) -> dict:
        flat = transforms.flatten_dict(data)
        paths = {
            "state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        if self.load_images:
            paths.update(
                {
                    "cam_right_top": "observation.images.right_hand_up",
                    "cam_left_top": "observation.images.left_hand_up",
                }
            )
            if self.use_head_camera:
                paths["cam_head"] = self.head_feature
        result = {name: flat[path] for name, path in paths.items()}
        mask_path = "observation.images.fixed_head_mask"
        if self.load_images and self.use_head_camera and self.use_head_mask and mask_path in flat:
            result[_HEAD_MASK_KEY] = flat[mask_path]
        return result


def fixed_head_mask_to_token_keep_mask(mask_frame: np.ndarray) -> np.ndarray:
    """Convert a white-is-masked frame to a row-major 16x16 SigLIP keep mask."""
    mask = np.asarray(mask_frame)
    if mask.ndim == 3:
        if mask.shape[0] in (1, 3, 4) and mask.shape[-1] not in (1, 3, 4):
            mask = np.moveaxis(mask, 0, -1)
        if mask.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Expected a 2D or HWC mask, got {mask.shape}")
        mask = mask[..., :3].mean(axis=-1)
    elif mask.ndim != 2:
        raise ValueError(f"Expected a 2D or HWC mask, got {mask.shape}")
    if not np.isfinite(mask).all():
        raise ValueError("Head mask contains NaN or infinity")
    if np.issubdtype(mask.dtype, np.floating) and float(mask.max(initial=0.0)) <= 1.0:
        mask = mask * 255.0

    masked = mask > 127.0
    height, width = masked.shape
    if height < _VISION_TOKEN_GRID or width < _VISION_TOKEN_GRID:
        raise ValueError(f"Head mask {width}x{height} is smaller than the 16x16 token grid")
    y_edges = np.linspace(0, height, _VISION_TOKEN_GRID + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, _VISION_TOKEN_GRID + 1, dtype=np.int64)
    ratios = np.empty((_VISION_TOKEN_GRID, _VISION_TOKEN_GRID), dtype=np.float32)
    for row in range(_VISION_TOKEN_GRID):
        for column in range(_VISION_TOKEN_GRID):
            ratios[row, column] = masked[
                y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]
            ].mean()
    keep = ratios < 0.5
    if int(keep.sum()) < _MIN_KEEP_HEAD_TOKENS:
        indices = np.argsort(ratios.reshape(-1), kind="stable")[:_MIN_KEEP_HEAD_TOKENS]
        keep.reshape(-1)[indices] = True
    return keep.reshape(-1).astype(np.bool_)


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    return image


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert the first two rotation-matrix rows to an orthonormal matrix."""
    rot6d = np.asarray(rot6d)
    row0 = rot6d[..., :3]
    row1 = rot6d[..., 3:6]
    norm0 = np.linalg.norm(row0, axis=-1, keepdims=True)
    if np.any(norm0 < 1e-8):
        raise ValueError("Invalid rotation_6d: first row has zero norm")
    row0 = row0 / norm0
    row1 = row1 - np.sum(row1 * row0, axis=-1, keepdims=True) * row0
    norm1 = np.linalg.norm(row1, axis=-1, keepdims=True)
    if np.any(norm1 < 1e-8):
        raise ValueError("Invalid rotation_6d: rows are collinear")
    row1 = row1 / norm1
    row2 = np.cross(row0, row1)
    return np.stack((row0, row1, row2), axis=-2)


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix)[..., :2, :].reshape(*matrix.shape[:-2], 6)


def _relative_pose(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Express absolute 9D poses in the coordinate frame of ``base``."""
    base_rotation = _rot6d_to_matrix(base[..., 3:9])
    target_rotation = _rot6d_to_matrix(target[..., 3:9])
    position = np.einsum("...ji,...j->...i", base_rotation, target[..., :3] - base[..., :3])
    rotation = np.einsum("...ji,...jk->...ik", base_rotation, target_rotation)
    return np.concatenate((position, _matrix_to_rot6d(rotation)), axis=-1)


def _split_state(data: dict) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(data["state"], dtype=np.float32)
    if state.shape == (2, _BIMANUAL_DIM):
        return state[0], state[1]
    if state.shape != (_BIMANUAL_DIM,):
        raise ValueError(f"Expected state shape (20,) or (2, 20), got {state.shape}")
    if "pre_state" not in data:
        raise ValueError("A single-frame state requires pre_state for real-robot inference")
    pre_state = np.asarray(data["pre_state"], dtype=np.float32)
    if pre_state.shape != (_BIMANUAL_DIM,):
        raise ValueError(f"Expected pre_state shape (20,), got {pre_state.shape}")
    return pre_state, state


def _relative_state(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    output = np.empty((_BIMANUAL_DIM,), dtype=np.float32)
    for offset in (0, _ARM_DIM):
        output[offset : offset + 9] = _relative_pose(current[offset : offset + 9], previous[offset : offset + 9])
        output[offset + 9] = current[offset + 9]
    return output


def _body_frame_actions(actions: np.ndarray, current: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != _BIMANUAL_DIM:
        raise ValueError(f"Expected actions shape (horizon, 20), got {actions.shape}")
    output = actions.copy()
    for offset in (0, _ARM_DIM):
        output[:, offset : offset + 9] = _relative_pose(actions[:, offset : offset + 9], current[offset : offset + 9])
    return output


@dataclasses.dataclass(frozen=True)
class UMIBimanualInputs(transforms.DataTransformFn):
    """Convert UMI observations and absolute TCP actions to the pi0.5 contract.

    Training passes ``state`` as a two-frame ``[previous, current]`` window.
    Real-robot inference passes one current ``state`` and one ``pre_state``.
    """

    model_type: _model.ModelType
    use_head_camera: bool = True
    load_images: bool = True

    def __call__(self, data: dict) -> dict:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported model type: {self.model_type}")

        previous, current = _split_state(data)
        inputs = {
            "state": _relative_state(previous, current),
        }
        if self.load_images:
            left_wrist = _parse_image(data[_CAMERAS["left_wrist_0_rgb"]])
            images = {
                "base_0_rgb": (
                    _parse_image(data[_CAMERAS["base_0_rgb"]])
                    if self.use_head_camera
                    else np.zeros_like(left_wrist)
                ),
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": _parse_image(data[_CAMERAS["right_wrist_0_rgb"]]),
            }
            image_masks = dict.fromkeys(images, np.True_)
            image_masks["base_0_rgb"] = np.bool_(self.use_head_camera)
            inputs["image"] = images
            inputs["image_mask"] = image_masks
            if self.use_head_camera and _HEAD_MASK_KEY in data:
                inputs["image_token_mask"] = {
                    name: (
                        fixed_head_mask_to_token_keep_mask(data[_HEAD_MASK_KEY])
                        if name == "base_0_rgb"
                        else np.ones(_VISION_TOKEN_GRID**2, dtype=np.bool_)
                    )
                    for name in images
                }
        if "actions" in data:
            inputs["actions"] = _body_frame_actions(data["actions"], current)
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class UMIBimanualOutputs(transforms.DataTransformFn):
    """Return the 20D body-frame targets consumed by the Marvin controller."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < _BIMANUAL_DIM:
            raise ValueError(f"Expected at least 20 action dimensions, got {actions.shape}")
        return {"actions": actions[..., :_BIMANUAL_DIM]}
