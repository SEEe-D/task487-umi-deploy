from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ProjectionFrame:
    image_bgr: np.ndarray
    episode_index: int
    source_episode: str
    relative_frame_index: int
    dataset_frame_index: int
    timestamp_s: float
    left_joint_angles_rad: dict[str, float]
    right_joint_angles_rad: dict[str, float]


class LeRobotProjectionDataset:
    def __init__(self, dataset_root: Path, camera_stream: str = "head_main") -> None:
        self.dataset_root = dataset_root.resolve()
        self.camera_stream = camera_stream
        self.source_index = pq.read_table(
            self.dataset_root / "meta/source_index.parquet"
        ).to_pandas()
        self.episode_table = pq.read_table(
            self.dataset_root / "meta/episodes/chunk-000/file-000.parquet"
        ).to_pandas()
        self.data_table = pq.read_table(
            self.dataset_root / "data/chunk-000/file-000.parquet",
            columns=[
                "observation.joint_position",
                "timestamp",
                "episode_index",
                "frame_index",
            ],
        )
        self.video_path = (
            self.dataset_root
            / f"videos/observation.images.{camera_stream}/chunk-000/file-000.mp4"
        )
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)

    def episode_length(self, episode_index: int) -> int:
        row = self._episode_row(episode_index)
        return int(row["length"])

    def source_episode(self, episode_index: int) -> str:
        rows = self.source_index[self.source_index["episode_index"] == episode_index]
        if len(rows) != 1:
            raise KeyError(f"episode_index {episode_index} not found in source index")
        return str(rows.iloc[0]["source_episode"])

    def evenly_spaced_indices(self, episode_index: int, count: int) -> list[int]:
        length = self.episode_length(episode_index)
        if count <= 1:
            return [length // 2]
        return sorted(set(np.linspace(0, length - 1, count, dtype=int).tolist()))

    def read_frames(
        self, episode_index: int, relative_frame_indices: list[int]
    ) -> list[ProjectionFrame]:
        episode_row = self._episode_row(episode_index)
        dataset_from_index = int(episode_row["dataset_from_index"])
        episode_length = int(episode_row["length"])
        source_episode = self.source_episode(episode_index)
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"failed to open {self.video_path}")
        output = []
        try:
            for relative_frame_index in relative_frame_indices:
                if not 0 <= relative_frame_index < episode_length:
                    raise IndexError(
                        f"frame {relative_frame_index} outside episode length {episode_length}"
                    )
                dataset_frame_index = dataset_from_index + relative_frame_index
                capture.set(cv2.CAP_PROP_POS_FRAMES, dataset_frame_index)
                ok, image_bgr = capture.read()
                if not ok or image_bgr is None:
                    raise RuntimeError(
                        f"failed to read video frame {dataset_frame_index}"
                    )
                joint_values_rad = np.asarray(
                    self.data_table["observation.joint_position"][dataset_frame_index].as_py(),
                    dtype=np.float64,
                )
                if joint_values_rad.shape != (16,):
                    raise ValueError(
                        f"expected 16 joint values, got {joint_values_rad.shape}"
                    )
                right_joint_angles_rad = {
                    f"Joint{index + 1}_R": float(joint_values_rad[index])
                    for index in range(7)
                }
                left_joint_angles_rad = {
                    f"Joint{index + 1}_L": float(joint_values_rad[8 + index])
                    for index in range(7)
                }
                output.append(
                    ProjectionFrame(
                        image_bgr=image_bgr,
                        episode_index=episode_index,
                        source_episode=source_episode,
                        relative_frame_index=relative_frame_index,
                        dataset_frame_index=dataset_frame_index,
                        timestamp_s=float(
                            self.data_table["timestamp"][dataset_frame_index].as_py()
                        ),
                        left_joint_angles_rad=left_joint_angles_rad,
                        right_joint_angles_rad=right_joint_angles_rad,
                    )
                )
        finally:
            capture.release()
        return output

    def _episode_row(self, episode_index: int):
        rows = self.episode_table[self.episode_table["episode_index"] == episode_index]
        if len(rows) != 1:
            raise KeyError(f"episode_index {episode_index} not found")
        return rows.iloc[0]

