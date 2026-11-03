#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Overlay the first frame of every episode to see the initial-state distribution of a dataset.

For each camera, all first frames are blended together. A pixel's opacity grows with its difference
from the median of the first frames, so the static scene stays clean and anything that changes
between episodes (objects, the arm) stays visible as a see-through layer. In a plain average, each of
N episodes would only contribute 1/N and small objects would disappear.

Only local v3 datasets with video cameras are supported.

Usage:
    uv run python examples/dataset/visualize_initial_frames.py \
        ~/.cache/huggingface/lerobot/boinkie/santa_ducky_20261103_145920

Writes one `initial_frames_overlay_<camera>.png` per camera to `--output-dir`, which defaults to
`outputs/initial_frames/<dataset name>`.
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd


def load_episodes(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata in {root / 'meta' / 'episodes'}. Is this a v3 dataset?")
    return pd.concat([pd.read_parquet(p) for p in paths]).sort_values("episode_index")


def read_frame(container: av.container.InputContainer, timestamp: float, fps: float) -> np.ndarray:
    """Returns the RGB frame at `timestamp`, in seconds from the start of the video file."""
    stream = container.streams.video[0]
    container.seek(int(timestamp / stream.time_base), stream=stream, backward=True)
    for frame in container.decode(stream):
        # Half a frame of tolerance, since timestamps are stored as rounded floats.
        if frame.time >= timestamp - 0.5 / fps:
            return frame.to_ndarray(format="rgb24")
    raise ValueError(f"No frame at {timestamp:.3f}s in {container.name}")


def load_first_frames(root: Path, info: dict, episodes: pd.DataFrame, camera: str) -> np.ndarray:
    """Returns the first frame of each episode as a (num_episodes, height, width, 3) array."""
    containers: dict[Path, av.container.InputContainer] = {}
    frames = []
    for _, episode in episodes.iterrows():
        path = root / info["video_path"].format(
            video_key=camera,
            chunk_index=episode[f"videos/{camera}/chunk_index"],
            file_index=episode[f"videos/{camera}/file_index"],
        )
        if path not in containers:
            containers[path] = av.open(str(path))
        timestamp = episode[f"videos/{camera}/from_timestamp"]
        frames.append(read_frame(containers[path], timestamp, info["fps"]))
    for container in containers.values():
        container.close()
    return np.stack(frames)


def make_overlay(frames: np.ndarray) -> np.ndarray:
    """Blends `frames` with a per-pixel opacity proportional to the squared distance to their median."""
    background = np.median(frames, axis=0)
    blended = np.zeros_like(background)
    total_opacity = np.zeros((*background.shape[:2], 1))
    for frame in frames:
        # The 1 keeps a pixel that never changes from dividing by zero.
        opacity = np.abs(frame.astype(np.float64) - background).sum(axis=-1, keepdims=True) ** 2 + 1
        blended += opacity * frame
        total_opacity += opacity
    return (blended / total_opacity).round().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay the first frame of every episode of a dataset.")
    parser.add_argument("root", type=Path, help="Path to a local LeRobot dataset.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to outputs/initial_frames/<dataset name>.")
    args = parser.parse_args()

    root = args.root.expanduser()
    info = json.loads((root / "meta" / "info.json").read_text())
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    if not cameras:
        raise ValueError(f"{root} has no video cameras.")
    episodes = load_episodes(root)
    output_dir = args.output_dir or Path("outputs") / "initial_frames" / root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    for camera in cameras:
        frames = load_first_frames(root, info, episodes, camera)
        overlay = make_overlay(frames)
        name = camera.removeprefix("observation.images.")
        path = output_dir / f"initial_frames_overlay_{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"{camera}: overlaid {len(frames)} episodes -> {path}")


if __name__ == "__main__":
    main()
