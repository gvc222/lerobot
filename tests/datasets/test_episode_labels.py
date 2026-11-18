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
import numpy as np
import pytest

from lerobot.datasets.factory import episodes_with_labels
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import EPISODE_LABELS

pytestmark = pytest.mark.dataset


def _make_dataset(tmp_path, labels: list[int] | None) -> LeRobotDataset:
    features = {"action": {"dtype": "float32", "shape": (2,), "names": None}}
    if labels is not None:
        features["episode_label"] = {"dtype": "int64", "shape": (1,), "names": None}
    dataset = LeRobotDataset.create("user/labels", fps=10, root=tmp_path / "ds", features=features)
    for label in labels or [0, 0]:
        for _ in range(3):
            frame = {"action": np.zeros(2, dtype=np.float32), "task": "t"}
            if labels is not None:
                frame["episode_label"] = np.array([label], dtype=np.int64)
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    return dataset


def test_episodes_with_labels(tmp_path):
    labels = [EPISODE_LABELS[name] for name in ["success", "recovery", "success", "failure", "recovery"]]
    dataset = _make_dataset(tmp_path, labels)

    assert episodes_with_labels(dataset.root, ["recovery", "failure"]) == [1, 3, 4]
    assert episodes_with_labels(dataset.root, ["success"]) == [0, 2]


def test_episodes_with_labels_requires_label_feature(tmp_path):
    dataset = _make_dataset(tmp_path, None)

    with pytest.raises(ValueError, match="episode_label"):
        episodes_with_labels(dataset.root, ["failure"])
