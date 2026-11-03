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

"""Starting values for the ``run-gui`` form.

The robot / teleop / camera / episode fields of the form are pre-filled from here, so edit these to
match your rig and you won't have to retype them.  Camera type and fps are not in the form: each
camera uses the entry with the same name below, or the first entry for extra cameras.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraDefaults:
    """One camera, passed to ``--robot.cameras``."""

    index_or_path: int | str
    width: int = 640
    height: int = 480
    fps: int = 30
    type: str = "opencv"
    # "fixed" or "moving" (e.g. a wrist camera).  Only saved in gui_session.json.
    mount: str = "fixed"


@dataclass(frozen=True)
class GuiDefaults:
    """Pre-filled form values, plus ``display_data`` which the form doesn't show."""

    robot_type: str = "so101_follower"
    robot_port: str = "/dev/ttyACM1"
    robot_id: str = "boinkie_follower_arm"
    teleop_type: str = "so101_leader"
    teleop_port: str = "/dev/ttyACM0"
    teleop_id: str = "boinkie_leader_arm"
    cameras: dict[str, CameraDefaults] = field(
        default_factory=lambda: {
            "wrist": CameraDefaults(index_or_path=2, mount="moving"),
            "top": CameraDefaults(index_or_path=4),
        }
    )
    # Timed mode only: manual mode is untimed and has no episode limit.
    num_episodes: int = 50
    episode_time_s: float = 30
    reset_time_s: float = 10
    # Show the camera feeds in the GUI while recording (lerobot-record --display_mode=stream).
    display_data: bool = True


DEFAULTS = GuiDefaults()
