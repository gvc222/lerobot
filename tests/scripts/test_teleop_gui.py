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

import dataclasses
import json
from datetime import datetime

import draccus
import pytest

from lerobot.gui.form import (
    MANUAL_EPISODE_CAP,
    PHASE_LABEL,
    PHASE_READY,
    PHASE_RECORDING,
    CameraValues,
    FormValues,
    build_record_command,
    clean_log_line,
    form_error,
    load_type_choices,
    make_repo_id,
    parse_form,
    parse_resolution,
    phase_from_log,
    suggested_mount,
    write_session_file,
)
from lerobot.scripts.lerobot_record import RecordConfig

pytestmark = [pytest.mark.dataset, pytest.mark.hardware]

NOW = datetime(2026, 11, 2, 15, 30, 0)

RAW_FIELDS = {
    "operator": " Gianna ",
    "dataset_name": "santa_ducky",
    "task": "Pick up santa ducky and place it in the plate",
    "manual_control": False,
    "push_to_hub": True,
    "robot_type": "so101_follower",
    "robot_port": "/dev/ttyACM1",
    "robot_id": "boinkie_follower_arm",
    "teleop_type": "so101_leader",
    "teleop_port": "/dev/ttyACM0",
    "teleop_id": "boinkie_leader_arm",
    "num_episodes": "50",
    "episode_time_s": "30",
    "reset_time_s": "10",
}
RAW_CAMERAS = [("wrist", "2", "640x480", "moving"), ("top", "4", "640x480", "fixed")]
WRIST = CameraValues(2, 640, 480, "moving")


def _values(**overrides) -> FormValues:
    return dataclasses.replace(parse_form(RAW_FIELDS, RAW_CAMERAS), **overrides)


def test_parse_form():
    values = parse_form(
        RAW_FIELDS, [("wrist", "2", "640x480", "moving"), ("side", "/dev/video6", "1280x720", "fixed")]
    )
    assert values.operator == "Gianna"
    assert values.cameras == {"wrist": WRIST, "side": CameraValues("/dev/video6", 1280, 720, "fixed")}
    assert (values.num_episodes, values.episode_time_s) == (50, 30.0)


@pytest.mark.parametrize(
    "fields, cameras, message",
    [
        ({"num_episodes": "lots"}, RAW_CAMERAS, "Number of episodes"),
        ({"reset_time_s": ""}, RAW_CAMERAS, "Reset time"),
        ({}, [("top", "2", "640x480", "fixed"), ("top", "4", "640x480", "fixed")], "unique"),
        ({}, [("top", "2", "big", "fixed")], "640x480"),
    ],
)
def test_parse_form_rejects(fields, cameras, message):
    with pytest.raises(ValueError, match=message):
        parse_form({**RAW_FIELDS, **fields}, cameras)


def test_parse_form_ignores_episode_fields_in_manual_mode():
    fields = {**RAW_FIELDS, "manual_control": True, "num_episodes": "", "episode_time_s": "x"}
    values = parse_form(fields, RAW_CAMERAS)
    assert form_error(values, "user") is None


@pytest.mark.parametrize(
    "overrides, hf_user, expect_error",
    [
        ({}, "user", False),
        ({"operator": ""}, "user", True),
        ({"dataset_name": "bad name/with slash"}, "user", True),
        ({"task": ""}, "user", True),
        ({"robot_port": ""}, "user", True),
        ({"teleop_id": ""}, "user", True),
        ({"cameras": {"2wrist": WRIST}}, "user", True),
        ({"cameras": {"wrist": CameraValues("", 640, 480, "moving")}}, "user", True),
        ({"cameras": {"wrist": CameraValues(2, 640, 480, "spinning")}}, "user", True),
        ({"cameras": {}}, "user", False),
        ({"num_episodes": 0}, "user", True),
        ({"manual_control": True, "num_episodes": 0}, "user", False),  # manual mode is uncapped
        ({"manual_control": False, "episode_time_s": 0.0}, "user", True),
        ({"manual_control": True, "episode_time_s": 0.0}, "user", False),  # ignored in manual mode
        ({}, None, True),  # upload without a login
        ({"push_to_hub": False}, None, False),  # local-only is fine without a login
    ],
)
def test_form_error(overrides, hf_user, expect_error):
    assert (form_error(_values(**overrides), hf_user) is not None) == expect_error


def test_camera_helpers():
    assert parse_resolution("1280x720") == (1280, 720)
    assert parse_resolution(" 640 × 480 ") == (640, 480)
    assert suggested_mount("wrist") == "moving"
    assert suggested_mount("left_wrist") == "moving"
    assert suggested_mount("front") == "fixed"


def test_make_repo_id():
    assert make_repo_id(" santa_ducky ", "boinkie", NOW) == "boinkie/santa_ducky_20261102_153000"
    assert make_repo_id("santa_ducky", None, NOW) == "local/santa_ducky_20261102_153000"


def test_type_choices_include_so101_and_skip_unconfigurable_types():
    robots, teleops = load_type_choices()
    assert "so101_follower" in robots and "koch_follower" in robots
    assert "so101_leader" in teleops
    # Bimanual / keyboard types have no single `port`, so the GUI can't configure them.
    assert "bi_so_follower" not in robots
    assert "keyboard" not in teleops


@pytest.mark.parametrize("manual, upload", [(True, True), (False, False)])
def test_command_parses_into_record_config(manual, upload):
    values = _values(
        manual_control=manual,
        push_to_hub=upload,
        operator="Gianna C",
        robot_type="koch_follower",
        teleop_type="koch_leader",
        cameras={
            "wrist": WRIST,
            "top": CameraValues(4, 1280, 720, "fixed"),
            "side": CameraValues("/dev/video6", 640, 480, "fixed"),
        },
        num_episodes=7,
    )
    repo_id = make_repo_id(values.dataset_name, "boinkie", NOW)
    cmd = build_record_command(values, repo_id)
    assert cmd[1:3] == ["-m", "lerobot.scripts.lerobot_record"]

    cfg = draccus.parse(RecordConfig, args=cmd[3:])
    assert cfg.robot.type == "koch_follower"
    assert cfg.robot.port == "/dev/ttyACM1"
    assert cfg.teleop.type == "koch_leader"
    assert cfg.teleop.id == "boinkie_leader_arm"
    assert cfg.manual_control is manual
    assert cfg.dataset.push_to_hub is upload
    assert cfg.dataset.repo_id == repo_id
    assert cfg.dataset.no_stamp is True
    assert cfg.dataset.single_task == values.task
    # Manual mode is uncapped: lerobot-record only stops on q / Stop.
    assert cfg.dataset.num_episodes == (MANUAL_EPISODE_CAP if manual else 7)
    assert cfg.display_data is False
    assert cfg.dataset.tags == ["operator:Gianna C"]
    assert list(cfg.robot.cameras) == ["wrist", "top", "side"]
    assert cfg.robot.cameras["top"].index_or_path == 4
    assert (cfg.robot.cameras["top"].width, cfg.robot.cameras["top"].height) == (1280, 720)
    # Cameras not in defaults.py borrow type / fps from its first camera.
    assert cfg.robot.cameras["side"].fps == 30


def test_write_session_file(tmp_path):
    values = _values()
    assert write_session_file(tmp_path / "missing", values, "u/x", NOW) is None

    path = write_session_file(tmp_path, values, "u/x", NOW)
    data = json.loads(path.read_text())
    assert data["operator"] == "Gianna"
    assert data["repo_id"] == "u/x"
    assert data["robot"]["type"] == "so101_follower"
    assert data["cameras"]["wrist"] == {"index_or_path": 2, "width": 640, "height": 480, "mount": "moving"}
    assert data["cameras"]["top"]["mount"] == "fixed"
    assert data["started_at"] == "2026-11-02T15:30:00"


def test_command_streams_cameras_to_the_gui():
    cmd = build_record_command(_values(), "u/x", stream_port=5555)
    cfg = draccus.parse(RecordConfig, args=cmd[3:])
    assert cfg.display_data is True
    assert cfg.display_mode == "stream"
    assert (cfg.display_ip, cfg.display_port) == ("127.0.0.1", 5555)


def test_clean_log_line():
    assert clean_log_line("\x1b[32mINFO\x1b[0m hello\n") == "INFO hello\n"
    assert clean_log_line("10%\r50%\r100%\n") == "100%\n"


@pytest.mark.parametrize(
    "line, manual, phase, text",
    [
        ("INFO 2026-11-02 utils.py:150 Recording episode 3", True, PHASE_RECORDING, "Space to stop"),
        ("INFO 2026-11-02 utils.py:150 Recording episode 3", False, PHASE_RECORDING, "Right to end"),
        ("INFO ... Label the episode: space=success  r=recovery", True, PHASE_LABEL, "r = recovery"),
        ("INFO ... Saved failure. 2 success, 0 recovery, 1 failure, 3 total", True, PHASE_READY, "3 total"),
        ("INFO ... Discarded", True, PHASE_READY, "Discarded"),
    ],
)
def test_phase_from_log(line, manual, phase, text):
    result = phase_from_log(line, manual)
    assert result is not None
    assert result[0] == phase
    assert text in result[1]


def test_phase_from_log_ignores_other_lines():
    assert phase_from_log("INFO connecting to /dev/ttyACM1", manual=True) is None
