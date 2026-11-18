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

"""Display-free logic of ``lerobot-run-gui``.

Parses and validates the forms, builds the ``lerobot-record`` / ``lerobot-replay`` commands and reads
their log output.  Kept free of Tk so it can be tested headless.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .defaults import DEFAULTS, CameraDefaults, GuiDefaults

SESSION_FILENAME = "gui_session.json"
LOCAL_NAMESPACE = "local"
MAX_CAMERAS = 8
STREAM_HOST = "127.0.0.1"
# lerobot-record's manual mode stops after --dataset.num_episodes saved episodes.  The GUI passes
# this instead, so a manual session is effectively uncapped: it ends with q or the Stop button.
MANUAL_EPISODE_CAP = 1_000_000

_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_CAMERA_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESOLUTION_RE = re.compile(r"^\s*(\d+)\s*[x×]\s*(\d+)\s*$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# Dropdown choices for each camera (AGENT_GUIDE.md §1 asks: how many, resolution, fixed or moving?).
CAMERA_NAMES = ("front", "top", "side", "wrist", "left_wrist", "right_wrist")
CAMERA_RESOLUTIONS = ("320x240", "640x480", "1280x720", "1920x1080")
CAMERA_MOUNTS = ("fixed", "moving")

# Fields the GUI passes on the command line; only types whose config has them are offered.
ROBOT_REQUIRED_FIELDS = frozenset({"port", "id", "cameras"})
TELEOP_REQUIRED_FIELDS = frozenset({"port", "id"})

MANUAL_KEYS_HELP = (
    "Space = start episode  ·  then end it with Space = success, r = recovery, f = failure  ·  "
    "Left = discard  ·  q = quit"
)
TIMED_KEYS_HELP = "Right = end episode early  ·  Left = re-record  ·  Esc / q = stop"
REPLAY_HELP = "The arm moves right after Start  ·  Support it: torque turns off when the replay ends or stops"


@dataclass(frozen=True)
class CameraValues:
    """One camera row of the form."""

    index_or_path: int | str
    width: int
    height: int
    # "fixed" or "moving".  Not a camera setting: only saved in gui_session.json.
    mount: str


@dataclass(frozen=True)
class FormValues:
    """What the operator entered in the form, parsed."""

    operator: str
    dataset_name: str
    task: str
    manual_control: bool
    push_to_hub: bool
    robot_type: str
    robot_port: str
    robot_id: str
    teleop_type: str
    teleop_port: str
    teleop_id: str
    # Camera name -> settings, in form order.
    cameras: dict[str, CameraValues]
    # The three below only apply to timed mode; manual mode is untimed and uncapped.
    num_episodes: int
    episode_time_s: float
    reset_time_s: float


@dataclass(frozen=True)
class ReplayValues:
    """What the operator entered in the replay form, parsed."""

    robot_type: str
    robot_port: str
    robot_id: str
    repo_id: str
    episode: int


# ---- type choices --------------------------------------------------------------------------------


def compatible_types(base: Any, required_fields: frozenset[str]) -> list[str]:
    """Registered choices of a draccus ``ChoiceRegistry`` whose config has all *required_fields*."""
    return sorted(
        name
        for name, cls in base.get_known_choices().items()
        if required_fields <= {f.name for f in dataclasses.fields(cls)}
    )


def load_type_choices() -> tuple[list[str], list[str]]:
    """``(robot types, teleop types)`` that lerobot-record supports and the GUI can configure.

    Imports lerobot-record (which registers every robot / teleop type it supports), so this takes a
    few seconds: the GUI calls it in a background thread.
    """
    import lerobot.scripts.lerobot_record  # noqa: F401  # heavy, deliberately lazy
    from lerobot.robots import RobotConfig
    from lerobot.teleoperators import TeleoperatorConfig

    return (
        compatible_types(RobotConfig, ROBOT_REQUIRED_FIELDS),
        compatible_types(TeleoperatorConfig, TELEOP_REQUIRED_FIELDS),
    )


# ---- parsing / validation ------------------------------------------------------------------------


def parse_camera_index(text: str) -> int | str:
    """``"2"`` -> ``2`` (an OpenCV index); anything else (e.g. ``/dev/video2``) stays a path."""
    text = text.strip()
    return int(text) if text.isdigit() else text


def suggested_mount(camera_name: str) -> str:
    """Wrist cameras move with the arm; everything else is assumed to be fixed."""
    return "moving" if "wrist" in camera_name else "fixed"


def parse_resolution(text: str) -> tuple[int, int]:
    """``"640x480"`` -> ``(640, 480)``.  Raises ValueError."""
    match = _RESOLUTION_RE.match(text)
    if not match:
        raise ValueError(f"Resolution {text!r} must look like 640x480.")
    return int(match.group(1)), int(match.group(2))


def parse_form(fields: dict[str, Any], cameras: list[tuple[str, str, str, str]]) -> FormValues:
    """Build :class:`FormValues` from raw widget values.  Raises ValueError with a user-facing message.

    *cameras* holds one ``(name, index_or_path, resolution, mount)`` tuple per camera row.
    """
    manual = bool(fields["manual_control"])

    def number(key: str, label: str, cast: type[int] | type[float]) -> Any:
        try:
            return cast(str(fields[key]).strip())
        except ValueError:
            if manual:  # greyed out and unused in manual mode
                return cast(0)
            raise ValueError(f"{label} must be a number.") from None

    names = [row[0].strip() for row in cameras]
    if len(set(names)) != len(names):
        raise ValueError("Camera names must be unique.")
    parsed_cameras = {}
    for name, (_, index, resolution, mount) in zip(names, cameras, strict=True):
        width, height = parse_resolution(resolution)
        parsed_cameras[name] = CameraValues(parse_camera_index(index), width, height, mount.strip())

    return FormValues(
        operator=fields["operator"].strip(),
        dataset_name=fields["dataset_name"].strip(),
        task=fields["task"].strip(),
        manual_control=manual,
        push_to_hub=bool(fields["push_to_hub"]),
        robot_type=fields["robot_type"].strip(),
        robot_port=fields["robot_port"].strip(),
        robot_id=fields["robot_id"].strip(),
        teleop_type=fields["teleop_type"].strip(),
        teleop_port=fields["teleop_port"].strip(),
        teleop_id=fields["teleop_id"].strip(),
        cameras=parsed_cameras,
        num_episodes=number("num_episodes", "Number of episodes", int),
        episode_time_s=number("episode_time_s", "Episode time", float),
        reset_time_s=number("reset_time_s", "Reset time", float),
    )


def form_error(values: FormValues, hf_user: str | None) -> str | None:
    """Return why the form can't be submitted yet, or None when it's ready."""
    required = [
        (values.operator, "Enter the operator name."),
        (values.dataset_name, "Enter a dataset name."),
        (values.task, "Enter the task description."),
        (values.robot_type, "Choose a robot type."),
        (values.robot_port, "Enter the robot port."),
        (values.robot_id, "Enter the robot id."),
        (values.teleop_type, "Choose a teleoperator type."),
        (values.teleop_port, "Enter the teleoperator port."),
        (values.teleop_id, "Enter the teleoperator id."),
    ]
    for value, message in required:
        if not value:
            return message
    if not _DATASET_NAME_RE.match(values.dataset_name):
        return "Dataset name may only contain letters, digits, '.', '_' and '-'."
    for name, cam in values.cameras.items():
        if not _CAMERA_NAME_RE.match(name):
            return f"Camera name {name!r} must be letters, digits and '_' (not starting with a digit)."
        if cam.index_or_path == "":
            return f"Enter the index or path of camera {name!r}."
        if cam.width <= 0 or cam.height <= 0:
            return f"Camera {name!r} needs a non-zero resolution."
        if cam.mount not in CAMERA_MOUNTS:
            return f"Camera {name!r} must be fixed or moving."
    if not values.manual_control:
        if values.num_episodes < 1:
            return "Number of episodes must be at least 1."
        if values.episode_time_s <= 0 or values.reset_time_s < 0:
            return "Episode time must be positive and reset time can't be negative."
    if values.push_to_hub and not hf_user:
        return "Auto-upload needs a Hugging Face login (run `hf auth login`)."
    return None


def parse_replay_form(fields: dict[str, Any]) -> ReplayValues:
    """Build :class:`ReplayValues` from raw widget values.  Raises ValueError with a user-facing message."""
    episode = str(fields["replay_episode"]).strip()
    if not episode.isdecimal():
        raise ValueError("Episode must be a whole number, starting at 0.")
    return ReplayValues(
        robot_type=fields["robot_type"].strip(),
        robot_port=fields["robot_port"].strip(),
        robot_id=fields["robot_id"].strip(),
        repo_id=fields["replay_repo_id"].strip(),
        episode=int(episode),
    )


def replay_form_error(values: ReplayValues) -> str | None:
    """Return why the replay form can't be submitted yet, or None when it's ready."""
    required = [
        (values.repo_id, "Enter a dataset repo id."),
        (values.robot_type, "Choose a robot type."),
        (values.robot_port, "Enter the robot port."),
        (values.robot_id, "Enter the robot id."),
    ]
    for value, message in required:
        if not value:
            return message
    if not _REPO_ID_RE.match(values.repo_id):
        return "Dataset repo id must look like <user>/<dataset>."
    return None


# ---- command line --------------------------------------------------------------------------------


def make_repo_id(dataset_name: str, hf_user: str | None, now: datetime) -> str:
    """``<user>/<name>_<timestamp>``, stamped here (not by lerobot-record) so the GUI knows the path."""
    namespace = hf_user or LOCAL_NAMESPACE
    return f"{namespace}/{dataset_name.strip()}_{now.strftime('%Y%m%d_%H%M%S')}"


def _cameras_arg(cameras: dict[str, CameraValues], defaults: GuiDefaults) -> str:
    """Cameras as ``--robot.cameras`` JSON; type / fps come from defaults.py (by name, else the first)."""
    template = next(iter(defaults.cameras.values()), CameraDefaults(index_or_path=0))
    config = {}
    for name, cam in cameras.items():
        base = defaults.cameras.get(name, template)
        config[name] = {
            "type": base.type,
            "index_or_path": cam.index_or_path,
            "width": cam.width,
            "height": cam.height,
            "fps": base.fps,
        }
    return json.dumps(config)


def build_record_command(
    values: FormValues, repo_id: str, defaults: GuiDefaults = DEFAULTS, stream_port: int | None = None
) -> list[str]:
    """The ``lerobot-record`` invocation for this form submission.

    With *stream_port*, camera frames are streamed (``--display_mode=stream``) to a viewer listening on
    ``STREAM_HOST:stream_port``; without it, nothing is displayed.
    """
    num_episodes = MANUAL_EPISODE_CAP if values.manual_control else values.num_episodes
    display = (
        [
            "--display_data=true",
            "--display_mode=stream",
            f"--display_ip={STREAM_HOST}",
            f"--display_port={stream_port}",
        ]
        if stream_port is not None
        else ["--display_data=false"]
    )
    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_record",
        f"--robot.type={values.robot_type}",
        f"--robot.port={values.robot_port}",
        f"--robot.id={values.robot_id}",
        f"--robot.cameras={_cameras_arg(values.cameras, defaults)}",
        f"--teleop.type={values.teleop_type}",
        f"--teleop.port={values.teleop_port}",
        f"--teleop.id={values.teleop_id}",
        f"--dataset.repo_id={repo_id}",
        "--dataset.no_stamp=true",
        f"--dataset.single_task={values.task}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={values.episode_time_s}",
        f"--dataset.reset_time_s={values.reset_time_s}",
        f"--dataset.push_to_hub={str(values.push_to_hub).lower()}",
        f"--dataset.tags={json.dumps([f'operator:{values.operator}'])}",
        # Without this, `save_episode` encodes the videos while nothing drives the robot, which freezes it.
        "--dataset.streaming_encoding=true",
        "--dataset.encoder_threads=2",
        *display,
        f"--manual_control={str(values.manual_control).lower()}",
        # The GUI speaks the state itself (see Announcer), with the episode number and label.
        "--play_sounds=false",
    ]


def build_replay_command(values: ReplayValues) -> list[str]:
    """The ``lerobot-replay`` invocation.  Replay only sends actions, so no cameras are opened."""
    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_replay",
        f"--robot.type={values.robot_type}",
        f"--robot.port={values.robot_port}",
        f"--robot.id={values.robot_id}",
        f"--dataset.repo_id={values.repo_id}",
        f"--dataset.episode={values.episode}",
        "--play_sounds=false",
    ]


def write_session_file(
    dataset_root: Path, values: FormValues, repo_id: str, started_at: datetime
) -> Path | None:
    """Record who collected the dataset next to it.  Skipped if lerobot-record never created it."""
    if not dataset_root.is_dir():
        return None
    path = dataset_root / SESSION_FILENAME
    payload = {
        "operator": values.operator,
        "task": values.task,
        "repo_id": repo_id,
        "robot": {"type": values.robot_type, "id": values.robot_id},
        "teleop": {"type": values.teleop_type, "id": values.teleop_id},
        "cameras": {name: asdict(cam) for name, cam in values.cameras.items()},
        "manual_control": values.manual_control,
        "num_episodes": None if values.manual_control else values.num_episodes,
        "push_to_hub": values.push_to_hub,
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def resolve_hf_user() -> str | None:
    """The logged-in Hugging Face username (``$HF_USER`` wins), or None when not logged in."""
    if env_user := os.environ.get("HF_USER", "").strip():
        return env_user
    try:
        from huggingface_hub import whoami

        return str(whoami()["name"])
    except Exception:  # noqa: BLE001 - offline / not logged in: the GUI just disables upload
        return None


# ---- log output ----------------------------------------------------------------------------------


def clean_log_line(line: str) -> str:
    r"""Drop ANSI colors and keep only the last redraw of a ``\\r`` progress line."""
    line = _ANSI_RE.sub("", line)
    if "\r" in line.rstrip("\n"):
        ending = "\n" if line.endswith("\n") else ""
        parts = [p for p in line.rstrip("\n").split("\r") if p.strip()]
        line = (parts[-1] if parts else "") + ending
    return line


# Phases shown in the live view, driven by what the subprocess logs (log_say -> logging.info).
PHASE_STARTING = "starting"
PHASE_READY = "ready"
PHASE_RECORDING = "recording"
PHASE_RESET = "reset"
PHASE_REPLAYING = "replaying"
PHASE_STOPPING = "stopping"

_PHASE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Recording episode (\d+)"), PHASE_RECORDING),
    (re.compile(r"Saved (success|recovery|failure)\. .*?(\d+) total"), PHASE_READY),
    (re.compile(r"\bDiscarded\b"), PHASE_READY),
    (re.compile(r"Reset the environment"), PHASE_RESET),
    (re.compile(r"Re-record episode"), PHASE_RESET),
    (re.compile(r"Stop recording"), PHASE_STOPPING),
    (re.compile(r"Replaying episode"), PHASE_REPLAYING),
]


def phase_from_log(line: str, manual: bool) -> tuple[str, str] | None:
    """``(phase, message)`` if *line* marks a recording or replay phase change, else None."""
    for pattern, phase in _PHASE_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        if phase == PHASE_RECORDING:
            stop = (
                "End with Space = success, r = recovery, f = failure  ·  Left = discard"
                if manual
                else "Right to end early, Left to re-record"
            )
            return phase, f"Recording episode {match.group(1)}  ·  {stop}"
        if phase == PHASE_READY and match.groups():
            return phase, f"Saved {match.group(1)} ({match.group(2)} total)  ·  Space to start the next one"
        if phase == PHASE_READY:
            return phase, "Discarded  ·  Space to start the next episode"
        if phase == PHASE_RESET:
            return phase, "Reset the scene"
        if phase == PHASE_REPLAYING:
            return phase, "Replaying the episode on the robot"
        return phase, "Stopping...  finalizing the dataset"
    return None


# ---- voice announcements -------------------------------------------------------------------------

_SAY_RECORDING = re.compile(r"Recording episode (\d+)")
_SAY_LABELED = re.compile(r"Saved (success|recovery|failure)\.")
_SAY_SAVED = re.compile(r"Saved episode (\d+)")
_SAY_DISCARDED = re.compile(r"\bDiscarded\b")
_SAY_RESET = re.compile(r"Reset the environment")
_SAY_RERECORD = re.compile(r"Re-record episode")
_SAY_STOP = re.compile(r"Stop recording")
_SAY_REPLAY = re.compile(r"Replaying episode")


class Announcer:
    """Turns ``lerobot-record`` / ``lerobot-replay`` log lines into short sentences to speak.

    Remembers the episode being recorded, so the end of an episode can say which one it was.
    """

    def __init__(self) -> None:
        """Start with no episode seen yet."""
        self.episode: int | None = None

    def _ep(self) -> str:
        return f"Episode {self.episode}" if self.episode is not None else "Episode"

    def __call__(self, line: str) -> str | None:
        """The sentence to speak for *line*, or None."""
        if match := _SAY_RECORDING.search(line):
            self.episode = int(match.group(1))
            return f"Recording episode {self.episode}"
        if match := _SAY_LABELED.search(line):  # manual mode
            return f"Recording stopped. {self._ep()} saved as {match.group(1)}"
        if _SAY_DISCARDED.search(line):  # manual mode
            return f"Recording stopped. {self._ep()} discarded"
        if _SAY_RESET.search(line):  # timed mode: the save comes after the reset phase
            return "Recording stopped. Reset the scene"
        if _SAY_RERECORD.search(line):
            return f"{self._ep()} discarded. Recording it again"
        if match := _SAY_SAVED.search(line):  # timed mode
            return f"Episode {match.group(1)} saved"
        if _SAY_STOP.search(line):
            return "Recording stopped. Saving the dataset"
        if _SAY_REPLAY.search(line):
            return "Replaying"
        return None
