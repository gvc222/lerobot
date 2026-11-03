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

"""A Tk window that launches ``lerobot-record`` for teleoperated data collection.

Run it with ``run-gui``.  Fill in the session (operator, dataset name, task), pick the robot and
teleoperator types, check the ports / ids / cameras / episode settings (pre-filled from
:mod:`lerobot.gui.defaults`), choose manual mode and/or auto-upload, and press Start.

The recording runs as a ``lerobot-record`` subprocess.  While it runs, the form is replaced by the live
camera feeds (streamed with ``--display_mode=stream``) and a badge for the current phase; its output
goes to the terminal pane at the bottom, which can be dragged taller or maximized.  Stop sends it
Ctrl+C, so it finalizes the dataset (and uploads it when auto-upload is on) exactly as a terminal
Ctrl+C would.  The operator name is stored as an ``operator:<name>`` Hub tag and in a
``gui_session.json`` file at the dataset root.
"""

from __future__ import annotations

import io
import os
import queue
import shlex
import signal
import socket
import subprocess  # nosec B404
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.stream_visualization import read_frame_messages

from . import theme
from .defaults import DEFAULTS, CameraDefaults, GuiDefaults
from .form import (
    CAMERA_MOUNTS,
    CAMERA_NAMES,
    CAMERA_RESOLUTIONS,
    LOCAL_NAMESPACE,
    MANUAL_KEYS_HELP,
    MAX_CAMERAS,
    PHASE_READY,
    PHASE_STARTING,
    PHASE_STOPPING,
    STREAM_HOST,
    TIMED_KEYS_HELP,
    FormValues,
    build_record_command,
    clean_log_line,
    form_error,
    load_type_choices,
    make_repo_id,
    parse_form,
    phase_from_log,
    resolve_hf_user,
    suggested_mount,
    write_session_file,
)

if TYPE_CHECKING:
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image

RENDER_INTERVAL_MS = 66  # ~15 fps, the rate lerobot-record streams at


def _echo(text: str) -> None:
    """Write *text* to the terminal ``run-gui`` was launched from (if any)."""
    if sys.stdout is None:  # launched without a console
        return
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except (OSError, ValueError):  # closed terminal / broken pipe: the GUI terminal still has it
        pass


class _CameraTile:
    """One live camera view: a caption above a dark box the frame is fitted into."""

    def __init__(self, gui: TeleopGui, parent: tk.Misc, name: str, caption: str) -> None:
        tk_ = gui.tk
        self.name = name
        self.frame = tk_.Frame(parent, background=theme.WHITE)
        tk_.Label(
            self.frame,
            text=caption,
            font=gui.fonts.bold,
            background=theme.WHITE,
            foreground=theme.TEXT,
            anchor="w",
        ).pack(fill="x", pady=(0, 4))
        self.body = tk_.Frame(self.frame, background=theme.DARK)
        self.body.pack(fill="both", expand=True)
        self.image_label = tk_.Label(
            self.body,
            text="Waiting for camera...",
            font=gui.fonts.small,
            background=theme.DARK,
            foreground=theme.DARK_TEXT,
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")
        self.shown: Image.Image | None = None
        self.photo: Any = None

    def show(self, image: Image.Image) -> None:
        from PIL import Image, ImageTk

        width, height = self.body.winfo_width(), self.body.winfo_height()
        if width < 10 or height < 10:
            return
        scale = min(width / image.width, height / image.height)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        self.photo = ImageTk.PhotoImage(image.resize(size, Image.Resampling.BILINEAR))
        self.image_label.configure(image=self.photo, text="")
        self.shown = image


class TeleopGui:
    """The recorder window; runs one ``lerobot-record`` subprocess at a time."""

    def __init__(self, defaults: GuiDefaults = DEFAULTS) -> None:
        """Build the widgets and start the Hugging Face / type-list lookups in the background."""
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.defaults = defaults
        self.hf_user: str | None = None
        self.proc: subprocess.Popen[str] | None = None
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.run_values: FormValues | None = None
        self.run_repo_id = ""
        self.run_started_at = datetime.now()
        self.terminal_maximized = False
        # Camera stream (filled by the receiver thread, drawn by the Tk loop).
        self.frames: dict[str, Image.Image] = {}
        self.stream_server: socket.socket | None = None
        self.stream_stop = threading.Event()
        self.tiles: list[_CameraTile] = []
        self.phase = PHASE_STARTING

        self.root = tk.Tk()
        self.root.title("LeRobot Recorder")
        self.root.geometry("1040x960")
        self.root.minsize(860, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.fonts = theme.apply_theme(self.root)

        d = defaults
        self.vars: dict[str, tk.Variable] = {
            "operator": tk.StringVar(),
            "dataset_name": tk.StringVar(),
            "task": tk.StringVar(),
            "manual_control": tk.BooleanVar(value=False),
            "push_to_hub": tk.BooleanVar(value=True),
            "robot_type": tk.StringVar(value=d.robot_type),
            "robot_port": tk.StringVar(value=d.robot_port),
            "robot_id": tk.StringVar(value=d.robot_id),
            "teleop_type": tk.StringVar(value=d.teleop_type),
            "teleop_port": tk.StringVar(value=d.teleop_port),
            "teleop_id": tk.StringVar(value=d.teleop_id),
            "num_episodes": tk.StringVar(value=str(d.num_episodes)),
            "episode_time_s": tk.StringVar(value=f"{d.episode_time_s:g}"),
            "reset_time_s": tk.StringVar(value=f"{d.reset_time_s:g}"),
        }
        for var in self.vars.values():
            var.trace_add("write", lambda *_: self._refresh())
        self.status = tk.StringVar()
        self.user_text = tk.StringVar(value="Checking Hugging Face login...")
        self.inputs: list[ttk.Widget] = []

        self._build_header()
        self._build_footer()
        self.paned = ttk.Panedwindow(self.root, orient="vertical")
        self.paned.pack(fill="both", expand=True)
        self.top = ttk.Frame(self.paned, padding=(16, 8, 16, 6))
        self.paned.add(self.top, weight=4)
        self._build_terminal()
        self.setup_view = ttk.Frame(self.top)
        self.live_view = ttk.Frame(self.top)
        self._build_setup_view(self.setup_view)
        self._build_live_view(self.live_view)
        self.setup_view.pack(fill="both", expand=True)

        self._sync_camera_rows()
        self._refresh()
        threading.Thread(target=self._lookup_hf_user, daemon=True).start()
        threading.Thread(target=self._lookup_type_choices, daemon=True).start()

    # ---- layout ----------------------------------------------------------------------------

    def _build_header(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        header = ttk.Frame(self.root, padding=(20, 14, 20, 12))
        header.pack(fill="x")
        ttk.Label(header, text="LeRobot Recorder", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Teleoperated data collection", style="Muted.TLabel").pack(
            side="left", padx=(12, 0), pady=(6, 0)
        )
        ttk.Label(header, textvariable=self.user_text, style="Muted.TLabel").pack(side="right", pady=(6, 0))
        tk.Frame(self.root, height=3, background=theme.HF_YELLOW).pack(fill="x")

    def _build_footer(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        footer = ttk.Frame(self.root, padding=(20, 10))
        footer.pack(side="bottom", fill="x")
        tk.Frame(self.root, height=1, background=theme.BORDER).pack(side="bottom", fill="x")
        # takefocus=0: lerobot-record listens for Space globally, which would also "press" a focused button.
        self.button = ttk.Button(
            footer, text="Start recording", style="Primary.TButton", command=self._on_button, takefocus=0
        )
        self.button.pack(side="right")
        text = ttk.Frame(footer)
        text.pack(side="left", fill="x", expand=True)
        self.status_label = ttk.Label(text, textvariable=self.status, style="Status.TLabel")
        self.status_label.pack(anchor="w")
        self.hint = ttk.Label(text, style="Muted.TLabel")
        self.hint.pack(anchor="w")

    def _build_terminal(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        terminal = tk.Frame(self.paned, background=theme.DARK)
        self.paned.add(terminal, weight=1)
        bar = tk.Frame(terminal, background=theme.DARK_RAISED)
        bar.pack(fill="x")
        title = tk.Label(
            bar,
            text="Terminal",
            font=self.fonts.bold,
            background=theme.DARK_RAISED,
            foreground=theme.DARK_TEXT,
            padx=12,
            pady=5,
        )
        title.pack(side="left")

        def bar_button(text: str, command: Any) -> tk.Button:
            return tk.Button(
                bar,
                text=text,
                command=command,
                font=self.fonts.small,
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
                takefocus=0,
                padx=10,
                background=theme.DARK_RAISED,
                foreground=theme.DARK_TEXT,
                activebackground=theme.HF_YELLOW,
                activeforeground=theme.TEXT,
            )

        self.maximize_button = bar_button("Maximize", self._toggle_terminal)
        self.maximize_button.pack(side="right", padx=(0, 6))
        bar_button("Clear", self._clear_log).pack(side="right")
        for widget in (bar, title):
            widget.bind("<Double-Button-1>", lambda _e: self._toggle_terminal())

        body = tk.Frame(terminal, background=theme.DARK)
        body.pack(fill="both", expand=True)
        self.log = tk.Text(
            body,
            height=6,
            wrap="word",
            state="disabled",
            takefocus=0,
            font=self.fonts.mono,
            background=theme.DARK,
            foreground=theme.DARK_TEXT,
            insertbackground=theme.DARK_TEXT,
            selectbackground=theme.HF_ORANGE,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=12,
            pady=8,
        )
        self.log.tag_configure("cmd", foreground=theme.HF_YELLOW)
        self.log.tag_configure("error", foreground="#F87171")
        scroll = ttk.Scrollbar(body, command=self.log.yview, style="Vertical.TScrollbar")
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

    def _build_setup_view(self, parent: ttk.Frame) -> None:
        import tkinter as tk
        from tkinter import ttk

        d = self.defaults
        parent.columnconfigure(0, weight=1, uniform="col")
        parent.columnconfigure(1, weight=1, uniform="col")

        session = self._section(parent, "Session", row=0, column=0, columnspan=2)
        self._entry(session, 0, "Operator name", "operator")
        self._entry(session, 1, "Dataset name", "dataset_name")
        self._entry(session, 2, "Task", "task")

        robot = self._section(parent, "Robot  ·  follower", row=1, column=0)
        self.robot_type_box = self._dropdown(robot, 0, "Type", "robot_type")
        self._entry(robot, 1, "Port", "robot_port")
        self._entry(robot, 2, "ID", "robot_id")

        teleop = self._section(parent, "Teleoperator  ·  leader", row=1, column=1)
        self.teleop_type_box = self._dropdown(teleop, 0, "Type", "teleop_type")
        self._entry(teleop, 1, "Port", "teleop_port")
        self._entry(teleop, 2, "ID", "teleop_id")

        cameras = self._section(parent, "Cameras", row=2, column=0, columnspan=2)
        ttk.Label(cameras, text="Number of cameras").grid(row=0, column=0, sticky="w", pady=4, padx=(0, 12))
        self.camera_count = tk.StringVar(value=str(len(d.cameras)))
        count_box = ttk.Spinbox(
            cameras, from_=0, to=MAX_CAMERAS, width=4, state="readonly", textvariable=self.camera_count
        )
        count_box.grid(row=0, column=1, sticky="w", pady=4)
        self.inputs.append(count_box)
        self.camera_rows_frame = ttk.Frame(cameras)
        self.camera_rows_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        for column, header in enumerate(["Name", "Index / path", "Resolution", "Mount"]):
            self.camera_rows_frame.columnconfigure(column, weight=1, uniform="cam")
            ttk.Label(self.camera_rows_frame, text=header, style="Header.TLabel").grid(
                row=0, column=column, sticky="w", padx=(0, 8)
            )
        # Per camera: (name, index, resolution, mount) variables.
        self.camera_vars: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar, tk.StringVar]] = []
        self.camera_widgets: list[list[ttk.Widget]] = []
        self.camera_count.trace_add("write", lambda *_: self._sync_camera_rows())

        episodes = self._section(parent, "Episodes", row=3, column=0)
        self.episode_entries = [
            self._entry(episodes, 0, "Number of episodes", "num_episodes"),
            self._entry(episodes, 1, "Episode time (s)", "episode_time_s"),
            self._entry(episodes, 2, "Reset time (s)", "reset_time_s"),
        ]
        self.episodes_note = ttk.Label(episodes, style="Muted.TLabel")
        self.episodes_note.grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))

        mode = self._section(parent, "Mode", row=3, column=1)
        for row, (text, key) in enumerate(
            [("Manual mode", "manual_control"), ("Auto-upload to Hugging Face", "push_to_hub")]
        ):
            check = ttk.Checkbutton(mode, text=text, variable=self.vars[key], takefocus=0)
            check.grid(row=row, column=0, sticky="w", pady=6)
            self.inputs.append(check)

    def _build_live_view(self, parent: ttk.Frame) -> None:
        import tkinter as tk
        from tkinter import ttk

        top = ttk.Frame(parent)
        top.pack(fill="x", pady=(0, 10))
        self.phase_badge = tk.Label(top, font=self.fonts.bold, padx=12, pady=4)
        self.phase_badge.pack(side="left")
        self.phase_text = ttk.Label(top, font=self.fonts.bold)
        self.phase_text.pack(side="left", padx=12)
        self.live_repo = ttk.Label(top, style="Muted.TLabel")
        self.live_repo.pack(side="right")
        self.tiles_frame = ttk.Frame(parent)
        self.tiles_frame.pack(fill="both", expand=True)

    # ---- widget helpers --------------------------------------------------------------------

    def _section(
        self, parent: ttk.Frame, title: str, row: int, column: int, columnspan: int = 1
    ) -> ttk.Labelframe:
        frame = self.ttk.LabelFrame(
            parent, text=f" {title} ", style="Card.TLabelframe", padding=(14, 4, 14, 6)
        )
        frame.grid(row=row, column=column, columnspan=columnspan, sticky="nsew", padx=5, pady=3)
        frame.columnconfigure(1, weight=1)
        return frame

    def _entry(self, parent: ttk.Frame | ttk.Labelframe, row: int, label: str, key: str) -> ttk.Entry:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 12))
        entry = self.ttk.Entry(parent, textvariable=self.vars[key])
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        self.inputs.append(entry)
        return entry

    def _dropdown(self, parent: ttk.Frame | ttk.Labelframe, row: int, label: str, key: str) -> ttk.Combobox:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 12))
        # Until the registry finishes loading, only the default from defaults.py is offered.
        box = self.ttk.Combobox(
            parent, textvariable=self.vars[key], values=[self.vars[key].get()], state="readonly"
        )
        box.grid(row=row, column=1, sticky="ew", pady=3)
        self.inputs.append(box)
        return box

    def _sync_camera_rows(self) -> None:
        """Show one row per camera, keeping what was entered in rows that get hidden."""
        count = self._camera_count()
        while len(self.camera_vars) < count:
            self._add_camera_row()
        for i, row_widgets in enumerate(self.camera_widgets):
            for column, widget in enumerate(row_widgets):
                if i < count:
                    widget.grid(row=i + 1, column=column, sticky="ew", pady=3, padx=(0, 8))
                else:
                    widget.grid_remove()
        self._refresh()

    def _add_camera_row(self) -> None:
        i = len(self.camera_vars)
        defaults = list(self.defaults.cameras.items())
        template = defaults[0][1] if defaults else CameraDefaults(index_or_path=0)
        if i < len(defaults):
            name, cam = defaults[i]
            index, mount = str(cam.index_or_path), cam.mount
        else:
            used = {name_var.get() for name_var, *_ in self.camera_vars}
            name = next((n for n in CAMERA_NAMES if n not in used), f"camera{i + 1}")
            cam, index, mount = template, "", suggested_mount(name)
        name_var = self.tk.StringVar(value=name)
        index_var = self.tk.StringVar(value=index)
        resolution_var = self.tk.StringVar(value=f"{cam.width}x{cam.height}")
        mount_var = self.tk.StringVar(value=mount)
        for var in (index_var, resolution_var, mount_var):
            var.trace_add("write", lambda *_: self._refresh())

        def on_name_change(*_: object) -> None:
            mount_var.set(suggested_mount(name_var.get()))  # also triggers _refresh

        name_var.trace_add("write", on_name_change)

        frame = self.camera_rows_frame
        names = list(CAMERA_NAMES) if name in CAMERA_NAMES else [name, *CAMERA_NAMES]
        resolutions = list(CAMERA_RESOLUTIONS)
        if resolution_var.get() not in resolutions:
            resolutions.insert(0, resolution_var.get())
        row_widgets: list[ttk.Widget] = [
            self.ttk.Combobox(frame, textvariable=name_var, values=names, state="readonly", width=12),
            self.ttk.Entry(frame, textvariable=index_var, width=12),
            self.ttk.Combobox(
                frame, textvariable=resolution_var, values=resolutions, state="readonly", width=11
            ),
            self.ttk.Combobox(
                frame, textvariable=mount_var, values=list(CAMERA_MOUNTS), state="readonly", width=8
            ),
        ]
        self.camera_vars.append((name_var, index_var, resolution_var, mount_var))
        self.camera_widgets.append(row_widgets)

    # ---- form state ------------------------------------------------------------------------

    def _camera_count(self) -> int:
        try:
            return max(0, min(MAX_CAMERAS, int(self.camera_count.get())))
        except ValueError:
            return 0

    def _read_form(self) -> tuple[FormValues | None, str | None]:
        cameras = [
            (name.get(), index.get(), resolution.get(), mount.get())
            for name, index, resolution, mount in self.camera_vars[: self._camera_count()]
        ]
        try:
            values = parse_form({key: var.get() for key, var in self.vars.items()}, cameras)
        except ValueError as e:
            return None, str(e)
        return values, form_error(values, self.hf_user)

    def _lookup_hf_user(self) -> None:
        user = resolve_hf_user()
        self.root.after(0, self._set_hf_user, user)

    def _set_hf_user(self, user: str | None) -> None:
        self.hf_user = user
        if user is None:
            self.vars["push_to_hub"].set(False)
            self.user_text.set("Not signed in to Hugging Face  ·  saving locally")
        else:
            self.user_text.set(f"Signed in as {user}")
        self._refresh()

    def _lookup_type_choices(self) -> None:
        try:
            robots, teleops = load_type_choices()
        except Exception as e:  # noqa: BLE001 - keep the defaults usable if an optional import breaks
            self.root.after(0, self._append_log, f"Could not list robot/teleop types: {e}\n", "error")
            return
        self.root.after(0, self._set_type_choices, robots, teleops)

    def _set_type_choices(self, robots: list[str], teleops: list[str]) -> None:
        for box, choices, key in [
            (self.robot_type_box, robots, "robot_type"),
            (self.teleop_type_box, teleops, "teleop_type"),
        ]:
            current = self.vars[key].get()
            box.configure(values=choices if current in choices else [current, *choices])

    def _refresh(self) -> None:
        if self.proc is not None or not hasattr(self, "episodes_note"):
            return
        manual = bool(self.vars["manual_control"].get())
        self.hint.configure(text=MANUAL_KEYS_HELP if manual else TIMED_KEYS_HELP)
        # Manual episodes are untimed and uncapped: the episode settings don't apply.
        for entry in self.episode_entries:
            entry.state(["disabled"] if manual else ["!disabled"])
        self.episodes_note.configure(
            text="Manual mode: untimed, no episode limit. Stop with q or Stop." if manual else ""
        )

        values, error = self._read_form()
        self.button.configure(state="disabled" if error else "normal")
        if values is not None and not error:
            namespace = self.hf_user or LOCAL_NAMESPACE
            self._set_status(f"Ready to record: {namespace}/{values.dataset_name}_<timestamp>")
        elif error and any(self.vars[k].get() for k in ("operator", "dataset_name", "task")):
            self._set_status(error, error=True)
        else:
            self._set_status("Fill in the operator, dataset name and task to start.")

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.set(text)
        self.status_label.configure(style="Error.TLabel" if error else "Status.TLabel")

    def _set_inputs_enabled(self, enabled: bool) -> None:
        widgets = self.inputs + [w for row in self.camera_widgets for w in row]
        for widget in widgets:
            widget.state(["!disabled"] if enabled else ["disabled"])

    # ---- views -----------------------------------------------------------------------------

    def _show_live_view(self, values: FormValues) -> None:
        for tile in self.tiles:
            tile.frame.destroy()
        self.tiles = []
        self.frames = {}
        names = list(values.cameras)
        columns = 1 if len(names) <= 1 else 2 if len(names) <= 4 else 3
        rows = max(1, -(-len(names) // columns))
        for c in range(3):
            used = c < columns
            self.tiles_frame.columnconfigure(c, weight=int(used), uniform="tile" if used else "")
        for r in range(3):
            used = r < rows
            self.tiles_frame.rowconfigure(r, weight=int(used), uniform="tile" if used else "")
        for i, name in enumerate(names):
            cam = values.cameras[name]
            caption = f"{name}   ·   {cam.mount}   ·   {cam.width}×{cam.height}"
            tile = _CameraTile(self, self.tiles_frame, name, caption)
            tile.frame.grid(row=i // columns, column=i % columns, sticky="nsew", padx=6, pady=6)
            self.tiles.append(tile)
        if not names:
            self.ttk.Label(self.tiles_frame, text="No cameras configured.", style="Muted.TLabel").grid()
        self.live_repo.configure(text=self.run_repo_id)
        self._set_phase(PHASE_STARTING, "Connecting to the robot and cameras...")
        self.setup_view.pack_forget()
        self.live_view.pack(fill="both", expand=True)

    def _show_setup_view(self) -> None:
        self.live_view.pack_forget()
        for child in self.tiles_frame.winfo_children():
            child.destroy()
        self.tiles = []
        self.setup_view.pack(fill="both", expand=True)

    def _set_phase(self, phase: str, message: str) -> None:
        self.phase = phase
        background, foreground = theme.PHASE_COLORS.get(phase, (theme.HF_GRAY, theme.WHITE))
        self.phase_badge.configure(text=phase.upper(), background=background, foreground=foreground)
        self.phase_text.configure(text=message)

    def _toggle_terminal(self) -> None:
        if self.terminal_maximized:
            self.paned.insert(0, self.top, weight=4)
            self.maximize_button.configure(text="Maximize")
        else:
            self.paned.forget(self.top)
            self.maximize_button.configure(text="Restore")
        self.terminal_maximized = not self.terminal_maximized

    # ---- camera stream ---------------------------------------------------------------------

    def _open_stream_server(self) -> int:
        """Listen for lerobot-record's camera stream; returns the port to pass it."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((STREAM_HOST, 0))
        server.listen(1)
        server.settimeout(0.5)
        self.stream_server = server
        self.stream_stop.clear()
        threading.Thread(target=self._receive_stream, args=(server,), daemon=True).start()
        return int(server.getsockname()[1])

    def _receive_stream(self, server: socket.socket) -> None:
        from PIL import Image

        while not self.stream_stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with conn:
                for name, jpeg in read_frame_messages(conn):
                    if self.stream_stop.is_set():
                        return
                    try:
                        image = Image.open(io.BytesIO(jpeg))
                        image.load()
                    except OSError:
                        continue
                    self.frames[name] = image

    def _close_stream(self) -> None:
        self.stream_stop.set()
        if self.stream_server is not None:
            self.stream_server.close()
            self.stream_server = None

    def _render_frames(self) -> None:
        if self.proc is None:
            return
        for tile in self.tiles:
            image = self.frames.get(tile.name)
            if image is not None and image is not tile.shown:
                tile.show(image)
        if (
            self.frames
            and self.phase == PHASE_STARTING
            and self.run_values
            and self.run_values.manual_control
        ):
            self._set_phase(PHASE_READY, "Press Space to start an episode")
        self.root.after(RENDER_INTERVAL_MS, self._render_frames)

    # ---- recording -------------------------------------------------------------------------

    def _on_button(self) -> None:
        if self.proc is None:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        values, error = self._read_form()
        if values is None or error:
            return
        self.run_values = values
        self.run_started_at = datetime.now()
        self.run_repo_id = make_repo_id(values.dataset_name, self.hf_user, self.run_started_at)
        stream_port = self._open_stream_server() if self.defaults.display_data and values.cameras else None
        cmd = build_record_command(values, self.run_repo_id, self.defaults, stream_port=stream_port)

        self._clear_log()
        self._append_log(f"$ lerobot-record ... --dataset.repo_id={self.run_repo_id}\n", "cmd", echo=False)
        _echo(f"$ {shlex.join(cmd)}\n")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen(  # nosec B603 - argv list built from our own config
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._pump_output, args=(self.proc,), daemon=True).start()

        self._set_inputs_enabled(False)
        self._show_live_view(values)
        self.button.configure(text="Stop recording", style="Stop.TButton", state="normal")
        self._set_status("Recording  ·  keys work while any window is focused")
        self.hint.configure(text=MANUAL_KEYS_HELP if values.manual_control else TIMED_KEYS_HELP)
        self.root.focus_set()
        self.root.after(100, self._poll)
        self.root.after(RENDER_INTERVAL_MS, self._render_frames)

    def _stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            self.button.configure(state="disabled")
            upload = self.run_values is not None and self.run_values.push_to_hub
            message = "Stopping...  finalizing" + (" and uploading" if upload else "") + " the dataset"
            self._set_phase(PHASE_STOPPING, message)
            self._set_status(message)

    def _pump_output(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            _echo(line)  # mirror to the terminal run-gui was started from, as it arrives
            self.lines.put(line)
        self.lines.put(None)

    def _poll(self) -> None:
        finished = False
        manual = self.run_values is not None and self.run_values.manual_control
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                break
            if line is None:
                finished = True
                continue
            line = clean_log_line(line)
            is_error = "ERROR" in line or "Traceback" in line
            self._append_log(line, "error" if is_error else None, echo=False)
            if (phase := phase_from_log(line, manual)) is not None:
                self._set_phase(*phase)
        if finished:
            self._on_finished()
        else:
            self.root.after(100, self._poll)

    def _save_session(self) -> Path | None:
        assert self.run_values is not None
        return write_session_file(
            HF_LEROBOT_HOME / self.run_repo_id, self.run_values, self.run_repo_id, self.run_started_at
        )

    def _on_finished(self) -> None:
        assert self.proc is not None
        code = self.proc.wait()
        self.proc = None
        self._close_stream()
        session = self._save_session()
        if session is not None:
            self._append_log(f"\nOperator info saved to {session}\n", "cmd")
        self._show_setup_view()
        self._set_inputs_enabled(True)
        self.button.configure(text="Start recording", style="Primary.TButton")
        self._refresh()
        # SIGINT exits with -2 / 130; that's a normal Stop, not a failure.
        if code in (0, -signal.SIGINT, 130):
            self._set_status(f"Done: {self.run_repo_id}")
        else:
            self._set_status(f"lerobot-record exited with code {code}. See the terminal.", error=True)
            if not self.terminal_maximized:
                self._toggle_terminal()

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, text: str, tag: str | None = None, echo: bool = True) -> None:
        """Show *text* in the GUI terminal and, with *echo*, in the terminal run-gui was started from."""
        if echo:
            _echo(text)
        at_bottom = self.log.yview()[1] >= 0.999
        self.log.configure(state="normal")
        self.log.insert("end", text, tag or ())
        self.log.configure(state="disabled")
        if at_bottom:  # don't yank the view while the operator scrolls back
            self.log.see("end")

    def _on_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            from tkinter import messagebox

            if not messagebox.askyesno(
                "Recording in progress", "Stop the recording and quit? The dataset will be finalized first."
            ):
                return
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait()
            self._save_session()
        self._close_stream()
        self.root.destroy()

    def run(self) -> None:
        """Show the window until it is closed."""
        self.root.mainloop()


def main() -> None:
    """Entry point of ``run-gui``."""
    TeleopGui().run()


if __name__ == "__main__":
    main()
