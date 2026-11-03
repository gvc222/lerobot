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

"""``--display_mode=stream``: send camera frames as JPEG over TCP to a local viewer (e.g. ``run-gui``).

The viewer listens on ``display_ip:display_port``; :func:`init_stream` connects to it.  The record
loop only hands over references to the latest frames (:func:`log_stream_data` never blocks); a
background thread JPEG-encodes and sends them at up to :data:`STREAM_MAX_FPS`.  If the viewer goes
away the stream stops with a warning and recording carries on.

Wire format, repeated: ``uint32 name_len | name (utf-8) | uint32 jpeg_len | jpeg`` (big-endian).
"""

from __future__ import annotations

import contextlib
import logging
import socket
import struct
import threading
import time
from collections.abc import Iterator

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation

STREAM_MAX_FPS = 15.0
JPEG_QUALITY = 80
_HEADER = struct.Struct(">I")
_IMAGE_PREFIX = "observation.images."

logger = logging.getLogger(__name__)


def encode_frame_message(name: str, jpeg: bytes) -> bytes:
    """One frame on the wire."""
    raw_name = name.encode()
    return _HEADER.pack(len(raw_name)) + raw_name + _HEADER.pack(len(jpeg)) + jpeg


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            return None
        chunks += chunk
    return bytes(chunks)


def read_frame_messages(sock: socket.socket) -> Iterator[tuple[str, bytes]]:
    """Yield ``(camera name, jpeg bytes)`` until the sender closes the connection."""
    while True:
        header = _recv_exact(sock, _HEADER.size)
        if header is None:
            return
        name = _recv_exact(sock, _HEADER.unpack(header)[0])
        size = _recv_exact(sock, _HEADER.size)
        if name is None or size is None:
            return
        jpeg = _recv_exact(sock, _HEADER.unpack(size)[0])
        if jpeg is None:
            return
        yield name.decode(), jpeg


def camera_frames(observation: RobotObservation | None) -> dict[str, np.ndarray]:
    """The RGB camera images of *observation* as HWC uint8 arrays, keyed by camera name."""
    frames: dict[str, np.ndarray] = {}
    for key, value in (observation or {}).items():
        if not isinstance(value, np.ndarray) or value.ndim != 3:
            continue
        arr = value
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):  # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] != 3:  # depth / alpha: not streamed
            continue
        if arr.dtype != np.uint8:
            arr = (
                (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.dtype.kind == "f" else arr.astype(np.uint8)
            )
        name = str(key).removeprefix(_IMAGE_PREFIX)
        frames[name] = arr
    return frames


class _StreamSender:
    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.settimeout(None)
        self.lock = threading.Lock()
        self.latest: dict[str, np.ndarray] = {}
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name="stream-visualization", daemon=True)
        self.thread.start()

    def offer(self, frames: dict[str, np.ndarray]) -> None:
        with self.lock:
            self.latest.update(frames)

    def _run(self) -> None:
        import cv2

        period = 1.0 / STREAM_MAX_FPS
        while not self.stopped.is_set():
            start = time.monotonic()
            with self.lock:
                frames, self.latest = self.latest, {}
            try:
                for name, rgb in frames.items():
                    ok, jpeg = cv2.imencode(
                        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )
                    if ok:
                        self.sock.sendall(encode_frame_message(name, jpeg.tobytes()))
            except OSError as e:
                if not self.stopped.is_set():
                    logger.warning(f"Camera stream viewer disconnected ({e}); recording continues.")
                self.stopped.set()
                return
            self.stopped.wait(max(0.0, period - (time.monotonic() - start)))

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=2)
        with contextlib.suppress(OSError):
            self.sock.close()


_sender: _StreamSender | None = None


def init_stream(host: str = "127.0.0.1", port: int | None = None) -> None:
    """Connect to the viewer listening on ``host:port``."""
    global _sender
    if port is None:
        raise ValueError("display_mode='stream' needs --display_port (the viewer's listening port).")
    shutdown_stream()
    _sender = _StreamSender(host, port)


def log_stream_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """Hand the latest camera frames to the sender thread.  ``action`` / ``compress_images`` are unused."""
    del action, compress_images
    if _sender is not None and not _sender.stopped.is_set():
        _sender.offer(camera_frames(observation))


def shutdown_stream() -> None:
    """Stop the sender thread and close the connection."""
    global _sender
    if _sender is not None:
        _sender.close()
        _sender = None
