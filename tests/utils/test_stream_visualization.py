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

import socket

import cv2
import numpy as np
import pytest

from lerobot.utils import stream_visualization as sv, visualization_utils as vu


def test_camera_frames_picks_rgb_images():
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    chw = np.zeros((3, 48, 64), dtype=np.uint8)
    frames = sv.camera_frames(
        {
            "wrist": rgb,
            "observation.images.top": chw,
            "depth": np.zeros((48, 64, 1), dtype=np.uint16),
            "shoulder_pan.pos": np.float32(1.0),
        }
    )
    assert set(frames) == {"wrist", "top"}
    assert frames["top"].shape == (48, 64, 3)


def test_wire_format_roundtrip():
    a, b = socket.socketpair()
    with a, b:
        a.sendall(sv.encode_frame_message("wrist", b"jpeg-1") + sv.encode_frame_message("top", b"jpeg-22"))
        a.shutdown(socket.SHUT_WR)
        assert list(sv.read_frame_messages(b)) == [("wrist", b"jpeg-1"), ("top", b"jpeg-22")]


def test_stream_sends_jpeg_frames_to_a_listening_viewer():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(5)
    with server:
        vu.init_visualization("stream", ip="127.0.0.1", port=server.getsockname()[1])
        conn, _ = server.accept()
        conn.settimeout(5)
        with conn:
            image = np.full((48, 64, 3), (255, 0, 0), dtype=np.uint8)  # pure red, RGB
            vu.log_visualization_data("stream", observation={"wrist": image})
            name, jpeg = next(sv.read_frame_messages(conn))
            vu.shutdown_visualization("stream")
    assert name == "wrist"
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)  # BGR
    assert decoded.shape == (48, 64, 3)
    assert decoded[24, 32, 2] > 200 and decoded[24, 32, 0] < 50  # still red


def test_stream_requires_a_port():
    with pytest.raises(ValueError, match="display_port"):
        vu.init_visualization("stream", ip="127.0.0.1", port=None)


def test_logging_after_viewer_left_does_not_raise():
    vu.shutdown_visualization("stream")
    vu.log_visualization_data("stream", observation={"wrist": np.zeros((4, 4, 3), dtype=np.uint8)})
