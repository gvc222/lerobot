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

"""Text-to-speech for ``lerobot-run-gui``, so the operator hears the recording state without looking.

Uses the system speech tool (``spd-say`` / ``espeak-ng`` / ``espeak`` on Linux, ``say`` on macOS,
System.Speech on Windows).  Sentences are spoken one at a time on a background thread, so they never
talk over each other and never block the Tk loop.
"""

from __future__ import annotations

import logging
import platform
import queue
import shutil
import subprocess  # nosec B404
import threading

logger = logging.getLogger(__name__)

# A sentence that waits behind more than this many others is stale: drop the oldest.
MAX_BACKLOG = 3
SPEAK_TIMEOUT_S = 20


def speech_command(text: str) -> list[str] | None:
    """The argv that speaks *text* and returns once it's done, or None if no speech tool is available."""
    system = platform.system()
    if system == "Darwin":
        return ["say", text]
    if system == "Windows":
        escaped = text.replace("'", "''")
        return [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{escaped}')",
        ]
    if shutil.which("spd-say"):
        return ["spd-say", "--wait", text]
    for tool in ("espeak-ng", "espeak"):
        if shutil.which(tool):
            return [tool, text]
    return None


class Voice:
    """Speaks sentences in order on a background thread.  ``say()`` never blocks."""

    def __init__(self) -> None:
        """Start the speaking thread."""
        self.pending: queue.Queue[str] = queue.Queue()
        self._warned = False
        threading.Thread(target=self._run, name="gui-voice", daemon=True).start()

    def say(self, text: str) -> None:
        """Queue *text* to be spoken after the sentences before it."""
        while self.pending.qsize() >= MAX_BACKLOG:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                break
        self.pending.put(text)

    def _run(self) -> None:
        while True:
            text = self.pending.get()
            cmd = speech_command(text)
            if cmd is None:
                if not self._warned:
                    logger.warning("No text-to-speech tool found (install speech-dispatcher or espeak-ng).")
                    self._warned = True
                continue
            try:
                subprocess.run(  # nosec B603 - fixed tool, text passed as one argv entry
                    cmd,
                    check=False,
                    timeout=SPEAK_TIMEOUT_S,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                logger.warning(f"Text-to-speech failed: {e}")
