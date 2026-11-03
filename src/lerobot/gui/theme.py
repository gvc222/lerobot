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

"""Hugging Face look for ``run-gui``.

Brand colors from https://huggingface.co/brand (yellow, orange, gray); the neutrals around them are
the gray scale the huggingface.co site pairs them with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tkinter as tk

# Brand colors.
HF_YELLOW = "#FFD21E"
HF_ORANGE = "#FF9D00"
HF_GRAY = "#6B7280"

# Neutrals.
WHITE = "#FFFFFF"
SURFACE = "#F9FAFB"
BORDER = "#E5E7EB"
TEXT = "#111827"
TEXT_MUTED = HF_GRAY
DISABLED_BG = "#F3F4F6"
DISABLED_TEXT = "#9CA3AF"
DARK = "#0B0F19"
DARK_RAISED = "#1F2937"
DARK_TEXT = "#E5E7EB"
RED = "#EF4444"

# Badge colors per recording phase: (background, foreground).
PHASE_COLORS = {
    "starting": (HF_GRAY, WHITE),
    "ready": (HF_YELLOW, TEXT),
    "recording": (RED, WHITE),
    "label": (HF_ORANGE, TEXT),
    "reset": (HF_YELLOW, TEXT),
    "stopping": (HF_GRAY, WHITE),
}

# huggingface.co's typefaces first.  Some Tk builds (e.g. uv's standalone Python) only see the X11 core
# fonts, hence the Nimbus / Helvetica / Courier fallbacks.
_SANS = (
    "Source Sans Pro",
    "Source Sans 3",
    "Inter",
    "Noto Sans",
    "Ubuntu",
    "DejaVu Sans",
    "Nimbus Sans L",
    "Helvetica",
)
_MONO = ("IBM Plex Mono", "Source Code Pro", "Ubuntu Mono", "DejaVu Sans Mono", "Nimbus Mono L", "Courier")


@dataclass(frozen=True)
class Fonts:
    """Font tuples for the widgets; families fall back to whatever is installed."""

    body: tuple[str, int]
    bold: tuple[str, int, str]
    title: tuple[str, int, str]
    small: tuple[str, int]
    mono: tuple[str, int]


def _first_available(root: tk.Misc, candidates: tuple[str, ...], fallback_font: str) -> str:
    """First installed family of *candidates*, else the family behind the named Tk font *fallback_font*."""
    from tkinter import font

    installed = {family.lower(): family for family in font.families(root)}
    for family in candidates:
        if family.lower() in installed:
            return installed[family.lower()]
    return str(font.nametofont(fallback_font, root=root).actual("family"))


def apply_theme(root: tk.Tk) -> Fonts:
    """Style every ttk widget class the GUI uses and return the fonts to use for tk widgets."""
    from tkinter import ttk

    sans = _first_available(root, _SANS, "TkDefaultFont")
    mono = _first_available(root, _MONO, "TkFixedFont")
    fonts = Fonts(
        body=(sans, 11), bold=(sans, 11, "bold"), title=(sans, 17, "bold"), small=(sans, 10), mono=(mono, 10)
    )

    root.configure(background=WHITE)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=WHITE, foreground=TEXT, font=fonts.body, bordercolor=BORDER)
    style.configure("TFrame", background=WHITE)
    style.configure("TLabel", background=WHITE, foreground=TEXT)
    style.configure("Muted.TLabel", foreground=TEXT_MUTED, font=fonts.small)
    style.configure("Header.TLabel", foreground=TEXT_MUTED, font=fonts.small)
    style.configure("Title.TLabel", font=fonts.title)
    style.configure("Status.TLabel", foreground=TEXT_MUTED)
    style.configure("Error.TLabel", foreground=RED)

    style.configure("Card.TLabelframe", background=WHITE, bordercolor=BORDER, relief="solid", borderwidth=1)
    style.configure("Card.TLabelframe.Label", background=WHITE, foreground=TEXT, font=fonts.bold)

    field = {"fieldbackground": WHITE, "bordercolor": BORDER, "lightcolor": WHITE, "darkcolor": WHITE}
    for widget in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(widget, padding=(6, 4), arrowcolor=TEXT_MUTED, **field)
        style.map(
            widget,
            bordercolor=[("focus", HF_ORANGE)],
            lightcolor=[("focus", HF_YELLOW)],
            fieldbackground=[("disabled", DISABLED_BG), ("readonly", WHITE)],
            foreground=[("disabled", DISABLED_TEXT)],
        )
    root.option_add("*TCombobox*Listbox.background", WHITE)
    root.option_add("*TCombobox*Listbox.selectBackground", HF_YELLOW)
    root.option_add("*TCombobox*Listbox.selectForeground", TEXT)
    root.option_add("*TCombobox*Listbox.font", fonts.body)

    style.configure("TCheckbutton", background=WHITE, indicatorbackground=WHITE, indicatorforeground=TEXT)
    style.map(
        "TCheckbutton",
        background=[("active", WHITE)],
        indicatorbackground=[("selected", HF_YELLOW), ("disabled", DISABLED_BG)],
        foreground=[("disabled", DISABLED_TEXT)],
    )

    button: dict[str, Any] = {"borderwidth": 0, "focusthickness": 0, "padding": (20, 9), "font": fonts.bold}
    style.configure("Primary.TButton", background=HF_YELLOW, foreground=TEXT, **button)
    style.map(
        "Primary.TButton",
        background=[("disabled", DISABLED_BG), ("pressed", HF_ORANGE), ("active", HF_ORANGE)],
        foreground=[("disabled", DISABLED_TEXT)],
    )
    style.configure("Stop.TButton", background=TEXT, foreground=WHITE, **button)
    style.map(
        "Stop.TButton",
        background=[("disabled", DISABLED_BG), ("active", DARK_RAISED)],
        foreground=[("disabled", DISABLED_TEXT)],
    )

    style.configure("TPanedwindow", background=BORDER)
    style.configure("Sash", sashthickness=6, gripcount=0, background=BORDER)
    style.configure(
        "Vertical.TScrollbar",
        background=DARK_RAISED,
        troughcolor=DARK,
        bordercolor=DARK,
        arrowcolor=DARK_TEXT,
    )
    return fonts
