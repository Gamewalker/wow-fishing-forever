"""Screen capture boundary: find the game window and grab its pixels as BGR arrays."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

Box = tuple[int, int, int, int]  # x, y, width, height in window pixels

GAME_TITLE = "World of Warcraft"
# The game's executable (release, beta); unlike the title it is the same in every language.
GAME_EXES = ("wow.exe", "wowb.exe")


@dataclass(frozen=True)
class WindowRef:
    title: str
    pid: int | None
    left: int
    top: int
    width: int
    height: int
    handle: int | None = None  # native window id where the platform has one


@runtime_checkable
class Capturer(Protocol):
    def open(self, window: WindowRef) -> None: ...

    def grab(self, box: Box | None = None) -> np.ndarray:
        """Return the window (or `box` inside it) as HxWx3 uint8 BGR."""
        ...

    def close(self) -> None: ...


def linux_x11() -> bool:
    return sys.platform == "linux" and bool(os.environ.get("DISPLAY"))


def window_matches(title: str, exe: str, title_substring: str = GAME_TITLE) -> bool:
    """Whether the title contains the text (ignoring case), or, when looking for the game by
    its default title, whether the window belongs to the game's executable."""
    if title_substring.lower() in title.lower():
        return True
    return title_substring == GAME_TITLE and exe.lower() in GAME_EXES


def find_window(title_substring: str = GAME_TITLE) -> WindowRef | None:
    """The largest viewable window that `window_matches` the text (X11, Windows; else None)."""
    if linux_x11():
        from wowfish.capture._x11 import find_window_x11

        return find_window_x11(title_substring)
    if sys.platform == "win32":
        from wowfish.capture import _win32

        return _win32.find_window_win32(title_substring)
    return None


def get_capturer() -> Capturer:
    if linux_x11():
        from wowfish.capture._x11 import X11WindowCapturer

        return X11WindowCapturer()
    try:
        import mss  # noqa: F401
    except Exception:
        from wowfish.capture._stub import StubCapturer

        return StubCapturer()
    from wowfish.capture._mss import MssCapturer

    return MssCapturer()


__all__ = [
    "Box",
    "Capturer",
    "WindowRef",
    "find_window",
    "get_capturer",
    "linux_x11",
    "window_matches",
]
