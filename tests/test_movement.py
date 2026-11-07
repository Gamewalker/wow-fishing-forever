"""The optional small step after a catch: off by default, validated, and always undone."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from wowfish.capture import WindowRef
from wowfish.config import AudioConfig, Config, ConfigError, GameConfig, VisionConfig, load_config
from wowfish.engine import events as ev
from wowfish.engine.engine import FishingEngine
from wowfish.engine.state import State


class _Input:
    def __init__(self) -> None:
        self.focused = True
        self.presses: list[tuple[str, float]] = []

    def press_key(self, key: str, hold_s: float = 0.06) -> None:
        self.presses.append((key, hold_s))

    def is_game_focused(self, window: WindowRef) -> bool:
        return self.focused


class _Audio:
    sample_rate = 48000

    def read(self, n: int) -> np.ndarray:
        return np.zeros(n, np.float32)


class _Gone:
    """A matcher that never sees the bobber, so the engine counts a catch after Interact."""

    def match(self, crop: np.ndarray, full_width: int) -> object:
        return type("M", (), {"accepted": lambda self, t: False})()


class _Capturer:
    def grab(self, box: object = None) -> np.ndarray:
        return np.zeros((10, 10, 3), np.uint8)


def _catch(move_chance: float, *, focused: bool = True, seed: int = 1):
    """Run the engine from Interact to the catch and the pause after it."""
    cfg = Config(GameConfig(), VisionConfig(), AudioConfig())
    cfg = replace(cfg, run=replace(cfg.run, move_chance=move_chance))
    inp = _Input()
    inp.focused = focused
    eng = FishingEngine(
        cfg,
        WindowRef("World of Warcraft", None, 0, 0, 1920, 1080),
        capturer=_Capturer(),
        input_device=inp,
        audio_source=_Audio(),
        detector=None,
        matcher=_Gone(),
        clock=lambda: 0.0,
        sleep=lambda s: None,
        rng=random.Random(seed),
        lost_after=1,
    )
    eng.state, eng._icon_xy, eng.started = State.INTERACT, (100, 100), 0.0
    eng._step(0.0)
    events = []
    while not eng.events.empty():
        events.append(eng.events.get())
    return eng, inp, events


def test_step_after_catch_holds_both_keys_equally_long():
    eng, inp, events = _catch(1.0)
    assert eng.state is State.IDLE
    assert [k for k, _ in inp.presses] == ["q", "e"]
    hold = inp.presses[0][1]
    assert inp.presses[1][1] == hold and 0.15 <= hold <= 0.3
    assert [e.why for e in events if isinstance(e, ev.KeyPressed)] == ["move", "move"]
    idle = [e for e in events if isinstance(e, ev.StateChanged)][-1]
    assert "small step" in idle.next


def test_no_step_by_default_or_when_unfocused():
    for chance, focused in ((0.0, True), (1.0, False)):
        _, inp, _ = _catch(chance, focused=focused)
        assert inp.presses == []
    # move_chance = 0 draws nothing extra from the rng, so pauses stay as they were
    rng = random.Random(1)
    expected = f"Casting again in {rng.uniform(0.5, 2.0):.1f} s"
    idle = [e for e in _catch(0.0)[2] if isinstance(e, ev.StateChanged)][-1]
    assert idle.next == expected


def test_step_stops_when_focus_is_lost_after_the_first_key():
    class _LosesFocus(_Input):
        def is_game_focused(self, window: WindowRef) -> bool:
            return not self.presses  # the user clicks away while the first key is held

    eng, _, _ = _catch(1.0)
    inp = eng.input = _LosesFocus()
    eng._step_and_back()
    assert [k for k, _ in inp.presses] == ["q"]


@pytest.mark.parametrize(
    "line",
    ["move_chance = 1.5", 'move_keys = "q"', 'move_keys = "q,q"', 'move_keys = "q,ctrl"',
     "move_max_ms = 20"],
)  # fmt: skip
def test_bad_movement_settings_are_rejected(tmp_path, line):
    path = tmp_path / "config.toml"
    path.write_text(f"[run]\n{line}\n")
    with pytest.raises(ConfigError):
        load_config(path)
