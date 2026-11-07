"""The optional random delay between hearing a bite and pressing Interact."""

from __future__ import annotations

import random

import numpy as np
import pytest

from wowfish.audio.detector import BiteEvent
from wowfish.capture import WindowRef
from wowfish.config import (
    AudioConfig,
    Config,
    ConfigError,
    GameConfig,
    RunConfig,
    VisionConfig,
    ensure_config_file,
    load_config,
)
from wowfish.engine import events as ev
from wowfish.engine.engine import SLEEP_SLICE_S, FishingEngine
from wowfish.engine.state import State


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


class _Input:
    def __init__(self) -> None:
        self.pressed: list[str] = []

    def is_game_focused(self, window: WindowRef) -> bool:
        return True

    def press_key(self, key: str, hold_s: float) -> None:
        self.pressed.append(key)


class _Audio:
    sample_rate = 16000

    def read(self, n: int) -> np.ndarray:
        return np.zeros(n, np.float32)


class _Detector:
    def feed(self, chunk: object) -> list[BiteEvent]:
        return [BiteEvent("splash", 0.9, 0.0)]


def _bite(run: RunConfig, *, stop: bool = False) -> tuple[FishingEngine, _Clock, _Input]:
    """One engine step that hears a bite; `stop` presses Stop during the first sleep."""
    cfg = Config(GameConfig(), VisionConfig(), AudioConfig(), run)
    clock, inp = _Clock(), _Input()
    engine = FishingEngine(
        cfg,
        WindowRef("World of Warcraft", None, 0, 0, 800, 600),
        capturer=None,
        input_device=inp,
        audio_source=_Audio(),
        detector=_Detector(),
        clock=clock,
        sleep=lambda s: (clock.sleep(s), stop and engine.stop()),
        rng=random.Random(1),
    )
    engine.state = State.WAIT_BITE
    engine.step()
    return engine, clock, inp


def _changes(engine: FishingEngine) -> list[ev.StateChanged]:
    events = []
    while not engine.events.empty():
        events.append(engine.events.get())
    return [e for e in events if isinstance(e, ev.StateChanged)]


def test_reaction_delay_waits_in_slices_before_pressing():
    engine, clock, inp = _bite(RunConfig(reaction_min_s=0.3, reaction_max_s=0.6))
    delay = random.Random(1).uniform(0.3, 0.6)
    assert inp.pressed == ["f"]
    assert sum(clock.sleeps) == pytest.approx(delay)
    assert max(clock.sleeps) <= SLEEP_SLICE_S
    assert engine._t_interact == pytest.approx(delay)
    assert f"waited {delay:.2f} s" in _changes(engine)[-1].did


def test_no_reaction_delay_by_default():
    engine, clock, inp = _bite(RunConfig())
    assert inp.pressed == ["f"] and clock.sleeps == []
    assert engine._rng.random() == random.Random(1).random()  # the rng was not used
    assert "waited" not in _changes(engine)[-1].did


def test_stop_during_reaction_delay_skips_the_press():
    _, clock, inp = _bite(RunConfig(reaction_min_s=1.0, reaction_max_s=1.0), stop=True)
    assert inp.pressed == [] and clock.sleeps == [SLEEP_SLICE_S]


@pytest.mark.parametrize(
    "lines",
    [
        "reaction_min_s = -0.1\nreaction_max_s = 0.5",
        "reaction_min_s = 0.6\nreaction_max_s = 0.5",
        "reaction_min_s = 0.0\nreaction_max_s = 2.5",
    ],
)
def test_reaction_delay_config_is_validated(tmp_path, lines):
    path = tmp_path / "config.toml"
    path.write_text(f"[run]\n{lines}\n")
    with pytest.raises(ConfigError, match=r"run\.reaction_m"):
        load_config(path)


def test_reaction_delay_is_in_the_default_config(tmp_path):
    path = tmp_path / "config.toml"
    ensure_config_file(path)
    assert "reaction_max_s = 0.0" in path.read_text()
