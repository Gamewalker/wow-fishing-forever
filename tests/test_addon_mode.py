"""Addon mode: one key casts and reels in, the bot only listens; no screen capture at all."""

from __future__ import annotations

import random

import numpy as np
import pytest

from wowfish.audio.detector import BiteEvent
from wowfish.capture import WindowRef
from wowfish.config import (
    AudioConfig,
    Config,
    GameConfig,
    RunConfig,
    VisionConfig,
    ensure_config_file,
    load_config,
)
from wowfish.engine import events as ev
from wowfish.engine.engine import ADDON_ESCAPE_S, ADDON_LOOT_S, FishingEngine
from wowfish.engine.state import State


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
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
    def __init__(self) -> None:
        self.bite = False

    def reset(self) -> None:
        self.bite = False

    def feed(self, chunk: object) -> list[BiteEvent]:
        return [BiteEvent("splash", 0.9, 0.0)] if self.bite else []


def _engine(run: RunConfig) -> tuple[FishingEngine, _Clock, _Input, _Detector]:
    cfg = Config(GameConfig(addon_mode=True), VisionConfig(), AudioConfig(), run)
    clock, inp, det = _Clock(), _Input(), _Detector()
    engine = FishingEngine(
        cfg,
        WindowRef("World of Warcraft", None, 0, 0, 800, 600),
        capturer=None,  # any screen grab would fail the test
        input_device=inp,
        audio_source=_Audio(),
        detector=det,
        clock=clock,
        sleep=clock.sleep,
        rng=random.Random(1),
    )
    return engine, clock, inp, det


def _events(engine: FishingEngine) -> list[ev.Event]:
    out = []
    while not engine.events.empty():
        out.append(engine.events.get())
    return out


def test_one_key_casts_listens_and_reels_in():
    engine, clock, inp, det = _engine(RunConfig())
    engine.step()
    assert inp.pressed == ["1"] and engine.state is State.WAIT_BITE
    engine.step()  # nothing heard yet: keep listening, no icon search
    assert engine.state is State.WAIT_BITE
    det.bite = True
    engine.step()
    assert inp.pressed == ["1", "1"] and engine.state is State.INTERACT
    engine.step()
    assert engine.state is State.INTERACT  # Auto Loot gets its moment
    clock.t += ADDON_LOOT_S
    engine.step()
    assert engine.state is State.IDLE and engine.stats.catches == 1
    assert any(isinstance(e, ev.Caught) for e in _events(engine))


def test_humanizing_options_still_apply():
    run = RunConfig(reaction_min_s=0.5, reaction_max_s=0.5, move_chance=1.0, pause_min_s=3.0,
                    pause_max_s=3.0)  # fmt: skip
    engine, clock, inp, det = _engine(run)
    engine.step()
    det.bite = True
    engine.step()
    assert clock.t >= 0.5 and inp.pressed == ["1", "1"]  # waited before reeling in
    clock.t += ADDON_LOOT_S
    t = clock.t
    engine.step()
    assert inp.pressed == ["1", "1", "q", "e"]  # the step and back after the catch
    assert clock.t - t == pytest.approx(3.0)  # and the pause before the next cast


def test_skipped_bite_waits_for_the_fish_to_get_away():
    engine, clock, inp, det = _engine(RunConfig(skip_bite_chance=0.99, pause_min_s=0.0,
                                                pause_max_s=0.0))  # fmt: skip
    engine.step()
    det.bite = True
    engine.step()
    assert inp.pressed == ["1"] and engine.state is State.IDLE and engine.stats.misses == 1
    assert clock.t == pytest.approx(ADDON_ESCAPE_S)  # so the next press casts, not reels in


def test_no_splash_times_out_as_a_miss():
    engine, clock, inp, _ = _engine(RunConfig(cast_timeout_s=5.0))
    engine.step()
    clock.t += 6.0
    engine.step()
    assert engine.state is State.IDLE and engine.stats.misses == 1 and inp.pressed == ["1"]


def test_addon_mode_is_off_by_default_and_in_the_default_config(tmp_path):
    path = tmp_path / "config.toml"
    ensure_config_file(path)
    assert "addon_mode = false" in path.read_text()
    assert load_config(path)[0].game.addon_mode is False
    path.write_text("[game]\naddon_mode = true\n")
    assert load_config(path)[0].game.addon_mode is True
