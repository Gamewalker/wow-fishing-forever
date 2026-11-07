"""Optional breaks between casts: off by default, Stop and the session limit still apply."""

from __future__ import annotations

import random
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from wowfish.capture import WindowRef
from wowfish.config import AudioConfig, Config, ConfigError, GameConfig, VisionConfig, load_config
from wowfish.engine import events as ev
from wowfish.engine.engine import SLEEP_SLICE_S, FishingEngine


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def _engine(t: FakeTime, seed: int = 0, session_s: float = 0, **run) -> FishingEngine:
    cfg = Config(game=GameConfig(), vision=VisionConfig(), audio=AudioConfig())
    cfg = replace(cfg, run=replace(cfg.run, **run))
    window = MagicMock(spec=WindowRef, width=1920, height=1080)
    return FishingEngine(
        cfg, window, capturer=MagicMock(), input_device=MagicMock(), audio_source=MagicMock(),
        detector=MagicMock(), clock=t.clock, sleep=t.sleep, rng=random.Random(seed),
        session_s=session_s,
    )  # fmt: skip


def _next_texts(engine: FishingEngine) -> list[str]:
    out = []
    while not engine.events.empty():
        e = engine.events.get()
        if isinstance(e, ev.StateChanged):
            out.append(e.next)
    return out


def test_no_break_by_default_and_rng_untouched():
    t = FakeTime()
    engine = _engine(t, seed=1)
    engine._pause_then_idle("done")
    expected = random.Random(1)
    assert t.now == pytest.approx(expected.uniform(0.5, 2.0))
    assert engine._rng.random() == expected.random()
    assert _next_texts(engine)[0].startswith("Casting again in")


def test_break_is_announced_and_slept_in_slices():
    t = FakeTime()
    engine = _engine(t, break_chance=1.0, break_min_s=10.0, break_max_s=10.0)
    engine._pause_then_idle("done")
    assert 10.5 <= t.now <= 12.0
    assert max(t.sleeps) <= SLEEP_SLICE_S
    assert "Taking a break" in _next_texts(engine)[0]


def test_break_ends_at_session_limit_and_on_stop():
    t = FakeTime()
    engine = _engine(t, session_s=5.0, break_chance=1.0, break_min_s=60.0, break_max_s=60.0)
    engine.started = 0.0
    engine._pause_then_idle("done")
    assert 5.0 <= t.now < 5.0 + 2 * SLEEP_SLICE_S
    engine.step()
    assert isinstance(list(engine.events.queue)[-1], ev.Stopped)

    t = FakeTime()
    engine = _engine(t, break_chance=1.0, break_min_s=60.0, break_max_s=60.0)
    engine._sleep = lambda s: (t.sleep(s), engine.stop())
    engine._pause_then_idle("done")
    assert len(t.sleeps) == 1


@pytest.mark.parametrize(
    "line", ["break_chance = 1.5", "break_chance = -0.1", "break_min_s = -1.0",
             "break_min_s = 10.0\nbreak_max_s = 5.0"],
)  # fmt: skip
def test_break_settings_are_validated(tmp_path, line):
    path = tmp_path / "config.toml"
    path.write_text(f"[run]\n{line}\n")
    with pytest.raises(ConfigError, match=r"run\.break_"):
        load_config(path)
