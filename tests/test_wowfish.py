"""Seven checks that protect what the bot actually relies on: the shipped assets, the default
config, logging in the console-less GUI executable, and that the two detectors say yes to the
real thing and no to nothing.

Everything here runs on bundled files and synthetic signals. No screen, audio device, window
system, or Tk is touched, so a failure means the code or the assets changed, not the machine.
"""

from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from wowfish.assets import CLIP_NAMES, ICON_SIZES
from wowfish.audio import AudioChunk
from wowfish.audio.clips import build_fingerprints, load_reference_clips
from wowfish.audio.detector import BiteDetector, BiteEvent, detector_for
from wowfish.audio.features import DEFAULT_MEL
from wowfish.capture import WindowRef
from wowfish.cli import _setup_logging
from wowfish.config import (
    AudioConfig,
    Config,
    ConfigError,
    GameConfig,
    RunConfig,
    VisionConfig,
    _validate,
    ensure_config_file,
    load_config,
    save_config,
)
from wowfish.engine import events as ev
from wowfish.engine.engine import FishingEngine
from wowfish.engine.state import State
from wowfish.vision.matcher import IconMatcher
from wowfish.vision.template import load_icon_templates, template_for_size

REPO_ROOT = Path(__file__).resolve().parents[1]
RATE = DEFAULT_MEL.sample_rate


def test_bundled_assets_load():
    """Every sprite and clip the code expects is shipped and readable; the release check agrees."""
    check = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_release.py")],
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr
    assert tuple(t.native_size for t in load_icon_templates()) == ICON_SIZES
    assert tuple(fp.name for fp in build_fingerprints()) == CLIP_NAMES


def test_default_config_matches_code_defaults(tmp_path):
    """The shipped default_config file and the dataclass defaults describe the same settings,
    and a config survives a save/load round trip unchanged."""
    path = tmp_path / "config.toml"
    assert ensure_config_file(path) is True
    assert ensure_config_file(path) is False  # never overwrites an existing file

    from_file, warnings = load_config(path)
    assert warnings == []
    assert from_file == Config(game=GameConfig(), vision=VisionConfig(), audio=AudioConfig())

    save_config(from_file, path)
    assert load_config(path) == (from_file, [])


def test_logging_works_without_stderr(tmp_path, monkeypatch):
    """wowfish-gui.exe is built without a console, so on Windows Python sets sys.stderr to None.
    Logging setup must not crash there, and the log file must still be written."""
    monkeypatch.setenv("WOWFISH_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "stderr", None)
    try:
        log_path = _setup_logging()
        logger.info("started without a console")
    finally:
        logger.remove()  # closes the file sink so tmp_path can be deleted on Windows
    assert "started without a console" in log_path.read_text()


def _hear(stream: np.ndarray) -> tuple[list[BiteEvent], BiteDetector]:
    """Run a stream through the detector the bot builds from the default config, in modest
    chunks like the live loop does."""
    det = detector_for(AudioConfig())
    events: list[BiteEvent] = []
    chunk = 2048
    for i in range(0, len(stream), chunk):
        part = stream[i : i + chunk]
        events += det.feed(AudioChunk(part, RATE, (i + len(part)) / RATE))
    return events, det


def test_bite_detector_hears_each_clip_once():
    """Each of the four bundled splashes, padded with silence, yields exactly one bite naming
    that clip, above the shipped threshold, timed inside the clip, and the "best match" the
    status line reports and the peak the Scores graph plots are that same peak. Two splashes
    well apart yield two."""
    threshold = AudioConfig().bite_threshold
    gap = np.zeros(RATE, dtype=np.float32)  # 1 s
    for name, clip in load_reference_clips(RATE).items():
        events, det = _hear(np.concatenate([gap, clip, gap]))
        assert [e.clip for e in events] == [name], name
        assert events[0].score >= threshold, (name, events[0].score)
        assert 1.0 - 0.1 <= events[0].t <= 1.0 + len(clip) / RATE, (name, events[0].t)
        best = det.best_heard
        assert (best.name, best.score) == (name, events[0].score), (name, best)
        assert det.take_peaks()[name] == events[0].score, name  # what the Scores graph plots
        assert det.take_peaks()[name] == -1.0, name  # taking the peaks clears them

    clip = load_reference_clips(RATE)[CLIP_NAMES[0]]
    apart = np.zeros(4 * RATE, dtype=np.float32)  # longer than the default refractory period
    assert len(_hear(np.concatenate([gap, clip, apart, clip, gap]))[0]) == 2


def test_bite_detector_stays_quiet_on_silence_and_noise():
    """Nothing fires on silence or broadband noise, and the scores stay well under threshold."""
    threshold = AudioConfig().bite_threshold
    rng = np.random.default_rng(0)
    silence = np.zeros(3 * RATE, dtype=np.float32)
    noise = (rng.standard_normal(3 * RATE) * 0.1).astype(np.float32)
    for stream in (silence, noise):
        det = detector_for(AudioConfig())
        events: list[BiteEvent] = []
        for i in range(0, len(stream), 2048):
            part = stream[i : i + 2048]
            events += det.feed(AudioChunk(part, RATE, (i + len(part)) / RATE))
        assert events == []
        assert max(s.score for s in det.last_scores) < threshold - 0.2


def test_icon_matcher_finds_sprite_and_rejects_water():
    """Every sprite pasted onto noisy blue water is found at the right spot, at 1080p and on a
    4K-wide frame that the matcher downscales. The same water without a sprite is rejected."""
    threshold = VisionConfig().icon_threshold
    templates = load_icon_templates()
    rng = np.random.default_rng(0)
    for frame_w, frame_h in ((1920, 1080), (3840, 2160)):
        water = np.full((frame_h, frame_w, 3), (90, 60, 30), np.uint8)  # BGR: bluish
        water += rng.integers(0, 20, water.shape, dtype=np.uint8)
        for size in ICON_SIZES:
            t = template_for_size(templates, size)
            x, y = frame_w // 3, frame_h // 3
            frame = water.copy()
            frame[y : y + size, x : x + size][t.mask > 0] = t.bgr[t.mask > 0]

            hit = IconMatcher(t).match(frame)
            assert hit.accepted(threshold), (frame_w, size, hit)
            assert abs(hit.x - (x + size / 2)) <= 2 and abs(hit.y - (y + size / 2)) <= 2

            miss = IconMatcher(t).match(water)
            assert not miss.accepted(threshold), (frame_w, size, miss)


def test_key_hold_jitter_varies_presses_within_bounds():
    """key_hold_jitter_ms varies each hold around key_hold_ms, clamped to 10..500 ms; at 0 it
    leaves the hold exact and draws nothing from the rng, so the pause sequence is unchanged."""

    class Keys:
        def __init__(self):
            self.holds = []

        def press_key(self, key, hold_s):
            self.holds.append(hold_s)

    class Rng:
        def __init__(self, *draws):
            self.draws = list(draws)

        def uniform(self, lo, hi):
            return self.draws.pop(0)

    def holds(run, rng):
        cfg = Config(GameConfig(), VisionConfig(), AudioConfig(), run)
        keys = Keys()
        engine = FishingEngine(
            cfg, None, capturer=None, input_device=keys, audio_source=None, detector=None, rng=rng
        )
        for _ in range(len(rng.draws) or 1):
            engine._press("1", "cast")
        return keys.holds

    assert holds(RunConfig(key_hold_ms=60), Rng()) == [0.06]
    assert holds(RunConfig(key_hold_ms=60, key_hold_jitter_ms=30), Rng(-30, 25)) == [0.03, 0.085]
    assert holds(RunConfig(key_hold_ms=20, key_hold_jitter_ms=50), Rng(-50)) == [0.01]
    assert holds(RunConfig(key_hold_ms=480, key_hold_jitter_ms=50), Rng(50)) == [0.5]

    for bad in (-1, 201):
        cfg = Config(GameConfig(), VisionConfig(), AudioConfig(), RunConfig(key_hold_jitter_ms=bad))
        with pytest.raises(ConfigError, match="key_hold_jitter_ms"):
            _validate(Path("config.toml"), cfg)


def test_skip_bite_chance_must_be_below_one(tmp_path):
    path = tmp_path / "config.toml"
    ensure_config_file(path)
    cfg, _ = load_config(path)
    for chance, ok in ((0.0, True), (0.3, True), (1.0, False), (-0.1, False)):
        save_config(replace(cfg, run=RunConfig(skip_bite_chance=chance)), path)
        try:
            load_config(path)
            assert ok, chance
        except ConfigError:
            assert not ok, chance


class _Keys:
    def __init__(self) -> None:
        self.pressed: list[str] = []

    def press_key(self, key: str, hold_s: float = 0.06) -> None:
        self.pressed.append(key)

    def is_game_focused(self, window: WindowRef) -> bool | None:
        return True


def _hear_bite(skip_bite_chance: float, roll: float) -> tuple[FishingEngine, _Keys]:
    rng = random.Random()
    rng.random = lambda: roll  # type: ignore[method-assign]
    keys = _Keys()
    run = RunConfig(skip_bite_chance=skip_bite_chance)
    cfg = Config(GameConfig(), VisionConfig(), AudioConfig(), run)
    engine = FishingEngine(
        cfg, WindowRef("WoW", None, 0, 0, 800, 600), capturer=None, input_device=keys,
        audio_source=None, detector=None, sleep=lambda s: None, rng=rng,
    )  # fmt: skip
    engine.state = State.WAIT_BITE
    engine._wait_bite(0.0, [BiteEvent("splash", 0.9, 0.0)])
    return engine, keys


def test_skipped_bite_counts_as_heard_and_missed_without_pressing():
    engine, keys = _hear_bite(0.5, roll=0.2)
    assert keys.pressed == []
    assert engine.state is State.IDLE
    assert (engine.stats.bites_heard, engine.stats.misses, engine.stats.catches) == (1, 1, 0)
    events = []
    while not engine.events.empty():
        events.append(engine.events.get())
    assert any(isinstance(e, ev.BiteHeard) for e in events)
    assert any(isinstance(e, ev.StateChanged) and "on purpose" in e.did for e in events)

    engine, keys = _hear_bite(0.5, roll=0.7)
    assert keys.pressed == [engine.cfg.game.interact_key]
    assert engine.state is State.INTERACT
