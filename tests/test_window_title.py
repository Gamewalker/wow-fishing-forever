"""The game window is found by its title or, for any client language, by its executable."""

from __future__ import annotations

from wowfish.capture import window_matches


def test_default_finds_game_by_title_or_executable():
    assert window_matches("World of Warcraft", "")
    assert window_matches("魔兽世界", "Wow.exe")  # Chinese client: title has no English
    assert window_matches("魔兽世界", "wowb.exe")
    assert not window_matches("魔兽世界", "")
    assert not window_matches("wowfish 1.0.1", "wowfish-gui.exe")


def test_custom_title_ignores_executable():
    assert window_matches("My WoW Window", "", "my wow")
    assert not window_matches("魔兽世界", "Wow.exe", "my wow")
