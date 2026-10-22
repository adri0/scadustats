"""Covers scadustats.overlay.layout.LAYOUT_SEASON_3 -- the Season 3 (2v2, base game)
broadcast, which cuts between a team view (grid at the bottom center) and a one-player
focus view (grid on the right), neither with claim counts or names on screen, so each
scene is recognized by its timer reading (see scadustats/overlay/layout.py).
"""

import cv2

from scadustats.overlay import board, commentators, layout, scoreboard, timer
from scadustats.pipeline import extract

# Two 5s, 3fps clips trimmed from Jo W Bush vs Monkey Ballers (Season 3) with
# scripts/redact_fixture.py, each cut with its own scene's layout so only that scene's
# grid, timer and nameplates are kept. The team view at video 00:45:00 reads 00:40:33-
# 00:40:38 with red 9, blue 7; the focus view at video 00:30:00 reads 00:25:33-00:25:38
# with red 6, blue 5.
_TEAM_CLIP = "tests/fixtures/clip_layout_season_3_team.mp4"
_FOCUS_CLIP = "tests/fixtures/clip_layout_season_3_focus.mp4"


def _frames(path: str) -> list:
    cap = cv2.VideoCapture(path)
    try:
        result = []
        while True:
            ok, frame = cap.read()
            if not ok:
                return result
            result.append(frame)
    finally:
        cap.release()


def _count(colors, value: str) -> int:
    return sum(cell.value == value for row in colors for cell in row)


def test_layouts_registry_includes_season_3_and_its_scenes():
    assert layout.LAYOUTS["layout_season_3"] is layout.LAYOUT_SEASON_3
    assert layout.LAYOUT_SEASON_3.scenes == (
        layout.LAYOUT_SEASON_3_FOCUS,
        layout.LAYOUT_SEASON_3_TEAM,
    )
    for scene in layout.LAYOUT_SEASON_3.scenes:
        assert layout.LAYOUTS[scene.name] is scene
        assert scene.score_box_red is None and scene.name_box_red is None
        assert scene.game_type_box is None
        assert not scene.scenes


def test_scene_for_picks_the_scene_on_screen():
    for path, scene in (
        (_TEAM_CLIP, layout.LAYOUT_SEASON_3_TEAM),
        (_FOCUS_CLIP, layout.LAYOUT_SEASON_3_FOCUS),
    ):
        frames = _frames(path)
        assert frames, f"no frames decoded from {path}"
        for frame in frames:
            assert board.scene_for(frame, layout.LAYOUT_SEASON_3) is scene
            assert board.is_gameplay_frame(frame, layout.LAYOUT_SEASON_3)
        assert extract._detect_layout(frames[0]) is layout.LAYOUT_SEASON_3


def test_other_layouts_reject_season_3_frames():
    for path in (_TEAM_CLIP, _FOCUS_CLIP):
        for frame in _frames(path):
            for other in layout.LAYOUTS.values():
                if other.name.value.startswith("layout_season_3"):
                    continue
                assert not board.is_gameplay_frame(frame, other)


def test_scene_for_without_scenes_is_the_layout_itself_or_none():
    frame = _frames(_TEAM_CLIP)[0]
    assert board.scene_for(frame, layout.LAYOUT_SEASON_3_TEAM) is layout.LAYOUT_SEASON_3_TEAM
    assert board.scene_for(frame, layout.LAYOUT_SEASON_3_FOCUS) is None


def test_read_timer_tracks_the_running_game_clock():
    for path, scene, first, last in (
        (_TEAM_CLIP, layout.LAYOUT_SEASON_3_TEAM, 2433, 2438),
        (_FOCUS_CLIP, layout.LAYOUT_SEASON_3_FOCUS, 1533, 1538),
    ):
        readings = [timer.read_timer(frame, scene) for frame in _frames(path)]
        assert readings[0] == first
        assert readings[-1] == last
        assert readings == sorted(readings)


def test_cell_colors_in_each_scene():
    for path, scene, red, blue in (
        (_TEAM_CLIP, layout.LAYOUT_SEASON_3_TEAM, 9, 7),
        (_FOCUS_CLIP, layout.LAYOUT_SEASON_3_FOCUS, 6, 5),
    ):
        for frame in _frames(path):
            colors = board.cell_colors(frame, scene)
            assert (_count(colors, "red"), _count(colors, "blue")) == (red, blue)


def test_no_scores_or_names_but_commentators():
    for path, scene in (
        (_TEAM_CLIP, layout.LAYOUT_SEASON_3_TEAM),
        (_FOCUS_CLIP, layout.LAYOUT_SEASON_3_FOCUS),
    ):
        frame = _frames(path)[8]
        assert scoreboard.read_scores(frame, scene) == (None, None)
        assert scoreboard.read_player_names(frame, scene) == ("", "")
        assert commentators.read_commentator_names(frame, scene) == ("blanxz", "Captain_Domo")
