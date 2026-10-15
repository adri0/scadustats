"""Covers scadustats.overlay.layout.LAYOUT_SEASON_4 -- the Season 4 (1v1, DLC only)
broadcast's overlay: both gameplay feeds across the top with the timer between them, a
strip of claim-count badges and player names under them, and the grid at the bottom
center between the commentators' webcams (see scadustats/overlay/layout.py).
"""

import cv2

from scadustats.overlay import board, commentators, layout, scoreboard, timer
from scadustats.pipeline import extract

# A 5s, 3fps clip trimmed from parkenharbor vs GrandPooBear (Season 4) at 01:20:00 (see
# scripts/redact_fixture.py), with everything outside the grid column and the overlay's
# own boxes blacked out -- and the gameplay feeds above the grid too, since that part of
# the column holds a player's webcam. The game clock reads 01:17:03-01:17:08;
# parkenharbor (red) holds 9 squares and GrandPooBear (blue) 6.
_CLIP = "tests/fixtures/clip_layout_season_4.mp4"


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


def test_layouts_registry_includes_season_4():
    assert layout.LAYOUTS["layout_season_4"] is layout.LAYOUT_SEASON_4
    assert not layout.LAYOUT_SEASON_4.square_type_tags
    assert not layout.LAYOUT_SEASON_4.team_names
    assert layout.LAYOUT_SEASON_4.game_type_box is None


def test_is_gameplay_frame_and_auto_detection():
    frames = _frames(_CLIP)
    assert frames, f"no frames decoded from {_CLIP}"
    for frame in frames:
        assert board.is_gameplay_frame(frame, layout.LAYOUT_SEASON_4)
        for other in layout.LAYOUTS.values():
            if other is not layout.LAYOUT_SEASON_4:
                assert not board.is_gameplay_frame(frame, other)
    assert extract._detect_layout(frames[0]) is layout.LAYOUT_SEASON_4


def test_read_timer_tracks_the_running_game_clock():
    readings = [timer.read_timer(frame, layout.LAYOUT_SEASON_4) for frame in _frames(_CLIP)]
    assert readings[0] == 4623
    assert readings[-1] == 4628
    assert readings == sorted(readings)


def test_read_scores_and_cell_colors_agree():
    for frame in _frames(_CLIP):
        assert scoreboard.read_scores(frame, layout.LAYOUT_SEASON_4) == (9, 6)
        colors = board.cell_colors(frame, layout.LAYOUT_SEASON_4)
        assert sum(cell.value == "red" for row in colors for cell in row) == 9
        assert sum(cell.value == "blue" for row in colors for cell in row) == 6


def test_read_player_and_commentator_names():
    frame = _frames(_CLIP)[8]
    assert scoreboard.read_player_names(frame, layout.LAYOUT_SEASON_4) == (
        "parkenharbor",
        "GrandPooBear",
    )
    assert commentators.read_commentator_names(frame, layout.LAYOUT_SEASON_4) == (
        "YoJosherino",
        "Captain_Domo",
    )
