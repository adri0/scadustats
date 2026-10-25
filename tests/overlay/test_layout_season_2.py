"""Covers scadustats.overlay.layout.LAYOUT_SEASON_2 -- the Season 2 (1v1, base game, 2023)
broadcast's overlay: both gameplay feeds side by side across the top with the timer in a
box between them, the player names and claim-count badges under the feeds, and the grid at
the bottom center between the commentators' webcams (see scadustats/overlay/layout.py).
"""

import cv2

from scadustats.overlay import board, commentators, layout, scoreboard, timer
from scadustats.pipeline import extract

# A 5s, 3fps clip trimmed from Puppery vs blueberrybrioche (Season 2) at 00:30:00 (see
# scripts/redact_fixture.py), with everything outside the grid and the overlay's own boxes
# blacked out. The game clock reads 00:26:52-00:26:57; Puppery (red) holds 2 squares and
# blueberrybrioche (blue) 4.
_CLIP = "tests/fixtures/clip_layout_season_2.mp4"


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


def test_layouts_registry_includes_season_2():
    assert layout.LAYOUTS["layout_season_2"] is layout.LAYOUT_SEASON_2
    assert not layout.LAYOUT_SEASON_2.square_type_tags
    assert not layout.LAYOUT_SEASON_2.team_names
    assert layout.LAYOUT_SEASON_2.game_type_box is None


def test_is_gameplay_frame_and_auto_detection():
    frames = _frames(_CLIP)
    assert frames, f"no frames decoded from {_CLIP}"
    for frame in frames:
        assert board.is_gameplay_frame(frame, layout.LAYOUT_SEASON_2)
        for other in layout.LAYOUTS.values():
            # V2 shares the badges, so it passes too (and is registered after).
            if other not in (layout.LAYOUT_SEASON_2, layout.LAYOUT_SEASON_2_V2):
                assert not board.is_gameplay_frame(frame, other)
    assert extract._detect_layout(frames[0]) is layout.LAYOUT_SEASON_2


def test_read_timer_tracks_the_running_game_clock():
    readings = [timer.read_timer(frame, layout.LAYOUT_SEASON_2) for frame in _frames(_CLIP)]
    assert readings[0] == 1612
    assert readings[-1] == 1617
    assert readings == sorted(readings)


def test_read_scores_and_cell_colors_agree():
    for frame in _frames(_CLIP):
        assert scoreboard.read_scores(frame, layout.LAYOUT_SEASON_2) == (2, 4)
        colors = board.cell_colors(frame, layout.LAYOUT_SEASON_2)
        assert sum(cell.value == "red" for row in colors for cell in row) == 2
        assert sum(cell.value == "blue" for row in colors for cell in row) == 4


def test_read_player_and_commentator_names():
    frame = _frames(_CLIP)[8]
    assert scoreboard.read_player_names(frame, layout.LAYOUT_SEASON_2) == (
        "Puppery",
        "blueberrybrioche",
    )
    assert commentators.read_commentator_names(frame, layout.LAYOUT_SEASON_2) == (
        "Zoodle",
        "Captain_Domo",
    )


# Season 2's second template (LAYOUT_SEASON_2_V2), used from week 2 on: the timer box moved
# up to the top edge and the names on yellow bars. A 5s, 3fps redacted clip at 00:45:00 of
# bushy vs catalystz: the game clock reads 00:41:47-00:41:51, bushy (red) holds 4 squares and
# catalystz (blue) 3.
_CLIP_V2 = "tests/fixtures/clip_layout_season_2_v2.mp4"


def test_layouts_registry_includes_season_2_v2():
    assert layout.LAYOUTS["layout_season_2_v2"] is layout.LAYOUT_SEASON_2_V2
    # Only the timer moved.
    assert layout.LAYOUT_SEASON_2_V2.timer_box.top < layout.LAYOUT_SEASON_2.timer_box.top
    assert layout.LAYOUT_SEASON_2_V2.grid_box == layout.LAYOUT_SEASON_2.grid_box
    assert layout.LAYOUT_SEASON_2_V2.score_box_red == layout.LAYOUT_SEASON_2.score_box_red


def test_v2_reads_the_timer_and_scores_and_the_board_agrees():
    frames = _frames(_CLIP_V2)
    assert frames, f"no frames decoded from {_CLIP_V2}"
    readings = [timer.read_timer(frame, layout.LAYOUT_SEASON_2_V2) for frame in frames]
    assert readings[0] == 2507
    assert readings[-1] == 2511
    assert readings == sorted(readings)
    for frame in frames:
        # Season 2's own timer box reads nothing here.
        assert timer.read_timer(frame, layout.LAYOUT_SEASON_2) is None
        assert board.is_gameplay_frame(frame, layout.LAYOUT_SEASON_2_V2)
        assert scoreboard.read_scores(frame, layout.LAYOUT_SEASON_2_V2) == (4, 3)
        colors = board.cell_colors(frame, layout.LAYOUT_SEASON_2_V2)
        assert sum(cell.value == "red" for row in colors for cell in row) == 4
        assert sum(cell.value == "blue" for row in colors for cell in row) == 3


def test_v2_reads_blue_name_and_commentators():
    frame = _frames(_CLIP_V2)[8]
    # Red's dark name on a yellow bar isn't read; blue's is.
    assert scoreboard.read_player_names(frame, layout.LAYOUT_SEASON_2_V2)[1] == "catalystz"
    assert commentators.read_commentator_names(frame, layout.LAYOUT_SEASON_2_V2) == (
        "starOchris",
        "Captain_Domo",
    )
