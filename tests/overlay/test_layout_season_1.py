"""Covers scadustats.overlay.layout.LAYOUT_SEASON_1 -- the Season 1 (1v1, base game, 2023)
broadcast's overlay: an unboxed H:MM:SS clock at the top, and each player's name printed in the
colour they claim in, which changes from match to match (see Layout.player_colors and
models.ClaimColor). It cuts between two scenes: the main one, both feeds side by side with the
grid at the bottom center over them, and a focus one, one feed full screen with the grid at the
top right corner.
"""

import cv2
import numpy as np

from scadustats.models import CellColor, ClaimColor
from scadustats.overlay import board, layout, scoreboard, timer
from scadustats.pipeline import extract

# A 5s, 3fps clip trimmed from Ainrun vs Mitchriz (Season 1) at 01:23:20 (see
# scripts/redact_fixture.py), with everything outside the grid and the overlay's own boxes
# blacked out. The game clock reads 1:01:45-1:01:50; Ainrun (left, so the red side) claims in
# blue and holds 8 squares, Mitchriz (blue side) claims in green and holds 5.
_CLIP = "tests/fixtures/clip_layout_season_1.mp4"


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


def _count(colors, color: CellColor) -> int:
    return sum(cell is color for row in colors for cell in row)


def test_layouts_registry_includes_season_1():
    assert layout.LAYOUTS["layout_season_1"] is layout.LAYOUT_SEASON_1
    assert layout.LAYOUT_SEASON_1.scenes == (
        layout.LAYOUT_SEASON_1_MAIN,
        layout.LAYOUT_SEASON_1_FOCUS,
    )
    assert layout.LAYOUT_SEASON_1_MAIN.score_box_red is None
    # No other layout reads its players' own colours.
    assert {lay.name for lay in layout.LAYOUTS.values() if lay.player_colors} == {
        "layout_season_1",
        "layout_season_1_main",
        "layout_season_1_focus",
    }


def test_scene_for_tells_the_main_scene_by_its_wordmark():
    for frame in _frames(_CLIP):
        assert board.scene_for(frame, layout.LAYOUT_SEASON_1) is layout.LAYOUT_SEASON_1_MAIN


def test_auto_detection_never_picks_season_1():
    # Only --layout picks it (see extract._detect_layout), even on its own frames.
    for frame in [*_frames(_CLIP), cv2.imread(_FOCUS)]:
        detected = extract._detect_layout(frame)
        assert detected is None or not detected.player_colors


def test_read_timer_reads_a_one_digit_hour():
    frames = _frames(_CLIP)
    assert frames, f"no frames decoded from {_CLIP}"
    readings = [timer.read_timer(frame, layout.LAYOUT_SEASON_1_MAIN) for frame in frames]
    assert readings[0] == 3705
    assert readings[-1] == 3710
    assert readings == sorted(readings)
    assert all(board.is_gameplay_frame(frame, layout.LAYOUT_SEASON_1) for frame in frames)


def test_read_claim_colors_off_the_names():
    for frame in _frames(_CLIP):
        assert board.read_claim_colors(frame, layout.LAYOUT_SEASON_1_MAIN) == (
            ClaimColor.BLUE,
            ClaimColor.GREEN,
        )


def test_cell_colors_against_the_claim_colors():
    for frame in _frames(_CLIP):
        colors = board.cell_colors(
            frame, layout.LAYOUT_SEASON_1_MAIN, (ClaimColor.BLUE, ClaimColor.GREEN)
        )
        assert _count(colors, CellColor.RED) == 8
        assert _count(colors, CellColor.BLUE) == 5
        # The usual red/blue reading takes the blue squares for the blue side and misses
        # the green ones altogether.
        usual = board.cell_colors(frame, layout.LAYOUT_SEASON_1_MAIN)
        assert _count(usual, CellColor.RED) == 0
        assert _count(usual, CellColor.BLUE) == 8


def test_read_player_names():
    frame = _frames(_CLIP)[8]
    assert scoreboard.read_player_names(frame, layout.LAYOUT_SEASON_1_MAIN) == (
        "Ainrun",
        "Mitchriz",
    )


# The focus scene, from the same match at 00:58:20 (redacted, see scripts/redact_fixture.py):
# the game clock reads 0:36:45; Ainrun (blue) holds 4 squares and Mitchriz (green) 1.
_FOCUS = "tests/fixtures/hud_season_1_focus.png"


def test_focus_scene_reads_the_board_in_the_corner():
    frame = cv2.imread(_FOCUS)
    assert board.scene_for(frame, layout.LAYOUT_SEASON_1) is layout.LAYOUT_SEASON_1_FOCUS
    assert timer.read_timer(frame, layout.LAYOUT_SEASON_1_FOCUS) == 36 * 60 + 45
    colors = board.cell_colors(
        frame, layout.LAYOUT_SEASON_1_FOCUS, (ClaimColor.BLUE, ClaimColor.GREEN)
    )
    assert _count(colors, CellColor.RED) == 4
    assert _count(colors, CellColor.BLUE) == 1
    # It prints only one name, so the colours aren't read off it.
    assert board.read_claim_colors(frame, layout.LAYOUT_SEASON_1_FOCUS) is None


def test_collect_observations_settles_the_colors_then_reads_the_board():
    observations, selected, _, claim_colors = extract._collect_observations(
        _CLIP, 3.0, forced_layout=layout.LAYOUT_SEASON_1
    )
    assert selected is layout.LAYOUT_SEASON_1
    assert claim_colors == (ClaimColor.BLUE, ClaimColor.GREEN)
    # The first samples only settle the colours (see _CLAIM_COLOR_CONFIRM_SAMPLES).
    assert len(observations) == len(_frames(_CLIP)) - extract._CLAIM_COLOR_CONFIRM_SAMPLES + 1
    for observation in observations:
        assert _count(observation.board, CellColor.RED) == 8
        assert _count(observation.board, CellColor.BLUE) == 5


def test_collect_observations_takes_given_colors():
    observations, _, _, claim_colors = extract._collect_observations(
        _CLIP,
        3.0,
        forced_layout=layout.LAYOUT_SEASON_1,
        claim_colors=(ClaimColor.BLUE, ClaimColor.RED),
    )
    assert claim_colors == (ClaimColor.BLUE, ClaimColor.RED)
    assert len(observations) == len(_frames(_CLIP))
    # Read against the wrong pair, the green squares are nobody's.
    assert _count(observations[0].board, CellColor.BLUE) == 0


def _patch(bgr: tuple[int, int, int]) -> np.ndarray:
    return np.full((8, 8, 3), bgr, dtype=np.uint8)


def test_classify_patch_as_takes_the_nearer_of_the_pair():
    pair = (ClaimColor.RED, ClaimColor.ORANGE)  # the closest pair, 81 apart
    assert board.classify_patch_as(_patch((52, 55, 209)), pair) is CellColor.RED
    assert board.classify_patch_as(_patch((34, 132, 224)), pair) is CellColor.BLUE
    # Unclaimed: the dark grey background, or a colour the match doesn't use.
    assert board.classify_patch_as(_patch((79, 75, 76)), pair) is CellColor.UNCLAIMED
    assert board.classify_patch_as(_patch((201, 147, 35)), pair) is CellColor.UNCLAIMED
