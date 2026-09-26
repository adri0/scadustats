"""Covers scadustats.overlay.layout.LAYOUT_SEASON_6_FINAL -- the Season 6 finals
broadcast's overlay template, visually unrelated to the standard one (see
scadustats/overlay/layout.py): the grid moves to the bottom-right quadrant, the timer/
player-names/live-claim-counts move to a bottom strip, and there are no commentator
nameplates at all.
"""

import cv2
import pytest

from scadustats.models import CellColor, GameType
from scadustats.overlay import board, commentators, game_type_label, layout, scoreboard, timer

# A 5s, 3fps clip trimmed from a real Season 6 finals VOD at 00:41:32 (see
# scripts/redact_fixture.py for how it was produced), with the webcam/face regions
# blacked out. Covers one already-in-progress base game between NUCLEARPASTATOM (red)
# and SERIOUSCHALLENGES (blue).
_CLIP = "tests/fixtures/clip_layout_season_6_final.mp4"

U, R, B = CellColor.UNCLAIMED, CellColor.RED, CellColor.BLUE


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


def test_layouts_registry_has_both_templates():
    assert set(layout.LAYOUTS) == {"standard", "layout_season_6_final"}
    assert layout.LAYOUTS["layout_season_6_final"] is layout.LAYOUT_SEASON_6_FINAL


def test_layout_season_6_final_has_no_commentator_nameplates():
    """Unlike the standard template, this broadcast never prints a commentator
    nameplate at all -- see Layout.commentator_box_left/right's docstring."""
    assert layout.LAYOUT_SEASON_6_FINAL.commentator_box_left is None
    assert layout.LAYOUT_SEASON_6_FINAL.commentator_box_right is None


def test_is_gameplay_frame_matches_only_this_layout():
    """Every frame of this clip should satisfy is_gameplay_frame under
    LAYOUT_SEASON_6_FINAL and under no other registered layout -- this is exactly the
    signal pipeline.extract._detect_layout auto-detects against."""
    frames = _frames(_CLIP)
    assert frames, f"no frames decoded from {_CLIP}"
    for frame in frames:
        assert board.is_gameplay_frame(frame, layout.LAYOUT_SEASON_6_FINAL)
        assert not board.is_gameplay_frame(frame, layout.STANDARD)


def test_read_timer_tracks_the_running_game_clock():
    """The clip covers a few seconds of an already-running game, so the reading should
    climb by roughly a second per sampled frame, not jump or reset."""
    frames = _frames(_CLIP)
    readings = [timer.read_timer(frame, layout.LAYOUT_SEASON_6_FINAL) for frame in frames]
    assert all(reading is not None for reading in readings)
    assert readings[0] == 1478
    assert readings[-1] == 1483
    assert readings == sorted(readings)


def test_read_game_type_label_reads_every_frame():
    frames = _frames(_CLIP)
    for frame in frames:
        reading = game_type_label.read_game_type_label(frame, layout.LAYOUT_SEASON_6_FINAL)
        assert reading is GameType.BASE


def test_read_player_names():
    frames = _frames(_CLIP)
    # A representative mid-clip frame, the same way extract_video itself only reads
    # player names off one frame per video (see its docstring) -- not majority-voted, so
    # this is exactly the single-frame read production code relies on.
    red, blue = scoreboard.read_player_names(frames[2], layout.LAYOUT_SEASON_6_FINAL)
    assert (red, blue) == ("NUCLEARPASTATOM", "SERIOUSCHALLENGES")


def test_read_scores_matches_the_live_claim_counts():
    frames = _frames(_CLIP)
    for frame in frames:
        assert scoreboard.read_scores(frame, layout.LAYOUT_SEASON_6_FINAL) == (4, 4)


def test_read_commentator_names_short_circuits_to_none():
    """No nameplate boxes are calibrated for this layout at all, so this should never
    even attempt an OCR call -- see commentators.read_commentator_names."""
    frames = _frames(_CLIP)
    for frame in frames:
        assert commentators.read_commentator_names(frame, layout.LAYOUT_SEASON_6_FINAL) == (
            None,
            None,
        )


def test_majority_commentator_names_is_empty():
    assert (
        commentators.majority_commentator_names(_frames(_CLIP), layout.LAYOUT_SEASON_6_FINAL) == []
    )


@pytest.mark.parametrize("frame_index", [0, 2, 8, 15])
def test_cell_colors_match_the_known_board_state(frame_index):
    """The board doesn't change across this short clip -- 8 squares already claimed
    before the clip starts, matching the events a full extract_video run over this same
    fixture records (see tests/pipeline/test_extract_integration.py)."""
    frames = _frames(_CLIP)
    expected = [
        [B, U, U, U, U],
        [U, U, U, U, R],
        [R, U, U, B, B],
        [U, U, U, U, R],
        [U, U, B, U, R],
    ]
    assert board.cell_colors(frames[frame_index], layout.LAYOUT_SEASON_6_FINAL) == expected
