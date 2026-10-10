"""Covers scadustats.overlay.layout.LAYOUT_PRE_SEASON_6 -- the pre-season 6 broadcast's
overlay template: STANDARD's, except that the timer sits higher and there is no
"BASE GAME"/"DLC" subtitle at all (see scadustats/overlay/layout.py).
"""

import cv2

from scadustats.overlay import board, commentators, game_type_label, layout, scoreboard, timer

# A 5s, 3fps clip trimmed from a real pre-season 6 VOD at 00:50:00 (see
# scripts/redact_fixture.py for how it was produced), with the webcam/face regions
# blacked out. Covers one already-in-progress base game between star0chris (red) and
# itzCBD (blue), at 00:47:06 on the game clock.
_CLIP = "tests/fixtures/clip_layout_pre_season_6.mp4"


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


def test_layouts_registry_includes_pre_season_6():
    assert layout.LAYOUTS["layout_pre_season_6"] is layout.LAYOUT_PRE_SEASON_6


def test_only_the_timer_and_game_type_boxes_differ_from_standard():
    for field in ("grid_box", "score_box_red", "score_box_blue", "name_box_red", "name_box_blue"):
        assert getattr(layout.LAYOUT_PRE_SEASON_6, field) == getattr(layout.STANDARD, field)
    assert layout.LAYOUT_PRE_SEASON_6.timer_box != layout.STANDARD.timer_box
    assert layout.LAYOUT_PRE_SEASON_6.game_type_box is None


def test_is_gameplay_frame_also_matches_standard():
    """The score strip is STANDARD's, so auto-detection can't tell the two apart --
    which is why a pre-season 6 VOD needs `--layout layout_pre_season_6`."""
    frames = _frames(_CLIP)
    assert frames, f"no frames decoded from {_CLIP}"
    for frame in frames:
        assert board.is_gameplay_frame(frame, layout.LAYOUT_PRE_SEASON_6)
        assert board.is_gameplay_frame(frame, layout.STANDARD)


def test_read_timer_tracks_the_running_game_clock():
    frames = _frames(_CLIP)
    readings = [timer.read_timer(frame, layout.LAYOUT_PRE_SEASON_6) for frame in frames]
    assert all(reading is not None for reading in readings)
    assert readings[0] == 2826
    assert readings[-1] == 2831
    assert readings == sorted(readings)


def test_standard_timer_box_misses_this_template():
    """The regression this layout exists for: STANDARD's timer box reads nothing here."""
    frames = _frames(_CLIP)
    assert all(timer.read_timer(frame, layout.STANDARD) is None for frame in frames)


def test_read_game_type_label_is_none_without_a_subtitle_box():
    frames = _frames(_CLIP)
    assert game_type_label.majority_game_type_label(frames, layout.LAYOUT_PRE_SEASON_6) is None


def test_read_player_names():
    frame = _frames(_CLIP)[len(_frames(_CLIP)) // 2]
    red, blue = scoreboard.read_player_names(frame, layout.LAYOUT_PRE_SEASON_6)
    # The handle is star0chris (a zero), but OCR can't tell 0 from O in this font.
    assert red.lower() == "starochris"
    assert blue.lower() == "itzcbd"


def test_read_commentator_names():
    frame = _frames(_CLIP)[len(_frames(_CLIP)) // 2]
    assert commentators.read_commentator_names(frame, layout.LAYOUT_PRE_SEASON_6) == (
        "Zoodle",
        "Captain_Domo",
    )
