"""Covers scadustats.overlay.layout.LAYOUT_SEASON_5 -- the Season 5 (2v2) broadcast's
overlay: four gameplay feeds around a central grid, team names and commentators in a
bottom strip, and a "(B)"/"(D)"/"(C)" pool tag on every square instead of a game-type
subtitle (see scadustats/overlay/layout.py and board.cell_square_types_majority).
"""

import cv2

from scadustats.models import GameType
from scadustats.overlay import board, commentators, layout, scoreboard, timer
from scadustats.pipeline import extract

# A 5s, 3fps clip trimmed from Positive Vibes vs Bushy Stache (Season 5) at 00:30:00
# (see scripts/redact_fixture.py), with everything outside the grid column and the
# overlay's own boxes blacked out. The game clock reads 00:25:21-00:25:26; Positive
# Vibes (red) hold 5 squares and Bushy Stache (blue) 3.
_CLIP = "tests/fixtures/clip_layout_season_5.mp4"

B, D, C = GameType.BASE, GameType.DLC, GameType.COOP_MIXED
# Each square's pool, as tagged on screen.
_TYPES = [
    [D, B, C, B, B],
    [D, C, C, D, C],
    [D, D, D, C, C],
    [B, B, D, B, B],
    [B, C, B, D, D],
]


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


def test_layouts_registry_includes_season_5():
    assert layout.LAYOUTS["layout_season_5"] is layout.LAYOUT_SEASON_5
    assert layout.LAYOUT_SEASON_5.square_type_tags
    assert layout.LAYOUT_SEASON_5.team_names
    assert layout.LAYOUT_SEASON_5.game_type_box is None


def test_is_gameplay_frame_and_auto_detection():
    frames = _frames(_CLIP)
    assert frames, f"no frames decoded from {_CLIP}"
    for frame in frames:
        assert board.is_gameplay_frame(frame, layout.LAYOUT_SEASON_5)
        assert not board.is_gameplay_frame(frame, layout.STANDARD)
    assert extract._detect_layout(frames[0]) is layout.LAYOUT_SEASON_5


def test_read_timer_tracks_the_running_game_clock():
    readings = [timer.read_timer(frame, layout.LAYOUT_SEASON_5) for frame in _frames(_CLIP)]
    assert readings[0] == 1521
    assert readings[-1] == 1526
    assert readings == sorted(readings)


def test_read_scores_and_cell_colors_agree():
    frame = _frames(_CLIP)[8]
    assert scoreboard.read_scores(frame, layout.LAYOUT_SEASON_5) == (5, 3)
    colors = board.cell_colors(frame, layout.LAYOUT_SEASON_5)
    assert sum(cell.value == "red" for row in colors for cell in row) == 5
    assert sum(cell.value == "blue" for row in colors for cell in row) == 3


def test_read_team_names():
    frame = _frames(_CLIP)[8]
    assert scoreboard.read_player_names(frame, layout.LAYOUT_SEASON_5) == (
        "POSITIVE VIBES",
        "BUSHY STACHE",
    )


def test_read_commentator_names():
    """The plates read "CAPTAIN_DOMO" and "ZOODLE", but OCR drops the underscore and
    keeps the longest run -- a known gap, fixed by hand in the match file."""
    frame = _frames(_CLIP)[8]
    assert commentators.read_commentator_names(frame, layout.LAYOUT_SEASON_5) == (
        "CAPTAIN",
        "ZOODLE",
    )


def test_square_types_read_every_cell_from_its_text_color_or_tag():
    frames = _frames(_CLIP)[:5]
    texts = board.cell_square_texts_majority(frames, layout.LAYOUT_SEASON_5)
    stripped, types = board.cell_square_types_majority(frames, texts, layout.LAYOUT_SEASON_5)
    assert types == _TYPES
    # The tag is stripped from the text.
    assert stripped[0][3] == "Kill Borealis"
    assert stripped[3][4] == "Kill Malenia"
