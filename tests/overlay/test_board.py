import cv2

from scadustats.overlay.board import CellColor, _sanitize_square_text, cell_colors

_FIXTURES = "tests/fixtures"


def test_sanitize_square_text_strips_line_breaks_and_punctuation():
    assert _sanitize_square_text("Deal 100\ndamage w/ a\n\nbow!") == "Deal 100 damage w/ a bow"


def test_sanitize_square_text_keeps_apostrophes():
    assert _sanitize_square_text("Defeat Rennala's\nknight") == "Defeat Rennala's knight"


def test_sanitize_square_text_collapses_whitespace():
    assert _sanitize_square_text("  Kill   3  enemies  ") == "Kill 3 enemies"


def test_cell_colors_all_unclaimed():
    frame = cv2.imread(f"{_FIXTURES}/hud_start.png")
    colors = cell_colors(frame)
    assert all(c is CellColor.UNCLAIMED for row in colors for c in row)


def test_cell_colors_partial_claims():
    frame = cv2.imread(f"{_FIXTURES}/hud_partial.png")
    colors = cell_colors(frame)

    red = sum(c is CellColor.RED for row in colors for c in row)
    blue = sum(c is CellColor.BLUE for row in colors for c in row)
    assert (red, blue) == (3, 4)

    assert colors[0][3] is CellColor.BLUE
    assert colors[0][4] is CellColor.RED
    assert colors[1][0] is CellColor.UNCLAIMED
