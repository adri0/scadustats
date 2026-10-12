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


def test_split_square_type_tag_reads_each_pool_and_its_common_misreads():
    from scadustats.models import GameType
    from scadustats.overlay.board import split_square_type_tag

    assert split_square_type_tag("Kill Malenia (B)") == ("Kill Malenia", GameType.BASE)
    assert split_square_type_tag("Rune Level 60 (8)") == ("Rune Level 60", GameType.BASE)
    assert split_square_type_tag("Acquire Moore Armor set (D)") == (
        "Acquire Moore Armor set",
        GameType.DLC,
    )
    assert split_square_type_tag("Acquire 3 DLC Talismans (0)") == (
        "Acquire 3 DLC Talismans",
        GameType.DLC,
    )
    assert split_square_type_tag("Kill Fia's Champs and Leda (C)") == (
        "Kill Fia's Champs and Leda",
        GameType.COOP_MIXED,
    )


def test_split_square_type_tag_strips_an_unreadable_tag_and_keeps_a_longer_parenthetical():
    from scadustats.overlay.board import split_square_type_tag

    assert split_square_type_tag("Kill Ancient Dragon Senessax ()") == (
        "Kill Ancient Dragon Senessax",
        None,
    )
    # Not a tag: a real square's own parenthetical.
    assert split_square_type_tag("30 Faith (no rebirth or buffs)") == (
        "30 Faith (no rebirth or buffs)",
        None,
    )
    assert split_square_type_tag("Kill Malenia") == ("Kill Malenia", None)


def test_text_color_type_tells_the_three_pools_apart():
    import numpy as np

    from scadustats.models import GameType
    from scadustats.overlay.board import _text_color_type

    def cell(bgr: tuple[int, int, int]) -> np.ndarray:
        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        crop[5:15, 5:15] = bgr  # "text" on a black cell
        return crop

    assert _text_color_type(cell((110, 100, 160))) is GameType.BASE
    assert _text_color_type(cell((130, 175, 175))) is GameType.DLC
    assert _text_color_type(cell((160, 165, 120))) is GameType.COOP_MIXED
    assert _text_color_type(cell((200, 200, 200))) is None  # white: no pool says
    assert _text_color_type(np.zeros((20, 20, 3), dtype=np.uint8)) is None


def test_majority_text_ignores_empty_readings():
    from scadustats.overlay.board import _majority_text

    assert _majority_text(["", "", "Kill Malenia", ""]) == "Kill Malenia"
    assert _majority_text(["", ""]) == ""
