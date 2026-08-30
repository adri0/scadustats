from scadustats.models import CellColor, WinType
from scadustats.winner import check_line_winner, determine_winner, majority_winner

U, R, B = CellColor.UNCLAIMED, CellColor.RED, CellColor.BLUE


def _board(*rows: list[CellColor]) -> list[list[CellColor]]:
    assert len(rows) == 5 and all(len(row) == 5 for row in rows)
    return list(rows)


def test_row_win():
    board = _board(
        [R, R, R, R, R],
        [U] * 5,
        [U] * 5,
        [U] * 5,
        [U] * 5,
    )
    assert check_line_winner(board) is CellColor.RED
    assert determine_winner(board) == (CellColor.RED, WinType.LINE)


def test_column_win():
    board = _board(
        [B, U, U, U, U],
        [B, U, U, U, U],
        [B, U, U, U, U],
        [B, U, U, U, U],
        [B, U, U, U, U],
    )
    assert check_line_winner(board) is CellColor.BLUE
    assert determine_winner(board) == (CellColor.BLUE, WinType.LINE)


def test_diagonal_win():
    board = _board(
        [R, U, U, U, U],
        [U, R, U, U, U],
        [U, U, R, U, U],
        [U, U, U, R, U],
        [U, U, U, U, R],
    )
    assert check_line_winner(board) is CellColor.RED


def test_anti_diagonal_win():
    board = _board(
        [U, U, U, U, B],
        [U, U, U, B, U],
        [U, U, B, U, U],
        [U, B, U, U, U],
        [B, U, U, U, U],
    )
    assert check_line_winner(board) is CellColor.BLUE


def test_no_winner_mid_game():
    board = _board(
        [R, B, U, U, U],
        [U] * 5,
        [U] * 5,
        [U] * 5,
        [U] * 5,
    )
    assert check_line_winner(board) is None
    assert determine_winner(board) == (None, WinType.NONE)


def test_majority_winner_when_all_lines_blocked():
    # A red cell plus one blue permutation cell (touching every row/column, and the
    # center cell covering both diagonals) in every line, so no line can still be
    # completed by one color; red holds more squares overall (20 vs 5).
    board = _board(
        [R, B, R, R, R],
        [B, R, R, R, R],
        [R, R, B, R, R],
        [R, R, R, R, B],
        [R, R, R, B, R],
    )
    assert check_line_winner(board) is None
    assert majority_winner(board) is CellColor.RED
    assert determine_winner(board) == (CellColor.RED, WinType.MAJORITY)


def test_tie_when_blocked_and_equal():
    # Every line still has both colors present (blocked), with an equal 12/12 split
    # and one square left unclaimed (25 is odd, so an exact split needs one gap).
    board = _board(
        [B, B, U, R, R],
        [B, R, R, R, R],
        [R, R, B, B, B],
        [R, R, B, R, B],
        [B, B, B, B, R],
    )
    assert check_line_winner(board) is None
    assert majority_winner(board) is None
    assert determine_winner(board) == (None, WinType.TIE)
