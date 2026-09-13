import pytest

from scadustats.models import CellColor, EventType, GameEvent, WinLine, WinType
from scadustats.rules import winner
from scadustats.rules.winner import LINES, determine_winner, majority_winner, winning_line

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
    assert winning_line(board) == (CellColor.RED, WinLine.ROW_0)
    assert determine_winner(board) == (CellColor.RED, WinType.LINE, WinLine.ROW_0)


def test_column_win():
    board = _board(
        [B, U, U, U, U],
        [B, U, U, U, U],
        [B, U, U, U, U],
        [B, U, U, U, U],
        [B, U, U, U, U],
    )
    assert winning_line(board) == (CellColor.BLUE, WinLine.COL_0)
    assert determine_winner(board) == (CellColor.BLUE, WinType.LINE, WinLine.COL_0)


def test_diagonal_win():
    board = _board(
        [R, U, U, U, U],
        [U, R, U, U, U],
        [U, U, R, U, U],
        [U, U, U, R, U],
        [U, U, U, U, R],
    )
    assert winning_line(board) == (CellColor.RED, WinLine.DIAGONAL_TL_BR)


def test_anti_diagonal_win():
    board = _board(
        [U, U, U, U, B],
        [U, U, U, B, U],
        [U, U, B, U, U],
        [U, B, U, U, U],
        [B, U, U, U, U],
    )
    assert winning_line(board) == (CellColor.BLUE, WinLine.DIAGONAL_BL_TR)


def test_inner_row_and_column_wins_name_their_own_index():
    row_3 = _board(
        [U] * 5,
        [U] * 5,
        [U] * 5,
        [R, R, R, R, R],
        [U] * 5,
    )
    assert winning_line(row_3) == (CellColor.RED, WinLine.ROW_3)

    col_2 = _board(
        [U, U, B, U, U],
        [U, U, B, U, U],
        [U, U, B, U, U],
        [U, U, B, U, U],
        [U, U, B, U, U],
    )
    assert winning_line(col_2) == (CellColor.BLUE, WinLine.COL_2)


@pytest.mark.parametrize("win_line", list(WinLine))
def test_every_win_line_is_detected_on_its_own_coordinates(win_line):
    """Guards the LINES table against a key whose coordinates don't match its name --
    the whole point of building it from the indices rather than writing it out."""
    board = _board(*([U] * 5 for _ in range(5)))
    for row, col in LINES[win_line]:
        board[row][col] = R

    assert winning_line(board) == (CellColor.RED, win_line)


def test_no_winner_mid_game():
    board = _board(
        [R, B, U, U, U],
        [U] * 5,
        [U] * 5,
        [U] * 5,
        [U] * 5,
    )
    assert winning_line(board) is None
    assert determine_winner(board) == (None, WinType.NONE, None)


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
    assert winning_line(board) is None
    assert majority_winner(board) is CellColor.RED
    # A majority win has no line to name.
    assert determine_winner(board) == (CellColor.RED, WinType.MAJORITY, None)


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
    assert winning_line(board) is None
    assert majority_winner(board) is None
    assert determine_winner(board) == (None, WinType.TIE, None)


@pytest.mark.parametrize(
    ("win_line", "expected"),
    [
        (WinLine.ROW_0, "row 0"),
        (WinLine.COL_4, "column 4"),
        (WinLine.DIAGONAL_TL_BR, "diagonal (top-left to bottom-right)"),
        (WinLine.DIAGONAL_BL_TR, "diagonal (bottom-left to top-right)"),
    ],
)
def test_win_line_labels_read_as_prose(win_line, expected):
    assert win_line.label == expected


def test_replay_applies_marks_and_unmarks_to_an_empty_board():
    events = [
        GameEvent(row=0, col=0, color=CellColor.RED, video_ts_s=1.0, game_elapsed_s=1),
        GameEvent(row=1, col=1, color=CellColor.BLUE, video_ts_s=2.0, game_elapsed_s=2),
        GameEvent(
            row=0,
            col=0,
            color=CellColor.RED,
            video_ts_s=3.0,
            game_elapsed_s=3,
            event_type=EventType.UNMARK,
        ),
    ]

    board = winner.replay(events)

    assert board[0][0] is CellColor.UNCLAIMED
    assert board[1][1] is CellColor.BLUE


def test_replay_ignores_an_event_that_is_not_about_a_square():
    """GAME_START carries no row/col (see models.GameEvent) -- replaying a game means
    walking its whole event list, so this has to be a no-op rather than an error."""
    game_start = GameEvent(
        row=None,
        col=None,
        color=None,
        video_ts_s=1.0,
        game_elapsed_s=180,
        event_type=EventType.GAME_START,
    )

    assert winner.replay([game_start]) == winner.empty_board()
