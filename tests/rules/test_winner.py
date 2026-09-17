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
    assert winning_line(board) == (CellColor.RED, WinLine.ROW_1)
    assert determine_winner(board) == (CellColor.RED, WinType.LINE, WinLine.ROW_1)


def test_column_win():
    board = _board(
        [B, U, U, U, U],
        [B, U, U, U, U],
        [B, U, U, U, U],
        [B, U, U, U, U],
        [B, U, U, U, U],
    )
    assert winning_line(board) == (CellColor.BLUE, WinLine.COL_1)
    assert determine_winner(board) == (CellColor.BLUE, WinType.LINE, WinLine.COL_1)


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
    row_4 = _board(
        [U] * 5,
        [U] * 5,
        [U] * 5,
        [R, R, R, R, R],
        [U] * 5,
    )
    assert winning_line(row_4) == (CellColor.RED, WinLine.ROW_4)

    col_3 = _board(
        [U, U, B, U, U],
        [U, U, B, U, U],
        [U, U, B, U, U],
        [U, U, B, U, U],
        [U, U, B, U, U],
    )
    assert winning_line(col_3) == (CellColor.BLUE, WinLine.COL_3)


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


def test_majority_winner_when_unbeatable():
    # Red holds 20 of 25 squares -- blue couldn't catch up even by claiming every one of
    # the remaining 5, so the majority is already settled.
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


def test_no_winner_when_a_line_is_blocked_but_the_majority_could_still_flip():
    # Every line already holds both colors, but the 12/12 split still has one square
    # unclaimed (25 is odd, so an exact split needs one gap) -- whichever color claims
    # it takes an unbeatable 13/12 majority, so the result isn't settled yet even though
    # no line can still be completed by either color. Blocking every line used to be
    # treated as "the majority is decided" (see winner._majority_settled), which is a
    # different -- and looser -- question than whether the count itself is unbeatable.
    board = _board(
        [B, B, U, R, R],
        [B, R, R, R, R],
        [R, R, B, B, B],
        [R, R, B, R, B],
        [B, B, B, B, R],
    )
    assert winning_line(board) is None
    assert majority_winner(board) is None
    assert determine_winner(board) == (None, WinType.NONE, None)


def test_majority_settles_the_instant_it_becomes_unbeatable_even_with_an_open_line():
    # Row 4 is blue-or-unclaimed -- still structurally completable by blue -- but blue
    # already holds 13 of 25 squares, an unbeatable majority no matter what happens to
    # the remaining unclaimed cells (including that row). This is the real-match case
    # (2026-07-25-TwistieT-vs-Grey) that motivated dropping the "every line blocked"
    # gate: a goal square nobody ever attempted stays unclaimed for the rest of the
    # game, so a line resting on it looks permanently "open" even though the game is
    # over and majority has already been won.
    board = _board(
        [R, B, U, B, R],
        [R, R, U, B, B],
        [U, U, B, R, B],
        [B, B, B, U, B],
        [U, B, R, B, B],
    )
    assert winning_line(board) is None
    assert majority_winner(board) is CellColor.BLUE
    assert determine_winner(board) == (CellColor.BLUE, WinType.MAJORITY, None)


@pytest.mark.parametrize(
    ("win_line", "expected"),
    [
        (WinLine.ROW_1, "row 1"),
        (WinLine.COL_4, "column 4"),
        (WinLine.DIAGONAL_TL_BR, "diagonal (top-left to bottom-right)"),
        (WinLine.DIAGONAL_BL_TR, "diagonal (bottom-left to top-right)"),
    ],
)
def test_win_line_labels_read_as_prose(win_line, expected):
    assert win_line.label == expected


def test_replay_applies_marks_and_unmarks_to_an_empty_board():
    events = [
        GameEvent(row=1, col=1, color=CellColor.RED, video_ts_s=1.0, game_elapsed_s=1),
        GameEvent(row=2, col=2, color=CellColor.BLUE, video_ts_s=2.0, game_elapsed_s=2),
        GameEvent(
            row=1,
            col=1,
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


def _mark(row: int, col: int, color: CellColor, ts: float) -> GameEvent:
    """row/col here are 0-based board positions, for readability at call sites --
    converted to the 1-based GameEvent.row/col at construction."""
    return GameEvent(row=row + 1, col=col + 1, color=color, video_ts_s=ts, game_elapsed_s=int(ts))


def _unmark(row: int, col: int, color: CellColor, ts: float) -> GameEvent:
    return GameEvent(
        row=row + 1,
        col=col + 1,
        color=color,
        video_ts_s=ts,
        game_elapsed_s=int(ts),
        event_type=EventType.UNMARK,
    )


def test_board_states_keeps_every_intermediate_board():
    events = [_mark(0, 0, R, 1.0), _mark(1, 1, B, 2.0)]

    states = winner.board_states(events)

    assert len(states) == 2
    # After the first event only (0, 0) has changed.
    assert states[0][0][0] is R
    assert states[0][1][1] is U
    # The second event doesn't undo the first.
    assert states[1][0][0] is R
    assert states[1][1][1] is B


def test_settled_result_index_finds_the_mark_that_completes_the_line():
    # Row 1 filled left to right, then an unrelated mark elsewhere that doesn't affect it.
    events = [_mark(0, c, R, float(c)) for c in range(5)] + [_mark(1, 0, B, 10.0)]

    states = winner.board_states(events)

    assert winner.settled_result_index(states, R, WinType.LINE) == 4


def test_settled_result_index_ignores_a_result_undone_before_the_end():
    events = [_mark(0, c, R, float(c)) for c in range(5)] + [_unmark(0, 4, R, 10.0)]

    states = winner.board_states(events)

    assert winner.settled_result_index(states, R, WinType.LINE) is None


def test_settled_result_index_finds_the_later_settling_point_after_an_undo():
    events = [
        *(_mark(0, c, R, float(c)) for c in range(5)),
        _unmark(0, 4, R, 10.0),
        _mark(0, 4, R, 11.0),
    ]

    states = winner.board_states(events)

    assert winner.settled_result_index(states, R, WinType.LINE) == 6


def test_settled_result_index_returns_none_for_no_states():
    assert winner.settled_result_index([], R, WinType.LINE) is None
