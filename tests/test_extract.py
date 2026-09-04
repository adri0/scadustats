import logging

from scadustats.extract import _detect_game_start, _determine_winner, _extract_events
from scadustats.models import CellColor, EventType
from scadustats.segmentation import Observation

U, R, B = CellColor.UNCLAIMED, CellColor.RED, CellColor.BLUE


def _board(*colors: tuple[int, int, CellColor]) -> list[list[CellColor]]:
    board = [[U] * 5 for _ in range(5)]
    for row, col, color in colors:
        board[row][col] = color
    return board


def _debounced(board: list[list[CellColor]], count: int = 2) -> list[Observation]:
    """Repeat the same board state enough times to satisfy the 2-consecutive-sample debounce."""
    return [Observation(i, float(i), board, 10 + i) for i in range(count)]


def test_claim_then_mistaken_unclaim_produces_both_events(caplog):
    segment = [
        *_debounced(_board()),
        *_debounced(_board((0, 0, R))),
        *_debounced(_board()),  # player undoes the mistaken mark
    ]
    with caplog.at_level(logging.WARNING):
        events = _extract_events(segment)

    assert [(e.row, e.col, e.color, e.event_type) for e in events] == [
        (0, 0, R, EventType.MARK),
        (0, 0, R, EventType.UNMARK),
    ]
    assert not caplog.records


def test_direct_color_swap_is_logged_not_recorded_as_event(caplog):
    segment = [
        *_debounced(_board((0, 0, R))),
        *_debounced(_board((0, 0, B))),  # red -> blue with no intervening unclaim
    ]
    with caplog.at_level(logging.WARNING):
        events = _extract_events(segment)

    # The initial unclaimed -> red is a real claim; the direct red -> blue swap that
    # follows is logged as a data-quality warning rather than recorded as a second event.
    assert [(e.row, e.col, e.color, e.event_type) for e in events] == [(0, 0, R, EventType.MARK)]
    assert any("unexpected direct color swap" in r.message for r in caplog.records)


def test_winner_ignores_a_line_undone_by_unclaim():
    # Four reds already in a row; the fifth is claimed (completing the line) then
    # immediately unclaimed -- the line was never actually held, so no winner yet.
    base = [(0, c, R) for c in range(4)]
    segment = [
        *_debounced(_board(*base)),
        *_debounced(_board(*base, (0, 4, R))),
        *_debounced(_board(*base)),
    ]
    events = _extract_events(segment)
    winner_color, win_type = _determine_winner(events)
    assert winner_color is None
    from scadustats.models import WinType

    assert win_type is WinType.NONE


def test_detect_game_start_finds_ascent_from_countdown_minimum():
    # A pre-game countdown ticking down to zero, then the real game clock ascending --
    # GAME_START is the sample at the bottom of that dip.
    board = _board()
    segment = [
        Observation(0, 0.0, board, 5),
        Observation(1, 1.0, board, 3),
        Observation(2, 2.0, board, 1),
        Observation(3, 3.0, board, 0),
        Observation(4, 4.0, board, 1),
        Observation(5, 5.0, board, 2),
    ]

    event = _detect_game_start(segment)

    assert event is not None
    assert event.event_type is EventType.GAME_START
    assert (event.row, event.col, event.color) == (None, None, None)
    assert event.game_elapsed_s == 0
    assert event.video_ts_s == 3.0


def test_detect_game_start_returns_none_without_a_local_minimum():
    # No countdown captured -- the timer only ever ascends, so there's no dip to find.
    board = _board()
    segment = [
        Observation(0, 0.0, board, 0),
        Observation(1, 1.0, board, 1),
        Observation(2, 2.0, board, 2),
    ]

    assert _detect_game_start(segment) is None
