import logging

from scadustats.extract import _determine_winner, _extract_events
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
