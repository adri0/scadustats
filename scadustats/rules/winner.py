"""Win-condition logic over a 5x5 board of CellColor, plus the event replay that
produces such a board. Pure functions, no I/O."""

from collections.abc import Iterable, Sequence

from scadustats.models import CellColor, EventType, GameEvent, WinLine, WinType

Board = list[list[CellColor]]
Coord = tuple[int, int]

# Keyed by WinLine so a win can name the line it was completed on, not just the color
# that completed it. Built from the indices rather than written out 12 times, which is
# what keeps each key's coordinates guaranteed to match the name it's filed under. The
# WinLine names are 1-based (see models.WinLine), but Coord stays a 0-based index into
# the internal Board list -- `r`/`c` here are plain Python list indices, never a
# GameEvent.row/col value, so they're offset by one from the name built alongside them.
LINES: dict[WinLine, list[Coord]] = {
    **{WinLine(f"row_{r + 1}"): [(r, c) for c in range(5)] for r in range(5)},
    **{WinLine(f"col_{c + 1}"): [(r, c) for r in range(5)] for c in range(5)},
    WinLine.DIAGONAL_TL_BR: [(i, i) for i in range(5)],
    # Written bottom-left first so it reads as its name; a line's cells are compared as a
    # set, so the order within one has no effect either way.
    WinLine.DIAGONAL_BL_TR: [(4 - i, i) for i in range(5)],
}


def empty_board() -> Board:
    return [[CellColor.UNCLAIMED] * 5 for _ in range(5)]


def apply_event(board: Board, event: GameEvent) -> None:
    """Apply one event to `board`, in place. A non-square event (GAME_START carries no
    row/col -- see models.GameEvent) leaves the board alone rather than being an error:
    replaying a game means walking its whole event list, not a filtered copy of it.

    event.row/col are 1-based (see models.GameEvent); board is a plain 0-based Python
    list, hence the -1s.
    """
    if event.row is None or event.col is None:
        return
    board[event.row - 1][event.col - 1] = (
        event.color if event.event_type is EventType.MARK else CellColor.UNCLAIMED
    )


def replay(events: Iterable[GameEvent]) -> Board:
    """The board state a game's events add up to -- unclaims included, so this is what
    the board genuinely looked like at the end, not every square ever touched."""
    board = empty_board()
    for event in events:
        apply_event(board, event)
    return board


def winning_line(board: Board) -> tuple[CellColor, WinLine] | None:
    """The color holding a completed line and which line that is, or None if no line is
    complete. Returns both together so callers can't end up with a line attributed to the
    wrong color -- everything that needs one needs the other."""
    for win_line, coords in LINES.items():
        colors = {board[r][c] for r, c in coords}
        if colors == {CellColor.RED}:
            return CellColor.RED, win_line
        if colors == {CellColor.BLUE}:
            return CellColor.BLUE, win_line
    return None


def _majority_settled(board: Board) -> bool:
    """True once no distribution of the board's still-unclaimed cells could change which
    color -- if either -- holds the majority: the leading color's tally already beats
    whatever the trailing color could reach by claiming every remaining cell, or there
    are no remaining cells left to claim.

    This used to instead require every *line* to already hold both colors (i.e. that no
    line could still be completed by one color), on the theory that a majority result
    only means anything once line wins are structurally impossible. That's a stronger --
    and, it turns out, wrong -- condition: on a real match
    (2026-07-25-TwistieT-vs-Grey), one goal square went unattempted by both players for
    the whole game, leaving one row all-blue-or-unclaimed right up to the game's own end.
    That row was never actually going to be completed -- the game had already ended --
    but the old check couldn't tell "nobody will ever claim this" apart from "this could
    still be claimed," so it withheld the majority verdict indefinitely even though
    blue's other claims already made the outcome mathematically unbeatable. Counting the
    unclaimed cells directly sidesteps the question of which lines they sit on."""
    red = sum(cell is CellColor.RED for row in board for cell in row)
    blue = sum(cell is CellColor.BLUE for row in board for cell in row)
    unclaimed = sum(len(row) for row in board) - red - blue
    return unclaimed == 0 or abs(red - blue) > unclaimed


def majority_winner(board: Board) -> CellColor | None:
    red = sum(cell is CellColor.RED for row in board for cell in row)
    blue = sum(cell is CellColor.BLUE for row in board for cell in row)
    if red == blue:
        return None
    return CellColor.RED if red > blue else CellColor.BLUE


def determine_winner(board: Board) -> tuple[CellColor | None, WinType, WinLine | None]:
    """The board's result: who won, how, and -- for a LINE win only -- on which line.
    The third element is None for every other outcome, since there's no line to name."""
    if (line_win := winning_line(board)) is not None:
        color, win_line = line_win
        return color, WinType.LINE, win_line
    if _majority_settled(board):
        winner = majority_winner(board)
        if winner is not None:
            return winner, WinType.MAJORITY, None
        return None, WinType.TIE, None
    return None, WinType.NONE, None


def board_states(events: Iterable[GameEvent]) -> list[Board]:
    """The board state after each event, in the given order -- index-aligned with
    `events`. Like `replay`, but keeps every intermediate state instead of collapsing to
    only the final one, for a caller that needs to know *when* the board first reached a
    particular result, not just what it ended as (see `settled_result_index`)."""
    board = empty_board()
    states = []
    for event in events:
        apply_event(board, event)
        states.append([row[:] for row in board])
    return states


def settled_result_index(
    states: Sequence[Board], color: CellColor, win_type: WinType
) -> int | None:
    """Index of the earliest state in `states` from which `(color, win_type)` is the
    result `determine_winner` keeps giving all the way through to the end -- i.e. when
    that outcome first became locked in, rather than merely reached and later undone.

    Walked backward from the end, not forward from the start: a line -- or a majority --
    can be reached and then undone by a later unclaim (a player can mark the wrong square
    and undo it), and a transient reach-and-undo isn't the point that decided the game.
    Returns None if the result never holds continuously through to the end, including
    for an empty `states`.
    """
    settled = None
    for index in reversed(range(len(states))):
        if determine_winner(states[index])[:2] == (color, win_type):
            settled = index
        else:
            break
    return settled
