"""Win-condition logic over a 5x5 board of CellColor. Pure functions, no I/O."""

from scadustats.models import CellColor, WinLine, WinType

Board = list[list[CellColor]]
Coord = tuple[int, int]

# Keyed by WinLine so a win can name the line it was completed on, not just the color
# that completed it. Built from the indices rather than written out 12 times, which is
# what keeps each key's coordinates guaranteed to match the name it's filed under.
LINES: dict[WinLine, list[Coord]] = {
    **{WinLine(f"row_{r}"): [(r, c) for c in range(5)] for r in range(5)},
    **{WinLine(f"col_{c}"): [(r, c) for r in range(5)] for c in range(5)},
    WinLine.DIAGONAL_TL_BR: [(i, i) for i in range(5)],
    # Written bottom-left first so it reads as its name; a line's cells are compared as a
    # set, so the order within one has no effect either way.
    WinLine.DIAGONAL_BL_TR: [(4 - i, i) for i in range(5)],
}


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


def _all_lines_blocked(board: Board) -> bool:
    """True once every line contains both colors (or an unclaimed cell alongside a color
    that the other player has already touched elsewhere), i.e. no line can still be
    completed by a single color."""
    for coords in LINES.values():
        colors = {board[r][c] for r, c in coords}
        if colors <= {CellColor.UNCLAIMED, CellColor.RED}:
            return False
        if colors <= {CellColor.UNCLAIMED, CellColor.BLUE}:
            return False
    return True


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
    if _all_lines_blocked(board):
        winner = majority_winner(board)
        if winner is not None:
            return winner, WinType.MAJORITY, None
        return None, WinType.TIE, None
    return None, WinType.NONE, None
