"""Win-condition logic over a 5x5 board of CellColor. Pure functions, no I/O."""

from scadustats.models import CellColor, WinType

Board = list[list[CellColor]]
Coord = tuple[int, int]

LINES: list[list[Coord]] = (
    [[(r, c) for c in range(5)] for r in range(5)]  # rows
    + [[(r, c) for r in range(5)] for c in range(5)]  # columns
    + [[(i, i) for i in range(5)], [(i, 4 - i) for i in range(5)]]  # diagonals
)


def check_line_winner(board: Board) -> CellColor | None:
    for line in LINES:
        colors = {board[r][c] for r, c in line}
        if colors == {CellColor.RED}:
            return CellColor.RED
        if colors == {CellColor.BLUE}:
            return CellColor.BLUE
    return None


def _all_lines_blocked(board: Board) -> bool:
    """True once every line contains both colors (or an unclaimed cell alongside a color
    that the other player has already touched elsewhere), i.e. no line can still be
    completed by a single color."""
    for line in LINES:
        colors = {board[r][c] for r, c in line}
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


def determine_winner(board: Board) -> tuple[CellColor | None, WinType]:
    if (winner := check_line_winner(board)) is not None:
        return winner, WinType.LINE
    if _all_lines_blocked(board):
        winner = majority_winner(board)
        return (winner, WinType.MAJORITY) if winner is not None else (None, WinType.TIE)
    return None, WinType.NONE
