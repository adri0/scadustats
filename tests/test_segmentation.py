from scadustats.models import CellColor
from scadustats.segmentation import Observation, detect_boundaries

U, R = CellColor.UNCLAIMED, CellColor.RED


def _unclaimed_board():
    return [[U] * 5 for _ in range(5)]


def _board_with_claim(row: int, col: int):
    board = _unclaimed_board()
    board[row][col] = R
    return board


def test_single_game_has_no_boundaries():
    observations = [
        Observation(0, 0.0, _unclaimed_board(), 0, "GAME 1"),
        Observation(1, 1.0, _unclaimed_board(), 1, None),
        Observation(2, 2.0, _board_with_claim(0, 0), 2, None),
        Observation(3, 3.0, _board_with_claim(0, 0), 3, None),
    ]
    assert detect_boundaries(observations) == []


def test_multi_game_boundary_detected_when_signals_agree():
    # Game 1 ends with a claimed board; game 2 starts with board reset + timer reset +
    # label change all landing at/near the same sample index.
    observations = [
        Observation(0, 0.0, _unclaimed_board(), 0, "GAME 1"),
        Observation(1, 1.0, _board_with_claim(0, 0), 100, None),
        Observation(2, 2.0, _board_with_claim(0, 0), 101, None),
        Observation(3, 3.0, _unclaimed_board(), 0, "GAME 2"),
        Observation(4, 4.0, _unclaimed_board(), 1, None),
    ]
    assert detect_boundaries(observations) == [3]


def test_single_noisy_signal_does_not_trigger_boundary():
    # A single bad label read (flips and flips back) with no board/timer reset should
    # not be mistaken for a new game.
    observations = [
        Observation(0, 0.0, _board_with_claim(0, 0), 100, "GAME 1"),
        Observation(1, 1.0, _board_with_claim(0, 0), 101, "GAME X"),  # noisy OCR blip
        Observation(2, 2.0, _board_with_claim(0, 0), 102, None),
        Observation(3, 3.0, _board_with_claim(0, 0), 103, "GAME 1"),
    ]
    assert detect_boundaries(observations) == []
