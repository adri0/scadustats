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
        Observation(0, 0.0, _unclaimed_board(), 0),
        Observation(1, 1.0, _unclaimed_board(), 1),
        Observation(2, 2.0, _board_with_claim(0, 0), 2),
        Observation(3, 3.0, _board_with_claim(0, 0), 3),
    ]
    assert detect_boundaries(observations) == []


def test_multi_game_boundary_detected_when_signals_agree():
    # Game 1 ends with a claimed board; game 2 starts with a board reset and a timer
    # reset landing on the same sample.
    observations = [
        Observation(0, 0.0, _unclaimed_board(), 0),
        Observation(1, 1.0, _board_with_claim(0, 0), 100),
        Observation(2, 2.0, _board_with_claim(0, 0), 101),
        Observation(3, 3.0, _unclaimed_board(), 0),
        Observation(4, 4.0, _unclaimed_board(), 1),
    ]
    assert detect_boundaries(observations) == [3]


def test_boundary_detected_with_a_nonzero_timer_reset():
    # The new game's timer doesn't restart near zero here (e.g. a prep phase before the
    # race timer conceptually starts) -- a large decrease must still count as a reset
    # signal without requiring the new value to be near 0.
    observations = [
        Observation(0, 0.0, _board_with_claim(0, 0), 3753),
        Observation(1, 1.0, _board_with_claim(0, 0), 3753),
        Observation(2, 2.0, _unclaimed_board(), 179),
        Observation(3, 3.0, _unclaimed_board(), 178),
        Observation(4, 4.0, _unclaimed_board(), 177),
    ]
    assert detect_boundaries(observations) == [2]


def test_a_board_reset_alone_does_not_trigger_boundary():
    # A one-sample classification flicker to an all-unclaimed board, with the clock
    # running on through it, isn't a new game.
    observations = [
        Observation(0, 0.0, _board_with_claim(0, 0), 100),
        Observation(1, 1.0, _unclaimed_board(), 101),  # flicker
        Observation(2, 2.0, _board_with_claim(0, 0), 102),
        Observation(3, 3.0, _board_with_claim(0, 0), 103),
    ]
    assert detect_boundaries(observations) == []


def test_a_timer_reset_alone_does_not_trigger_boundary():
    # A single bad timer OCR read, with the board untouched either side of it, isn't a
    # new game either.
    observations = [
        Observation(0, 0.0, _board_with_claim(0, 0), 3000),
        Observation(1, 1.0, _board_with_claim(0, 0), 12),  # misread
        Observation(2, 2.0, _board_with_claim(0, 0), 3002),
        Observation(3, 3.0, _board_with_claim(0, 0), 3003),
    ]
    assert detect_boundaries(observations) == []


def test_timer_reset_is_seen_across_unreadable_timer_samples():
    # Regression test: on real footage the timer is unreadable for several samples during
    # exactly the overlay transition a reset happens in (a frozen final reading, then a
    # run of None, then a fresh countdown). Comparing only strictly-adjacent observations
    # sees a None on one side of the discontinuity and never registers the drop -- each
    # reading has to be compared against the last one actually *seen*.
    observations = [
        Observation(0, 0.0, _board_with_claim(0, 0), 3123),
        Observation(1, 1.0, _board_with_claim(0, 0), None),
        Observation(2, 2.0, _board_with_claim(0, 0), None),
        Observation(3, 3.0, _unclaimed_board(), 7),
        Observation(4, 4.0, _unclaimed_board(), 6),
    ]
    assert detect_boundaries(observations) == [3]


def test_boundaries_are_positions_not_sample_indices():
    # Regression test: non-gameplay frames (a "POST GAME" recap, which is exactly what
    # sits between two games) are dropped before segmentation, so sample_index runs ahead
    # of list position. Callers slice the observation list with these, so a boundary must
    # be the position of the new game's first observation -- returning its sample_index
    # would point past the end of the list here and silently collapse the whole video
    # into one game.
    observations = [
        Observation(0, 0.0, _board_with_claim(0, 0), 100),
        Observation(1, 1.0, _board_with_claim(0, 0), 101),
        # samples 2-49 were a recap screen and never became observations
        Observation(50, 50.0, _unclaimed_board(), 5),
        Observation(51, 51.0, _unclaimed_board(), 4),
    ]
    assert detect_boundaries(observations) == [2]
