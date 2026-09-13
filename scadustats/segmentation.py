"""Detect game boundaries within a video from a stream of per-sample observations.

A single video can contain multiple back-to-back games. Two independent signals hint at
a new game starting: the board resetting to fully unclaimed, and the timer jumping
backward by a large amount. Either alone is noisy (a classification flicker, a bad OCR
read), so a boundary is only trusted where both agree within a small window of samples.

The overlay's "GAME N" label used to be a third signal, dropped because it earned
nothing: on a real two-game video it produced 18 "confirmed" changes, 17 of them OCR
noise ("GARE Zz", "GANIC Zz", ... for a board reading "GAME 2"), while the one genuine
change was only confirmed 15 samples after the fact -- well outside the agreement window
-- so it never once contributed to detecting the real boundary it was watching for.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from scadustats.models import CellColor

Board = list[list[CellColor]]

_AGREEMENT_WINDOW = 5  # samples
_TIMER_JUMP_THRESHOLD = 30  # seconds; distinguishes a genuine reset from a per-second tick


@dataclass
class Observation:
    sample_index: int
    video_ts_s: float
    board: Board
    timer_s: int | None


def _is_all_unclaimed(board: Board) -> bool:
    return all(cell is CellColor.UNCLAIMED for row in board for cell in row)


def detect_boundaries(observations: Iterable[Observation]) -> list[int]:
    """Positions *within `observations`* at which a new game starts.

    These are list positions, not `sample_index` values: the two only coincide when every
    sampled frame was a gameplay frame, and in a real video they diverge badly, since
    every non-gameplay frame (a "POST GAME" recap between games -- exactly where the
    boundaries are) is dropped before it ever gets here. Callers slice the observation
    list with these, so positions are what they need; `sample_index` is still what the
    agreement/collapse windows below are measured in, since a window of "5 samples"
    should mean 5 sampled frames of real footage, not 5 surviving observations with an
    arbitrary stretch of skipped recap between them.
    """
    observations = list(observations)

    def _close(position_a: int, position_b: int) -> bool:
        gap = observations[position_a].sample_index - observations[position_b].sample_index
        return abs(gap) <= _AGREEMENT_WINDOW

    board_reset_at = [
        position
        for position, (prev, obs) in enumerate(
            zip(observations, observations[1:], strict=False), start=1
        )
        if _is_all_unclaimed(obs.board) and not _is_all_unclaimed(prev.board)
    ]
    # A continuously-running stopwatch should never decrease during live play -- but a
    # game apparently has its own multi-minute pre-game countdown that legitimately
    # ticks down by ~1/sample, which a plain "any decrease" check can't tell apart from
    # a genuine reset. A real reset is a large, one-off discontinuity (e.g. a ~63-minute
    # elapsed reading dropping straight to a fresh countdown value), so only count a
    # decrease past a threshold no ordinary per-second tick could produce.
    #
    # Observation.timer_s is optional, and an unreadable run of it clusters exactly where
    # resets happen (the overlay transition between two games). Comparing each reading
    # against the most recently *seen* one keeps such a run from swallowing the drop
    # across it, which an adjacent-pair comparison would, seeing a None on one side of
    # the discontinuity and skipping the pair entirely.
    timer_reset_at = []
    last_timer_s = None
    for position, obs in enumerate(observations):
        if obs.timer_s is None:
            continue
        if last_timer_s is not None and last_timer_s - obs.timer_s > _TIMER_JUMP_THRESHOLD:
            timer_reset_at.append(position)
        last_timer_s = obs.timer_s

    # Every signal has to agree, since there are only two of them left -- a lone board
    # reset (a classification flicker) or a lone timer jump (a bad OCR read) isn't
    # evidence of anything on its own.
    signal_lists = [board_reset_at, timer_reset_at]
    candidates = sorted(set(board_reset_at) | set(timer_reset_at))

    boundaries = [
        candidate
        for candidate in candidates
        if all(any(_close(candidate, position) for position in signal) for signal in signal_lists)
    ]

    # Collapse boundaries that landed within the agreement window of each other (the same
    # real event triggering multiple nearby candidate positions).
    collapsed: list[int] = []
    for boundary in boundaries:
        if not collapsed or not _close(boundary, collapsed[-1]):
            collapsed.append(boundary)
    return collapsed
