"""Detect game boundaries within a video from a stream of per-sample observations.

A single video can contain multiple back-to-back games. Three independent signals hint
at a new game starting: the board resetting to fully unclaimed, the timer resetting near
zero, and the "GAME N" label text changing. Each is individually noisy (a classification
flicker, a bad OCR read), so a boundary is only trusted when at least two of the three
signals agree within a small window of samples.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from scadustats.models import CellColor

Board = list[list[CellColor]]

_AGREEMENT_WINDOW = 5  # samples


@dataclass
class Observation:
    sample_index: int
    video_ts_s: float
    board: Board
    timer_s: int | None
    label: str | None = None  # None if the label wasn't OCR'd on this sample


def _is_all_unclaimed(board: Board) -> bool:
    return all(cell is CellColor.UNCLAIMED for row in board for cell in row)


def detect_boundaries(observations: Iterable[Observation]) -> list[int]:
    observations = list(observations)

    board_reset_at = [
        obs.sample_index
        for prev, obs in zip(observations, observations[1:], strict=False)
        if _is_all_unclaimed(obs.board) and not _is_all_unclaimed(prev.board)
    ]
    timer_reset_at = [
        obs.sample_index
        for prev, obs in zip(observations, observations[1:], strict=False)
        if obs.timer_s is not None
        and prev.timer_s is not None
        and obs.timer_s < prev.timer_s
        and obs.timer_s <= 5
    ]
    label_change_at = [
        obs.sample_index
        for prev, obs in zip(observations, observations[1:], strict=False)
        if obs.label is not None and prev.label is not None and obs.label != prev.label
    ]

    signal_lists = [board_reset_at, timer_reset_at, label_change_at]
    candidates = sorted(set(board_reset_at) | set(timer_reset_at) | set(label_change_at))

    boundaries = []
    for candidate in candidates:
        agreeing = sum(
            any(abs(candidate - idx) <= _AGREEMENT_WINDOW for idx in signal)
            for signal in signal_lists
        )
        if agreeing >= 2:
            boundaries.append(candidate)

    # Collapse boundaries that landed within the agreement window of each other (the same
    # real event triggering multiple nearby candidate indices).
    collapsed: list[int] = []
    for boundary in boundaries:
        if not collapsed or boundary - collapsed[-1] > _AGREEMENT_WINDOW:
            collapsed.append(boundary)
    return collapsed
