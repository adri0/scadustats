"""Detect game boundaries within a video from a stream of per-sample observations.

A single video can contain multiple back-to-back games. Three independent signals hint
at a new game starting: the board resetting to fully unclaimed, the timer jumping
backward by a large amount, and the "GAME N" label text changing. Each is individually
noisy (a classification flicker, a bad OCR read), so a boundary is only trusted when at
least two of the three signals agree within a small window of samples.
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
    # A continuously-running stopwatch should never decrease during live play -- but a
    # game apparently has its own multi-minute pre-game countdown that legitimately
    # ticks down by ~1/sample, which a plain "any decrease" check can't tell apart from
    # a genuine reset. A real reset is a large, one-off discontinuity (e.g. a ~63-minute
    # elapsed reading dropping straight to a fresh countdown value), so only count a
    # decrease past a threshold no ordinary per-second tick could produce.
    timer_reset_at = [
        obs.sample_index
        for prev, obs in zip(observations, observations[1:], strict=False)
        if obs.timer_s is not None
        and prev.timer_s is not None
        and prev.timer_s - obs.timer_s > _TIMER_JUMP_THRESHOLD
    ]

    # The label is only OCR'd on a fraction of samples (it's comparatively expensive),
    # so most samples have label=None. Comparing strictly-adjacent observations would
    # almost never see two non-None labels next to each other -- compare each non-None
    # label against the most recently *seen* non-None label instead. OCR noise on this
    # field is severe in practice (a stable "GAME 2" was read as a different garbled
    # string on nearly every sample), so a new label is only trusted once it has been
    # read identically twice in a row -- cheap noise floor, since a real label is stable
    # for many consecutive reads while noise essentially never repeats itself exactly.
    label_change_at = []
    last_confirmed_label = None
    pending_label = None
    pending_count = 0
    for obs in observations:
        if obs.label is None:
            continue
        if obs.label == pending_label:
            pending_count += 1
        else:
            pending_label = obs.label
            pending_count = 1
        if pending_count >= 2 and pending_label != last_confirmed_label:
            label_change_at.append(obs.sample_index)
            last_confirmed_label = pending_label

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
