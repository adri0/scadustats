"""Orchestrates the full pipeline: frames -> board/timer/scoreboard -> segmentation ->
claim detection -> winner determination -> JSON. Writing that JSON into DuckDB is a
separate, optional step -- see db.load_json_dir -- not something extract_video does.
"""

import logging
import math
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from scadustats import board, frames, json_export, layout, ocr, scoreboard, timer, winner
from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameResult,
    MatchMetadata,
    WinType,
)
from scadustats.segmentation import Observation, detect_boundaries

logger = logging.getLogger(__name__)

# Samples (~seconds, at the default 1Hz sample rate) between game-label OCR reads --
# the label only needs to be read occasionally since it's one of three redundant
# game-boundary signals, and label OCR is comparatively expensive.
_LABEL_SAMPLE_INTERVAL = 5

# Frames to majority-vote per game when reading square text -- a single frame's OCR
# occasionally misreads a cell, so square text is read from this many frames spread
# across the game and, per cell, whichever exact text came back most often wins (see
# board.cell_square_texts_majority).
_SQUARE_TEXT_FRAME_COUNT = 10

# Profiling showed OCR (pytesseract shelling out to the tesseract binary) is 80-90% of
# extraction's wall time, dominated by the per-sample timer read -- each call has a fixed
# ~70ms process-spawn/IPC cost regardless of crop size. tesseract runs as a subprocess, so
# the calling thread releases the GIL while waiting on it; running several concurrently
# overlaps that wait instead of serializing it. Capped well below "one per sample" so a
# long video doesn't queue thousands of frames/tesseract processes at once.
_OCR_WORKERS = min(8, os.cpu_count() or 4)


@dataclass
class ExtractionSummary:
    video_id: str
    num_games: int
    num_claims: int


def _read_label(frame: np.ndarray) -> str:
    height, width = frame.shape[:2]
    crop = layout.crop(frame, layout.GAME_LABEL_BOX, width, height)
    return ocr.read_text(crop, psm=7)


def estimate_sample_count(video_path: str | Path, sample_rate_hz: float = 1.0) -> int:
    """Estimate how many samples `extract_video` will process, for sizing a progress bar
    upfront -- an estimate only, since the true count depends on exactly where decoding
    stops near the end of the video."""
    video_info = frames.probe(video_path)
    return math.ceil(video_info.duration_s * sample_rate_hz)


def _collect_observations(
    video_path: Path,
    sample_rate_hz: float,
    on_progress: Callable[[], None] | None = None,
) -> list[Observation]:
    prev_counts = (0, 0)
    # (index, video_ts_s, colors, timer_future, label_future) per gameplay sample, in
    # sample order -- OCR runs in the background via the futures below, so this fills in
    # while later samples are still being decoded, and is drained into Observations only
    # once every future for it has resolved.
    pending: list[
        tuple[int, float, list[list[CellColor]], Future[int | None], Future[str] | None]
    ] = []
    ocr_slots = threading.Semaphore(_OCR_WORKERS)

    def _release_slot(_future: Future) -> None:
        ocr_slots.release()

    with ThreadPoolExecutor(max_workers=_OCR_WORKERS) as executor:
        for i, (video_ts_s, frame) in enumerate(frames.sample_frames(video_path, sample_rate_hz)):
            if on_progress is not None:
                on_progress()

            if not board.is_gameplay_frame(frame):
                # Not showing the live overlay (e.g. a "POST GAME" recap screen, which
                # reuses the same grid coordinates to cycle through other completed games'
                # boards) -- cell colors and timer/label crops would be meaningless here.
                continue

            colors = board.cell_colors(frame)

            red = sum(cell is CellColor.RED for row in colors for cell in row)
            blue = sum(cell is CellColor.BLUE for row in colors for cell in row)
            if (red, blue) != prev_counts:
                board_red, board_blue = scoreboard.read_scores(frame)
                if board_red is not None and board_red != red:
                    logger.warning(
                        "red claim count mismatch at %.1fs: board=%d scoreboard=%d",
                        video_ts_s,
                        red,
                        board_red,
                    )
                if board_blue is not None and board_blue != blue:
                    logger.warning(
                        "blue claim count mismatch at %.1fs: board=%d scoreboard=%d",
                        video_ts_s,
                        blue,
                        board_blue,
                    )
                prev_counts = (red, blue)

            # Acquiring a slot (rather than submitting unboundedly) is what keeps this a
            # bounded pipeline instead of decoding/queueing the whole video's frames in
            # memory before OCR has processed any of them.
            ocr_slots.acquire()
            timer_future = executor.submit(timer.read_timer, frame)
            timer_future.add_done_callback(_release_slot)

            label_future = None
            if i % _LABEL_SAMPLE_INTERVAL == 0:
                ocr_slots.acquire()
                label_future = executor.submit(_read_label, frame)
                label_future.add_done_callback(_release_slot)

            pending.append((i, video_ts_s, colors, timer_future, label_future))

        return [
            Observation(
                i,
                video_ts_s,
                colors,
                timer_future.result(),
                label_future.result() if label_future is not None else None,
            )
            for i, video_ts_s, colors, timer_future, label_future in pending
        ]


def _segment_games(observations: list[Observation]) -> list[list[Observation]]:
    boundaries = detect_boundaries(observations)
    starts = [0, *boundaries]
    ends = [*boundaries, len(observations)]
    return [observations[start:end] for start, end in zip(starts, ends, strict=True) if start < end]


def _grab_frame(video_path: Path, video_ts_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, video_ts_s * 1000)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"could not read a frame at {video_ts_s}s from {video_path}")
        return frame
    finally:
        cap.release()


def _grab_frames(video_path: Path, video_ts_s_list: list[float]) -> list[np.ndarray]:
    """Like `_grab_frame`, but shares one `VideoCapture` across several timestamps --
    used to pull the handful of frames majority-voted for square text, rather than
    reopening the video file per frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        result = []
        for video_ts_s in video_ts_s_list:
            cap.set(cv2.CAP_PROP_POS_MSEC, video_ts_s * 1000)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"could not read a frame at {video_ts_s}s from {video_path}")
            result.append(frame)
        return result
    finally:
        cap.release()


def _select_square_text_observations(segment: list[Observation]) -> list[Observation]:
    """Up to `_SQUARE_TEXT_FRAME_COUNT` observations spread evenly across the segment, so
    square text can be majority-voted across multiple frames instead of trusting whichever
    single frame happened to land on the segment's midpoint.
    """
    if len(segment) <= _SQUARE_TEXT_FRAME_COUNT:
        return segment
    step = (len(segment) - 1) / (_SQUARE_TEXT_FRAME_COUNT - 1)
    indices = sorted({round(i * step) for i in range(_SQUARE_TEXT_FRAME_COUNT)})
    return [segment[i] for i in indices]


def _detect_game_start(segment: list[Observation]) -> GameEvent | None:
    """The overlay's stopwatch isn't the game clock until the pre-game countdown (which
    ticks down by ~1/sample, see segmentation.py) bottoms out -- the real GAME_START
    moment is where the reading stops decreasing and starts ascending again, i.e. the
    first strict local minimum in the segment's timer readings. Returns None if no such
    minimum is observed (e.g. the segment's footage starts mid-countdown-descent, or
    already mid-game with no countdown captured at all)."""
    timed = [obs for obs in segment if obs.timer_s is not None]
    for prev, curr, nxt in zip(timed, timed[1:], timed[2:], strict=False):
        if curr.timer_s < prev.timer_s and curr.timer_s < nxt.timer_s:
            return GameEvent(
                row=None,
                col=None,
                color=None,
                video_ts_s=curr.video_ts_s,
                game_elapsed_s=curr.timer_s,
                event_type=EventType.GAME_START,
            )
    return None


def _extract_events(segment: list[Observation]) -> list[GameEvent]:
    """Diff consecutive board states into claim/unclaim events.

    A color change is only accepted once the same new color has been observed on two
    consecutive samples -- this debounce rejects single-frame classification flicker
    (e.g. a transitional stream wipe/flash whose color transiently matches a reference
    hue) without meaningfully hurting timestamp precision at the ~1 sample/second rate
    this pipeline samples at.

    UNCLAIMED -> RED/BLUE is a claim. RED/BLUE -> UNCLAIMED is a legitimate unclaim, not
    an anomaly: a player can inadvertently mark the wrong square and undo it. A direct
    RED <-> BLUE swap without an intervening unclaim isn't a real game mechanic though,
    so that's still logged as a data-quality warning rather than recorded as an event.
    """
    events = []
    confirmed: list[list[CellColor]] = [[CellColor.UNCLAIMED] * 5 for _ in range(5)]
    previous_observed: list[list[CellColor]] = [[CellColor.UNCLAIMED] * 5 for _ in range(5)]
    for obs in segment:
        for r in range(5):
            for c in range(5):
                observed = obs.board[r][c]
                if observed == previous_observed[r][c] and observed != confirmed[r][c]:
                    old = confirmed[r][c]
                    game_elapsed = obs.timer_s if obs.timer_s is not None else 0
                    if old is CellColor.UNCLAIMED and observed is not CellColor.UNCLAIMED:
                        events.append(
                            GameEvent(
                                r, c, observed, obs.video_ts_s, game_elapsed, EventType.MARK
                            )
                        )
                    elif old is not CellColor.UNCLAIMED and observed is CellColor.UNCLAIMED:
                        events.append(
                            GameEvent(r, c, old, obs.video_ts_s, game_elapsed, EventType.UNMARK)
                        )
                    else:
                        logger.warning(
                            "unexpected direct color swap %s -> %s at (%d,%d), ts=%.1fs",
                            old,
                            observed,
                            r,
                            c,
                            obs.video_ts_s,
                        )
                    confirmed[r][c] = observed
                previous_observed[r][c] = observed
    return events


def _determine_winner(events: list[GameEvent]) -> tuple[CellColor | None, WinType]:
    """Replay every event (including unclaims) to the final board state and evaluate the
    win condition once there -- rather than stopping at the first claim that completes a
    line, since a mistaken claim can complete a line and then be immediately unclaimed
    (the game doesn't actually end in that case). The schema doesn't record a win
    timestamp, so evaluating only the final state loses nothing for the normal case
    either: a genuine win ends the game, so no further events should follow it anyway.

    Non-square events (e.g. GAME_START) carry no row/col and don't affect board state,
    so they're skipped here.
    """
    state: list[list[CellColor]] = [[CellColor.UNCLAIMED] * 5 for _ in range(5)]
    for event in events:
        if event.row is None or event.col is None:
            continue
        state[event.row][event.col] = (
            event.color if event.event_type is EventType.MARK else CellColor.UNCLAIMED
        )
    return winner.determine_winner(state)


def _video_id(match_metadata: MatchMetadata, red_name: str | None, blue_name: str | None) -> str:
    """`<match-date>-<red player>-vs-<blue-player>`, e.g. `2026-03-05-blanxz-vs-Serious`
    -- human-readable, and unique per match (not per game: every game in a multi-game
    match shares one video_id, matching one video -> one match). red_name/blue_name fall
    back to "unknown" for an empty/failed OCR read (scoreboard.read_player_names can
    return "" but never None) or when a segment has no games at all, rather than leaving
    the id with a blank component. "/" is replaced since it would otherwise split into a
    spurious path segment when used as a filename.
    """
    red = (red_name or "unknown").strip().replace("/", "-")
    blue = (blue_name or "unknown").strip().replace("/", "-")
    return f"{match_metadata.match_date.isoformat()}-{red}-vs-{blue}"


def extract_video(
    video_path: str | Path,
    *,
    match_metadata: MatchMetadata | Future[MatchMetadata],
    json_dir: str | Path = "matches",
    if_exists: str = "replace",
    sample_rate_hz: float = 1.0,
    on_progress: Callable[[], None] | None = None,
) -> ExtractionSummary:
    video_path = Path(video_path)
    observations = _collect_observations(video_path, sample_rate_hz, on_progress)
    segments = _segment_games(observations)

    games = []
    for game_index, segment in enumerate(segments, start=1):
        representative_frame = _grab_frame(video_path, segment[len(segment) // 2].video_ts_s)
        square_text_observations = _select_square_text_observations(segment)
        square_text_frames = _grab_frames(
            video_path, [obs.video_ts_s for obs in square_text_observations]
        )
        square_texts = board.cell_square_texts_majority(square_text_frames)
        player_red_name, player_blue_name = scoreboard.read_player_names(representative_frame)
        label = next((obs.label for obs in segment if obs.label), None)

        events = _extract_events(segment)
        game_start = _detect_game_start(segment)
        if game_start is not None:
            events = [game_start, *events]
        winner_color, win_type = _determine_winner(events)

        games.append(
            GameResult(
                game_index=game_index,
                label=label,
                start_video_ts_s=segment[0].video_ts_s,
                end_video_ts_s=segment[-1].video_ts_s,
                player_red_name=player_red_name,
                player_blue_name=player_blue_name,
                square_texts=square_texts,
                events=events,
                winner_color=winner_color,
                win_type=win_type,
            )
        )

    # Resolved here -- the last step before persistence -- rather than at the top of the
    # function, so a caller can hand in a Future that's still being filled in (e.g. the
    # CLI prompting the user interactively) and have it overlap with everything above
    # instead of blocking extraction from starting. In practice this essentially never
    # actually blocks: prompting takes seconds, extraction takes minutes.
    if isinstance(match_metadata, Future):
        match_metadata = match_metadata.result()

    first_game = games[0] if games else None
    video_id = _video_id(
        match_metadata,
        first_game.player_red_name if first_game else None,
        first_game.player_blue_name if first_game else None,
    )
    for game in games:
        json_export.write_game(json_dir, video_id, game, match_metadata, if_exists=if_exists)

    return ExtractionSummary(
        video_id=video_id,
        num_games=len(games),
        num_claims=sum(
            1 for game in games for event in game.events if event.event_type is EventType.MARK
        ),
    )
