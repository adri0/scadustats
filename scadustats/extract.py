"""Orchestrates the full pipeline: frames -> board/timer/scoreboard -> segmentation ->
claim detection -> winner determination -> DuckDB.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from scadustats import board, db, frames, json_export, layout, ocr, scoreboard, timer, winner
from scadustats.models import CellColor, ClaimEvent, EventType, GameResult, WinType
from scadustats.segmentation import Observation, detect_boundaries

logger = logging.getLogger(__name__)

# Samples (~seconds, at the default 1Hz sample rate) between game-label OCR reads --
# the label only needs to be read occasionally since it's one of three redundant
# game-boundary signals, and label OCR is comparatively expensive.
_LABEL_SAMPLE_INTERVAL = 5


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
    observations = []
    prev_counts = (0, 0)
    for i, (video_ts_s, frame) in enumerate(frames.sample_frames(video_path, sample_rate_hz)):
        if on_progress is not None:
            on_progress()

        if not board.is_gameplay_frame(frame):
            # Not showing the live overlay (e.g. a "POST GAME" recap screen, which
            # reuses the same grid coordinates to cycle through other completed games'
            # boards) -- cell colors and timer/label crops would be meaningless here.
            continue

        colors = board.cell_colors(frame)
        timer_s = timer.read_timer(frame)
        label = _read_label(frame) if i % _LABEL_SAMPLE_INTERVAL == 0 else None

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

        observations.append(Observation(i, video_ts_s, colors, timer_s, label))
    return observations


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


def _extract_events(segment: list[Observation]) -> list[ClaimEvent]:
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
                            ClaimEvent(
                                r, c, observed, obs.video_ts_s, game_elapsed, EventType.CLAIM
                            )
                        )
                    elif old is not CellColor.UNCLAIMED and observed is CellColor.UNCLAIMED:
                        events.append(
                            ClaimEvent(r, c, old, obs.video_ts_s, game_elapsed, EventType.UNCLAIM)
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


def _determine_winner(events: list[ClaimEvent]) -> tuple[CellColor | None, WinType]:
    """Replay every event (including unclaims) to the final board state and evaluate the
    win condition once there -- rather than stopping at the first claim that completes a
    line, since a mistaken claim can complete a line and then be immediately unclaimed
    (the game doesn't actually end in that case). The schema doesn't record a win
    timestamp, so evaluating only the final state loses nothing for the normal case
    either: a genuine win ends the game, so no further events should follow it anyway.
    """
    state: list[list[CellColor]] = [[CellColor.UNCLAIMED] * 5 for _ in range(5)]
    for event in events:
        state[event.row][event.col] = (
            event.color if event.event_type is EventType.CLAIM else CellColor.UNCLAIMED
        )
    return winner.determine_winner(state)


def extract_video(
    video_path: str | Path,
    db_path: str | Path = "scadustats.duckdb",
    if_exists: str = "replace",
    json_dir: str | Path | None = None,
    sample_rate_hz: float = 1.0,
    on_progress: Callable[[], None] | None = None,
) -> ExtractionSummary:
    video_path = Path(video_path)
    video_info = frames.probe(video_path)
    observations = _collect_observations(video_path, sample_rate_hz, on_progress)
    segments = _segment_games(observations)

    games = []
    for game_index, segment in enumerate(segments, start=1):
        representative_frame = _grab_frame(video_path, segment[len(segment) // 2].video_ts_s)
        goal_texts = board.cell_goal_texts(representative_frame)
        player_red_name, player_blue_name = scoreboard.read_player_names(representative_frame)
        label = next((obs.label for obs in segment if obs.label), None)

        events = _extract_events(segment)
        winner_color, win_type = _determine_winner(events)

        games.append(
            GameResult(
                game_index=game_index,
                label=label,
                start_video_ts_s=segment[0].video_ts_s,
                end_video_ts_s=segment[-1].video_ts_s,
                player_red_name=player_red_name,
                player_blue_name=player_blue_name,
                goal_texts=goal_texts,
                claims=events,
                winner_color=winner_color,
                win_type=win_type,
            )
        )

    video_id = video_path.stem
    db.write_extraction(db_path, video_id, str(video_path), video_info, games, if_exists=if_exists)

    if json_dir is not None:
        for game in games:
            json_export.write_game(json_dir, video_id, game, if_exists=if_exists)

    return ExtractionSummary(
        video_id=video_id,
        num_games=len(games),
        num_claims=sum(
            1 for game in games for event in game.claims if event.event_type is EventType.CLAIM
        ),
    )
