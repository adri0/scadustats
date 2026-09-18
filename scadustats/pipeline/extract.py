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
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameResult,
    GameType,
    MatchMetadata,
    VideoExtraction,
    WinLine,
    WinType,
)
from scadustats.overlay import board, commentators, game_type_label, scoreboard, timer
from scadustats.pipeline import squares
from scadustats.pipeline.segmentation import Observation, detect_boundaries
from scadustats.rules import winner
from scadustats.storage import json_export
from scadustats.video import frames

logger = logging.getLogger(__name__)

# Frames to majority-vote per game when reading square text -- a single frame's OCR
# occasionally misreads a cell, so square text is read from this many frames spread
# across the game and, per cell, whichever exact text came back most often wins (see
# board.cell_square_texts_majority).
_SQUARE_TEXT_FRAME_COUNT = 10

# Profiling showed OCR is 80-90% of extraction's wall time, dominated by the per-sample
# timer read. ocr.py binds Tesseract's C++ engine in-process via tesserocr rather than
# shelling out per call (see CLAUDE.md), so concurrency here now overlaps a much smaller
# per-call cost -- tesserocr's Cython layer only releases the GIL for part of its work,
# not the full wait a subprocess gave for free, so this is a smaller win than it used to
# be, but still a net positive with no accuracy cost. Capped well below "one per sample"
# so a long video doesn't queue thousands of decoded frames in memory at once.
_OCR_WORKERS = min(8, os.cpu_count() or 4)

# Largest one-sample rise the game clock can plausibly show once it has started ticking
# up (~1s/sample at the default rate, with slack for a run of unreadable samples in
# between). Anything larger is the overlay swapping which timer it displays -- see
# _detect_game_start.
_CLOCK_STEP_TOLERANCE_S = 30

# Share of gameplay samples whose timer may be unreadable before that stops looking like
# the usual between-game transition (well under 1% on real videos) and starts looking
# like the timer simply isn't being read -- see _collect_observations.
_UNREADABLE_TIMER_WARN_FRACTION = 0.05

# Consecutive matching samples required to confirm a color change -- see _extract_events.
# A claim (UNCLAIMED -> RED/BLUE) uses the baseline debounce, same as a direct color swap.
# An unclaim (RED/BLUE -> UNCLAIMED) requires more: it's the rarer, more consequential
# direction (a real one is a deliberate undo that persists for the rest of the game, not
# a couple of samples), and it's the direction a stray couple of misread frames falls
# into by default, since a dulled/compressed color patch reads as UNCLAIMED far more
# readily than it invents a false RED/BLUE out of nothing (the splash-transition
# regression documented in CLAUDE.md is exactly this failure mode). Doubling the baseline
# meaningfully cuts that flicker through without materially delaying detection of a
# genuine, sustained unclaim.
_MARK_DEBOUNCE_SAMPLES = 2
_UNMARK_DEBOUNCE_SAMPLES = 2 * _MARK_DEBOUNCE_SAMPLES


@dataclass
class ExtractionSummary:
    video_id: str
    num_games: int
    num_claims: int
    # The full extraction, for a caller that wants to show/inspect it (e.g. the CLI
    # printing the same output as `match show` right after extracting) without a separate
    # read back off disk. Optional since it's meaningless when skipped=True -- the
    # existing file on disk was left untouched, and this would otherwise describe the
    # replacement that didn't happen.
    extraction: VideoExtraction | None = None
    # True when a duplicate was found and on_duplicate declined to replace it --
    # num_games/num_claims are meaningless (left at 0) in that case, since nothing was
    # written.
    skipped: bool = False


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
    # (index, video_ts_s, colors, timer_future) per gameplay sample, in sample order --
    # the timer OCR runs in the background via the future, so this fills in while later
    # samples are still being decoded, and is drained into Observations only once every
    # future for it has resolved.
    pending: list[tuple[int, float, list[list[CellColor]], Future[int | None]]] = []
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
                # boards) -- the cell colors and the timer crop would be meaningless here.
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

            pending.append((i, video_ts_s, colors, timer_future))

        # An unreadable timer means the overlay isn't in a live-gameplay state, so the
        # sample is dropped here for the same reason board.is_gameplay_frame drops a
        # recap screen -- it just isn't detectable until the OCR result is in hand. The
        # case this catches is the splash *between* two games: the score bars stay
        # colored (so is_gameplay_frame still passes), but the grid and the timer are
        # both torn down for a few samples. Reading cell colors off that splash produced
        # a burst of spurious unmarks that wiped the previous game's winning line out of
        # its final board state, costing game 1 of a real two-game video its winner.
        # Unreadable timers are otherwise vanishingly rare in practice (8 samples out of
        # 6280 on that video -- all 8 of them this exact transition), so this drops
        # essentially nothing else.
        observations = [
            Observation(i, video_ts_s, colors, timer_s)
            for i, video_ts_s, colors, timer_future in pending
            if (timer_s := timer_future.result()) is not None
        ]

    # Dropping a handful of samples is the point; dropping a large share of them means
    # the timer isn't being read at all (a miscalibrated TIMER_BOX, a source too low-
    # bitrate for it) and claims are silently going missing with it -- which would
    # otherwise look like a quiet, successful extraction of a near-empty game.
    dropped = len(pending) - len(observations)
    if dropped > len(pending) * _UNREADABLE_TIMER_WARN_FRACTION:
        logger.warning(
            "timer unreadable on %d of %d gameplay samples (%.0f%%) -- those samples were "
            "skipped, so claims in them are missing; check the timer layout box",
            dropped,
            len(pending),
            100 * dropped / len(pending),
        )
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
    already mid-game with no countdown captured at all).

    The ascent out of the minimum has to be *continuous* to count, though. A segment can
    open on the tail of a short countdown belonging to the previous screen, which then
    gives way to the game's own countdown -- observed on a real match as ...3, 2, 1 and
    then straight to 180, a 3-minute countdown that only then descends to 0 and ascends
    as the game clock. That reading of 1 is a strict local minimum, but the +179 step out
    of it is the overlay swapping which timer it's showing, not a clock ticking up.
    """
    timed = [obs for obs in segment if obs.timer_s is not None]
    for prev, curr, nxt in zip(timed, timed[1:], timed[2:], strict=False):
        is_local_minimum = curr.timer_s < prev.timer_s and curr.timer_s < nxt.timer_s
        clock_continues = nxt.timer_s - curr.timer_s <= _CLOCK_STEP_TOLERANCE_S
        if is_local_minimum and clock_continues:
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

    A color change is only accepted once the same new color has been observed on enough
    consecutive samples in a row to reach its required debounce (see
    _MARK_DEBOUNCE_SAMPLES/_UNMARK_DEBOUNCE_SAMPLES) -- this rejects short runs of
    classification flicker (e.g. a transitional stream wipe/flash whose color transiently
    matches a reference hue, or a dulled/compressed color patch briefly misreading as
    UNCLAIMED) without meaningfully hurting timestamp precision at the ~1 sample/second
    rate this pipeline samples at. Unclaiming requires a longer run than claiming -- see
    _UNMARK_DEBOUNCE_SAMPLES.

    UNCLAIMED -> RED/BLUE is a claim. RED/BLUE -> UNCLAIMED is a legitimate unclaim, not
    an anomaly: a player can inadvertently mark the wrong square and undo it. A direct
    RED <-> BLUE swap without an intervening unclaim isn't a real game mechanic though,
    so that's still logged as a data-quality warning rather than recorded as an event
    (debounced the same as a claim).
    """
    events = []
    confirmed: list[list[CellColor]] = [[CellColor.UNCLAIMED] * 5 for _ in range(5)]
    # Per-cell (candidate color, consecutive-run-length) for whatever color currently
    # differs from confirmed -- reset whenever the observed color changes again before
    # reaching its required run length.
    candidate: list[list[CellColor | None]] = [[None] * 5 for _ in range(5)]
    run_length: list[list[int]] = [[0] * 5 for _ in range(5)]
    for obs in segment:
        for r in range(5):
            for c in range(5):
                observed = obs.board[r][c]
                if observed == confirmed[r][c]:
                    candidate[r][c] = None
                    run_length[r][c] = 0
                    continue
                if observed == candidate[r][c]:
                    run_length[r][c] += 1
                else:
                    candidate[r][c] = observed
                    run_length[r][c] = 1

                required = (
                    _UNMARK_DEBOUNCE_SAMPLES
                    if observed is CellColor.UNCLAIMED
                    else _MARK_DEBOUNCE_SAMPLES
                )
                if run_length[r][c] < required:
                    continue

                old = confirmed[r][c]
                game_elapsed = obs.timer_s if obs.timer_s is not None else 0
                # r/c are 0-based list indices; GameEvent.row/col are 1-based (see
                # models.GameEvent), hence the +1s below.
                if old is CellColor.UNCLAIMED and observed is not CellColor.UNCLAIMED:
                    events.append(
                        GameEvent(
                            r + 1, c + 1, observed, obs.video_ts_s, game_elapsed, EventType.MARK
                        )
                    )
                elif old is not CellColor.UNCLAIMED and observed is CellColor.UNCLAIMED:
                    events.append(
                        GameEvent(
                            r + 1, c + 1, old, obs.video_ts_s, game_elapsed, EventType.UNMARK
                        )
                    )
                else:
                    logger.warning(
                        "unexpected direct color swap %s -> %s at (%d,%d), ts=%.1fs",
                        old,
                        observed,
                        r + 1,
                        c + 1,
                        obs.video_ts_s,
                    )
                confirmed[r][c] = observed
                candidate[r][c] = None
                run_length[r][c] = 0
    return events


def _detect_game_end(
    events: list[GameEvent], winner_color: CellColor | None, win_type: WinType
) -> GameEvent | None:
    """The GAME_END event for a game with a recorded winner (LINE or MAJORITY) --
    timestamped at the one MARK event that locks that outcome in for the rest of the
    game (see winner.settled_result_index), the same way _detect_game_start locates
    GAME_START at a signal in the existing events rather than inventing a new timestamp.

    `events` need not be in video-timestamp order on the way in (GAME_START is prepended
    to the front of the list regardless of its own timestamp -- see extract_video), so
    this sorts its own copy before replaying. Returns None when there's no winner to
    settle (TIE/NONE, where winner_color is None) or the settling point can't be found --
    including, in principle, a settling point that lands on something other than a MARK
    (an UNMARK can't complete a line, but could in principle tip a majority swing; that
    isn't a real "square marked" ending and isn't reported as one).
    """
    if winner_color is None:
        return None
    ordered = sorted(events, key=lambda event: event.video_ts_s)
    states = winner.board_states(ordered)
    index = winner.settled_result_index(states, winner_color, win_type)
    if index is None or ordered[index].event_type is not EventType.MARK:
        return None
    settling = ordered[index]
    return GameEvent(
        row=None,
        col=None,
        color=None,
        video_ts_s=settling.video_ts_s,
        game_elapsed_s=settling.game_elapsed_s,
        event_type=EventType.GAME_END,
    )


def _determine_winner(
    events: list[GameEvent],
) -> tuple[CellColor | None, WinType, WinLine | None]:
    """Replay every event (including unclaims) to the final board state and evaluate the
    win condition once there -- rather than stopping at the first claim that completes a
    line, since a mistaken claim can complete a line and then be immediately unclaimed
    (the game doesn't actually end in that case). The schema doesn't record a win
    timestamp, so evaluating only the final state loses nothing for the normal case
    either: a genuine win ends the game, so no further events should follow it anyway.

    Non-square events (e.g. GAME_START) carry no row/col and don't affect board state,
    so winner.replay skips them.
    """
    return winner.determine_winner(winner.replay(events))


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
    data_dir: str | Path = "data",
    if_exists: str | None = None,
    sample_rate_hz: float = 1.0,
    on_progress: Callable[[], None] | None = None,
    on_missing_game_type: Callable[[GameResult], GameType] | None = None,
    on_duplicate: Callable[[Path], bool] | None = None,
    known_squares: dict[str, GameType] | None = None,
    # The video's YouTube upload date, when known -- read off the same yt-dlp call that
    # downloaded the video (see video.download.DownloadResult), not fetched here: a
    # locally-supplied video was never downloaded by this run, so there's no metadata to
    # read it from, and this stays None rather than extract_video making its own separate
    # network call just to look it up.
    published_at: date | None = None,
) -> ExtractionSummary:
    video_path = Path(video_path)
    if known_squares is None:
        known_squares = squares.load_known_squares(Path(data_dir) / "squares")
    # The whole broadcast's length, recorded alongside the games -- a match's games only
    # cover part of it (intros, between-game recaps and post-game are all in there too),
    # so this can't be derived from the game segments after the fact. Non-positive means
    # frames.probe couldn't read an fps to divide the frame count by; None ("unknown")
    # rather than a fabricated 0.0 in that case.
    duration_s = frames.probe(video_path).duration_s
    observations = _collect_observations(video_path, sample_rate_hz, on_progress)
    segments = _segment_games(observations)

    games = []
    # Player names are read once, from the first game, rather than once per game -- a
    # video is one match, so the same two players hold for every game in it (see
    # VideoExtraction), and re-reading per game would just be redundant OCR cost.
    player_red_name: str | None = None
    player_blue_name: str | None = None
    # Read once too, and for the same reason -- one video is one broadcast, cast by the
    # same people throughout.
    casters: list[str] = []
    for game_index, segment in enumerate(segments, start=1):
        square_text_observations = _select_square_text_observations(segment)
        square_text_frames = _grab_frames(
            video_path, [obs.video_ts_s for obs in square_text_observations]
        )
        square_texts = board.cell_square_texts_majority(square_text_frames)
        # The overlay's own "BASE GAME"/"DLC" subtitle (see game_type_label.py) is tried
        # first -- it's read directly off the same frames already grabbed for square-text
        # majority voting, no extra decode cost. Only falls back to inferring from square
        # texts (squares.py) if none of those frames produced a clean reading, e.g. a
        # transition moment or -- see CLAUDE.md -- source footage too compressed for this
        # small a subtitle to OCR reliably.
        game_type = game_type_label.majority_game_type_label(square_text_frames)
        if game_type is None:
            game_type = squares.infer_game_type(square_texts, known_squares)
        if game_index == 1:
            representative_frame = _grab_frame(video_path, segment[len(segment) // 2].video_ts_s)
            player_red_name, player_blue_name = scoreboard.read_player_names(representative_frame)
            # Majority-voted over the frames already grabbed above for square text, so
            # this costs no extra decoding -- only the OCR of two small crops per frame,
            # once for the whole video. Unlike the player names, which come off one
            # frame, there are no two independent readings to cross-check a commentator
            # nameplate against (see the scoreboard count check in _collect_observations),
            # so voting is the only guard against a single bad frame here.
            casters = commentators.majority_commentator_names(square_text_frames)

        events = _extract_events(segment)
        game_start = _detect_game_start(segment)
        if game_start is not None:
            events = [game_start, *events]
        winner_color, win_type, win_line = _determine_winner(events)
        game_end = _detect_game_end(events, winner_color, win_type)
        if game_end is not None:
            events = [*events, game_end]

        games.append(
            GameResult(
                game_index=game_index,
                start_video_ts_s=segment[0].video_ts_s,
                end_video_ts_s=segment[-1].video_ts_s,
                square_texts=square_texts,
                events=events,
                winner_color=winner_color,
                win_type=win_type,
                win_line=win_line,
                game_type=game_type,
            )
        )

    # Resolved here -- the last step before persistence -- rather than at the top of the
    # function, so a caller can hand in a Future that's still being filled in (e.g. the
    # CLI prompting the user interactively) and have it overlap with everything above
    # instead of blocking extraction from starting. In practice this essentially never
    # actually blocks: prompting takes seconds, extraction takes minutes.
    if isinstance(match_metadata, Future):
        match_metadata = match_metadata.result()

    # Deferred to here (rather than resolved inline with the inference above) so it runs
    # only after match_metadata's own prompting (if any) has fully finished -- both can
    # ultimately be interactive, and doing this one first would mean two things prompting
    # over each other on the terminal at once.
    for game in games:
        if game.game_type is None:
            if on_missing_game_type is None:
                raise ValueError(
                    f"couldn't infer game type for game {game.game_index} from its "
                    "squares, and no on_missing_game_type callback was given to supply one"
                )
            game.game_type = on_missing_game_type(game)

    video_id = _video_id(match_metadata, player_red_name, player_blue_name)
    extraction = VideoExtraction(
        video_id=video_id,
        video_url=match_metadata.video_url,
        match_date=match_metadata.match_date,
        season=match_metadata.season,
        match_type=match_metadata.match_type,
        player_red_name=player_red_name,
        player_blue_name=player_blue_name,
        extracted_at=date.today(),
        games=games,
        commentators=casters,
        duration_s=duration_s if duration_s > 0 else None,
        published_at=published_at,
        # The local file this extraction actually ran against -- whether it was supplied
        # directly or downloaded from a URL first (video_path is the same either way by
        # this point; see cli.app.extract). Recorded so a later run against the same file
        # can be recognized and offered this run's details as defaults.
        source_path=str(video_path),
    )
    # A match is unique by match_date + player names (see _video_id), which video_id
    # already encodes -- so a same-name file here means this exact match was already
    # extracted. if_exists=None ("not explicitly forced by the caller") is the only case
    # that asks about it: "error"/"replace" (an explicit --if-exists) skip straight to
    # json_export.write_video below, which already knows how to raise or overwrite
    # unconditionally for those.
    if if_exists is None:
        target_path = json_export.video_path(data_dir, extraction.season, video_id)
        if target_path.exists():
            if on_duplicate is None:
                raise ValueError(
                    f"{target_path} already exists, and no on_duplicate callback was "
                    "given to ask whether to replace it"
                )
            if not on_duplicate(target_path):
                return ExtractionSummary(
                    video_id=video_id, num_games=0, num_claims=0, skipped=True
                )
        if_exists = "replace"

    json_export.write_video(data_dir, extraction, if_exists=if_exists)

    return ExtractionSummary(
        video_id=video_id,
        num_games=len(games),
        num_claims=sum(
            1 for game in games for event in game.events if event.event_type is EventType.MARK
        ),
        extraction=extraction,
    )
