import datetime
import logging

import numpy as np

from scadustats.models import CellColor, EventType, MatchMetadata, MatchType, WinType
from scadustats.pipeline import extract
from scadustats.pipeline.extract import (
    _collect_observations,
    _detect_game_end,
    _detect_game_start,
    _determine_winner,
    _extract_events,
    _match_id,
    _segment_games,
)
from scadustats.pipeline.segmentation import Observation

U, R, B = CellColor.UNCLAIMED, CellColor.RED, CellColor.BLUE


def _board(*colors: tuple[int, int, CellColor]) -> list[list[CellColor]]:
    board = [[U] * 5 for _ in range(5)]
    for row, col, color in colors:
        board[row][col] = color
    return board


def _stub_sampling(monkeypatch, timer_readings: list[int | None]) -> None:
    """Run _collect_observations over synthetic frames whose timer reads as given, with
    every frame passing the gameplay check and showing an empty board."""
    # Each frame is painted with its own sample number so the stubbed timer read can tell
    # which frame it was handed -- OCR runs concurrently, so a plain call counter would
    # race. (Sample number has to fit in a uint8 pixel, capping this at 256 samples.)
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(len(timer_readings))]

    monkeypatch.setattr(
        extract.frames,
        "sample_frames",
        lambda path, rate: ((float(i), frame) for i, frame in enumerate(frames)),
    )
    monkeypatch.setattr(extract.board, "is_gameplay_frame", lambda frame: True)
    monkeypatch.setattr(extract.board, "cell_colors", lambda frame: _board())
    monkeypatch.setattr(extract.scoreboard, "read_scores", lambda frame: (None, None))
    monkeypatch.setattr(extract.timer, "read_timer", lambda frame: timer_readings[frame[0, 0, 0]])


def _debounced(
    board: list[list[CellColor]], count: int = extract._MARK_DEBOUNCE_SAMPLES
) -> list[Observation]:
    """Repeat the same board state enough times to satisfy the mark debounce (by default --
    pass extract._UNMARK_DEBOUNCE_SAMPLES for a block meant to confirm an unclaim)."""
    return [Observation(i, float(i), board, 10 + i) for i in range(count)]


def test_claim_then_mistaken_unclaim_produces_both_events(caplog):
    segment = [
        *_debounced(_board()),
        *_debounced(_board((0, 0, R))),
        # player undoes the mistaken mark -- unclaiming needs the longer debounce run
        *_debounced(_board(), count=extract._UNMARK_DEBOUNCE_SAMPLES),
    ]
    with caplog.at_level(logging.WARNING):
        events = _extract_events(segment)

    assert [(e.row, e.col, e.color, e.event_type) for e in events] == [
        (1, 1, R, EventType.MARK),
        (1, 1, R, EventType.UNMARK),
    ]
    assert not caplog.records


def test_brief_reversion_to_unclaimed_is_not_recorded_as_an_unmark(caplog):
    # A couple of misread frames reverting a claimed cell to UNCLAIMED -- exactly the
    # kind of flicker the splash-transition regression (see CLAUDE.md) produced -- should
    # not be enough to confirm an unclaim on its own, unlike a genuine, sustained undo.
    segment = [
        *_debounced(_board((0, 0, R))),
        *_debounced(_board(), count=extract._UNMARK_DEBOUNCE_SAMPLES - 1),
        *_debounced(_board((0, 0, R))),  # the true state reasserts itself
    ]
    with caplog.at_level(logging.WARNING):
        events = _extract_events(segment)

    assert [(e.row, e.col, e.color, e.event_type) for e in events] == [(1, 1, R, EventType.MARK)]
    assert not caplog.records


def test_direct_color_swap_is_logged_not_recorded_as_event(caplog):
    segment = [
        *_debounced(_board((0, 0, R))),
        *_debounced(_board((0, 0, B))),  # red -> blue with no intervening unclaim
    ]
    with caplog.at_level(logging.WARNING):
        events = _extract_events(segment)

    # The initial unclaimed -> red is a real claim; the direct red -> blue swap that
    # follows is logged as a data-quality warning rather than recorded as a second event.
    assert [(e.row, e.col, e.color, e.event_type) for e in events] == [(1, 1, R, EventType.MARK)]
    assert any("unexpected direct color swap" in r.message for r in caplog.records)


def test_winner_ignores_a_line_undone_by_unclaim():
    # Four reds already in a row; the fifth is claimed (completing the line) then
    # immediately unclaimed -- the line was never actually held, so no winner yet.
    base = [(0, c, R) for c in range(4)]
    segment = [
        *_debounced(_board(*base)),
        *_debounced(_board(*base, (0, 4, R))),
        *_debounced(_board(*base), count=extract._UNMARK_DEBOUNCE_SAMPLES),
    ]
    events = _extract_events(segment)
    winner_color, win_type, win_line = _determine_winner(events)
    assert winner_color is None
    from scadustats.models import WinType

    assert win_type is WinType.NONE
    assert win_line is None


def test_winner_names_the_line_it_was_completed_on():
    segment = _debounced(_board(*[(2, c, R) for c in range(5)]))

    winner_color, win_type, win_line = _determine_winner(_extract_events(segment))

    from scadustats.models import WinLine, WinType

    assert (winner_color, win_type, win_line) == (R, WinType.LINE, WinLine.ROW_3)


def test_detect_game_end_locates_the_mark_that_completes_the_winning_line():
    segment = _debounced(_board(*[(2, c, R) for c in range(5)]))
    events = _extract_events(segment)
    winner_color, win_type, _ = _determine_winner(events)

    game_end = _detect_game_end(events, winner_color, win_type)

    # The row is filled left to right within the same observation, so the last event in
    # the list is the one that actually completes it (col 5, 1-based).
    completing_mark = events[-1]
    assert (completing_mark.row, completing_mark.col) == (3, 5)

    assert game_end is not None
    assert game_end.event_type is EventType.GAME_END
    assert (game_end.row, game_end.col, game_end.color) == (None, None, None)
    assert game_end.video_ts_s == completing_mark.video_ts_s
    assert game_end.game_elapsed_s == completing_mark.game_elapsed_s


def test_detect_game_end_locates_the_settling_mark_of_a_majority_win():
    # Same board as rules.winner's own majority test: a red cell plus one blue cell (both
    # touching every row/column, the center covering both diagonals) in every line, so no
    # line can still be completed by one color; red holds more squares overall.
    layout = [
        "RBRRR",
        "BRRRR",
        "RRBRR",
        "RRRRB",
        "RRRBR",
    ]
    colors = [(r, c, R if layout[r][c] == "R" else B) for r in range(5) for c in range(5)]
    segment = _debounced(_board(*colors))
    events = _extract_events(segment)
    winner_color, win_type, _ = _determine_winner(events)
    assert (winner_color, win_type) == (R, WinType.MAJORITY)

    game_end = _detect_game_end(events, winner_color, win_type)

    assert game_end is not None
    assert game_end.event_type is EventType.GAME_END
    assert (game_end.row, game_end.col, game_end.color) == (None, None, None)
    # It's timestamped off one real mark event -- the one that locked the majority in.
    assert any(
        e.event_type is EventType.MARK
        and e.video_ts_s == game_end.video_ts_s
        and e.game_elapsed_s == game_end.game_elapsed_s
        for e in events
    )


def test_detect_game_end_returns_none_without_a_winner():
    segment = _debounced(_board((0, 0, R), (0, 1, B)))
    events = _extract_events(segment)
    winner_color, win_type, _ = _determine_winner(events)

    assert _detect_game_end(events, winner_color, win_type) is None


def test_detect_game_end_returns_none_when_the_win_was_undone():
    base = [(0, c, R) for c in range(4)]
    segment = [
        *_debounced(_board(*base)),
        *_debounced(_board(*base, (0, 4, R))),
        *_debounced(_board(*base), count=extract._UNMARK_DEBOUNCE_SAMPLES),
    ]
    events = _extract_events(segment)
    winner_color, win_type, _ = _determine_winner(events)

    assert _detect_game_end(events, winner_color, win_type) is None


def test_detect_game_start_finds_ascent_from_countdown_minimum():
    # A pre-game countdown ticking down to zero, then the real game clock ascending --
    # GAME_START is the sample at the bottom of that dip.
    board = _board()
    segment = [
        Observation(0, 0.0, board, 5),
        Observation(1, 1.0, board, 3),
        Observation(2, 2.0, board, 1),
        Observation(3, 3.0, board, 0),
        Observation(4, 4.0, board, 1),
        Observation(5, 5.0, board, 2),
    ]

    event = _detect_game_start(segment)

    assert event is not None
    assert event.event_type is EventType.GAME_START
    assert (event.row, event.col, event.color) == (None, None, None)
    assert event.game_elapsed_s == 0
    assert event.video_ts_s == 3.0


def test_detect_game_start_skips_a_minimum_the_clock_does_not_continue_from():
    # A segment can open on the tail of a previous screen's short countdown, which then
    # gives way to the game's own 3-minute one: ...2, 1, then straight to 180. That 1 is
    # a strict local minimum, but the +179 step out of it is the overlay swapping timers,
    # not a clock ticking up -- the real start is the bottom of the countdown that
    # follows.
    board = _board()
    segment = [
        Observation(0, 0.0, board, 3),
        Observation(1, 1.0, board, 2),
        Observation(2, 2.0, board, 1),
        Observation(3, 3.0, board, 180),
        Observation(4, 4.0, board, 179),
        Observation(5, 5.0, board, 0),
        Observation(6, 6.0, board, 1),
        Observation(7, 7.0, board, 2),
    ]

    event = _detect_game_start(segment)

    assert event is not None
    assert event.game_elapsed_s == 0
    assert event.video_ts_s == 5.0


def test_detect_game_start_returns_none_without_a_local_minimum():
    # No countdown captured -- the timer only ever ascends, so there's no dip to find.
    board = _board()
    segment = [
        Observation(0, 0.0, board, 0),
        Observation(1, 1.0, board, 1),
        Observation(2, 2.0, board, 2),
    ]

    assert _detect_game_start(segment) is None


def test_collect_observations_drops_samples_with_an_unreadable_timer(monkeypatch, caplog):
    # The splash between two games keeps the score bars colored (so is_gameplay_frame
    # still passes) while the grid and timer are both torn down -- reading cell colors
    # off it produced a burst of spurious unmarks that cost the finished game its
    # winner. An unreadable timer is the tell, so those samples don't become
    # observations at all.
    _stub_sampling(monkeypatch, [10, None, None, 12])

    with caplog.at_level(logging.WARNING):
        observations = _collect_observations("video.mp4", 1.0)

    assert [(obs.sample_index, obs.timer_s) for obs in observations] == [(0, 10), (3, 12)]
    # Half the samples gone is far past "the usual transition" and worth flagging.
    assert "timer unreadable on 2 of 4" in caplog.text


def test_collect_observations_does_not_warn_about_a_few_unreadable_timers(monkeypatch, caplog):
    _stub_sampling(monkeypatch, [*range(50), None, *range(50)])

    with caplog.at_level(logging.WARNING):
        observations = _collect_observations("video.mp4", 1.0)

    assert len(observations) == 100
    assert not caplog.records


def test_segment_games_splits_a_video_whose_samples_have_gaps():
    # Regression test for the whole-video-is-one-game bug: non-gameplay frames (the
    # recap between games) never become observations, so sample_index runs ahead of list
    # position -- slicing the observation list by sample_index put the split past the end
    # of the list and yielded a single segment spanning both games.
    observations = [
        Observation(0, 0.0, _board((0, 0, R)), 100),
        Observation(1, 1.0, _board((0, 0, R)), 101),
        # samples 2-49 were a recap screen and never became observations
        Observation(50, 50.0, _board(), 5),
        Observation(51, 51.0, _board(), 4),
    ]

    segments = _segment_games(observations)

    assert [[obs.video_ts_s for obs in segment] for segment in segments] == [
        [0.0, 1.0],
        [50.0, 51.0],
    ]


_MATCH_METADATA = MatchMetadata(
    match_date=datetime.date(2026, 3, 5), season="6", match_type=MatchType.ROUND_ROBIN
)


def test_match_id_formats_date_and_players():
    assert _match_id(_MATCH_METADATA, "blanxz", "Serious") == "2026-03-05-blanxz-vs-Serious"


def test_match_id_falls_back_to_unknown_for_missing_names():
    assert _match_id(_MATCH_METADATA, None, "") == "2026-03-05-unknown-vs-unknown"


def test_match_id_strips_slashes_from_names():
    assert _match_id(_MATCH_METADATA, "blanxz/2", "Serious") == "2026-03-05-blanxz-2-vs-Serious"
