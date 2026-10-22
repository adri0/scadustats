import datetime
import logging

import numpy as np

from scadustats.models import (
    CellColor,
    EventType,
    MatchMetadata,
    MatchType,
    Scoring,
    WinType,
    format_video_timestamp,
)
from scadustats.pipeline import extract
from scadustats.pipeline.extract import (
    _collect_observations,
    _detect_game_end,
    _detect_game_start,
    _determine_winner,
    _extract_events,
    _game_events,
    _match_id,
    _segment_games,
)
from scadustats.pipeline.segmentation import Observation
from scadustats.rules import winner

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
    monkeypatch.setattr(extract.board, "is_gameplay_frame", lambda frame, layout_: True)
    monkeypatch.setattr(extract.board, "cell_colors", lambda frame, layout_: _board())
    monkeypatch.setattr(extract.scoreboard, "read_scores", lambda frame, layout_: (None, None))
    monkeypatch.setattr(
        extract.timer, "read_timer", lambda frame, layout_: timer_readings[frame[0, 0, 0]]
    )


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
    assert game_end.video_timestamp == completing_mark.video_timestamp
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
        and e.video_timestamp == game_end.video_timestamp
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


def _clocked(*boards: list[list[CellColor]], start_s: int) -> list[Observation]:
    """One sample per board, a second apart, with the game clock ticking up from
    `start_s` -- for the samples around the time limit."""
    return [
        Observation(i, float(start_s + i), board, start_s + i) for i, board in enumerate(boards)
    ]


_LIMIT = extract._TIME_LIMIT_S


def _played(segment: list[Observation]):
    """A lockout game's events and result, as extract_video gets them from a segment."""
    events, stopped = _game_events(segment)
    game = _unsettled_game(events)
    if stopped is None:
        extract._settle_game(game, segment[-1], Scoring.LOCKOUT)
    else:
        extract._settle_game(game, stopped, Scoring.LOCKOUT, time_limit=True)
    return game.events, game.winner_color, game.win_type, game.win_line


def test_game_stopped_by_the_time_limit_is_decided_on_the_count():
    two_one = _board((0, 0, R), (2, 2, R), (4, 4, B))
    segment = _clocked(*[two_one] * 6, start_s=_LIMIT - 3)

    events, winner_color, win_type, win_line = _played(segment)

    assert (winner_color, win_type, win_line) == (R, WinType.MAJORITY, None)
    game_end = events[-1]
    assert game_end.event_type is EventType.GAME_END
    assert (game_end.row, game_end.col, game_end.color) == (None, None, None)
    # At the last sample kept: the one at the limit plus the mark debounce.
    stopped_s = _LIMIT + extract._MARK_DEBOUNCE_SAMPLES - 1
    assert game_end.game_elapsed_s == stopped_s
    assert game_end.video_timestamp == format_video_timestamp(stopped_s)
    assert winner.time_up(events)


def test_level_count_at_the_time_limit_is_a_tie():
    one_one = _board((0, 0, R), (4, 4, B))
    segment = _clocked(*[one_one] * 6, start_s=_LIMIT - 3)

    events, winner_color, win_type, _ = _played(segment)

    assert (winner_color, win_type) == (None, WinType.TIE)
    assert events[-1].event_type is EventType.GAME_END


def test_claim_after_the_time_limit_is_dropped():
    one_one = _board((0, 0, R), (4, 4, B))
    late = _board((0, 0, R), (4, 4, B), (1, 1, B), (3, 3, B))
    # The clock reaches the limit on the fourth sample; blue's two squares only show up
    # well after it.
    segment = _clocked(*[one_one] * 6, *[late] * 4, start_s=_LIMIT - 3)

    events, winner_color, win_type, _ = _played(segment)

    assert (winner_color, win_type) == (None, WinType.TIE)
    assert [(e.row, e.col) for e in events if e.event_type is EventType.MARK] == [
        (1, 1),
        (5, 5),
    ]


def test_claim_showing_when_the_clock_reaches_the_limit_counts():
    one_one = _board((0, 0, R), (4, 4, B))
    two_one = _board((0, 0, R), (2, 2, R), (4, 4, B))
    # Red's second square first shows on the sample where the clock reads the limit.
    segment = _clocked(*[one_one] * 3, *[two_one] * 4, start_s=_LIMIT - 3)

    _, winner_color, win_type, _ = _played(segment)

    assert (winner_color, win_type) == (R, WinType.MAJORITY)


def test_misread_timer_jump_to_the_limit_does_not_stop_the_game():
    two_one = _board((0, 0, R), (2, 2, R), (4, 4, B))
    segment = _clocked(*[two_one] * 6, start_s=600)
    # One sample reads two hours more than the clock really shows.
    segment[3] = Observation(3, segment[3].video_ts_s, two_one, segment[3].timer_s + _LIMIT)

    events, winner_color, win_type, _ = _played(segment)

    assert (winner_color, win_type) == (None, WinType.NONE)
    assert all(e.event_type is not EventType.GAME_END for e in events)


def test_game_settled_before_the_time_limit_keeps_its_own_game_end():
    row = _board(*[(2, c, R) for c in range(5)])
    segment = _clocked(*[row] * 4, start_s=600)

    events, winner_color, win_type, _ = _played(segment)

    assert (winner_color, win_type) == (R, WinType.LINE)
    assert [e.event_type for e in events].count(EventType.GAME_END) == 1
    assert events[-1].game_elapsed_s == 600 + extract._MARK_DEBOUNCE_SAMPLES - 1


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
    assert event.video_timestamp == format_video_timestamp(3.0)


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
    assert event.video_timestamp == format_video_timestamp(5.0)


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
        observations, _, _ = _collect_observations("video.mp4", 1.0)

    assert [(obs.sample_index, obs.timer_s) for obs in observations] == [(0, 10), (3, 12)]
    # Half the samples gone is far past "the usual transition" and worth flagging.
    assert "timer unreadable on 2 of 4" in caplog.text


def test_collect_observations_does_not_warn_about_a_few_unreadable_timers(monkeypatch, caplog):
    _stub_sampling(monkeypatch, [*range(50), None, *range(50)])

    with caplog.at_level(logging.WARNING):
        observations, _, _ = _collect_observations("video.mp4", 1.0)

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


def test_match_id_joins_a_team_name_with_underscores():
    assert (
        _match_id(_MATCH_METADATA, "Positive Vibes", "Bushy  Stache")
        == "2026-03-05-Positive_Vibes-vs-Bushy_Stache"
    )


def test_find_team_matches_the_overlays_capitals_and_a_stray_misread():
    from scadustats.models import Team

    roster = [
        Team(name="Positive Vibes", base_player="star0chris", dlc_player="Adef"),
        Team(name="Bushy Stache", base_player="Bushy", dlc_player="Vswed"),
    ]

    assert extract._find_team("POSITIVE VIBES", roster) is roster[0]
    assert extract._find_team("BUSHY STACHF", roster) is roster[1]
    assert extract._find_team("MONKEY BUSINESS", roster) is None
    assert extract._find_team("POSITIVE VIBES", []) is None


def _unsettled_game(events):
    from scadustats.models import GameResult, GameType

    return GameResult(
        game_index=1,
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        square_texts=[[""] * 5 for _ in range(5)],
        events=events,
        winner_color=None,
        win_type=WinType.NONE,
        game_type=GameType.BASE,
    )


def _square_event(row, col, color, ts):
    from scadustats.models import GameEvent

    return GameEvent(
        row=row,
        col=col,
        color=color,
        video_timestamp=format_video_timestamp(ts),
        game_elapsed_s=int(ts),
        event_type=EventType.MARK,
    )


def test_settle_game_decides_a_points_game_at_the_segments_last_reading():
    from scadustats.models import Scoring

    # Red completes row 1 early (7 points); blue claims 8 squares by the end (8 points).
    events = [_square_event(1, col, R, col) for col in range(1, 6)]
    events += [_square_event(2 + i // 4, 1 + i % 4, B, 10 + i) for i in range(8)]
    game = _unsettled_game(events)

    extract._settle_game(game, Observation(99, 3700.0, _board(), 3598), Scoring.POINTS)

    assert game.scoring is Scoring.POINTS
    assert (game.winner_color, game.win_type, game.win_line) == (B, WinType.POINTS, None)
    assert (game.red_points, game.blue_points) == (7, 8)
    assert game.events[-1].event_type is EventType.GAME_END
    assert game.events[-1].game_elapsed_s == 3598
    assert game.events[-1].video_timestamp == "01:01:40"


def test_settle_game_decides_a_lockout_game_on_its_line():
    from scadustats.models import Scoring, WinLine

    game = _unsettled_game([_square_event(1, col, R, col) for col in range(1, 6)])

    extract._settle_game(game, Observation(99, 100.0, _board(), 90), Scoring.LOCKOUT)

    assert (game.winner_color, game.win_type, game.win_line) == (R, WinType.LINE, WinLine.ROW_1)
    assert game.red_points is None and game.blue_points is None
    assert game.events[-1].event_type is EventType.GAME_END
    assert game.events[-1].video_timestamp == format_video_timestamp(5)


def test_text_scene_prefers_the_larger_grid_seen_long_enough():
    from scadustats.overlay import layout

    team, focus = layout.LAYOUT_SEASON_3_TEAM, layout.LAYOUT_SEASON_3_FOCUS
    segment = [Observation(i, float(i), _board(), i) for i in range(40)]
    # Mostly the team view, with the focus view for 12 samples.
    sample_layouts = {i: (focus if 10 <= i < 22 else team) for i in range(40)}

    scene, part = extract._text_scene(segment, sample_layouts, layout.LAYOUT_SEASON_3)
    assert scene is focus
    assert [obs.sample_index for obs in part] == list(range(10, 22))

    # Seen for too few samples to vote over, the larger grid loses to the most-seen scene.
    sample_layouts = {i: (focus if i < 3 else team) for i in range(40)}
    scene, part = extract._text_scene(segment, sample_layouts, layout.LAYOUT_SEASON_3)
    assert scene is team
    assert len(part) == 37


def test_text_scene_is_the_whole_segment_without_scenes():
    from scadustats.overlay import layout

    segment = [Observation(i, float(i), _board(), i) for i in range(5)]
    assert extract._text_scene(segment, {}, layout.LAYOUT_SEASON_4) == (
        layout.LAYOUT_SEASON_4,
        segment,
    )


def test_settle_game_ends_a_points_game_on_the_claim_that_reaches_13():
    from scadustats.models import Scoring

    # Red: row 1 (5 + 2 = 7), then six more squares -- the sixth, at 30s, makes 13.
    events = [_square_event(1, col, R, col) for col in range(1, 6)]
    events += [_square_event(2 + i // 4, 1 + i % 4, R, 25 + i) for i in range(6)]
    events += [_square_event(5, 5, B, 40)]
    game = _unsettled_game(events)

    extract._settle_game(game, Observation(99, 3700.0, _board(), 3598), Scoring.POINTS)

    assert (game.winner_color, game.win_type) == (R, WinType.POINTS)
    assert (game.red_points, game.blue_points) == (13, 1)
    assert game.events[-1].event_type is EventType.GAME_END
    assert game.events[-1].video_timestamp == format_video_timestamp(30)
