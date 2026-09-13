import datetime

from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameResult,
    GameType,
    MatchType,
    VideoExtraction,
    WinLine,
    WinType,
)
from scadustats.validation import validate_extraction, validate_game


def _mark(row: int, col: int, color: CellColor, ts: float) -> GameEvent:
    return GameEvent(
        row=row,
        col=col,
        color=color,
        video_ts_s=ts,
        game_elapsed_s=int(ts),
        event_type=EventType.MARK,
    )


def _unmark(row: int, col: int, color: CellColor, ts: float) -> GameEvent:
    return GameEvent(
        row=row,
        col=col,
        color=color,
        video_ts_s=ts,
        game_elapsed_s=int(ts),
        event_type=EventType.UNMARK,
    )


def _game_start(ts: float = 0.0) -> GameEvent:
    return GameEvent(
        row=None,
        col=None,
        color=None,
        video_ts_s=ts,
        game_elapsed_s=0,
        event_type=EventType.GAME_START,
    )


def _line_win_events(color: CellColor = CellColor.RED, start_ts: float = 1.0) -> list[GameEvent]:
    """A game_start plus the top row claimed by one color -- the minimum that replays to
    a genuine line win."""
    return [_game_start(), *(_mark(0, col, color, start_ts + col) for col in range(5))]


def _game(game_index: int = 1, **overrides) -> GameResult:
    defaults = dict(
        game_index=game_index,
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        square_texts=[[f"goal {r}-{c}" for c in range(5)] for r in range(5)],
        events=_line_win_events(),
        winner_color=CellColor.RED,
        win_type=WinType.LINE,
        # _line_win_events claims the top row, whichever color it's asked for.
        win_line=WinLine.ROW_0,
        game_type=GameType.BASE if game_index == 1 else GameType.DLC,
    )
    defaults.update(overrides)
    return GameResult(**defaults)


def _extraction(**overrides) -> VideoExtraction:
    defaults = dict(
        video_id="2026-03-05-alice-vs-bob",
        video_url=None,
        match_date=datetime.date(2026, 3, 5),
        season=6,
        match_type=MatchType.PLAYOFFS,
        player_red_name="alice",
        player_blue_name="bob",
        extracted_at=datetime.date(2026, 3, 6),
        games=[
            _game(1),
            _game(2, events=_line_win_events(CellColor.BLUE), winner_color=CellColor.BLUE),
        ],
    )
    defaults.update(overrides)
    return VideoExtraction(**defaults)


def _codes(issues) -> list[str]:
    return [issue.code for issue in issues]


def test_a_clean_playoffs_match_has_no_issues():
    assert validate_extraction(_extraction()) == []


def test_a_clean_double_elimination_sweep_has_no_issues():
    """Two games to the same player ends a best-of-3 -- no decider is expected."""
    extraction = _extraction(
        match_type=MatchType.DOUBLE_ELIMINATION,
        games=[_game(1), _game(2, game_type=GameType.DLC)],
    )

    assert validate_extraction(extraction) == []


def test_playoffs_match_must_have_exactly_two_games():
    issues = validate_extraction(_extraction(games=[_game(1)]))

    assert _codes(issues) == ["game_count"]
    assert "found 1" in issues[0].message


def test_double_elimination_match_must_have_two_or_three_games():
    extraction = _extraction(
        match_type=MatchType.DOUBLE_ELIMINATION,
        games=[
            _game(1),
            _game(2, game_type=GameType.DLC),
            _game(3, game_type=GameType.BASE),
            _game(4, game_type=GameType.DLC),
        ],
    )

    assert "game_count" in _codes(validate_extraction(extraction))


def test_first_two_games_must_be_base_then_dlc():
    extraction = _extraction(
        games=[
            _game(1, game_type=GameType.DLC),
            _game(
                2,
                game_type=GameType.BASE,
                events=_line_win_events(CellColor.BLUE),
                winner_color=CellColor.BLUE,
            ),
        ]
    )

    issues = validate_extraction(extraction)

    assert _codes(issues) == ["opening_game_type", "opening_game_type"]
    assert "game 1 should be base, found dlc" in issues[0].message


def test_a_third_game_may_be_either_type():
    """Only the decider is free -- games 1 and 2 are still checked."""
    extraction = _extraction(
        match_type=MatchType.DOUBLE_ELIMINATION,
        games=[
            _game(1),
            _game(
                2,
                game_type=GameType.DLC,
                events=_line_win_events(CellColor.BLUE),
                winner_color=CellColor.BLUE,
            ),
            _game(3, game_type=GameType.BASE),
        ],
    )

    assert validate_extraction(extraction) == []


def test_a_game_without_a_game_type_is_reported():
    issues = validate_extraction(_extraction(games=[_game(1, game_type=None), _game(2)]))

    assert "opening_game_type" in _codes(issues)
    assert "found no game type" in issues[0].message


def test_a_game_without_a_winner_leaves_the_match_outcome_undetermined():
    extraction = _extraction(
        games=[
            _game(1, events=[_game_start()], winner_color=None, win_type=WinType.NONE),
            _game(2),
        ]
    )

    issues = validate_extraction(extraction)

    assert "match_outcome_undetermined" in _codes(issues)
    assert "game 1" in next(i for i in issues if i.code == "match_outcome_undetermined").message


def test_an_undetermined_outcome_suppresses_the_decider_rule():
    """With the score unknown, "a decider is missing" would be a guess -- and would bury
    the one real problem under a second, derived complaint."""
    extraction = _extraction(
        match_type=MatchType.DOUBLE_ELIMINATION,
        games=[
            _game(1, events=[_game_start()], winner_color=None, win_type=WinType.NONE),
            _game(2, game_type=GameType.DLC),
        ],
    )

    codes = _codes(validate_extraction(extraction))

    assert "match_outcome_undetermined" in codes
    assert "missing_decider_game" not in codes


def test_split_double_elimination_games_require_a_decider():
    extraction = _extraction(
        match_type=MatchType.DOUBLE_ELIMINATION,
        games=[
            _game(1),
            _game(
                2,
                game_type=GameType.DLC,
                events=_line_win_events(CellColor.BLUE),
                winner_color=CellColor.BLUE,
            ),
        ],
    )

    assert _codes(validate_extraction(extraction)) == [
        "double_elimination_draw",
        "missing_decider_game",
    ]


def test_a_decided_double_elimination_match_needs_no_third_game():
    extraction = _extraction(
        match_type=MatchType.DOUBLE_ELIMINATION,
        games=[_game(1), _game(2, game_type=GameType.DLC), _game(3, game_type=GameType.DLC)],
    )

    assert _codes(validate_extraction(extraction)) == ["unnecessary_decider_game"]


def test_playoffs_may_end_one_apiece():
    """A 1-1 playoffs match is a draw, not a rule violation -- unlike double
    elimination."""
    assert validate_extraction(_extraction()) == []


def test_recorded_winner_must_match_the_replayed_board():
    """Four marks in a row is no line -- the recorded LINE win has nothing behind it."""
    events = [_game_start(), *(_mark(0, col, CellColor.RED, 1.0 + col) for col in range(4))]
    issues = validate_game(_game(events=events))

    assert _codes(issues) == ["winner_board_mismatch"]
    assert "red (line on row 0)" in issues[0].message
    assert "no winner (none)" in issues[0].message


def test_a_missing_win_line_is_reported_under_its_own_code():
    """A file written before win_line existed agrees with its board in every other
    respect -- that's a recoverable gap, not the board contradicting the result."""
    issues = validate_game(_game(win_line=None))

    assert _codes(issues) == ["win_line_not_recorded"]
    assert "row 0" in issues[0].message


def test_a_win_recorded_on_the_wrong_line_is_reported():
    """The board holds the top row, not the left column -- the same defect as a win
    recorded for the wrong color, so it's the same rule."""
    issues = validate_game(_game(win_line=WinLine.COL_0))

    assert _codes(issues) == ["winner_board_mismatch"]
    assert "red (line on column 0)" in issues[0].message
    assert "red (line on row 0)" in issues[0].message


def test_a_line_completed_and_then_undone_is_not_a_win():
    events = [
        *_line_win_events(),
        _unmark(0, 4, CellColor.RED, 10.0),
    ]

    assert _codes(validate_game(_game(events=events))) == ["winner_board_mismatch"]


def test_game_must_have_exactly_one_game_start():
    assert _codes(validate_game(_game(events=_line_win_events()[1:]))) == ["game_start_count"]

    doubled = [_game_start(0.0), _game_start(0.5), *_line_win_events()[1:]]
    issues = validate_game(_game(events=doubled))
    assert _codes(issues) == ["game_start_count"]
    assert "found 2" in issues[0].message


def test_marks_after_a_line_win_are_reported():
    events = [*_line_win_events(), _mark(1, 1, CellColor.BLUE, 20.0)]

    issues = validate_game(_game(events=events))

    assert _codes(issues) == ["mark_after_win"]
    assert "1 square(s) marked" in issues[0].message


def test_a_mark_before_a_transient_win_is_not_counted_as_after_it():
    """The line at ts=5 is undone at ts=6 and only re-completed at ts=8, so the mark at
    ts=7 happened while the game was still live."""
    events = [
        *_line_win_events(),
        _unmark(0, 4, CellColor.RED, 6.0),
        _mark(1, 1, CellColor.BLUE, 7.0),
        _mark(0, 4, CellColor.RED, 8.0),
    ]

    assert validate_game(_game(events=events)) == []


def test_marks_after_a_majority_win_are_allowed():
    """A majority win is decided when time runs out, and every line is typically blocked
    long before that -- players keep claiming squares in between, so the rule only
    applies to line wins."""
    # Every square claimed, every line blocked, red ahead 13-12. The middle square breaks
    # up the two diagonals a plain alternating pattern would hand to one color.
    layout = [
        "BRBRB",
        "RBRBR",
        "BRRRB",
        "RBRBR",
        "BRBRB",
    ]
    events = [
        _game_start(),
        *(
            _mark(
                r,
                c,
                CellColor.RED if layout[r][c] == "R" else CellColor.BLUE,
                float(r * 5 + c + 1),
            )
            for r in range(5)
            for c in range(5)
        ),
    ]

    majority_game = _game(
        events=events,
        winner_color=CellColor.RED,
        win_type=WinType.MAJORITY,
        # A majority win has no line to name; leaving the fixture's default in place is
        # itself a winner_board_mismatch, which the next test pins down.
        win_line=None,
    )

    assert validate_game(majority_game) == []


def test_a_non_line_win_may_not_name_a_line():
    majority_win = _game(win_type=WinType.MAJORITY, win_line=WinLine.ROW_0)

    assert _codes(validate_game(majority_win)) == ["winner_board_mismatch"]


def test_games_are_validated_in_index_order_after_the_match_rules():
    extraction = _extraction(
        games=[
            # Deliberately out of index order, and each game missing its game_start.
            _game(2, events=_line_win_events()[1:], winner_color=CellColor.RED),
            _game(1, game_type=None, events=_line_win_events()[1:]),
        ]
    )

    issues = validate_extraction(extraction)

    assert [issue.scope for issue in issues] == ["match", "game 1", "game 2"]
    assert issues[0].code == "opening_game_type"


def test_issue_scope_labels_match_and_game_level_rules():
    extraction = _extraction(games=[_game(1)])

    issues = validate_extraction(extraction)

    assert issues[0].scope == "match"
    assert issues[0].game_index is None
