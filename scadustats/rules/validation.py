"""Rule checks over one already-extracted match (a models.VideoExtraction).

This is a review aid, not a gate: extraction never calls it, and a failing rule doesn't
mean the video is bad -- it means the JSON says something the tournament's own rules say
can't happen, so an OCR misread or a missed game boundary probably slipped through and a
contributor should look at that file (see `match validate` in cli/app.py). The checks are
therefore written to point at *what* looks wrong rather than to guess a correction.

Two scopes, since a match is one video but several games (see models.VideoExtraction):
match-level rules are about the games as a set (how many, in what order, who took the
match), game-level rules are about one game's own events and board.
"""

from dataclasses import dataclass
from pathlib import Path

from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameResult,
    GameType,
    MatchType,
    MatchWinner,
    Scoring,
    VideoExtraction,
    WinLine,
    WinType,
    square_type_tag,
)
from scadustats.rules import winner


@dataclass(frozen=True)
class SeasonFormat:
    """How a season's matches are played, as far as the match-level rules care. Each
    season sets its own: the format can't be told from a match's games alone, since a
    Season 6 match that lost its base game to a missed boundary looks exactly like a
    Season 4 one (one DLC game).

    `game_types` is every game type the season's boards can be, `round_robin_games` the
    exact number of games in one of its round robin matches, and `opening_game_types`
    the game type each of the first games must be, in order (empty: not checked). A
    best of 3 is 2 or 3 games in every season, so it isn't part of the format.
    `scoring` is how every game of the season is decided, round robin and best of 3
    alike (see models.Scoring)."""

    name: str
    game_types: frozenset[GameType]
    round_robin_games: int
    opening_game_types: tuple[GameType, ...] = ()
    scoring: Scoring = Scoring.DEFAULT


# 1v1, a base game then a DLC game; only a best-of-3 decider (game 3) is free to be either.
_BASE_THEN_DLC = SeasonFormat(
    name="1v1",
    game_types=frozenset({GameType.BASE, GameType.DLC}),
    round_robin_games=2,
    opening_game_types=(GameType.BASE, GameType.DLC),
)

# Keyed by VideoExtraction.season. A season missing here is reported (unknown_season)
# and the rules that depend on its format are skipped.
SEASON_FORMATS: dict[str, SeasonFormat] = {
    # 2v2, both teammates on the same base game board (no DLC yet); every game, playoffs
    # included, is a race to 13 points (see models.Scoring).
    "3": SeasonFormat(
        name="2v2 base game",
        game_types=frozenset({GameType.BASE}),
        round_robin_games=1,
        scoring=Scoring.POINTS,
    ),
    "4": SeasonFormat(
        name="DLC-only 1v1", game_types=frozenset({GameType.DLC}), round_robin_games=1
    ),
    "5": SeasonFormat(name="2v2", game_types=frozenset({GameType.TWO_V_TWO}), round_robin_games=1),
    "pre-season-6": _BASE_THEN_DLC,
    "6": _BASE_THEN_DLC,
}


def scoring_for(season: str) -> Scoring:
    """How a game of this season is decided: the season's scoring, or line for a
    season with no known format."""
    format_ = SEASON_FORMATS.get(season)
    return format_.scoring if format_ is not None else Scoring.DEFAULT


@dataclass(frozen=True)
class ValidationIssue:
    """`code` is a stable, greppable identifier for the rule that failed; `message` is
    the human-facing explanation of this particular failure. `game_index` is None for a
    match-level rule -- including a match-level rule whose message happens to name a game
    (e.g. "the match outcome is undetermined because game 1 has no winner": the defect is
    in the match as a whole, not in that game's own data)."""

    code: str
    message: str
    game_index: int | None = None

    @property
    def scope(self) -> str:
        return "match" if self.game_index is None else f"game {self.game_index}"


Board = list[list[CellColor]]


def _ordered_events(game: GameResult) -> list[GameEvent]:
    """Events in the order they actually happened. Sorted by video timestamp rather than
    game_elapsed_s: the latter is the overlay's stopwatch, which runs a pre-game countdown
    *down* before the game clock runs up (see extract._detect_game_start), so sorting by
    it would put a GAME_START event in the wrong place relative to the claims.
    """
    return sorted(game.events, key=lambda event: event.video_timestamp)


def _check_recorded_winner_matches_board(
    game: GameResult, states: list[Board]
) -> list[ValidationIssue]:
    """Rule: the recorded winner, win type and winning line must be what the game's own
    events replay to. A mismatch means the two disagree about the same game -- either the
    events are incomplete (a claim missed at extraction time) or the result was
    hand-edited without the board behind it. The line is part of the check, not a
    separate rule: a win recorded on a line the board doesn't hold is the same defect as
    a win recorded for the wrong color. A game that ends on its GAME_END ran until it was
    called, so an unsettled board there is decided on the count (winner.time_up)."""
    final = states[-1] if states else winner.empty_board()
    expected = winner.determine_winner(
        final, time_up=winner.time_up(game.events), scoring=game.scoring
    )
    if (game.winner_color, game.win_type, game.win_line) == expected:
        return []

    # A win the board agrees with, minus the line it was won on, isn't a disagreement --
    # it's a JSON file written before win_line was recorded (or hand-edited to drop it).
    # Worth reporting, since the field is recoverable and missing, but under its own code
    # so it doesn't read as the board contradicting the result.
    expected_color, expected_type, expected_line = expected
    if (
        game.win_line is None
        and expected_line is not None
        and (game.winner_color, game.win_type) == (expected_color, expected_type)
    ):
        return [
            ValidationIssue(
                code="win_line_not_recorded",
                message=(
                    f"no win_line recorded for this line win -- the board gives "
                    f"{expected_line.label}; re-extract the video to fill it in"
                ),
                game_index=game.game_index,
            )
        ]

    return [
        ValidationIssue(
            code="winner_board_mismatch",
            message=(
                f"recorded result is "
                f"{_describe_result(game.winner_color, game.win_type, game.win_line)}, "
                f"but replaying this game's events gives {_describe_result(*expected)}"
            ),
            game_index=game.game_index,
        )
    ]


def _check_points(game: GameResult, states: list[Board]) -> list[ValidationIssue]:
    """Rule: a points game's recorded points are what its final board scores (see
    winner.points), and a default game records none."""
    if game.scoring is not Scoring.POINTS:
        if game.red_points is None and game.blue_points is None:
            return []
        return [
            ValidationIssue(
                code="points_mismatch",
                message="a default game records no points",
                game_index=game.game_index,
            )
        ]
    final = states[-1] if states else winner.empty_board()
    expected = winner.points(final)
    if (game.red_points, game.blue_points) == expected:
        return []
    return [
        ValidationIssue(
            code="points_mismatch",
            message=(
                f"recorded points are red {game.red_points}, blue {game.blue_points}, but "
                f"the final board scores red {expected[0]}, blue {expected[1]}"
            ),
            game_index=game.game_index,
        )
    ]


def _check_events_sorted(game: GameResult) -> list[ValidationIssue]:
    """Rule: events must be stored in ascending video_timestamp order -- the order a
    contributor hand-reviewing the JSON reads them in, and the order chronological
    replay assumes. Checked against the raw, on-disk list (game.events) rather than the
    chronologically reordered copy every other check here works from (_ordered_events):
    this rule is specifically about that reordering being unnecessary in the first
    place."""
    timestamps = [event.video_timestamp for event in game.events]
    if timestamps == sorted(timestamps):
        return []
    return [
        ValidationIssue(
            code="events_not_sorted",
            message="events are not sorted by ascending video_timestamp",
            game_index=game.game_index,
        )
    ]


def _check_first_event_is_game_start(
    game: GameResult, events: list[GameEvent]
) -> list[ValidationIssue]:
    """Rule: the first event in chronological order must be game_start -- anything
    claimed or unclaimed before the game clock is confirmed running (see
    extract._detect_game_start) means the segment boundary is probably wrong, or
    game_start itself was never detected."""
    if not events or events[0].event_type is EventType.GAME_START:
        return []
    return [
        ValidationIssue(
            code="first_event_not_game_start",
            message=(
                f"expected the first event to be game_start, found {events[0].event_type.value}"
            ),
            game_index=game.game_index,
        )
    ]


def _check_last_event_is_game_end(
    game: GameResult, events: list[GameEvent], states: list[Board]
) -> list[ValidationIssue]:
    """Rule: a game that ends on a completed line should end on its own GAME_END marker
    (see extract._detect_game_end). Only checked for a LINE win, the same exemption
    _check_no_marks_after_win has and for the same reason: a MAJORITY win is decided the
    instant a lead becomes mathematically unbeatable, which -- unlike a completed line
    -- isn't a structural stopping point, so marks can legitimately continue being
    recorded after it and there's no guarantee the last event is (or should be) its
    GAME_END. Also skipped, the same as that rule, when the settling event can't be
    found or wasn't itself a MARK (an UNMARK can't complete a line) -- there's nothing
    for extract._detect_game_end to have emitted a GAME_END for in the first place.

    A points game (Scoring.POINTS) always ends on its GAME_END: at the claim that took a
    side to winner.POINTS_TO_WIN, or where the clock stopped if neither got there."""
    if game.scoring is Scoring.POINTS:
        if events and events[-1].event_type is EventType.GAME_END:
            return []
        found = events[-1].event_type.value if events else "no events"
        return [
            ValidationIssue(
                code="last_event_not_game_end",
                message=f"a points game ends on game_end where the clock stopped, found {found}",
                game_index=game.game_index,
            )
        ]
    if game.win_type is not WinType.LINE or game.winner_color is None:
        return []
    settled = winner.settled_result_index(states, game.winner_color, WinType.LINE)
    if settled is None or events[settled].event_type is not EventType.MARK:
        return []
    if not events or events[-1].event_type is EventType.GAME_END:
        return []
    return [
        ValidationIssue(
            code="last_event_not_game_end",
            message=(
                f"expected the last event to be game_end, found {events[-1].event_type.value}"
            ),
            game_index=game.game_index,
        )
    ]


def _check_game_timer_monotonic(game: GameResult, events: list[GameEvent]) -> list[ValidationIssue]:
    """Rule: the overlay's stopwatch must never go backward across a game's events in
    chronological order -- game_timer in the JSON (GameEvent.game_elapsed_s on the
    model; see json_export._dict_to_event/_event_to_dict for the reshaping between the
    two). The pre-game countdown that ticks down is over by game_start (see
    extract._detect_game_start), so every event from there on should only ever read a
    later, larger clock value -- a decrease means two events were misordered or the
    timer was misread on one of them. Reports only the first decrease found, the same
    "point at the one real problem" shape _check_game_count etc. already have. The
    message names game_timer, not game_elapsed_s: that's the key a contributor
    hand-reviewing the JSON actually sees."""
    for prev, curr in zip(events, events[1:], strict=False):
        if curr.game_elapsed_s < prev.game_elapsed_s:
            return [
                ValidationIssue(
                    code="game_timer_not_monotonic",
                    message=(
                        f"game_timer went from {prev.game_elapsed_s}s at "
                        f"{prev.video_timestamp} to {curr.game_elapsed_s}s at "
                        f"{curr.video_timestamp}"
                    ),
                    game_index=game.game_index,
                )
            ]
    return []


def _check_single_game_start(game: GameResult) -> list[ValidationIssue]:
    """Rule: exactly one GAME_START event per game. Zero means the clock's countdown-to-
    game-clock turn was never seen (footage starting mid-game, or a missed dip); more than
    one means two games were probably merged into one segment."""
    count = sum(event.event_type is EventType.GAME_START for event in game.events)
    if count == 1:
        return []
    return [
        ValidationIssue(
            code="game_start_count",
            message=f"expected exactly 1 game_start event, found {count}",
            game_index=game.game_index,
        )
    ]


def _check_no_marks_after_win(
    game: GameResult, events: list[GameEvent], states: list[Board]
) -> list[ValidationIssue]:
    """Rule: nothing is claimed once the game has been won.

    Only checked for a LINE win, and for a points game won by reaching
    winner.POINTS_TO_WIN. Either ends the game the moment it happens, so a later mark
    means the segment ran past the end of its game -- but a MAJORITY win is decided by
    who's ahead when time runs out, and the board's lines are typically all blocked long
    before that, so players legitimately keep claiming squares afterward.
    """
    if game.winner_color is None:
        return []
    if game.scoring is Scoring.POINTS and game.win_type is WinType.POINTS:
        # Reaching winner.POINTS_TO_WIN ends a points game the way a completed line ends a
        # default game. A points game won on time has no such moment (settled is None below).
        settled = winner.settled_result_index(
            states, game.winner_color, WinType.POINTS, Scoring.POINTS
        )
        what = f"reached {winner.POINTS_TO_WIN} points"
    elif game.win_type is WinType.LINE:
        settled = winner.settled_result_index(states, game.winner_color, WinType.LINE)
        # Falls back to "a line" only when win_line is missing from an otherwise line-won
        # game -- _check_recorded_winner_matches_board reports that separately.
        what = f"completed {game.win_line.label if game.win_line else 'a line'}"
    else:
        return []
    if settled is None:
        # The board doesn't end on a line at all -- _check_recorded_winner_matches_board
        # already reports that, and re-reporting it here would just be noise.
        return []

    after = [event for event in events[settled + 1 :] if event.event_type is EventType.MARK]
    if not after:
        return []
    return [
        ValidationIssue(
            code="mark_after_win",
            message=(
                f"{len(after)} square(s) marked after {game.winner_color.value} {what} "
                f"at {events[settled].video_timestamp} "
                f"(first at {after[0].video_timestamp})"
            ),
            game_index=game.game_index,
        )
    ]


def _check_square_type_tags(game: GameResult) -> list[ValidationIssue]:
    """Rule: on a 2v2 mixed board (TWO_V_TWO), every square's text ends in its pool tag
    -- " (B)" base game, " (D)" DLC or " (C)" co-op, exactly as the overlay prints it.
    The tag is the only record of which pool a square is in (see
    models.GameResult.cell_goal): an untagged square isn't counted toward any square's
    stats or any player's marks, so it's reported rather than guessed."""
    if game.game_type is not GameType.TWO_V_TWO:
        return []
    untagged = [
        f"[{r + 1},{c + 1}]"
        for r, row in enumerate(game.square_texts)
        for c, text in enumerate(row)
        if square_type_tag(text) is None
    ]
    if not untagged:
        return []
    return [
        ValidationIssue(
            code="square_type_tag_missing",
            message=(
                f"{len(untagged)} square(s) don't end in a (B)/(D)/(C) tag: {', '.join(untagged)}"
            ),
            game_index=game.game_index,
        )
    ]


def _describe_result(color: CellColor | None, win_type: WinType, win_line: WinLine | None) -> str:
    line = f" on {win_line.label}" if win_line else ""
    return f"{color.value if color else 'no winner'} ({win_type.value}{line})"


def validate_game(game: GameResult) -> list[ValidationIssue]:
    events = _ordered_events(game)
    states = winner.board_states(events)
    return [
        *_check_recorded_winner_matches_board(game, states),
        *_check_points(game, states),
        *_check_single_game_start(game),
        *_check_first_event_is_game_start(game, events),
        *_check_last_event_is_game_end(game, events, states),
        *_check_events_sorted(game),
        *_check_game_timer_monotonic(game, events),
        *_check_no_marks_after_win(game, events, states),
        *_check_square_type_tags(game),
    ]


def _check_season_known(extraction: VideoExtraction) -> list[ValidationIssue]:
    """Rule: the match's season has a known format (see SEASON_FORMATS). Without one the
    game types, game count and opening game types can't be checked, so those rules are
    skipped rather than guessing the format from the games."""
    if extraction.season in SEASON_FORMATS:
        return []
    return [
        ValidationIssue(
            code="unknown_season",
            message=(
                f"season {extraction.season!r} has no known format, so its game types "
                "and game count aren't checked"
            ),
        )
    ]


def _check_season_game_types(
    extraction: VideoExtraction, season: SeasonFormat
) -> list[ValidationIssue]:
    """Rule: every game is one of the season's game types -- a DLC board in Season 4, a
    2v2 mixed board in Season 5, a base game or DLC board otherwise. Anything else is a
    misread or mis-entered game type."""
    allowed = ", ".join(sorted(game_type.value for game_type in season.game_types))
    issues = []
    for game in extraction.games:
        if game.game_type in season.game_types:
            continue
        found = game.game_type.value if game.game_type else "no game type"
        issues.append(
            ValidationIssue(
                code="season_game_type",
                message=(
                    f"game {game.game_index} is {found}, but season {extraction.season} "
                    f"games are {allowed}"
                ),
            )
        )
    return issues


def _check_game_count(extraction: VideoExtraction, season: SeasonFormat) -> list[ValidationIssue]:
    """Rule: how many games a match has depends on its type and its season's format. A
    round robin match is exactly the season's round_robin_games: 2 in Season 6 (a base
    game and a DLC game), 1 in Season 5 (2v2) and Season 4 (DLC only). A best of 3 is 2 or
    3 games in every season. A wrong count is the loudest sign of a segmentation problem
    -- games merged together, a game lost to a missed boundary, or a recap segment read as
    a game of its own."""
    count = extraction.num_games
    if extraction.match_type is MatchType.ROUND_ROBIN:
        n = season.round_robin_games
        expected = f"exactly {n} game{'s' if n != 1 else ''}"
        ok = count == n
    else:
        expected = "2 or 3 games (best of 3)"
        ok = count in (2, 3)
    if ok:
        return []
    return [
        ValidationIssue(
            code="game_count",
            message=(
                f"a {season.name} {extraction.match_type.value} match has {expected}, found {count}"
            ),
        )
    ]


def _check_opening_game_types(
    extraction: VideoExtraction, season: SeasonFormat
) -> list[ValidationIssue]:
    """Rule: the first games are the season's opening_game_types, in order -- game 1 the
    base game and game 2 DLC in a 1v1 base + DLC season. (A best-of-3 decider -- game 3 --
    can be either, so it isn't checked.) Nothing to check in a season with a single game
    type (Season 4, Season 5). A game whose type isn't one of the season's at all (or is
    missing) is _check_season_game_types' issue, not reported again here."""
    issues = []
    for game, expected in zip(extraction.games, season.opening_game_types, strict=False):
        if game.game_type is expected or game.game_type not in season.game_types:
            continue
        found = game.game_type.value
        issues.append(
            ValidationIssue(
                code="opening_game_type",
                message=f"game {game.game_index} should be {expected.value}, found {found}",
            )
        )
    return issues


def _check_season_scoring(extraction: VideoExtraction) -> list[ValidationIssue]:
    """Rule: every game is scored the way its season is (see scoring_for) -- points
    in Season 3, default everywhere else. A game
    scored the other way has the wrong result rules applied to it."""
    expected = scoring_for(extraction.season)
    return [
        ValidationIssue(
            code="season_scoring",
            message=(
                f"game {game.game_index} is a {game.scoring.value} game, but a season "
                f"{extraction.season} game is a {expected.value} game"
            ),
        )
        for game in extraction.games
        if game.scoring is not expected
    ]


def _check_season_format(extraction: VideoExtraction) -> list[ValidationIssue]:
    """The rules that depend on the season's format, or unknown_season alone when the
    season has none."""
    season = SEASON_FORMATS.get(extraction.season)
    if season is None:
        return _check_season_known(extraction)
    return [
        *_check_season_game_types(extraction, season),
        *_check_season_scoring(extraction),
        *_check_game_count(extraction, season),
        *_check_opening_game_types(extraction, season),
    ]


def _check_match_outcome(extraction: VideoExtraction) -> list[ValidationIssue]:
    """Rule: the match must resolve to a winner or (round robin only) a draw.

    Every game has to have a winner for that to be decidable, so an undecided game is
    reported here and stops the rules below it: with the score unknown, "a decider was
    missing" or "this was a draw" would both be guesses, and reporting them anyway would
    bury the one real problem under cascading noise.
    """
    undecided = [game for game in extraction.games if game.winner_color is None]
    if undecided:
        return [
            ValidationIssue(
                code="match_outcome_undetermined",
                message=(
                    "the match outcome can't be determined: no winner recorded for "
                    + ", ".join(f"game {game.game_index}" for game in undecided)
                ),
            )
        ]

    # A round robin match is allowed to end 1-1 (the issue's rule 2 calls that a draw), so
    # only double elimination has anything left to check here. Every game has a winner by
    # this point, so extraction.winner is guaranteed non-None.
    if (
        extraction.match_type is MatchType.DOUBLE_ELIMINATION
        and extraction.winner is MatchWinner.DRAW
    ):
        return [
            ValidationIssue(
                code="double_elimination_draw",
                message=(
                    "a double_elimination match can't end in a draw, but the games "
                    f"split {extraction.red_score}-{extraction.blue_score}"
                ),
            )
        ]
    return []


def _check_decider_game(extraction: VideoExtraction) -> list[ValidationIssue]:
    """Rule: a double-elimination match plays a third game exactly when the first two
    were split. Skipped unless the game count and every game's winner are themselves
    valid -- this rule is derived from both, so it can only repeat their complaints."""
    games = extraction.games
    if extraction.match_type is not MatchType.DOUBLE_ELIMINATION:
        return []
    if len(games) not in (2, 3) or any(game.winner_color is None for game in games):
        return []

    split = games[0].winner_color is not games[1].winner_color
    if split and len(games) == 2:
        return [
            ValidationIssue(
                code="missing_decider_game",
                message=(
                    "the first two games were split, so a third (decider) game should "
                    "follow, but the match has only 2"
                ),
            )
        ]
    if not split and len(games) == 3:
        return [
            ValidationIssue(
                code="unnecessary_decider_game",
                message=(
                    f"{games[0].winner_color.value} won the first two games, so the match "
                    "was already over, but a third game follows"
                ),
            )
        ]
    return []


def _check_match_id_matches_filename(
    extraction: VideoExtraction, path: Path | None
) -> list[ValidationIssue]:
    """Rule: the file's own name must match the match_id recorded inside it.
    json_export.write_video always names a fresh extraction's file after its match_id
    (see json_export.video_path), so a mismatch means the file was renamed, or match_id
    was hand-edited, after the fact -- and every command that looks a match up by
    filename (match show, match validate <match_id>, square/player consolidate
    <match_id> -- see cli._match_path) would then silently return a VideoExtraction
    whose own match_id disagrees with the id that was actually asked for.

    Only checked when a path is given: nothing about the JSON payload itself says what
    file it lives in, so a caller with no path in hand (a VideoExtraction built
    in-memory, e.g. in tests) gets no complaint from this rule.
    """
    if path is None or path.stem == extraction.match_id:
        return []
    return [
        ValidationIssue(
            code="match_id_filename_mismatch",
            message=f"file is named {path.name!r} but match_id is {extraction.match_id!r}",
        )
    ]


def validate_extraction(
    extraction: VideoExtraction, path: Path | None = None
) -> list[ValidationIssue]:
    """Every rule violation found in one extracted match, match-level rules first and
    then each game's own, in game order. An empty list means the match looks valid.

    `path` is the file this extraction was read from, if known -- passed through only
    for _check_match_id_matches_filename, which is the one rule here that needs to know
    anything about the filesystem rather than just the JSON payload.

    The extraction isn't mutated -- games are sorted into index order for checking (the
    order rules only mean anything against that) without touching the caller's object.
    model_copy(update=...), not dataclasses.replace: VideoExtraction is a pydantic model
    (see CLAUDE.md), and update= is a shallow, unvalidated copy -- fine here since the
    replacement value (a re-sort of the caller's own already-valid games list) needs no
    re-validation.
    """
    ordered = extraction.model_copy(
        update={"games": sorted(extraction.games, key=lambda g: g.game_index)}
    )
    issues = [
        *_check_match_id_matches_filename(ordered, path),
        *_check_season_format(ordered),
        *_check_match_outcome(ordered),
        *_check_decider_game(ordered),
    ]
    for game in ordered.games:
        issues.extend(validate_game(game))
    return issues
