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

from dataclasses import dataclass, replace

from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameResult,
    GameType,
    MatchType,
    MatchWinner,
    VideoExtraction,
    WinLine,
    WinType,
)
from scadustats.rules import winner

# Both formats open with a base game and follow it with a DLC game; only a
# double-elimination decider (game 3) is free to be either.
_OPENING_GAME_TYPES = (GameType.BASE, GameType.DLC)


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
    return sorted(game.events, key=lambda event: event.video_ts_s)


def _board_states(events: list[GameEvent]) -> list[Board]:
    """The board state after each event, index-aligned with `events`. Mirrors
    extract._determine_winner's replay, but keeps every intermediate state instead of
    only the final one -- the "no marks after the win" rule needs to know *when* the
    board became won, not just that it ended that way."""
    state = winner.empty_board()
    states = []
    for event in events:
        winner.apply_event(state, event)
        states.append([row[:] for row in state])
    return states


def _settled_win_index(states: list[Board], winner_color: CellColor) -> int | None:
    """Index of the earliest event after which `winner_color` holds the win for the rest
    of the game. Found by walking back from the end rather than forward from the start,
    because a line can be completed and then immediately undone -- a player can claim the
    wrong square and unclaim it (see extract._extract_events) -- and that transient win
    isn't the one that ended the game. Returns None if the board isn't won at the end.
    """
    settled = None
    for index in reversed(range(len(states))):
        line_win = winner.winning_line(states[index])
        if line_win is not None and line_win[0] is winner_color:
            settled = index
        else:
            break
    return settled


def _check_recorded_winner_matches_board(
    game: GameResult, states: list[Board]
) -> list[ValidationIssue]:
    """Rule: the recorded winner, win type and winning line must be what the game's own
    events replay to. A mismatch means the two disagree about the same game -- either the
    events are incomplete (a claim missed at extraction time) or the result was
    hand-edited without the board behind it. The line is part of the check, not a
    separate rule: a win recorded on a line the board doesn't hold is the same defect as
    a win recorded for the wrong color."""
    final = states[-1] if states else winner.empty_board()
    expected = winner.determine_winner(final)
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

    Only checked for a LINE win. A line ends the game the moment it completes, so a later
    mark means the segment ran past the end of its game -- but a MAJORITY win is decided
    by who's ahead when time runs out, and the board's lines are typically all blocked
    long before that, so players legitimately keep claiming squares afterward.
    """
    if game.win_type is not WinType.LINE or game.winner_color is None:
        return []

    settled = _settled_win_index(states, game.winner_color)
    if settled is None:
        # The board doesn't end on a line at all -- _check_recorded_winner_matches_board
        # already reports that, and re-reporting it here would just be noise.
        return []

    after = [event for event in events[settled + 1 :] if event.event_type is EventType.MARK]
    if not after:
        return []
    # Falls back to "a line" only when win_line is missing from an otherwise line-won
    # game -- _check_recorded_winner_matches_board reports that separately.
    line = game.win_line.label if game.win_line else "a line"
    return [
        ValidationIssue(
            code="mark_after_win",
            message=(
                f"{len(after)} square(s) marked after {game.winner_color.value} completed "
                f"{line} at {events[settled].video_ts_s:.1f}s "
                f"(first at {after[0].video_ts_s:.1f}s)"
            ),
            game_index=game.game_index,
        )
    ]


def _describe_result(color: CellColor | None, win_type: WinType, win_line: WinLine | None) -> str:
    line = f" on {win_line.label}" if win_line else ""
    return f"{color.value if color else 'no winner'} ({win_type.value}{line})"


def validate_game(game: GameResult) -> list[ValidationIssue]:
    events = _ordered_events(game)
    states = _board_states(events)
    return [
        *_check_recorded_winner_matches_board(game, states),
        *_check_single_game_start(game),
        *_check_no_marks_after_win(game, events, states),
    ]


def _check_game_count(extraction: VideoExtraction) -> list[ValidationIssue]:
    """Rule: a playoffs match is exactly 2 games; a double-elimination match (best of 3)
    is 2 or 3. A wrong count is the loudest sign of a segmentation problem -- games merged
    together, or a recap segment read as a game of its own."""
    count = extraction.num_games
    if extraction.match_type is MatchType.PLAYOFFS:
        expected = "exactly 2 games"
        ok = count == 2
    else:
        expected = "2 or 3 games (best of 3)"
        ok = count in (2, 3)
    if ok:
        return []
    return [
        ValidationIssue(
            code="game_count",
            message=f"a {extraction.match_type.value} match has {expected}, found {count}",
        )
    ]


def _check_opening_game_types(extraction: VideoExtraction) -> list[ValidationIssue]:
    """Rule: game 1 is the base game and game 2 is DLC, in both formats. (A
    double-elimination decider -- game 3 -- can be either, so it isn't checked.)"""
    issues = []
    for game, expected in zip(extraction.games, _OPENING_GAME_TYPES, strict=False):
        if game.game_type is expected:
            continue
        found = game.game_type.value if game.game_type else "no game type"
        issues.append(
            ValidationIssue(
                code="opening_game_type",
                message=f"game {game.game_index} should be {expected.value}, found {found}",
            )
        )
    return issues


def _check_match_outcome(extraction: VideoExtraction) -> list[ValidationIssue]:
    """Rule: the match must resolve to a winner or (playoffs only) a draw.

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

    # A playoffs match is allowed to end 1-1 (the issue's rule 2 calls that a draw), so
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


def validate_extraction(extraction: VideoExtraction) -> list[ValidationIssue]:
    """Every rule violation found in one extracted match, match-level rules first and
    then each game's own, in game order. An empty list means the match looks valid.

    The extraction isn't mutated -- games are sorted into index order for checking (the
    order rules only mean anything against that) without touching the caller's object.
    """
    ordered = replace(extraction, games=sorted(extraction.games, key=lambda g: g.game_index))
    issues = [
        *_check_game_count(ordered),
        *_check_opening_game_types(ordered),
        *_check_match_outcome(ordered),
        *_check_decider_game(ordered),
    ]
    for game in ordered.games:
        issues.extend(validate_game(game))
    return issues
