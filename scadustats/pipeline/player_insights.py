"""Player insights read from the boards and claims of every match: pace, when each
claim of a game comes, how games are won, first squares, comebacks, form and signature
squares -- the fields of models.PlayerStats that pipeline.consolidate.consolidate_players
leaves at their defaults, filled in by add_player_insights (`player consolidate` runs
both).

A claim is a MARK still held on the game's final board (consolidate._final_marks), the
same as square stats: a misclick that was undone wasn't a claim. Times are the game
clock (GameEvent.game_elapsed_s).

In a 2v2 match (VideoExtraction.team_red/team_blue) everything about the side -- its
wins, first squares, comebacks and results -- counts for both players, while a
claim is only the teammate's whose game the square was in (both, for a co-op square),
as in consolidate_players' top squares.
"""

import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import NamedTuple

from scadustats.models import (
    CellColor,
    ClaimRef,
    Comebacks,
    EventType,
    FirstSquare,
    FormResult,
    GameEvent,
    GameRef,
    GameResult,
    GameType,
    MatchWinner,
    NthClaim,
    PlayerForm,
    PlayerPace,
    PlayerStats,
    SignatureSquare,
    Square,
    VideoExtraction,
    WinType,
    WinTypeCounts,
)
from scadustats.pipeline.consolidate import _final_marks, slugify_name
from scadustats.rules.winner import apply_event, empty_board

# A win counts as a comeback when the winner was this many squares behind at some point.
COMEBACK_DEFICIT = 3
# Signature squares need this many of the player's own claims (quickest) or deals
# (surest) to count, and this many claims by everyone else to compare against.
SIGNATURE_MIN_CLAIMS = 3
SIGNATURE_MIN_DEALT = 4
SIGNATURE_MIN_FIELD_CLAIMS = 3
SIGNATURE_LIMIT = 3
FORM_LENGTH = 5
# claim_order goes up to a player's 13th claim of a game: 13 squares win on majority.
CLAIM_ORDER_LIMIT = 13


class _Claim(NamedTuple):
    """One of a player's claims in a game: when, the square's pool, and its text (the
    catalogue's wording when the square is known, else the board's)."""

    time: int
    game_type: GameType
    text: str


@dataclass
class _Tally:
    """One player's running totals while walking every match."""

    first_claim_times: list[int] = field(default_factory=list)
    # Per game, the player's claims in order -- see claim_order.
    claim_sequences: list[tuple[str, int, list[_Claim]]] = field(default_factory=list)
    claims: int = 0
    game_seconds: int = 0
    win_times: list[GameRef] = field(default_factory=list)
    win_types: Counter[WinType] = field(default_factory=Counter)
    decided_games: int = 0
    first_claims: int = 0
    wins_after_first_claim: int = 0
    comebacks: list[tuple[int, GameRef]] = field(default_factory=list)
    results: list[tuple[date, str, str]] = field(default_factory=list)
    # (game_type, square id) -> the player's claim times, and how often it was dealt to them.
    square_times: dict[tuple[GameType, str], list[int]] = field(default_factory=dict)
    square_dealt: Counter[tuple[GameType, str]] = field(default_factory=Counter)


def _ordered(game: GameResult) -> list[GameEvent]:
    return sorted(game.events, key=lambda event: event.video_timestamp)


def _game_length_s(game: GameResult) -> int:
    """The game clock at its end: the GAME_END event's, or else its last event's."""
    events = _ordered(game)
    for event in events:
        if event.event_type is EventType.GAME_END:
            return event.game_elapsed_s
    return max((event.game_elapsed_s for event in events), default=0)


def _largest_deficit(game: GameResult, color: CellColor) -> int:
    """The most squares `color` was ever behind by, replaying the game's events."""
    board = empty_board()
    worst = 0
    for event in _ordered(game):
        apply_event(board, event)
        own = sum(cell is color for row in board for cell in row)
        other = sum(cell not in (color, CellColor.UNCLAIMED) for row in board for cell in row)
        worst = max(worst, other - own)
    return worst


def _first_claim(game: GameResult) -> GameEvent | None:
    held = _final_marks(game).values()
    return min(held, key=lambda event: (event.game_elapsed_s, event.video_timestamp), default=None)


def _players(extraction: VideoExtraction, color: CellColor) -> list[tuple[str, str]]:
    """(name, slug) for each player on a side."""
    return [(name, slugify_name(name)) for name in extraction.side_players(color)]


def _owns(extraction: VideoExtraction, color: CellColor, name: str, cell_type: GameType) -> bool:
    """Whether a square of cell_type on `color`'s side is `name`'s to claim."""
    team = extraction.team(color)
    return team is None or name in team.players_for(cell_type)


def _result(extraction: VideoExtraction, color: CellColor) -> str | None:
    winner = extraction.winner
    if winner is None:
        return None
    if winner is MatchWinner.DRAW:
        return "draw"
    return "win" if winner.value == color.value else "loss"


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.median(values)) if values else None


def _form(results: list[tuple[date, str, str]]) -> PlayerForm:
    ordered = sorted(results)
    longest = run = 0
    for _, _, result in ordered:
        run = run + 1 if result == "win" else 0
        longest = max(longest, run)
    current_result = ordered[-1][2] if ordered else None
    current = 0
    for _, _, result in reversed(ordered):
        if result != current_result:
            break
        current += 1
    return PlayerForm(
        last=[FormResult(match_id=m, result=r) for _, m, r in reversed(ordered[-FORM_LENGTH:])],
        longest_win_streak=longest,
        current_streak_result=current_result,
        current_streak=current,
    )


def _of_type(claims: list[_Claim], game_type: GameType | None) -> list[_Claim]:
    """A game's claims, in order, keeping only one pool's squares (all of them for None)."""
    return [claim for claim in claims if game_type is None or claim.game_type is game_type]


def _claim_order(
    slug: str,
    tally: _Tally,
    nth_times: dict[tuple[GameType | None, int], list[tuple[str, int]]],
    game_type: GameType | None = None,
) -> list[NthClaim]:
    """The player's n-th claims (of one pool's squares, or of all for None), for every n
    they ever reached, with everyone else's median n-th claim to compare against."""
    order: list[NthClaim] = []
    for n in range(1, CLAIM_ORDER_LIMIT + 1):
        refs = []
        for match_id, game_index, claims in tally.claim_sequences:
            claims = _of_type(claims, game_type)
            if len(claims) >= n:
                claim = claims[n - 1]
                refs.append(
                    ClaimRef(
                        match_id=match_id,
                        game_index=game_index,
                        time_s=claim.time,
                        game_type=claim.game_type,
                        text=claim.text,
                    )
                )
        if not refs:
            break
        refs.sort(key=lambda ref: (ref.time_s, ref.match_id, ref.game_index))
        others = [t for other, t in nth_times.get((game_type, n), []) if other != slug]
        order.append(
            NthClaim(
                n=n,
                claims=refs,
                median_s=_median(ref.time_s for ref in refs),
                field_median_s=_median(others),
            )
        )
    return order


def add_player_insights(
    stats: dict[str, PlayerStats],
    extractions: list[VideoExtraction],
    known_squares: dict[GameType, list[Square]] | None = None,
) -> dict[str, PlayerStats]:
    """`stats` (consolidate_players' output, keyed by slug) with every insight field
    filled in from `extractions`. Signature squares need the squares reference, to tell
    which square a cell is (by any of its wordings, like consolidate_square_stats); a cell
    it doesn't know is left out of them."""
    by_text = {
        (game_type, wording): square
        for game_type, squares in (known_squares or {}).items()
        for square in squares
        for wording in square.wordings
    }
    tallies: dict[str, _Tally] = {}
    # (game_type, square id) -> every claim's (time, claimant slugs), for the field medians.
    all_claims: dict[tuple[GameType, str], list[tuple[int, list[str]]]] = {}
    squares: dict[tuple[GameType, str], Square] = {}

    for extraction in extractions:
        sides = {color: _players(extraction, color) for color in (CellColor.RED, CellColor.BLUE)}
        for color, players in sides.items():
            result = _result(extraction, color)
            for _, slug in players:
                if result is not None:
                    tallies.setdefault(slug, _Tally()).results.append(
                        (extraction.match_date, extraction.match_id, result)
                    )

        for game in extraction.games:
            held = _final_marks(game)
            length = _game_length_s(game)
            first = _first_claim(game)
            # The square behind each cell, with its pool.
            cells: dict[tuple[int, int], tuple[GameType, Square | None]] = {}
            for row_index, row in enumerate(game.square_texts):
                for col_index in range(len(row)):
                    cell_type, text = game.cell_goal(row_index, col_index)
                    if cell_type is not None:
                        square = by_text.get((cell_type, text))
                        cells[(row_index + 1, col_index + 1)] = (cell_type, square)
                        if square is not None:
                            squares[(square.game_type, square.id)] = square

            for (row, col), mark in held.items():
                cell_type, square = cells.get((row, col), (None, None))
                if square is None or mark.color not in sides:
                    continue
                claimants = [
                    slug
                    for name, slug in sides[mark.color]
                    if _owns(extraction, mark.color, name, cell_type)
                ]
                all_claims.setdefault((square.game_type, square.id), []).append(
                    (mark.game_elapsed_s, claimants)
                )

            for color, players in sides.items():
                won = game.winner_color is color
                decided = game.winner_color in (CellColor.RED, CellColor.BLUE)
                ref = GameRef(
                    match_id=extraction.match_id, game_index=game.game_index, time_s=length
                )
                deficit = _largest_deficit(game, color) if won else 0
                for name, slug in players:
                    tally = tallies.setdefault(slug, _Tally())
                    tally.game_seconds += length
                    for cell_type, square in cells.values():
                        if square is not None and _owns(extraction, color, name, cell_type):
                            tally.square_dealt[(square.game_type, square.id)] += 1
                    own_claims: list[_Claim] = []
                    for (row, col), mark in held.items():
                        cell_type, square = cells.get((row, col), (None, None))
                        if mark.color is not color or cell_type is None:
                            continue
                        if not _owns(extraction, color, name, cell_type):
                            continue
                        text = square.text if square else game.cell_goal(row - 1, col - 1)[1]
                        own_claims.append(_Claim(mark.game_elapsed_s, cell_type, text))
                        if square is not None:
                            tally.square_times.setdefault((square.game_type, square.id), []).append(
                                mark.game_elapsed_s
                            )
                    tally.claims += len(own_claims)
                    if own_claims:
                        own_claims.sort(key=lambda claim: claim.time)
                        tally.first_claim_times.append(own_claims[0].time)
                        tally.claim_sequences.append(
                            (extraction.match_id, game.game_index, own_claims)
                        )

                    if not decided:
                        continue
                    tally.decided_games += 1
                    if first is not None and first.color is color:
                        tally.first_claims += 1
                        tally.wins_after_first_claim += won
                    if won:
                        tally.win_times.append(ref)
                        tally.win_types[game.win_type] += 1
                        if deficit >= COMEBACK_DEFICIT:
                            tally.comebacks.append((deficit, ref))

    # (pool or None for all, n) -> every player's n-th claim times, as (slug, time), for
    # the field medians.
    nth_times: dict[tuple[GameType | None, int], list[tuple[str, int]]] = {}
    for slug, tally in tallies.items():
        for _, _, claims in tally.claim_sequences:
            for game_type in (None, *{claim.game_type for claim in claims}):
                for n, claim in enumerate(_of_type(claims, game_type)[:CLAIM_ORDER_LIMIT], 1):
                    nth_times.setdefault((game_type, n), []).append((slug, claim.time))

    updated: dict[str, PlayerStats] = {}
    for slug, player in stats.items():
        tally = tallies.get(slug, _Tally())
        fastest = min(tally.win_times, key=lambda ref: (ref.time_s, ref.match_id), default=None)
        largest = max(tally.comebacks, key=lambda c: (c[0], -c[1].time_s), default=None)

        signatures: list[SignatureSquare] = []
        for key, dealt in tally.square_dealt.items():
            times = tally.square_times.get(key, [])
            others = [time for time, slugs in all_claims.get(key, []) if slug not in slugs]
            square = squares[key]
            signatures.append(
                SignatureSquare(
                    id=square.id,
                    text=square.text,
                    game_type=square.game_type,
                    claims=len(times),
                    dealt=dealt,
                    median_time_s=_median(times),
                    field_median_time_s=_median(others)
                    if len(others) >= SIGNATURE_MIN_FIELD_CLAIMS
                    else None,
                )
            )
        quickest = sorted(
            (
                s
                for s in signatures
                if s.claims >= SIGNATURE_MIN_CLAIMS
                and s.field_median_time_s
                and s.median_time_s < s.field_median_time_s
            ),
            key=lambda s: (s.median_time_s / s.field_median_time_s, -s.claims, s.id),
        )
        surest = sorted(
            (s for s in signatures if s.dealt >= SIGNATURE_MIN_DEALT and s.claims),
            key=lambda s: (-s.claims / s.dealt, -s.dealt, s.id),
        )

        updated[slug] = player.model_copy(
            update={
                "pace": PlayerPace(
                    games=len(tally.first_claim_times),
                    median_first_claim_s=_median(tally.first_claim_times),
                    claims_per_hour=round(tally.claims * 3600 / tally.game_seconds, 2)
                    if tally.game_seconds
                    else None,
                    median_win_time_s=_median(ref.time_s for ref in tally.win_times),
                    fastest_win=fastest,
                ),
                "claim_order": _claim_order(slug, tally, nth_times),
                "claim_order_by_type": {
                    game_type: _claim_order(slug, tally, nth_times, game_type)
                    for game_type in sorted(
                        {c.game_type for _, _, claims in tally.claim_sequences for c in claims}
                    )
                },
                "win_types": WinTypeCounts(
                    line=tally.win_types[WinType.LINE],
                    majority=tally.win_types[WinType.MAJORITY],
                    points=tally.win_types[WinType.POINTS],
                ),
                "first_square": FirstSquare(
                    games=tally.decided_games,
                    first_claims=tally.first_claims,
                    wins_after_first_claim=tally.wins_after_first_claim,
                ),
                "comebacks": Comebacks(
                    count=len(tally.comebacks),
                    largest_deficit=largest[0] if largest else 0,
                    largest=largest[1] if largest else None,
                ),
                "form": _form(tally.results),
                "quickest_squares": quickest[:SIGNATURE_LIMIT],
                "surest_squares": surest[:SIGNATURE_LIMIT],
            }
        )
    return updated
