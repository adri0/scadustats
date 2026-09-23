"""Builds a per-game-type reference of every goal square seen across a match history,
from already-extracted matches (see storage.json_export) -- the source data for
<data_dir>/squares/base_game.json and <data_dir>/squares/dlc.json
(storage.json_export.write_squares). This is a one-shot bootstrap/refresh run on demand
(`square consolidate`), not something the extraction pipeline itself calls; pipeline.
squares.py reads its output back (via storage.json_export.read_squares) as
extract_video's fallback for inferring a game's type.

consolidate_match_squares is the one operation this module actually performs: it
reconciles a single already-extracted match's OCR'd square_texts against a reference
built so far (correcting a misread in place, or growing the reference with a square it's
never seen), the `square consolidate <match_id>` half of the command (issue #76). A
from-scratch rebuild across the whole match history (`square consolidate` with no
match_id, issue #86) is just this same operation run once per match in turn, starting
from an empty reference -- cli.app.square_consolidate is what loops it, so there's no
separate whole-history function here to keep in step with it.

consolidate_players (issue #72) builds the analogous per-player reference -- one
consolidated stats record per player, covering every match they've appeared in -- for
`player consolidate` (see storage.player_stats). consolidate_player_info (issue #82) is
its sibling for the hand-curated half of a player's profile (storage.player_info): it
only ever assigns a brand-new player's id, never touching one that already exists.
"""

import difflib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from scadustats.models import (
    CellColor,
    EventType,
    GameType,
    MatchRecord,
    MatchWinner,
    PlayerInfo,
    PlayerStats,
    Square,
    SquareMarks,
    VideoExtraction,
    WinLoss,
)

# Filler verbs and connectors common to Bingo Brawlers' goal-text phrasing ("Acquire 3
# ...", "Kill 5 ..."), stripped before picking a square's slug word so the slug lands on
# the word that actually identifies the square rather than its boilerplate. Not
# exhaustive -- a text made only of these (or of words entirely missed here) still gets a
# usable, if less apt, slug -- see _slugify's fallback.
_SLUG_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "or",
    "and",
    "with",
    "from",
    "in",
    "on",
    "at",
    "to",
    "for",
    "both",
    "all",
    "non",
    "different",
    "unique",
    "s",
    "acquire",
    "obtain",
    "get",
    "collect",
    "kill",
    "defeat",
    "give",
    "restore",
    "level",
    "craft",
    "deal",
    "reach",
    "open",
    "light",
    "find",
    "take",
    "return",
    "finish",
    "drain",
    "wake",
    "put",
    "win",
    "unlock",
    "discover",
    "join",
    "equip",
    "wear",
    "cure",
    "remove",
    "break",
    "drink",
    "burn",
    "blow",
    "duplicate",
    "complete",
    "use",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _slugify(text: str) -> str:
    """A short, human-legible id candidate for a square's goal text -- e.g. "Complete 3
    Tunnels or Precipices" -> "tunnels_3": the first goal-identifying word (filler verbs
    and connectors skipped, see _SLUG_STOPWORDS) plus the first number in the text, if
    any. Falls back to the text's own first word when every word is a stopword or a
    number, so this never returns empty.
    """
    words = _WORD_RE.findall(text.lower())
    numbers = [word for word in words if word.isdigit()]
    significant = [word for word in words if word not in _SLUG_STOPWORDS and not word.isdigit()]
    base = significant[0] if significant else (words[0] if words else "square")
    return f"{base}_{numbers[0]}" if numbers else base


def _unique_id(base: str, used: set[str]) -> str:
    """base, disambiguated against ids already assigned within the same game type's list
    -- a plain numeric suffix (_2, _3, ...), since two different goal texts slugifying to
    the same base word (most often two different counts of the same item, once the count
    itself is already part of the slug) is the expected case here, not a bug to chase.
    """
    if base not in used:
        return base
    suffix = 2
    while f"{base}_{suffix}" in used:
        suffix += 1
    return f"{base}_{suffix}"


def validate_squares(squares: list[Square]) -> None:
    """Raises ValueError if `squares` holds a duplicated id or text. consolidate_match_squares'
    own construction already prevents both within one game type -- this is the final
    sanity check the reference is built to have (see issue #71), catching a future
    regression in that construction rather than anything expected to fire in practice.
    """
    id_counts = Counter(square.id for square in squares)
    duplicate_ids = sorted(id for id, count in id_counts.items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"duplicate square id(s): {', '.join(duplicate_ids)}")

    text_counts = Counter(square.text for square in squares)
    duplicate_texts = sorted(text for text, count in text_counts.items() if count > 1)
    if duplicate_texts:
        raise ValueError(f"duplicate square text(s): {', '.join(duplicate_texts)}")


@dataclass
class SquareValidationIssue:
    """One thing wrong with a consolidated squares reference, as found by
    find_square_issues -- the `square validate` CLI command's report (issue #87). `code`
    is a stable, greppable rule id; `message` is the human-facing explanation.
    """

    code: str
    message: str


def find_square_issues(squares: list[Square]) -> list[SquareValidationIssue]:
    """Every problem found in one game type's consolidated square list (issue #87): a
    duplicated `id` or `text`, and any square listed out of the text's own alphabetical
    order.

    This is the same duplicate check validate_squares already enforces as a hard
    regression backstop *during* consolidation (see its own docstring) -- but reported
    here as data rather than raised, since `square validate` is a read-only diagnostic
    over whatever's already on disk (the same "point at what looks wrong, never guess a
    correction" stance rules.validation.validate_extraction takes for matches), not a
    step in building the reference where a raise is the right way to stop.

    The order check is new: `storage.json_export.write_squares` sorts each file by `id`
    for a stable, diffable file, reasoning that an id -- derived from its own square's
    text -- already reads "close to alphabetical-by-text" without deliberately sorting by
    text itself. That's usually true but not guaranteed (an id's slug word is whichever
    one `_slugify` picked as goal-identifying, which isn't always the text's first word),
    so this flags wherever it actually isn't, one issue per adjacent pair found out of
    order.
    """
    issues: list[SquareValidationIssue] = []

    id_counts = Counter(square.id for square in squares)
    for id in sorted(id for id, count in id_counts.items() if count > 1):
        issues.append(SquareValidationIssue("duplicate_id", f"duplicate square id: {id}"))

    text_counts = Counter(square.text for square in squares)
    for text in sorted(text for text, count in text_counts.items() if count > 1):
        issues.append(SquareValidationIssue("duplicate_text", f"duplicate square text: {text!r}"))

    texts = [square.text for square in squares]
    for previous, current in zip(texts, texts[1:], strict=False):
        if current < previous:
            issues.append(
                SquareValidationIssue(
                    "square_order", f"{current!r} is out of alphabetical order (after {previous!r})"
                )
            )

    return issues


# Below this, a candidate is treated as a genuinely different square rather than an OCR
# misread of one already in the reference -- see consolidate_match_squares. Calibrated
# against real OCR failures logged in issue #76 (a dropped/swapped letter, a stray
# leading token, a missing apostrophe: "Kilt 3 Friendly NPCs..." vs. "Kill 3 Friendly
# NPCs...", 0.976), which all land well above this, and against two real, *distinct*
# squares that happen to share most of their wording ("Kill a Crystalian" vs. "Kill a
# Crucible Knight", 0.513), which lands well below it.
_FUZZY_MATCH_CUTOFF = 0.85

# \b on both sides, not a bare \d+: a goal count is always its own token ("Kill 3 ..."),
# but an OCR misread can also turn a letter inside an alphanumeric code into a digit --
# e.g. "BBK" (a boss's initials) misread as "B8K" -- and \d+ alone would pull that "8" out
# as if it were a second goal count, disagreeing with the correct candidate's "4" and
# blocking an otherwise-obvious fix ("Kil 4 Unique Gargoyles B8K" vs. "Kill 4 Unique
# Gargoyles / BBK", issue reported 2026-09-19). \b requires a non-word/word boundary, and
# "B" and "8" are both word characters, so the digit run embedded inside "B8K" never
# matches here in the first place -- only a digit run standing on its own does.
#
# A standalone "S"/"s" is included alongside a digit run for the same reason: at the
# glyph height square text is OCR'd at (see ocr.py), Tesseract routinely renders the
# digit "5" as the letter "S" -- a real "Kill 5 ..." square has come back read as "Kill
# S ..." in practice (issue #98). Left as a bare digit check, that misread would either
# wrongly refuse to match a correct "Kill 5 ..." reading of the same square in a later
# match (no digit found in "Kill S ...", so goal_count comes back None where the other
# reading's is "5" -- an unfixable disagreement that adds a permanent duplicate), or
# worse, wrongly fuzzy-match a misread "Kill S ..." against a genuinely different "Kill 3
# ..." square, since neither looked like it carried a count at all (see
# test_consolidate_match_squares_adds_rather_than_changes_a_different_goal_count for the
# case this guards). Normalizing a standalone "S"/"s" to "5" (_goal_count, below) treats
# both readings of the same square as agreeing, while still keeping them apart from any
# other, genuinely different count.
_NUMBER_RE = re.compile(r"\b(?:\d+|[Ss])\b")


def _goal_count(text: str) -> str | None:
    """`text`'s own leading goal count -- see _best_match -- or None if it doesn't have
    one. A standalone "S"/"s" token is normalized to "5" (see _NUMBER_RE above); any other
    match is the digit run verbatim.
    """
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    token = matches[0]
    return "5" if token in ("S", "s") else token


@dataclass
class SquareTextChange:
    """One cell of a match's square_texts that consolidate_match_squares looked at
    because its text wasn't already an exact match in its game's reference pool.

    A FIX (is_new=False) corrects an OCR misread against an existing reference entry --
    resolved_text is that entry's text, and ratio records how close the match was. An ADD
    (is_new=True, ratio=None) is a square the reference had never seen before, appended to
    it rather than silently dropped -- resolved_text is just original_text unchanged.

    game_type is carried alongside game_index/row/col rather than left for a caller to
    re-derive from the match -- a CLI reporting which squares.json file an ADD landed in
    (issue #76) needs exactly this and nothing else about the game."""

    game_index: int
    # 1-based, matching GameEvent.row/col and the squares DB table -- see CLAUDE.md.
    row: int
    col: int
    original_text: str
    resolved_text: str
    is_new: bool
    ratio: float | None
    game_type: GameType


def _best_match(text: str, candidates: list[str]) -> tuple[str, float] | None:
    """The candidate text closest to `text`, or None if nothing clears
    _FUZZY_MATCH_CUTOFF -- meaning `text` should be treated as its own, new square rather
    than a misread of one of these candidates.

    A candidate whose goal count -- the first standalone digit run (a standalone "S"/"s"
    counts too, normalized to "5"; see _goal_count), since Bingo Brawlers' own goal texts
    always put it right after the verb ("Kill 4 ...", "Acquire 3 ...") --
    disagrees with text's own is never considered, however similar the surrounding wording
    is: "Kill 3 Friendly NPCs" and "Kill 5 Friendly NPCs" can both be real, distinct
    squares, and a wrong digit is exactly the kind of high-similarity-looking mismatch text
    similarity alone can't tell apart from an OCR typo. Only that leading count is compared
    -- not every digit run in the text -- so a later, incidental number (a "+0 Weapon Only"
    restriction) or a stray digit OCR invents out of unrelated noise (e.g. a trailing "...
    in their name 1" misread of a text that should end "... in their name", issue reported
    2026-09-19) doesn't get treated as a second goal count to disagree over. If text's
    leading count has no candidate that shares it, there's nothing safe to match against,
    full stop -- unlike a stray letter, silently changing a goal's count is the one class
    of "fix" worth refusing outright.
    """
    goal_count = _goal_count(text)
    pool = (
        [c for c in candidates if _goal_count(c) == goal_count]
        if goal_count is not None
        else candidates
    )
    if not pool:
        return None

    matches = difflib.get_close_matches(text, pool, n=1, cutoff=_FUZZY_MATCH_CUTOFF)
    if not matches:
        return None
    match = matches[0]
    return match, difflib.SequenceMatcher(None, text, match).ratio()


def consolidate_match_squares(
    extraction: VideoExtraction, known_squares: dict[GameType, list[Square]]
) -> list[SquareTextChange]:
    """Reconciles one already-extracted match's own square_texts against the
    consolidated reference (known_squares, storage.json_export.read_squares' output) --
    the operation `square consolidate <match_id>` performs (issue #76). A from-scratch
    rebuild across the whole match history (`square consolidate` with no match_id, issue
    #86) is cli.app.square_consolidate calling this once per match in turn, starting from
    an empty known_squares, rather than a separate whole-history algorithm -- so the two
    forms of the command can't drift apart the way a second, independent implementation
    of "what squares exist" risked.

    A cell whose text is already an exact match in its game's pool needs nothing and
    isn't reported. Anything else is one of two things: an OCR misread of a square the
    reference already knows (fuzzy-matched via _best_match and corrected in place on
    `extraction`, mutating game.square_texts -- the same object the caller goes on to
    write back with json_export.write_video), or a square the reference has never seen,
    appended to `known_squares` (mutated in place, the same object the caller goes on to
    write back with json_export.write_squares) with a freshly assigned id via
    _slugify/_unique_id. That's also why a genuinely new square is added at all rather
    than left unresolved the way an unmatched cell used to be: doing so gives this
    one-match path the same end effect on the reference that including this match in a
    full consolidate run would have had.

    A game with no resolved game_type is skipped entirely -- there's no pool to check or
    add its squares to.
    """
    changes: list[SquareTextChange] = []
    used_ids = {
        game_type: {square.id for square in squares} for game_type, squares in known_squares.items()
    }

    for game in extraction.games:
        if game.game_type is None:
            continue
        pool = known_squares.setdefault(game.game_type, [])
        used = used_ids.setdefault(game.game_type, set())
        candidates = [square.text for square in pool]
        candidate_set = set(candidates)

        for row_index, row in enumerate(game.square_texts):
            for col_index, text in enumerate(row):
                if not text or text in candidate_set:
                    continue

                match = _best_match(text, candidates)
                if match is not None:
                    matched_text, ratio = match
                    game.square_texts[row_index][col_index] = matched_text
                    changes.append(
                        SquareTextChange(
                            game_index=game.game_index,
                            row=row_index + 1,
                            col=col_index + 1,
                            original_text=text,
                            resolved_text=matched_text,
                            is_new=False,
                            ratio=ratio,
                            game_type=game.game_type,
                        )
                    )
                    continue

                square_id = _unique_id(_slugify(text), used)
                used.add(square_id)
                pool.append(Square(id=square_id, text=text, game_type=game.game_type))
                candidates.append(text)
                candidate_set.add(text)
                changes.append(
                    SquareTextChange(
                        game_index=game.game_index,
                        row=row_index + 1,
                        col=col_index + 1,
                        original_text=text,
                        resolved_text=text,
                        is_new=True,
                        ratio=None,
                        game_type=game.game_type,
                    )
                )

    return changes


_NAME_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_name(name: str) -> str:
    """A player's display name, lowercased with any run of non-alphanumeric characters
    (spaces, punctuation) collapsed to a single underscore and stripped from either end
    -- e.g. "TwistieT" -> "twistiet", "Star0 Chris" -> "star0_chris". This is the identity
    two OCR'd name readings are considered "the same player" under (see
    consolidate_players): a casing/spacing difference between two readings of the same
    overlay is common, but two different players sharing a slug isn't expected in
    practice.

    Public (not `_`-prefixed) since `cli.app`'s match_id-scoped `player consolidate`
    (issue #79) needs it too, to know which of consolidate_players' output profiles --
    keyed by slug -- belong to one match's two players.
    """
    return _NAME_SLUG_RE.sub("_", name.strip().lower()).strip("_")


def _event_square_text(square_texts: list[list[str]], row: int | None, col: int | None) -> str:
    """The goal text at one event's square, or "" when there isn't one -- the same lookup
    as storage.json_export._event_square_text/cli.display._square_text, duplicated here
    for the same reason those two already are rather than shared: it's a couple of lines,
    and none of these three modules should import each other just for it.
    """
    if row is None or col is None:
        return ""
    try:
        # row/col are 1-based (see models.GameEvent); square_texts is a plain 0-based grid.
        return square_texts[row - 1][col - 1]
    except IndexError:
        return ""


def _top_squares(counts: Counter[str], limit: int = 5) -> list[SquareMarks]:
    """The `limit` most-marked texts in `counts`, each paired with its mark count, ranked
    by count descending and then alphabetically -- the same tie-break
    storage.json_export.write_squares' own sort-by-id relies on for a stable, diffable
    file, rather than Counter.most_common's insertion-order tie-break."""
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [SquareMarks(text=text, marks=count) for text, count in ranked[:limit]]


def consolidate_player_info(
    slugs: Iterable[str], existing: dict[str, PlayerInfo] | None = None
) -> dict[str, PlayerInfo]:
    """A freshly assigned models.PlayerInfo for every slug in `slugs` not already in
    `existing` -- the identity half of a player's profile that, once created,
    pipeline.consolidate never touches again (see PlayerInfo, issue #82).

    Only ever returns *new* entries: a PlayerInfo already in `existing` is hand-owned
    from the moment it was written (storage.player_info.write_player_info won't
    overwrite one either), so a caller has nothing to do for it but leave it alone.

    ids are assigned in slug order, starting one past whatever id `existing` already
    uses (1 if `existing` is empty) -- the same scheme consolidate_players' own id
    assignment used before this split.
    """
    existing = existing or {}
    next_id = max((info.id for info in existing.values()), default=0) + 1

    new_info: dict[str, PlayerInfo] = {}
    for slug in sorted(set(slugs) - existing.keys()):
        new_info[slug] = PlayerInfo(id=next_id, slug=slug)
        next_id += 1
    return new_info


def consolidate_players(extractions: list[VideoExtraction]) -> dict[str, PlayerStats]:
    """Every player's consolidated stats (see models.PlayerStats) across `extractions`,
    keyed by slug (see slugify_name) -- the source data for `player consolidate`
    (storage.player_stats.write_player_stats, issue #72).

    Two OCR'd name strings that slugify the same are treated as the same player, and
    display_name is that player's most commonly seen exact spelling across every match
    they appear in -- the same majority-vote shape overlay.commentators.
    majority_commentator_names uses for its own static, per-broadcast text reads.

    Wholly regenerated from `extractions` every call, the same as a from-scratch
    `square consolidate` run's reference -- there's no `existing` to carry anything
    forward from, unlike consolidate_player_info's identity half of a player's profile
    (issue #82): every field here is derived from match history, with nothing hand-owned
    to protect.

    A match whose winner can't be named (VideoExtraction.winner is None -- see its own
    docstring) contributes nothing to season_records, and a game with no winner_color
    contributes nothing to game_record/game_type_records/the top-squares tallies -- in
    both cases there's no result yet to attribute to either player, so guessing one would
    be inventing data the same way VideoExtraction.winner itself declines to.
    """
    name_votes: dict[str, Counter[str]] = {}
    season_records: dict[str, dict[str, MatchRecord]] = {}
    game_records: dict[str, WinLoss] = {}
    game_type_records: dict[str, dict[GameType, WinLoss]] = {}
    all_matches: dict[str, list[tuple[date, str]]] = {}
    square_marks: dict[str, dict[GameType, Counter[str]]] = {}

    for extraction in extractions:
        for color, name in (
            (CellColor.RED, extraction.player_red_name),
            (CellColor.BLUE, extraction.player_blue_name),
        ):
            if not name:
                continue
            slug = slugify_name(name)
            name_votes.setdefault(slug, Counter())[name] += 1
            all_matches.setdefault(slug, []).append((extraction.match_date, extraction.match_id))

            if extraction.winner is not None:
                record = season_records.setdefault(slug, {}).setdefault(
                    extraction.season, MatchRecord()
                )
                if extraction.winner is MatchWinner.DRAW:
                    record.draws += 1
                elif (extraction.winner is MatchWinner.RED and color is CellColor.RED) or (
                    extraction.winner is MatchWinner.BLUE and color is CellColor.BLUE
                ):
                    record.wins += 1
                else:
                    record.losses += 1

            for game in extraction.games:
                if game.winner_color is None:
                    continue
                won = game.winner_color is color

                overall = game_records.setdefault(slug, WinLoss())
                if won:
                    overall.wins += 1
                else:
                    overall.losses += 1

                if game.game_type is not None:
                    by_type = game_type_records.setdefault(slug, {}).setdefault(
                        game.game_type, WinLoss()
                    )
                    if won:
                        by_type.wins += 1
                    else:
                        by_type.losses += 1

                    marks = square_marks.setdefault(slug, {}).setdefault(game.game_type, Counter())
                    for event in game.events:
                        if event.event_type is EventType.MARK and event.color is color:
                            text = _event_square_text(game.square_texts, event.row, event.col)
                            if text:
                                marks[text] += 1

    stats: dict[str, PlayerStats] = {}
    for slug in sorted(name_votes):
        ordered_matches = sorted(all_matches[slug], key=lambda pair: pair[0], reverse=True)

        stats[slug] = PlayerStats(
            slug=slug,
            display_name=name_votes[slug].most_common(1)[0][0],
            season_records=season_records.get(slug, {}),
            game_record=game_records.get(slug, WinLoss()),
            game_type_records=game_type_records.get(slug, {}),
            all_matches=[match_id for _, match_id in ordered_matches],
            top_squares_base_game=_top_squares(
                square_marks.get(slug, {}).get(GameType.BASE, Counter())
            ),
            top_squares_dlc=_top_squares(square_marks.get(slug, {}).get(GameType.DLC, Counter())),
        )
    return stats
