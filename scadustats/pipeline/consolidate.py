"""Builds a per-game-type reference of every goal square seen across a match history,
from already-extracted matches (see storage.json_export) -- the source data for
<data_dir>/squares/base_game.json and <data_dir>/squares/dlc.json
(storage.json_export.write_squares). This is a one-shot bootstrap/refresh run on demand
(`square consolidate`), not something the extraction pipeline itself calls; pipeline.
squares.py reads its output back (via storage.json_export.read_squares) as
extract_video's fallback for inferring a game's type.

consolidate_match_squares runs that same reference the other direction, scoped to one
already-extracted match (`square consolidate <video_id>`, issue #76): it corrects that
match's own OCR'd square_texts against whatever the reference already knows, and grows
the reference with whatever it doesn't -- the same end effect on squares/ that including
this match in a from-scratch consolidate_squares run would have had.
"""

import difflib
import logging
import re
from collections import Counter
from dataclasses import dataclass

from scadustats.models import GameType, Square, VideoExtraction

logger = logging.getLogger(__name__)

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


def consolidate_squares(extractions: list[VideoExtraction]) -> dict[GameType, list[Square]]:
    """Every distinct goal square text seen across `extractions`, split by game type and
    tagged with a short slug id.

    A game with no resolved game_type contributes nothing (there's no pool to file its
    squares under), and a blank square_texts cell (a ragged, hand-edited grid) is skipped.
    The same text seen under more than one game type across the match history is a real
    data problem -- a board's squares are drawn from one pool, not a mix (see
    squares.infer_game_type) -- so it's filed under whichever type it was seen under most,
    with a warning, rather than appearing in both references.

    Output is sorted by text within each game type, so re-running this against a growing
    match history produces a stable, diffable file rather than one reordered by whatever
    order the source files happened to be read in.
    """
    votes: dict[str, Counter[GameType]] = {}
    for extraction in extractions:
        for game in extraction.games:
            if game.game_type is None:
                continue
            for row in game.square_texts:
                for text in row:
                    if text:
                        votes.setdefault(text, Counter())[game.game_type] += 1

    squares: dict[GameType, list[Square]] = {GameType.BASE: [], GameType.DLC: []}
    used_ids: dict[GameType, set[str]] = {GameType.BASE: set(), GameType.DLC: set()}
    for text in sorted(votes):
        counts = votes[text]
        if len(counts) > 1:
            logger.warning(
                "square seen under more than one game type: %r (%s)", text, dict(counts)
            )
        game_type = counts.most_common(1)[0][0]

        square_id = _unique_id(_slugify(text), used_ids[game_type])
        used_ids[game_type].add(square_id)
        squares[game_type].append(Square(id=square_id, text=text, game_type=game_type))

    return squares


def validate_squares(squares: list[Square]) -> None:
    """Raises ValueError if `squares` holds a duplicated id or text. consolidate_squares'
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


# Below this, a candidate is treated as a genuinely different square rather than an OCR
# misread of one already in the reference -- see consolidate_match_squares. Calibrated
# against real OCR failures logged in issue #76 (a dropped/swapped letter, a stray
# leading token, a missing apostrophe: "Kilt 3 Friendly NPCs..." vs. "Kill 3 Friendly
# NPCs...", 0.976), which all land well above this, and against two real, *distinct*
# squares that happen to share most of their wording ("Kill a Crystalian" vs. "Kill a
# Crucible Knight", 0.513), which lands well below it.
_FUZZY_MATCH_CUTOFF = 0.85

_NUMBER_RE = re.compile(r"\d+")


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

    A candidate whose goal-count digits ("Kill 3 ...") disagree with text's own is never
    considered, however similar the surrounding wording is -- "Kill 3 Friendly NPCs" and
    "Kill 5 Friendly NPCs" can both be real, distinct squares, and a wrong digit is
    exactly the kind of high-similarity-looking mismatch text similarity alone can't tell
    apart from an OCR typo. If text has digits and no candidate shares them, there's
    nothing safe to match against, full stop -- unlike a stray letter, silently changing
    a goal's count is the one class of "fix" worth refusing outright.
    """
    numbers = _NUMBER_RE.findall(text)
    pool = [c for c in candidates if _NUMBER_RE.findall(c) == numbers] if numbers else candidates
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
    the video_id-scoped form of `square consolidate` (issue #76), as opposed to
    consolidate_squares' from-scratch rebuild across every match.

    A cell whose text is already an exact match in its game's pool needs nothing and
    isn't reported. Anything else is one of two things: an OCR misread of a square the
    reference already knows (fuzzy-matched via _best_match and corrected in place on
    `extraction`, mutating game.square_texts -- the same object the caller goes on to
    write back with json_export.write_video), or a square the reference has never seen,
    appended to `known_squares` (mutated in place, the same object the caller goes on to
    write back with json_export.write_squares) with a freshly assigned id -- the same
    _slugify/_unique_id scheme consolidate_squares itself uses, so a squares file grown
    one match at a time this way ids its entries exactly as a from-scratch
    consolidate_squares run would. That's also why a genuinely new square is added at all
    rather than left unresolved the way an unmatched cell used to be: doing so gives this
    one-match path the same end effect on the reference that including this match in a
    full consolidate run would have had.

    A game with no resolved game_type is skipped entirely, like consolidate_squares
    itself -- there's no pool to check or add its squares to.
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
