"""Builds a per-game-type reference of every goal square seen across a match history,
from already-extracted matches (see storage.json_export) -- the source data for
squares/base_game.json and squares/dlc.json (storage.json_export.write_squares). This is
a one-shot bootstrap/refresh run on demand (`square consolidate`), not something the
extraction pipeline itself calls; it's a different, newer reference shape (per-square id,
split by game type) than pipeline/squares.py's squares.json, and doesn't feed back into
it.

fix_square_texts runs the same reference the other direction (`square fix`, issue #76):
given one already-extracted match, it corrects that match's own OCR'd square_texts
against the reference, treating it as source of truth for the exact text once a game's
squares have been consolidated into it at least once before.
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


# Below this, no candidate is offered as a fix -- see fix_square_texts. Calibrated
# against real OCR failures logged in issue #76 (a dropped/swapped letter, a stray
# leading token, a missing apostrophe: "Kilt 3 Friendly NPCs..." vs. "Kill 3 Friendly
# NPCs...", 0.976), which all land well above this, and against two real, *distinct*
# squares that happen to share most of their wording ("Kill a Crystalian" vs. "Kill a
# Crucible Knight", 0.513), which lands well below it.
_FUZZY_MATCH_CUTOFF = 0.85

_NUMBER_RE = re.compile(r"\d+")


@dataclass
class SquareTextFix:
    """One cell of a match's square_texts that fix_square_texts looked at because its
    text wasn't already an exact match in the consolidated reference.

    matched_text/ratio are None when no reference entry was a close enough match to act
    on -- the cell is reported but left untouched, since a wrong guess here would corrupt
    a hand-reviewable JSON file (see fix_square_texts)."""

    game_index: int
    # 1-based, matching GameEvent.row/col and the squares DB table -- see CLAUDE.md.
    row: int
    col: int
    original_text: str
    matched_text: str | None
    ratio: float | None

    @property
    def resolved(self) -> bool:
        return self.matched_text is not None


def _best_match(text: str, candidates: list[str]) -> tuple[str, float] | None:
    """The candidate text closest to `text`, or None if nothing clears
    _FUZZY_MATCH_CUTOFF.

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


def fix_square_texts(
    extraction: VideoExtraction, known_squares: dict[GameType, list[Square]]
) -> list[SquareTextFix]:
    """Corrects OCR misreads in `extraction`'s own square_texts against `known_squares`
    (storage.json_export.read_squares' output, the reference pipeline.consolidate itself
    builds) -- issue #76. A cell whose text is already an exact match in its game's pool
    is left alone and not reported at all; anything else is fuzzy-matched against that
    same pool (see _best_match) and, on a close enough hit, corrected in place on the
    passed-in `extraction` (mutating game.square_texts, the same object the caller will
    go on to write back with json_export.write_video).

    A game with no resolved game_type is skipped entirely -- like consolidate_squares
    itself, there's no pool to check its squares against. Every other cell is reported,
    resolved or not, so a caller can tell a contributor about a cell nothing in the
    reference came close to, rather than silently leaving a bad OCR read in place with no
    record that it was even looked at.
    """
    fixes: list[SquareTextFix] = []
    for game in extraction.games:
        if game.game_type is None:
            continue
        candidates = [square.text for square in known_squares.get(game.game_type, [])]
        if not candidates:
            continue
        candidate_set = set(candidates)

        for row_index, row in enumerate(game.square_texts):
            for col_index, text in enumerate(row):
                if not text or text in candidate_set:
                    continue

                match = _best_match(text, candidates)
                matched_text, ratio = match if match else (None, None)
                if matched_text is not None:
                    game.square_texts[row_index][col_index] = matched_text

                fixes.append(
                    SquareTextFix(
                        game_index=game.game_index,
                        row=row_index + 1,
                        col=col_index + 1,
                        original_text=text,
                        matched_text=matched_text,
                        ratio=ratio,
                    )
                )

    return fixes
