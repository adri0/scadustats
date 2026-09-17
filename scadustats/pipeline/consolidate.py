"""Builds a per-game-type reference of every goal square seen across a match history,
from already-extracted matches (see storage.json_export) -- the source data for
squares/base_game.json and squares/dlc.json (storage.json_export.write_squares). This is
a one-shot bootstrap/refresh run on demand (`square consolidate`), not something the
extraction pipeline itself calls; it's a different, newer reference shape (per-square id,
split by game type) than pipeline/squares.py's squares.json, and doesn't feed back into
it.
"""

import logging
import re
from collections import Counter

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
