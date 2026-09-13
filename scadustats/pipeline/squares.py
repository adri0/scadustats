"""Base-game vs DLC classification for a game's goal squares.

`squares.json` (alongside this module) is a growing, hand-maintained reference mapping
each square text ever seen to the game type ("base" or "dlc") it belongs to -- there's
no programmatic way to know this, so the reference only knows what's been added to it.
It ships empty: until a match's squares have been added, every game in it fails to
infer here. This is only extract_video's fallback, though -- game_type_label.py reads
the overlay's own "BASE GAME"/"DLC" subtitle directly and normally succeeds first, so in
practice this reference (and, failing both, prompting the user) only covers whatever the
direct read misses (see extract.py and cli/app.py).
"""

import json
import logging
from collections import Counter
from pathlib import Path

from scadustats.models import GameType

logger = logging.getLogger(__name__)

DEFAULT_SQUARES_PATH = Path(__file__).parent / "squares.json"


def load_known_squares(path: str | Path = DEFAULT_SQUARES_PATH) -> dict[str, GameType]:
    data = json.loads(Path(path).read_text())
    return {text: GameType(value) for text, value in data.items()}


def infer_game_type(
    square_texts: list[list[str]], known_squares: dict[str, GameType]
) -> GameType | None:
    """None if none of the board's 25 squares are in known_squares (including when
    known_squares is empty) -- there's nothing to infer from. If the matched squares
    disagree (which shouldn't happen: a board's squares are drawn from one pool, not
    mixed), the majority wins and a warning is logged, rather than failing outright."""
    matched = [
        known_squares[text] for row in square_texts for text in row if text in known_squares
    ]
    if not matched:
        return None

    counts = Counter(matched)
    if len(counts) > 1:
        logger.warning("board has squares from more than one game type: %s", dict(counts))
    return counts.most_common(1)[0][0]
