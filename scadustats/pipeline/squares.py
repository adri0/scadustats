"""Base-game vs DLC classification for a game's goal squares.

`<data_dir>/squares/squares.json` (see squares_path) is a growing, hand-maintained
reference mapping each square text ever seen to the game type ("base" or "dlc") it
belongs to -- there's no programmatic way to know this, so the reference only knows
what's been added to it. It lives under the same data directory as extracted matches
(issue #73) rather than bundled inside the installed package, since it's user-editable
data that grows over time, not code -- this repo tracks its own copy at
`data/squares/squares.json` (starting empty), the way `scadustats/pipeline/squares.json`
used to before the move. Until a match's squares have been added, every game fails to
infer here; a data directory that doesn't have the file at all yet (a fresh checkout, or
one that's never run `extract`) is treated the same way -- load_known_squares returns an
empty reference rather than raising. This is only extract_video's fallback, though --
game_type_label.py reads the overlay's own "BASE GAME"/"DLC" subtitle directly and
normally succeeds first, so in practice this reference (and, failing both, prompting the
user) only covers whatever the direct read misses (see extract.py and cli/app.py).
"""

import json
import logging
from collections import Counter
from pathlib import Path

from scadustats.models import GameType

logger = logging.getLogger(__name__)


def squares_path(data_dir: str | Path) -> Path:
    """`<data_dir>/squares/squares.json` -- shared with extract.py, the same way
    json_export.video_path is shared with its own callers, so every caller agrees on
    where the reference lives under a given data directory without hand-rolling the
    layout."""
    return Path(data_dir) / "squares" / "squares.json"


def load_known_squares(path: str | Path) -> dict[str, GameType]:
    """Empty when path doesn't exist yet -- a data directory that's never had squares
    added to it (or never run extract at all) is exactly the "ships empty" case this
    reference has always supported, just as a missing file now rather than a present,
    empty one."""
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
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
