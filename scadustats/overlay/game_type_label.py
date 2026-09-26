"""Reads the overlay's own "BASE GAME"/"DLC" subtitle, printed directly under the
"GAME N" label, as a direct, independent way to know a game's type -- see squares.py for
the other way (inferring it from which goal squares appear on the board), used as a
fallback when this doesn't produce a clean match.
"""

from collections import Counter

import numpy as np

from scadustats.models import GameType
from scadustats.overlay import layout, ocr
from scadustats.overlay.layout import Layout


def parse_game_type_label(text: str) -> GameType | None:
    """None for anything other than an exact "BASE GAME"/"DLC" reading -- including a
    misread or a moment the subtitle isn't rendered at all (e.g. a transition). Guessing
    wrong here would silently mislabel a game, so callers fall back to squares.py's
    inference instead of trusting a fuzzy match."""
    normalized = text.strip().upper()
    if normalized == "BASE GAME":
        return GameType.BASE
    if normalized == "DLC":
        return GameType.DLC
    return None


def read_game_type_label(frame: np.ndarray, layout_: Layout = layout.STANDARD) -> GameType | None:
    height, width = frame.shape[:2]
    crop = layout.crop(frame, layout_.game_type_box, width, height)
    return parse_game_type_label(ocr.read_text(crop, psm=7, scale=3))


def majority_game_type_label(
    frames: list[np.ndarray], layout_: Layout = layout.STANDARD
) -> GameType | None:
    """Votes `read_game_type_label` across several frames of the same game -- same
    rationale as board.cell_square_texts_majority: a single frame's OCR can misread (or
    land on a moment the subtitle briefly isn't rendered), so whichever type comes back
    most often across a handful of frames wins. None if none of them produced a match."""
    votes = [t for t in (read_game_type_label(frame, layout_) for frame in frames) if t is not None]
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]
