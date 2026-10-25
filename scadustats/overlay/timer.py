"""In-game elapsed-time stopwatch OCR."""

import re

import numpy as np

from scadustats.overlay import layout, ocr
from scadustats.overlay.layout import Layout

_TIMER_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})")

# No game's clock gets near this (the longest are stopped at two hours), so a reading at or
# above it is a misread -- e.g. a mouse cursor over the first digit turning "00:00:42" into
# "60:00:42" for a minute and a half, which would otherwise stop the game on time.
_MAX_CLOCK_S = 3 * 60 * 60


def parse_timer(text: str) -> int | None:
    """Parse an HH:MM:SS reading into elapsed seconds, tolerating OCR noise around it.
    A reading no game clock can show (minutes or seconds past 59, or `_MAX_CLOCK_S` and
    over) is None."""
    match = _TIMER_RE.search(text)
    if not match:
        return None
    hours, minutes, seconds = (int(g) for g in match.groups())
    if minutes >= 60 or seconds >= 60:
        return None
    elapsed = hours * 3600 + minutes * 60 + seconds
    return elapsed if elapsed < _MAX_CLOCK_S else None


def read_timer(frame: np.ndarray, layout_: Layout = layout.STANDARD) -> int | None:
    height, width = frame.shape[:2]
    crop = layout.crop(frame, layout_.timer_box, width, height)
    text = ocr.read_text(crop, psm=7, whitelist="0123456789:")
    return parse_timer(text)
