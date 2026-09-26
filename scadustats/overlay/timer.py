"""In-game elapsed-time stopwatch OCR."""

import re

import numpy as np

from scadustats.overlay import layout, ocr
from scadustats.overlay.layout import Layout

_TIMER_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})")


def parse_timer(text: str) -> int | None:
    """Parse an HH:MM:SS reading into elapsed seconds, tolerating OCR noise around it."""
    match = _TIMER_RE.search(text)
    if not match:
        return None
    hours, minutes, seconds = (int(g) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def read_timer(frame: np.ndarray, layout_: Layout = layout.STANDARD) -> int | None:
    height, width = frame.shape[:2]
    crop = layout.crop(frame, layout_.timer_box, width, height)
    text = ocr.read_text(crop, psm=7, whitelist="0123456789:")
    return parse_timer(text)
