"""Player name and running-score OCR from the score bars above the grid."""

import re

import numpy as np

from scadustats.overlay import layout, ocr
from scadustats.overlay.layout import Layout


def _clean_name(text: str) -> str:
    """Keep the longest run of letters, dropping flag-icon/border OCR noise around it."""
    words = re.findall(r"[A-Za-z']{2,}", text)
    return max(words, key=len) if words else text.strip()


def _clean_team_name(text: str) -> str:
    """Every word of a team name ("POSITIVE VIBES"), dropping one-letter noise -- a
    single longest word, as _clean_name keeps, would cut it to "POSITIVE". The overlay
    prints it in capitals; storage.teams matches it against the roster's own spelling."""
    words = re.findall(r"[A-Za-z']{2,}", text)
    return " ".join(words) if words else text.strip()


def read_player_names(frame: np.ndarray, layout_: Layout = layout.STANDARD) -> tuple[str, str]:
    """The two sides' names as (red, blue): players' names, or teams' when the layout
    shows team names (Layout.team_names)."""
    height, width = frame.shape[:2]
    red_crop = layout.crop(frame, layout_.name_box_red, width, height)
    blue_crop = layout.crop(frame, layout_.name_box_blue, width, height)
    clean = _clean_team_name if layout_.team_names else _clean_name
    return (
        clean(ocr.read_text(red_crop, psm=7)),
        clean(ocr.read_text(blue_crop, psm=7)),
    )


def _read_score(frame: np.ndarray, box: layout.FractionalBox) -> int | None:
    height, width = frame.shape[:2]
    crop = layout.crop(frame, box, width, height)
    text = ocr.read_text(crop, psm=7, whitelist="0123456789")
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def read_scores(
    frame: np.ndarray, layout_: Layout = layout.STANDARD
) -> tuple[int | None, int | None]:
    return (
        _read_score(frame, layout_.score_box_red),
        _read_score(frame, layout_.score_box_blue),
    )
