"""Player name and running-score OCR from the score bars above the grid."""

import re

import numpy as np

from scadustats import layout, ocr


def _clean_name(text: str) -> str:
    """Keep the longest run of letters, dropping flag-icon/border OCR noise around it."""
    words = re.findall(r"[A-Za-z']{2,}", text)
    return max(words, key=len) if words else text.strip()


def read_player_names(frame: np.ndarray) -> tuple[str, str]:
    height, width = frame.shape[:2]
    red_crop = layout.crop(frame, layout.NAME_BOX_RED, width, height)
    blue_crop = layout.crop(frame, layout.NAME_BOX_BLUE, width, height)
    return (
        _clean_name(ocr.read_text(red_crop, psm=7)),
        _clean_name(ocr.read_text(blue_crop, psm=7)),
    )


def _read_score(frame: np.ndarray, box: layout.FractionalBox) -> int | None:
    height, width = frame.shape[:2]
    crop = layout.crop(frame, box, width, height)
    text = ocr.read_text(crop, psm=7, whitelist="0123456789")
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def read_scores(frame: np.ndarray) -> tuple[int | None, int | None]:
    return (
        _read_score(frame, layout.SCORE_BOX_RED),
        _read_score(frame, layout.SCORE_BOX_BLUE),
    )
