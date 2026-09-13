"""Reads the commentators' names off the nameplates the broadcast prints under the two
inner webcams (see layout.COMMENTATOR_BOX_LEFT/RIGHT).

Unlike the players, whose names are in the score strip above the grid (scoreboard.py),
the commentators are only ever named by these plates -- which also carry a mic icon that
OCRs as junk ("&", "|", "E") ahead of the name, hence the token-based cleaning below.
"""

import re
from collections import Counter

import cv2
import numpy as np

from scadustats import layout, ocr

# A commentator's name is a streaming handle -- letters, digits, and the separators
# Twitch/YouTube allow in one ("Captain_Domo", "star0chris") -- so the name is recovered
# as the longest such token in the reading, which drops the mic icon's one-character
# OCR noise without needing to know what that noise will look like. Same idea as
# scoreboard._clean_name, but digits and "_" are part of a handle and have to survive:
# that function's letters-only rule would cut "star0chris" down to "chris".
_NAME_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]+")

# A nameplate is bright text on a dark bar, so the share of near-white pixels in the crop
# says whether there's a plate there at all. Measured on real 720p frames: 8-10% of a
# plate's pixels clear _TEXT_PIXEL_VALUE, against exactly 0% where the region is blank
# (a redacted test fixture, or a broadcast moment showing no commentator cams). The
# threshold sits well inside that gap.
#
# Without this gate, a blank region isn't simply "unreadable": Tesseract hallucinates on
# a uniform crop -- a fully black one reads as "re", which is a plausible enough token to
# survive _clean_name and be recorded as somebody's name. Gating up front (like
# board.is_gameplay_frame does before reading cells) is also what keeps that from costing
# an OCR call in the first place.
_TEXT_PIXEL_VALUE = 180
_MIN_TEXT_PIXEL_FRACTION = 0.01


def _has_nameplate(crop: np.ndarray) -> bool:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return bool((gray > _TEXT_PIXEL_VALUE).mean() >= _MIN_TEXT_PIXEL_FRACTION)


def _read_nameplate(crop: np.ndarray) -> str | None:
    if not _has_nameplate(crop):
        return None
    return _clean_name(ocr.read_text(crop, psm=7))


def _clean_name(text: str) -> str | None:
    """The name in an OCR reading of one nameplate, or None if it holds no plausible
    name at all -- None (rather than "") so a failed read can be dropped instead of
    recorded as a nameless commentator."""
    tokens = _NAME_TOKEN.findall(text)
    return max(tokens, key=len) if tokens else None


def read_commentator_names(frame: np.ndarray) -> tuple[str | None, str | None]:
    """The (left, right) nameplate readings, None where the frame shows no plate or none
    that reads as a name. Seat order is kept -- rather than returning just the names
    found -- so majority_commentator_names can vote per seat.

    psm 7 (one line of text) with no upscaling: the plate text is only ~10-20px tall,
    below where ocr.read_text's docstring says Tesseract gets comfortable, but upscaling
    was measured on real frames to change nothing at 2x and to start introducing stray
    punctuation at 3x ("Captain_.Domo"), so there's nothing to buy here.
    """
    height, width = frame.shape[:2]
    left_crop = layout.crop(frame, layout.COMMENTATOR_BOX_LEFT, width, height)
    right_crop = layout.crop(frame, layout.COMMENTATOR_BOX_RIGHT, width, height)
    return (_read_nameplate(left_crop), _read_nameplate(right_crop))


def majority_commentator_names(frames: list[np.ndarray]) -> list[str]:
    """Vote `read_commentator_names` across several frames of the same video and return
    the winning name per seat, left to right, dropping any seat that never read as a
    name.

    Same rationale as game_type_label.majority_game_type_label and
    board.cell_square_texts_majority: a single frame's OCR can misread, and a nameplate
    is static for the whole broadcast, so there's no debounce to lean on -- whichever
    exact string came back most often across a handful of frames wins.

    The result is a flat list because seat position isn't a property of a commentator
    the way red/blue is of a player: it's just where the broadcast happened to put their
    webcam. A video with one unreadable plate yields a one-name list, not a gap.
    """
    per_frame = [read_commentator_names(frame) for frame in frames]
    names = []
    for seat in range(2):
        votes = [reading[seat] for reading in per_frame if reading[seat]]
        if votes:
            names.append(Counter(votes).most_common(1)[0][0])
    return names
