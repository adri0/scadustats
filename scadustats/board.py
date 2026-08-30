"""5x5 bingo grid analysis: per-cell claim color and goal text."""

import cv2
import numpy as np

from scadustats import layout, ocr
from scadustats.models import CellColor

# Classification is done in HSV rather than raw BGR distance: an unclaimed cell is a
# low-saturation neutral gray regardless of overall brightness, while claimed cells are
# strongly saturated red/blue at a near-constant hue -- calibrated by sampling real
# claimed/unclaimed cells from both sample videos (720p mp4 and 1080p webm), which agree
# closely (hue ~177 for red, ~104 for blue; unclaimed saturation 20-40 vs claimed 140-255).
# This is far more robust than a raw-color distance to whole-frame brightness swings (e.g.
# a stream fade-in/flash before the overlay appears), which raw BGR distance was fooled by.
_SATURATION_THRESHOLD = 80  # below this -> neutral/unclaimed, regardless of hue
# OpenCV hue is 0-179. Real samples cluster tightly (red ~177, blue ~104) across both
# calibration videos, so these bands are kept narrow to reject near-miss contamination
# (e.g. a progress-count badge bleeding into the sampled patch) that a wider band would
# otherwise accept.
_RED_HUE_RANGE = (168, 180)
_BLUE_HUE_RANGE = (95, 115)

# Inset patch anchored at a cell's bottom-left corner, as a fraction of cell width/height.
# Chosen to avoid the centered goal text and the numeric progress badge observed near
# cell centers/corners for multi-count goals.
_PATCH_LEFT, _PATCH_RIGHT = 0.05, 0.25
_PATCH_TOP, _PATCH_BOTTOM = 0.75, 0.93


def classify_patch(patch: np.ndarray) -> CellColor:
    hue, saturation, _ = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(axis=0)
    if saturation < _SATURATION_THRESHOLD:
        return CellColor.UNCLAIMED
    if _RED_HUE_RANGE[0] <= hue <= _RED_HUE_RANGE[1]:
        return CellColor.RED
    if _BLUE_HUE_RANGE[0] <= hue <= _BLUE_HUE_RANGE[1]:
        return CellColor.BLUE
    return CellColor.UNCLAIMED


def _cell_patch(frame: np.ndarray, row: int, col: int) -> np.ndarray:
    height, width = frame.shape[:2]
    box = layout.grid_cell_box(row, col)
    left, top, right, bottom = layout.to_pixel_box(box, width, height)
    cell_w, cell_h = right - left, bottom - top

    px0 = left + int(cell_w * _PATCH_LEFT)
    px1 = left + int(cell_w * _PATCH_RIGHT)
    py0 = top + int(cell_h * _PATCH_TOP)
    py1 = top + int(cell_h * _PATCH_BOTTOM)
    return frame[py0:py1, px0:px1]


def cell_colors(frame: np.ndarray) -> list[list[CellColor]]:
    return [[classify_patch(_cell_patch(frame, r, c)) for c in range(5)] for r in range(5)]


def cell_goal_texts(frame: np.ndarray) -> list[list[str]]:
    height, width = frame.shape[:2]
    texts = []
    for r in range(5):
        row_texts = []
        for c in range(5):
            crop = layout.crop(frame, layout.grid_cell_box(r, c), width, height)
            row_texts.append(ocr.read_text(crop, psm=6))
        texts.append(row_texts)
    return texts
