"""5x5 bingo grid analysis: per-cell claim color and square text."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scadustats import layout, ocr
from scadustats.models import CellColor

# Classification uses raw BGR channel dominance rather than HSV hue. HSV hue was tried
# first and looked solid on a handful of calibration frames (red clustering tightly
# around hue~177, blue~104), but a full-video scan (one real ~4-hour match, sampled
# every 3s across all 25 cells) showed hue is surprisingly unstable for this specific
# palette: a genuinely, continuously red cell was observed drifting from hue 177 down to
# 152 within a few seconds of real compressed footage, with saturation staying high the
# whole time -- any hue band tight enough to reject a progress-count badge's contaminated
# color (hue ~120, saturation ~100) was also tight enough to intermittently drop that
# genuine red reading, causing claim-detection flicker. The red reference color (18,0,164
# in BGR) has near-zero green, while blue (154,100,39) has substantial green -- so
# red>blue dominance plus a red>green contrast check (to reject an unrelated high-green
# "orange flash" artifact seen during a stream transition) turned out far more stable
# across the full video than any hue band tried.
_DOMINANCE_THRESHOLD = 50  # required margin of the dominant channel over the other color
_RED_GREEN_CONTRAST_THRESHOLD = 80  # required R-over-G margin, to reject high-G false reds

# Inset patch anchored at a cell's bottom-left corner, as a fraction of cell width/height.
# Chosen to avoid the centered square text and the numeric progress badge observed near
# cell centers/corners for multi-count squares.
_PATCH_LEFT, _PATCH_RIGHT = 0.05, 0.25
_PATCH_TOP, _PATCH_BOTTOM = 0.75, 0.93


def classify_patch(patch: np.ndarray) -> CellColor:
    blue, green, red = patch.reshape(-1, 3).mean(axis=0)
    if red - blue > _DOMINANCE_THRESHOLD and red - green > _RED_GREEN_CONTRAST_THRESHOLD:
        return CellColor.RED
    if blue - red > _DOMINANCE_THRESHOLD:
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


def is_gameplay_frame(frame: np.ndarray) -> bool:
    """Whether this frame is showing the live-gameplay overlay (grid + colored score
    bars + bottom-left timer), as opposed to e.g. a "POST GAME" recap screen that
    reuses the same grid coordinates to cycle through completed games' final boards
    but replaces the score bars with plain background and moves the timer elsewhere.
    Cell colors and the timer/label crops are only meaningful when this is true.
    """
    height, width = frame.shape[:2]
    red_bar = layout.crop(frame, layout.SCORE_BOX_RED, width, height)
    blue_bar = layout.crop(frame, layout.SCORE_BOX_BLUE, width, height)
    return classify_patch(red_bar) is CellColor.RED and classify_patch(blue_bar) is CellColor.BLUE


def cell_square_texts(frame: np.ndarray) -> list[list[str]]:
    """OCR all 25 cells' square text.

    Each cell is an independent tesseract subprocess call (~70ms fixed overhead
    regardless of crop size, per profiling -- see CLAUDE.md), so this is run as one
    bounded batch of 25 concurrent calls rather than sequentially -- there's no per-video
    stream to bound here, just a fixed 5x5 grid read once per game.
    """
    height, width = frame.shape[:2]
    crops = [
        layout.crop(frame, layout.grid_cell_box(r, c), width, height)
        for r in range(5)
        for c in range(5)
    ]
    with ThreadPoolExecutor(max_workers=len(crops)) as executor:
        flat_texts = list(executor.map(lambda crop: ocr.read_text(crop, psm=6), crops))
    return [flat_texts[r * 5 : r * 5 + 5] for r in range(5)]
