"""Overlay geometry for the Bingo Brawlers broadcast template.

All regions are stored as fractions (0-1) of frame width/height rather than
pixels, because the overlay is composited proportionally to the output
resolution (confirmed by comparing a 1280x720 and a 1920x1080 recording of
the same template) -- fixed pixel coordinates would only work for one
resolution.

These constants are starting guesses reasoned from the observed composition
and must be hand-tuned against real frames with scripts/calibrate_layout.py
before anything downstream (color classification, OCR) can work.
"""

import numpy as np

from scadustats.models import FractionalBox

GRID_BOX = FractionalBox(left=0.354, top=0.504, right=0.646, bottom=0.988)
TIMER_BOX = FractionalBox(left=0.008, top=0.878, right=0.172, bottom=0.972)
# The small "BASE GAME"/"DLC" subtitle printed directly under the overlay's "GAME N"
# label -- see game_type_label.py. The label itself has no box here: a game is identified
# by its index within the video (see models.GameResult), so N is never read.
GAME_TYPE_BOX = FractionalBox(left=0.86, top=0.905, right=1.0, bottom=0.945)
SCORE_BOX_RED = FractionalBox(left=0.0, top=0.5, right=0.06, bottom=0.556)
SCORE_BOX_BLUE = FractionalBox(left=0.94, top=0.5, right=1.0, bottom=0.556)
NAME_BOX_RED = FractionalBox(left=0.08, top=0.5, right=0.3, bottom=0.556)
NAME_BOX_BLUE = FractionalBox(left=0.7, top=0.5, right=0.96, bottom=0.556)
# The dark nameplates printed across the bottom of the two *inner* webcams -- the
# commentators', flanking the grid. The players' own webcams are the outer pair and carry
# no nameplate (their names are in the score strip, see NAME_BOX_* above). Measured on a
# 720p broadcast frame as x 220-450/830-1060, y 671-706, which is exactly symmetric about
# the frame's center line -- see commentators.py.
COMMENTATOR_BOX_LEFT = FractionalBox(left=0.172, top=0.932, right=0.352, bottom=0.980)
COMMENTATOR_BOX_RIGHT = FractionalBox(left=0.648, top=0.932, right=0.828, bottom=0.980)


def to_pixel_box(box: FractionalBox, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a fractional box to pixel (left, top, right, bottom) for a given frame size."""
    return (
        round(box.left * width),
        round(box.top * height),
        round(box.right * width),
        round(box.bottom * height),
    )


def crop(frame: np.ndarray, box: FractionalBox, width: int, height: int) -> np.ndarray:
    """Crop a frame to a fractional box. width/height must match frame.shape[1]/[0]."""
    left, top, right, bottom = to_pixel_box(box, width, height)
    return frame[top:bottom, left:right]


def grid_cell_box(row: int, col: int, grid: FractionalBox = GRID_BOX) -> FractionalBox:
    """Fractional box for grid cell (row, col), 0-indexed, within a 5x5 board."""
    cell_w = (grid.right - grid.left) / 5
    cell_h = (grid.bottom - grid.top) / 5
    left = grid.left + col * cell_w
    top = grid.top + row * cell_h
    return FractionalBox(left, top, left + cell_w, top + cell_h)
