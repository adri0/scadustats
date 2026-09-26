"""Overlay geometry for the Bingo Brawlers broadcast templates.

All regions are stored as fractions (0-1) of frame width/height rather than
pixels, because each template is composited proportionally to the output
resolution (confirmed for the standard template by comparing a 1280x720 and a
1920x1080 recording of it) -- fixed pixel coordinates would only work for one
resolution.

A broadcast can use more than one template: `STANDARD` is the season's usual
layout, while `LAYOUT_SEASON_6_FINAL` is a visually unrelated template used
for the Season 6 finals broadcast -- same grid/timer/score-bar/player-name/
game-type concepts, but at entirely different positions on the frame (the
grid moves from a full-width strip to the bottom-right quadrant, the timer
and player names move to the very bottom, and there are no commentator
nameplates at all). `Layout` bundles one template's full set of regions so
the rest of `overlay/` can be parameterized over which one applies to a given
video, rather than every module hard-coding a single set of module-level
constants. `LAYOUTS` is the registry `pipeline.extract` auto-detects against
(and a `--layout` CLI override looks up by name).

These constants are starting guesses reasoned from the observed composition
and must be hand-tuned against real frames with scripts/calibrate_layout.py
before anything downstream (color classification, OCR) can work.
"""

from dataclasses import dataclass

import numpy as np

from scadustats.models import FractionalBox


@dataclass(frozen=True)
class Layout:
    """One broadcast template's full set of overlay regions.

    `commentator_box_left`/`right` are `None` for a template that prints no
    commentator nameplates at all (as opposed to a nameplate box that's simply
    blank in a given frame, which `commentators._has_nameplate` already
    handles) -- `commentators.read_commentator_names` skips straight to
    `(None, None)` in that case rather than cropping and OCR'ing a region that
    was never calibrated to hold a nameplate.
    """

    name: str
    grid_box: FractionalBox
    timer_box: FractionalBox
    # The small "BASE GAME"/"DLC" subtitle -- see game_type_label.py. The "GAME N" label
    # itself has no box here: a game is identified by its index within the video (see
    # models.GameResult), so N is never read.
    game_type_box: FractionalBox
    score_box_red: FractionalBox
    score_box_blue: FractionalBox
    name_box_red: FractionalBox
    name_box_blue: FractionalBox
    commentator_box_left: FractionalBox | None
    commentator_box_right: FractionalBox | None


STANDARD = Layout(
    name="standard",
    grid_box=FractionalBox(left=0.354, top=0.504, right=0.646, bottom=0.988),
    timer_box=FractionalBox(left=0.008, top=0.878, right=0.172, bottom=0.972),
    game_type_box=FractionalBox(left=0.86, top=0.905, right=1.0, bottom=0.945),
    score_box_red=FractionalBox(left=0.0, top=0.5, right=0.06, bottom=0.556),
    score_box_blue=FractionalBox(left=0.94, top=0.5, right=1.0, bottom=0.556),
    name_box_red=FractionalBox(left=0.08, top=0.5, right=0.3, bottom=0.556),
    name_box_blue=FractionalBox(left=0.7, top=0.5, right=0.96, bottom=0.556),
    # The dark nameplates printed across the bottom of the two *inner* webcams -- the
    # commentators', flanking the grid. The players' own webcams are the outer pair and
    # carry no nameplate (their names are in the score strip, see NAME_BOX_* above).
    # Measured on a 720p broadcast frame as x 220-450/830-1060, y 671-706, which is
    # exactly symmetric about the frame's center line -- see commentators.py.
    commentator_box_left=FractionalBox(left=0.172, top=0.932, right=0.352, bottom=0.980),
    commentator_box_right=FractionalBox(left=0.648, top=0.932, right=0.828, bottom=0.980),
)

# Season 6 finals broadcast: two full-width gameplay feeds fill the top half, the two
# commentators (no nameplates) and both players' own webcams occupy a middle row, and the
# 5x5 grid is confined to the bottom-right quadrant -- with the timer, "GAME N"/game-type
# label, player names and live claim counts all moved to a bottom strip below everything
# else. Measured on a 1280x720 frame at 00:41:32 of the Season 6 finals VOD (see
# tests/fixtures/clip_layout_season_6_final.mp4) by cropping each region and confirming it
# OCRs cleanly with the exact call each module already makes (ocr.read_text with that
# call's own psm/whitelist/scale), the same way the standard template's own boxes are
# meant to be hand-tuned (see the module docstring).
LAYOUT_SEASON_6_FINAL = Layout(
    name="layout_season_6_final",
    grid_box=FractionalBox(left=0.695, top=0.497, right=1.0, bottom=1.0),
    timer_box=FractionalBox(left=0.203, top=0.928, right=0.297, bottom=0.972),
    game_type_box=FractionalBox(left=0.104, top=0.953, right=0.166, bottom=0.982),
    # Tight around the live claim-count badge itself (a small ~28x20px square, unlike the
    # standard template's much wider score bar) -- the first pass measured here padded a
    # few pixels of the surrounding dark background into the crop, which diluted the
    # badge's own red/blue enough to fail is_gameplay_frame's R/B-vs-G contrast check
    # (tuned against the standard layout's much larger, less digit-dominated score bar)
    # even though the "4" digit inside still OCR'd fine either way. Re-measured by
    # scanning nearby box bounds for the widest margin that both classifies correctly and
    # OCRs the digit, then centering within it -- confirmed stable across every second of
    # tests/fixtures/clip_layout_season_6_final.mp4.
    score_box_red=FractionalBox(left=0.3609, top=0.8514, right=0.382, bottom=0.8792),
    score_box_blue=FractionalBox(left=0.6234, top=0.8528, right=0.6469, bottom=0.8778),
    name_box_red=FractionalBox(left=0.328, top=0.907, right=0.473, bottom=0.944),
    name_box_blue=FractionalBox(left=0.520, top=0.907, right=0.676, bottom=0.944),
    commentator_box_left=None,
    commentator_box_right=None,
)

# Keyed by Layout.name -- the registry pipeline.extract auto-detects against (STANDARD
# first, so a frame that happens to satisfy both checks favors the far more common
# template) and a --layout CLI override looks up by name.
LAYOUTS: dict[str, Layout] = {layout.name: layout for layout in (STANDARD, LAYOUT_SEASON_6_FINAL)}


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


def grid_cell_box(row: int, col: int, grid: FractionalBox) -> FractionalBox:
    """Fractional box for grid cell (row, col), 0-indexed, within a 5x5 board."""
    cell_w = (grid.right - grid.left) / 5
    cell_h = (grid.bottom - grid.top) / 5
    left = grid.left + col * cell_w
    top = grid.top + row * cell_h
    return FractionalBox(left, top, left + cell_w, top + cell_h)
