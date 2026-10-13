"""5x5 bingo grid analysis: per-cell claim color and square text."""

import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scadustats.models import CellColor, GameType, tag_square_text
from scadustats.overlay import layout, ocr
from scadustats.overlay.layout import Layout

# Tesseract's PSM 6 reads square text as multiple lines wrapped to the cell width; only
# letters, digits, spaces, apostrophes (e.g. "Rennala's"), periods (e.g. "Mt. Gelmir"),
# parentheses (e.g. "(Volcano Manor)"), "+" (e.g. "+0 Weapon Only"), double-quotes, "/",
# square brackets, ":", ";", "-", and "," are meaningful goal text, so anything else (line
# breaks, stray punctuation from OCR noise) is stripped rather than kept as literal output.
_NON_ALPHANUMERIC_SPACE = re.compile(r'[^A-Za-z0-9 \'.()+"/\[\]:;,-]+')


def _sanitize_square_text(text: str) -> str:
    collapsed = _NON_ALPHANUMERIC_SPACE.sub(" ", text)
    return " ".join(collapsed.split())


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


def _cell_patch(frame: np.ndarray, row: int, col: int, layout_: Layout) -> np.ndarray:
    height, width = frame.shape[:2]
    box = layout.grid_cell_box(row, col, layout_.grid_box)
    left, top, right, bottom = layout.to_pixel_box(box, width, height)
    cell_w, cell_h = right - left, bottom - top

    px0 = left + int(cell_w * _PATCH_LEFT)
    px1 = left + int(cell_w * _PATCH_RIGHT)
    py0 = top + int(cell_h * _PATCH_TOP)
    py1 = top + int(cell_h * _PATCH_BOTTOM)
    return frame[py0:py1, px0:px1]


def cell_colors(frame: np.ndarray, layout_: Layout = layout.STANDARD) -> list[list[CellColor]]:
    return [[classify_patch(_cell_patch(frame, r, c, layout_)) for c in range(5)] for r in range(5)]


def is_gameplay_frame(frame: np.ndarray, layout_: Layout = layout.STANDARD) -> bool:
    """Whether this frame is showing the live-gameplay overlay (grid + colored score
    bars + timer), as opposed to e.g. a "POST GAME" recap screen that reuses the same
    grid coordinates to cycle through other completed games' final boards but replaces
    the score bars with plain background and moves the timer elsewhere. Cell colors and
    the timer/label crops are only meaningful when this is true.

    Also doubles as `pipeline.extract`'s layout-detection probe: since each `Layout`'s
    score-bar boxes sit at template-specific coordinates, a frame only satisfies this
    check under the one layout whose boxes actually line up with the broadcast's overlay.
    """
    height, width = frame.shape[:2]
    red_bar = layout.crop(frame, layout_.score_box_red, width, height)
    blue_bar = layout.crop(frame, layout_.score_box_blue, width, height)
    return classify_patch(red_bar) is CellColor.RED and classify_patch(blue_bar) is CellColor.BLUE


# Cell crops are tiny (e.g. ~75x69px at 720p) holding up to 4 wrapped lines of goal text,
# putting real glyph height well below what Tesseract reads reliably -- upscale before OCR
# to compensate (see ocr.read_text).
_SQUARE_TEXT_OCR_SCALE = 4.0


def cell_square_texts(frame: np.ndarray, layout_: Layout = layout.STANDARD) -> list[list[str]]:
    """OCR all 25 cells' square text.

    Each cell is an independent OCR call, so this is run as one bounded batch of
    concurrent calls rather than sequentially -- there's no per-video stream to bound
    here, just a fixed 5x5 grid read once per game. The batch is capped at the machine's
    CPU count rather than always all 25 at once: this dates from when ocr.py shelled out
    to a `tesseract` subprocess per call (see CLAUDE.md) and a CPU-constrained CI runner
    (e.g. GitHub's 4-vCPU ubuntu-latest) was oversubscribed 6x by 25 concurrent
    subprocesses, turning every game's square-text read into minutes of context-switch
    thrashing instead of the sub-second read this is meant to be -- with OCR now
    in-process via tesserocr, 25 concurrent threads is cheaper (no process-per-call
    overhead), but the same cap still bounds how many threads spin up their own
    tessdata-loaded engine (see ocr._engine) at once, so it's kept rather than removed.
    """
    height, width = frame.shape[:2]
    crops = [
        layout.crop(frame, layout.grid_cell_box(r, c, layout_.grid_box), width, height)
        for r in range(5)
        for c in range(5)
    ]
    with ThreadPoolExecutor(max_workers=min(len(crops), os.cpu_count() or 4)) as executor:
        flat_texts = list(
            executor.map(
                lambda crop: ocr.read_text(crop, psm=6, scale=_SQUARE_TEXT_OCR_SCALE), crops
            )
        )
    flat_texts = [_sanitize_square_text(text) for text in flat_texts]
    return [flat_texts[r * 5 : r * 5 + 5] for r in range(5)]


def cell_square_texts_majority(
    frames: list[np.ndarray], layout_: Layout = layout.STANDARD
) -> list[list[str]]:
    """OCR all 25 cells across several frames of the same game and, per cell, keep
    whichever exact text was read most often.

    A single frame's OCR occasionally misreads a cell (motion blur, a compression
    artifact on that frame), and unlike cell color/claims there's no debounce here since
    square text never changes mid-game -- so voting across multiple frames instead of
    trusting whichever one frame happened to be sampled is a straightforward way to drop
    that noise. An empty reading doesn't get a vote (see _majority_text).
    """
    per_frame_texts = [cell_square_texts(frame, layout_) for frame in frames]
    return [
        [_majority_text([pf[r][c] for pf in per_frame_texts]) for c in range(5)] for r in range(5)
    ]


def _majority_text(readings: list[str]) -> str:
    """The most common non-empty reading -- a cell whose small text OCR'd as nothing on
    most frames still keeps what it read on the others (Season 5's smallest-font squares
    came back blank on most frames) -- or "" if every reading was empty."""
    counts = Counter(text for text in readings if text)
    return counts.most_common(1)[0][0] if counts else ""


# Season 5's mixed boards (see models.GameType.TWO_V_TWO) end every square's text in a
# pool tag: "(B)" base game, "(D)" DLC, "(C)" co-op. At the glyph size square text is read
# at, Tesseract often gets the letter wrong -- "(B)" comes back "(8)" or "(6)", "(D)" as
# "(0)" -- or drops it, so a one- or two-character parenthesized token at the very end is
# treated as the tag whatever it reads as, and the letter only decides the pool when it's
# one of these known readings.
_SQUARE_TYPE_TAG = re.compile(r"\s*\(\s*(\w{0,2})\s*\)?\s*$")
_TAG_READINGS = {
    "B": GameType.BASE,
    "8": GameType.BASE,
    "6": GameType.BASE,
    "D": GameType.DLC,
    "0": GameType.DLC,
    "O": GameType.DLC,
    "C": GameType.TWO_V_TWO,
}


def split_square_type_tag(text: str) -> tuple[str, GameType | None]:
    """A mixed board's square text without its trailing pool tag, and the pool the tag
    reads as -- None when there's no tag or it didn't read as a known letter."""
    match = _SQUARE_TYPE_TAG.search(text)
    if match is None:
        return text, None
    return text[: match.start()].rstrip(), _TAG_READINGS.get(match.group(1).upper())


# An unclaimed cell on a mixed board is black, with its text colored by pool: pink-red for
# a base-game square, yellow for DLC and cyan for co-op (a claimed cell turns red/blue with
# white text, so only an unclaimed cell says anything). Measured on Positive Vibes vs
# Bushy Stache (youtube.com/watch?v=YX3OSxKTJE8): the mean BGR of a cell's text pixels
# was about (110, 100, 160) for base game, (130, 175, 175) for DLC and (160, 165, 120)
# for co-op -- each pool has one channel well clear of the other two, which is what the
# scores below compare.
_TEXT_PIXEL_MIN = 140  # brightest channel of a pixel counted as text, not background
_TEXT_COLOR_MARGIN = 10  # how far the winning score must clear zero to count


def _text_color_type(crop: np.ndarray) -> GameType | None:
    pixels = crop.reshape(-1, 3).astype(int)
    text = pixels[pixels.max(axis=1) > _TEXT_PIXEL_MIN]
    if len(text) == 0:
        return None
    blue, green, red = text.mean(axis=0)
    scores = {
        GameType.BASE: red - (green + blue) / 2,
        GameType.DLC: (red + green) / 2 - blue,
        GameType.TWO_V_TWO: (green + blue) / 2 - red,
    }
    best = max(scores, key=lambda game_type: scores[game_type])
    return best if scores[best] > _TEXT_COLOR_MARGIN else None


def cell_text_color_types(
    frame: np.ndarray, layout_: Layout = layout.STANDARD
) -> list[list[GameType | None]]:
    """Each cell's pool as its text color says (see _text_color_type) -- None for a
    claimed cell, whose text is white, or one whose color is unclear."""
    height, width = frame.shape[:2]
    colors = cell_colors(frame, layout_)
    return [
        [
            _text_color_type(
                layout.crop(frame, layout.grid_cell_box(r, c, layout_.grid_box), width, height)
            )
            if colors[r][c] is CellColor.UNCLAIMED
            else None
            for c in range(5)
        ]
        for r in range(5)
    ]


def tag_square_texts_majority(
    frames: list[np.ndarray], square_texts: list[list[str]], layout_: Layout = layout.STANDARD
) -> list[list[str]]:
    """A mixed board's square texts, each ending in a clean pool tag -- " (B)", " (D)" or
    " (C)", the form models.square_type_tag reads the cell's pool from.

    Which pool is voted by text color across `frames` first: it's read off far more
    pixels than the tag's one small letter, and a square unclaimed in any of the frames
    -- every square, early in the game -- gets a vote. The OCR'd tag is the fallback for a
    cell with no color vote at all. Whatever the OCR read as the tag is replaced by the
    decided one; a cell where neither says anything is left untagged, which
    rules.validation reports, to be fixed by hand.
    """
    votes = [cell_text_color_types(frame, layout_) for frame in frames]
    tagged_texts: list[list[str]] = []
    for r, row in enumerate(square_texts):
        tagged_row: list[str] = []
        for c, text in enumerate(row):
            stripped, tagged = split_square_type_tag(text)
            colored = Counter(vote[r][c] for vote in votes if vote[r][c] is not None)
            game_type = colored.most_common(1)[0][0] if colored else tagged
            tagged_row.append(tag_square_text(stripped, game_type) if game_type else stripped)
        tagged_texts.append(tagged_row)
    return tagged_texts
