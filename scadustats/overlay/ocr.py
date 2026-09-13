"""Thin pytesseract wrapper shared by every OCR call site in the pipeline."""

import cv2
import numpy as np
import pytesseract


def read_text(
    image: np.ndarray, psm: int = 6, whitelist: str | None = None, scale: float = 1.0
) -> str:
    """OCR an image crop, after optional upscaling and grayscale + Otsu thresholding for
    contrast against the HUD.

    `scale` upscales tiny crops (e.g. wrapped multi-line goal text squeezed into a ~75x69px
    grid cell) before OCR -- Tesseract's accuracy drops sharply below roughly a 30px glyph
    height, and its per-call cost is a fixed subprocess overhead regardless of crop size
    (see CLAUDE.md), so upscaling costs no meaningful extra time.
    """
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    return pytesseract.image_to_string(thresh, config=config).strip()
