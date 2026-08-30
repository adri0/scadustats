"""Thin pytesseract wrapper shared by every OCR call site in the pipeline."""

import cv2
import numpy as np
import pytesseract


def read_text(image: np.ndarray, psm: int = 6, whitelist: str | None = None) -> str:
    """OCR an image crop, after grayscale + Otsu thresholding for contrast against the HUD."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    return pytesseract.image_to_string(thresh, config=config).strip()
