"""Thin tesserocr wrapper shared by every OCR call site in the pipeline.

Binds directly to Tesseract's C++ engine via `tesserocr` rather than shelling out to the
`tesseract` CLI per call (as this module did via `pytesseract` until it was benchmarked
against tesserocr, see CLAUDE.md): re-spawning a subprocess and re-initializing the
engine (reloading its language data from disk) from scratch on every single call turned
out to account for ~97% of a call's wall time on a real crop, with the actual character
recognition only ~3% of it -- so a persistent engine, reused across calls instead of
rebuilt each time, measured ~11x faster per call on the same crops with identical output.
"""

import os
import shutil
import threading
from pathlib import Path

import cv2
import numpy as np
import tesserocr
from PIL import Image

_local = threading.local()


def _resolve_tessdata_prefix() -> str:
    """Find the directory holding Tesseract's .traineddata language files.

    The `tesseract` CLI binary locates its own tessdata relative to argv[0] when nothing
    else is configured, but a Python process embedding libtesseract has no such anchor --
    left unset, tesserocr silently searches the current working directory and finds
    nothing there, rather than raising (confirmed empirically: `get_languages()` came
    back `('./', [])` on a real Homebrew install with TESSDATA_PREFIX unset).
    """
    env = os.environ.get("TESSDATA_PREFIX")
    if env:
        return env

    tesseract_bin = shutil.which("tesseract")
    if tesseract_bin:
        # Both a Homebrew install (<prefix>/bin/tesseract, tessdata under
        # <prefix>/share/tessdata) and a Debian/Ubuntu one (/usr/bin/tesseract, tessdata
        # under /usr/share/tesseract-ocr/<version>/tessdata, as apt's tesseract-ocr
        # package -- the one CI installs -- lays it out) resolve from the binary's own
        # location the same way: walk up to the install prefix, then look under share/.
        prefix = Path(tesseract_bin).resolve().parent.parent
        candidates = [
            prefix / "share" / "tessdata",
            prefix / "share" / "tesseract-ocr" / "tessdata",
            *sorted(prefix.glob("share/tesseract-ocr/*/tessdata"), reverse=True),
        ]
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.traineddata")):
                return str(candidate)

    raise RuntimeError(
        "Could not locate Tesseract's tessdata directory. Set the TESSDATA_PREFIX "
        "environment variable to it -- e.g. `$(brew --prefix tesseract)/share/tessdata` "
        "on macOS, or /usr/share/tesseract-ocr/<version>/tessdata on Debian/Ubuntu."
    )


def _engine() -> tesserocr.PyTessBaseAPI:
    """One PyTessBaseAPI instance per thread, created lazily on first use and reused for
    every later call on that same thread.

    PyTessBaseAPI isn't thread-safe (tesserocr's own docs recommend exactly this
    thread-local-instance pattern for concurrent use), and creating a fresh one per call
    would throw away the entire point of not re-initializing Tesseract every time -- the
    thread pools in extract.py/board.py are exactly where this pays off, since the same
    handful of worker threads live across many OCR calls.
    """
    engine = getattr(_local, "engine", None)
    if engine is None:
        engine = tesserocr.PyTessBaseAPI(path=_resolve_tessdata_prefix())
        _local.engine = engine
    return engine


def read_text(
    image: np.ndarray, psm: int = 6, whitelist: str | None = None, scale: float = 1.0
) -> str:
    """OCR an image crop, after optional upscaling and grayscale + Otsu thresholding for
    contrast against the HUD.

    `scale` upscales tiny crops (e.g. wrapped multi-line goal text squeezed into a ~75x69px
    grid cell) before OCR -- Tesseract's accuracy drops sharply below roughly a 30px glyph
    height, and a whole HUD crop is tiny next to Tesseract's own recognition work either
    way, so upscaling still costs no meaningful extra time.
    """
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    engine = _engine()
    engine.SetPageSegMode(psm)
    # Set (or clear) the whitelist on every call, whether or not this call wants one --
    # the engine persists across calls on this thread, so a whitelist left over from a
    # previous call would otherwise silently leak into one that never asked for it.
    engine.SetVariable("tessedit_char_whitelist", whitelist or "")
    engine.SetImage(Image.fromarray(thresh))
    return engine.GetUTF8Text().strip()
