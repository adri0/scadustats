"""Dev-only tool: produce a committable test fixture from real match footage, with the
webcam/face regions blacked out.

Not part of the installed package. Usage:

    uv run python scripts/redact_fixture.py <video_or_image> <out> [--start 600] [--duration 5]

Players' and commentators' likenesses aren't needed to test the overlay-reading logic,
so a fixture keeps only the regions the pipeline actually reads and zeroes everything
else. The kept regions are derived from the boxes in scadustats/layout.py rather than
hand-measured pixels, so a future recalibration of the overlay template keeps fixture
generation correct for free:

  * the grid column (GRID_BOX's x-range, full height),
  * the score-bar strip (SCORE_BOX_RED's y-range, full width -- this is what carries the
    scores, player names and flags),
  * the timer, game-type and commentator-nameplate boxes, which sit inside the
    blacked-out area below the strip and so have to be restored on top of it. A nameplate
    is printed text rather than a likeness -- it carries the commentator's broadcast
    handle, which commentators.py reads and the extraction records.

Video output is re-encoded through ffmpeg at a high quality (low CRF) on purpose: the
"BASE GAME"/"DLC" subtitle the game-type read depends on is small enough that aggressive
recompression destroys it (see CLAUDE.md). The redacted frame is overwhelmingly black,
so this stays small on disk regardless.
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from scadustats.overlay import layout

# Kept on top of the blacked-out region below the score strip -- each sits inside the
# area the mask would otherwise zero, and each is read by the pipeline.
_RESTORED_BOXES = (
    layout.TIMER_BOX,
    layout.GAME_TYPE_BOX,
    layout.COMMENTATOR_BOX_LEFT,
    layout.COMMENTATOR_BOX_RIGHT,
)

_CRF = "16"


def redact(frame: np.ndarray) -> np.ndarray:
    """Zero everything outside the overlay regions the pipeline reads."""
    height, width = frame.shape[:2]
    out = np.zeros_like(frame)

    grid_left, _, grid_right, _ = layout.to_pixel_box(layout.GRID_BOX, width, height)
    out[:, grid_left:grid_right] = frame[:, grid_left:grid_right]

    _, strip_top, _, strip_bottom = layout.to_pixel_box(layout.SCORE_BOX_RED, width, height)
    out[strip_top:strip_bottom, :] = frame[strip_top:strip_bottom, :]

    for box in _RESTORED_BOXES:
        left, top, right, bottom = layout.to_pixel_box(box, width, height)
        out[top:bottom, left:right] = frame[top:bottom, left:right]

    return out


def redact_image(source: Path, out: Path) -> None:
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"could not read an image from {source}")
    cv2.imwrite(str(out), redact(frame))


def redact_video(source: Path, out: Path, start_s: float, duration_s: float) -> None:
    cap = cv2.VideoCapture(str(source))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"could not read a frame at {start_s}s from {source}")
        height, width = frame.shape[:2]

        # Raw frames are piped to ffmpeg rather than written with cv2.VideoWriter so the
        # encoder and its quality setting are explicit -- see the module docstring on why
        # quality matters here.
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-crf",
                _CRF,
                "-pix_fmt",
                "yuv420p",
                str(out),
            ],
            stdin=subprocess.PIPE,
        )
        assert ffmpeg.stdin is not None

        end_ms = (start_s + duration_s) * 1000
        written = 0
        while ok and cap.get(cv2.CAP_PROP_POS_MSEC) <= end_ms:
            ffmpeg.stdin.write(redact(frame).tobytes())
            written += 1
            ok, frame = cap.read()

        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding the redacted clip")
        print(f"{out} ({written} frames)")
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("--start", type=float, default=0.0, help="clip start (video only)")
    parser.add_argument("--duration", type=float, default=5.0, help="clip length (video only)")
    args = parser.parse_args()

    source, out = Path(args.source), Path(args.out)
    if source.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        redact_image(source, out)
        print(out)
    else:
        redact_video(source, out, args.start, args.duration)


if __name__ == "__main__":
    main()
