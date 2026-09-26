"""Dev-only tool: produce a committable test fixture from real match footage, with the
webcam/face regions blacked out.

Not part of the installed package. Usage:

    uv run python scripts/redact_fixture.py <video_or_image> <out> [--start 600] [--duration 5]
        [--layout standard]

Players' and commentators' likenesses aren't needed to test the overlay-reading logic,
so a fixture keeps only the regions the pipeline actually reads and zeroes everything
else. The kept regions are derived from the given --layout's own boxes (see
scadustats/overlay/layout.py) rather than hand-measured pixels, so a future
recalibration of that template -- or picking a different registered Layout entirely --
keeps fixture generation correct for free:

  * the grid column (grid_box's x-range, full height),
  * every other box the layout defines (timer, game type, both player names, both live
    claim counts, and the commentator nameplates when the layout has them at all --
    Layout.commentator_box_left/right is None for one that doesn't, e.g.
    layout_season_6_final), restored individually on top of the otherwise blacked-out
    frame. A nameplate or a player name is printed text rather than a likeness, so
    keeping it doesn't reintroduce what the redaction is for.

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

_CRF = "16"


def _restored_boxes(layout_: layout.Layout) -> tuple[layout.FractionalBox, ...]:
    """Every one of the layout's own boxes other than the grid (which is kept as a whole
    column, not box-by-box -- see redact()), skipping a commentator box the layout
    doesn't have at all."""
    boxes = [
        layout_.timer_box,
        layout_.game_type_box,
        layout_.score_box_red,
        layout_.score_box_blue,
        layout_.name_box_red,
        layout_.name_box_blue,
    ]
    if layout_.commentator_box_left is not None:
        boxes.append(layout_.commentator_box_left)
    if layout_.commentator_box_right is not None:
        boxes.append(layout_.commentator_box_right)
    return tuple(boxes)


def redact(frame: np.ndarray, layout_: layout.Layout) -> np.ndarray:
    """Zero everything outside the overlay regions the pipeline reads."""
    height, width = frame.shape[:2]
    out = np.zeros_like(frame)

    grid_left, _, grid_right, _ = layout.to_pixel_box(layout_.grid_box, width, height)
    out[:, grid_left:grid_right] = frame[:, grid_left:grid_right]

    for box in _restored_boxes(layout_):
        left, top, right, bottom = layout.to_pixel_box(box, width, height)
        out[top:bottom, left:right] = frame[top:bottom, left:right]

    return out


def redact_image(source: Path, out: Path, layout_: layout.Layout) -> None:
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"could not read an image from {source}")
    cv2.imwrite(str(out), redact(frame, layout_))


def redact_video(
    source: Path,
    out: Path,
    start_s: float,
    duration_s: float,
    layout_: layout.Layout,
    fps: float | None = None,
) -> None:
    """fps (None: keep the source's own rate) subsamples the source rather than just
    setting ffmpeg's output rate -- a high-fps source (e.g. this project's own real VOD
    downloads, unlike the ~3fps source most existing fixtures were cut from) would
    otherwise write hundreds of near-duplicate frames for a few seconds of footage,
    bloating a committed fixture for no real gain: extraction only ever samples at 1Hz by
    default, so a handful of frames per second is already far more than any test needs.
    """
    cap = cv2.VideoCapture(str(source))
    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        output_fps = fps if fps is not None else source_fps
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
                str(output_fps),
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
        # Nearest-frame decimation: write source frame i only when doing so keeps the
        # running count of written frames in step with how many output-rate frames
        # should have elapsed by that source frame's own timestamp -- handles a
        # non-integer source/output ratio (e.g. 59.94fps down to 3fps) without drift.
        source_index = 0
        while ok and cap.get(cv2.CAP_PROP_POS_MSEC) <= end_ms:
            if round(source_index * output_fps / source_fps) >= written:
                ffmpeg.stdin.write(redact(frame, layout_).tobytes())
                written += 1
            source_index += 1
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
    parser.add_argument("--layout", default=layout.STANDARD.name, choices=sorted(layout.LAYOUTS))
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="subsample to this rate (video only, default: source's own)",
    )
    args = parser.parse_args()

    source, out = Path(args.source), Path(args.out)
    layout_ = layout.LAYOUTS[args.layout]
    if source.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        redact_image(source, out, layout_)
        print(out)
    else:
        redact_video(source, out, args.start, args.duration, layout_, args.fps)


if __name__ == "__main__":
    main()
