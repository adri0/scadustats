"""Dev-only tool: draw the configured overlay boxes on a sample frame for visual calibration.

Not part of the installed package. Usage:

    uv run python scripts/calibrate_layout.py <video_path> [--timestamp 60] [--out calibration.png]

Run against sample videos at different resolutions and visually diff the annotated
image against the source video to confirm the fractional boxes in scadustats/layout.py
line up, then adjust the constants and re-run.
"""

import argparse

import cv2

from scadustats import layout


def annotate(video_path: str, timestamp_s: float) -> "cv2.typing.MatLike":
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame at {timestamp_s}s from {video_path}")

    height, width = frame.shape[:2]

    def draw(box: layout.FractionalBox, color: tuple[int, int, int], thickness: int = 2) -> None:
        left, top, right, bottom = layout.to_pixel_box(box, width, height)
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

    draw(layout.GRID_BOX, (0, 255, 0))
    for row in range(5):
        for col in range(5):
            draw(layout.grid_cell_box(row, col), (0, 200, 0), 1)
    draw(layout.TIMER_BOX, (0, 255, 255))
    draw(layout.GAME_TYPE_BOX, (255, 0, 128))
    draw(layout.SCORE_BOX_RED, (0, 0, 255))
    draw(layout.SCORE_BOX_BLUE, (255, 0, 0))
    draw(layout.NAME_BOX_RED, (0, 128, 255))
    draw(layout.NAME_BOX_BLUE, (255, 128, 0))

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    parser.add_argument("--timestamp", type=float, default=60.0)
    parser.add_argument("--out", default="calibration.png")
    args = parser.parse_args()

    frame = annotate(args.video_path, args.timestamp)
    cv2.imwrite(args.out, frame)
    print(args.out)


if __name__ == "__main__":
    main()
