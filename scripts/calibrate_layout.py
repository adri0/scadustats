"""Dev-only tool: draw the configured overlay boxes on a sample frame for visual calibration.

Not part of the installed package. Usage:

    uv run python scripts/calibrate_layout.py <video_path> [--timestamp 60] [--out calibration.png]
        [--layout standard]

Run against sample videos at different resolutions and visually diff the annotated
image against the source video to confirm the fractional boxes in scadustats/layout.py
line up, then adjust the constants and re-run. --layout picks which registered Layout's
boxes to draw (see layout.LAYOUTS) -- e.g. `--layout layout_season_6_final` for a broadcast
using that alternate template.
"""

import argparse

import cv2

from scadustats.overlay import layout


def annotate(video_path: str, timestamp_s: float, layout_: layout.Layout) -> "cv2.typing.MatLike":
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

    draw(layout_.grid_box, (0, 255, 0))
    for row in range(5):
        for col in range(5):
            draw(layout.grid_cell_box(row, col, layout_.grid_box), (0, 200, 0), 1)
    draw(layout_.timer_box, (0, 255, 255))
    draw(layout_.game_type_box, (255, 0, 128))
    draw(layout_.score_box_red, (0, 0, 255))
    draw(layout_.score_box_blue, (255, 0, 0))
    draw(layout_.name_box_red, (0, 128, 255))
    draw(layout_.name_box_blue, (255, 128, 0))
    if layout_.commentator_box_left is not None:
        draw(layout_.commentator_box_left, (128, 255, 128))
    if layout_.commentator_box_right is not None:
        draw(layout_.commentator_box_right, (128, 255, 128))

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    parser.add_argument("--timestamp", type=float, default=60.0)
    parser.add_argument("--out", default="calibration.png")
    parser.add_argument("--layout", default=layout.STANDARD.name, choices=sorted(layout.LAYOUTS))
    args = parser.parse_args()

    frame = annotate(args.video_path, args.timestamp, layout.LAYOUTS[args.layout])
    cv2.imwrite(args.out, frame)
    print(args.out)


if __name__ == "__main__":
    main()
