"""Frame iteration over a video file. No knowledge of the overlay lives here."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from scadustats.models import VideoInfo


def probe(video_path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_s = frame_count / fps if fps else 0.0
        return VideoInfo(width=width, height=height, fps=fps, duration_s=duration_s)
    finally:
        cap.release()


def sample_frames(
    video_path: str | Path, sample_rate_hz: float = 1.0
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (video_timestamp_s, frame) pairs at roughly sample_rate_hz."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    step_ms = 1000.0 / sample_rate_hz
    try:
        timestamp_ms = 0.0
        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            ok, frame = cap.read()
            if not ok:
                return
            yield timestamp_ms / 1000.0, frame
            timestamp_ms += step_ms
    finally:
        cap.release()
