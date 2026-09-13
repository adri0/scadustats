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
    """Yield (video_timestamp_s, frame) pairs at roughly sample_rate_hz.

    Decodes sequentially and skips unwanted frames with `grab()` (cheap -- decodes but
    doesn't convert/copy the frame) rather than seeking to each target timestamp with
    `CAP_PROP_POS_MSEC`. Seeking forces the codec to jump back to the nearest preceding
    keyframe and decode forward to the target on *every single sample*, which measured
    ~2.5x slower than sequential grab+skip on a real clip -- and OpenCV's frame-accurate
    seek support is inconsistent across codecs/containers to begin with.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        raise ValueError(f"could not read fps for video: {video_path}")
    step_frames = fps / sample_rate_hz

    try:
        frame_index = 0
        next_sample_index = 0.0
        while True:
            ok = cap.grab()
            if not ok:
                return
            if frame_index >= next_sample_index:
                ok, frame = cap.retrieve()
                if not ok:
                    return
                yield frame_index / fps, frame
                next_sample_index += step_frames
            frame_index += 1
    finally:
        cap.release()
