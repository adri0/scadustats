from scadustats.frames import probe, sample_frames

_CLIP_PATH = "tests/fixtures/clip_claim.mp4"


def test_sample_frames_yields_expected_count_and_timestamps():
    info = probe(_CLIP_PATH)

    samples = list(sample_frames(_CLIP_PATH, sample_rate_hz=1.0))

    assert len(samples) == int(info.duration_s)
    timestamps = [ts for ts, _ in samples]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 0.0
    # Sequential grab-based sampling lands on the frame nearest each 1s mark rather than
    # an exact multiple, so allow a sub-frame tolerance instead of requiring equality.
    assert all(abs(ts - round(ts)) < 0.1 for ts in timestamps)


def test_sample_frames_frames_have_video_resolution():
    info = probe(_CLIP_PATH)

    _, frame = next(sample_frames(_CLIP_PATH, sample_rate_hz=1.0))

    assert frame.shape[:2] == (info.height, info.width)
