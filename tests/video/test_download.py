import pytest

from scadustats.video.download import download_video

VIDEO_URL = "https://www.youtube.com/watch?v=tPEE9ZwTmy0"


@pytest.mark.integration
def test_download_video(tmp_path):
    path = download_video(VIDEO_URL, output_dir=tmp_path, timeout=120)

    assert path.exists()
    assert path.parent == tmp_path
    assert path.stat().st_size > 0
