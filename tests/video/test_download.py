import os
import stat
import subprocess
import sys
import textwrap

import pytest

from scadustats.video.download import download_video

VIDEO_URL = "https://www.youtube.com/watch?v=tPEE9ZwTmy0"


@pytest.mark.integration
def test_download_video(tmp_path):
    path = download_video(VIDEO_URL, output_dir=tmp_path, timeout=120)

    assert path.exists()
    assert path.parent == tmp_path
    assert path.stat().st_size > 0


def _install_fake_yt_dlp(tmp_path, monkeypatch, body):
    """Puts a fake `yt-dlp` executable on PATH so download_video's subprocess plumbing
    (progress parsing, error handling, timeout) can be exercised without the network.
    `body` is the Python source of the fake script, run with the real yt-dlp's argv.
    """
    script = tmp_path / "bin" / "yt-dlp"
    script.parent.mkdir()
    script.write_text(f"#!{sys.executable}\n{textwrap.dedent(body)}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")


def test_download_video_reports_progress(tmp_path, monkeypatch):
    _install_fake_yt_dlp(
        tmp_path,
        monkeypatch,
        """
        import sys
        print("[download]   0.0% of 1.00MiB at 1.00MiB/s ETA 00:01")
        print("[download]  50.0% of 1.00MiB at 1.00MiB/s ETA 00:01")
        print("[download] 100.0% of 1.00MiB at 1.00MiB/s ETA 00:00")
        outtmpl = sys.argv[sys.argv.index("-o") + 1]
        print(outtmpl.replace("%(id)s", "fake").replace("%(ext)s", "mp4"))
        """,
    )
    updates = []

    path = download_video(
        VIDEO_URL, output_dir=tmp_path / "out", on_progress=updates.append
    )

    assert updates == [0.0, 50.0, 100.0]
    assert path == tmp_path / "out" / "fake.mp4"


def test_download_video_raises_on_yt_dlp_failure(tmp_path, monkeypatch):
    _install_fake_yt_dlp(
        tmp_path,
        monkeypatch,
        """
        import sys
        print("ERROR: something went wrong", file=sys.stderr)
        sys.exit(1)
        """,
    )

    with pytest.raises(subprocess.CalledProcessError):
        download_video(VIDEO_URL, output_dir=tmp_path / "out")


def test_download_video_raises_on_timeout(tmp_path, monkeypatch):
    _install_fake_yt_dlp(
        tmp_path,
        monkeypatch,
        """
        import time
        time.sleep(30)
        """,
    )

    with pytest.raises(subprocess.TimeoutExpired):
        download_video(VIDEO_URL, output_dir=tmp_path / "out", timeout=0.5)
