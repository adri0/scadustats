import datetime
import os
import stat
import subprocess
import sys
import textwrap

import pytest

from scadustats.video.download import download_video, fetch_published_date

VIDEO_URL = "https://www.youtube.com/watch?v=tPEE9ZwTmy0"


@pytest.mark.integration
def test_download_video(tmp_path):
    path = download_video(VIDEO_URL, output_dir=tmp_path, timeout=120)

    assert path.exists()
    assert path.parent == tmp_path
    assert path.stat().st_size > 0


def _install_fake_yt_dlp(tmp_path, monkeypatch, body):
    """Puts a fake `yt-dlp` executable on PATH so download_video's subprocess plumbing
    (error handling, timeout) can be exercised without the network.
    `body` is the Python source of the fake script, run with the real yt-dlp's argv.
    """
    script = tmp_path / "bin" / "yt-dlp"
    script.parent.mkdir()
    script.write_text(f"#!{sys.executable}\n{textwrap.dedent(body)}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")


def test_download_video_returns_the_final_path(tmp_path, monkeypatch):
    _install_fake_yt_dlp(
        tmp_path,
        monkeypatch,
        """
        import sys
        print("[download]   0.0% of 1.00MiB at 1.00MiB/s ETA 00:01")
        print("[download] 100.0% of 1.00MiB at 1.00MiB/s ETA 00:00")
        outtmpl = sys.argv[sys.argv.index("-o") + 1]
        path = outtmpl.replace("%(id)s", "fake").replace("%(ext)s", "mp4")
        print_to_file = sys.argv[sys.argv.index("--print-to-file") + 2]
        with open(print_to_file, "w") as f:
            f.write(path)
        """,
    )

    path = download_video(VIDEO_URL, output_dir=tmp_path / "out")

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


def test_fetch_published_date_parses_yt_dlps_upload_date(tmp_path, monkeypatch):
    _install_fake_yt_dlp(
        tmp_path,
        monkeypatch,
        """
        print("20260301")
        """,
    )

    assert fetch_published_date(VIDEO_URL) == datetime.date(2026, 3, 1)


def test_fetch_published_date_returns_none_on_yt_dlp_failure(tmp_path, monkeypatch):
    """Unlike download_video, a failed lookup here is best-effort: the caller
    (extract_video) treats a missing published date as just another unknown field, not a
    reason to abort an otherwise-successful extraction."""
    _install_fake_yt_dlp(
        tmp_path,
        monkeypatch,
        """
        import sys
        print("ERROR: Video unavailable", file=sys.stderr)
        sys.exit(1)
        """,
    )

    assert fetch_published_date(VIDEO_URL) is None


def test_fetch_published_date_returns_none_on_timeout(tmp_path, monkeypatch):
    _install_fake_yt_dlp(
        tmp_path,
        monkeypatch,
        """
        import time
        time.sleep(30)
        """,
    )

    assert fetch_published_date(VIDEO_URL, timeout=0.5) is None


def test_fetch_published_date_returns_none_for_unparseable_output(tmp_path, monkeypatch):
    """yt-dlp prints "NA" for a video with no upload date on record, rather than an
    empty line -- neither is a valid YYYYMMDD, so both come back as unknown."""
    _install_fake_yt_dlp(
        tmp_path,
        monkeypatch,
        """
        print("NA")
        """,
    )

    assert fetch_published_date(VIDEO_URL) is None
