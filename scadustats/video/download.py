"""Download YouTube videos via yt-dlp."""

import subprocess
import tempfile
from datetime import date, datetime
from pathlib import Path

_FORMAT = "bestvideo[height<=720]"


def download_video(
    url: str,
    output_dir: str | Path = "downloads",
    timeout: float | None = None,
) -> Path:
    """Download the best video-only (no audio) stream up to 720p.

    `timeout` (seconds) bounds the yt-dlp subprocess; by default there is none,
    since real match videos can legitimately take a long time to download.

    stdout/stderr are left inherited rather than captured, so yt-dlp's own progress
    bar prints directly to the terminal as it would from a plain CLI invocation. That
    means the final file path can't be recovered from captured output, so
    `--print-to-file` is used instead to have yt-dlp write just that one line to a
    temp file rather than to stdout, where it would otherwise land in the middle of
    the progress output.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(output_dir / "%(id)s.%(ext)s")
    with tempfile.NamedTemporaryFile(mode="r+") as path_file:
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                _FORMAT,
                "-o",
                outtmpl,
                "--print-to-file",
                "after_move:filepath",
                path_file.name,
                url,
            ],
            timeout=timeout,
            check=True,
        )
        return Path(path_file.read().strip())


def fetch_published_date(url: str, timeout: float | None = None) -> date | None:
    """The video's upload date on YouTube (yt-dlp's `upload_date` field, YYYYMMDD) --
    fetched with `--skip-download`, so this doesn't pull the video itself, just its
    metadata. Used to populate VideoExtraction.published_at.

    Returns None on any failure -- yt-dlp erroring (an unreachable/removed/private
    video), a timeout, or a response with no parseable upload date -- rather than
    raising: unlike download_video, this is a best-effort supplementary field, not
    something the rest of extraction depends on.
    """
    try:
        result = subprocess.run(
            ["yt-dlp", "--skip-download", "--print", "upload_date", url],
            timeout=timeout,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None

    raw = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None
