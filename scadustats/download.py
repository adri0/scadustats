"""Download YouTube videos via yt-dlp."""

import subprocess
from pathlib import Path

_FORMAT = "bestvideo[height<=720]"


def download_video(
    url: str, output_dir: str | Path = "downloads", timeout: float | None = None
) -> Path:
    """Download the best video-only (no audio) stream up to 720p.

    `timeout` (seconds) bounds the yt-dlp subprocess; by default there is none,
    since real match videos can legitimately take a long time to download.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(output_dir / "%(id)s.%(ext)s")
    result = subprocess.run(
        [
            "yt-dlp",
            "-f",
            _FORMAT,
            "-o",
            outtmpl,
            "--print",
            "after_move:filepath",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    return Path(result.stdout.strip().splitlines()[-1])
