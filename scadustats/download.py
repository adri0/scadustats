"""Download YouTube videos via yt-dlp."""

import subprocess
from pathlib import Path

_FORMAT = "bestvideo[height<=720]"


def download_video(url: str, output_dir: str | Path = "downloads") -> Path:
    """Download the best video-only (no audio) stream up to 720p."""
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
    )

    return Path(result.stdout.strip().splitlines()[-1])
