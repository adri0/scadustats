"""Download YouTube videos via yt-dlp."""

import subprocess
import tempfile
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
