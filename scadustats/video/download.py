"""Download YouTube videos via yt-dlp."""

import re
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

_FORMAT = "bestvideo[height<=720]"

# yt-dlp's own progress bar normally redraws one line in place with `\r`, which is both
# invisible once the subprocess's output is captured (see below) and awkward to parse --
# `--newline` makes it emit one line per update instead, in the form
# "[download]  45.2% of ...".
_PROGRESS_RE = re.compile(r"\[download\]\s+(\d{1,3}\.\d)%")


def download_video(
    url: str,
    output_dir: str | Path = "downloads",
    timeout: float | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """Download the best video-only (no audio) stream up to 720p.

    `timeout` (seconds) bounds the yt-dlp subprocess; by default there is none,
    since real match videos can legitimately take a long time to download.

    `on_progress`, if given, is called with yt-dlp's own reported download percentage
    (0-100) as each update arrives, so a caller can drive its own progress bar --
    capturing the subprocess's output (needed to recover the final file path below)
    already defeats yt-dlp's own in-place progress bar, so this is how one gets
    reconstructed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(output_dir / "%(id)s.%(ext)s")
    process = subprocess.Popen(
        [
            "yt-dlp",
            "-f",
            _FORMAT,
            "-o",
            outtmpl,
            # yt-dlp only draws its progress bar when stdout is a tty; --progress forces
            # it on regardless (needed since stdout is piped below), and --newline makes
            # it emit one line per update instead of redrawing a single line with `\r`.
            "--progress",
            "--newline",
            "--print",
            "after_move:filepath",
            url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []

    def _read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            if on_progress is not None:
                match = _PROGRESS_RE.search(line)
                if match:
                    on_progress(float(match.group(1)))

    # Read on a background thread rather than iterating process.stdout on the main one --
    # that blocks until the pipe closes (i.e. until the process exits), which would leave
    # the timeout below unable to interrupt a hung download the way subprocess.run's did.
    reader = threading.Thread(target=_read_output, daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        reader.join()
        raise
    reader.join()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, process.args, output="".join(lines))

    return Path(next(line for line in reversed(lines) if line.strip()).strip())
