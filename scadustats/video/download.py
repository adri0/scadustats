"""Download YouTube videos via yt-dlp."""

import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

_FORMAT = "bestvideo[height<=720]"

# Separates the two fields written by the single --print-to-file call below. A tab can't
# appear in either field (a filesystem path or an 8-digit date), so this is safe to split
# on unconditionally rather than needing a second temp file/subprocess call.
_FIELD_SEPARATOR = "\t"


@dataclass
class DownloadResult:
    path: Path
    # The video's upload date on YouTube (yt-dlp's `upload_date` field), read off the
    # same download -- no separate network call. None when yt-dlp didn't report one (a
    # video with no upload date on record), not when the download itself fails, which
    # still raises like before.
    published_at: date | None


def _parse_upload_date(raw: str) -> date | None:
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def download_video(
    url: str,
    output_dir: str | Path = "downloads",
    timeout: float | None = None,
    cookies_file: str | Path | None = None,
) -> DownloadResult:
    """Download the best video-only (no audio) stream up to 720p.

    `timeout` (seconds) bounds the yt-dlp subprocess; by default there is none,
    since real match videos can legitimately take a long time to download.

    `cookies_file`, when given, is passed through as yt-dlp's `--cookies` (a
    Netscape-format cookies.txt, e.g. exported from a browser) -- YouTube sometimes
    throttles/bot-detects anonymous requests, and authenticating this way is yt-dlp's
    own documented workaround rather than something this project reimplements.

    stdout/stderr are left inherited rather than captured, so yt-dlp's own progress
    bar prints directly to the terminal as it would from a plain CLI invocation. That
    means the final file path (and the upload date alongside it) can't be recovered
    from captured output, so `--print-to-file` is used instead to have yt-dlp write
    both to a temp file rather than to stdout, where they'd otherwise land in the
    middle of the progress output. One `--print-to-file` call rather than two: both
    fields come off the same already-fetched info dict, so there's no reason to shell
    out to yt-dlp a second time just to read the upload date separately.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(output_dir / "%(id)s.%(ext)s")
    with tempfile.NamedTemporaryFile(mode="r+") as info_file:
        cookies_args = ["--cookies", str(cookies_file)] if cookies_file else []
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                _FORMAT,
                "-o",
                outtmpl,
                *cookies_args,
                "--print-to-file",
                f"after_move:%(filepath)s{_FIELD_SEPARATOR}%(upload_date)s",
                info_file.name,
                url,
            ],
            timeout=timeout,
            check=True,
        )
        path_str, _, upload_date_str = info_file.read().strip().partition(_FIELD_SEPARATOR)
        return DownloadResult(path=Path(path_str), published_at=_parse_upload_date(upload_date_str))
