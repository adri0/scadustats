"""Command-line entry point for scadustats."""

import enum
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from scadustats.download import download_video
from scadustats.extract import estimate_sample_count, extract_video
from scadustats.models import MatchMetadata, MatchType

app = typer.Typer()


class IfExists(enum.StrEnum):
    replace = "replace"
    append = "append"
    error = "error"


def _prompt_match_metadata(
    match_date_opt: str | None,
    season_opt: int | None,
    match_type_opt: MatchType | None,
) -> MatchMetadata:
    """Prompts for whichever of the three match details weren't already supplied as CLI
    options. Run on a background thread by the `extract` command so it overlaps with the
    (much longer) extraction pipeline instead of blocking it."""
    match_date_val = date.fromisoformat(match_date_opt) if match_date_opt else _prompt_match_date()

    # One season per year, and 2026 is season 6 -- so season = year - 2020. Just a
    # default: the user can override it at the prompt.
    default_season = match_date_val.year - 2020
    season = (
        season_opt
        if season_opt is not None
        else typer.prompt("Season", default=default_season, type=int)
    )

    match_type = match_type_opt or _prompt_match_type()

    return MatchMetadata(match_date=match_date_val, season=season, match_type=match_type)


def _prompt_match_date() -> date:
    # No default here -- unlike season/match type, the match date can't be reasonably
    # guessed (defaulting to "today" silently produced wrong data whenever extraction
    # wasn't run the same day the match was played), so the user must type one in.
    while True:
        raw = typer.prompt("Match date (YYYY-MM-DD)")
        try:
            return date.fromisoformat(raw)
        except ValueError:
            typer.echo(f"Invalid date {raw!r}, expected YYYY-MM-DD")


def _prompt_match_type() -> MatchType:
    choices = [t.value for t in MatchType]
    while True:
        raw = typer.prompt(
            f"Match type ({'/'.join(choices)})", default=MatchType.DOUBLE_ELIMINATION.value
        )
        try:
            return MatchType(raw)
        except ValueError:
            typer.echo(f"Invalid match type {raw!r}, must be one of: {', '.join(choices)}")


@app.command()
def download(
    url: Annotated[str, typer.Argument(help="YouTube video URL")],
    output_dir: Annotated[
        Path,
        typer.Option("-o", "--output-dir", help="Directory to save the video to"),
    ] = Path("downloads"),
) -> None:
    """Download a YouTube video."""
    path = download_video(url, output_dir=output_dir)
    print(path)


@app.command()
def extract(
    video_path: Annotated[Path, typer.Argument(help="Path to a downloaded match video")],
    db: Annotated[
        Path, typer.Option(help="DuckDB database file path")
    ] = Path("scadustats.duckdb"),
    if_exists: Annotated[
        IfExists,
        typer.Option(help="Behavior when this video was already extracted into the DB"),
    ] = IfExists.replace,
    json_dir: Annotated[
        Path | None,
        typer.Option(
            help="Directory to also write one human-reviewable JSON file per game into "
            "(optional; omit to skip JSON export)"
        ),
    ] = None,
    match_date: Annotated[
        str | None,
        typer.Option(help="Match date, YYYY-MM-DD (prompted if omitted)"),
    ] = None,
    season: Annotated[
        int | None,
        typer.Option(help="Season number (prompted, defaulted from match date, if omitted)"),
    ] = None,
    match_type: Annotated[
        MatchType | None,
        typer.Option(help="double_elimination or playoffs (prompted if omitted)"),
    ] = None,
) -> None:
    """Extract bingo board stats from a match video into DuckDB."""
    total = estimate_sample_count(video_path)
    # The prompts run on a background thread concurrently with extraction itself (which
    # runs on the main thread below, as before) rather than before/after it -- prompting
    # takes seconds, extraction takes minutes, so this lets the user answer while it's
    # running instead of waiting idle. `progress_lock` keeps the bar and the prompts from
    # writing to the terminal at the same time without making extraction itself wait on
    # it: the prompt thread holds the lock for its whole Q&A sequence, and on_progress
    # -- called synchronously from the extraction loop -- only *tries* to acquire it,
    # accumulating ticks instead of blocking when it can't. Entering the progress bar
    # before starting the prompt thread means its first render can't race the prompts'
    # first line either. Net effect: the bar keeps redrawing in place on its own line
    # before and after the prompts, and the prompts print as their own clean lines
    # rather than getting interleaved into the bar's line.
    progress_lock = threading.Lock()
    with typer.progressbar(length=total, label="Extracting") as progress:
        pending = 0

        def on_progress() -> None:
            nonlocal pending
            pending += 1
            if progress_lock.acquire(blocking=False):
                try:
                    progress.update(pending)
                    pending = 0
                finally:
                    progress_lock.release()

        def prompt_for_metadata() -> MatchMetadata:
            with progress_lock:
                typer.echo()  # fresh line, so this doesn't run into the bar's last redraw
                return _prompt_match_metadata(match_date, season, match_type)

        with ThreadPoolExecutor(max_workers=1) as prompt_executor:
            metadata_future: Future[MatchMetadata] = prompt_executor.submit(prompt_for_metadata)
            summary = extract_video(
                video_path,
                db_path=db,
                if_exists=if_exists.value,
                json_dir=json_dir,
                match_metadata=metadata_future,
                on_progress=on_progress,
            )

        # If extraction (typically minutes) finishes faster than the prompts (a handful
        # of seconds), on_progress can spend its whole run accumulating ticks it never
        # gets a chance to flush -- there's no sample left to trigger one, since sampling
        # is what calls on_progress in the first place. extract_video only returns once
        # the prompts are answered (it awaits match_metadata internally), so the lock is
        # guaranteed free here -- a plain flush is enough to land the bar on its true
        # final value instead of leaving it stuck wherever it was when prompting started.
        if pending:
            progress.update(pending)
            pending = 0
    print(summary)


def main() -> None:
    app()
