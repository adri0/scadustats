"""Command-line entry point for scadustats."""

import enum
from pathlib import Path
from typing import Annotated

import typer

from scadustats.download import download_video
from scadustats.extract import estimate_sample_count, extract_video

app = typer.Typer()


class IfExists(enum.StrEnum):
    replace = "replace"
    append = "append"
    error = "error"


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
) -> None:
    """Extract bingo board stats from a match video into DuckDB."""
    total = estimate_sample_count(video_path)
    with typer.progressbar(length=total, label="Extracting") as progress:
        summary = extract_video(
            video_path,
            db_path=db,
            if_exists=if_exists.value,
            json_dir=json_dir,
            on_progress=lambda: progress.update(1),
        )
    print(summary)


def main() -> None:
    app()
