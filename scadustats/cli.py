"""Command-line entry point for scadustats."""

import enum
from pathlib import Path
from typing import Annotated

import typer

from scadustats.download import download_video
from scadustats.extract import extract_video

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
) -> None:
    """Extract bingo board stats from a match video into DuckDB."""
    summary = extract_video(video_path, db_path=db, if_exists=if_exists.value)
    print(summary)


def main() -> None:
    app()
