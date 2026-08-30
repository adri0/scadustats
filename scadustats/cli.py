"""Command-line entry point for scadustats."""

import argparse

from scadustats.download import download_video
from scadustats.extract import extract_video


def _download(args: argparse.Namespace) -> None:
    path = download_video(args.url, output_dir=args.output_dir)
    print(path)


def _extract(args: argparse.Namespace) -> None:
    summary = extract_video(args.video_path, db_path=args.db, if_exists=args.if_exists)
    print(summary)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scadustats")
    subparsers = parser.add_subparsers(required=True)

    download_parser = subparsers.add_parser("download", help="Download a YouTube video.")
    download_parser.add_argument("url", help="YouTube video URL")
    download_parser.add_argument(
        "-o",
        "--output-dir",
        default="downloads",
        help="Directory to save the video to (default: downloads)",
    )
    download_parser.set_defaults(func=_download)

    extract_parser = subparsers.add_parser(
        "extract", help="Extract bingo board stats from a match video into DuckDB."
    )
    extract_parser.add_argument("video_path", help="Path to a downloaded match video")
    extract_parser.add_argument(
        "--db",
        default="scadustats.duckdb",
        help="DuckDB database file path (default: scadustats.duckdb)",
    )
    extract_parser.add_argument(
        "--if-exists",
        choices=["replace", "append", "error"],
        default="replace",
        help="Behavior when this video was already extracted into the DB (default: replace)",
    )
    extract_parser.set_defaults(func=_extract)

    args = parser.parse_args(argv)
    args.func(args)
