"""Command-line entry point for scadustats."""

import argparse

from scadustats.download import download_video


def _download(args: argparse.Namespace) -> None:
    path = download_video(args.url, output_dir=args.output_dir)
    print(path)


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

    args = parser.parse_args(argv)
    args.func(args)
