"""Dev-only tool: rewrite match JSON files from the old raw-seconds `video_ts_s` event
key to the current `video_timestamp` (hh:mm:ss.ms) key (issue #109).

Not part of the installed package -- and not a required step, either: `json_export.
read_video` already tolerates the old key transparently, reformatting it in memory on
every read (see `json_export._dict_to_event`), so a match extracted before the rename
keeps working with every command as-is. This script is only for a contributor who wants
their on-disk files actually rewritten to the current format -- e.g. before hand-editing
one, so a diff against git history shows just their edit rather than the edit plus a
format migration. Usage:

    uv run python scripts/migrate_video_timestamp.py [--data-dir data] [--dry-run]

Walks <data_dir>/matches the same way `match list`/`db.load_json_dir` do. Only a file
that still carries the old `video_ts_s` key on at least one event is touched -- everything
else (a file already on the current format, or a non-match `*.json` living elsewhere
under data_dir, e.g. `<data_dir>/squares/`) is left alone. --dry-run lists what would be
rewritten without touching anything.
"""

import argparse
import json
from pathlib import Path

from scadustats.storage import json_export


def needs_migration(path: Path) -> bool:
    """True if any event in this match JSON still carries the old `video_ts_s` key --
    the exact condition `json_export._dict_to_event` falls back on when `video_timestamp`
    is absent, so this is precisely the set of files that read differently than they'd
    write today."""
    data = json.loads(path.read_text())
    return any(
        "video_ts_s" in event for game in data.get("games", []) for event in game.get("events", [])
    )


def migrate(data_dir: str | Path, *, dry_run: bool = False) -> list[Path]:
    """Rewrites every match JSON under <data_dir>/matches still using the old
    `video_ts_s` key to the current `video_timestamp` format, in place. Returns the paths
    touched (or, under dry_run, the paths that would be) -- an already-migrated file isn't
    included, so an empty return means there was nothing to do.
    """
    matches_dir = Path(data_dir) / "matches"
    touched = []
    for path in sorted(matches_dir.rglob("*.json")):
        if not needs_migration(path):
            continue
        touched.append(path)
        if dry_run:
            continue
        # read_video tolerates the old key (reformatting it), write_video always writes
        # the current format -- the round trip is the migration. write_video resolves the
        # same path this file already lives at (season + match_id, both unaffected by
        # this change), so this rewrites path in place rather than writing elsewhere.
        extraction = json_export.read_video(path)
        json_export.write_video(data_dir, extraction, if_exists="replace")
    return touched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="data directory written by `extract`")
    parser.add_argument(
        "--dry-run", action="store_true", help="list files that would be rewritten, without writing"
    )
    args = parser.parse_args()

    touched = migrate(args.data_dir, dry_run=args.dry_run)
    if not touched:
        print("No files using the old video_ts_s key found -- nothing to do.")
        return

    verb = "Would rewrite" if args.dry_run else "Rewrote"
    for path in touched:
        print(f"{verb} {path}")
    print(f"\n{verb.lower()} {len(touched)} file{'s' if len(touched) != 1 else ''}")


if __name__ == "__main__":
    main()
