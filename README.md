# scadustats

Extract match data from [Bingo Brawlers](https://bingobrawlers.com/) match VODs. Such as:

- Board layout & square texts
- Square mark/unmark events
- Start/end events for all games in a match video
- Game type (base game / DLC)
- Game result (winner of each game)
- Game duration
- Win type (by line [row, column, diagonal], by majority)
- Players (names and colours)
- Commentators
- Timer

From that data, derive consolidated views and statistics for each player, squares and game types.

> **Disclaimer**: This project is fan initiative, built out of appreciation for Bingo Brawlers and Elden Ring. Its only purpose is to provide more insights into the tournement. It is not affiliated with or endorsed by Bingo Brawlers, FromSoftware, or Bandai Namco.

## Requirements

- `ffmpeg` - required by yt-dlp to merge downloaded video/audio streams (e.g. `brew install ffmpeg` or `sudo apt install ffmpeg`)
- `tesseract` - OCR engine for extracting square text, player names, commentator names, the game label, and the timer (e.g. `brew install tesseract`)
- `leptonica`, `pkgconfig` - requirements for installing `tesserocr`, Python tesseract binding. (e.g. `brew install pkgconfig leptonica` on macOS, or `sudo apt install libtesseract-dev libleptonica-dev pkg-config` on Debian/Ubuntu).

## Installation

Requires Python 3.13+.

With [`uv`](https://docs.astral.sh/uv):

```
uv tool install scadustats
```

Or pip:

```
pip install scadustats
```

## Usage

```
scadustats extract <video_path_or_url> [--data-dir data] [--if-exists replace|append|error] [--consolidate]
scadustats match list [--data-dir data]
scadustats match show <match_id> [--events] [--data-dir data]
scadustats match validate [match_id] [--data-dir data]
scadustats match consolidate <match_id> [--data-dir data]
scadustats square consolidate [match_id] [--data-dir data]
scadustats player consolidate [match_id] [--data-dir data]
```

### Extract match data

`scadustats extract <video_path_or_url>` reads match data from a video and writes one JSON file per video (all its games, players, commentators, match date, and video link included) under `<data_dir>/matches`. `<video_path_or_url>` can be a local file, or a youtube.com/youtu.be URL to download and extract in one step (deleted afterward on success; kept, on request, if extraction fails). 

At the end of each extraction a validation is displayed for any inconsistencies. Equivalent of running `scadustats match validate <match_id>` for the recently ingested match.

### View extracted matches

The `match` sub-commands are read-only lookups over what's already been extracted, straight from the JSON files (no database needed) — except `consolidate`, which writes.

- `list` prints a table of every match: who played, the format, how many games, and how the match ended.
- `show` prints a summarised version of a match — per game its recorded result, how the squares ended up split, and the final board with the winning line marked. `--events` adds every mark and unmark with its timestamp, player, and goal text.
- `validate` checks if data of a match is consistent — game counts and order, a winner the board actually supports, one game-start per game, nothing claimed after a line was completed — printing every issue it finds and exiting non-zero if there were any. A reported issue means an extraction mistake probably slipped through and that file needs a look; with no `match_id` given, it checks every match.
- `consolidate` reconciles one match's squares against the squares reference and refreshes its two players' profiles — shorthand for running `square consolidate` and `player consolidate` scoped to that match.

### Square data

`square consolidate` reconciles squares against `<data_dir>/squares/base_game.json` and `<data_dir>/squares/dlc.json`: every distinct square text seen, split by game type and tagged with unique slug id (e.g. `{"id": "tunnels_3", "text": "Complete 3 Tunnels or Precipices", "game_type": "base"}`).

The goal is that square texts can be manually fixed for inconsistencies. Then future extractions use `squares/base_game.json` and `squares/dlc.json` for correcting OCR misreads in matches along the way. 

Given a `match_id`, it reconciles just that match; with none, it walks every match under `data_dir`, growing and correcting the reference incrementally rather than rebuilding it from scratch.

### Player data

`player consolidate` gives each player their own `<data_dir>/players/<slug>/` directory, holding two files:

- `stats.yaml`: their win/loss record (per season, and overall/per game type for individual games), every match they've played (most recent first), and their 5 most-claimed squares per game type, each with its mark count. Wholly regenerated every run -- this file is gitignored along with the rest of `data_dir`, it's disposable output re-derived from match history each time.
- `info.yaml`: just an `id`, assigned once for a newly seen player and kept stable after that. `twitch`/`avatar`/`bio` are never set by the tool at all -- fill those in by hand once the file exists, and this command never touches an existing player's `info.yaml` again, so hand edits always survive a re-run. Unlike `stats.yaml`, this one file is carved back out of `.gitignore` and meant to be committed to the repo.

Given a `match_id`, only that match's two players get their directories written/updated.

## Troubleshooting

### `DownloadError`

Sometimes `extract` fails with a `DownloadError` mentioning that YouTube requires sign-in/authentication. Pass a cookies file with `--cookies` (a Netscape-format `cookies.txt`, e.g. exported from a browser) — see [yt-dlp's guide to exporting YouTube cookies](https://github.com/yt-dlp/yt-dlp/wiki/extractors#exporting-youtube-cookies). Installing the optional `ejs` extra (`uv sync --extra ejs`, or `pip install scadustats[ejs]`), which pulls in [`yt-dlp-ejs`](https://github.com/yt-dlp/yt-dlp-ejs), can also help resolve this class of failure.
