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

From that data, derive consolidated views of each player and existing squares and statistics.

## Requirements

- `ffmpeg` - required by yt-dlp to merge downloaded video/audio streams (e.g. `brew install ffmpeg` or `sudo apt install ffmpeg`)
- `tesseract` - OCR engine for extracting square text, player names, commentator names, the game label, and the timer (e.g. `brew install tesseract`)
- `leptonica`, `pkgconfig` - requirements for installing `tesserocr`, Python tesseract binding. (e.g. `brew install pkgconfig leptonica` on macOS, or `sudo apt install libtesseract-dev libleptonica-dev pkg-config` on Debian/Ubuntu).

## Usage

```
scadustats extract <video_path_or_url>
scadustats extract <video_path_or_url> [--data-dir data] [--if-exists replace|append|error] [--consolidate]
scadustats match list [--data-dir data]
scadustats match show <match_id> [--events] [--data-dir data]
scadustats match validate [match_id] [--data-dir data]
scadustats match consolidate <match_id> [--data-dir data]
scadustats square consolidate [match_id] [--data-dir data]
scadustats player consolidate [match_id] [--data-dir data]
```

Run `scadustats` (or `scadustats match`) with no command to see this list with full help for each command.

`extract` reads the bingo-board overlay from a match video and writes one JSON file per video (all its games, players, commentators, match date, and video link included) under `<data_dir>/matches` — it never touches a database. `video_path_or_url` can be a local file, or a youtube.com/youtu.be URL to download and extract in one step (deleted afterward on success; kept, on request, if extraction fails). Each game is also classified as `base` or `dlc`, primarily by reading the overlay's own "BASE GAME"/"DLC" subtitle, falling back to matching its goal squares against the consolidated squares reference (see `square consolidate` below); when neither resolves it, an interactive prompt asks per game. `--consolidate` runs `match consolidate` for the extracted match right after, as long as it passes validation cleanly.

The `match` sub-commands are read-only lookups over what's already been extracted, straight from the JSON files (no database needed) — except `consolidate`, which writes.

- `list` prints a table of every match: who played, the format, how many games, and how the match ended.
- `show` prints one match in full — per game its recorded result, how the squares ended up split, and the final board with the winning line marked. `--events` adds every mark and unmark with its timestamp, player, and goal text.
- `validate` checks matches against the tournament's own rules — game counts and order, a winner the board actually supports, one game-start per game, nothing claimed after a line was completed — printing every issue it finds and exiting non-zero if there were any. A reported issue means an extraction mistake probably slipped through and that file needs a look; with no `match_id` given, it checks every match in the directory.
- `consolidate` reconciles one match's squares against the squares reference and refreshes its two players' profiles — shorthand for running `square consolidate` and `player consolidate` scoped to that match.

`load-db` is a separate, optional step: it reflects those JSON files into a DuckDB database file. Run it whenever you want the JSON's current contents (including any manual corrections) written into the DB.

`square consolidate` reconciles goal squares against `<data_dir>/squares/base_game.json` and `<data_dir>/squares/dlc.json`: every distinct square text seen, split by game type and tagged with a short, unique id (e.g. `{"id": "tunnels_3", "text": "Complete 3 Tunnels or Precipices", "game_type": "base"}`), correcting OCR misreads in matches against it along the way. Given a `match_id`, it reconciles just that match; with none, it walks every match under `data_dir`, growing and correcting the reference incrementally rather than rebuilding it from scratch.

`player consolidate` rebuilds one YAML file per player under `<data_dir>/players/<slug>.yaml`: their win/loss record (per season, and overall/per game type for individual games), every match they've played (most recent first), and their 5 most-claimed squares per game type, each with its mark count. `id` is assigned once, the first time a player is seen, and kept stable after that; `twitch`/`avatar`/`bio` are never set by the tool — fill those in by hand and they survive every later re-run. Given a `match_id`, only that match's two players are written.

## Troubleshooting

### `DownloadError`

Sometimes `extract` fails with a `DownloadError` mentioning that YouTube requires sign-in/authentication. Pass a cookies file with `--cookies` (a Netscape-format `cookies.txt`, e.g. exported from a browser) — see [yt-dlp's guide to exporting YouTube cookies](https://github.com/yt-dlp/yt-dlp/wiki/extractors#exporting-youtube-cookies). Installing the optional `ejs` extra (`uv sync --extra ejs`, or `pip install scadustats[ejs]`), which pulls in [`yt-dlp-ejs`](https://github.com/yt-dlp/yt-dlp-ejs), can also help resolve this class of failure.
