# scadustats

Extract statistics from Elden Ring "Bingo Brawlers" match videos.

## Requirements

- `ffmpeg` must be installed on the system (e.g. `brew install ffmpeg`) — required by yt-dlp to merge downloaded video/audio streams.
- `tesseract` must be installed on the system (e.g. `brew install tesseract`) — required by `pytesseract` to OCR goal text, player names, commentator names, the game label, and the timer during extraction.
- If `download`/`extract` fails with a `DownloadError` mentioning that YouTube requires sign-in/authentication, pass a cookies file with `--cookies` (a Netscape-format `cookies.txt`, e.g. exported from a browser) — see [yt-dlp's guide to exporting YouTube cookies](https://github.com/yt-dlp/yt-dlp/wiki/extractors#exporting-youtube-cookies). Installing the optional `ejs` extra (`uv sync --extra ejs`, or `pip install scadustats[ejs]`), which pulls in [`yt-dlp-ejs`](https://github.com/yt-dlp/yt-dlp-ejs), can also help resolve this class of failure.

## Usage

```
scadustats download <youtube-url> [-o downloads]
scadustats extract <video_path_or_url> [--data-dir data] [--if-exists replace|append|error]
scadustats load-db [data_dir] [--db scadustats.duckdb] [--if-exists replace|append|error]
scadustats match list [--data-dir data]
scadustats match show <video_id> [--events] [--data-dir data]
scadustats match validate [video_id] [--data-dir data]
```

Run `scadustats` (or `scadustats match`) with no command to see this list with full help for each command.

`extract` reads the bingo-board overlay from a match video and writes one JSON file per video (all its games, players, commentators, match date, and video link included) under `<data_dir>/matches` — it never touches a database. `video_path_or_url` can be a local file, or a youtube.com/youtu.be URL to download and extract in one step (deleted afterward on success; kept, on request, if extraction fails). Each game is also classified as `base` or `dlc`, primarily by reading the overlay's own "BASE GAME"/"DLC" subtitle, falling back to matching its goal squares against `<data_dir>/squares/squares.json`, a growing reference of previously seen squares; when neither resolves it, an interactive prompt asks per game.

The `match` sub-commands are read-only lookups over what's already been extracted, straight from the JSON files (no database needed).

- `list` prints a table of every match: who played, the format, how many games, and how the match ended.
- `show` prints one match in full — per game its recorded result, how the squares ended up split, and the final board with the winning line marked. `--events` adds every mark and unmark with its timestamp, player, and goal text.
- `validate` checks matches against the tournament's own rules — game counts and order, a winner the board actually supports, one game-start per game, nothing claimed after a line was completed — printing every issue it finds and exiting non-zero if there were any. A reported issue means an extraction mistake probably slipped through and that file needs a look; with no `video_id` given, it checks every match in the directory.

`load-db` is a separate, optional step: it reflects those JSON files into a DuckDB database file. Run it whenever you want the JSON's current contents (including any manual corrections) written into the DB.
