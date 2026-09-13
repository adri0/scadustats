# scadustats

Extract statistics from Elden Ring "Bingo Brawlers" match videos.

## Requirements

- `ffmpeg` must be installed on the system (e.g. `brew install ffmpeg`) — required by yt-dlp to merge downloaded video/audio streams.
- `tesseract` must be installed on the system (e.g. `brew install tesseract`) — required by `pytesseract` to OCR goal text, player names, commentator names, the game label, and the timer during extraction.

## Usage

```
scadustats download <youtube-url> [-o downloads]
scadustats extract <video_path_or_url> [--json-dir matches] [--if-exists replace|append|error]
scadustats load-db <json_dir> [--db scadustats.duckdb] [--if-exists replace|append|error]
scadustats match list [json_dir]
scadustats match show <video_id> [--json-dir matches]
scadustats match validate [video_id] [--json-dir matches]
```

`extract` reads the bingo-board overlay from a match video and writes one JSON file per video (all its games, players, commentators, match date, and video link included) — it never touches a database. `video_path_or_url` can be a local file, or a youtube.com/youtu.be URL to download and extract in one step (deleted afterward on success; kept, on request, if extraction fails). Each game is also classified as `base` or `dlc` by matching its goal squares against `scadustats/squares.json`, a growing reference of previously seen squares; when a game's squares aren't in there yet, `--game-type` (or an interactive prompt) supplies it.

The `match` sub-commands are read-only lookups over what's already been extracted, straight from the JSON files (no database needed): `list` prints one line per match, `show` prints one match's full breakdown, and `validate` checks matches against the tournament's own rules — game counts and order, a winner the board actually supports, one game-start per game, nothing claimed after a line was completed — printing every issue it finds and exiting non-zero if there were any. A reported issue means an extraction mistake probably slipped through and that file needs a look; with no `video_id` given, `validate` checks every match in the directory.

`load-db` is a separate, optional step: it reflects those JSON files into a DuckDB database file. Run it whenever you want the JSON's current contents (including any manual corrections) written into the DB.
