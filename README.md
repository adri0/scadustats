# scadustats

Extract statistics from Elden Ring "Bingo Brawlers" match videos.

## Requirements

- `ffmpeg` must be installed on the system (e.g. `brew install ffmpeg`) — required by yt-dlp to merge downloaded video/audio streams.
- `tesseract` must be installed on the system (e.g. `brew install tesseract`) — required by `pytesseract` to OCR goal text, player names, the game label, and the timer during extraction.

## Usage

```
scadustats download <youtube-url> [-o downloads]
scadustats extract <video_path> [--json-dir matches] [--if-exists replace|append|error]
scadustats load-db <json_dir> [--db scadustats.duckdb] [--if-exists replace|append|error]
```

`extract` reads the bingo-board overlay from a downloaded match video and writes one JSON file per video (all its games, players, match date, and video link included) — it never touches a database. Each game is also classified as `base` or `dlc` by matching its goal squares against `scadustats/squares.json`, a growing reference of previously seen squares; when a game's squares aren't in there yet, `--game-type` (or an interactive prompt) supplies it.

`load-db` is a separate, optional step: it reflects those JSON files into a DuckDB database file. Run it whenever you want the JSON's current contents (including any manual corrections) written into the DB.
