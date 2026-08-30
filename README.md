# scadustats

Extract statistics from Elden Ring "Bingo Brawlers" match videos.

## Requirements

- `ffmpeg` must be installed on the system (e.g. `brew install ffmpeg`) — required by yt-dlp to merge downloaded video/audio streams.
- `tesseract` must be installed on the system (e.g. `brew install tesseract`) — required by `pytesseract` to OCR goal text, player names, the game label, and the timer during extraction.

## Usage

```
scadustats download <youtube-url> [-o downloads]
scadustats extract <video_path> [--db scadustats.duckdb] [--if-exists replace|append|error]
```

`extract` reads the bingo-board overlay from a downloaded match video and writes each claimed square (position, player color, in-game timestamp) plus computed game winners into a DuckDB database file.
