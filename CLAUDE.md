# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

scadustats extracts statistics from Elden Ring "Bingo Brawlers" match videos.

## Setup and commands

- `ffmpeg` must be installed on the system (e.g. `brew install ffmpeg`) — yt-dlp needs it to merge downloaded video/audio streams; without it, downloads fail with `DownloadError: ... ffmpeg is not installed`.
- `tesseract` must be installed on the system (e.g. `brew install tesseract`) — `pytesseract` shells out to it for OCR (goal text, player names, the game label, and the timer) during extraction.
- Dependency management and running: `uv` (not pip/venv directly).
- Install/sync environment: `uv sync`
- Run a script in the project environment: `uv run python -c "..."` or `uv run <entry point>`
- Build distributable (wheel/sdist): `uv build` (backed by hatchling)

## Architecture

- Build backend: `hatchling`, configured in `pyproject.toml`.
- Package layout is flat (no `src/` directory) — the importable package `scadustats` lives at the repo root, alongside `pyproject.toml`.
- `pyproject.toml` explicitly declares `[tool.hatch.build.targets.wheel] packages = ["scadustats"]` since the flat layout requires hatchling to be told which directory is the package.

### Downloading videos (`scadustats/download.py`)

- `download_video` shells out to the `yt-dlp` CLI binary via `subprocess.run` rather than driving the `yt_dlp` Python API. This keeps the invocation a direct, readable equivalent of the documented CLI command (`yt-dlp -f "bestvideo[height<=720]" <URL>`).
- Format is fixed to `bestvideo[height<=720]` — video-only, no audio track, capped at 720p — since this project only needs footage for stats extraction, not playback.
- The final path is recovered from `--print after_move:filepath`, which yt-dlp resolves after its internal move-to-destination step, rather than reconstructing the filename manually (extensions can vary by source format).

### Extracting stats (`scadustats/extract.py` and friends)

The Bingo Brawlers broadcast overlay (5x5 goal grid with per-cell claim color, a score bar per player, a stopwatch timer, a "GAME N" label) is a fixed template that scales proportionally with output resolution — confirmed by comparing a 1280x720 and a 1920x1080 recording of the same template. This shapes most of the design below:

- `layout.py` stores every overlay region (grid, timer, label, score bars) as **fractional** bounding boxes (0-1 of frame width/height), not pixels, so one calibrated config works across resolutions instead of needing per-resolution constants. Recalibrate with `scripts/calibrate_layout.py <video_path>` (draws the configured boxes on a frame for visual verification) if a future season changes the template.
- `board.py` classifies each cell's claim color in **HSV**, not raw BGR distance: an unclaimed cell is a low-saturation neutral gray at any brightness, while claimed cells are strongly saturated at a near-constant hue (red ≈177°, blue ≈104°, consistent across both calibration videos). A raw-BGR nearest-centroid approach was tried first and was fooled by a whole-frame brightness swing (a stream fade-in/flash before the overlay appears) that happened to land numerically closer to the blue centroid than to gray. Hue/saturation thresholds are kept narrow (not just "somewhat red/blue") to reject near-miss contamination from a progress-count badge that sometimes overlaps the sampled corner patch.
- Color is sampled from a small patch inset at each cell's **corner**, not its center, specifically to dodge the centered goal text and that progress badge.
- `extract.py`'s claim detection only accepts a color change once the **same new color persists across two consecutive samples** — this debounce is what actually stops the transient fade-in/flash artifact above (and similar single-frame flicker) from being recorded as a real claim, since color classification alone can't distinguish "real color" from "a transition effect that happens to match a reference hue."
- Any cell transition other than `UNCLAIMED -> RED/BLUE` (e.g. `RED -> BLUE`) is logged as a warning, not raised — a single misclassified frame must never abort a multi-hour extraction.
- `segmentation.py` requires **at least 2 of 3 independent signals** (board reset to all-unclaimed, timer reset near zero, "GAME N" label text change) to agree before declaring a new game boundary, since a single video can contain multiple back-to-back games and each signal alone is individually noisy (OCR misreads, classification flicker).
- The live claim count from `board.py` is cross-checked against the OCR'd scoreboard numbers on every count change, logging a warning on mismatch — free validation since the broadcast already renders that number.
- Re-running `extract` on an already-extracted video defaults to `--if-exists replace` (overwrite that video's prior rows) rather than accumulating duplicates, since iterating on calibration/bugfixes and re-extracting is the expected common workflow.
- `tests/fixtures/clip_claim.mp4` is a short (80s), re-encoded, full-resolution clip trimmed from a real match, covering exactly one known claim transition — downscaling it for size once broke goal-text OCR (crops became too small to read reliably), so it's kept at source resolution instead and just trimmed/re-encoded to stay small.
- All three committed fixtures (`hud_start.png`, `hud_partial.png`, `clip_claim.mp4`) have the webcam/face regions blacked out (players' and commentators' likenesses aren't needed for testing the overlay-reading logic) — done by zeroing everything outside the grid column and score-bar strip except the timer and game-label boxes, which sit inside those columns and are still needed. Regenerate via the same crop math in `layout.py` (grid/score-bar boxes define the columns to redact; timer/label boxes are restored on top) if new fixtures are ever captured, rather than committing raw frames.

## Testing

- Test runner: `pytest` (dev dependency, declared under `[dependency-groups] dev` in `pyproject.toml`).
- Tests that hit real external services (currently: downloading an actual YouTube video) are marked `@pytest.mark.integration`.
- `[tool.pytest.ini_options] addopts = "-m 'not integration'"` excludes integration tests by default, so a plain `uv run pytest` (or local dev loop) stays fast and offline.
- Run integration tests explicitly with `uv run pytest -m integration`.

## CI

- GitHub Actions workflow at `.github/workflows/ci.yml`, triggered on push to `main` and on pull requests.
- Steps: install `ffmpeg`, install `uv` (`astral-sh/setup-uv`), `uv sync --locked`, `uv build`, then run tests in two steps — a plain `uv run pytest` (unit tests, default marker filter applies) followed by `uv run pytest -m integration` (explicitly runs the integration suite) — so CI covers both, while local runs stay integration-free by default.
