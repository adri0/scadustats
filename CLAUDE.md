# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

scadustats extracts statistics from Elden Ring "Bingo Brawlers" match videos.

## Setup and commands

- `ffmpeg` must be installed on the system (e.g. `brew install ffmpeg`) — yt-dlp needs it to merge downloaded video/audio streams; without it, downloads fail with `DownloadError: ... ffmpeg is not installed`.
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

## Testing

- Test runner: `pytest` (dev dependency, declared under `[dependency-groups] dev` in `pyproject.toml`).
- Tests that hit real external services (currently: downloading an actual YouTube video) are marked `@pytest.mark.integration`.
- `[tool.pytest.ini_options] addopts = "-m 'not integration'"` excludes integration tests by default, so a plain `uv run pytest` (or local dev loop) stays fast and offline.
- Run integration tests explicitly with `uv run pytest -m integration`.

## CI

- GitHub Actions workflow at `.github/workflows/ci.yml`, triggered on push to `main` and on pull requests.
- Steps: install `ffmpeg`, install `uv` (`astral-sh/setup-uv`), `uv sync --locked`, `uv build`, then run tests in two steps — a plain `uv run pytest` (unit tests, default marker filter applies) followed by `uv run pytest -m integration` (explicitly runs the integration suite) — so CI covers both, while local runs stay integration-free by default.
