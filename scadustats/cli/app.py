"""Command-line entry point for scadustats."""

import enum
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import typer

from scadustats.cli.display import render_match, render_match_table
from scadustats.models import GameResult, GameType, MatchMetadata, MatchType, VideoExtraction
from scadustats.pipeline.extract import estimate_sample_count, extract_video
from scadustats.rules.validation import validate_extraction
from scadustats.storage.db import load_json_dir
from scadustats.storage.json_export import read_video
from scadustats.video.download import download_video

# no_args_is_help: a bare `scadustats` prints the full command list rather than Typer's
# default "Missing command" error, which tells a first-time user nothing about what the
# commands actually are. Set on every Typer group here (see match_app below), so a bare
# sub-command group behaves the same way its parent does.
app = typer.Typer(no_args_is_help=True)


class IfExists(enum.StrEnum):
    replace = "replace"
    append = "append"
    error = "error"


# Both the full domain and its short-link form are genuinely YouTube -- this is a light,
# host-only check (not a reachability/existence check, which would need a network call
# and is more than "light"), so a link with a typo'd video id still passes.
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def _is_youtube_url(url: str) -> bool:
    return urlparse(url).hostname in _YOUTUBE_HOSTS


def _prompt_match_metadata(
    match_date_opt: str | None,
    season_opt: str | None,
    match_type_opt: MatchType | None,
    video_url_opt: str | None,
    existing_match: VideoExtraction | None = None,
) -> MatchMetadata:
    """Prompts for whichever of the match details weren't already supplied as CLI
    options. Run on a background thread by the `extract` command so it overlaps with the
    (much longer) extraction pipeline instead of blocking it.

    existing_match, when given (see _find_existing_match), is a previously-extracted match
    that came from this same source link (the same local path, or the same YouTube URL) --
    its match_date/season/video_url are offered as prompt defaults instead of the usual
    from-scratch ones, since re-extracting the same source (e.g. after a calibration fix)
    shouldn't mean retyping details already on record. Still just defaults: any of them
    can be overridden at the prompt, or by the matching --option, same as always.
    """
    match_date_val = (
        date.fromisoformat(match_date_opt)
        if match_date_opt
        else _prompt_match_date(existing_match.match_date if existing_match else None)
    )

    # season is free text (e.g. a one-off "Off-Season Cup"), not a number -- but most
    # seasons are numbered one per year, and 2026 is season 6, so that numbering still
    # makes a reasonable default. Just a default: the user can override it at the prompt.
    default_season = (
        existing_match.season if existing_match else str(match_date_val.year - 2020)
    )
    season = (
        season_opt
        if season_opt is not None
        else typer.prompt("Season", default=default_season, type=str)
    )

    match_type = match_type_opt or _prompt_match_type()

    video_url = (
        (video_url_opt.strip() or None)
        if video_url_opt is not None
        else _prompt_video_url(existing_match.video_url if existing_match else None)
    )

    return MatchMetadata(
        match_date=match_date_val, season=season, match_type=match_type, video_url=video_url
    )


def _prompt_match_date(default: date | None = None) -> date:
    # No default here, ordinarily -- unlike season/match type, the match date can't be
    # reasonably guessed (defaulting to "today" silently produced wrong data whenever
    # extraction wasn't run the same day the match was played), so the user must type one
    # in. The exception is a previously-extracted match found from the same source link
    # (see _find_existing_match): its match_date is a real recorded fact about this same
    # source, not a guess, so it's safe to offer as a default here.
    while True:
        raw = typer.prompt(
            "Match date (YYYY-MM-DD)", default=default.isoformat() if default else None
        )
        try:
            return date.fromisoformat(raw)
        except ValueError:
            typer.echo(f"Invalid date {raw!r}, expected YYYY-MM-DD")


def _prompt_match_type() -> MatchType:
    choices = [t.value for t in MatchType]
    while True:
        raw = typer.prompt(
            f"Match type ({'/'.join(choices)})", default=MatchType.DOUBLE_ELIMINATION.value
        )
        try:
            return MatchType(raw)
        except ValueError:
            typer.echo(f"Invalid match type {raw!r}, must be one of: {', '.join(choices)}")


def _prompt_video_url(default: str | None = None) -> str | None:
    # Unlike match date, this one really can be left blank -- e.g. extracting from a
    # video someone else downloaded, with no link handy. default, when given (see
    # _find_existing_match), is the video_url recorded on a previously-extracted match
    # from this same source link.
    while True:
        raw = typer.prompt(
            "Video URL (optional)", default=default or "", show_default=bool(default)
        ).strip()
        if not raw:
            return None
        if _is_youtube_url(raw):
            return raw
        typer.echo(f"{raw!r} doesn't look like a youtube.com/youtu.be URL")


def _prompt_game_type(game: GameResult, game_type_opt: GameType | None) -> GameType:
    """Answers extract_video's on_missing_game_type callback: game_type_opt (--game-type)
    is used for every game that needs it, if given, so the common case (a whole match is
    one type) doesn't mean answering the same question once per game. Otherwise prompts,
    per game, since squares.py couldn't infer it (see extract.py) -- no default, same
    reasoning as match date: guessing wrong here would silently mislabel real data.
    """
    if game_type_opt is not None:
        return game_type_opt

    choices = [t.value for t in GameType]
    while True:
        raw = typer.prompt(
            f"Game {game.game_index} type -- couldn't infer from its squares "
            f"({'/'.join(choices)})"
        )
        try:
            return GameType(raw)
        except ValueError:
            typer.echo(f"Invalid game type {raw!r}, must be one of: {', '.join(choices)}")


@app.command()
def download(
    url: Annotated[str, typer.Argument(help="YouTube video URL")],
    output_dir: Annotated[
        Path,
        typer.Option("-o", "--output-dir", help="Directory to save the video to"),
    ] = Path("downloads"),
) -> None:
    """Download a YouTube video."""
    result = download_video(url, output_dir=output_dir)
    print(result.path)


@app.command()
def extract(
    video_path_or_url: Annotated[
        str,
        typer.Argument(
            help="Path to a downloaded match video, or a youtube.com/youtu.be URL to "
            "download and extract in one step"
        ),
    ],
    download_dir: Annotated[
        Path,
        typer.Option(
            help="Where to download video_path_or_url to, if it's a URL (deleted "
            "afterward -- kept only if extraction fails and you choose to keep it)"
        ),
    ] = Path("downloads"),
    if_exists: Annotated[
        IfExists | None,
        typer.Option(
            help="Behavior when this exact match (same players and match date) was "
            "already extracted into json_dir -- omitted asks interactively"
        ),
    ] = None,
    json_dir: Annotated[
        Path,
        typer.Option(
            help="Directory to write one human-reviewable JSON file per game into"
        ),
    ] = Path("matches"),
    match_date: Annotated[
        str | None,
        typer.Option(help="Match date, YYYY-MM-DD (prompted if omitted)"),
    ] = None,
    season: Annotated[
        str | None,
        typer.Option(help="Season (prompted, defaulted from match date, if omitted)"),
    ] = None,
    match_type: Annotated[
        MatchType | None,
        typer.Option(help="double_elimination or playoffs (prompted if omitted)"),
    ] = None,
    video_url: Annotated[
        str | None,
        typer.Option(help="Source video URL, for provenance (prompted if omitted; may be empty)"),
    ] = None,
    game_type: Annotated[
        GameType | None,
        typer.Option(
            help="base or dlc, used for every game whose type can't be inferred from its "
            "squares (prompted per such game if omitted)"
        ),
    ] = None,
    keep_video: Annotated[
        bool,
        typer.Option(
            help="Don't delete the video downloaded from video_path_or_url after "
            "extraction (ignored for a locally-supplied video, which is never deleted "
            "either way)"
        ),
    ] = False,
) -> None:
    """Extract bingo board stats from a match video into JSON files in json_dir.

    video_path_or_url may be a local file (as before) or a youtube.com/youtu.be URL --
    given a URL, the video is downloaded first, then extracted, then deleted (unless
    --keep-video was given, or extraction fails, in which case you're asked whether to
    keep it).

    A match is identified by its players and match date (see video_id in the JSON) --
    if this exact match already has a JSON extraction in json_dir, you're asked whether
    to replace it, unless --if-exists was given to decide that upfront.

    If video_path_or_url matches the video_url or local source path of a match already
    extracted into json_dir, its match_date/season/video_url are offered as defaults at
    the prompts instead of the usual from-scratch ones -- handy when re-extracting the
    same source after a calibration fix.

    This only writes JSON -- it never touches a database. Run `load-db` separately
    (and optionally) to reflect that JSON into DuckDB.
    """
    # Checked here, upfront, rather than left to _prompt_match_metadata -- that runs on
    # a background thread overlapping the (multi-minute) extraction pipeline, so a bad
    # --video-url wouldn't surface until everything else had already finished. A blank
    # value is still fine unvalidated: video_url stays optional either way (see
    # _prompt_video_url), so only reject a value that was actually supplied.
    if video_url and not _is_youtube_url(video_url):
        raise typer.BadParameter(
            f"{video_url!r} doesn't look like a youtube.com/youtu.be URL",
            param_hint="--video-url",
        )

    # A previously-extracted match from this exact same source link (its video_url for a
    # YouTube URL, its source_path for a local file) -- if found, its match_date/season/
    # video_url are offered as prompt defaults below instead of asking from scratch, e.g.
    # when re-extracting the same source after a calibration fix. Looked up here, upfront
    # (a quick directory walk, unlike the metadata prompts), rather than inside
    # prompt_for_metadata on the background thread below, so its own JSON-parsing warnings
    # (see _read_matches) don't get interleaved with the progress bar the way a prompt
    # would need progress_lock to avoid.
    existing_match = _find_existing_match(json_dir, video_path_or_url)

    # video_path_or_url doubles as the source of the --video-url provenance field when
    # it's itself a YouTube URL and --video-url wasn't separately given -- the two are
    # the same link, so there's no reason to make a user paste it twice.
    downloaded_path: Path | None = None
    # Only known when this run actually downloaded the video -- read off that same
    # yt-dlp call (see video.download.DownloadResult), not looked up separately. A
    # locally-supplied video has no metadata to read it from, so this stays None and
    # VideoExtraction.published_at is simply left unrecorded for it.
    published_at: date | None = None
    if _is_youtube_url(video_path_or_url):
        if video_url is None:
            video_url = video_path_or_url
        typer.echo(f"Downloading {video_path_or_url}...")
        download_result = download_video(video_path_or_url, output_dir=download_dir)
        video_path = download_result.path
        published_at = download_result.published_at
        downloaded_path = video_path
        typer.echo(f"Downloaded to {video_path}")
    else:
        video_path = Path(video_path_or_url)

    # Wraps the whole pipeline (not just extract_video) so a failure anywhere -- a bad
    # video file, an OCR crash, whatever -- still gets the same downloaded-video cleanup
    # decision below, rather than only covering part of the run.
    try:
        total = estimate_sample_count(video_path)
        # The prompts run on a background thread concurrently with extraction itself (which
        # runs on the main thread below, as before) rather than before/after it -- prompting
        # takes seconds, extraction takes minutes, so this lets the user answer while it's
        # running instead of waiting idle. `progress_lock` keeps the bar and the prompts from
        # writing to the terminal at the same time without making extraction itself wait on
        # it: the prompt thread holds the lock for its whole Q&A sequence, and on_progress
        # -- called synchronously from the extraction loop -- only *tries* to acquire it,
        # accumulating ticks instead of blocking when it can't. Entering the progress bar
        # before starting the prompt thread means its first render can't race the prompts'
        # first line either. Net effect: the bar keeps redrawing in place on its own line
        # before and after the prompts, and the prompts print as their own clean lines
        # rather than getting interleaved into the bar's line.
        progress_lock = threading.Lock()
        with typer.progressbar(length=total, label="Extracting") as progress:
            pending = 0

            def on_progress() -> None:
                nonlocal pending
                pending += 1
                if progress_lock.acquire(blocking=False):
                    try:
                        progress.update(pending)
                        pending = 0
                    finally:
                        progress_lock.release()

            def prompt_for_metadata() -> MatchMetadata:
                with progress_lock:
                    typer.echo()  # fresh line, so this doesn't run into the bar's last redraw
                    return _prompt_match_metadata(
                        match_date, season, match_type, video_url, existing_match
                    )

            def on_missing_game_type(game: GameResult) -> GameType:
                # Runs on the main thread, after match_metadata has been awaited (see
                # extract_video) -- the prompt thread above is guaranteed done by then, so
                # progress_lock is free and this can't race its prompts on the terminal.
                with progress_lock:
                    typer.echo()
                    return _prompt_game_type(game, game_type)

            def on_duplicate(path: Path) -> bool:
                # Same timing guarantee as on_missing_game_type above -- this only runs
                # once video_id (and so path) is known, which is after match_metadata's
                # prompt thread has long since finished.
                with progress_lock:
                    typer.echo()
                    return typer.confirm(
                        f"{path} already exists -- replace it?", default=False
                    )

            with ThreadPoolExecutor(max_workers=1) as prompt_executor:
                metadata_future: Future[MatchMetadata] = prompt_executor.submit(
                    prompt_for_metadata
                )
                summary = extract_video(
                    video_path,
                    json_dir=json_dir,
                    if_exists=if_exists.value if if_exists is not None else None,
                    match_metadata=metadata_future,
                    on_progress=on_progress,
                    on_missing_game_type=on_missing_game_type,
                    on_duplicate=on_duplicate,
                    published_at=published_at,
                )

            # If extraction (typically minutes) finishes faster than the prompts (a handful
            # of seconds), on_progress can spend its whole run accumulating ticks it never
            # gets a chance to flush -- there's no sample left to trigger one, since sampling
            # is what calls on_progress in the first place. extract_video only returns once
            # the prompts are answered (it awaits match_metadata internally), so the lock is
            # guaranteed free here -- a plain flush is enough to land the bar on its true
            # final value instead of leaving it stuck wherever it was when prompting started.
            if pending:
                progress.update(pending)
                pending = 0
        if summary.skipped:
            typer.echo(f"Skipped -- kept the existing JSON extraction for {summary.video_id}")
        else:
            print(summary)
    except Exception:
        if (
            downloaded_path is not None
            and not keep_video
            and typer.confirm(
                f"Extraction failed -- delete the downloaded video at {downloaded_path}?",
                default=False,
            )
        ):
            downloaded_path.unlink(missing_ok=True)
        raise
    else:
        if downloaded_path is not None and not keep_video:
            downloaded_path.unlink(missing_ok=True)


@app.command("load-db")
def load_db(
    json_dir: Annotated[
        Path, typer.Argument(help="Directory of JSON files written by `extract`")
    ],
    db: Annotated[
        Path, typer.Option(help="DuckDB database file path")
    ] = Path("scadustats.duckdb"),
    if_exists: Annotated[
        IfExists,
        typer.Option(help="Behavior when a video's JSON was already loaded into the DB"),
    ] = IfExists.replace,
) -> None:
    """Reflect previously extracted JSON files into DuckDB.

    Separate from, and optional after, `extract` -- run this whenever you want the
    JSON's current contents (including any manual corrections) written into the DB.
    """
    video_ids = load_json_dir(db, json_dir, if_exists=if_exists.value)
    print(f"Loaded {len(video_ids)} video(s) into {db}")


def _match_path(json_dir: Path, video_id: str) -> Path:
    """The JSON file one video_id names, as a CLI error rather than a traceback when
    it isn't there -- a mistyped id is a user mistake, not a bug.

    video_id alone doesn't say which season subdirectory the file lives under (see
    json_export.video_path), so this searches json_dir for it rather than building the
    path directly -- a video_id is unique across the whole match history, so at most one
    match is ever expected.
    """
    matches = sorted(Path(json_dir).rglob(f"{video_id}.json"))
    if not matches:
        raise typer.BadParameter(
            f"no match file for {video_id!r} under {json_dir}", param_hint="video_id"
        )
    return matches[0]


match_app = typer.Typer(
    help="List or inspect previously extracted matches.", no_args_is_help=True
)
app.add_typer(match_app, name="match")


def _read_matches(json_dir: Path) -> list[VideoExtraction]:
    """Every match in json_dir, oldest first. A file that doesn't parse as a
    current-format extraction (e.g. one of the pre-existing one-file-per-game files
    predating the one-file-per-video format) is skipped with a warning on stderr rather
    than aborting the whole run -- shared by `match list` and `match validate`, which
    both walk the directory and both want that tolerance.

    Searches every season subdirectory (see json_export.video_path), not just json_dir
    itself. Sorted by filename alone rather than the full path: a filename is
    "<video_id>.json" and video_id is "<match-date>-<red>-vs-<blue>" (see
    extract._video_id), so filename order already is chronological order, whereas a
    season subdirectory's name (free text, not necessarily a sortable number) would not
    sort that way against another season's.
    """
    extractions = []
    for path in sorted(Path(json_dir).rglob("*.json"), key=lambda p: p.name):
        try:
            extractions.append(read_video(path))
        except (KeyError, ValueError) as exc:
            typer.echo(f"Skipping {path.name}: {exc}", err=True)
    return extractions


def _find_existing_match(json_dir: Path, link: str) -> VideoExtraction | None:
    """The previously-extracted match, if any, that came from the same source link as
    this `extract` run's video_path_or_url -- a YouTube URL is matched against that
    match's video_url, a local path against its source_path (see
    extract_video/models.VideoExtraction). Used by `extract` to offer that match's
    match_date/season/video_url as prompt defaults, so re-extracting the same source
    (e.g. after a calibration fix) doesn't mean retyping details already on record.

    Reuses _read_matches' directory walk, so an unparseable file is skipped the same way
    match list/validate already tolerate it, rather than failing this lookup outright.
    """
    field = "video_url" if _is_youtube_url(link) else "source_path"
    for extraction in _read_matches(json_dir):
        if getattr(extraction, field) == link:
            return extraction
    return None


@match_app.command("list")
def match_list(
    json_dir: Annotated[
        Path, typer.Option(help="Directory of JSON files written by `extract`")
    ] = Path("matches"),
) -> None:
    """List every match extracted into json_dir as a table: who played, the format, how
    many games, and how the match ended.

    Reads the JSON files directly rather than a database -- the JSON is this project's
    source of truth (see CLAUDE.md), so this works whether or not `load-db` has ever
    been run.
    """
    extractions = _read_matches(json_dir)
    if not extractions:
        typer.echo(f"No matches found in {json_dir}")
        return

    for line in render_match_table(extractions):
        typer.echo(line)
    count = len(extractions)
    typer.echo(f"\n{count} match{'es' if count != 1 else ''}")


@match_app.command("validate")
def match_validate(
    video_id: Annotated[
        str | None,
        typer.Argument(
            help="video_id to validate, as printed by `match list` (omitted validates "
            "every match in json_dir)"
        ),
    ] = None,
    json_dir: Annotated[
        Path, typer.Option(help="Directory of JSON files written by `extract`")
    ] = Path("matches"),
) -> None:
    """Check extracted matches against the tournament's own rules and report what
    doesn't add up.

    A reported issue means the JSON says something the rules say can't happen (a playoffs
    match with one game, a winner the board doesn't support, squares claimed after a line
    was completed) -- so an extraction mistake probably slipped through and that file
    needs a look. Exits non-zero if any match has issues, so this can gate a batch of
    extractions. A clean match is marked green, an issue red -- typer.echo (via click)
    strips the color codes automatically when the output isn't a terminal (piped to a
    file, or under CliRunner in tests), so this doesn't need its own --no-color flag.
    """
    if video_id is not None:
        extractions = [read_video(_match_path(json_dir, video_id))]
    else:
        extractions = _read_matches(json_dir)
        if not extractions:
            typer.echo(f"No matches found in {json_dir}")
            return

    checked = 0
    with_issues = 0
    for extraction in extractions:
        checked += 1
        issues = validate_extraction(extraction)
        if not issues:
            mark = typer.style("✓", fg=typer.colors.GREEN)
            status = typer.style("ok", fg=typer.colors.GREEN, bold=True)
            typer.echo(f"{mark} {extraction.video_id}: {status}")
            continue
        with_issues += 1
        mark = typer.style("✗", fg=typer.colors.RED)
        count = f"{len(issues)} issue{'s' if len(issues) != 1 else ''}"
        status = typer.style(count, fg=typer.colors.RED, bold=True)
        typer.echo(f"{mark} {extraction.video_id}: {status}")
        for issue in issues:
            scope = typer.style(f"{issue.scope}:", fg=typer.colors.CYAN)
            code = typer.style(f"[{issue.code}]", dim=True)
            typer.echo(f"    {scope} {issue.message} {code}")

    if checked > 1:
        summary = f"{with_issues} of {checked} matches have issues"
        color = typer.colors.RED if with_issues else typer.colors.GREEN
        typer.echo(f"\n{typer.style(summary, fg=color, bold=True)}")
    if with_issues:
        raise typer.Exit(1)


@match_app.command("show")
def match_show(
    video_id: Annotated[
        str, typer.Argument(help="video_id to show, as printed by `match list`")
    ],
    events: Annotated[
        bool,
        typer.Option(
            "--events/--no-events",
            help="Also print every game's events (marks, unmarks, game start) with the "
            "goal text of each square they touched",
        ),
    ] = False,
    json_dir: Annotated[
        Path, typer.Option(help="Directory of JSON files written by `extract`")
    ] = Path("matches"),
) -> None:
    """Print one match in full: its details, then per game the recorded result, how the
    squares ended up split, and the final board (with the winning line marked).

    The board is replayed from the game's own events, so a result the board doesn't
    support is visible right next to it -- `match validate` is what states that in so
    many words.
    """
    extraction = read_video(_match_path(json_dir, video_id))
    for line in render_match(extraction, events=events):
        typer.echo(line)


def main() -> None:
    app()
