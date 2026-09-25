"""Human-reviewable JSON export of one video's full extraction, for manual
review/correction outside DuckDB, and the reverse: read_video parses one of these files
back into the VideoExtraction it was serialized from, for db.load_json_dir to reflect
into DuckDB.
"""

import json
from pathlib import Path

from scadustats.models import GameEvent, GameResult, GameType, Square, VideoExtraction

# File a game type's consolidated squares reference is written to, keyed by GameType --
# see pipeline.consolidate.consolidate_match_squares and write_squares below.
# "base_game.json" rather than "base.json" so the filename reads unambiguously on its
# own, next to "dlc.json", in a directory listing.
_SQUARES_FILENAMES = {GameType.BASE: "base_game.json", GameType.DLC: "dlc.json"}


def _format_hms(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _event_square_text(
    square_texts: list[list[str]], row: int | None, col: int | None
) -> str | None:
    """The goal text at an event's square, looked up from the game's own square_texts
    grid -- so a reviewer reading the events list doesn't have to cross-reference row/col
    against that grid by hand. None for a game-level event (GAME_START, which carries no
    row/col) or when the grid doesn't reach that far -- square_texts is OCR output and
    can be hand-edited into a ragged grid, and that shouldn't take down export over one
    missing cell.
    """
    if row is None or col is None:
        return None
    try:
        # row/col are 1-based (see models.GameEvent); square_texts is a plain 0-based grid.
        return square_texts[row - 1][col - 1]
    except IndexError:
        return None


def _event_to_dict(event: GameEvent, square_texts: list[list[str]]) -> dict:
    # model_dump(mode="json") already turns color/event_type into their JSON-safe values
    # (None passed through as-is for color) -- game_elapsed_s is the one field that needs
    # reshaping (formatted as game_timer) rather than carrying straight through, and
    # square_text isn't a GameEvent field at all (see its own docstring above).
    dumped = event.model_dump(mode="json")
    return {
        "row": dumped["row"],
        "col": dumped["col"],
        "square_text": _event_square_text(square_texts, event.row, event.col),
        "color": dumped["color"],
        "event_type": dumped["event_type"],
        "game_timer": _format_hms(dumped["game_elapsed_s"]),
        "video_ts_s": dumped["video_ts_s"],
    }


def _game_to_dict(game: GameResult) -> dict:
    dumped = game.model_dump(mode="json")
    return {
        "game_index": dumped["game_index"],
        "start_video_ts_s": dumped["start_video_ts_s"],
        "end_video_ts_s": dumped["end_video_ts_s"],
        "game_type": dumped["game_type"],
        "winner_color": dumped["winner_color"],
        "win_type": dumped["win_type"],
        "win_line": dumped["win_line"],
        "square_texts": dumped["square_texts"],
        "events": [
            _event_to_dict(event, game.square_texts)
            for event in sorted(game.events, key=lambda event: event.game_elapsed_s)
        ],
    }


def _metadata_to_dict(extraction: VideoExtraction) -> dict:
    """The fields that describe the *video file* itself, as opposed to the match it
    records -- grouped under their own "metadata" key so a reviewer skimming the file's
    match-identifying fields (match_id, players, season, ...) isn't interleaved with this
    provenance/technical detail. See CLAUDE.md and issue #47.
    """
    dumped = extraction.model_dump(mode="json")
    return {
        "video_url": dumped["video_url"],
        "duration_s": dumped["duration_s"],
        "published_at": dumped["published_at"],
        "extracted_at": dumped["extracted_at"],
    }


def _extraction_to_dict(extraction: VideoExtraction) -> dict:
    dumped = extraction.model_dump(mode="json")
    return {
        "match_id": dumped["match_id"],
        "match_date": dumped["match_date"],
        "season": dumped["season"],
        "match_type": dumped["match_type"],
        "player_red_name": dumped["player_red_name"],
        "player_blue_name": dumped["player_blue_name"],
        "commentators": dumped["commentators"],
        "metadata": _metadata_to_dict(extraction),
        # Written as a header for the list below, so a reviewer (or a SQL query against
        # the matching videos columns) sees the match's game count and result without
        # tallying the games by hand. All four are derived properties, not model fields --
        # model_dump() above doesn't see them -- and are never read back, see
        # VideoExtraction.num_games and read_video.
        "num_games": extraction.num_games,
        "red_score": extraction.red_score,
        "blue_score": extraction.blue_score,
        # null when no winner can be named: a game (or the whole match) has none recorded.
        "winner": extraction.winner.value if extraction.winner else None,
        "games": [
            _game_to_dict(game)
            for game in sorted(extraction.games, key=lambda game: game.game_index)
        ],
    }


def video_path(data_dir: str | Path, season: str, match_id: str) -> Path:
    """The path write_video writes (or would write) a video's extraction to --
    `<data_dir>/matches/season-<season>/<match_id>.json`. Shared with extract.py's
    duplicate check, so both agree on where a given (season, match_id) lives without
    either hand-rolling the layout: files are grouped one directory per season, rather
    than flat across every season a tournament has run, since the matches folder
    otherwise only grows across seasons and a season is a natural, already-recorded
    grouping to browse it by. The `season-` prefix keeps the directory name unambiguous
    now that season is free text (see models.MatchMetadata.season) -- a bare season value
    could otherwise collide with another entry under the matches folder, or (for a purely
    numeric season) look like something else entirely when browsing the directory tree.

    The `matches` subfolder under data_dir (issue #73) is what makes data_dir a root a
    contributor can also point other entities at -- e.g. `<data_dir>/squares/` (see
    pipeline.consolidate/pipeline.squares) -- rather than a single flat directory that
    only ever held match JSON.
    """
    return Path(data_dir) / "matches" / f"season-{season}" / f"{match_id}.json"


def write_video(
    data_dir: str | Path,
    extraction: VideoExtraction,
    if_exists: str = "replace",
) -> Path:
    """Write one video's full extraction (every game it contains) to
    `<data_dir>/matches/season-<season>/<match_id>.json` (see video_path), creating any
    missing directories. Returns the path written.

    if_exists="error" raises FileExistsError if the target file already exists.
    Any other value (including "append") overwrites unconditionally -- there's nothing
    meaningful to append into a single already-complete video's extraction file.
    """
    path = video_path(data_dir, extraction.season, extraction.match_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    if if_exists == "error" and path.exists():
        raise FileExistsError(f"match file {path} already exists")

    path.write_text(json.dumps(_extraction_to_dict(extraction), indent=2) + "\n")
    return path


def _parse_hms(text: str) -> int:
    hours, minutes, seconds = (int(part) for part in text.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _dict_to_event(data: dict) -> GameEvent:
    """game_timer (a formatted HH:MM:SS string) is the one key that doesn't map directly
    onto a GameEvent field -- parsed into game_elapsed_s here before handing off to
    pydantic. Everything else (including tolerating a missing/null color, and ignoring the
    square_text key entirely -- it isn't a GameEvent field, see the module docstring above
    GameEvent) is handled by GameEvent's own validation for free.
    """
    return GameEvent.model_validate({**data, "game_elapsed_s": _parse_hms(data["game_timer"])})


def _dict_to_game(data: dict) -> GameResult:
    """win_line/game_type missing or explicitly null (a file written before either was
    recorded) both resolve to GameResult's own None default automatically -- no .get()
    needed here the way the hand-written version used to."""
    return GameResult.model_validate(
        {**data, "events": [_dict_to_event(event) for event in data["events"]]}
    )


def read_video(path: str | Path) -> VideoExtraction:
    """Inverse of write_video: parses a JSON file it wrote back into the
    VideoExtraction it was serialized from.

    The file's `num_games`/`red_score`/`blue_score`/`winner` keys are deliberately ignored
    (like a pre-existing file's per-game `label`): all four are derived properties, not
    VideoExtraction fields, so pydantic's default extra="ignore" already drops them here --
    a hand-edited file that added a game, or corrected one's winner, is re-tallied from
    what it actually holds rather than trusted to have had every place updated in step.

    video_url/duration_s/published_at/extracted_at moved under a "metadata" key (issue
    #47). A file written before that lacks the key entirely; `metadata` falls back to
    `data` itself in that case, since those fields lived at the top level there -- the
    explicit metadata.get() calls below then behave exactly as they did against the flat
    layout, and everything else -- season's int-to-str coercion, match_date/extracted_at/
    published_at's ISO-string parsing, match_type's enum lookup, commentators defaulting to
    [] when absent -- is handled by VideoExtraction's own validation.
    """
    data = json.loads(Path(path).read_text())
    metadata = data.get("metadata", data)

    return VideoExtraction.model_validate(
        {
            **data,
            "video_url": metadata.get("video_url"),
            "extracted_at": metadata["extracted_at"],
            "duration_s": metadata.get("duration_s"),
            "published_at": metadata.get("published_at"),
            "games": [_dict_to_game(game) for game in data["games"]],
        }
    )


def squares_path(squares_dir: str | Path, game_type: GameType) -> Path:
    """The path write_squares writes (or would write) one game type's consolidated
    squares reference to -- `<squares_dir>/base_game.json` or `<squares_dir>/dlc.json`.
    """
    return Path(squares_dir) / _SQUARES_FILENAMES[game_type]


def write_squares(
    squares_dir: str | Path, squares: dict[GameType, list[Square]]
) -> dict[GameType, Path]:
    """Write pipeline.consolidate.consolidate_match_squares' grown/corrected reference as
    one JSON file per game type (see squares_path), creating squares_dir if it doesn't
    exist yet. Each square is written sorted by id -- id already reads close to
    alphabetical-by-text since it's derived from the text, so this doesn't reorder the
    list in any surprising way -- purely so a re-run against an unchanged reference
    produces a byte-identical, diffable file rather than one reordered by whatever order
    the squares happened to be added in this run.

    Unconditionally overwrites, unlike write_video: the caller (cli.app's
    _consolidate_one_match) already read whatever this held before via read_squares and
    folded its own changes into that same dict before calling this, so there's no prior
    on-disk version left to ask about before replacing -- this only ever writes what the
    caller already knows supersedes it. Returns the path written per game type.
    """
    Path(squares_dir).mkdir(parents=True, exist_ok=True)

    written = {}
    for game_type, game_squares in squares.items():
        path = squares_path(squares_dir, game_type)
        ordered = sorted(game_squares, key=lambda square: square.id)
        body = json.dumps([square.model_dump(mode="json") for square in ordered], indent=2)
        path.write_text(body + "\n")
        written[game_type] = path
    return written


def read_squares(squares_dir: str | Path) -> dict[GameType, list[Square]]:
    """Inverse of write_squares: reads squares_dir/base_game.json and
    squares_dir/dlc.json back into the Square lists they were serialized from. A missing
    file -- squares_dir hasn't had `square consolidate` run against it yet -- contributes
    an empty list for that game type rather than raising, the same "ships empty" tolerance
    squares.json used to have (issue #75). This is also what
    pipeline.consolidate.consolidate_match_squares uses to match a match's own OCR'd
    square_texts against (issue #76).
    """
    squares: dict[GameType, list[Square]] = {}
    for game_type in GameType:
        try:
            data = json.loads(squares_path(squares_dir, game_type).read_text())
        except FileNotFoundError:
            data = []
        squares[game_type] = [Square.model_validate(entry) for entry in data]
    return squares
