"""Human-reviewable JSON export of one video's full extraction, for manual
review/correction outside DuckDB, and the reverse: read_video parses one of these files
back into the VideoExtraction it was serialized from, for db.load_json_dir to reflect
into DuckDB.
"""

import json
from datetime import date
from pathlib import Path

from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameResult,
    GameType,
    MatchType,
    Square,
    VideoExtraction,
    WinLine,
    WinType,
)

# File a game type's consolidated squares reference is written to, keyed by GameType --
# see pipeline.consolidate.consolidate_squares and write_squares below. "base_game.json"
# rather than "base.json" so the filename reads unambiguously on its own, next to
# "dlc.json", in a directory listing.
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
    return {
        "row": event.row,
        "col": event.col,
        "square_text": _event_square_text(square_texts, event.row, event.col),
        "color": event.color.value if event.color else None,
        "event_type": event.event_type.value,
        "game_timer": _format_hms(event.game_elapsed_s),
        "video_ts_s": event.video_ts_s,
    }


def _game_to_dict(game: GameResult) -> dict:
    return {
        "game_index": game.game_index,
        "start_video_ts_s": game.start_video_ts_s,
        "end_video_ts_s": game.end_video_ts_s,
        "game_type": game.game_type.value if game.game_type else None,
        "winner_color": game.winner_color.value if game.winner_color else None,
        "win_type": game.win_type.value,
        "win_line": game.win_line.value if game.win_line else None,
        "square_texts": game.square_texts,
        "events": [
            _event_to_dict(event, game.square_texts)
            for event in sorted(game.events, key=lambda event: event.game_elapsed_s)
        ],
    }


def _metadata_to_dict(extraction: VideoExtraction) -> dict:
    """The fields that describe the *video file* itself, as opposed to the match it
    records -- grouped under their own "metadata" key so a reviewer skimming the file's
    match-identifying fields (video_id, players, season, ...) isn't interleaved with this
    provenance/technical detail. See CLAUDE.md and issue #47.
    """
    return {
        "video_url": extraction.video_url,
        "duration_s": extraction.duration_s,
        "published_at": extraction.published_at.isoformat() if extraction.published_at else None,
        "source_path": extraction.source_path,
        "extracted_at": extraction.extracted_at.isoformat(),
    }


def _extraction_to_dict(extraction: VideoExtraction) -> dict:
    return {
        "video_id": extraction.video_id,
        "match_date": extraction.match_date.isoformat(),
        "season": extraction.season,
        "match_type": extraction.match_type.value,
        "player_red_name": extraction.player_red_name,
        "player_blue_name": extraction.player_blue_name,
        "commentators": extraction.commentators,
        "metadata": _metadata_to_dict(extraction),
        # Written as a header for the list below, so a reviewer (or a SQL query against
        # the matching videos columns) sees the match's game count and result without
        # tallying the games by hand. All four are derived from `games` every time they're
        # written, and never read back -- see VideoExtraction.num_games and read_video.
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


def video_path(data_dir: str | Path, season: str, video_id: str) -> Path:
    """The path write_video writes (or would write) a video's extraction to --
    `<data_dir>/matches/season-<season>/<video_id>.json`. Shared with extract.py's
    duplicate check, so both agree on where a given (season, video_id) lives without
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
    return Path(data_dir) / "matches" / f"season-{season}" / f"{video_id}.json"


def write_video(
    data_dir: str | Path,
    extraction: VideoExtraction,
    if_exists: str = "replace",
) -> Path:
    """Write one video's full extraction (every game it contains) to
    `<data_dir>/matches/season-<season>/<video_id>.json` (see video_path), creating any
    missing directories. Returns the path written.

    if_exists="error" raises FileExistsError if the target file already exists.
    Any other value (including "append") overwrites unconditionally -- there's nothing
    meaningful to append into a single already-complete video's extraction file.
    """
    path = video_path(data_dir, extraction.season, extraction.video_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    if if_exists == "error" and path.exists():
        raise FileExistsError(f"match file {path} already exists")

    path.write_text(json.dumps(_extraction_to_dict(extraction), indent=2) + "\n")
    return path


def _parse_hms(text: str) -> int:
    hours, minutes, seconds = (int(part) for part in text.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _dict_to_event(data: dict) -> GameEvent:
    return GameEvent(
        row=data["row"],
        col=data["col"],
        color=CellColor(data["color"]) if data["color"] else None,
        video_ts_s=data["video_ts_s"],
        game_elapsed_s=_parse_hms(data["game_timer"]),
        event_type=EventType(data["event_type"]),
    )


def _dict_to_game(data: dict) -> GameResult:
    return GameResult(
        game_index=data["game_index"],
        start_video_ts_s=data["start_video_ts_s"],
        end_video_ts_s=data["end_video_ts_s"],
        square_texts=data["square_texts"],
        events=[_dict_to_event(event) for event in data["events"]],
        winner_color=CellColor(data["winner_color"]) if data["winner_color"] else None,
        win_type=WinType(data["win_type"]),
        # .get for the same reason as the extraction's duration_s: a file written before
        # win_line was recorded is still a valid current-format extraction.
        win_line=WinLine(data["win_line"]) if data.get("win_line") else None,
        game_type=GameType(data["game_type"]) if data["game_type"] else None,
    )


def read_video(path: str | Path) -> VideoExtraction:
    """Inverse of write_video: parses a JSON file it wrote back into the
    VideoExtraction it was serialized from.

    The file's `num_games`/`red_score`/`blue_score`/`winner` keys are deliberately ignored
    (like a pre-existing file's per-game `label`): all four are derived from `games`, so a
    hand-edited file that added a game, or corrected one's winner, is re-tallied from what
    it actually holds rather than trusted to have had every place updated in step.

    video_url/duration_s/published_at/source_path/extracted_at moved under a "metadata"
    key (issue #47). A file written before that lacks the key entirely; `metadata` falls
    back to `data` itself in that case, since those fields lived at the top level there --
    the .get() calls below then behave exactly as they did against the flat layout.
    """
    data = json.loads(Path(path).read_text())
    metadata = data.get("metadata", data)

    return VideoExtraction(
        video_id=data["video_id"],
        # .get, unlike every other field read directly off `data` here: a pre-#47 file
        # always had this key, but it's genuinely optional (see VideoExtraction.video_url)
        # so there's no reason to demand it be present.
        video_url=metadata.get("video_url"),
        match_date=date.fromisoformat(data["match_date"]),
        season=data["season"],
        match_type=MatchType(data["match_type"]),
        player_red_name=data["player_red_name"],
        player_blue_name=data["player_blue_name"],
        extracted_at=date.fromisoformat(metadata["extracted_at"]),
        games=[_dict_to_game(game) for game in data["games"]],
        # .get, for the same reason as duration_s below: a file written before
        # commentators were recorded reads back with none rather than a KeyError.
        commentators=data.get("commentators", []),
        # .get, unlike every other field here: a file written before duration_s existed
        # is still a valid current-format extraction and reads back as "unknown length",
        # rather than a KeyError that `match list` would report as an unparseable file.
        duration_s=metadata.get("duration_s"),
        # .get, for the same reason as duration_s: a file written before published_at
        # existed (or one whose fetch failed/was never attempted) still reads back
        # cleanly, as "unknown".
        published_at=(
            date.fromisoformat(metadata["published_at"]) if metadata.get("published_at") else None
        ),
        # .get, for the same reason as duration_s/published_at: a file written before
        # source_path existed still reads back cleanly, as "unknown".
        source_path=metadata.get("source_path"),
    )


def squares_path(squares_dir: str | Path, game_type: GameType) -> Path:
    """The path write_squares writes (or would write) one game type's consolidated
    squares reference to -- `<squares_dir>/base_game.json` or `<squares_dir>/dlc.json`.
    """
    return Path(squares_dir) / _SQUARES_FILENAMES[game_type]


def _square_to_dict(square: Square) -> dict:
    return {"id": square.id, "text": square.text, "game_type": square.game_type.value}


def write_squares(
    squares_dir: str | Path, squares: dict[GameType, list[Square]]
) -> dict[GameType, Path]:
    """Write pipeline.consolidate.consolidate_squares' output as one JSON file per game
    type (see squares_path), creating squares_dir if it doesn't exist yet. Each square is
    written sorted by id, for the same diffability reason consolidate_squares itself
    sorts by text -- id already reads close to alphabetical-by-text since it's derived
    from the text, so this doesn't reorder the list in any surprising way.

    Unconditionally overwrites, unlike write_video: this reference is wholly regenerated
    from the current match history every run, not incrementally appended to, so there's
    no prior version worth asking about before replacing. Returns the path written per
    game type.
    """
    Path(squares_dir).mkdir(parents=True, exist_ok=True)

    written = {}
    for game_type, game_squares in squares.items():
        path = squares_path(squares_dir, game_type)
        ordered = sorted(game_squares, key=lambda square: square.id)
        body = json.dumps([_square_to_dict(square) for square in ordered], indent=2)
        path.write_text(body + "\n")
        written[game_type] = path
    return written


def _dict_to_square(data: dict) -> Square:
    return Square(id=data["id"], text=data["text"], game_type=GameType(data["game_type"]))


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
        squares[game_type] = [_dict_to_square(entry) for entry in data]
    return squares
