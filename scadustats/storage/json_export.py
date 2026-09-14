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
    VideoExtraction,
    WinLine,
    WinType,
)


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


def video_path(json_dir: str | Path, season: str, video_id: str) -> Path:
    """The path write_video writes (or would write) a video's extraction to --
    `<json_dir>/season-<season>/<video_id>.json`. Shared with extract.py's duplicate
    check, so both agree on where a given (season, video_id) lives without either
    hand-rolling the layout: files are grouped one directory per season, rather than flat
    across every season a tournament has run, since json_dir otherwise only grows across
    seasons and a season is a natural, already-recorded grouping to browse it by. The
    `season-` prefix keeps the directory name unambiguous now that season is free text
    (see models.MatchMetadata.season) -- a bare season value could otherwise collide with
    another entry under json_dir, or (for a purely numeric season) look like something
    else entirely when browsing the directory tree.
    """
    return Path(json_dir) / f"season-{season}" / f"{video_id}.json"


def write_video(
    json_dir: str | Path,
    extraction: VideoExtraction,
    if_exists: str = "replace",
) -> Path:
    """Write one video's full extraction (every game it contains) to
    `<json_dir>/season-<season>/<video_id>.json` (see video_path), creating any missing
    directories. Returns the path written.

    if_exists="error" raises FileExistsError if the target file already exists.
    Any other value (including "append") overwrites unconditionally -- there's nothing
    meaningful to append into a single already-complete video's extraction file.
    """
    path = video_path(json_dir, extraction.season, extraction.video_id)
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
