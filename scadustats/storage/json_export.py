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
        return square_texts[row][col]
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


def _extraction_to_dict(extraction: VideoExtraction) -> dict:
    return {
        "video_id": extraction.video_id,
        "video_url": extraction.video_url,
        "match_date": extraction.match_date.isoformat(),
        "season": extraction.season,
        "match_type": extraction.match_type.value,
        "duration_s": extraction.duration_s,
        "player_red_name": extraction.player_red_name,
        "player_blue_name": extraction.player_blue_name,
        "commentators": extraction.commentators,
        "extracted_at": extraction.extracted_at.isoformat(),
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


def write_video(
    json_dir: str | Path,
    extraction: VideoExtraction,
    if_exists: str = "replace",
) -> Path:
    """Write one video's full extraction (every game it contains) to
    `<json_dir>/<video_id>.json`, creating json_dir if needed. Returns the path written.

    if_exists="error" raises FileExistsError if the target file already exists.
    Any other value (including "append") overwrites unconditionally -- there's nothing
    meaningful to append into a single already-complete video's extraction file.
    """
    json_dir = Path(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    path = json_dir / f"{extraction.video_id}.json"

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
    it actually holds rather than trusted to have had every place updated in step."""
    data = json.loads(Path(path).read_text())

    return VideoExtraction(
        video_id=data["video_id"],
        video_url=data["video_url"],
        match_date=date.fromisoformat(data["match_date"]),
        season=data["season"],
        match_type=MatchType(data["match_type"]),
        player_red_name=data["player_red_name"],
        player_blue_name=data["player_blue_name"],
        extracted_at=date.fromisoformat(data["extracted_at"]),
        games=[_dict_to_game(game) for game in data["games"]],
        # .get, for the same reason as duration_s below: a file written before
        # commentators were recorded reads back with none rather than a KeyError.
        commentators=data.get("commentators", []),
        # .get, unlike every other field here: a file written before duration_s existed
        # is still a valid current-format extraction and reads back as "unknown length",
        # rather than a KeyError that `match list` would report as an unparseable file.
        duration_s=data.get("duration_s"),
    )
