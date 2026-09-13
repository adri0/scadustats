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


def _event_to_dict(event: GameEvent) -> dict:
    return {
        "row": event.row,
        "col": event.col,
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
            _event_to_dict(event)
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
    VideoExtraction it was serialized from."""
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
