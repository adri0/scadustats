"""Human-reviewable JSON export of a single extracted game, for manual review/correction
outside DuckDB, and the reverse: read_game parses one of these files back into the
models it was serialized from, for db.load_json_dir to reflect into DuckDB.
"""

import json
from datetime import date
from pathlib import Path

from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameResult,
    MatchMetadata,
    MatchType,
    WinType,
)


def game_id(video_id: str, game_index: int) -> str:
    """Same `<video_id>-<game_index>` convention as db.py's game_id, so JSON
    filenames and DB primary keys stay cross-referenceable."""
    return f"{video_id}-{game_index}"


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


def _game_to_dict(
    video_id: str, game: GameResult, match_metadata: MatchMetadata | None = None
) -> dict:
    return {
        "game_id": game_id(video_id, game.game_index),
        "video_id": video_id,
        "game_index": game.game_index,
        "label": game.label,
        "start_video_ts_s": game.start_video_ts_s,
        "end_video_ts_s": game.end_video_ts_s,
        "player_red_name": game.player_red_name,
        "player_blue_name": game.player_blue_name,
        "match_date": match_metadata.match_date.isoformat() if match_metadata else None,
        "season": match_metadata.season if match_metadata else None,
        "match_type": match_metadata.match_type.value if match_metadata else None,
        "winner_color": game.winner_color.value if game.winner_color else None,
        "win_type": game.win_type.value,
        "square_texts": game.square_texts,
        "events": [
            _event_to_dict(event)
            for event in sorted(game.events, key=lambda event: event.game_elapsed_s)
        ],
    }


def write_game(
    json_dir: str | Path,
    video_id: str,
    game: GameResult,
    match_metadata: MatchMetadata | None = None,
    if_exists: str = "replace",
) -> Path:
    """Write one game to `<json_dir>/<video_id>-<game_index>.json`, creating json_dir if
    needed. Returns the path written.

    if_exists="error" raises FileExistsError if the target file already exists.
    Any other value (including "append") overwrites unconditionally -- there's nothing
    meaningful to append into a single already-complete match file.
    """
    json_dir = Path(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    path = json_dir / f"{game_id(video_id, game.game_index)}.json"

    if if_exists == "error" and path.exists():
        raise FileExistsError(f"match file {path} already exists")

    path.write_text(json.dumps(_game_to_dict(video_id, game, match_metadata), indent=2) + "\n")
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


def read_game(path: str | Path) -> tuple[str, GameResult, MatchMetadata | None]:
    """Inverse of write_game: parses a JSON file it wrote back into the (video_id,
    GameResult, MatchMetadata) it was serialized from.

    Reads match_date/season/match_type with .get() rather than direct indexing --
    files written before match metadata collection existed (see matches/ for real
    examples) simply don't have those keys, which should read the same as the
    metadata-requested-but-declined case: no MatchMetadata, not an error.
    """
    data = json.loads(Path(path).read_text())

    match_metadata = None
    if data.get("match_date") is not None:
        match_metadata = MatchMetadata(
            match_date=date.fromisoformat(data["match_date"]),
            season=data["season"],
            match_type=MatchType(data["match_type"]),
        )

    game = GameResult(
        game_index=data["game_index"],
        label=data["label"],
        start_video_ts_s=data["start_video_ts_s"],
        end_video_ts_s=data["end_video_ts_s"],
        player_red_name=data["player_red_name"],
        player_blue_name=data["player_blue_name"],
        square_texts=data["square_texts"],
        events=[_dict_to_event(event) for event in data["events"]],
        winner_color=CellColor(data["winner_color"]) if data["winner_color"] else None,
        win_type=WinType(data["win_type"]),
    )
    return data["video_id"], game, match_metadata
