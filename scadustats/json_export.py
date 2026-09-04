"""Human-reviewable JSON export of a single extracted game, for manual review/correction
outside DuckDB. Export only -- there's no importer back from JSON into the DB yet.
"""

import json
from pathlib import Path

from scadustats.models import GameEvent, GameResult


def game_id(video_id: str, game_index: int) -> str:
    """Same `<video_id>-<game_index>` convention as db.py's game_id (db.py:102), so JSON
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


def _game_to_dict(video_id: str, game: GameResult) -> dict:
    return {
        "game_id": game_id(video_id, game.game_index),
        "video_id": video_id,
        "game_index": game.game_index,
        "label": game.label,
        "start_video_ts_s": game.start_video_ts_s,
        "end_video_ts_s": game.end_video_ts_s,
        "player_red_name": game.player_red_name,
        "player_blue_name": game.player_blue_name,
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

    path.write_text(json.dumps(_game_to_dict(video_id, game), indent=2) + "\n")
    return path
