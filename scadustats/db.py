"""DuckDB schema and persistence for extracted match data."""

from pathlib import Path

import duckdb

from scadustats import json_export
from scadustats.models import VideoExtraction, VideoInfo

_SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS events_seq START 1;

CREATE TABLE IF NOT EXISTS videos (
    video_id VARCHAR PRIMARY KEY,
    -- Nullable: load_json_dir writes a video row from previously-extracted JSON alone,
    -- with no source video file (or its resolution/fps) in hand.
    source_path VARCHAR,
    resolution_width INTEGER,
    resolution_height INTEGER,
    fps DOUBLE,
    video_url VARCHAR,
    match_date DATE NOT NULL,
    season INTEGER NOT NULL,
    match_type VARCHAR NOT NULL,
    -- Player names live here, not on games -- one video is one match, and the same two
    -- players hold for every game in it (see models.VideoExtraction).
    player_red_name VARCHAR,
    player_blue_name VARCHAR,
    -- The date the video was (last) extracted/updated, as recorded in its JSON --
    -- *not* whatever moment load_json_dir happens to run at, which could be much later.
    extracted_at DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
    game_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    game_index INTEGER NOT NULL,
    start_video_ts_s DOUBLE NOT NULL,
    end_video_ts_s DOUBLE,
    -- Nullable: extract_video always resolves this before writing JSON (inferred, or
    -- supplied when inference fails), but a hand-edited JSON file could clear it back
    -- to unknown, and that should load cleanly rather than fail a NOT NULL constraint.
    game_type VARCHAR CHECK (game_type IN ('base', 'dlc')),
    winner_color VARCHAR CHECK (winner_color IN ('red', 'blue')),
    win_type VARCHAR CHECK (win_type IN ('line', 'majority', 'tie', 'none'))
);

CREATE TABLE IF NOT EXISTS squares (
    game_id VARCHAR NOT NULL REFERENCES games(game_id),
    row INTEGER NOT NULL CHECK (row BETWEEN 0 AND 4),
    col INTEGER NOT NULL CHECK (col BETWEEN 0 AND 4),
    square_text VARCHAR NOT NULL,
    PRIMARY KEY (game_id, row, col)
);

CREATE TABLE IF NOT EXISTS events (
    event_id BIGINT PRIMARY KEY DEFAULT nextval('events_seq'),
    game_id VARCHAR NOT NULL REFERENCES games(game_id),
    row INTEGER,
    col INTEGER,
    color VARCHAR CHECK (color IN ('red', 'blue')),
    event_type VARCHAR NOT NULL CHECK (event_type IN ('mark', 'unmark', 'game_start')),
    game_elapsed_s INTEGER NOT NULL,
    video_ts_s DOUBLE NOT NULL,
    FOREIGN KEY (game_id, row, col) REFERENCES squares(game_id, row, col)
);
"""


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SCHEMA)


def _delete_video(con: duckdb.DuckDBPyConnection, video_id: str) -> None:
    con.execute(
        "DELETE FROM events WHERE game_id IN (SELECT game_id FROM games WHERE video_id = ?)",
        [video_id],
    )
    con.execute(
        "DELETE FROM squares WHERE game_id IN (SELECT game_id FROM games WHERE video_id = ?)",
        [video_id],
    )
    con.execute("DELETE FROM games WHERE video_id = ?", [video_id])
    con.execute("DELETE FROM videos WHERE video_id = ?", [video_id])


def write_extraction(
    db_path: str | Path,
    extraction: VideoExtraction,
    source_path: str | None = None,
    video_info: VideoInfo | None = None,
    if_exists: str = "replace",
) -> None:
    """source_path/video_info are optional since load_json_dir calls this from
    previously-extracted JSON alone, with no source video file in hand -- resolution/fps/
    source_path are stored as NULL in that case."""
    con = duckdb.connect(str(db_path))
    try:
        init_schema(con)

        exists = con.execute(
            "SELECT 1 FROM videos WHERE video_id = ?", [extraction.video_id]
        ).fetchone()
        if exists:
            if if_exists == "error":
                raise ValueError(f"video {extraction.video_id!r} already extracted into {db_path}")
            if if_exists == "replace":
                _delete_video(con, extraction.video_id)

        con.execute(
            """INSERT INTO videos
               (video_id, source_path, resolution_width, resolution_height, fps,
                video_url, match_date, season, match_type, player_red_name,
                player_blue_name, extracted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                extraction.video_id,
                source_path,
                video_info.width if video_info else None,
                video_info.height if video_info else None,
                video_info.fps if video_info else None,
                extraction.video_url,
                extraction.match_date,
                extraction.season,
                extraction.match_type.value,
                extraction.player_red_name,
                extraction.player_blue_name,
                extraction.extracted_at,
            ],
        )

        for game in extraction.games:
            game_id = f"{extraction.video_id}-{game.game_index}"
            con.execute(
                """INSERT INTO games
                   (game_id, video_id, game_index, start_video_ts_s, end_video_ts_s,
                    game_type, winner_color, win_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    game_id,
                    extraction.video_id,
                    game.game_index,
                    game.start_video_ts_s,
                    game.end_video_ts_s,
                    game.game_type.value if game.game_type else None,
                    game.winner_color.value if game.winner_color else None,
                    game.win_type.value,
                ],
            )

            for row in range(5):
                for col in range(5):
                    con.execute(
                        "INSERT INTO squares (game_id, row, col, square_text) VALUES (?, ?, ?, ?)",
                        [game_id, row, col, game.square_texts[row][col]],
                    )

            for event in game.events:
                con.execute(
                    """INSERT INTO events
                       (game_id, row, col, color, event_type, game_elapsed_s, video_ts_s)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [
                        game_id,
                        event.row,
                        event.col,
                        event.color.value if event.color else None,
                        event.event_type.value,
                        event.game_elapsed_s,
                        event.video_ts_s,
                    ],
                )
    finally:
        con.close()


def load_json_dir(
    db_path: str | Path,
    json_dir: str | Path,
    if_exists: str = "replace",
) -> list[str]:
    """Reflects every `*.json` video file (as written by json_export.write_video) under
    json_dir into the DuckDB at db_path -- the separate, optional process that turns
    extracted JSON into database rows. extract_video itself never touches the database;
    this is the only path that does. Returns the video_ids written, in filename order.
    """
    json_dir = Path(json_dir)
    video_ids = []

    for path in sorted(json_dir.glob("*.json")):
        extraction = json_export.read_video(path)
        write_extraction(db_path, extraction, if_exists=if_exists)
        video_ids.append(extraction.video_id)

    return video_ids
