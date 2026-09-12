"""DuckDB schema and persistence for extracted match data."""

from pathlib import Path

import duckdb

from scadustats import json_export
from scadustats.models import GameResult, MatchMetadata, VideoInfo

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
    match_date DATE,
    season INTEGER,
    match_type VARCHAR,
    extracted_at TIMESTAMP NOT NULL
);

-- Backfills the three columns above onto a videos table created before match metadata
-- existed. IF NOT EXISTS makes this a no-op on a freshly created table too, so it's
-- safe to always run alongside the CREATE TABLE above. No CHECK constraint on
-- match_type here (unlike win_type/event_type below) -- retrofitting one via ALTER
-- TABLE ADD COLUMN isn't reliably supported, and MatchType already validates on the
-- Python side before insert.
ALTER TABLE videos ADD COLUMN IF NOT EXISTS match_date DATE;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS season INTEGER;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS match_type VARCHAR;

CREATE TABLE IF NOT EXISTS games (
    game_id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL REFERENCES videos(video_id),
    game_index INTEGER NOT NULL,
    label VARCHAR,
    start_video_ts_s DOUBLE NOT NULL,
    end_video_ts_s DOUBLE,
    player_red_name VARCHAR,
    player_blue_name VARCHAR,
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
    video_id: str,
    source_path: str | None,
    video_info: VideoInfo | None,
    games: list[GameResult],
    match_metadata: MatchMetadata | None = None,
    if_exists: str = "replace",
) -> None:
    """source_path/video_info are optional since load_json_dir calls this from
    previously-extracted JSON alone, with no source video file in hand -- resolution/fps/
    source_path are stored as NULL in that case."""
    con = duckdb.connect(str(db_path))
    try:
        init_schema(con)

        exists = con.execute(
            "SELECT 1 FROM videos WHERE video_id = ?", [video_id]
        ).fetchone()
        if exists:
            if if_exists == "error":
                raise ValueError(f"video {video_id!r} already extracted into {db_path}")
            if if_exists == "replace":
                _delete_video(con, video_id)

        con.execute(
            """INSERT INTO videos
               (video_id, source_path, resolution_width, resolution_height, fps,
                match_date, season, match_type, extracted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, now())""",
            [
                video_id,
                source_path,
                video_info.width if video_info else None,
                video_info.height if video_info else None,
                video_info.fps if video_info else None,
                match_metadata.match_date if match_metadata else None,
                match_metadata.season if match_metadata else None,
                match_metadata.match_type.value if match_metadata else None,
            ],
        )

        for game in games:
            game_id = f"{video_id}-{game.game_index}"
            con.execute(
                """INSERT INTO games
                   (game_id, video_id, game_index, label, start_video_ts_s, end_video_ts_s,
                    player_red_name, player_blue_name, winner_color, win_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    game_id,
                    video_id,
                    game.game_index,
                    game.label,
                    game.start_video_ts_s,
                    game.end_video_ts_s,
                    game.player_red_name,
                    game.player_blue_name,
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
    """Reflects every `*.json` game file (as written by json_export.write_game) under
    json_dir into the DuckDB at db_path -- the separate, optional process that turns
    extracted JSON into database rows. extract_video itself never touches the database;
    this is the only path that does.

    Games are grouped by video_id (one match's JSON files can span several games) before
    writing, since write_extraction persists one video at a time. Returns the video_ids
    written, in the order first encountered.
    """
    json_dir = Path(json_dir)
    games_by_video: dict[str, list[GameResult]] = {}
    metadata_by_video: dict[str, MatchMetadata | None] = {}

    for path in sorted(json_dir.glob("*.json")):
        video_id, game, match_metadata = json_export.read_game(path)
        games_by_video.setdefault(video_id, []).append(game)
        metadata_by_video[video_id] = match_metadata

    for video_id, games in games_by_video.items():
        games.sort(key=lambda game: game.game_index)
        write_extraction(
            db_path,
            video_id,
            None,
            None,
            games,
            metadata_by_video[video_id],
            if_exists=if_exists,
        )

    return list(games_by_video)
