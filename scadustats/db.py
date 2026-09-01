"""DuckDB schema and persistence for extracted match data."""

from pathlib import Path

import duckdb

from scadustats.models import GameResult, VideoInfo

_SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS claims_seq START 1;

CREATE TABLE IF NOT EXISTS videos (
    video_id VARCHAR PRIMARY KEY,
    source_path VARCHAR NOT NULL,
    resolution_width INTEGER,
    resolution_height INTEGER,
    fps DOUBLE,
    extracted_at TIMESTAMP NOT NULL
);

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

CREATE TABLE IF NOT EXISTS claims (
    claim_id BIGINT PRIMARY KEY DEFAULT nextval('claims_seq'),
    game_id VARCHAR NOT NULL REFERENCES games(game_id),
    row INTEGER NOT NULL,
    col INTEGER NOT NULL,
    color VARCHAR NOT NULL CHECK (color IN ('red', 'blue')),
    event_type VARCHAR NOT NULL CHECK (event_type IN ('mark', 'unmark')),
    game_elapsed_s INTEGER NOT NULL,
    video_ts_s DOUBLE NOT NULL,
    FOREIGN KEY (game_id, row, col) REFERENCES squares(game_id, row, col)
);
"""


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SCHEMA)


def _delete_video(con: duckdb.DuckDBPyConnection, video_id: str) -> None:
    con.execute(
        "DELETE FROM claims WHERE game_id IN (SELECT game_id FROM games WHERE video_id = ?)",
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
    source_path: str,
    video_info: VideoInfo,
    games: list[GameResult],
    if_exists: str = "replace",
) -> None:
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
               (video_id, source_path, resolution_width, resolution_height, fps, extracted_at)
               VALUES (?, ?, ?, ?, ?, now())""",
            [video_id, source_path, video_info.width, video_info.height, video_info.fps],
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

            for claim in game.claims:
                con.execute(
                    """INSERT INTO claims
                       (game_id, row, col, color, event_type, game_elapsed_s, video_ts_s)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [
                        game_id,
                        claim.row,
                        claim.col,
                        claim.color.value,
                        claim.event_type.value,
                        claim.game_elapsed_s,
                        claim.video_ts_s,
                    ],
                )
    finally:
        con.close()
