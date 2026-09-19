"""DuckDB schema and persistence for extracted match data."""

from pathlib import Path

import duckdb

from scadustats.models import VideoExtraction, VideoInfo
from scadustats.storage import json_export

_SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS events_seq START 1;

CREATE TABLE IF NOT EXISTS videos (
    match_id VARCHAR PRIMARY KEY,
    -- Nullable: load_json_dir writes a video row from previously-extracted JSON alone,
    -- with no source video file (or its resolution/fps) in hand.
    source_path VARCHAR,
    resolution_width INTEGER,
    resolution_height INTEGER,
    fps DOUBLE,
    -- The source video's full length. Nullable like the columns above, but for a
    -- different reason: it comes from the extraction itself (VideoExtraction.duration_s),
    -- not from a source video file, so it's present even via load_json_dir -- unless the
    -- JSON predates the field or had it cleared by hand.
    duration_s DOUBLE,
    video_url VARCHAR,
    match_date DATE NOT NULL,
    -- Free text, not a number: seasons aren't always named with a plain integer (see
    -- models.MatchMetadata.season).
    season VARCHAR NOT NULL,
    match_type VARCHAR NOT NULL,
    -- Player names live here, not on games -- one video is one match, and the same two
    -- players hold for every game in it (see models.VideoExtraction).
    player_red_name VARCHAR,
    player_blue_name VARCHAR,
    -- The date the video was (last) extracted/updated, as recorded in its JSON --
    -- *not* whatever moment load_json_dir happens to run at, which could be much later.
    extracted_at DATE NOT NULL,
    -- How many games the match consists of. Denormalized from the games table (it's
    -- always COUNT(*) of this video's rows there, see models.VideoExtraction.num_games)
    -- so the common "how long was this match" question is answerable off videos alone,
    -- without a join and a GROUP BY.
    num_games INTEGER NOT NULL,
    -- The match score and who took it, denormalized from games.winner_color for the same
    -- reason (see models.VideoExtraction.red_score/blue_score/winner). winner is nullable
    -- and the scores are not: the games always tally to *some* score, but no winner can
    -- be named when one of them has no winner recorded.
    red_score INTEGER NOT NULL,
    blue_score INTEGER NOT NULL,
    winner VARCHAR CHECK (winner IN ('red', 'blue', 'draw'))
);

CREATE TABLE IF NOT EXISTS commentators (
    match_id VARCHAR NOT NULL REFERENCES videos(match_id),
    -- Order within VideoExtraction.commentators (left nameplate first), not a seat id:
    -- a commentator whose plate didn't read is dropped from the list rather than left as
    -- a gap, so position 1 isn't necessarily the right-hand webcam. Its only job is to
    -- keep the list's order and the primary key unique -- which webcam someone sat in
    -- isn't a property of the match (see models.VideoExtraction).
    position INTEGER NOT NULL,
    name VARCHAR NOT NULL,
    PRIMARY KEY (match_id, position)
);

CREATE TABLE IF NOT EXISTS games (
    game_id VARCHAR PRIMARY KEY,
    match_id VARCHAR NOT NULL REFERENCES videos(match_id),
    game_index INTEGER NOT NULL,
    start_video_ts_s DOUBLE NOT NULL,
    end_video_ts_s DOUBLE,
    -- Nullable: extract_video always resolves this before writing JSON (inferred, or
    -- supplied when inference fails), but a hand-edited JSON file could clear it back
    -- to unknown, and that should load cleanly rather than fail a NOT NULL constraint.
    game_type VARCHAR CHECK (game_type IN ('base', 'dlc')),
    winner_color VARCHAR CHECK (winner_color IN ('red', 'blue')),
    win_type VARCHAR CHECK (win_type IN ('line', 'majority', 'tie', 'none')),
    -- Which line a win_type='line' win was completed on; NULL for every other win type,
    -- since there's no line to name. Row/column indices are 1-based, like the row/col
    -- columns on squares/events (see models.WinLine).
    win_line VARCHAR CHECK (win_line IN (
        'row_1', 'row_2', 'row_3', 'row_4', 'row_5',
        'col_1', 'col_2', 'col_3', 'col_4', 'col_5',
        'diagonal_tl_br', 'diagonal_bl_tr'
    ))
);

CREATE TABLE IF NOT EXISTS squares (
    game_id VARCHAR NOT NULL REFERENCES games(game_id),
    row INTEGER NOT NULL CHECK (row BETWEEN 1 AND 5),
    col INTEGER NOT NULL CHECK (col BETWEEN 1 AND 5),
    square_text VARCHAR NOT NULL,
    PRIMARY KEY (game_id, row, col)
);

CREATE TABLE IF NOT EXISTS events (
    event_id BIGINT PRIMARY KEY DEFAULT nextval('events_seq'),
    game_id VARCHAR NOT NULL REFERENCES games(game_id),
    row INTEGER,
    col INTEGER,
    color VARCHAR CHECK (color IN ('red', 'blue')),
    event_type VARCHAR NOT NULL
        CHECK (event_type IN ('mark', 'unmark', 'game_start', 'game_end')),
    game_elapsed_s INTEGER NOT NULL,
    video_ts_s DOUBLE NOT NULL,
    FOREIGN KEY (game_id, row, col) REFERENCES squares(game_id, row, col)
);

-- CREATE TABLE IF NOT EXISTS above leaves an already-created videos table alone, so a
-- database file made before duration_s existed would fail every insert below on an
-- unknown column. The database is disposable (it regenerates from JSON via load_json_dir
-- at any time), but failing rather than picking the new column up costs a contributor a
-- confusing error for no reason.
ALTER TABLE videos ADD COLUMN IF NOT EXISTS duration_s DOUBLE;
-- Same story for games.win_line. (A column added this way carries no CHECK constraint,
-- unlike one created with the table above -- the values still come from models.WinLine
-- either way, so the constraint is a backstop, not the thing keeping them valid.)
ALTER TABLE games ADD COLUMN IF NOT EXISTS win_line VARCHAR;
-- And for videos.num_games. (Nullable when added this way, unlike the NOT NULL above --
-- an ALTER can't retroactively fill a value in for rows already there. Every row written
-- from here on gets one regardless, since it comes from the extraction.)
ALTER TABLE videos ADD COLUMN IF NOT EXISTS num_games INTEGER;
-- Likewise for the match result. (Same two caveats as every ALTER above: nullable, since
-- a value can't be invented for rows already written, and no CHECK on winner -- the
-- values come from models.MatchWinner regardless.)
ALTER TABLE videos ADD COLUMN IF NOT EXISTS red_score INTEGER;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS blue_score INTEGER;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS winner VARCHAR;
-- And for the video's YouTube upload date (models.VideoExtraction.published_at) --
-- nullable for the usual ALTER reason, and also because it's only known when the video
-- was downloaded as part of this run (see video.download.DownloadResult) -- a locally-
-- supplied video leaves it empty even for one extracted after this field existed.
ALTER TABLE videos ADD COLUMN IF NOT EXISTS published_at DATE;
-- The commentators table needs no such fixup: CREATE TABLE IF NOT EXISTS does create a
-- table that an older database file simply doesn't have yet. Only *columns* added to a
-- table that already exists need the ALTERs above.
"""


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SCHEMA)


def _delete_video(con: duckdb.DuckDBPyConnection, match_id: str) -> None:
    con.execute(
        "DELETE FROM events WHERE game_id IN (SELECT game_id FROM games WHERE match_id = ?)",
        [match_id],
    )
    con.execute(
        "DELETE FROM squares WHERE game_id IN (SELECT game_id FROM games WHERE match_id = ?)",
        [match_id],
    )
    con.execute("DELETE FROM games WHERE match_id = ?", [match_id])
    con.execute("DELETE FROM commentators WHERE match_id = ?", [match_id])
    con.execute("DELETE FROM videos WHERE match_id = ?", [match_id])


def write_extraction(
    db_path: str | Path,
    extraction: VideoExtraction,
    source_path: str | None = None,
    video_info: VideoInfo | None = None,
    if_exists: str = "replace",
) -> None:
    """source_path/video_info are optional since load_json_dir calls this from
    previously-extracted JSON alone, with no source video file in hand -- resolution/fps
    are stored as NULL in that case."""
    con = duckdb.connect(str(db_path))
    try:
        init_schema(con)

        exists = con.execute(
            "SELECT 1 FROM videos WHERE match_id = ?", [extraction.match_id]
        ).fetchone()
        if exists:
            if if_exists == "error":
                raise ValueError(f"video {extraction.match_id!r} already extracted into {db_path}")
            if if_exists == "replace":
                _delete_video(con, extraction.match_id)

        # The extraction's own duration/source_path win over the caller-supplied ones when
        # both are in hand: they're what the JSON -- this project's source of truth --
        # records, including any hand correction. The caller-supplied values only fill in
        # for JSON written before those fields existed, where they're the one thing still
        # able to answer.
        duration_s = extraction.duration_s
        if duration_s is None and video_info is not None:
            duration_s = video_info.duration_s
        if extraction.source_path is not None:
            source_path = extraction.source_path

        con.execute(
            """INSERT INTO videos
               (match_id, source_path, resolution_width, resolution_height, fps,
                duration_s, video_url, match_date, season, match_type, player_red_name,
                player_blue_name, extracted_at, num_games, red_score, blue_score, winner,
                published_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                extraction.match_id,
                source_path,
                video_info.width if video_info else None,
                video_info.height if video_info else None,
                video_info.fps if video_info else None,
                duration_s,
                extraction.video_url,
                extraction.match_date,
                extraction.season,
                extraction.match_type.value,
                extraction.player_red_name,
                extraction.player_blue_name,
                extraction.extracted_at,
                extraction.num_games,
                extraction.red_score,
                extraction.blue_score,
                extraction.winner.value if extraction.winner else None,
                extraction.published_at,
            ],
        )

        # One row per commentator rather than a column on videos: the broadcast has
        # shown two, but nothing about the overlay guarantees that number, and a list
        # column would make "which matches did X cast?" a string search.
        for position, name in enumerate(extraction.commentators):
            con.execute(
                "INSERT INTO commentators (match_id, position, name) VALUES (?, ?, ?)",
                [extraction.match_id, position, name],
            )

        for game in extraction.games:
            game_id = f"{extraction.match_id}-{game.game_index}"
            con.execute(
                """INSERT INTO games
                   (game_id, match_id, game_index, start_video_ts_s, end_video_ts_s,
                    game_type, winner_color, win_type, win_line)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    game_id,
                    extraction.match_id,
                    game.game_index,
                    game.start_video_ts_s,
                    game.end_video_ts_s,
                    game.game_type.value if game.game_type else None,
                    game.winner_color.value if game.winner_color else None,
                    game.win_type.value,
                    game.win_line.value if game.win_line else None,
                ],
            )

            for row in range(5):
                for col in range(5):
                    # game.square_texts is a plain 0-based grid; the squares table's
                    # row/col columns are 1-based (see models.GameEvent), hence the +1s.
                    con.execute(
                        "INSERT INTO squares (game_id, row, col, square_text) VALUES (?, ?, ?, ?)",
                        [game_id, row + 1, col + 1, game.square_texts[row][col]],
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
    data_dir: str | Path,
    if_exists: str = "replace",
) -> list[str]:
    """Reflects every `*.json` video file (as written by json_export.write_video, one
    per season subdirectory -- see json_export.video_path) under `<data_dir>/matches`
    into the DuckDB at db_path -- the separate, optional process that turns extracted
    JSON into database rows. extract_video itself never touches the database; this is
    the only path that does. Returns the match_ids written, in filename order -- sorted
    by filename alone, not the full path, since a season subdirectory's name (free text,
    not necessarily a sortable number) doesn't sort chronologically against another
    season's.

    Walks `<data_dir>/matches` specifically, not data_dir itself -- data_dir can also
    hold other entities (e.g. `<data_dir>/squares/`, see pipeline.squares/pipeline.consolidate,
    issue #73), and read_video would raise trying to parse one of those as a video
    extraction.
    """
    matches_dir = Path(data_dir) / "matches"
    match_ids = []

    for path in sorted(matches_dir.rglob("*.json"), key=lambda p: p.name):
        extraction = json_export.read_video(path)
        write_extraction(db_path, extraction, if_exists=if_exists)
        match_ids.append(extraction.match_id)

    return match_ids
