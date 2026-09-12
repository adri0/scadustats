import datetime

import duckdb
import pytest

from scadustats.db import load_json_dir, write_extraction
from scadustats.json_export import write_video
from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameResult,
    MatchType,
    VideoExtraction,
    VideoInfo,
    WinType,
)


def _sample_game(game_index: int = 1) -> GameResult:
    square_texts = [[f"goal {r}-{c}" for c in range(5)] for r in range(5)]
    return GameResult(
        game_index=game_index,
        label="GAME 1",
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        square_texts=square_texts,
        events=[GameEvent(row=0, col=0, color=CellColor.RED, video_ts_s=10.0, game_elapsed_s=9)],
        winner_color=None,
        win_type=WinType.NONE,
    )


def _sample_extraction(**overrides) -> VideoExtraction:
    defaults = dict(
        video_id="2026-03-05-alice-vs-bob",
        video_url="https://youtu.be/abc123",
        match_date=datetime.date(2026, 3, 5),
        season=6,
        match_type=MatchType.PLAYOFFS,
        player_red_name="alice",
        player_blue_name="bob",
        extracted_at=datetime.date(2026, 3, 6),
        games=[_sample_game()],
    )
    defaults.update(overrides)
    return VideoExtraction(**defaults)


def test_write_and_read_extraction(tmp_path):
    db_path = tmp_path / "test.duckdb"
    video_info = VideoInfo(width=1280, height=720, fps=60.0, duration_s=100.0)

    write_extraction(db_path, _sample_extraction(), "downloads/vid1.mp4", video_info)

    con = duckdb.connect(str(db_path))
    try:
        assert con.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM squares").fetchone()[0] == 25
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    finally:
        con.close()


def test_replace_is_idempotent(tmp_path):
    db_path = tmp_path / "test.duckdb"

    write_extraction(db_path, _sample_extraction())
    write_extraction(db_path, _sample_extraction())

    con = duckdb.connect(str(db_path))
    try:
        assert con.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM squares").fetchone()[0] == 25
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    finally:
        con.close()


def test_game_start_event_writes_with_null_row_col_color(tmp_path):
    db_path = tmp_path / "test.duckdb"
    game = _sample_game()
    game.events = [
        GameEvent(
            row=None,
            col=None,
            color=None,
            video_ts_s=1.0,
            game_elapsed_s=0,
            event_type=EventType.GAME_START,
        ),
        *game.events,
    ]

    write_extraction(db_path, _sample_extraction(games=[game]))

    con = duckdb.connect(str(db_path))
    try:
        row, col, color, event_type = con.execute(
            "SELECT row, col, color, event_type FROM events WHERE event_type = 'game_start'"
        ).fetchone()
        assert (row, col, color, event_type) == (None, None, None, "game_start")
    finally:
        con.close()


def test_video_row_carries_match_metadata_and_players(tmp_path):
    db_path = tmp_path / "test.duckdb"

    write_extraction(db_path, _sample_extraction())

    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT video_url, match_date, season, match_type, player_red_name, "
            "player_blue_name, extracted_at FROM videos WHERE video_id = ?",
            ["2026-03-05-alice-vs-bob"],
        ).fetchone()
        assert row == (
            "https://youtu.be/abc123",
            datetime.date(2026, 3, 5),
            6,
            "playoffs",
            "alice",
            "bob",
            datetime.date(2026, 3, 6),
        )
    finally:
        con.close()


def test_if_exists_error_raises_on_duplicate(tmp_path):
    db_path = tmp_path / "test.duckdb"

    write_extraction(db_path, _sample_extraction())
    with pytest.raises(ValueError):
        write_extraction(db_path, _sample_extraction(), if_exists="error")


def test_write_extraction_allows_missing_source_path_and_video_info(tmp_path):
    """load_json_dir has no source video file in hand -- both are optional and stored
    as NULL, not required."""
    db_path = tmp_path / "test.duckdb"

    write_extraction(db_path, _sample_extraction())

    con = duckdb.connect(str(db_path))
    try:
        source_path, width, height, fps = con.execute(
            "SELECT source_path, resolution_width, resolution_height, fps FROM videos "
            "WHERE video_id = ?",
            ["2026-03-05-alice-vs-bob"],
        ).fetchone()
        assert (source_path, width, height, fps) == (None, None, None, None)
    finally:
        con.close()


def test_load_json_dir_writes_all_games_from_one_video_file(tmp_path):
    json_dir = tmp_path / "json"
    db_path = tmp_path / "test.duckdb"
    extraction = _sample_extraction(games=[_sample_game(1), _sample_game(2)])
    write_video(json_dir, extraction)

    video_ids = load_json_dir(db_path, json_dir)

    assert video_ids == [extraction.video_id]
    con = duckdb.connect(str(db_path))
    try:
        assert con.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
        assert (
            con.execute(
                "SELECT COUNT(*) FROM games WHERE video_id = ?", [extraction.video_id]
            ).fetchone()[0]
            == 2
        )
        match_date, season, match_type = con.execute(
            "SELECT match_date, season, match_type FROM videos WHERE video_id = ?",
            [extraction.video_id],
        ).fetchone()
        assert match_date == datetime.date(2026, 3, 5)
        assert season == 6
        assert match_type == "playoffs"
    finally:
        con.close()


def test_load_json_dir_if_exists_error_raises_on_duplicate(tmp_path):
    json_dir = tmp_path / "json"
    db_path = tmp_path / "test.duckdb"
    write_video(json_dir, _sample_extraction())

    load_json_dir(db_path, json_dir)
    with pytest.raises(ValueError):
        load_json_dir(db_path, json_dir, if_exists="error")
