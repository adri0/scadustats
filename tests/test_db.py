import duckdb
import pytest

from scadustats.db import write_extraction
from scadustats.models import CellColor, ClaimEvent, GameResult, VideoInfo, WinType


def _sample_game() -> GameResult:
    square_texts = [[f"goal {r}-{c}" for c in range(5)] for r in range(5)]
    return GameResult(
        game_index=1,
        label="GAME 1",
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        player_red_name="alice",
        player_blue_name="bob",
        square_texts=square_texts,
        claims=[ClaimEvent(row=0, col=0, color=CellColor.RED, video_ts_s=10.0, game_elapsed_s=9)],
        winner_color=None,
        win_type=WinType.NONE,
    )


def test_write_and_read_extraction(tmp_path):
    db_path = tmp_path / "test.duckdb"
    video_info = VideoInfo(width=1280, height=720, fps=60.0, duration_s=100.0)

    write_extraction(db_path, "vid1", "downloads/vid1.mp4", video_info, [_sample_game()])

    con = duckdb.connect(str(db_path))
    try:
        assert con.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM squares").fetchone()[0] == 25
        assert con.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
    finally:
        con.close()


def test_replace_is_idempotent(tmp_path):
    db_path = tmp_path / "test.duckdb"
    video_info = VideoInfo(width=1280, height=720, fps=60.0, duration_s=100.0)

    write_extraction(db_path, "vid1", "downloads/vid1.mp4", video_info, [_sample_game()])
    write_extraction(db_path, "vid1", "downloads/vid1.mp4", video_info, [_sample_game()])

    con = duckdb.connect(str(db_path))
    try:
        assert con.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM squares").fetchone()[0] == 25
        assert con.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
    finally:
        con.close()


def test_if_exists_error_raises_on_duplicate(tmp_path):
    db_path = tmp_path / "test.duckdb"
    video_info = VideoInfo(width=1280, height=720, fps=60.0, duration_s=100.0)

    write_extraction(db_path, "vid1", "downloads/vid1.mp4", video_info, [_sample_game()])
    with pytest.raises(ValueError):
        write_extraction(
            db_path, "vid1", "downloads/vid1.mp4", video_info, [_sample_game()], if_exists="error"
        )
