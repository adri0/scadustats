import duckdb
import pytest

from scadustats.extract import extract_video

# A short (80s), downscaled, re-encoded clip trimmed from a real match video, covering
# exactly one known claim transition (see CLAUDE.md for how it was produced) -- small
# enough to commit directly, unlike the multi-GB originals in downloads/.
_CLIP_PATH = "tests/fixtures/clip_claim.mp4"


@pytest.mark.integration
def test_extract_video_end_to_end(tmp_path):
    db_path = tmp_path / "extract.duckdb"

    summary = extract_video(_CLIP_PATH, db_path=db_path)

    assert summary.num_games == 1
    assert summary.num_claims == 1

    con = duckdb.connect(str(db_path))
    try:
        squares = con.execute(
            "SELECT goal_text FROM squares WHERE game_id = ?", [f"{summary.video_id}-1"]
        ).fetchall()
        assert len(squares) == 25
        assert all(text for (text,) in squares)

        claims = con.execute(
            "SELECT row, col, color FROM claims WHERE game_id = ?", [f"{summary.video_id}-1"]
        ).fetchall()
        assert claims == [(0, 4, "red")]
    finally:
        con.close()
