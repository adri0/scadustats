import json

import duckdb
import pytest

from scadustats.extract import estimate_sample_count, extract_video

# A short (80s), downscaled, re-encoded clip trimmed from a real match video, covering
# exactly one known claim transition (see CLAUDE.md for how it was produced) -- small
# enough to commit directly, unlike the multi-GB originals in downloads/.
_CLIP_PATH = "tests/fixtures/clip_claim.mp4"


@pytest.mark.integration
def test_extract_video_end_to_end(tmp_path):
    db_path = tmp_path / "extract.duckdb"
    json_dir = tmp_path / "json"

    progress_calls = 0

    def on_progress() -> None:
        nonlocal progress_calls
        progress_calls += 1

    summary = extract_video(
        _CLIP_PATH, db_path=db_path, json_dir=json_dir, on_progress=on_progress
    )

    assert summary.num_games == 1
    assert summary.num_claims == 1
    # on_progress fires once per sampled frame, so this should track estimate_sample_count
    # (an estimate, not exact -- see its docstring) within a sample or two.
    assert progress_calls == pytest.approx(estimate_sample_count(_CLIP_PATH), abs=2)

    json_path = json_dir / f"{summary.video_id}-1.json"
    assert json_path.exists()
    game_data = json.loads(json_path.read_text())
    assert [(c["row"], c["col"], c["color"]) for c in game_data["claims"]] == [(0, 4, "red")]

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
