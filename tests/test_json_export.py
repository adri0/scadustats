import json

import pytest

from scadustats.json_export import write_game
from scadustats.models import CellColor, ClaimEvent, GameResult, WinType


def _sample_game() -> GameResult:
    goal_texts = [[f"goal {r}-{c}" for c in range(5)] for r in range(5)]
    return GameResult(
        game_index=1,
        label="GAME 1",
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        player_red_name="alice",
        player_blue_name="bob",
        goal_texts=goal_texts,
        claims=[ClaimEvent(row=0, col=0, color=CellColor.RED, video_ts_s=10.0, game_elapsed_s=9)],
        winner_color=CellColor.RED,
        win_type=WinType.LINE,
    )


def test_write_game_creates_file_with_expected_content(tmp_path):
    path = write_game(tmp_path, "vid1", _sample_game())

    assert path == tmp_path / "vid1-1.json"
    data = json.loads(path.read_text())

    assert data["game_id"] == "vid1-1"
    assert data["video_id"] == "vid1"
    assert data["game_index"] == 1
    assert data["label"] == "GAME 1"
    assert data["player_red_name"] == "alice"
    assert data["player_blue_name"] == "bob"
    assert data["winner_color"] == "red"
    assert data["win_type"] == "line"
    assert len(data["goal_texts"]) == 5
    assert all(len(row) == 5 for row in data["goal_texts"])
    assert data["claims"] == [
        {
            "row": 0,
            "col": 0,
            "color": "red",
            "event_type": "claim",
            "game_elapsed_s": 9,
            "video_ts_s": 10.0,
        }
    ]


def test_write_game_handles_no_winner(tmp_path):
    game = _sample_game()
    game.winner_color = None
    game.win_type = WinType.NONE

    path = write_game(tmp_path, "vid1", game)
    data = json.loads(path.read_text())

    assert data["winner_color"] is None
    assert data["win_type"] == "none"


def test_write_game_creates_json_dir_if_missing(tmp_path):
    nested = tmp_path / "a" / "b"

    path = write_game(nested, "vid1", _sample_game())

    assert path.exists()


def test_if_exists_error_raises_on_existing_file(tmp_path):
    write_game(tmp_path, "vid1", _sample_game())

    with pytest.raises(FileExistsError):
        write_game(tmp_path, "vid1", _sample_game(), if_exists="error")


def test_if_exists_replace_overwrites(tmp_path):
    write_game(tmp_path, "vid1", _sample_game())

    game = _sample_game()
    game.label = "GAME 1 UPDATED"
    path = write_game(tmp_path, "vid1", game, if_exists="replace")

    data = json.loads(path.read_text())
    assert data["label"] == "GAME 1 UPDATED"


def test_if_exists_append_behaves_like_replace(tmp_path):
    write_game(tmp_path, "vid1", _sample_game())

    game = _sample_game()
    game.label = "GAME 1 UPDATED"
    path = write_game(tmp_path, "vid1", game, if_exists="append")

    data = json.loads(path.read_text())
    assert data["label"] == "GAME 1 UPDATED"
