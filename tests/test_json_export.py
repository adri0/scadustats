import json

import pytest

from scadustats.json_export import write_game
from scadustats.models import CellColor, EventType, GameEvent, GameResult, WinType


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
        events=[GameEvent(row=0, col=0, color=CellColor.RED, video_ts_s=10.0, game_elapsed_s=9)],
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
    assert len(data["square_texts"]) == 5
    assert all(len(row) == 5 for row in data["square_texts"])
    assert data["events"] == [
        {
            "row": 0,
            "col": 0,
            "color": "red",
            "event_type": "mark",
            "game_timer": "00:00:09",
            "video_ts_s": 10.0,
        }
    ]


def test_write_game_sorts_events_by_game_timer(tmp_path):
    game = _sample_game()
    game.events = [
        GameEvent(row=1, col=1, color=CellColor.BLUE, video_ts_s=30.0, game_elapsed_s=29),
        GameEvent(row=0, col=0, color=CellColor.RED, video_ts_s=10.0, game_elapsed_s=9),
        GameEvent(row=2, col=2, color=CellColor.RED, video_ts_s=20.0, game_elapsed_s=19),
    ]

    path = write_game(tmp_path, "vid1", game)
    data = json.loads(path.read_text())

    assert [event["game_timer"] for event in data["events"]] == [
        "00:00:09",
        "00:00:19",
        "00:00:29",
    ]


def test_write_game_serializes_game_start_event_with_null_fields(tmp_path):
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

    path = write_game(tmp_path, "vid1", game)
    data = json.loads(path.read_text())

    game_start = next(e for e in data["events"] if e["event_type"] == "game_start")
    assert (game_start["row"], game_start["col"], game_start["color"]) == (None, None, None)
    assert game_start["game_timer"] == "00:00:00"


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
