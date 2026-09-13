import datetime
import json

import pytest

from scadustats.json_export import read_video, write_video
from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameResult,
    GameType,
    MatchType,
    VideoExtraction,
    WinLine,
    WinType,
)


def _sample_game(game_index: int = 1) -> GameResult:
    square_texts = [[f"goal {r}-{c}" for c in range(5)] for r in range(5)]
    return GameResult(
        game_index=game_index,
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        square_texts=square_texts,
        events=[GameEvent(row=0, col=0, color=CellColor.RED, video_ts_s=10.0, game_elapsed_s=9)],
        winner_color=CellColor.RED,
        win_type=WinType.LINE,
        win_line=WinLine.DIAGONAL_BL_TR,
        game_type=GameType.BASE,
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
        duration_s=4321.0,
    )
    defaults.update(overrides)
    return VideoExtraction(**defaults)


def test_write_video_creates_file_with_expected_content(tmp_path):
    extraction = _sample_extraction()
    path = write_video(tmp_path, extraction)

    assert path == tmp_path / "2026-03-05-alice-vs-bob.json"
    data = json.loads(path.read_text())

    assert data["video_id"] == "2026-03-05-alice-vs-bob"
    assert data["video_url"] == "https://youtu.be/abc123"
    assert data["match_date"] == "2026-03-05"
    assert data["season"] == 6
    assert data["match_type"] == "playoffs"
    assert data["player_red_name"] == "alice"
    assert data["player_blue_name"] == "bob"
    assert data["extracted_at"] == "2026-03-06"
    assert data["duration_s"] == 4321.0
    assert data["num_games"] == 1
    assert len(data["games"]) == 1

    game_data = data["games"][0]
    assert game_data["game_index"] == 1
    assert game_data["game_type"] == "base"
    assert game_data["winner_color"] == "red"
    assert game_data["win_type"] == "line"
    assert game_data["win_line"] == "diagonal_bl_tr"
    assert len(game_data["square_texts"]) == 5
    assert all(len(row) == 5 for row in game_data["square_texts"])
    assert game_data["events"] == [
        {
            "row": 0,
            "col": 0,
            "color": "red",
            "event_type": "mark",
            "game_timer": "00:00:09",
            "video_ts_s": 10.0,
        }
    ]


def test_write_video_includes_every_game_sorted_by_index(tmp_path):
    extraction = _sample_extraction(games=[_sample_game(2), _sample_game(1)])
    path = write_video(tmp_path, extraction)
    data = json.loads(path.read_text())

    assert [game["game_index"] for game in data["games"]] == [1, 2]
    assert data["num_games"] == 2


def test_read_video_recounts_games_rather_than_trusting_num_games(tmp_path):
    """num_games is a derived count, so a hand-edited file that added or dropped a game
    without updating it reads back as what it actually holds -- never as the stale
    number."""
    path = write_video(tmp_path, _sample_extraction(games=[_sample_game(1), _sample_game(2)]))
    data = json.loads(path.read_text())
    data["num_games"] = 99
    path.write_text(json.dumps(data))

    assert read_video(path).num_games == 2


def test_read_video_handles_a_file_without_num_games(tmp_path):
    """A file written before num_games existed is still a valid current-format extraction
    -- like duration_s, it must read back rather than raise the KeyError `match list`
    reports as an unparseable file."""
    path = write_video(tmp_path, _sample_extraction())
    data = json.loads(path.read_text())
    del data["num_games"]
    path.write_text(json.dumps(data))

    assert read_video(path).num_games == 1


def test_write_video_sorts_events_by_game_timer(tmp_path):
    game = _sample_game()
    game.events = [
        GameEvent(row=1, col=1, color=CellColor.BLUE, video_ts_s=30.0, game_elapsed_s=29),
        GameEvent(row=0, col=0, color=CellColor.RED, video_ts_s=10.0, game_elapsed_s=9),
        GameEvent(row=2, col=2, color=CellColor.RED, video_ts_s=20.0, game_elapsed_s=19),
    ]

    path = write_video(tmp_path, _sample_extraction(games=[game]))
    data = json.loads(path.read_text())

    assert [event["game_timer"] for event in data["games"][0]["events"]] == [
        "00:00:09",
        "00:00:19",
        "00:00:29",
    ]


def test_write_video_serializes_game_start_event_with_null_fields(tmp_path):
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

    path = write_video(tmp_path, _sample_extraction(games=[game]))
    data = json.loads(path.read_text())

    game_start = next(e for e in data["games"][0]["events"] if e["event_type"] == "game_start")
    assert (game_start["row"], game_start["col"], game_start["color"]) == (None, None, None)
    assert game_start["game_timer"] == "00:00:00"


def test_write_video_handles_no_winner(tmp_path):
    game = _sample_game()
    game.winner_color = None
    game.win_type = WinType.NONE
    game.win_line = None

    path = write_video(tmp_path, _sample_extraction(games=[game]))
    data = json.loads(path.read_text())

    assert data["games"][0]["winner_color"] is None
    assert data["games"][0]["win_type"] == "none"
    assert data["games"][0]["win_line"] is None

    read_back = read_video(path)
    assert read_back.games[0].win_line is None


def test_read_video_treats_a_file_without_a_win_line_as_unknown(tmp_path):
    """A file written before win_line was recorded still has to read back, like
    duration_s -- see test_read_video_treats_a_file_without_duration_as_unknown_length."""
    path = write_video(tmp_path, _sample_extraction())
    data = json.loads(path.read_text())
    del data["games"][0]["win_line"]
    path.write_text(json.dumps(data))

    assert read_video(path).games[0].win_line is None


def test_write_video_handles_unresolved_game_type(tmp_path):
    game = _sample_game()
    game.game_type = None

    path = write_video(tmp_path, _sample_extraction(games=[game]))
    data = json.loads(path.read_text())

    assert data["games"][0]["game_type"] is None

    read_back = read_video(path)
    assert read_back.games[0].game_type is None


def test_write_video_video_url_defaults_to_null(tmp_path):
    path = write_video(tmp_path, _sample_extraction(video_url=None))
    data = json.loads(path.read_text())

    assert data["video_url"] is None


def test_write_video_handles_unknown_duration(tmp_path):
    path = write_video(tmp_path, _sample_extraction(duration_s=None))
    data = json.loads(path.read_text())

    assert data["duration_s"] is None
    assert read_video(path).duration_s is None


def test_read_video_treats_a_file_without_duration_as_unknown_length(tmp_path):
    """A file written before duration_s existed is still a valid current-format
    extraction -- it must read back, not raise the KeyError `match list` reports as an
    unparseable file."""
    path = write_video(tmp_path, _sample_extraction())
    data = json.loads(path.read_text())
    del data["duration_s"]
    path.write_text(json.dumps(data))

    assert read_video(path).duration_s is None


def test_write_video_creates_json_dir_if_missing(tmp_path):
    nested = tmp_path / "a" / "b"

    path = write_video(nested, _sample_extraction())

    assert path.exists()


def test_if_exists_error_raises_on_existing_file(tmp_path):
    write_video(tmp_path, _sample_extraction())

    with pytest.raises(FileExistsError):
        write_video(tmp_path, _sample_extraction(), if_exists="error")


def test_if_exists_replace_overwrites(tmp_path):
    write_video(tmp_path, _sample_extraction())

    game = _sample_game()
    game.win_type = WinType.MAJORITY
    path = write_video(tmp_path, _sample_extraction(games=[game]), if_exists="replace")

    data = json.loads(path.read_text())
    assert data["games"][0]["win_type"] == "majority"


def test_if_exists_append_behaves_like_replace(tmp_path):
    write_video(tmp_path, _sample_extraction())

    game = _sample_game()
    game.win_type = WinType.MAJORITY
    path = write_video(tmp_path, _sample_extraction(games=[game]), if_exists="append")

    data = json.loads(path.read_text())
    assert data["games"][0]["win_type"] == "majority"


def test_read_video_round_trips_write_video(tmp_path):
    extraction = _sample_extraction(games=[_sample_game(1), _sample_game(2)])
    path = write_video(tmp_path, extraction)

    read_back = read_video(path)

    assert read_back == extraction


def test_read_video_round_trips_game_start_event_and_no_url(tmp_path):
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
    extraction = _sample_extraction(video_url=None, games=[game])
    path = write_video(tmp_path, extraction)

    read_back = read_video(path)

    assert read_back == extraction
