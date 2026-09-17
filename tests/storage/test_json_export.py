import datetime
import json

import pytest

from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameResult,
    GameType,
    MatchType,
    MatchWinner,
    Square,
    VideoExtraction,
    WinLine,
    WinType,
)
from scadustats.storage.json_export import (
    read_squares,
    read_video,
    squares_path,
    video_path,
    write_squares,
    write_video,
)


def _sample_game(game_index: int = 1) -> GameResult:
    square_texts = [[f"goal {r}-{c}" for c in range(5)] for r in range(5)]
    return GameResult(
        game_index=game_index,
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        square_texts=square_texts,
        events=[GameEvent(row=1, col=1, color=CellColor.RED, video_ts_s=10.0, game_elapsed_s=9)],
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
        season="6",
        match_type=MatchType.ROUND_ROBIN,
        player_red_name="alice",
        player_blue_name="bob",
        extracted_at=datetime.date(2026, 3, 6),
        games=[_sample_game()],
        commentators=["star0chris", "Captain_Domo"],
        duration_s=4321.0,
        published_at=datetime.date(2026, 3, 1),
    )
    defaults.update(overrides)
    return VideoExtraction(**defaults)


def test_write_video_creates_file_with_expected_content(tmp_path):
    extraction = _sample_extraction()
    path = write_video(tmp_path, extraction)

    assert path == video_path(tmp_path, season="6", video_id="2026-03-05-alice-vs-bob")
    data = json.loads(path.read_text())

    assert data["video_id"] == "2026-03-05-alice-vs-bob"
    assert data["match_date"] == "2026-03-05"
    assert data["season"] == "6"
    assert data["match_type"] == "round_robin"
    assert data["player_red_name"] == "alice"
    assert data["player_blue_name"] == "bob"
    assert data["commentators"] == ["star0chris", "Captain_Domo"]
    assert data["metadata"]["video_url"] == "https://youtu.be/abc123"
    assert data["metadata"]["extracted_at"] == "2026-03-06"
    assert data["metadata"]["duration_s"] == 4321.0
    assert data["metadata"]["published_at"] == "2026-03-01"
    assert data["num_games"] == 1
    assert data["red_score"] == 1
    assert data["blue_score"] == 0
    assert data["winner"] == "red"
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
            "row": 1,
            "col": 1,
            "square_text": "goal 0-0",
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


def test_read_video_retallies_the_score_rather_than_trusting_the_file(tmp_path):
    """Same for red_score/blue_score/winner: correcting a game's winner by hand is enough,
    the match-level tally follows from the games rather than needing its own edit."""
    blue_win = _sample_game(2)
    blue_win.winner_color = CellColor.BLUE
    path = write_video(tmp_path, _sample_extraction(games=[_sample_game(1), blue_win]))
    data = json.loads(path.read_text())
    data["red_score"], data["blue_score"], data["winner"] = 7, 0, "red"
    path.write_text(json.dumps(data))

    read_back = read_video(path)
    assert (read_back.red_score, read_back.blue_score) == (1, 1)
    assert read_back.winner is MatchWinner.DRAW


def test_write_video_records_an_undetermined_match_result_as_null(tmp_path):
    """A game with no winner leaves the match result unknowable -- the score is still
    whatever the decided games say, but no winner is named."""
    undecided = _sample_game(2)
    undecided.winner_color = None
    undecided.win_type = WinType.NONE
    undecided.win_line = None

    path = write_video(tmp_path, _sample_extraction(games=[_sample_game(1), undecided]))
    data = json.loads(path.read_text())

    assert (data["red_score"], data["blue_score"]) == (1, 0)
    assert data["winner"] is None
    assert read_video(path).winner is None


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
        GameEvent(row=2, col=2, color=CellColor.BLUE, video_ts_s=30.0, game_elapsed_s=29),
        GameEvent(row=1, col=1, color=CellColor.RED, video_ts_s=10.0, game_elapsed_s=9),
        GameEvent(row=3, col=3, color=CellColor.RED, video_ts_s=20.0, game_elapsed_s=19),
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
    assert game_start["square_text"] is None
    assert game_start["game_timer"] == "00:00:00"


def test_write_video_serializes_game_end_event_with_null_fields(tmp_path):
    game = _sample_game()
    game.events = [
        *game.events,
        GameEvent(
            row=None,
            col=None,
            color=None,
            video_ts_s=30.0,
            game_elapsed_s=29,
            event_type=EventType.GAME_END,
        ),
    ]

    path = write_video(tmp_path, _sample_extraction(games=[game]))
    data = json.loads(path.read_text())

    game_end = next(e for e in data["games"][0]["events"] if e["event_type"] == "game_end")
    assert (game_end["row"], game_end["col"], game_end["color"]) == (None, None, None)
    assert game_end["square_text"] is None
    assert game_end["game_timer"] == "00:00:29"


def test_write_video_yields_null_square_text_for_a_ragged_square_texts_grid(tmp_path):
    """square_texts is OCR output and can be hand-edited into a ragged grid -- an event
    whose square falls outside it gets a null square_text rather than an IndexError."""
    game = _sample_game()
    game.square_texts = []

    path = write_video(tmp_path, _sample_extraction(games=[game]))
    data = json.loads(path.read_text())

    assert data["games"][0]["events"][0]["square_text"] is None


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

    assert data["metadata"]["video_url"] is None


def test_write_video_handles_unknown_duration(tmp_path):
    path = write_video(tmp_path, _sample_extraction(duration_s=None))
    data = json.loads(path.read_text())

    assert data["metadata"]["duration_s"] is None
    assert read_video(path).duration_s is None


def test_commentators_round_trip_in_order(tmp_path):
    """Order is part of the data (it's the nameplates' left-to-right order), so it has to
    survive the round trip rather than being treated as an unordered set."""
    path = write_video(tmp_path, _sample_extraction())

    assert read_video(path).commentators == ["star0chris", "Captain_Domo"]


def test_write_video_handles_a_match_with_no_commentators_read(tmp_path):
    path = write_video(tmp_path, _sample_extraction(commentators=[]))
    data = json.loads(path.read_text())

    assert data["commentators"] == []
    assert read_video(path).commentators == []


def test_read_video_treats_a_file_without_commentators_as_none_recorded(tmp_path):
    """Same story as duration_s below: a file written before commentators were recorded
    is still a valid current-format extraction and has to read back."""
    path = write_video(tmp_path, _sample_extraction())
    data = json.loads(path.read_text())
    del data["commentators"]
    path.write_text(json.dumps(data))

    assert read_video(path).commentators == []


def test_read_video_treats_a_file_without_duration_as_unknown_length(tmp_path):
    """A file written before duration_s existed is still a valid current-format
    extraction -- it must read back, not raise the KeyError `match list` reports as an
    unparseable file."""
    path = write_video(tmp_path, _sample_extraction())
    data = json.loads(path.read_text())
    del data["metadata"]["duration_s"]
    path.write_text(json.dumps(data))

    assert read_video(path).duration_s is None


def test_published_at_round_trips(tmp_path):
    path = write_video(tmp_path, _sample_extraction())

    assert read_video(path).published_at == datetime.date(2026, 3, 1)


def test_write_video_handles_unknown_published_date(tmp_path):
    """published_at is only known when the video was downloaded as part of this run
    (see video.download.DownloadResult) -- a locally-supplied video never has one, so
    this has to serialize/read back as null, like duration_s."""
    path = write_video(tmp_path, _sample_extraction(published_at=None))
    data = json.loads(path.read_text())

    assert data["metadata"]["published_at"] is None
    assert read_video(path).published_at is None


def test_read_video_treats_a_file_without_published_at_as_unknown(tmp_path):
    """A file written before published_at existed is still a valid current-format
    extraction -- same story as duration_s above."""
    path = write_video(tmp_path, _sample_extraction())
    data = json.loads(path.read_text())
    del data["metadata"]["published_at"]
    path.write_text(json.dumps(data))

    assert read_video(path).published_at is None


def test_source_path_round_trips(tmp_path):
    path = write_video(tmp_path, _sample_extraction(source_path="downloads/abc123.mp4"))
    data = json.loads(path.read_text())

    assert data["metadata"]["source_path"] == "downloads/abc123.mp4"
    assert read_video(path).source_path == "downloads/abc123.mp4"


def test_write_video_handles_unknown_source_path(tmp_path):
    path = write_video(tmp_path, _sample_extraction(source_path=None))
    data = json.loads(path.read_text())

    assert data["metadata"]["source_path"] is None
    assert read_video(path).source_path is None


def test_read_video_treats_a_file_without_source_path_as_unknown(tmp_path):
    """A file written before source_path existed is still a valid current-format
    extraction -- same story as duration_s/published_at above."""
    path = write_video(tmp_path, _sample_extraction())
    data = json.loads(path.read_text())
    del data["metadata"]["source_path"]
    path.write_text(json.dumps(data))

    assert read_video(path).source_path is None


def test_read_video_reads_a_pre_grouping_file_with_flat_metadata_fields(tmp_path):
    """A file written before issue #47 grouped these fields under "metadata" has them at
    the top level instead -- it still has to read back cleanly."""
    path = write_video(tmp_path, _sample_extraction())
    data = json.loads(path.read_text())
    metadata = data.pop("metadata")
    data.update(metadata)
    path.write_text(json.dumps(data))

    read_back = read_video(path)
    assert read_back.video_url == "https://youtu.be/abc123"
    assert read_back.duration_s == 4321.0
    assert read_back.published_at == datetime.date(2026, 3, 1)
    assert read_back.extracted_at == datetime.date(2026, 3, 6)


def test_write_video_creates_json_dir_if_missing(tmp_path):
    nested = tmp_path / "a" / "b"

    path = write_video(nested, _sample_extraction())

    assert path.exists()


def test_write_video_groups_matches_under_a_season_subdirectory(tmp_path):
    path = write_video(tmp_path, _sample_extraction(season="Off-Season Cup"))

    assert path == tmp_path / "season-Off-Season Cup" / "2026-03-05-alice-vs-bob.json"
    assert path.parent.parent == tmp_path


def test_write_video_puts_different_seasons_in_different_subdirectories(tmp_path):
    season_6 = write_video(tmp_path, _sample_extraction(season="6"))
    season_7 = write_video(
        tmp_path, _sample_extraction(season="7", video_id="2027-03-05-alice-vs-bob")
    )

    assert season_6.parent != season_7.parent
    assert season_6.exists()
    assert season_7.exists()


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


def test_read_video_round_trips_game_end_event(tmp_path):
    game = _sample_game()
    game.events = [
        *game.events,
        GameEvent(
            row=None,
            col=None,
            color=None,
            video_ts_s=30.0,
            game_elapsed_s=29,
            event_type=EventType.GAME_END,
        ),
    ]
    extraction = _sample_extraction(games=[game])
    path = write_video(tmp_path, extraction)

    read_back = read_video(path)

    assert read_back == extraction


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


def test_squares_path_names_a_file_per_game_type(tmp_path):
    assert squares_path(tmp_path, GameType.BASE) == tmp_path / "base_game.json"
    assert squares_path(tmp_path, GameType.DLC) == tmp_path / "dlc.json"


def test_write_squares_writes_one_file_per_game_type(tmp_path):
    squares = {
        GameType.BASE: [Square(id="wormface", text="Kill Wormface", game_type=GameType.BASE)],
        GameType.DLC: [
            Square(id="hearts_2", text="Acquire 2 Dragon Hearts", game_type=GameType.DLC)
        ],
    }

    paths = write_squares(tmp_path, squares)

    assert paths == {
        GameType.BASE: tmp_path / "base_game.json",
        GameType.DLC: tmp_path / "dlc.json",
    }
    base_data = json.loads(paths[GameType.BASE].read_text())
    assert base_data == [{"id": "wormface", "text": "Kill Wormface", "game_type": "base"}]
    dlc_data = json.loads(paths[GameType.DLC].read_text())
    assert dlc_data == [
        {"id": "hearts_2", "text": "Acquire 2 Dragon Hearts", "game_type": "dlc"}
    ]


def test_write_squares_creates_the_directory(tmp_path):
    squares_dir = tmp_path / "squares"
    squares = {GameType.BASE: [], GameType.DLC: []}

    write_squares(squares_dir, squares)

    assert squares_dir.is_dir()


def test_write_squares_sorts_each_file_by_id(tmp_path):
    squares = {
        GameType.BASE: [
            Square(id="zebra", text="Zebra goal", game_type=GameType.BASE),
            Square(id="apple", text="Apple goal", game_type=GameType.BASE),
        ],
        GameType.DLC: [],
    }

    paths = write_squares(tmp_path, squares)

    data = json.loads(paths[GameType.BASE].read_text())
    assert [entry["id"] for entry in data] == ["apple", "zebra"]


def test_read_squares_is_the_inverse_of_write_squares(tmp_path):
    squares = {
        GameType.BASE: [Square(id="wormface", text="Kill Wormface", game_type=GameType.BASE)],
        GameType.DLC: [
            Square(id="hearts_2", text="Acquire 2 Dragon Hearts", game_type=GameType.DLC)
        ],
    }
    write_squares(tmp_path, squares)

    read_back = read_squares(tmp_path)

    assert read_back == squares


def test_read_squares_treats_a_missing_file_as_an_empty_list(tmp_path):
    read_back = read_squares(tmp_path)

    assert read_back == {GameType.BASE: [], GameType.DLC: []}
