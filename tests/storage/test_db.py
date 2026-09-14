import datetime

import duckdb
import pytest

from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameResult,
    GameType,
    MatchType,
    MatchWinner,
    VideoExtraction,
    VideoInfo,
    WinLine,
    WinType,
)
from scadustats.storage.db import load_json_dir, write_extraction
from scadustats.storage.json_export import write_video


def _sample_game(game_index: int = 1) -> GameResult:
    square_texts = [[f"goal {r}-{c}" for c in range(5)] for r in range(5)]
    return GameResult(
        game_index=game_index,
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        square_texts=square_texts,
        events=[GameEvent(row=0, col=0, color=CellColor.RED, video_ts_s=10.0, game_elapsed_s=9)],
        winner_color=None,
        win_type=WinType.NONE,
        game_type=GameType.BASE,
    )


def _sample_extraction(**overrides) -> VideoExtraction:
    defaults = dict(
        video_id="2026-03-05-alice-vs-bob",
        video_url="https://youtu.be/abc123",
        match_date=datetime.date(2026, 3, 5),
        season="6",
        match_type=MatchType.PLAYOFFS,
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
        assert con.execute("SELECT COUNT(*) FROM commentators").fetchone()[0] == 2
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
        # Re-writing a video has to clear its commentator rows too, or they'd accumulate
        # (and collide on the (video_id, position) primary key).
        assert con.execute("SELECT COUNT(*) FROM commentators").fetchone()[0] == 2
    finally:
        con.close()


def test_commentator_rows_keep_the_extractions_order(tmp_path):
    db_path = tmp_path / "test.duckdb"

    write_extraction(db_path, _sample_extraction())

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT position, name FROM commentators WHERE video_id = ? ORDER BY position",
            ["2026-03-05-alice-vs-bob"],
        ).fetchall()
        assert rows == [(0, "star0chris"), (1, "Captain_Domo")]
    finally:
        con.close()


def test_a_match_with_no_commentators_read_writes_no_rows(tmp_path):
    db_path = tmp_path / "test.duckdb"

    write_extraction(db_path, _sample_extraction(commentators=[]))

    con = duckdb.connect(str(db_path))
    try:
        assert con.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM commentators").fetchone()[0] == 0
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
            "6",
            "playoffs",
            "alice",
            "bob",
            datetime.date(2026, 3, 6),
        )
    finally:
        con.close()


def test_game_type_round_trips(tmp_path):
    db_path = tmp_path / "test.duckdb"

    write_extraction(db_path, _sample_extraction())

    con = duckdb.connect(str(db_path))
    try:
        game_type = con.execute(
            "SELECT game_type FROM games WHERE game_id = '2026-03-05-alice-vs-bob-1'"
        ).fetchone()[0]
        assert game_type == "base"
    finally:
        con.close()


def test_game_type_defaults_to_null(tmp_path):
    """A hand-edited JSON file could clear game_type back to unknown -- this should
    load cleanly, not fail a NOT NULL constraint."""
    db_path = tmp_path / "test.duckdb"
    game = _sample_game()
    game.game_type = None

    write_extraction(db_path, _sample_extraction(games=[game]))

    con = duckdb.connect(str(db_path))
    try:
        game_type = con.execute(
            "SELECT game_type FROM games WHERE game_id = '2026-03-05-alice-vs-bob-1'"
        ).fetchone()[0]
        assert game_type is None
    finally:
        con.close()


def test_win_line_round_trips_and_is_null_without_a_line_win(tmp_path):
    db_path = tmp_path / "test.duckdb"
    line_win = _sample_game(1)
    line_win.winner_color = CellColor.RED
    line_win.win_type = WinType.LINE
    line_win.win_line = WinLine.DIAGONAL_TL_BR

    write_extraction(db_path, _sample_extraction(games=[line_win, _sample_game(2)]))

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute("SELECT game_index, win_line FROM games ORDER BY game_index").fetchall()
        # Game 2 is the fixture's default: no winner, so no line to name.
        assert rows == [(1, "diagonal_tl_br"), (2, None)]
    finally:
        con.close()


@pytest.mark.parametrize("win_line", list(WinLine))
def test_schema_accepts_every_win_line_value(win_line, tmp_path):
    """The games.win_line CHECK constraint spells its 12 values out in SQL, so it can
    drift from models.WinLine -- this is what catches that."""
    db_path = tmp_path / "test.duckdb"
    game = _sample_game()
    game.winner_color = CellColor.RED
    game.win_type = WinType.LINE
    game.win_line = win_line

    write_extraction(db_path, _sample_extraction(games=[game]))

    con = duckdb.connect(str(db_path))
    try:
        assert con.execute("SELECT win_line FROM games").fetchone()[0] == win_line.value
    finally:
        con.close()


def test_init_schema_adds_win_line_to_a_database_predating_it(tmp_path):
    """Like duration_s on videos: CREATE TABLE IF NOT EXISTS leaves an existing games
    table alone, so the column has to be added explicitly."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            "CREATE TABLE games (game_id VARCHAR PRIMARY KEY, video_id VARCHAR NOT NULL, "
            "game_index INTEGER NOT NULL, start_video_ts_s DOUBLE NOT NULL, "
            "end_video_ts_s DOUBLE, game_type VARCHAR, winner_color VARCHAR, "
            "win_type VARCHAR)"
        )
    finally:
        con.close()

    game = _sample_game()
    game.winner_color = CellColor.RED
    game.win_type = WinType.LINE
    game.win_line = WinLine.COL_3
    write_extraction(db_path, _sample_extraction(games=[game]))

    con = duckdb.connect(str(db_path))
    try:
        assert con.execute("SELECT win_line FROM games").fetchone()[0] == "col_3"
    finally:
        con.close()


def test_if_exists_error_raises_on_duplicate(tmp_path):
    db_path = tmp_path / "test.duckdb"

    write_extraction(db_path, _sample_extraction())
    with pytest.raises(ValueError):
        write_extraction(db_path, _sample_extraction(), if_exists="error")


def test_video_row_carries_duration(tmp_path):
    """Written from the extraction itself, so it survives load_json_dir's no-video-file
    path -- unlike resolution/fps, which only a source video can supply."""
    db_path = tmp_path / "test.duckdb"

    write_extraction(db_path, _sample_extraction())

    con = duckdb.connect(str(db_path))
    try:
        duration_s = con.execute(
            "SELECT duration_s FROM videos WHERE video_id = ?", ["2026-03-05-alice-vs-bob"]
        ).fetchone()[0]
        assert duration_s == 4321.0
    finally:
        con.close()


def test_video_row_carries_the_game_count(tmp_path):
    """Denormalized from the games table so "how many games" is answerable off videos
    alone -- it must agree with what actually landed there."""
    db_path = tmp_path / "test.duckdb"

    write_extraction(db_path, _sample_extraction(games=[_sample_game(1), _sample_game(2)]))

    con = duckdb.connect(str(db_path))
    try:
        num_games, counted = con.execute(
            "SELECT v.num_games, COUNT(g.game_id) FROM videos v JOIN games g "
            "USING (video_id) WHERE v.video_id = ? GROUP BY v.num_games",
            ["2026-03-05-alice-vs-bob"],
        ).fetchone()
        assert num_games == 2
        assert counted == 2
    finally:
        con.close()


def test_video_row_carries_the_match_score_and_winner(tmp_path):
    db_path = tmp_path / "test.duckdb"
    red_win = _sample_game(1)
    red_win.winner_color = CellColor.RED
    red_win.win_type = WinType.MAJORITY
    blue_win = _sample_game(2)
    blue_win.winner_color = CellColor.BLUE
    blue_win.win_type = WinType.MAJORITY
    decider = _sample_game(3)
    decider.winner_color = CellColor.RED
    decider.win_type = WinType.MAJORITY

    write_extraction(db_path, _sample_extraction(games=[red_win, blue_win, decider]))

    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT red_score, blue_score, winner FROM videos WHERE video_id = ?",
            ["2026-03-05-alice-vs-bob"],
        ).fetchone()
        assert row == (2, 1, "red")
    finally:
        con.close()


def test_video_winner_is_null_when_a_game_has_no_winner(tmp_path):
    """The sample game has no winner recorded, so the match outcome can't be named --
    the column takes NULL rather than a guessed result."""
    db_path = tmp_path / "test.duckdb"

    write_extraction(db_path, _sample_extraction())

    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT red_score, blue_score, winner FROM videos WHERE video_id = ?",
            ["2026-03-05-alice-vs-bob"],
        ).fetchone()
        assert row == (0, 0, None)
    finally:
        con.close()


@pytest.mark.parametrize("match_winner", list(MatchWinner))
def test_schema_accepts_every_match_winner_value(match_winner, tmp_path):
    """The videos.winner CHECK constraint spells its values out in SQL, so it can drift
    from models.MatchWinner -- this is what catches that."""
    db_path = tmp_path / "test.duckdb"
    games = [_sample_game(1)]
    games[0].winner_color = CellColor.RED
    if match_winner is MatchWinner.BLUE:
        games[0].winner_color = CellColor.BLUE
    elif match_winner is MatchWinner.DRAW:
        blue_win = _sample_game(2)
        blue_win.winner_color = CellColor.BLUE
        games.append(blue_win)

    write_extraction(db_path, _sample_extraction(games=games))

    con = duckdb.connect(str(db_path))
    try:
        assert con.execute("SELECT winner FROM videos").fetchone()[0] == match_winner.value
    finally:
        con.close()


def test_init_schema_adds_num_games_to_a_database_predating_it(tmp_path):
    """Same story as duration_s/win_line: CREATE TABLE IF NOT EXISTS leaves an existing
    videos table alone, so the column has to be added explicitly."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            "CREATE TABLE videos (video_id VARCHAR PRIMARY KEY, source_path VARCHAR, "
            "resolution_width INTEGER, resolution_height INTEGER, fps DOUBLE, "
            "duration_s DOUBLE, video_url VARCHAR, match_date DATE NOT NULL, "
            "season INTEGER NOT NULL, match_type VARCHAR NOT NULL, "
            "player_red_name VARCHAR, player_blue_name VARCHAR, extracted_at DATE NOT NULL)"
        )
    finally:
        con.close()

    write_extraction(db_path, _sample_extraction())

    con = duckdb.connect(str(db_path))
    try:
        # The match-result columns are added by the same ALTER block, so they come along.
        row = con.execute(
            "SELECT num_games, red_score, blue_score, winner FROM videos WHERE video_id = ?",
            ["2026-03-05-alice-vs-bob"],
        ).fetchone()
        assert row == (1, 0, 0, None)
    finally:
        con.close()


def test_published_at_round_trips_through_the_db(tmp_path):
    db_path = tmp_path / "test.duckdb"

    write_extraction(db_path, _sample_extraction())

    con = duckdb.connect(str(db_path))
    try:
        published_at = con.execute(
            "SELECT published_at FROM videos WHERE video_id = ?", ["2026-03-05-alice-vs-bob"]
        ).fetchone()[0]
        assert published_at == datetime.date(2026, 3, 1)
    finally:
        con.close()


def test_init_schema_adds_published_at_to_a_database_predating_it(tmp_path):
    """Same story as duration_s/win_line/num_games above: a database file created before
    this column existed still has to accept an insert, not fail on an unknown column."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            "CREATE TABLE videos (video_id VARCHAR PRIMARY KEY, source_path VARCHAR, "
            "resolution_width INTEGER, resolution_height INTEGER, fps DOUBLE, "
            "duration_s DOUBLE, video_url VARCHAR, match_date DATE NOT NULL, "
            "season VARCHAR NOT NULL, match_type VARCHAR NOT NULL, "
            "player_red_name VARCHAR, player_blue_name VARCHAR, extracted_at DATE NOT NULL, "
            "num_games INTEGER, red_score INTEGER, blue_score INTEGER, winner VARCHAR)"
        )
    finally:
        con.close()

    write_extraction(db_path, _sample_extraction())

    con = duckdb.connect(str(db_path))
    try:
        published_at = con.execute(
            "SELECT published_at FROM videos WHERE video_id = ?", ["2026-03-05-alice-vs-bob"]
        ).fetchone()[0]
        assert published_at == datetime.date(2026, 3, 1)
    finally:
        con.close()


def test_video_duration_falls_back_to_the_probed_video(tmp_path):
    """JSON predating duration_s has none to write, but a caller extracting from a real
    file still has the probed VideoInfo in hand."""
    db_path = tmp_path / "test.duckdb"
    video_info = VideoInfo(width=1280, height=720, fps=60.0, duration_s=100.0)

    write_extraction(db_path, _sample_extraction(duration_s=None), "vid.mp4", video_info)

    con = duckdb.connect(str(db_path))
    try:
        duration_s = con.execute(
            "SELECT duration_s FROM videos WHERE video_id = ?", ["2026-03-05-alice-vs-bob"]
        ).fetchone()[0]
        assert duration_s == 100.0
    finally:
        con.close()


def test_init_schema_adds_duration_to_a_database_predating_it(tmp_path):
    """CREATE TABLE IF NOT EXISTS leaves an existing videos table alone, so the column
    has to be added explicitly or every insert into an older database file fails."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            "CREATE TABLE videos (video_id VARCHAR PRIMARY KEY, source_path VARCHAR, "
            "resolution_width INTEGER, resolution_height INTEGER, fps DOUBLE, "
            "video_url VARCHAR, match_date DATE NOT NULL, season INTEGER NOT NULL, "
            "match_type VARCHAR NOT NULL, player_red_name VARCHAR, "
            "player_blue_name VARCHAR, extracted_at DATE NOT NULL)"
        )
    finally:
        con.close()

    write_extraction(db_path, _sample_extraction())

    con = duckdb.connect(str(db_path))
    try:
        duration_s = con.execute(
            "SELECT duration_s FROM videos WHERE video_id = ?", ["2026-03-05-alice-vs-bob"]
        ).fetchone()[0]
        assert duration_s == 4321.0
    finally:
        con.close()


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
        assert season == "6"
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
