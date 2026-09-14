import datetime

import pytest

from scadustats.cli.display import (
    format_clock,
    format_duration,
    format_outcome,
    format_result,
    render_board,
    render_events,
    render_match,
    render_match_table,
    render_table,
    undecided_games,
)
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
from scadustats.rules.winner import empty_board


def _game(game_index: int = 1, **overrides) -> GameResult:
    defaults = dict(
        game_index=game_index,
        start_video_ts_s=300.0,
        end_video_ts_s=900.0,
        square_texts=[[f"goal {r}-{c}" for c in range(5)] for r in range(5)],
        events=[
            GameEvent(
                row=1,
                col=col + 1,
                color=CellColor.RED,
                video_ts_s=310.0 + col,
                game_elapsed_s=col,
            )
            for col in range(5)
        ],
        winner_color=CellColor.RED,
        win_type=WinType.LINE,
        win_line=WinLine.ROW_1,
        game_type=GameType.BASE,
    )
    defaults.update(overrides)
    return GameResult(**defaults)


def _extraction(**overrides) -> VideoExtraction:
    defaults = dict(
        video_id="2026-03-05-alice-vs-bob",
        video_url="https://youtu.be/abc123",
        match_date=datetime.date(2026, 3, 5),
        season="6",
        match_type=MatchType.DOUBLE_ELIMINATION,
        player_red_name="alice",
        player_blue_name="bob",
        extracted_at=datetime.date(2026, 3, 6),
        games=[_game()],
        duration_s=4321.0,
    )
    defaults.update(overrides)
    return VideoExtraction(**defaults)


def test_format_duration_is_hours_minutes_seconds():
    assert format_duration(4321.0) == "01:12:01"


def test_format_clock_drops_the_hour_a_game_clock_never_reaches():
    assert format_clock(185) == "03:05"
    # The pre-game countdown is recorded as it reads; a negative would be a data problem,
    # but it should still print as a number rather than as a mangled one.
    assert format_clock(-5) == "-00:05"


def test_undecided_games_counts_games_with_no_winner():
    games = [
        _game(1),
        _game(2, winner_color=CellColor.BLUE),
        _game(3, winner_color=None, win_type=WinType.NONE, win_line=None),
    ]

    assert undecided_games(_extraction(games=games)) == 1


def test_format_outcome_names_the_match_winner():
    games = [_game(1), _game(2)]

    assert format_outcome(_extraction(games=games)) == "alice wins 2-0"


def test_format_outcome_says_leads_while_a_game_has_no_winner():
    """A score that isn't final shouldn't read as if it were -- an undecided game is
    usually an extraction problem, and calling it a win would paper over that."""
    games = [_game(1), _game(2, winner_color=None, win_type=WinType.NONE, win_line=None)]

    assert format_outcome(_extraction(games=games)) == "alice leads 1-0, 1 undecided"


def test_format_outcome_reports_a_draw():
    games = [_game(1), _game(2, winner_color=CellColor.BLUE)]

    assert format_outcome(_extraction(games=games)) == "draw 1-1"


def test_format_outcome_falls_back_to_colors_when_a_player_name_is_missing():
    assert format_outcome(_extraction(player_red_name=None)) == "red wins 1-0"


def test_format_result_names_the_line_a_game_was_won_on():
    assert format_result(_game(), _extraction()) == "alice (red) by line on row 1"


def test_format_result_describes_a_tie():
    game = _game(winner_color=None, win_type=WinType.TIE, win_line=None)

    assert "tie" in format_result(game, _extraction())


def test_render_table_pads_every_column_to_its_widest_value():
    lines = render_table(["A", "LONGER"], [["xx", "y"], ["z", "wwwwwww"]])

    assert lines == ["A   LONGER", "xx  y", "z   wwwwwww"]


def test_render_board_brackets_the_winning_line():
    board = empty_board()
    for col in range(5):
        board[2][col] = CellColor.RED
    board[0][0] = CellColor.BLUE

    lines = render_board(board, WinLine.ROW_3)

    assert lines[0] == " B  .  .  .  ."
    assert lines[2] == "[R][R][R][R][R]"


def test_render_board_brackets_nothing_without_a_winning_line():
    assert "[" not in "".join(render_board(empty_board()))


def test_render_match_table_has_a_header_and_a_row_per_match():
    lines = render_match_table([_extraction(), _extraction(video_id="2026-04-01-c-vs-d")])

    assert lines[0].split() == ["MATCH", "SEASON", "TYPE", "GAMES", "RESULT"]
    assert len(lines) == 3
    assert lines[1].startswith("2026-03-05-alice-vs-bob")
    assert lines[1].endswith("alice wins 1-0")


def test_render_events_orders_by_video_timestamp_not_the_game_clock():
    """The clock counts *down* through the pre-game countdown, so a game_start carrying a
    3-minute reading has to still come first -- sorting by the clock would bury it."""
    game = _game(
        events=[
            GameEvent(row=1, col=1, color=CellColor.RED, video_ts_s=310.0, game_elapsed_s=10),
            GameEvent(
                row=None,
                col=None,
                color=None,
                video_ts_s=300.0,
                game_elapsed_s=180,
                event_type=EventType.GAME_START,
            ),
        ]
    )

    rows = render_events(game, _extraction())

    assert rows[1].startswith("00:05:00  03:00  game start")
    assert "mark" in rows[2] and "alice" in rows[2] and "goal 0-0" in rows[2]


def test_render_events_truncates_a_long_goal_text():
    game = _game(square_texts=[[("x" * 80) for _ in range(5)] for _ in range(5)])

    rows = render_events(game, _extraction())

    assert rows[1].endswith("...")
    assert len(rows[1]) < 120


def test_render_events_tolerates_a_square_text_grid_that_is_missing_cells():
    """square_texts is OCR output and can be hand-edited -- a ragged grid shouldn't take
    down a read-only listing."""
    game = _game(square_texts=[])

    rows = render_events(game, _extraction())

    assert len(rows) == 6  # header plus the game's five marks


def test_render_match_shows_the_board_and_the_square_split():
    lines = render_match(_extraction())
    text = "\n".join(lines)

    assert "alice (red) vs bob (blue) -- alice wins 1-0" in text
    assert "Game 1  base game  00:05:00-00:15:00 (10:00)" in text
    assert "squares  alice 5, bob 0, unclaimed 20" in text
    assert "[R][R][R][R][R]" in text


def test_render_match_omits_length_and_url_when_they_are_unknown():
    text = "\n".join(render_match(_extraction(duration_s=None, video_url=None)))

    assert "length" not in text
    assert "video " not in text
    assert "extracted 2026-03-06" in text


def test_render_match_names_the_commentators_in_order():
    text = "\n".join(render_match(_extraction(commentators=["star0chris", "Captain_Domo"])))

    assert "commentary star0chris, Captain_Domo" in text


def test_render_match_omits_commentary_when_no_nameplate_read():
    text = "\n".join(render_match(_extraction(commentators=[])))

    assert "commentary" not in text


def test_render_match_leaves_an_open_ended_game_span_open():
    text = "\n".join(render_match(_extraction(games=[_game(end_video_ts_s=None)])))

    assert "00:05:00-?" in text


@pytest.mark.parametrize("events", [True, False])
def test_render_match_includes_the_event_listing_only_when_asked(events):
    text = "\n".join(render_match(_extraction(), events=events))

    assert ("goal 0-0" in text) is events
