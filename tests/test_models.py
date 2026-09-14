"""Tests for the derived match-level summaries on VideoExtraction -- the fields that are
properties over `games` rather than stored data (see models.VideoExtraction)."""

import datetime

from scadustats.models import (
    CellColor,
    GameResult,
    MatchType,
    MatchWinner,
    VideoExtraction,
    WinType,
)


def _game(game_index: int, winner_color: CellColor | None) -> GameResult:
    return GameResult(
        game_index=game_index,
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        square_texts=[[""] * 5 for _ in range(5)],
        events=[],
        winner_color=winner_color,
        win_type=WinType.MAJORITY if winner_color else WinType.NONE,
    )


def _extraction(*games: GameResult) -> VideoExtraction:
    return VideoExtraction(
        video_id="2026-03-05-alice-vs-bob",
        video_url=None,
        match_date=datetime.date(2026, 3, 5),
        season="6",
        match_type=MatchType.DOUBLE_ELIMINATION,
        player_red_name="alice",
        player_blue_name="bob",
        extracted_at=datetime.date(2026, 3, 6),
        games=list(games),
    )


def test_scores_count_each_players_game_wins():
    extraction = _extraction(
        _game(1, CellColor.RED), _game(2, CellColor.BLUE), _game(3, CellColor.RED)
    )

    assert (extraction.red_score, extraction.blue_score) == (2, 1)
    assert extraction.num_games == 3


def test_winner_is_whoever_won_more_games():
    red_match = _extraction(_game(1, CellColor.RED), _game(2, CellColor.RED))
    blue_match = _extraction(_game(1, CellColor.BLUE), _game(2, CellColor.BLUE))

    assert red_match.winner is MatchWinner.RED
    assert blue_match.winner is MatchWinner.BLUE


def test_an_even_split_is_a_draw():
    """Legal in playoffs (two games, 1-1); validation's double_elimination_draw rule is
    what flags it in the other format -- the property reports what the games say either
    way."""
    extraction = _extraction(_game(1, CellColor.RED), _game(2, CellColor.BLUE))

    assert extraction.winner is MatchWinner.DRAW


def test_winner_is_undetermined_when_a_game_has_no_winner():
    """The decided games still tally to a score, but naming a winner off an incomplete
    set would invent an outcome -- that's what validation reports, not something to guess
    at here."""
    extraction = _extraction(_game(1, CellColor.RED), _game(2, None))

    assert (extraction.red_score, extraction.blue_score) == (1, 0)
    assert extraction.winner is None


def test_a_match_with_no_games_has_no_winner():
    """0-0 is not a draw -- there was no match to draw."""
    extraction = _extraction()

    assert (extraction.red_score, extraction.blue_score, extraction.num_games) == (0, 0, 0)
    assert extraction.winner is None
