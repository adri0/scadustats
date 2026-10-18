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
        match_id="2026-03-05-alice-vs-bob",
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
    """Legal in round robin (two games, 1-1); validation's double_elimination_draw rule is
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


def test_team_players_for_each_pool():
    from scadustats.models import GameType, Team

    team = Team(name="Positive Vibes", base_player="star0chris", dlc_player="Adef")
    assert team.players_for(GameType.BASE) == ["star0chris"]
    assert team.players_for(GameType.DLC) == ["Adef"]
    assert team.players_for(GameType.TWO_V_TWO) == ["star0chris", "Adef"]
    assert team.players_for(None) == []


def test_a_team_without_roles_shares_every_claim():
    from scadustats.models import GameType, Team

    team = Team(name="Cattery", members=["catalystz", "Puppery"])
    assert team.players == ["catalystz", "Puppery"]
    assert team.players_for(GameType.BASE) == ["catalystz", "Puppery"]
    assert team.players_for(None) == []


def test_a_team_has_roles_or_two_members():
    import pytest
    from pydantic import ValidationError

    from scadustats.models import Team

    with pytest.raises(ValidationError):
        Team(name="Cattery", members=["catalystz"])
    with pytest.raises(ValidationError):
        Team(name="Cattery", base_player="catalystz", members=["catalystz", "Puppery"])
    with pytest.raises(ValidationError):
        Team(name="Cattery", base_player="catalystz")


def test_square_type_tag_helpers():
    from scadustats.models import (
        GameType,
        square_type_tag,
        strip_square_type_tag,
        tag_square_text,
    )

    assert square_type_tag("Kill Malenia (B)") is GameType.BASE
    assert square_type_tag("Kill Bols in front of Castle Ensis (D)") is GameType.DLC
    assert square_type_tag("Kill Fia's Champs and Leda (C)") is GameType.TWO_V_TWO
    # Only the exact form counts: an OCR misread or a longer parenthetical isn't a tag.
    assert square_type_tag("Kill Malenia (8)") is None
    assert square_type_tag("30 Faith (no rebirth or buffs)") is None
    assert square_type_tag("Kill Malenia") is None
    assert strip_square_type_tag("Kill Malenia (B)") == "Kill Malenia"
    assert strip_square_type_tag("Kill Malenia") == "Kill Malenia"
    assert tag_square_text("Kill Malenia", GameType.BASE) == "Kill Malenia (B)"
    assert tag_square_text("Kill Malenia (D)", GameType.TWO_V_TWO) == "Kill Malenia (C)"


def _game_with(game_type, texts):
    from scadustats.models import GameResult, WinType

    flat = list(texts) + [""] * (25 - len(texts))
    return GameResult(
        game_index=1,
        start_video_ts_s=0.0,
        end_video_ts_s=1.0,
        square_texts=[flat[r * 5 : r * 5 + 5] for r in range(5)],
        events=[],
        winner_color=None,
        win_type=WinType.NONE,
        game_type=game_type,
    )


def test_cell_goal_reads_a_mixed_board_cells_pool_from_its_tag():
    from scadustats.models import GameType

    game = _game_with(GameType.TWO_V_TWO, ["Kill Malenia (B)", "Kill a Red Wolf (C)", "Kill Bols"])

    assert game.cell_goal(0, 0) == (GameType.BASE, "Kill Malenia")
    assert game.cell_goal(0, 1) == (GameType.TWO_V_TWO, "Kill a Red Wolf")
    assert game.cell_goal(0, 2) == (None, "Kill Bols")  # untagged: no pool
    assert game.cell_goal(9, 9) == (None, "")


def test_cell_goal_is_the_games_own_type_and_text_on_any_other_board():
    from scadustats.models import GameType

    game = _game_with(GameType.BASE, ["Kill Malenia (B)"])

    # Not a mixed board, so nothing is read as a tag.
    assert game.cell_goal(0, 0) == (GameType.BASE, "Kill Malenia (B)")
