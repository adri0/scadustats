import datetime
import logging

import pytest

from scadustats.models import GameResult, GameType, MatchType, Square, VideoExtraction, WinType
from scadustats.pipeline.consolidate import consolidate_squares, validate_squares


def _board(*texts: str) -> list[list[str]]:
    flat = list(texts) + [""] * (25 - len(texts))
    return [flat[r * 5 : r * 5 + 5] for r in range(5)]


def _game(game_type: GameType | None, *texts: str, game_index: int = 1) -> GameResult:
    return GameResult(
        game_index=game_index,
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        square_texts=_board(*texts),
        events=[],
        winner_color=None,
        win_type=WinType.NONE,
        game_type=game_type,
    )


def _extraction(*games: GameResult, video_id: str = "2026-03-05-alice-vs-bob") -> VideoExtraction:
    return VideoExtraction(
        video_id=video_id,
        video_url=None,
        match_date=datetime.date(2026, 3, 5),
        season="6",
        match_type=MatchType.ROUND_ROBIN,
        player_red_name="alice",
        player_blue_name="bob",
        extracted_at=datetime.date(2026, 3, 6),
        games=list(games),
    )


def test_consolidate_squares_splits_by_game_type():
    extraction = _extraction(
        _game(GameType.BASE, "Complete 3 Tunnels or Precipices"),
        _game(GameType.DLC, "Acquire 2 Dragon Hearts", game_index=2),
    )

    squares = consolidate_squares([extraction])

    assert [s.text for s in squares[GameType.BASE]] == ["Complete 3 Tunnels or Precipices"]
    assert [s.text for s in squares[GameType.DLC]] == ["Acquire 2 Dragon Hearts"]


def test_consolidate_squares_slugifies_a_short_readable_id():
    extraction = _extraction(_game(GameType.BASE, "Complete 3 Tunnels or Precipices"))

    squares = consolidate_squares([extraction])

    assert squares[GameType.BASE][0].id == "tunnels_3"


def test_consolidate_squares_skips_games_with_no_resolved_game_type():
    extraction = _extraction(_game(None, "Some unresolved goal"))

    squares = consolidate_squares([extraction])

    assert squares[GameType.BASE] == []
    assert squares[GameType.DLC] == []


def test_consolidate_squares_skips_blank_cells():
    extraction = _extraction(_game(GameType.BASE, "Real goal"))

    squares = consolidate_squares([extraction])

    assert [s.text for s in squares[GameType.BASE]] == ["Real goal"]


def test_consolidate_squares_deduplicates_the_same_text_across_matches():
    extractions = [
        _extraction(_game(GameType.BASE, "Kill Wormface"), video_id="2026-01-01-a-vs-b"),
        _extraction(_game(GameType.BASE, "Kill Wormface"), video_id="2026-02-01-c-vs-d"),
    ]

    squares = consolidate_squares(extractions)

    assert [s.text for s in squares[GameType.BASE]] == ["Kill Wormface"]


def test_consolidate_squares_disambiguates_colliding_slugs():
    extraction = _extraction(
        _game(GameType.BASE, "Acquire 14 Unique Incantations", "Acquire 14 Unique incantations")
    )

    squares = consolidate_squares([extraction])

    ids = {s.id for s in squares[GameType.BASE]}
    assert ids == {"incantations_14", "incantations_14_2"}


def test_consolidate_squares_sorts_output_by_text():
    extraction = _extraction(_game(GameType.BASE, "Zebra goal", "Apple goal"))

    squares = consolidate_squares([extraction])

    assert [s.text for s in squares[GameType.BASE]] == ["Apple goal", "Zebra goal"]


def test_consolidate_squares_warns_and_takes_the_majority_on_a_conflicting_game_type(caplog):
    extractions = [
        _extraction(_game(GameType.BASE, "Ambiguous goal"), video_id="2026-01-01-a-vs-b"),
        _extraction(_game(GameType.BASE, "Ambiguous goal"), video_id="2026-02-01-c-vs-d"),
        _extraction(_game(GameType.DLC, "Ambiguous goal"), video_id="2026-03-01-e-vs-f"),
    ]

    with caplog.at_level(logging.WARNING):
        squares = consolidate_squares(extractions)

    assert [s.text for s in squares[GameType.BASE]] == ["Ambiguous goal"]
    assert squares[GameType.DLC] == []
    assert any("more than one game type" in r.message for r in caplog.records)


def test_validate_squares_passes_for_unique_ids_and_texts():
    extraction = _extraction(_game(GameType.BASE, "Goal one", "Goal two"))

    validate_squares(consolidate_squares([extraction])[GameType.BASE])


def test_validate_squares_raises_on_duplicate_id():
    squares = [
        Square(id="dup", text="Goal one", game_type=GameType.BASE),
        Square(id="dup", text="Goal two", game_type=GameType.BASE),
    ]

    with pytest.raises(ValueError, match="duplicate square id"):
        validate_squares(squares)


def test_validate_squares_raises_on_duplicate_text():
    squares = [
        Square(id="id_1", text="Same goal", game_type=GameType.BASE),
        Square(id="id_2", text="Same goal", game_type=GameType.BASE),
    ]

    with pytest.raises(ValueError, match="duplicate square text"):
        validate_squares(squares)
