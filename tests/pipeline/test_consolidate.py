import datetime
import logging

import pytest

from scadustats.models import GameResult, GameType, MatchType, Square, VideoExtraction, WinType
from scadustats.pipeline.consolidate import (
    consolidate_match_squares,
    consolidate_squares,
    validate_squares,
)


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


def _known(*texts: str, game_type: GameType = GameType.BASE) -> dict[GameType, list[Square]]:
    others = GameType.DLC if game_type is GameType.BASE else GameType.BASE
    return {
        game_type: [Square(id=f"id_{i}", text=t, game_type=game_type) for i, t in enumerate(texts)],
        others: [],
    }


def test_consolidate_match_squares_corrects_a_dropped_letter():
    extraction = _extraction(
        _game(GameType.BASE, "Kill Tree Sentinel (imgrave) with a +0 Weapon Only")
    )
    known = _known("Kill Tree Sentinel (Limgrave) with a +0 Weapon Only")

    changes = consolidate_match_squares(extraction, known)

    assert len(changes) == 1
    assert not changes[0].is_new
    assert changes[0].resolved_text == "Kill Tree Sentinel (Limgrave) with a +0 Weapon Only"
    assert extraction.games[0].square_texts[0][0] == (
        "Kill Tree Sentinel (Limgrave) with a +0 Weapon Only"
    )


def test_consolidate_match_squares_corrects_a_stray_leading_token():
    extraction = _extraction(_game(GameType.BASE, "x Kill Borealis"))
    known = _known("Kill Borealis")

    changes = consolidate_match_squares(extraction, known)

    assert not changes[0].is_new
    assert changes[0].resolved_text == "Kill Borealis"


def test_consolidate_match_squares_corrects_a_missing_apostrophe():
    extraction = _extraction(_game(GameType.BASE, "Restore Rykard s Great Rune"))
    known = _known("Restore Rykard's Great Rune")

    changes = consolidate_match_squares(extraction, known)

    assert changes[0].resolved_text == "Restore Rykard's Great Rune"


def test_consolidate_match_squares_corrects_a_letter_swap():
    extraction = _extraction(
        _game(GameType.BASE, "Kilt 3 Friendly NPCs (No Hermit Merchants)")
    )
    known = _known("Kill 3 Friendly NPCs (No Hermit Merchants)")

    changes = consolidate_match_squares(extraction, known)

    assert changes[0].resolved_text == "Kill 3 Friendly NPCs (No Hermit Merchants)"


def test_consolidate_match_squares_corrects_a_missing_space():
    extraction = _extraction(_game(GameType.BASE, "Killa Death Knight"))
    known = _known("Kill a Death Knight")

    changes = consolidate_match_squares(extraction, known)

    assert changes[0].resolved_text == "Kill a Death Knight"


def test_consolidate_match_squares_leaves_an_exact_match_unreported():
    extraction = _extraction(_game(GameType.BASE, "Kill Wormface"))
    known = _known("Kill Wormface")

    changes = consolidate_match_squares(extraction, known)

    assert changes == []


def test_consolidate_match_squares_adds_rather_than_changes_a_different_goal_count():
    """"Kill 3 Friendly NPCs" and "Kill 5 Friendly NPCs" can both be real, distinct
    squares -- a wrong digit shouldn't be "fixed" just because the rest of the wording is
    a close textual match, so this is filed as its own new square instead."""
    extraction = _extraction(_game(GameType.BASE, "Kill 3 Friendly NPCs (No Hermit Merchants)"))
    known = _known("Kill 5 Friendly NPCs (No Hermit Merchants)")

    changes = consolidate_match_squares(extraction, known)

    assert changes[0].is_new
    assert changes[0].resolved_text == "Kill 3 Friendly NPCs (No Hermit Merchants)"
    assert extraction.games[0].square_texts[0][0] == "Kill 3 Friendly NPCs (No Hermit Merchants)"
    assert known[GameType.BASE][-1].text == "Kill 3 Friendly NPCs (No Hermit Merchants)"


def test_consolidate_match_squares_adds_an_unmatched_square_to_the_reference():
    extraction = _extraction(_game(GameType.BASE, "Some completely unrelated goal text"))
    known = _known("Kill Wormface")

    changes = consolidate_match_squares(extraction, known)

    assert len(changes) == 1
    assert changes[0].original_text == "Some completely unrelated goal text"
    assert changes[0].is_new
    assert changes[0].game_type is GameType.BASE
    assert extraction.games[0].square_texts[0][0] == "Some completely unrelated goal text"
    added = known[GameType.BASE][-1]
    assert added.text == "Some completely unrelated goal text"
    assert added.game_type is GameType.BASE
    assert added.id


def test_consolidate_match_squares_assigns_a_slug_id_to_a_new_square():
    extraction = _extraction(_game(GameType.BASE, "Complete 3 Tunnels or Precipices"))
    known = _known()

    consolidate_match_squares(extraction, known)

    assert known[GameType.BASE][-1].id == "tunnels_3"


def test_consolidate_match_squares_disambiguates_a_new_squares_id_against_the_reference():
    """The new square's own text ("Kill 3 Wolves of the Forest", no relation to the
    unrelated placeholder already on file) is nothing like the existing entry's, so this
    is squarely an ADD, not a fuzzy-matched FIX -- but it still slugifies to the same id
    ("wolves_3") the existing entry was hand-assigned, which _unique_id must notice."""
    extraction = _extraction(_game(GameType.BASE, "Kill 3 Wolves of the Forest"))
    known = _known("Totally unrelated placeholder text")
    known[GameType.BASE][0].id = "wolves_3"

    changes = consolidate_match_squares(extraction, known)

    assert changes[0].is_new
    assert known[GameType.BASE][-1].id == "wolves_3_2"


def test_consolidate_match_squares_skips_games_with_no_resolved_game_type():
    extraction = _extraction(_game(None, "x Kill Borealis"))
    known = _known("Kill Borealis")

    changes = consolidate_match_squares(extraction, known)

    assert changes == []
    assert extraction.games[0].square_texts[0][0] == "x Kill Borealis"
    assert known[GameType.BASE] == [
        Square(id="id_0", text="Kill Borealis", game_type=GameType.BASE)
    ]


def test_consolidate_match_squares_skips_blank_cells():
    extraction = _extraction(_game(GameType.BASE, "Kill Wormface"))
    known = _known("Kill Wormface", "Kill Borealis")

    changes = consolidate_match_squares(extraction, known)

    assert changes == []


def test_consolidate_match_squares_only_matches_within_the_games_own_game_type_pool():
    """A BASE game's squares are never checked against the DLC pool (or vice versa) --
    a board's squares are drawn from one pool, not a mix (see squares.infer_game_type).
    An unmatched square is added to its own game type's pool, not the other one's."""
    extraction = _extraction(_game(GameType.BASE, "Kill Borealis"))
    known = _known("Kill Borealis", game_type=GameType.DLC)

    changes = consolidate_match_squares(extraction, known)

    assert changes[0].is_new
    assert known[GameType.BASE] == [
        Square(id="borealis", text="Kill Borealis", game_type=GameType.BASE)
    ]
    assert known[GameType.DLC] == [Square(id="id_0", text="Kill Borealis", game_type=GameType.DLC)]
