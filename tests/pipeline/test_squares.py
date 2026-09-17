import logging

from scadustats.models import GameType, Square
from scadustats.pipeline.squares import infer_game_type, load_known_squares
from scadustats.storage.json_export import write_squares


def _board(*texts: str) -> list[list[str]]:
    flat = list(texts) + [""] * (25 - len(texts))
    return [flat[r * 5 : r * 5 + 5] for r in range(5)]


def test_infer_game_type_returns_none_for_empty_known_squares():
    board = _board("Kill Wormface", "Acquire 3 Memory Stones")

    assert infer_game_type(board, {}) is None


def test_infer_game_type_returns_none_when_no_square_matches():
    board = _board("Some unseen goal")
    known = {"Kill Wormface": GameType.BASE}

    assert infer_game_type(board, known) is None


def test_infer_game_type_returns_matched_type():
    board = _board("Kill Wormface", "Acquire 3 Memory Stones")
    known = {"Kill Wormface": GameType.BASE, "Acquire 3 Memory Stones": GameType.BASE}

    assert infer_game_type(board, known) is GameType.BASE


def test_infer_game_type_majority_vote_on_conflicting_matches(caplog):
    board = _board("Base goal 1", "Base goal 2", "DLC goal")
    known = {
        "Base goal 1": GameType.BASE,
        "Base goal 2": GameType.BASE,
        "DLC goal": GameType.DLC,
    }

    with caplog.at_level(logging.WARNING):
        result = infer_game_type(board, known)

    assert result is GameType.BASE
    assert any("more than one game type" in r.message for r in caplog.records)


def test_load_known_squares_flattens_both_game_types(tmp_path):
    write_squares(
        tmp_path,
        {
            GameType.BASE: [Square(id="wormface", text="Kill Wormface", game_type=GameType.BASE)],
            GameType.DLC: [Square(id="dlc_goal", text="Some DLC goal", game_type=GameType.DLC)],
        },
    )

    known = load_known_squares(tmp_path)

    assert known == {"Kill Wormface": GameType.BASE, "Some DLC goal": GameType.DLC}


def test_load_known_squares_handles_missing_files(tmp_path):
    """A data_dir/squares that's never had `square consolidate` run against it (a fresh
    checkout, or one that's never extracted anything) ships the same "empty reference"
    behavior squares.json used to -- see issue #75."""
    assert load_known_squares(tmp_path / "squares") == {}
