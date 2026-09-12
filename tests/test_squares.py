import json
import logging

from scadustats.models import GameType
from scadustats.squares import infer_game_type, load_known_squares


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


def test_load_known_squares_parses_json_file(tmp_path):
    path = tmp_path / "squares.json"
    path.write_text(json.dumps({"Kill Wormface": "base", "Some DLC goal": "dlc"}))

    known = load_known_squares(path)

    assert known == {"Kill Wormface": GameType.BASE, "Some DLC goal": GameType.DLC}


def test_load_known_squares_handles_empty_file(tmp_path):
    path = tmp_path / "squares.json"
    path.write_text("{}")

    assert load_known_squares(path) == {}
