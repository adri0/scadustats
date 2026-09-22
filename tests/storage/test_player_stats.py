import yaml

from scadustats.models import GameType, MatchRecord, PlayerStats, SquareMarks, WinLoss
from scadustats.storage.player_stats import (
    player_stats_path,
    read_player_stats,
    read_players_stats,
    write_player_stats,
)


def _stats(**overrides) -> PlayerStats:
    defaults = dict(
        slug="alice",
        display_name="alice",
        season_records={"6": MatchRecord(wins=2, losses=1, draws=1)},
        game_record=WinLoss(wins=5, losses=3),
        game_type_records={
            GameType.BASE: WinLoss(wins=3, losses=1),
            GameType.DLC: WinLoss(wins=2, losses=2),
        },
        all_matches=["2026-03-05-alice-vs-bob", "2026-02-01-alice-vs-carol"],
        top_squares_base_game=[SquareMarks(text="Kill Wormface", marks=4)],
        top_squares_dlc=[SquareMarks(text="Acquire 2 Dragon Hearts", marks=2)],
    )
    defaults.update(overrides)
    return PlayerStats(**defaults)


def test_write_player_stats_creates_file_at_slug_path(tmp_path):
    path = write_player_stats(tmp_path, _stats())

    assert path == player_stats_path(tmp_path, "alice")
    assert path.exists()


def test_write_player_stats_creates_players_dir(tmp_path):
    players_dir = tmp_path / "players"

    write_player_stats(players_dir, _stats())

    assert players_dir.is_dir()


def test_read_player_stats_round_trips_every_field(tmp_path):
    stats = _stats()
    path = write_player_stats(tmp_path, stats)

    result = read_player_stats(path)

    assert result == stats


def test_write_player_stats_overwrites_an_existing_file(tmp_path):
    write_player_stats(tmp_path, _stats(display_name="alice"))
    write_player_stats(tmp_path, _stats(display_name="Alice"))

    result = read_player_stats(player_stats_path(tmp_path, "alice"))

    assert result.display_name == "Alice"


def test_read_players_stats_is_empty_for_a_missing_directory(tmp_path):
    assert read_players_stats(tmp_path / "does-not-exist") == {}


def test_read_players_stats_is_empty_for_a_directory_with_no_yaml_files(tmp_path):
    assert read_players_stats(tmp_path) == {}


def test_read_player_stats_treats_a_plain_text_top_square_as_an_unknown_mark_count(tmp_path):
    """A file written before mark counts were added to top_squares_base_game/dlc holds a
    plain list of strings there -- it still has to read back cleanly, as an unknown (zero)
    count, rather than raising."""
    path = write_player_stats(tmp_path, _stats())
    data = yaml.safe_load(path.read_text())
    data["top_squares_base_game"] = ["Kill Wormface"]
    path.write_text(yaml.safe_dump(data, sort_keys=False))

    result = read_player_stats(path)

    assert result.top_squares_base_game == [SquareMarks(text="Kill Wormface", marks=0)]


def test_read_players_stats_keys_by_slug(tmp_path):
    write_player_stats(tmp_path, _stats(slug="alice", display_name="alice"))
    write_player_stats(tmp_path, _stats(slug="bob", display_name="bob"))

    players = read_players_stats(tmp_path)

    assert set(players) == {"alice", "bob"}
    assert players["bob"].display_name == "bob"
