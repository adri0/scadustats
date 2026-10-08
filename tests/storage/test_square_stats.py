from scadustats.models import GameType, PlayerMarks, SquareClaim, SquareStats
from scadustats.storage.square_stats import (
    read_square_stats,
    square_stats_path,
    write_square_stats,
)


def _stats(**overrides) -> SquareStats:
    defaults = dict(
        id="crystalian",
        text="Kill a Crystalian",
        game_type=GameType.BASE,
        num_matches=2,
        matches=["2026-03-06-alice-vs-carol", "2026-03-05-alice-vs-bob"],
        num_games=3,
        games_marked=2,
        mark_rate=0.6667,
        top_players=[PlayerMarks(slug="alice", marks=2)],
        mark_times_s=[120, 480],
        median_mark_time_s=300.0,
        claims=[
            SquareClaim(match_id="2026-03-05-alice-vs-bob", game_index=1, time_s=120, slug="alice"),
            SquareClaim(match_id="2026-03-06-alice-vs-carol", game_index=2, time_s=480, slug=None),
        ],
    )
    defaults.update(overrides)
    return SquareStats(**defaults)


def test_write_square_stats_files_each_game_type_in_its_own_directory(tmp_path):
    base = write_square_stats(tmp_path, _stats())
    dlc = write_square_stats(tmp_path, _stats(game_type=GameType.DLC))

    assert base == tmp_path / "base" / "crystalian.yaml"
    assert dlc == tmp_path / "dlc" / "crystalian.yaml"
    assert base == square_stats_path(tmp_path, GameType.BASE, "crystalian")


def test_read_square_stats_round_trips_every_field(tmp_path):
    stats = _stats()

    assert read_square_stats(write_square_stats(tmp_path, stats)) == stats


def test_read_square_stats_round_trips_a_square_never_dealt(tmp_path):
    stats = SquareStats(id="merchant", text="Find a Merchant", game_type=GameType.BASE)

    assert read_square_stats(write_square_stats(tmp_path, stats)) == stats
