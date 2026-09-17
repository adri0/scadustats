import yaml

from scadustats.models import GameType, MatchRecord, PlayerProfile, SquareMarks, WinLoss
from scadustats.storage.player_export import player_path, read_player, read_players, write_player


def _profile(**overrides) -> PlayerProfile:
    defaults = dict(
        id=1,
        slug="alice",
        display_name="alice",
        twitch_url="https://twitch.tv/alice",
        avatar="alice.png",
        bio="hand-written bio",
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
    return PlayerProfile(**defaults)


def test_write_player_creates_file_at_slug_path(tmp_path):
    path = write_player(tmp_path, _profile())

    assert path == player_path(tmp_path, "alice")
    assert path.exists()


def test_write_player_creates_players_dir(tmp_path):
    players_dir = tmp_path / "players"

    write_player(players_dir, _profile())

    assert players_dir.is_dir()


def test_read_player_round_trips_every_field(tmp_path):
    profile = _profile()
    path = write_player(tmp_path, profile)

    result = read_player(path)

    assert result == profile


def test_read_player_round_trips_manual_fields_left_blank(tmp_path):
    profile = _profile(twitch_url=None, avatar=None, bio=None)
    path = write_player(tmp_path, profile)

    result = read_player(path)

    assert result.twitch_url is None
    assert result.avatar is None
    assert result.bio is None


def test_write_player_overwrites_an_existing_file(tmp_path):
    write_player(tmp_path, _profile(display_name="alice"))
    write_player(tmp_path, _profile(display_name="Alice"))

    result = read_player(player_path(tmp_path, "alice"))

    assert result.display_name == "Alice"


def test_read_players_is_empty_for_a_missing_directory(tmp_path):
    assert read_players(tmp_path / "does-not-exist") == {}


def test_read_players_is_empty_for_a_directory_with_no_yaml_files(tmp_path):
    assert read_players(tmp_path) == {}


def test_write_player_writes_a_multiline_bio_as_a_literal_block(tmp_path):
    profile = _profile(bio="Line one.\nLine two.")

    path = write_player(tmp_path, profile)

    body = path.read_text()
    assert "bio: |" in body
    assert read_player(path).bio == "Line one.\nLine two."


def test_read_player_treats_a_plain_text_top_square_as_an_unknown_mark_count(tmp_path):
    """A file written before mark counts were added to top_squares_base_game/dlc holds a
    plain list of strings there -- it still has to read back cleanly, as an unknown (zero)
    count, rather than raising."""
    path = write_player(tmp_path, _profile())
    data = yaml.safe_load(path.read_text())
    data["top_squares_base_game"] = ["Kill Wormface"]
    path.write_text(yaml.safe_dump(data, sort_keys=False))

    result = read_player(path)

    assert result.top_squares_base_game == [SquareMarks(text="Kill Wormface", marks=0)]


def test_read_players_keys_by_slug(tmp_path):
    write_player(tmp_path, _profile(slug="alice", display_name="alice"))
    write_player(tmp_path, _profile(slug="bob", display_name="bob"))

    players = read_players(tmp_path)

    assert set(players) == {"alice", "bob"}
    assert players["bob"].display_name == "bob"
