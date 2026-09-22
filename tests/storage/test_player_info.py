import yaml

from scadustats.models import PlayerInfo
from scadustats.storage.player_info import (
    player_info_path,
    read_player_info,
    read_players_info,
    write_player_info,
)


def _info(**overrides) -> PlayerInfo:
    defaults = dict(id=1, slug="alice", twitch="alice", avatar="alice.png", bio="hand-written bio")
    defaults.update(overrides)
    return PlayerInfo(**defaults)


def test_write_player_info_creates_file_at_slug_path(tmp_path):
    path = write_player_info(tmp_path, _info())

    assert path == player_info_path(tmp_path, "alice")
    assert path.exists()


def test_write_player_info_creates_players_dir(tmp_path):
    players_dir = tmp_path / "players"

    write_player_info(players_dir, _info())

    assert players_dir.is_dir()


def test_read_player_info_round_trips_every_field(tmp_path):
    info = _info()
    path = write_player_info(tmp_path, info)

    result = read_player_info(path)

    assert result == info


def test_read_player_info_round_trips_manual_fields_left_blank(tmp_path):
    info = _info(twitch=None, avatar=None, bio=None)
    path = write_player_info(tmp_path, info)

    result = read_player_info(path)

    assert result.twitch is None
    assert result.avatar is None
    assert result.bio is None


def test_write_player_info_does_not_overwrite_an_existing_file(tmp_path):
    write_player_info(tmp_path, _info(twitch="alice"))

    written = write_player_info(tmp_path, _info(twitch="someone-else"))

    assert written is None
    assert read_player_info(player_info_path(tmp_path, "alice")).twitch == "alice"


def test_read_players_info_is_empty_for_a_missing_directory(tmp_path):
    assert read_players_info(tmp_path / "does-not-exist") == {}


def test_read_players_info_is_empty_for_a_directory_with_no_yaml_files(tmp_path):
    assert read_players_info(tmp_path) == {}


def test_write_player_info_writes_a_multiline_bio_as_a_literal_block(tmp_path):
    info = _info(bio="Line one.\nLine two.")

    path = write_player_info(tmp_path, info)

    body = path.read_text()
    assert "bio: |" in body
    assert read_player_info(path).bio == "Line one.\nLine two."


def test_read_player_info_recovers_a_handle_from_a_legacy_twitch_url(tmp_path):
    """A file written before `twitch` held just the handle has a `twitch_url` full URL
    instead -- a handle already filled in by hand under the old field name shouldn't be
    silently dropped by the rename."""
    path = write_player_info(tmp_path, _info(twitch=None))
    data = yaml.safe_load(path.read_text())
    del data["twitch"]
    data["twitch_url"] = "https://www.twitch.tv/blanxz"
    path.write_text(yaml.safe_dump(data, sort_keys=False))

    assert read_player_info(path).twitch == "blanxz"


def test_read_player_info_treats_a_blank_legacy_twitch_url_as_no_handle(tmp_path):
    path = write_player_info(tmp_path, _info(twitch=None))
    data = yaml.safe_load(path.read_text())
    del data["twitch"]
    data["twitch_url"] = None
    path.write_text(yaml.safe_dump(data, sort_keys=False))

    assert read_player_info(path).twitch is None


def test_read_players_info_keys_by_slug(tmp_path):
    write_player_info(tmp_path, _info(id=1, slug="alice"))
    write_player_info(tmp_path, _info(id=2, slug="bob"))

    players = read_players_info(tmp_path)

    assert set(players) == {"alice", "bob"}
    assert players["bob"].id == 2
