"""YAML persistence for one player's static, hand-curated identity (see
models.PlayerInfo) -- the half of a player's profile pipeline.consolidate creates once
and never touches again (issue #82). One `info.yaml` per player, under
`<players_dir>/<slug>/info.yaml` -- the same `<data_dir>/players/<slug>/` directory
storage.player_stats writes its own `stats.yaml` into, so a player's profile lives in
one place rather than two separate directory trees. The two files are still tracked
completely differently, though: info.yaml is the one committed to the repo (see the
`!data/players/*/info.yaml` carve-out in .gitignore) -- twitch/avatar/bio are genuinely
hand-authored content a contributor fills in and expects to survive forever, unlike
stats.yaml's disposable, gitignored output.

write_player_info only ever creates a file, never overwrites one that already exists --
there's nothing here for a re-run to legitimately change: id is assigned once and never
revisited, and every other field is meant to be filled in and kept by hand. A caller is
expected to only construct a PlayerInfo for a slug that isn't already on disk (see
pipeline.consolidate.consolidate_player_info); this module's own "already exists" guard
is a backstop against a hand-edited file being clobbered even if that expectation is
ever violated, not something normal use is meant to rely on.
"""

from pathlib import Path
from urllib.parse import urlparse

import yaml

from scadustats.models import PlayerInfo


class _PlayerInfoDumper(yaml.SafeDumper):
    """A SafeDumper subclass carrying its own string representer (see below) rather than
    registering it on yaml.SafeDumper directly -- this module is one of two things in
    the codebase that write YAML (see storage.player_stats), and a representer added to
    the class the yaml module itself hands out would leak into the other's use of
    yaml.safe_dump.
    """


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """A multi-line string (the free-text `bio` field is the only one expected to be) is
    written as a literal block ("|"), not PyYAML's default single-quoted scalar with
    embedded "\\n"s -- bio exists to be hand-written/edited as ordinary multi-line text
    (see the module docstring), and a quoted form would flatten a contributor's own
    block-style edit back into a much less readable form.
    """
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_PlayerInfoDumper.add_representer(str, _represent_str)


def player_info_path(players_dir: str | Path, slug: str) -> Path:
    """The path write_player_info writes (or would write) one player's info to --
    `<players_dir>/<slug>/info.yaml`."""
    return Path(players_dir) / slug / "info.yaml"


def _info_to_dict(info: PlayerInfo) -> dict:
    return {
        "id": info.id,
        "slug": info.slug,
        "twitch": info.twitch,
        "avatar": info.avatar,
        "bio": info.bio,
    }


def write_player_info(players_dir: str | Path, info: PlayerInfo) -> Path | None:
    """Create `<players_dir>/<slug>/info.yaml` for a brand-new player (see
    player_info_path), creating that player's directory if it doesn't exist yet.
    Returns None without writing anything if the file already exists, rather than
    overwriting it -- see the module docstring.
    """
    path = player_info_path(players_dir, info.slug)
    if path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.dump(
        _info_to_dict(info), Dumper=_PlayerInfoDumper, sort_keys=False, allow_unicode=True
    )
    path.write_text(body)
    return path


def _twitch_handle(data: dict) -> str | None:
    """The `twitch` field, or -- for a file written before the field held just the
    handle -- the handle recovered from a legacy `twitch_url` full URL (e.g.
    "https://www.twitch.tv/blanxz" -> "blanxz"), so a handle already filled in by hand
    under the old field name isn't silently dropped by this rename."""
    if "twitch" in data:
        return data["twitch"]
    legacy_url = data.get("twitch_url")
    return urlparse(legacy_url).path.strip("/") or None if legacy_url else None


def _dict_to_info(data: dict) -> PlayerInfo:
    return PlayerInfo(
        id=data["id"],
        slug=data["slug"],
        twitch=_twitch_handle(data),
        avatar=data.get("avatar"),
        bio=data.get("bio"),
    )


def read_player_info(path: str | Path) -> PlayerInfo:
    """Inverse of write_player_info: parses one player's YAML info file back into the
    PlayerInfo it was serialized from."""
    data = yaml.safe_load(Path(path).read_text())
    return _dict_to_info(data)


def read_players_info(players_dir: str | Path) -> dict[str, PlayerInfo]:
    """Every player's `info.yaml` already on disk under players_dir, keyed by its
    parent directory's name (the slug) -- empty if players_dir doesn't exist yet (a
    fresh checkout, or one that's never had a new player consolidated into it), the
    same "ships empty" tolerance pipeline.squares/read_squares has for a data directory
    with no squares reference yet.
    """
    directory = Path(players_dir)
    if not directory.is_dir():
        return {}
    return {
        path.parent.name: read_player_info(path) for path in sorted(directory.glob("*/info.yaml"))
    }
