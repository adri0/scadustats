"""Human-reviewable YAML export of one player's consolidated profile (see
pipeline.consolidate.consolidate_players), one file per player under
`<data_dir>/players` -- the same "one file per unit, plain text, hand-editable" shape
storage.json_export uses for matches, but YAML rather than JSON specifically so the
free-text `bio` field can be written as an ordinary multi-line block rather than an
escaped JSON string.

Unlike a match file, a player file mixes fields `player consolidate` wholly regenerates
every run (slug, display_name, the win/loss records, all_matches, top squares) with a
few it never sets at all (twitch, avatar, bio) and one it sets once and never touches
again (id) -- read_players is what lets a re-run recover those three from the file
already on disk instead of overwriting them with nothing (see
pipeline.consolidate.consolidate_players's `existing` parameter and cli.app.
player_consolidate).
"""

from pathlib import Path
from urllib.parse import urlparse

import yaml

from scadustats.models import GameType, MatchRecord, PlayerProfile, SquareMarks, WinLoss


class _PlayerDumper(yaml.SafeDumper):
    """A SafeDumper subclass carrying its own string representer (see below) rather than
    registering it on yaml.SafeDumper directly -- this module is the only thing in the
    codebase that writes YAML, but a representer added to the class the yaml module
    itself hands out would still leak into any future unrelated use of yaml.safe_dump.
    """


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """A multi-line string (the free-text `bio` field is the only one expected to be) is
    written as a literal block ("|"), not PyYAML's default single-quoted scalar with
    embedded "\\n"s -- bio exists to be hand-written/edited as ordinary multi-line text
    (see the module docstring), and write_player would otherwise flatten a contributor's
    own block-style edit back into a much less readable quoted form on every re-run.
    """
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_PlayerDumper.add_representer(str, _represent_str)


def player_path(players_dir: str | Path, slug: str) -> Path:
    """The path write_player writes (or would write) one player's profile to --
    `<players_dir>/<slug>.yaml`."""
    return Path(players_dir) / f"{slug}.yaml"


def _profile_to_dict(profile: PlayerProfile) -> dict:
    return {
        "id": profile.id,
        "slug": profile.slug,
        "display_name": profile.display_name,
        "twitch": profile.twitch,
        "avatar": profile.avatar,
        "bio": profile.bio,
        "season_records": {
            season: {"wins": record.wins, "losses": record.losses, "draws": record.draws}
            for season, record in sorted(profile.season_records.items())
        },
        "game_record": {"wins": profile.game_record.wins, "losses": profile.game_record.losses},
        "game_type_records": {
            game_type.value: {"wins": record.wins, "losses": record.losses}
            for game_type, record in profile.game_type_records.items()
        },
        "all_matches": profile.all_matches,
        "top_squares_base_game": [
            {"text": square.text, "marks": square.marks} for square in profile.top_squares_base_game
        ],
        "top_squares_dlc": [
            {"text": square.text, "marks": square.marks} for square in profile.top_squares_dlc
        ],
    }


def write_player(players_dir: str | Path, profile: PlayerProfile) -> Path:
    """Write one player's profile to `<players_dir>/<slug>.yaml` (see player_path),
    creating players_dir if it doesn't exist yet. Unconditionally overwrites, the same as
    write_squares -- every regenerated field is wholly replaced by this run's
    consolidate_players output; only the caller's use of read_players beforehand (see
    consolidate_players's `existing` parameter) is what carries id/twitch/avatar/bio
    forward across that replacement.

    Field order in the written file follows models.PlayerProfile's own declaration order
    (sort_keys=False) rather than alphabetical, so a contributor reading the file top to
    bottom sees identity, hand-filled fields, then derived stats -- not an arbitrary
    re-sort of them.
    """
    Path(players_dir).mkdir(parents=True, exist_ok=True)
    path = player_path(players_dir, profile.slug)
    body = yaml.dump(
        _profile_to_dict(profile), Dumper=_PlayerDumper, sort_keys=False, allow_unicode=True
    )
    path.write_text(body)
    return path


def _dict_to_square_marks(entry: dict | str) -> SquareMarks:
    """One top_squares_base_game/top_squares_dlc entry -- a plain text string, for a file
    written before mark counts were recorded here, reads back as an unknown (zero) count
    rather than failing to parse; every field consolidate_players fills in here is
    wholly regenerated on the next `player consolidate` run anyway (see PlayerProfile),
    so a stale mark count surviving even one extra run costs nothing."""
    if isinstance(entry, str):
        return SquareMarks(text=entry, marks=0)
    return SquareMarks(**entry)


def _twitch_handle(data: dict) -> str | None:
    """The `twitch` field, or -- for a file written before the field held just the
    handle -- the handle recovered from a legacy `twitch_url` full URL (e.g.
    "https://www.twitch.tv/blanxz" -> "blanxz"), so a handle already filled in by hand
    under the old field name isn't silently dropped by this rename."""
    if "twitch" in data:
        return data["twitch"]
    legacy_url = data.get("twitch_url")
    return urlparse(legacy_url).path.strip("/") or None if legacy_url else None


def _dict_to_profile(data: dict) -> PlayerProfile:
    return PlayerProfile(
        id=data["id"],
        slug=data["slug"],
        display_name=data["display_name"],
        twitch=_twitch_handle(data),
        avatar=data.get("avatar"),
        bio=data.get("bio"),
        season_records={
            season: MatchRecord(**record)
            for season, record in data.get("season_records", {}).items()
        },
        game_record=WinLoss(**data["game_record"]) if data.get("game_record") else WinLoss(),
        game_type_records={
            GameType(game_type): WinLoss(**record)
            for game_type, record in data.get("game_type_records", {}).items()
        },
        all_matches=data.get("all_matches", []),
        top_squares_base_game=[
            _dict_to_square_marks(square) for square in data.get("top_squares_base_game", [])
        ],
        top_squares_dlc=[
            _dict_to_square_marks(square) for square in data.get("top_squares_dlc", [])
        ],
    )


def read_player(path: str | Path) -> PlayerProfile:
    """Inverse of write_player: parses one player's YAML file back into the
    PlayerProfile it was serialized from."""
    data = yaml.safe_load(Path(path).read_text())
    return _dict_to_profile(data)


def read_players(players_dir: str | Path) -> dict[str, PlayerProfile]:
    """Every player profile already on disk under players_dir, keyed by slug -- empty if
    the directory doesn't exist yet (a fresh checkout, or one `player consolidate` has
    never run against), the same "ships empty" tolerance pipeline.squares/read_squares
    has for a data directory with no squares reference yet.
    """
    directory = Path(players_dir)
    if not directory.is_dir():
        return {}
    return {path.stem: read_player(path) for path in sorted(directory.glob("*.yaml"))}
