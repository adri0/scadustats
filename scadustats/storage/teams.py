"""The 2v2 team rosters (see models.Team), hand-curated in `<data_dir>/teams.yaml` --
committed alongside the matches, like a player's info.yaml, since the overlay only shows
team names and who plays in a team (and in which game) can't be read off the video.

Keyed by season, each a list of teams:

    "5":
      - name: Positive Vibes
        base_player: star0chris
        dlc_player: Adef
"""

from pathlib import Path

import yaml

from scadustats.models import Team


def teams_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "teams.yaml"


def read_teams(data_dir: str | Path) -> dict[str, list[Team]]:
    """Every season's roster, or an empty dict when there's no teams.yaml."""
    try:
        data = yaml.safe_load(teams_path(data_dir).read_text()) or {}
    except FileNotFoundError:
        return {}
    return {
        str(season): [Team.model_validate(team) for team in teams] for season, teams in data.items()
    }
