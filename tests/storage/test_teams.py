from scadustats.models import Team
from scadustats.storage.teams import read_teams


def test_read_teams_by_season(tmp_path):
    (tmp_path / "teams.yaml").write_text(
        '"5":\n  - name: Positive Vibes\n    base_player: star0chris\n    dlc_player: Adef\n'
    )
    assert read_teams(tmp_path) == {
        "5": [Team(name="Positive Vibes", base_player="star0chris", dlc_player="Adef")]
    }


def test_read_teams_without_a_file_is_empty(tmp_path):
    assert read_teams(tmp_path) == {}


def test_read_teams_without_roles(tmp_path):
    (tmp_path / "teams.yaml").write_text(
        '"3":\n  - name: Cattery\n    members: [catalystz, Puppery]\n'
    )
    assert read_teams(tmp_path) == {"3": [Team(name="Cattery", members=["catalystz", "Puppery"])]}
