import datetime

from scadustats.cli import _prompt_match_date, _prompt_match_metadata, _prompt_video_url
from scadustats.models import MatchType


def test_prompt_match_metadata_uses_supplied_options_without_prompting(monkeypatch):
    def _fail_prompt(*args, **kwargs):
        raise AssertionError("typer.prompt should not be called when all options are supplied")

    monkeypatch.setattr("scadustats.cli.typer.prompt", _fail_prompt)

    metadata = _prompt_match_metadata(
        "2026-03-05", 6, MatchType.PLAYOFFS, "https://youtu.be/abc123"
    )

    assert metadata.match_date == datetime.date(2026, 3, 5)
    assert metadata.season == 6
    assert metadata.match_type == MatchType.PLAYOFFS
    assert metadata.video_url == "https://youtu.be/abc123"


def test_prompt_match_metadata_defaults_season_from_match_date(monkeypatch):
    prompts = []

    def _fake_prompt(text, default=None, type=None, **kwargs):
        prompts.append((text, default))
        return default

    monkeypatch.setattr("scadustats.cli.typer.prompt", _fake_prompt)

    # 2026 is season 6, one season per year -- so 2029 should default to season 9.
    metadata = _prompt_match_metadata("2029-01-01", None, MatchType.DOUBLE_ELIMINATION, None)

    assert metadata.season == 9
    assert ("Season", 9) in prompts
    assert metadata.video_url is None


def test_prompt_match_date_has_no_default(monkeypatch):
    calls = []

    def _fake_prompt(text, default=None, type=None):
        calls.append((text, default))
        return "2026-03-05"

    monkeypatch.setattr("scadustats.cli.typer.prompt", _fake_prompt)

    result = _prompt_match_date()

    assert result == datetime.date(2026, 3, 5)
    assert calls == [("Match date (YYYY-MM-DD)", None)]


def test_prompt_match_date_reprompts_on_invalid_input(monkeypatch):
    responses = iter(["not-a-date", "2026-03-05"])
    monkeypatch.setattr("scadustats.cli.typer.prompt", lambda *a, **k: next(responses))
    echoed = []
    monkeypatch.setattr("scadustats.cli.typer.echo", echoed.append)

    result = _prompt_match_date()

    assert result == datetime.date(2026, 3, 5)
    assert any("Invalid date" in msg for msg in echoed)


def test_prompt_match_metadata_requires_match_date_when_not_supplied(monkeypatch):
    monkeypatch.setattr("scadustats.cli.typer.prompt", lambda *a, **k: "2026-03-05")

    metadata = _prompt_match_metadata(None, 6, MatchType.PLAYOFFS, "")

    assert metadata.match_date == datetime.date(2026, 3, 5)


def test_prompt_video_url_returns_none_for_blank_input(monkeypatch):
    monkeypatch.setattr("scadustats.cli.typer.prompt", lambda *a, **k: "")

    assert _prompt_video_url() is None


def test_prompt_video_url_strips_and_returns_value(monkeypatch):
    monkeypatch.setattr("scadustats.cli.typer.prompt", lambda *a, **k: "  https://youtu.be/x  ")

    assert _prompt_video_url() == "https://youtu.be/x"
