import datetime

import pytest
from typer.testing import CliRunner

from scadustats.cli import (
    _is_youtube_url,
    _prompt_game_type,
    _prompt_match_date,
    _prompt_match_metadata,
    _prompt_video_url,
    app,
)
from scadustats.models import GameResult, GameType, MatchType, WinType


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


def test_prompt_video_url_reprompts_on_non_youtube_url(monkeypatch):
    responses = iter(["https://vimeo.com/123", "https://youtu.be/x"])
    monkeypatch.setattr("scadustats.cli.typer.prompt", lambda *a, **k: next(responses))
    echoed = []
    monkeypatch.setattr("scadustats.cli.typer.echo", echoed.append)

    result = _prompt_video_url()

    assert result == "https://youtu.be/x"
    assert any("vimeo.com" in msg for msg in echoed)


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com/watch?v=abc123",
        "https://www.youtube.com/watch?v=abc123",
        "https://m.youtube.com/watch?v=abc123",
        "https://youtu.be/abc123",
        "http://youtu.be/abc123",
    ],
)
def test_is_youtube_url_accepts_youtube_hosts(url):
    assert _is_youtube_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://vimeo.com/123",
        "https://notyoutube.com/watch?v=abc123",
        "https://evil.com/?u=youtube.com",
        "not a url",
        "",
    ],
)
def test_is_youtube_url_rejects_non_youtube_urls(url):
    assert not _is_youtube_url(url)


def _sample_game() -> GameResult:
    return GameResult(
        game_index=1,
        label="GAME 1",
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        square_texts=[[""] * 5 for _ in range(5)],
        events=[],
        winner_color=None,
        win_type=WinType.NONE,
    )


def test_prompt_game_type_returns_supplied_option(monkeypatch):
    def _fail_prompt(*args, **kwargs):
        raise AssertionError("typer.prompt should not be called when game_type_opt is given")

    monkeypatch.setattr("scadustats.cli.typer.prompt", _fail_prompt)

    assert _prompt_game_type(_sample_game(), GameType.DLC) is GameType.DLC


def test_prompt_game_type_prompts_when_not_supplied(monkeypatch):
    prompts = []

    def _fake_prompt(text, **kwargs):
        prompts.append(text)
        return "dlc"

    monkeypatch.setattr("scadustats.cli.typer.prompt", _fake_prompt)

    result = _prompt_game_type(_sample_game(), None)

    assert result is GameType.DLC
    assert len(prompts) == 1
    assert "GAME 1" in prompts[0]


def test_prompt_game_type_reprompts_on_invalid_input(monkeypatch):
    responses = iter(["not-a-type", "base"])
    monkeypatch.setattr("scadustats.cli.typer.prompt", lambda *a, **k: next(responses))
    echoed = []
    monkeypatch.setattr("scadustats.cli.typer.echo", echoed.append)

    result = _prompt_game_type(_sample_game(), None)

    assert result is GameType.BASE
    assert any("Invalid game type" in msg for msg in echoed)


def test_extract_rejects_non_youtube_video_url_before_doing_any_work():
    # video_path doesn't need to exist -- --video-url is checked before the video is
    # ever touched, so this never gets far enough to notice.
    result = CliRunner().invoke(
        app,
        [
            "extract",
            "nonexistent.mp4",
            "--video-url",
            "https://vimeo.com/123",
            "--match-date",
            "2026-03-05",
            "--season",
            "6",
            "--match-type",
            "playoffs",
        ],
    )

    assert result.exit_code != 0
    assert "youtube.com" in result.output
