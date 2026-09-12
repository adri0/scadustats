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
from scadustats.extract import ExtractionSummary
from scadustats.models import GameResult, GameType, MatchMetadata, MatchType, WinType


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


_EXTRACT_ARGS = [
    "--match-date",
    "2026-03-05",
    "--season",
    "6",
    "--match-type",
    "playoffs",
    "--game-type",
    "base",
]


def _fake_extract_video_success(*args, **kwargs):
    return ExtractionSummary(video_id="v1", num_games=1, num_claims=0)


def test_extract_downloads_and_deletes_video_on_success(tmp_path, monkeypatch):
    downloaded = tmp_path / "abc123.mp4"
    downloaded.write_bytes(b"fake video")

    monkeypatch.setattr("scadustats.cli.download_video", lambda url, output_dir: downloaded)
    monkeypatch.setattr("scadustats.cli.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.extract_video", _fake_extract_video_success)

    result = CliRunner().invoke(
        app,
        [
            "extract",
            "https://youtu.be/abc123",
            "--download-dir",
            str(tmp_path),
            *_EXTRACT_ARGS,
        ],
    )

    assert result.exit_code == 0, result.output
    assert not downloaded.exists()


def test_extract_does_not_download_a_local_video_path(tmp_path, monkeypatch):
    local = tmp_path / "local.mp4"
    local.write_bytes(b"fake video")

    def _fail_download(*args, **kwargs):
        raise AssertionError("download_video should not be called for a local path")

    monkeypatch.setattr("scadustats.cli.download_video", _fail_download)
    monkeypatch.setattr("scadustats.cli.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.extract_video", _fake_extract_video_success)

    result = CliRunner().invoke(app, ["extract", str(local), *_EXTRACT_ARGS])

    assert result.exit_code == 0, result.output
    # A local video is never deleted, whatever happens -- only ones we downloaded are.
    assert local.exists()


def test_extract_deletes_downloaded_video_when_extraction_fails_and_confirmed(
    tmp_path, monkeypatch
):
    downloaded = tmp_path / "abc123.mp4"
    downloaded.write_bytes(b"fake video")

    monkeypatch.setattr("scadustats.cli.download_video", lambda url, output_dir: downloaded)
    monkeypatch.setattr("scadustats.cli.estimate_sample_count", lambda path: 1)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("scadustats.cli.extract_video", _boom)

    result = CliRunner().invoke(
        app,
        ["extract", "https://youtu.be/abc123", "--download-dir", str(tmp_path), *_EXTRACT_ARGS],
        input="y\n",
    )

    assert result.exit_code != 0
    assert not downloaded.exists()


def test_extract_keeps_downloaded_video_when_extraction_fails_and_declined(tmp_path, monkeypatch):
    downloaded = tmp_path / "abc123.mp4"
    downloaded.write_bytes(b"fake video")

    monkeypatch.setattr("scadustats.cli.download_video", lambda url, output_dir: downloaded)
    monkeypatch.setattr("scadustats.cli.estimate_sample_count", lambda path: 1)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("scadustats.cli.extract_video", _boom)

    result = CliRunner().invoke(
        app,
        ["extract", "https://youtu.be/abc123", "--download-dir", str(tmp_path), *_EXTRACT_ARGS],
        input="n\n",
    )

    assert result.exit_code != 0
    assert downloaded.exists()


def test_extract_uses_video_url_argument_as_provenance_when_not_separately_given(
    tmp_path, monkeypatch
):
    downloaded = tmp_path / "abc123.mp4"
    downloaded.write_bytes(b"fake video")
    captured: dict[str, MatchMetadata] = {}

    def _fake_extract_video(video_path, *, match_metadata, **kwargs):
        captured["match_metadata"] = match_metadata.result()
        return ExtractionSummary(video_id="v1", num_games=1, num_claims=0)

    monkeypatch.setattr("scadustats.cli.download_video", lambda url, output_dir: downloaded)
    monkeypatch.setattr("scadustats.cli.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.extract_video", _fake_extract_video)

    result = CliRunner().invoke(
        app,
        [
            "extract",
            "https://youtu.be/abc123",
            "--download-dir",
            str(tmp_path),
            *_EXTRACT_ARGS,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["match_metadata"].video_url == "https://youtu.be/abc123"
