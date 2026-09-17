import datetime

import pytest
from typer.testing import CliRunner

from scadustats.cli.app import (
    _find_existing_match,
    _is_youtube_url,
    _prompt_game_type,
    _prompt_match_date,
    _prompt_match_metadata,
    _prompt_video_url,
    app,
)
from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameResult,
    GameType,
    MatchMetadata,
    MatchType,
    VideoExtraction,
    WinLine,
    WinType,
)
from scadustats.pipeline.extract import ExtractionSummary
from scadustats.storage.json_export import write_video
from scadustats.video.download import DownloadResult


def test_prompt_match_metadata_uses_supplied_options_without_prompting(monkeypatch):
    def _fail_prompt(*args, **kwargs):
        raise AssertionError("typer.prompt should not be called when all options are supplied")

    monkeypatch.setattr("scadustats.cli.app.typer.prompt", _fail_prompt)

    metadata = _prompt_match_metadata(
        "2026-03-05", "6", MatchType.ROUND_ROBIN, "https://youtu.be/abc123"
    )

    assert metadata.match_date == datetime.date(2026, 3, 5)
    assert metadata.season == "6"
    assert metadata.match_type == MatchType.ROUND_ROBIN
    assert metadata.video_url == "https://youtu.be/abc123"


def test_prompt_match_metadata_defaults_season_from_match_date(monkeypatch):
    prompts = []

    def _fake_prompt(text, default=None, type=None, **kwargs):
        prompts.append((text, default))
        return default

    monkeypatch.setattr("scadustats.cli.app.typer.prompt", _fake_prompt)

    # 2026 is season 6, one season per year -- so 2029 should default to season 9.
    metadata = _prompt_match_metadata("2029-01-01", None, MatchType.DOUBLE_ELIMINATION, None)

    assert metadata.season == "9"
    assert ("Season", "9") in prompts
    assert metadata.video_url is None


def test_prompt_match_metadata_offers_existing_match_details_as_defaults(monkeypatch):
    """existing_match (see _find_existing_match) comes from a previously-extracted match
    that shares this run's source link -- its date/season/video_url are real recorded
    facts about that same source, so they're safe to offer as defaults rather than the
    usual from-scratch ones (no default date, a computed season, no default video_url)."""
    prompts = []

    def _fake_prompt(text, default=None, type=None, **kwargs):
        prompts.append((text, default))
        return default if default else "unused"

    monkeypatch.setattr("scadustats.cli.app.typer.prompt", _fake_prompt)

    existing = _sample_extraction(
        match_date=datetime.date(2026, 1, 1), season="Off-Season Cup"
    )
    metadata = _prompt_match_metadata(None, None, MatchType.DOUBLE_ELIMINATION, None, existing)

    assert metadata.match_date == datetime.date(2026, 1, 1)
    assert metadata.season == "Off-Season Cup"
    assert metadata.video_url == "https://youtu.be/abc123"
    assert ("Match date (YYYY-MM-DD)", "2026-01-01") in prompts
    assert ("Season", "Off-Season Cup") in prompts


def test_prompt_match_metadata_explicit_options_override_existing_match_defaults(
    monkeypatch,
):
    """An existing match only supplies defaults -- an explicitly-given --option still
    wins, same as when there's no existing match at all."""

    def _fail_prompt(*args, **kwargs):
        raise AssertionError("typer.prompt should not be called when all options are supplied")

    monkeypatch.setattr("scadustats.cli.app.typer.prompt", _fail_prompt)

    existing = _sample_extraction(match_date=datetime.date(2026, 1, 1), season="1")
    metadata = _prompt_match_metadata(
        "2026-03-05", "6", MatchType.ROUND_ROBIN, "https://youtu.be/xyz789", existing
    )

    assert metadata.match_date == datetime.date(2026, 3, 5)
    assert metadata.season == "6"
    assert metadata.video_url == "https://youtu.be/xyz789"


def test_prompt_match_date_has_no_default(monkeypatch):
    calls = []

    def _fake_prompt(text, default=None, type=None):
        calls.append((text, default))
        return "2026-03-05"

    monkeypatch.setattr("scadustats.cli.app.typer.prompt", _fake_prompt)

    result = _prompt_match_date()

    assert result == datetime.date(2026, 3, 5)
    assert calls == [("Match date (YYYY-MM-DD)", None)]


def test_prompt_match_date_reprompts_on_invalid_input(monkeypatch):
    responses = iter(["not-a-date", "2026-03-05"])
    monkeypatch.setattr("scadustats.cli.app.typer.prompt", lambda *a, **k: next(responses))
    echoed = []
    monkeypatch.setattr("scadustats.cli.app.typer.echo", echoed.append)

    result = _prompt_match_date()

    assert result == datetime.date(2026, 3, 5)
    assert any("Invalid date" in msg for msg in echoed)


def test_prompt_match_date_offers_a_default_when_given(monkeypatch):
    calls = []

    def _fake_prompt(text, default=None, type=None):
        calls.append((text, default))
        return default

    monkeypatch.setattr("scadustats.cli.app.typer.prompt", _fake_prompt)

    result = _prompt_match_date(datetime.date(2026, 1, 1))

    assert result == datetime.date(2026, 1, 1)
    assert calls == [("Match date (YYYY-MM-DD)", "2026-01-01")]


def test_prompt_match_metadata_requires_match_date_when_not_supplied(monkeypatch):
    monkeypatch.setattr("scadustats.cli.app.typer.prompt", lambda *a, **k: "2026-03-05")

    metadata = _prompt_match_metadata(None, 6, MatchType.ROUND_ROBIN, "")

    assert metadata.match_date == datetime.date(2026, 3, 5)


def test_prompt_video_url_returns_none_for_blank_input(monkeypatch):
    monkeypatch.setattr("scadustats.cli.app.typer.prompt", lambda *a, **k: "")

    assert _prompt_video_url() is None


def test_prompt_video_url_strips_and_returns_value(monkeypatch):
    monkeypatch.setattr("scadustats.cli.app.typer.prompt", lambda *a, **k: "  https://youtu.be/x  ")

    assert _prompt_video_url() == "https://youtu.be/x"


def test_prompt_video_url_offers_a_default_when_given(monkeypatch):
    monkeypatch.setattr("scadustats.cli.app.typer.prompt", lambda *a, **k: k.get("default", ""))

    assert _prompt_video_url("https://youtu.be/abc123") == "https://youtu.be/abc123"


def test_prompt_video_url_reprompts_on_non_youtube_url(monkeypatch):
    responses = iter(["https://vimeo.com/123", "https://youtu.be/x"])
    monkeypatch.setattr("scadustats.cli.app.typer.prompt", lambda *a, **k: next(responses))
    echoed = []
    monkeypatch.setattr("scadustats.cli.app.typer.echo", echoed.append)

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
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        square_texts=[[""] * 5 for _ in range(5)],
        events=[],
        winner_color=None,
        win_type=WinType.NONE,
    )


def test_prompt_game_type_prompts(monkeypatch):
    prompts = []

    def _fake_prompt(text, **kwargs):
        prompts.append(text)
        return "dlc"

    monkeypatch.setattr("scadustats.cli.app.typer.prompt", _fake_prompt)

    result = _prompt_game_type(_sample_game())

    assert result is GameType.DLC
    assert len(prompts) == 1
    # The prompt names the game by its index -- that's the only identity a game has.
    assert "Game 1" in prompts[0]


def test_prompt_game_type_reprompts_on_invalid_input(monkeypatch):
    responses = iter(["not-a-type", "base"])
    monkeypatch.setattr("scadustats.cli.app.typer.prompt", lambda *a, **k: next(responses))
    echoed = []
    monkeypatch.setattr("scadustats.cli.app.typer.echo", echoed.append)

    result = _prompt_game_type(_sample_game())

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
            "round_robin",
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
    "round_robin",
]


def _fake_extract_video_success(*args, **kwargs):
    return ExtractionSummary(
        video_id="v1", num_games=1, num_claims=0, extraction=_sample_extraction(video_id="v1")
    )


def test_extract_asks_on_duplicate_and_replaces_when_confirmed(tmp_path, monkeypatch):
    local = tmp_path / "local.mp4"
    local.write_bytes(b"fake video")
    dup_path = tmp_path / "matches" / "v1.json"
    captured: dict = {}

    def _fake_extract_video(video_path, *, if_exists, on_duplicate, **kwargs):
        captured["if_exists"] = if_exists
        approved = on_duplicate(dup_path)
        return ExtractionSummary(
            video_id="v1",
            num_games=1,
            num_claims=0,
            extraction=_sample_extraction(video_id="v1"),
            skipped=not approved,
        )

    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.app.extract_video", _fake_extract_video)

    result = CliRunner().invoke(
        app, ["extract", str(local), *_EXTRACT_ARGS, "--video-url", ""], input="y\n"
    )

    assert result.exit_code == 0, result.output
    # Not explicitly given on the command line -- the CLI leaves this None so
    # extract_video knows to ask rather than deciding for itself.
    assert captured["if_exists"] is None
    assert str(dup_path) in result.output
    # Prints the same report as `match show` for the match just extracted.
    assert "v1" in result.output
    assert "alice (red) vs bob (blue)" in result.output


def test_extract_reports_skip_when_duplicate_declined(tmp_path, monkeypatch):
    local = tmp_path / "local.mp4"
    local.write_bytes(b"fake video")
    dup_path = tmp_path / "matches" / "v1.json"

    def _fake_extract_video(video_path, *, on_duplicate, **kwargs):
        approved = on_duplicate(dup_path)
        return ExtractionSummary(
            video_id="v1", num_games=0, num_claims=0, skipped=not approved
        )

    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.app.extract_video", _fake_extract_video)

    result = CliRunner().invoke(
        app, ["extract", str(local), *_EXTRACT_ARGS, "--video-url", ""], input="n\n"
    )

    assert result.exit_code == 0, result.output
    assert "Skipped" in result.output
    assert "v1" in result.output


def test_extract_if_exists_flag_skips_the_duplicate_prompt(tmp_path, monkeypatch):
    local = tmp_path / "local.mp4"
    local.write_bytes(b"fake video")
    captured: dict = {}

    def _fake_extract_video(video_path, *, if_exists, on_duplicate, **kwargs):
        captured["if_exists"] = if_exists
        return ExtractionSummary(
        video_id="v1", num_games=1, num_claims=0, extraction=_sample_extraction(video_id="v1")
    )

    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.app.extract_video", _fake_extract_video)

    # No input given -- if this prompted, there'd be nothing to answer with.
    result = CliRunner().invoke(
        app, ["extract", str(local), "--if-exists", "replace", *_EXTRACT_ARGS]
    )

    assert result.exit_code == 0, result.output
    assert captured["if_exists"] == "replace"


def test_extract_downloads_and_deletes_video_on_success(tmp_path, monkeypatch):
    downloaded = tmp_path / "abc123.mp4"
    downloaded.write_bytes(b"fake video")

    monkeypatch.setattr(
        "scadustats.cli.app.download_video",
        lambda url, output_dir, cookies=None: DownloadResult(
            path=downloaded, published_at=None
        ),
    )
    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.app.extract_video", _fake_extract_video_success)

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


def test_extract_keeps_downloaded_video_on_success_with_keep_video_flag(tmp_path, monkeypatch):
    downloaded = tmp_path / "abc123.mp4"
    downloaded.write_bytes(b"fake video")

    monkeypatch.setattr(
        "scadustats.cli.app.download_video",
        lambda url, output_dir, cookies=None: DownloadResult(
            path=downloaded, published_at=None
        ),
    )
    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.app.extract_video", _fake_extract_video_success)

    result = CliRunner().invoke(
        app,
        [
            "extract",
            "https://youtu.be/abc123",
            "--download-dir",
            str(tmp_path),
            "--keep-video",
            *_EXTRACT_ARGS,
        ],
    )

    assert result.exit_code == 0, result.output
    assert downloaded.exists()


def test_extract_keeps_downloaded_video_on_failure_with_keep_video_flag_without_prompting(
    tmp_path, monkeypatch
):
    downloaded = tmp_path / "abc123.mp4"
    downloaded.write_bytes(b"fake video")

    monkeypatch.setattr(
        "scadustats.cli.app.download_video",
        lambda url, output_dir, cookies=None: DownloadResult(
            path=downloaded, published_at=None
        ),
    )
    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("scadustats.cli.app.extract_video", _boom)

    # No input given -- if this were prompted, CliRunner would raise on EOF, so a clean
    # failure here proves --keep-video skips the confirm entirely.
    result = CliRunner().invoke(
        app,
        [
            "extract",
            "https://youtu.be/abc123",
            "--download-dir",
            str(tmp_path),
            "--keep-video",
            *_EXTRACT_ARGS,
        ],
    )

    assert result.exit_code != 0
    assert downloaded.exists()


def test_extract_does_not_download_a_local_video_path(tmp_path, monkeypatch):
    local = tmp_path / "local.mp4"
    local.write_bytes(b"fake video")

    def _fail_download(*args, **kwargs):
        raise AssertionError("download_video should not be called for a local path")

    monkeypatch.setattr("scadustats.cli.app.download_video", _fail_download)
    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.app.extract_video", _fake_extract_video_success)

    result = CliRunner().invoke(app, ["extract", str(local), *_EXTRACT_ARGS])

    assert result.exit_code == 0, result.output
    # A local video is never deleted, whatever happens -- only ones we downloaded are.
    assert local.exists()


def test_extract_deletes_downloaded_video_when_extraction_fails_and_confirmed(
    tmp_path, monkeypatch
):
    downloaded = tmp_path / "abc123.mp4"
    downloaded.write_bytes(b"fake video")

    monkeypatch.setattr(
        "scadustats.cli.app.download_video",
        lambda url, output_dir, cookies=None: DownloadResult(
            path=downloaded, published_at=None
        ),
    )
    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("scadustats.cli.app.extract_video", _boom)

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

    monkeypatch.setattr(
        "scadustats.cli.app.download_video",
        lambda url, output_dir, cookies=None: DownloadResult(
            path=downloaded, published_at=None
        ),
    )
    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("scadustats.cli.app.extract_video", _boom)

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
        return ExtractionSummary(
        video_id="v1", num_games=1, num_claims=0, extraction=_sample_extraction(video_id="v1")
    )

    monkeypatch.setattr(
        "scadustats.cli.app.download_video",
        lambda url, output_dir, cookies=None: DownloadResult(
            path=downloaded, published_at=None
        ),
    )
    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.app.extract_video", _fake_extract_video)

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


def test_extract_offers_existing_match_details_as_defaults_for_the_same_local_path(
    tmp_path, monkeypatch
):
    """Re-extracting the same local file a match was already extracted from (e.g. after
    a calibration fix) offers that earlier extraction's date/season/video_url as prompt
    defaults instead of asking from scratch -- see _find_existing_match."""
    local = tmp_path / "local.mp4"
    local.write_bytes(b"fake video")
    json_dir = tmp_path / "matches"
    write_video(
        json_dir,
        _sample_extraction(
            video_url="https://youtu.be/abc123",
            source_path=str(local),
            match_date=datetime.date(2026, 1, 1),
            season="Off-Season Cup",
        ),
    )
    captured: dict = {}

    def _fake_extract_video(video_path, *, match_metadata, **kwargs):
        captured["match_metadata"] = match_metadata.result()
        return ExtractionSummary(
        video_id="v1", num_games=1, num_claims=0, extraction=_sample_extraction(video_id="v1")
    )

    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.app.extract_video", _fake_extract_video)

    result = CliRunner().invoke(
        app,
        [
            "extract",
            str(local),
            "--json-dir",
            str(json_dir),
            "--match-type",
            "round_robin",
        ],
        # Blank lines accept whatever default each prompt offers -- match date, season,
        # video URL, in that order.
        input="\n\n\n",
    )

    assert result.exit_code == 0, result.output
    metadata = captured["match_metadata"]
    assert metadata.match_date == datetime.date(2026, 1, 1)
    assert metadata.season == "Off-Season Cup"
    assert metadata.video_url == "https://youtu.be/abc123"


def test_extract_passes_the_downloaded_published_date_through_to_extract_video(
    tmp_path, monkeypatch
):
    downloaded = tmp_path / "abc123.mp4"
    downloaded.write_bytes(b"fake video")
    captured: dict = {}

    def _fake_extract_video(video_path, *, published_at, **kwargs):
        captured["published_at"] = published_at
        return ExtractionSummary(
        video_id="v1", num_games=1, num_claims=0, extraction=_sample_extraction(video_id="v1")
    )

    monkeypatch.setattr(
        "scadustats.cli.app.download_video",
        lambda url, output_dir, cookies=None: DownloadResult(
            path=downloaded, published_at=datetime.date(2026, 2, 20)
        ),
    )
    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.app.extract_video", _fake_extract_video)

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
    assert captured["published_at"] == datetime.date(2026, 2, 20)


def test_extract_leaves_published_date_unset_for_a_local_video(tmp_path, monkeypatch):
    """A locally-supplied video was never downloaded by this run, so there's no yt-dlp
    metadata to read a published date from -- even when --video-url is also given."""
    local = tmp_path / "local.mp4"
    local.write_bytes(b"fake video")
    captured: dict = {}

    def _fake_extract_video(video_path, *, published_at, **kwargs):
        captured["published_at"] = published_at
        return ExtractionSummary(
        video_id="v1", num_games=1, num_claims=0, extraction=_sample_extraction(video_id="v1")
    )

    def _fail_download(*args, **kwargs):
        raise AssertionError("download_video should not be called for a local path")

    monkeypatch.setattr("scadustats.cli.app.download_video", _fail_download)
    monkeypatch.setattr("scadustats.cli.app.estimate_sample_count", lambda path: 1)
    monkeypatch.setattr("scadustats.cli.app.extract_video", _fake_extract_video)

    result = CliRunner().invoke(
        app,
        ["extract", str(local), *_EXTRACT_ARGS, "--video-url", "https://youtu.be/abc123"],
    )

    assert result.exit_code == 0, result.output
    assert captured["published_at"] is None


def _match_sample_game(game_index: int = 1, **overrides) -> GameResult:
    square_texts = [[f"goal {r}-{c}" for c in range(5)] for r in range(5)]
    defaults = dict(
        game_index=game_index,
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        square_texts=square_texts,
        events=[
            GameEvent(row=1, col=1, color=CellColor.RED, video_ts_s=10.0, game_elapsed_s=9),
            GameEvent(
                row=2,
                col=2,
                color=CellColor.BLUE,
                video_ts_s=20.0,
                game_elapsed_s=19,
                event_type=EventType.UNMARK,
            ),
        ],
        winner_color=CellColor.RED,
        win_type=WinType.LINE,
        win_line=WinLine.ROW_2,
        game_type=GameType.BASE,
    )
    defaults.update(overrides)
    return GameResult(**defaults)


def _sample_extraction(**overrides) -> VideoExtraction:
    defaults = dict(
        video_id="2026-03-05-alice-vs-bob",
        video_url="https://youtu.be/abc123",
        match_date=datetime.date(2026, 3, 5),
        season="6",
        match_type=MatchType.ROUND_ROBIN,
        player_red_name="alice",
        player_blue_name="bob",
        extracted_at=datetime.date(2026, 3, 6),
        games=[_match_sample_game()],
        duration_s=4321.0,
    )
    defaults.update(overrides)
    return VideoExtraction(**defaults)


def test_find_existing_match_matches_a_youtube_link_by_video_url(tmp_path):
    write_video(tmp_path, _sample_extraction(video_url="https://youtu.be/abc123"))

    found = _find_existing_match(tmp_path, "https://youtu.be/abc123")

    assert found is not None
    assert found.video_id == "2026-03-05-alice-vs-bob"


def test_find_existing_match_matches_a_local_path_by_source_path(tmp_path):
    write_video(
        tmp_path, _sample_extraction(video_url=None, source_path="downloads/local.mp4")
    )

    found = _find_existing_match(tmp_path, "downloads/local.mp4")

    assert found is not None
    assert found.video_id == "2026-03-05-alice-vs-bob"


def test_find_existing_match_does_not_match_a_local_path_against_video_url(tmp_path):
    """A local path is only ever matched against source_path -- a match whose video_url
    happens to equal the given string (e.g. by coincidence in a test) shouldn't count."""
    write_video(
        tmp_path, _sample_extraction(video_url="downloads/local.mp4", source_path=None)
    )

    assert _find_existing_match(tmp_path, "downloads/local.mp4") is None


def test_find_existing_match_returns_none_when_nothing_matches(tmp_path):
    write_video(tmp_path, _sample_extraction())

    assert _find_existing_match(tmp_path, "https://youtu.be/other") is None
    assert _find_existing_match(tmp_path, "downloads/nonexistent.mp4") is None


def test_find_existing_match_returns_none_for_an_empty_directory(tmp_path):
    assert _find_existing_match(tmp_path, "https://youtu.be/abc123") is None


def test_list_matches_prints_a_row_per_video_under_a_header(tmp_path):
    write_video(tmp_path, _sample_extraction())
    write_video(tmp_path, _sample_extraction(video_id="2026-01-01-carol-vs-dave"))

    result = CliRunner().invoke(app, ["match", "list", "--json-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("MATCH")
    # Filenames (== video_id) sort chronologically -- the earlier date comes first.
    assert lines[1].startswith("2026-01-01-carol-vs-dave")
    assert lines[2].startswith("2026-03-05-alice-vs-bob")
    # The result of the match is the column the old listing didn't have at all.
    assert lines[2].endswith("alice wins 1-0")
    assert "2 matches" in result.output


def test_list_matches_aligns_its_columns(tmp_path):
    write_video(tmp_path, _sample_extraction())
    write_video(tmp_path, _sample_extraction(video_id="2026-01-01-a-vs-b"))

    result = CliRunner().invoke(app, ["match", "list", "--json-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    header, *rows = result.output.splitlines()[:3]
    season_column = header.index("SEASON")
    assert all(row[season_column:].startswith("6") for row in rows)


def test_list_matches_skips_unparseable_files_with_a_warning(tmp_path):
    write_video(tmp_path, _sample_extraction())
    (tmp_path / "old-format.json").write_text('{"game_id": "x"}')

    result = CliRunner().invoke(app, ["match", "list", "--json-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "2026-03-05-alice-vs-bob" in result.output
    assert "Skipping old-format.json" in result.output


def test_list_matches_reports_when_directory_has_no_matches(tmp_path):
    result = CliRunner().invoke(app, ["match", "list", "--json-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "No matches found" in result.output


def test_show_match_prints_metadata_and_per_game_breakdown(tmp_path):
    write_video(tmp_path, _sample_extraction())

    result = CliRunner().invoke(
        app, ["match", "show", "2026-03-05-alice-vs-bob", "--json-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "2026-03-05-alice-vs-bob" in result.output
    assert "alice (red) vs bob (blue) -- alice wins 1-0" in result.output
    assert "Game 1  base game" in result.output
    # A line win names the line it was won on; see test_show_match_names_no_line... below
    # for the other case.
    assert "alice (red) by line on row 2" in result.output
    assert "1 mark, 1 unmark" in result.output
    assert "length 01:12:01" in result.output


def test_show_match_draws_the_final_board_with_the_winning_line_marked(tmp_path):
    # A genuine row-1 line win, so the board actually holds the line it's shown on.
    game = _match_sample_game(
        events=[
            GameEvent(
                row=1,
                col=col + 1,
                color=CellColor.RED,
                video_ts_s=1.0 + col,
                game_elapsed_s=col,
            )
            for col in range(5)
        ],
        win_line=WinLine.ROW_1,
    )
    write_video(tmp_path, _sample_extraction(games=[game]))

    result = CliRunner().invoke(
        app, ["match", "show", "2026-03-05-alice-vs-bob", "--json-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "board key: 🟥 = alice, 🟦 = bob" in result.output
    # The won row prints as five red squares; the four rows nobody touched as unclaimed.
    assert "🟥 🟥 🟥 🟥 🟥" in result.output
    assert result.output.count("⬛ ⬛ ⬛ ⬛ ⬛") == 4
    assert "squares  alice 5, bob 0, unclaimed 20" in result.output


def test_show_match_reports_a_draw_and_an_undetermined_result(tmp_path):
    drawn = [_match_sample_game(1), _match_sample_game(2, winner_color=CellColor.BLUE)]
    write_video(tmp_path, _sample_extraction(games=drawn))

    result = CliRunner().invoke(
        app, ["match", "show", "2026-03-05-alice-vs-bob", "--json-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "draw 1-1" in result.output

    undecided = [_match_sample_game(1), _match_sample_game(2, winner_color=None)]
    write_video(tmp_path, _sample_extraction(games=undecided))

    result = CliRunner().invoke(
        app, ["match", "show", "2026-03-05-alice-vs-bob", "--json-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    # "leads", not "wins": one game has no winner, so the score isn't the whole story.
    assert "alice leads 1-0, 1 undecided" in result.output


def test_show_match_names_no_line_for_a_majority_win(tmp_path):
    game = _match_sample_game(win_type=WinType.MAJORITY, win_line=None)
    write_video(tmp_path, _sample_extraction(games=[game]))

    result = CliRunner().invoke(
        app, ["match", "show", "2026-03-05-alice-vs-bob", "--json-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "alice (red) by majority" in result.output


def test_show_match_lists_events_only_when_asked(tmp_path):
    write_video(tmp_path, _sample_extraction())

    without = CliRunner().invoke(
        app, ["match", "show", "2026-03-05-alice-vs-bob", "--json-dir", str(tmp_path)]
    )
    with_events = CliRunner().invoke(
        app,
        ["match", "show", "2026-03-05-alice-vs-bob", "--events", "--json-dir", str(tmp_path)],
    )

    assert without.exit_code == 0, without.output
    assert "goal 0-0" not in without.output
    assert with_events.exit_code == 0, with_events.output
    # The mark's own row: its video timestamp, game clock, player, square and goal text.
    assert "00:00:10  00:09  mark    alice   r1 c1   goal 0-0" in with_events.output
    assert "unmark  bob     r2 c2   goal 1-1" in with_events.output


def test_show_match_headers_are_colorized_when_color_is_supported(tmp_path):
    """CliRunner strips ANSI codes by default (see match_validate's docstring), so this
    forces color on to confirm the section headers (video_id, each "Game N" line, and
    the events header) actually emit bold codes for a real terminal."""
    write_video(tmp_path, _sample_extraction())

    result = CliRunner().invoke(
        app,
        ["match", "show", "2026-03-05-alice-vs-bob", "--events", "--json-dir", str(tmp_path)],
        color=True,
    )

    assert "\x1b[1m2026-03-05-alice-vs-bob\x1b[0m" in result.output
    assert "\x1b[1mGame 1  base game" in result.output
    assert "\x1b[1mevent log\x1b[0m" in result.output


def test_show_match_reports_a_clean_match_as_valid(tmp_path):
    write_video(tmp_path, _valid_match_extraction())

    result = CliRunner().invoke(
        app, ["match", "show", "2026-03-05-alice-vs-bob", "--json-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "validation\n" in result.output
    assert "2026-03-05-alice-vs-bob: ok" in result.output


def test_show_match_lists_validation_issues_for_an_invalid_match(tmp_path):
    write_video(tmp_path, _sample_extraction())

    result = CliRunner().invoke(
        app, ["match", "show", "2026-03-05-alice-vs-bob", "--json-dir", str(tmp_path)]
    )

    # Unlike `match validate`, a bad result here isn't a CLI failure -- `show` always
    # prints the report, the validation section just says whether it holds up.
    assert result.exit_code == 0, result.output
    assert "[game_count]" in result.output
    assert "[winner_board_mismatch]" in result.output


def _valid_match_extraction(**overrides) -> VideoExtraction:
    """A match that passes every validation rule: round robin, two games (base then DLC),
    each with exactly one game_start and a genuine line win. The shared
    `_sample_extraction` above is deliberately *not* valid (one game, a recorded winner
    with no line behind it), which is what the failure cases below reuse."""

    def game(game_index: int, color: CellColor, game_type: GameType) -> GameResult:
        return _match_sample_game(
            game_index,
            events=[
                GameEvent(
                    row=None,
                    col=None,
                    color=None,
                    video_ts_s=0.0,
                    game_elapsed_s=0,
                    event_type=EventType.GAME_START,
                ),
                *(
                    GameEvent(
                        row=1,
                        col=col + 1,
                        color=color,
                        video_ts_s=1.0 + col,
                        game_elapsed_s=col,
                    )
                    for col in range(5)
                ),
            ],
            winner_color=color,
            win_type=WinType.LINE,
            win_line=WinLine.ROW_1,
            game_type=game_type,
        )

    overrides.setdefault(
        "games",
        [game(1, CellColor.RED, GameType.BASE), game(2, CellColor.BLUE, GameType.DLC)],
    )
    return _sample_extraction(**overrides)


def test_validate_match_reports_a_clean_match_as_ok(tmp_path):
    write_video(tmp_path, _valid_match_extraction())

    result = CliRunner().invoke(
        app, ["match", "validate", "2026-03-05-alice-vs-bob", "--json-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "2026-03-05-alice-vs-bob: ok" in result.output


def test_validate_match_lists_issues_and_exits_nonzero(tmp_path):
    write_video(tmp_path, _sample_extraction())

    result = CliRunner().invoke(
        app, ["match", "validate", "2026-03-05-alice-vs-bob", "--json-dir", str(tmp_path)]
    )

    assert result.exit_code == 1
    # A round robin match with one game, whose single recorded line win has no line behind
    # it -- one match-level and one game-level rule, each naming its own scope.
    assert "[game_count]" in result.output
    assert "game 1: " in result.output
    assert "[winner_board_mismatch]" in result.output


def test_validate_output_is_colorized_when_color_is_supported(tmp_path):
    """CliRunner strips ANSI codes by default, matching a non-terminal pipe (see
    match_validate's docstring) -- every other test here reads that stripped text, so this
    one forces color on to confirm the codes are actually emitted for a real terminal, not
    just that the plain text still reads correctly once they're gone."""
    write_video(tmp_path, _valid_match_extraction())
    write_video(tmp_path, _sample_extraction(video_id="2026-01-01-carol-vs-dave"))

    result = CliRunner().invoke(
        app, ["match", "validate", "--json-dir", str(tmp_path)], color=True
    )

    assert "\x1b[32m" in result.output  # green -- the clean match's "ok"
    assert "\x1b[31m" in result.output  # red -- the other match's issue count


def test_validate_checks_every_match_in_the_directory_by_default(tmp_path):
    write_video(tmp_path, _valid_match_extraction())
    write_video(tmp_path, _sample_extraction(video_id="2026-01-01-carol-vs-dave"))

    result = CliRunner().invoke(app, ["match", "validate", "--json-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "2026-03-05-alice-vs-bob: ok" in result.output
    assert "2026-01-01-carol-vs-dave: " in result.output
    assert "1 of 2 matches have issues" in result.output


def test_validate_skips_unparseable_files_with_a_warning(tmp_path):
    write_video(tmp_path, _valid_match_extraction())
    (tmp_path / "old-format.json").write_text('{"game_id": "x"}')

    result = CliRunner().invoke(app, ["match", "validate", "--json-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Skipping old-format.json" in result.output
    assert "2026-03-05-alice-vs-bob: ok" in result.output


def test_validate_reports_when_directory_has_no_matches(tmp_path):
    result = CliRunner().invoke(app, ["match", "validate", "--json-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "No matches found" in result.output


def test_validate_errors_on_unknown_video_id(tmp_path):
    result = CliRunner().invoke(
        app, ["match", "validate", "nonexistent", "--json-dir", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert "nonexistent" in result.output


def test_show_match_errors_on_unknown_video_id(tmp_path):
    result = CliRunner().invoke(
        app, ["match", "show", "nonexistent", "--json-dir", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert "nonexistent" in result.output


def test_no_command_prints_the_command_list():
    """A bare `scadustats` should show what the commands are, not just Typer's default
    "Missing command" error, which names none of them."""
    result = CliRunner().invoke(app, [])

    assert "Commands" in result.output
    for command in ("download", "extract", "load-db", "match"):
        assert command in result.output
    assert "Missing command" not in result.output


def test_no_subcommand_prints_the_group_command_list():
    """Same for a bare sub-command group -- `scadustats match` lists its own commands."""
    result = CliRunner().invoke(app, ["match"])

    assert "Commands" in result.output
    for command in ("list", "show", "validate"):
        assert command in result.output
    assert "Missing command" not in result.output
