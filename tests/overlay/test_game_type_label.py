import cv2
import pytest

from scadustats.models import GameType
from scadustats.overlay.game_type_label import (
    majority_game_type_label,
    parse_game_type_label,
    read_game_type_label,
)

# 5s clips trimmed from a real match -- one from its base-game half, one from its DLC
# half -- with the webcam regions blacked out (see scripts/redact_fixture.py). Unlike the
# older, more heavily recompressed fixtures, these retain enough quality for the small
# "BASE GAME"/"DLC" subtitle to actually OCR, so they're what covers the real read path.
_BASE_CLIP = "tests/fixtures/clip_base_game.mp4"
_DLC_CLIP = "tests/fixtures/clip_dlc_game.mp4"


def _frames(path: str) -> list:
    cap = cv2.VideoCapture(path)
    try:
        result = []
        while True:
            ok, frame = cap.read()
            if not ok:
                return result
            result.append(frame)
    finally:
        cap.release()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("BASE GAME", GameType.BASE),
        ("base game", GameType.BASE),
        ("  Base Game  ", GameType.BASE),
        ("DLC", GameType.DLC),
        ("dlc", GameType.DLC),
        ("DACE RAME", None),  # a real misread this box has produced on low-bitrate footage
        ("", None),
        ("GAME 1", None),
    ],
)
def test_parse_game_type_label(text, expected):
    assert parse_game_type_label(text) == expected


def test_majority_game_type_label_returns_none_when_no_frame_votes(monkeypatch):
    monkeypatch.setattr(
        "scadustats.overlay.game_type_label.read_game_type_label", lambda frame: None
    )
    assert majority_game_type_label([object(), object()]) is None


def test_majority_game_type_label_picks_the_most_common_vote(monkeypatch):
    votes = iter([GameType.BASE, GameType.DLC, GameType.BASE])
    monkeypatch.setattr(
        "scadustats.overlay.game_type_label.read_game_type_label", lambda frame: next(votes)
    )
    assert majority_game_type_label([object(), object(), object()]) is GameType.BASE


@pytest.mark.parametrize(
    ("clip", "expected"),
    [(_BASE_CLIP, GameType.BASE), (_DLC_CLIP, GameType.DLC)],
)
def test_read_game_type_label_reads_every_frame_of_a_real_clip(clip, expected):
    """The subtitle is static for a whole game, so every frame should read the same --
    this is what would catch GAME_TYPE_BOX drifting out of calibration."""
    frames = _frames(clip)
    assert frames, f"no frames decoded from {clip}"
    assert [read_game_type_label(frame) for frame in frames] == [expected] * len(frames)


@pytest.mark.parametrize(
    ("clip", "expected"),
    [(_BASE_CLIP, GameType.BASE), (_DLC_CLIP, GameType.DLC)],
)
def test_majority_game_type_label_on_a_real_clip(clip, expected):
    assert majority_game_type_label(_frames(clip)) is expected
