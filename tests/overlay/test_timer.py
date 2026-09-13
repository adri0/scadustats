import cv2
import pytest

from scadustats.overlay.timer import parse_timer, read_timer

_FIXTURES = "tests/fixtures"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("00:01:49", 109),
        ("00:27:11", 1631),
        ("01:00:00", 3600),
        ("not a timer", None),
        ("", None),
        ("12:34", None),
    ],
)
def test_parse_timer(text, expected):
    assert parse_timer(text) == expected


def test_read_timer_start_fixture():
    frame = cv2.imread(f"{_FIXTURES}/hud_start.png")
    assert read_timer(frame) == 109


def test_read_timer_partial_fixture():
    frame = cv2.imread(f"{_FIXTURES}/hud_partial.png")
    assert read_timer(frame) == 1631
