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
        ("02:59:59", 10799),
        # Season 1's clock has a one-digit hour.
        ("0:38:25", 2305),
        ("1:01:47", 3707),
        # Readings no game clock shows: a cursor over the first digit, impossible minutes.
        ("60:00:42", None),
        ("03:00:00", None),
        ("00:61:00", None),
        ("00:00:75", None),
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
