import cv2

from scadustats.overlay import commentators
from scadustats.overlay.commentators import (
    _clean_name,
    majority_commentator_names,
    read_commentator_names,
)

_FIXTURES = "tests/fixtures"


def test_clean_name_drops_the_mic_icons_ocr_noise():
    # The plate's mic icon comes back as a stray one-character token ahead of the name,
    # and the plate's own border sometimes as a "|" after it.
    assert _clean_name("& starOchris") == "starOchris"
    assert _clean_name("| & Captain_Domo") == "Captain_Domo"
    assert _clean_name("& starOchris |") == "starOchris"


def test_clean_name_keeps_digits_and_underscores():
    # The whole reason this isn't scoreboard._clean_name: that one keeps only letters,
    # which would cut "star0chris" down to "chris" and "Captain_Domo" to "Captain".
    assert _clean_name("star0chris") == "star0chris"
    assert _clean_name("Captain_Domo") == "Captain_Domo"


def test_clean_name_returns_none_when_the_reading_holds_no_name():
    assert _clean_name("") is None
    assert _clean_name("& |") is None


def test_read_commentator_names_returns_none_for_a_blacked_out_region():
    """Every other committed fixture predates the nameplate boxes being restored (see
    scripts/redact_fixture.py) and so has this whole region zeroed -- read_commentator_names
    has to report "no plate here", not hallucinate a name off a blank crop (a fully black
    crop reads back from tesseract as the plausible-looking token "re" without the gate --
    see commentators._has_nameplate)."""
    frame = cv2.imread(f"{_FIXTURES}/hud_start.png")

    assert read_commentator_names(frame) == (None, None)


def test_read_commentator_names_reads_both_nameplates_from_a_real_frame():
    frame = cv2.imread(f"{_FIXTURES}/hud_commentators.png")

    # "starOchris" with a capital O, though the broadcaster's actual handle is
    # "star0chris" with a zero: the plate's font draws an unslashed zero, and Tesseract
    # reads it as an O inside a word. Upscaling doesn't fix it (see
    # read_commentator_names), and no whitelist can -- both characters are legal in a
    # handle. Asserting what's really read keeps this test honest about a known limit;
    # the JSON is hand-correctable, which is what that's for.
    assert read_commentator_names(frame) == ("starOchris", "Captain_Domo")


def test_majority_vote_picks_the_most_common_reading_per_seat(monkeypatch):
    readings = {
        1: ("star0chris", "Captain_Domo"),
        2: ("starOchris", "Captain_Domo"),
        3: ("star0chris", "Captain_Dom0"),
    }
    monkeypatch.setattr(commentators, "read_commentator_names", lambda frame: readings[frame])

    assert majority_commentator_names([1, 2, 3]) == ["star0chris", "Captain_Domo"]


def test_majority_vote_drops_a_seat_that_never_read_as_a_name(monkeypatch):
    """One unreadable plate yields a shorter list, not a gap -- position in the list
    isn't a seat (see models.VideoExtraction.commentators)."""
    monkeypatch.setattr(
        commentators, "read_commentator_names", lambda frame: (None, "Captain_Domo")
    )

    assert majority_commentator_names([1, 2]) == ["Captain_Domo"]


def test_majority_vote_returns_nothing_when_neither_plate_reads(monkeypatch):
    monkeypatch.setattr(commentators, "read_commentator_names", lambda frame: (None, None))

    assert majority_commentator_names([1, 2]) == []
