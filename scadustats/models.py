"""Shared types used across the extraction pipeline."""

from dataclasses import dataclass
from datetime import date
from enum import Enum, StrEnum
from typing import NamedTuple


class CellColor(Enum):
    UNCLAIMED = "unclaimed"
    RED = "red"
    BLUE = "blue"


class MatchType(StrEnum):
    """User-provided, not derivable from the video. StrEnum (unlike the plain Enums
    above) so it doubles directly as a Typer/Click CLI choice type -- see cli.py."""

    DOUBLE_ELIMINATION = "double_elimination"
    PLAYOFFS = "playoffs"


class WinType(Enum):
    LINE = "line"
    MAJORITY = "majority"
    TIE = "tie"
    NONE = "none"


class EventType(Enum):
    """A square can be unmarked after being marked -- a player can inadvertently claim
    the wrong square and undo it -- so a claim's lifecycle is an event, not just a
    one-time transition. Not every event is about a square, though: GAME_START marks a
    whole-game moment (the stopwatch turning from the pre-game countdown into the
    ascending game clock) and so carries no row/col/color."""

    MARK = "mark"
    UNMARK = "unmark"
    GAME_START = "game_start"


class FractionalBox(NamedTuple):
    """A bounding box expressed as fractions (0-1) of frame width/height."""

    left: float
    top: float
    right: float
    bottom: float


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_s: float


@dataclass
class GameEvent:
    # None for a game-level event (GAME_START) that isn't about any one square.
    row: int | None
    col: int | None
    color: CellColor | None  # for an UNMARK, the color that was removed; None for GAME_START
    video_ts_s: float
    game_elapsed_s: int
    event_type: EventType = EventType.MARK


@dataclass
class MatchMetadata:
    """User-supplied details about a match that can't be read from the video itself --
    collected interactively by the CLI (see cli.py) and attached once per video, since a
    match is one video even when it contains multiple game segments."""

    match_date: date
    season: int
    match_type: MatchType
    # The video's URL, if extraction started from a local file rather than the URL
    # itself -- optional since a contributor may not have it handy at prompt time.
    video_url: str | None = None


@dataclass
class GameResult:
    game_index: int
    label: str | None
    start_video_ts_s: float
    end_video_ts_s: float | None
    square_texts: list[list[str]]
    events: list[GameEvent]
    winner_color: CellColor | None
    win_type: WinType


@dataclass
class VideoExtraction:
    """Everything extracted from one video, and the unit `json_export`/`db` persist:
    one video is one match, and a match can contain several games (GameResult), but
    player names/match metadata are read/collected once per video, not once per game --
    see extract._video_id and CLAUDE.md."""

    video_id: str
    video_url: str | None
    match_date: date
    season: int
    match_type: MatchType
    player_red_name: str | None
    player_blue_name: str | None
    extracted_at: date
    games: list[GameResult]
