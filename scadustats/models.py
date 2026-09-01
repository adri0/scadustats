"""Shared types used across the extraction pipeline."""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple


class CellColor(Enum):
    UNCLAIMED = "unclaimed"
    RED = "red"
    BLUE = "blue"


class WinType(Enum):
    LINE = "line"
    MAJORITY = "majority"
    TIE = "tie"
    NONE = "none"


class EventType(Enum):
    """A square can be unmarked after being marked -- a player can inadvertently claim
    the wrong square and undo it -- so a claim's lifecycle is an event, not just a
    one-time transition."""

    MARK = "mark"
    UNMARK = "unmark"


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
class ClaimEvent:
    row: int
    col: int
    color: CellColor  # for an UNMARK, the color that was removed
    video_ts_s: float
    game_elapsed_s: int
    event_type: EventType = EventType.MARK


@dataclass
class GameResult:
    game_index: int
    label: str | None
    start_video_ts_s: float
    end_video_ts_s: float | None
    player_red_name: str | None
    player_blue_name: str | None
    square_texts: list[list[str]]
    claims: list[ClaimEvent]
    winner_color: CellColor | None
    win_type: WinType
