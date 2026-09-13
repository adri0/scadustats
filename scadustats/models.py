"""Shared types used across the extraction pipeline."""

from dataclasses import dataclass, field
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


class GameType(StrEnum):
    """Whether a game's goal squares are drawn from the base game's pool or the Shadow
    of the Erdtree DLC's -- read per game directly off the overlay's own "BASE GAME"/
    "DLC" subtitle where that OCRs cleanly (see game_type_label.py), falling back to
    inferring it from square texts (see squares.py) and finally to a user-provided
    answer. StrEnum for the same reason as MatchType: it doubles as a Typer/Click CLI
    choice type."""

    BASE = "base"
    DLC = "dlc"


class WinType(Enum):
    LINE = "line"
    MAJORITY = "majority"
    TIE = "tie"
    NONE = "none"


class WinLine(StrEnum):
    """Which of the board's 12 lines a LINE win was completed on (see winner.LINES).

    Row/column indices are 0-based, matching GameEvent.row/col and the `squares` DB
    table: a JSON file shows `"win_line": "row_0"` a few lines from a claim's `"row": 0`,
    and a prettier 1-based label would line up with nothing else in the file. StrEnum
    like the other persisted enums, so it serializes as its own value.
    """

    ROW_0 = "row_0"
    ROW_1 = "row_1"
    ROW_2 = "row_2"
    ROW_3 = "row_3"
    ROW_4 = "row_4"
    COL_0 = "col_0"
    COL_1 = "col_1"
    COL_2 = "col_2"
    COL_3 = "col_3"
    COL_4 = "col_4"
    DIAGONAL_TL_BR = "diagonal_tl_br"
    DIAGONAL_BL_TR = "diagonal_bl_tr"

    @property
    def label(self) -> str:
        """Human-facing description, for CLI output and validation messages -- the stored
        value stays the machine-readable one."""
        if self is WinLine.DIAGONAL_TL_BR:
            return "diagonal (top-left to bottom-right)"
        if self is WinLine.DIAGONAL_BL_TR:
            return "diagonal (bottom-left to top-right)"
        kind, index = self.value.split("_")
        return f"{'row' if kind == 'row' else 'column'} {index}"


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
    # A game is identified by its position within the video. The overlay prints a
    # "GAME N" label too, but it isn't read: N is game_index, and OCR of that box was
    # noise in practice (a stable "GAME 2" came back as a different garbled string on
    # nearly every sample).
    game_index: int
    start_video_ts_s: float
    end_video_ts_s: float | None
    square_texts: list[list[str]]
    events: list[GameEvent]
    winner_color: CellColor | None
    win_type: WinType
    # Which line the win was completed on -- None for any non-LINE win (there's no line
    # to name), and for a JSON file written before this was recorded.
    win_line: WinLine | None = None
    # None until it's resolved from the overlay's "BASE GAME"/"DLC" subtitle
    # (game_type_label.py) or, failing that, inferred from square_texts (squares.py).
    # Both can fail (a transient/low-quality frame for the former, no matching squares
    # yet in the reference for the latter), in which case the caller must supply one (see
    # extract_video's on_missing_game_type).
    game_type: GameType | None = None


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
    # The commentators casting this match, read off the broadcast's own nameplates
    # (commentators.py), left seat to right. A flat list rather than named seats: which
    # webcam a commentator sat in isn't a property of the match. Empty when neither plate
    # read as a name -- including for a JSON file written before this was recorded, and
    # for footage whose nameplate regions are blacked out (see the test fixtures).
    commentators: list[str] = field(default_factory=list)
    # The source video's full length in seconds (frames.probe), covering the whole
    # broadcast -- intros, between-game recaps and all -- not just the segments that
    # became games. Optional/defaulted since a JSON file written before this field
    # existed (or hand-edited to drop it) still has to read back cleanly, and since
    # a video whose fps can't be read has no duration to record.
    duration_s: float | None = None
