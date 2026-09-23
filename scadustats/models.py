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
    above) so it doubles directly as a Typer/Click CLI choice type -- see cli/app.py."""

    DOUBLE_ELIMINATION = "double_elimination"
    ROUND_ROBIN = "round_robin"


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


class MatchWinner(StrEnum):
    """Who took the *match* (as opposed to WinType/CellColor, which are about one game).

    DRAW is a real outcome here, unlike in a single game: a round robin match is two games
    and is allowed to end 1-1. Its own enum rather than reusing CellColor for that reason
    -- and StrEnum like the other persisted enums, so it serializes as its own value.
    """

    RED = "red"
    BLUE = "blue"
    DRAW = "draw"


class WinLine(StrEnum):
    """Which of the board's 12 lines a LINE win was completed on (see winner.LINES).

    Row/column indices are 1-based, matching GameEvent.row/col and the `squares` DB
    table: a JSON file shows `"win_line": "row_1"` a few lines from a claim's `"row": 1`,
    and a 0-based value here would line up with nothing else in the file -- a contributor
    hand-reviewing the JSON otherwise has no reason to expect a square's row/col to be
    off by one from what the board visually looks like. StrEnum like the other persisted
    enums, so it serializes as its own value.
    """

    ROW_1 = "row_1"
    ROW_2 = "row_2"
    ROW_3 = "row_3"
    ROW_4 = "row_4"
    ROW_5 = "row_5"
    COL_1 = "col_1"
    COL_2 = "col_2"
    COL_3 = "col_3"
    COL_4 = "col_4"
    COL_5 = "col_5"
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
    one-time transition. Not every event is about a square, though: GAME_START and
    GAME_END mark whole-game moments and so carry no row/col/color. GAME_START is the
    stopwatch turning from the pre-game countdown into the ascending game clock.
    GAME_END is the moment the game's recorded result (a completed line, or a majority
    once it's mathematically unbeatable -- see rules.winner.determine_winner) first
    became locked in for the rest of the game, timestamped at the settling MARK event
    itself (see extract._detect_game_end) -- a TIE or an undetermined game has no
    line/majority to settle on, so those have no GAME_END event at all."""

    MARK = "mark"
    UNMARK = "unmark"
    GAME_START = "game_start"
    GAME_END = "game_end"


class FractionalBox(NamedTuple):
    """A bounding box expressed as fractions (0-1) of frame width/height."""

    left: float
    top: float
    right: float
    bottom: float


def format_video_timestamp(seconds: float) -> str:
    """Render a raw video-offset second count as `hh:mm:ss.ms` for GameEvent.video_timestamp
    -- the format a contributor hand-reviewing the JSON reads directly, rather than a raw
    float they'd have to convert in their head (see CLAUDE.md and issue #109). Milliseconds
    are only appended when the value doesn't fall exactly on a whole second -- the debounced
    ~1 sample/second events this is built from routinely do, and a trailing `.000` on every
    one of them would be pure noise.
    """
    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, ms = divmod(remainder_ms, 1000)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{base}.{ms:03d}" if ms else base


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_s: float


@dataclass
class GameEvent:
    # 1-based (1-5), matching the board as a reviewer sees it and WinLine's own values --
    # None for a game-level event (GAME_START/GAME_END) that isn't about any one square.
    row: int | None
    col: int | None
    # for an UNMARK, the color that was removed; None for GAME_START/GAME_END
    color: CellColor | None
    # `hh:mm:ss.ms` (see format_video_timestamp), milliseconds omitted when the reading
    # falls exactly on a whole second -- e.g. "00:12:34.500" or "00:12:34". A plain string
    # rather than a raw float: it's assigned once, at event-construction time from the raw
    # sample timestamp (see extract.py), and every later use (sorting, JSON/DB storage,
    # CLI display) either reads it as-is or -- for sorting -- relies on the fixed-width
    # hh:mm:ss prefix (and the fact that a present-vs-absent `.ms` suffix only ever makes
    # the shorter, no-milliseconds string sort first, exactly where "no fractional second"
    # belongs) to keep chronological and lexicographic order the same, so no float version
    # needs to survive alongside it.
    video_timestamp: str
    game_elapsed_s: int
    event_type: EventType = EventType.MARK


@dataclass
class MatchMetadata:
    """User-supplied details about a match that can't be read from the video itself --
    collected interactively by the CLI (see cli/app.py) and attached once per video, since a
    match is one video even when it contains multiple game segments."""

    match_date: date
    # Free text, not a number: seasons aren't always named with a plain integer (e.g. a
    # one-off "Off-Season Cup"), and nothing here needs to do arithmetic on it -- see
    # cli.app._prompt_match_metadata for the numbered-season default most matches use.
    season: str
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
class Square:
    """One goal square in the consolidated, per-game-type reference built by
    pipeline.consolidate from previously extracted matches (see
    storage.json_export.write_squares) -- distinct from GameResult.square_texts, which is
    just the raw OCR text read off one game's board, with no id or cross-match dedup.
    """

    id: str
    text: str
    game_type: GameType


@dataclass
class VideoExtraction:
    """Everything extracted from one video, and the unit `json_export`/`db` persist:
    one video is one match, and a match can contain several games (GameResult), but
    player names/match metadata are read/collected once per video, not once per game --
    see extract._match_id and CLAUDE.md."""

    match_id: str
    video_url: str | None
    match_date: date
    season: str
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
    # When the video was uploaded to YouTube (yt-dlp's `upload_date`), as opposed to
    # match_date (when the match was played, user-supplied) or extracted_at (when this
    # extraction ran) -- distinct from both, and useful for e.g. sanity-checking
    # match_date against it. Read off the same yt-dlp call that downloaded the video (see
    # video.download.DownloadResult), so it's only known when this run did the
    # downloading -- a locally-supplied video has no metadata to read it from, and a JSON
    # file written before this field existed still has to read back cleanly.
    published_at: date | None = None

    @property
    def red_score(self) -> int:
        """Games the red player won in this match. Derived from `games`, like num_games."""
        return sum(game.winner_color is CellColor.RED for game in self.games)

    @property
    def blue_score(self) -> int:
        """Games the blue player won in this match."""
        return sum(game.winner_color is CellColor.BLUE for game in self.games)

    @property
    def winner(self) -> MatchWinner | None:
        """Who took the match, by game score -- or None when that can't be said.

        None ("undetermined") rather than a guess in two cases: a match with no games at
        all, and one where any game has no winner recorded. In the latter, the scores
        below don't account for every game, so calling the current leader the winner --
        or an even split a draw -- would be inventing an outcome from incomplete data.
        That's exactly what validation's `match_outcome_undetermined` rule reports, and a
        reviewer should fix the extraction rather than read a number here.
        """
        if not self.games or any(game.winner_color is None for game in self.games):
            return None
        if self.red_score > self.blue_score:
            return MatchWinner.RED
        if self.blue_score > self.red_score:
            return MatchWinner.BLUE
        # Only legal in round robin (validation's double_elimination_draw rule catches the
        # other format) -- but what the games *say* happened is reported either way.
        return MatchWinner.DRAW

    @property
    def num_games(self) -> int:
        """How many games the match consists of. Derived from `games` rather than stored
        alongside it, so the two can't disagree -- it's persisted (JSON `num_games`, DB
        `videos.num_games`) for a reader's/query's convenience, but a hand-edited file
        that adds or drops a game is re-counted on read, not believed."""
        return len(self.games)


@dataclass
class MatchRecord:
    """Wins, losses and draws for one season's worth of a player's matches (see
    PlayerProfile.season_records). A draw is a real *match* outcome -- a round robin
    match can end 1-1, see MatchWinner.DRAW -- unlike for a single game, where a tie can
    no longer even occur on a 5x5 board (see winner._majority_settled's note on
    WinType.TIE), so this carries a draws count that WinLoss (used for the game-level
    tallies below) doesn't need."""

    wins: int = 0
    losses: int = 0
    draws: int = 0


@dataclass
class WinLoss:
    """Wins and losses for one bucket of a player's individual games (see
    PlayerProfile.game_record/game_type_records) -- no draws field, since a single game
    can't end in one, unlike a match (see MatchRecord)."""

    wins: int = 0
    losses: int = 0


@dataclass
class SquareMarks:
    """One square text and how many times a player has personally marked it (see
    PlayerProfile.top_squares_base_game/top_squares_dlc) -- the count travels with the
    text rather than being left for a reader to re-derive, since it's the very thing that
    ranked this square into the top 5 in the first place."""

    text: str
    marks: int


@dataclass
class PlayerProfile:
    """One player's consolidated profile, built by pipeline.consolidate.consolidate_players
    from every match they appear in and persisted as one YAML file per player (see
    storage.player_export, issue #72).

    slug/display_name/season_records/game_record/game_type_records/all_matches/
    top_squares_base_game/top_squares_dlc are wholly regenerated from match history on
    every `player consolidate` run -- none of them is meant to be hand-edited and
    expected to survive a re-run.
    id/twitch/avatar/bio are the exception: id is assigned once, the first time a
    player is seen, and kept stable across every later re-run, and twitch/avatar/bio
    are never set by the tool at all -- they're filled in by hand and preserved the same
    way (see storage.player_export.read_players).
    """

    id: int
    # Lowercase, underscores as separator -- e.g. "twistiet" -- derived from display_name
    # and what two OCR'd readings of the same player's name are considered equal under
    # (see pipeline.consolidate.slugify_name). Also the file's own basename (see
    # storage.player_export.player_path).
    slug: str
    # The exact spelling seen most often across this player's matches -- see
    # consolidate_players.
    display_name: str
    # Just the handle (e.g. "twistiet"), not the full https://twitch.tv/... URL -- filled
    # in by hand, like avatar/bio below.
    twitch: str | None = None
    avatar: str | None = None
    bio: str | None = None
    season_records: dict[str, MatchRecord] = field(default_factory=dict)
    game_record: WinLoss = field(default_factory=WinLoss)
    game_type_records: dict[GameType, WinLoss] = field(default_factory=dict)
    # match_ids the player appears in, ordered by match_date descending (most recent
    # first) -- see consolidate_players.
    all_matches: list[str] = field(default_factory=list)
    # The 5 squares this player has personally marked most often, per game type, each
    # with its mark count -- see consolidate_players.
    top_squares_base_game: list[SquareMarks] = field(default_factory=list)
    top_squares_dlc: list[SquareMarks] = field(default_factory=list)
