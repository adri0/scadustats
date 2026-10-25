"""Shared types used across the extraction pipeline.

The models that get persisted as JSON/YAML data files (everything from GameEvent down to
PlayerStats below) are pydantic BaseModels rather than plain dataclasses -- validation on
construction (an enum value that isn't one of its members, a date string that doesn't
parse, ...) catches a malformed hand-edit or a programming mistake at the point a model is
built, rather than letting a bad value silently ride along until something downstream
trips over it. Pydantic's own (de)serialization (`model_dump(mode="json")`/
`model_validate`) also replaces most of the hand-written dict-munging that used to live in
storage/json_export.py, storage/player_info.py and storage/player_stats.py: enum <->
string and date <-> ISO-string conversion, and Optional-field defaulting for a key an
older file doesn't have, all happen for free instead of being spelled out per field at
every read/write call site. Purely in-memory value objects that are never round-tripped
through a dict/JSON/YAML (VideoInfo, segmentation.Observation, validation.ValidationIssue,
consolidate.SquareValidationIssue/SquareTextChange, extract.ExtractionSummary,
video.download.DownloadResult) are deliberately left as plain dataclasses -- pydantic earns
its keep at a (de)serialization boundary, not on a value that only ever exists in Python.
"""

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum, StrEnum
from typing import NamedTuple

from pydantic import BaseModel, Field, field_validator, model_validator


class CellColor(Enum):
    UNCLAIMED = "unclaimed"
    RED = "red"
    BLUE = "blue"


class MatchType(StrEnum):
    """User-provided, not derivable from the video. StrEnum (unlike the plain Enums
    above) so it doubles directly as a Typer/Click CLI choice type -- see cli/app.py."""

    DOUBLE_ELIMINATION = "double_elimination"
    ROUND_ROBIN = "round_robin"


class LayoutName(StrEnum):
    """Identifies one registered overlay template (see overlay.layout.Layout/LAYOUTS) --
    a --layout CLI override or extract_video's own layout_name parameter names one of
    these rather than a bare string, for the same reason as MatchType: it doubles as a
    Typer/Click CLI choice type, so an unknown name is rejected by Typer itself rather
    than needing its own manual validation."""

    STANDARD = "standard"
    SEASON_6_FINAL = "layout_season_6_final"
    PRE_SEASON_6 = "layout_pre_season_6"
    SEASON_5 = "layout_season_5"
    SEASON_5_V2 = "layout_season_5_v2"
    SEASON_5_V3 = "layout_season_5_v3"
    SEASON_4 = "layout_season_4"
    SEASON_3 = "layout_season_3"
    SEASON_3_TEAM = "layout_season_3_team"
    SEASON_3_FOCUS = "layout_season_3_focus"


class GameType(StrEnum):
    """Whether a game's goal squares are drawn from the base game's pool or the Shadow
    of the Erdtree DLC's -- read per game directly off the overlay's own "BASE GAME"/
    "DLC" subtitle where that OCRs cleanly (see game_type_label.py), falling back to
    inferring it from square texts (see squares.py) and finally to a user-provided
    answer. StrEnum for the same reason as MatchType: it doubles as a Typer/Click CLI
    choice type.

    TWO_V_TWO is Season 5's 2v2 format: each team has one player in the base game and
    one in the DLC, and the board mixes base game, DLC and co-op squares -- the overlay
    tags each square "(B)", "(D)" or "(C)", and a mixed board's square texts keep that tag
    as the last thing in the text (see square_type_tag). A co-op square needs both
    teammates to complete a goal (one in each game) before it can be claimed. As a
    Square's game_type it names the co-op pool (see Square), since a base-game or DLC
    square on a mixed board belongs to the BASE/DLC pools like any other.
    """

    BASE = "base"
    DLC = "dlc"
    TWO_V_TWO = "2v2"


# The pool tag every square on a mixed board (GameType.TWO_V_TWO) ends in, exactly as
# the overlay prints it: " (B)" base game, " (D)" DLC, " (C)" co-op.
_SQUARE_TYPE_TAGS = {"B": GameType.BASE, "D": GameType.DLC, "C": GameType.TWO_V_TWO}
_SQUARE_TYPE_TAG_LETTERS = {game_type: letter for letter, game_type in _SQUARE_TYPE_TAGS.items()}
_SQUARE_TYPE_TAG_RE = re.compile(r" \(([BDC])\)$")


def square_type_tag(text: str) -> GameType | None:
    """The pool a mixed board's square text is tagged with -- "Kill Malenia (B)" is
    BASE -- or None when it doesn't end in a tag."""
    match = _SQUARE_TYPE_TAG_RE.search(text)
    return _SQUARE_TYPE_TAGS[match.group(1)] if match else None


def strip_square_type_tag(text: str) -> str:
    """A mixed board's square text without its pool tag -- the goal as the squares
    reference knows it ("Kill Malenia (B)" is the base game's "Kill Malenia")."""
    return _SQUARE_TYPE_TAG_RE.sub("", text)


def tag_square_text(text: str, game_type: GameType) -> str:
    """`text` ending in game_type's pool tag, replacing any it already has."""
    return f"{strip_square_type_tag(text)} ({_SQUARE_TYPE_TAG_LETTERS[game_type]})"


class WinType(Enum):
    LINE = "line"
    MAJORITY = "majority"
    # A points game (see Scoring.POINTS) won on points: first to 13, or ahead at time up.
    POINTS = "points"
    TIE = "tie"
    NONE = "none"


class Scoring(StrEnum):
    """How a game is decided. Every game is lockout -- a square one side claims can't be
    claimed by the other -- and that alone doesn't say who wins. LINE is the usual bingo:
    a completed line wins, and without one the most squares (see
    rules.winner.determine_winner). POINTS is Season 3, round robin and playoffs alike:
    each side scores one point per square plus two per completed line (see
    rules.winner.points), a line alone doesn't win, and the first side to 13 points wins
    (rules.winner.POINTS_TO_WIN). A game stopped by the clock before
    either side gets there goes to the higher total; equal totals are a TIE. Set per game
    at extraction from the season's format (see rules.validation.SEASON_FORMATS).
    StrEnum like the other persisted enums, so it serializes as its own value."""

    LINE = "line"
    POINTS = "points"


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
    line/majority to settle on, so those have no GAME_END event at all. The exception is
    a game stopped by a time limit before either settled: a GAME_END where the clock
    stopped, as the game's last event, makes the count at that point its result (see
    rules.winner.time_up). Extraction adds it when the game clock reaches two hours (see
    extract._detect_time_limit); a contributor can add one by hand for any other limit."""

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


class GameEvent(BaseModel):
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


class MatchMetadata(BaseModel):
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


class GameResult(BaseModel):
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
    # How the game is decided (see Scoring) -- LINE for every game but Season 3's round
    # robin, and for a JSON file written before this was recorded.
    scoring: Scoring = Scoring.LINE
    # A points game's points per side at its end (see rules.winner.points) --
    # None for a line game. Stored, not derived on read, so a reader (the site) doesn't
    # need the line rules to show them; validation checks them against the board.
    red_points: int | None = None
    blue_points: int | None = None

    def cell_goal(self, row: int, col: int) -> tuple[GameType | None, str]:
        """The square at 0-based (row, col): the pool it belongs to and its text as the
        squares reference knows it. On a mixed board (TWO_V_TWO) the pool is the text's
        own tag, which is stripped from the text (see square_type_tag); a cell with no tag
        has no pool. Every other game's cells are all in game_type's pool, text as is. A
        cell outside a ragged, hand-edited grid is (None, "")."""
        try:
            text = self.square_texts[row][col]
        except IndexError:
            return None, ""
        if self.game_type is GameType.TWO_V_TWO:
            return square_type_tag(text), strip_square_type_tag(text)
        return self.game_type, text


class Square(BaseModel):
    """One goal square in the consolidated, per-game-type reference built by
    pipeline.consolidate from previously extracted matches (see
    storage.json_export.write_squares) -- distinct from GameResult.square_texts, which is
    just the raw OCR text read off one game's board, with no id or cross-match dedup.

    `text` is the square's canonical wording. `aliases` are other wordings of the same
    goal seen on other broadcasts (e.g. "Kill Midra, Lord of Frenzied Flame" for "Kill
    Midra", issue #125) -- added by hand, never by consolidation, since only a person can
    tell a rewording from a genuinely different goal. A board cell reading any of them is
    this square: consolidation corrects OCR towards whichever wording it's closest to,
    leaving the cell in the wording actually shown on screen, and every stat groups the
    wordings under this one square (see wordings / pipeline.consolidate).
    """

    id: str
    text: str
    game_type: GameType
    aliases: list[str] = Field(default_factory=list)

    @property
    def wordings(self) -> list[str]:
        """The canonical text first, then every alias."""
        return [self.text, *self.aliases]


class Team(BaseModel):
    """One side of a 2v2 match: the team's name, as shown on the overlay, and its two
    players. Filled in at extraction from the season's roster (see storage.teams), since
    the overlay only shows team names.

    In Season 5 (see GameType.TWO_V_TWO) each team keeps the same roles for a whole season
    -- one player always plays the base game and the other the DLC -- so a claim on a
    base-game square is the base player's, a DLC square the DLC player's, and a co-op
    square both of theirs: `base_player`/`dlc_player`. In Season 3 both teammates play the
    base game on the same board, and the overlay doesn't say which of them completed a
    square, so the team lists its two `members` without roles and every claim is both of
    theirs. A team has one or the other, never both."""

    name: str
    base_player: str | None = None
    dlc_player: str | None = None
    members: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _roles_or_members(self) -> "Team":
        has_roles = self.base_player is not None or self.dlc_player is not None
        if self.members:
            if has_roles:
                raise ValueError(
                    f"team {self.name!r} has both members and base/dlc players -- pick one"
                )
            if len(self.members) != 2:
                raise ValueError(f"team {self.name!r} needs 2 members, has {len(self.members)}")
        elif self.base_player is None or self.dlc_player is None:
            raise ValueError(f"team {self.name!r} needs a base_player and a dlc_player, or members")
        return self

    @property
    def players(self) -> list[str]:
        if self.members:
            return list(self.members)
        return [self.base_player, self.dlc_player]

    def players_for(self, game_type: GameType | None) -> list[str]:
        """Whose claim a square of this pool is: the base or the DLC player, both for a
        co-op square, or nobody when the square's pool isn't known. A team without roles
        (members) shares every claim."""
        if game_type is None:
            return []
        if self.members:
            return list(self.members)
        if game_type is GameType.BASE:
            return [self.base_player]
        if game_type is GameType.DLC:
            return [self.dlc_player]
        if game_type is GameType.TWO_V_TWO:
            return self.players
        return []


class VideoExtraction(BaseModel):
    """Everything extracted from one video, and the unit `json_export`/`db` persist:
    one video is one match, and a match can contain several games (GameResult), but
    player names/match metadata are read/collected once per video, not once per game --
    see extract._match_id and CLAUDE.md."""

    match_id: str
    video_url: str | None
    match_date: date
    season: str
    match_type: MatchType
    # The two sides' names as the overlay shows them -- a player's name, or a team's in a
    # 2v2 match (with team_red/team_blue below naming its players).
    player_red_name: str | None
    player_blue_name: str | None
    extracted_at: date
    games: list[GameResult]
    # The commentators casting this match, read off the broadcast's own nameplates
    # (commentators.py), left seat to right. A flat list rather than named seats: which
    # webcam a commentator sat in isn't a property of the match. Empty when neither plate
    # read as a name -- including for a JSON file written before this was recorded, and
    # for footage whose nameplate regions are blacked out (see the test fixtures).
    commentators: list[str] = Field(default_factory=list)
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
    # A 2v2 match's teams (see Team) -- None for a 1v1 match, and for a team whose name
    # isn't in the season's roster.
    team_red: Team | None = None
    team_blue: Team | None = None

    def team(self, color: CellColor) -> Team | None:
        return self.team_red if color is CellColor.RED else self.team_blue

    def side_players(self, color: CellColor) -> list[str]:
        """The players on one side: a team's two players, or the side's own player name
        in a 1v1 match (empty when it isn't known)."""
        team = self.team(color)
        if team is not None:
            return team.players
        name = self.player_red_name if color is CellColor.RED else self.player_blue_name
        return [name] if name else []

    @field_validator("season", mode="before")
    @classmethod
    def _coerce_season_to_str(cls, value: object) -> object:
        """season is free text (see MatchMetadata.season above), but a hand-edited JSON
        file can write an all-digit season unquoted -- json.loads then hands that back as
        a Python int, not str. Normalized here so every VideoExtraction.season really is
        the str its type declares, regardless of how the file spelled it -- without this,
        a match history mixing a quoted "6" (always written this way by `extract`) with an
        unquoted 6 (a hand edit) fails to sort in pipeline.consolidate.consolidate_players,
        which groups by season across every match.
        """
        return str(value) if value is not None else value

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


class MatchRecord(BaseModel):
    """Wins, losses and draws for one season's worth of a player's matches (see
    PlayerStats.season_records). A draw is a real *match* outcome -- a round robin
    match can end 1-1, see MatchWinner.DRAW -- unlike for a single game, where a tie can
    no longer even occur on a 5x5 board (see winner._majority_settled's note on
    WinType.TIE), so this carries a draws count that WinLoss (used for the game-level
    tallies below) doesn't need."""

    wins: int = 0
    losses: int = 0
    draws: int = 0


class WinLoss(BaseModel):
    """Wins and losses for one bucket of a player's individual games (see
    PlayerStats.game_record/game_type_records) -- no draws field, since a single game
    can't end in one, unlike a match (see MatchRecord)."""

    wins: int = 0
    losses: int = 0


class SquareMarks(BaseModel):
    """One square text and how many times a player has personally marked it (see
    PlayerStats.top_squares_base_game/top_squares_dlc) -- the count travels with the
    text rather than being left for a reader to re-derive, since it's the very thing that
    ranked this square into the top 5 in the first place."""

    text: str
    marks: int


class PlayerMarks(BaseModel):
    """One player and how many games they finished holding one square in (see
    SquareStats.top_players) -- SquareMarks' mirror image, from the square's side."""

    # The join key to that player's PlayerInfo/PlayerStats -- see
    # pipeline.consolidate.slugify_name.
    slug: str
    marks: int


class SquareClaim(BaseModel):
    """One game that ended with a square claimed (see SquareStats.claims): which match
    and game it was, when the claim that stuck was made, and by whom -- the provenance of
    one entry in SquareStats.mark_times_s."""

    match_id: str
    game_index: int
    # GameEvent.game_elapsed_s of the persisting MARK, as in SquareStats.mark_times_s.
    time_s: int
    # The claiming player's slug (pipeline.consolidate.slugify_name), or None when the
    # match doesn't name the player on that side -- or when the claim was a 2v2 team's
    # co-op square, which is both players' (see slugs).
    slug: str | None = None
    # Every player behind the claim, when that's more than one: a co-op square's two
    # teammates. Empty otherwise, where slug says it all.
    slugs: list[str] = Field(default_factory=list)


class SquareStats(BaseModel):
    """One goal square's dynamically computed stats, built by
    pipeline.consolidate.consolidate_square_stats from every game it appeared in and
    persisted as one YAML file per square under `<data_dir>/square_stats` (see
    storage.square_stats) -- wholly regenerated by `square stats` on every run, like
    PlayerStats.

    "Marked" means the square was still claimed on the game's *final* board, not just
    that a MARK event touched it: a misclicked square that was immediately unmarked
    wasn't achieved, and counting it would let marks exceed games and push mark_rate
    past 1.
    """

    # The square's identity in the consolidated reference (see Square) -- id alone isn't
    # unique, since ids are only disambiguated within one game type.
    id: str
    text: str
    game_type: GameType
    # Distinct matches the square was dealt in, and those match_ids ordered by
    # match_date descending (most recent first), like PlayerStats.all_matches.
    num_matches: int = 0
    matches: list[str] = Field(default_factory=list)
    # Games the square was dealt in (a match can deal it in more than one game), and how
    # many of those ended with it marked.
    num_games: int = 0
    games_marked: int = 0
    # games_marked / num_games, or None when the square was never dealt at all.
    mark_rate: float | None = None
    # The 5 players who finished a game holding this square most often -- see
    # consolidate_square_stats.
    top_players: list[PlayerMarks] = Field(default_factory=list)
    # The game clock (GameEvent.game_elapsed_s) at the moment of each marked game's
    # persisting MARK, sorted ascending -- one entry per games_marked, the raw
    # distribution for the site to bucket however it likes.
    mark_times_s: list[int] = Field(default_factory=list)
    median_mark_time_s: float | None = None
    # The same claims as mark_times_s, one per marked game and in the same order (by
    # time, then match_id, then game_index), with the match, game and player behind each
    # -- so a consumer can link a point of the distribution back to its match.
    claims: list[SquareClaim] = Field(default_factory=list)


class PlayerInfo(BaseModel):
    """One player's static, hand-curated identity -- the half of a player's profile that
    pipeline.consolidate creates once and never touches again (see storage.player_info,
    issue #82). Persisted as one YAML file per player under a directory meant to be
    committed to the repo, unlike PlayerStats's fully-regenerated, gitignored
    counterpart: twitch/avatar/bio are genuinely hand-authored content a contributor
    fills in and expects to survive forever, not disposable output a re-run can
    reconstruct.

    id is the one field here the tool itself sets -- assigned once, the first time a
    player is ever seen across any extracted match, and kept stable after that (see
    pipeline.consolidate.consolidate_player_info). twitch/avatar/bio are never set by
    the tool at all.
    """

    id: int
    # Lowercase, underscores as separator -- e.g. "twistiet" -- what two OCR'd readings
    # of the same player's name are considered equal under (see
    # pipeline.consolidate.slugify_name). Also the file's own basename (see
    # storage.player_info.player_info_path).
    slug: str
    # Just the handle (e.g. "twistiet"), not the full https://twitch.tv/... URL.
    twitch: str | None = None
    avatar: str | None = None
    bio: str | None = None


class GameRef(BaseModel):
    """One game of one match, and a moment in it on the game clock (GameEvent.
    game_elapsed_s) -- what a player stat points back to, so a consumer can link it."""

    match_id: str
    game_index: int
    time_s: int


class PlayerPace(BaseModel):
    """How fast a player plays (see pipeline.player_insights). Claims are the player's own
    (a 2v2 player's: their own game's squares plus the team's co-op squares), counted only
    if still held at the end of the game; win times are their side's."""

    # Games they made at least one claim in, and the median game clock of their first.
    games: int = 0
    median_first_claim_s: float | None = None
    # Their claims per hour of game clock, over every game they played.
    claims_per_hour: float | None = None
    # Median game clock at the end of the games they won, and the quickest of those.
    median_win_time_s: float | None = None
    fastest_win: GameRef | None = None


class ClaimRef(GameRef):
    """A claim in one game, and the square claimed: its pool (its game type; on a 2v2
    board, the square's own tag -- see GameResult.cell_goal) and its text (the squares
    reference's wording when it knows the square, otherwise as read off the board,
    without a 2v2 tag)."""

    game_type: GameType
    text: str = ""


class NthClaim(BaseModel):
    """A player's n-th claim of a game (their 1st, 2nd... square), over every game they
    made that many claims in (see pipeline.player_insights)."""

    n: int
    # One per game, by time.
    claims: list[ClaimRef] = Field(default_factory=list)
    median_s: float | None = None
    # Everyone else's median for their own n-th claim.
    field_median_s: float | None = None


class WinTypeCounts(BaseModel):
    """A player's game wins by how they were won (models.WinType)."""

    line: int = 0
    majority: int = 0
    # Points games (Season 3), won on points rather than by a line.
    points: int = 0


class FirstSquare(BaseModel):
    """Who claimed a game's first square (the earliest claim still held at the end), over
    the decided games a player played -- a 2v2 team's first square counts for both."""

    games: int = 0
    first_claims: int = 0
    # Games won out of the first_claims ones.
    wins_after_first_claim: int = 0


class Comebacks(BaseModel):
    """Games a player's side won after trailing on the board by at least
    pipeline.player_insights.COMEBACK_DEFICIT squares, and the largest deficit they
    overcame."""

    count: int = 0
    largest_deficit: int = 0
    largest: GameRef | None = None


class FormResult(BaseModel):
    """One decided match's result for a player: "win", "loss" or "draw"."""

    match_id: str
    result: str


class PlayerForm(BaseModel):
    """A player's recent results and streaks, over decided matches by match date."""

    # Their last five results, most recent first.
    last: list[FormResult] = Field(default_factory=list)
    longest_win_streak: int = 0
    # The run they're on now: its result and how many matches long it is.
    current_streak_result: str | None = None
    current_streak: int = 0


class SignatureSquare(BaseModel):
    """A square a player stands out on (see pipeline.player_insights): claimed faster
    than everyone else, or claimed more often when it's on their board."""

    id: str
    text: str
    game_type: GameType
    # Their claims of it (held at the end of the game) and the games it was dealt to them.
    claims: int
    dealt: int
    median_time_s: float | None = None
    # Everyone else's median claim time for it.
    field_median_time_s: float | None = None


class PlayerStats(BaseModel):
    """One player's dynamically computed stats, built by
    pipeline.consolidate.consolidate_players from every match they appear in and
    persisted as one YAML file per player under `<data_dir>/players` (see
    storage.player_stats, issue #72) -- wholly regenerated by `player consolidate` on
    every run, unlike PlayerInfo's hand-curated identity (issue #82), which this
    deliberately carries no fields for and never touches.
    """

    # Lowercase, underscores as separator -- e.g. "twistiet" -- derived from
    # display_name and what two OCR'd readings of the same player's name are considered
    # equal under (see pipeline.consolidate.slugify_name). Also the file's own basename
    # (see storage.player_stats.player_stats_path), and the join key back to this
    # player's PlayerInfo.
    slug: str
    # The exact spelling seen most often across this player's matches -- see
    # consolidate_players.
    display_name: str
    season_records: dict[str, MatchRecord] = Field(default_factory=dict)
    game_record: WinLoss = Field(default_factory=WinLoss)
    game_type_records: dict[GameType, WinLoss] = Field(default_factory=dict)
    # match_ids the player appears in, ordered by match_date descending (most recent
    # first) -- see consolidate_players.
    all_matches: list[str] = Field(default_factory=list)
    # The 5 squares this player has personally marked most often, per game type, each
    # with its mark count -- see consolidate_players.
    top_squares_base_game: list[SquareMarks] = Field(default_factory=list)
    top_squares_dlc: list[SquareMarks] = Field(default_factory=list)
    # Co-op squares on a 2v2 mixed board (see GameType.TWO_V_TWO), which count for both
    # teammates.
    top_squares_coop: list[SquareMarks] = Field(default_factory=list)
    # Insights from the boards and claims -- see pipeline.player_insights.
    pace: PlayerPace = Field(default_factory=PlayerPace)
    # When they made their 1st, 2nd... claim of a game, up to
    # pipeline.player_insights.CLAIM_ORDER_LIMIT.
    claim_order: list[NthClaim] = Field(default_factory=list)
    # The same, counting only squares of one pool: their 1st, 2nd... DLC square.
    claim_order_by_type: dict[GameType, list[NthClaim]] = Field(default_factory=dict)
    win_types: WinTypeCounts = Field(default_factory=WinTypeCounts)
    first_square: FirstSquare = Field(default_factory=FirstSquare)
    comebacks: Comebacks = Field(default_factory=Comebacks)
    form: PlayerForm = Field(default_factory=PlayerForm)
    # Up to 3 squares each: claimed fastest against everyone else's median, and claimed
    # most often when dealt.
    quickest_squares: list[SignatureSquare] = Field(default_factory=list)
    surest_squares: list[SignatureSquare] = Field(default_factory=list)

    @field_validator("top_squares_base_game", "top_squares_dlc", mode="before")
    @classmethod
    def _normalize_legacy_plain_text_marks(cls, value: object) -> object:
        """A YAML file written before mark counts were added to top_squares_base_game/dlc
        (see storage.player_stats) holds a plain list of strings there instead of
        {text, marks} entries -- normalized to a mark count of 0 (unknown) so it still
        reads back rather than failing validation. Every field consolidate_players fills
        in here is wholly regenerated on the next `player consolidate` run anyway, so a
        stale/zeroed mark count surviving even one extra run costs nothing.
        """
        if not isinstance(value, list):
            return value
        return [{"text": entry, "marks": 0} if isinstance(entry, str) else entry for entry in value]
