"""Rendering for the read-only `match` CLI commands (see cli/app.py).

Kept out of cli/app.py so the formatting is plain data-in/strings-out and can be tested
without a CliRunner: every function here returns lines (or one line) and prints nothing,
and cli/app.py's job is reduced to reading the JSON and echoing what comes back.

Nothing here interprets the extraction -- it only presents what the JSON already says.
In particular a game's result is shown exactly as recorded, never recomputed from its
events: `match validate` is where the two are cross-checked, and silently showing a
"corrected" result here would hide the very disagreement that command exists to find.
"""

from collections.abc import Sequence

from scadustats.models import (
    CellColor,
    EventType,
    GameResult,
    GameType,
    MatchWinner,
    VideoExtraction,
    WinLine,
    WinType,
)
from scadustats.rules import winner

# Colored square emoji rather than letters or ANSI-styled text: each glyph is inherently
# colored, so the board reads at a glance and stays legible even when piped/redirected
# (unlike ANSI codes, which typer.echo strips outside a terminal).
_CELL_SYMBOLS = {CellColor.RED: "🟥", CellColor.BLUE: "🟦", CellColor.UNCLAIMED: "⬛"}

_GAME_TYPE_LABELS = {GameType.BASE: "base game", GameType.DLC: "DLC game"}

# Goal texts run long (a full sentence, sometimes an OCR'd mess); an events listing is
# read as a column, so they're cut to keep the rows scannable. The JSON has the full text.
_GOAL_TEXT_WIDTH = 44


def format_duration(seconds: float) -> str:
    """A video length or a span within one, as HH:MM:SS -- display only, the JSON keeps
    raw seconds."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_clock(seconds: float) -> str:
    """A game-clock reading, as MM:SS -- the overlay's own stopwatch never reaches an
    hour, so the leading `00:` an HH:MM:SS would carry is pure noise here."""
    total = int(seconds)
    sign = "-" if total < 0 else ""
    minutes, secs = divmod(abs(total), 60)
    return f"{sign}{minutes:02d}:{secs:02d}"


def undecided_games(extraction: VideoExtraction) -> int:
    """How many of a match's games have no winner recorded. VideoExtraction.winner is
    None for such a match (it won't name a leader from an incomplete score), so this is
    what says *why* -- and it's usually an extraction problem rather than a played
    result."""
    return sum(game.winner_color is None for game in extraction.games)


def player_names(extraction: VideoExtraction) -> tuple[str, str]:
    """The two players, falling back to their colors -- a name can be missing from the
    JSON (a failed OCR read, see extract._video_id), and "red" still identifies a side."""
    return (extraction.player_red_name or "red", extraction.player_blue_name or "blue")


def format_outcome(extraction: VideoExtraction) -> str:
    """Who took the match, e.g. "alice wins 2-1". Defers to VideoExtraction.winner rather
    than reading the scores itself, so the CLI can't disagree with the JSON and the DB
    about who won -- and says "leads" rather than "wins" for the case that property
    refuses to call (a game with no winner recorded), so a score that only looks final
    isn't presented as if it were.
    """
    red, blue = player_names(extraction)
    if not extraction.games:
        return "no games"

    red_score, blue_score = extraction.red_score, extraction.blue_score
    if extraction.winner is MatchWinner.RED:
        return f"{red} wins {red_score}-{blue_score}"
    if extraction.winner is MatchWinner.BLUE:
        return f"{blue} wins {blue_score}-{red_score}"
    if extraction.winner is MatchWinner.DRAW:
        return f"draw {red_score}-{blue_score}"

    # Undetermined: some game has no winner, so the score so far doesn't account for
    # every game played. Report it as a standing, with what's missing spelled out.
    if red_score > blue_score:
        standing = f"{red} leads {red_score}-{blue_score}"
    elif blue_score > red_score:
        standing = f"{blue} leads {blue_score}-{red_score}"
    else:
        standing = f"level {red_score}-{blue_score}"
    return f"{standing}, {undecided_games(extraction)} undecided"


def format_result(game: GameResult, extraction: VideoExtraction) -> str:
    """One game's recorded result in prose, e.g. "alice (red) by line on row 2"."""
    red, blue = player_names(extraction)
    if game.winner_color is None:
        if game.win_type is WinType.TIE:
            return "tie -- every line blocked, neither player ahead"
        return "no winner recorded"

    name = red if game.winner_color is CellColor.RED else blue
    how = game.win_type.value
    if game.win_line is not None:
        how += f" on {game.win_line.label}"
    return f"{name} ({game.winner_color.value}) by {how}"


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A header row plus `rows`, each column padded to its own widest value. Ragged
    columns are what made the old one-line-per-match listing hard to read across rows."""
    widths = [
        max(len(str(row[column])) for row in [headers, *rows]) for column in range(len(headers))
    ]
    lines = []
    for row in [headers, *rows]:
        cells = [str(value).ljust(width) for value, width in zip(row, widths, strict=True)]
        lines.append("  ".join(cells).rstrip())
    return lines


def render_match_table(extractions: Sequence[VideoExtraction]) -> list[str]:
    """The `match list` table. The date and both player names are deliberately not their
    own columns: video_id already spells out `<date>-<red>-vs-<blue>` (see
    extract._video_id) and repeating them crowded out the one thing the old listing
    didn't say at all, which is how each match actually ended."""
    rows = [
        [
            extraction.video_id,
            extraction.season,
            extraction.match_type.value,
            str(extraction.num_games),
            format_outcome(extraction),
        ]
        for extraction in extractions
    ]
    return render_table(["MATCH", "SEASON", "TYPE", "GAMES", "RESULT"], rows)


def render_board(board: winner.Board) -> list[str]:
    """The 5x5 board as five lines of colored square emoji (red/blue/black, see
    _CELL_SYMBOLS) -- a completed line is already visible as five squares of one color
    in a row, so no extra marker (e.g. brackets) is needed to call it out, and one would
    only break the grid's symmetry.
    """
    return [" ".join(_CELL_SYMBOLS[cell] for cell in row) for row in board]


def _square_text(game: GameResult, row: int | None, col: int | None) -> str:
    """The goal text at one square, or "" when it isn't there -- square_texts comes from
    OCR and can be hand-edited, so a short/ragged grid is a real possibility and isn't
    worth crashing a read-only listing over."""
    if row is None or col is None:
        return ""
    try:
        # row/col are 1-based (see models.GameEvent); square_texts is a plain 0-based grid.
        return game.square_texts[row - 1][col - 1]
    except IndexError:
        return ""


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def render_events(game: GameResult, extraction: VideoExtraction) -> list[str]:
    """One game's events as a table, in the order they happened.

    Ordered by video timestamp rather than the game clock, for the same reason
    validation._ordered_events is: the clock counts *down* through the pre-game countdown,
    so sorting by it would file GAME_START in among the claims instead of ahead of them.
    """
    red, blue = player_names(extraction)
    rows = []
    for event in sorted(game.events, key=lambda event: event.video_ts_s):
        if event.event_type is EventType.GAME_START:
            rows.append(
                [
                    format_duration(event.video_ts_s),
                    format_clock(event.game_elapsed_s),
                    "game start",
                    "",
                    "",
                    "",
                ]
            )
            continue
        player = red if event.color is CellColor.RED else blue
        rows.append(
            [
                format_duration(event.video_ts_s),
                format_clock(event.game_elapsed_s),
                "mark" if event.event_type is EventType.MARK else "unmark",
                player,
                f"r{event.row} c{event.col}",
                _truncate(_square_text(game, event.row, event.col), _GOAL_TEXT_WIDTH),
            ]
        )
    return render_table(["VIDEO", "CLOCK", "EVENT", "PLAYER", "SQUARE", "GOAL"], rows)


def _format_span(game: GameResult) -> str:
    """When a game runs within the video, e.g. "00:06:12-00:21:30 (15:18)". The end is
    optional in the model (GameResult.end_video_ts_s), so both it and the length it
    implies drop out rather than being guessed."""
    start = format_duration(game.start_video_ts_s)
    if game.end_video_ts_s is None:
        return f"{start}-?"
    length = format_clock(game.end_video_ts_s - game.start_video_ts_s)
    return f"{start}-{format_duration(game.end_video_ts_s)} ({length})"


def render_game(game: GameResult, extraction: VideoExtraction, *, events: bool) -> list[str]:
    red, blue = player_names(extraction)
    board = winner.replay(game.events)
    claimed = {
        color: sum(cell is color for row in board for cell in row)
        for color in (CellColor.RED, CellColor.BLUE, CellColor.UNCLAIMED)
    }
    marks = sum(event.event_type is EventType.MARK for event in game.events)
    unmarks = sum(event.event_type is EventType.UNMARK for event in game.events)
    game_type = _GAME_TYPE_LABELS.get(game.game_type, "unknown game type")

    lines = [
        f"  Game {game.game_index}  {game_type}  {_format_span(game)}",
        f"    result   {format_result(game, extraction)}",
        f"    squares  {red} {claimed[CellColor.RED]}, {blue} {claimed[CellColor.BLUE]}, "
        f"unclaimed {claimed[CellColor.UNCLAIMED]}",
        f"    events   {marks} mark{'s' if marks != 1 else ''}, "
        f"{unmarks} unmark{'s' if unmarks != 1 else ''}",
        "",
    ]
    lines.extend(f"    {line}" for line in render_board(board))
    if events:
        lines.append("")
        lines.extend(f"    {line}" for line in render_events(game, extraction))
    return lines


def render_match(extraction: VideoExtraction, *, events: bool = False) -> list[str]:
    """The whole `match show` report: the match's own details, then one block per game.

    The board printed per game is the one its events replay to -- i.e. the board as it
    stood when the game ended, unclaims and all -- which is also what a recorded winner
    is supposed to agree with (see validation._check_recorded_winner_matches_board), so
    a disagreement is visible right there next to the result it contradicts.
    """
    red, blue = player_names(extraction)
    games = sorted(extraction.games, key=lambda game: game.game_index)
    lines = [
        extraction.video_id,
        f"  {red} (red) vs {blue} (blue) -- {format_outcome(extraction)}",
        f"  {extraction.match_date}  season {extraction.season}  "
        f"{extraction.match_type.value}  "
        f"{extraction.num_games} game{'s' if extraction.num_games != 1 else ''}",
    ]

    # Only when there are names to print: a nameplate that never read (or a video with no
    # commentator cams) leaves the list empty, and an empty "commentary" line would read
    # as a match nobody cast rather than as one whose plates weren't legible.
    if extraction.commentators:
        lines.append(f"  commentary {', '.join(extraction.commentators)}")

    details = []
    if extraction.duration_s is not None:
        details.append(f"length {format_duration(extraction.duration_s)}")
    if extraction.published_at is not None:
        details.append(f"published {extraction.published_at}")
    details.append(f"extracted {extraction.extracted_at}")
    lines.append("  " + "  ".join(details))
    if extraction.video_url:
        lines.append(f"  video {extraction.video_url}")

    if games:
        lines.append("")
        red_symbol, blue_symbol = _CELL_SYMBOLS[CellColor.RED], _CELL_SYMBOLS[CellColor.BLUE]
        unclaimed_symbol = _CELL_SYMBOLS[CellColor.UNCLAIMED]
        lines.append(
            f"  board key: {red_symbol} = {red}, {blue_symbol} = {blue}, "
            f"{unclaimed_symbol} = unclaimed"
        )
    for game in games:
        lines.append("")
        lines.extend(render_game(game, extraction, events=events))
    return lines
