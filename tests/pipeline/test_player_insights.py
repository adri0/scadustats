import datetime

from scadustats.models import (
    CellColor,
    EventType,
    GameEvent,
    GameRef,
    GameResult,
    GameType,
    MatchType,
    PlayerStats,
    Square,
    Team,
    VideoExtraction,
    WinType,
    format_video_timestamp,
)
from scadustats.pipeline.consolidate import consolidate_players
from scadustats.pipeline.player_insights import add_player_insights

RED, BLUE = CellColor.RED, CellColor.BLUE


def _event(
    row: int | None,
    col: int | None,
    color: CellColor | None,
    elapsed: int,
    event_type: EventType = EventType.MARK,
) -> GameEvent:
    return GameEvent(
        row=row,
        col=col,
        color=color,
        video_timestamp=format_video_timestamp(float(elapsed)),
        game_elapsed_s=elapsed,
        event_type=event_type,
    )


def _end(elapsed: int) -> GameEvent:
    return _event(None, None, None, elapsed, EventType.GAME_END)


def _game(
    *events: GameEvent,
    winner: CellColor | None = RED,
    win_type: WinType = WinType.LINE,
    texts: list[str] | None = None,
    game_index: int = 1,
    game_type: GameType = GameType.BASE,
) -> GameResult:
    flat = (texts or []) + [""] * (25 - len(texts or []))
    return GameResult(
        game_index=game_index,
        start_video_ts_s=0.0,
        end_video_ts_s=100.0,
        square_texts=[flat[r * 5 : r * 5 + 5] for r in range(5)],
        events=list(events),
        winner_color=winner,
        win_type=win_type if winner else WinType.NONE,
        game_type=game_type,
    )


def _match(
    *games: GameResult,
    match_id: str = "2026-03-05-alice-vs-bob",
    match_date: datetime.date = datetime.date(2026, 3, 5),
    red: str = "alice",
    blue: str = "bob",
    **teams: Team,
) -> VideoExtraction:
    return VideoExtraction(
        match_id=match_id,
        video_url=None,
        match_date=match_date,
        season="6",
        match_type=MatchType.ROUND_ROBIN,
        player_red_name=red,
        player_blue_name=blue,
        extracted_at=datetime.date(2026, 3, 6),
        games=list(games),
        **teams,
    )


def _insights(*matches: VideoExtraction, known=None) -> dict[str, PlayerStats]:
    extractions = list(matches)
    return add_player_insights(consolidate_players(extractions, known), extractions, known)


def test_pace_counts_held_claims_only():
    game = _game(
        _event(1, 1, RED, 600),
        _event(1, 2, RED, 900),
        _event(1, 2, RED, 950, EventType.UNMARK),
        _event(2, 2, BLUE, 300),
        _end(3600),
    )
    stats = _insights(_match(game))

    alice = stats["alice"].pace
    assert alice.games == 1
    assert alice.median_first_claim_s == 600
    assert alice.claims_per_hour == 1.0
    assert alice.median_win_time_s == 3600
    assert alice.fastest_win == GameRef(
        match_id="2026-03-05-alice-vs-bob", game_index=1, time_s=3600
    )
    assert stats["bob"].pace.median_first_claim_s == 300
    assert stats["bob"].pace.fastest_win is None


def test_fastest_win_is_the_shortest_won_game():
    stats = _insights(
        _match(
            _game(_event(1, 1, RED, 10), _end(3000), game_index=1),
            _game(_event(1, 1, RED, 10), _end(1800), game_index=2),
            _game(_event(1, 1, BLUE, 10), _end(900), game_index=3, winner=BLUE),
        )
    )

    assert stats["alice"].pace.fastest_win.game_index == 2
    assert stats["alice"].pace.median_win_time_s == 2400


def test_win_types_count_game_wins_by_line_and_majority():
    stats = _insights(
        _match(
            _game(_end(10), game_index=1, win_type=WinType.LINE),
            _game(_end(10), game_index=2, win_type=WinType.MAJORITY),
            _game(_end(10), game_index=3, win_type=WinType.MAJORITY),
            _game(_end(10), game_index=4, winner=BLUE, win_type=WinType.LINE),
            _game(_end(10), game_index=5, winner=BLUE, win_type=WinType.POINTS),
        )
    )

    alice, bob = stats["alice"].win_types, stats["bob"].win_types
    assert (alice.line, alice.majority, alice.points) == (1, 2, 0)
    assert (bob.line, bob.majority, bob.points) == (1, 0, 1)


def test_first_square_is_the_earliest_held_claim():
    stats = _insights(
        _match(
            # Bob's first mark is undone, so Alice holds the first square.
            _game(
                _event(1, 1, BLUE, 100),
                _event(1, 1, BLUE, 120, EventType.UNMARK),
                _event(2, 2, RED, 200),
                game_index=1,
                winner=RED,
            ),
            _game(_event(1, 1, RED, 50), game_index=2, winner=BLUE),
            # An undecided game doesn't count.
            _game(_event(1, 1, RED, 50), game_index=3, winner=None),
        )
    )

    alice = stats["alice"].first_square
    assert (alice.games, alice.first_claims, alice.wins_after_first_claim) == (2, 2, 1)
    assert stats["bob"].first_square.first_claims == 0


def test_comeback_after_trailing_by_three():
    stats = _insights(
        _match(
            _game(
                _event(1, 1, BLUE, 10),
                _event(1, 2, BLUE, 20),
                _event(1, 3, BLUE, 30),
                *(_event(r, 5, RED, 40 + r) for r in range(1, 6)),
                _end(100),
                winner=RED,
            ),
            _game(_event(1, 1, BLUE, 10), _event(1, 2, BLUE, 20), _end(50), game_index=2),
        )
    )

    comebacks = stats["alice"].comebacks
    assert (comebacks.count, comebacks.largest_deficit) == (1, 3)
    assert comebacks.largest.game_index == 1
    assert stats["bob"].comebacks.count == 0


def test_form_lists_last_five_and_streaks():
    def played(day: int, winner: CellColor) -> VideoExtraction:
        return _match(
            _game(_end(10), winner=winner),
            match_id=f"m{day:02d}",
            match_date=datetime.date(2026, 3, day),
        )

    results = [RED, RED, RED, BLUE, RED, RED]
    stats = _insights(*(played(day, winner) for day, winner in enumerate(results, start=1)))

    form = stats["alice"].form
    assert [(r.match_id, r.result) for r in form.last] == [
        ("m06", "win"),
        ("m05", "win"),
        ("m04", "loss"),
        ("m03", "win"),
        ("m02", "win"),
    ]
    assert form.longest_win_streak == 3
    assert (form.current_streak_result, form.current_streak) == ("win", 2)
    assert stats["bob"].form.longest_win_streak == 1


def test_signature_squares_compare_against_everyone_else():
    known = {
        GameType.BASE: [
            Square(id="crystalian", text="Kill a Crystalian", game_type=GameType.BASE),
            Square(id="merchant", text="Find a Merchant", game_type=GameType.BASE),
        ]
    }
    games = []
    for index in range(1, 5):
        # Alice takes the Crystalian at 100s every time; Bob takes the merchant at 500s.
        games.append(
            _game(
                _event(1, 1, RED, 100),
                _event(1, 2, BLUE, 500),
                texts=["Kill a Crystalian", "Find a Merchant"],
                game_index=index,
            )
        )
    # Bob takes the Crystalian at 1000s in three more games.
    for index in range(5, 8):
        games.append(_game(_event(1, 1, BLUE, 1000), texts=["Kill a Crystalian"], game_index=index))
    stats = _insights(_match(*games), known=known)

    quickest = stats["alice"].quickest_squares
    assert [s.id for s in quickest] == ["crystalian"]
    assert (quickest[0].median_time_s, quickest[0].field_median_time_s) == (100, 1000)
    assert (quickest[0].claims, quickest[0].dealt) == (4, 7)

    surest = {s.id: s for s in stats["bob"].surest_squares}
    assert (surest["merchant"].claims, surest["merchant"].dealt) == (4, 4)
    assert surest["merchant"] == stats["bob"].surest_squares[0]


def test_2v2_side_stats_count_for_both_but_claims_for_the_squares_owner():
    teams = {
        "team_red": Team(name="Reds", base_player="alice", dlc_player="carol"),
        "team_blue": Team(name="Blues", base_player="bob", dlc_player="dave"),
    }
    game = _game(
        _event(1, 1, RED, 100),
        _event(1, 2, RED, 400),
        _end(3600),
        texts=["Kill a Crystalian (B)", "Kill Rellana (D)"],
        game_type=GameType.TWO_V_TWO,
    )
    stats = _insights(_match(game, red="Reds", blue="Blues", **teams))

    assert stats["alice"].pace.median_first_claim_s == 100
    assert stats["carol"].pace.median_first_claim_s == 400
    assert stats["alice"].first_square.first_claims == 1
    assert stats["carol"].first_square.first_claims == 1
    assert stats["carol"].form.last[0].result == "win"
    assert stats["dave"].form.last[0].result == "loss"


def test_claim_order_lists_each_nth_claim_against_everyone_else():
    stats = _insights(
        _match(
            _game(
                _event(1, 1, RED, 100),
                _event(1, 2, RED, 300),
                # Undone, so not a claim.
                _event(1, 3, RED, 350),
                _event(1, 3, RED, 360, EventType.UNMARK),
                _event(2, 1, BLUE, 200),
                game_index=1,
            ),
            _game(_event(1, 1, RED, 50), _event(2, 1, BLUE, 400), game_index=2),
        )
    )

    order = stats["alice"].claim_order
    assert [nth.n for nth in order] == [1, 2]
    assert [(ref.game_index, ref.time_s) for ref in order[0].claims] == [(2, 50), (1, 100)]
    assert order[0].median_s == 75
    assert order[0].field_median_s == 300
    assert order[1].claims[0].time_s == 300
    assert order[1].field_median_s is None


def test_claim_order_by_type_counts_one_pools_squares():
    teams = {
        "team_red": Team(name="Reds", base_player="alice", dlc_player="carol"),
        "team_blue": Team(name="Blues", base_player="bob", dlc_player="dave"),
    }
    game = _game(
        # Carol's: a co-op square, then two DLC squares.
        _event(1, 1, RED, 100),
        _event(1, 2, RED, 200),
        _event(1, 3, RED, 300),
        # Dave's DLC square.
        _event(2, 2, BLUE, 250),
        texts=[
            "Kill a Wyrm (C)",
            "Kill Rellana (D)",
            "Kill Messmer (D)",
            "",
            "",
            "",
            "Kill Bayle (D)",
        ],
        game_type=GameType.TWO_V_TWO,
    )
    stats = _insights(_match(game, red="Reds", blue="Blues", **teams))

    carol = stats["carol"]
    assert [(ref.time_s, ref.game_type) for nth in carol.claim_order for ref in nth.claims] == [
        (100, GameType.TWO_V_TWO),
        (200, GameType.DLC),
        (300, GameType.DLC),
    ]
    assert [ref.text for nth in carol.claim_order for ref in nth.claims] == [
        "Kill a Wyrm",
        "Kill Rellana",
        "Kill Messmer",
    ]
    assert set(carol.claim_order_by_type) == {GameType.DLC, GameType.TWO_V_TWO}
    dlc = carol.claim_order_by_type[GameType.DLC]
    assert [nth.claims[0].time_s for nth in dlc] == [200, 300]
    assert dlc[0].field_median_s == 250
    assert carol.claim_order_by_type[GameType.TWO_V_TWO][0].field_median_s == 100
