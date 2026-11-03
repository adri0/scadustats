"""Overlay geometry for the Bingo Brawlers broadcast templates.

All regions are stored as fractions (0-1) of frame width/height rather than
pixels, because each template is composited proportionally to the output
resolution (confirmed for the standard template by comparing a 1280x720 and a
1920x1080 recording of it) -- fixed pixel coordinates would only work for one
resolution.

A broadcast can use more than one template: `STANDARD` is the season's usual
layout, while `LAYOUT_SEASON_6_FINAL` is a visually unrelated template used
for the Season 6 finals broadcast -- same grid/timer/score-bar/player-name/
game-type concepts, but at entirely different positions on the frame (the
grid moves from a full-width strip to the bottom-right quadrant, the timer
and player names move to the very bottom, and there are no commentator
nameplates at all). `Layout` bundles one template's full set of regions so
the rest of `overlay/` can be parameterized over which one applies to a given
video, rather than every module hard-coding a single set of module-level
constants. `LAYOUTS` is the registry `pipeline.extract` auto-detects against
(and a `--layout` CLI override looks up by name).

These constants are starting guesses reasoned from the observed composition
and must be hand-tuned against real frames with scripts/calibrate_layout.py
before anything downstream (color classification, OCR) can work.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from scadustats.models import FractionalBox, LayoutName


@dataclass(frozen=True)
class Layout:
    """One broadcast template's full set of overlay regions.

    `commentator_box_left`/`right` are `None` for a template that prints no
    commentator nameplates at all (as opposed to a nameplate box that's simply
    blank in a given frame, which `commentators._has_nameplate` already
    handles) -- `commentators.read_commentator_names` skips straight to
    `(None, None)` in that case rather than cropping and OCR'ing a region that
    was never calibrated to hold a nameplate. `game_type_box` is `None` the same
    way for a template with no "BASE GAME"/"DLC" subtitle at all, leaving the
    game type to be inferred from the board's square texts (see
    pipeline.squares.infer_game_type).

    `score_box_red`/`blue` and `name_box_red`/`blue` are `None` for a template with no
    claim counts or names on its gameplay scene (Season 3): `board.is_gameplay_frame` then
    recognizes that scene by its timer reading instead, the counts aren't cross-checked
    and the names are left for the contributor to fill in.

    `scenes` is for a broadcast that cuts between several gameplay scenes, each with the
    grid and timer somewhere else (Season 3: a team view and a one-player focus view).
    Such a Layout is the set of them: every frame is read with whichever of its scenes it
    shows (see board.scene_for), and its own boxes are unused.
    """

    name: LayoutName
    grid_box: FractionalBox
    timer_box: FractionalBox
    # The small "BASE GAME"/"DLC" subtitle -- see game_type_label.py. The "GAME N" label
    # itself has no box here: a game is identified by its index within the video (see
    # models.GameResult), so N is never read.
    game_type_box: FractionalBox | None
    score_box_red: FractionalBox | None
    score_box_blue: FractionalBox | None
    name_box_red: FractionalBox | None
    name_box_blue: FractionalBox | None
    commentator_box_left: FractionalBox | None
    commentator_box_right: FractionalBox | None
    # Whether each square's text ends in a "(B)"/"(D)"/"(C)" pool tag -- a Season 5 mixed
    # board (see board.tag_square_texts_majority and models.GameType.TWO_V_TWO).
    square_type_tags: bool = False
    # Whether the name boxes hold team names (several words, e.g. "POSITIVE VIBES")
    # rather than a single player's name -- see scoreboard.read_player_names.
    team_names: bool = False
    scenes: tuple["Layout", ...] = ()
    # Whether each player claims in their own colour, printed in their name too, rather than
    # the overlay's red/blue (Season 1, see models.ClaimColor): the board is then read
    # against the two colours the names show (see board.read_claim_colors).
    player_colors: bool = False
    # A box holding printed white text (a wordmark) that only this scene shows -- what tells
    # Season 1's main scene from its focus scene, whose clock is in the same place (see
    # board.is_gameplay_frame).
    logo_box: FractionalBox | None = None


STANDARD = Layout(
    name=LayoutName.STANDARD,
    grid_box=FractionalBox(left=0.354, top=0.504, right=0.646, bottom=0.988),
    timer_box=FractionalBox(left=0.008, top=0.878, right=0.172, bottom=0.972),
    game_type_box=FractionalBox(left=0.86, top=0.905, right=1.0, bottom=0.945),
    score_box_red=FractionalBox(left=0.0, top=0.5, right=0.06, bottom=0.556),
    score_box_blue=FractionalBox(left=0.94, top=0.5, right=1.0, bottom=0.556),
    name_box_red=FractionalBox(left=0.08, top=0.5, right=0.3, bottom=0.556),
    name_box_blue=FractionalBox(left=0.7, top=0.5, right=0.96, bottom=0.556),
    # The dark nameplates printed across the bottom of the two *inner* webcams -- the
    # commentators', flanking the grid. The players' own webcams are the outer pair and
    # carry no nameplate (their names are in the score strip, see NAME_BOX_* above).
    # Measured on a 720p broadcast frame as x 220-450/830-1060, y 671-706, which is
    # exactly symmetric about the frame's center line -- see commentators.py.
    commentator_box_left=FractionalBox(left=0.172, top=0.932, right=0.352, bottom=0.980),
    commentator_box_right=FractionalBox(left=0.648, top=0.932, right=0.828, bottom=0.980),
)

# Season 6 finals broadcast: two full-width gameplay feeds fill the top half, the two
# commentators (no nameplates) and both players' own webcams occupy a middle row, and the
# 5x5 grid is confined to the bottom-right quadrant -- with the timer, "GAME N"/game-type
# label, player names and live claim counts all moved to a bottom strip below everything
# else. Measured on a 1280x720 frame at 00:41:32 of the Season 6 finals VOD (see
# tests/fixtures/clip_layout_season_6_final.mp4) by cropping each region and confirming it
# OCRs cleanly with the exact call each module already makes (ocr.read_text with that
# call's own psm/whitelist/scale), the same way the standard template's own boxes are
# meant to be hand-tuned (see the module docstring).
LAYOUT_SEASON_6_FINAL = Layout(
    name=LayoutName.SEASON_6_FINAL,
    grid_box=FractionalBox(left=0.695, top=0.497, right=1.0, bottom=1.0),
    timer_box=FractionalBox(left=0.203, top=0.928, right=0.297, bottom=0.972),
    game_type_box=FractionalBox(left=0.104, top=0.953, right=0.166, bottom=0.982),
    # Tight around the live claim-count badge itself (a small ~28x20px square, unlike the
    # standard template's much wider score bar) -- the first pass measured here padded a
    # few pixels of the surrounding dark background into the crop, which diluted the
    # badge's own red/blue enough to fail is_gameplay_frame's R/B-vs-G contrast check
    # (tuned against the standard layout's much larger, less digit-dominated score bar)
    # even though the "4" digit inside still OCR'd fine either way. Re-measured by
    # scanning nearby box bounds for the widest margin that both classifies correctly and
    # OCRs the digit, then centering within it -- confirmed stable across every second of
    # tests/fixtures/clip_layout_season_6_final.mp4.
    score_box_red=FractionalBox(left=0.3609, top=0.8514, right=0.382, bottom=0.8792),
    score_box_blue=FractionalBox(left=0.6234, top=0.8528, right=0.6469, bottom=0.8778),
    name_box_red=FractionalBox(left=0.328, top=0.907, right=0.473, bottom=0.944),
    name_box_blue=FractionalBox(left=0.520, top=0.907, right=0.676, bottom=0.944),
    commentator_box_left=None,
    commentator_box_right=None,
)

# Pre-season 6 broadcast: the same template as STANDARD (grid, score strip, player
# names and commentator nameplates all in the same place), except that the timer sits
# ~28px (at 720p) higher -- with a row of per-player game-win indicators above it and an
# empty panel below -- and there is no "BASE GAME"/"DLC" subtitle anywhere on screen
# (the bottom-right corner holds a webcam instead), so game_type_box is None. Measured on
# the pre-season 6 VOD star0chris vs itzCBD (youtube.com/watch?v=Wyz0jB_fQ0M, 1280x720):
# the timer reads cleanly anywhere from 16px to 40px above STANDARD's timer_box, so this
# box sits in the middle of that range.
#
# Because its score strip is STANDARD's, is_gameplay_frame can't tell the two apart, so
# _detect_layout always picks STANDARD (registered first) on this footage. Pass
# `--layout layout_pre_season_6` explicitly to extract a pre-season 6 VOD.
LAYOUT_PRE_SEASON_6 = dataclasses.replace(
    STANDARD,
    name=LayoutName.PRE_SEASON_6,
    timer_box=FractionalBox(left=0.008, top=0.839, right=0.172, bottom=0.933),
    game_type_box=None,
)

# Season 5 broadcast (2v2 teams, see models.GameType.TWO_V_TWO): four gameplay feeds fill
# the frame's quadrants, with the 5x5 grid in the very center between them, the timer
# just above it under the "BINGO BRAWLERS SEASON 5" logo, and the live claim counts as two
# badges further down the center column -- blue on the left, red on the right. A strip
# along the bottom holds the commentators' names at its far ends (with mic icons, no
# webcam nameplates) and the two team names in its middle, blue then red. There's no
# "BASE GAME"/"DLC" subtitle: every game is TWO_V_TWO and each square's own text ends in
# a "(B)"/"(D)"/"(C)" tag instead (see board.tag_square_texts_majority). Measured on a 1280x720
# frame at 00:30:00 of Positive Vibes vs Bushy Stache (youtube.com/watch?v=YX3OSxKTJE8):
# the grid sits inside a yellow border at x 510-770, y 214-452, and the badges are flat
# blue/red at x 601-639/642-681, y 632-667.
LAYOUT_SEASON_5 = Layout(
    name=LayoutName.SEASON_5,
    grid_box=FractionalBox(left=0.4, top=0.3014, right=0.6, bottom=0.625),
    timer_box=FractionalBox(left=0.466, top=0.261, right=0.534, bottom=0.292),
    game_type_box=None,
    score_box_red=FractionalBox(left=0.503, top=0.881, right=0.53, bottom=0.925),
    score_box_blue=FractionalBox(left=0.471, top=0.881, right=0.498, bottom=0.925),
    name_box_red=FractionalBox(left=0.502, top=0.944, right=0.677, bottom=0.99),
    name_box_blue=FractionalBox(left=0.327, top=0.944, right=0.498, bottom=0.99),
    commentator_box_left=FractionalBox(left=0.0, top=0.944, right=0.21, bottom=0.99),
    commentator_box_right=FractionalBox(left=0.85, top=0.944, right=1.0, bottom=0.99),
    square_type_tags=True,
    team_names=True,
)

# Season 5's second broadcast template, used from Positive Vibes vs Itzerino
# (youtube.com/watch?v=zl8vqTjgCEw, 2025-10-12) on: the same 2v2 mixed board, at the same
# size, but moved to the bottom center under a "BASE GAME / DLC / CO-OP" header strip,
# with the timer at the top center, each player's name printed over their webcam, and a
# bottom strip holding both team names around their claim-count badges -- red on the
# left this time, blue on the right -- plus a single "A & B" commentator line in the
# bottom-right corner. That line moves with the names' lengths, so it has no boxes here
# (commentators are left for a contributor to fill in). Measured on a 1280x720 frame at
# 00:30:00 of that video: grid x 512-768 / y 409-642, timer x 599-681 / y 94-120, badges
# flat red x 612-638 and blue x 641-666 at y 692-719.
LAYOUT_SEASON_5_V2 = dataclasses.replace(
    LAYOUT_SEASON_5,
    name=LayoutName.SEASON_5_V2,
    grid_box=FractionalBox(left=0.4, top=0.568, right=0.6, bottom=0.8917),
    timer_box=FractionalBox(left=0.468, top=0.1306, right=0.532, bottom=0.1667),
    score_box_red=FractionalBox(left=0.481, top=0.964, right=0.497, bottom=0.996),
    score_box_blue=FractionalBox(left=0.503, top=0.964, right=0.519, bottom=0.996),
    name_box_red=FractionalBox(left=0.3, top=0.961, right=0.477, bottom=0.999),
    name_box_blue=FractionalBox(left=0.522, top=0.961, right=0.7, bottom=0.999),
    commentator_box_left=None,
    commentator_box_right=None,
)

# Season 5's third template, used from the second round robin weekend (Monkey Business vs
# Blanxonetta, youtube.com/watch?v=EwSBtdl4ZuE, 2025-10-24) on: LAYOUT_SEASON_5_V2 with the
# "BASE GAME / DLC / CO-OP" strip moved from above the board to below it, which lifts the
# board ~18px (at 720p). Nothing else moved. Measured on a 1280x720 frame at 00:50:00 of
# that video: the grid's top border is at y 391; the cell colors agree with both claim-count
# badges for any lift from 12px to 26px, so the box sits at 18px, in the middle of that range.
#
# Its score strip is LAYOUT_SEASON_5_V2's, so is_gameplay_frame can't tell the two apart and
# _detect_layout always picks V2 (registered first) on this footage. Pass
# `--layout layout_season_5_v3` explicitly.
LAYOUT_SEASON_5_V3 = dataclasses.replace(
    LAYOUT_SEASON_5_V2,
    name=LayoutName.SEASON_5_V3,
    grid_box=FractionalBox(left=0.4, top=0.5431, right=0.6, bottom=0.8667),
)

# Season 4 broadcast (1v1, every game DLC): the two players' gameplay feeds side by side
# across the top half, with the timer in a light box at the top center between them. Below
# the feeds a strip holds each player's claim-count badge at the frame's outer edge (red on
# the left, blue on the right) with their name next to it, and under that the two
# commentators' webcams flank the 5x5 grid at the bottom center, each with a nameplate
# along its bottom edge. There's no "BASE GAME"/"DLC" subtitle. Measured on a 1280x720
# frame at 00:10:00 of parkenharbor vs GrandPooBear (youtube.com/watch?v=5F-Lrmmti6I):
# grid inside a light border at x 469-811 / y 397-709, badges at y 390-417 (red x 6-41,
# blue x 1241-1271), timer box x 586-692 / y 6-42, nameplates x 199-449 / 830-1080 at
# y 652-682. The red badge's box is wider than the badge (x 6-44, y 390-417): measured
# tight around it, the "9" in the fixture OCR'd on only 6 of 16 frames; this box reads it
# on all 16 and still classifies as red.
#
# Every game is DLC, inferred from the squares (see pipeline.squares.infer_game_type), and
# a round robin match is a single game (see rules.validation.SEASON_FORMATS).
LAYOUT_SEASON_4 = Layout(
    name=LayoutName.SEASON_4,
    grid_box=FractionalBox(left=0.3672, top=0.5528, right=0.6336, bottom=0.9847),
    timer_box=FractionalBox(left=0.461, top=0.0125, right=0.5375, bottom=0.0556),
    game_type_box=None,
    score_box_red=FractionalBox(left=0.0047, top=0.5417, right=0.0344, bottom=0.5792),
    score_box_blue=FractionalBox(left=0.971, top=0.5444, right=0.993, bottom=0.5778),
    name_box_red=FractionalBox(left=0.033, top=0.5444, right=0.3, bottom=0.5778),
    name_box_blue=FractionalBox(left=0.7, top=0.5444, right=0.967, bottom=0.5778),
    commentator_box_left=FractionalBox(left=0.156, top=0.906, right=0.351, bottom=0.947),
    commentator_box_right=FractionalBox(left=0.649, top=0.906, right=0.844, bottom=0.947),
)

# Season 3 broadcast (2v2, both teammates in the base game, April 2024). The broadcast cuts
# between scenes, often every minute or two, and neither shows claim counts or names
# while a game is on (team names only appear on the end screen), so each scene is
# recognized by its timer reading (see board.is_gameplay_frame). Measured on 1280x720
# frames of Jo W Bush vs Monkey Ballers (youtube.com/watch?v=miUkeOU8ue8).
#
# The team view, most of the time: one team's two gameplay feeds across the top (team
# name above them), and the other team's two feeds small along the bottom on either side
# of the grid -- grid x 507-773 / y 416-657, timer box right of the casters' nameplates at
# x 704-842 / y 659-702, nameplates x 194-398 / 404-612 at y 672-703.
LAYOUT_SEASON_3_TEAM = Layout(
    name=LayoutName.SEASON_3_TEAM,
    grid_box=FractionalBox(left=0.3961, top=0.5778, right=0.6039, bottom=0.9125),
    timer_box=FractionalBox(left=0.5531, top=0.9194, right=0.6547, bottom=0.9708),
    game_type_box=None,
    score_box_red=None,
    score_box_blue=None,
    name_box_red=None,
    name_box_blue=None,
    commentator_box_left=FractionalBox(left=0.1516, top=0.9333, right=0.3109, bottom=0.9764),
    commentator_box_right=FractionalBox(left=0.3156, top=0.9333, right=0.4781, bottom=0.9764),
)

# The focus view: one player's feed large on the left with the other three small under
# it, and the grid on the right -- grid x 888-1259 / y 104-440, timer box above it at
# x 1009-1159 / y 57-101, the casters' nameplates under it at x 890-1072 / 1078-1258,
# y 445-461.
LAYOUT_SEASON_3_FOCUS = Layout(
    name=LayoutName.SEASON_3_FOCUS,
    grid_box=FractionalBox(left=0.6938, top=0.1444, right=0.9836, bottom=0.6111),
    timer_box=FractionalBox(left=0.7914, top=0.0833, right=0.9023, bottom=0.1361),
    game_type_box=None,
    score_box_red=None,
    score_box_blue=None,
    name_box_red=None,
    name_box_blue=None,
    commentator_box_left=FractionalBox(left=0.6953, top=0.6181, right=0.8375, bottom=0.6403),
    commentator_box_right=FractionalBox(left=0.8422, top=0.6181, right=0.9828, bottom=0.6403),
)

# Both scenes; the focus view first, since its grid is the larger, clearer read when a
# frame could pass as either. A four-feed scene before the game and the end screen have
# no live grid and match neither. The boxes here are the team view's, but unused (see
# Layout.scenes).
LAYOUT_SEASON_3 = dataclasses.replace(
    LAYOUT_SEASON_3_TEAM,
    name=LayoutName.SEASON_3,
    scenes=(LAYOUT_SEASON_3_FOCUS, LAYOUT_SEASON_3_TEAM),
)

# Season 2 broadcast (1v1, base game, September-November 2023). Both gameplay feeds side by
# side across the top with the timer in a box between them, the player names under the
# feeds (red left-aligned, blue right-aligned) with a claim-count badge under each, and the
# grid at the bottom center between the commentators' webcams, each with a nameplate.
# Measured on 1280x720 frames of Puppery vs blueberrybrioche
# (youtube.com/watch?v=CxKs4KtjClY): grid x 468-810 / y 398-708 (where Season 4's is),
# timer x 575-704 / y 19-54, badges x 32-65 / 1198-1230 at y 427-450, names at y 399-415,
# nameplates x 232-408 / 905-1072 at y 643-678 (right of each Twitch icon).
LAYOUT_SEASON_2 = Layout(
    name=LayoutName.SEASON_2,
    grid_box=FractionalBox(left=0.3656, top=0.5528, right=0.6328, bottom=0.9833),
    timer_box=FractionalBox(left=0.4492, top=0.0264, right=0.55, bottom=0.075),
    game_type_box=None,
    score_box_red=FractionalBox(left=0.025, top=0.5931, right=0.0508, bottom=0.625),
    score_box_blue=FractionalBox(left=0.9359, top=0.5931, right=0.9609, bottom=0.625),
    name_box_red=FractionalBox(left=0.0063, top=0.55, right=0.328, bottom=0.5806),
    name_box_blue=FractionalBox(left=0.672, top=0.55, right=0.9938, bottom=0.5806),
    commentator_box_left=FractionalBox(left=0.1813, top=0.8931, right=0.3188, bottom=0.9417),
    commentator_box_right=FractionalBox(left=0.707, top=0.8931, right=0.8375, bottom=0.9417),
)

# Season 2's second template, used from week 2 (bushy vs catalystz,
# youtube.com/watch?v=job6WgY9AzM, 2023-09-09) on: LAYOUT_SEASON_2 with the timer box moved up
# to the top edge, where Season 4's is, and the player names in dark text on yellow bars
# (which the name reader misses for red). The board, badges and nameplates didn't move.
# Measured on a 1280x720 frame at 00:45:00 of that video: timer box x 586-692 / y 8-42.
# Short focus scenes (one feed large, the grid on the right) cut in now and then; they match
# neither template, so a claim made during one is timed when the broadcast cuts back.
#
# Its badges are LAYOUT_SEASON_2's, so is_gameplay_frame can't tell the two apart and
# _detect_layout always picks LAYOUT_SEASON_2 (registered first) on this footage, whose timer
# box then reads nothing. Pass `--layout layout_season_2_v2` explicitly.
LAYOUT_SEASON_2_V2 = dataclasses.replace(
    LAYOUT_SEASON_2,
    name=LayoutName.SEASON_2_V2,
    timer_box=LAYOUT_SEASON_4.timer_box,
)

# Season 1 (2023), measured on 1280x720 frames of Ainrun vs Mitchriz
# (youtube.com/watch?v=MIxT4JcGxIk). Each player claims in their own colour, printed in
# their name too (player_colors) -- blue vs green, orange vs blue, ... changing from match to
# match. The broadcast cuts between two scenes, with the unboxed H:MM:SS clock at the top
# centre (x 579-703 / y 46-72) in both:
#
# The main scene: both feeds side by side, the grid at the bottom centre over them (x 466-815
# / y 403-719), the names either side of it at y 486-515, and the "Bingo Brawlers" wordmark
# at the top left (logo_box), which the focus scene doesn't print. From week 2 two claim
# counts sit stacked right of the grid, but in no fixed side order (each is in its player's
# colour), so they aren't read. The commentators' names are printed over their webcams and
# read poorly (often a fragment), so check them by hand; their boxes hold just the text line.
LAYOUT_SEASON_1_MAIN = Layout(
    name=LayoutName.SEASON_1_MAIN,
    grid_box=FractionalBox(left=0.3641, top=0.5597, right=0.6367, bottom=0.9986),
    timer_box=FractionalBox(left=0.4375, top=0.05, right=0.5625, bottom=0.1139),
    game_type_box=None,
    score_box_red=None,
    score_box_blue=None,
    name_box_red=FractionalBox(left=0.1094, top=0.6611, right=0.3633, bottom=0.7333),
    name_box_blue=FractionalBox(left=0.6367, top=0.6611, right=0.8438, bottom=0.7333),
    commentator_box_left=FractionalBox(left=0.0039, top=0.93, right=0.2578, bottom=0.99),
    commentator_box_right=FractionalBox(left=0.7422, top=0.93, right=0.9984, bottom=0.99),
    player_colors=True,
    logo_box=FractionalBox(left=0.09, top=0.03, right=0.21, bottom=0.14),
)

# The focus scene: one player's feed full screen, the grid at the top right corner at the
# same size (x 932-1280 / y 0-316), and only that player's name, at the bottom. Its grid is
# a hair narrower than the main scene's, so square text and names are read off the main
# scene (see extract._text_scene).
LAYOUT_SEASON_1_FOCUS = dataclasses.replace(
    LAYOUT_SEASON_1_MAIN,
    name=LayoutName.SEASON_1_FOCUS,
    grid_box=FractionalBox(left=0.7281, top=0.0, right=1.0, bottom=0.4389),
    name_box_red=None,
    name_box_blue=None,
    logo_box=None,
)

# Both scenes, the main one first: a frame passes as the focus scene on its clock alone, so
# the main scene's wordmark has to be checked before.
LAYOUT_SEASON_1 = dataclasses.replace(
    LAYOUT_SEASON_1_MAIN,
    name=LayoutName.SEASON_1,
    scenes=(LAYOUT_SEASON_1_MAIN, LAYOUT_SEASON_1_FOCUS),
)

# Keyed by Layout.name -- the registry pipeline.extract auto-detects against (STANDARD
# first, so a frame that happens to satisfy more than one check favors the far more
# common template) and a --layout CLI override looks up by name.
LAYOUTS: dict[LayoutName, Layout] = {
    layout.name: layout
    for layout in (
        STANDARD,
        LAYOUT_SEASON_6_FINAL,
        LAYOUT_PRE_SEASON_6,
        LAYOUT_SEASON_5,
        LAYOUT_SEASON_5_V2,
        LAYOUT_SEASON_5_V3,
        LAYOUT_SEASON_4,
        LAYOUT_SEASON_3,
        LAYOUT_SEASON_3_TEAM,
        LAYOUT_SEASON_3_FOCUS,
        LAYOUT_SEASON_2,
        LAYOUT_SEASON_2_V2,
        LAYOUT_SEASON_1,
        LAYOUT_SEASON_1_MAIN,
        LAYOUT_SEASON_1_FOCUS,
    )
}


def to_pixel_box(box: FractionalBox, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a fractional box to pixel (left, top, right, bottom) for a given frame size."""
    return (
        round(box.left * width),
        round(box.top * height),
        round(box.right * width),
        round(box.bottom * height),
    )


def crop(frame: np.ndarray, box: FractionalBox, width: int, height: int) -> np.ndarray:
    """Crop a frame to a fractional box. width/height must match frame.shape[1]/[0]."""
    left, top, right, bottom = to_pixel_box(box, width, height)
    return frame[top:bottom, left:right]


def grid_cell_box(row: int, col: int, grid: FractionalBox) -> FractionalBox:
    """Fractional box for grid cell (row, col), 0-indexed, within a 5x5 board."""
    cell_w = (grid.right - grid.left) / 5
    cell_h = (grid.bottom - grid.top) / 5
    left = grid.left + col * cell_w
    top = grid.top + row * cell_h
    return FractionalBox(left, top, left + cell_w, top + cell_h)
