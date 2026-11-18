"""Challenge 1: The Sealed Notes
================================
"Then three in four is simply the most anyone can do."  -- Achilles

Claim: sealed notes -- a note written inside each box before the players part --
win the CHSH game at most three times in four (equivalently S <= 2).  With the
card's labelling the four demands are a0(+)b0=0, a0(+)b1=0, a1(+)b0=0,
a1(+)b1=1; the left sides sum to 0 and the right sides to 1, so an odd number
of demands is always broken: at least one, never none.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
from fractions import Fraction

import numpy as np
from dfp import CUP, Picture, matrix_of, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch07 as P

# The game (Brunner II.2.4): questions x,y in {0,1}, answers a,b in {0,1},
# win iff a (+) b = x.y.  Card labelling: the one "unlike" pair is x=1,y=1.
QUESTIONS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def win(a, b, x, y):
    return (a ^ b) == (x & y)


def det_of(state):
    """c00*c11 - c01*c10, top wire first: zero exactly for a product of two."""
    c = np.asarray(state).reshape(2, 2)
    return c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0]


def blackboard():
    rng = np.random.default_rng(71)

    # --- the 16 deterministic strategies (a0,a1,b0,b1) -----------------------
    strategies = list(itertools.product((0, 1), repeat=4))
    assert len(strategies) == 16
    win_counts = []
    for a0, a1, b0, b1 in strategies:
        a = {0: a0, 1: a1}
        b = {0: b0, 1: b1}
        wins = sum(win(a[x], b[y], x, y) for (x, y) in QUESTIONS)
        win_counts.append(wins)
    # best is exactly three of four; never four, never even
    assert max(win_counts) == 3
    assert all(w in (1, 3) for w in win_counts)          # losses odd: 1 or 3
    assert win_counts.count(3) == 8 and win_counts.count(1) == 8   # 8 and 8

    # --- the number of broken pair-masks is odd for every one of the 16 -------
    # a pair-mask is broken when its demand fails; losses = broken masks
    losses = [4 - w for w in win_counts]
    assert all(L % 2 == 1 for L in losses)               # always odd
    assert losses.count(1) == 8 and losses.count(3) == 8

    # --- every strategy has |S| = 2 (deterministic CHSH) ---------------------
    for a0, a1, b0, b1 in strategies:
        a = {0: a0, 1: a1}
        b = {0: b0, 1: b1}
        E = {(x, y): (-1) ** a[x] * (-1) ** b[y] for (x, y) in QUESTIONS}
        S = E[(0, 0)] + E[(0, 1)] + E[(1, 0)] - E[(1, 1)]
        assert abs(S) == 2

    # --- one flip changes exactly two of the four verdicts -------------------
    for strat in strategies:
        a = {0: strat[0], 1: strat[1]}
        b = {0: strat[2], 1: strat[3]}
        base = [win(a[x], b[y], x, y) for (x, y) in QUESTIONS]
        for note in range(4):
            flipped = list(strat)
            flipped[note] ^= 1
            af = {0: flipped[0], 1: flipped[1]}
            bf = {0: flipped[2], 1: flipped[3]}
            new = [win(af[x], bf[y], x, y) for (x, y) in QUESTIONS]
            changed = sum(1 for u, v in zip(base, new) if u != v)
            assert changed == 2                          # never one, never three

    # --- a shared coin (mixture) cannot beat the best pure strategy ----------
    fractions = np.array(win_counts) / 4.0
    assert np.isclose(fractions.max(), 3 / 4)
    for _ in range(200):
        weights = rng.random(16)
        weights /= weights.sum()
        assert weights @ fractions <= 3 / 4 + 1e-12       # never more than 3/4

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws Achilles's card, the masks and the buttons
#   \notecard[opts]{state}{caption}                       (pdf/props/ch07.tex)
#   \cardsteps[opts]{state}{steps}{captions}              (a step strip of it)
# The Lua COMPUTES the buttons: one on every mask whose word the counters in
# its windows break; and, round 4 ("Show the process"), the state after each
# step (the masks laid one at a time, a counter turned), the step's marks (a
# laid mask's ghost on the card and its arc to its place below; a counter's
# flip; a button's ghost where it left and its arc where it crossed), and what
# the named wrong rule `mended` predicts.  tools/props_ch07.py transcribes it
# (proved against texlua; P.check_lua holds the Lua and the .tex to the text
# it was proved on); the calls are read out of this challenge's section of
# dialog.md and dialog.raw.md, never copied.
#
# The state model (identity card, sections 3 and 3e): on the front a column
# for each box (left, right), the lid square above the door square; white =
# bright = answer 0, black = dark = answer 1.  The card's labelling of the
# game: Alice (the left box) lid x=0, door x=1; Bob (the right box) door y=0,
# lid y=1.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
FRONT_Q = {"left lid": ("alice", 0), "left door": ("alice", 1),
           "right door": ("bob", 0), "right lid": ("bob", 1)}
BIT = {"white": 0, "black": 1}


def mask_question(m):
    """The question pair (x, y) a front mask reads: Alice's square and Bob's."""
    (_, x), (_, y) = sorted(FRONT_Q[s] for s in m.sq)
    return x, y


def verdicts(side, cs):
    return {m.name: P.broken(m, cs) for m in P.MASKS[side]}


def turned(before, after):
    """The squares whose counter was turned between two pictures."""
    assert before.keys() == after.keys()
    return sorted(s for s in before if before[s] != after[s])


def check_transcription():
    """The Lua's mask rule against the pair game (a structural check, not a
    second proof): each front mask reads one question pair, and it is kept
    exactly when that pair is won (a xor b = x.y); so "never all four kept" is
    the blackboard's "best 3 of 4", and the buttons are the losses."""
    P.check_lua()
    P.check_sites()
    front = P.MASKS["front"]
    assert [mask_question(m) for m in front] == QUESTIONS
    assert [m.rule for m in front] == ["alike", "alike", "alike", "unlike"]
    assert [m.name for m in front if m.rule == "unlike"] == [P.MASKALIAS["unlike"]]
    assert all(len(P.masks_over("front", s)) == 2 for s in P.squares("front"))
    nbuttons = []
    for faces in itertools.product(P.FACES, repeat=4):
        cs = dict(zip(P.squares("front"), faces))
        a = {FRONT_Q[s][1]: BIT[f] for s, f in cs.items() if FRONT_Q[s][0] == "alice"}
        b = {FRONT_Q[s][1]: BIT[f] for s, f in cs.items() if FRONT_Q[s][0] == "bob"}
        for m in front:
            x, y = mask_question(m)
            assert (not P.broken(m, cs)) == win(a[x], b[y], x, y)
        on = P.buttons("front", cs)
        assert 4 - len(on) == sum(win(a[x], b[y], x, y) for (x, y) in QUESTIONS)
        nbuttons.append(len(on))
        for s in cs:                                   # a flip: exactly two verdicts change
            flip = {**cs, s: "black" if cs[s] == "white" else "white"}
            v0, v1 = verdicts("front", cs), verdicts("front", flip)
            assert sorted(k for k in v0 if v0[k] != v1[k]) == sorted(P.masks_over("front", s))
    assert set(nbuttons) == {1, 3}                     # one or three, never none, never two
    assert nbuttons.count(1) == 8 and nbuttons.count(3) == 8
    # the macro's own refusals (its tex.error paths for the card and the masks)
    assert P.faults(P.notecard, "", "side; all white") != []            # a bad side
    assert P.faults(P.notecard, "", "front; all white; all black") != []   # one semicolon
    assert P.faults(P.notecard, "", "front; middle lid white") != []    # a bad square
    assert P.faults(P.notecard, "", "front; left lid grey") != []       # a bad face
    assert P.faults(P.notecard, "sideways", "front; all white") != []   # a bad view
    assert P.crashes(P.notecard, "under", "front; all white")           # under what? (the Lua stops)
    assert P.faults(P.notecard, "under two drawers", "front; all white") != []   # no such mask
    assert P.faults(P.notecard, "masks, button on two lids", "front; all white") != []
    assert P.faults(P.notecard, "masks", "front; left lid white") != []   # the masks read every square
    assert P.faults(P.notecard, "masks", "front; all white") == []
    assert P.faults(P.notecard, "under unlike", "front; all white") == []   # the alias
    # round 4's refusals on this section's steps
    assert P.faults(P.notecard, "masks; turn=left lid", "front; right lid white") != []   # nothing to turn
    assert P.faults(P.notecard, "masks; lay=next mask", "front; all white") != []         # past four
    assert P.faults(P.notecard, "card; lay=all", "front; all white") != []                # one at a time
    assert P.faults(P.notecard, "card; expects=mended", "front; all white") != []         # no turn
    assert P.faults(P.notecard, "card; lay=next mask; turn=left lid", "front; all white") != []   # one step
    assert P.faults(P.notecard, "masks; turn=left lid; expects=gaps", "front; all white") != []   # the rule
    assert P.faults(P.notecard, "masks; marks=off", "front; all white") != []             # marks need a step


def lay_frames(found):
    """Strip 1.2's frames: (picture, caption) four times."""
    (p12, frames), = found[1]
    return p12, frames


def pictures():
    check_transcription()
    found = P.check_section(DIALOGUE, "First", sizes=[1, 1, 2, 1], frames=[1, 4, 2, 1])
    (p11, [(c11, cap11)]), = found[0]
    p12, [(f02, cap02), (f03, cap03), (f04, cap04), (f05, cap05)] = lay_frames(found)
    (p13e, [(f06, cap06)]), (p13, [(f07, cap07)]) = found[2]
    (p14, [(c14, cap14)]), = found[3]
    assert [c.macro for c in (p11, p12, p13e, p13, p14)] == \
        ["notecard", "cardsteps", "notecard", "notecard", "notecard"]
    assert (cap11, cap02, cap03, cap04, cap05, cap06, cap07, cap14) == \
        ("the sealed notes", "the first mask", "the second", "the third", "one button",
         "as Achilles expects", "one counter turned", "turned again")
    lays = (f02, f03, f04, f05)
    first = DIALOGUE.section("First")

    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a1, a2, a3, a4 = DIALOGUE.anchors("First", path)
        assert a1.startswith("**Achilles:** *[laying a white counter in each]*")
        assert a1.endswith("three askings in four, won on the spot.")
        assert a2 == "**Tortoise:** A button, where a rule is broken. One broken, three kept."
        assert "He turns one counter, the right-lid square, white to black" in a3
        assert "there is still exactly one button on the card" in a3
        assert a4.startswith("**Achilles:** *[turning the left-lid counter to black]*")
        assert a4.endswith("it walks to the next square along and sits down again.")
    # the directions edited this round (D1, D2 with the Dramatist's amendment), in both files
    P.check_edits(DIALOGUE, "First", [
        ("*[The card as it stands: four white counters. She lays the first mask;",
         "*[She lays the first mask;"),
        ("*[The card as it was: four white counters, one button on the unlike mask. He turns one counter,",
         "*[The one turn, followed through. He turns one counter,")])

    # P1.1: the front, keyed; a column for each box, the lid above the door; all white
    assert p11.keyed and c11.side == "front" and c11.view == "card"
    assert c11.labels == ("lid", "door", "left box", "right box")
    assert "a column for each box, the lid square above the door square" in first
    assert P.place("front", "left lid") == (0, 0) and P.place("front", "right door") == (1, 1)
    assert c11.counters == {s: "white" for s in P.squares("front")} == c11.shown
    assert "white face up for a bright note, black for a dark one" in first
    # "three askings in four, won on the spot": the all-white card keeps three masks
    assert len(P.buttons("front", c11.counters)) == 1
    assert not c11.marked and c11.step is None

    # Block 1.2: the four masks laid one at a time and set out below the card.  Its
    # before-frame is P1.1 (from=off): the same card, the same counters, untouched.
    assert P.options(p12)["from"] == "off" and p12.args[0] == p11.args[0]
    spec = P.card_spec("front", c11.counters)
    assert [(s.opts, s.state) for s in P.frames_of(p12)] == [
        ("card; lay=next mask; keyroom=on", "front; all white"),
        ("one mask; lay=next mask; keyroom=on", spec),
        ("two masks; lay=next mask; keyroom=on", spec),
        ("three masks; lay=next mask; keyroom=on", spec)]
    assert all(f.counters == c11.counters and f.view == "masks" for f in lays)   # nothing turned
    assert [f.nmasks for f in lays] == [1, 2, 3, 4]
    # "one asking to a mask ... one at a time": each frame's step lays the next, its ghost on
    # the card and one arc from the card down to its place, in the order the Tortoise reads
    names = [m.name for m in P.MASKS["front"]]
    assert names == ["left lid right door", "two lids", "two doors", "left door right lid"]
    w, h = P.cardwh("front")
    for k, f in enumerate(lays):
        assert f.step == ("lay", "next mask") and f.pins == [] and f.expects is None
        assert [(g.what, g.name) for g in f.ghosts] == [("mask", names[k])]
        (lay,) = f.arcs
        assert lay.kind == "swing" and lay.tag == "lay " + names[k]
        sx, sy = P.mask_spots("front")[k]
        assert np.isclose(lay.start[1], P.MCARDY - h / 2 - 0.03)       # from the card's foot
        assert np.isclose(lay.end[1], sy + h / 2 + 0.03)               # to the spot's head
        assert np.sign(lay.end[0]) == np.sign(sx)                      # on the spot's side
        assert f.spots[names[k]] == divmod(k, 2)                       # two by two, in order
        assert list(f.spots) == names[:k + 1]
    assert "set each mask out below the card as she read it, two by two in the order laid" in first
    # "below the card": the card's foot above every spot's head
    assert all(sy + h / 2 < P.MCARDY - h / 2 for _, sy in P.mask_spots("front"))
    # the words the Tortoise reads, in order: ALIKE, ALIKE, ALIKE, UNLIKE
    assert [m.rule for m in P.MASKS["front"]] == ["alike", "alike", "alike", "unlike"]
    assert "the word on it is ALIKE" in first and "its word is UNLIKE" in first
    # no button until the fourth is down; then one, on the unlike mask, from her pocket:
    # no mark (F05's only marks are the fourth mask's)
    assert [f.buttons for f in lays] == [(), (), (), ("left door right lid",)]
    assert f05.spots["left door right lid"] == (1, 1)                 # the fourth laid
    assert "stands one on the fourth mask" in first and "From her pocket" in first
    # keyed: frame 1 only (Q5): the rows and the first mask's word; the other frames keep
    # the room as vertical phantoms, so the four stand level
    assert p12.keyed and f02.keyed and not any(f.keyed for f in lays[1:])
    assert f02.labels == ("lid", "door", "alike") and not f02.phantom
    assert all(f.phantom and f.showkey for f in lays[1:])
    assert [f.labels for f in lays[1:]] == [("lid", "door", "alike", "alike"),
                                            ("lid", "door", "alike", "alike", "alike"),
                                            ("lid", "door", "alike", "alike", "alike", "unlike")]
    assert all(n.endswith("{\\vphantom{" + l + "}};") for f in lays[1:]
               for n, l in zip(f.label_nodes, f.labels))

    # Block 1.3: the turn, Achilles's forecast beside it.  Both start from F05's state
    # (written {front; all white}: the same counters, the same one button).
    for c in (p13e, p13):
        side, cs = P.parse_card(c.args[0])
        assert side == "front" and cs == f05.counters
        assert P.buttons(side, cs) == f05.buttons
        assert P.options(c)["turn"] == "right lid" and P.options(c)["bare"] == "masks"
    # F06, the prediction: the named wrong rule, `mended`, its words the Lua's own comment's;
    # the counters as turned, no button: "unlike, mended" -- paler, unkeyed, no marks
    quotes = P.rule_quotes()
    assert quotes[0] in first and quotes[2] in first                   # said, and refuted
    assert first.index(quotes[0]) < first.index(quotes[2])
    assert P.is_prediction(p13e) and f06.expects == "mended" and f06.pale and not f06.keyed
    assert not f06.marked and f06.labels == () and f06.step == ("turn", "right lid")
    assert f06.counters == f07.counters
    assert f06.buttons == tuple(P.rule_mended("front", f05.counters, ["right lid"])) == ()
    # F07, the truth: the right lid black; the button leaves the unlike mask and crosses to
    # the two-lids mask -- one counter, two verdicts, still one button
    assert not p13.keyed and f07.view == "masks" and f07.expects is None and not f07.pale
    assert turned(f05.counters, f07.counters) == ["right lid"] and f07.counters["right lid"] == "black"
    assert f07.buttons == ("two lids",) != f06.buttons
    v2, v3 = verdicts("front", f05.counters), verdicts("front", f07.counters)
    assert {k for k in v2 if v2[k] != v3[k]} == {"left door right lid", "two lids"}   # two changed
    assert "Now the unlike mask has a white and a black — unlike, mended." in first
    assert "It has gone to the two-lids mask." in first
    # its marks: the counter's flip in its own window; the button's ghost where it stood; one
    # arc from there to the mask it stands on now
    assert f07.step == ("turn", "right lid") and f07.pins == []
    assert [(g.what, g.name) for g in f07.ghosts] == [("button", "left door right lid")]
    flip, cross = f07.arcs
    assert flip.tag == "flip right lid" and cross.tag == "button left door right lid -> two lids"
    x0, y0 = P.button_at("front", "left door right lid")
    x1, y1 = P.button_at("front", "two lids")
    assert np.isclose(np.hypot(cross.start[0] - x0, cross.start[1] - y0), 0.11)
    assert np.isclose(np.hypot(cross.end[0] - x1, cross.end[1] - y1), 0.12)
    assert np.allclose(f07.ghosts[0].at, (x0, y0))
    fx, fy = P.sqpos("front", "right lid")
    assert abs(flip.start[0] - fx) < 0.13 and abs(flip.start[1] - (fy + P.MCARDY)) < 0.01

    # P1.4: turned again (the left lid); the button walks to the left-lid-right-door mask.
    # A repeat of the turn: unmarked, as the card rules; its before-frame is F07.
    assert not p14.keyed and c14.view == "masks" and not c14.marked and c14.step is None
    assert turned(f07.counters, c14.counters) == ["left lid"] and c14.counters["left lid"] == "black"
    assert c14.buttons == ("left lid right door",)
    v4 = verdicts("front", c14.counters)
    assert {k for k in v3 if v3[k] != v4[k]} == {"two lids", "left lid right door"}
    assert "the left-lid-right-door mask has gone wrong instead, and the button is sitting on *that*" in first
    # the Anteater's fingertip: the left-door counter shows through two masks
    assert set(P.masks_over("front", "left door")) == {"two doors", "left door right lid"}
    assert "two masks at once — the two-doors mask and the unlike mask" in first
    # one button every time: "Count the broken masks, ever. One, or three."
    assert [len(c.buttons) for c in (f05, f07, c14)] == [1, 1, 1]

    # the log the pictures write, line for line as the build writes it
    rest = "front; left lid white, left door white, right lid white, right door white"
    turned_rl = "front; left lid white, left door white, right lid black, right door white"
    assert P.section_dumps(found) == [
        "notecard: [card] {front; all white} | " + rest,
        "cardsteps: strip of 4 frames",
        "notecard: [card; lay=next mask; keyroom=on] {front; all white} | " + rest + " | buttons none",
        "notecard: + step lay=next mask -> [one mask] {" + rest + "}",
        "notecard: + ghost mask left lid right door on the card; arc to its place below (1 of 4)",
        "notecard: [one mask; lay=next mask; keyroom=on] {" + rest + "} | " + rest + " | buttons none",
        "notecard: + step lay=next mask -> [two masks] {" + rest + "}",
        "notecard: + ghost mask two lids on the card; arc to its place below (2 of 4)",
        "notecard: [two masks; lay=next mask; keyroom=on] {" + rest + "} | " + rest + " | buttons none",
        "notecard: + step lay=next mask -> [three masks] {" + rest + "}",
        "notecard: + ghost mask two doors on the card; arc to its place below (3 of 4)",
        "notecard: [three masks; lay=next mask; keyroom=on] {" + rest + "} | " + rest
        + " | buttons left door right lid",
        "notecard: + step lay=next mask -> [masks] {" + rest + "}",
        "notecard: + ghost mask left door right lid on the card; arc to its place below (4 of 4)",
        "notecard: [masks; turn=right lid; expects=mended] {front; all white} | " + turned_rl + " | buttons none",
        "notecard: + step turn=right lid -> [masks] {" + turned_rl + "}",
        "notecard: + expects mended: buttons none (true: two lids)",
        "notecard: [masks; turn=right lid] {front; all white} | " + turned_rl + " | buttons two lids",
        "notecard: + step turn=right lid -> [masks] {" + turned_rl + "}",
        "notecard: + turned right lid white to black",
        "notecard: + ghost button left door right lid",
        "notecard: + arc button left door right lid -> two lids",
        "notecard: [masks] {front; all white, right lid black, left lid black} | front; left lid black, "
        "left door white, right lid black, right door white | buttons left lid right door"]


# --------------------------------------------------------------------------
# The sketchbook.  A sealed-note strategy is a PRODUCT picture -- two boxes
# that never share a wire -- so every joint number it makes is a product of two
# one-wire numbers.  The cup does not fall apart this way (Chapter 6's test).
# Rows are wires (row 0 is the top), and a state wired straight into an effect
# on the same row is a closed picture: one number.
# --------------------------------------------------------------------------
def one_wire(state_phase, effect_phase):
    """A one-wire state met by an effect: a single number."""
    p = Picture()
    s = p.z(0, 1, state_phase)   # PyZX draws this with the spiders you meet in Chapter 8.
    e = p.z(0, 2, effect_phase)
    p.wire(s, e)
    return p.done()


def product_with_effects(sp0, ep0, sp1, ep1):
    """Two disconnected one-wire closed pictures, stacked."""
    p = Picture()
    s0, e0 = p.z(0, 1, sp0), p.z(0, 2, ep0)   # PyZX draws this with the spiders you meet in Chapter 8.
    s1, e1 = p.z(1, 1, sp1), p.z(1, 2, ep1)
    p.wire(s0, e0)
    p.wire(s1, e1)
    return p.done()


def product_state(sp0, sp1):
    """Two disconnected one-wire states: two recipe cards."""
    p = Picture()
    p.wire(p.z(0, 1, sp0), p.out(0))   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(p.z(1, 1, sp1), p.out(1))
    return p.done()


def cup_state():
    """A wire bent round: the two legs are one connected picture."""
    p = Picture()
    bend = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(bend, p.out(0))
    p.wire(bend, p.out(1))
    return p.done()


def sketchbook():
    ph = (Fraction(1, 4), Fraction(1, 3), Fraction(2, 5), Fraction(3, 7))

    # the sealed-note number factors, exactly, into two one-wire numbers
    joint = matrix_of(product_with_effects(*ph)).ravel()[0]
    left = matrix_of(one_wire(ph[0], ph[1])).ravel()[0]
    right = matrix_of(one_wire(ph[2], ph[3])).ravel()[0]
    assert np.isclose(joint, left * right)

    # Chapter 6's test: two cards make a rank-one (det 0) table; the cup does not
    cards = matrix_of(product_state(ph[0], ph[2])).ravel()
    cup = matrix_of(cup_state()).ravel()
    assert np.isclose(det_of(cards), 0)          # a product: falls apart
    assert np.isclose(det_of(cup), 1)            # the (unnormalized) cup: does not
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))   # cup is unnormalized
    assert np.allclose(cup, CUP)
    assert not same_picture(product_state(ph[0], ph[2]), cup_state())


if __name__ == "__main__":
    verdict("SealedNotes", blackboard, sketchbook)
