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
#   \notecard[view]{state}{caption}              (pdf/props/ch07.tex)
# The Lua COMPUTES the buttons: one on every mask whose word the counters in
# its windows break.  tools/props_ch07.py transcribes it; the calls are read
# out of this challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3): on the front a column for each
# box (left, right), the lid square above the door square; white = bright =
# answer 0, black = dark = answer 1.  The card's labelling of the game: Alice
# (the left box) lid x=0, door x=1; Bob (the right box) door y=0, lid y=1.
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
    assert P.faults(P.notecard, "under", "front; all white") != []      # under what?
    assert P.faults(P.notecard, "under two drawers", "front; all white") != []   # no such mask
    assert P.faults(P.notecard, "masks, button on two lids", "front; all white") != []
    assert P.faults(P.notecard, "masks", "front; left lid white") != []   # the masks read every square
    assert P.faults(P.notecard, "masks", "front; all white") == []
    assert P.faults(P.notecard, "under unlike", "front; all white") == []   # the alias


def pictures():
    check_transcription()
    p11, p12, p13, p14 = DIALOGUE.check_layout("First", sizes=[1, 1, 1, 1])
    for c in (p11, p12, p13, p14):
        assert c.macro == "notecard"
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS), c.caption
    c11, c12, c13, c14 = (P.picture(c) for c in (p11, p12, p13, p14))
    first = DIALOGUE.section("First")

    a1, a2, a3, a4 = P.anchors(DIALOGUE, "First")
    assert a1.startswith("**Achilles:** *[laying a white counter in each]*")
    assert a1.endswith("three askings in four, won on the spot.")
    assert a2 == "**Tortoise:** A button, where a rule is broken. One broken, three kept."
    assert "He turns one counter, the right-lid square, white to black" in a3
    assert "there is still exactly one button on the card" in a3
    assert a4.startswith("**Achilles:** *[turning the left-lid counter to black]*")
    assert a4.endswith("it walks to the next square along and sits down again.")

    # P1.1: the front, keyed; a column for each box, the lid above the door; all white
    assert p11.keyed and c11.side == "front" and c11.view == "card"
    assert c11.labels == ("lid", "door", "left box", "right box")
    assert "a column for each box, the lid square above the door square" in first
    assert P.place("front", "left lid") == (0, 0) and P.place("front", "right door") == (1, 1)
    assert c11.counters == {s: "white" for s in P.squares("front")} == c11.shown
    assert "white face up for a bright note, black for a dark one" in first
    # "three askings in four, won on the spot": the all-white card keeps three masks
    assert len(P.buttons("front", c11.counters)) == 1

    # P1.2: the same card, the masks set out below it, keyed; one button, on the unlike mask
    assert p12.keyed and c12.view == "masks" and p12.args == p11.args
    assert c12.labels == ("lid", "door", "alike", "alike", "alike", "unlike")
    assert c12.buttons == ("left door right lid",)
    assert c12.spots["left door right lid"] == (1, 1)               # the fourth laid
    assert "stands one on the fourth mask" in first
    assert "set each mask out below the card as she read it, two by two in the order laid" in first

    # P1.3: one counter turned (the right lid); the button moves to the two-lids mask
    assert not p13.keyed and c13.view == "masks"
    assert turned(c12.counters, c13.counters) == ["right lid"] and c13.counters["right lid"] == "black"
    assert c13.buttons == ("two lids",)
    v2, v3 = verdicts("front", c12.counters), verdicts("front", c13.counters)
    assert {k for k in v2 if v2[k] != v3[k]} == {"left door right lid", "two lids"}   # two changed
    assert "Now the unlike mask has a white and a black — unlike, mended." in first
    assert "It has gone to the two-lids mask." in first

    # P1.4: turned again (the left lid); the button walks to the left-lid-right-door mask
    assert not p14.keyed and c14.view == "masks"
    assert turned(c13.counters, c14.counters) == ["left lid"] and c14.counters["left lid"] == "black"
    assert c14.buttons == ("left lid right door",)
    v4 = verdicts("front", c14.counters)
    assert {k for k in v3 if v3[k] != v4[k]} == {"two lids", "left lid right door"}
    assert "the left-lid-right-door mask has gone wrong instead, and the button is sitting on *that*" in first
    # the Anteater's fingertip: the left-door counter shows through two masks
    assert set(P.masks_over("front", "left door")) == {"two doors", "left door right lid"}
    assert "two masks at once — the two-doors mask and the unlike mask" in first
    # one button every time: "Count the broken masks, ever. One, or three."
    assert [len(c.buttons) for c in (c12, c13, c14)] == [1, 1, 1]


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
