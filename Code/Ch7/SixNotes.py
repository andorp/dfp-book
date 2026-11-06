"""Challenge 4: Six Notes, Four Masks
======================================
"One mask wants even and three want odd, so the four together must come out odd."  -- Anteater

Claim (the GHZ paradox).  (i) On N = |000> + |111>:  X(x)X(x)X N = +N and
X(x)Y(x)Y N = Y(x)X(x)Y N = Y(x)Y(x)X N = -N, exactly.  (ii) No six numbers
x_A,y_A,x_B,y_B,x_C,y_C in {+-1} meet all four demands x_A x_B x_C = +1,
x_A y_B y_C = y_A x_B y_C = y_A y_B x_C = -1: the product of the four left
sides is +1 and of the right sides -1, so an odd number is always broken --
at most three of the four hold.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
from fractions import Fraction

import numpy as np
from dfp import CUP, Picture, ket, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch07 as P

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)     # S  = diag(1,  i)
Sd = np.array([[1, 0], [0, -1j]], dtype=complex)   # S* = diag(1, -i)
N = ket(0, 0, 0) + ket(1, 1, 1)


def bra(bit):
    e = np.zeros((1, 2), dtype=complex)
    e[0, bit] = 1
    return e


# the four demands, as products of the six written notes (+1 = bright, -1 = dark)
def demands(x, y):
    return [
        x[0] * x[1] * x[2] == 1,      # EVEN mask: all three doors
        x[0] * y[1] * y[2] == -1,     # ODD masks: one door, two drawers
        y[0] * x[1] * y[2] == -1,
        y[0] * y[1] * x[2] == -1,
    ]


def blackboard():
    # the node and cup are unnormalized (needed for the closed-picture values)
    assert np.isclose(np.linalg.norm(N), np.sqrt(2)) and np.isclose(np.vdot(N, N), 2)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))

    # --- (i) the four eigen-relations, with exact signs ----------------------
    assert np.allclose(kron(X, X, X) @ N, +N)
    assert np.allclose(kron(X, Y, Y) @ N, -N)
    assert np.allclose(kron(Y, X, Y) @ N, -N)
    assert np.allclose(kron(Y, Y, X) @ N, -N)
    # the operator identity behind the contradiction
    assert np.allclose(kron(X, Y, Y) @ kron(Y, X, Y) @ kron(Y, Y, X), -kron(X, X, X))

    # --- the drawer question is the door question behind S-dagger ------------
    plus, minus = bra(0) @ H, bra(1) @ H
    ket_pi = (ket(0) + 1j * ket(1)) / np.sqrt(2)      # |+i>, a NON-real state
    ket_mi = (ket(0) - 1j * ket(1)) / np.sqrt(2)      # |-i>
    assert not np.allclose(ket_pi, ket_pi.conj())     # genuinely complex
    assert np.allclose(plus @ Sd, ket_pi.conj().reshape(1, 2))    # <+i| = <+|S*
    assert np.allclose(minus @ Sd, ket_mi.conj().reshape(1, 2))   # <-i| = <-|S*
    # the conjugate matters: S (not S*) gives the wrong bra ...
    assert not np.allclose(plus @ S, ket_pi.conj().reshape(1, 2))
    # ... and |+i> transposed (not conjugated) is not the drawer bra either
    assert not np.allclose(ket_pi.reshape(1, 2), plus @ Sd)

    # a pair opened by the drawer on both sides disagrees: Y(x)Y cup = -cup
    assert np.allclose(kron(Y, Y) @ CUP, -CUP)

    # --- (ii) 64 cards: none meets four; at most three; odd many broken -------
    signs = (1, -1)
    kept_counts, broken_counts = [], []
    for xA, yA, xB, yB, xC, yC in itertools.product(signs, repeat=6):
        x, y = (xA, xB, xC), (yA, yB, yC)
        held = demands(x, y)
        kept_counts.append(sum(held))
        broken_counts.append(4 - sum(held))
    assert len(kept_counts) == 64
    assert max(kept_counts) == 3                       # never all four
    assert all(b % 2 == 1 for b in broken_counts)      # always odd
    assert broken_counts.count(1) == 32 and broken_counts.count(3) == 32

    # --- one flip changes exactly two of the four verdicts -------------------
    for card in itertools.product(signs, repeat=6):
        base = demands((card[0], card[2], card[4]), (card[1], card[3], card[5]))
        for note in range(6):
            flipped = list(card)
            flipped[note] *= -1
            new = demands((flipped[0], flipped[2], flipped[4]),
                          (flipped[1], flipped[3], flipped[5]))
            assert sum(1 for u, v in zip(base, new) if u != v) == 2

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the back of Achilles's card with the four
# trio-masks set out below it (\notecard[masks]{state}{caption}).  The Lua
# COMPUTES the buttons: one on every mask whose word (EVEN or ODD: the number
# of black counters in its windows) the counters break.  tools/props_ch07.py
# transcribes it; the calls are read out of this challenge's section of
# dialog.md, never copied.
#
# The state model (identity card, section 3): the back's columns are the boxes
# as they stand on the cloth (left, middle, right = legs A, B, C), the door row
# above the drawer row; door = the X question, drawer = the Y question; white
# = bright = +1, black = dark = -1.  The masks are laid, and drawn two by two,
# in the order even, odd left, odd middle, odd right.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
COL = {"left": 0, "middle": 1, "right": 2}
SIGN = {"white": 1, "black": -1}
QUESTION = {"door": X, "drawer": Y}
EIGEN = {"even": +1, "odd": -1}


def deal_of(m):
    ways = [None] * 3
    for s in m.sq:
        col, way = s.split(" ")
        ways[COL[col]] = way
    return tuple(ways)


def verdicts(cs):
    return {m.name: P.broken(m, cs) for m in P.MASKS["back"]}


def turned(before, after):
    assert before.keys() == after.keys()
    return sorted(s for s in before if before[s] != after[s])


def check_transcription():
    """The Lua's trio-masks against the four guarantees (a structural check, not
    a second proof): each mask's windows are one deal, and the product of the
    deal's questions sends the node's state N to +N for the EVEN mask and to -N
    for the ODD ones -- the four eigen-relations.  A mask is kept exactly when
    the product of its windows' faces is that eigenvalue, which is the
    blackboard's `demands`; so the buttons are part (ii), as buttons."""
    back = P.MASKS["back"]
    for m in back:
        op = kron(*(QUESTION[w] for w in deal_of(m)))
        assert np.allclose(op @ N, EIGEN[m.rule] * N)
        assert not np.allclose(op @ N, -EIGEN[m.rule] * N)
    assert all(len(P.masks_over("back", s)) == 2 for s in P.squares("back"))
    nbuttons = []
    for faces in itertools.product(P.FACES, repeat=6):
        cs = dict(zip(P.squares("back"), faces))
        for m in back:
            prod = np.prod([SIGN[cs[s]] for s in m.sq])
            assert (not P.broken(m, cs)) == (prod == EIGEN[m.rule])
        x = tuple(SIGN[cs[f"{c} door"]] for c in COL)
        y = tuple(SIGN[cs[f"{c} drawer"]] for c in COL)
        assert [not P.broken(m, cs) for m in back] == demands(x, y)
        nbuttons.append(len(P.buttons("back", cs)))
        # the Anteater's count: every counter shows through two masks, so the
        # blacks seen, summed over the four masks, are even whatever is written
        assert sum(sum(cs[s] == "black" for s in m.sq) for m in back) % 2 == 0
    assert sum(EIGEN[m.rule] == -1 for m in back) == 3        # one wants even, three odd
    assert set(nbuttons) == {1, 3}
    assert nbuttons.count(1) == 32 and nbuttons.count(3) == 32
    assert P.faults(P.notecard, "masks", "back; all white, left door empty") != []
    assert P.faults(P.notecard, "masks", "back; all black") == []


def pictures():
    check_transcription()
    p41, p42, p43 = DIALOGUE.check_layout("Fourth", sizes=[1, 1, 1])
    for c in (p41, p42, p43):
        assert c.macro == "notecard" and c.opt == "masks"
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS), c.caption
    c41, c42, c43 = (P.picture(c) for c in (p41, p42, p43))
    sec = DIALOGUE.section("Fourth")

    a1, a2, a3 = P.anchors(DIALOGUE, "Fourth")
    assert a1 == "**Tortoise:** *[standing a button on each of the three]* Three broken, one kept."
    assert a2.startswith("**Achilles:** Then a dark is wanted.") and a2.endswith("Two gone at a stroke!")
    assert "He turns the middle box's drawer counter, white to black" in a2
    assert a3.startswith("**Achilles:** Now mend the last.")
    assert "the left box's, one of the two under the single surviving button" in a3
    assert a3.endswith("The very same walk as the front of the card, six squares wide instead of four.")

    # the masks laid as l.322 names them: even, then odd left, odd middle, odd right
    assert [m.name for m in P.MASKS["back"]] == ["even", "odd left", "odd middle", "odd right"]
    assert [deal_of(m) for m in P.MASKS["back"]] == [
        ("door", "door", "door"), ("door", "drawer", "drawer"),
        ("drawer", "door", "drawer"), ("drawer", "drawer", "door")]
    order = ["The EVEN mask first, its three windows over the three door-squares",
             "Then the first ODD mask, one window on the left box's door-square",
             "The second ODD mask, its windows over the middle box's door and the left and right drawers",
             "The third, over the right box's door and the other two drawers"]
    assert [sec.index(t) for t in order] == sorted(sec.index(t) for t in order)

    # P4.1: six white, the masks set out below, keyed; buttons on the three ODD masks
    assert p41.keyed and c41.side == "back" and c41.view == "masks"
    assert c41.labels == ("door", "drawer", "even", "odd", "odd", "odd")
    assert c41.counters == {s: "white" for s in P.squares("back")}
    assert "six white counters" in sec
    assert c41.buttons == ("odd left", "odd middle", "odd right")       # three broken, one kept

    # P4.2: the middle box's drawer turned black; two buttons lift at once, one stays
    assert not p42.keyed and turned(c41.counters, c42.counters) == ["middle drawer"]
    assert c42.counters["middle drawer"] == "black"
    assert set(P.masks_over("back", "middle drawer")) == {"odd left", "odd right"}
    assert c42.buttons == ("odd middle",)                               # three become one
    survivor = dict((m.name, m) for m in P.MASKS["back"])["odd middle"]
    assert "middle door" in survivor.sq and "middle drawer" not in survivor.sq
    assert "One is left standing, on the ODD mask over the middle door, which cannot see this square." in sec

    # P4.3: the left box's drawer, one of the two drawers under the surviving
    # button; the button crosses to the mask next door
    assert not p43.keyed and turned(c42.counters, c43.counters) == ["left drawer"]
    assert [s for s in survivor.sq if s.endswith("drawer")] == ["left drawer", "right drawer"]
    assert set(P.masks_over("back", "left drawer")) == {"odd middle", "odd right"}
    assert "odd right" not in c42.buttons                               # already kept
    assert c43.buttons == ("odd right",)
    v2, v3 = verdicts(c42.counters), verdicts(c43.counters)
    assert {k for k in v2 if v2[k] != v3[k]} == {"odd middle", "odd right"}   # two verdicts changed
    (r0, k0), (r1, k1) = c43.spots["odd middle"], c43.spots["odd right"]
    assert r0 == r1 and abs(k0 - k1) == 1                               # side by side: next door
    assert "it crosses to the mask next door" in sec
    assert [len(c.buttons) for c in (c41, c42, c43)] == [3, 1, 1]

    # the flip after the Anteater's count (not pictured): the right box's door
    assert "Achilles turns one more counter — the right box's door" in sec
    assert P.buttons("back", {**c43.counters, "right door": "black"}) == ("even",)


# --------------------------------------------------------------------------
# The sketchbook.  The Theorem, Twice: two drawer questions are two S-daggers
# on two legs; they slide to one leg and merge into a Z; the Z flips one door
# answer.  A drawer-effect is a Z box (phase -1/2 = S-dagger) then a door-
# effect.  A Z box on a leg is a phase node of phase 1.  Every one-legged
# effect costs a sqrt(2), paid back with add_power(-1).  Closed diagrams here
# are evaluated with the "naive" strategy inside matrix_of / same_picture.
# --------------------------------------------------------------------------
def node_boxes(box_rows):
    """The node with a diagonal Z box (given phase) on the named legs, open."""
    p = Picture()
    dot = p.z(1, 1)   # the node: PyZX draws this with the spiders you meet in Chapter 8.
    for r in range(3):
        chain = [p.z(r, 3, box_rows[r])] if r in box_rows else []
        p.wire(dot, *chain, p.out(r))
    return p.done()


def closed_doors(a, b, c, z_on=None):
    """The node with all-door effects <a|H,<b|H,<c|H (Z-effects), and an optional
    Z box on leg z_on."""
    p = Picture()
    dot = p.z(1, 1)   # the node: PyZX draws this with the spiders you meet in Chapter 8.
    ans = (a, b, c)
    for r in range(3):
        col = 3
        chain = []
        if z_on == r:
            chain.append(p.z(r, col, Fraction(1)))       # a Z box
            col += 2
        eff = p.z(r, col, ans[r])                        # door effect <ans|H
        p.wire(dot, *chain, eff)
    g = p.done()
    g.scalar.add_power(-3)                               # three one-legged effects
    return g


def lone_door_effect(ans):
    """A single door-effect on an input wire (before paying its sqrt(2))."""
    p = Picture()
    e = p.z(0, 1, ans)   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(p.inp(0), e)
    return p.done()


def lone_z_box():
    """A single Z box (phase 1) on a wire."""
    p = Picture()
    b = p.z(0, 1, Fraction(1))   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(p.inp(0), b, p.out(0))
    return p.done()


def sketchbook():
    # the pieces, literally, against NumPy: a door-effect node is sqrt(2)<a|H
    # and the phase-1 node is exactly Z (the box that flips a door-answer).
    Hm = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    Zm = np.array([[1, 0], [0, -1]], dtype=complex)
    for ans in (0, 1):
        assert np.allclose(matrix_of(lone_door_effect(ans)), np.sqrt(2) * (bra(ans) @ Hm))
    assert np.allclose(matrix_of(lone_z_box()), Zm)

    # the rewrite, step by step: two S-daggers = one Z, and the Z may sit anywhere
    sd_on_BC = node_boxes({1: Fraction(-1, 2), 2: Fraction(-1, 2)})
    z_on_C = node_boxes({2: Fraction(1)})
    z_on_A = node_boxes({0: Fraction(1)})
    assert same_picture(sd_on_BC, z_on_C)               # slide-and-merge
    assert same_picture(z_on_C, z_on_A)                 # a Z sits on any leg
    assert not same_picture(sd_on_BC, node_boxes({}))   # ... but it is not nothing

    # all-door effects on N: 1/sqrt(2) on even parity, 0 on odd
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                val = complex(matrix_of(closed_doors(a, b, c)).ravel()[0])
                if (a + b + c) % 2 == 0:
                    assert np.isclose(val, 1 / np.sqrt(2))
                else:
                    assert np.isclose(val, 0)

    # with a Z on one leg (one drawer's worth) the parities exchange
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                val = complex(matrix_of(closed_doors(a, b, c, z_on=0)).ravel()[0])
                if (a + b + c) % 2 == 1:
                    assert np.isclose(val, 1 / np.sqrt(2))
                else:
                    assert np.isclose(val, 0)

    # the sign lives in that Z: the all-bright reading is certain without it,
    # impossible with it
    assert np.isclose(complex(matrix_of(closed_doors(0, 0, 0)).ravel()[0]), 1 / np.sqrt(2))
    assert np.isclose(complex(matrix_of(closed_doors(0, 0, 0, z_on=0)).ravel()[0]), 0)


if __name__ == "__main__":
    verdict("SixNotes", blackboard, sketchbook)
