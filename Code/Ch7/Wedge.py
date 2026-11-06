"""Challenge 2: The Wedge
=========================
"tilted halfway between lid and door"  -- Tortoise

Claim: with the shared state |Phi+> = cup/sqrt(2) and one box tilted by the
wedge -- the real rotation R_{pi/8} on Bob's leg -- every one of the four
question pairs is won with probability cos^2(pi/8) = (1 + 1/sqrt(2))/2, so
E = (+,+,+,-)/sqrt(2) and S = 2*sqrt(2) > 2, and no kind of evening is the
losing one.  R_{pi/8} is real but NOT symmetric (R^T = R_{-pi/8}), so sliding
it round the cup is a genuine transpose.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
from collections import Counter
from fractions import Fraction

import numpy as np
from dfp import BELL, CUP, I2, Picture, kron, matrix_of, random_matrix, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch07 as P

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
THETA = np.pi / 8
R = np.array([[np.cos(THETA), -np.sin(THETA)],
              [np.sin(THETA), np.cos(THETA)]], dtype=complex)


def bra(bit):
    e = np.zeros((1, 2), dtype=complex)
    e[0, bit] = 1
    return e


def alice_effect(x, a):
    """Alice: x=0 lid (<a|), x=1 door (<a|H)."""
    return bra(a) @ (H if x == 1 else I2)


def bob_effect(y, b):
    """Bob opens plainly after the wedge: y=0 door (<b|H), y=1 lid (<b|)."""
    return bra(b) @ (H if y == 0 else I2)


def blackboard():
    rng = np.random.default_rng(72)

    # --- R is real but lopsided: R^T = R_{-theta} is a genuine transpose -----
    Rminus = np.array([[np.cos(THETA), np.sin(THETA)],
                       [-np.sin(THETA), np.cos(THETA)]], dtype=complex)
    assert np.allclose(R.T, Rminus) and not np.allclose(R, R.T)
    assert np.allclose(R.conj(), R)                       # real

    # sliding round the cup is a TRANSPOSE -- shown first with a lopsided
    # complex box, where transpose, dagger and conjugate all differ
    A = random_matrix(rng)
    assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))    # unnormalized cup
    assert np.allclose(BELL, CUP / np.sqrt(2))            # the Bell state rides the 1/sqrt(2)
    lhs = kron(I2, A) @ CUP                               # A on Bob's (bottom) leg
    assert np.allclose(lhs, kron(A.T, I2) @ CUP)          # = A^T on Alice's: yes
    assert not np.allclose(lhs, kron(A.conj().T, I2) @ CUP)   # dagger: no
    assert not np.allclose(lhs, kron(A, I2) @ CUP)            # unchanged: no
    # for R the transpose is the wedge tilted the other way
    assert np.allclose(kron(I2, R) @ CUP, kron(R.T, I2) @ CUP)

    # --- all 16 amplitudes on |Phi+> with the wedge on Bob's leg -------------
    state = kron(I2, R) @ BELL                            # wedge = R on Bob's leg
    amp = {}
    for x in (0, 1):
        for y in (0, 1):
            for a in (0, 1):
                for b in (0, 1):
                    e = kron(alice_effect(x, a), bob_effect(y, b))
                    amp[(x, y, a, b)] = complex((e @ state).ravel()[0])

    # each question pair is a proper probability distribution
    for x in (0, 1):
        for y in (0, 1):
            total = sum(abs(amp[(x, y, a, b)]) ** 2 for a in (0, 1) for b in (0, 1))
            assert np.isclose(total, 1.0)

    # correlations, win probabilities, and S
    cos2 = np.cos(THETA) ** 2
    assert np.isclose(cos2, (1 + 1 / np.sqrt(2)) / 2)
    E = {}
    for x in (0, 1):
        for y in (0, 1):
            E[(x, y)] = sum((-1) ** (a ^ b) * abs(amp[(x, y, a, b)]) ** 2
                            for a in (0, 1) for b in (0, 1))
            # win iff a (+) b = x.y
            pwin = sum(abs(amp[(x, y, a, b)]) ** 2
                       for a in (0, 1) for b in (0, 1) if (a ^ b) == (x & y))
            assert np.isclose(pwin, cos2)                # every pair wins cos^2(pi/8)
    assert np.isclose(E[(0, 0)], +1 / np.sqrt(2))
    assert np.isclose(E[(0, 1)], +1 / np.sqrt(2))
    assert np.isclose(E[(1, 0)], +1 / np.sqrt(2))
    assert np.isclose(E[(1, 1)], -1 / np.sqrt(2))        # the one "unlike" pair
    S = E[(0, 0)] + E[(0, 1)] + E[(1, 0)] - E[(1, 1)]
    assert np.isclose(S, 2 * np.sqrt(2)) and S > 2       # past the sealed-note ceiling

    # --- nothing to read: each player's own answer is 1/2, 1/2 every pair ----
    for x in (0, 1):
        for y in (0, 1):
            pa0 = sum(abs(amp[(x, y, 0, b)]) ** 2 for b in (0, 1))
            pb0 = sum(abs(amp[(x, y, a, 0)]) ** 2 for a in (0, 1))
            assert np.isclose(pa0, 1 / 2) and np.isclose(pb0, 1 / 2)

    pictures(cos2)


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws a page of Aunt Hillary's score-book
#   \scorebook[view]{caption}                    (pdf/props/ch07.tex)
# and, when the case is opened, the hall table \halltable{state}{caption}.
# The Lua COMPUTES Bob's strokes by the pair game's rule from the four words
# of each row (the notes alike, save east door with west lid, when unlike);
# the rows are the identity card's (section 3, M1 and M2), a fiction (section
# 9): the page is checked for what it shows, and no probability is asserted
# from it.  tools/props_ch07.py transcribes the Lua; the calls are read out of
# this challenge's section of dialog.md, never copied.
#
# The state model: east is Alice's column (x: lid 0, door 1), west is Bob's
# (y: door 0, lid 1), each entry the way asked and the note heard (bright 0,
# dark 1); the stroke stands in the space between.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
EAST_X = {"lid": 0, "door": 1}
WEST_Y = {"door": 0, "lid": 1}
NOTE_BIT = {"bright": 0, "dark": 1}
KINDS = [("lid", "door"), ("lid", "lid"), ("door", "door"), ("door", "lid")]   # the hook's order


def check_transcription():
    """The Lua's stroke rule against the game (a structural check, not a second
    proof): it is the CHSH win rule a xor b = x.y under the card's labelling,
    and the very rule of the First's masks, the left box being Alice's."""
    ways, notes = ("lid", "door"), ("bright", "dark")
    for ea, eh, wa, wh in itertools.product(ways, notes, ways, notes):
        a, b, x, y = NOTE_BIT[eh], NOTE_BIT[wh], EAST_X[ea], WEST_Y[wa]
        assert P.won(ea, eh, wa, wh) == ((a ^ b) == (x & y))
    for m in P.MASKS["front"]:
        (ls,) = [s for s in m.sq if s.startswith("left ")]
        (rs,) = [s for s in m.sq if s.startswith("right ")]
        for fl, fr in itertools.product(P.FACES, repeat=2):
            kept = not P.broken(m, {ls: fl, rs: fr})
            assert kept == P.won(ls.split()[1], P.NOTE[fl], rs.split()[1], P.NOTE[fr])
    # the handwriting keeps apart the words a reader must tell apart on the page
    assert P.OUTLINES["lid"] != P.OUTLINES["door"] and P.OUTLINES["bright"] != P.OUTLINES["dark"]
    assert P.faults(P.scorebook, "sideways") != [] and P.faults(P.scorebook, "") == []


def pictures(cos2):
    check_transcription()
    p21, p22, p23, p24 = DIALOGUE.check_layout("Second", sizes=[1, 1, 1, 1])
    assert [c.macro for c in (p21, p22, p23, p24)] == ["scorebook"] * 3 + ["halltable"]
    for c in (p21, p22, p23, p24):
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS), c.caption
    assert not P.quantity_words(p24.caption)   # the wire-objects
    b21, b22, b23, t24 = (P.picture(c) for c in (p21, p22, p23, p24))
    sec = DIALOGUE.section("Second")

    a1, a2, a3, a4 = P.anchors(DIALOGUE, "Second")
    assert a1.startswith("**Anteater:** And here is the book he and Alice filled.")
    assert ("Each of them wrote down the way they were asked and the note they heard; "
            "Bob's stroke goes in the space between.") in a1
    assert a2.startswith("**Tortoise:** Let Bob draw a stroke for every evening they won")
    assert "to the last page of the book, where Bob wrote the same evenings out again sorted by kind" in a2
    assert a2.endswith("Lid-and-door here, lid-and-lid here, door-and-door, door-and-lid. Count each heap.")
    assert a3.startswith("**Anteater:** Ask the book.")
    assert "He turns back to the page the book lay open at and covers Alice's column with his hand" in a3
    assert "Inside, in three shaped nests, lie three small brass boxes." in a4
    paras = P.paragraphs(DIALOGUE, "Second")
    after = paras[[i for i, q in enumerate(paras) if "\\halltable" in q][0] + 1]
    assert after == "**Tortoise:** These never lose."            # the exit hook, right after

    # no picture among the COECKE carriers, the hooks that rule the book, or the wedge line
    first_block = sec.index("\\begin{center}")
    for line in ("Consider Bob, west of the hill", "Even a slant is a definite thing",
                 "tilted halfway between lid and door", "Let Bob draw the two columns",
                 "Let Bob draw the hundred rows"):
        assert sec.index(line) < first_block, line

    # P2.1: the open page, keyed east, won, west -- the stroke in the space between
    assert p21.keyed and b21.view == "open" and b21.labels == ("east", "won", "west")
    assert "Let Bob draw the two columns, then, east and west, with a space for the mark between." in sec
    assert len(b21.rows) == 12 and sum(b21.column("stroke")) == 10          # computed
    assert [i + 1 for i, r in enumerate(b21.rows) if not r.stroke] == [4, 9]
    assert {(r.east[0], r.west[0]) for r in b21.rows} == set(KINDS)          # every kind of evening
    for r in b21.rows:                                                      # the stroke is the game
        a, b = NOTE_BIT[r.east[1]], NOTE_BIT[r.west[1]]
        assert r.stroke == ((a ^ b) == (EAST_X[r.east[0]] & WEST_Y[r.west[0]]))
    assert sum(b21.column("stroke")) / len(b21.rows) > 3 / 4                # past the card's ceiling
    # the spoken tally is fiction (card section 9), chosen near cos^2(pi/8), past 75
    assert "Eighty-five wins in the hundred." in sec
    assert 85 / 100 > 3 / 4 and abs(85 / 100 - cos2) < 0.01

    # P2.2: Bob's sorted page -- four heaps of six, in the order the hook names them
    assert not p22.keyed and b22.view == "heaps" and len(b22.rows) == 24
    assert [(h[0], h[1]) for h in P.HEAPS] == KINDS
    for k, kind in enumerate(KINDS):
        heap = [r for r in b22.rows if r.heap == k]
        assert len(heap) == 6 and all((r.east[0], r.west[0]) == kind for r in heap)
        assert sum(r.stroke for r in heap) == 5                 # "About the same in each."
        assert 5 / 6 > 3 / 4                                    # every heap past the ceiling
    losses = [[i for i, r in enumerate(b22.rows) if r.heap == k and not r.stroke] for k in range(4)]
    assert [l[0] - 6 * k for k, l in enumerate(losses)] == [2, 4, 1, 5]   # a different row each
    assert "About the same in each." in sec and "There is no heap they gave up." in sec
    # "the same evenings out again sorted by kind": the open page's evenings, by kind,
    # are among the heaps' (not a stroke added or lost), its two losses among theirs
    for k, kind in enumerate(KINDS):
        page = Counter((r.east[1], r.west[1], r.stroke) for r in b21.rows
                       if (r.east[0], r.west[0]) == kind)
        heap = Counter((r.east[1], r.west[1], r.stroke) for r in b22.rows if r.heap == k)
        assert not page - heap, kind

    # P2.3: the open page, Alice's column under the Anteater's hand; Bob's alone
    assert not p23.keyed and b23.view == "east covered" and b23.covered == ("east",)
    assert b23.rows == b21.rows                                  # not a word of it changed
    west_dark = [r.west[1] == "dark" for r in b23.rows]
    assert sum(west_dark) == 6                                   # half and half
    for asked in ("lid", "door"):                                # whatever Alice was asked
        col = [r.west[1] == "dark" for r in b23.rows if r.east[0] == asked]
        assert len(col) == 6 and sum(col) == 3
    assert sum(r.east[1] == "dark" for r in b23.rows) == 7       # Alice's: near enough
    assert "Bob's own notes, whatever Alice was asked that evening: bright and dark, near enough half and half" in sec

    # P2.4: the hall table, keyed; the case open, the boxes in their nests; the
    # envelopes sealed, names up, left to right; the slip folded under its glass
    assert p24.keyed and t24.labels == ("the case", "the slip", "the envelopes")
    assert t24.trio == "case open" and t24.case_lid == "up" and t24.nests == ("box",) * 3
    assert t24.cloth == () and t24.deal == () and not t24.spent
    assert t24.envelopes == "sealed" and t24.envelope_faces == P.WHO
    assert t24.slip == "sealed" and t24.glass == "on the slip" and t24.slip_shows is None


# --------------------------------------------------------------------------
# The sketchbook.  The state is the cup with 1/sqrt(2) beside it; the wedge is
# R_{pi/8} on Bob's (bottom) leg.  A door effect <a|H is a Z-effect phase a;
# a lid effect <a| is an X-effect phase a; each one-legged piece costs a
# sqrt(2), paid back with add_power(-1).
# --------------------------------------------------------------------------
def r_chain(p, row, col, reverse=False):
    """R_{pi/8} as a chain, input side first; reversed gives R^T = R_{-pi/8}.
    The caller adds the phase -1/8 to the whole diagram's scalar."""
    if reverse:
        return [p.z(row, col, Fraction(1, 2)), p.x(row, col + 1, Fraction(1, 4)),
                p.z(row, col + 2, Fraction(-1, 2))]
    return [p.z(row, col, Fraction(-1, 2)), p.x(row, col + 1, Fraction(1, 4)),
            p.z(row, col + 2, Fraction(1, 2))]


def r_box(reverse=False):
    p = Picture()
    chain = r_chain(p, 0, 1, reverse)   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(p.inp(0), *chain, p.out(0))
    g = p.done()
    g.scalar.add_phase(Fraction(-1, 8))
    return g


def effect_node(p, row, col, kind, ans):
    """A door effect <a|H = Z-effect phase a; a lid effect <a| = X-effect phase a."""
    return p.z(row, col, ans) if kind == "door" else p.x(row, col, ans)


def game_pic(x, y, a, b):
    """Closed picture: cup, R on Bob's leg, Alice's effect, Bob's effect."""
    p = Picture()
    bend = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    a_eff = effect_node(p, 0, 3, "door" if x == 1 else "lid", a)
    p.wire(bend, a_eff)
    chain = r_chain(p, 1, 3)
    b_eff = effect_node(p, 1, 7, "door" if y == 0 else "lid", b)
    p.wire(bend, *chain, b_eff)
    g = p.done()
    g.scalar.add_phase(Fraction(-1, 8))     # R's phase
    g.scalar.add_power(-3)                   # cup 1/sqrt(2), two effects
    return g


def wedge_on_bob():
    """The cup with the wedge on Bob's (bottom) leg: a two-legged state."""
    p = Picture()
    bend = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    chain = r_chain(p, 1, 3)
    p.wire(bend, p.out(0))
    p.wire(bend, *chain, p.out(1))
    g = p.done()
    g.scalar.add_phase(Fraction(-1, 8))
    return g


def wedge_on_alice():
    """The same, with R turned the other way (R_{-pi/8}) on Alice's leg."""
    p = Picture()
    bend = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    chain = r_chain(p, 0, 3, reverse=True)
    p.wire(bend, *chain, p.out(0))
    p.wire(bend, p.out(1))
    g = p.done()
    g.scalar.add_phase(Fraction(-1, 8))
    return g


def lone_effect(kind, ans):
    """A single one-legged effect on an input wire (before paying its sqrt(2))."""
    p = Picture()
    e = effect_node(p, 0, 1, kind, ans)   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(p.inp(0), e)
    return p.done()


def sketchbook():
    # the one-legged pieces, literally, against NumPy: each stands for sqrt(2)
    # times its bra, which add_power(-1) later pays back.
    for ans in (0, 1):
        assert np.allclose(matrix_of(lone_effect("door", ans)), np.sqrt(2) * (bra(ans) @ H))
        assert np.allclose(matrix_of(lone_effect("lid", ans)), np.sqrt(2) * bra(ans))

    # the pieces check out on their own
    assert np.allclose(matrix_of(r_box()), R)                 # R_{pi/8}
    assert np.allclose(matrix_of(r_box(reverse=True)), R.T)   # reversed = transpose
    bell = Picture()
    b = bell.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    bell.wire(b, bell.out(0))
    bell.wire(b, bell.out(1))
    g = bell.done()
    g.scalar.add_power(-1)
    assert np.allclose(matrix_of(g).ravel(), BELL)            # cup/sqrt(2)

    # every closed picture reproduces its NumPy amplitude, scalar included
    state = kron(I2, R) @ BELL
    for x in (0, 1):
        for y in (0, 1):
            for a in (0, 1):
                for b in (0, 1):
                    e = kron(alice_effect(x, a), bob_effect(y, b))
                    want = complex((e @ state).ravel()[0])
                    got = complex(matrix_of(game_pic(x, y, a, b)).ravel()[0])
                    assert np.isclose(got, want)

    # the wedge slides: R on Bob's leg = R turned the other way on Alice's
    assert same_picture(wedge_on_bob(), wedge_on_alice())
    # ... and it is not the untilted cup, nor R the same way up on Alice's leg
    plain = Picture()
    pb = plain.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    plain.wire(pb, plain.out(0))
    plain.wire(pb, plain.out(1))
    assert not same_picture(wedge_on_bob(), plain.done())

    same_way = Picture()
    sb = same_way.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    chain = r_chain(same_way, 0, 3)
    same_way.wire(sb, *chain, same_way.out(0))
    same_way.wire(sb, same_way.out(1))
    sg = same_way.done()
    sg.scalar.add_phase(Fraction(-1, 8))
    assert not same_picture(wedge_on_bob(), sg)   # R (not R^T) on Alice's leg: no


if __name__ == "__main__":
    verdict("Wedge", blackboard, sketchbook)
