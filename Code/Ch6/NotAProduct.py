"""Challenge 1: Two Recipe Cards
==============================
"Two cards won't do it. It needs one card for the pair."  -- Achilles

Claim: the Bell state is not a product of two one-wire states: a two-wire
state c00|00> + c01|01> + c10|10> + c11|11> is a product exactly when
c00*c11 - c01*c10 = 0, and for the Bell state that number is 1/2.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction

import numpy as np
import pyzx
from dfp import BELL, CUP, I2, Picture, ket, kron, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch06 as P


def det_of(state):
    """c00*c11 - c01*c10.  Top wire first: the TOP wire is the row index."""
    c = np.asarray(state).reshape(2, 2)
    return c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0]


def blackboard():
    rng = np.random.default_rng(61)

    # conventions: top wire first; the cup is unnormalized; Bell = cup / sqrt(2)
    assert ket(0, 1)[1] == 1 and ket(1, 0)[2] == 1
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))
    assert np.allclose(BELL, CUP / np.sqrt(2))

    # two recipe cards: a product state always has determinant 0
    alpha, beta = random_state(rng), random_state(rng)
    two_cards = np.kron(alpha, beta)
    assert np.isclose(det_of(two_cards), 0)

    # the pair: determinant 1/2 (and 1 for the unnormalized cup) -- no two cards
    assert np.isclose(det_of(BELL), 1 / 2)
    assert np.isclose(det_of(CUP), 1)

    # the converse, on a sample: determinant 0 means rank one, and a rank-one
    # table of amplitudes IS a product (column times row)
    rank = lambda state: np.linalg.matrix_rank(np.asarray(state).reshape(2, 2))
    assert rank(two_cards) == 1 and rank(BELL) == 2
    generic = random_state(rng, 4)
    assert not np.isclose(det_of(generic), 0) and rank(generic) == 2

    # the Bell state is nothing exotic: H on the top wire, then CNOT, on |00>
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert np.allclose(CNOT @ kron(H, I2) @ ket(0, 0), BELL)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the Crab's boxes and his ledger with
#   \musicboxes[string]{boxes}{caption}   \crabledger[blotter]{lines}{caption}
# (pdf/props/ch06.tex).  The Lua COMPUTES the ledger's right-hand column (a
# pair opened by the same door plays one word: the left word again) and draws
# a spent box shut; tools/props_ch06.py transcribes it.  The calls are read out
# of this challenge's section of dialog.md (tools/dfp_props.py), never copied,
# so an edit to the dialogue that breaks a picture breaks here.
#
# The state model (identity card, section 3): wire map left pearwood = 1 ...
# right ebony = 6; box 1 open at the first note and spent from then on; box 2
# lifted apart in the Crab's claw.  A ledger line is one pair's joint outcome,
# the left box's word then the right box's; the lid's words (high, low) are
# one pair of outcomes, the side door's (bright, dark) another.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("musicboxes", "walnutstring", "beadstring", "crabledger"))


def two_cards_disagree():
    """Achilles's two cards: two fair coins tossed separately.  How often
    would a ledger line built from them disagree?"""
    coin = np.array([1 / 2, 1 / 2])
    joint = np.outer(coin, coin)                                  # the two cards, side by side
    return joint[0, 1] + joint[1, 0]


def the_pair_disagrees():
    """The same, from the one card for the pair (the Bell state), by the lid."""
    joint = [[abs((kron(ket(i), ket(j)).conj().T @ BELL).item()) ** 2 for j in (0, 1)]
             for i in (0, 1)]                                     # top wire = left box
    return joint[0][1] + joint[1][0]


def check_transcription():
    """The macros' own rules (the Lua's tex.error), as the transcription has them."""
    assert P.faults(P.musicboxes, "", "table; left pearwood open") != []        # tray or row
    assert P.faults(P.musicboxes, "", "tray; left pearwood open; left pearwood spent") != []
    assert P.faults(P.musicboxes, "", "tray; left pearwood kept; right pearwood kept") != []
    assert P.faults(P.musicboxes, "", "tray; left pearwood spent; left pearwood kept") != []
    assert P.faults(P.musicboxes, "three bends", "tray") != []                  # no string on the tray
    assert P.faults(P.musicboxes, "", "row; right pearwood kept") != []         # kept: tray only
    assert P.faults(P.crabledger, "", "high, tonight low, high") != []          # tonight is last
    assert P.faults(P.crabledger, "", "low completed") != []                    # only tonight waits
    assert P.faults(P.crabledger, "", ", ".join(["high"] * 9)) != []            # eight to a page
    assert P.faults(P.crabledger, "blotter across", "high") != []
    # a spent box is drawn shut: only the lamp's glint on its lid is missing
    b = P.musicboxes("", "tray; left pearwood spent")
    assert b.boxes[1].looks_shut and not b.boxes[1].glint and b.boxes[2].glint


def pictures():
    check_transcription()
    p11, p12, p13, p14 = DIALOGUE.check_layout("First", sizes=[1, 2, 1])
    for c in (p11, p12, p13, p14):
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS), c.caption
        assert "chain" not in c.caption.lower()
    assert [c.macro for c in (p11, p12, p13, p14)] == \
        ["musicboxes", "musicboxes", "crabledger", "crabledger"]
    b11, b12, t13, c14 = (P.picture(c) for c in (p11, p12, p13, p14))

    # anchors: each block sits after the moment it shows
    a1, a2, a3 = P.anchors(DIALOGUE, "First")
    assert a1.endswith("a single high tone, clear as a struck glass.]*")      # the lid still up
    assert "you have had high" in a2 and a2.endswith("And nobody touches this one.")
    assert a3.endswith("I was told so very firmly.")
    after = DIALOGUE.section("First").split(p14.text)[1]
    assert after.split("**Achilles:**")[1].lstrip().startswith("They never disagree.")

    # P1.1: the tray, keyed; the left pearwood box (wire 1) open, the other five shut and full
    assert p11.keyed and b11.layout == "tray" and b11.string is None
    assert b11.labels == ("pearwood", "walnut", "ebony") and b11.labels_at == "beside the rows"
    assert b11.boxes[1].lid_up
    assert all(b11.state(w) == "shut" and b11.boxes[w].glint for w in range(2, 7))

    # P1.2: box 1 spent (drawn shut, no glint), box 2 held apart, boxes 3-6 as they were
    assert not p12.keyed and b12.layout == "tray"
    assert b12.state(1) == "spent" and b12.boxes[1].looks_shut and not b12.boxes[1].glint
    assert b12.state(2) == "kept" and b12.boxes[2].glint                  # shut and full
    assert all(b12.state(w) == "shut" for w in range(3, 7))

    # P1.3: page T, keyed -- four finished lines, then tonight's lone word in fresh ink
    assert p13.keyed and t13.labels == ("left boxes", "right boxes")
    assert t13.written == ("low", "high", "high", "low", "tonight high")
    assert all(l.right == l.left and l.ink == "old" for l in t13.lines[:4])
    tonight = t13.lines[-1]
    assert (tonight.left, tonight.right, tonight.ink) == ("high", None, "fresh")
    assert "a single high tone" in a1                     # the word is the note just heard

    # P1.4: page C, eight finished lines, each the same word twice, door words and lid words
    assert not c14.keyed and len(c14.lines) == 8 and c14.blotter == "open"
    assert all(l.right == l.left for l in c14.lines)
    assert {l.left for l in c14.lines} >= {"bright", "dark", "high", "low"}
    # the leaf after page A (the old page, pictured in the Second): Achilles's
    # spoken reading of A and of C's first line fits the drawn pages
    page_a = P.picture(DIALOGUE.blocks("Second")[0][0])
    reading = re.search(r"\*\*Achilles:\*\* ((?:(?:High|Low), (?:high|low)\. ?)+)",
                        DIALOGUE.section("First")).group(1)
    pairs = re.findall(r"(High|Low), (high|low)\.", reading)
    assert [(a.lower(), b) for a, b in pairs] == \
        [(l.left, l.right) for l in page_a.lines[:len(pairs)]] and len(pairs) == 5
    assert "Here's a \"bright, bright\"" in DIALOGUE.section("First")
    assert (c14.lines[0].left, c14.lines[0].right) == ("bright", "bright")
    assert c14.written != page_a.written != t13.written

    # the surprise, against the blackboard: two cards (two separate coins)
    # would write a disagreeing line half the time; the pair, never -- and no
    # drawn page has one
    assert np.isclose(two_cards_disagree(), 1 / 2)
    assert np.isclose(the_pair_disagrees(), 0)
    for page in (t13, c14):
        assert not any(l.right is not None and l.right != l.left for l in page.lines)


def bent_wire():
    """The pair as ONE connected picture: a wire bent round, both ends out."""
    p = Picture()
    bend = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(bend, p.out(0))
    p.wire(bend, p.out(1))
    return p.done()


def two_separate_states(a, b):
    """Two recipe cards: a DISCONNECTED picture, one one-wire state per row."""
    p = Picture()
    p.wire(p.z(0, 1, a), p.out(0))
    p.wire(p.z(1, 1, b), p.out(1))
    return p.done()


def prepared_pair():
    """|00>, then H on the top wire, then CNOT -- as a picture."""
    p = Picture()
    p.wire(p.x(0, 1), p.out(0))          # PyZX's phase-free one-legged x node
    p.wire(p.x(1, 1), p.out(1))          # is sqrt(2)|0>, so ...
    g = p.done()
    g.scalar.add_power(-2)               # ... divide by sqrt(2), twice: now |00>
    assert np.allclose(matrix_of(g).ravel(), ket(0, 0))
    circuit = pyzx.Circuit(2)
    circuit.add_gate("HAD", 0)           # row 0 is the top wire
    circuit.add_gate("CNOT", 0, 1)
    g.compose(circuit.to_graph())        # first the state, then the circuit
    return g


def sketchbook():
    # (i) a disconnected picture is a product, and is not the bent wire
    cards = two_separate_states(Fraction(1, 4), Fraction(1, 3))
    assert np.isclose(det_of(matrix_of(cards)), 0)
    assert np.isclose(det_of(matrix_of(bent_wire())), 1)
    assert not same_picture(cards, bent_wire())

    # (ii) the preparation circuit draws 1/sqrt(2) times the bent wire -- exactly
    bell = bent_wire()
    bell.scalar.add_power(-1)            # the number 1/sqrt(2) written beside it
    assert np.allclose(matrix_of(bell).ravel(), BELL)
    assert same_picture(prepared_pair(), bell)
    assert not same_picture(prepared_pair(), bent_wire())   # the 1/sqrt(2) matters


if __name__ == "__main__":
    verdict("NotAProduct", blackboard, sketchbook)
