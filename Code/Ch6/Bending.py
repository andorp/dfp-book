"""Challenge 5: A Key in the Post
===============================
"...Oh. Every key is a pair."  -- Achilles

Claim: bending a box's input wire round turns the box A into the two-wire
state (A (x) I) cup, whose four amplitudes are A's four entries; bending the
wire back with a cap, and pulling straight, gives the box again.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import BELL, CAP, CUP, I2, ket, Picture, kron, matrix_of, random_matrix, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch06 as P


def blackboard():
    rng = np.random.default_rng(65)
    A = random_matrix(rng)
    assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())   # generic
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))                # unnormalized cup
    assert np.allclose(CAP, CUP.conj().reshape(1, 4))                 # ... and cap

    # the key becomes a pair: A on the TOP leg
    psi = kron(A, I2) @ CUP
    assert np.allclose(psi.reshape(2, 2), A)          # amplitude of |ik> is the entry A_ik
    assert not np.allclose(psi.reshape(2, 2), A.T)

    # the pitfall: with A on the BOTTOM leg the amplitudes are the transpose's entries
    psi_bottom = kron(I2, A) @ CUP
    assert np.allclose(psi_bottom.reshape(2, 2), A.T)
    assert not np.allclose(psi_bottom.reshape(2, 2), A)

    # the pair becomes the key again: bend back with a cap (and pull straight)
    assert np.allclose(kron(I2, CAP) @ kron(psi, I2), A)

    # nothing is lost either way: every two-wire state is some box, bent
    phi = random_state(rng, 4)
    assert np.allclose(kron(phi.reshape(2, 2), I2) @ CUP, phi)

    # the identity box is the cup itself
    assert np.allclose(kron(I2, I2) @ CUP, CUP)

    # product states are exactly the rank-one boxes (Challenge 1's test again)
    alpha, beta = random_state(rng), random_state(rng)
    rank_one = np.outer(alpha, beta)
    assert np.isclose(np.linalg.det(rank_one), 0)
    assert np.allclose(kron(rank_one, I2) @ CUP, np.kron(alpha, beta))
    assert not np.isclose(np.linalg.det(A), 0)        # a generic box: not a product

    # The dialogue's claim about the pair "with the key already turned in one of them"
    # (Achilles: "On a lid evening it would read high-low, low-high ... never agreeing").
    # Take the key to be the bit-flip.  By the lid (0/1) the pair now never agrees;
    # by the side door (+/-) it still always agrees -- hence "on a lid evening".
    flip = np.array([[0, 1], [1, 0]], dtype=complex)
    turned = (kron(flip, I2) @ BELL).ravel()
    plus, minus = np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)
    agree = lambda basis: sum(abs(np.kron(b, b).conj() @ turned) ** 2 for b in basis)
    assert np.isclose(agree([ket(0), ket(1)]), 0.0)   # lid: high-low, low-high
    assert np.isclose(agree([plus, minus]), 1.0)      # side door: still agreeing

    pictures(A)


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the key on the string at the walnut pair with
#   \walnutstring[view]{laid}{caption}        (pdf/props/ch06.tex)
# The Lua draws the key at one fixed spot (it "never moves"): riding the middle
# of the straight string, and on the leg that ends at the left walnut box when
# the string is a hairpin.  tools/props_ch06.py transcribes it; the calls are
# read out of this challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3): the walnut pair is wires 3
# (left) and 4 (right), the left one the top wire of the pair; the hairpin's
# ends rest at the boxes and its bend lies towards the company.  The picture
# does not show which way round the key sits; the words do: "a note comes in
# along the string at the left end" (the key's input side faces the string's
# left end), and "She carries the right end across and up to the left walnut
# box, and swings the left end down, round and up to the right one".
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("musicboxes", "walnutstring", "beadstring", "crabledger"))


def string_map(s, early, keys=None):
    """A laid string as a map, read with time running away from the `early`
    side of the table: a bend on the early side is a cup, on the late side a
    cap; a free end on the early side is an input, on the late side an output;
    the legs, left to right, are the wires, top first.  `keys` puts a box A on
    a leg, {leg: (A, facing)}, its input side facing the drawn path's "start" or
    "finish" end: where time meets the box input side first the leg carries A,
    where it meets the output side first it carries A turned end for end, A^T."""
    keys = keys or {}
    late = {"far": "near", "near": "far"}[early]
    order = s.names
    heading = {n: {"up": "far", "down": "near"}[d] for n, d in s.legs}

    def layer(side, joined):
        out, i = [], 0
        while i < len(order):
            bend = [b for b in s.bends if b.side == side and order[i] in b.legs]
            if bend:
                assert bend[0].legs == (order[i], order[i + 1])
                out.append(joined)
                i += 2
            else:
                out.append(I2)
                i += 1
        return kron(*out)

    def on(leg):
        if leg not in keys:
            return I2
        A, facing = keys[leg]
        with_path = heading[leg] == late          # the path runs with time on this leg
        return A if with_path == (facing == "start") else A.T

    return layer(late, CAP) @ kron(*[on(n) for n in order]) @ layer(early, CUP)


def check_transcription():
    """The key's spot, as the Lua draws it: the same in both pictures."""
    straight = P.walnutstring("", "straight, key")
    hairpin = P.walnutstring("", "hairpin, key")
    assert straight.key_spot == hairpin.key_spot == (P.KEYX, P.KEYY)
    assert straight.key_leg == "line" and np.isclose(straight.key_at, 1 / 2)   # the middle
    assert hairpin.key_leg == "left box" and 0 < hairpin.key_at < 1           # before the bend
    assert np.isclose(P.BOXX["left"], P.KEYX)                                  # under the left box
    # the macro's own rule: the key rides a straight string or a hairpin, nothing else
    assert P.faults(P.walnutstring, "", "out back out, key") != []
    assert P.faults(P.walnutstring, "", "the other way about, key") != []
    assert P.faults(P.walnutstring, "", "hairpin, key") == []


def pictures(A):
    check_transcription()
    (p51, p52), = DIALOGUE.blocks("Fifth")
    assert [p51, p52] == DIALOGUE.check_layout("Fifth", sizes=[2])
    for c in (p51, p52):
        assert c.macro == "walnutstring" and not c.keyed and c.opt == ""
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS), c.caption
        assert "chain" not in c.caption.lower()
        assert "nudged" not in P.split(c.args[0], ",")
    w51, w52 = P.picture(p51), P.picture(p52)

    (a1,) = P.anchors(DIALOGUE, "Fifth")
    assert "a note comes in along the string at the left end, passes through the key, " \
           "and leaves at the right end" in a1
    assert "She carries the right end across and up to the left walnut box, and swings " \
           "the left end down, round and up to the right one" in a1
    assert "The key has not moved a hair" in a1
    assert "slips the bead off the string" in a1              # no bead in these pictures

    # P5.1: the key at work -- the straight string, the key at the middle of the line
    assert w51.shape == "straight" and w51.key and w51.key_leg == "line"
    assert w51.string.ends == ("left", "right") and w51.string.bends == ()
    # in at the left end (the drawn line's start), through the key, out at the right: A itself
    # P5.2: a pair now -- the hairpin; the key at the SAME spot, on the leg to the left box
    assert w52.shape == "hairpin" and w52.key and w52.key_spot == w51.key_spot
    assert w52.key_leg == "left box" and w52.ends_at == ("left box", "right box")
    assert w52.string.names[0] == "left box"                  # the drawn path starts at box 3,
    # where the words put the string's RIGHT end: its left end (the key's input
    # side) is the path's finish, at box 4
    assert P.WIRE[("left", "walnut")] == 3 < P.WIRE[("right", "walnut")] == 4   # the top wire

    # what the pair is, from the picture and the words (a reading, checked here):
    # the key's output side faces box 3, and the bend lies on its input side --
    # it is the key's INPUT wire that is bent round to box 4: the top-leg state.
    psi = string_map(w52.string, "near", keys={w52.key_leg: (A, "finish")}).ravel()
    assert np.allclose(psi, kron(A, I2) @ CUP)                # A on the top leg, as in 6.5
    assert np.allclose(psi.reshape(2, 2), A)                  # box 3's word first: A's entries
    assert not np.allclose(psi.reshape(2, 2), A.T)            # not the transposed one
    assert not np.allclose(psi, kron(I2, A) @ CUP)            # not A on the bottom leg
    assert not np.isclose(np.linalg.det(psi.reshape(2, 2)), 0)   # a pair, not two cards
    # had the left end gone to box 3 instead, the pair would be the transposed one
    wrong = string_map(w52.string, "near", keys={w52.key_leg: (A, "start")}).ravel()
    assert np.allclose(wrong.reshape(2, 2), A.T) and not np.allclose(wrong, psi)
    # the key again: bend the left end (box 4, the input) back and pull -- the yank gives A
    assert "bending the left end back the way it came and laying the right end back down" \
        in DIALOGUE.section("Fifth")
    assert np.allclose(kron(I2, CAP) @ kron(psi, I2), A)
    # the bare hairpin (the Third's pair) is the identity key, bent
    assert np.allclose(string_map(w52.string, "near").ravel(), CUP)
    assert "The key again." in DIALOGUE.section("Fifth")
    # the Crab's key, the flip: the pair that never agrees by the lid
    flip = np.array([[0, 1], [1, 0]], dtype=complex)
    assert np.allclose(string_map(w52.string, "near", keys={w52.key_leg: (flip, "finish")}).ravel(),
                       kron(flip, I2) @ CUP)


KEY = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))   # a generic box, input side first


def thread(p, row, box):
    a, b, c = box
    return [p.z(row, 3, a), p.x(row, 4, b), p.z(row, 5, c)]


def key():
    """The box alone: one wire in, one wire out."""
    p = Picture()
    p.wire(p.inp(0), *thread(p, 0, KEY), p.out(0))
    return p.done()


def key_as_pair():
    """The input wire bent round: no inputs, two outputs."""
    p = Picture()
    bend = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(bend, *thread(p, 0, KEY), p.out(0))
    p.wire(bend, p.out(1))
    return p.done()


def pair_bent_back():
    """Three rows: the pair on rows 0-1, then a cap joining row 1 to a fresh
    input on row 2.  in -- cap -- cup -- box -- out."""
    p = Picture()
    cup, cap = p.bend(0.5, 1), p.bend(1.5, 7)
    p.wire(p.inp(2), cap, cup, *thread(p, 0, KEY), p.out(0))
    return p.done()


def sketchbook():
    M = matrix_of(key())
    assert not np.allclose(M, M.T) and not np.allclose(M, M.conj())   # generic
    # the pair carries the key's four numbers, top wire first
    assert np.allclose(matrix_of(key_as_pair()).reshape(2, 2), M)
    assert not np.allclose(matrix_of(key_as_pair()).reshape(2, 2), M.T)
    # bend back and pull: the key again, exactly
    assert same_picture(pair_bent_back(), key())


if __name__ == "__main__":
    verdict("Bending", blackboard, sketchbook)
