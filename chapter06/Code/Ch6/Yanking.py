"""Challenge 3: The String
========================
"Take that end. I have this one. Pull."  -- Tortoise

Claim: a wire bent out, back and out again is a straight wire, whichever way
it is bent, with no stray number: (cap (x) I)(I (x) cup) = I = (I (x) cap)(cup (x) I).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
from dfp import BELL, CAP, CUP, I2, Picture, ket, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch06 as P


def blackboard():
    # conventions: cup = |00> + |11> and cap = <00| + <11|, both UNNORMALIZED
    assert np.allclose(CUP, ket(0, 0) + ket(1, 1))
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))
    assert np.allclose(CAP, CUP.conj().reshape(1, 4))       # the cap is the cup's bra
    assert np.allclose(BELL, CUP / np.sqrt(2))

    # the Z-bend: in on the top row, cup below, cap above, out on the bottom row
    assert np.allclose(kron(CAP, I2) @ kron(I2, CUP), I2)
    # the S-bend: the same with the rows in the other order
    assert np.allclose(kron(I2, CAP) @ kron(CUP, I2), I2)

    # why the cup carries no 1/sqrt(2): with the Bell state and its bra,
    # a wire with a kink in it would be HALF a wire
    bell_bra = BELL.conj().reshape(1, 4)
    kinked = kron(bell_bra, I2) @ kron(I2, BELL)
    assert np.allclose(kinked, I2 / 2)
    assert not np.allclose(kinked, I2)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the string at the walnut pair with
#   \walnutstring[view]{laid}{caption}        (pdf/props/ch06.tex)
# The Lua COMPUTES the pulled string: bends of opposite kinds run into one
# another and cancel, so a pulled string keeps (far bends) - (near bends) of
# them.  tools/props_ch06.py transcribes it, as legs and bends in the order the
# string runs; the calls are read out of this challenge's section of
# dialog.md, never copied.
#
# The state model (identity card, section 3): the window at the top (far), the
# company at the bottom (near); the S (out, back, out) is a far bend behind
# the left box, then a near bend in front of the right box; the Z the same
# with far and near exchanged.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("musicboxes", "walnutstring", "beadstring", "crabledger"))


def string_map(s, early):
    """A laid string as a map, read with time running away from the `early`
    side of the table: a bend on the early side is a cup (it makes a pair), a
    bend on the late side is a cap; a free end on the early side is an input,
    on the late side an output.  The legs, left to right, are the wires, top
    first.  (This is our reading of the Lua's words, not the Lua's own.)"""
    late = {"far": "near", "near": "far"}[early]
    order = s.names                           # every string drawn here runs left to right

    def layer(side, joined):
        out, i = [], 0
        while i < len(order):
            bend = [b for b in s.bends if b.side == side and order[i] in b.legs]
            if bend:
                assert bend[0].legs == (order[i], order[i + 1])   # neighbours only
                out.append(joined)
                i += 2
            else:
                out.append(I2)                    # a free end: a wire passing through
                i += 1
        return kron(*out)

    return layer(late, CAP) @ layer(early, CUP)


def check_transcription():
    """The Lua's pull rule against the blackboard (a structural check, not a
    second proof): the out-back-out string and the other-way-about string are
    the two bent wires of the yanking equation, one for each chirality; the pull
    leaves far - near = 0 bends, and the pulled straight string is the same map."""
    s_laid, _ = P.walnut_strand("S")
    z_laid, _ = P.walnut_strand("Z")
    for laid in (s_laid, z_laid):
        assert P.continuous(laid) and len(laid.legs) == 3 and len(laid.bends) == 2
        assert laid.count("far") == laid.count("near") == 1
        assert P.faults(P.pulled, laid, 0) == []              # the Lua's own arithmetic
        straight = P.pulled(laid, drawn=0)
        assert P.continuous(straight)
        assert straight.bends == () and straight.ends == laid.ends   # the ends stay put
        for early in ("near", "far"):             # whichever way time is read
            assert np.allclose(string_map(laid, early), I2)
            assert np.allclose(string_map(straight, early), I2)
    # read with time running up the page (the company's side first), the S is
    # (cap (x) I)(I (x) cup) and the Z is (I (x) cap)(cup (x) I)
    assert np.allclose(string_map(s_laid, "near"), kron(CAP, I2) @ kron(I2, CUP))
    assert np.allclose(string_map(z_laid, "near"), kron(I2, CAP) @ kron(CUP, I2))
    # the hairpin: one near bend, its ends at the boxes -- a pair, the cup
    hairpin, ends_at = P.walnut_strand("hairpin")
    assert ends_at == ("left box", "right box") and P.continuous(hairpin)
    assert np.allclose(string_map(hairpin, "near").ravel(), CUP)
    # the macro's own rules
    assert P.faults(P.walnutstring, "pulled", "hairpin") != []            # only S and Z pull
    assert P.faults(P.walnutstring, "pulled", "straight") != []
    assert P.faults(P.walnutstring, "", "out back out, the other way about") != []
    assert P.faults(P.walnutstring, "", "key") != []                      # no shape
    assert P.faults(P.walnutstring, "", "out back out, key") != []
    assert P.faults(P.walnutstring, "sideways", "hairpin") != []


FORBIDDEN_WORDS = ("nudged", "hairpin out", "hairpin close", "hairpin aside")


def pictures():
    check_transcription()
    p31, p32, p33, p34 = DIALOGUE.check_layout("Third", sizes=[1, 1, 1, 1])
    for c in (p31, p32, p33, p34):
        assert c.macro == "walnutstring"
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS), c.caption
        assert "chain" not in c.caption.lower()
        assert not any(w in P.split(c.args[0], ",") for w in FORBIDDEN_WORDS)   # never nudged
    w31, w32, w33, w34 = (P.picture(c) for c in (p31, p32, p33, p34))
    assert not any(w.key or w.nudged for w in (w31, w32, w33, w34))

    a1, a2, a3, a4 = P.anchors(DIALOGUE, "Third")
    assert a1.endswith("tell me what the other is the end of.")
    assert "round in a first bend behind it, towards the window" in a2
    assert "in front of the right one" in a2 and "and still one string" in a2
    assert "The kinks are gone" in a3
    assert a4.endswith("And bent like this?")
    assert "the first bend now facing down where it faced up" in a4

    # P3.1: the hairpin, keyed: the ends at the two boxes, one bend towards the company
    assert p31.keyed and w31.labels == ("the window", "the company")
    assert w31.shape == "hairpin" and w31.ends_at == ("left box", "right box")
    assert [b.side for b in w31.string.bends] == ["near"]
    assert np.allclose(string_map(w31.string, "near").ravel(), CUP)      # "There is a pair."

    # P3.2: out, back, out -- a far bend behind the left box, then a near one in front of the right
    assert not p32.keyed and w32.view == "laid" and w32.shape == "S"
    assert [(b.side, b.around) for b in w32.string.bends] == [("far", "left"), ("near", "right")]
    assert np.allclose(string_map(w32.string, "near"), I2)

    # P3.3: the same string pulled: far - near = 0 bends, one straight line (the yank)
    assert w33.view == "pulled" and p33.args == p32.args
    assert w33.string.bends == () and len(w33.string.legs) == 1
    assert w33.string.ends == w32.string.ends                      # the same two ends
    assert np.allclose(string_map(w33.string, "near"), string_map(w32.string, "near"))
    assert np.allclose(kron(CAP, I2) @ kron(I2, CUP), I2)

    # P3.4: the other way about -- the first bend now near, the second far;
    # its pull (spoken, not drawn) leaves 0 bends too: the other chirality
    assert w34.view == "laid" and w34.shape == "Z"
    assert [(b.side, b.around) for b in w34.string.bends] == [("near", "left"), ("far", "right")]
    assert w32.string.bends[0].side == "far" and w34.string.bends[0].side == "near"   # up, down
    assert w34.string != w32.string and w34.string.ends == ("far", "near")
    assert P.faults(P.pulled, w34.string, 0) == [] and P.pulled(w34.string, 0).bends == ()
    assert np.allclose(string_map(w34.string, "near"), I2)
    assert "Straight again." in DIALOGUE.section("Third")


def bent(row_in, row_out):
    """Three rows.  The wire comes in on `row_in`, goes right, doubles back along
    the middle row, and leaves on `row_out`:  in -- cap -- cup -- out."""
    p = Picture()
    cup = p.bend((1 + row_out) / 2, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    cap = p.bend((row_in + 1) / 2, 3)    # same kind of node; wired the other way round
    p.wire(p.inp(row_in), cap, cup, p.out(row_out))
    return p.done()


def straight():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def sketchbook():
    z_bend = bent(row_in=0, row_out=2)
    s_bend = bent(row_in=2, row_out=0)
    assert np.allclose(matrix_of(straight()), I2)
    assert same_picture(z_bend, straight())                 # pull: exact,
    assert same_picture(s_bend, straight())                 # both ways round

    # the normalized version: write 1/sqrt(2) beside the cup and beside the cap
    kinked = bent(row_in=0, row_out=2)
    kinked.scalar.add_power(-2)
    assert np.allclose(matrix_of(kinked), I2 / 2)
    assert not same_picture(kinked, straight())


if __name__ == "__main__":
    verdict("Yanking", blackboard, sketchbook)
