"""Challenge 2: The Ledger
========================
"At which level are you asking?"  -- Anteater

Claim: each half of a Bell pair, asked alone, is a fair coin in every basis:
(<b| (x) I) Bell = |b*> / sqrt(2), so every outcome has probability 1/2 and
the other half is left in |b*>; in every real basis the two halves agree.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction

import numpy as np
from dfp import BELL, CUP, I2, Picture, ket, kron, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch06 as P


def bra(v):
    """The effect <v|: conjugate the amplitudes and lay them in a row."""
    return np.asarray(v).conj().reshape(1, -1)


def joint_table(basis):
    """Probabilities of the four joint outcomes when both wires are asked in `basis`."""
    return [abs((kron(bra(top), bra(bottom)) @ BELL).item()) ** 2
            for top in basis for bottom in basis]          # top wire first


def blackboard():
    rng = np.random.default_rng(62)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))     # unnormalized cup
    assert np.allclose(BELL, CUP / np.sqrt(2))

    b = random_state(rng)
    assert not np.allclose(b, b.conj())                    # generic: not a real vector

    # one part alone (the blotter over the other column): a fair coin
    left_behind = (kron(bra(b), I2) @ BELL).ravel()        # ask the TOP wire for b
    assert np.allclose(left_behind, b.conj() / np.sqrt(2))       # |b*> / sqrt(2)
    assert not np.allclose(left_behind, b / np.sqrt(2))          # not |b> / sqrt(2)
    assert np.isclose(np.linalg.norm(left_behind) ** 2, 1 / 2)   # probability 1/2

    # the same from the other side
    left_behind = (kron(I2, bra(b)) @ BELL).ravel()        # ask the BOTTOM wire for b
    assert np.allclose(left_behind, b.conj() / np.sqrt(2))
    assert np.isclose(np.linalg.norm(left_behind) ** 2, 1 / 2)

    # the whole (the blotter lifted): in a real basis the columns are identical
    lid = [ket(0), ket(1)]
    side_door = [(ket(0) + ket(1)) / np.sqrt(2), (ket(0) - ket(1)) / np.sqrt(2)]
    assert np.allclose(joint_table(lid), [1 / 2, 0, 0, 1 / 2])
    assert np.allclose(joint_table(side_door), [1 / 2, 0, 0, 1 / 2])

    # ... and only in a real basis: here b* is the OTHER basis vector
    not_real = [(ket(0) + 1j * ket(1)) / np.sqrt(2), (ket(0) - 1j * ket(1)) / np.sqrt(2)]
    assert np.allclose(joint_table(not_real), [0, 1 / 2, 1 / 2, 0])

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the Crab's ledger with
#   \crabledger[blotter]{lines}{caption}      (pdf/props/ch06.tex)
# The Lua COMPUTES the right-hand column (the left word again) and lays the
# blotter over one column; tools/props_ch06.py transcribes it.  The calls are
# read out of this challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3): one page, A, under the blotter
# on the right, then on the left, then lifted; a line is one pair's joint
# outcome, left box first (the top wire).  The lid asks in one basis (high,
# low), the side door in another (bright, dark).
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("musicboxes", "walnutstring", "beadstring", "crabledger"))
DOORS = {"lid": (("high", "low"), [ket(0), ket(1)]),
         "side door": (("bright", "dark"),
                       [(ket(0) + ket(1)) / np.sqrt(2), (ket(0) - ket(1)) / np.sqrt(2)])}


def check_transcription():
    """The Lua's ledger against the blackboard (a structural check, not a
    second proof): the rule 'right = left on every finished line, by either
    door' is the pair's joint table having nothing off its diagonal, in the
    lid's basis and in the door's; and either column alone is a fair coin."""
    for words, basis in DOORS.values():
        table = np.array(joint_table(basis)).reshape(2, 2)       # [left word][right word]
        possible = {(words[i], words[j]) for i in range(2) for j in range(2)
                    if not np.isclose(table[i, j], 0)}
        page = P.crabledger("", ", ".join(words))
        assert possible == {(l.left, l.right) for l in page.lines}   # what the Lua writes
        assert np.allclose(table.sum(axis=1), [1 / 2, 1 / 2])        # the left boxes alone
        assert np.allclose(table.sum(axis=0), [1 / 2, 1 / 2])        # the right boxes alone
    assert P.faults(P.crabledger, "blotter middle", "high") != []
    assert P.crabledger("blotter right", "high").visible == ("left",)
    assert P.crabledger("blotter left", "high").visible == ("right",)
    assert P.crabledger("", "high").visible == ("left", "right")


def pictures():
    check_transcription()
    p21, p22, p23 = DIALOGUE.check_layout("Second", sizes=[1, 1, 1])
    for c in (p21, p22, p23):
        assert c.macro == "crabledger" and not c.keyed
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS), c.caption
        assert "chain" not in c.caption.lower()
    a21, a22, a23 = (P.picture(c) for c in (p21, p22, p23))

    a1, a2, a3 = P.anchors(DIALOGUE, "Second")
    assert "lays the sheet of blotting paper down the right-hand column" in a1
    assert a2.endswith("And the right boxes.") and "bury the left column" in a2
    assert a3.endswith("And now.") and "lifting the blotter clean off" in a3

    # one page, A, in all three: the blotter right, then left, then lifted
    assert a21.written == a22.written == a23.written
    assert (a21.blotter, a22.blotter, a23.blotter) == ("blotter right", "blotter left", "open")
    assert (a21.visible, a22.visible, a23.visible) == (("left",), ("right",), ("left", "right"))
    # page A is neither tonight's page nor the leaf after it (both pictured in the First)
    first = [P.picture(c) for b in DIALOGUE.blocks("First") for c in b
             if c.macro == "crabledger"]
    assert len(first) == 2 and all(a21.written != q.written for q in first)
    assert first[0].written != first[1].written

    # P2.1: the left boxes alone -- Achilles's tally is the drawn column
    left = a21.column("left")
    tally = re.search(r"(\w+) highs and (\w+) lows on this page", DIALOGUE.section("Second"))
    numbers = {w: n for n, w in enumerate("no one two three four five six seven eight".split())}
    assert (left.count("high"), left.count("low")) == \
        (numbers[tally.group(1).lower()], numbers[tally.group(2)]) == (5, 3)
    # ... and the leaf he peeks at is page C of the First: "bright, dark, dark, bright"
    assert "bright, dark, dark, bright" in DIALOGUE.section("Second")
    assert first[1].column("left")[:4] == ("bright", "dark", "dark", "bright")

    # P2.2: the right boxes alone -- the same run as P2.1's left column, computed
    assert a22.column("right") == left

    # P2.3: the blotter lifted -- the same word twice on every line
    assert all(l.left == l.right for l in a23.lines)
    assert "It's the same column twice." in DIALOGUE.section("Second")


def effect_on_top_leg(a):
    """A one-legged node plugged into the top leg of a bent wire."""
    p = Picture()
    bend = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(bend, p.z(0, 2, a))
    p.wire(bend, p.out(1))
    return p.done()


def state_on_bottom_wire(a):
    """The same one-legged node, as a state on the bottom wire."""
    p = Picture()
    p.wire(p.z(1, 2, a), p.out(1))
    return p.done()


def sketchbook():
    a = Fraction(1, 3)
    s = np.array([1, np.exp(1j * np.pi * a)])              # the node, read as a state
    assert np.allclose(matrix_of(state_on_bottom_wire(a)).ravel(), s)
    assert not np.allclose(s, s.conj())                    # generic: not real

    # the effect slides round the bend and comes out as a state -- exactly
    assert same_picture(effect_on_top_leg(a), state_on_bottom_wire(a))
    assert not same_picture(effect_on_top_leg(a), state_on_bottom_wire(-a))

    # PyZX does not conjugate a node when it turns it from a state into an
    # effect: the plugged-in node is the row (1, e^{ia}), which is <s*|, not <s|.
    # So the picture says "ask for s*, find s" -- the blackboard's b -> b*.
    assert np.allclose((kron(bra(s.conj()), I2) @ CUP).ravel(), s)
    assert np.allclose(matrix_of(effect_on_top_leg(a)).ravel(), s)


if __name__ == "__main__":
    verdict("Marginals", blackboard, sketchbook)
