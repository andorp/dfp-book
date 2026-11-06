"""Challenge I: The Look That Forces  (projective measurement + the classical wire)
==================================================================================
"The look didn't read the stone, it *made* it."  -- Achilles

Claim: a projective measurement returns outcome i with the Born probability
p_i = <psi|P_i|psi> = |psi_i|^2 and leaves the qubit in P_i|psi>/sqrt(p_i); and
the read-spider Z(2->1) = sum_i |i><ii|, fed a state on one leg and its CONJUGATE
on the other, puts the Born-probability vector (|psi_0|^2, |psi_1|^2) on a single
classical wire -- exactly, scalar 1.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re

import numpy as np
import pyzx as zx
from pyzx.utils import VertexType as VT

from dfp import I2, ket, matrix_of, random_state, verdict
from dfp_props import Dialogue, words_only
import props_ch09 as P


def blackboard():
    rng = np.random.default_rng(91)

    P0 = np.outer(ket(0), ket(0).conj())      # |0><0|
    P1 = np.outer(ket(1), ket(1).conj())      # |1><1|
    assert np.allclose(P0 + P1, I2)           # a complete projective measurement
    assert np.allclose(P0 @ P0, P0) and np.allclose(P1 @ P1, P1)   # projectors

    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())   # generic: a NON-REAL state

    # the Born rule: outcome i with probability p_i = <psi|P_i|psi> = |psi_i|^2
    p0 = (psi.conj() @ P0 @ psi).real
    p1 = (psi.conj() @ P1 @ psi).real
    assert np.isclose(p0, abs(psi[0]) ** 2) and np.isclose(p1, abs(psi[1]) ** 2)
    assert np.isclose(p0 + p1, 1.0)           # a genuine coin: the weights sum to 1

    # post-measurement states are normalized, and the look is idempotent
    post0 = (P0 @ psi) / np.sqrt(p0)
    post1 = (P1 @ psi) / np.sqrt(p1)
    assert np.isclose(np.linalg.norm(post0), 1) and np.isclose(np.linalg.norm(post1), 1)
    # post-state is |i> up to a global phase: it lies wholly along the |i> axis
    assert np.isclose(abs(post0[0]), 1) and np.isclose(post0[1], 0)
    assert np.isclose(post1[0], 0) and np.isclose(abs(post1[1]), 1)
    # measuring a second time returns the same i (P_i|i> = |i>): the answer is settled
    assert np.allclose(P0 @ post0, post0) and np.allclose(P1 @ post1, post1)

    # |+> is the fair coin; |0> is the certain answer
    plus = (ket(0) + ket(1)) / np.sqrt(2)
    assert np.isclose((plus.conj() @ P0 @ plus).real, 0.5)
    assert np.isclose((ket(0).conj() @ P0 @ ket(0)).real, 1.0)

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the Crab's viewer and the slate its
# looks are cut on,
#   \crabviewer[view]{state}{caption}   \notchslate[view]{notches}{caption}
# (pdf/props/ch09.tex).  tools/props_ch09.py transcribes the Lua; the calls are
# read out of this challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3): a restless stone is a state the
# look has not settled (both outcomes possible); a look is the projective
# measurement {P0, P1}, outcome 0 = the needle HIGH (the right stop), 1 = LOW
# (the left); the stone is then spent -- the post-measurement basis state; the
# stylus cuts one notch an answer, only when a look throws the needle from the
# middle to a stop (high into the top edge, low into the foot).  A structural
# check of the pictures against this file's blackboard, not a second proof.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
QUOTE = "The look didn't read the stone, it *made* it."
PROJ = {"high": np.outer(ket(0), ket(0).conj()), "low": np.outer(ket(1), ket(1).conj())}


class ViewerModel:
    """The viewer and its slate, as this file models them (NumPy)."""

    def __init__(self):
        self.stone, self.needle, self.slate = None, "middle", []

    def fresh(self, psi):
        """A fresh stone in the dish (the last one tipped onto the heap: the dish
        emptied, the needle falls back to the middle; no notch)."""
        self.stone, self.needle = psi, "middle"

    def restless(self):
        p = [np.vdot(self.stone, PROJ[w] @ self.stone).real for w in ("high", "low")]
        return all(q > 1e-9 for q in p)          # neither answer settled yet

    def look(self, word):
        """A look that forces `word`: it must have a Born probability > 0; the
        stone is left in the answer's basis state; a notch is cut only if the
        needle was in the middle."""
        p = np.vdot(self.stone, PROJ[word] @ self.stone).real
        assert p > 1e-9                                           # a possible answer
        self.stone = PROJ[word] @ self.stone / np.sqrt(p)
        if self.needle == "middle":
            self.slate.append(word)
        self.needle = word
        return p


def check_faults():
    """The Lua's refusals on the viewer and the slate: each the state the chapter
    rules out."""
    assert P.faults(P.crabviewer, "", "restless stone, needle high") == ["viewer-restless"]
    assert P.faults(P.crabviewer, "", "spent stone") == ["viewer-spent"]   # the needle in the middle
    assert P.faults(P.crabviewer, "bare", "restless stone") == []           # a view, unused here
    assert P.faults(P.notchslate, "", "high, unread") == ["slate-unread"]  # read only under the cloth
    assert P.faults(P.notchslate, "", ", ".join(["low"] * 13)) == ["notch-many"]
    assert P.faults(P.notchslate, "", "") == ["notch-none"]


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    check_faults()
    calls = DIALOGUE.check_layout("First", sizes=[1, 2, 1])
    p11, p12, p13, p14 = calls
    blocks = DIALOGUE.blocks("First")
    for c in calls:
        assert words_only(c.caption, forbidden=P.CAPTION_FORBIDDEN), c.caption
        assert P.place_ok("First", c), c.text
    for b in blocks:
        assert len({P.height(c) for c in b}) == 1                 # no block mixes frames
    assert [c.macro for c in calls] == ["crabviewer", "crabviewer", "notchslate", "notchslate"]
    assert [c.keyed for c in calls] == [True, False, True, False]
    assert all(c.opt == "" for c in calls)                         # the viewer shut, the slate plain
    v11, v12, s13, s14 = (P.picture(c) for c in calls)
    sec = DIALOGUE.section("First")
    a1, a2, a3 = DIALOGUE.anchors("First")
    f1, f2, f3 = DIALOGUE.following("First")
    assert QUOTE in a3 and QUOTE in sec                            # the docstring's line

    # the model, run through the First as the dialogue tells it
    rng = np.random.default_rng(911)
    V = ViewerModel()
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())                       # a non-real stone
    V.fresh(psi)

    # P1.1: the viewer, keyed; a restless stone in the dish, the needle in the middle
    assert a1.startswith("**Crab:** And this is how you look.")
    assert ("a stoppered flask sunk in a well at the left of the base and a shallow dish "
            "at the right") in a1
    assert P.PLACES == {"flask": "left", "dish": "right"}
    assert (v11.view, v11.dish, v11.needle) == ("shut", "restless", "middle")
    assert v11.labels == ("eyepiece", "needle", "flask", "stone")
    assert V.restless() and V.needle == v11.needle and V.slate == []
    assert f1.startswith("**Crab:** The answer it forces is cut here.")   # the slate, after
    assert "at the top edge for high, at the foot for low" in f1

    # P1.2, P1.3: the look -- the needle hard against the right stop, one notch in the top edge
    assert "swings hard over to the high stop, on the right" in a2 and P.STOP["high"] == "right"
    assert "The stylus scratches a single notch into the slate's top edge." in a2
    assert P.EDGE["high"] == "top" and s13.labels == ("high", "low")
    p = V.look("high")
    assert 0 < p < 1                                              # it might have gone either way
    assert (v12.dish, v12.needle) == ("spent", V.needle) == ("spent", "high")
    assert not V.restless()                                       # spent: settled
    assert list(s13.shown) == V.slate == ["high"]

    # the re-look: the needle holds, the stylus cuts nothing, the answer is certain
    assert f2.startswith("**Achilles:** High.")
    assert "the needle stays hard against its stop, and the stylus cuts nothing more" in f2
    before = list(V.slate)
    assert np.isclose(V.look("high"), 1.0)                       # P_i P_i = P_i: probability 1
    assert V.slate == before and V.needle == "high"               # no notch, needle held
    assert np.isclose(np.vdot(V.stone, PROJ["low"] @ V.stone).real, 0)   # low is now impossible

    # P1.4: fresh stone after fresh stone, each read aloud as it is cut
    assert a3.startswith("**Achilles:** ...Oh.")
    spoken = a3[a3.rindex("]*") + 2:]
    heard = [w.lower() for w in re.findall(r"\b(High|Low)\.", spoken)]
    assert heard == ["high", "low", "low", "high", "low"]
    for word in heard:
        V.fresh(random_state(rng))                                # the dish emptied: needle middle
        assert V.restless()
        V.look(word)
    assert list(s14.shown) == V.slate == list(s13.shown) + heard   # P1.3's row is P1.4's head
    assert len(s14.notches) == 1 + len(heard)                      # a notch an answer, per fresh stone
    assert "fern" in f3 and f3.startswith("*[")                    # the frost, after the block


def read_spider():
    """The read-spider: a phase-free green Z-spider, two legs in, one classical
    leg out.  Z(2->1) = |0><00| + |1><11|.  (The Z-spider of Chapter 8, put to a
    new use -- it keeps the branches where the two legs agree.)"""
    g = zx.Graph()
    i1 = g.add_vertex(VT.BOUNDARY, 0, 0)
    i2 = g.add_vertex(VT.BOUNDARY, 1, 0)
    z = g.add_vertex(VT.Z, 0.5, 1)
    o = g.add_vertex(VT.BOUNDARY, 0.5, 2)     # the single CLASSICAL wire out
    g.add_edge((i1, z)); g.add_edge((i2, z)); g.add_edge((z, o))
    g.set_inputs((i1, i2)); g.set_outputs((o,))
    return g


def sketchbook():
    # translation check (the picture evaluates to the NumPy matrix), not a second proof:
    # this file evaluates what the read-spider primitive EQUALS, with no rewrite to derive.
    R = matrix_of(read_spider())              # shape (2, 4)
    # the read-spider is exactly sum_i |i><ii| -- scalar 1, nothing dropped
    target = np.outer(ket(0), ket(0, 0).conj()) + np.outer(ket(1), ket(1, 1).conj())
    assert np.allclose(R, target)

    # fed a state on one leg and its CONJUGATE on the other -- the doubled state,
    # built by hand as |psi> beside its mirror copy |psi*> -- the classical wire
    # carries the Born probabilities (|psi_0|^2, |psi_1|^2), exactly.
    for psi, born in [((ket(0) + ket(1)) / np.sqrt(2), np.array([0.5, 0.5])),
                      (ket(0), np.array([1.0, 0.0]))]:
        doubled = np.kron(psi, psi.conj())
        assert np.allclose((R @ doubled).ravel(), born)

    # a generic NON-REAL state: the conjugate leg is what turns amplitudes into
    # probabilities.  Feeding |psi> on BOTH legs (no conjugation) gives sum psi_i^2,
    # which is NOT the Born vector for a non-real state.
    rng = np.random.default_rng(910)
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())
    assert np.allclose((R @ np.kron(psi, psi.conj())).ravel(), np.abs(psi) ** 2)
    assert not np.allclose((R @ np.kron(psi, psi)).ravel(), np.abs(psi) ** 2)   # no conj: wrong
    # the classical wire carries a probability vector: real, non-negative, sums to 1
    out = (R @ np.kron(psi, psi.conj())).ravel()
    assert np.allclose(out.imag, 0) and np.all(out.real >= -1e-12)
    assert np.isclose(out.real.sum(), 1.0)


if __name__ == "__main__":
    verdict("Measurement", blackboard, sketchbook)
