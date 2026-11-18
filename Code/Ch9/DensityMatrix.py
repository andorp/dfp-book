"""Challenge II: A Stone That's Been Looked At  (density matrix + doubling)
=========================================================================
"A thing and its reflection."  -- Tortoise

Claim: a measured-but-unread state is a MIXTURE rho = sum_i p_i |psi_i><psi_i|
(measured, unread |+> is 1/2 I); and DOUBLING draws a state as itself beside its
CONJUGATE (mirror) copy -- a pure |psi> reshapes to |psi><psi|, and a unitary
acts as U on the ket copy and Ubar (the plain conjugate, not U^T or U^dagger) on
the mirror copy.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import copy
from fractions import Fraction

import numpy as np
import pyzx as zx
from pyzx.utils import VertexType as VT

from dfp import BELL, I2, ket, matrix_of, random_matrix, random_state, verdict
from dfp_props import Dialogue, words_only
import props_ch09 as P


def blackboard():
    rng = np.random.default_rng(92)

    # a density matrix is a weighted list of possibilities
    ps = np.array([0.7, 0.3])
    kets = [random_state(rng), random_state(rng)]
    rho = sum(p * np.outer(k, k.conj()) for p, k in zip(ps, kets))
    assert np.isclose(np.trace(rho), 1)               # unit trace: total probability
    assert np.allclose(rho, rho.conj().T)             # Hermitian
    evals = np.linalg.eigvalsh(rho)
    assert np.all(evals >= -1e-12)                    # positive

    # Achilles's measured-but-unread |+> is the maximally mixed state 1/2 I
    unread = 0.5 * np.outer(ket(0), ket(0)) + 0.5 * np.outer(ket(1), ket(1))
    assert np.allclose(unread, I2 / 2)
    # ... and it differs from the pure |+> he began with exactly in the off-diagonals
    # (the coherence the look destroyed)
    plus = (ket(0) + ket(1)) / np.sqrt(2)
    pure_plus = np.outer(plus, plus.conj())
    assert np.allclose(pure_plus, 0.5 * np.ones((2, 2)))
    assert not np.allclose(pure_plus, unread)
    assert np.allclose(np.diag(pure_plus), np.diag(unread))   # same diagonal, lost phase

    # a PURE state is the rank-one case; purity tr(rho^2) = 1 pure, 1/d mixed
    psi = random_state(rng)
    pure = np.outer(psi, psi.conj())
    assert np.isclose(np.trace(pure @ pure), 1) and np.linalg.matrix_rank(pure) == 1
    assert np.isclose(np.trace(unread @ unread), 0.5)         # = 1/d, d = 2

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the viewer and the slate under their
# cloths, and the black slab with a thing and its reflection,
#   \crabviewer[step=look]{state}{caption}   \notchslate[covered; step=cut unread]{notches}{caption}
#   \blackslab[<operations>; expects=tighter]{scene}{caption}   (pdf/props/ch09.tex)
# tools/props_ch09.py transcribes the Lua; the calls are read out of this
# challenge's section of dialog.md, never copied.  Round 4 ("Show the process"):
# the unseen look and its unread notch are drawn as steps (the state written is
# the state before; under the cloth no mark shows the swing), and the looked-at
# stone is drawn beside, paler, as Achilles's "tighter still" foresaw it.
#
# The state model (identity card, section 3): a reflection is TIGHT when the
# piece's own state -- the reduced state of that piece alone, as it stands -- is
# pure (tr rho^2 = 1) and SLACK when it is mixed; the gloss (restless or spent)
# shows the history.  The tie below runs over every scene of the slab and every
# piece drawn with a reflection that carries a state: the stones, and the needle
# and the flask standing ready (|0> each, before the works bind them).  The cloth
# and the works are left out: the cloth hides the far stone and its reflection
# (nothing of the far stone's state is drawn), and the works are a process, not a
# state.  A structural check against this file's blackboard, not a second proof.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
QUOTE = "A thing and its reflection."
DRAWN_STATE = ("stone", "needle", "flask")       # the kinds whose reflection shows a state


def purity(rho):
    return np.trace(rho @ rho).real


def looked_unread(rho):
    """A look taken and not read: sum_i P_i rho P_i (the weighted list)."""
    Ps = [np.outer(ket(i), ket(i).conj()) for i in (0, 1)]
    return sum(Pi @ rho @ Pi for Pi in Ps)


def halves(rho4):
    """The two reduced states of a two-stone state, far (top wire) first."""
    r = rho4.reshape(2, 2, 2, 2)                 # far, near, far', near'
    return np.einsum("abcb->ac", r), np.einsum("abad->bd", r)


def piece_states(rng):
    """Every scene's pieces with a drawn state, as this file models them."""
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())                       # non-real
    alone = np.outer(psi, psi.conj())                             # a stone cut alone, restless
    far, near = random_state(rng), random_state(rng)
    plain = np.outer(np.kron(far, near), np.kron(far, near).conj())   # two stones each cut alone
    one_piece = np.outer(BELL, BELL.conj())                       # a pair of one piece
    zero = np.outer(ket(0), ket(0).conj())                        # the needle, the flask, ready
    pf, pn = halves(plain)
    assert np.allclose(pf, np.outer(far, far.conj())) and np.allclose(pn, np.outer(near, near.conj()))
    _, bell_near = halves(one_piece)
    return {
        "restless stone": {"stone": alone},
        "looked-at stone": {"stone": looked_unread(alone)},
        "plain pair": {"far stone": pf, "near stone": pn},
        "pair of one piece": {"near stone": bell_near},
        "pair of one piece, strung": {"near stone": bell_near},
        "the look laid out": {"near stone": bell_near, "needle": zero, "flask": zero},
        "the look on a string": {},
    }


def tie_holds(scenes, states):
    """tight <=> pure, on every piece of every scene that shows a state."""
    for name, S in scenes.items():
        drawn = [t for t in P.visible(S).things if t.kind in DRAWN_STATE]
        if {t.name for t in drawn} != set(states[name]):
            return False
        for t in drawn:
            if (t.refl == "tight") != bool(np.isclose(purity(states[name][t.name]), 1)):
                return False
    return True


def check_faults():
    """The round-4 refusals the Second's steps meet: under the cloth the look is unseen
    and the notch unread; a prediction carries no marks; the tighter rule is said of the
    looked-at stone alone."""
    assert P.faults(P.crabviewer, "step=look high", "restless stone, dial covered") == ["vstep-covered"]
    assert P.faults(P.notchslate, "covered; step=cut high", "clean") == ["cut-covered"]
    assert P.faults(P.notchslate, "step=cut unread", "clean") == ["slate-unread"]
    assert P.faults(P.notchslate, "step=cut sideways", "clean") == ["cut-word"]
    assert P.faults(P.blackslab, "expects=tighter", "restless stone") == ["wrong-tighter"]
    assert P.faults(P.blackslab, "expects=tighter; marks=on; step=pull",
                    "looked-at stone") == ["expects-marks", "pull-none"]
    assert P.faults(P.blackslab, "expects=tight", "looked-at stone") == ["slab-rule"]


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    P.check_lua()
    P.check_sites()
    check_faults()
    found = P.check_section(DIALOGUE, "Second", sizes=[2, 1, 2], frames=[2, 1, 2])
    calls = [c for b in found for c, _ in b]
    p21, p22, p23, e24, p24 = calls
    assert [c.macro for c in calls] == ["crabviewer", "notchslate", "blackslab", "blackslab", "blackslab"]
    assert [c.keyed for c in calls] == [False, False, True, False, False]
    assert [P.is_prediction(c) for c in calls] == [False, False, False, True, False]
    assert [P.options(c) for c in calls] == [
        {"step": "look", "bare": ""}, {"step": "cut unread", "bare": "covered"}, {"bare": ""},
        {"expects": "tighter", "bare": ""}, {"bare": ""}]
    v21, s22, b23, w24, b24 = (P.picture(c) for c in calls)
    paras = DIALOGUE.paragraphs("Second")
    sec = DIALOGUE.section("Second")
    a1, a2, a3 = DIALOGUE.anchors("Second")
    f1, f2, f3 = DIALOGUE.following("Second")
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        here = [(n, o) for n, _, o in P.quote_places(DIALOGUE, path) if o == "Second"]
        assert sorted(here) == [("THE SLAB'S WRONG RULES", "Second")] * 2 + [
            ("THE SLATE'S STEP", "Second")] + [("THE VIEWER'S STEPS", "Second")] * 2

    # the whole slab: tight <=> pure, every scene, every piece with a state (L1 in force)
    rng = np.random.default_rng(921)
    states = piece_states(rng)
    assert set(states) == set(P.SCENES)
    assert tie_holds(P.SCENES, states)
    assert np.isclose(purity(states["looked-at stone"]["stone"]), np.sum(np.abs(
        np.diag(states["restless stone"]["stone"])) ** 2))       # p0^2 + p1^2 < 1
    assert np.allclose(states["pair of one piece"]["near stone"], I2 / 2)   # a Bell half: 1/2 I
    # NEGATIVE: the Lua before L1 drew the near stone of a pair of one piece TIGHT; on
    # that model the tie fails (a Bell half is mixed, tr rho^2 = 1/2)
    pre_l1 = copy.deepcopy(P.SCENES)
    for name in ("pair of one piece", "pair of one piece, strung", "the look laid out"):
        for it in pre_l1[name].items:
            if it.name == "near stone":
                it.refl = "tight"
    assert not tie_holds(pre_l1, states)
    assert np.isclose(purity(states["pair of one piece"]["near stone"]), 0.5)

    # B2.1: looked through with the dial and the slate under their cloths -- each a step
    # from the state the First left: the restless stone, the six notches
    covering = [q for q in paras if "draping one cloth over the needle's dial" in q]
    assert len(covering) == 1 and "another over the slate and its stylus arm" in covering[0]
    assert "never read" in a1 and "the stylus cuts a notch" in a1   # a mechanism cut it (R5)
    assert v21.before == ("restless", "covered") and (v21.dish, v21.needle) == ("spent", "covered")
    assert v21.moved.get("unseen") and v21.marks == ()             # the cloth hides the swing
    first_row = P.picture(DIALOGUE.blocks("First")[-1][-1])       # P1.4, as the First left it
    assert s22.view == "covered" and s22.shown == () and s22.marks == ()   # no notch shows
    assert s22.before == first_row.notches                         # the slate the First left
    assert s22.notches == first_row.notches + ("unread",)          # the row, and one unread after
    assert P.faults(P.notchslate, "", ", ".join(s22.notches)) == ["slate-unread"]   # never read
    # = round 3's P2.1 and P2.2, the state words written out
    r1 = P.crabviewer("", "spent stone, dial covered")
    r2 = P.notchslate("covered", ", ".join(s22.notches))
    assert (r1.view, r1.dish, r1.needle, r1.labels) == (v21.view, v21.dish, v21.needle, v21.labels)
    assert (r2.view, r2.notches, r2.shown, r2.labels) == (s22.view, s22.notches, s22.shown, s22.labels)
    assert f1.startswith("**Achilles:** So it has an answer, and I can't get at it.")

    # P2.3: a stone cut alone, restless; its reflection tight (keyed: a thing and its twin)
    assert "moving as one thing" in a2 and "cut alone, with no partner" in a2
    (t23,) = b23.shown.things
    assert (t23.kind, t23.gloss, t23.refl) == ("stone", "restless", "tight")
    assert b23.labels == ("stone", "reflection", "black slab") and b23.shown.frame == P.LOW_FRAME
    assert np.isclose(purity(states["restless stone"]["stone"]), 1)
    assert f2.startswith("**Tortoise:** " + QUOTE + " While the thing is restless, the two are tight")

    # B2.4: the looked-at, unread stone set where it stood; its reflection slack
    assert "gone loose" in a3 and "its reflection answers slack" in a3
    (t24,) = b24.shown.things
    assert (t24.kind, t24.gloss, t24.refl, t24.x) == ("stone", "spent", "slack", t23.x)
    assert purity(states["looked-at stone"]["stone"]) < 1 - 1e-6
    # NEGATIVE (the prediction N2, "tighter still"): the look LOWERS the purity
    assert purity(states["looked-at stone"]["stone"]) < purity(states["restless stone"]["stone"])
    assert "like a rower a beat behind the stroke" in f3
    # the prediction, paler, beside the truth: Achilles's rule draws the spent stone
    # TIGHT -- the same stone, still, at the same place; only its reflection differs.
    # The tie refutes it: a tight reflection on a mixed stone.
    assert "Then a spent stone will be tighter still — it has stopped trembling altogether." in sec
    assert sec.index("tighter still") < sec.index(a3)             # said before the rock
    (w,) = w24.shown.things
    assert w24.expects == "tighter" and w24.pale and w24.marks == () and w24.labels == ()
    assert (w.kind, w.gloss, w.x, w.mode) == (t24.kind, t24.gloss, t24.x, t24.mode)
    assert (w.refl, t24.refl) == ("tight", "slack")
    assert not tie_holds({"looked-at stone": w24.state}, states)  # tight, yet tr rho^2 < 1
    assert tie_holds({"looked-at stone": b24.state}, states)
    assert calls.index(e24) == calls.index(p24) - 1               # the prediction before the truth


def doubled_state(alpha):
    """A doubled pure state as a picture: a non-real one-qubit state on the top (ket)
    wire, and its CONJUGATE on the bottom (mirror) wire.  The equatorial state is a
    one-leg Z-spider Z(alpha) = (1, e^{i alpha pi}); its conjugate is the SAME picture
    with the phase negated, Z(-alpha).  (A restricted, non-real family; the general
    random-state coverage is the numpy check below.)"""
    g = zx.Graph()
    top = g.add_vertex(VT.Z, 0, 0, phase=Fraction(alpha))     # ket copy   |psi>
    bot = g.add_vertex(VT.Z, 1, 0, phase=Fraction(-alpha))    # mirror copy |psi*>
    ot = g.add_vertex(VT.BOUNDARY, 0, 1)
    ob = g.add_vertex(VT.BOUNDARY, 1, 1)
    g.add_edge((top, ot)); g.add_edge((bot, ob))
    g.set_inputs(()); g.set_outputs((ot, ob))
    return g


def sketchbook():
    # translation check (the picture evaluates to the NumPy matrix), not a second proof:
    # this file evaluates what doubling EQUALS (a state beside its conjugate), no rewrite.
    # the doubling picture: the two copies of a NON-REAL state genuinely differ
    a = Fraction(1, 3)
    v = matrix_of(doubled_state(a)).ravel()           # length-4 doubled vector
    # read the two copies off the two wires
    top = np.array([1, np.exp(1j * np.pi * float(a))])       # Z(alpha)  = |psi>  (unnormalized)
    bot = np.array([1, np.exp(-1j * np.pi * float(a))])      # Z(-alpha) = |psi*>
    assert not np.allclose(top, bot)                  # conjugate copy != ket copy
    assert np.allclose(bot, top.conj())               # mirror = CONJUGATE

    # the doubled pure state reshapes to the density matrix |psi><psi| (unnormalized;
    # trace 2, so rho = reshape / trace)
    dm = v.reshape(2, 2)
    assert np.allclose(dm, np.outer(top, top.conj()))
    assert np.isclose(np.trace(dm), 2)
    assert np.allclose(dm / np.trace(dm), np.outer(top, top.conj()) / 2)

    # a general random NON-REAL state, doubled by hand as |psi> (x) |psi*>:
    rng = np.random.default_rng(920)
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())
    doubled = np.kron(psi, psi.conj())
    assert np.allclose(doubled.reshape(2, 2), np.outer(psi, psi.conj()))   # = |psi><psi|
    # NEGATIVE: without the conjugate copy, |psi> (x) |psi> reshapes to psi psi^T,
    # which is NOT the density matrix for a non-real state
    assert not np.allclose(np.kron(psi, psi).reshape(2, 2), np.outer(psi, psi.conj()))

    # a MIXTURE is a sum of doubled pure states -- and it matches the blackboard rho
    ps, kets = np.array([0.6, 0.4]), [random_state(rng), random_state(rng)]
    rho = sum(p * np.outer(k, k.conj()) for p, k in zip(ps, kets))
    doubled_mix = sum(p * np.kron(k, k.conj()) for p, k in zip(ps, kets))
    assert np.allclose(doubled_mix.reshape(2, 2), rho)

    # doubled EVOLUTION is U (x) Ubar (conjugate) -- rho -> U rho U^dagger
    U = random_matrix(rng)
    U = np.linalg.qr(U)[0]                             # a non-real unitary
    assert not np.allclose(U, U.conj())
    evolved = np.kron(U, U.conj()) @ doubled           # (U (x) Ubar) vec(rho)
    assert np.allclose(evolved.reshape(2, 2), U @ np.outer(psi, psi.conj()) @ U.conj().T)
    # NEGATIVE: the transpose and the dagger on the mirror copy give the wrong rho'
    assert not np.allclose((np.kron(U, U.T) @ doubled).reshape(2, 2),
                           U @ np.outer(psi, psi.conj()) @ U.conj().T)
    assert not np.allclose((np.kron(U, U.conj().T) @ doubled).reshape(2, 2),
                           U @ np.outer(psi, psi.conj()) @ U.conj().T)


if __name__ == "__main__":
    verdict("DensityMatrix", blackboard, sketchbook)
