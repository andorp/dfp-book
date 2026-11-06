"""Challenge IV: What the Twin Cannot Tell  (no-signalling / The Theorem, Twice)
==============================================================================
"There's nothing I can do here that reaches there."  -- Achilles

Claim: half a Bell pair is maximally mixed -- tr_2 |Phi+><Phi+| = 1/2 I, exact,
the 1/2 included -- and it stays 1/2 I after ANY trace-preserving channel on the
near (acted-on) half (no-signalling / causality).  In the picture: the unfolded
chain S-cup-M-cap-Mbar-cup-Sbar yanks in ONE pull to the far half cupped to its
own mirror, times 1/2; and the discard-cap swallows any doubled channel U (x) Ubar.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx.simplify as simp

from dfp import BELL, CAP, CUP, I2, ket, matrix_of, same_picture, Picture, verdict
from dfp_props import Dialogue, words_only
import props_ch09 as P


def random_kraus(rng, r=3, n=2):
    """A random trace-preserving channel: r non-real Kraus operators K_i with
    sum_i K_i^dagger K_i = I (normalized by a K -> K S^{-1/2} rescaling)."""
    Ks = [rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)) for _ in range(r)]
    S = sum(K.conj().T @ K for K in Ks)
    L = np.linalg.cholesky(np.linalg.inv(S))
    return [K @ L for K in Ks]


def apply_on_B(rho4, Ks):
    """Apply a channel (Kraus list) to the NEAR half B = qubit 2 (bottom wire)."""
    out = np.zeros_like(rho4)
    for K in Ks:
        out += np.einsum('bB,aBcD,dD->abcd', K, rho4, K.conj())
    return out


def apply_on_A(rho4, Ks):
    """Apply a channel (Kraus list) to the FAR half A = qubit 1 (top wire)."""
    out = np.zeros_like(rho4)
    for K in Ks:
        out += np.einsum('aA,ABcD,cC->aBCD', K, rho4, K.conj())
    return out


def blackboard():
    rng = np.random.default_rng(94)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))       # cups are UNNORMALIZED
    assert np.allclose(BELL, CUP / np.sqrt(2))               # |Phi+> = cup / sqrt2

    # evaluate BOTH sides: tr over the NEAR half (qubit 2) of |Phi+><Phi+| = 1/2 I
    rho = np.outer(BELL, BELL.conj()).reshape(2, 2, 2, 2)     # A, B, A', B'
    far = np.einsum('abcb->ac', rho)
    assert np.allclose(far, I2 / 2)                           # exact, the 1/2 included
    assert not np.allclose(far, I2)                           # NOT I: the 1/2 is the point

    # the chapter's own examples: do nothing / flip the near half with X / dephase it --
    # each leaves the far half 1/2 I (Achilles's "look, fold, leave")
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    for chan in [[I2], [X], [np.outer(ket(0), ket(0)), np.outer(ket(1), ket(1))]]:
        assert np.allclose(sum(K.conj().T @ K for K in chan), I2)        # trace-preserving
        assert np.allclose(np.einsum('abcb->ac', apply_on_B(rho, chan)), I2 / 2)

    # a random NON-REAL, non-unitary trace-preserving channel on the near half
    Ks = random_kraus(rng)
    assert np.allclose(sum(K.conj().T @ K for K in Ks), I2)   # trace-preserving
    assert not np.allclose(Ks[0], Ks[0].conj())              # non-real
    assert not np.allclose(Ks[0].conj().T @ Ks[0], I2)       # non-unitary
    far_after = np.einsum('abcb->ac', apply_on_B(rho, Ks))
    assert np.allclose(far_after, I2 / 2)                    # the far half NEVER moves

    # causality: a trace-preserving channel preserves the trace of any state
    M = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    sigma = M @ M.conj().T
    sigma /= np.trace(sigma)
    assert np.isclose(np.trace(sum(K @ sigma @ K.conj().T for K in Ks)), np.trace(sigma))

    # NEGATIVE: a NON-trace-preserving map (drop a Kraus operator) does move the far half
    far_bad = np.einsum('abcb->ac', apply_on_B(rho, Ks[:-1]))
    assert not np.allclose(far_bad, I2 / 2)

    # near vs. far (top wire first): the theorem is about acting on the NEAR half.  Act on
    # your OWN (far) half instead, and its marginal is Lambda(1/2 I), which for a non-unital
    # channel is NOT 1/2 I -- so which wire the channel hits genuinely matters.
    assert not np.allclose(sum(K @ K.conj().T for K in Ks), I2)          # non-unital
    far_own = np.einsum('abcb->ac', apply_on_A(rho, Ks))
    assert not np.allclose(far_own, I2 / 2)

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the black slab with a pair of one piece
# on it, the far stone under its cloth, the Tortoise's string laid across, the
# near stone folded and the string pulled,
#   \blackslab[operations]{scene}{caption}                 (pdf/props/ch09.tex)
# tools/props_ch09.py transcribes the Lua; the calls are read out of this
# challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3): a pair of one piece is the Bell
# state, the far stone on the top wire; its near stone alone is 1/2 I (slack,
# L1); the fold is the discard of the near half; the pulled string -- the far
# stone bent round to its own reflection, "half" chalked beside it -- is 1/2 the
# cup on the far stone and its mirror, = 1/2 I.  Whatever is done to the near
# stone before the fold, the pulled picture is the same, as the far marginal is
# 1/2 I after any trace-preserving channel on the near half.  A structural check
# against this file's blackboard, not a second proof.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
QUOTE = "There's nothing I can do here that reaches there."


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    calls = DIALOGUE.check_layout("Fourth", sizes=[1, 2, 1])
    p41, p42, p43, p44 = calls
    for c in calls:
        assert words_only(c.caption, forbidden=P.CAPTION_FORBIDDEN), c.caption
        assert P.place_ok("Fourth", c), c.text
    for b in DIALOGUE.blocks("Fourth"):
        assert len({P.height(c) for c in b}) == 1                 # no block mixes frames
    assert [c.macro for c in calls] == ["blackslab"] * 4
    assert [c.keyed for c in calls] == [True, False, False, False]
    assert [c.args[0] for c in calls] == ["pair of one piece"] + ["pair of one piece, strung"] * 3
    assert [c.opt for c in calls] == ["", "", "fold near stone", "fold near stone, pull"]
    b41, b42, b43, b44 = (P.picture(c) for c in calls)
    paras = DIALOGUE.paragraphs("Fourth")
    a1, a2, a3 = DIALOGUE.anchors("Fourth")
    f1, f2, f3 = DIALOGUE.following("Fourth")
    assert QUOTE in DIALOGUE.section("Fourth")                    # the docstring's line

    # the model: the Bell pair, far on the top wire
    rho = np.outer(BELL, BELL.conj()).reshape(2, 2, 2, 2)        # far, near, far', near'
    near_alone = np.einsum("abad->bd", rho)                       # the near stone, far covered
    far_alone = np.einsum("abcb->ac", rho)                        # the far stone, near folded
    assert np.allclose(near_alone, I2 / 2) and np.isclose(np.trace(near_alone @ near_alone).real, 0.5)

    # P4.1: the pair of one piece, keyed -- the far stone under its cloth at the far
    # end, the near stone by Achilles's hand, restless and SLACK (a Bell half is mixed)
    assert "no one has looked at this one" in a1 and "its reflection answers it slack" in a1
    assert "Those are the rules." in paras[paras.index(a1) - 1]   # D20 follows the rules
    cloth, near = b41.shown.things
    assert (cloth.kind, cloth.name, cloth.x) == ("cloth", "far stone", -0.8)
    assert (near.name, near.gloss, near.refl) == ("near stone", "restless", "slack")
    assert cloth.x < near.x and b41.labels == ("cloth", "near", "black slab")
    assert (near.refl == "tight") == bool(np.isclose(np.trace(near_alone @ near_alone).real, 1))

    # P4.2, P4.3: the string laid far to near, reflection beneath; the near stone folded
    assert "pinched away" in a2
    assert b42.shown.string == "laid" and b42.shown.things == b41.shown.things
    assert b42.shown.frame == b43.shown.frame == P.TALL_FRAME
    assert b43.shown.string == "folded" and b43.shown.things == (cloth,)   # only the cloth stands

    # P4.4: pulled -- one bend at the far stone's foot, nothing else on the black
    assert "chalks the one word: *half*" in a3
    assert b44.shown.string == "pulled" and b44.shown.things == (cloth,)
    assert f3.startswith("**Achilles:** The far half, bent round to its own reflection, "
                         "and a half beside it.")
    assert "a partner out of reach is loosening enough" in f3     # S5, the right naming
    # the pulled picture <-> 1/2 cup(far, far mirror) = 1/2 I = the far stone, near folded
    assert np.allclose(0.5 * CUP.reshape(2, 2), I2 / 2)
    assert np.allclose(0.5 * CUP.reshape(2, 2), far_alone)        # the chalked half drawn in
    assert not np.allclose(CUP.reshape(2, 2), far_alone)          # without the half: I, not 1/2 I
    # "any box you please on that middle leg": a tip before the fold pulls to the same
    # picture, as a non-real channel on the near half leaves the far half 1/2 I
    tipped = P.blackslab("tip near stone, fold near stone, pull",
                           "pair of one piece, strung")
    assert tipped.shown == b44.shown
    rng = np.random.default_rng(941)
    Ks = random_kraus(rng)
    assert not np.allclose(Ks[0], Ks[0].conj())
    assert np.allclose(np.einsum("abcb->ac", apply_on_B(rho, Ks)), far_alone)

    # the Lua refuses the moves the chapter rules out: nobody tips or folds the far
    # stone; the pull draws the slack through the fold, so the fold comes first
    assert P.faults(P.blackslab, "fold far stone", "pair of one piece, strung") == ["far-stone"]
    assert P.faults(P.blackslab, "tip far stone", "pair of one piece") == ["far-stone"]
    assert P.faults(P.blackslab, "pull", "pair of one piece, strung") == ["pull-laid"]
    assert P.faults(P.blackslab, "fold near stone, pull, pull",
                    "pair of one piece, strung") == ["pull-pulled"]
    assert P.faults(P.blackslab, "pull", "pair of one piece") == ["pull-none"]
    assert "washing line" in DIALOGUE.section("Fourth")           # S6 defines the pull


F = Fraction
PH = (F(1, 4), F(1, 3), F(2, 5))              # a generic, non-real, non-symmetric box U


def chain(p, row, phases):
    """A generic one-wire box as a z-x-z chain; returns (left, right) endpoints."""
    a, b, c = phases
    v0, v1, v2 = p.z(row, 2, a), p.x(row, 3, b), p.z(row, 4, c)
    p.wire(v0, v1, v2)
    return v0, v2


def zigzag(mbox=None, mbarbox=None):
    """The unfolded no-signalling chain, rows S=0, M=1, Mbar=2, Sbar=3:
    S -- cup(S,M) -- [box on M] -- cap(M,Mbar) -- [box on Mbar] -- cup(Mbar,Sbar) -- Sbar."""
    p = Picture()
    cupSM, capMM, cupSbMb = p.bend(0.5, 1), p.bend(1.5, 6), p.bend(2.5, 1)
    outS, outSb = p.out(0), p.out(3)
    if mbox is None:
        p.wire(outS, cupSM, capMM)
    else:
        l, r = chain(p, 1, mbox); p.wire(outS, cupSM, l); p.wire(r, capMM)
    if mbarbox is None:
        p.wire(capMM, cupSbMb, outSb)
    else:
        l, r = chain(p, 2, mbarbox); p.wire(capMM, l); p.wire(r, cupSbMb, outSb)
    return p.done()


def one_box(phases):
    """The box U alone, as a 1->1 map, so we can read its matrix."""
    p = Picture()
    l, r = chain(p, 0, phases)
    p.wire(p.inp(0), l); p.wire(r, p.out(0))
    return matrix_of(p.done())


def sketchbook():
    # --- THE ONE YANK ---  the doubled Bell state (scalar 1/2) with its near half
    # discarded pulls straight to the far half cupped to its own mirror, times 1/2.
    half = zigzag(); half.scalar.add_power(-2)               # cup = |00>+|11>, |Phi+> = cup/sqrt2
    p = Picture(); p.wire(p.out(0), p.bend(1.5, 1), p.out(3))
    far_cup = p.done(); far_cup.scalar.add_power(-2)         # 1/2 cup(S, Sbar)
    assert same_picture(half, far_cup)                       # one yank, scalar carried
    assert np.allclose(matrix_of(far_cup).reshape(2, 2), I2 / 2)   # = 1/2 I (maximally mixed)
    assert not same_picture(zigzag(), far_cup)               # without the 1/2 it is I, not 1/2 I

    # derive the yank with the rewrite rules (they carry the scalar unchanged)
    g = zigzag(); g.scalar.add_power(-2); simp.full_reduce(g)
    assert same_picture(g, far_cup)

    # --- CAUSALITY, non-real, in the picture ---  a doubled channel U (x) Ubar on the
    # near half is swallowed by the discard-cap: cap (U (x) Ubar) = cap, because
    # sum_i U_ik conj(U_il) = (U^dagger U)_lk = delta.  Ubar is the CONJUGATE, not U, U^T, U^dagger.
    U = one_box(PH)
    assert not np.allclose(U, U.conj()) and not np.allclose(U, U.T)     # non-real, non-symmetric
    assert np.allclose(U.conj().T @ U, I2)                              # a unitary box
    assert np.allclose((CAP @ np.kron(U, U.conj())).ravel(), CAP.ravel())          # Ubar: swallowed
    assert not np.allclose((CAP @ np.kron(U, U)).ravel(), CAP.ravel())             # bare U: no
    assert not np.allclose((CAP @ np.kron(U, U.T)).ravel(), CAP.ravel())           # transpose: no
    assert not np.allclose((CAP @ np.kron(U, U.conj().T)).ravel(), CAP.ravel())    # dagger: no

    # --- the same, fully in-picture ---  the box U rides the near half M; its conjugate
    # reflection rides the mirror leg Mbar (which the cap reads as Ubar^T = U^dagger, the
    # chapter's slide -- in phase-free chains, the reversed and negated phase sequence).
    a, b, c = PH
    conj_reflection = (-c, -b, -a)
    assert np.allclose(one_box(conj_reflection), U.conj().T)            # = U^dagger, as drawn
    channel = zigzag(mbox=PH, mbarbox=conj_reflection)
    assert same_picture(channel, zigzag())                             # far half unchanged
    # NEGATIVE: the plain (U), the transpose (reversed, U^T), and the same-order conjugate
    # do NOT cancel -- only the true conjugate reflection is swallowed
    assert not same_picture(zigzag(mbox=PH, mbarbox=PH), zigzag())          # bare U
    assert not same_picture(zigzag(mbox=PH, mbarbox=(c, b, a)), zigzag())   # reversed = U^T
    assert not same_picture(zigzag(mbox=PH, mbarbox=(-a, -b, -c)), zigzag())
    # and the far half is still 1/2 I after the channel (scalar 1/2)
    ch = zigzag(mbox=PH, mbarbox=conj_reflection); ch.scalar.add_power(-2)
    assert same_picture(ch, far_cup)


if __name__ == "__main__":
    verdict("NoSignalling", blackboard, sketchbook)
