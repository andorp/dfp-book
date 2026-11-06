"""Challenge III: Cover the Other Half  (partial trace / discard)
================================================================
That fold is "cover the other half."  -- Tortoise

Claim: covering (discarding) a subsystem IS the partial trace -- rho_A =
tr_B rho_AB -- and in the doubled picture it is a CAP folding the discarded
system's two copies together.  The scalar is load-bearing: the doubled bare cup
discards to I (a closed value of 2), the doubled Bell state to 1/2 I.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
import pyzx.simplify as simp

from dfp import BELL, CAP, CUP, I2, matrix_of, random_state, same_picture, Picture, verdict
from dfp_props import Dialogue, words_only
import props_ch09 as P


def blackboard():
    rng = np.random.default_rng(93)

    # a genuinely MIXED random joint state rho_AB (positive, Hermitian, unit trace)
    X = rng.normal(size=(4, 6)) + 1j * rng.normal(size=(4, 6))
    rho = X @ X.conj().T
    rho /= np.trace(rho)
    assert np.allclose(rho, rho.conj().T) and np.linalg.matrix_rank(rho) > 1

    r = rho.reshape(2, 2, 2, 2)                        # indices A, B, A', B'
    trB = np.einsum('abcb->ac', r)                     # partial trace over B

    # discard = the CAP <cc| joining B to its mirror copy: cap[b,b'] = delta_{b b'}
    cap = CAP.reshape(2, 2)                            # = <00|+<11| read on (b, b')
    discard = np.einsum('abAB,bB->aA', r, cap)
    assert np.allclose(discard, trB)                  # discard IS the partial trace
    assert np.isclose(np.trace(trB), 1)               # ... and it preserves the trace

    # the two load-bearing scalars, on the blackboard
    bare = np.outer(CUP, CUP.conj())                  # doubled bare cup (unnormalized)
    bare_discard = np.einsum('abcb->ac', bare.reshape(2, 2, 2, 2))
    assert np.allclose(bare_discard, I2) and np.isclose(np.trace(bare_discard), 2)   # -> I
    bell = np.outer(BELL, BELL.conj())                # doubled Bell = |Phi+><Phi+|
    assert np.allclose(np.einsum('abcb->ac', bell.reshape(2, 2, 2, 2)), I2 / 2)      # -> 1/2 I

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the black slab, a plain pair on it,
# the near stone tipped and folded,
#   \blackslab[operations]{scene}{caption}                 (pdf/props/ch09.tex)
# tools/props_ch09.py transcribes the Lua; the calls are read out of this
# challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3): the plain pair is two stones each
# cut alone -- a product of two pure states, far (the slab's left, the top wire)
# and near; a fold is the discard, the partial trace over the folded stone.  The
# far stone's drawn entry must come through the fold unchanged, as
# tr_near(rho_far (x) rho_near) = rho_far.  A structural check against this
# file's blackboard, not a second proof.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
QUOTE = 'That fold is "cover the other half."'


def pictures():
    assert QUOTE in __doc__                                      # quoted byte for byte
    calls = DIALOGUE.check_layout("Third", sizes=[3])
    p31, p32, p33 = calls
    for c in calls:
        assert words_only(c.caption, forbidden=P.CAPTION_FORBIDDEN), c.caption
        assert P.place_ok("Third", c), c.text
    (block,) = DIALOGUE.blocks("Third")
    assert len({P.height(c) for c in block}) == 1                # no block mixes frames
    assert [c.macro for c in calls] == ["blackslab"] * 3
    assert [c.keyed for c in calls] == [True, False, False]
    assert [c.args for c in calls] == [("plain pair",)] * 3       # one scene, one motion
    assert [c.opt for c in calls] == ["", "tip near stone", "fold near stone"]
    b31, b32, b33 = (P.picture(c) for c in calls)
    (a1,) = DIALOGUE.anchors("Third")
    (f1,) = DIALOGUE.following("Third")
    assert QUOTE in DIALOGUE.section("Third")                     # the docstring's line

    # the model: two stones each cut alone, non-real; far on the top wire
    rng = np.random.default_rng(931)
    far, near = random_state(rng), random_state(rng)
    assert not np.allclose(far, far.conj()) and not np.allclose(near, near.conj())
    rho_far, rho_near = np.outer(far, far.conj()), np.outer(near, near.conj())
    joint = np.kron(rho_far, rho_near).reshape(2, 2, 2, 2)       # far, near, far', near'
    kept = np.einsum("abcb->ac", joint)                           # fold the near stone
    assert np.allclose(kept, rho_far)                             # the far stone unstirred
    assert np.isclose(np.trace(rho_near @ rho_near).real, 1)      # each stone pure: tight
    assert np.isclose(np.trace(kept @ kept).real, 1)

    # P3.1: the plain pair, keyed -- far at the left, near at the right, both restless, tight
    assert "not stirred by a hair" in a1
    assert "the far one at the slab's far end and the near one at Achilles's" in a1
    assert "each cut alone" in a1
    far1, near1 = b31.shown.things
    assert (far1.name, near1.name) == ("far stone", "near stone") and far1.x < near1.x
    assert all((t.kind, t.gloss, t.refl, t.mode) == ("stone", "restless", "tight", "standing")
               for t in (far1, near1))
    assert b31.labels == ("far", "near", "black slab") and b31.shown.frame == P.LOW_FRAME

    # P3.2: the near stone tipped -- it and its reflection meeting at the surface
    far2, near2 = b32.shown.things
    assert near2.mode == "tipped" and near2.x == near1.x and far2 == far1

    # P3.3: folded -- the near stone gone, the far stone's entry exactly P3.1's
    (far3,) = b33.shown.things
    assert far3 == far1                                           # unstirred, to the letter
    assert b33.shown.frame == b31.shown.frame and b33.shown.ends == b31.shown.ends
    assert "simply gone, folded into its own mirror" in a1

    # the Lua refuses the moves the chapter rules out: the far stone is never folded
    # (only a stone of one's own is covered), and a folded stone is gone for good
    assert P.faults(P.blackslab, "fold far stone", "plain pair") == ["far-stone"]
    assert P.faults(P.blackslab, "tip far stone", "plain pair") == ["far-stone"]
    assert P.faults(P.blackslab, "fold near stone, fold near stone", "plain pair") == ["gone"]
    assert P.faults(P.blackslab, "fold near stone, tip near stone", "plain pair") == ["gone"]
    assert P.faults(P.blackslab, "tip near stone, fold near stone", "plain pair") == []
    assert P.faults(P.blackslab, "fold near stone", "restless stone") == ["single"]
    assert "near slope" in f1 and f1.startswith("*[")             # the roof, after the block


def discard_cap():
    """The discard drawn in doubled form: a CAP on a system's two copies.
    Two legs in, nothing out -- the effect <00|+<11| (unnormalized)."""
    p = Picture()
    cap = p.bend(0.5, 1)
    p.wire(p.inp(0), cap, p.inp(1))
    return p.done()


def discarded(scalar_power=0):
    """A doubled two-qubit state with its NEAR half discarded, unfolded:
    S -- cup(S,M) -- cap(M,Mbar) -- cup(Mbar,Sbar) -- Sbar.  One yank leaves the far
    half cupped to its own mirror.  scalar_power = -2 turns the bare cup into Bell."""
    p = Picture()
    cupSM = p.bend(0.5, 1)
    capMM = p.bend(1.5, 3)
    cupSbMb = p.bend(2.5, 1)
    p.wire(p.out(0), cupSM, capMM, cupSbMb, p.out(3))
    g = p.done()
    if scalar_power:
        g.scalar.add_power(scalar_power)
    return g


def cup_ref(scalar_power=0):
    """The far half cupped to its own mirror: cup(S, Sbar)."""
    p = Picture()
    p.wire(p.out(0), p.bend(1.5, 1), p.out(3))
    g = p.done()
    if scalar_power:
        g.scalar.add_power(scalar_power)
    return g


def sketchbook():
    # the discard is the UNNORMALIZED cap <00|+<11|, not the normalized one
    assert np.allclose(matrix_of(discard_cap()).ravel(), CAP.ravel())
    assert not np.allclose(matrix_of(discard_cap()).ravel(), CAP.ravel() / 2)

    # discard = partial trace, exactly, scalar included:
    # doubled bare cup, near half discarded, yanks to cup(S,Sbar) = vec(I)   (trace 2)
    assert same_picture(discarded(), cup_ref())
    assert np.allclose(matrix_of(cup_ref()).reshape(2, 2), I2)              # = I
    # doubled Bell (the 1/2), discarded, yanks to 1/2 cup(S,Sbar) = vec(1/2 I)
    assert same_picture(discarded(scalar_power=-2), cup_ref(scalar_power=-2))
    assert np.allclose(matrix_of(cup_ref(scalar_power=-2)).reshape(2, 2), I2 / 2)  # = 1/2 I
    # the rewrite rules DERIVE the yank (cup-cap-cup straightens), carrying the scalar
    g = discarded(); simp.full_reduce(g)
    assert same_picture(g, cup_ref())
    g2 = discarded(scalar_power=-2); simp.full_reduce(g2)
    assert same_picture(g2, cup_ref(scalar_power=-2))                       # 1/2 carried through
    # NEGATIVE: the Bell discard is NOT I -- the 1/2 is the whole point (maximally mixed)
    assert not same_picture(discarded(scalar_power=-2), cup_ref())

    # a generic NON-REAL doubled state (built by hand as the state beside its conjugate
    # mirror image): the discard-cap on the B copies matches the numpy partial trace
    rng = np.random.default_rng(930)
    phi = rng.normal(size=4) + 1j * rng.normal(size=4)
    phi /= np.linalg.norm(phi)
    assert not np.allclose(phi, phi.conj())
    doubled = np.kron(phi, phi.conj()).reshape(2, 2, 2, 2)   # ket (a,b), mirror (a',b')
    capped = np.einsum('abAB,bB->aA', doubled, CAP.reshape(2, 2))
    trB = np.einsum('abcb->ac', np.outer(phi, phi.conj()).reshape(2, 2, 2, 2))
    assert np.allclose(capped, trB)


if __name__ == "__main__":
    verdict("PartialTrace", blackboard, sketchbook)
