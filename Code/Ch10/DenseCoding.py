"""Challenge IV: The Other Way Round  (superdense coding; the mirror)
===================================================================
"...Oh. It's the letter's own table."  -- Achilles

Claim: Alice turns her half of the pair by Z^k1 X^k2 and sends it; Bob's Bell
reading answers j with amplitude <B_j|(Z^k1 X^k2 (x) I)|Phi+> =
1/2 tr(X^j2 Z^j1 Z^k1 X^k2) = delta_jk, exactly, for all sixteen (j, k): a
closed loop is a trace, a clean loop is 1 and a loop with a ring left over is
0.  The codebook is teleportation's correction table, and the reading is the
mirror (dagger) of the preparation -- not its transpose, and dense coding is
not the dagger of the teleportation protocol.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx.simplify as simp

from dfp import BELL, CAP, CUP, I2, Picture, kron, ket, matrix_of, random_matrix, \
    random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch10 as PC


X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
KS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def P(M, m):
    return np.linalg.matrix_power(M, m)


def bell(m1, m2):
    """|B_{m1m2}> = CNOT (H (x) I)|m1 m2>."""
    return (CNOT @ kron(H, I2) @ ket(m1, m2).reshape(4, 1)).ravel()


def codebook(k1, k2):
    """Alice's writing for message k1 k2: flip if k2, then spin if k1."""
    return P(Z, k1) @ P(X, k2)


def random_unitary(rng, d=2):
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def blackboard():
    rng = np.random.default_rng(1004)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))       # cup unnormalized
    assert np.isclose((CAP @ CUP).item(), 2)                  # the bare loop is 2

    # Result 10.6: a loop through ANY box is its trace -- on a generic box
    A = random_matrix(rng)
    assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())
    assert np.isclose((CAP @ kron(A, I2) @ CUP).item(), np.trace(A))
    # with the NORMALIZED pair and effect it is half the trace: the 1/2 written in front
    assert np.isclose((BELL.conj() @ kron(A, I2) @ BELL).item(), np.trace(A) / 2)
    assert not np.isclose((BELL.conj() @ kron(A, I2) @ BELL).item(), np.trace(A))

    # Result 10.7: all sixteen, exactly delta_jk
    for j in KS:
        for k in KS:
            sent = (kron(codebook(*k), I2) @ BELL).ravel()
            amp = bell(*j).conj() @ sent
            assert np.isclose(amp, 1.0 if j == k else 0.0)
            # the loop form: 1/2 tr(X^j2 Z^j1 Z^k1 X^k2)
            ring = P(X, j[1]) @ P(Z, j[0]) @ P(Z, k[0]) @ P(X, k[1])
            assert np.isclose(amp, 0.5 * np.trace(ring))
            # tr(sigma_j^dagger sigma_k) = 2 delta_jk
            assert np.isclose(np.trace(codebook(*j).conj().T @ codebook(*k)), 2.0 * (j == k))
    # a leftover flip, spin or both is traceless: the loop is 0
    for R in (X, Z, Z @ X):
        assert np.isclose(np.trace(R), 0)

    # the codebook IS teleportation's correction table: entry k undoes the turn
    # X^k2 Z^k1 that answer k leaves on Bob's wire in teleportation
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())
    for k1, k2 in KS:
        held = (kron(bell(k1, k2).conj().reshape(1, 4), I2) @ kron(psi, BELL)).ravel()
        assert np.allclose(held, 0.5 * P(X, k2) @ P(Z, k1) @ psi)       # teleportation, answer k
        assert np.allclose(codebook(k1, k2) @ held, 0.5 * psi)          # corrected by entry k
        assert np.allclose(codebook(k1, k2) @ P(X, k2) @ P(Z, k1), I2)

    # the reading is the MIRROR of the preparation -- tested where mirror and half-turn
    # differ: a twisted basis U_k = Z^k1 X^k2 W, W non-real and non-symmetric
    W = random_unitary(rng)
    assert not np.allclose(W, W.conj()) and not np.allclose(W, W.T)
    Es = {k: (kron(codebook(*k) @ W, I2) @ BELL).ravel() for k in KS}   # |E_k>
    for j in KS:
        read = Es[j].conj()                                  # <E_j| = |E_j>^dagger
        assert not np.allclose(read, Es[j])                  # ... which is NOT |E_j>^T
        for k in KS:
            sent = (kron(codebook(*k) @ W, I2) @ BELL).ravel()
            assert np.isclose(read @ sent, 1.0 if j == k else 0.0)      # decodes exactly
    # reading with the half-turn (transpose) instead of the mirror does not decode
    trans = np.array([[Es[j] @ Es[k] for k in KS] for j in KS])
    assert not np.allclose(np.abs(trans) ** 2, np.eye(4))

    # NOT the dagger of the protocol.  Teleportation's corrected branch is the one-qubit
    # map 1/2 I; its dagger is 1/2 I again, a qubit-to-qubit map (teleportation backwards).
    # Dense coding is a map from two bits to two bits: the 4x4 table of |amplitude|^2.
    branch_map = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        e = ket(i)
        branch_map[:, i] = codebook(1, 1) @ (kron(bell(1, 1).conj().reshape(1, 4), I2)
                                             @ kron(e, BELL)).ravel()
    assert np.allclose(branch_map, 0.5 * I2)
    assert np.allclose(branch_map.conj().T, 0.5 * I2)        # its dagger: 1/2 I on a qubit
    dense = np.array([[abs(bell(*j).conj() @ (kron(codebook(*k), I2) @ BELL).ravel()) ** 2
                       for k in KS] for j in KS])
    assert np.allclose(dense, np.eye(4))                     # two bits in, the same two out
    assert branch_map.conj().T.shape != dense.shape          # not the same kind of thing

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the Tortoise's closed loop -- the pair's
# bend at the left, Bob's glass's bend at the right, Alice's leg along the top
# carrying her writing and then the answer asked about -- with two greens side by
# side, lifted off to a bare loop; then a red answer carried the long way round,
# meeting nothing,
#   \ringstring[operations]{scene}{caption}              (pdf/props/ch10.tex)
# tools/props_ch10.py transcribes the Lua (the string's layout read by the
# shared cloth kit, tools/props_cloth.py); the calls are read out of this
# challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3, state model 6): the loop's ring
# sequence from the near bend -- the writing's red, its green, the answer's
# green, its red -- read as the order the turns act in, is the operator
# X^j2 Z^j1 Z^k1 X^k2 whose half-trace is the dense-coding amplitude
# <B_j|(Z^k1 X^k2 (x) I)|Phi+>; like rings meet cyclically (a trace is cyclic),
# and the loop comes clean exactly when the writing is the answer, for all
# sixteen pairs.  A structural check against this file's blackboard, not a
# second proof.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = "...Oh. It's the letter's own table."
RING = {"red": X, "green": Z}


def applied(colours):
    """The operator a ring sequence makes, read in the order the turns act."""
    R = I2
    for c in colours:
        R = RING[c] @ R
    return R


def loop_scene(w, a):
    return "the loop, writing %s, answer %s" % (PC.patname(w), PC.patname(a))


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    calls = DIALOGUE.check_layout("Fourth", sizes=[1, 1, 1])
    p41, p42, p43 = calls
    for c in calls:
        assert words_only(c.caption, forbidden=PC.CAPTION_FORBIDDEN), c.caption
        assert PC.place_ok("Fourth", c), c.text
    for b in DIALOGUE.blocks("Fourth"):
        assert len({PC.height(c) for c in b}) == 1                # no block mixes frames
    assert [c.macro for c in calls] == ["ringstring"] * 3
    assert not any(c.keyed for c in calls)                        # the string was keyed at P1.3
    assert [(c.opt, c.args[0]) for c in calls] == [
        ("", "the loop, writing copper up, answer copper up"),
        ("kin to kin", "the loop, writing copper up, answer copper up"),
        ("slide round", "the loop, writing copper up, answer brass up")]
    assert [c.caption for c in calls] == [
        "green beside green", "lifted off, the loop bare", "round the loop, and nothing meets"]
    s41, s42, s43 = (PC.picture(c) for c in calls)
    a1, a2, a3 = DIALOGUE.anchors("Fourth")
    f1, f2, f3 = DIALOGUE.following("Fourth")
    section = DIALOGUE.section("Fourth")
    assert QUOTE in section                                       # the docstring's line

    # P4.1: the loop, closed, two bends; on Alice's leg her writing (no red, a green) and
    # the answer (a green): two greens side by side
    assert s41.dump == "loop; places [- green green -]"
    assert len(s41.cloth["cords"]) == 1 and s41.cloth["cords"][0]["closed"]
    assert PC.bends_of(s41.cloth) == 2 and s41.frame == 1.54
    assert "no red, for the brass lay down, and a green ring for the copper" in a1
    assert "copper up, brass down: one green ring" in a1
    assert "two green rings side by side on it" in a1
    assert f1 == "**Tortoise:** Kin to kin."
    # the writing is the one Alice chose (the Crab turned the copper, the brass lies blank)
    assert "turns the copper one over" in section
    # ... and its rings are the letter's: the same rule threads Bob's corrections
    w = PC.Pattern(brass=False, copper=True)
    assert [c for c in s41.state.places[:2] if c] == PC.legrings(w)
    bob = [r.c for r in PC.apply_string(PC.apply_string(PC.scene_state("the tale, copper up"), "pull"),
                                       "corrections").seq if r.who == "bob"]
    assert bob == PC.legrings(w)
    assert np.allclose(applied(PC.legrings(w)), codebook(*PC.bits(w)))   # the codebook entry

    # P4.2: kin to kin -- the answer's green meets the writing's and comes off: a clean loop
    assert s42.dump == "loop; places [- - - -]" and not s42.cloth["things"]
    assert s42.frame == s41.frame
    assert "nothing on it anywhere" in a2
    assert f2.startswith("**Tortoise:** A clean loop")

    # P4.3: the answer brass up, its red carried round the far bend, along Bob's leg, round
    # the near bend, up to beside the writing's green -- and nothing meets
    assert s43.dump == "loop; places [red green - -]" and s43.state.slid
    assert PC.dump(PC.apply_string(s43.state, "kin to kin")) == s43.dump   # nothing comes off
    assert "until it sits beside the green" in a3 and "Nothing comes off." in a3
    assert f3.startswith("**Tortoise:** A loop with a ring left over")

    # the loop, all sixteen: its rings from the near bend, read as the order of
    # application, make X^j2 Z^j1 Z^k1 X^k2, whose half-trace is the amplitude; kin to
    # kin (and slide round first, where the Lua allows it) keeps the half-trace, and the
    # loop is bare exactly when the writing is the answer
    for w in PC.PATTERNS:
        for a in PC.PATTERNS:
            k, j = PC.bits(w), PC.bits(a)
            amp = bell(*j).conj() @ (kron(codebook(*k), I2) @ BELL).ravel()
            assert np.isclose(amp, 1.0 if j == k else 0.0)
            M = PC.scene_state(loop_scene(w, a))
            R = applied(PC.sequence(M))
            assert np.allclose(R, P(X, j[1]) @ P(Z, j[0]) @ P(Z, k[0]) @ P(X, k[1]))
            assert np.isclose(0.5 * np.trace(R), amp)
            runs = [[]]
            if PC.faults(PC.ringstring, "slide round", loop_scene(w, a)) == []:
                runs.append(["slide round"])
            for run in runs:
                N = M
                for op in run:
                    N = PC.apply_string(N, op)
                assert np.isclose(0.5 * np.trace(applied(PC.sequence(N))), amp)   # trace is cyclic
                if PC.sequence(N):
                    N = PC.apply_string(N, "kin to kin")
                assert np.isclose(0.5 * np.trace(applied(PC.sequence(N))), amp)
                assert (PC.sequence(N) == []) == (w == a)          # clean iff writing = answer
    # the Lua's loop rules: slide round carries the answer's ONE ring, once, into a free place
    assert PC.faults(PC.ringstring, "slide round", "the loop, writing copper up, answer both up") == \
        ["slide-count"]
    assert PC.faults(PC.ringstring, "slide round", "the loop, writing brass up, answer copper up") == \
        ["slide-room"]
    assert PC.faults(PC.ringstring, "slide round, slide round",
                     "the loop, writing copper up, answer brass up") == ["slid-twice"]


# --- pictures -----------------------------------------------------------------

def loop(k, j):
    """The closed loop of Section 10.4: from the cup (the shared pair), along Alice's leg
    her writing (red k2*pi, then green k1*pi), then the reading's rings (green j1*pi,
    then red j2*pi), round the cap, and back along Bob's leg to the cup.  The return
    leg carries a phase-0 green dot as a spacer: pyzx's add_edge between the same two
    vertices twice CANCELS, so the loop must never join two vertices by two edges.
    Scalar: the pair's 1/sqrt2 and the reading's 1/sqrt2 make 1/2 -> add_power(-2)."""
    (k1, k2), (j1, j2) = k, j
    p = Picture()
    cup, cap = p.bend(0.5, 0), p.bend(0.5, 6)
    p.wire(cup, p.x(0, 1, k2), p.z(0, 2, k1), p.z(0, 3, j1), p.x(0, 4, j2), cap)
    p.wire(cap, p.z(1, 3, 0), cup)
    g = p.done()
    g.scalar.add_power(-2)
    return g


def loop_through(chain):
    """A bare cup-then-cap loop with a chain of dots on its top leg (no scalar)."""
    p = Picture()
    cup, cap = p.bend(0.5, 0), p.bend(0.5, 6)
    dots = [p.z(0, 1 + i, a) if c == "Z" else p.x(0, 1 + i, a) for i, (c, a) in enumerate(chain)]
    p.wire(cup, *dots, cap)
    p.wire(cap, p.z(1, 3, 0), cup)                           # spacer, as above
    return p.done()


def wire_through(chain):
    p = Picture()
    dots = [p.z(0, 1 + i, a) if c == "Z" else p.x(0, 1 + i, a) for i, (c, a) in enumerate(chain)]
    p.wire(p.inp(0), *dots, p.out(0))
    return p.done()


def preparation(chain, scale=0):
    """A cup with a chain of dots on its top leg, listed outward from the bend."""
    p = Picture()
    bend = p.bend(0.5, 0)
    dots = [p.z(0, 1 + i, a) if c == "Z" else p.x(0, 1 + i, a) for i, (c, a) in enumerate(chain)]
    p.wire(bend, *dots, p.out(0))
    p.wire(bend, p.out(1))
    g = p.done()
    g.scalar.add_power(scale)
    return g


def reading(chain, scale=0):
    """A cap with a chain of dots on its incoming top leg, listed from the input."""
    p = Picture()
    bend = p.bend(0.5, 9)
    dots = [p.z(0, 1 + i, a) if c == "Z" else p.x(0, 1 + i, a) for i, (c, a) in enumerate(chain)]
    p.wire(p.inp(0), *dots, bend)
    p.wire(p.inp(1), bend)
    g = p.done()
    g.scalar.add_power(scale)
    return g


def sketchbook():
    # the bare loop is the number 2 (no scalar added); a loop through a box is its trace
    assert np.isclose(matrix_of(loop_through([("Z", 0)])).item(), 2)
    chain = [("Z", Fraction(1, 5)), ("X", Fraction(2, 7)), ("Z", Fraction(3, 4))]
    A = matrix_of(wire_through(chain))
    assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())       # a generic box
    assert np.isclose(matrix_of(loop_through(chain)).item(), np.trace(A))

    # Result 10.7 in the picture: sixteen closed loops (closed diagrams -- evaluated only
    # through matrix_of, i.e. the naive contraction), exactly delta_jk
    for k in KS:
        for j in KS:
            g = loop(k, j)
            want = 1.0 if j == k else 0.0
            assert np.isclose(matrix_of(g).item(), want)
            # DERIVED: full_reduce fuses like with like and removes the whole turns; a clean
            # loop reduces to the number 1 (= 1/2 x the bare loop's 2), a loop with a ring
            # left over to 0.  Nothing of the picture is left but its scalar.
            h = g.copy()
            simp.full_reduce(h)
            assert h.num_vertices() == 0
            assert np.isclose(h.scalar.to_number(), want)
            assert np.isclose(matrix_of(h).item(), want)

    # the reading is the MIRROR of the preparation: reflect left to right and conjugate
    # every phase.  Half-turn rings are their own mirror images (pi = -pi), so the Bell
    # effect is the mirror of the Bell state, rings reversed along the page.
    for m1, m2 in KS:
        state = preparation([("X", m2), ("Z", m1)], scale=-1)          # |B_k>
        effect = reading([("Z", m1), ("X", m2)], scale=-1)             # <B_k|
        assert np.allclose(matrix_of(state).ravel(), bell(m1, m2))
        assert same_picture(state.adjoint(), effect)
    # ... and on a generic chain, where mirror and half-turn differ: the mirror conjugates
    # the phases, the half-turn (transpose) does not
    a, b, c = Fraction(1, 5), Fraction(2, 7), Fraction(3, 4)
    prep = preparation([("Z", a), ("X", b), ("Z", c)])
    mirror = reading([("Z", -c), ("X", -b), ("Z", -a)])
    half_turn = reading([("Z", c), ("X", b), ("Z", a)])
    assert same_picture(prep.adjoint(), mirror)                          # the dagger: yes
    assert np.allclose(matrix_of(half_turn), matrix_of(prep).T)          # this is the transpose
    assert not same_picture(prep.adjoint(), half_turn)                   # ... and it is not the mirror


if __name__ == "__main__":
    verdict("DenseCoding", blackboard, sketchbook)
