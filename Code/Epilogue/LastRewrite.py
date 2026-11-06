"""The exchange: Postlude on the Same Sheet
==========================================
"Heads a half, tails a half, and it lies one way or the other for certain. A half and a half: once. Had it been loaded five to one, five sixths and one sixth: once still. Whatever way it was set, once."  -- Achilles

Claim: a state, let go, is the empty diagram: for every state psi of one wire,
tr |psi><psi| = <psi|psi> = 1, exactly, scalar included.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from dfp import CAP, CUP, Picture, matrix_of, same_picture, verdict

I = np.eye(2, dtype=complex)


def phases():
    """Two seeded phases (units of pi), shared by both strands, so they check one psi."""
    rng = np.random.default_rng(154)
    a, b = rng.integers(1, 97, size=2)
    return Fraction(int(a), 97), Fraction(int(b), 97)


def psi_of(a, b):
    """psi = 1/2 (1 + e^{i pi a}, (1 - e^{i pi a}) e^{i pi b}): normalized, and non-real."""
    ea, eb = np.exp(1j * np.pi * float(a)), np.exp(1j * np.pi * float(b))
    return 0.5 * np.array([1 + ea, (1 - ea) * eb])


def blackboard():
    a, b = phases()
    psi = psi_of(a, b)
    assert not np.allclose(psi, psi.conj())                  # non-real: the mirror matters
    assert not np.isclose((psi[0] * psi[1].conj()).imag, 0)  # non-real beyond a global phase
    assert np.isclose(np.linalg.norm(psi), 1)                # a state

    # the cap is UNNORMALIZED: <00| + <11|, norm sqrt 2
    assert np.isclose(np.linalg.norm(CAP), np.sqrt(2))
    assert np.allclose(CAP.ravel(), CUP.conj())

    # --- Dirac's Blackboard: letting go is the trace -------------------------
    alpha, beta = psi
    rho = np.outer(psi, psi.conj())                          # |psi><psi|
    assert np.isclose(np.trace(rho), alpha * alpha.conj() + beta * beta.conj())
    assert np.isclose(np.trace(rho), psi.conj() @ psi)       # = <psi|psi>
    assert np.isclose(np.trace(rho), 1)                      # = 1, exactly

    # for every state: a handful more, seeded and non-real
    rng = np.random.default_rng(1540)
    for _ in range(8):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v /= np.linalg.norm(v)
        assert not np.allclose(v, v.conj())
        assert np.isclose(np.trace(np.outer(v, v.conj())), 1)

    # the doubled picture: psi above, its conjugate copy psi* below, closed by the cap.
    # The answer is the 1x1 matrix (1): a process from no wire to no wire.
    doubled = np.kron(psi, psi.conj()).reshape(4, 1)        # system first, copy second
    closed = CAP @ doubled
    assert closed.shape == (1, 1)
    assert np.allclose(closed, [[1]])
    assert not np.isclose((CAP / 2) @ doubled, 1)            # the normalized cap is wrong
    assert not np.isclose((CAP / np.sqrt(2)) @ doubled, 1)

    # --- The bend: the copy carried round the cap arrives as psi-dagger ------
    bent = CAP @ np.kron(I, psi.conj().reshape(2, 1))       # an effect on the upper wire
    assert bent.shape == (1, 2)
    assert np.allclose(bent.ravel(), psi.conj())             # psi-dagger: the mirror conjugates
    assert not np.allclose(bent.ravel(), psi)                # not psi-transpose
    assert np.isclose(bent @ psi, 1)                         # it closes psi to 1
    assert not np.isclose(psi @ psi, 1)                      # psi-transpose as the effect: not 1

    # --- The example: |+> let go --------------------------------------------
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho_plus = np.outer(plus, plus.conj())
    assert np.allclose(np.diag(rho_plus), [0.5, 0.5])
    assert np.isclose(np.trace(rho_plus), 1)                 # a half + a half = 1

    # --- Letting go is not looking ------------------------------------------
    bra0 = np.array([1, 0], dtype=complex)
    amplitude = bra0 @ plus
    assert np.isclose(amplitude, 1 / np.sqrt(2))             # <0|+> = 1/sqrt 2
    look = np.kron(bra0, bra0) @ np.kron(plus, plus.conj())  # effect 0 on each copy
    assert np.isclose(look, abs(amplitude) ** 2)
    assert np.isclose(look, 0.5)                             # the look weighs one answer: 1/2
    let_go = CAP @ np.kron(plus, plus.conj())
    assert np.isclose(let_go, 1)                             # letting go weighs them all: 1
    assert not np.isclose(look, let_go)

    # --- The coin, reckoned (exact fractions) -------------------------------
    heads, tails = Fraction(1, 2), Fraction(1, 2)
    assert heads + tails == 1                                # "a half and a half: once"
    assert heads != 1                                        # "A half. Heads, a half." -- heads alone
    loaded = (Fraction(5, 6), Fraction(1, 6))
    assert sum(loaded) == 1                                  # "five sixths and one sixth: once still"
    assert np.isclose(np.trace(np.diag([5 / 6, 1 / 6])), 1)

    # --- Scalar honesty: a wire does not always go for nothing ---------------
    assert np.isclose(CAP @ CUP, 2)                          # a bare loop is 2
    assert np.isclose(np.trace(I), 2)                        # the doubled bare cup, let go: tr I
    assert np.isclose(CAP @ (CUP / 2), 1)                    # the half-cup, let go: 1/2 x 2 = 1
    assert np.allclose((CUP / 2).reshape(2, 2), I / 2)       # the half-cup is 1/2 I
    point = np.array([1, 1], dtype=complex)                  # |0> + |1>, unnormalized
    assert np.isclose(CAP @ np.kron(point, point.conj()), 2) # an unnormalized point let go: 2

    # mixed states: a weighted list of pure ones, weights summing to 1, lets go to 1
    weights = rng.random(3)
    weights /= weights.sum()
    pures = [rng.normal(size=2) + 1j * rng.normal(size=2) for _ in range(3)]
    pures = [v / np.linalg.norm(v) for v in pures]
    mixed = sum(w * np.outer(v, v.conj()) for w, v in zip(weights, pures))
    assert np.linalg.matrix_rank(mixed) > 1                  # genuinely mixed
    assert np.all(np.linalg.eigvalsh(mixed) > -1e-12)        # positive
    assert np.isclose(np.trace(mixed), 1)


def state(a, b):
    """The state box psi on one output wire."""
    p = Picture()
    # PyZX draws the state box ψ with a red and a green spider (Chapter 8).
    p.wire(p.x(0, 1, a), p.z(0, 2, b), p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    assert g.num_edges() == 2
    return g


def doubled_state(a, b):
    """psi on the upper line, its conjugate copy psi* (phases negated) on the lower."""
    p = Picture()
    p.wire(p.x(0, 1, a), p.z(0, 2, b), p.out(0))
    p.wire(p.x(1, 1, -a), p.z(1, 2, -b), p.out(1))
    g = p.done()
    g.scalar.add_power(-2)
    assert g.num_edges() == 4
    return g


def unfolded(a, b, power=-2):
    """P2: psi and psi*, the two lines joined on the right by the unnormalized cap.
    P1 (psi, a wire, the discard) is not built: it IS this picture, by the definition of
    the discard as the cap on a system's two copies (Chapter 9, row 67)."""
    p = Picture()
    # PyZX 0.10.6 does not evaluate a graph with a ground vertex, so the discard is drawn unfolded, as the cap on the two copies (Chapter 9, §9.3), as Chapter 9's code does.
    cap = p.bend(0.5, 3)
    p.wire(p.x(0, 1, a), p.z(0, 2, b), cap, p.z(1, 2, -b), p.x(1, 1, -a))
    g = p.done()
    if power:
        g.scalar.add_power(power)
    assert g.num_edges() == 4
    return g


def bent_copy(a, b):
    """The copy psi* carried round the cap: an effect on the upper wire."""
    p = Picture()
    p.wire(p.inp(0), p.bend(0.5, 3), p.z(1, 2, -b), p.x(1, 1, -a))
    g = p.done()
    g.scalar.add_power(-1)
    assert g.num_edges() == 3
    return g


def transposed_effect(a, b):
    """The plausible mix-up: psi bent round WITHOUT conjugating -- psi-transpose."""
    p = Picture()
    p.wire(p.inp(0), p.z(0, 2, b), p.x(0, 3, a))
    g = p.done()
    g.scalar.add_power(-1)
    assert g.num_edges() == 2
    return g


def lone_cap():
    """A cap on two wires: the effect <00| + <11|."""
    p = Picture()
    p.wire(p.inp(0), p.bend(0.5, 1), p.inp(1))
    g = p.done()
    assert g.num_edges() == 2
    return g


def loop(power=0):
    """A bare loop: a cup closed by a cap (four phase-free nodes in a ring)."""
    p = Picture()
    cup, top, cap, bottom = p.bend(0.5, 1), p.z(0, 2), p.bend(0.5, 3), p.z(1, 2)
    p.wire(cup, top, cap, bottom, cup)
    g = p.done()
    if power:
        g.scalar.add_power(power)
    assert g.num_edges() == 4
    return g


def unnormalized_point_let_go():
    """|0> + |1> (a phase-free point), capped to its own conjugate copy."""
    p = Picture()
    p.wire(p.z(0, 1), p.bend(0.5, 3), p.z(1, 1))
    g = p.done()
    assert g.num_edges() == 2
    return g


def look_at_plus():
    """The effect 0 on each copy of the doubled |+>."""
    p = Picture()
    p.wire(p.z(0, 1), p.x(0, 2))
    p.wire(p.z(1, 1), p.x(1, 2))
    g = p.done()
    g.scalar.add_power(-4)
    assert g.num_edges() == 2
    return g


def sketchbook():
    a, b = phases()
    psi = psi_of(a, b)

    # the empty diagram
    # PyZX draws pictures with the spiders you meet in Chapter 8; this one has none.
    E = zx.Graph()
    assert E.num_vertices() == 0 and E.num_edges() == 0
    assert matrix_of(E).shape == (1, 1)
    assert np.allclose(matrix_of(E), [[1]])

    # the cap is the UNNORMALIZED <00| + <11|
    assert np.allclose(matrix_of(lone_cap()).ravel(), CAP.ravel())
    assert not np.allclose(matrix_of(lone_cap()).ravel(), CAP.ravel() / 2)
    assert not np.allclose(matrix_of(lone_cap()).ravel(), CAP.ravel() / np.sqrt(2))

    # the state box is the blackboard's psi; its copy (phases negated) is psi*
    s = state(a, b)
    assert np.allclose(matrix_of(s).ravel(), psi)
    copy = state(-a, -b)
    assert np.allclose(matrix_of(copy).ravel(), psi.conj())  # the mirror conjugates
    assert not np.allclose(matrix_of(copy).ravel(), psi)
    assert np.allclose(matrix_of(doubled_state(a, b)).ravel(), np.kron(psi, psi.conj()))

    # P2: unfold the discard -- exactly 1
    P2 = unfolded(a, b)
    assert matrix_of(P2).shape == (1, 1)
    assert np.allclose(matrix_of(P2), [[1]])

    # the bend: the copy carried round the cap arrives as psi-dagger, the effect psi
    bent = bent_copy(a, b)
    assert np.allclose(matrix_of(bent).ravel(), psi.conj())
    assert same_picture(bent, s.adjoint())                   # psi-dagger
    assert np.allclose(matrix_of(transposed_effect(a, b)).ravel(), psi)
    assert not same_picture(bent, transposed_effect(a, b))   # not psi-transpose

    # P3: the state, then its mirror image -- exactly 1
    P3 = s.copy()
    P3.compose(s.adjoint())                                  # compose mutates in place
    assert np.allclose(matrix_of(P3), [[1]])

    # P1 = P2 = P3 = P4, scalar included
    assert same_picture(P2, E)
    assert same_picture(P3, E)
    assert same_picture(P2, P3)

    # the rewrites DERIVE it: the last rewrite leaves nothing
    for picture in (P2, P3):
        g = picture.copy()
        simp.full_reduce(g)
        assert g.num_vertices() == 0 and g.num_edges() == 0
        assert np.isclose(g.scalar.to_number(), 1)
        assert same_picture(g, E)

    # --- the traps: a wire does not always go for nothing --------------------
    L = loop()
    assert np.allclose(matrix_of(L), [[2]])                  # a bare loop is 2
    assert not same_picture(L, E)
    g = L.copy()
    simp.full_reduce(g)
    assert g.num_vertices() == 0 and np.isclose(g.scalar.to_number(), 2)

    P2_bare = unfolded(a, b, power=0)                        # P2 with its scalar left off
    assert np.allclose(matrix_of(P2_bare), [[2]])
    assert not same_picture(P2_bare, E)

    point = unnormalized_point_let_go()                      # an unnormalized point let go
    assert np.allclose(matrix_of(point), [[2]])
    assert not same_picture(point, E)

    half = loop(power=-2)                                    # the half-cup let go: 1/2 x 2
    assert np.allclose(matrix_of(half), [[1]])
    assert same_picture(half, E)

    # --- letting go is not looking -----------------------------------------
    look = look_at_plus()
    assert np.allclose(matrix_of(look), [[0.5]])             # |<0|+>|^2 = 1/2
    assert not same_picture(look, E)
    p = Picture()
    p.wire(p.z(0, 1), p.x(0, 2))                             # one copy: the amplitude
    one = p.done()
    one.scalar.add_power(-2)
    assert np.allclose(matrix_of(one), [[1 / np.sqrt(2)]])   # <0|+> = 1/sqrt 2


if __name__ == "__main__":
    verdict("LastRewrite", blackboard, sketchbook)
