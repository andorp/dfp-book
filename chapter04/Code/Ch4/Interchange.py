"""Challenge 4: Which First?
=============================
"The same again. First both firsts and then both seconds; or all of the silk and
then all of the hemp. But between the two walks nobody touched a clip. ...Oh.
Nothing was moved. It was one frame, and I read it two ways."  -- Achilles

Claim (the Interchange Law, Theorem 4.2): for any four matrices of compatible
shapes, (A (x) B)(C (x) D) = AC (x) BD, exactly and scalar included, with the
top wire as the left factor.  Corollary (Sliding): (A (x) I)(I (x) B) = A (x) B
= (I (x) B)(A (x) I).  On ONE wire the order is still part of what is said.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import I2, Picture, ket, matrix_of, random_matrix, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch04 as P

# the chapter's named matrices (Sec 4.4, Sec 4.5)
A = np.array([[1, 1j], [0, 1]], dtype=complex)
B = np.array([[0, 1], [1j, 0]], dtype=complex)
C = np.array([[1, 0], [1, 1j]], dtype=complex)
D = np.array([[1j, 2], [0, 1]], dtype=complex)

X = np.array([[0, 1], [1, 0]], dtype=complex)          # "turn it over"
S0 = np.array([[1, 1], [0, 0]], dtype=complex)         # "heads up, whatever it was"
S1 = np.array([[0, 0], [1, 1]], dtype=complex)


def generic(*ms):
    """Non-real, non-symmetric, non-Hermitian, and pairwise different."""
    for m in ms:
        assert not np.allclose(m, m.conj())
        assert not np.allclose(m, m.T)
        assert not np.allclose(m, m.conj().T)
    for i, m in enumerate(ms):
        for n in ms[i + 1:]:
            assert not np.allclose(m, n)


def blackboard():
    rng = np.random.default_rng(44)
    generic(A, B, C, D)

    # --- the chapter's printed matrices --------------------------------------
    assert np.allclose(np.kron(A, B),
                       [[0, 1, 0, 1j], [1j, 0, -1, 0], [0, 0, 0, 1], [0, 0, 1j, 0]])
    assert np.allclose(np.kron(B, A),
                       [[0, 0, 1, 1j], [0, 0, 0, 1], [1j, -1, 0, 0], [0, 1j, 0, 0]])
    assert np.allclose(np.kron(C, D),
                       [[1j, 2, 0, 0], [0, 1, 0, 0], [1j, 2, -1, 2j], [0, 1, 0, 1j]])
    assert np.allclose(A @ C, [[1 + 1j, -1], [1, 1j]])
    assert np.allclose(B @ D, [[0, 1], [-1, 2j]])
    product = np.kron(A, B) @ np.kron(C, D)
    assert np.allclose(product,
                       [[0, 1 + 1j, 0, -1], [-1 - 1j, -2 + 2j, 1, -2j],
                        [0, 1, 0, 1j], [-1, 2j, -1j, -2]])
    assert np.isclose(product[1, 1], -2 + 2j)          # the traced entry, row 01 col 01
    assert np.isclose(np.kron(A, B)[1] @ np.kron(C, D)[:, 1], -2 + 2j)
    assert np.isclose((A @ C)[0, 0] * (B @ D)[1, 1], -2 + 2j)

    # --- the law, on the named matrices and on matrices drawn at random ------
    quads = [(A, B, C, D)]
    for _ in range(30):
        quad = tuple(random_matrix(rng) for _ in range(4))
        generic(*quad)
        quads.append(quad)
    for a, b, c, d in quads:
        assert np.allclose(np.kron(a, b) @ np.kron(c, d), np.kron(a @ c, b @ d))
        # every cheaper guess fails
        assert not np.allclose(np.kron(a, b) @ np.kron(c, d), np.kron(c @ a, d @ b))
        assert not np.allclose(np.kron(a, b) @ np.kron(c, d), np.kron(a @ d, b @ c))
        assert not np.allclose(np.kron(a, b) @ np.kron(c, d), np.kron(b @ d, a @ c))
        assert not np.allclose(np.kron(a, b) @ np.kron(c, d), np.kron(a @ c, b @ d).T)
        assert not np.allclose(np.kron(a, b) @ np.kron(c, d),
                               np.kron((a @ c).conj().T, (b @ d).conj().T))
        assert not np.allclose(np.kron(a, b) @ np.kron(c, d), np.kron(a @ c, b @ d).conj())
        assert not np.allclose(np.kron(a, b) @ np.kron(c, d),
                               np.kron(a.T @ c.T, b.T @ d.T))
        # Sliding, both ways round
        assert np.allclose(np.kron(a, I2) @ np.kron(I2, b), np.kron(a, b))
        assert np.allclose(np.kron(I2, b) @ np.kron(a, I2), np.kron(a, b))
        # and the mirror of a stack is the stack of the mirrors
        assert np.allclose(np.kron(a, b).conj().T, np.kron(a.conj().T, b.conj().T))
        assert not np.allclose(np.kron(a, b).conj().T, np.kron(b.conj().T, a.conj().T))

    # --- unitary above unitary is unitary ------------------------------------
    for _ in range(10):
        U = np.linalg.qr(random_matrix(rng))[0]
        V = np.linalg.qr(random_matrix(rng))[0]
        generic(U, V)
        assert np.allclose(U.conj().T @ U, I2) and np.allclose(V.conj().T @ V, I2)
        UV = np.kron(U, V)
        assert np.allclose(UV.conj().T @ UV, np.eye(4))
        assert np.allclose(UV.conj().T @ UV, np.kron(I2, I2))

    # --- on ONE wire the order is still part of what is said -----------------
    assert np.allclose(S0 @ X, S0) and np.allclose(X @ S0, S1)
    assert not np.allclose(S0 @ X, X @ S0)
    # but on two wires those same two boxes have nothing to quarrel about
    assert np.allclose(np.kron(X, I2) @ np.kron(I2, S0), np.kron(X, S0))
    assert np.allclose(np.kron(I2, S0) @ np.kron(X, I2), np.kron(X, S0))

    # --- the dialogue's coin walks -------------------------------------------
    # one clip each: "turn it over" on silk, "heads up" on hemp.  Left, right
    # and level are three matrices, and they are one matrix.
    left = np.kron(X, I2) @ np.kron(I2, S0)            # hemp clip met first
    right = np.kron(I2, S0) @ np.kron(X, I2)           # silk clip met first
    level = np.kron(X, S0)
    assert np.allclose(left, right) and np.allclose(left, level)
    assert np.allclose(level @ ket(0, 1), ket(1, 0))   # "They arrive tails above, heads below."
    for a in (0, 1):
        for b in (0, 1):
            assert np.allclose(left @ ket(a, b), right @ ket(a, b))
            assert np.allclose(left @ ket(a, b), ket(1 - a, 0))

    # two clips each: silk "turn it over" then "heads up"; hemp the other order
    silk = S0 @ X                                      # read right to left
    hemp = X @ S0
    firsts = np.kron(X, S0)                            # both first clips
    seconds = np.kron(S0, X)                           # both second clips
    both_firsts_then_seconds = seconds @ firsts
    all_silk_then_all_hemp = np.kron(I2, hemp) @ np.kron(silk, I2)
    assert np.allclose(both_firsts_then_seconds, all_silk_then_all_hemp)
    assert np.allclose(both_firsts_then_seconds, np.kron(silk, hemp))
    for a in (0, 1):
        for b in (0, 1):
            assert np.allclose(np.kron(silk, hemp) @ ket(a, b), ket(0, 1))
    assert np.allclose(np.kron(silk, hemp) @ ket(1, 0), ket(0, 1))   # the walk as spoken
    # the silk clips in the other order: the silk coin arrives tails
    assert np.allclose(np.kron(X @ S0, hemp) @ ket(1, 0), ket(1, 1))
    assert not np.allclose(np.kron(X @ S0, hemp), np.kron(silk, hemp))

    # --- the law at larger shapes --------------------------------------------
    for _ in range(10):
        a, b, c, d = (random_matrix(rng, 4) for _ in range(4))
        assert np.allclose(np.kron(a, b) @ np.kron(c, d), np.kron(a @ c, b @ d))
        assert np.kron(a, b).shape == (16, 16)
        assert not np.allclose(np.kron(a, b) @ np.kron(c, d), np.kron(c @ a, d @ b))
    # and with one factor a column: each box acts on its own wire
    for _ in range(10):
        a, b = random_matrix(rng), random_matrix(rng)
        psi = random_matrix(rng)[:, 0:1]
        phi = random_matrix(rng)[:, 0:1]
        assert np.allclose(np.kron(a, b) @ np.kron(psi, phi), np.kron(a @ psi, b @ phi))
    # and with rows above and columns below: Proposition 4.1 is the small case
    psi = random_matrix(rng)[:, 0:1]
    phi = random_matrix(rng)[:, 0:1]
    r1, r2 = random_matrix(rng)[0:1], random_matrix(rng)[0:1]
    assert np.allclose(np.kron(r1, r2) @ np.kron(psi, phi), (r1 @ psi) * (r2 @ phi))

    pictures()


# --------------------------------------------------------------------------
# The pictures (pdf/props/ch04.tex; the Lua's rules transcribed in
# tools/props_ch04.py), read out of this challenge's section of dialog.md.
# Each frame's walk is checked, from every start, against the columns'
# matrices; then against the matrix the chapter writes for it.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("stringframe", "foldsheet"))
NOT_IN_CAPTIONS = ("strings", "wires", "wire", "diagram", "diagrams", "drawing", "drawings",
                   "picture", "pictures", "box", "boxes")
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
GATES = {"over": X, "heads": S0, "yoke": CNOT, "cross": SWAP}
I4 = np.eye(4, dtype=complex)


def clips_on(fr, name):
    return [(col, rules) for col, n, row, rules in fr.clips() if n == name]


def pictures():
    p41, p42, p43, p44 = DIALOGUE.check_layout("Fourth", sizes=[1, 1, 1, 1])
    for c in (p41, p42, p43, p44):
        assert c.macro == "stringframe" and not c.keyed
        assert words_only(c.caption, forbidden=NOT_IN_CAPTIONS), c.caption
    f1, f2, f3, f4 = (P.picture(c) for c in (p41, p42, p43, p44))
    for fr in (f1, f2, f3, f4):
        assert fr.place == ("end", None) and fr.hooked == ("silk", "hemp", "gut", "horsehair")
        assert fr.yokes() == [] and fr.crossings() == []

    # P4.1: her clip to the left of his; heads above, tails below
    assert clips_on(f1, "silk") == [(2, ("over",))] and clips_on(f1, "hemp") == [(1, ("heads",))]
    M1 = P.check_walks(f1, GATES)
    assert np.allclose(M1, np.kron(np.kron(X, I2) @ np.kron(I2, S0), I4))   # hemp's first
    assert f1.coins() == {"silk": ("tails", 1), "hemp": ("heads", 2)}
    assert np.allclose(np.kron(X, S0) @ ket(0, 1), ket(1, 0))

    # P4.2: slid past his; the same arrival, from every start (Sliding)
    assert clips_on(f2, "silk") == [(2, ("over",))] and clips_on(f2, "hemp") == [(3, ("heads",))]
    M2 = P.check_walks(f2, GATES)
    assert np.allclose(M2, np.kron(np.kron(I2, S0) @ np.kron(X, I2), I4))   # silk's first
    assert np.allclose(M1, M2) and np.allclose(M1, np.kron(np.kron(X, S0), I4))
    assert f2.coins() == f1.coins()
    for start in P.starts(f1):
        assert f1.walk(f1.ncol, start) == f2.walk(f2.ncol, start)

    # P4.3: nothing moved -- first clips level, second clips level
    assert clips_on(f3, "silk") == [(2, ("over",)), (3, ("heads",))]
    assert clips_on(f3, "hemp") == [(2, ("heads",)), (3, ("over",))]
    M3 = P.check_walks(f3, GATES)
    by_columns = np.kron(S0, X) @ np.kron(X, S0)             # both firsts, then both seconds
    by_strings = np.kron(I2, X @ S0) @ np.kron(S0 @ X, I2)   # all of the silk, then the hemp
    assert np.allclose(by_columns, by_strings)               # the Interchange Law
    assert np.allclose(M3, np.kron(by_columns, I4))
    assert np.allclose(M3, np.kron(np.kron(S0 @ X, X @ S0), I4))
    assert f3.coins() == {"silk": ("heads", 1), "hemp": ("tails", 2)}
    for start in P.starts(f3):                               # whatever the four coins
        face = f3.walk(f3.ncol, start)
        assert (face["silk"], face["hemp"]) == ("heads", "tails")
    # the transcribed places, on this frame: stopped before a column, and past it
    for k, word in ((1, "first"), (2, "second"), (3, "third")):
        at = P.stringframe("at-" + word, p43.args[0]).coins()
        past = P.stringframe("past-" + word, p43.args[0]).coins()
        assert at == {n: (f, f3.row_after(n, k - 1))
                      for n, f in f3.walk(k - 1).items() if f}
        assert past == {n: (f, f3.row_after(n, k)) for n, f in f3.walk(k).items() if f}
    assert P.stringframe("off", p43.args[0]).coins() == {}
    assert P.faults(P.stringframe, "at-fourth", p43.args[0]) != []     # no fourth column

    # P4.4: off and on again -- heads up now before turn it over, on the silk
    assert clips_on(f4, "silk") == [(3, ("heads",)), (4, ("over",))]
    assert [r for c, r in clips_on(f4, "hemp")] == [r for c, r in clips_on(f3, "hemp")]
    assert [c for c, r in clips_on(f4, "hemp")] == [2, 3]   # the hemp clips unmoved
    assert f4.face["hemp"] is None                           # the hemp coin laid aside
    assert f4.coins() == {"silk": ("tails", 1)}               # silk arrives tails ...
    assert f3.coins()["silk"] == ("heads", 1)                 # ... a moment ago heads
    M4 = P.check_walks(f4, GATES)
    assert np.allclose(M4, np.kron(np.kron(X @ S0, X @ S0), I4))
    assert not np.allclose(X @ S0, S0 @ X)                   # along one string, an order
    assert np.allclose(X @ S0 @ ket(1), ket(1)) and np.allclose(S0 @ X @ ket(1), ket(0))


# --------------------------------------------------------------------------
# The sketchbook.  Rows are wires, row 0 is the top one; `g1 @ g2` stacks g1
# above g2, `g1 + g2` puts g1 first along the wire.  The chains below are
# unitary, so the law for boxes that keep no promise is checked on the
# blackboard only; what the pictures check is that the two READINGS of one
# drawing agree.
# --------------------------------------------------------------------------
def box(phases):
    p = Picture()
    a, b, c = phases
    # PyZX draws this with the spiders you meet in Chapter 8.
    vs = [p.z(0, 1, a), p.x(0, 2, b), p.z(0, 3, c)]
    p.wire(p.inp(0), *vs, p.out(0))
    return p.done()


def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def sketchbook():
    W = bare_wire()
    a = box((Fraction(1, 4), Fraction(1, 3), Fraction(3, 4)))
    b = box((Fraction(1, 2), Fraction(1, 4), Fraction(1, 3)))
    c = box((Fraction(1, 3), Fraction(1, 2), Fraction(1, 4)))
    d = box((Fraction(3, 4), Fraction(1, 3), Fraction(1, 2)))
    ms = [matrix_of(g) for g in (a, b, c, d)]
    generic(*ms)

    # --- the theorem: one drawing, two readings ------------------------------
    by_columns = (c @ d) + (a @ b)          # the column C,D and then the column A,B
    by_wires = (c + a) @ (d + b)            # the top wire C then A, the bottom D then B
    assert same_picture(by_columns, by_wires)
    assert np.allclose(matrix_of(by_columns),
                       np.kron(ms[0] @ ms[2], ms[1] @ ms[3]))
    # the wrong pairings, drawn: the boxes crossed from wire to wire
    assert not same_picture(by_columns, (c + a) @ (b + d))
    assert not same_picture(by_columns, (a + c) @ (d + b))
    assert not same_picture(by_columns, (d + b) @ (c + a))

    # --- Sliding: left, right and level are one drawing ----------------------
    slid_left = (W @ b) + (a @ W)
    slid_right = (a @ W) + (W @ b)
    level = a @ b
    assert same_picture(slid_left, level) and same_picture(slid_right, level)
    assert same_picture(slid_left, slid_right)
    assert not same_picture(level, b @ a)              # up and down is not free

    # --- along one wire, two boxes cannot pass each other --------------------
    assert not same_picture(a + c, c + a)
    assert not same_picture((a + c) @ W, (c + a) @ W)
    assert np.allclose(matrix_of(a + c), ms[2] @ ms[0])     # C after A: the row reads left
    # and a box on a different wire may still be slid past either of them
    assert same_picture((a + c) @ b, (a @ b) + (c @ W))
    assert same_picture((a + c) @ b, (a @ W) + (c @ b))


if __name__ == "__main__":
    verdict("Interchange", blackboard, sketchbook)
