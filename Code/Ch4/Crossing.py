"""Challenge 5: The Crossing
=============================
"The clip kept its string; the string changed its place."  -- Achilles

Claim: the crossing is SWAP, the matrix that exchanges the two labels of every
pair.  Its two rules: a box slides through a crossing and changes row,
SWAP (A (x) B) = (B (x) A) SWAP; and two crossings are two bare wires,
SWAP.SWAP = I.  SWAP is itself no pair of boxes: it is not A (x) B for any
A and B.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import I2, Picture, ket, matrix_of, random_matrix, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch04 as P

X = np.array([[0, 1], [1, 0]], dtype=complex)


def reshuffle(M):
    """The 4x4 matrix rewritten so that A (x) B becomes a rank-one matrix."""
    return M.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)


def blackboard():
    rng = np.random.default_rng(45)

    # --- SWAP, read off the rule |ab> -> |ba> --------------------------------
    SWAP = np.column_stack([ket(b, a) for a in (0, 1) for b in (0, 1)])
    assert np.allclose(SWAP, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    assert np.allclose(SWAP @ ket(0, 1), ket(1, 0))
    assert np.allclose(SWAP @ ket(1, 0), ket(0, 1))
    assert np.allclose(SWAP @ ket(0, 0), ket(0, 0))
    assert np.allclose(SWAP @ ket(1, 1), ket(1, 1))
    assert np.allclose(SWAP, SWAP.real) and np.allclose(SWAP, SWAP.T)
    assert np.allclose(SWAP.conj().T @ SWAP, np.eye(4))       # a unitary box
    assert np.allclose(SWAP @ SWAP, np.eye(4))                # twice is nothing

    # --- it exchanges the factors of a product ------------------------------
    for _ in range(30):
        psi, phi = random_state(rng), random_state(rng)
        assert not np.allclose(psi.imag, 0)
        assert np.allclose(SWAP @ np.kron(psi, phi), np.kron(phi, psi))
        assert not np.allclose(SWAP @ np.kron(psi, phi), np.kron(psi, phi))
        assert np.allclose(SWAP @ SWAP @ np.kron(psi, phi), np.kron(psi, phi))
    arrow = np.array([3 / 5, 4 / 5], dtype=complex)
    omega = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    assert np.allclose(SWAP @ np.kron(arrow, omega), np.kron(omega, arrow))
    assert np.allclose(SWAP @ np.kron(arrow, omega),
                       np.array([3, 4, 3j, 4j]) / (5 * np.sqrt(2)))

    # --- a box slides through and changes row --------------------------------
    for _ in range(30):
        A, B = random_matrix(rng), random_matrix(rng)
        assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())
        assert np.allclose(SWAP @ np.kron(A, B), np.kron(B, A) @ SWAP)
        assert not np.allclose(SWAP @ np.kron(A, B), np.kron(A, B) @ SWAP)   # must change row
        assert not np.allclose(SWAP @ np.kron(A, B), np.kron(B.T, A.T) @ SWAP)
        assert not np.allclose(SWAP @ np.kron(A, B), np.kron(B, A).conj().T @ SWAP)
        assert np.allclose(SWAP @ np.kron(A, B) @ SWAP, np.kron(B, A))
        assert not np.allclose(np.kron(A, B), np.kron(B, A))
    # the only way two stacked boxes commute is one being a multiple of the other
    A = random_matrix(rng)
    lam = 1.3 - 0.7j
    assert np.allclose(np.kron(A, lam * A), np.kron(lam * A, A))

    # --- SWAP is not a pair of boxes -----------------------------------------
    for _ in range(20):
        A, B = random_matrix(rng), random_matrix(rng)
        assert np.linalg.matrix_rank(reshuffle(np.kron(A, B))) == 1
    assert np.linalg.matrix_rank(reshuffle(SWAP)) == 4
    assert np.linalg.matrix_rank(reshuffle(np.kron(X, I2))) == 1
    # the elementary argument of Sec 4.6, in numbers: the upper left block of
    # a product is a multiple of the lower right block, and SWAP's are not
    upper_left, lower_right = SWAP[:2, :2], SWAP[2:, 2:]
    assert np.allclose(upper_left, [[1, 0], [0, 0]])
    assert np.allclose(lower_right, [[0, 0], [0, 1]])
    assert np.linalg.matrix_rank(np.stack([upper_left.ravel(), lower_right.ravel()])) == 2

    # --- the dialogue's coin walks -------------------------------------------
    # crossed once: the coins change hooks, and neither changes its face
    assert np.allclose(SWAP @ ket(0, 1), ket(1, 0))
    # "turn it over" on the gut string, left of the crossing, then slid right
    before = SWAP @ np.kron(X, I2)
    after = np.kron(I2, X) @ SWAP
    assert np.allclose(before, after)
    assert np.allclose(before @ ket(0, 0), ket(0, 1))     # heads on the gut's hook
    for a in (0, 1):
        for b in (0, 1):
            assert np.allclose(before @ ket(a, b), after @ ket(a, b))
    # the same clip on the horsehair string instead
    lower_before = SWAP @ np.kron(I2, X)
    lower_after = np.kron(X, I2) @ SWAP
    assert np.allclose(lower_before, lower_after)
    assert not np.allclose(before, lower_before)
    # crossed again the same way round: every coin on its own hook, face unchanged
    assert np.allclose(SWAP @ SWAP, np.kron(I2, I2))
    for a in (0, 1):
        for b in (0, 1):
            assert np.allclose(SWAP @ SWAP @ ket(a, b), ket(a, b))

    pictures(SWAP)


# --------------------------------------------------------------------------
# The pictures (pdf/props/ch04.tex; the Lua's rules transcribed in
# tools/props_ch04.py), read out of this challenge's section of dialog.md.
# The macro computes each string's row column by column, the row a clip sits
# on after a crossing, and which string passes in front at each crossing (the
# one that was upper there).  The coins' arrival never reads that last fact:
# the string remembers which went in front, the coin cannot tell.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("stringframe", "foldsheet"))
NOT_IN_CAPTIONS = ("strings", "wires", "wire", "diagram", "diagrams", "drawing", "drawings",
                   "picture", "pictures", "box", "boxes")
S0 = np.array([[1, 1], [0, 0]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
I4 = np.eye(4, dtype=complex)


def pictures(SWAP):
    GATES = {"over": X, "heads": S0, "yoke": CNOT, "cross": SWAP}
    p51, p52, p53 = DIALOGUE.check_layout("Fifth", sizes=[1, 1, 1])
    for c in (p51, p52, p53):
        assert c.macro == "stringframe" and not c.keyed
        assert words_only(c.caption, forbidden=NOT_IN_CAPTIONS), c.caption
    f1, f2, f3 = (P.picture(c) for c in (p51, p52, p53))
    for fr in (f1, f2, f3):
        assert fr.place == ("end", None) and fr.hooked == ("silk", "hemp", "gut", "horsehair")
        assert fr.yokes() == []

    # P5.1: crossed once, midway, the gut (upper) in front; hooks exchanged, faces kept
    assert f1.ncol == 3 and f1.crossings() == [(2, "gut", "horsehair")]
    assert f1.rows["gut"] == [3, 3, 4, 4] and f1.rows["horsehair"] == [4, 4, 3, 3]
    assert f1.clips() == []
    M1 = P.check_walks(f1, GATES)
    assert np.allclose(M1, np.kron(I4, SWAP))
    assert not np.allclose(M1, np.eye(16))
    assert f1.coins() == {"gut": ("heads", 4), "horsehair": ("tails", 3)}
    assert np.allclose(P.column({1: "tails", 2: "heads"}, 2), SWAP @ ket(0, 1))

    # P5.2: the clip slid through -- still on the gut, now on the lower row
    assert f2.crossings() == [(2, "gut", "horsehair")]
    assert f2.clips() == [(3, "gut", 4, ("over",))]
    M2 = P.check_walks(f2, GATES)
    assert np.allclose(M2, np.kron(I4, np.kron(I2, X) @ SWAP))
    assert np.allclose(M2, np.kron(I4, SWAP @ np.kron(X, I2)))     # slid: row changed
    assert f2.coins() == {"gut": ("tails", 4), "horsehair": ("heads", 3)}
    # the clip before the crossing (l.421): the same arrival, face for face
    before = P.stringframe("end", "silk; hemp; gut heads: over, cross, -; "
                                  "horsehair heads: -, cross, -")
    assert before.clips() == [(1, "gut", 3, ("over",))]
    assert np.allclose(P.check_walks(before, GATES), M2)
    assert before.coins() == f2.coins()
    for start in P.starts(f2):
        assert before.walk(3, start) == f2.walk(3, start)
    # Achilles's guess -- the clip turns the horsehair coin -- is another frame
    guess = P.stringframe("end", "silk; hemp; gut heads: -, cross, -; "
                                 "horsehair heads: -, cross, over")
    assert guess.clips() == [(3, "horsehair", 3, ("over",))]
    assert guess.coins() != f2.coins()
    assert not np.allclose(P.check_walks(guess, GATES), M2)

    # P5.3: crossed again the same way round -- wound, each string in front once
    assert f3.ncol == 4
    assert f3.crossings() == [(2, "gut", "horsehair"), (3, "horsehair", "gut")]
    assert sorted(front for col, front, behind in f3.crossings()) == ["gut", "horsehair"]
    assert f3.rows["gut"][-1] == 3 and f3.rows["horsehair"][-1] == 4   # own hooks
    M3 = P.check_walks(f3, GATES)
    assert np.allclose(M3, np.kron(I4, SWAP @ SWAP)) and np.allclose(M3, np.eye(16))
    assert f3.coins() == {"silk": ("heads", 1), "hemp": ("tails", 2),
                          "gut": ("heads", 3), "horsehair": ("tails", 4)}
    # the same report as the straight pair: every coin home with its face
    for n, (face, row) in f3.coins().items():
        assert face == f3.face[n] and row == P.HOOKROW[n]
    # and the coin cannot tell wound from straight: the walk never reads the front
    straight = P.stringframe("end", "silk heads; hemp tails; gut heads; horsehair tails")
    assert straight.crossings() == []
    for start in P.starts(f3):
        assert f3.coins(start) == straight.coins(start)
    assert np.allclose(P.check_walks(straight, GATES), M3)

    # the macro's own faults: a crossing needs its partner on the next row
    assert P.faults(P.stringframe, "end", "gut: cross; horsehair") != []
    assert P.faults(P.stringframe, "end", "silk: cross; gut: cross") != []


# --------------------------------------------------------------------------
# The sketchbook.  Rows are wires, row 0 is the top one; `g1 @ g2` stacks g1
# above g2, `g1 + g2` puts g1 first along the wire.  A crossing is drawn by
# wiring the upper input to the lower output and the lower input to the upper:
# flat, with no node at all, so the drawing records which output each input
# reaches and nothing else.  "SWAP is no pair of boxes" is a property of one
# matrix and stays on the blackboard.
# --------------------------------------------------------------------------
def crossing():
    p = Picture()
    i0, i1 = p.inp(0), p.inp(1)
    o0, o1 = p.out(0), p.out(1)
    p.wire(i0, o1)
    p.wire(i1, o0)
    return p.done()


def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def box(phases):
    p = Picture()
    a, b, c = phases
    # PyZX draws this with the spiders you meet in Chapter 8.
    vs = [p.z(0, 1, a), p.x(0, 2, b), p.z(0, 3, c)]
    p.wire(p.inp(0), *vs, p.out(0))
    return p.done()


def not_box():
    p = Picture()
    n = p.x(0, 1, 1)
    p.wire(p.inp(0), n, p.out(0))
    return p.done()


def basis(b):
    p = Picture()
    n = p.x(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def state(phases):
    p = Picture()
    n = p.x(0, 1, 0)
    a, b, c = phases
    vs = [p.z(0, 2, a), p.x(0, 3, b), p.z(0, 4, c)]
    p.wire(n, *vs, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def sketchbook():
    S, W = crossing(), bare_wire()
    assert np.allclose(matrix_of(S),
                       [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    # --- two states stacked, crossed, are the two states the other way round -
    s1 = state((Fraction(1, 4), Fraction(1, 3), Fraction(3, 4)))
    s2 = state((Fraction(1, 2), Fraction(1, 4), Fraction(1, 3)))
    assert not np.allclose(matrix_of(s1).ravel(), matrix_of(s2).ravel())
    assert same_picture((s1 @ s2) + S, s2 @ s1)
    assert not same_picture((s1 @ s2) + S, s1 @ s2)
    for a in (0, 1):
        for b in (0, 1):
            assert same_picture((basis(a) @ basis(b)) + S, basis(b) @ basis(a))

    # --- a box slides through a crossing and changes row ---------------------
    A = box((Fraction(1, 4), Fraction(1, 3), Fraction(3, 4)))
    Bx = box((Fraction(1, 2), Fraction(1, 4), Fraction(1, 3)))
    assert same_picture((A @ Bx) + S, S + (Bx @ A))
    assert not same_picture((A @ Bx) + S, S + (A @ Bx))     # it must change row
    assert not same_picture((A @ Bx) + S, S + (Bx @ Bx))
    # the clip on either string, left of the crossing and right of it
    assert same_picture((A @ W) + S, S + (W @ A))
    assert same_picture((W @ A) + S, S + (A @ W))
    assert not same_picture((A @ W) + S, S + (A @ W))
    N = not_box()
    assert same_picture((N @ W) + S, S + (W @ N))           # the gut clip, slid through
    assert same_picture((basis(0) @ basis(0)) + (N @ W) + S, basis(0) @ basis(1))

    # --- two crossings are two bare wires ------------------------------------
    assert same_picture(S + S, W @ W)
    assert not same_picture(S, W @ W)                       # one crossing is not
    assert same_picture(S + S + S, S)
    # and a box may be slid through both, arriving on the row it set out from
    assert same_picture((A @ Bx) + S + S, A @ Bx)


if __name__ == "__main__":
    verdict("Crossing", blackboard, sketchbook)
