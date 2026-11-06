"""Challenge 2: Sixteen Panels
===============================
"Eight. ...Oh. The new fold didn't add two panels at the edge. It went through
every panel that was already there."  -- Achilles

Claim: each further wire DOUBLES the column -- n wires carry 2^n entries, not
2n -- while n separate states are fixed by 2n numbers, two apiece.  The two
counts agree at one wire and at two, and part for good from three on.

The sketchbook half of this file is a size claim, not a second proof: it checks
that the picture's own tensor has 2^n entries while the drawing costs n lines.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import I2, Picture, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch04 as P


def blackboard():
    rng = np.random.default_rng(42)

    # --- the two counts, side by side ---------------------------------------
    for n in (1, 2, 3, 4):
        states = [random_state(rng) for _ in range(n)]
        column = np.array([1.0 + 0j])
        for s in states:
            column = np.kron(column, s)
        assert column.shape == (2 ** n,)                  # one entry per string
        assert np.isclose(np.linalg.norm(column), 1)      # and it is a state
        separate = np.concatenate(states)
        assert separate.shape == (2 * n,)                 # two numbers apiece
        if n <= 2:
            assert 2 ** n == 2 * n                        # the trap is real here
        else:
            assert 2 ** n > 2 * n                         # and gone from three on
    assert (2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4) == (2, 4, 8, 16)
    assert 2 ** 3 != 6 and 2 ** 4 != 8                    # "So a third string makes six."
    assert 2 ** 10 == 1024 and 2 * 10 == 20

    # --- bracketing does not matter -----------------------------------------
    p, q, r = (random_state(rng) for _ in range(3))
    left = np.kron(np.kron(p, q), r)
    right = np.kron(p, np.kron(q, r))
    assert np.allclose(left, right)
    for i in (0, 1):
        for k in (0, 1):
            for m in (0, 1):
                assert np.isclose(left[4 * i + 2 * k + m], p[i] * q[k] * r[m])
    assert not np.allclose(left, np.kron(np.kron(r, q), p))   # order still matters

    # --- nothing on any of four wires ---------------------------------------
    ident = np.array([[1.0 + 0j]])
    for _ in range(4):
        ident = np.kron(ident, I2)
    assert ident.shape == (16, 16)
    assert np.allclose(ident, np.eye(16))

    # a product column of four states is fixed by eight numbers: rebuild it
    four = [random_state(rng) for _ in range(4)]
    eight = np.concatenate(four)
    rebuilt = np.array([1.0 + 0j])
    for j in range(4):
        rebuilt = np.kron(rebuilt, eight[2 * j:2 * j + 2])
    big = np.array([1.0 + 0j])
    for s in four:
        big = np.kron(big, s)
    assert rebuilt.shape == (16,) and np.allclose(rebuilt, big)

    pictures()


# --------------------------------------------------------------------------
# The pictures (pdf/props/ch04.tex; the Lua's rules transcribed in
# tools/props_ch04.py), read out of this challenge's section of dialog.md.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("stringframe", "foldsheet"))
NOT_IN_CAPTIONS = ("strings", "wires", "wire", "diagram", "diagrams", "drawing", "drawings",
                   "picture", "pictures", "box", "boxes")
X = np.array([[0, 1], [1, 0]], dtype=complex)
S0 = np.array([[1, 1], [0, 0]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
GATES = {"over": X, "heads": S0, "yoke": CNOT, "cross": SWAP}


def pictures():
    p21, p22, p23 = DIALOGUE.check_layout("Second", sizes=[1, 1, 1])
    for c in (p21, p22, p23):
        assert words_only(c.caption, forbidden=NOT_IN_CAPTIONS), c.caption

    # the macro's folds: each fold goes through every panel there is
    for n, word in enumerate(("none", "once", "twice", "three", "four")):
        assert P.foldsheet(word, "blank").panels == 2 ** n
    assert P.faults(P.foldsheet, "five", "blank") != []    # no fifth crease

    # P2.1: the third fold, across again -- eight panels, not six
    sh = P.picture(p21)
    assert p21.macro == "foldsheet" and not p21.keyed
    assert sh.shape == (4, 2)                               # four rows of two
    assert sh.panels == 2 ** 3 == 8 and sh.panels != 6      # "Two, four, six... eight."
    assert sh.shares is None and sh.arrows == [] and sh.tallies is None

    # P2.2: the whole frame, the names at their hooks, bare, no coin
    fr = P.picture(p22)
    assert p22.macro == "stringframe" and p22.keyed and fr.place == ("start", None)
    assert fr.hooked == ("silk", "hemp", "gut", "horsehair") and fr.ncol == 0
    assert fr.rows == {"silk": [1], "hemp": [2], "gut": [3], "horsehair": [4]}
    assert fr.labels == {"silk": 1, "hemp": 2, "gut": 3, "horsehair": 4}
    assert fr.clips() == [] and fr.yokes() == [] and fr.crossings() == [] and fr.coins() == {}
    bare = P.check_walks(fr, GATES)
    assert bare.shape == (2 ** 4, 2 ** 4) and np.allclose(bare, np.eye(16))

    # P2.3: the fourth fold -- sixteen, where the separate coins need eight lines
    sh = P.picture(p23)
    assert p23.macro == "foldsheet" and not p23.keyed
    assert sh.shape == (4, 4) and sh.panels == 2 ** len(fr.hooked) == 16
    assert 2 * len(fr.hooked) == 8 != sh.panels             # "Eight lines in all."


# --------------------------------------------------------------------------
# The sketchbook.  Rows are wires, row 0 is the top one; `g1 @ g2` stacks g1
# above g2.
# --------------------------------------------------------------------------
def bare_wires(n):
    p = Picture()
    for row in range(n):
        p.wire(p.inp(row), p.out(row))
    return p.done()


def basis_state(b, row=0):
    p = Picture()
    n = p.x(row, 1, b)      # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(n, p.out(row))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def some_state(phases):
    p = Picture()
    n = p.x(0, 1, 0)
    a, b, c = phases
    vs = [p.z(0, 2, a), p.x(0, 3, b), p.z(0, 4, c)]
    p.wire(n, *vs, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def sketchbook():
    # --- n lines of ink, 2^n entries ----------------------------------------
    for n in (1, 2, 3, 4):
        g = bare_wires(n)
        assert g.num_edges() == n                       # n lines drawn
        m = matrix_of(g)
        assert m.shape == (2 ** n, 2 ** n)              # 2^n entries down each side
        assert np.allclose(m, np.eye(2 ** n))
    # stacking is what does it: one wire below another, not two more entries
    assert same_picture(bare_wires(1) @ bare_wires(1), bare_wires(2))
    assert same_picture(bare_wires(2) @ bare_wires(2), bare_wires(4))
    assert matrix_of(bare_wires(3)).shape != (6, 6)

    # --- stacked states: the column doubles with each -------------------------
    phases = [(Fraction(1, 4), Fraction(1, 3), Fraction(3, 4)),
              (Fraction(1, 2), Fraction(1, 4), Fraction(1, 3)),
              (Fraction(1, 3), Fraction(1, 2), Fraction(1, 4)),
              (Fraction(3, 4), Fraction(1, 3), Fraction(1, 2))]
    pics = [some_state(ph) for ph in phases]
    cols = [matrix_of(g).ravel() for g in pics]
    assert not np.allclose(cols[0].imag, 0)
    stack = pics[0]
    want = cols[0]
    for n in (2, 3, 4):
        stack = stack @ pics[n - 1]
        want = np.kron(want, cols[n - 1])
        got = matrix_of(stack).ravel()
        assert got.shape == (2 ** n,)
        assert np.allclose(got, want)

    # --- and the basis states stacked, for all sixteen strings of four -------
    for j in range(16):
        bits = [(j >> k) & 1 for k in (3, 2, 1, 0)]
        g = basis_state(bits[0])
        for b in bits[1:]:
            g = g @ basis_state(b)
        col = np.zeros(16, dtype=complex)
        col[j] = 1
        assert np.allclose(matrix_of(g).ravel(), col)   # one entry per string


if __name__ == "__main__":
    verdict("Doubling", blackboard, sketchbook)
