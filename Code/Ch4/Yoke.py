"""Challenge 6: The Yoke
=========================
"The stick isn't holding two clips together. The stick is how the lower one
looks up."  -- Achilles

Claim: CNOT, the box across two wires with the rule |a,b> -> |a, a (+) b>, is
not A (x) B for any pair of one-wire boxes; done twice it is nothing; and from
a tossed coin above and a plain one below it makes a matched pair, a list of
shares that is no product of two lists.  The master picture, Twice Yoked, Twice
Crossed, is bare wire on every row: (CNOT (x) SWAP)^2 = I_16.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import I2, Picture, ket, matrix_of, random_matrix, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch04 as P

X = np.array([[0, 1], [1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def reshuffle(M):
    """The 4x4 matrix rewritten so that A (x) B becomes a rank-one matrix."""
    return M.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)


def blackboard():
    rng = np.random.default_rng(46)

    # --- CNOT, read off the rule |a,b> -> |a, a+b mod 2> --------------------
    CNOT = np.column_stack([ket(a, a ^ b) for a in (0, 1) for b in (0, 1)])
    assert np.allclose(CNOT, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert np.allclose(CNOT[:2, :2], I2) and np.allclose(CNOT[2:, 2:], X)
    assert np.allclose(CNOT[:2, 2:], 0) and np.allclose(CNOT[2:, :2], 0)
    # the dialogue's walks, with heads = 0: the upper coin is never touched
    assert np.allclose(CNOT @ ket(0, 0), ket(0, 0))
    assert np.allclose(CNOT @ ket(0, 1), ket(0, 1))
    assert np.allclose(CNOT @ ket(1, 0), ket(1, 1))
    assert np.allclose(CNOT @ ket(1, 1), ket(1, 0))

    # --- it is real, symmetric, unitary, stochastic, and its own undoing -----
    assert np.allclose(CNOT, CNOT.real) and np.allclose(CNOT, CNOT.T)
    assert np.allclose(CNOT.conj().T @ CNOT, np.eye(4))
    assert np.all(CNOT.real >= 0) and np.allclose(CNOT.real.sum(axis=0), 1)
    assert np.allclose(CNOT @ CNOT, np.eye(4))
    assert np.allclose(CNOT @ CNOT, np.kron(I2, I2))

    # --- no pair of one-wire boxes does it -----------------------------------
    assert np.linalg.matrix_rank(reshuffle(CNOT)) == 2
    for _ in range(30):
        P, Q = random_matrix(rng), random_matrix(rng)
        assert np.linalg.matrix_rank(reshuffle(np.kron(P, Q))) == 1
        assert not np.allclose(np.kron(P, Q), CNOT)
    # Achilles's two attempts, exactly as he made them
    clip_on_hemp = np.kron(I2, X)             # "turn it over" on the lower string
    no_clip = np.kron(I2, I2)
    assert np.allclose(clip_on_hemp @ ket(1, 0), CNOT @ ket(1, 0))      # "Right!"
    assert not np.allclose(clip_on_hemp @ ket(0, 0), CNOT @ ket(0, 0))  # "Wrong."
    assert np.allclose(no_clip @ ket(0, 0), CNOT @ ket(0, 0))
    assert not np.allclose(no_clip @ ket(1, 0), CNOT @ ket(1, 0))
    # the block argument of Proposition 4.5, in numbers
    assert np.linalg.matrix_rank(np.stack([CNOT[:2, :2].ravel(), CNOT[2:, 2:].ravel()])) == 2

    # --- the tossed coin and the plain one, on shares ------------------------
    tossed = np.array([1 / 2, 1 / 2])
    heads = np.array([1.0, 0.0])
    joint = np.kron(tossed, heads)
    assert np.allclose(joint, [1 / 2, 0, 1 / 2, 0])
    assert np.linalg.matrix_rank(joint.reshape(2, 2)) == 1      # still a product of two lists
    matched = CNOT.real @ joint
    assert np.allclose(matched, [1 / 2, 0, 0, 1 / 2])      # both heads or both tails
    assert np.isclose(matched.sum(), 1)
    # not a product of two lists: the empty panel would have to empty a full one
    assert np.linalg.matrix_rank(matched.reshape(2, 2)) == 2
    for _ in range(30):
        p = rng.random(2)
        q = rng.random(2)
        assert not np.allclose(np.kron(p, q), matched)
        # a pair of clips carries a product list to a product list
        P, Q = random_matrix(rng), random_matrix(rng)
        out = np.kron(P, Q) @ np.kron(p, q)
        assert np.linalg.matrix_rank(out.reshape(2, 2)) <= 1
        assert np.allclose(out, np.kron(P @ p, Q @ q))
    # the second yoke gives the coins back: "the hemp coin arrives heads every time"
    assert np.allclose(CNOT.real @ matched, joint)
    for a in (0, 1):
        for b in (0, 1):
            assert np.allclose(CNOT @ CNOT @ ket(a, b), ket(a, b))

    # --- the master picture: Twice Yoked, Twice Crossed ----------------------
    I4 = np.eye(4, dtype=complex)
    paired = np.kron(CNOT, SWAP)
    assert paired.shape == (16, 16)
    assert np.allclose(paired @ paired, np.eye(16))
    alternating = (np.kron(I4, SWAP) @ np.kron(CNOT, I4)
                   @ np.kron(I4, SWAP) @ np.kron(CNOT, I4))
    assert np.allclose(alternating, np.eye(16))
    assert np.allclose(alternating, paired @ paired)
    assert np.allclose(np.kron(I4, SWAP) @ np.kron(CNOT, I4), paired)   # sliding
    assert np.allclose(np.kron(CNOT, I4) @ np.kron(I4, SWAP), paired)
    # with any two boxes on the upper pair: the crossed pair contributes nothing
    for _ in range(10):
        P, Q = random_matrix(rng, 4), random_matrix(rng, 4)
        whole = (np.kron(I4, SWAP) @ np.kron(Q, I4)
                 @ np.kron(I4, SWAP) @ np.kron(P, I4))
        assert np.allclose(whole, np.kron(Q @ P, I4))
        assert not np.allclose(whole, np.kron(P @ Q, I4))
        assert not np.allclose(whole, np.kron(Q @ P, SWAP))

    pictures(CNOT, matched, alternating)


# --------------------------------------------------------------------------
# The pictures (pdf/props/ch04.tex; the Lua's rules transcribed in
# tools/props_ch04.py), read out of this challenge's section of dialog.md.
# The macro's yoke looks at the coin on the upper of its two rows as it
# arrives at the bar and turns the lower one for tails; here that walk is
# checked against CNOT from every start, and the closing frame against the
# master picture.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("stringframe", "foldsheet"))
NOT_IN_CAPTIONS = ("strings", "wires", "wire", "diagram", "diagrams", "drawing", "drawings",
                   "picture", "pictures", "box", "boxes")
S0 = np.array([[1, 1], [0, 0]], dtype=complex)
I4 = np.eye(4, dtype=complex)
CLIP_CELLS = ("-", "over", "heads", "over+heads", "heads+over")


def pictures(CNOT, matched, alternating):
    GATES = {"over": X, "heads": S0, "yoke": CNOT, "cross": SWAP}
    p61, p62, p63, p64 = DIALOGUE.check_layout("Sixth", sizes=[1, 1, 1, 1])
    for c in (p61, p62, p63, p64):
        assert not c.keyed
        assert words_only(c.caption, forbidden=NOT_IN_CAPTIONS), c.caption
    assert "yoked" not in p64.caption.lower().split()       # never the master's name
    f1, f2, sh, f4 = (P.picture(c) for c in (p61, p62, p63, p64))
    wound = [(2, "gut", "horsehair"), (3, "horsehair", "gut")]

    # P6.1: the bar near the left upright; tails above, heads below
    assert p61.macro == "stringframe" and f1.place == ("end", None)
    assert f1.yokes() == [(1, "silk", "hemp", 1, 2)] and f1.clips() == []
    assert f1.crossings() == wound                          # the lower pair still wound
    M1 = P.check_walks(f1, GATES)
    assert np.allclose(M1, np.kron(CNOT, I4) @ np.kron(I4, SWAP @ SWAP))
    assert np.allclose(M1, np.kron(CNOT, I4))
    assert f1.coins() == {"silk": ("tails", 1), "hemp": ("tails", 2)}   # no coins below
    assert np.allclose(P.column({1: "tails", 2: "tails"}, 2), CNOT @ ket(1, 0))

    # P6.2: no clip, no bar -- tails above, heads below arrives unchanged
    assert f2.yokes() == [] and f2.clips() == [] and f2.crossings() == wound
    assert f2.face == f1.face                               # the same pair laid
    assert [f2.cells[n] for n in ("gut", "horsehair")] == [f1.cells[n] for n in ("gut", "horsehair")]
    M2 = P.check_walks(f2, GATES)
    assert np.allclose(M2, np.eye(16))
    assert f2.coins() == {"silk": ("tails", 1), "hemp": ("heads", 2)}
    assert f2.coins()["hemp"] != f1.coins()["hemp"]         # wrong for tails
    # no clips on the silk and the hemp, of any kind in any order, walk as the bar does
    for cs in CLIP_CELLS:
        for ch in CLIP_CELLS:
            fr = P.stringframe("end", f"silk: {cs}; hemp: {ch}")
            bar = P.stringframe("end", "silk: yoke; hemp: yoke")
            assert any(fr.walk(1, st) != bar.walk(1, st) for st in P.starts(bar))
            assert not np.allclose(P.check_walks(fr, GATES), P.check_walks(bar, GATES))

    # P6.3: a dozen tallies, the shares shaded, the border slips empty
    assert p63.macro == "foldsheet" and sh.shape == (2, 2) and sh.arrows == []
    assert np.allclose(sh.shares, [1 / 2, 0, 0, 1 / 2])
    assert np.allclose(sh.shares, matched)                  # CNOT on (1/2,1/2) (x) (1,0)
    assert np.allclose(sh.shares, CNOT.real @ np.kron([1 / 2, 1 / 2], [1, 0]))
    assert np.linalg.matrix_rank(np.reshape(sh.shares, (2, 2))) == 2   # not a patchwork
    assert sh.tallies == (7, 0, 0, 5) and sum(sh.tallies) == 12
    assert all((t == 0) == np.isclose(s, 0) for t, s in zip(sh.tallies, sh.shares))
    assert sh.borders and sh.border is None                 # slips laid, nothing on them

    # P6.4: every one home -- bar, crossing, bar, crossing
    assert p64.macro == "stringframe" and f4.place == ("end", None) and f4.ncol == 4
    kinds = [sorted({rule for rule, rows in f4.layers(j)}) for j in range(1, 5)]
    assert kinds == [["yoke"], ["cross"], ["yoke"], ["cross"]]
    assert f4.yokes() == [(1, "silk", "hemp", 1, 2), (3, "silk", "hemp", 1, 2)]
    assert f4.crossings() == [(2, "gut", "horsehair"), (4, "horsehair", "gut")]
    assert f4.clips() == []
    M4 = P.check_walks(f4, GATES)
    assert np.allclose(M4, alternating)                     # the master picture
    assert np.allclose(M4, np.eye(16))
    assert f4.coins() == {"silk": ("tails", 1), "hemp": ("heads", 2),
                          "gut": ("heads", 3), "horsehair": ("tails", 4)}
    for start in P.starts(f4):                              # all sixteen starts
        assert f4.coins(start) == {n: (start[n], P.HOOKROW[n]) for n in f4.hooked}
    # on the way: the hemp coin turned at the first bar, back at the second
    assert f4.walk(1)["hemp"] == "tails" and f4.walk(3)["hemp"] == "heads"
    assert f4.row_after("gut", 2) == 4 and f4.row_after("gut", 4) == 3

    # the macro's own faults: a yoke needs its partner, and a coin on both or neither
    assert P.faults(P.stringframe, "end", "silk tails: yoke; hemp: yoke") != []
    assert P.faults(P.stringframe, "end", "silk: yoke; hemp") != []


# --------------------------------------------------------------------------
# The sketchbook.  Rows are wires, row 0 is the top one; `g1 @ g2` stacks g1
# above g2, `g1 + g2` puts g1 first along the wire.  "CNOT is no pair of boxes"
# is a property of one matrix, with no picture equation at this stage, and so
# it stays on the blackboard.
# --------------------------------------------------------------------------
def cnot():
    p = Picture()
    i0, i1 = p.inp(0), p.inp(1)
    # PyZX draws this with the spiders you meet in Chapter 8.
    a, b = p.z(0, 1), p.x(1, 1)
    p.wire(i0, a, p.out(0))
    p.wire(i1, b, p.out(1))
    p.wire(a, b)
    g = p.done()
    g.scalar.add_power(1)
    return g


def crossing():
    p = Picture()
    i0, i1 = p.inp(0), p.inp(1)
    o0, o1 = p.out(0), p.out(1)
    p.wire(i0, o1)
    p.wire(i1, o0)
    return p.done()


def bare_wires(n):
    p = Picture()
    for row in range(n):
        p.wire(p.inp(row), p.out(row))
    return p.done()


def basis(b):
    p = Picture()
    n = p.x(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def not_box():
    p = Picture()
    n = p.x(0, 1, 1)
    p.wire(p.inp(0), n, p.out(0))
    return p.done()


def box(phases):
    p = Picture()
    a, b, c = phases
    vs = [p.z(0, 1, a), p.x(0, 2, b), p.z(0, 3, c)]
    p.wire(p.inp(0), *vs, p.out(0))
    return p.done()


def sketchbook():
    cn, S = cnot(), crossing()
    W1, W2 = bare_wires(1), bare_wires(2)
    assert np.allclose(matrix_of(cn),
                       [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    # --- the rule, as four picture equations --------------------------------
    for a in (0, 1):
        for b in (0, 1):
            pair = basis(a) @ basis(b)
            assert same_picture(pair + cn, basis(a) @ basis(a ^ b))
    assert not same_picture((basis(1) @ basis(0)) + cn, basis(1) @ basis(0))

    # --- a two-wire box is not a stack of two one-wire boxes ----------------
    A = box((Fraction(1, 4), Fraction(1, 3), Fraction(3, 4)))
    Bx = box((Fraction(1, 2), Fraction(1, 4), Fraction(1, 3)))
    assert not same_picture(cn, A @ Bx)
    assert not same_picture(cn, W1 @ A)
    assert not same_picture(cn, W2)
    # Achilles's clip on the lower string, drawn: right on one start, wrong on another
    N = not_box()
    assert not same_picture(cn, W1 @ N)
    assert same_picture((basis(1) @ basis(0)) + (W1 @ N),
                        (basis(1) @ basis(0)) + cn)              # "Right!"
    assert not same_picture((basis(0) @ basis(0)) + (W1 @ N),
                            (basis(0) @ basis(0)) + cn)          # "Wrong."

    # --- CNOT twice is nothing ----------------------------------------------
    assert same_picture(cn + cn, W2)
    assert not same_picture(cn, W2)
    assert same_picture(S + S, W2)

    # --- Twice Yoked, Twice Crossed ------------------------------------------
    alternating = (cn @ W2) + (W2 @ S) + (cn @ W2) + (W2 @ S)
    paired = (cn @ S) + (cn @ S)
    assert matrix_of(alternating).shape == (16, 16)
    assert same_picture(alternating, paired)                 # sliding
    assert same_picture(alternating, bare_wires(4))          # bare wire on every row
    assert np.allclose(matrix_of(alternating), np.eye(16))
    # and with the columns in the other order
    assert same_picture((W2 @ S) + (cn @ W2) + (W2 @ S) + (cn @ W2), bare_wires(4))
    # one yoke and two crossings is not nothing
    assert not same_picture((cn @ W2) + (W2 @ S) + (W2 @ S), bare_wires(4))

    # --- with any two boxes above: the crossed pair contributes nothing ------
    P = cn + (A @ Bx)
    Q = (Bx @ A) + cn
    whole = (P @ W2) + (W2 @ S) + (Q @ W2) + (W2 @ S)
    assert same_picture(whole, (P + Q) @ W2)
    assert not same_picture(whole, (Q + P) @ W2)
    assert not same_picture(whole, (P + Q) @ S)


if __name__ == "__main__":
    verdict("Yoke", blackboard, sketchbook)
