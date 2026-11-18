"""Challenge 3: One Clip, Nothing on the Other
===============================================
"Moved all over, both times, and differently. There isn't a panel anywhere that
belongs to one coin."  -- Achilles

Claim: a box on the top wire of two is A (x) I and on the bottom wire I (x) A;
these are different matrices unless A is a multiple of I, and neither of them
leaves any entry of the column alone.  A (x) I mixes the two halves of the
column as whole halves; I (x) A acts inside each half.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import I2, Picture, ket, matrix_of, random_matrix, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch04 as P

X = np.array([[0, 1], [1, 0]], dtype=complex)
NAMED_A = np.array([[1, 1j], [0, 1]], dtype=complex)        # Sec 4.4's test matrix


def blackboard():
    rng = np.random.default_rng(43)

    # --- the two matrices of Sec 4.4, from the rule on the four pairs -------
    top = np.column_stack([np.kron(X @ ket(a), ket(b))
                           for a in (0, 1) for b in (0, 1)])
    bottom = np.column_stack([np.kron(ket(a), X @ ket(b))
                              for a in (0, 1) for b in (0, 1)])
    assert np.allclose(top, [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    assert np.allclose(bottom, [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert np.allclose(top, np.kron(X, I2))              # NOT on the TOP wire
    assert np.allclose(bottom, np.kron(I2, X))           # NOT on the bottom wire
    assert not np.allclose(top, bottom)
    # X (x) I exchanges 00<->10 and 01<->11; I (x) X exchanges 00<->01 and 10<->11
    assert np.allclose(top @ ket(0, 0), ket(1, 0)) and np.allclose(top @ ket(0, 1), ket(1, 1))
    assert np.allclose(bottom @ ket(0, 0), ket(0, 1)) and np.allclose(bottom @ ket(1, 0), ket(1, 1))

    # --- the block forms, on a matrix that is neither real nor symmetric ----
    for A in (NAMED_A, random_matrix(rng)):
        assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())
        assert np.allclose(np.kron(A, I2),
                           np.block([[A[0, 0] * I2, A[0, 1] * I2],
                                     [A[1, 0] * I2, A[1, 1] * I2]]))
        assert np.allclose(np.kron(I2, A),
                           np.block([[A, np.zeros((2, 2))], [np.zeros((2, 2)), A]]))
        assert not np.allclose(np.kron(A, I2), np.kron(I2, A))
        # and the mix-ups a stray reflection would make
        assert not np.allclose(np.kron(A, I2), np.kron(A.T, I2))
        assert not np.allclose(np.kron(A, I2), np.kron(A.conj().T, I2))
    assert np.allclose(np.kron(NAMED_A, I2),
                       [[1, 0, 1j, 0], [0, 1, 0, 1j], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.allclose(np.kron(I2, NAMED_A),
                       [[1, 1j, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1j], [0, 0, 0, 1]])

    # --- the two agree exactly when A is a multiple of I --------------------
    lam = 0.4 - 2.1j
    assert np.allclose(np.kron(lam * I2, I2), np.kron(I2, lam * I2))
    for _ in range(20):
        A = random_matrix(rng)
        assert not np.allclose(np.kron(A, I2), np.kron(I2, A))

    # --- each factor acts on its own wire -----------------------------------
    for _ in range(20):
        A, B = random_matrix(rng), random_matrix(rng)
        psi, phi = random_state(rng), random_state(rng)
        assert np.allclose(np.kron(A, B) @ np.kron(psi, phi),
                           np.kron(A @ psi, B @ phi))
        assert not np.allclose(np.kron(A, B) @ np.kron(psi, phi),
                               np.kron(B @ psi, A @ phi))
        assert np.allclose(np.kron(A, I2) @ np.kron(psi, phi), np.kron(A @ psi, phi))
    for j in (0, 1):
        for l in (0, 1):
            assert np.allclose(np.kron(A, B) @ ket(j, l), np.kron(A @ ket(j), B @ ket(l)))

    # --- the trap: the numbers look alike, the panels do not -----------------
    panels = np.kron([1 / 2, 1 / 2], [3 / 4, 1 / 4])
    turned = np.kron(X, I2) @ panels
    assert np.allclose(turned, panels)                   # "the whole sheet is untouched!"
    for a in (0, 1):                                     # and yet no panel stays
        for b in (0, 1):
            assert not np.allclose(np.kron(X, I2) @ ket(a, b), ket(a, b))
            assert not np.allclose(np.kron(I2, X) @ ket(a, b), ket(a, b))
    # the two clips move the panels differently: halves swapped vs inside halves
    assert np.allclose(np.kron(X, I2) @ ket(0, 0), ket(1, 0))    # crosses the first fold
    assert np.allclose(np.kron(I2, X) @ ket(0, 0), ket(0, 1))    # stays in its half
    # a sheet that is not made of twin halves shows the difference in the numbers
    lopsided = np.kron([1 / 3, 2 / 3], [3 / 4, 1 / 4])
    assert not np.allclose(np.kron(X, I2) @ lopsided, lopsided)

    pictures(panels)


# --------------------------------------------------------------------------
# The pictures (pdf/props/ch04.tex; the Lua's rules transcribed in
# tools/props_ch04.py), read out of this challenge's section of dialog.md.
# A pencil arrow runs from the panel a pair of coins starts in to the panel it
# arrives in; the macro walks the pairs itself.  Here each rule's arrows are
# checked against the matrix of that rule on the four panels, and the walk
# strip frame by frame against the matrix of the clip.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("stringframe", "foldsheet", "framewalk", "framesteps",
                                      "foldsteps"))
S0 = np.array([[1, 1], [0, 0]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
GATES = {"over": X, "heads": S0, "yoke": CNOT, "cross": SWAP}
RULES = {"silk-over": np.kron(X, I2), "hemp-over": np.kron(I2, X),
         "silk-heads": np.kron(S0, I2), "hemp-heads": np.kron(I2, S0), "yoke": CNOT}


def arrows_are(arrow, M):
    """An arrow for exactly the panels M moves, each to where M sends it."""
    moved = dict(arrow)
    for a in range(4):
        e = np.eye(4)[a]
        assert np.allclose(M @ e, np.eye(4)[moved.get(a, a)]), (a, moved)
    return moved


def faces(v):
    """The faces a basis column shows, top row first (heads = 0)."""
    (k,) = np.flatnonzero(np.abs(v) > 0.5)
    n = int(np.log2(len(v)))
    return [P.FACE[int(b)] for b in format(int(k), "0%db" % n)]


def pictures(panels):
    P.check_lua()                     # ch04.lua and ch04.tex as the transcription was proved
    # Three moments.  Block 3.1, a step strip: the coins laid on, just past the
    # clip, at the right-hand hooks.  Block 3.2: the sheet as Achilles expects it
    # beside the sheet with its blue arrows.  P3.3: the red arrows added.
    found = P.check_section(DIALOGUE, "Third", sizes=[1, 2, 1], frames=[3, 2, 1])
    (p31, walk), = found[0]
    (p32e, [(still, _)]), (p32, [(sh, _)]) = found[1]
    (p33, [(sh3, _)]), = found[2]

    # the macro's five arrow rules against their matrices, and their pencils
    for rule, M in RULES.items():
        drawn = P.foldsheet("twice", "arrows: " + rule).arrows
        assert len(drawn) == 1
        (r, colour, arrow), = drawn
        assert r == rule and arrows_are(arrow, M) is not None
        assert colour == {"silk": "blue", "hemp": "red"}.get(rule.split("-")[0], "graphite")

    # Block 3.1: "turn it over" on the silk, heads above and tails below walked
    # (the second walk: D4 sets the coins back at the left upright first)
    assert p31.macro == "framewalk" and not p31.keyed and P.options(p31) == {"to": "end"}
    (f0, _), (f1, _), (f2, _) = walk
    assert [f.place for f in (f0, f1, f2)] == [("start", None), ("past", 1), ("end", None)]
    for f in (f0, f1, f2):                                    # one frame, three moments
        assert f.hooked == ("silk", "hemp", "gut", "horsehair") and f.ncol == 1
        assert f.clips() == [(1, "silk", 1, ("over",))]       # on the silk, nothing on the hemp
        assert f.face == {"silk": "heads", "hemp": "tails", "gut": None, "horsehair": None}
        M = P.check_walks(f, GATES)
        assert np.allclose(M, np.kron(np.kron(X, I2), np.eye(4)))   # X (x) I on silk and hemp
        assert not np.allclose(M, np.kron(np.kron(I2, X), np.eye(4)))
    start = ket(0, 1)                                         # heads above, tails below
    walked = [start, np.kron(X, I2) @ start, np.kron(X, I2) @ start]
    for f, v in zip((f0, f1, f2), walked):                    # each frame, the model's coins
        assert np.allclose(P.drawn_column(f, rows=[1, 2]), v)
    assert np.allclose(walked[2], ket(1, 1))                  # "arrives tails above, tails below"
    assert not np.allclose(walked[2], np.kron(I2, X) @ start)
    # the coins laid on at the left upright, and at the right-hand hooks
    assert all(np.isclose(x, 0.2) for _, _, _, x, _ in f0.drawn_coins)
    assert all(np.isclose(x, P.L - 0.2) for _, _, _, x, _ in f2.drawn_coins)
    # the step through the clip: a ghost of the coin it turned, as it was just
    # before (the model's column before the step), and the hop; the hemp coin,
    # which passes untouched, has neither
    assert f0.ghost_coins == [] and f0.hops == [] and f2.ghost_coins == [] and f2.hops == []
    changed = [r for r, a, b in zip((1, 2), faces(walked[0]), faces(walked[1])) if a != b]
    assert changed == [1]
    assert [(n, r) for n, _, r, _, _ in f1.ghost_coins] == [("silk", 1)]
    assert np.allclose(P.drawn_column(f1, rows=changed, ghosts=True), ket(0))   # as laid
    assert [n for n, _, _ in f1.hops] == ["silk"]
    # the ghost stands just before the clip and the coin just past it, both clear of it
    (_, _, _, cx, cy), = [c for c in f1.clips_drawn]
    ghost = [P.coin_extent(x, y, ghost=True) for _, _, _, x, y in f1.ghost_coins]
    after = [P.coin_extent(x, y) for n, _, _, x, y in f1.drawn_coins if n == "silk"]
    clip = P.clip_extent(cx, cy)
    assert ghost[0][1] < clip[0] < clip[1] < after[0][0]
    # F12 is the old P3.1, unchanged
    old = P.stringframe("end", p31.args[0])
    assert f2.dump == old.dump and f2.drawn_coins == old.drawn_coins
    D4 = ("*[He draws a blue arrow on the sheet.]*",
          "*[He draws a blue arrow on the sheet, sets the coins back at the left upright, and "
          "walks again.]*")
    P.check_edits(DIALOGUE, "Third", [D4])
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a31, a32, a33 = DIALOGUE.anchors("Third", path)
        assert a31.endswith("Heads above, tails below: arrives tails above, tails below. "
                            "Upper right goes to lower right.")
        assert a31.index(D4[1]) < a31.index("Heads above, tails below: arrives")
        assert a32.endswith("The shares looked the same only because the two halves were twins.")
        assert "Every panel has moved!" in a32

    # Block 3.2: the patchwork unchanged, and four blue arrows, all up or down
    assert p32.macro == "foldsheet" and p32.opt == "" and not p32.keyed
    assert p32.caption == "blue: across the first fold"       # C1
    assert np.allclose(sh.shares, panels) and sh.border is not None
    assert len(sh.arrows) == 1                                # one rule, one pencil
    (rule, colour, blue), = sh.arrows
    assert (rule, colour) == ("silk-over", "blue")
    moved = arrows_are(blue, np.kron(X, I2))
    assert len(blue) == 4                                     # "Every panel has moved!"
    assert all(a % 2 == b % 2 and a // 2 != b // 2 for a, b in blue)   # across the first fold
    assert np.allclose([panels[moved[a]] for a in range(4)], panels)  # twin halves: same numbers
    # the prediction beside it ("The whole sheet is untouched!"): the rule's
    # sheet, the same patchwork with no panel moved -- the arrows of the
    # identity, which are none -- and not the sheet beside it
    assert p32e.macro == "foldsheet" and p32e.args == p32.args and not p32e.keyed
    assert P.options(p32e) == {"expects": "untouched"} and still.expects == "untouched"
    assert still.untouched == ("silk-over",) and still.arrows == []
    arrows_are((), np.eye(4))                                 # the identity moves nothing
    assert np.allclose(still.shares, sh.shares) and still.border == sh.border
    assert still.arrows != sh.arrows
    assert not np.allclose(np.kron(X, I2), np.eye(4))         # the clip is no identity ...
    assert np.allclose(np.kron(X, I2).real @ panels, panels)   # ... though the numbers agree
    # the rule says nothing only where no border list changes: never for a clip
    # on a string whose slip is not a half and a half
    assert P.faults(P.foldsheet, "twice", "patchwork: a half, three-quarters; arrows: hemp-over",
                    False, "expects=untouched") == [
        "foldsheet: expects=untouched: hemp-over changes a border list; the rule says nothing"]

    # P3.3: the blue arrows kept, and four red ones, all sideways
    assert np.allclose(sh3.shares, panels) and len(sh3.arrows) == 2
    (r1, c1, blue3), (r2, c2, red) = sh3.arrows
    assert (r1, c1, blue3) == ("silk-over", "blue", blue)
    assert (r2, c2) == ("hemp-over", "red")
    arrows_are(red, np.kron(I2, X))
    assert len(red) == 4                                      # every panel moves again
    assert all(a // 2 == b // 2 and a % 2 != b % 2 for a, b in red)    # inside each half
    assert not set(red) & set(blue)                           # both times, and differently
    assert not np.allclose(np.kron(X, I2), np.kron(I2, X))

    assert P.section_dumps(found) == [
        "stringframe: start",
        "stringframe: + coin silk heads row 1",
        "stringframe: + coin hemp tails row 2",
        "stringframe: + cells silk over",
        "stringframe: + cells hemp -",
        "stringframe: + cells gut -",
        "stringframe: + cells horsehair -",
        "stringframe: past-first",
        "stringframe: + coin silk tails row 1",
        "stringframe: + coin hemp tails row 2",
        "stringframe: + cells silk over",
        "stringframe: + cells hemp -",
        "stringframe: + cells gut -",
        "stringframe: + cells horsehair -",
        "stringframe: + ghost coin silk heads row 1",
        "stringframe: end",
        "stringframe: + coin silk tails row 1",
        "stringframe: + coin hemp tails row 2",
        "stringframe: + cells silk over",
        "stringframe: + cells hemp -",
        "stringframe: + cells gut -",
        "stringframe: + cells horsehair -",
        "foldsheet: twice rows 2 cols 2 expects untouched",
        "foldsheet: + shares 0.3750,0.1250,0.3750,0.1250",
        "foldsheet: + slips 0.5000,0.7500",
        "foldsheet: + untouched silk-over",
        "foldsheet: twice rows 2 cols 2",
        "foldsheet: + shares 0.3750,0.1250,0.3750,0.1250",
        "foldsheet: + slips 0.5000,0.7500",
        "foldsheet: + arrows silk-over 4",
        "foldsheet: twice rows 2 cols 2",
        "foldsheet: + shares 0.3750,0.1250,0.3750,0.1250",
        "foldsheet: + slips 0.5000,0.7500",
        "foldsheet: + arrows silk-over 4",
        "foldsheet: + arrows hemp-over 4"]


# --------------------------------------------------------------------------
# The sketchbook.  Rows are wires, row 0 is the top one; `g1 @ g2` stacks g1
# above g2, `g1 + g2` puts g1 first along the wire.
# --------------------------------------------------------------------------
def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def box(phases):
    """A box with no name: a chain of three nodes, non-symmetric and non-real."""
    p = Picture()
    a, b, c = phases
    # PyZX draws this with the spiders you meet in Chapter 8.
    vs = [p.z(0, 1, a), p.x(0, 2, b), p.z(0, 3, c)]
    p.wire(p.inp(0), *vs, p.out(0))
    return p.done()


def basis(b):
    p = Picture()
    n = p.x(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def state(phases):
    """A state with no name: the state 0 followed by a chain of three nodes."""
    p = Picture()
    n = p.x(0, 1, 0)
    a, b, c = phases
    vs = [p.z(0, 2, a), p.x(0, 3, b), p.z(0, 4, c)]
    p.wire(n, *vs, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def not_box():
    p = Picture()
    n = p.x(0, 1, 1)
    p.wire(p.inp(0), n, p.out(0))
    return p.done()


def sketchbook():
    W = bare_wire()
    A = box((Fraction(1, 4), Fraction(1, 3), Fraction(3, 4)))
    Am = matrix_of(A)
    assert not np.allclose(Am, Am.T) and not np.allclose(Am, Am.conj())
    assert np.allclose(matrix_of(W), I2)

    # --- which wire is the left factor?  The picture decides, and we ask it --
    assert np.allclose(matrix_of(A @ W), np.kron(Am, I2))     # box on the TOP wire
    assert not np.allclose(matrix_of(A @ W), np.kron(I2, Am))
    assert np.allclose(matrix_of(W @ A), np.kron(I2, Am))     # box on the bottom wire
    assert not same_picture(A @ W, W @ A)

    # --- nothing on a wire is the box I on it -------------------------------
    B = box((Fraction(1, 2), Fraction(1, 4), Fraction(1, 3)))
    Bm = matrix_of(B)
    assert np.allclose(matrix_of(A @ B), np.kron(Am, Bm))
    assert not same_picture(A @ B, B @ A)

    # --- each box minds its own wire, as a picture equation ------------------
    s1 = state((Fraction(1, 3), Fraction(1, 2), Fraction(1, 4)))
    s2 = state((Fraction(3, 4), Fraction(1, 3), Fraction(1, 2)))
    assert same_picture((s1 @ s2) + (A @ B), (s1 + A) @ (s2 + B))
    assert not same_picture((s1 @ s2) + (A @ B), (s1 + B) @ (s2 + A))
    assert same_picture((s1 @ s2) + (A @ W), (s1 + A) @ s2)   # nothing on the lower wire

    # --- the two clips of the dialogue, drawn ------------------------------
    N = not_box()
    assert np.allclose(matrix_of(N), X)
    assert not same_picture(N @ W, W @ N)             # silk clip vs hemp clip
    assert np.allclose(matrix_of(N @ W), np.kron(X, I2))
    assert np.allclose(matrix_of(W @ N), np.kron(I2, X))
    for a in (0, 1):
        for b in (0, 1):
            pair = basis(a) @ basis(b)
            assert same_picture(pair + (N @ W), basis(1 - a) @ basis(b))
            assert same_picture(pair + (W @ N), basis(a) @ basis(1 - b))
            assert not same_picture(pair + (N @ W), basis(a) @ basis(1 - b))


if __name__ == "__main__":
    verdict("OneClip", blackboard, sketchbook)
