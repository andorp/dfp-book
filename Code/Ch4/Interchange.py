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
# matrices; then against the matrix the chapter writes for it.  A step's
# frame is checked against the state before it (the call's own state) and the
# state after it, and its marks against that step.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("stringframe", "foldsheet", "framewalk", "framesteps",
                                      "foldsteps"))
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
GATES = {"over": X, "heads": S0, "yoke": CNOT, "cross": SWAP}
I4 = np.eye(4, dtype=complex)


def clips_on(fr, name):
    return [(col, rules) for col, n, row, rules in fr.clips() if n == name]


def drawn(fr, name):
    """The clips drawn on a string: (column, rule, x, y), left to right."""
    return [(j, rule, x, y) for j, n, rule, x, y in fr.clips_drawn if n == name]


def pictures():
    P.check_lua()                     # ch04.lua and ch04.tex as the transcription was proved
    # Four moments: the slide past his (4.1), nothing moved (4.2), the knock
    # (4.3), the lift beside its result (4.4).  Every frame on four columns.
    found = P.check_section(DIALOGUE, "Fourth", sizes=[1, 1, 1, 2], frames=[1, 1, 1, 2])
    (p41, [(f1, _)]), = found[0]
    (p42, [(f3, _)]), = found[1]
    (p43, [(fk, _)]), = found[2]
    (p44a, [(fl, _)]), (p44, [(f4, _)]) = found[3]
    calls = (p41, p42, p43, p44a, p44)
    for c in calls:
        assert c.macro == "stringframe" and not c.keyed and not P.is_prediction(c)
    for c in calls[:4]:                                      # the layout's columns fixed ...
        assert P.options(c)["cols"] == "4"
    assert "cols" not in P.options(p44)                      # ... the last has four cells
    for fr in (f1, f3, fk, fl, f4):
        assert fr.ncol == 4 and fr.hooked == ("silk", "hemp", "gut", "horsehair")
        assert fr.yokes() == [] and fr.crossings() == [] and fr.pins == []
    assert [fr.place[0] for fr in (f1, f3, fk, fl, f4)] == ["end", "end", "off", "off", "end"]
    assert [fr.step for fr in (f1, f3, fk, fl, f4)] == ["slide", None, "slide", "lift", None]

    # Block 4.1: her clip slid past his.  Before the step (the call's state, the
    # old P4.1): her clip to the left of his; heads above, tails below
    before = P.clean(P.stringframe, "end;cols=4", p41.args[0])
    assert clips_on(before, "silk") == [(2, ("over",))]
    assert clips_on(before, "hemp") == [(1, ("heads",))]
    M1 = P.check_walks(before, GATES)
    assert np.allclose(M1, np.kron(np.kron(X, I2) @ np.kron(I2, S0), I4))   # hemp's first
    # after it: slid past his; the same arrival, from every start (Sliding)
    assert clips_on(f1, "silk") == [(2, ("over",))] and clips_on(f1, "hemp") == [(3, ("heads",))]
    M2 = P.check_walks(f1, GATES)
    assert np.allclose(M2, np.kron(np.kron(I2, S0) @ np.kron(X, I2), I4))   # silk's first
    assert np.allclose(M1, M2) and np.allclose(M1, np.kron(np.kron(X, S0), I4))
    assert f1.coins() == before.coins() == {"silk": ("tails", 1), "hemp": ("heads", 2)}
    assert np.allclose(np.kron(X, S0) @ ket(0, 1), ket(1, 0))
    assert np.allclose(P.drawn_column(f1, rows=[1, 2]), np.kron(X, S0) @ ket(0, 1))
    for start in P.starts(f1):
        assert before.walk(4, start) == f1.walk(4, start)
    # the marks: the ghost of her clip where it stood before the step, and an
    # arc to where it stands; low (a slide), rightwards, under his clip
    assert f1.moved["kind"] == "slide" and not f1.moved["knocked"]
    (_, _, gx0, gy0), = drawn(before, "hemp")
    assert f1.ghost_clip == ("hemp", "heads", 1, 2, gx0, gy0)
    (j, rule, nx, ny), = drawn(f1, "hemp")
    (sj, _, sx, sy), = drawn(f1, "silk")
    a = f1.move_arc
    assert (j, sj) == (3, 2) and gx0 < sx < nx                 # "beyond it": to the right of his
    assert a[0][0] < sx < a[3][0] and not P.leftward(a)
    # "passes clean under his": the arc, its stroke included, keeps a clear gap below
    # his clip, its drop shadow included -- lowered for it, starting beside her
    # clip's ghost and ending beside the clip where it lands
    lo, hi = P.bezier_y_extent(a)
    assert hi + P.SWING / 2 < P.clip_extent(sx, sy)[2]
    assert P.clip_extent(sx, sy)[2] - (hi + P.SWING / 2) > 0.04
    assert f1.lowered and hi <= f1.slide_ceiling                 # under the ceiling
    assert f1.follow_arc is None                                 # no crossing on the way
    assert np.allclose(a, ((0.6365, -0.38), (0.7245, -0.38 + 0.10 / 0.75),
                           (0.9005, -0.38 + 0.10 / 0.75), (0.9885, -0.38)))
    assert a[0][0] - P.SWING / 2 > gx0 + P.CLIPW / 2 + 0.014     # the ghost's outline
    assert a[3][0] + P.SWING / 2 < P.clip_extent(nx, ny)[0]      # the clip landed
    assert a[0][1] > gy0 and a[3][1] > ny                        # just above the string
    D5 = ("being on strings one above the other — and comes to rest beyond it.]*",
          "being on strings one above the other.]*")
    P.check_edits(DIALOGUE, "Fourth", [D5])
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        at, blocks = P.where(DIALOGUE, "Fourth", D5[1], path)
        assert at < blocks[0]
        assert "passes clean under his" in DIALOGUE.paragraphs("Fourth", path)[at]
        assert DIALOGUE.anchors("Fourth", path)[0].endswith("The silk coin first, this time. "
                                                             "The same!")
        assert "{hers to the left}" not in DIALOGUE.read(path)          # P4.1 removed

    # Block 4.2: nothing moved -- first clips level, second clips level
    assert clips_on(f3, "silk") == [(2, ("over",)), (3, ("heads",))]
    assert clips_on(f3, "hemp") == [(2, ("heads",)), (3, ("over",))]
    assert f3.moved is None and f3.ghost_clip is None and f3.move_arc is None
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
        at = P.stringframe("at-" + word, p42.args[0]).coins()
        past = P.stringframe("past-" + word, p42.args[0]).coins()
        assert at == {n: (f, f3.row_after(n, k - 1))
                      for n, f in f3.walk(k - 1).items() if f}
        assert past == {n: (f, f3.row_after(n, k)) for n, f in f3.walk(k).items() if f}
    assert P.stringframe("off", p42.args[0]).coins() == {}
    assert P.faults(P.stringframe, "at-fourth", p42.args[0]) != []     # no fourth column
    # the macro refuses a walk strip of these two-clip strings: no coin fits
    # between two columns of clips
    assert any("too crowded to walk" in e for e in
               P.faults(P.run_call, "framewalk", False, "to=end", (p42.args[0],),
                        "laid on; past the first; past the second; at the hooks"))

    # Block 4.3: the knock.  D6 has taken the coins off; she pushes the silk's
    # "turn it over" to the right, towards its neighbour: it stops against it
    # ("It won't go.") and the two stand struck together, the neighbour where
    # it stood, at its column's centre
    assert P.options(p43)["slide"] == "silk over second to fourth"
    assert fk.drawn_coins == [] and fk.coins() == {}
    assert fk.moved["knocked"] and (fk.moved["from"], fk.moved["to"]) == (2, 3)
    assert fk.moved["to"] < 4                                  # it won't go where it was sent
    assert fk.cellwords("silk") == ["-", "-", "over>heads", "-"]
    assert fk.cellwords("hemp") == f3.cellwords("hemp")
    (_, r1, x1, _), (_, r2, x2, y2) = drawn(fk, "silk")
    assert (r1, r2) == ("over", "heads")                      # its order along the string kept
    assert np.isclose(x2, P.colx(fk.state, 3))                # struck: the neighbour unmoved
    (_, _, h3x, _), = [d for d in drawn(f3, "silk") if d[1] == "heads"]
    assert np.isclose(x2, h3x)
    gap = P.clip_extent(x2, y2, shadow=False)[0] - P.clip_extent(x1, y2, shadow=False)[1]
    assert 0 < gap < 0.01                                     # "They knock together"
    (o3x,) = [x for _, r, x, _ in drawn(f3, "silk") if r == "over"]
    assert fk.ghost_clip[:4] == ("silk", "over", 2, 1) and np.isclose(fk.ghost_clip[4], o3x)
    assert fk.move_arc[0][0] < fk.move_arc[3][0]               # to the right
    assert fk.slide_ceiling is None and not fk.lowered        # nothing above the silk
    assert fk.follow_arc is None                              # and no crossing: the old arc
    gx, gy = fk.ghost_clip[4:]
    assert fk.move_arc == P.arc(gx + 0.02, gy + P.CLIPTOP + 0.03, x1 - 0.02,
                                y2 + P.CLIPTOP + 0.03, gy + P.CLIPTOP + 0.1, gy + P.CLIPTOP + 0.1)
    # along one string the order cannot change by sliding: the struck pair walks
    # as the pair it was
    Mk = P.check_walks(fk, GATES)
    assert np.allclose(Mk, M3)
    # a slide that would knock against two clips, or past its own struck
    # neighbour, is refused
    assert P.faults(P.stringframe, "off;slide=silk over second to fourth", "silk: -, over, heads+over") == [
        "stringframe: slide: it would knock against two clips (one cell holds two at most)"]
    assert P.faults(P.stringframe, "off;slide=silk over third to fourth", "silk: -, -, over>heads") == [
        "stringframe: slide: the clip it knocks against is in its way (lift it)"]
    D6 = ('*[On the silk string she pushes the "turn it over" clip to the right, towards its '
          'neighbour.]*',
          '*[Achilles takes the two coins off the right-hand hooks. On the silk string she pushes '
          'the "turn it over" clip to the right, towards its neighbour.]*')
    P.check_edits(DIALOGUE, "Fourth", [D6])
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        at, blocks = P.where(DIALOGUE, "Fourth", D6[1], path)
        assert blocks[1] < at < blocks[2]                     # after nothing moved, before the knock
        assert DIALOGUE.anchors("Fourth", path)[2].endswith(
            "It won't go. They knock together. They are on the same string!")

    # Block 4.4: lifted clean off and pinched on again beyond its neighbour, from
    # where the knock left it; then the silk walked alone from tails
    assert p44a.args[0].startswith("silk: -, -, over>heads;")   # the knock frame's struck cell
    assert P.options(p44a)["lift"] == "silk over third to fourth"
    assert fl.drawn_coins == [] and fl.moved["kind"] == "lift" and not fl.moved["knocked"]
    assert fl.cellwords("silk") == ["-", "-", "heads", "over"]
    assert fl.ghost_clip[:4] == ("silk", "over", 3, 1) and np.isclose(fl.ghost_clip[4], x1)
    (_, hr, hx, hy), (_, orule, ox, oy) = drawn(fl, "silk")
    assert (hr, orule) == ("heads", "over") and hx < ox        # beyond its neighbour
    assert np.isclose(hx, x2)                                  # the neighbour still unmoved
    lo, hi = P.bezier_y_extent(fl.move_arc)
    assert hi > P.clip_extent(hx, hy)[3]                       # up and over it
    assert hi > P.bezier_y_extent(fk.move_arc)[1]              # a lift goes higher than a slide
    assert clips_on(f4, "silk") == [(3, ("heads",)), (4, ("over",))]
    assert f4.cells == fl.cells                                # the frame beside it: its result
    assert f4.moved is None and f4.ghost_clip is None
    for fr in (f3, fk, fl, f4):                                # the hemp clips unmoved
        assert fr.cellwords("hemp") == ["-", "heads", "over", "-"]
    assert [r for c, r in clips_on(f3, "silk")] == [("over",), ("heads",)]
    assert [r for c, r in clips_on(f4, "silk")] == [("heads",), ("over",)]   # changed places
    assert f4.face["hemp"] is None                           # the hemp coin laid aside (D7)
    assert f4.coins() == {"silk": ("tails", 1)}               # silk arrives tails ...
    assert f3.coins()["silk"] == ("heads", 1)                 # ... a moment ago heads
    M4 = P.check_walks(f4, GATES)
    assert np.allclose(M4, np.kron(np.kron(X @ S0, X @ S0), I4))
    assert np.allclose(P.drawn_column(f4, rows=[1]), X @ S0 @ ket(1))
    assert not np.allclose(X @ S0, S0 @ X)                   # along one string, an order
    assert np.allclose(X @ S0 @ ket(1), ket(1)) and np.allclose(S0 @ X @ ket(1), ket(0))
    assert any("too crowded to walk" in e for e in
               P.faults(P.run_call, "framewalk", False, "to=end", (p44.args[0],),
                        "laid on; past the third; past the fourth; at the hooks"))
    D7 = ("*[He lifts the hemp coin off the frame and lays it aside, and walks the silk coin alone "
          "from tails:",
          "*[He lays the hemp coin aside, and walks the silk coin alone from tails:")
    P.check_edits(DIALOGUE, "Fourth", [D7])
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        at, blocks = P.where(DIALOGUE, "Fourth", D7[1], path)
        assert blocks[2] < at < blocks[3]
        assert DIALOGUE.anchors("Fourth", path)[3] == DIALOGUE.paragraphs("Fourth", path)[at]

    assert P.section_dumps(found) == [
        "stringframe: end slide",
        "stringframe: + coin silk tails row 1",
        "stringframe: + coin hemp heads row 2",
        "stringframe: + cells silk -,over,-,-",
        "stringframe: + cells hemp -,-,heads,-",
        "stringframe: + cells gut -,-,-,-",
        "stringframe: + cells horsehair -,-,-,-",
        "stringframe: + slid hemp 1 to 3",
        "stringframe: + ghost clip hemp heads col 1 row 2",
        "stringframe: end",
        "stringframe: + coin silk heads row 1",
        "stringframe: + coin hemp tails row 2",
        "stringframe: + cells silk -,over,heads,-",
        "stringframe: + cells hemp -,heads,over,-",
        "stringframe: + cells gut -,-,-,-",
        "stringframe: + cells horsehair -,-,-,-",
        "stringframe: off slide",
        "stringframe: + cells silk -,-,over>heads,-",
        "stringframe: + cells hemp -,heads,over,-",
        "stringframe: + cells gut -,-,-,-",
        "stringframe: + cells horsehair -,-,-,-",
        "stringframe: + slid silk 2 to 3 knocked",
        "stringframe: + ghost clip silk over col 2 row 1",
        "stringframe: off lift",
        "stringframe: + cells silk -,-,heads,over",
        "stringframe: + cells hemp -,heads,over,-",
        "stringframe: + cells gut -,-,-,-",
        "stringframe: + cells horsehair -,-,-,-",
        "stringframe: + lifted silk 3 to 4",
        "stringframe: + ghost clip silk over col 3 row 1",
        "stringframe: end",
        "stringframe: + coin silk tails row 1",
        "stringframe: + cells silk -,-,heads,over",
        "stringframe: + cells hemp -,heads,over,-",
        "stringframe: + cells gut -,-,-,-",
        "stringframe: + cells horsehair -,-,-,-"]


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
