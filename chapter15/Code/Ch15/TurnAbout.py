"""Challenge II: The Turn-About  (Grover's diffusion; the headline; The Theorem, Twice)
======================================================================================
"...Oh. All the odds are on the second drawer."  -- Achilles

Claim: for every n >= 1, Grover's D (2/N off the diagonal, -1 + 2/N on it) equals
W R W = -W X^n C^{n-1}Z X^n W = 2|s><s| - I, so (Dv)_x = 2 vbar - v_x, D|s> = |s>, Du = -u
when the entries of u sum to 0, and D^2 = I (Result 15.5); I - 2|c><c| is the bare cords
minus 2^{1-n} times red c-effects beside red c-points (Result 15.6); for two cords
R = (Z (x) Z) CZ with no sign (Result 15.7 (i)), so the Tortoise's cloth build -- collars,
a half-ring on each cord, the controlled half-turn, collars -- is D / sqrt2 bare, leaves the
even setting alone and turns every tilted setting over; and for three cords the CCZ drawn
as phase gadgets is 2^{-5/2} CCZ, checked by matrix and not derived (Result 15.17).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from itertools import product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import ket, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch15 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
CCZ = np.diag([1] * 7 + [-1]).astype(complex)


def bits(x, n):
    return tuple((x >> (n - 1 - i)) & 1 for i in range(n))


def grover_D(N):
    """Grover's (4.0), entry by entry: 2/N off the diagonal, -1 + 2/N on it."""
    D = np.full((N, N), 2 / N, dtype=complex)
    np.fill_diagonal(D, -1 + 2 / N)
    return D


def grover_R(N):
    """Grover's R: R_00 = 1, R_xx = -1 for x != 0, zero off the diagonal."""
    return np.diag([1] + [-1] * (N - 1)).astype(complex)


def multi_cz(n):
    """C^{n-1}Z = I - 2|1..1><1..1|  (n = 1: Z itself)."""
    M = np.eye(2 ** n, dtype=complex)
    M[-1, -1] = -1
    return M


def even_state(n):
    return np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)


def blackboard():
    rng = np.random.default_rng(1502)

    # --- Result 15.5 for n = 1..6, exactly, against Grover's own entries
    for n in range(1, 7):
        N = 2 ** n
        W, Xn = kron(*([H] * n)), kron(*([X] * n))
        s = even_state(n)
        D = grover_D(N)
        proj = np.outer(s, s.conj())
        assert np.allclose(W @ grover_R(N) @ W, D)
        assert np.allclose(-W @ Xn @ multi_cz(n) @ Xn @ W, D)
        assert np.allclose(2 * proj - np.eye(N), D)
        assert np.allclose(Xn @ multi_cz(n) @ Xn, -grover_R(N))         # Step 1
        assert not np.allclose(np.eye(N) - 2 * proj, D)                 # the sign: no odds see it
        assert not np.allclose(W @ Xn @ multi_cz(n) @ Xn @ W, D)        # the -1 is needed
        assert np.allclose(D @ D, np.eye(N))                            # D^2 = I
        assert np.allclose(D @ s, s)                                    # D|s> = |s>
        # the Dirac's-Blackboard sum: (WRW)_xy = 2^{-n} (2 - sum_z (-1)^{(x xor y).z})
        for x, y in [(0, 0), (1, 2 % N), (N - 1, 0), (N - 1, N - 1)]:
            tot = sum((-1) ** (sum(a * b for a, b in zip(bits(x ^ y, n), bits(z, n))) % 2)
                      for z in range(N))
            assert np.isclose((W @ grover_R(N) @ W)[x, y], (2 - tot) / N)
        # inversion about average on a random COMPLEX vector
        v = rng.normal(size=N) + 1j * rng.normal(size=N)
        vbar = v.mean()
        assert not np.isclose(vbar, vbar.conjugate())
        assert np.allclose(D @ v, 2 * vbar - v)
        assert not np.allclose(D @ v, 2 * np.conj(vbar) - v)            # not the conjugate mean
        # "a peg standing at the average stays, and a peg far from it goes far"
        assert np.allclose(D @ v - vbar, -(v - vbar))
        # D u = -u when the entries of u sum to zero
        u = rng.normal(size=N) + 1j * rng.normal(size=N)
        u -= u.mean()
        assert np.isclose(u.sum(), 0) and np.allclose(D @ u, -u)
    # n = 1: C^0 Z is Z itself, and the turn-about on two drawers is X
    assert np.allclose(multi_cz(1), Z) and np.allclose(grover_D(2), X)

    # --- Result 15.6 for every c: I - 2|c><c| = I - 2^{1-n} (2^n |c><c|)
    for n in (1, 2, 3):
        N = 2 ** n
        for c in range(N):
            e = np.eye(N)[c]
            pts = kron(*[np.sqrt(2) * ket(b) for b in bits(c, n)]).ravel()  # n red points
            assert np.allclose(np.outer(pts, pts), 2 ** n * np.outer(e, e))
            assert np.allclose(np.eye(N) - 2 ** (1 - n) * np.outer(pts, pts), np.eye(N) - 2 * np.outer(e, e))
        # c = 1..1 is C^{n-1}Z; c = 0..0 is -R, so with its sign reversed it is R (§15.7 Step 2)
        assert np.allclose(-(np.eye(N) - 2 * np.outer(np.eye(N)[0], np.eye(N)[0])), grover_R(N))
        assert np.allclose(np.eye(N) - 2 * np.outer(np.eye(N)[-1], np.eye(N)[-1]), multi_cz(n))
        assert np.allclose(np.eye(N) - 2 * np.outer(np.eye(N)[0], np.eye(N)[0]), -grover_R(N))

    # --- Result 15.7 (i): two cords, no sign; X Z X = -Z
    ZZ = kron(Z, Z)
    assert np.allclose(ZZ @ CZ, grover_R(4)) and np.allclose(ZZ @ CZ, np.diag([1, -1, -1, -1]))
    assert np.allclose(X @ Z @ X, -Z)
    XX = kron(X, X)
    assert np.allclose(-XX @ CZ @ XX, ZZ @ CZ)
    assert not np.allclose(XX @ CZ @ XX, ZZ @ CZ)                       # the X-form carries -1
    for (c1, c2), (h1, h2) in product(product((0, 1), repeat=2), repeat=2):
        M = kron(np.linalg.matrix_power(Z, h1), np.linalg.matrix_power(Z, h2)) @ CZ
        assert np.isclose(M[2 * c1 + c2, 2 * c1 + c2], (-1) ** (c1 * c2 + h1 * c1 + h2 * c2))

    # --- the pegs of the Second Challenge
    D4 = grover_D(4).real
    pegs = np.array([1, -1, 1, 1])                                      # touched: the second under
    assert np.isclose(pegs.mean(), 0.5)                                 # "half a notch"
    assert np.allclose(D4 @ pegs, [0, 2, 0, 0])                         # three at the line, two up
    assert np.allclose((D4 @ pegs) ** 2, [0, 4, 0, 0])                  # "Nothing, four, nothing, nothing"
    assert np.isclose(np.sum((D4 @ pegs) ** 2), np.sum(pegs ** 2))      # the squares still four
    assert np.allclose(D4 @ np.ones(4), np.ones(4))                     # the even row stays
    assert np.isclose(np.array([1, 1, -1, -1]).mean(), 0)
    assert np.allclose(D4 @ [1, 1, -1, -1], [-1, -1, 1, 1])             # turned over
    # the tilted setting "upper cord at a half" is the row with drawers 10, 11 under
    tilted = np.kron(Z @ np.ones(2), np.ones(2))
    assert np.allclose(tilted, [1, 1, -1, -1])
    # and the other two tilted settings are turned over too; each averages to nothing
    for t in (np.kron(np.ones(2), Z @ np.ones(2)), np.kron(Z @ np.ones(2), Z @ np.ones(2))):
        assert np.isclose(t.mean(), 0) and np.allclose(D4 @ t, -t)

    # --- the CCZ bookkeeping of Result 15.17: x1x2x3 as a sum of parities, all eight cases
    for x1, x2, x3 in product((0, 1), repeat=3):
        rhs = Fraction((x1 + x2 + x3) - (x1 ^ x2) - (x2 ^ x3) - (x1 ^ x3) + (x1 ^ x2 ^ x3), 4)
        assert rhs == x1 * x2 * x3

    pictures()


# --- the prop pictures ------------------------------------------------------------------

def pictures():
    """The Second Challenge's prop pictures (identity card, section 4), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch15.py.  The
    ties hold the transcription against this file's blackboard (a structural check of the
    pictures, not a second proof): the turn-about on the pegs is Grover's inversion about
    the average, (Dv)_x = 2 vbar - v_x, and on the cloth it leaves the even setting alone
    and turns every tilted setting over, the halves it sets aside counting the sign."""
    calls = DIALOGUE.check_layout("Second", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("pegcard", False, "touch the second, mark the average, turn about", ("even row",), "turned about, a tall peg"),
        ("aboutcloth", True, "", ("even setting",), "the build, set even"),
        ("aboutcloth", False, "push through", ("tilted setting",), "the tilted setting, pushed through"),
        ("pegcard", False, "turn about", ("one, one, under, under; tick at half a notch",), "tilted, then turned about")]
    PC.check_captions(DIALOGUE, "Second", calls)
    PC.check_section(DIALOGUE, "Second", calls)
    pegs_h = round(PC.PFTOP - PC.PFBOT, 2)
    cloth_h = PC.height(calls[1])
    assert (pegs_h, cloth_h) == (1.48, 1.75)                            # read from the Lua and the kit
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Second")] == [[pegs_h], [cloth_h], [cloth_h], [pegs_h]]

    # the anchors (card section 4)
    a1, a2, a3, a4 = DIALOGUE.anchors("Second")
    f1, f2, f3, f4 = DIALOGUE.following("Second")
    assert "in the card's left margin, beside the first slit, and there it stays." in a1                 # D4
    assert "What has held: the average, still half a notch" in a1
    assert f1.startswith("**Tortoise:** Now the odds.")
    assert "on the left end of each cord.]*" in a2
    assert f2.startswith("*[The picture as it stands: the two green end-rings at rest at the left;")
    assert ("she lays a small green bob from her work-bag, its bead at a half, off the cloth just beyond its "
            "near edge at the right-hand end, by the candlestick.") in a3                                  # D9
    assert "each coming to rest just past where its far tube stood." in a3
    assert f3.startswith("**Tortoise:** The same setting, with a half set aside: turned over.")
    assert "below his pencilled tick, which still stands at half a notch." in a4                          # D10
    assert f4.startswith("**Achilles:** Under, under, one, one.")
    assert DIALOGUE.following("Second", DIALOGUE.raw) == [f1, f2, f3, f4]
    # the section carries D4 to D10's anchor phrases, once each, in both files
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        sec = DIALOGUE.section("Second", path)
        for phrase in ("in the card's left margin, beside the first slit, and there it stays.",          # D4
                       "on its right as Achilles sees it, between the card and the candlestick",          # D5
                       "from Achilles's left to his right, one above the other, the upper further from him",  # D6
                       "each from its cord's loose right-hand end",
                       "a bead at rest stands at the top of its ring as Achilles sees it",
                       "ties each by a tie of its own to a bare red hub set to their right, midway between the cords",
                       "a short pendant runs on to the right", "which runs on a little way past it.]*",
                       "as every end she pushes tonight swallows the cord it travels",                     # D7
                       "and each comes to rest just past where its far tube stood",
                       "with only the short loose end of its cord beyond it, bare; and nothing set aside.",
                       "She gathers the cloth clear, the end-rings and what is left of the cords into her work-bag, "
                       "and lays the whole build again from the work-bag",                                 # D8
                       "she lays a small green bob from her work-bag, its bead at a half, off the cloth just beyond "
                       "its near edge at the right-hand end, by the candlestick.",                         # D9
                       "each coming to rest just past where its far tube stood.",
                       "below his pencilled tick, which still stands at half a notch."):                   # D10
            assert sec.count(phrase) == 1, phrase

    p21, p22, p23, p24 = (PC.picture(c) for c in calls)
    D4 = grover_D(4).real
    s2 = 2 * even_state(2).real                                         # the even row: 2|s>

    # P2.1: touched, the average ticked, turned about: (0, 2, 0, 0) = D (1, -1, 1, 1), the
    # tick at the average 1/2, each peg turned about it -- Grover's inversion about the average
    v = np.array([1, -1, 1, 1])
    assert p21.dump == "pegs=0,2,0,0 ticks=0.5"
    assert np.allclose(p21.state.h, D4 @ v) and np.allclose(p21.state.h, 2 * v.mean() - v)
    assert p21.state.ticks == [v.mean()] == [0.5]
    assert np.argmax(p21.state.h) == 0b01                               # the second drawer stands tall
    assert not np.allclose(p21.state.h, -(D4 @ v))                      # 2 vbar - v, not v - 2 vbar
    assert np.isclose(np.square(p21.state.h).sum(), np.square(v).sum())  # the squares together, still four
    # the even row turned about is unmoved: D|s> = |s>
    E = PC.pegs_state("turn about", "even row")
    assert np.allclose(E.h, D4 @ s2) and np.allclose(E.h, s2)
    # P2.4: one, one, under, under -- 2|-+>, the tilted setting read in the drawers' basis --
    # turned about to under, under, one, one: D u = -u when the entries of u sum to nothing
    u = np.array([1, 1, -1, -1])
    MINUS, PLUS = np.array([1, -1]) / np.sqrt(2), np.array([1, 1]) / np.sqrt(2)
    assert np.allclose(u, 2 * np.kron(MINUS, PLUS))                     # top cord first
    assert not np.allclose(u, 2 * np.kron(PLUS, MINUS))
    assert p24.dump == "pegs=-1,-1,1,1 ticks=0.5"
    assert np.allclose(p24.state.h, D4 @ u) and np.allclose(p24.state.h, -u)
    assert u.mean() == 0 and p24.state.ticks == p21.state.ticks          # the tick still stands
    assert not PC.TEX_ERRORS

    # the pegs' fixed parts: the same card, slits, middle line and notches as in the First
    # Challenge's pictures; a move changes only the pegs and the ticks; the tick, once made,
    # stays, clear of the first slit and of every peg
    first = [PC.picture(c) for c in DIALOGUE.check_layout("First", sizes=[1, 1, 1]) if c.macro == "pegcard"]
    for p in first + [p21, p24]:
        w = p.shown
        assert (w.card, w.slits, w.notches, w.middle) == (p21.shown.card, p21.shown.slits, p21.shown.notches,
                                                          p21.shown.middle)
        assert [x for _, x, _ in w.pegs] == list(PC.SLITX)
        for (k, x, y), (_, x0, y0, x1, y1) in zip(w.pegs, w.slits):            # every peg inside its slit
            assert y0 + PC.PR + 0.3 * PC.PT / 2 < y < y1 - PC.PR - 0.3 * PC.PT / 2
    for p in (p21, p24):
        parts = PC.drawn_parts(p)[0]
        assert min(PC.O.gap(parts["tick 0"], parts[k]) for k in parts if k.startswith(("slit", "peg"))) > 0.3
        assert p.shown.ticks[0][3] == PC.MID + 0.5 * PC.NP

    # the cloth: the build pushed through in each of its four settings returns the setting
    # unchanged, with 0, 1, 1, 3 halves set aside; their parity is the sign by which D
    # multiplies |++>, |-+>, |+->, |-->  (a bead at a half is |->, at rest |+>)
    KET = {0: PLUS, 12: MINUS}
    for setting, halves in (("even setting", 0), ("upper at a half", 1), ("lower at a half", 1),
                            ("both at a half", 3)):
        M = PC.cloth_state("push through", setting)
        assert not PC.TEX_ERRORS
        st = PC.SETTINGS[setting]
        assert [(e["c"], e["t"], e["stage"]) for e in M["ends"]] == [("green", st[0], 4), ("green", st[1], 4)]
        assert M["aside"] == halves and not M["turn"] and M["landed"] is None
        ket_in = np.kron(KET[st[0]], KET[st[1]])
        assert np.allclose(D4 @ ket_in, (-1) ** halves * ket_in)
        assert not np.allclose(D4 @ ket_in, (-1) ** (halves + 1) * ket_in)
    assert PC.SETTINGS["tilted setting"] == PC.SETTINGS["upper at a half"]   # P2.3 against P2.4

    # P2.2: the build as laid, both end-rings green at rest on the left
    assert p22.dump == (
        "setting=even setting upper=green:0:laid lower=green:0:laid turn=whole landed=none aside=0 | "
        "cols=8 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:end green 0 @0,0; 2:ring green 12 @2.4,0; "
        "3:ring green 6 @3.8,0; 4:hub red 0 @4.8,0.5; 5:bob green 18 @5.8,0.5; 6:end green 0 @0,1; "
        "7:ring green 12 @2.4,1; 8:ring green 6 @3.8,1; tube @1,0 a0; tube @5.5,0 a0; tube @1,1 a0; "
        "tube @5.5,1 a0; tie 3-4 []; tie 4-8 []; pendant 4-5 []; cord loose-1 [7,0 0,0]; cord loose-6 [7,1 0,1]")
    assert p22.labels == ("end-ring", "tube", "hub", "bob", "tie")       # the kit's own keyed words
    assert set(p22.options["key"]) == {"end", "tube", "hub", "bob", "tie"}
    # its clearances, on drawn extents (rings with their strokes and shadows, cords with
    # theirs, tubes as the boxes they cover): no tie meets a ring but its own; the hub and
    # the bob clear every cord; along each cord every piece clears the next; the far
    # tubes clear the bob
    rings, cords, tubes = PC.cloth_parts(p22)
    R = {i: (x, y, reach) for i, kind, x, y, r, reach in rings}
    for kind, a, b, pts, reach in cords:
        if kind == "tie":
            for i, (x, y, rr) in R.items():
                if i not in (a, b):
                    assert PC.point_to_polyline((x, y), pts) - rr - reach > 0.05, (a, b, i)
    for i in (4, 5):                                                        # the hub and the bob
        for kind, a, b, pts, reach in cords:
            if kind == "cord":
                assert PC.point_to_polyline(R[i][:2], pts) - R[i][2] - reach > 0.1
    for row, chain in ((0, (1, "tube 1", 2, 3, "tube 5.5")), (1, (6, "tube 1", 7, 8, "tube 5.5"))):
        T = {"tube %g" % x: box for x, y, box in tubes if y == row}

        def gap(p, q):
            if isinstance(p, str) and isinstance(q, str):
                return PC.box_gap(T[p], T[q])
            if isinstance(p, str):
                p, q = q, p
            if isinstance(q, str):
                return PC.box_to_point(T[q], R[p][:2]) - R[p][2]
            return np.hypot(R[p][0] - R[q][0], R[p][1] - R[q][1]) - R[p][2] - R[q][2]
        assert min(gap(p, q) for p, q in zip(chain, chain[1:])) > 0.05, row
    for x, y, box in tubes:
        if x == 5.5:
            assert PC.box_to_point(box, R[5][:2]) - R[5][2] > 0.1

    # P2.3: the tilted setting pushed through -- each end swallowed its cord to just past
    # where its far tube stood (x = 6.0), every other piece gone, the upper still at a half,
    # one half set aside
    assert p23.dump == (
        "setting=tilted setting upper=green:12:far tubes lower=green:0:far tubes turn=gone landed=none aside=1 | "
        "cols=8 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:end green 12 @6,0; 2:end green 0 @6,1; "
        "cord loose-1 [7,0 6,0]; cord loose-2 [7,1 6,1]")
    assert PC.REST[3] == PC.AT["FAR"] + 0.5 == 6.0
    assert [t["kind"] for t in p23.cloth["things"]] == ["end", "end"] and p23.cloth["tubes"] == []
    # between P2.2 and P2.3 the cloth, its rows and its pitch are the same
    g2, g3 = (PC.C._geometry(p.cloth, PC.C.options(PC.CLOTHOPTS, PC.TWHO)) for p in (p22, p23))
    assert all(g2[k] == g3[k] for k in ("P", "Q", "PX", "EX", "y0", "ftop", "fbot", "cols", "rows")) and p22.frame == p23.frame
    # the set-aside bob lies wholly below the linen's near edge (its shadow and its stroke
    # included), inside the frame's band, at the right-hand end
    v3 = p23.shown
    assert len(v3.aside) == 1 and v3.aside == ((PC.C.cloth_xy(p23.scene, PC.CLOTHOPTS, 6, 1, PC.TWHO)[0], v3.linen - PC.ASIDE_DOWN),)
    (x, y), = v3.aside
    hw = 0.3 * PC.PT / 2
    top, bottom = y + PC.ASIDE_RO + hw, y - 0.018 - PC.ASIDE_RO
    assert v3.linen - 0.03 - hw - top > 0.05                               # the linen's own shadow too
    assert bottom - (v3.fbot - PC.ASIDE) > 0.05 and x + 0.014 + PC.ASIDE_RO < 1.25 and x > 0
    assert v3.bead == ((x, y - (PC.ASIDE_RO + PC.ASIDE_RI) / 2),)        # its bead at a half
    assert p22.shown.aside == ()


# --- the sketchbook ---------------------------------------------------------------------

class Cords:
    """n cords read left to right, row 0 the top cord.  Each cord starts at an input
    boundary, or at a point when `points` gives its (kind, phase).  A collar (the yellow
    box) is a Hadamard EDGE on the next stretch of its cord; two collars with nothing
    between them cancel (H H = I).  `cut` ends a cord in an effect and starts it again at
    a point: a term of a sum of pictures."""

    def __init__(self, n, points=None):
        self.g, self.n, self.col = zx.Graph(), n, 1
        self.ins, self.end, self.pend = [], [], [ET.SIMPLE] * n
        for r in range(n):
            if points is None:
                v = self.g.add_vertex(VT.BOUNDARY, r, 0)
                self.ins.append(v)
            else:
                kind, phase = points[r]
                v = self.g.add_vertex(kind, r, 0, phase=Fraction(phase))
            self.end.append(v)

    def dot(self, r, kind, phase=0):
        v = self.g.add_vertex(kind, r, self.col, phase=Fraction(phase))
        self.g.add_edge((self.end[r], v), self.pend[r])
        self.pend[r], self.end[r] = ET.SIMPLE, v
        self.col += 1
        return v

    def collar(self, *rows):
        for r in rows:
            self.pend[r] = ET.HADAMARD if self.pend[r] == ET.SIMPLE else ET.SIMPLE

    def cut(self, r, kind, phase=0):
        """An effect (kind, phase) closes the cord; a point (kind, phase) starts it again."""
        eff = self.dot(r, kind, phase)
        pt = self.g.add_vertex(kind, r, self.col, phase=Fraction(phase))
        self.end[r] = pt
        self.col += 1
        return eff, pt

    def gadget(self, rows, ring, pendant):
        """Rings (green, phase `ring`) on `rows`, tied to one phase-free red hub, with a
        pendant green point of phase `pendant`.  Bare, a hub of k ties with its pendant
        contributes 2^{-(k-1)/2}."""
        rings = [self.dot(r, VT.Z, ring) for r in rows]
        hub = self.g.add_vertex(VT.X, sum(rows) / len(rows) + 0.5, self.col)
        bob = self.g.add_vertex(VT.Z, sum(rows) / len(rows) + 0.5, self.col + 1, phase=Fraction(pendant))
        for v in rings:
            self.g.add_edge((v, hub))
        self.g.add_edge((hub, bob))
        self.col += 2
        return rings, hub, bob

    def ccz(self):
        """CCZ as phase gadgets (Result 15.17): a green eighth on each cord; three hubs of
        two ties (fresh phase-free rings) with pendants an eighth back; one hub of three
        ties with a pendant an eighth forward.  Bare: 2^{-3/2} * 2^{-1} = 2^{-5/2} CCZ."""
        for r in range(3):
            self.dot(r, VT.Z, Fraction(1, 4))
        for pair in ([0, 1], [1, 2], [0, 2]):
            self.gadget(pair, 0, Fraction(-1, 4))
        self.gadget([0, 1, 2], 0, Fraction(1, 4))

    def done(self):
        outs = []
        for r in range(self.n):
            o = self.g.add_vertex(VT.BOUNDARY, r, self.col + 1)
            self.g.add_edge((self.end[r], o), self.pend[r])
            outs.append(o)
        self.g.set_inputs(tuple(self.ins))
        self.g.set_outputs(tuple(outs))
        return self.g


def scaled(g, power=0, sign=0):
    """g times sqrt2^power times (-1)^sign (add_power counts powers of sqrt2)."""
    h = g.copy()
    h.scalar.add_power(power)
    if sign:
        h.scalar.add_phase(Fraction(1))
    return h


def bare_cords(n):
    return Cords(n).done()


def effects_points(n, kind, phases):
    """n effects beside n points, of one kind, the phases per cord (units of pi)."""
    c = Cords(n)
    for r in range(n):
        c.cut(r, kind, phases[r])
    return c.done()


def build(points=None):
    """The Tortoise's cloth build (two cords): collars, a half-ring on each cord, the
    controlled half-turn, collars.  Bare: D / sqrt2."""
    c = Cords(2, points)
    c.collar(0, 1)
    c.dot(0, VT.Z, 1)
    c.dot(1, VT.Z, 1)
    c.gadget([0, 1], Fraction(1, 2), Fraction(-1, 2))
    c.collar(0, 1)
    return c.done()


def xform_turnabout(n):
    """-W X^n C^{n-1}Z X^n W without its -1: collars, red pi's, the multi-controlled
    half-turn in dots (n = 2: the controlled half-turn; n = 3: the CCZ gadgets), red pi's,
    collars.  Bare: -D/sqrt2 (n = 2), -2^{-5/2} D (n = 3)."""
    c = Cords(n)
    c.collar(*range(n))
    for r in range(n):
        c.dot(r, VT.X, 1)
    if n == 2:
        c.gadget([0, 1], Fraction(1, 2), Fraction(-1, 2))
    else:
        c.ccz()
    for r in range(n):
        c.dot(r, VT.X, 1)
    c.collar(*range(n))
    return c.done()


def reduces_to_bare_cords(g, n):
    """full_reduce leaves every input joined straight to its output, scalar 1."""
    h = g.copy()
    simp.full_reduce(h)
    if any(h.type(v) != VT.BOUNDARY for v in h.vertices()):
        return False
    return len(h.inputs()) == n and np.isclose(h.scalar.to_number(), 1) and all(
        h.connected(i, o) and h.edge_type(h.edge(i, o)) == ET.SIMPLE
        for i, o in zip(h.inputs(), h.outputs()))


def sketchbook():
    # (a) The sum-of-pictures chain, n = 2 and n = 3, one picture at a time.
    for n in (2, 3):
        N = 2 ** n
        ones = [1] * n
        # Result 15.6 in pictures for every c: bare cords - 2^{1-n} (red c-effects beside
        # red c-points) is I - 2|c><c|; the lit picture is 2^n |1..1><1..1|
        for c in range(N):
            e = np.eye(N)[c]
            term = matrix_of(effects_points(n, VT.X, bits(c, n)))
            assert np.allclose(term, 2 ** n * np.outer(e, e))
            assert np.allclose(matrix_of(bare_cords(n)) - 2 ** (1 - n) * term, np.eye(N) - 2 * np.outer(e, e))
        assert np.allclose(np.eye(N) - 2 ** (1 - n) * matrix_of(effects_points(n, VT.X, ones)), multi_cz(n))

        # Step 1: D = -(term A) + 2^{1-n} (term B), term A the bare-cords term with its
        # collars and red pi's, term B the lit term with them
        A = Cords(n)
        A.collar(*range(n))
        for r in range(n):
            A.dot(r, VT.X, 1)
        for r in range(n):
            A.dot(r, VT.X, 1)
        A.collar(*range(n))
        A = A.done()
        B = Cords(n)
        B.collar(*range(n))
        red_pis, lit = [], []
        for r in range(n):
            red_pis.append(B.dot(r, VT.X, 1))
        for r in range(n):
            lit.append(B.cut(r, VT.X, 1))
        for r in range(n):
            red_pis.append(B.dot(r, VT.X, 1))
        B.collar(*range(n))
        B = B.done()
        D = grover_D(N)
        assert np.allclose(-matrix_of(A) + 2 ** (1 - n) * matrix_of(B), D)
        assert not np.allclose(matrix_of(A) - 2 ** (1 - n) * matrix_of(B), D)   # the sign

        # Step 2, the middle (between the two columns of collars): red pi's either side of
        # the tall box, with the -1, is R; after fusion it is 2^{1-n} DARK effects beside dark
        # points minus the bare cords -- Result 15.6 at c = 0..0 with its sign reversed, R.
        # So the chain passes through W R W.
        midA = Cords(n)
        for r in range(n):
            midA.dot(r, VT.X, 1)
        for r in range(n):
            midA.dot(r, VT.X, 1)
        midA = midA.done()
        midB = Cords(n)
        mid_pis, mid_lit = [], []
        for r in range(n):
            mid_pis.append(midB.dot(r, VT.X, 1))
        for r in range(n):
            mid_lit.append(midB.cut(r, VT.X, 1))
        for r in range(n):
            mid_pis.append(midB.dot(r, VT.X, 1))
        midB = midB.done()
        R_N = grover_R(N)
        assert np.allclose(-matrix_of(midA) + 2 ** (1 - n) * matrix_of(midB), R_N)
        assert reduces_to_bare_cords(midA, n)                           # red pi, red pi: a bare cord
        for r in range(n):                                              # the named rule, fusion
            for dot in (mid_pis[r], mid_pis[n + r]):
                for tgt in mid_lit[r]:
                    if midB.connected(dot, tgt):
                        assert simp.check_fuse(midB, tgt, dot) and simp.fuse(midB, tgt, dot)
                        break
            assert all(midB.phase(v) == 0 for v in mid_lit[r])          # dark
        darks = effects_points(n, VT.X, [0] * n)
        assert same_picture(midB, darks)
        assert np.allclose(2 ** (1 - n) * matrix_of(darks) - matrix_of(bare_cords(n)), R_N)
        assert not np.allclose(matrix_of(bare_cords(n)) - 2 ** (1 - n) * matrix_of(darks), R_N)   # -R
        W = kron(*([H] * n))
        assert np.allclose(W @ R_N @ W, D)                              # the chain passes through WRW

        # Step 2 on term A, DERIVED: the red pi's fuse to a red 0, the plain dot goes, the
        # collars meet and cancel -- full_reduce leaves bare cords, scalar 1
        assert reduces_to_bare_cords(A, n)
        # Step 2 on term B, by the named rule: each red pi FUSES into its lit effect or lit
        # point, and pi + pi = 0 leaves it dark
        Bf = B.copy()
        for r in range(n):
            eff, pt = lit[r]
            for dot in (red_pis[r], red_pis[n + r]):
                for tgt in (eff, pt):
                    if Bf.connected(dot, tgt):
                        assert simp.check_fuse(Bf, tgt, dot) and simp.fuse(Bf, tgt, dot)
                        break                                           # the red pi is gone
            assert red_pis[r] not in Bf.vertices() and red_pis[n + r] not in Bf.vertices()
            assert Bf.phase(eff) == 0 and Bf.phase(pt) == 0              # dark
        assert same_picture(Bf, B)                                      # fusion costs nothing
        dark = Cords(n)
        dark.collar(*range(n))
        for r in range(n):
            dark.cut(r, VT.X, 0)
        dark.collar(*range(n))
        assert same_picture(Bf, dark.done())
        # Step 3, by the named rule: a yellow box on the only leg of a dark effect or point
        # turns it green (colour change)
        for r in range(n):
            for v in lit[r]:
                assert simp.check_color_change(Bf, v) and simp.color_change(Bf, v)
                assert Bf.type(v) == VT.Z and Bf.phase(v) == 0
        greens = effects_points(n, VT.Z, [0] * n)
        assert same_picture(Bf, greens)
        # Step 4: the green points are 2^{n/2}|s>, so the term is 2^n |s><s| and
        # D = 2^{1-n} (green effects beside green points) - bare cords = 2|s><s| - I
        s = even_state(n)
        assert np.allclose(matrix_of(greens), 2 ** n * np.outer(s, s))
        assert np.allclose(2 ** (1 - n) * matrix_of(greens) - matrix_of(bare_cords(n)), D)
        # the entries as closed pictures: 2^n <x|D|y> = 2 - 2^n [x = y]
        for x, y in [(0, 0), (0, N - 1), (N - 1, N - 1)]:
            assert np.isclose(2 ** n * D[x, y], 2 - 2 ** n * (x == y))

    # (b) The two-cord build.  Bare it is D/sqrt2, so add_power(1).  DERIVED: the build
    #     composed with the mirror of the X-form (red pi's, controlled half-turn, red pi's,
    #     with its -1) full_reduces to bare cords -- the two drawings are one process; and
    #     the build composed with itself full_reduces to bare cords -- D^2 = I in pictures.
    D4 = grover_D(4)
    B2 = scaled(build(), 1)
    assert np.allclose(matrix_of(B2), D4)
    assert np.allclose(matrix_of(build()), D4 / np.sqrt(2))              # the scalar is drawn
    X2 = scaled(xform_turnabout(2), 1, sign=1)
    assert np.allclose(matrix_of(X2), D4)
    assert same_picture(B2, X2)
    assert not same_picture(B2, scaled(xform_turnabout(2), 1))           # the X-form's -1
    both = B2.copy()
    both.compose(X2.adjoint())
    assert reduces_to_bare_cords(both, 2)
    twice = B2.copy()
    twice.compose(B2.copy())
    assert reduces_to_bare_cords(twice, 2)
    h = B2.copy()
    simp.full_reduce(h)
    assert same_picture(h, B2)
    # Result 15.7 (i) by the basis-point lemma, DERIVED: feed a dark or lit point to each
    # cord of R's picture (a half-ring on each cord, the controlled half-turn); full_reduce
    # leaves the same two points, times 2^{-1/2} and a sign that is -1 unless both are dark
    for c1, c2 in product((0, 1), repeat=2):
        R = Cords(2, points=[(VT.X, c1), (VT.X, c2)])
        R.dot(0, VT.Z, 1)
        R.dot(1, VT.Z, 1)
        R.gadget([0, 1], Fraction(1, 2), Fraction(-1, 2))
        R = R.done()
        pair = Cords(2, points=[(VT.X, c1), (VT.X, c2)]).done()
        sign = (c1, c2) != (0, 0)
        assert same_picture(R, scaled(pair, -1, sign=sign))
        assert not same_picture(R, scaled(pair, -1, sign=not sign))
        h = R.copy()
        simp.full_reduce(h)
        assert same_picture(h, scaled(pair, -1, sign=sign))
        ends = [next(iter(h.neighbors(o))) for o in h.outputs()]
        assert [h.phase(v) for v in ends] == [c1, c2]                    # the points are kept
        assert all(h.vertex_degree(v) == 1 for v in ends)

    # (c) The cloth claims: the even setting (two green points, sqrt2|+> each) comes out as
    #     it went in; each tilted setting (a green pi point on the upper, the lower, or
    #     both) comes out turned over.  DERIVED by full_reduce on copies.
    for p0, p1 in product((0, 1), repeat=2):
        fed = scaled(build(points=[(VT.Z, p0), (VT.Z, p1)]), 1)
        same = Cords(2, points=[(VT.Z, p0), (VT.Z, p1)]).done()
        tilted = (p0, p1) != (0, 0)
        assert same_picture(fed, scaled(same, 0, sign=tilted))
        assert not same_picture(fed, scaled(same, 0, sign=not tilted))
        h = fed.copy()
        simp.full_reduce(h)
        assert same_picture(h, scaled(same, 0, sign=tilted))
        if not tilted:
            ends = [next(iter(h.neighbors(o))) for o in h.outputs()]
            assert all(h.type(v) == VT.Z and h.phase(v) == 0 and h.vertex_degree(v) == 1 for v in ends)

    # (d) Three cords.  The hub scalars first, each with a generic pendant 3/7 (not a
    #     Clifford turn): a hub of two ties is 2^{-1/2} (add_power(1)), a hub of three ties
    #     2^{-1} (add_power(2)).  Then the CCZ gadgets: bare 2^{-5/2} CCZ, add_power(5) --
    #     not 4, not 6.  CHECKED BY MATRIX: the chapter derives this drawing by no rule.
    #     Non-Clifford route: full_reduce a copy, compare the reduction by matrix_of.
    phi = Fraction(3, 7)
    for k in (2, 3):
        c = Cords(k)
        c.gadget(list(range(k)), 0, phi)
        g = c.done()
        diag = np.array([np.exp(1j * np.pi * float(phi) * (sum(bits(x, k)) % 2)) for x in range(2 ** k)])
        assert np.allclose(matrix_of(scaled(g, k - 1)), np.diag(diag))
        assert not np.allclose(matrix_of(g), np.diag(diag))
    c = Cords(3)
    c.ccz()
    G = c.done()
    assert max(Fraction(G.phase(v)).denominator for v in G.vertices()) == 4   # it needs eighths
    assert np.allclose(matrix_of(scaled(G, 5)), CCZ)
    assert np.allclose(matrix_of(G), 2 ** -2.5 * CCZ)
    for wrong in (4, 6):
        assert not np.allclose(matrix_of(scaled(G, wrong)), CCZ)        # never 2^{-2}, never 2^{-3}
    h = scaled(G, 5)
    simp.full_reduce(h)
    assert np.allclose(matrix_of(h), CCZ)
    # the three-cord X-form turn-about: add_power(5) and the -1, exactly D on eight drawers
    D8 = grover_D(8)
    X3 = scaled(xform_turnabout(3), 5, sign=1)
    assert np.allclose(matrix_of(X3), D8)
    assert not np.allclose(matrix_of(scaled(xform_turnabout(3), 5)), D8)
    h = X3.copy()
    simp.full_reduce(h)
    assert np.allclose(matrix_of(h), D8)
    # and one step of an eight-drawer search built from it (chapter Result 15.17's remark):
    # three dark points, collars, the marker for 011 (red pi on the top cord either side of
    # the CCZ gadgets), then the turn-about.  Two CCZ's are 2^{-5}, the points 2^{3/2}:
    # add_power(10 - 3); the X-form's -1: add_phase.  Marked amplitude sin(3 theta).
    c = Cords(3, points=[(VT.X, 0)] * 3)
    c.collar(0, 1, 2)
    c.dot(0, VT.X, 1)
    c.ccz()
    c.dot(0, VT.X, 1)
    c.collar(0, 1, 2)
    for r in range(3):
        c.dot(r, VT.X, 1)
    c.ccz()
    for r in range(3):
        c.dot(r, VT.X, 1)
    c.collar(0, 1, 2)
    step = scaled(c.done(), 10 - 3, sign=1)
    v = matrix_of(step).ravel()
    theta = np.arcsin(1 / np.sqrt(8))
    assert np.isclose(v[0b011], np.sin(3 * theta)) and np.isclose(v[0b011], 0.883883, atol=1e-6)
    assert np.isclose(v[0b110], np.cos(3 * theta) / np.sqrt(7))         # upside down: unmarked
    h = step.copy()
    simp.full_reduce(h)
    assert np.allclose(matrix_of(h).ravel(), v)


if __name__ == "__main__":
    verdict("TurnAbout", blackboard, sketchbook)
