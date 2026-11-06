"""Challenge I: Four Drawers, One Wish  (the classical cost; the marker is one query; the even state)
=====================================================================================================
"A quarter of the time the first is the one, and I have it at once. The other three quarters, I guess among three and I'm right one time in three, which is another quarter. Half the time, all told."  -- Achilles
"One, one, one, one. The squares haven't changed."  -- Achilles

Claim: asking one by one takes N-1 asks in the worst case and (N-1)/2 + (N-1)/N on average
(Result 15.1); every classical method with k asks names the marked drawer with probability
at most (k+1)/N, checked over every decision tree for four drawers, so one ask and a guess
is exactly 2/N (Result 15.2); the query box with its flag at |-> is the marker
O_w = I - 2|w><w| on the rods, which is unitary, its own inverse, and changes no odds
(Result 15.3); W|0..0> = |s>, W is real, symmetric and its own inverse (Result 15.4); and,
in pictures, the marker on two cords is the controlled half-turn with half-rings (Z-form)
or with red pi's either side (X-form), a red point through a yellow box is a green point,
and a basis point passes the marker with its sign and the price 2^{-1/2}.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import ket, kron, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch15 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
MINUS = (ket(0) - ket(1)) / np.sqrt(2)
PLUS = (ket(0) + ket(1)) / np.sqrt(2)


def bits(x, n):
    """The n bits of the drawer x, top cord first."""
    return tuple((x >> (n - 1 - i)) & 1 for i in range(n))


def marker(w, n):
    """O_w = I - 2|w><w|."""
    O = np.eye(2 ** n, dtype=complex)
    O[w, w] = -1
    return O


def query_box(w, n):
    """U_f |x>|y> = |x>|y xor f(x)>, f(x) = [x = w], the flag LAST (the bottom cord)."""
    U = np.zeros((2 ** (n + 1), 2 ** (n + 1)), dtype=complex)
    for x in range(2 ** n):
        for y in (0, 1):
            U[2 * x + (y ^ (x == w)), 2 * x + y] = 1
    return U


def walsh(n):
    return kron(*([H] * n))


def even_state(n):
    return np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)


# --- Result 15.2, over every deterministic decision tree -------------------------------
@lru_cache(maxsize=None)
def achievable(N, alive, asks):
    """Every number of marked drawers a deterministic method can find, over all decision
    trees with `asks` asks left, when the drawers still consistent with the answers so far
    are `alive` (a frozenset).  A node asks about ANY drawer q (even a ruled-out one); a
    leaf names ANY drawer z.  Coins are an average over such trees."""
    if asks == 0:
        return frozenset({0, 1}) if alive else frozenset({0})       # name z: hit iff z alive
    out = set()
    for q in range(N):
        yes, no = alive & {q}, alive - {q}
        for a in achievable(N, frozenset(yes), asks - 1):
            for b in achievable(N, frozenset(no), asks - 1):
                out.add(a + b)
    return frozenset(out)


def blackboard():
    rng = np.random.default_rng(1501)

    # --- Result 15.1: ask in a fixed order, stop at the first yes, name the last unasked
    for N in (4, 64):
        def asks_for(w, order):
            j = order.index(w) + 1
            return min(j, N - 1)                                        # the last is deduced
        order = list(range(N))
        counts = [asks_for(w, order) for w in range(N)]
        assert max(counts) == N - 1                                     # worst case N - 1
        assert Fraction(sum(counts), N) == Fraction(N - 1, 2) + Fraction(N - 1, N)
    assert Fraction(9, 4) == Fraction(3, 2) + Fraction(3, 4)            # 2.25 for four
    big = 2 ** 20                                                       # the opening's million:
    assert abs((big - 1) / 2 + (big - 1) / big - 5e5) / 5e5 < 0.05      # about half a million
    assert np.isclose(float(Fraction(63, 2) + Fraction(63, 64)), 32.48, atol=5e-3)   # 32.48
    # the chapter's worked order 00, 01, 10, 11: 1, 2, 3, 3 asks
    assert [min(j, 3) for j in (1, 2, 3, 4)] == [1, 2, 3, 3]
    # and the expectation does not depend on the order (every order of four drawers)
    for order in permutations(range(4)):
        assert Fraction(sum(min(order.index(w) + 1, 3) for w in range(4)), 4) == Fraction(9, 4)

    # --- Result 15.2, exhaustively: every decision tree for four drawers (and five, six)
    for N in (4, 5, 6):
        for k in range(0, N):
            best = max(achievable(N, frozenset(range(N)), k))
            assert best == min(k + 1, N)                                # (k+1)/N, attained
    four = {k: Fraction(max(achievable(4, frozenset(range(4)), k)), 4) for k in range(4)}
    assert four == {0: Fraction(1, 4), 1: Fraction(1, 2), 2: Fraction(3, 4), 3: Fraction(1)}
    # success 1/2 needs N/2 - 1 asks; certainty needs N - 1 (three wishes, to be sure)
    assert min(k for k in four if four[k] >= Fraction(1, 2)) == 4 // 2 - 1
    assert min(k for k in four if four[k] == 1) == 4 - 1 == 3
    # one ask about a fixed drawer, then a uniform guess among the others: exactly 2/N
    for N in (4, 64):
        p = Fraction(0)
        for w in range(N):
            p += Fraction(1, N) * (1 if w == 0 else Fraction(1, N - 1))
        assert p == Fraction(2, N)
    # Achilles's reckoning: "a quarter ... another quarter. Half the time, all told."
    assert Fraction(1, 4) + Fraction(3, 4) * Fraction(1, 3) == Fraction(1, 2)
    assert Fraction(2, 64) == Fraction(1, 32)

    # --- Result 15.3: the query box with its flag at |-> is the marker
    for n in (1, 2, 3):
        N = 2 ** n
        for w in range(N):
            U, O = query_box(w, n), marker(w, n)
            assert np.allclose(U @ U.conj().T, np.eye(2 * N))            # a box, unitary
            for x in range(N):                                          # every drawer
                e = np.eye(N)[x]
                assert np.allclose(U @ np.kron(e, MINUS), (-1) ** (x == w) * np.kron(e, MINUS))
            psi = random_state(rng, N)                                  # a random complex query
            assert not np.allclose(psi, psi.conj())
            assert np.allclose(U @ np.kron(psi, MINUS), np.kron(O @ psi, MINUS))
            # the flag at |+> kicks nothing back: the |-> is what makes it a marker
            assert np.allclose(U @ np.kron(psi, PLUS), np.kron(psi, PLUS))
            assert not np.allclose(U @ np.kron(psi, PLUS), np.kron(O @ psi, PLUS))
            # unitary, its own inverse, and no odds change, on a complex state
            assert np.allclose(O @ O.conj().T, np.eye(N)) and np.allclose(O @ O, np.eye(N))
            assert np.allclose(np.abs(O @ psi) ** 2, np.abs(psi) ** 2)
            assert not np.allclose(O @ psi, psi)                        # yet the state moved
            # the marker on the even state: |s> - (2/sqrt N)|w>, every odds 1/N
            s = even_state(n)
            assert np.allclose(O @ s, s - 2 / np.sqrt(N) * np.eye(N)[w])
            assert np.allclose(np.abs(O @ s) ** 2, 1 / N)

    # --- Result 15.4: the even state; W real, symmetric, its own inverse
    for n in range(1, 7):
        W = walsh(n)
        assert np.allclose((W @ ket(*([0] * n))).ravel(), even_state(n))
        assert np.allclose(W, W.real) and np.allclose(W, W.T) and np.allclose(W @ W, np.eye(2 ** n))
        for x, y in [(0, 1), (2 ** n - 1, 2 ** n - 1), (1, 2 ** n - 1)]:
            dot = sum(a * b for a, b in zip(bits(x, n), bits(y, n))) % 2
            assert np.isclose(W[x, y], 2 ** (-n / 2) * (-1) ** dot)
    assert np.allclose(np.kron(H @ ket(0), H @ ket(0)), 0.5 * np.ones(4))

    # --- the pegs: the even row, the second drawer (0b01, top cord first) touched
    assert bits(0b01, 2) == (0, 1)                                      # upper dark, lower lit
    pegs = np.array([1, 1, 1, 1])
    touched = (marker(0b01, 2) @ pegs).real
    assert np.allclose(touched, [1, -1, 1, 1])                          # one peg under
    assert np.allclose(touched ** 2, pegs ** 2)                         # "the squares haven't changed"
    assert not np.allclose(touched, marker(0b10, 2).real @ pegs)        # not the third drawer
    assert np.allclose(marker(0b01, 2) @ even_state(2), 0.5 * np.array([1, -1, 1, 1]))

    # --- Result 15.7 (ii), used by the sketchbook below: the two-cord forms, every w
    CZ = np.diag([1, 1, 1, -1]).astype(complex)
    for w in range(4):
        w1, w2 = bits(w, 2)
        nb1, nb2 = 1 - w1, 1 - w2
        zform = (-1) ** (nb1 * nb2) * kron(np.linalg.matrix_power(Z, nb2), np.linalg.matrix_power(Z, nb1)) @ CZ
        Xw = kron(np.linalg.matrix_power(X, nb1), np.linalg.matrix_power(X, nb2))
        assert np.allclose(zform, marker(w, 2))
        assert np.allclose(Xw @ CZ @ Xw, marker(w, 2))
    # the half-ring on the wrong cord marks the third drawer, not the second
    assert not np.allclose(kron(Z, np.eye(2)) @ CZ, marker(0b01, 2))
    assert np.allclose(kron(Z, np.eye(2)) @ CZ, marker(0b10, 2))

    pictures()


# --- the prop pictures ------------------------------------------------------------------

def pictures():
    """The First Challenge's prop pictures (identity card, section 4), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch15.py.  The
    ties below hold the transcription against this file's blackboard: a structural check
    of the pictures, not a second proof.  A peg's height is an unnormalized amplitude; a
    drawer is an index, the first |00>, the second |01> (top cord first)."""
    calls = DIALOGUE.check_layout("First", sizes=[1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("drawercase", True, "", ("",), "the Genie's drawer-case"),
        ("pegcard", True, "", ("even row",), "level across the card"),
        ("pegcard", False, "touch the second", ("even row",), "the marked peg under")]
    PC.check_captions(DIALOGUE, "First", calls)
    PC.check_section(DIALOGUE, "First", calls)
    case_h, pegs_h = round(PC.CFTOP - PC.CFBOT, 2), round(PC.PFTOP - PC.PFBOT, 2)
    assert (case_h, pegs_h) == (1.51, 1.48)                           # read from the Lua
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("First")] == [[case_h], [pegs_h], [pegs_h]]

    # the anchors (card section 4): the paragraph before each block and the one after
    a1, a2, a3 = DIALOGUE.anchors("First")
    f1, f2, f3 = DIALOGUE.following("First")
    assert "A spent case unlocks its drawers. The marked drawer holds the key to its back.\"" in a1
    assert f1.startswith("**Achilles:** So one drawer has the key in it")
    assert a2 == "*[She slides each of the four pegs up its slit to the first notch above the middle line.]*"
    assert f2.startswith("**Tortoise:** The even row: every peg one notch up.")
    assert "The second peg is exactly as far from the line as it was, only on the other side: under." in a3
    assert f3.startswith("**Achilles:** One, one, one, one. The squares haven't changed.")
    assert DIALOGUE.following("First", DIALOGUE.raw) == [f1, f2, f3]
    # the section carries D1, D2 and D3's anchor phrases, once each, in both files
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        sec = DIALOGUE.section("First", path)
        for phrase in ("bending at the shelf to the plate on the top of the case, its drawers towards him and "
                       "its windows at his right",                                                     # D1
                       "setting it at Achilles's right, a little beyond his place, as he takes his chair again",  # D2
                       "a strip of cream card, with a wooden peg in each of its slits, and a pencil",  # D3
                       "its first slit at his left, and every slit running away from him",
                       "The pencil she lays beside it.]*"):
            assert sec.count(phrase) == 1, phrase

    p11, p12, p13 = (PC.picture(c) for c in calls)

    # P1.1: the case before the wish -- sealed, both panes plain, every drawer shut
    assert p11.dump == "granted=false windows=-- drawers=ssss key=in back=sealed"
    assert p11.family == "front" and p11.state.windows is None and not any(p11.state.open)
    assert [g for *_, g in p11.shown.windows] == [False, False] and p11.shown.key is None
    assert len(p11.shown.pulls) == 4 and p11.shown.fronts == ()
    assert p11.labels == ("drawer", "plate", "upper window", "lower window")
    # the windows sit on the right-hand end, the upper above the lower; the drawers run
    # first to fourth from the left
    (_, ux, uy, _), (_, lx, ly, _) = p11.shown.windows
    assert ux == lx > PC.X1 and uy > ly
    xs = [(a + b) / 2 for _, a, b, _ in p11.shown.drawers]
    assert xs == sorted(xs) and all(PC.X0 < a and b < PC.X1 for _, a, b, _ in p11.shown.drawers)
    # the keyed words, as set (node boxes measured in the book's preamble, strokes and
    # shadows included): each clears the drawing, the others and the frame's edges; the
    # letters by more
    for ink, least in ((False, 0.005), (True, 0.03)):
        for w, g in PC.word_gaps(p11, ink).items():
            assert min(g.values()) > least, (w, min(g, key=g.get), min(g.values()))

    # P1.2: the even row, every peg one notch up, is 2|s> for four drawers
    ones = 2 * even_state(2)
    assert p12.dump == "pegs=1,1,1,1 ticks=none"
    assert np.allclose(p12.state.h, ones)
    assert p12.labels == ("notch", "peg", "slit", "middle line")
    for ink, least in ((False, 0.0005), (True, 0.03)):
        for w, g in PC.word_gaps(p12, ink).items():
            assert min(g.values()) > least, (w, min(g, key=g.get), min(g.values()))
    # "peg" stands just right of the fourth peg: its node box clears the peg's drop shadow
    # by a hair (the inner sep, which nothing draws), the letters by more than 0.03
    assert PC.word_gaps(p12)["peg"]["peg 4 shadow"] < 0.005

    # P1.3: the marker touched the second peg -- O_w 2|s> with w = 01, the second drawer
    assert p13.dump == "pegs=1,-1,1,1 ticks=none"
    assert bits(1, 2) == (0, 1)                                          # the second drawer, top cord first
    assert np.allclose(p13.state.h, marker(0b01, 2) @ ones)
    assert not np.allclose(p13.state.h, marker(0b10, 2) @ ones)        # not the third
    assert np.allclose(np.square(p13.state.h), np.square(p12.state.h))  # the squares unchanged
    assert np.isclose(np.linalg.norm(marker(0b01, 2) @ ones), np.linalg.norm(ones))
    # for every drawer w, touching its peg on the even row is O_w on 2|s>
    for w, name in enumerate(("first", "second", "third", "fourth")):
        P = PC.pegs_state("touch the " + name, "even row")
        assert np.allclose(P.h, marker(w, 2) @ ones) and P.ticks == []
    assert not PC.TEX_ERRORS

    # fixed parts: the card's slits, middle line and notches are the same in every picture
    # of it; the touch changes the second peg's height only
    v2, v3 = p12.shown, p13.shown
    assert (v2.card, v2.slits, v2.notches, v2.middle) == (v3.card, v3.slits, v3.notches, v3.middle)
    assert [y for _, _, y in v2.pegs][0::2] == [y for _, _, y in v3.pegs][0::2] and v2.pegs[3] == v3.pegs[3]
    assert v3.pegs[1][2] < PC.MID < v2.pegs[1][2]
    # every peg lies inside its slit, its head and stroke clear of the slit's ends
    for v in (v2, v3):
        for (k, x, y), (_, x0, y0, x1, y1) in zip(v.pegs, v.slits):
            assert x0 < x < x1 and y0 + PC.PR + 0.3 * PC.PT / 2 < y < y1 - PC.PR - 0.3 * PC.PT / 2


# --- the sketchbook ---------------------------------------------------------------------

class Cords:
    """n cords read left to right, row 0 the top cord.  Each cord starts at an input
    boundary, or at a point when `points` gives its (kind, phase).  A collar (the yellow
    box) is a Hadamard EDGE on the next stretch of its cord; two collars with nothing
    between them cancel (H H = I) and leave a plain edge."""

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

    def gadget(self, rows, ring, pendant):
        """Rings (green, phase `ring`) on `rows`, tied to one phase-free red hub, with a
        pendant green point of phase `pendant`.  At ring = 1/2, pendant = -1/2 on two
        rows this is the controlled half-turn, bare CZ / sqrt2."""
        rings = [self.dot(r, VT.Z, ring) for r in rows]
        hub = self.g.add_vertex(VT.X, sum(rows) / len(rows) + 0.5, self.col)
        bob = self.g.add_vertex(VT.Z, sum(rows) / len(rows) + 0.5, self.col + 1, phase=Fraction(pendant))
        for v in rings:
            self.g.add_edge((v, hub))
        self.g.add_edge((hub, bob))
        self.col += 2
        return rings, hub, bob

    def done(self):
        """Close every cord in an output boundary."""
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


def marker_zform(w, points=None):
    """Result 15.7 (ii): the controlled half-turn, a half-ring on the lower cord if
    w1 = 0 and on the upper cord if w2 = 0.  Bare: (-1)^{[w=00]} O_w / sqrt2."""
    w1, w2 = bits(w, 2)
    c = Cords(2, points)
    c.gadget([0, 1], Fraction(1, 2), Fraction(-1, 2))
    if w2 == 0:
        c.dot(0, VT.Z, 1)
    if w1 == 0:
        c.dot(1, VT.Z, 1)
    return c.done()


def marker_xform(w, points=None):
    """X^{w-bar} CZ X^{w-bar}: red pi's on the cords where w has a 0, either side of the
    controlled half-turn.  Bare: O_w / sqrt2, no sign."""
    c = Cords(2, points)
    flips = [r for r, b in enumerate(bits(w, 2)) if b == 0]
    for r in flips:
        c.dot(r, VT.X, 1)
    c.gadget([0, 1], Fraction(1, 2), Fraction(-1, 2))
    for r in flips:
        c.dot(r, VT.X, 1)
    return c.done()


def reduces_to_bare_cords(g, n):
    """Rule derivation: full_reduce leaves every input joined straight to its output by a
    plain edge, no dot in between, and the scalar 1."""
    h = g.copy()
    simp.full_reduce(h)
    inner = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
    if inner:
        return False
    for i, o in zip(h.inputs(), h.outputs()):
        if not h.connected(i, o) or h.edge_type(h.edge(i, o)) != ET.SIMPLE:
            return False
    return len(h.inputs()) == n and np.isclose(h.scalar.to_number(), 1)


def query_picture(w, flag_phase):
    """The query box on two rods and a flag (the bottom cord), U_f = X^{w-bar} . Toffoli .
    X^{w-bar}, the Toffoli being collars on the flag around the CCZ drawn as phase gadgets
    (a green eighth on each cord; three two-tie hubs, pendants an eighth back; one
    three-tie hub, pendant an eighth forward; bare 2^{-5/2} CCZ).  The flag starts at a
    green point with phase flag_phase (pi: sqrt2 |->; 0: sqrt2 |+>)."""
    c = Cords(3)
    # replace the flag's input by a green point
    g = c.g
    flag_in = c.ins.pop()
    g.set_type(flag_in, VT.Z)
    g.set_phase(flag_in, Fraction(flag_phase))
    flips = [r for r, b in enumerate(bits(w, 2)) if b == 0]
    for r in flips:
        c.dot(r, VT.X, 1)
    c.collar(2)
    for r in range(3):
        c.dot(r, VT.Z, Fraction(1, 4))
    for pair in ([0, 1], [1, 2], [0, 2]):
        c.gadget(pair, 0, Fraction(-1, 4))
    c.gadget([0, 1, 2], 0, Fraction(1, 4))
    c.collar(2)
    for r in flips:
        c.dot(r, VT.X, 1)
    return c.done()


def marker_beside_flag(w, flag_phase):
    """The marker on the two rods (X-form), and the flag's green point passing by."""
    c = Cords(3)
    g = c.g
    flag_in = c.ins.pop()
    g.set_type(flag_in, VT.Z)
    g.set_phase(flag_in, Fraction(flag_phase))
    flips = [r for r, b in enumerate(bits(w, 2)) if b == 0]
    for r in flips:
        c.dot(r, VT.X, 1)
    c.gadget([0, 1], Fraction(1, 2), Fraction(-1, 2))
    for r in flips:
        c.dot(r, VT.X, 1)
    return c.done()


def sketchbook():
    # (a) The marker in dots, every drawer, both forms.  The bare controlled half-turn is
    #     CZ/sqrt2, so each form needs add_power(1); the Z-form of the first drawer also
    #     carries the number -1 (add_phase).  DERIVED: the Z-form followed by the mirror of
    #     the X-form full_reduces to two bare cords with scalar 1, so the rules show the two
    #     drawings are one process; and each form full_reduces to a picture with the same
    #     value.
    for w in range(4):
        zf = scaled(marker_zform(w), 1, sign=(w == 0))
        xf = scaled(marker_xform(w), 1)
        assert np.allclose(matrix_of(zf), marker(w, 2))
        assert np.allclose(matrix_of(xf), marker(w, 2))
        assert same_picture(zf, xf)
        both = zf.copy()
        both.compose(xf.adjoint())
        assert reduces_to_bare_cords(both, 2)
        for f in (zf, xf):
            h = f.copy()
            simp.full_reduce(h)
            assert same_picture(h, f)
    # the first drawer's Z-form without its -1 is R, not the marker
    assert not same_picture(scaled(marker_zform(0b00), 1), scaled(marker_xform(0b00), 1))
    assert np.allclose(matrix_of(scaled(marker_zform(0b00), 1)), -marker(0b00, 2))
    # the scalar is drawn: bare, the marker is O_w / sqrt2, not O_w
    assert np.allclose(matrix_of(marker_xform(0b01)), marker(0b01, 2) / np.sqrt(2))
    assert not same_picture(marker_xform(0b01), scaled(marker_xform(0b01), 1))
    # the half-ring's cord names the drawer: 01 has it on the LOWER cord, 10 on the upper
    assert not same_picture(scaled(marker_zform(0b01), 1), scaled(marker_zform(0b10), 1))
    lower = marker_zform(0b01)
    halves = [v for v in lower.vertices() if lower.type(v) == VT.Z and lower.phase(v) == 1]
    assert len(halves) == 1 and lower.qubit(halves[0]) == 1           # qubit = the cord

    # (b) The even state: a red point through a yellow box is a green point (colour
    #     change), and two green points are 2|s>.  DERIVED by full_reduce on a copy.
    reds = Cords(2, points=[(VT.X, 0), (VT.X, 0)])
    reds.collar(0, 1)
    reds = reds.done()
    greens = Cords(2, points=[(VT.Z, 0), (VT.Z, 0)]).done()
    assert same_picture(reds, greens)
    assert np.allclose(matrix_of(greens).ravel(), 2 * even_state(2))    # 2|s>, not |s>
    assert not np.allclose(matrix_of(greens).ravel(), even_state(2))
    h = reds.copy()
    simp.full_reduce(h)
    assert same_picture(h, greens)
    ends = [next(iter(h.neighbors(o))) for o in h.outputs()]
    assert all(h.type(v) == VT.Z and h.phase(v) == 0 and h.vertex_degree(v) == 1 for v in ends)
    assert all(h.edge_type(h.edge(v, o)) == ET.SIMPLE for v, o in zip(ends, h.outputs()))

    # (c) The row-104 check of the marker for the second drawer: a dark or lit point on
    #     each cord (red, phases c1*pi, c2*pi) comes out as the same pair of points, times
    #     2^{-1/2} (-1)^{[c=01]} before compensation.  DERIVED by full_reduce on a copy.
    for c1, c2 in product((0, 1), repeat=2):
        fed = marker_zform(0b01, points=[(VT.X, c1), (VT.X, c2)])
        pair = Cords(2, points=[(VT.X, c1), (VT.X, c2)]).done()
        sign = (c1, c2) == (0, 1)
        assert same_picture(fed, scaled(pair, -1, sign=sign))
        assert not same_picture(fed, scaled(pair, -1, sign=not sign))   # the sign is real
        h = fed.copy()
        simp.full_reduce(h)
        assert same_picture(h, scaled(pair, -1, sign=sign))
        assert np.allclose(matrix_of(fed).ravel(),
                           2 ** -0.5 * (-1) ** sign * 2 * np.kron(ket(c1), ket(c2)))

    # (d) Result 15.3 in pictures: the query box with a green pi point (sqrt2 |->) on its
    #     flag equals the marker on the rods with the green pi point passing by.  The CCZ
    #     gadgets are bare 2^{-5/2} CCZ (add_power(5)); the marker's controlled half-turn
    #     is bare CZ/sqrt2 (add_power(1)).  Eighth-turns are not Clifford, so both sides
    #     are full_reduced on copies and compared by matrix_of.
    for w in range(4):
        lhs = scaled(query_picture(w, 1), 5)
        rhs = scaled(marker_beside_flag(w, 1), 1)
        assert same_picture(lhs, rhs)
        psi = random_state(np.random.default_rng(1510 + w), 4)
        assert np.allclose(matrix_of(lhs) @ psi, np.sqrt(2) * np.kron(marker(w, 2) @ psi, MINUS))
        a, b = lhs.copy(), rhs.copy()
        simp.full_reduce(a)
        simp.full_reduce(b)
        assert same_picture(a, b) and same_picture(a, rhs)
        # a green point with no turn (|+>) on the flag: the query box does nothing at all
        plus = scaled(query_picture(w, 0), 5)
        assert np.allclose(matrix_of(plus) @ psi, np.sqrt(2) * np.kron(psi, PLUS))
        assert not same_picture(plus, scaled(marker_beside_flag(w, 0), 1))


if __name__ == "__main__":
    verdict("FourDrawers", blackboard, sketchbook)
