"""Challenge II: Cooking Down  (phase gadgets; the headline; the master cooked)
============================================================================
"A quarter whenever the upper is lit, a half when the lower is — and when both are, a half taken back."  -- Achilles
"But it never promises the fewest."  -- Crab

Claim: U = (CNOT_12 (I (x) T))^4 = (S (x) Z) CZ exactly (Result 17.4).  In pictures, with
every number written: sandwich(a) = 2^(-1/2) hub(a) (Result 17.2); two hubs on the same
spiders = 2^(-1/2) one hub, ends added (Result 17.3); and the chain of Section 17.6 takes
the recipe's bare picture to 2^(-3/2) times the cooked picture (green quarter on the upper
wire, green half on the lower, one Hadamard edge), which is U/sqrt2.  PyZX's full_reduce
cooks The Uncooked Net to that picture with the number 2^(-3/2); extract_circuit returns
(S (x) Z) CZ exactly; teleport_reduce leaves quarters at the first and fourth eighths
(Result 17.7); the recipe followed by the adjoint cooks to bare wires.  On 20 seeded random
Clifford+T circuits, full_reduce keeps the value exactly (20/20) and lowers the T-count on
18 -- a heuristic: nothing here says the count is the fewest.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
import random
import re
from fractions import Fraction
from itertools import product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import I2, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch17 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

Z = np.diag([1, -1]).astype(complex)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)  # control on top
CZ = np.diag([1, 1, 1, -1]).astype(complex)
U = np.linalg.matrix_power(CNOT @ kron(I2, T), 4)
R2 = np.sqrt(2)


def Zr(a):
    return np.diag([1, np.exp(1j * a)])


def Gamma(a):
    """The two-wire phase gadget diag(1, e^{ia}, e^{ia}, 1)."""
    return CNOT @ kron(I2, Zr(a)) @ CNOT


def basis(x1, x2):
    v = np.zeros(4, dtype=complex)
    v[2 * x1 + x2] = 1
    return v


# ------------------------------------------------------------------ pictures
# PyZX draws the rings as green (Z) and red (X) spiders; a Hadamard edge is the yellow box.
class Two:
    """A two-wire picture built left to right: row 0 is the upper wire."""

    def __init__(self):
        self.g = zx.Graph()
        self.last = [self.g.add_vertex(VT.BOUNDARY, q, 0) for q in (0, 1)]
        self.ins = list(self.last)
        self.col = 1

    def on(self, q, kind, phase=0, et=ET.SIMPLE):
        v = self.g.add_vertex(kind, q, self.col, phase=Fraction(phase))
        self.g.add_edge((self.last[q], v), et)
        self.last[q] = v
        self.col += 1
        return v

    def joiner(self):                                   # green on the upper wire, red on the lower
        a, b = self.on(0, VT.Z), self.on(1, VT.X)
        self.g.add_edge((a, b))

    def hub(self, end, top=0, bottom=0):
        """A green foot on each wire, both tied to one red hub with a green end hanging."""
        a, b = self.on(0, VT.Z, top), self.on(1, VT.Z, bottom)
        h = self.g.add_vertex(VT.X, 0.5, self.col)
        e = self.g.add_vertex(VT.Z, 0.5, self.col + 1, phase=Fraction(end))
        self.g.add_edge((a, h))
        self.g.add_edge((b, h))
        self.g.add_edge((h, e))
        return a, b

    def extra_hub(self, feet, end):                     # another hub on the same two feet
        h = self.g.add_vertex(VT.X, 0.5, self.col)
        e = self.g.add_vertex(VT.Z, 0.5, self.col + 1, phase=Fraction(end))
        self.g.add_edge((feet[0], h))
        self.g.add_edge((feet[1], h))
        self.g.add_edge((h, e))

    def tie(self, et=ET.HADAMARD):                      # two greens at rest joined by a tie
        a, b = self.on(0, VT.Z), self.on(1, VT.Z)
        self.g.add_edge((a, b), et)

    def done(self, power=0):
        outs = [self.g.add_vertex(VT.BOUNDARY, q, self.col + 2) for q in (0, 1)]
        for q in (0, 1):
            self.g.add_edge((self.last[q], outs[q]))
        self.g.set_inputs(tuple(self.ins))
        self.g.set_outputs(tuple(outs))
        self.g.scalar.add_power(power)                  # the picture's number: sqrt2 ** power
        return self.g


def sandwich(a, where=1):
    p = Two()
    p.joiner()
    p.on(where, VT.Z, a)
    p.joiner()
    return p


def hub_picture(a, top=0, bottom=0):
    p = Two()
    p.hub(a, top, bottom)
    return p


def recipe():
    c = zx.Circuit(2)
    for _ in range(4):
        c.add_gate("T", 1)
        c.add_gate("CNOT", 0, 1)
    return c


def bare(g):
    g.scalar = zx.Scalar()
    return g


def uncooked_net():
    """The master, as in FourEighths.py (upper: in1 - small1 => small2 => big - out1;
    lower: in2 - e1 => p1 => ... => e4 => p4 => out2; ties big => p_k)."""
    g = zx.Graph()
    in1, in2 = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0)
    s1, s2, big = g.add_vertex(VT.Z, 0, 1), g.add_vertex(VT.Z, 0, 2), g.add_vertex(VT.Z, 0, 5)
    e = [g.add_vertex(VT.Z, 1, 2 * k - 1, phase=Fraction(1, 4)) for k in range(1, 5)]
    p = [g.add_vertex(VT.Z, 1, 2 * k) for k in range(1, 5)]
    out1, out2 = g.add_vertex(VT.BOUNDARY, 0, 10), g.add_vertex(VT.BOUNDARY, 1, 10)
    g.add_edge((in1, s1))
    g.add_edge((s1, s2), ET.HADAMARD)
    g.add_edge((s2, big), ET.HADAMARD)
    g.add_edge((big, out1))
    g.add_edge((in2, e[0]))
    for k in range(4):
        g.add_edge((e[k], p[k]), ET.HADAMARD)
        g.add_edge((big, p[k]), ET.HADAMARD)
        if k < 3:
            g.add_edge((p[k], e[k + 1]), ET.HADAMARD)
    g.add_edge((p[3], out2), ET.HADAMARD)
    g.set_inputs((in1, in2))
    g.set_outputs((out1, out2))
    return g


def blackboard():
    rng = np.random.default_rng(1702)
    # --- the phase gadget: Gamma(a) = diag(1, e^{ia}, e^{ia}, 1), for 20 random angles
    for a in rng.uniform(-np.pi, np.pi, 20):
        assert np.allclose(Gamma(a), np.diag([1, np.exp(1j * a), np.exp(1j * a), 1]))
        for x1, x2 in product((0, 1), repeat=2):
            assert np.allclose(Gamma(a) @ basis(x1, x2), np.exp(1j * a * (x1 ^ x2)) * basis(x1, x2))
        b = rng.uniform(-np.pi, np.pi)
        assert np.allclose(Gamma(a) @ Gamma(b), Gamma(a + b))              # gadget fusion (GF)
        # the turn on the UPPER wire (the control) is not a gadget: the joiners just cancel
        assert np.allclose(CNOT @ kron(Zr(a), I2) @ CNOT, kron(Zr(a), I2))
        assert not np.allclose(CNOT @ kron(Zr(a), I2) @ CNOT, Gamma(a))
        # Result 17.2 on the blackboard: the hub picture, summed over bits.  Each green foot
        # copies its wire's bit onto its tie; the red hub sends x1, x2 to x1 xor x2 with 1/sqrt2;
        # the end gives e^{ia (x1 xor x2)}.
        hub = sum(np.exp(1j * a * (x1 ^ x2)) / R2 * np.outer(basis(x1, x2), basis(x1, x2))
                  for x1, x2 in product((0, 1), repeat=2))
        assert np.allclose(hub, Gamma(a) / R2)
        assert np.allclose(Gamma(a) / 2, 2 ** -0.5 * hub)                  # sandwich = 2^-1/2 hub

    # --- the headline through the gadgets
    assert np.allclose(U, Gamma(np.pi / 4) @ kron(I2, T) @ Gamma(np.pi / 4) @ kron(I2, T))
    assert np.allclose(U, kron(I2, S) @ Gamma(np.pi / 2))
    assert np.allclose(Gamma(np.pi / 2), kron(S, S) @ CZ)
    assert np.allclose(U, kron(S, Z) @ CZ) and np.allclose(U, np.diag([1, -1, 1j, 1j]))
    assert not np.allclose(U, kron(I2, Z))                                  # not the trap
    assert not np.allclose(U, kron(Z, S) @ CZ)                              # wires swapped: no

    # --- Achilles's four settings (the eighths that turn, as angles of U's diagonal)
    eighths_turned = {}
    for x1, x2 in product((0, 1), repeat=2):
        lower, n = x2, 0
        for _ in range(4):
            n += lower                                                      # the eighth sees it
            lower ^= x1                                                     # the joiner
        eighths_turned[(x1, x2)] = n
    assert eighths_turned == {(0, 0): 0, (0, 1): 4, (1, 0): 2, (1, 1): 2}  # none, half, quarter, quarter
    for (x1, x2), n in eighths_turned.items():
        assert np.isclose(U[2 * x1 + x2, 2 * x1 + x2], np.exp(1j * np.pi / 4 * n))
        # "A quarter whenever the upper is lit, a half when the lower is -- and when both are,
        # a half taken back": i^{x1} (-1)^{x2} (-1)^{x1 x2}
        assert np.isclose(np.exp(1j * np.pi / 4 * n), 1j ** x1 * (-1) ** x2 * (-1) ** (x1 * x2))

    # --- Result 17.7 (phase teleportation): one quarter in {1st, 3rd} and one in {2nd, 4th}
    def with_quarters(places):
        M = np.eye(4, dtype=complex)
        for k in range(1, 5):
            M = CNOT @ kron(I2, S if k in places else I2) @ M
        return M
    for p1, p2 in product((1, 3), (2, 4)):
        assert np.allclose(with_quarters({p1, p2}), U)
    assert np.allclose(with_quarters({1, 3}), kron(I2, Z))                 # the trap again
    assert not np.allclose(with_quarters({2, 4}), U)

    pictures()


# every frame's dump, in the dialogue's order (the log's, card 3e.6 item 6: r17/build/def/dumps_md.txt)
FRAME_DUMPS = (
    # Second block 1 frame 1: The card laid again
    'card aside=0 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @1.32,0; 2:ring green'
    ' 0 @3.66,0; 3:ring green 0 @6,0; 4:ring green 0 @8.34,0; 5:ring green 3 @0.15,1; 6:ring red 0 @1.32,1; '
    '7:ring green 3 @2.49,1; 8:ring red 0 @3.66,1; 9:ring green 3 @4.83,1; 10:ring red 0 @6,1; 11:ring green 3 '
    '@7.17,1; 12:ring red 0 @8.34,1; tie 1-6 []; tie 2-8 []; tie 3-10 []; tie 4-12 []; cord loose-loose [-0.5,0 '
    '9.5,0]; cord loose-loose [-0.5,1 9.5,1]',
    # Second block 1 frame 2: Greens met above
    'hang2:greens aside=0 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @2.49,0; '
    '2:ring green 0 @6,0; 3:ring green 0 @8.34,0; 4:ring green 3 @0.15,1; 5:ring red 0 @1.32,1; 6:ring green 3 '
    '@2.49,1; 7:ring red 0 @3.66,1; 8:ring green 3 @4.83,1; 9:ring red 0 @6,1; 10:ring green 3 @7.17,1; 11:ring '
    'red 0 @8.34,1; tie 1-5 []; tie 1-7 []; tie 2-9 []; tie 3-11 []; cord loose-loose [-0.5,0 9.5,0]; cord '
    'loose-loose [-0.5,1 9.5,1]',
    # Second block 1 frame 3: Bead sent to an end
    'hang2:apart aside=0 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @2.49,0; '
    '2:ring green 0 @6,0; 3:ring green 0 @8.34,0; 4:bob green 3 @2.49,0.5; 5:ring green 3 @0.15,1; 6:ring red 0 '
    '@1.32,1; 7:ring green 0 @2.49,1; 8:ring red 0 @3.66,1; 9:ring green 3 @4.83,1; 10:ring red 0 @6,1; 11:ring '
    'green 3 @7.17,1; 12:ring red 0 @8.34,1; tie 1-6 []; tie 1-8 []; tie 2-10 []; tie 3-12 []; pendant 7-4 []; '
    'cord loose-loose [-0.5,0 9.5,0]; cord loose-loose [-0.5,1 9.5,1]',
    # Second block 1 frame 4: The cradle, undone
    'hung2 aside=1 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @2.49,0; 2:ring '
    'green 0 @6,0; 3:ring green 0 @8.34,0; 4:bob green 3 @1.49,0.5; 5:hub red 0 @2.49,0.5; 6:ring green 3 '
    '@0.15,1; 7:ring green 0 @2.49,1; 8:ring green 3 @4.83,1; 9:ring red 0 @6,1; 10:ring green 3 @7.17,1; 11:ring'
    ' red 0 @8.34,1; tie 1-5 []; tie 2-9 []; tie 3-11 []; tie 5-7 []; pendant 5-4 []; cord loose-loose [-0.5,0 '
    '9.5,0]; cord loose-loose [-0.5,1 9.5,1]',
    # Second block 2 frame 1: A hub at each side
    'hung2,hung4 aside=2 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @2.49,0; '
    '2:ring green 0 @7.17,0; 3:bob green 3 @1.49,0.5; 4:hub red 0 @2.49,0.5; 5:hub red 0 @7.17,0.5; 6:bob green 3'
    ' @8.17,0.5; 7:ring green 3 @0.15,1; 8:ring green 0 @2.49,1; 9:ring green 3 @4.83,1; 10:ring green 0 @7.17,1;'
    ' tie 1-4 []; tie 2-5 []; tie 4-8 []; tie 5-10 []; pendant 4-3 []; pendant 5-6 []; cord loose-loose [-0.5,0 '
    '9.5,0]; cord loose-loose [-0.5,1 9.5,1]',
    # Second block 2 frame 2: The upper feet met
    'hung2,hung4,slid:upper aside=2 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 '
    '@4.83,0; 2:bob green 3 @1.49,0.5; 3:hub red 0 @2.49,0.5; 4:hub red 0 @7.17,0.5; 5:bob green 3 @8.17,0.5; '
    '6:ring green 3 @0.15,1; 7:ring green 0 @2.49,1; 8:ring green 3 @4.83,1; 9:ring green 0 @7.17,1; tie 1-3 []; '
    'tie 1-4 []; tie 3-7 []; tie 4-9 []; pendant 3-2 []; pendant 4-5 []; cord loose-loose [-0.5,0 9.5,0]; cord '
    'loose-loose [-0.5,1 9.5,1]',
    # Second block 2 frame 3: Lower feet slid together
    'hung2,hung4,slid aside=2 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @4.83,0; '
    '2:bob green 3 @1.49,0.5; 3:hub red 0 @2.49,0.5; 4:hub red 0 @7.17,0.5; 5:bob green 3 @8.17,0.5; 6:ring green'
    ' 6 @4.83,1; tie 1-3 []; tie 1-4 []; tie 3-6 []; tie 4-6 []; pendant 3-2 []; pendant 4-5 []; cord loose-loose'
    ' [-0.5,0 9.5,0]; cord loose-loose [-0.5,1 9.5,1]',
    # Second block 3 frame 1: The hubs merged
    'hung2,hung4,slid,onehub aside=3 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 '
    '@4.83,0; 2:hub red 0 @4.83,0.5; 3:bob green 6 @5.83,0.5; 4:ring green 6 @4.83,1; tie 1-2 []; tie 2-4 []; '
    'pendant 2-3 []; cord loose-loose [-0.5,0 9.5,0]; cord loose-loose [-0.5,1 9.5,1]',
    # Second block 4 frame 1: as Achilles expected
    'card aside=0 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @1.32,0; 2:ring green'
    ' 0 @3.66,0; 3:ring green 0 @6,0; 4:ring green 0 @8.34,0; 5:ring green 3 @0.15,1; 6:ring red 0 @1.32,1; '
    '7:ring green 3 @2.49,1; 8:ring red 0 @3.66,1; 9:ring green 3 @4.83,1; 10:ring red 0 @6,1; 11:ring green 3 '
    '@7.17,1; 12:ring red 0 @8.34,1; tie 1-6 []; tie 2-8 []; tie 3-10 []; tie 4-12 []; cord loose-loose [-0.5,0 '
    '9.5,0]; cord loose-loose [-0.5,1 9.5,1]',
    # Second block 4 frame 1: Feet slid left
    'hung2,hung4,slid,onehub,tie:feet aside=3 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring '
    'green 18 @2.49,0; 2:ring green 6 @4.83,0; 3:hub red 0 @4.83,0.5; 4:bob green 6 @5.83,0.5; 5:ring green 18 '
    '@2.49,1; 6:ring green 12 @4.83,1; tie 1-3 []; tie 3-5 []; pendant 3-4 []; cord loose-loose [-0.5,0 9.5,0]; '
    'cord loose-loose [-0.5,1 9.5,1]',
    # Second block 4 frame 2: Hub lifted, sleeve laid
    'hung2,hung4,slid,onehub,tie:lifted aside=3 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring'
    ' green 0 @2.49,0; 2:ring green 6 @4.83,0; 3:ring green 0 @2.49,1; 4:ring green 12 @4.83,1; tie 1-3 tubes1 '
    '[]; cord loose-loose [-0.5,0 9.5,0]; cord loose-loose [-0.5,1 9.5,1]',
    # Second block 4 frame 3: Nothing hanging
    'hung2,hung4,slid,onehub,tie aside=3 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green '
    '6 @4.83,0; 2:ring green 12 @4.83,1; tie 1-2 tubes1 []; cord loose-loose [-0.5,0 9.5,0]; cord loose-loose '
    '[-0.5,1 9.5,1]',
    # Second block 5 frame 1: What the pan made
    'upper=2 lower=4 sleeved',
)
SECOND_DUMPS = (
    "hung2 aside=1 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @2.49,0; 2:ring green 0 @6,0; "
    "3:ring green 0 @8.34,0; 4:bob green 3 @1.49,0.5; 5:hub red 0 @2.49,0.5; 6:ring green 3 @0.15,1; 7:ring green 0 @2.49,1; "
    "8:ring green 3 @4.83,1; 9:ring red 0 @6,1; 10:ring green 3 @7.17,1; 11:ring red 0 @8.34,1; tie 1-5 []; tie 2-9 []; "
    "tie 3-11 []; tie 5-7 []; pendant 5-4 []; cord loose-loose [-0.5,0 9.5,0]; cord loose-loose [-0.5,1 9.5,1]",
    "hung2,hung4,slid aside=2 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @4.83,0; "
    "2:bob green 3 @1.49,0.5; 3:hub red 0 @2.49,0.5; 4:hub red 0 @7.17,0.5; 5:bob green 3 @8.17,0.5; 6:ring green 6 @4.83,1; "
    "tie 1-3 []; tie 1-4 []; tie 3-6 []; tie 4-6 []; pendant 3-2 []; pendant 4-5 []; cord loose-loose [-0.5,0 9.5,0]; "
    "cord loose-loose [-0.5,1 9.5,1]",
    "hung2,hung4,slid,onehub,tie aside=3 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 6 @4.83,0; "
    "2:ring green 12 @4.83,1; tie 1-2 tubes1 []; cord loose-loose [-0.5,0 9.5,0]; cord loose-loose [-0.5,1 9.5,1]",
    "upper=2 lower=4 sleeved",
)


def hooks_graph(H, et=ET.HADAMARD):
    """The pair on the second hooks as a picture: each ring a green spider at its turn
    (eighths, so phase turn/4 in units of pi) on its own line, hung by a lead from the
    left-hand hook and to the right-hand one, the sleeved cord between them."""
    p = Two()
    a = p.on(0, VT.Z, Fraction(H.upper, 4))
    b = p.on(1, VT.Z, Fraction(H.lower, 4))
    p.g.add_edge((a, b), et)
    g = p.done()
    assert g.num_edges() == 5
    return g


def pieces_at(dump):
    """The pieces of a cloth dump, as (kind, colour, line, column) places (a hub's line 0.5)."""
    return {(kind, col, y, round(x, 3)) for kind, col, t, x, y in PC.parse_dump(dump)[0].values()}


def ghost_grid(q):
    """The grid place a piece's ghost names in the log ("the green ring (grid 1.32,0)")."""
    x, y = re.search(r"\(grid ([-\d.]+),([-\d.]+)\)", q.what).groups()
    return float(y), float(x)


def slide_of(a):
    y, x0, x1 = a.slide
    return y, round(x0, 3), round(x1, 3)


def bead_ghost_ok(q, before_dump, g):
    """A bead's ghost stands on the bead as it was: on the ring (or end) nearest it in the frame
    before, at the bead's radius round that ring's centre, at the ring's turn then (in
    twenty-fourths; an eighth is three)."""
    nodes = [v for v in PC.parse_dump(before_dump)[0].values()]
    d, (kind, col, t, x, y) = min(((np.hypot(g.XY((v[3], v[4]))[0] - q[0], g.XY((v[3], v[4]))[1] - q[1]), v) for v in nodes),
                                  key=lambda u: u[0])
    cx, cy = g.XY((x, y))
    a = np.radians(90 - 45 * t / 3)
    return d < g.RO and np.hypot(q[0] - cx - g.RM * np.cos(a), q[1] - cy - g.RM * np.sin(a)) < 1e-3


def check_marks(R):
    """A card step's marks, on the drawn extent.  Every hop runs from a ring of the state before to
    a ring of the step's state (or of the state before, a chained move), the way the ring went, on
    the line's outer side, bowed at least the Lua's hop floor; every piece that left its place has a
    ghost drawn there (or the log says what hides it), and a ring that left its place starts a hop;
    every ring that came to a new place ends one.  Returns (hops, ghosts at places)."""
    g = PC.G(PC.card_scene(R.after), PC.CARDOPTS, PC.CWHO)
    old, new = pieces_at(PC.card_dump(R.before)), pieces_at(PC.card_dump(R.after))
    ring = lambda s: {(y, x) for kind, col, y, x in s if kind == "ring"}
    hops = [a for a in R.arcs if getattr(a, "slide", None) is not None]
    for a in hops:
        y, x0, x1 = slide_of(a)
        assert a.kind == "slide" and (y, x0) in ring(old) and (y, x1) in ring(old) | ring(new) and x0 != x1, a.what
        assert np.sign(a.to[0] - a.frm[0]) == np.sign(x1 - x0)
        assert abs(a.frm[0] - g.X(x0)) < abs(a.frm[0] - g.X(x1)) and abs(a.to[0] - g.X(x1)) < abs(a.to[0] - g.X(x0))
        assert (min(p[1] for p in a.body) > g.Y(0)) if y == 0 else (max(p[1] for p in a.body) < g.Y(1)), a.what
        assert a.minsag == PC.HOPSAG and a.sag >= PC.HOPSAG - 1e-9, (a.what, a.sag)
    places = {ghost_grid(q) for q in R.ghosts if not q.bead}
    for q in R.ghosts:
        if not q.bead:
            y, x = ghost_grid(q)
            assert np.allclose(q[:2], g.XY((x, y)), atol=1e-3), q.what
    hidden = {(float(y), float(x)) for x, y in re.findall(r"no ghost of [^(]*\(grid ([-\d.]+),([-\d.]+)\): the piece lying "
                                                          r"on its place now hides it", " ".join(R.notes))}
    gone = {(y, x) for kind, col, y, x in old - new}                     # a piece no longer where it was
    assert gone - hidden == places - hidden and hidden <= gone, (gone, places, hidden)
    for y, x in ring(old) - ring(new):
        assert any(slide_of(a)[:2] == (y, x) for a in hops) or not hops, (y, x)
    for y, x in ring(new) - ring(old):
        assert any(slide_of(a)[0] == y and slide_of(a)[2] == x for a in hops) or not hops, (y, x)
    return hops, places


def pictures():
    """The Second Challenge's prop pictures (identity card, sections 4 and 3e), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch17.py, and held against
    this file's blackboard (graph_of: a structural reading of each cloth dump, not a proof).  Round
    4 draws the cooking move by move: every frame, times what its counters stand for, is the recipe,
    U/4, exactly; and Achilles's prediction (the joiners cancel) is drawn pale: I (x) Z, which is not
    the recipe."""
    calls = DIALOGUE.check_layout("Second", sizes=[1, 1, 1, 2, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("cardclothsteps", False, "",
         ("upper greens together above the second eighth; second eighth slid apart; cradle undone at the second eighth",),
         "The card laid again; Greens met above; Bead sent to an end; The cradle, undone"),
        ("cardclothsteps", False, "after=second eighth hung, fourth eighth hung",
         ("upper feet slid together; lower line slid together",), "A hub at each side; The upper feet met; Lower feet slid together"),
        ("cardcloth", False, "second eighth hung, fourth eighth hung, slid together; step=one hub", (), "The hubs merged"),
        ("cardcloth", False, "expects=the joiners cancel", (), "as Achilles expected"),
        ("cardclothsteps", False, "after=second eighth hung, fourth eighth hung, slid together, one hub; from=off",
         ("the feet slid apart; the hub lifted away; into their neighbours",), "Feet slid left; Hub lifted, sleeve laid; Nothing hanging"),
        ("secondhooks", True, "a quarter round, at a half", (), "What the pan made")]
    PC.check_captions(DIALOGUE, "Second", calls)
    PC.check_section(DIALOGUE, "Second", calls)
    assert PC.check_anchors(DIALOGUE, "Second") == (10, 18)               # P2.1, P2.2, B2.2a, P2.3, P2.4
    frames = [PC.frame_chain(c) for c in calls]
    assert [len(f) for f in frames] == [4, 3, 1, 1, 3, 1]                 # card 3e.6: 5 blocks, 13 frames
    assert tuple(R.head for f in frames for R in f) == FRAME_DUMPS        # the log's dump, frame by frame
    b21, b22, (b22a,), (pale,), b23, (p24,) = frames
    p21, p22, p23 = b21[-1], b22[-1], b23[-1]
    pics = [p21, p22, p23, p24]
    # every strip's last frame is round 3's picture of its block (round 3's assertions carry over)
    for p, br in zip((p21, p22, p23), ("second eighth hung", "second eighth hung, fourth eighth hung, slid together", PC.COOKED)):
        assert p.dump == PC.cardcloth(br, False).dump
    assert tuple(p.dump for p in pics) == SECOND_DUMPS
    assert [p.frame for p in pics] == [1.37, 1.37, 1.37, 1.5]          # the card's frame as in I (band included)
    assert {R.frame for f in frames[:5] for R in f} == {1.37}            # B2.3's two calls: one height
    assert [p.state.aside for p in (p21, p22, p23)] == [1, 2, 3]
    assert [len(p.shown.counters) for p in (p21, p22, p23)] == [1, 2, 3]

    # the counters' number, computed from this file's blackboard, not read: a sandwich is
    # 2^-1/2 times its hub picture; two hubs on the same feet 2^-1/2 times one hub
    a, b = 0.37, -1.21
    hub = lambda t: Gamma(t) / R2                                     # the hub picture's value
    per_hang = (Gamma(a) / 2)[1, 1] / hub(a)[1, 1]                     # sandwich / hub
    per_merge = (hub(a) @ hub(b))[1, 1] / hub(a + b)[1, 1]             # two hubs / one hub
    assert np.isclose(per_hang, 2 ** -0.5) and np.isclose(per_merge, 2 ** -0.5)
    number = {0: 1, 1: per_hang, 2: per_hang ** 2, 3: per_hang ** 2 * per_merge}
    assert np.isclose(number[3], 2 ** -1.5)
    # the mathematics: each cloth picture times what its counters stand for is the recipe, U/4
    for p in (p21, p22, p23):
        g = PC.graph_of(p.dump)
        assert np.allclose(number[p.state.aside] * matrix_of(g), U / 4)
        assert not np.allclose(matrix_of(g), U / 4)                    # the counters are not idle
    cooked = PC.graph_of(p23.dump)
    assert np.allclose(matrix_of(cooked), U / R2)                      # P2.3 alone: the cooked picture
    assert np.allclose(matrix_of(cooked), kron(S, Z) @ CZ / R2)
    # the plausible mix-ups: the tie unsleeved, or the turns on the wrong lines
    assert not np.allclose(matrix_of(PC.graph_of(p23.dump.replace("tie 1-2 tubes1", "tie 1-2"))), U / R2)
    swapped = p23.dump.replace("1:ring green 6", "1:ring green X").replace("2:ring green 12", "2:ring green 6")
    swapped = swapped.replace("1:ring green X", "1:ring green 12")
    assert not np.allclose(matrix_of(PC.graph_of(swapped)), U / R2)
    assert np.allclose(matrix_of(PC.graph_of(swapped)), kron(Z, S) @ CZ / R2)

    # card 3e.6, item 7: every frame of B2.1, B2.2, B2.2a and B2.3's strip, times 2^(-aside/2), is
    # exactly U/4 -- each move a rewrite with its number written (a counter by the teapot)
    cooking = b21 + b22 + [b22a] + b23
    assert [R.drawn.aside for R in cooking] == [0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3]
    for R in cooking:
        g = PC.graph_of(R.dump)
        assert np.allclose(number[R.drawn.aside] * matrix_of(g), U / 4), R.step
        assert not np.allclose(matrix_of(g), kron(I2, Z) / 4)
        # the mix-up: an end's turn lost on the way (its bead left at rest) is not the recipe
        n = PC.parse_dump(R.dump)[0]
        k = next((k for k, v in n.items() if v[0] == "bob"), None)
        if k is not None:
            lost = re.sub(r"\b%d:bob green \d+ " % k, "%d:bob green 0 " % k, R.dump)
            assert not np.allclose(number[R.drawn.aside] * matrix_of(PC.graph_of(lost)), U / 4), R.step
    # every bead's ghost stands on the bead as it was, in the frame before (its ring's turn then)
    beads = 0
    for P, R in zip(cooking, cooking[1:]):
        if R.step is None:                                               # B2.2's before-frame: unmarked
            assert R.ghosts == []
            continue
        assert R.before == P.drawn, R.step
        g_ = PC.G(PC.card_scene(R.after), PC.CARDOPTS, PC.CWHO)
        for q in (q for q in R.ghosts if q.bead):
            assert bead_ghost_ok(q, P.dump, g_), (R.step, q.what)
            beads += 1
    assert beads == 3                                                    # B2.1 f3's eighth; B2.3 f1's two rings
    # a move that sets no counter aside leaves the picture's value exactly as it was; a counter set
    # aside is the move's number, 2^-1/2
    for P, R in zip(cooking, cooking[1:]):
        if R.before == P.drawn and R.drawn.aside == P.drawn.aside:
            assert same_picture(PC.graph_of(R.dump), PC.graph_of(P.dump)), R.step
    # the strips take up where the block before stops: B2.2 from the two hubs hung, B2.2a from the
    # feet slid together, B2.3's strip from the hubs made one
    assert b22[0].before == PC.card_state("second eighth hung, fourth eighth hung")
    assert b22a.before == p22.drawn and b23[0].before == b22a.drawn
    assert b21[0].before == PC.card_state("")                            # B2.1's before-frame: the card laid again
    assert b21[0].dump == PC.cardcloth("", False).dump and not b21[0].marked

    # B2.1, the second eighth hung: the greens met (two hops), the eighth slid apart (its bead's
    # ghost and the end's arc: L5), the cradle undone (the red rings' ghosts; the end hidden under
    # the hub that takes its place; the counter from her work-bag, unmarked)
    assert [R.step for R in b21] == [None, "upper greens together above second eighth", "second eighth slid apart",
                                     "cradle undone at second eighth"]
    hops, places = check_marks(b21[1])
    assert [slide_of(a) for a in hops] == [(0, 1.32, 2.49), (0, 3.66, 2.49)] and places == {(0, 1.32), (0, 3.66)}
    f3 = b21[2]
    (arc,), (bead,) = f3.arcs, f3.ghosts
    assert getattr(arc, "slide", None) is None and arc.kind == "slide" and bead.bead and bead.what == "the second eighth"
    assert np.isclose(arc.len, 0.152, atol=5e-4) and arc.sag >= PC.MINSAG - 1e-9 and np.isclose(arc.sag, 0.030, atol=5e-4)
    assert arc.fit[0] >= PC.MINCLEAR - 1e-9                              # clear of the net (L5: 0.020 cords, 0.022 rings)
    assert re.match(r"cords and ties 0\.020, tubes -, rings 0\.022", arc.fit[1]) and arc.fit[2] > 0
    g3 = PC.G(PC.card_scene(f3.after), PC.CARDOPTS, PC.CWHO)
    ring_at, end_at = g3.XY((2.49, 1)), g3.XY((2.49, 0.5))
    assert abs(arc.frm[1] - ring_at[1]) < abs(arc.frm[1] - end_at[1]) and arc.to[1] > arc.frm[1]   # up from the ring
    assert arc.side == "on the pendant's right, under the end" and min(p[0] for p in arc.body) > ring_at[0]   # the right side
    # E3: "The end lies up from the ring, towards the upper line, between the short strings."
    (bob,) = [v for v in PC.parse_dump(f3.dump)[0].values() if v[0] == "bob"]
    assert bob[3] == 2.49 and 0 < bob[4] < 1                             # above its ring, below the upper line
    ties = PC.parse_dump(f3.dump)[2]
    n3 = PC.parse_dump(f3.dump)[0]
    short = [(n3[a_][3], n3[b_][3]) for kind, a_, b_, k in ties if n3[a_][3] == 2.49 and n3[b_][1] == "red"]
    assert sorted(x for _, x in short) == [1.32, 3.66] and 1.32 < bob[3] < 3.66        # between the short strings
    for path in (DIALOGUE.dialog, DIALOGUE.raw):                      # in the direction before B2.1's
        text, sec = DIALOGUE.read(path), DIALOGUE.section("Second", path)
        assert text.count(PC.E3) == 1 and "and the ring left on the line is at rest. " + PC.E3 in sec
        assert sec.index(PC.E3) < sec.index(DIALOGUE.anchors("Second", path)[0]) < sec.index(calls[0].text)
    hops, places = check_marks(b21[3])
    assert hops == [] and b21[3].arcs == [] and places == {(1, 1.32), (1, 3.66)}
    assert any("no mark for the counter set aside" in n for n in b21[3].notes)

    # B2.2: from the two hubs, the upper feet met (two hops), then the lower line slid together
    # (three hops, the first eighth into the first foot, those into the third eighth, the second
    # foot brought along)
    assert [R.step for R in b22] == [None, "upper feet slid together", "lower line slid together"]
    assert not b22[0].marked and b22[0].arcs == []
    hops, _ = check_marks(b22[1])
    assert [slide_of(a) for a in hops] == [(0, 2.49, 4.83), (0, 7.17, 4.83)]
    hops, _ = check_marks(b22[2])
    assert [slide_of(a) for a in hops] == [(1, 0.15, 2.49), (1, 2.49, 4.83), (1, 7.17, 4.83)]
    # B2.2a: the two hubs made one -- each hub's and each end's ghost where it hung, no arc
    hops, places = check_marks(b22a)
    assert hops == [] and places == {(0.5, 2.49), (0.5, 1.49), (0.5, 7.17), (0.5, 8.17)}
    assert b22a.marked and b22a.step == "one hub" and b22a.drawn.aside == 3
    # B2.3's strip: the feet slid apart (a quarter back on each: two hops off the rings, the rings'
    # beads ghosted), the hub lifted (its ghost and its end's), the rings at rest slid in (two hops)
    assert [R.step for R in b23] == ["feet slid apart", "hub lifted away", "into their neighbours"]
    hops, places = check_marks(b23[0])
    assert [slide_of(a) for a in hops] == [(0, 4.83, 2.49), (1, 4.83, 2.49)] and places == set()
    assert [q.what for q in b23[0].ghosts] == ["the upper ring", "the lower ring"] and all(q.bead for q in b23[0].ghosts)
    feet = {(v[4], v[2]) for v in PC.parse_dump(b23[0].dump)[0].values() if v[0] == "ring" and v[3] == 2.49}
    assert feet == {(0, 18), (1, 18)}                                    # each foot a quarter back (18/24 = -6/24)
    hops, places = check_marks(b23[1])
    assert hops == [] and places == {(0.5, 4.83), (0.5, 5.83)}
    hops, places = check_marks(b23[2])
    assert [slide_of(a) for a in hops] == [(0, 2.49, 4.83), (1, 2.49, 4.83)] and places == {(0, 2.49), (1, 2.49)}
    # every arc of the cooking clear of the net; every hop bowed 0.058 (L4)
    assert all(a.fit[0] >= PC.MINCLEAR - 1e-9 for R in cooking for a in R.arcs)
    assert {round(a.sag, 3) for R in cooking for a in R.arcs if getattr(a, "slide", None) is not None} == {0.058}
    assert all(R.marked == (R.step is not None) for R in cooking) and not any(R.keyed for R in cooking)

    # the pale frame: the joiners cancel, Achilles's wrong rule, applied by the model -- every
    # joiner taken off (in pairs: four), the four eighths summed on the lower line into one ring at
    # a half, the upper line bare, no tie, no counter.  The Lua draws the guess over the card as
    # laid (its log's dump is the card's), so the guess is the model's, never read from the dump.
    assert PC.is_prediction(calls[3]) and pale.wrong is not None and not pale.marked and pale.step is None
    assert pale.head == FRAME_DUMPS[0] == b21[0].head                    # the card, as laid
    W, ring, joiners, eighths = PC.card_wrong(PC.card_state(""))
    assert joiners == 4 and joiners % 2 == 0 and eighths == 4
    assert (ring["c"], ring["t"], ring["x"], ring["y"]) == ("green", 12, PC.CX[4], 1)   # a half (24ths), mid-card, lower line
    assert pale.drawn[0] == W and pale.dump == PC.kit_dump(W, PC.CWHO)
    pd = "the guess | " + pale.dump                                      # the kit's dump of the guessed scene
    guess = PC.graph_of(pd)
    G = matrix_of(guess)
    assert np.allclose(G, kron(I2, Z))                                   # the trap, I (x) Z (the card's 1/4 dropped)
    assert np.allclose(G, np.linalg.matrix_power(kron(I2, T), 4))        # the eighths alone, the joiners gone
    # wrong about the value, not only the drawing: it is not the recipe, at any normalization
    for scale in (1, 1 / 4, 1 / R2, 2 ** -1.5):
        assert not np.allclose(scale * G, U / 4) and not np.allclose(scale * G, U)
    for x2 in (0, 1):                                                    # they agree exactly with the upper dark,
        assert np.allclose(U @ basis(0, x2), G @ basis(0, x2))
        assert not np.allclose(U @ basis(1, x2), G @ basis(1, x2))       # and differ with it lit
    # its telling parts are what it lacks against the true state beside it: the upper ring and the
    # sleeved tie of "Nothing hanging"
    up = lambda d: [v for v in PC.parse_dump(d)[0].values() if v[4] == 0]
    assert up(pd) == [] and PC.parse_dump(pd)[2] == [] and len(PC.parse_dump(pd)[0]) == 1
    assert len(up(p23.dump)) == 1 and [k for _, _, _, k in PC.parse_dump(p23.dump)[2]] == [1]
    assert not same_picture(guess, cooked)
    # drawn: the cloth under a white veil, the ring at a half at full strength through a hole in it,
    # and a touch heavier (L8): a second edge just outside its own, so that it reads at least 0.15 cm
    # across under the veil (0.169)
    gp = PC.G(W, PC.CARDOPTS, PC.CWHO)
    cx, cy = gp.XY((PC.CX[4], 1))
    veil, edge = pale.own[1], pale.own[4]
    hole = re.search(r"\((-?[\d.]+),(-?[\d.]+)\) circle \(([\d.]+)\);$", veil)
    assert np.allclose([float(v) for v in hole.groups()[:2]], (cx, cy), atol=1e-3) and float(hole.group(3)) > gp.RO
    assert pale.own[2].startswith("\\fill[white, opacity=") and pale.own[0] == "\\begin{scope}[even odd rule]"
    assert PC.PALE == 0.5 and "white, opacity=0.500" in pale.own[2]
    m = re.fullmatch(r"\\draw\[propclothgreenedge, line width=([\d.]+)pt\] \((-?[\d.]+),(-?[\d.]+)\) circle \(([\d.]+)\);", edge)
    assert m and float(m.group(1)) == PC.PALEEDGE and np.isclose(float(m.group(4)), gp.RO + PC.PALEOUT, atol=1e-3)
    across = 2 * float(m.group(4)) + float(m.group(1)) * 2.54 / 72.27
    assert across >= 0.15 and np.isclose(across, 0.169, atol=5e-4), across
    # the wrong rule's comment in the Lua, word for word as the card copies it (section 3e.1), and
    # every phrase it quotes spoken in both dialogue files
    card = (pathlib.Path(DIALOGUE.dialog).parent / "identity.md").read_text(encoding="utf-8")
    assert ("```\n" + PC.rule_comment().rstrip("\n") + "\n```") in card
    assert PC.rule_comment().startswith("-- " + PC.RULE_COMMENT)
    quotes = PC.comment_quotes()
    assert quotes[0] == "the joiners cancel" and len(quotes) >= 6
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        text = pathlib.Path(path).read_text(encoding="utf-8")
        for q in quotes[1:]:
            assert q in text or q.replace(" — ", " -- ") in text, (path, q)
    assert "the joiners, in pairs, undo each other" in PC.comment_prose()
    # a prediction frame is never keyed, carries no marks, and has no step before it or in it
    assert PC.faults(PC.run_frame, "card", "expects=the joiners cancel", "", True) == ["expects-keyed"]
    assert PC.faults(PC.run_frame, "card", "reds made green; expects=the joiners cancel", "", False) == ["card-expects-bare"]
    PC.TEX_ERRORS.clear()

    # P2.4: the pan's answer on the hooks is the cloth's answer, exactly -- the same two rings,
    # the same turns, the same sleeved tie
    assert p24.state.upper == 2 and p24.state.lower == 4 and not p24.state.empty
    n23 = PC.parse_dump(p23.dump)[0]
    assert (n23[1][2], n23[2][2]) == (3 * p24.state.upper, 3 * p24.state.lower)   # twenty-fourths
    assert same_picture(hooks_graph(p24.state), cooked)
    assert not same_picture(hooks_graph(p24.state, ET.SIMPLE), cooked)
    assert not same_picture(hooks_graph(PC.hooks_state("at a half, a quarter round")), cooked)
    hv = p24.shown
    assert len(hv.rings) == 2 and len(hv.cords) == 5 and hv.sleeve == (0, (PC.RUY - PC.HRO + PC.RLY + PC.HRO) / 2, 90)
    for (x, y), (rx, ry, t) in zip(hv.beads, hv.rings):                # each bead at its ring's turn, clockwise from the top
        assert np.isclose(np.arctan2(y - ry, x - rx), np.pi / 2 - np.pi / 4 * t)
    assert not p24.marked and p24.arcs == [] and p24.ghosts == []      # the press draws no marks (Q-G: plain)

    # the captions' truth: a hub stands; one upper and one lower ring with both hubs tied to
    # each; no hub and no bob left; a pair on the hooks
    kinds = lambda p: [k for k, c, t, x, y in PC.parse_dump(p.dump)[0].values()]
    assert "hub" in kinds(p21)
    n22, _, t22, _ = PC.parse_dump(p22.dump)
    up_ = [k for k, v in n22.items() if v[0] == "ring" and v[4] == 0]
    lo = [k for k, v in n22.items() if v[0] == "ring" and v[4] == 1]
    hubs = [k for k, v in n22.items() if v[0] == "hub"]
    assert len(up_) == len(lo) == 1 and len(hubs) == 2
    assert {(a_, b_) for kind, a_, b_, _ in t22 if kind == "tie"} == {(up_[0], h) for h in hubs} | {(h, lo[0]) for h in hubs}
    assert "hub" not in kinds(p23) and "bob" not in kinds(p23)
    assert kinds(b22a).count("hub") == 1 and kinds(b22[0]).count("hub") == 2          # "The hubs merged"
    assert "hub" not in kinds(b23[1]) and any(k == 1 for _, _, _, k in PC.parse_dump(b23[1].dump)[2])  # sleeve laid
    assert PC.hooks_dump(p24.state) != "empty"

    # fixed parts, frame to frame: the first hub and its end stay where they hung from P2.1 through
    # B2.2; the other pieces of each frame a step does not move keep their places
    hung = lambda p: sorted((k, x, y) for k, c, t, x, y in PC.parse_dump(p.dump)[0].values() if k in ("hub", "bob") and x < 4)
    assert hung(p21) == [("bob", 1.49, 0.5), ("hub", 2.49, 0.5)]
    for R in b22:
        assert hung(R) == hung(p21), R.step
    for P, R in zip(cooking, cooking[1:]):
        if R.before == P.drawn:
            moved = {slide_of(a)[:2] for a in R.arcs if getattr(a, "slide", None) is not None} | {ghost_grid(q) for q in R.ghosts if not q.bead}
            keep = {(y, x) for k, c, y, x in pieces_at(P.dump)} - moved
            assert keep <= {(y, x) for k, c, y, x in pieces_at(R.dump)}, R.step
    # the counters lie in the band below the linen, clear of the linen's shadow and inside the frame
    for p in cooking:
        v = p.shown
        g = PC.C._geometry(p.cloth, p.options)
        s = g["P"] / 0.30
        L = v.linen
        shadow = (L[0] + PC.LINEN_SHADOW * s, L[1] - PC.LINEN_SHADOW * s, L[2] + PC.LINEN_SHADOW * s, L[3] - PC.LINEN_SHADOW * s)
        for c in v.counters:
            box = PC.counter_box(*c)
            assert PC.box_gap(box, shadow) >= 0.02 and PC.box_gap(box, L) >= 0.02
            assert box[1] - (v.fbot - PC.ASIDE) >= 0.025 and 1.25 - box[2] >= 0.025
        assert v.counters == p23.shown.counters[:len(v.counters)]      # each counter laid on the others, where it was

    # the keyed words (P2.4): second hooks, sleeve; set as the compiled pictures set them, their
    # letters clear of every part drawn (second hooks stands over the frame's top, by design)
    assert p24.labels == ("second hooks", "sleeve") and all(R.labels == () for f in frames[:5] for R in f)
    PC.check_compiled_boxes(p24.picture)
    least = PC.check_keyed_words(p24.picture)
    assert least["sleeve"] >= 0.02 and least["second hooks"] >= 0.1
    # the Lua's refusals: the two ways through do not mix; one hub before slid together
    PC.TEX_ERRORS.clear()
    PC.card_state("second eighth hung, reds made green")
    PC.card_state("second eighth hung, one hub")
    PC.hooks_state("a quarter round")
    assert [k for k, _ in PC.TEX_ERRORS] == ["card-twopaths-first", "card-onehub", "hooks-pair"]
    PC.TEX_ERRORS.clear()
    check_faults()


# Every tex.error site of ch17.lua (PC.TEX_SITES, in the Lua's file order), each tripped by a call of
# its own: (site, macro, keyed, bracket, state[, the sites raised]) -- a strip's case is (site,
# "steps", keyed, kind, state, options, steps, captions[, the sites raised]).  Two sites are reached
# only with a second fault each, named: expects=...; marks=on raises marks=on's own fault too, and an
# unknown strip has no captions to count.
B = "second eighth hung, fourth eighth hung"
PAN, NET, V, RD = "the pan's answer", "the second net", "the V", "lit, dark, lit"
FAULT_CASES = [
    # \onewaytin (terr)
    ("tin-lifted", "tin", False, "first lit, first dark", ""),
    ("tin-order", "tin", False, "second lit", ""),
    ("tin-spent", "tin", False, "first lit, first dial the other way", ""),
    ("tin-step", "tin", False, "frobbed", ""),
    # \secondhooks (herr)
    ("hooks-pair", "hooks", False, "a quarter round", ""),
    ("hooks-turn", "hooks", False, "a quarter round, sideways", ""),
    # \cardcloth (cerr): card_part's, then the state's
    ("card-twopaths-part", "card", False, "second eighth hung, first and second into one", ""),
    ("card-up-green", "card", False, "first and second into one", ""),
    ("card-up-done", "card", False, "reds made green, upper rings slid together, first and second into one", ""),
    ("card-up-turn", "card", False, "reds made green, third into those", ""),
    ("card-h-done", "card", False, "second eighth hung, upper greens together above the second eighth", ""),
    ("card-h-turn", "card", False, "second eighth slid apart", ""),
    ("card-s-hang", "card", False, "second eighth hung, upper feet slid together", ""),
    ("card-s-done", "card", False, B + ", slid together, upper feet slid together", ""),
    ("card-s-turn", "card", False, B + ", lower line slid together", ""),
    ("card-t-hub", "card", False, B + ", the feet slid apart", ""),
    ("card-t-turn", "card", False, B + ", slid together, one hub, the hub lifted away", ""),
    ("card-open", "card", False, "upper greens together above the second eighth, fourth eighth hung", ""),
    ("card-twopaths-first", "card", False, "second eighth hung, reds made green", ""),
    ("card-twice", "card", False, "reds made green, reds made green", ""),
    ("card-after", "card", False, "upper rings slid together", ""),
    ("card-twopaths-hung", "card", False, "reds made green, second eighth hung", ""),
    ("card-hung-twice", "card", False, "second eighth hung, second eighth hung", ""),
    ("card-twopaths-slid", "card", False, "reds made green, slid together", ""),
    ("card-slid-hang", "card", False, "slid together", ""),
    ("card-slid-twice", "card", False, B + ", slid together, slid together", ""),
    ("card-onehub", "card", False, "one hub", ""),
    ("card-tie", "card", False, "sleeved tie", ""),
    ("card-step", "card", False, "frobbed", ""),
    # \tincloth (rerr)
    ("row-readings", "row", False, "", "lit, dark"),
    ("row-reading", "row", False, "", "lit, dark, grey"),
    ("row-open", "row", False, "first half off its end, slid in", RD),
    ("row-off-slid", "row", False, "slid in, first half off its end", RD),
    ("row-off-dark", "row", False, "second half off its end", RD),
    ("row-off-sent", "row", False, "first half sent on, first half off its end", RD),
    ("row-none", "row", False, "through the tube", RD),
    ("row-slid", "row", False, "slid in, first half sent on", RD),
    ("row-dark", "row", False, "second half sent on", RD),
    ("row-sent", "row", False, "first half sent on, first half sent on", RD),
    ("row-slid-twice", "row", False, "slid in, slid in", RD),
    ("row-step", "row", False, "frobbed", RD),
    # \outcloth (oerr)
    ("out-start", "out", False, "", "sideways"),
    ("out-open", "out", False, "upper ring slid apart, the joiner laid at the ends", PAN),
    ("out-part-pan", "out", False, "upper ring slid apart", NET),
    ("out-part-done", "out", False, "slid apart, upper ring slid apart", PAN),
    ("out-apart-pan", "out", False, "slid apart", V),
    ("out-apart-twice", "out", False, "slid apart, slid apart", PAN),
    ("out-joiner-net", "out", False, "the joiner laid at the ends", PAN),
    ("out-joiner-twice", "out", False, "the joiner laid at the ends, the joiner laid at the ends", NET),
    ("out-step", "out", False, "frobbed", PAN),
    # round 4 (xerr): the bracket and the step
    ("opt-unknown", "card", False, "step=reds made green; colour=red", ""),
    ("opt-twice", "card", False, "step=reds made green; step=reds made green", ""),
    ("opt-bare-two", "card", False, "reds made green; reds made green; marks=off", ""),
    ("opt-onoff", "card", False, "step=reds made green; marks=maybe", ""),
    ("marks-no-step", "card", False, "reds made green; marks=on", ""),
    ("expects-keyed", "card", True, "expects=the joiners cancel", ""),
    # expects-marked fires only past another fault: without a step, marks-no-step; with one,
    # card-expects-step (the guess is foreseen from the card as laid)
    ("expects-marked", "card", False, "expects=the joiners cancel; marks=on", "", ("marks-no-step", "expects-marked")),
    ("marks-keyed", "tin", True, "step=first lit", ""),
    ("step-commas", "card", False, "step=reds made green, one hub", ""),
    ("step-empty", "card", False, "reds made green; step=", ""),
    # round 4 (xerr): the five macros
    ("tin-expects", "tin", False, "expects=coin", ""),
    ("hooks-pressed", "hooks", False, "a quarter round, a half; step=pressed", ""),
    ("hooks-lifted", "hooks", False, "step=lifted off", ""),
    ("hooks-expects", "hooks", False, "expects=less", ""),
    ("hooks-step-empty", "hooks", False, "step=", ""),
    ("hooks-step-unknown", "hooks", False, "step=jammed", ""),
    ("card-partkey-old", "card", True, "reds made green, first and second into one", ""),
    ("card-partkey", "card", True, "reds made green; step=first and second into one; marks=off", ""),
    ("card-expects-rule", "card", False, "expects=they add up", ""),
    ("card-expects-step", "card", False, "expects=the joiners cancel; step=reds made green", ""),
    ("card-expects-bare", "card", False, "reds made green; expects=the joiners cancel", ""),
    ("card-marks-tangle", "card", False, "reds made green, upper rings slid together; step=small rings put on; marks=on", ""),
    ("row-halves-keyed-old", "row", True, "first half off its end", RD),
    ("row-expects", "row", False, "expects=red", RD),
    ("row-halves-keyed", "row", True, "step=first half off its end", RD),
    ("out-expects", "out", False, "expects=card", V),
    # round 4 (xerr): the strips
    # strip-unknown fires only past strip-captions: an unknown strip has no frames, and any
    # captions (even none: one empty item) are one too many
    ("strip-unknown", "steps", False, "nope", "", "", "x", "a", ("strip-unknown", "strip-captions")),
    ("strip-bare", "steps", False, "card", "", "loud", "reds made green", "a; b"),
    ("strip-star-from", "steps", True, "card", "", "from=off", "reds made green", "a"),
    ("strip-none", "steps", False, "card", "", "", "", "a"),
    ("strip-commas", "steps", False, "card", "", "", "reds made green, one hub", "a; b"),
    ("strip-long", "steps", False, "card", "", "", "reds made green; first and second into one; third into those; "
     "fourth into those", "a; b; c; d; e"),
    ("strip-captions", "steps", False, "card", "", "", "reds made green", "a"),
]


def check_faults():
    """Every fault site of the chapter's Lua raised by its own call (the strip faults by the strip
    alone, as TeX raises them before any frame runs), each with messages of its own; the 83 sites
    are the Lua's, all of them; a refused step draws no marks; the ';' bracket is order-free, each
    order run."""
    seen = {}
    for case in FAULT_CASES:
        site = case[0]
        if case[1] == "steps":
            _, _, keyed, kind, scene, opt, stepspec, caps = case[:8]
            want = case[8] if len(case) > 8 else (site,)
            fn, args = PC.steps, (kind, keyed, scene, opt, stepspec, caps)
        else:
            _, kind, keyed, br, state = case[:5]
            want = case[5] if len(case) > 5 else (site,)
            fn, args = PC.run_frame, (kind, br, state, keyed)
        assert tuple(dict.fromkeys(PC.faults(fn, *args))) == tuple(want), (site, PC.TEX_ERRORS)
        msgs = {m for s_, m in zip(PC.faults(fn, *args), PC.messages(fn, *args)) if s_ == site}
        assert msgs, site
        for m in msgs:
            assert m not in seen, (site, seen.get(m))                    # no two sites share a message
            seen[m] = site
    # the mutual-exclusion checks are made on the whole parse: every order of a refused bracket's
    # items raises the same faults (and a strip's options, likewise)
    orders = 0
    for case in FAULT_CASES:
        if case[1] == "steps":
            _, _, keyed, kind, scene, opt, stepspec, caps = case[:8]
            items, fn = opt.split("; "), lambda br: PC.faults(PC.steps, kind, keyed, scene, br, stepspec, caps)
        else:
            _, kind, keyed, br, state = case[:5]
            items, fn = br.split("; "), lambda br: PC.faults(PC.run_frame, kind, br, state, keyed)
        if len(items) < 2:
            continue
        want = sorted(fn("; ".join(items)))
        for order in itertools.permutations(items):
            assert sorted(fn("; ".join(order))) == want, (case[0], order)
            orders += 1
    assert orders >= 20
    # a step the Lua refuses draws no marks, in each family's step= form; the same kind of step
    # taken draws them
    refused = 0
    for kind, br, state, site in (("tin", "step=second lit", "", "tin-order"),
                                  ("tin", "first lit; step=first dark", "", "tin-lifted"),
                                  ("hooks", "step=lifted off", "", "hooks-lifted"),
                                  ("hooks", "a quarter round, a half; step=pressed", "", "hooks-pressed"),
                                  ("card", "step=upper rings slid together", "", "card-after"),
                                  ("card", "second eighth hung; step=reds made green", "", "card-twopaths-first"),
                                  ("card", "second eighth hung; step=upper feet slid together", "", "card-s-hang"),
                                  ("row", "step=second half sent on", RD, "row-dark"),
                                  ("row", "step=the second half off its end", RD, "row-off-dark"),
                                  ("out", "step=the joiner laid at the ends", PAN, "out-joiner-net"),
                                  ("out", "step=upper ring slid apart", NET, "out-part-pan")):
        PC.TEX_ERRORS.clear()
        p = PC.run_frame(kind, br, state, False)
        assert [s_ for s_, _ in PC.TEX_ERRORS] == [site] and p.refused, (kind, br, PC.TEX_ERRORS)
        assert p.mark_tikz == [] and p.ghosts == [] and p.arcs == [] and p.halves == [], (kind, br)
        assert "no marks: the step was refused" in p.notes, (kind, br)
        refused += 1
    assert refused == 11
    for kind, br, state in (("tin", "step=first lit", ""), ("card", "step=upper greens together above the second eighth", ""),
                            ("row", "step=the first half off its end", RD), ("out", "step=upper ring slid apart", PAN)):
        PC.TEX_ERRORS.clear()
        p = PC.run_frame(kind, br, state, False)
        assert p.mark_tikz and p.arcs and not PC.TEX_ERRORS, kind
    # a call with a fault draws no marks either
    for case in FAULT_CASES:
        if case[1] != "steps" and "step=" in case[3]:
            PC.TEX_ERRORS.clear()
            p = PC.run_frame(case[1], case[3], case[4], case[2])
            if PC.TEX_ERRORS:
                assert p.mark_tikz == [] and p.arcs == [] and p.ghosts == [], case[0]
    # the second hooks: the press works unwatched, so pressed draws nothing of its own; lifted off
    # leaves the two rings' ghosts on the empty hooks, and no arc
    PC.TEX_ERRORS.clear()
    p = PC.run_frame("hooks", "step=pressed", "", False)
    assert p.mark_tikz == p.arcs == p.ghosts == [] and not PC.TEX_ERRORS and "It jams if watched" in p.notes[1]
    p = PC.run_frame("hooks", "a quarter round, at a half; step=lifted off", "", False)
    assert p.arcs == [] and [q.what for q in p.ghosts] == ["the upper ring", "the lower ring"] and not PC.TEX_ERRORS
    hung = PC.run_frame("hooks", "a quarter round, at a half", "", False).picture.shown.rings   # the frame before
    assert [(q[0], q[1], q[2]) for q in p.ghosts] == [(x, y, PC.HRO) for x, y, t in hung]       # where they hung
    assert p.dump == "empty"
    assert [c[0] for c in FAULT_CASES] == [k for k, _ in PC.TEX_SITES] and len(PC.ERROR_SITES) == 83
    assert {k.rstrip("("): v for k, v in PC.SITE_COUNTS.items()} == {"terr": 4, "herr": 2, "cerr": 23, "rerr": 12, "oerr": 9, "xerr": 33}
    assert sum(PC.SITE_COUNTS.values()) == 83
    PC.check_sites()                                                     # the wrappers' tex.error calls
    PC.check_lua()                                                       # every rule and place, in the Lua's words
    PC.check_pins()                                                      # every chunk as transcribed, pinned
    # the bracket is order-free: every order of a frame's items draws the same frame
    for kind, state, items in (("card", "", ["reds made green", "step=first and second into one", "marks=off"]),
                               ("tin", "", ["first lit", "step=second dark"]),
                               ("row", RD, ["step=the first half off its end", "marks=on"]),
                               ("out", PAN, ["upper ring slid apart", "step=lower ring slid apart"])):
        drawn = set()
        for order in itertools.permutations(items):
            PC.TEX_ERRORS.clear()
            p = PC.run_frame(kind, "; ".join(order), state, False)
            assert not PC.TEX_ERRORS, (kind, order)
            drawn.add((p.head, tuple(p.own)))
        assert len(drawn) == 1, kind
    PC.TEX_ERRORS.clear()


def sketchbook():
    c = recipe()
    circuit = bare(c.to_graph())
    assert np.allclose(matrix_of(circuit), U / 4)
    rng = np.random.default_rng(1703)
    rand = [Fraction(int(k), 97) for k in rng.integers(1, 194, size=6)]    # generic angles (units of pi)
    assert all(a.denominator == 97 for a in rand)

    # --- Result 17.2: sandwich(a) = 2^-1/2 hub(a), at pi/4 and at a random angle
    for a in (Fraction(1, 4), rand[0]):
        sw = sandwich(a).done()
        assert np.allclose(matrix_of(sw), Gamma(float(a) * np.pi) / 2)
        assert np.allclose(matrix_of(hub_picture(a).done()), Gamma(float(a) * np.pi) / R2)
        assert same_picture(sw, hub_picture(a).done(-1))                    # with its number
        assert not same_picture(sw, hub_picture(a).done())                  # without it: no
        assert not same_picture(sandwich(a, where=0).done(), hub_picture(a).done(-1))  # turn on top: no

    # --- Result 17.3: two hubs on the same spiders = 2^-1/2 one hub, ends added,
    #     whatever turns the spiders on the wires carry (random here)
    al, be, top, bot = rand[1], rand[2], rand[3], rand[4]
    p = Two()
    feet = p.hub(al, top, bot)
    p.extra_hub(feet, be)
    two = p.done()
    assert same_picture(two, hub_picture(al + be, top, bot).done(-1))
    assert not same_picture(two, hub_picture(al + be, top, bot).done())
    assert not same_picture(two, hub_picture(al - be, top, bot).done(-1))

    # --- the chain of Section 17.6, every picture with its number
    q = Fraction(1, 4)
    # step (i): the sandwiches -> 1/2 (eighth, hub, eighth, hub)
    p = Two()
    p.on(1, VT.Z, q)
    p.hub(q)
    p.on(1, VT.Z, q)
    p.hub(q)
    step1 = p.done()
    assert not same_picture(circuit, step1)                                  # the half is owed
    step1_n = step1.copy()
    step1_n.scalar.add_power(-2)
    assert same_picture(circuit, step1_n)                                    # = 1/2 x (eighth, hub, eighth, hub)
    # step (ii): slide together (row 57) -- DERIVED by fusion; no number
    step2 = step1.copy()
    assert simp.spider_simp(step2)
    assert same_picture(step2, step1)
    zs = sorted(Fraction(step2.phase(v)) for v in step2.vertices()
                if step2.type(v) == VT.Z and step2.vertex_degree(v) > 1)
    assert zs == [0, Fraction(1, 2)]                                         # upper at rest, lower a quarter
    p = Two()
    feet = p.hub(q, 0, Fraction(1, 2))
    p.extra_hub(feet, q)
    two_hubs = p.done()
    assert same_picture(step2, two_hubs)
    assert np.allclose(matrix_of(two_hubs), U / 2)
    # step (iii): two hubs into one (row 130): 2^-1/2, the end a quarter
    one_hub = hub_picture(Fraction(1, 2), 0, Fraction(1, 2)).done()
    assert np.allclose(matrix_of(one_hub), U / R2)
    assert same_picture(two_hubs, hub_picture(Fraction(1, 2), 0, Fraction(1, 2)).done(-1))
    # step (iv): slide each wire's spider apart so the feet are a quarter back ...
    p = Two()
    p.hub(Fraction(1, 2), Fraction(-1, 2), Fraction(-1, 2))
    p.on(0, VT.Z, Fraction(1, 2))
    p.on(1, VT.Z, 1)
    apart = p.done()
    fused = apart.copy()
    assert simp.spider_simp(fused)                                           # DERIVED: fusion undoes it
    assert same_picture(fused, one_hub) and same_picture(apart, one_hub)
    # ... then the quarter hub is Ch 14's controlled turn at -pi: bare CZ/sqrt2, and so is a
    # Hadamard edge between two spiders at rest.  A TRANSLATION checked by evaluation, not a rewrite.
    qhub = hub_picture(Fraction(1, 2), Fraction(-1, 2), Fraction(-1, 2)).done()
    p = Two()
    p.tie()
    hedge = p.done()
    assert np.allclose(matrix_of(qhub), CZ / R2) and np.allclose(matrix_of(hedge), CZ / R2)
    assert not same_picture(hub_picture(Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)).done(), hedge)
    p = Two()
    p.tie()
    p.on(0, VT.Z, Fraction(1, 2))
    p.on(1, VT.Z, 1)
    tied = p.done()
    assert same_picture(apart, tied)
    # step (v): slide together -- DERIVED by fusion: the cooked picture
    cooked_derived = tied.copy()
    assert simp.spider_simp(cooked_derived)
    p = Two()
    a0, b0 = p.on(0, VT.Z, Fraction(1, 2)), p.on(1, VT.Z, 1)
    p.g.add_edge((a0, b0), ET.HADAMARD)
    cooked = p.done()
    assert same_picture(cooked_derived, cooked)
    assert cooked_derived.num_vertices() == 6 and cooked_derived.num_edges() == 5
    # step (vi): read off
    assert np.allclose(matrix_of(cooked), U / R2)
    assert np.allclose(matrix_of(cooked), kron(S, Z) @ CZ / R2)
    cooked_n = cooked.copy()
    cooked_n.scalar.add_power(-3)
    assert same_picture(circuit, cooked_n)                                   # circuit = 2^-3/2 cooked
    assert not same_picture(circuit, cooked)

    # --- PyZX cooks The Uncooked Net by its own moves (re-tyings, pivots, gadget moves)
    net = uncooked_net()
    # the upper wire's four lines, the lower wire's nine and the four ties, as in FourEighths.py
    assert net.num_edges() == 17 and same_picture(net, circuit)
    h = net.copy()
    simp.full_reduce(h)
    assert same_picture(h, net)                                              # value kept, number tracked
    assert zx.tcount(net) == 4 and zx.tcount(h) == 0
    spiders = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
    assert len(spiders) == 2 and all(h.type(v) == VT.Z for v in spiders)
    up = next(v for v in spiders if h.connected(v, h.inputs()[0]))          # the upper wire's
    down = next(v for v in spiders if h.connected(v, h.inputs()[1]))
    assert h.phase(up) == Fraction(1, 2) and h.phase(down) == 1              # a quarter, a half
    assert h.connected(up, down) and h.edge_type(h.edge(up, down)) == ET.HADAMARD
    assert sum(1 for e in h.edges() if h.edge_type(e) == ET.HADAMARD) == 1
    assert np.isclose(h.scalar.to_number(), 2 ** -1.5)                       # the number written beside it
    assert same_picture(h, cooked_n)

    # --- extraction: exactly (S (x) Z) CZ, T-count 0
    ce = zx.extract_circuit(h.copy())
    assert zx.tcount(ce) == 0
    assert np.allclose(matrix_of(ce.to_graph()), U)                          # exact, not up to a phase
    assert sorted((gt.name, gt.target, str(getattr(gt, "phase", ""))) for gt in ce.gates) == \
        sorted([("CZ", 1, ""), ("ZPhase", 1, "1"), ("ZPhase", 0, "1/2")])

    # --- phase teleportation (Result 17.7): teleport_reduce returns a NEW graph
    g0 = c.to_graph()
    before = matrix_of(g0)
    nv = g0.num_vertices()
    tg = simp.teleport_reduce(g0)
    assert g0.num_vertices() == nv and np.allclose(matrix_of(g0), before)   # input untouched
    ct = zx.Circuit.from_graph(tg)
    assert zx.tcount(ct) == 0
    assert np.allclose(matrix_of(ct.to_graph()), U)                          # exact
    names = [(gt.name, gt.target) for gt in ct.gates]
    assert names == [("S", 1), ("CNOT", 1), ("CNOT", 1), ("CNOT", 1), ("S", 1), ("CNOT", 1)]
    # every joiner kept; quarters in the first and fourth eighths' places, none in the 2nd, 3rd

    # --- validation: the recipe followed by the adjoint of the new one cooks to bare wires
    v = c.copy()
    v.add_circuit(ce.adjoint())
    vg = v.to_graph()
    simp.full_reduce(vg)
    assert all(vg.type(u) == VT.BOUNDARY for u in vg.vertices()) and vg.num_edges() == 2
    assert np.allclose(matrix_of(vg), np.eye(4))
    assert c.verify_equality(ce)
    bad = recipe()
    bad.add_gate("T", 1)
    assert not bad.verify_equality(ce)                                       # one eighth more: no

    # --- PyZX in general: 20 seeded random circuits, 3 wires, 40 gates, T with probability 0.2
    lowered, flat = 0, []
    for k in range(1, 21):
        random.seed(k)
        np.random.seed(k)
        g = zx.generate.cliffordT(3, 40, p_t=0.2)
        M = matrix_of(g)
        assert np.allclose(M @ M.conj().T, np.eye(8))                        # a unitary
        r = g.copy()
        simp.full_reduce(r)
        assert same_picture(r, g)                                            # the value, exactly
        assert zx.tcount(r) <= zx.tcount(g)
        if zx.tcount(r) < zx.tcount(g):
            lowered += 1
        else:
            flat.append(k)
    assert lowered == 18 and flat == [2, 13]                                 # lowered, never proved least


if __name__ == "__main__":
    verdict("CookingDown", blackboard, sketchbook)
