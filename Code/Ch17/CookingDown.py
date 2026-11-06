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

import random
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


def pictures():
    """The Second Challenge's prop pictures (identity card, section 4), read from the dialogue
    through tools/dfp_props.py and the transcription tools/props_ch17.py, and held against
    this file's blackboard (graph_of: a structural reading of each cloth dump, not a proof)."""
    calls = DIALOGUE.check_layout("Second", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("cardcloth", False, "second eighth hung", (), "The cradle, undone"),
        ("cardcloth", False, "second eighth hung, fourth eighth hung, slid together", (), "Feet slid together"),
        ("cardcloth", False, "second eighth hung, fourth eighth hung, slid together, one hub, sleeved tie", (),
         "Nothing hanging"),
        ("secondhooks", True, "a quarter round, at a half", (), "What the pan made")]
    PC.check_captions(DIALOGUE, "Second", calls)
    PC.check_section(DIALOGUE, "Second", calls)
    PC.check_anchors(DIALOGUE, "Second", ["P2.1", "P2.2", "P2.3", "P2.4"])
    pics = [PC.picture(c) for c in calls]
    p21, p22, p23, p24 = pics
    assert tuple(p.dump for p in pics) == SECOND_DUMPS
    assert [p.frame for p in pics] == [1.37, 1.37, 1.37, 1.5]          # the card's frame as in I (band included)
    assert [p.state.aside for p in (p21, p22, p23)] == [1, 2, 3]
    assert [len(p.shown.counters) for p in (p21, p22, p23)] == [1, 2, 3]

    # the counters' number, computed from this file's blackboard, not read: a sandwich is
    # 2^-1/2 times its hub picture; two hubs on the same feet 2^-1/2 times one hub
    a, b = 0.37, -1.21
    hub = lambda t: Gamma(t) / R2                                     # the hub picture's value
    per_hang = (Gamma(a) / 2)[1, 1] / hub(a)[1, 1]                     # sandwich / hub
    per_merge = (hub(a) @ hub(b))[1, 1] / hub(a + b)[1, 1]             # two hubs / one hub
    assert np.isclose(per_hang, 2 ** -0.5) and np.isclose(per_merge, 2 ** -0.5)
    number = {1: per_hang, 2: per_hang ** 2, 3: per_hang ** 2 * per_merge}
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

    # the captions' truth: a hub stands; one upper and one lower ring with both hubs tied to
    # each; no hub and no bob left; a pair on the hooks
    kinds = lambda p: [k for k, c, t, x, y in PC.parse_dump(p.dump)[0].values()]
    assert "hub" in kinds(p21)
    n22, _, t22, _ = PC.parse_dump(p22.dump)
    up = [k for k, v in n22.items() if v[0] == "ring" and v[4] == 0]
    lo = [k for k, v in n22.items() if v[0] == "ring" and v[4] == 1]
    hubs = [k for k, v in n22.items() if v[0] == "hub"]
    assert len(up) == len(lo) == 1 and len(hubs) == 2
    assert {(a_, b_) for kind, a_, b_, _ in t22 if kind == "tie"} == {(up[0], h) for h in hubs} | {(h, lo[0]) for h in hubs}
    assert "hub" not in kinds(p23) and "bob" not in kinds(p23)
    assert PC.hooks_dump(p24.state) != "empty"

    # fixed parts: the first hub and its end stay where they hung from P2.1 to P2.2
    hung = lambda p: sorted((k, x, y) for k, c, t, x, y in PC.parse_dump(p.dump)[0].values() if k in ("hub", "bob") and x < 4)
    assert hung(p21) == hung(p22) == [("bob", 1.49, 0.5), ("hub", 2.49, 0.5)]
    # the counters lie in the band below the linen, clear of the linen's shadow and inside the frame
    for p in (p21, p22, p23):
        v = p.shown
        g = PC.C._geometry(p.cloth, p.options)
        s = g["P"] / 0.30
        L = v.linen
        shadow = (L[0] + PC.LINEN_SHADOW * s, L[1] - PC.LINEN_SHADOW * s, L[2] + PC.LINEN_SHADOW * s, L[3] - PC.LINEN_SHADOW * s)
        for c in v.counters:
            box = PC.counter_box(*c)
            assert PC.box_gap(box, shadow) >= 0.02 and PC.box_gap(box, L) >= 0.02
            assert box[1] - (v.fbot - PC.ASIDE) >= 0.025 and 1.25 - box[2] >= 0.025
        assert v.counters == pics[2].shown.counters[:len(v.counters)]  # each counter laid on the others, where it was

    # the keyed words (P2.4): second hooks, sleeve; set as the compiled pictures set them, their
    # letters clear of every part drawn (second hooks stands over the frame's top, by design)
    assert p24.labels == ("second hooks", "sleeve") and all(p.labels == () for p in (p21, p22, p23))
    PC.check_compiled_boxes(p24)
    least = PC.check_keyed_words(p24)
    assert least["sleeve"] >= 0.02 and least["second hooks"] >= 0.1
    # the Lua's refusals: the two ways through do not mix; one hub before slid together
    PC.TEX_ERRORS.clear()
    PC.card_state("second eighth hung, reds made green")
    PC.card_state("second eighth hung, one hub")
    PC.hooks_state("a quarter round")
    assert [k for k, _ in PC.TEX_ERRORS] == ["card-twopaths-first", "card-onehub", "hooks-pair"]
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
