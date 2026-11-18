"""Challenge I: Four Eighths  (the T-count; the net of a circuit; the master)
===========================================================================
"Two of them turned. A quarter — and on the upper line's account. ...Oh."  -- Achilles
"Slide them together."  -- Tortoise

Claim: the recipe U = (CNOT_12 (I (x) T))^4 has four eighths (T-count 4) and is NOT the
trap I (x) Z: U = diag(1, -1, i, i), U|x1 x2> = i^(x2 + x1 xor x2) |x1 x2>.  Its bare
picture is exactly U/4, and three exact moves (colour change, fusion, two dummies at rest
with yellow boxes) turn it into its net, The Uncooked Net: all green, Hadamard ties, one end
per spider, value exactly U/4 -- the same net PyZX's own to_graph_like makes (isomorphic:
types, phases, ends, edge types).  Result 17.1's instance; Result 17.18.
"""
import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from itertools import product

import networkx as nx
import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import I2, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch17 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)  # control on top
CZ = np.diag([1, 1, 1, -1]).astype(complex)

U = np.linalg.matrix_power(CNOT @ kron(I2, T), 4)       # time right to left: T first, then the joiner


def basis(x1, x2):
    v = np.zeros(4, dtype=complex)
    v[2 * x1 + x2] = 1                                  # ket order: top wire first
    return v


# ---------------------------------------------------------------- the net, by hand
# The card's names.  Upper wire: in1 - small1 => small2 => big - out1.  Lower wire:
# in2 - e1 => p1 => e2 => p2 => e3 => p3 => e4 => p4 => out2.  Ties: big => p_k.
# (- a plain lead; => a Hadamard edge, the yellow sleeve)
NET_EDGES = (
    [("in1", "small1", ET.SIMPLE), ("small1", "small2", ET.HADAMARD),
     ("small2", "big", ET.HADAMARD), ("big", "out1", ET.SIMPLE)]            # the upper wire's four lines
    + [("in2", "e1", ET.SIMPLE)]
    + [(f"e{k}", f"p{k}", ET.HADAMARD) for k in range(1, 5)]
    + [(f"p{k}", f"e{k + 1}", ET.HADAMARD) for k in range(1, 4)]
    + [("p4", "out2", ET.HADAMARD)]                                         # the lower wire's nine lines
    + [("big", f"p{k}", ET.HADAMARD) for k in range(1, 5)]                  # the four ties
)
NET_PHASE = {"small1": 0, "small2": 0, "big": 0,
             **{f"e{k}": Fraction(1, 4) for k in range(1, 5)},
             **{f"p{k}": 0 for k in range(1, 5)}}


def uncooked_net():
    g = zx.Graph()
    v = {}
    v["in1"], v["in2"] = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0)
    v["small1"] = g.add_vertex(VT.Z, 0, 1)
    v["small2"] = g.add_vertex(VT.Z, 0, 2)
    v["big"] = g.add_vertex(VT.Z, 0, 5)
    for k in range(1, 5):
        v[f"e{k}"] = g.add_vertex(VT.Z, 1, 2 * k - 1, phase=Fraction(1, 4))
        v[f"p{k}"] = g.add_vertex(VT.Z, 1, 2 * k)
    v["out1"], v["out2"] = g.add_vertex(VT.BOUNDARY, 0, 10), g.add_vertex(VT.BOUNDARY, 1, 10)
    for a, b, et in NET_EDGES:
        g.add_edge((v[a], v[b]), et)
    g.set_inputs((v["in1"], v["in2"]))
    g.set_outputs((v["out1"], v["out2"]))
    return g, v


def recipe_circuit():
    c = zx.Circuit(2)
    for _ in range(4):
        c.add_gate("T", 1)            # an eighth on the lower line (qubit 1)
        c.add_gate("CNOT", 0, 1)      # then a joiner from the upper line (qubit 0) to the lower
    return c


def bare(g):
    g.scalar = zx.Scalar()            # drop pyzx's compensating number: the bare picture
    return g


def is_graph_like(g):
    """Duncan et al. Def. 3.1, checked by hand (a local helper)."""
    bnd = [v for v in g.vertices() if g.type(v) == VT.BOUNDARY]
    spiders = [v for v in g.vertices() if g.type(v) != VT.BOUNDARY]
    if any(g.type(v) != VT.Z for v in spiders):                                  # 1. all green
        return False
    for e in g.edges():
        a, b = g.edge_st(e)
        if a in spiders and b in spiders and g.edge_type(e) != ET.HADAMARD:       # 2. H-edges only
            return False
        if a == b:                                                              # 3. no self-loops
            return False
    for b in bnd:                                                               # 4. ends on greens,
        if len(list(g.neighbors(b))) != 1 or g.type(next(iter(g.neighbors(b)))) != VT.Z:
            return False
    return all(sum(1 for n in g.neighbors(v) if g.type(n) == VT.BOUNDARY) <= 1  #    one end each
               for v in spiders)


def as_nx(g):
    side = {v: f"in{k}" for k, v in enumerate(g.inputs())}
    side.update({v: f"out{k}" for k, v in enumerate(g.outputs())})
    G = nx.Graph()
    for v in g.vertices():
        G.add_node(v, t=g.type(v), ph=Fraction(g.phase(v)), side=side.get(v), q=float(g.qubit(v)))
    for e in g.edges():
        G.add_edge(*g.edge_st(e), et=g.edge_type(e))
    return G


def isomorphic(g1, g2):
    return nx.is_isomorphic(as_nx(g1), as_nx(g2),
                            node_match=lambda a, b: a == b, edge_match=lambda a, b: a == b)


def blackboard():
    # --- the gates and their identities
    assert np.allclose(T @ T, S) and np.allclose(S @ S, Z)
    assert np.allclose(np.linalg.matrix_power(T, 8), I2)
    assert not np.allclose(np.linalg.matrix_power(T, 4), I2)
    assert np.allclose(CNOT @ CNOT, np.eye(4))
    for x1, x2 in product((0, 1), repeat=2):                  # CNOT_12 |x1 x2> = |x1, x1 xor x2>
        assert np.allclose(CNOT @ basis(x1, x2), basis(x1, x1 ^ x2))

    # --- the recipe card: four eighths
    c = recipe_circuit()
    assert [gt.name for gt in c.gates].count("T") == 4 and zx.tcount(c) == 4

    # --- the recipe multiplied out (the Blackboard sidebar): M^2 = diag(1, i, w, w)
    w = np.exp(1j * np.pi / 4)
    M = CNOT @ kron(I2, T)
    assert np.allclose(M @ M, np.diag([1, 1j, w, w]))
    assert np.allclose(U, np.diag([1, -1, 1j, 1j]))
    assert np.allclose(U, kron(S, Z) @ CZ)                     # the headline, exact, scalar 1

    # --- the trap: "the joiners cancel, four eighths make a half on the lower line"
    trap = kron(I2, Z)
    assert np.allclose(np.linalg.matrix_power(T, 4), Z)       # four eighths ARE a half...
    assert not np.allclose(U, trap)                           # ...but the recipe is not I (x) Z
    for x2 in (0, 1):                                         # agree exactly when the upper is dark
        assert np.allclose(U @ basis(0, x2), trap @ basis(0, x2))
    for x2 in (0, 1):                                         # differ on both inputs with it lit
        assert not np.allclose(U @ basis(1, x2), trap @ basis(1, x2))
    # Achilles follows the card from |10>: two of the eighths turn, "on the upper line's account"
    state, turned = basis(1, 0), 0
    for _ in range(4):
        lower_lit = abs(state[3]) > 0.5                       # |11> carries the amplitude
        state = kron(I2, T) @ state
        turned += int(lower_lit)
        state = CNOT @ state
    assert turned == 2
    assert np.allclose(state, 1j * basis(1, 0))               # a quarter: i|10>
    assert np.allclose(trap @ basis(1, 0), basis(1, 0))       # the trap says nothing turns

    # --- the parity count: after k joiners the lower wire carries x2 xor (k mod 2) x1
    for x1, x2 in product((0, 1), repeat=2):
        lower, seen = x2, []
        for k in range(1, 5):
            seen.append(lower)                                # the k-th eighth sees this
            lower ^= x1                                       # the k-th joiner
            assert lower == x2 ^ ((k % 2) * x1)
        assert lower == x2
        assert seen == [x2, x1 ^ x2, x2, x1 ^ x2]
        assert np.allclose(U @ basis(x1, x2), 1j ** (x2 + (x1 ^ x2)) * basis(x1, x2))

    # --- Result 17.18 on the blackboard: sum over the bits of the net's eleven spiders.
    # A green spider is a bit b with the weight e^{i phase b}; a plain lead ties its spider's
    # bit to the boundary bit; a Hadamard edge between bits a, b gives (-1)^{ab}/sqrt2.
    spiders = list(NET_PHASE)
    assert len(spiders) == 11
    net_matrix = np.zeros((4, 4), dtype=complex)
    for y1, y2, x1, x2 in product((0, 1), repeat=4):          # <y1 y2| net |x1 x2>
        ends = {"in1": x1, "in2": x2, "out1": y1, "out2": y2}
        total = 0
        for bits in product((0, 1), repeat=11):
            b = dict(zip(spiders, bits))
            b.update(ends)
            term = np.prod([np.exp(1j * np.pi * float(NET_PHASE[s]) * b[s]) for s in spiders])
            for u, v, et in NET_EDGES:
                if et == ET.SIMPLE:
                    term *= float(b[u] == b[v])
                else:
                    term *= (-1) ** (b[u] * b[v]) / np.sqrt(2)
            total += term
        net_matrix[2 * y1 + y2, 2 * x1 + x2] = total
    assert np.allclose(net_matrix, U / 4)                     # exactly a quarter of U
    assert not np.allclose(net_matrix, U) and not np.allclose(net_matrix, trap / 4)

    pictures()


# every frame's dump, in the dialogue's order (the log's, card 3e.6 item 6: r17/build/def/dumps_md.txt)
FRAME_DUMPS = (
    # First block 1 frame 1: The card, laid out
    'card aside=0 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @1.32,0; 2:ring green'
    ' 0 @3.66,0; 3:ring green 0 @6,0; 4:ring green 0 @8.34,0; 5:ring green 3 @0.15,1; 6:ring red 0 @1.32,1; '
    '7:ring green 3 @2.49,1; 8:ring red 0 @3.66,1; 9:ring green 3 @4.83,1; 10:ring red 0 @6,1; 11:ring green 3 '
    '@7.17,1; 12:ring red 0 @8.34,1; tie 1-6 []; tie 2-8 []; tie 3-10 []; tie 4-12 []; cord loose-loose [-0.5,0 '
    '9.5,0]; cord loose-loose [-0.5,1 9.5,1]',
    # First block 2 frame 1: Every ring green
    'green aside=0 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @1.32,0; 2:ring '
    'green 0 @3.66,0; 3:ring green 0 @6,0; 4:ring green 0 @8.34,0; 5:ring green 3 @0.15,1; 6:ring green 0 '
    '@1.32,1; 7:ring green 3 @2.49,1; 8:ring green 0 @3.66,1; 9:ring green 3 @4.83,1; 10:ring green 0 @6,1; '
    '11:ring green 3 @7.17,1; 12:ring green 0 @8.34,1; tube @0.735,1 a0; tube @1.905,1 a0; tube @3.075,1 a0; tube'
    ' @4.245,1 a0; tube @5.415,1 a0; tube @6.585,1 a0; tube @7.755,1 a0; tube @8.925,1 a0; tie 1-6 tubes1 []; tie'
    ' 2-8 tubes1 []; tie 3-10 tubes1 []; tie 4-12 tubes1 []; cord loose-loose [-0.5,0 9.5,0]; cord loose-loose '
    '[-0.5,1 9.5,1]',
    # First block 3 frame 1: Leftmost rings met
    'green,upper:2 aside=0 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @3.66,0; '
    '2:ring green 0 @6,0; 3:ring green 0 @8.34,0; 4:ring green 3 @0.15,1; 5:ring green 0 @1.32,1; 6:ring green 3 '
    '@2.49,1; 7:ring green 0 @3.66,1; 8:ring green 3 @4.83,1; 9:ring green 0 @6,1; 10:ring green 3 @7.17,1; '
    '11:ring green 0 @8.34,1; tube @0.735,1 a0; tube @1.905,1 a0; tube @3.075,1 a0; tube @4.245,1 a0; tube '
    '@5.415,1 a0; tube @6.585,1 a0; tube @7.755,1 a0; tube @8.925,1 a0; tie 1-5 tubes1 []; tie 1-7 tubes1 []; tie'
    ' 2-9 tubes1 []; tie 3-11 tubes1 []; cord loose-loose [-0.5,0 9.5,0]; cord loose-loose [-0.5,1 9.5,1]',
    # First block 3 frame 2: Joined midway
    'green,upper:3 aside=0 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @4.83,0; '
    '2:ring green 0 @8.34,0; 3:ring green 3 @0.15,1; 4:ring green 0 @1.32,1; 5:ring green 3 @2.49,1; 6:ring green'
    ' 0 @3.66,1; 7:ring green 3 @4.83,1; 8:ring green 0 @6,1; 9:ring green 3 @7.17,1; 10:ring green 0 @8.34,1; '
    'tube @0.735,1 a0; tube @1.905,1 a0; tube @3.075,1 a0; tube @4.245,1 a0; tube @5.415,1 a0; tube @6.585,1 a0; '
    'tube @7.755,1 a0; tube @8.925,1 a0; tie 1-4 tubes1 []; tie 1-6 tubes1 []; tie 1-8 tubes1 []; tie 2-10 tubes1'
    ' []; cord loose-loose [-0.5,0 9.5,0]; cord loose-loose [-0.5,1 9.5,1]',
    # First block 3 frame 3: A fan of sleeves
    'green,big aside=0 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @4.83,0; 2:ring '
    'green 3 @0.15,1; 3:ring green 0 @1.32,1; 4:ring green 3 @2.49,1; 5:ring green 0 @3.66,1; 6:ring green 3 '
    '@4.83,1; 7:ring green 0 @6,1; 8:ring green 3 @7.17,1; 9:ring green 0 @8.34,1; tube @0.735,1 a0; tube '
    '@1.905,1 a0; tube @3.075,1 a0; tube @4.245,1 a0; tube @5.415,1 a0; tube @6.585,1 a0; tube @7.755,1 a0; tube '
    '@8.925,1 a0; tie 1-3 tubes1 []; tie 1-5 tubes1 []; tie 1-7 tubes1 []; tie 1-9 tubes1 []; cord loose-loose '
    '[-0.5,0 9.5,0]; cord loose-loose [-0.5,1 9.5,1]',
    # First block 4 frame 1: Rings clipped on
    'green,big,small aside=0 | cols=10 rows=2 y=0..1 near=false folded=false; gaps=0.5; 1:ring green 0 @1.83,0; '
    '2:ring green 0 @3.33,0; 3:ring green 0 @4.83,0; 4:ring green 3 @0.15,1; 5:ring green 0 @1.32,1; 6:ring green'
    ' 3 @2.49,1; 7:ring green 0 @3.66,1; 8:ring green 3 @4.83,1; 9:ring green 0 @6,1; 10:ring green 3 @7.17,1; '
    '11:ring green 0 @8.34,1; tube @2.58,0 a0; tube @4.08,0 a0; tube @0.735,1 a0; tube @1.905,1 a0; tube @3.075,1'
    ' a0; tube @4.245,1 a0; tube @5.415,1 a0; tube @6.585,1 a0; tube @7.755,1 a0; tube @8.925,1 a0; tie 3-11 '
    'tubes1 []; tie 3-5 tubes1 []; tie 3-7 tubes1 []; tie 3-9 tubes1 []; cord loose-loose [-0.5,0 9.5,0]; cord '
    'loose-loose [-0.5,1 9.5,1]',
)


def row_of(dump, y):
    """The things on one line of a cloth dump, left to right: (kind, colour, turn, x)."""
    nodes = PC.parse_dump(dump)[0]
    return sorted((x, kind, col, t) for kind, col, t, x, yy in nodes.values() if yy == y)


def rings_at(dump):
    """The rings of a cloth dump, as (line, column) places."""
    return {(y, round(x, 3)) for kind, col, t, x, y in PC.parse_dump(dump)[0].values() if kind == "ring"}


def slide_of(a):
    """A hop's (line, from column, to column), as the dump writes columns."""
    y, x0, x1 = a.slide
    return y, round(x0, 3), round(x1, 3)


def ghost_grid(q):
    """The grid place a ring's ghost names in the log ("the green ring (grid 1.32,0)")."""
    x, y = re.search(r"\(grid ([-\d.]+),([-\d.]+)\)", q.what).groups()
    return float(y), float(x)


def check_slides(R):
    """A slide step's marks, on the drawn extent: every hop runs from a ring of the state before to
    a ring of the step's state (or of the state before, a chained move), the way the ring went, on
    the line's outer side, bowed at least the Lua's hop floor; every ring that left its place has a
    ghost there, and is the start of a hop; every ring that came to a new place is the end of one."""
    g = PC.G(PC.card_scene(R.after), PC.CARDOPTS, PC.CWHO)
    old, new = rings_at(PC.card_dump(R.before)), rings_at(PC.card_dump(R.after))
    hops = [a for a in R.arcs if getattr(a, "slide", None) is not None]
    for a in hops:
        y, x0, x1 = slide_of(a)
        assert a.kind == "slide" and (y, x0) in old and (y, x1) in old | new and x0 != x1, a.what
        assert np.sign(a.to[0] - a.frm[0]) == np.sign(x1 - x0)                      # the way it went
        assert abs(a.frm[0] - g.X(x0)) < abs(a.frm[0] - g.X(x1)) and abs(a.to[0] - g.X(x1)) < abs(a.to[0] - g.X(x0))
        line = g.Y(y)
        assert (min(p[1] for p in a.body) > line) if y == 0 else (max(p[1] for p in a.body) < line), a.what
        assert a.minsag == PC.HOPSAG and a.sag >= PC.HOPSAG - 1e-9, (a.what, a.sag)
    places = {ghost_grid(q) for q in R.ghosts if not q.bead}
    for q in R.ghosts:
        if not q.bead:
            y, x = ghost_grid(q)
            assert np.allclose(q[:2], g.XY((x, y)), atol=1e-3), q.what               # drawn where it lay
    for y, x in old - new:
        assert (y, x) in places and any(slide_of(a)[:2] == (y, x) for a in hops), (y, x)
    for y, x in new - old:
        assert any(slide_of(a)[0] == y and slide_of(a)[2] == x for a in hops), (y, x)
    return hops


def pictures():
    """The First Challenge's prop pictures (identity card, sections 4 and 3e), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch17.py.  The ties
    below hold the transcription against this file's blackboard: graph_of reads a cloth dump as
    a PyZX graph, a structural check of the pictures, not a second proof.  Round 4 draws the
    upper rings slid together move by move (B1.3): each part-way frame is the recipe's value,
    exactly."""
    calls = DIALOGUE.check_layout("First", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("cardcloth", True, "", (), "The card, laid out"),
        ("cardcloth", False, "reds made green", (), "Every ring green"),
        ("cardclothsteps", False, "after=reds made green; from=off",
         ("first and second into one; third into those; fourth into those",), "Leftmost rings met; Joined midway; A fan of sleeves"),
        ("cardcloth", False, "reds made green, upper rings slid together, small rings put on", (), "Rings clipped on")]
    PC.check_captions(DIALOGUE, "First", calls)
    PC.check_section(DIALOGUE, "First", calls)
    assert PC.check_anchors(DIALOGUE, "First") == (4, 6)                 # P1.1's and P1.3's, both files
    frames = [PC.frame_chain(c) for c in calls]
    assert [len(f) for f in frames] == [1, 1, 3, 1]                      # card 3e.6: 4 blocks, 6 frames
    assert tuple(R.dump for f in frames for R in f) == FRAME_DUMPS
    (p11,), (p12,), b13, (p14,) = frames
    p13 = b13[-1]
    pics = [p11, p12, p13, p14]
    # every strip's last frame is round 3's picture of its block: P1.3 as round 3 drew it
    assert p13.dump == PC.cardcloth("reds made green, upper rings slid together", False).dump
    # one frame in every state: the kit's (columns 10, rows 2, fine) and the band below it
    assert {R.frame for f in frames for R in f} == {1.37}
    assert all(p.shown.counters == () and p.state.aside == 0 for p in pics)     # nothing set aside in I

    # the mathematics: every picture of the First Challenge is the recipe's net, exactly U/4
    graphs = [PC.graph_of(p.dump) for p in pics]
    for g in graphs:
        assert np.allclose(matrix_of(g), U / 4)
        assert not np.allclose(matrix_of(g), kron(I2, Z) / 4)                  # never the trap
    assert same_picture(graphs[0], bare(recipe_circuit().to_graph()))         # P1.1: the bare card
    net, _ = uncooked_net()
    assert graphs[3].num_edges() == 17                                         # P1.4: seventeen lines
    assert isomorphic(graphs[3], net) and is_graph_like(graphs[3])            # the master, spider for spider
    pg = recipe_circuit().to_graph()
    simp.to_graph_like(pg)
    assert isomorphic(graphs[3], pg)                                           # PyZX's own net
    assert not is_graph_like(graphs[2])                                        # P1.3: the big ring holds two ends
    assert graphs[2].num_edges() == 15 and not isomorphic(graphs[2], net)
    # the plausible mix-ups fail: one small ring's tube missing (a yellow box on the upper lead),
    # or the lower right lead's tube missing; both small rings bare is the two dummies taken off
    # again, no change (a tube twice is no tube)
    slack = p14.dump.replace("tube @2.58,0 a0; ", "")
    assert np.allclose(matrix_of(PC.graph_of(slack)), U @ kron(H, I2) / 4)
    short = p12.dump.replace("tube @8.925,1 a0; ", "")
    assert np.allclose(matrix_of(PC.graph_of(short)), kron(I2, H) @ U / 4)
    assert not np.allclose(kron(I2, H) @ U, U) and not np.allclose(U @ kron(H, I2), U)
    bare_small = p14.dump.replace("tube @2.58,0 a0; tube @4.08,0 a0; ", "")
    assert np.allclose(matrix_of(PC.graph_of(bare_small)), U / 4) and not is_graph_like(PC.graph_of(bare_small))

    # B1.3, move by move (card 3e.6, item 7): the leftmost two upper rings into one, then the third
    # into those, then the fourth -- each frame a state of the slide part-way, its value exactly U/4
    # (a ring slid into a ring of its colour, their turns added, is the same picture), and none but
    # P1.4 the master
    assert [R.step for R in b13] == ["first and second into one", "third into those", "fourth into those"]
    assert b13[0].before == p12.state                                          # after=reds made green
    assert [len([x for y, x in rings_at(R.dump) if y == 0]) for R in b13] == [3, 2, 1]
    assert [R.after.upn for R in b13[:2]] == [2, 3] and b13[2].after.big
    for R in b13:
        g = PC.graph_of(R.dump)
        assert np.allclose(matrix_of(g), U / 4) and R.after.aside == 0, R.step
        assert not isomorphic(g, net)
        assert not np.allclose(matrix_of(g), kron(I2, Z) / 4)
        # the mix-up: the slid ring's ties left behind (a tie dropped) is not the recipe
        cut = re.sub(r"tie 1-\d+ tubes1 \[\]; ", "", R.dump, count=1)
        assert cut != R.dump and not np.allclose(matrix_of(PC.graph_of(cut)), U / 4)
    # their marks: a ghost at each ring's old place and a hop to its new one, every hop drawn
    assert [len(check_slides(R)) for R in b13] == [1, 2, 1]
    assert all(R.marked and not R.keyed for R in b13)
    assert [len(R.ghosts) for R in b13] == [1, 2, 1]
    assert [round(a.sag, 3) for R in b13 for a in R.arcs] == [0.058] * 4               # L4: every hop 0.058
    assert all(a.fit[0] >= PC.MINCLEAR - 1e-9 for R in b13 for a in R.arcs)            # clear of the net
    # the lone pictures carry no marks; P1.4, the drawn copy of the tangle, is never marked: its
    # step (the small rings put on) draws none, and marks=on there is refused
    for p in (p11, p12, p14):
        assert not p.marked and p.arcs == [] and p.ghosts == [] and p.step is None
    R = PC.run_frame("card", "reds made green, upper rings slid together; step=small rings put on", "", False)
    assert R.arcs == R.ghosts == [] and R.notes[1].startswith("no marks: the rings and tubes come from her box")
    assert R.dump == p14.dump
    assert PC.faults(PC.run_frame, "card", "reds made green, upper rings slid together; step=small rings put on; "
                     "marks=on", "", False) == ["card-marks-tangle"]
    # a part-way frame is never keyed (the card's words are set for its whole states)
    assert PC.faults(PC.run_frame, "card", "reds made green; step=first and second into one; marks=off", "",
                     True) == ["card-partkey"]
    PC.TEX_ERRORS.clear()

    # the card's details (section 4): P1.1's eighths at turn 3 (an eighth round, in twenty-fourths)
    # on the lower line and its joiners' ties plain; P1.2's ties all sleeved, the lower left lead
    # plain and the lower right lead with a tube; P1.3's one upper ring at the third eighth's
    # column; P1.4's upper line ring, tube, ring, tube, ring from the left
    nodes, tubes, ties, cords = PC.parse_dump(p11.dump)
    assert [t for x, kind, col, t in row_of(p11.dump, 1) if col == "green"] == [3, 3, 3, 3]
    assert all(k == 0 for _, _, _, k in ties) and not tubes
    nodes, tubes, ties, cords = PC.parse_dump(p12.dump)
    assert all(k == 1 for _, _, _, k in ties) and len(ties) == 4
    lower = [x for x, *_ in row_of(p12.dump, 1)]
    assert not any(tx < lower[0] for tx, ty in tubes if ty == 1)               # the left lead plain
    assert any(tx > lower[-1] for tx, ty in tubes if ty == 1)                  # the right lead sleeved
    assert [x for x, *_ in row_of(p13.dump, 0)] == [PC.CX[2] + 2 * PC.CD] == [4.83]
    up = sorted([(x, "ring") for x, *_ in row_of(p14.dump, 0)] + [(tx, "tube") for tx, ty in PC.parse_dump(p14.dump)[1] if ty == 0])
    assert [k for _, k in up] == ["ring", "tube", "ring", "tube", "ring"]

    # the captions' truth: the card as it stands; no red ring; the two leftmost upper rings met
    # (at the second's place), then midway (the third eighth's column), then one upper ring and four
    # sleeved ties fanning from it; the small rings put on
    assert PC.card_words(p11.state) == "card"
    assert all(col == "green" for kind, col, t, x, y in PC.parse_dump(p12.dump)[0].values())
    assert any(col == "red" for kind, col, t, x, y in PC.parse_dump(p11.dump)[0].values())
    assert [x for y, x in sorted(rings_at(b13[0].dump)) if y == 0][0] == round(PC.CX[3], 3)   # where the second stood
    assert (0, round(PC.CX[4], 3)) in rings_at(b13[1].dump)                               # midway
    n13, _, t13, _ = PC.parse_dump(p13.dump)
    assert len(row_of(p13.dump, 0)) == 1 and [(a, k) for _, a, b, k in t13] == [(1, 1)] * 4
    assert PC.card_words(p14.state) == "green,big,small"

    # fixed parts, frame to frame: the card's columns from P1.1 to P1.2 (only the reds change
    # colour); the lower line, every ring, turn and tube where it was, from P1.2 through B1.3 to P1.4
    strip = lambda r: [(x, kind, t) for x, kind, col, t in r]
    assert [strip(row_of(p11.dump, y)) for y in (0, 1)] == [strip(row_of(p12.dump, y)) for y in (0, 1)]
    for p in b13 + [p14]:
        assert row_of(p.dump, 1) == row_of(p12.dump, 1), p.step
        assert [u for u in PC.parse_dump(p.dump)[1] if u[1] == 1] == [u for u in PC.parse_dump(p12.dump)[1] if u[1] == 1]
    # and the upper rings a step does not move keep their places
    for P, R in zip([p12] + b13, b13):
        moved = {slide_of(a)[:2] for a in R.arcs}
        assert {u for u in rings_at(P.dump) if u[0] == 0} - moved <= rings_at(R.dump), R.step
    # a picture that moved a fixed part would fail: the lower line shifted by a column
    moved = PC.card_scene(p13.state)
    moved["things"][1]["x"] += 0.5
    assert strip(row_of("moved | " + PC.kit_dump(moved, PC.CWHO), 1)) != strip(row_of(p12.dump, 1))

    # the keyed words (P1.1): the kit's ring and line (placed by the kit, not re-measured here: the
    # identity card's section 4 gives every kit word clear by at least 0.028 cm, ink to ink), and
    # the Lua's own bead, measured: clear of the drawing, and above the kit's frame by design, in
    # the band over the linen
    assert p11.labels == ("ring", "line", "bead") and all(R.labels == () for f in frames[1:] for R in f)
    PC.check_compiled_boxes(p11.picture)
    least = PC.check_keyed_words(p11.picture)
    assert least["bead"] >= 0.06
    # a fault the Lua raises is a fault here: a step out of order, or a step twice
    PC.TEX_ERRORS.clear()
    PC.card_state("upper rings slid together")
    PC.card_state("reds made green, reds made green")
    assert [k for k, _ in PC.TEX_ERRORS] == ["card-after", "card-twice"]
    PC.TEX_ERRORS.clear()
    # a picture of the card out of its place would fail (none in the Third)
    third = [c for b in DIALOGUE.blocks("Third") for c in b][0]
    fake = calls[0].__class__(calls[0].macro, False, calls[0].opt, (), calls[0].caption, third.text)
    assert not PC.place_ok("Third", fake, DIALOGUE.read(DIALOGUE.dialog))


def sketchbook():
    c = recipe_circuit()

    # --- the recipe's bare picture: four joiners, four eighths, exactly U/4
    g = bare(c.to_graph())
    assert np.allclose(matrix_of(g), U / 4)
    assert np.allclose(matrix_of(c.to_graph()), U)            # pyzx's number is 4: (sqrt2)^4
    assert not np.allclose(matrix_of(g), kron(I2, Z) / 4)     # the trap, in pictures

    # --- two joiners with nothing between: 1/2 times two plain wires (row 127).
    # DERIVED: fusion slides the greens together and the reds together; the doubled cord
    # between a green and a red comes away (Hopf, row 62) with the half.
    two = zx.Circuit(2)
    two.add_gate("CNOT", 0, 1)
    two.add_gate("CNOT", 0, 1)
    tg = bare(two.to_graph())
    assert np.allclose(matrix_of(tg), np.eye(4) / 2)
    d = tg.copy()
    simp.spider_simp(d)
    simp.id_simp(d)
    assert all(d.type(v) == VT.BOUNDARY for v in d.vertices())      # bare wires...
    assert np.isclose(d.scalar.to_number(), 0.5)                    # ...and the half
    assert same_picture(d, tg)
    # an eighth between them: the reds can no longer meet (red, green, red on the lower wire)
    sand = zx.Circuit(2)
    sand.add_gate("CNOT", 0, 1)
    sand.add_gate("T", 1)
    sand.add_gate("CNOT", 0, 1)
    sg = bare(sand.to_graph())
    simp.spider_simp(sg)
    assert sum(1 for v in sg.vertices() if sg.type(v) == VT.X) == 2
    assert not np.allclose(matrix_of(sg), kron(I2, T) / 2)          # not just the eighth

    # --- the three moves, as rewrites, the picture checked after each (Result 17.1's instance)
    # PyZX draws the rings as green (Z) and red (X) spiders; a Hadamard edge is the sleeve.
    h = g.copy()
    reds = [v for v in h.vertices() if h.type(v) == VT.X]
    assert len(reds) == 4
    # (1) colour change (row 58): each red target turns green, its three legs sleeved
    for v in reds:
        assert simp.check_color_change(h, v) and simp.color_change(h, v)
        assert h.type(v) == VT.Z
        assert all(h.edge_type(e) == ET.HADAMARD for e in h.incident_edges(v))
        assert same_picture(h, g)
    # (2) fusion (row 57): the four upper greens slide into one big ring -- "Slide them together."
    assert simp.spider_simp(h)
    assert same_picture(h, g)
    upper = [v for v in h.vertices() if h.type(v) == VT.Z and h.qubit(v) == 0]
    assert len(upper) == 1
    big = upper[0]
    in1, out1 = h.inputs()[0], h.outputs()[0]
    assert h.connected(big, in1) and h.connected(big, out1)         # it holds two ends:
    assert not is_graph_like(h)                                     # condition 4 fails
    assert not simp.is_graph_like(h, strict=True)
    # (3) the dummies (rows 56, 57): a ring at rest, a sleeve, a ring at rest, a sleeve
    h.remove_edge(h.edge(in1, big))
    s1 = h.add_vertex(VT.Z, 0, 0.3)
    s2 = h.add_vertex(VT.Z, 0, 0.6)
    h.add_edge((in1, s1), ET.SIMPLE)
    h.add_edge((s1, s2), ET.HADAMARD)
    h.add_edge((s2, big), ET.HADAMARD)
    assert same_picture(h, g)                                       # no number spent
    assert is_graph_like(h) and simp.is_graph_like(h, strict=True)
    # one dummy alone (a ring and a single sleeve) is a yellow box on the lead, not a plain lead
    one, w = uncooked_net()
    one.remove_vertex(w["small1"])
    one.add_edge((w["in1"], w["small2"]), ET.SIMPLE)
    assert not same_picture(one, g)
    assert np.allclose(matrix_of(one), U @ kron(H, I2) / 4)

    # --- The Uncooked Net, built by hand: the master
    net, v = uncooked_net()
    # the upper wire's four lines, the lower wire's nine and the four ties: 4 + 9 + 4
    assert net.num_edges() == 17
    assert net.num_vertices() == 15
    assert sum(1 for u in net.vertices() if net.type(u) == VT.Z) == 11
    assert np.allclose(matrix_of(net), U / 4)                       # Result 17.18, exact
    assert same_picture(net, g)                                     # = the recipe's bare picture
    assert is_graph_like(net) and simp.is_graph_like(net, strict=True)
    assert zx.tcount(net) == 4                                      # four eighths
    assert isomorphic(net, h)                                       # = the three moves' result
    # PyZX's own graph-like form of the recipe is the same net, spider for spider
    pg = c.to_graph()
    simp.to_graph_like(pg)
    assert isomorphic(net, pg)
    assert np.isclose(pg.scalar.to_number(), 4)                     # its number is still 4:
    assert same_picture(bare(pg), net)                              # bare, the same value
    # negative controls: an eighth on a plain ring, or the lower right lead unsleeved, is not it
    wrong, w = uncooked_net()
    wrong.set_phase(w["p1"], Fraction(1, 4))
    wrong.set_phase(w["e1"], 0)
    assert not isomorphic(net, wrong) and not same_picture(net, wrong)
    plain, w = uncooked_net()
    plain.set_edge_type(plain.edge(w["p4"], w["out2"]), ET.SIMPLE)
    assert not isomorphic(net, plain) and not same_picture(net, plain)


if __name__ == "__main__":
    verdict("FourEighths", blackboard, sketchbook)
