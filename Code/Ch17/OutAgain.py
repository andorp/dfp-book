"""Challenge IV: Out Again  (flow; gFlow; extraction; the synthesis)
==================================================================
"Slide them apart."  -- Achilles
"...Oh. Each has to go before the other."  -- Achilles

Claim: the row 1-2-3 has a flow, f(1) = 2, f(2) = 3, and a gFlow; the V (edges a-c, b-c; c
kept) has neither, with no input or with a as an input (Result 17.13; Danos-Kashefi Fig. 2).
A joiner at the ends unties a doubled tie: the frontier picture with B = [[1,1],[0,1]] equals
the B = I picture (exactly H (x) H) followed by a bare joiner, green on the upper wire and red
on the lower, and not the other way up (Result 17.15; Duncan et al. Prop. 7.1, the instance).
The cooked picture is a Hadamard edge between two spiders at rest followed by a green quarter
and a green half (Result 17.16): the new recipe (S (x) Z) CZ, exactly U.  The Uncooked Net has
the causal flow of Section 17.4 and PyZX finds a gFlow on it (Lemma 3.7).  On the 20 random
circuits of CookingDown.py extraction succeeds 20/20, each equal to the original only up to a
unit number r, |r| = 1 (none exactly: Duncan et al. Remark 2.2); PyZX's finder finds a gFlow
on the 16 cooked nets without a phase gadget and none on the 4 with one (seeds 6, 8, 10, 18) --
a statement about PyZX's finder and ordinary gFlow only, nothing about Backens et al.'s
extended notion.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import random
import re
from fractions import Fraction
from itertools import combinations, permutations, product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.gflow import gflow
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import I2, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch17 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
ORD = ("First", "Second", "Third", "Fourth")

Z = np.diag([1, -1]).astype(complex)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)  # control on top
CNOT21 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)  # control below
CZ = np.diag([1, 1, 1, -1]).astype(complex)
U = np.linalg.matrix_power(CNOT @ kron(I2, T), 4)
R2 = np.sqrt(2)


# ------------------------------------------------ flow and gFlow, by brute force (local helpers)
def nbrs(E, v):
    return {b for a, b in E if a == v} | {a for a, b in E if b == v}


def odd(E, A):
    """Odd(A): the vertices with an odd number of neighbours in A."""
    V = {x for e in E for x in e}
    return {v for v in V if len(nbrs(E, v) & set(A)) % 2 == 1}


def is_flow(E, V_in, V_out, f, order):
    """Danos-Kashefi Def. 2 in the book's letters; `order` is a list, earlier = smaller."""
    rank = {v: k for k, v in enumerate(order)}
    for i in (v for v in order if v not in V_out):
        if f[i] in V_in or f[i] not in nbrs(E, i):                       # f into non-inputs; (F0)
            return False
        if not rank[i] < rank[f[i]]:                                      # (F1)
            return False
        if any(not rank[i] < rank[k] for k in nbrs(E, f[i]) - {i}):       # (F2)
            return False
    return True


def find_flow(V, E, V_in, V_out):
    """Every map f and every total order (a partial order works iff one of its linear
    extensions does)."""
    meas = [v for v in V if v not in V_out]
    choices = [sorted(nbrs(E, i) - set(V_in)) for i in meas]
    for order in permutations(V):
        for fs in product(*choices):
            f = dict(zip(meas, fs))
            if is_flow(E, V_in, V_out, f, list(order)):
                return f, list(order)
    return None


def find_gflow(V, E, V_in, V_out):
    """The chapter's three conditions: u in Odd(g(u)); u < v for v in g(u); u < v for
    v != u in Odd(g(u)).  Every total order, every set of non-inputs."""
    cand = [v for v in V if v not in V_in]
    subsets = [set(c) for r in range(1, len(cand) + 1) for c in combinations(cand, r)]
    for order in permutations(V):
        rank = {v: k for k, v in enumerate(order)}
        g = {}
        for u in (v for v in V if v not in V_out):
            for A in subsets:
                od = odd(E, A)
                if u in od and all(rank[u] < rank[v] for v in A) and \
                        all(rank[u] < rank[v] for v in od - {u}):
                    g[u] = A
                    break
            else:
                break
        else:
            return g, list(order)
    return None


def open_graph(V, E, V_in, V_out):
    """The open graph as a graph-like PyZX diagram (spiders at rest, Hadamard edges)."""
    g = zx.Graph()
    v = {x: g.add_vertex(VT.Z, k, 1) for k, x in enumerate(V)}
    for a, b in E:
        g.add_edge((v[a], v[b]), ET.HADAMARD)
    ins, outs = [], []
    for x in V_in:
        i = g.add_vertex(VT.BOUNDARY, V.index(x), 0)
        g.add_edge((i, v[x]))
        ins.append(i)
    for x in V_out:
        o = g.add_vertex(VT.BOUNDARY, V.index(x), 2)
        g.add_edge((v[x], o))
        outs.append(o)
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


CHAIN = ([1, 2, 3], [(1, 2), (2, 3)])
VEE = (["a", "b", "c"], [("a", "c"), ("b", "c")])


def blackboard():
    # --- the row has a flow, and it is f(1) = 2, f(2) = 3 with 1 < 2 < 3
    V, E = CHAIN
    fl = find_flow(V, E, [], [3])
    assert fl is not None and fl[0] == {1: 2, 2: 3}
    assert is_flow(E, [], [3], {1: 2, 2: 3}, [1, 2, 3])
    assert not is_flow(E, [], [3], {1: 2, 2: 3}, [2, 1, 3])              # read 2 first: no
    assert find_gflow(V, E, [], [3]) is not None
    # --- Result 17.13: the V has no flow and no gFlow, with no input and with a an input
    V, E = VEE
    for V_in in ([], ["a"]):
        assert find_flow(V, E, V_in, ["c"]) is None
        assert find_gflow(V, E, V_in, ["c"]) is None
    # ... because a lit half from a lands on b's end, and from b on a's: Odd({c}) = {a, b}
    assert odd(E, {"c"}) == {"a", "b"}
    # the V with both arms kept is fine: then nothing is read
    assert find_flow(V, E, [], ["a", "b", "c"]) is not None

    # --- Gaussian elimination: B = [[1,1],[0,1]] -> I by adding row 2 to row 1: one CNOT
    B = np.array([[1, 1], [0, 1]])
    E_row = np.array([[1, 1], [0, 1]])                                    # add row 2 to row 1
    assert np.array_equal(E_row @ B % 2, np.eye(2, dtype=int))
    assert not np.array_equal(np.array([[1, 0], [1, 1]]) @ B % 2, np.eye(2, dtype=int))  # row 1 into row 2: no
    # which joiner that row operation is, is Result 17.15 (below and in the sketchbook)

    # --- Result 17.15 on the blackboard: sum over bits
    left = np.zeros((4, 4), dtype=complex)
    for y1, y2, x1, x2 in product((0, 1), repeat=4):
        left[2 * y1 + y2, 2 * x1 + x2] = 2 ** -1.5 * (-1) ** (x1 * y1 + x2 * y1 + x2 * y2)
    assert np.allclose(left, CNOT @ kron(H, H) / R2)
    assert not np.allclose(left, CNOT21 @ kron(H, H) / R2)               # the other way up: no
    # it differs at x = (1,0), y = (0,1)
    other = CNOT21 @ kron(H, H) / R2
    assert not np.isclose(left[0b01, 0b10], other[0b01, 0b10])

    # --- Result 17.16: the new card equals U, T-count 0
    assert np.allclose(kron(S, Z) @ CZ, U)
    assert not np.allclose(CZ @ kron(Z, S), U)                            # quarter on the wrong wire

    pictures()


FOURTH_DUMPS = (
    "pan,slid apart | cols=6 rows=2 y=0..1 near=false folded=false; 1:ring green 0 @1,0; 2:ring green 6 @3,0; "
    "3:ring green 0 @1,1; 4:ring green 12 @3,1; tie 1-3 tubes1 []; cord loose-loose [-0.5,0 5.5,0]; "
    "cord loose-loose [-0.5,1 5.5,1]",
    "net | cols=6 rows=2 y=0..1 near=false folded=false; 1:ring green 0 @1,0; 2:ring green 0 @3,0; 3:ring green 0 @1,1; "
    "4:ring green 0 @3,1; tie 1-2 tubes1 []; tie 2-3 tubes1 []; tie 3-4 tubes1 []; cord loose-1 [-0.5,0 1,0]; "
    "cord loose-2 [5.5,0 3,0]; cord loose-3 [-0.5,1 1,1]; cord loose-4 [5.5,1 3,1]",
    "net,joiner | cols=6 rows=2 y=0..1 near=false folded=false; 1:ring green 0 @1,0; 2:ring green 0 @3,0; "
    "3:ring green 0 @4.2,0; 4:ring green 0 @1,1; 5:ring green 0 @3,1; 6:ring red 0 @4.2,1; tie 1-2 tubes1 []; tie 3-6 []; "
    "tie 4-5 tubes1 []; cord loose-1 [-0.5,0 1,0]; cord loose-2 [5.5,0 3,0]; cord loose-4 [-0.5,1 1,1]; "
    "cord loose-5 [5.5,1 3,1]",
    "v | cols=6 rows=2 y=0..1 near=false folded=false; 1:ring green 0 @1,0; 2:ring green 0 @4,0; 3:ring green 0 @2.5,1; "
    "tie 1-3 tubes1 []; tie 2-3 tubes1 []; cord loose-1 [-0.5,0 1,0]; cord loose-2 [5.5,0 4,0]; cord loose-3 [5.5,1 2.5,1]",
)


def pictures():
    """The Fourth Challenge's prop pictures (identity card, section 4), read from the dialogue
    through tools/dfp_props.py and the transcription tools/props_ch17.py, and held against
    this file's blackboard and pictures (graph_of: a structural reading of each cloth dump,
    not a proof); then the whole dialogue's checks (section 4 and the form check of 1c)."""
    calls = DIALOGUE.check_layout("Fourth", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("outcloth", True, "slid apart", ("the pan's answer",), "The turns slid off"),
        ("outcloth", False, "", ("the second net",), "A harder net"),
        ("outcloth", False, "the joiner laid at the ends", ("the second net",), "Out at the ends"),
        ("outcloth", False, "", ("the V",), "Arms and a corner")]
    PC.check_captions(DIALOGUE, "Fourth", calls)
    PC.check_section(DIALOGUE, "Fourth", calls)
    PC.check_anchors(DIALOGUE, "Fourth", ["P4.1", "P4.2", "P4.3", "P4.4"])
    pics = [PC.picture(c) for c in calls]
    p41, p42, p43, p44 = pics
    assert tuple(p.dump for p in pics) == FOURTH_DUMPS
    assert {p.frame for p in pics} == {1.0}
    g41, g42, g43, g44 = (PC.graph_of(p.dump) for p in pics)

    # P4.1: the pan's answer slid apart is exactly U/sqrt2, and reads as the new card -- a
    # sleeved tie between two rings at rest, then a quarter on the upper line, a half on the lower
    assert np.allclose(matrix_of(g41), U / R2)
    assert same_picture(g41, two_wire([(0, Fraction(1, 2)), (1, 1)]))
    assert same_picture(g41, PC.graph_of(PC.outcloth("", "the pan's answer").dump))   # slid apart: no number
    assert not same_picture(g41, two_wire([(0, 1), (1, Fraction(1, 2))]))             # turns swapped: no
    # P4.2 and P4.3: the second net is the frontier picture with B = [[1,1],[0,1]], and the
    # joiner at the ends unties its crossing tie exactly -- green on the upper line, red below
    B = [[1, 1], [0, 1]]
    assert same_picture(g42, frontier(B)) and same_picture(g43, frontier([[1, 0], [0, 1]], joiner=(VT.Z, VT.X)))
    assert same_picture(g42, g43)
    swapped = p43.dump.replace("3:ring green 0 @4.2,0", "3:ring red 0 @4.2,0").replace("6:ring red 0 @4.2,1",
                                                                                       "6:ring green 0 @4.2,1")
    assert swapped != p43.dump and not same_picture(g42, PC.graph_of(swapped))         # the other way up fails
    assert np.allclose(matrix_of(g42), CNOT @ kron(H, H) / R2)
    # P4.4: the V as an open graph -- its rings the vertices, its ties the edges, the arm with a
    # lead in from the left an input; the arms read and the corner kept: no flow, no gFlow
    nodes, tubes, ties, cords = PC.parse_dump(p44.dump)
    V = sorted(nodes)
    E = [(a, b) for kind, a, b, k in ties]
    assert all(k == 1 for *_, k in ties) and len(E) == 2
    corner = [v for v in V if nodes[v][4] == 1]
    arms = [v for v in V if nodes[v][4] == 0]
    assert len(corner) == 1 and len(arms) == 2
    assert set(E) == {(a, corner[0]) for a in arms}                                    # arms not tied
    V_in = [v for v in V if any(a == "loose" and b == str(v) and min(x0, x1) == -0.5 for a, b, x0, y0, x1, y1 in cords)]
    assert V_in == [arms[0]]
    for vin in ([], V_in):
        assert find_flow(V, E, vin, corner) is None and find_gflow(V, E, vin, corner) is None
    # with the right arm's lead kept as well (nothing read there) a flow exists: it is the
    # reading of both arms that has no order
    outs = [v for v in V if any("loose" in (a, b) and str(v) in (a, b) and max(x0, x1) == 5.5
                                for a, b, x0, y0, x1, y1 in cords)]
    assert sorted(outs) == sorted([arms[1], corner[0]])
    assert find_flow(V, E, V_in, outs) is not None
    assert g44.num_inputs() == 1 and g44.num_outputs() == 2

    # the captions' truth: the turns slid off; the crossing tie present; the joiner at the ends;
    # the V
    assert p41.state.apart and p41.dump.startswith("pan,slid apart")
    n42 = PC.parse_dump(p42.dump)[0]
    assert any(n42[a][3] == 3 and n42[a][4] == 0 and n42[b][3] == 1 and n42[b][4] == 1 for _, a, b, _ in PC.parse_dump(p42.dump)[2])
    assert p43.state.joiner and p43.dump.startswith("net,joiner")
    assert p44.state.start == "v"
    # fixed parts: the second net's four rings at columns 1 and 3, in both its pictures
    four = lambda p: sorted((x, y) for k, c, t, x, y in PC.parse_dump(p.dump)[0].values() if x in (1, 3))
    assert four(p42) == four(p43) == [(1, 0), (1, 1), (3, 0), (3, 1)]
    # the keyed words (P4.1): the kit's tube, ring and lead (the kit places them and is not
    # re-measured here; the identity card's section 4 gives them clear by ring 0.034, tube 0.072,
    # lead 0.038 cm, ink to ink); the Lua sets no word of its own here
    assert p41.labels == ("tube", "ring", "lead") and all(p.labels == () for p in (p42, p43, p44))
    assert PC.own_words(p41) == () and PC.check_keyed_words(p41) == {}
    # the Lua's refusals
    PC.TEX_ERRORS.clear()
    PC.out_state("slid apart", "the V")
    PC.out_state("the joiner laid at the ends", "the pan's answer")
    assert [k for k, _ in PC.TEX_ERRORS] == ["out-apart-pan", "out-joiner-net"]
    PC.TEX_ERRORS.clear()

    # --- the whole dialogue (card section 4): the sections in order, every picture inside one;
    # the starred form on the first picture of each macro; no negative anchor in either file
    assert DIALOGUE.check_whole(ORD) == 16
    DIALOGUE.keyed_first(ORD, kind=PC.keyed_kind)
    starred = [(o, c.caption) for o in ORD for b in DIALOGUE.blocks(o) for c in b if c.keyed]
    assert starred == [("First", "The card, laid out"), ("Second", "What the pan made"), ("Third", "Every lid lifted"),
                       ("Third", "Ends waiting"), ("Fourth", "The turns slid off")]
    PC.check_negative_anchors(DIALOGUE)
    # the form check of card section 1c, on both files
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        ok, seen = PC.form_check(pathlib.Path(path).read_text(encoding="utf-8"))
        assert ok, seen
        assert seen["together"] == [("First Challenge", "Tortoise"), ("Second Challenge", "Tortoise"),
                                    ("Third Challenge", "Tortoise")]
        assert seen["apart"] == [("Fourth Challenge", "Achilles")] and seen["near"] == []
        assert seen["koekpan"] >= 1
    # the greps of card section 1c: the signal (none), the hooks (none), no Genie, no gloss
    text = pathlib.Path(DIALOGUE.dialog).read_text(encoding="utf-8")
    assert not re.search(r"(?<![A-Za-z0-9_])(seventeen|seventeenth|17)(?![A-Za-z0-9_])", text, re.I)
    assert text.count("Let Bob draw") == 0 and text.count("**Genie") == 0
    here = pathlib.Path(__file__).resolve().parents[2]
    gloss = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(PC.GLOSS) + r")(?![A-Za-z0-9_])", re.I)
    for f in [here / "dialog.md", here / "chapter.md"] + sorted(p for p in (here / "Code").rglob("*") if p.is_file()):
        assert not gloss.search(f.read_text(encoding="utf-8", errors="replace")), f


def frontier(B, joiner=None):
    """Back spiders on the inputs, frontier spiders on the outputs, a Hadamard edge for
    each 1 in B (rows: frontier, columns: back); then optionally a bare joiner at the ends."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, q, 0) for q in (0, 1)]
    back = [g.add_vertex(VT.Z, q, 1) for q in (0, 1)]
    front = [g.add_vertex(VT.Z, q, 2) for q in (0, 1)]
    for q in (0, 1):
        g.add_edge((ins[q], back[q]))
    for r, c in product((0, 1), repeat=2):
        if B[r][c]:
            g.add_edge((back[c], front[r]), ET.HADAMARD)
    last = list(front)
    if joiner is not None:                          # joiner = (colour on q0, colour on q1)
        j = [g.add_vertex(joiner[q], q, 3) for q in (0, 1)]
        for q in (0, 1):
            g.add_edge((front[q], j[q]))
        g.add_edge((j[0], j[1]))
        last = j
    outs = [g.add_vertex(VT.BOUNDARY, q, 4) for q in (0, 1)]
    for q in (0, 1):
        g.add_edge((last[q], outs[q]))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


def two_wire(ops):
    """A Hadamard edge between two spiders at rest, then the given (qubit, phase) greens."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, q, 0) for q in (0, 1)]
    a, b = g.add_vertex(VT.Z, 0, 1), g.add_vertex(VT.Z, 1, 1)
    g.add_edge((ins[0], a))
    g.add_edge((ins[1], b))
    g.add_edge((a, b), ET.HADAMARD)
    last, col = [a, b], 2
    for q, ph in ops:
        v = g.add_vertex(VT.Z, q, col, phase=Fraction(ph))
        g.add_edge((last[q], v))
        last[q], col = v, col + 1
    outs = [g.add_vertex(VT.BOUNDARY, q, col) for q in (0, 1)]
    for q in (0, 1):
        g.add_edge((last[q], outs[q]))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


def uncooked_net():
    """The master, as in FourEighths.py; returns the graph and its named spiders."""
    g = zx.Graph()
    n = {}
    in1, in2 = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0)
    n["small1"], n["small2"], n["big"] = (g.add_vertex(VT.Z, 0, c) for c in (1, 2, 5))
    for k in range(1, 5):
        n[f"e{k}"] = g.add_vertex(VT.Z, 1, 2 * k - 1, phase=Fraction(1, 4))
        n[f"p{k}"] = g.add_vertex(VT.Z, 1, 2 * k)
    out1, out2 = g.add_vertex(VT.BOUNDARY, 0, 10), g.add_vertex(VT.BOUNDARY, 1, 10)
    E = ([("small1", "small2"), ("small2", "big")]
         + [(f"e{k}", f"p{k}") for k in range(1, 5)]
         + [(f"p{k}", f"e{k + 1}") for k in range(1, 4)]
         + [("big", f"p{k}") for k in range(1, 5)])
    for a, b in E:
        g.add_edge((n[a], n[b]), ET.HADAMARD)
    g.add_edge((in1, n["small1"]))
    g.add_edge((n["big"], out1))
    g.add_edge((in2, n["e1"]))
    g.add_edge((n["p4"], out2), ET.HADAMARD)
    g.set_inputs((in1, in2))
    g.set_outputs((out1, out2))
    return g, n, E


def vee(ends, half_on_output=False):
    """The V closed with ends turned ends[0], ends[1] on a, b; c's output left open."""
    g = zx.Graph()
    a, b, c = g.add_vertex(VT.Z, 0, 1), g.add_vertex(VT.Z, 2, 1), g.add_vertex(VT.Z, 1, 2)
    ea = g.add_vertex(VT.Z, 0, 0, phase=Fraction(ends[0]))
    eb = g.add_vertex(VT.Z, 2, 0, phase=Fraction(ends[1]))
    g.add_edge((ea, a))
    g.add_edge((eb, b))
    g.add_edge((a, c), ET.HADAMARD)
    g.add_edge((b, c), ET.HADAMARD)
    o = g.add_vertex(VT.BOUNDARY, 1, 4)
    if half_on_output:
        x = g.add_vertex(VT.X, 1, 3, phase=Fraction(1))
        g.add_edge((c, x))
        g.add_edge((x, o))
    else:
        g.add_edge((c, o))
    g.set_inputs(())
    g.set_outputs((o,))
    return g


def sketchbook():
    rng = np.random.default_rng(1706)

    # --- the V on the cloth: a lit half on a's end travels through c (at rest) and lands on
    #     b's end, and on c's output as a red half; exactly, with no number
    for _ in range(3):
        pa, pb = (Fraction(int(k), 97) for k in rng.integers(1, 194, size=2))
        lhs = vee((pa + 1, pb))
        rhs = vee((pa, pb + 1), half_on_output=True)
        assert same_picture(lhs, rhs)
        assert not same_picture(lhs, vee((pa, pb + 1)))                  # the output's half is owed
        assert not same_picture(lhs, vee((pa, pb), half_on_output=True))  # and so is b's
        # and the same picture turned over: b's half lands on a's end
        assert same_picture(vee((pa, pb + 1)), vee((pa + 1, pb), half_on_output=True))

    # --- the row and the V, through PyZX's gFlow finder: the same verdicts
    assert gflow(open_graph(*CHAIN, [], [3])) is not None
    assert gflow(open_graph(*VEE, [], ["c"])) is None
    assert gflow(open_graph(*VEE, ["a"], ["c"])) is None

    # --- Result 17.15: a joiner at the ends
    B = [[1, 1], [0, 1]]
    lhs = frontier(B)
    assert np.allclose(matrix_of(frontier([[1, 0], [0, 1]])), kron(H, H))       # exactly H (x) H
    good = frontier([[1, 0], [0, 1]], joiner=(VT.Z, VT.X))                         # green up, red down
    flip = frontier([[1, 0], [0, 1]], joiner=(VT.X, VT.Z))
    assert same_picture(lhs, good)
    assert not same_picture(lhs, flip)
    assert np.allclose(matrix_of(lhs), CNOT @ kron(H, H) / R2)
    # DERIVED (checked by the rules, as validation is): lhs followed by the adjoint of the
    # right side cooks to two bare wires, with the number 1/2 (each side is 1/sqrt2 a unitary)
    v = lhs.copy()
    v.compose(good.adjoint())
    simp.full_reduce(v)
    assert all(v.type(u) == VT.BOUNDARY for u in v.vertices()) and v.num_edges() == 2
    assert np.allclose(matrix_of(v), np.eye(4) / 2)
    w = lhs.copy()
    w.compose(flip.adjoint())
    simp.full_reduce(w)
    assert not np.allclose(matrix_of(w), np.eye(4) / 2)

    # --- Result 17.16: "Slide them apart." -- the cooked picture's turns slide off its spiders
    apart = two_wire([(0, Fraction(1, 2)), (1, 1)])
    fused = apart.copy()
    assert simp.spider_simp(fused)                                          # DERIVED: fusion, row 57
    sp = [u for u in fused.vertices() if fused.type(u) != VT.BOUNDARY]
    assert len(sp) == 2
    up = next(u for u in sp if fused.connected(u, fused.inputs()[0]))
    down = next(u for u in sp if fused.connected(u, fused.inputs()[1]))
    assert fused.phase(up) == Fraction(1, 2) and fused.phase(down) == 1
    assert fused.edge_type(fused.edge(up, down)) == ET.HADAMARD
    assert same_picture(fused, apart)
    assert np.allclose(matrix_of(two_wire([])), CZ / R2)                     # the tie: CZ/sqrt2
    assert np.allclose(matrix_of(apart), kron(S, Z) @ CZ / R2)
    assert np.allclose(matrix_of(apart) * R2, U)                             # the new card is U
    assert not same_picture(apart, two_wire([(0, 1), (1, Fraction(1, 2))]))  # turns swapped: no

    # --- the master: the causal flow of Section 17.4, and PyZX's gFlow (Lemma 3.7)
    net, n, E = uncooked_net()
    assert net.num_edges() == 17 and np.allclose(matrix_of(net), U / 4)   # the master, as in FourEighths.py
    V_in, V_out = ["small1", "e1"], ["big", "p4"]
    f = {"small1": "small2", "small2": "big"}
    f.update({f"e{k}": f"p{k}" for k in range(1, 5)})
    f.update({f"p{k}": f"e{k + 1}" for k in range(1, 4)})
    lower = [x for k in range(1, 5) for x in (f"e{k}", f"p{k}")]
    order = ["small1", "small2"] + lower + ["big"]
    assert is_flow(E, V_in, V_out, f, order)
    assert not is_flow(E, V_in, V_out, f, ["big"] + order[:-1])             # big first: no
    assert gflow(net) is not None
    # extraction of the cooked master: cz; rz(pi) on q1; rz(pi/2) on q0 -- exactly U
    h = net.copy()
    simp.full_reduce(h)
    ce = zx.extract_circuit(h.copy())
    assert np.allclose(matrix_of(ce.to_graph()), U)
    assert sorted((gt.name, gt.target, str(getattr(gt, "phase", ""))) for gt in ce.gates) == \
        sorted([("CZ", 1, ""), ("ZPhase", 1, "1"), ("ZPhase", 0, "1/2")])

    # --- the 20 random circuits of CookingDown.py
    gadget_seeds, exact = [], 0
    for k in range(1, 21):
        random.seed(k)
        np.random.seed(k)
        g = zx.generate.cliffordT(3, 40, p_t=0.2)
        M = matrix_of(g)
        r_g = g.copy()
        simp.full_reduce(r_g)
        has_gadget = any(r_g.type(u) != VT.BOUNDARY and r_g.vertex_degree(u) == 1
                         for u in r_g.vertices())
        if has_gadget:
            gadget_seeds.append(k)
            # PyZX's finder finds no ORDINARY gFlow on a net that kept a gadget (Section 17.4);
            # nothing is claimed here about Backens et al.'s extended notion (not read)
            assert gflow(r_g) is None
        else:
            assert gflow(r_g) is not None                                   # Duncan et al. Cor. 5.5
        ce = zx.extract_circuit(r_g.copy())                                 # succeeds on all 20
        Mx = matrix_of(ce.to_graph())
        # THE LICENSED BLOCK: extraction gives the circuit back only up to a global phase
        # (Duncan et al. Remark 2.2 neglect non-zero scalars; for unitaries |z| = 1)
        idx = np.unravel_index(np.argmax(np.abs(M)), M.shape)
        r = Mx[idx] / M[idx]
        assert np.isclose(abs(r), 1)
        assert np.allclose(Mx, r * M)
        exact += int(np.allclose(Mx, M))
    assert gadget_seeds == [6, 8, 10, 18]
    assert exact == 0                                                       # none exact, here


if __name__ == "__main__":
    verdict("OutAgain", blackboard, sketchbook)
