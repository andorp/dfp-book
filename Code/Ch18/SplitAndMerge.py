"""Challenge V: Split and Merge  (lattice surgery; the synthesis)
================================================================
"A joiner again."  -- Crab
"Everything tonight was one ring, asked in two colours."  -- Tortoise

Claim: splits and merges are spiders, exactly (de Beaudrap and Horsman, Lemma 1; Result
18.21); a merge's negative branch is its positive branch after a half of the other colour on
its first incoming leg (Lemma 3, Corollary 4, first convention; Result 18.22); and the joiner
is a smooth split then a rough merge: (I (x) K_j,ro)(U_sm (x) I) = (Z^j (x) I) CNOT/sqrt2, each
branch with probability 1/2 for every input, the negative one mended by a green half on the
upper wire (Result 18.23).  A bare merge is not always a coin: on |++> it is certain.
"""
import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from collections import Counter
from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import I2, ket, kron, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch18 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)  # control on top
plus, minus = (ket(0) + ket(1)) / np.sqrt(2), (ket(0) - ket(1)) / np.sqrt(2)


def op(*pairs):
    """sum of |out><in| for (out, in) pairs of vectors."""
    return sum(np.outer(o, i.conj()) for o, i in pairs)


pp, mm = kron(plus, plus).ravel(), kron(minus, minus).ravel()
mp, pm = kron(minus, plus).ravel(), kron(plus, minus).ravel()

U_SM = op((ket(0, 0), ket(0)), (ket(1, 1), ket(1)))                   # smooth split = delta
U_RO = op((pp, plus), (mm, minus))                                    # rough split
K0_SM = op((ket(0), ket(0, 0)), (ket(1), ket(1, 1)))                  # smooth merge, j = 0
K1_SM = op((ket(0), ket(1, 0)), (ket(1), ket(0, 1)))                  #               j = 1
K0_RO = op((plus, pp), (minus, mm))                                   # rough merge,  j = 0
K1_RO = op((plus, mp), (minus, pm))                                   #               j = 1
ENC = op((ket(0, 0, 0), ket(0)), (ket(1, 1, 1), ket(1)))


def joiner(j):
    """(I (x) K_j,ro)(U_sm (x) I): split the upper wire, merge its second half with the lower."""
    K = K0_RO if j == 0 else K1_RO
    return kron(I2, K) @ kron(U_SM, I2)


def blackboard():
    rng = np.random.default_rng(1851)

    # --- the negative branches, first convention: the merge after a flip of its first input
    assert np.allclose(K1_SM, K0_SM @ kron(X, I2))
    assert np.allclose(K1_RO, K0_RO @ kron(Z, I2))
    assert not np.allclose(K1_RO, K0_RO @ kron(I2, Z))       # the other convention differs...
    assert np.allclose(Z @ K1_RO, K0_RO @ kron(I2, Z))       # ...by a half on the child
    # each merge is one channel with two Kraus operators
    for K0, K1 in ((K0_SM, K1_SM), (K0_RO, K1_RO)):
        assert np.allclose(K0.conj().T @ K0 + K1.conj().T @ K1, np.eye(4))
    # the smooth merge asks whether two messages agree: outcome a xor b, with certainty
    for a in (0, 1):
        for b in (0, 1):
            assert np.isclose(np.linalg.norm(K0_SM @ ket(a, b)) ** 2, float(a == b))
            assert np.isclose(np.linalg.norm(K1_SM @ ket(a, b)) ** 2, float(a != b))
    # the split is the copier: U_sm = delta, and it carries a Z through: U_sm Z = (I (x) Z) U_sm
    assert np.allclose(U_SM, np.outer(ket(0, 0), ket(0)) + np.outer(ket(1, 1), ket(1)))
    assert np.allclose(U_SM @ Z, kron(I2, Z) @ U_SM)

    # --- Result 18.23: the joiner by surgery, both branches, exact
    for a in (0, 1):                                         # the rough merge: |ab> -> |a xor b>/sqrt2
        for b in (0, 1):
            assert np.allclose(K0_RO @ ket(a, b), ket(a ^ b) / np.sqrt(2))
    B0, B1 = joiner(0), joiner(1)
    assert np.allclose(B0, CNOT / np.sqrt(2))
    assert np.allclose(B1, kron(Z, I2) @ CNOT / np.sqrt(2))
    assert np.allclose(B1, CNOT @ kron(Z, I2) / np.sqrt(2))  # Z on the control commutes
    assert not np.allclose(B0, CNOT)                         # the 1/sqrt2 is the point
    assert not np.allclose(B1, B0)                           # the negative branch is not the joiner...
    assert np.allclose(kron(Z, I2) @ B1, B0)                 # ...until a green half mends it
    assert np.allclose(B0.conj().T @ B0 + B1.conj().T @ B1, np.eye(4))   # the branches complete
    for B in (B0, B1):                                       # probability 1/2 for EVERY input:
        assert np.allclose(B.conj().T @ B, np.eye(4) / 2)
    # ...on five random product inputs and one random entangled input (all non-real)
    inputs = [np.kron(random_state(rng), random_state(rng)) for _ in range(5)]
    ent = random_state(rng, 4)
    assert np.linalg.matrix_rank(ent.reshape(2, 2)) == 2    # entangled: not a product
    inputs.append(ent)
    for psi in inputs:
        assert not np.allclose(psi, psi.real)
        for B in (B0, B1):
            assert np.isclose(np.linalg.norm(B @ psi) ** 2, 0.5)

    # --- a merge is not always a coin: a bare rough merge on |++> is certain
    assert np.allclose(K0_RO @ pp, plus)
    assert np.allclose(K1_RO @ pp, 0)
    assert np.isclose(np.linalg.norm(K0_RO @ pp) ** 2, 1)

    pictures()


# ------------------------------------------------------------------------------ prop pictures

FIFTH_DUMPS = (
    "joiner | cols=6 rows=3 y=0..2 near=false folded=false; gaps=1.5; 1:ring green 0 @1,0; 2:ring red 0 @3,2; "
    "tie 1-2 []; cord loose-loose [-0.3,0 5.3,0]; cord loose-loose [-0.3,2 5.3,2]",
    "out | cols=6 rows=3 y=0..2 near=false folded=false; gaps=1.5; 1:ring green 0 @1,0; 2:ring red 0 @3,2; "
    "cord 1-2 [1,0 1,1 3,1 3,2]; cord loose-loose [-0.3,0 5.3,0]; cord loose-loose [-0.3,2 5.3,2]",
    "out,half,into,apart | cols=6 rows=3 y=0..2 near=false folded=false; gaps=1.5; 1:ring green 0 @1,0; "
    "2:ring green 12 @2,0; 3:ring red 0 @3,2; cord 1-3 [1,0 1,1 3,1 3,2]; cord loose-loose [-0.3,0 5.3,0]; "
    "cord loose-loose [-0.3,2 5.3,2]",
    "out,half,into,apart,crab,together | cols=6 rows=3 y=0..2 near=false folded=false; gaps=1.5; "
    "1:ring green 0 @1,0; 2:ring red 0 @3,2; cord 1-2 [1,0 1,1 3,1 3,2]; cord loose-loose [-0.3,0 5.3,0]; "
    "cord loose-loose [-0.3,2 5.3,2]",
)


def pictures():
    """The Fifth Challenge's prop pictures (identity card, section 4), read from the dialogue
    through tools/dfp_props.py and the transcription tools/props_ch18.py, and held against this
    file's joiner pictures (graph_of: a structural reading of each cloth dump, not a proof);
    then the checks of card section 4 on the whole dialogue."""
    calls = DIALOGUE.check_layout("Fifth", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("splitcloth", True, "", (), "Clipped across"),
        ("splitcloth", False, "drawn out", (), "Drawn out long"),
        ("splitcloth", False, "drawn out, half on the string, slid into the split, slid apart", (), "Slid back and apart"),
        ("splitcloth", False, "drawn out, half on the string, slid into the split, slid apart, crab's half beside, "
                              "slid together", (), "Nothing left over")]
    PC.check_captions(DIALOGUE, "Fifth", calls)
    PC.check_section(DIALOGUE, "Fifth", calls)
    PC.check_anchors(DIALOGUE, "Fifth", ["P5.1", "P5.2", "P5.3", "P5.4"])
    pics = [PC.picture(c) for c in calls]
    p51, p52, p53, p54 = pics
    assert tuple(p.dump for p in pics) == FIFTH_DUMPS
    steps = ["drawn out", "half on the string", "slid into the split", "slid apart", "crab's half beside",
             "slid together"]
    every = [PC.splitcloth(", ".join(steps[:k])) for k in range(7)]
    assert {s.frame for s in every} == {p51.frame}                 # one frame in every state

    # P5.1: the joiner clipped across is the bare joiner, CNOT / sqrt2, exactly
    bare = bare_cnot_picture()
    g = [PC.graph_of(p.dump) for p in pics]
    assert same_picture(g[0], bare) and np.allclose(matrix_of(g[0]), CNOT / np.sqrt(2))
    assert not np.allclose(matrix_of(g[0]), CNOT)                  # the joiner's number is in the picture
    # P5.2: drawn out long changes only how it lies: the same picture, exactly; a split and a merge
    assert same_picture(g[1], bare) and same_picture(g[1], surgery(0))
    # the unpictured steps: the half on the string, and slid into the split, each (Z (x) I) CNOT / sqrt2
    zc = kron(Z, I2) @ CNOT / np.sqrt(2)
    assert np.allclose(zc, CNOT @ kron(Z, I2) / np.sqrt(2))       # the half slides through the split
    for k in (2, 3):
        gk = PC.graph_of(every[k].dump)
        assert np.allclose(matrix_of(gk), zc) and not same_picture(gk, bare)
    assert same_picture(PC.graph_of(every[2].dump), surgery(1))
    # P5.3: slid back and apart: the half on the upper line, (Z (x) I) CNOT / sqrt2, exactly; the
    # tails branch before the Crab's hand
    assert same_picture(g[2], surgery(1, half_on_upper=True)) and np.allclose(matrix_of(g[2]), zc)
    assert not same_picture(g[2], bare)
    # P5.4: the Crab's half beside hers, slid together: nothing left over, the joiner again
    assert same_picture(g[3], surgery(1, mend=True, half_on_upper=True))
    assert same_picture(g[3], bare)
    crab = PC.graph_of(every[5].dump)                              # beside, not yet slid: the same
    assert same_picture(crab, bare)

    # captions against their dumps
    head = [p.dump.split(" | ")[0].split(",") for p in pics]
    assert head[0] == ["joiner"] and head[1][-1] == "out"          # "Clipped across", "Drawn out long"
    assert "apart" in head[2] and "together" in head[3]           # "Slid back and apart", "Nothing left over"
    assert p54.dump.split(" | ")[1] == p52.dump.split(" | ")[1]    # the table equal to out's
    # fixed parts: the split's green ring at (1,0) and the merge's red ring at (3,2) in every picture
    for p in pics + every:
        places = {(c, x, y) for kind, c, t, x, y in PC.parse_dump(p.dump)[0].values()}
        assert ("green", 1.0, 0.0) in places and ("red", 3.0, 2.0) in places
    nodes = PC.parse_dump(p53.dump)[0]                             # the half just past the split
    assert sorted((c, t, x, y) for kind, c, t, x, y in nodes.values()) == [
        ("green", 0, 1.0, 0.0), ("green", 12, 2.0, 0.0), ("red", 0, 3.0, 2.0)]

    # the keyed word string is the cloth kit's own placement (its no-room fault is the check)
    assert p51.labels == ("string",) and PC.own_words(p51) == () and all(p.labels == () for p in pics[1:])
    # the words split and merge, which the Lua sets itself in the starred form once the string is
    # drawn out (no dialogue picture draws them): measured clear in every such state, at the
    # compiled boxes.  With the half on the string, split stands below the string's first
    # corner, clear of the half (it once sat on it: a near-touch the Lua has since moved).
    for k in range(1, 7):
        pic = PC.splitcloth(", ".join(steps[:k]), True)
        assert [w for w, *_ in PC.own_words(pic)] == ["split", "merge"]
        PC.check_compiled_boxes(pic)
        assert min(PC.check_keyed_words(pic).values()) >= PC.LETTER_FLOOR
    corner = PC.own_words(PC.splitcloth("drawn out, half on the string", True))[0]
    usual = PC.own_words(PC.splitcloth("drawn out", True))[0]
    assert corner[3] == "north east" and usual[3] == "north west" and corner[2] < usual[2]

    # the Lua's refusals: a step out of order; an unknown step
    PC.TEX_ERRORS.clear()
    PC.split_state("half on the string")
    PC.split_state("drawn out, half on the penny")
    assert [k for k, _ in PC.TEX_ERRORS] == ["split-order", "split-step"]
    PC.TEX_ERRORS.clear()

    # --- the whole dialogue (card section 4): the sections in order, every picture inside one;
    # the starred form on the first picture of each macro; no negative anchor in either file
    ORD = PC.ORDER
    assert DIALOGUE.check_whole(ORD) == 17
    DIALOGUE.keyed_first(ORD, kind=PC.keyed_kind)
    starred = [(o, c.macro) for o in ORD for b in DIALOGUE.blocks(o) for c in b if c.keyed]
    assert starred == [("First", "plaintin"), ("First", "partcloth"), ("First", "stormline"),
                       ("Second", "linecloth"), ("Third", "tubecloth"), ("Fifth", "splitcloth")]
    PC.check_negative_anchors(DIALOGUE)
    every_call = [c for o in ORD for b in DIALOGUE.blocks(o) for c in b]
    for c in every_call:                                           # the Cat is never drawn or named
        assert not re.search(r"\bcat\b", c.caption + " " + " ".join(PC.picture(c).labels), re.I)
    # the form check of card section 1c, the signal grep, the hooks, the Genie, the Cat, on both files
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        text = pathlib.Path(path).read_text(encoding="utf-8")
        E, seen = PC.form_check(text)
        assert E == [], E
        assert seen["subject"] == [("Second Challenge", "Tortoise"), ("Second Challenge", "Crab"),
                                   ("Second Challenge", "Achilles"), ("Fourth Challenge", "Crab")]
        assert seen["inversion"] == [("Fourth Challenge", "Crab")]
        assert seen["speakers"] == ["Achilles", "Cat", "Crab", "Tortoise"]
        assert not re.search(r"\b(eighteen|eighteenth|18)\b", text, re.I)
        assert text.count("Let Bob draw") == 0 and text.count("**Genie") == 0
        assert PC.cat_count(text) == 4


# ------------------------------------------------------------------------------ pictures

def three_legged(colour, n_in, phase_on_first_in=None, half_colour=None):
    """A spider at rest with n_in legs in and 3 - n_in out; optionally a two-legged half of
    `half_colour` on its first incoming leg (Result 18.22)."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in range(n_in)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 4) for r in range(3 - n_in)]
    v = g.add_vertex(colour, 0.5, 2)
    for k, b in enumerate(ins):
        if k == phase_on_first_in:
            half = g.add_vertex(half_colour, 0, 1, phase=Fraction(1))
            g.add_edge((b, half))
            g.add_edge((half, v))
        else:
            g.add_edge((b, v))
    for b in outs:
        g.add_edge((v, b))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


def surgery(j, mend=False, half_on_upper=False):
    """The joiner by surgery: a green split on the upper wire, its second leg into a red merge
    with the lower wire.  j = 1 puts a green half on the merge's first incoming leg; mend puts
    a green half on the upper wire after it; half_on_upper puts the negative branch's half on
    the split's upper output instead (the chapter's first picture)."""
    g = zx.Graph()
    i0, i1 = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0)
    o0, o1 = g.add_vertex(VT.BOUNDARY, 0, 6), g.add_vertex(VT.BOUNDARY, 1, 6)
    split = g.add_vertex(VT.Z, 0, 1)
    merge = g.add_vertex(VT.X, 1, 3)
    g.add_edge((i0, split))
    g.add_edge((i1, merge))
    g.add_edge((merge, o1))
    if j == 1 and not half_on_upper:
        h = g.add_vertex(VT.Z, 0.5, 2, phase=Fraction(1))
        g.add_edge((split, h))
        g.add_edge((h, merge))
    else:
        g.add_edge((split, merge))
    upper = [split]
    if j == 1 and half_on_upper:
        upper.append(g.add_vertex(VT.Z, 0, 4, phase=Fraction(1)))
    if mend:
        upper.append(g.add_vertex(VT.Z, 0, 5, phase=Fraction(1)))
    upper.append(o0)
    for a, b in zip(upper, upper[1:]):
        g.add_edge((a, b))
    g.set_inputs((i0, i1))
    g.set_outputs((o0, o1))
    return g


def merge_on_plus_plus(j):
    g = zx.Graph()
    o = g.add_vertex(VT.BOUNDARY, 0, 4)
    mg = g.add_vertex(VT.X, 0.5, 2)
    first, second = g.add_vertex(VT.Z, 0, 0), g.add_vertex(VT.Z, 1, 0)
    if j == 1:
        half = g.add_vertex(VT.Z, 0, 1, phase=Fraction(1))
        g.add_edge((first, half))
        g.add_edge((half, mg))
    else:
        g.add_edge((first, mg))
    g.add_edge((second, mg))
    g.add_edge((mg, o))
    g.set_inputs(())
    g.set_outputs((o,))
    g.scalar.add_power(-2)                               # the two points normalized: |++>
    return g


def bare_cnot_picture():
    c = zx.Circuit(2)
    c.add_gate("CNOT", 0, 1)
    g = c.to_graph()
    assert np.allclose(matrix_of(g), CNOT)                  # to_graph() carries sqrt2: exact CNOT
    g.scalar = zx.Scalar()                                  # the bare joiner: CNOT / sqrt2
    return g


def shape(g):
    """What a picture is made of: its spiders (colour, phase), its edge count and edge types."""
    spiders = Counter((g.type(v), Fraction(g.phase(v))) for v in g.vertices() if g.type(v) != VT.BOUNDARY)
    return spiders, g.num_edges(), Counter(g.edge_type(e) for e in g.edges())


def sketchbook():
    # --- Result 18.21: the four splits and merges are spiders, exactly, no number
    assert np.allclose(matrix_of(three_legged(VT.Z, 1)), U_SM)          # green 1 -> 2
    assert np.allclose(matrix_of(three_legged(VT.Z, 2)), K0_SM)         # green 2 -> 1
    assert np.allclose(matrix_of(three_legged(VT.X, 2)), K0_RO)         # red 2 -> 1
    assert np.allclose(matrix_of(three_legged(VT.X, 1)), U_RO)          # red 1 -> 2
    assert not same_picture(three_legged(VT.Z, 1), three_legged(VT.X, 1))   # the colours differ
    # --- Result 18.22: the negative branches, a half of the other colour on the first leg in
    assert np.allclose(matrix_of(three_legged(VT.X, 2, 0, VT.Z)), K1_RO)
    assert np.allclose(matrix_of(three_legged(VT.Z, 2, 0, VT.X)), K1_SM)
    assert not np.allclose(matrix_of(three_legged(VT.X, 2, 1, VT.Z)), K1_RO)   # the other convention

    # --- Result 18.23, the positive branch: the surgery picture is the joiner, CNOT/sqrt2
    s0 = surgery(0)
    assert np.allclose(matrix_of(s0), CNOT / np.sqrt(2))
    circ = bare_cnot_picture()
    assert same_picture(s0, circ)
    # DERIVED: full_reduce brings the surgery picture and the circuit picture to the same picture
    a, b = s0.copy(), circ.copy()
    simp.full_reduce(a)
    simp.full_reduce(b)
    assert same_picture(a, b) and same_picture(a, s0)
    assert shape(a) == shape(b)

    # --- the negative branch: a green half on the merge's first incoming leg
    s1 = surgery(1)
    assert np.allclose(matrix_of(s1), kron(Z, I2) @ CNOT / np.sqrt(2))
    assert not same_picture(s1, s0)
    up = surgery(1, half_on_upper=True)                  # the chapter's first picture: the half
    assert same_picture(s1, up)                          # slid back onto the upper wire
    # DERIVED: fusion (row 57) slides both halves into the split; the two pictures become one
    x, y = s1.copy(), up.copy()
    assert simp.spider_simp(x) and simp.spider_simp(y)
    assert shape(x) == shape(y)
    assert Counter({(VT.Z, Fraction(1)): 1, (VT.X, Fraction(0)): 1}) == shape(x)[0]
    # the negative branch as the circuit: the bare joiner then a green pi on qubit 0
    c = zx.Circuit(2)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("Z", 0)
    cz = c.to_graph()
    cz.scalar = zx.Scalar()
    assert same_picture(s1, cz)
    a, b = s1.copy(), cz.copy()
    simp.full_reduce(a)
    simp.full_reduce(b)
    assert same_picture(a, b) and shape(a) == shape(b)

    # --- the mending: a green half on the upper wire after the negative branch.
    # DERIVED: the two halves fuse into the split, pi + pi = 0: the bare joiner, spider for spider
    m = surgery(1, mend=True)
    h = m.copy()
    assert simp.spider_simp(h)
    ref = s0.copy()
    simp.spider_simp(ref)
    assert shape(h) == shape(ref)
    assert Counter({(VT.Z, Fraction(0)): 1, (VT.X, Fraction(0)): 1}) == shape(h)[0]
    assert same_picture(h, s0) and np.allclose(matrix_of(m), CNOT / np.sqrt(2))

    # --- probability 1/2 for every input, in picture form: each branch followed by its mirror
    # reduces to two bare wires with the number 1/2 (DERIVED by full_reduce)
    for j in (0, 1):
        g = surgery(j)
        gg = g.copy()
        gg.compose(g.adjoint())                          # B^dag after B
        assert np.allclose(matrix_of(gg), np.eye(4) / 2)
        r = gg.copy()
        simp.full_reduce(r)
        assert all(r.type(v) == VT.BOUNDARY for v in r.vertices()) and r.num_edges() == 2
        assert np.isclose(r.scalar.to_number(), 0.5)
    # a bare rough merge on |++> is certain.  |+> is a green point at rest divided by sqrt2
    # (row 53); two of them into the red merge: the positive branch gives |+>, the negative
    # branch (a green half on the first incoming leg) gives nothing
    for j in (0, 1):
        g = merge_on_plus_plus(j)
        assert np.allclose(matrix_of(g).ravel(), (K0_RO if j == 0 else K1_RO) @ pp)
        h = g.copy()
        simp.full_reduce(h)                                              # DERIVED
        if j == 0:
            assert same_picture(h, g) and np.isclose(np.linalg.norm(matrix_of(h)), 1)
        else:
            assert h.scalar.is_zero

    # --- the synthesis: one kind of ring.  The split is the copier delta and the parting ring
    # is the same spider with one more leg
    assert np.allclose(matrix_of(three_legged(VT.Z, 1)), np.outer(ket(0, 0), ket(0)) + np.outer(ket(1, 1), ket(1)))
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 1, 0)
    v = g.add_vertex(VT.Z, 1, 1)
    outs = [g.add_vertex(VT.BOUNDARY, k, 3) for k in range(3)]
    g.add_edge((i, v))
    for ob in outs:
        g.add_edge((v, ob))
    g.set_inputs((i,))
    g.set_outputs(tuple(outs))
    assert np.allclose(matrix_of(g), ENC)
    # DERIVED: the parting ring is a split followed by a split (fusion backwards, row 57)
    g2 = zx.Graph()
    i = g2.add_vertex(VT.BOUNDARY, 1, 0)
    s1v, s2v = g2.add_vertex(VT.Z, 0.5, 1), g2.add_vertex(VT.Z, 1.5, 2)
    outs = [g2.add_vertex(VT.BOUNDARY, k, 3) for k in range(3)]
    for u, w in ((i, s1v), (s1v, outs[0]), (s1v, s2v), (s2v, outs[1]), (s2v, outs[2])):
        g2.add_edge((u, w))
    g2.set_inputs((i,))
    g2.set_outputs(tuple(outs))
    assert np.allclose(matrix_of(g2), kron(I2, U_SM) @ U_SM)
    assert simp.spider_simp(g2)
    assert shape(g2) == shape(g) and same_picture(g2, g)


if __name__ == "__main__":
    verdict("SplitAndMerge", blackboard, sketchbook)
