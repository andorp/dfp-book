"""Challenge II: The Blind Vote  (the headline; The Theorem, Twice; the master)
===============================================================================
"Ask the pair, not the one."  -- Tortoise
"Both bells rang. So the middle strand disagrees with both its neighbours, and it's the middle I mend."  -- Crab

Claim: one gust, mended blind -- for every X^e in E_X = {I, X1, X2, X3} and every outcome s,
Enc^dag X^c(s) P_s X^e Enc = [s = sigma(e)] I, exactly, scalar 1 (Result 18.7).  The checks
hear the gust and never the message (Result 18.6); two gusts give X (Result 18.9); every
mixture of single gusts is undone (Result 18.10).  The master, The Storm Line, is built as
drawn and equals [s = sigma(e)] (I | X | 0) for all 2^8 labellings (Result 18.24); PyZX's
full_reduce derives the bare wire, the single X or zero.  A bell tied twice to one green
spider comes off with (1 + e^{i theta})/2 (Result 18.8); a bell asks a pair (Result 18.5).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from itertools import product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import I2, ket, kron, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch18 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
I8 = np.eye(8)

ENC = np.outer(ket(0, 0, 0), ket(0)) + np.outer(ket(1, 1, 1), ket(1))   # 8 x 2, top wire first
DEC = ENC.conj().T                                                     # the good-branch decoder


def on(k, A):
    ops = [I2, I2, I2]
    ops[k] = A
    return kron(*ops)


def X_pattern(e):
    return kron(*[X if b else I2 for b in e])


Z1Z2, Z2Z3 = on(0, Z) @ on(1, Z), on(1, Z) @ on(2, Z)


def P(s):
    s1, s2 = s
    return (I8 + (-1) ** s1 * Z1Z2) @ (I8 + (-1) ** s2 * Z2Z3) / 4


def sigma(e):
    return (e[0] ^ e[1], e[1] ^ e[2])


CORR = {(0, 0): (0, 0, 0), (1, 0): (1, 0, 0), (1, 1): (0, 1, 0), (0, 1): (0, 0, 1)}
SINGLE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]                  # E_X as patterns
OUTCOMES = list(product((0, 1), repeat=2))
BITS3 = list(product((0, 1), repeat=3))


def storm_value(e, s, c):
    """Result 18.24's formula: [s = sigma(e)] * (I if e+c = 000, X if 111, 0 otherwise)."""
    if tuple(s) != sigma(e):
        return np.zeros((2, 2))
    f = tuple(a ^ b for a, b in zip(e, c))
    return I2 if f == (0, 0, 0) else X if f == (1, 1, 1) else np.zeros((2, 2))


MARKS = {
    "dark": ket(0), "lit": ket(1),
    "at rest": (ket(0) + ket(1)) / np.sqrt(2), "a half": (ket(0) - ket(1)) / np.sqrt(2),
    "a quarter round": (ket(0) + 1j * ket(1)) / np.sqrt(2),
    "a quarter back": (ket(0) - 1j * ket(1)) / np.sqrt(2),
}


def blackboard():
    rng = np.random.default_rng(1811)
    # Z, X, the P_s and Enc are real, so no transpose/dagger mix-up can hide in them; the
    # messages and the Kraus weights below are the non-real, generic inputs.

    # --- the checks: projectors, orthogonal, complete
    assert np.allclose(sum(P(s) for s in OUTCOMES), I8)
    for s, t in product(OUTCOMES, repeat=2):
        assert np.allclose(P(s) @ P(t), P(s) if s == t else 0)
    assert np.allclose(P((0, 0)) @ ENC, ENC)                  # the code space is "agree, agree"

    # --- Result 18.6: the bells hear the gust, not the message (all 8 patterns, all 4 outcomes)
    psi = random_state(rng)
    assert not np.allclose(psi, psi.real)
    for e in BITS3:
        for s in OUTCOMES:
            lhs = P(s) @ X_pattern(e) @ ENC
            assert np.allclose(lhs, (s == sigma(e)) * X_pattern(e) @ ENC)
            prob = np.linalg.norm(lhs @ psi) ** 2
            assert np.isclose(prob, float(s == sigma(e)))     # certain, whatever the message

    # the dialogue's headline run: a quarter round, a gust on the middle strand, both bells
    q = MARKS["a quarter round"]
    state = X_pattern((0, 1, 0)) @ ENC @ q
    assert np.allclose(state, (ket(0, 1, 0) + 1j * ket(1, 0, 1)) / np.sqrt(2))
    assert np.isclose(np.linalg.norm(P((1, 1)) @ state), 1)   # both ring, with certainty
    assert np.allclose(X_pattern(CORR[(1, 1)]) @ state, ENC @ q)   # the middle mended: exact

    # --- Result 18.7, the headline: exact, scalar 1, branch by branch
    for e in SINGLE:
        for s in OUTCOMES:
            lhs = DEC @ X_pattern(CORR[s]) @ P(s) @ X_pattern(e) @ ENC
            assert np.allclose(lhs, (s == sigma(e)) * I2)
    assert sorted(sigma(e) for e in SINGLE) == sorted(OUTCOMES)      # sigma one-to-one on E_X
    assert all(CORR[sigma(e)] == e for e in SINGLE)                  # c(sigma(e)) = e
    # negative controls: no mending, or mending the wrong strand, does not give the message back
    for e in SINGLE[1:]:
        s = sigma(e)
        assert not np.allclose(DEC @ P(s) @ X_pattern(e) @ ENC, I2)
        other = next(f for f in SINGLE[1:] if f != e)
        assert not np.allclose(DEC @ X_pattern(other) @ P(s) @ X_pattern(e) @ ENC, I2)

    # --- Result 18.9: every pattern, [s = sigma(e)] X^[wt(e) >= 2]
    for e in BITS3:
        for s in OUTCOMES:
            lhs = DEC @ X_pattern(CORR[s]) @ P(s) @ X_pattern(e) @ ENC
            assert np.allclose(lhs, (s == sigma(e)) * (X if sum(e) >= 2 else I2))

    # --- Result 18.24 on the blackboard: the free-c formula for all 2^8 labellings
    for e, s, c in product(BITS3, OUTCOMES, BITS3):
        lhs = DEC @ X_pattern(c) @ P(s) @ X_pattern(e) @ ENC
        assert np.allclose(lhs, storm_value(e, s, c))

    # --- Result 18.10: every mixture of single gusts, E = sum_e a_e X^e with complex a_e
    for _ in range(3):
        a = rng.normal(size=4) + 1j * rng.normal(size=4)
        E = sum(ae * X_pattern(e) for ae, e in zip(a, SINGLE))
        for s in OUTCOMES:
            k = SINGLE.index(next(e for e in SINGLE if sigma(e) == s))   # sigma^{-1}(s)
            assert np.allclose(X_pattern(CORR[s]) @ P(s) @ E @ ENC, a[k] * ENC)

    # the channel: five random non-real Kraus operators in span E_X, trace preserving on the
    # code (sum_i Enc^dag K_i^dag K_i Enc = I, i.e. the 5 x 4 weight matrix has norm 1)
    A = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
    A /= np.linalg.norm(A)
    kraus = [sum(A[i, j] * X_pattern(e) for j, e in enumerate(SINGLE)) for i in range(5)]
    assert np.allclose(sum(DEC @ K.conj().T @ K @ ENC for K in kraus), I2)
    B = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho = B @ B.conj().T
    rho /= np.trace(rho)
    assert not np.allclose(rho, rho.real)
    coded = ENC @ rho @ DEC
    noisy = sum(K @ coded @ K.conj().T for K in kraus)
    recovered = sum(X_pattern(CORR[s]) @ P(s) @ noisy @ P(s) @ X_pattern(CORR[s]) for s in OUTCOMES)
    assert np.allclose(recovered, coded)                       # undone exactly
    assert not np.allclose(noisy, coded)                       # (the storm did do something)

    # --- the storm line's props: the kitchen glass lifted at the mark sent (|<psi|out>|^2)
    def glass(msg, e):
        """Send msg, gust e, the Crab mends by the bells; return the clear-probability."""
        coded = ENC @ msg
        hit = X_pattern(e) @ coded
        clear = 0.0
        for s in OUTCOMES:
            branch = X_pattern(CORR[s]) @ P(s) @ hit
            if np.linalg.norm(branch) > 1e-12:
                out = DEC @ branch                              # the joining ring
                clear += abs(np.vdot(msg, out)) ** 2
        return clear
    for name, msg in MARKS.items():
        for e in SINGLE:                                        # one gust on any strand, or none
            assert np.isclose(glass(msg, e), 1), name
    two = (1, 1, 0)                                             # two gusts at once
    for name in ("dark", "lit", "a quarter round", "a quarter back"):
        assert np.isclose(glass(MARKS[name], two), 0)           # cloudy: the vote went wrong
    for name in ("at rest", "a half"):
        assert np.isclose(glass(MARKS[name], two), 1)           # clear: X leaves them alone

    pictures()


# ------------------------------------------------------------------------------ prop pictures

SECOND_DUMPS = (
    "mend=0,2,0 tell=0,0,0 clip=none glass=clear",
    "drawn | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; 1:ring red 0 @2.5,0; 2:ring green 0 @3.9,0; "
    "3:ring red 0 @6.6,0; 4:hub red 0 @5,0.5; 5:ring green 0 @1,1; 6:ring red 0 @2.5,1; 7:ring green 0 @3.9,1; "
    "8:ring red 0 @6.6,1; 9:ring green 0 @8,1; 10:hub red 0 @5,1.5; 11:ring red 0 @2.5,2; 12:ring green 0 @3.9,2; "
    "13:ring red 0 @6.6,2; tie 10-12 []; tie 2-4 []; tie 4-7 []; tie 7-10 []; cord 11-12 [2.5,2 3.9,2]; "
    "cord 12-13 [3.9,2 6.6,2]; cord 1-2 [2.5,0 3.9,0]; cord 1-5 [2.5,0 1,1]; cord 2-3 [3.9,0 6.6,0]; cord 3-9 [6.6,0 8,1]; "
    "cord 5-11 [1,1 2.5,2]; cord 5-6 [1,1 2.5,1]; cord 6-7 [2.5,1 3.9,1]; cord 7-8 [3.9,1 6.6,1]; cord 8-9 [6.6,1 8,1]; "
    "cord 9-13 [8,1 6.6,2]; cord loose-5 [0,1 1,1]; cord loose-9 [9,1 8,1]",
    "halves,through tiehalves=2 | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; 1:ring red 0 @2.5,0; "
    "2:ring green 0 @3.9,0; 3:ring red 0 @6.6,0; 4:hub red 12 @5,0.5; 5:ring green 0 @1,1; 6:ring red 0 @2.5,1; "
    "7:ring green 0 @3.9,1; 8:ring red 12 @5,1; 9:ring red 12 @6.6,1; 10:ring green 0 @8,1; 11:hub red 12 @5,1.5; "
    "12:ring red 0 @2.5,2; 13:ring green 0 @3.9,2; 14:ring red 0 @6.6,2; tie 11-13 []; tie 2-4 []; tie 4-7 []; "
    "tie 7-11 []; cord 10-14 [8,1 6.6,2]; cord 12-13 [2.5,2 3.9,2]; cord 13-14 [3.9,2 6.6,2]; cord 1-2 [2.5,0 3.9,0]; "
    "cord 1-5 [2.5,0 1,1]; cord 2-3 [3.9,0 6.6,0]; cord 3-10 [6.6,0 8,1]; cord 5-12 [1,1 2.5,2]; cord 5-6 [1,1 2.5,1]; "
    "cord 6-7 [2.5,1 3.9,1]; cord 7-9 [3.9,1 6.6,1]; cord 9-10 [6.6,1 8,1]; cord loose-10 [9,1 8,1]; cord loose-5 [0,1 1,1]",
    "halves,through,slidin,together,off,straight | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; "
    "cord loose-loose [0,1 9,1]",
)


def signature(g):
    """Each vertex's type, degree and the types of its neighbours, as a multiset."""
    return sorted((g.type(v), len(list(g.neighbors(v))), tuple(sorted(g.type(u) for u in g.neighbors(v))))
                  for v in g.vertices())


def pictures():
    """The Second Challenge's prop pictures (identity card, section 4), read from the dialogue
    through tools/dfp_props.py and the transcription tools/props_ch18.py, and held against this
    file's storm_line (graph_of: a structural reading of each cloth dump, not a proof)."""
    calls = DIALOGUE.check_layout("Second", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("stormline", False, "middle mending ring at a half, glass clear", (), "Lifted at the kitchen"),
        ("linecloth", True, "", (), "On the cloth, at rest"),
        ("linecloth", False, "halves set, gust's half sent through", (), "Gone through"),
        ("linecloth", False, "halves set, gust's half sent through, halves slid in, green rings slid together, "
                             "bells off, drawn straight", (), "Drawn straight")]
    PC.check_captions(DIALOGUE, "Second", calls)
    PC.check_section(DIALOGUE, "Second", calls)
    PC.check_anchors(DIALOGUE, "Second", ["P2.1", "P2.2", "P2.3", "P2.4"])
    pics = [PC.picture(c) for c in calls]
    p21, p22, p23, p24 = pics
    assert tuple(p.dump for p in pics) == SECOND_DUMPS
    lc = lambda s: PC.linecloth(s)
    steps = ["halves set", "gust's half sent through", "halves slid in", "green rings slid together", "bells off",
             "drawn straight"]
    states = [""] + [", ".join(steps[:k]) for k in range(1, 7)]
    assert {lc(s).frame for s in states} == {p22.frame}           # one frame in every state
    assert {PC.stormline(s).frame for s in ("", "glass cloudy")} == {p21.frame}

    # P2.1: the Crab's mend for the gust's pattern.  The gust turned the middle weather ring and
    # fell back (never drawn); both bells rang, s = (1, 1); the dump's mending ring at a half is
    # c(s) = (0, 1, 0); and the glass reads clear because that line is the identity, exactly.
    L = p21.state
    c = tuple(q // 2 for q in L.mend)
    assert all(q % 2 == 0 for q in L.mend) and c == CORR[(1, 1)] == (0, 1, 0)
    e, s = (0, 1, 0), (1, 1)
    assert sigma(e) == s
    assert np.allclose(matrix_of(storm_line(e, s, c)), I2)       # "glass=clear"
    assert not np.allclose(matrix_of(storm_line(e, s, (0, 0, 0))), I2)   # unmended: not clear
    assert L.glass == "clear" and L.tell == [0, 0, 0] and L.clip is None   # "Lifted at the kitchen"
    # fixed parts, P1.4 -> P2.1: only the middle mending ring's bead (by a half) and the glass
    # (lifted, clear) differ; every other ring, bead and cord the same, the dial shut in both
    before = PC.stormline("", True)
    assert before.dump == "mend=0,0,0 tell=0,0,0 clip=none glass=reset"
    vb, va = before.shown, p21.shown
    assert vb.spans == va.spans and vb.knots == va.knots and vb.bells == va.bells
    assert [(a[0], a[4], b[4]) for a, b in zip(vb.rings, va.rings) if a != b] == [("m2", 0, 2)]
    assert [a[:4] for a in vb.rings] == [b[:4] for b in va.rings]    # every ring where it was
    assert [k for k, (a, b) in enumerate(zip(va.beads, vb.beads)) if a != b] == [10]
    assert va.clip == vb.clip == () and (vb.glass, va.glass) == ("reset", "clear")

    # P2.2: the line on the cloth, ring for ring: its graph has the master's eighteen edges and
    # fifteen vertices, the same vertex signature, and equals the master exactly
    g22 = PC.graph_of(p22.dump)
    master = storm_line((0, 0, 0), (0, 0), (0, 0, 0))
    assert g22.num_edges() == 18 == master.num_edges() and g22.num_vertices() == 15 == master.num_vertices()
    assert signature(g22) == signature(master)
    assert same_picture(g22, master)
    nodes = PC.parse_dump(p22.dump)[0]                            # "On the cloth, at rest": drawn, every turn 0
    assert p22.dump.startswith("drawn |") and all(t == 0 for kind, col, t, x, y in nodes.values())
    # the bells between the strands, a little towards the kitchen end from the tell rings
    hubs = sorted((x, y) for kind, col, t, x, y in nodes.values() if kind == "hub")
    assert hubs == [(PC.LX["hub"], 0.5), (PC.LX["hub"], 1.5)] and PC.LX["hub"] > PC.LX["tell"]

    # P2.3: halves set equals the line with the gust, the bells and the mend at a half; the half
    # sent through (C4) leaves it exactly as it was
    halves = PC.graph_of(lc("halves set").dump)
    gust = storm_line((0, 1, 0), (1, 1), (0, 1, 0))
    assert same_picture(halves, gust)
    g23 = PC.graph_of(p23.dump)
    # the 18, the half on the mending cord splitting it, and a half on each tie to a bell
    assert g23.num_edges() == 21 and g23.num_vertices() == 18
    assert same_picture(g23, halves)
    assert np.allclose(matrix_of(g23), I2)                         # and it is the identity
    no_tie_halves = PC.graph_of(p23.dump.replace(" tiehalves=2", ""))
    assert not same_picture(no_tie_halves, halves)                 # the halves on the ties carry it
    assert "through" in p23.dump.split(" |")[0]                    # "Gone through"
    # the unpictured steps between, each the same line (C4, C1, C6): slid in, slid together
    # (each bell tied twice to the one ring), the bells off
    for k in (3, 4, 5):
        assert same_picture(PC.graph_of(lc(", ".join(steps[:k])).dump), master)
    together = PC.graph_of(lc(", ".join(steps[:4])).dump)       # one ring, each bell tied twice
    assert (together.num_vertices(), together.num_edges()) == (7, 8)
    # fixed parts: the parting and joining rings at their columns until the green rings slide
    for pic in (p22, p23):
        g = [(x, y) for kind, col, t, x, y in PC.parse_dump(pic.dump)[0].values() if col == "green" and y == 1]
        assert (PC.LX["part"], 1) in g and (PC.LX["join"], 1) in g

    # P2.4: the line drawn straight is the plain cord, and equals the line at rest exactly
    g24 = PC.graph_of(p24.dump)
    assert same_picture(g24, g22) and np.allclose(matrix_of(g24), I2)
    nodes, tubes, ties, cords, extra = PC.parse_dump(p24.dump)
    assert nodes == {} and tubes == [] and ties == [] and len(cords) == 1   # "Drawn straight"
    assert p24.dump.split(" |")[0].endswith("straight")

    # the keyed word (P2.2): bell, set by the Lua, measured clear; the kit is never keyed here
    assert p22.labels == ("bell",) and all(p.labels == () for p in (p21, p23, p24))
    PC.check_compiled_boxes(p22)
    assert min(PC.check_keyed_words(p22).values()) >= PC.LETTER_FLOOR

    # the Lua's refusals: the green rings slid together while a half is still out; a step
    # out of order; a mending ring turned a quarter
    PC.TEX_ERRORS.clear()
    PC.lc_state("halves set, green rings slid together")
    PC.lc_state("halves slid in")
    PC.lc_state("green rings slid together, halves set")          # a step that must come first, late
    PC.line_state("middle mending ring a quarter round")
    assert [k for k, _ in PC.TEX_ERRORS] == ["lc-together", "lc-after", "lc-after", "line-mend-quarter"]
    assert PC.TEX_ERRORS[2][1] == "linecloth: 'halves set' comes first, before every other step"
    PC.TEX_ERRORS.clear()


# ------------------------------------------------------------------------------ pictures

def storm_line(e, s, c, tell=(0, 0, 0)):
    """The Storm Line, built by hand as drawn (Section 18.8).  Rows: strand k on row 2k,
    bells between strands.  Every ring is drawn whatever its bit (row 75)."""
    g = zx.Graph()
    lead_in = g.add_vertex(VT.BOUNDARY, 2, 0)
    lead_out = g.add_vertex(VT.BOUNDARY, 2, 12)
    parting = g.add_vertex(VT.Z, 2, 1)                                        # Enc
    joining = g.add_vertex(VT.Z, 2, 11)                                       # Enc^dag
    weather = [g.add_vertex(VT.X, 2 * k, 3, phase=Fraction(e[k])) for k in range(3)]
    tellr = [g.add_vertex(VT.Z, 2 * k, 5, phase=Fraction(tell[k])) for k in range(3)]
    bells = [g.add_vertex(VT.X, 2 * j + 1, 6, phase=Fraction(s[j])) for j in range(2)]
    mending = [g.add_vertex(VT.X, 2 * k, 9, phase=Fraction(c[k])) for k in range(3)]
    g.add_edge((lead_in, parting), ET.SIMPLE)                                 # the lead-in
    for k in range(3):                                                        # four lengths a strand
        g.add_edge((parting, weather[k]), ET.SIMPLE)
        g.add_edge((weather[k], tellr[k]), ET.SIMPLE)
        g.add_edge((tellr[k], mending[k]), ET.SIMPLE)
        g.add_edge((mending[k], joining), ET.SIMPLE)
    for t, b in ((0, 0), (1, 0), (1, 1), (2, 1)):                             # the ties
        g.add_edge((tellr[t], bells[b]), ET.SIMPLE)
    g.add_edge((joining, lead_out), ET.SIMPLE)                                # the lead-out
    g.set_inputs((lead_in,))
    g.set_outputs((lead_out,))
    return g


def double_tied_bell(theta, phi, legs):
    """A red bell theta tied twice to one green spider phi with `legs` other legs.  PyZX's
    simple graphs keep one edge per pair, so the second tie carries a green ring at rest
    (bare cord, row 57): the tell ring not yet slid in."""
    g = zx.Graph()
    green = g.add_vertex(VT.Z, 1, 2, phase=Fraction(phi))
    bell = g.add_vertex(VT.X, 3, 2, phase=Fraction(theta))
    spacer = g.add_vertex(VT.Z, 2, 3)
    g.add_edge((green, bell))
    g.add_edge((green, spacer))
    g.add_edge((spacer, bell))
    ins, outs = [], []
    if legs >= 2:
        ins.append(g.add_vertex(VT.BOUNDARY, 0, 0))
    for k in range(legs - len(ins)):
        outs.append(g.add_vertex(VT.BOUNDARY, k, 4))
    for b in ins + outs:
        g.add_edge((b, green))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


def green_alone(phi, legs, theta=None):
    g = zx.Graph()
    green = g.add_vertex(VT.Z, 1, 2, phase=Fraction(phi))
    ins = [g.add_vertex(VT.BOUNDARY, 0, 0)] if legs >= 2 else []
    outs = [g.add_vertex(VT.BOUNDARY, k, 4) for k in range(legs - len(ins))]
    for b in ins + outs:
        g.add_edge((b, green))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    if theta is not None:                  # the number (1 + e^{i theta})/2
        g.scalar.add_node(Fraction(theta))
        g.scalar.add_power(-2)
    return g


def pair_check(s):
    """Result 18.5: a green ring at rest on each of two wires, tied to one red bell s*pi."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in (0, 1)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 4) for r in (0, 1)]
    feet = [g.add_vertex(VT.Z, r, 1) for r in (0, 1)]
    bell = g.add_vertex(VT.X, 0.5, 2, phase=Fraction(s))
    for r in (0, 1):
        g.add_edge((ins[r], feet[r]))
        g.add_edge((feet[r], outs[r]))
        g.add_edge((feet[r], bell))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


def two_bells(s):
    """The middle of the storm line: tell rings on three wires, the two bells."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, 2 * r, 0) for r in range(3)]
    outs = [g.add_vertex(VT.BOUNDARY, 2 * r, 4) for r in range(3)]
    tellr = [g.add_vertex(VT.Z, 2 * r, 1) for r in range(3)]
    bells = [g.add_vertex(VT.X, 2 * j + 1, 2, phase=Fraction(s[j])) for j in range(2)]
    for r in range(3):
        g.add_edge((ins[r], tellr[r]))
        g.add_edge((tellr[r], outs[r]))
    for t, b in ((0, 0), (1, 0), (1, 1), (2, 1)):
        g.add_edge((tellr[t], bells[b]))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


def sketchbook():
    rng = np.random.default_rng(1812)
    ZZ = np.kron(Z, Z)

    # --- Result 18.5 / row 138: a bell asks a pair, exactly, no number (the weight-2 case
    # of WhatABellMayAsk.py's T2; the derivation in this file is the master's)
    for s in (0, 1):
        m = matrix_of(pair_check(s))
        assert np.allclose(m, (np.eye(4) + (-1) ** s * ZZ) / 2)
        assert not np.allclose(m, (np.eye(4) + (-1) ** (1 - s) * ZZ) / 2)
    for s in OUTCOMES:                                                  # two bells: the P_s
        assert np.allclose(matrix_of(two_bells(s)), P(s))

    # --- The Storm Line: built as drawn, counted
    g0 = storm_line((0, 0, 0), (0, 0), (0, 0, 0))
    # the lead-in and the lead-out, twelve strand lengths (four on each strand), four ties
    assert g0.num_edges() == 18
    assert g0.num_vertices() == 15
    assert sum(1 for v in g0.vertices() if g0.type(v) == VT.BOUNDARY) == 2
    assert sum(1 for v in g0.vertices() if g0.type(v) == VT.Z) == 5
    assert sum(1 for v in g0.vertices() if g0.type(v) == VT.X) == 8
    assert all(g0.edge_type(ed) == ET.SIMPLE for ed in g0.edges())    # no Hadamard edge

    # --- Result 18.24: its value, for every one of the 2^8 labellings, exactly, scalar 1
    for e, s, c in product(BITS3, OUTCOMES, BITS3):
        g = storm_line(e, s, c)
        assert np.allclose(matrix_of(g), storm_value(e, s, c))
        # step (i) of Section 18.7, exact for every labelling: the gust's halves moved into
        # the bells and the mending rings, the weather rings left at rest
        moved = storm_line((0, 0, 0),
                           tuple(a ^ b for a, b in zip(s, sigma(e))),
                           tuple(a ^ b for a, b in zip(c, e)))
        assert same_picture(g, moved)

        # DERIVED: full_reduce takes each labelling to a bare wire, a single X, or zero
        h = g.copy()
        simp.full_reduce(h)
        f = tuple(a ^ b for a, b in zip(e, c))
        if s == sigma(e) and f == (0, 0, 0):
            assert h.num_vertices() == 2 and h.num_edges() == 1       # a bare wire...
            assert np.isclose(h.scalar.to_number(), 1)                # ...number 1
        elif s == sigma(e) and f == (1, 1, 1):
            inner = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
            assert len(inner) == 1                                    # one spider:
            v = inner[0]
            # PyZX's graph-like form of a red half: green pi between two Hadamard edges
            assert h.type(v) == VT.Z and h.phase(v) == 1
            assert [h.edge_type(ed) for ed in h.incident_edges(v)] == [ET.HADAMARD] * 2
            assert np.isclose(h.scalar.to_number(), 1)
            assert np.allclose(matrix_of(h), X)
        else:
            assert h.scalar.is_zero                                   # the branch is 0
    # the headline, as the picture: c = c(s) on E_X
    for e in SINGLE:
        for s in OUTCOMES:
            m = matrix_of(storm_line(e, s, CORR[s]))
            assert np.allclose(m, (s == sigma(e)) * I2)

    # --- Result 18.8: the double-tied bell comes off with (1 + e^{i theta})/2,
    # for a green spider with 1, 2 or 3 other legs and a random turn phi
    for legs in (1, 2, 3):
        phi = Fraction(int(rng.integers(1, 97)), 97)
        rand_theta = Fraction(int(rng.integers(1, 97)), 97)
        for theta in (Fraction(0), Fraction(1), rand_theta):
            g = double_tied_bell(theta, phi, legs)
            assert g.num_edges() == 3 + legs                         # two ties (one via the spacer)
            ref = green_alone(phi, legs, theta)
            assert same_picture(g, ref)
            expect = (1 + np.exp(1j * np.pi * float(theta))) / 2
            assert np.allclose(matrix_of(g), expect * matrix_of(green_alone(phi, legs)))
        # DERIVED by full_reduce: at rest, the green spider alone, number 1; at a half, zero
        h = double_tied_bell(0, phi, legs).copy()
        simp.full_reduce(h)
        assert same_picture(h, green_alone(phi, legs))
        assert np.isclose(h.scalar.to_number(), 1)
        h = double_tied_bell(1, phi, legs).copy()
        simp.full_reduce(h)
        assert h.scalar.is_zero
        # at a random theta, reduce both sides and compare
        h = double_tied_bell(rand_theta, phi, legs).copy()
        simp.full_reduce(h)
        r = green_alone(phi, legs, rand_theta)
        simp.full_reduce(r)
        assert same_picture(h, r)
    # the warning: a bell tied to two DIFFERENT green spiders does not come off
    for s in (0, 1):
        assert not np.allclose(matrix_of(pair_check(s)), np.eye(4))


if __name__ == "__main__":
    verdict("BlindVote", blackboard, sketchbook)
