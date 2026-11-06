"""Challenge II: The Hinge  (the Euler card; the headline; The Theorem, Twice)
=============================================================================
"A hub with a quarter hanging from it, standing on a quarter back on each line, is a tie in a sleeve. Yesterday I hadn't a move for it."  -- Tortoise

Claim: the quarter hub is the sleeved tie, exactly, with the number 1 -- Hub = Tie = CZ / sqrt2
(Result 19.6).  The Euler card Z(pi/2) X(pi/2) Z(pi/2) = e^{i pi/4} H is added to the deck;
its mirror (Result 19.3), Lemma P, a green turn passes a basis point with e^{i c alpha} for
every alpha (Result 19.4), and Lemma A, a green quarter point is e^{i pi/4} times a red
quarter-back point (Result 19.5), hold with their numbers.  Every picture of the five-move
chain of Section 19.7 is checked against the one before, with its number, and PyZX's own
simplifier (which is not the book's deck) also takes Hub to Tie.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from itertools import product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import I2, ket, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch19 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

PI = np.pi
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
S = np.diag([1, 1j])
PLUS, MINUS = (ket(0) + ket(1)) / np.sqrt(2), (ket(0) - ket(1)) / np.sqrt(2)


def Zt(a):
    """The bare two-legged green spider, turned a radians: diag(1, e^{ia})."""
    return np.diag([1, np.exp(1j * a)])


def Xt(a):
    """The bare two-legged red spider, from |+>, |-> (row 54)."""
    return np.outer(PLUS, PLUS) + np.exp(1j * a) * np.outer(MINUS, MINUS)


def hub_matrix(foot=-PI / 2, point=PI / 2):
    """Hub, contracted leg by leg as drawn: each foot a green spider (in, out, tie), both ties
    into one red hub at rest (two in, one out), closed by a green point used as an effect."""
    foot_t = np.zeros((2, 2, 2), dtype=complex)            # [wire out, tie, wire in]
    for x in (0, 1):
        foot_t[x, x, x] = np.exp(1j * foot * x)
    red = (np.einsum("o,a,b->oab", PLUS, PLUS.conj(), PLUS.conj())
           + np.einsum("o,a,b->oab", MINUS, MINUS.conj(), MINUS.conj()))                    # 2 -> 1
    eff = np.array([1, np.exp(1j * point)])                  # <0| + e^{i point}<1|
    M = np.einsum("pai,qbj,oab,o->pqij", foot_t, foot_t, red, eff)
    return M.reshape(4, 4)


def tie_matrix(edge=H):
    foot = np.zeros((2, 2, 2), dtype=complex)
    for x in (0, 1):
        foot[x, x, x] = 1
    M = np.einsum("pai,qbj,ab->pqij", foot, foot, edge)       # the tie runs from top to bottom
    return M.reshape(4, 4)


def blackboard():
    rng = np.random.default_rng(147)
    q = PI / 2
    # --- the Euler card (row 145), its mirror (Result 19.3) and the colour-swapped form
    assert np.allclose(Xt(q), H @ Zt(q) @ H)                          # red is green seen through tubes
    assert np.allclose(Zt(q) @ Xt(q) @ Zt(q), np.exp(1j * PI / 4) * H)
    assert np.allclose(Zt(-q) @ Xt(-q) @ Zt(-q), np.exp(-1j * PI / 4) * H)
    assert np.allclose(Xt(q) @ Zt(q) @ Xt(q), np.exp(1j * PI / 4) * H)
    assert not np.allclose(Zt(q) @ Xt(q) @ Zt(q), H)                   # the eighth is load-bearing
    assert not np.allclose(Zt(-q) @ Xt(-q) @ Zt(-q), np.exp(1j * PI / 4) * H)   # mirror: an eighth BACK
    # Chapter 3's SVS, with V = HSH
    V = H @ S @ H
    assert np.allclose(V, Xt(q)) and np.allclose(S @ V @ S, np.exp(1j * PI / 4) * H)
    # the mirror Euler by the card's own derivation: a tube pair after the three turns, the first
    # tube read backwards, the three opposite pairs fused away from the inside out
    lhs = Zt(-q) @ Xt(-q) @ Zt(-q)
    step = H @ (np.exp(-1j * PI / 4) * Zt(q) @ Xt(q) @ Zt(q)) @ lhs     # = H H lhs = lhs
    assert np.allclose(step, lhs)
    assert np.allclose(Zt(q) @ Zt(-q), I2) and np.allclose(Xt(q) @ Xt(-q), I2)
    assert np.allclose(step, np.exp(-1j * PI / 4) * H)

    # --- Lemma P (Result 19.4), for every alpha: seeded random turns, both basis points
    for _ in range(5):
        alpha = rng.uniform(0, 2 * PI)
        for c in (0, 1):
            pt = np.sqrt(2) * ket(c)
            assert np.allclose(Zt(alpha) @ pt, np.exp(1j * c * alpha) * pt)
            # the sketchbook's route: slide apart; the point copies through the green spider at
            # rest (row 81, number 1/sqrt2); the hanging copy meets the green alpha point
            copies = np.kron(pt, pt) / np.sqrt(2)
            assert np.allclose(copies, np.outer(ket(0, 0), ket(0)) @ pt + np.outer(ket(1, 1), ket(1)) @ pt)
            closed = copies.reshape(2, 2) @ np.array([1, np.exp(1j * alpha)])
            assert np.allclose(closed, np.exp(1j * c * alpha) * pt)
        assert not np.allclose(Zt(alpha) @ ket(1), ket(1))               # c = 1: the whole turns
        assert not np.allclose(Zt(alpha) @ PLUS, PLUS)                   # not for a green point

    # --- Lemma A (Result 19.5)
    red_qb = Xt(-q) @ (np.sqrt(2) * ket(0))
    green_q = ket(0) + 1j * ket(1)
    assert np.allclose(red_qb, np.exp(-1j * PI / 4) * green_q)
    assert np.allclose(green_q, np.exp(1j * PI / 4) * red_qb)
    assert not np.allclose(Xt(q) @ (np.sqrt(2) * ket(0)), np.exp(-1j * PI / 4) * green_q)  # a quarter ROUND: no
    # its five moves: red -q point = red point at rest then red -q; = e^{-i pi/4} Z(q) H Z(q)
    # on the red point at rest; Z(q) passes it (c = 0); H makes it green; Z(q) fuses in
    assert np.allclose(Xt(-q), np.exp(-1j * PI / 4) * Zt(q) @ H @ Zt(q))
    assert np.allclose(Zt(q) @ (np.sqrt(2) * ket(0)), np.sqrt(2) * ket(0))
    assert np.allclose(H @ (np.sqrt(2) * ket(0)), ket(0) + ket(1))

    # --- the hinge, the two lines: x1 + x2 - (x1 xor x2) = 2 x1 x2 on every basis word
    for x1, x2 in product((0, 1), repeat=2):
        assert x1 + x2 - (x1 ^ x2) == 2 * x1 * x2
        amp = 2 ** -0.5 * np.exp(1j * q * ((x1 ^ x2) - x1 - x2))
        assert np.isclose(amp, 2 ** -0.5 * (-1) ** (x1 * x2))
        assert np.isclose(H[x2, x1], 2 ** -0.5 * (-1) ** (x1 * x2))       # <x2|H|x1>
    Hub, Tie = hub_matrix(), tie_matrix()
    assert np.allclose(Hub, CZ / np.sqrt(2)) and np.allclose(Tie, CZ / np.sqrt(2))
    assert np.allclose(Hub, Tie)
    Gamma = np.diag([1, 1j, 1j, 1])                                       # row 129, Gamma(pi/2)
    assert np.allclose(hub_matrix(foot=0), Gamma / np.sqrt(2))            # feet at rest
    assert np.allclose(np.kron(S.conj().T, S.conj().T) @ Gamma / np.sqrt(2), CZ / np.sqrt(2))
    # NEGATIVE: the plausible slips
    assert not np.allclose(hub_matrix(foot=+q), Tie)                      # feet a quarter ROUND
    assert not np.allclose(hub_matrix(point=0), Tie)                      # pendant at rest
    assert not np.allclose(tie_matrix(I2), Tie)                           # a tie with no sleeve
    assert not np.allclose(Hub, CZ)                                       # the 1/sqrt2 is there
    assert np.allclose(Tie @ Tie.conj().T, np.eye(4) / 2)                 # Tie Tie^dag = 1/2
    assert np.allclose(kron(ket(0), ket(1)).ravel(), ket(0, 1))            # top wire first

    pictures()


# ------------------------------------------------------------------- the prop pictures (hinge)

SECOND_DUMPS = (
    "new face | left: g6 r6 g6 | right: tube | dial=right:3 | cross=none",
    "laid | cols=9 rows=5 y=0..4 near=false folded=false; corner=; 1:ring green 18 @4,0; 2:hub red 0 @4,2; "
    "3:bob green 6 @5,3; 4:ring green 18 @4,4; tie 1-2 []; tie 2-4 []; pendant 2-3 []; cord loose-loose [0,0 8,0]; "
    "cord loose-loose [0,4 8,4]",
    "red,slid,apart | cols=9 rows=5 y=0..4 near=false folded=false; corner=3; 1:ring green 0 @4,0; "
    "2:ring green 18 @4,1; 3:hub red 18 @4,2; 4:ring green 18 @4,3; 5:ring green 0 @4,4; tie 1-2 []; tie 2-3 []; "
    "tie 3-4 []; tie 4-5 []; cord loose-loose [0,0 8,0]; cord loose-loose [0,4 8,4]",
    "red,slid,apart,tube,gone | cols=9 rows=5 y=0..4 near=false folded=false; corner=; 1:ring green 0 @4,0; "
    "2:ring green 0 @4,4; tie 1-2 tubes1 []; cord loose-loose [0,0 8,0]; cord loose-loose [0,4 8,4]")
HINGE_STEPS = ("red point", "slid into the hub", "feet slid apart", "tube", "slips gone")


def with_slips(dump):
    """graph_of a cloth dump (a structural reading of the picture, not a second proof), with the
    corner's slips written on its scalar: the whole picture's number."""
    g = PC.graph_of(dump)
    g.scalar.add_phase(PC.slips_turn(dump))
    return g


def pictures():
    """The Second Challenge's prop pictures (identity card, section 4), read from the dialogue
    through tools/dfp_props.py and the transcription tools/props_ch19.py, and held against this
    file's hinge_picture and hub_matrix."""
    calls = DIALOGUE.check_layout("Second", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("rulecard", False, "", ("new",), "No longer blank"),
        ("hingecloth", True, "", (), "Laid out flat"),
        ("hingecloth", False, "red point, slid into the hub, feet slid apart", (), "Feet at rest"),
        ("hingecloth", False, "red point, slid into the hub, feet slid apart, tube, slips gone", (), "The corner empty")]
    PC.check_captions(DIALOGUE, "Second", calls)
    PC.check_section(DIALOGUE, "Second", calls)
    PC.check_anchors(DIALOGUE, "Second")
    pics = [PC.picture(c) for c in calls]
    p21, p22, p23, p24 = pics
    assert tuple(p.dump for p in pics) == SECOND_DUMPS
    states = [""] + [", ".join(HINGE_STEPS[:k]) for k in range(1, 6)]
    assert {PC.hingecloth(st).frame for st in states} == {p22.frame}      # one frame in every state

    # P2.1, the new card, no longer blank: both sides drawn, and the left side -- green, red,
    # green, each a quarter round -- is e^{i pi/4} times the tube, exactly (row 145)
    assert p21.state.view == "face" and p21.state.left and p21.state.right
    q = PI / 2
    assert [(x.kind, x.c, x.t) for x in p21.state.left] == [("ring", "green", 6), ("ring", "red", 6), ("ring", "green", 6)]
    assert [x.kind for x in p21.state.right] == ["tube"] and p21.state.dial == 3
    assert np.allclose(Zt(q) @ Xt(q) @ Zt(q), np.exp(1j * PI * p21.state.dial / 12) * H)
    assert not np.allclose(Zt(q) @ Xt(q) @ Zt(q), H)
    euler = one_wire([("Z", F(1, 2)), ("X", F(1, 2)), ("Z", F(1, 2))])
    assert same_picture(euler, one_wire([("H",)], phase=F(p21.state.dial, 12)))

    # P2.2, the hub as laid: graph_of reads it as this file's Hub, exactly, and Hub is hub_matrix()
    hub = hinge_picture("hub")
    g22 = PC.graph_of(p22.dump)
    assert same_picture(g22, hub) and np.allclose(matrix_of(g22), hub_matrix())
    assert g22.num_edges() == 7 and g22.num_vertices() == 8
    nodes, tubes, ties, cords, corner, knots = PC.parse_dump(p22.dump)
    assert corner == [] and tubes == []                                  # "Laid out flat", corner empty
    assert p22.dump.startswith("laid |")
    # the hub's point: its short cord running out and down to the right, the point on the cloth
    hubv = [v for v in nodes.values() if v[0] == "hub"][0]
    bob = [v for v in nodes.values() if v[0] == "bob"][0]
    assert bob[3] > hubv[3] and bob[4] > hubv[4] and ("pendant", 2, 3, 0) in ties
    # NEGATIVE: the point a quarter BACK, or the feet a quarter ROUND, is not the hub
    assert not same_picture(PC.graph_of(p22.dump.replace("bob green 6", "bob green 18")), hub)
    assert not same_picture(PC.graph_of(p22.dump.replace("ring green 18", "ring green 6")), hub)

    # P2.3, P2.4 and every unpictured state of the trace: each, with its corner's slips, is the
    # hub as laid, exactly (Lemma A, the fusions, the crossed card, the slips cancelling)
    for st in states:
        pic = PC.hingecloth(st)
        assert same_picture(with_slips(pic.dump), hub), st
    # ...and each picture between is an eighth off without its slips
    for st in states[1:4]:
        assert not same_picture(PC.graph_of(PC.hingecloth(st).dump), hub), st
    assert [PC.parse_dump(PC.hingecloth(st).dump)[4] for st in states] == [[], [3], [3], [3], [3, 21], []]
    assert same_picture(PC.graph_of(PC.hingecloth(states[1]).dump), hinge_picture("i"))
    assert same_picture(PC.graph_of(PC.hingecloth(states[2]).dump), hinge_picture("ii"))
    g23 = PC.graph_of(p23.dump)
    assert same_picture(g23, hinge_picture("iii"))
    # "Feet at rest": the rings on the lines at rest; a green quarter back on each tie
    nodes = PC.parse_dump(p23.dump)[0]
    assert sorted(t for kind, c, t, x, y in nodes.values() if y in (0, 4)) == [0, 0]
    assert sorted((c, t) for kind, c, t, x, y in nodes.values() if y in (1, 3)) == [("green", 18), ("green", 18)]

    # P2.4, the corner empty: the sleeved tie, the headline -- hinge_picture("tie"), CZ / sqrt2
    g24 = PC.graph_of(p24.dump)
    assert same_picture(g24, hinge_picture("tie")) and np.allclose(matrix_of(g24), CZ / np.sqrt(2))
    assert PC.parse_dump(p24.dump)[4] == [] and p24.dump.startswith("red,slid,apart,tube,gone |")
    assert not same_picture(PC.graph_of(p24.dump.replace(" tubes1", "")), hinge_picture("tie"))   # the sleeve

    # fixed parts: each line's loose ends and the column of the feet, in every picture
    for pic in (p22, p23, p24):
        nodes, tubes, ties, cords, corner, knots = PC.parse_dump(pic.dump)
        assert [(a, b, pts) for a, b, pts in cords] == [("loose", "loose", [(0, 0), (8, 0)]),
                                                         ("loose", "loose", [(0, 4), (8, 4)])]
        assert {x for kind, c, t, x, y in nodes.values() if y in (0, 4)} == {4}

    # the keyed words (P2.2): the kit's hub, point, tie and line (its no-room fault the check);
    # corner, set by the Lua, at its compiled box and measured clear of every part; the rest unkeyed
    assert p22.labels == ("hub", "point", "tie", "line", "corner")
    assert [w for w, *_ in PC.own_words(p22)] == ["corner"] and not PC.parse_dump(p22.dump)[4]
    PC.check_compiled_boxes(p22)
    assert min(PC.check_keyed_words(p22).values()) >= PC.LETTER_FLOOR
    assert p21.labels == p23.labels == p24.labels == ()

    # the Lua's refusals: an unknown step; a step twice; a step before the one it follows
    PC.TEX_ERRORS.clear()
    PC.hinge_state("hub swung round")
    PC.hinge_state("red point, red point")
    PC.hinge_state("red point, feet slid apart")
    assert [k for k, _ in PC.TEX_ERRORS] == ["step-unknown", "step-done", "step-after"]
    PC.TEX_ERRORS.clear()


# ----------------------------------------------------------------------------- the sketchbook
F = Fraction


def hinge_picture(stage):
    """The pictures of the chain, Section 19.7.  Rows: top wire 0, bottom wire 4; the tie runs
    down between them.  Turns are in units of pi.
      'hub'  : feet green -1/2, red hub at rest, green +1/2 point hanging  (Hub)
      'i'    : the hanging point is a red -1/2 point                      (number e^{-i pi/4} apart)
      'ii'   : the point fused into the hub: a red -1/2 on the tie
      'iii'  : each foot slid apart: green at rest on the wire, green -1/2 on the tie
      'tie'  : green at rest on each wire, joined by a Hadamard edge       (Tie)"""
    g = zx.Graph()
    i0, i1 = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 4, 0)
    o0, o1 = g.add_vertex(VT.BOUNDARY, 0, 4), g.add_vertex(VT.BOUNDARY, 4, 4)
    if stage in ("hub", "i", "ii"):
        top, bot = g.add_vertex(VT.Z, 0, 1, F(-1, 2)), g.add_vertex(VT.Z, 4, 1, F(-1, 2))
    else:
        top, bot = g.add_vertex(VT.Z, 0, 1), g.add_vertex(VT.Z, 4, 1)
    for e in ((i0, top), (top, o0), (i1, bot), (bot, o1)):
        g.add_edge(e)
    if stage in ("hub", "i"):
        hub = g.add_vertex(VT.X, 2, 1)
        if stage == "hub":
            pt = g.add_vertex(VT.Z, 2, 2, F(1, 2))
        else:
            pt = g.add_vertex(VT.X, 2, 2, F(-1, 2))
        for e in ((top, hub), (bot, hub), (hub, pt)):
            g.add_edge(e)
        expected = 7
    elif stage == "ii":
        hub = g.add_vertex(VT.X, 2, 1, F(-1, 2))
        g.add_edge((top, hub))
        g.add_edge((hub, bot))
        expected = 6
    elif stage == "iii":
        a, r, c = (g.add_vertex(VT.Z, 1, 1, F(-1, 2)), g.add_vertex(VT.X, 2, 1, F(-1, 2)),
                   g.add_vertex(VT.Z, 3, 1, F(-1, 2)))
        for e in ((top, a), (a, r), (r, c), (c, bot)):
            g.add_edge(e)
        expected = 8
    else:
        g.add_edge((top, bot), ET.HADAMARD)
        expected = 5
    g.set_inputs((i0, i1))
    g.set_outputs((o0, o1))
    assert g.num_edges() == expected
    return g


def one_wire(pieces, phase=0):
    """A chain of two-legged spiders on one wire; ('H',) is a yellow box (a Hadamard edge
    into the next piece, or into the output)."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    prev, et, col = i, ET.SIMPLE, 1
    n = 0
    for p in pieces:
        if p[0] == "H":
            et = ET.HADAMARD if et == ET.SIMPLE else ET.SIMPLE
            continue
        v = g.add_vertex(VT.Z if p[0] == "Z" else VT.X, 0, col, p[1])
        g.add_edge((prev, v), et)
        prev, et, col, n = v, ET.SIMPLE, col + 1, n + 1
    o = g.add_vertex(VT.BOUNDARY, 0, col)
    g.add_edge((prev, o), et)
    g.set_inputs((i,))
    g.set_outputs((o,))
    assert g.num_edges() == n + 1
    g.scalar.add_phase(F(phase))
    return g


def point(kind, turn, number_phase=0):
    """A point (a spider with one leg), as a state on one output."""
    g = zx.Graph()
    v = g.add_vertex(VT.Z if kind == "Z" else VT.X, 0, 0, turn)
    o = g.add_vertex(VT.BOUNDARY, 0, 2)
    g.add_edge((v, o))
    g.set_inputs(())
    g.set_outputs((o,))
    assert g.num_edges() == 1
    g.scalar.add_phase(F(number_phase))
    return g


def sketchbook():
    rng = np.random.default_rng(1470)
    q, mq = F(1, 2), F(-1, 2)

    # --- the Euler card as two pictures: three spiders on a wire, and a yellow box
    euler = one_wire([("Z", q), ("X", q), ("Z", q)])
    ybox = one_wire([("H",)], phase=F(1, 4))
    assert np.allclose(matrix_of(one_wire([("H",)])), H)
    assert same_picture(euler, ybox)                                  # the whole an eighth round
    assert not same_picture(euler, one_wire([("H",)]))                # not without the eighth
    mirror_euler = one_wire([("Z", mq), ("X", mq), ("Z", mq)])
    assert same_picture(mirror_euler, one_wire([("H",)], phase=F(-1, 4)))
    assert same_picture(euler.adjoint(), mirror_euler)                # its mirror, seen through
    # DERIVED by PyZX: full_reduce takes each chain to a lone yellow box and its number
    for chain, ph in ((euler, F(1, 4)), (mirror_euler, F(-1, 4))):
        h = chain.copy()
        simp.full_reduce(h)
        assert same_picture(h, one_wire([("H",)], phase=ph))

    # --- Lemma P, for random turns (non-dyadic) and c = 0, 1: red point, then green alpha
    for _ in range(3):
        alpha = F(int(rng.integers(1, 96)), 97)
        for c in (0, 1):
            g = zx.Graph()
            v = g.add_vertex(VT.X, 0, 0, F(c))
            w = g.add_vertex(VT.Z, 0, 1, alpha)
            o = g.add_vertex(VT.BOUNDARY, 0, 2)
            g.add_edge((v, w))
            g.add_edge((w, o))
            g.set_outputs((o,))
            assert g.num_edges() == 2
            assert same_picture(g, point("X", F(c), c * alpha))
            if c == 1:
                assert not same_picture(g, point("X", F(c)))        # the whole really turns

    # --- Lemma A: a green quarter point is e^{i pi/4} times a red quarter-back point
    assert same_picture(point("Z", q), point("X", mq, F(1, 4)))
    assert not same_picture(point("Z", q), point("X", q, F(1, 4)))    # a quarter ROUND: no
    h = point("X", mq, F(1, 4))
    simp.full_reduce(h)                                                # DERIVED by PyZX
    assert same_picture(h, point("Z", q))

    # --- the hinge: Hub and Tie as drawn, and the chain, each picture against the one before
    hub, tie = hinge_picture("hub"), hinge_picture("tie")
    assert np.allclose(matrix_of(hub), CZ / np.sqrt(2)) and np.allclose(matrix_of(tie), CZ / np.sqrt(2))
    s_i, s_ii, s_iii = hinge_picture("i"), hinge_picture("ii"), hinge_picture("iii")
    for g, ph in ((s_i, F(1, 4)), (s_ii, F(1, 4)), (s_iii, F(1, 4))):
        g.scalar.add_phase(ph)                                         # the slip: an eighth round
    assert same_picture(hub, s_i)                                      # (i)   Lemma A
    assert same_picture(s_i, s_ii)                                     # (ii)  fusion, row 57
    assert same_picture(s_ii, s_iii)                                   # (iii) fusion backwards
    tie_iv = hinge_picture("tie")
    tie_iv.scalar.add_phase(F(1, 4))
    tie_iv.scalar.add_phase(F(-1, 4))                                  # (iv) the mirror Euler's slip
    assert same_picture(s_iii, tie_iv)
    assert same_picture(hub, tie)                                      # (v)  the slips cancel: number 1
    # NEGATIVE: the unslipped middle pictures are an eighth off; a plain tie is not the hinge
    assert not same_picture(hub, hinge_picture("i"))
    assert not same_picture(hub, hinge_picture("iii"))
    plain = hinge_picture("tie")
    plain.remove_edge(plain.edge(*[v for v in plain.vertices() if plain.type(v) == VT.Z]))
    plain.add_edge(tuple(v for v in plain.vertices() if plain.type(v) == VT.Z))
    assert plain.num_edges() == 5 and not same_picture(hub, plain)

    # DERIVATION: PyZX's full_reduce (not the book's deck) takes Hub to a picture that is Tie
    h = hub.copy()
    simp.full_reduce(h)
    assert same_picture(h, tie)
    # ...and Hub meeting the mirror of Tie reduces to two bare wires, number 1/2 (Tie Tie^dag = I/2)
    h = hub.copy()
    h.compose(tie.adjoint())
    simp.full_reduce(h)
    assert h.num_vertices() == 4 and h.num_edges() == 2
    for i, o in zip(h.inputs(), h.outputs()):
        assert h.connected(i, o) and h.edge_type(h.edge(i, o)) == ET.SIMPLE
    assert np.isclose(h.scalar.to_number(), 0.5)


if __name__ == "__main__":
    verdict("TheHinge", blackboard, sketchbook)
