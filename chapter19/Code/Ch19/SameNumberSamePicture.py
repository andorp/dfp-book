"""Challenge III: Same Number, Same Picture  (completeness, by instance only)
============================================================================
"Here's one. A tube, then a quarter round, three times over, on one cord; and a bare cord with the whole turned an eighth round. My pad says they're the same. Get from one to the other with your cards, and don't look at my numbers."  -- Achilles

Claim: three instances of what a complete deck promises -- two pictures with the same matrix,
joined by rewrites.  (1) A Clifford pair: (SH)^3 = e^{i pi/4} I, and the tube-and-quarter
picture rewrites, move by move, to a bare wire with the number e^{i pi/4} (Result 19.7).
(2) An eighth-turn pair (van de Wetering's eq. (173), with its number worked out): two green
points at alpha and alpha + pi into a red two-in-one-out spider are (1 - e^{2i alpha})/2 times
a red point at rest, for every alpha (Result 19.11).  (3) A pair with letting go: a green, red,
green turn on a wire, the opposite turns on its mirror copy, then the cap, is the bare cap, for
all turns (Result 19.9).  No file checks completeness itself: the theorems are stated, not
proved (Result 19.8 -- Backens, Theorem 21; Jeandel, Perdrix and Vilmart, Theorem 1; Ng and
Wang, Theorem 5.2; Carette, Jeandel, Perdrix and Vilmart, Corollary 1).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import CAP, CUP, I2, ket, matrix_of, random_matrix, same_picture, verdict
from dfp_props import Dialogue
import props_ch19 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

PI = np.pi
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j])
PLUS, MINUS = (ket(0) + ket(1)) / np.sqrt(2), (ket(0) - ket(1)) / np.sqrt(2)
E8 = np.exp(1j * PI / 4)


def Zt(a):
    return np.diag([1, np.exp(1j * a)])


def Xt(a):
    return np.outer(PLUS, PLUS) + np.exp(1j * a) * np.outer(MINUS, MINUS)


RED_2_1 = np.outer(PLUS, np.kron(PLUS, PLUS)) + np.outer(MINUS, np.kron(MINUS, MINUS))   # 2 -> 1, at rest


def eighth_turn_lhs(alpha):
    """Two green points, at alpha and alpha + pi, into a red spider with two legs in, one out."""
    p1 = ket(0) + np.exp(1j * alpha) * ket(1)
    p2 = ket(0) + np.exp(1j * (alpha + PI)) * ket(1)
    return RED_2_1 @ np.kron(p1, p2)


def blackboard():
    rng = np.random.default_rng(150)
    q = PI / 2

    # --- (1) the dare: a yellow box then a green quarter, three times over (read left to right)
    SH = S @ H
    assert np.allclose(SH, np.array([[1, 1], [1j, -1j]]) / np.sqrt(2))
    assert np.allclose(SH @ SH, np.array([[1 + 1j, 1 - 1j], [1 + 1j, -1 + 1j]]) / 2)
    dare = SH @ SH @ SH
    assert np.allclose(dare, E8 * I2)
    assert not np.allclose(dare, I2) and not np.allclose(dare, np.conj(E8) * I2)   # an eighth ROUND
    # the five moves of Result 19.7, each picture's matrix against the first (product right to left)
    step_i = S @ H @ S @ Xt(q)                                            # (i)  H S H -> red quarter
    assert np.allclose(H @ S @ H, Xt(q)) and np.allclose(step_i, dare)
    step_ii = S @ (np.conj(E8) * Zt(q) @ Xt(q) @ Zt(q)) @ S @ Xt(q)         # (ii) the Euler card backwards
    assert np.allclose(H, np.conj(E8) * Zt(q) @ Xt(q) @ Zt(q)) and np.allclose(step_ii, dare)
    step_iii = np.conj(E8) * Zt(PI) @ Xt(q) @ Zt(PI) @ Xt(q)               # (iii) neighbours fuse
    assert np.allclose(step_iii, dare)
    assert np.allclose(Xt(q) @ Zt(PI), 1j * Zt(PI) @ Xt(-q))               # (iv) the half passes
    step_iv = np.conj(E8) * 1j * Zt(PI) @ Zt(PI) @ Xt(-q) @ Xt(q)
    assert np.allclose(step_iv, dare)
    assert np.allclose(np.conj(E8) * 1j, E8) and np.allclose(Zt(PI) @ Zt(PI), I2)   # (v)
    # the card's hand chain: Z(q) H Z(q) X(q) = e^{i pi/4} I, and Z(pi) X(q) Z(pi) = i X(-q)
    assert np.allclose(Zt(q) @ H @ Zt(q) @ Xt(q), E8 * I2)
    assert np.allclose(Zt(PI) @ Xt(q) @ Zt(PI), 1j * Xt(-q))

    # --- (2) the eighth-turn instance, exact number, at alpha = pi/4 and for random alpha
    red_pt = np.sqrt(2) * ket(0)
    assert np.allclose(PLUS + MINUS, red_pt)                              # |+> + |-> = sqrt2 |0>
    for alpha in [PI / 4] + list(rng.uniform(0, 2 * PI, size=5)):
        number = (1 - np.exp(2j * alpha)) / 2
        assert np.allclose(eighth_turn_lhs(alpha), number * red_pt)
        assert not np.allclose(eighth_turn_lhs(alpha), red_pt)            # "up to a scalar" is not enough
    assert np.isclose((1 - np.exp(2j * PI / 4)) / 2, (1 - 1j) / 2)          # the number at an eighth
    assert not np.isclose((1 - 1j) / 2, 0)                                 # so the equation says something

    # --- (3) letting go after any turn: cap (U (x) conj U) = cap, U = Z(c) X(b) Z(a)
    for _ in range(5):
        a, b, c = rng.uniform(0, 2 * PI, size=3)
        U = Zt(c) @ Xt(b) @ Zt(a)                                         # green a, red b, green c
        Ubar = Zt(-c) @ Xt(-b) @ Zt(-a)                                   # the mirror copy, turns reversed
        assert np.allclose(Ubar, U.conj())                                # ...is the conjugate
        assert not np.allclose(U, U.T) and not np.allclose(U, U.conj())   # a generic turn
        assert np.allclose(CAP @ np.kron(U, Ubar), CAP)
        assert not np.allclose(CAP @ np.kron(U, U), CAP)                  # turns not reversed: no
        assert not np.allclose(CAP @ np.kron(U, U.conj().T), CAP)         # reversed AND reordered: no
    # and for any unitary at all (row 72), a random one from QR
    Q, _ = np.linalg.qr(random_matrix(rng))
    assert np.allclose(CAP @ np.kron(Q, Q.conj()), CAP)
    rho = random_matrix(rng)
    rho = rho @ rho.conj().T
    assert np.isclose(np.trace(Q @ rho @ Q.conj().T), np.trace(rho))      # tr(U rho U^dag) = tr rho
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))                    # the cap is unnormalized

    pictures()


# -------------------------------------------------------------------- the prop pictures (dare)

_HEAD = " | cols=11 rows=1 y=0..0 near=false folded=false; "
THIRD_DUMPS = (
    "laid" + _HEAD + "corner=; 1:ring green 6 @2,0; 2:ring green 6 @4,0; 3:ring green 6 @8,0; tube @1,0 a0; "
    "tube @3,0 a0; tube @6,0 a0; cord loose-loose [0,0 10,0]",
    "through,backwards" + _HEAD + "corner=21; 1:ring red 6 @2,0; 2:ring green 6 @4,0; 3:ring green 6 @5,0; "
    "4:ring red 6 @6,0; 5:ring green 6 @7,0; 6:ring green 6 @8,0; cord loose-loose [0,0 10,0]",
    "through,backwards,halves,passed" + _HEAD + "corner=21 6; 1:ring red 6 @2,0; 2:ring red 18 @6,0; "
    "3:ring green 12 @7,0; 4:ring green 12 @8,0; cord loose-loose [0,0 10,0]",
    "through,backwards,halves,passed,bare,one" + _HEAD + "corner=3; cord loose-loose [0,0 10,0]")
DARE_STEPS = ("drawn through", "card backwards", "halves", "half passed", "bare cord", "one slip")


def with_slips(dump):
    """graph_of a cloth dump (a structural reading of the picture, not a second proof), with the
    corner's slips written on its scalar: the whole picture's number."""
    g = PC.graph_of(dump)
    g.scalar.add_phase(PC.slips_turn(dump))
    return g


def pictures():
    """The Third Challenge's prop pictures (identity card, section 4), read from the dialogue
    through tools/dfp_props.py and the transcription tools/props_ch19.py, and held against this
    file's dare and the Tortoise's five moves."""
    calls = DIALOGUE.check_layout("Third", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("darecloth", True, "", (), "Achilles's dare"),
        ("darecloth", False, "drawn through, card backwards", (), "Read backwards"),
        ("darecloth", False, "drawn through, card backwards, halves, half passed", (), "Slid past"),
        ("darecloth", False, "drawn through, card backwards, halves, half passed, bare cord, one slip", (), "A new slip")]
    PC.check_captions(DIALOGUE, "Third", calls)
    PC.check_section(DIALOGUE, "Third", calls)
    PC.check_anchors(DIALOGUE, "Third")
    pics = [PC.picture(c) for c in calls]
    p31, p32, p33, p34 = pics
    assert tuple(p.dump for p in pics) == THIRD_DUMPS
    states = [""] + [", ".join(DARE_STEPS[:k]) for k in range(1, 7)]
    assert {PC.darecloth(st).frame for st in states} == {p31.frame}      # one frame in every state
    # the captions, true of their dumps
    heads = [p.dump.split(" |")[0] for p in pics]
    assert heads[0] == "laid" and "backwards" in heads[1] and "passed" in heads[2]
    assert PC.parse_dump(p34.dump)[4] == [3] and PC.parse_dump(PC.darecloth(states[5]).dump)[4] == [21, 6]

    # P3.1, Achilles's dare as laid: this file's dare, read ring for ring -- a tube, then a green
    # quarter round, three times over -- and exactly e^{i pi/4} times the bare cord
    q = F(1, 2)
    dare = one_wire([("H",), ("Z", q), ("H",), ("Z", q), ("H",), ("Z", q)])
    g31 = PC.graph_of(p31.dump)
    assert same_picture(g31, dare) and np.allclose(matrix_of(g31), E8 * I2)
    assert not np.allclose(matrix_of(g31), I2)
    assert PC.parse_dump(p31.dump)[4] == []                              # the corner empty

    # every state of the trace, read ring for ring, is the picture of the Tortoise's moves (the
    # file's chain), and with its corner's slips equals P3.1, exactly
    moves = [
        [("H",), ("Z", q), ("H",), ("Z", q), ("H",), ("Z", q)],                    # laid
        [("X", q), ("Z", q), ("H",), ("Z", q)],                                    # (i)   drawn through
        [("X", q), ("Z", q), ("Z", q), ("X", q), ("Z", q), ("Z", q)],               # (ii)  backwards
        [("X", q), ("Z", 1), ("X", q), ("Z", 1)],                                  # (iii) halves
        [("X", q), ("X", -q), ("Z", 1), ("Z", 1)],                                 # (iv)  half passed
        [],                                                                        # (v)   bare cord
        []]                                                                        #       one slip
    for st, mv in zip(states, moves):
        dump = PC.darecloth(st).dump
        assert same_picture(PC.graph_of(dump), one_wire(mv)), st
        assert same_picture(with_slips(dump), g31), st
    # NEGATIVE: without its slips, every picture after the first two is off the dare
    for st in states[2:]:
        assert not same_picture(PC.graph_of(PC.darecloth(st).dump), g31), st
    # P3.4: the two slips become one, an eighth round, the whole the dare's
    assert PC.slips_turn(p34.dump) == PC.slips_turn(PC.darecloth(states[5]).dump) == F(1, 4)
    assert not same_picture(PC.graph_of(p34.dump), g31)

    # fixed parts: the cord's two loose ends at 0 and 10 in every picture; the red ring at 2 kept
    # from P3.2 to P3.3 (no step between moves it)
    for pic in pics:
        assert PC.parse_dump(pic.dump)[3] == [("loose", "loose", [(0, 0), (10, 0)])]
    red = lambda pic: [v for v in PC.parse_dump(pic.dump)[0].values() if v[3] == 2]
    assert red(p32) == red(p33) == [("ring", "red", 6, 2, 0)]

    # the keyed word (P3.1): the kit's tube, its no-room fault the check; the rest unkeyed
    assert p31.labels == ("tube",) and PC.own_words(p31) == ()
    assert p32.labels == p33.labels == p34.labels == ()

    # the Lua's refusals: an unknown step; a step twice; a step before the one it follows
    PC.TEX_ERRORS.clear()
    PC.dare_state("tubes taken off")
    PC.dare_state("drawn through, drawn through")
    PC.dare_state("drawn through, halves")
    assert [k for k, _ in PC.TEX_ERRORS] == ["step-unknown", "step-done", "step-after"]
    PC.TEX_ERRORS.clear()


# ----------------------------------------------------------------------------- the sketchbook
F = Fraction


def one_wire(pieces, number_phase=0):
    """A chain on one wire: ('Z', t), ('X', t) are two-legged spiders, ('H',) a yellow box
    (a Hadamard edge into the next piece)."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    prev, et, col, n = i, ET.SIMPLE, 1, 0
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
    g.scalar.add_phase(F(number_phase))
    return g


def eighth_turn_picture(alpha):
    g = zx.Graph()
    p1 = g.add_vertex(VT.Z, 0, 0, alpha)
    p2 = g.add_vertex(VT.Z, 2, 0, alpha + 1)
    r = g.add_vertex(VT.X, 1, 1)
    o = g.add_vertex(VT.BOUNDARY, 1, 2)
    for e in ((p1, r), (p2, r), (r, o)):
        g.add_edge(e)
    g.set_outputs((o,))
    assert g.num_edges() == 3
    return g


def red_point(number=None):
    g = zx.Graph()
    v, o = g.add_vertex(VT.X, 0, 0), g.add_vertex(VT.BOUNDARY, 0, 1)
    g.add_edge((v, o))
    g.set_outputs((o,))
    assert g.num_edges() == 1
    if number is not None:
        g.scalar.add_float(number)
    return g


def let_go(top, copy):
    """Doubled by hand (pyzx 0.10.6 has no discard): a wire carrying green, red, green turns
    `top`, its mirror copy carrying `copy`, and the cap joining them -- a two-legged green
    spider at rest.  Built as a STATE: the two open ends are outputs (wire first, copy second),
    so its vector has the entries of the effect."""
    g = zx.Graph()
    o_w, o_c = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0)
    chain_w = [g.add_vertex(k, 0, 1 + j, t) for j, (k, t) in enumerate(zip((VT.Z, VT.X, VT.Z), top))]
    chain_c = [g.add_vertex(k, 1, 1 + j, t) for j, (k, t) in enumerate(zip((VT.Z, VT.X, VT.Z), copy))]
    cap = g.add_vertex(VT.Z, 0.5, 5)
    n = 0
    for o, chain in ((o_w, chain_w), (o_c, chain_c)):
        for u, v in zip([o] + chain, chain + [cap]):
            g.add_edge((u, v))
            n += 1
    g.set_outputs((o_w, o_c))
    assert g.num_edges() == n == 8
    return g


def bare_cap():
    g = zx.Graph()
    o_w, o_c = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0)
    cap = g.add_vertex(VT.Z, 0.5, 1)
    g.add_edge((o_w, cap))
    g.add_edge((cap, o_c))
    g.set_outputs((o_w, o_c))
    assert g.num_edges() == 2
    return g


def sketchbook():
    rng = np.random.default_rng(1500)
    q = F(1, 2)

    # --- (1) DERIVATION, Clifford: the dare's picture, and PyZX's full_reduce takes it to a
    # bare wire with the number e^{i pi/4}
    dare = one_wire([("H",), ("Z", q), ("H",), ("Z", q), ("H",), ("Z", q)])
    assert np.allclose(matrix_of(dare), S @ H @ S @ H @ S @ H)
    bare8 = one_wire([], number_phase=F(1, 4))
    assert same_picture(dare, bare8)
    assert not same_picture(dare, one_wire([]))                         # not without the slip
    h = dare.copy()
    simp.full_reduce(h)
    assert h.num_vertices() == 2 and h.num_edges() == 1
    assert h.edge_type(h.edge(h.inputs()[0], h.outputs()[0])) == ET.SIMPLE
    assert np.isclose(h.scalar.to_number(), E8)
    # the Tortoise's five moves, each picture against the dare with its slip
    chain = [
        (one_wire([("X", q), ("Z", q), ("H",), ("Z", q)]), 0),                                   # (i)
        (one_wire([("X", q), ("Z", q), ("Z", q), ("X", q), ("Z", q), ("Z", q)]), F(-1, 4)),       # (ii)
        (one_wire([("X", q), ("Z", 1), ("X", q), ("Z", 1)]), F(-1, 4)),                           # (iii)
        (one_wire([("X", q), ("X", -q), ("Z", 1), ("Z", 1)]), F(-1, 4) + F(1, 2)),                # (iv)
        (one_wire([]), F(1, 4)),                                                                  # (v)
    ]
    for g, slip in chain:
        g.scalar.add_phase(slip)
        assert same_picture(g, dare)
    assert not same_picture(one_wire([("X", q), ("X", -q), ("Z", 1), ("Z", 1)]), dare)   # slips count

    # --- (2) (T2) TRANSLATION CHECK, the eighth-turn instance at alpha = pi/4: each side reduced
    # by full_reduce, the reductions compared through matrix_of, with the number (1 - i)/2.
    # PyZX's simplifier is NOT the book's deck: this is an instance compared after reduction,
    # not a derivation by the book's cards (Section 19.5: the book has derived no such chain).
    lhs = eighth_turn_picture(F(1, 4))
    rhs = red_point((1 - 1j) / 2)
    assert same_picture(lhs, rhs)
    assert not same_picture(lhs, red_point())                           # number 1: wrong
    lr, rr = lhs.copy(), rhs.copy()
    simp.full_reduce(lr)
    simp.full_reduce(rr)
    assert same_picture(lr, rr) and same_picture(lr, lhs)
    # ...and for random, non-dyadic alpha (Result 19.11 says every alpha): tensors compared
    for _ in range(3):
        alpha = F(int(rng.integers(1, 96)), 97)
        num = (1 - np.exp(2j * PI * float(alpha))) / 2
        assert same_picture(eighth_turn_picture(alpha), red_point(num))
        lr = eighth_turn_picture(alpha)
        simp.full_reduce(lr)
        assert same_picture(lr, red_point(num))

    # --- (3) DERIVATION, with letting go: a turn on the wire, the reversed turns on its copy,
    # the cap; full_reduce leaves the bare cap, number 1 -- for pi/4-multiples and generic turns
    turns = [(F(1, 4), F(3, 4), F(1, 2)), (F(7, 4), F(1, 4), F(5, 4))]
    turns += [tuple(F(int(k), 97) for k in rng.integers(1, 194, size=3)) for _ in range(3)]
    for a, b, c in turns:
        g = let_go((a, b, c), (-a, -b, -c))
        assert np.allclose(matrix_of(g).ravel(), CAP.ravel())
        assert same_picture(g, bare_cap())
        h = g.copy()
        simp.full_reduce(h)
        assert h.num_vertices() == 2 and h.num_edges() == 1
        assert np.isclose(h.scalar.to_number(), 1)
        assert same_picture(h, bare_cap())
        # NEGATIVE: the copy with the same turns, or with the reversed turns in reversed order
        # (the dagger, not the conjugate), is not let go
        assert not same_picture(let_go((a, b, c), (a, b, c)), bare_cap())
        assert not same_picture(let_go((a, b, c), (-c, -b, -a)), bare_cap())


if __name__ == "__main__":
    verdict("SameNumberSamePicture", blackboard, sketchbook)
