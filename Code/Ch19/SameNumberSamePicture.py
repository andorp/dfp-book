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

import re
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
THIRD_FRAMES = (
    ("Achilles's dare", True, None, THIRD_DUMPS[0]),
    ("Tubes into the ring", False, "drawn through",
     "through" + _HEAD + "corner=; 1:ring red 6 @2,0; 2:ring green 6 @4,0; 3:ring green 6 @8,0; tube @6,0 a0; "
     "cord loose-loose [0,0 10,0]"),
    ("Read backwards", False, None, THIRD_DUMPS[1]),
    ("Greens together", False, "halves",
     "through,backwards,halves" + _HEAD + "corner=21; 1:ring red 6 @2,0; 2:ring green 12 @5,0; 3:ring red 6 @6,0; "
     "4:ring green 12 @8,0; cord loose-loose [0,0 10,0]"),
    ("Green slid past red", False, "half passed", THIRD_DUMPS[2]),
    ("Rings gone", False, "bare cord", "through,backwards,halves,passed,bare" + _HEAD + "corner=21 6; cord loose-loose [0,0 10,0]"),
    ("A new slip", False, "one slip", THIRD_DUMPS[3]))
DARE_STEPS = ("drawn through", "card backwards", "halves", "half passed", "bare cord", "one slip")


def with_slips(dump):
    """graph_of a cloth dump (a structural reading of the picture, not a second proof), with the
    corner's slips written on its scalar: the whole picture's number."""
    g = PC.graph_of(dump)
    g.scalar.add_phase(PC.slips_turn(dump))
    return g


def nearest(pt, places):
    """The place nearest a mark's end, and how far: (distance, place)."""
    return min((float(np.hypot(pt[0] - q[0], pt[1] - q[1])), tuple(q)) for q in places)


def mark_bounds(R, g):
    """Every mark on its drawn extent (the stroke's half width, a ghost's outline included) inside
    the picture's box by at least 0.01 cm (the spec's margin rule); every arc bowed at least the
    Lua's floor (MINSAG) and clear of the cloth's pieces by its clearance (MINCLEAR); no mark of
    any other kind (no pin; a ghost, a slide arc or a lift arc only)."""
    fb, ft, half = g._g["fbot"], g._g["ftop"], PC.C.FRAME_WIDTH / 2
    gap = lambda x, y, r: min(x - r + half, half - x - r, y - r - fb, ft - y - r)
    for a in R.arcs:
        assert a.kind in ("slide", "swing") and a.sag >= PC.MINSAG - 1e-9 and a.fit[0] >= PC.MINCLEAR - 1e-9, a.what
        assert min(gap(x, y, PC.MARKHALF) for x, y in a.pts) >= 0.01, a.what
    for q in R.ghosts:
        assert gap(q[0], q[1], q[2] + PC.GHOSTHALF) >= 0.01, q.what
    assert len(R.mark_tikz) >= len(R.arcs) + len(R.ghosts)
    draws = [t for t in R.mark_tikz if t.startswith("\\draw")]
    assert bool(draws) == bool(R.arcs or R.ghosts) and all(re.match(r"\\draw\[propcnineteen(ghost|slide|swing)[,\]]", t) for t in draws)
    for style in ("ghost", "slide", "swing"):                              # grey, thin, dotted (ch19.tex)
        assert re.search(r"propcnineteen" + style + r"/\.style=\{(densely )?dotted, black!\d\d, line width=0\.\d+pt\}",
                         PC.TEX_TEXT), style


# the dare's marks, step by step (the Lua's comment, pinned): (arcs as (from column, to column),
# ghosted columns); the lift arc of one slip runs from the second slip onto the first
DARE_MARKS = {"through": ([(1, 2), (3, 2)], []),          # each tube hops into the ring; no tube ghosted
              "backwards": None,                           # no marks (her work-bag, her box, her packet)
              "halves": ([(4, 5), (7, 8)], [4, 7]),        # each left-hand green into the right-hand one
              "passed": ([(5, 7)], [5]),                   # the half past the red ring to just beyond it
              "bare": ([(2, 6), (7, 8)], [2, 6, 7, 8]),    # E2: left into right, the reds and the green halves
              "one": "lift"}                               # E3: the slip a quarter round onto the slip an eighth back


def dare_marks(R):
    S0, S1 = PC.dare_scene(R.before), PC.dare_scene(R.after)
    g = PC.G(S1, PC.DOPTS, PC.DWHO)
    mark_bounds(R, g)
    key = PC._key_of(PC.DKEYS, R.after)
    want = DARE_MARKS[key]
    if want is None:
        assert not R.marked and R.arcs == R.ghosts == [] and R.notes[-1] == "no marks: " + PC.DMARKLESS[key]
        return
    assert R.marked
    col = lambda x: min((abs(x - g.X(c)), c) for c in range(S1["cols"]))     # (distance, column)
    if want == "lift":
        assert S0["corner"] == [21, 6] and S1["corner"] == [3]           # an eighth back and a quarter round: one slip
        (a,) = R.arcs
        (q,) = R.ghosts                                                    # the first's place holds the new slip
        s1, s2 = PC.slipxy(g, 1), PC.slipxy(g, 2)
        assert a.kind == "swing" and np.allclose(q[:2], s2)
        assert np.isclose(a.frm[0], s2[0]) and np.isclose(a.to[0], s1[0]) and a.frm[1] > 0 and a.to[1] > 0
        return
    arcs, ghosted = want
    got = []
    for a in R.arcs:
        (d0, c0), (d1, c1) = col(a.frm[0]), col(a.to[0])
        assert a.kind == "slide" and d0 < 0.1 and d1 < 0.1 and a.frm[1] > g.Y(0) and a.to[1] > g.Y(0), a.what
        got.append((c0, c1))
    assert got == arcs, (key, got)
    assert sorted(round(q[0], 6) for q in R.ghosts) == sorted(round(g.X(c), 6) for c in ghosted), key
    old = {t["x"] for t in S0["things"]} | {t["x"] for t in S0["tubes"]}
    new = {t["x"] for t in S1["things"]}
    assert all(a in old and b in new | set(ghosted) for a, b in arcs)       # into a piece after, or one this step ghosts
    assert set(ghosted) <= {t["x"] for t in S0["things"]} - new
    if key == "bare":
        assert all(a < b for a, b in arcs)                                 # E2: each time left into right
    if key in ("halves", "bare"):                                          # like moves: every one or none, alike
        assert np.isclose(R.arcs[0].sag, R.arcs[1].sag)


def pictures():
    """The Third Challenge's prop pictures (identity card, sections 4 and 3e), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch19.py, and held against
    this file's dare and the Tortoise's five moves.  Round 4 draws every move of the dare, each frame
    the state after one step with its marks; the last two moves a strip."""
    calls = DIALOGUE.check_layout("Third", sizes=[1] * 6)
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("darecloth", True, "", (), "Achilles's dare"),
        ("darecloth", False, "; step=drawn through", (), "Tubes into the ring"),
        ("darecloth", False, "drawn through, card backwards", (), "Read backwards"),
        ("darecloth", False, "drawn through, card backwards; step=halves", (), "Greens together"),
        ("darecloth", False, "drawn through, card backwards, halves; step=half passed", (), "Green slid past red"),
        ("dareclothsteps", False, "after=drawn through, card backwards, halves, half passed; from=off", ("bare cord; one slip",),
         "Rings gone; A new slip")]
    assert PC.check_captions(DIALOGUE, "Third", calls) == 7
    blocks = PC.check_section(DIALOGUE, "Third", calls, [1, 1, 1, 1, 1, 2])
    assert PC.check_anchors(DIALOGUE, "Third") == (8, 20)              # next-starts, phrases before (E2, E3 among them)
    for f_ in (DIALOGUE.dialog, DIALOGUE.raw):
        assert PC.E_ITEMS["E2"] in DIALOGUE.anchors("Third", f_)[5] and PC.E_ITEMS["E3"] in DIALOGUE.anchors("Third", f_)[5]
    assert '"' + PC.E_ITEMS["E2"].rstrip(".") + '", E2' in PC.comment_prose("round 4: \\darecloth[")
    frames = [R for b in blocks for R, _, _ in b]
    assert tuple((cap, k, R.step, R.dump) for b in blocks for R, cap, k in b) == THIRD_FRAMES      # the readings
    # round 3's four pictures are their blocks' last frames, unchanged
    pics = [frames[i] for i in (0, 2, 4, 6)]
    p31, p32, p33, p34 = pics
    assert tuple(p.dump for p in pics) == THIRD_DUMPS == tuple(R.picture.dump for R in pics)
    states = [""] + [", ".join(DARE_STEPS[:k]) for k in range(1, 7)]
    assert {PC.darecloth(st).frame for st in states} == {p31.frame} == {R.frame for R in frames}     # one frame
    # the chain, frame to frame: each frame the dare as its bracket leaves it, each step frame the step
    # applied to the frame before (the strip's first: to the state its after= names)
    for k, R in enumerate(frames):
        assert R.dump == PC.darecloth(states[k]).dump, k
        if R.step is not None:
            assert repr(R.before) == repr(frames[k - 1].drawn), k
    assert [R.step for R in frames] == [None, "drawn through", None, "halves", "half passed", "bare cord", "one slip"]
    for R in frames[1:]:
        if R.step is not None:
            dare_marks(R)
    assert [(len(R.ghosts), len(R.arcs)) for R in frames] == [(0, 0), (0, 2), (0, 0), (2, 2), (1, 1), (4, 2), (1, 1)]
    # card backwards, drawn alone, has no marks (the Lua's markless step)
    dare_marks(PC.run_frame("dare", "drawn through; step=card backwards", "", False))
    # no prediction frame: expects= refused with the Lua's reason, which quotes the dialogue
    assert PC.faults(PC.run_frame, "dare", "; expects=a bare cord", "", False) == ["dare-expects"]
    for q in re.findall(r'"([^"]+)"', PC.DEXPECTS):
        assert all(q in DIALOGUE.read(f) for f in (DIALOGUE.dialog, DIALOGUE.raw)), q
    PC.TEX_ERRORS.clear()
    # the captions, true of their dumps
    heads = [p.dump.split(" |")[0] for p in pics]
    assert heads[0] == "laid" and "backwards" in heads[1] and "passed" in heads[2]
    # round 4's captions: the ring red where the tubes went in; each green pair one green half
    n = PC.parse_dump(frames[1].dump)[0]
    assert [v[:2] for v in n.values() if v[3] == 2] == [("ring", "red")] and len(PC.parse_dump(frames[1].dump)[1]) == 1
    assert sorted((v[1], v[2]) for v in PC.parse_dump(frames[3].dump)[0].values() if v[1] == "green") == [("green", 12)] * 2
    assert PC.parse_dump(frames[5].dump)[0] == {} and PC.parse_dump(frames[5].dump)[4] == [21, 6]     # rings gone
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
    for pic in frames:
        assert PC.parse_dump(pic.dump)[3] == [("loose", "loose", [(0, 0), (10, 0)])]
    red = lambda pic: [v for v in PC.parse_dump(pic.dump)[0].values() if v[3] == 2]
    assert red(p32) == red(p33) == [("ring", "red", 6, 2, 0)]

    # the keyed word (P3.1): the kit's tube, its no-room fault the check; the rest unkeyed
    assert p31.labels == ("tube",) and PC.own_words(p31) == ()
    assert all(R.labels == () for R in frames[1:])

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
