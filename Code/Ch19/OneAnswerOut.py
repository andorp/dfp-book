"""Challenge IV: One Answer Out  (a run is a closed box; the master, The Closed Box)
===================================================================================
"There's the Cat's own point, on the answer's cord, just as it lay before anybody looked."  -- Tortoise
"Not any look. Yours copied asleep-or-awake, and mine only watched you. A look that asked the Cat some other question could change them."  -- Tortoise

Claim: The Closed Box, built as drawn, equals the Cat's own green point at rest on the answer
wire, number 1: the classical vector (1, 1), odds 1/2 and 1/2 (Result 19.12); PyZX's
full_reduce derives it.  Whoever watched, by a look that copies the Cat's asleep-or-awake or
watches only the watchers, the Cat's odds are unchanged, for any state and any unitaries in the
looks (Result 19.13) -- and a look that copies the Cat in another basis can change them.  One
run of a closed box gives the answer's odds; a majority of three runs each right with
probability p is right with probability f(p) = 3p^2 - 2p^3, and p >= 2/3 gives f(p) >= 20/27
(Result 19.10).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction
from itertools import product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import CAP, I2, ket, kron, matrix_of, random_matrix, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch19 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)
CNOT = np.eye(4, dtype=complex)[[0, 1, 3, 2]]
PLUS, MINUS = (ket(0) + ket(1)) / np.sqrt(2), (ket(0) - ket(1)) / np.sqrt(2)
P0, P1 = np.outer(ket(0), ket(0)), np.outer(ket(1), ket(1))

GREEN_1_2 = np.outer(ket(0, 0), ket(0)) + np.outer(ket(1, 1), ket(1))            # copies 0/1
RED_2_1 = np.outer(PLUS, np.kron(PLUS, PLUS)) + np.outer(MINUS, np.kron(MINUS, MINUS))
READ = np.outer(ket(0), ket(0, 0)) + np.outer(ket(1), ket(1, 1))                  # green 2 -> 1


def unitary(rng, n):
    Q, R = np.linalg.qr(random_matrix(rng, n))
    return Q @ np.diag(np.diag(R) / abs(np.diag(R)))


def cat_odds(rho):
    """The Cat is the top wire of three; the odds are the diagonal of its part."""
    r = rho.reshape(2, 4, 2, 4)
    return np.real(np.diag(np.einsum("iaja->ij", r))), np.einsum("iaja->ij", r)


def f(p):
    """Majority of three independent runs, each right with probability p: count the ways."""
    total = 0
    for runs in product((True, False), repeat=3):
        weight = 1
        for right in runs:
            weight *= p if right else 1 - p
        if sum(runs) >= 2:
            total += weight
    return total


def blackboard():
    rng = np.random.default_rng(151)

    # --- the bare look (row 64): a green spider tied to a red one is CNOT / sqrt2
    joiner = np.kron(I2, RED_2_1) @ np.kron(GREEN_1_2, I2)
    assert np.allclose(joiner, CNOT / np.sqrt(2))
    assert np.allclose(np.sqrt(2) * joiner, np.kron(P0, I2) + np.kron(P1, X))     # kind (a)
    # --- The Closed Box, undoubled: Cat |0> + |1>, both pages sqrt2 |0>, top wire first
    cat, page = ket(0) + ket(1), np.sqrt(2) * ket(0)
    start = kron(cat, page, page).ravel()
    after_achilles = np.kron(joiner, I2) @ start
    assert np.allclose(after_achilles, np.sqrt(2) * np.kron(ket(0, 0) + ket(1, 1), ket(0)))
    psi = np.kron(I2, joiner) @ after_achilles
    assert np.allclose(psi, ket(0, 0, 0) + ket(1, 1, 1))                     # all three records agree
    # doubled: psi (x) conj(psi); both pages let go (caps), the Cat read (the read-spider)
    doubled = np.einsum("cat,CAT->catCAT", psi.reshape(2, 2, 2), psi.conj().reshape(2, 2, 2))
    value = np.einsum("catCAT,aA,tT,rcC->r", doubled, CAP.reshape(2, 2), CAP.reshape(2, 2),
                      READ.reshape(2, 2, 2))
    assert np.allclose(value, [1, 1])                                        # the Cat's own point
    rho_cat = np.einsum("catCat->cC", doubled)                               # partial trace
    assert np.allclose(rho_cat, I2) and np.allclose(np.diag(rho_cat), value)
    assert np.isclose(value.sum(), 2) and not np.isclose(value.sum(), 1)     # closed: 2, not 1
    assert np.allclose(value / value.sum(), [0.5, 0.5])                      # normalized: even odds
    before = np.abs(cat) ** 2
    assert np.allclose(before / before.sum(), value / value.sum())           # as before anyone looked

    # --- Result 19.13: whoever watched, by these two kinds of look.  Random non-real states
    # (an entangled pure state of Cat, Achilles, Tortoise, and a random mixed one) and random
    # unitaries inside the looks.
    phi = random_state(rng, 8)
    assert not np.allclose(phi, phi.conj())
    B = random_matrix(rng, 8)
    mixed = B @ B.conj().T
    mixed /= np.trace(mixed)
    for rho in (np.outer(phi, phi.conj()), mixed):
        odds0, part0 = cat_odds(rho)
        state = rho
        for step in range(4):
            U0, U1 = unitary(rng, 4), unitary(rng, 4)
            kind_a = np.kron(P0, U0) + np.kron(P1, U1)                  # copies asleep-or-awake
            Ut = unitary(rng, 2)
            tortoise = np.kron(I2, np.kron(P0, I2) + np.kron(P1, Ut))   # watches only Achilles
            kind_b = np.kron(I2, unitary(rng, 4))                       # anything on the pages
            for W in (kind_a, tortoise, kind_b):
                assert np.allclose(W.conj().T @ W, np.eye(8))
                state = W @ state @ W.conj().T
                odds, part = cat_odds(state)
                assert np.allclose(odds, odds0)                         # the odds never move
        assert not np.allclose(part, part0)                             # ...though the Cat's part does
    # NEGATIVE: a look that asks the Cat another question.  The chapter's case: the Cat surely
    # asleep, a look that copies its |+>/|-> onto a blank page
    look_pm = np.kron(np.outer(PLUS, PLUS), I2) + np.kron(np.outer(MINUS, MINUS), X)
    out = look_pm @ np.kron(ket(0), ket(0))
    assert np.allclose(out, (np.kron(PLUS, ket(0)) + np.kron(MINUS, ket(1))) / np.sqrt(2))
    rho_c = np.einsum("iaja->ij", np.outer(out, out.conj()).reshape(2, 2, 2, 2))
    assert np.allclose(rho_c, I2 / 2)                                   # asleep or awake, evenly
    assert not np.allclose(np.diag(rho_c), [1, 0])                      # it was surely asleep
    # and generically: a look controlled on |+>/|-> with random unitaries moves random odds
    rho = np.outer(phi, phi.conj())
    W = (np.kron(np.outer(PLUS, PLUS), unitary(rng, 4))
         + np.kron(np.outer(MINUS, MINUS), unitary(rng, 4)))
    assert not np.allclose(cat_odds(W @ rho @ W.conj().T)[0], cat_odds(rho)[0])

    # --- a run (Section 19.4): points in, U, the first wire read, the second let go; with the
    # number 1/4 for the four red points the value is the answer's odds (p0, p1)
    U = unitary(rng, 4)
    out = U @ kron(np.sqrt(2) * ket(0), np.sqrt(2) * ket(0)).ravel()
    doubled = np.einsum("ab,AB->abAB", out.reshape(2, 2), out.conj().reshape(2, 2))
    odds = np.einsum("abAB,bB,raA->r", doubled, CAP.reshape(2, 2), READ.reshape(2, 2, 2)) / 4
    zero = kron(ket(0), ket(0)).ravel()
    p1 = np.vdot(zero, U.conj().T @ np.kron(P1, I2) @ U @ zero)
    assert np.allclose(odds, [1 - p1.real, p1.real]) and np.isclose(odds.sum(), 1)
    assert np.isclose(p1.imag, 0)

    # --- Result 19.10: a majority of three, exactly, with Fractions
    Fr = Fraction
    for p in (Fr(2, 3), Fr(1, 2), Fr(9, 10), Fr(0), Fr(1)):
        assert f(p) == 3 * p ** 2 - 2 * p ** 3
    assert f(Fr(2, 3)) == Fr(20, 27) and Fr(20, 27) > Fr(2, 3)
    assert f(Fr(2, 3)) == Fr(8, 27) + 3 * Fr(4, 27)
    grid = [Fr(2, 3) + Fr(k, 300) for k in range(101)]                 # from 2/3 to 1
    assert grid[-1] == 1
    for p in grid:
        assert f(p) >= Fr(20, 27)
        assert 6 * p * (1 - p) >= 0                                     # f'(p) = 6p(1 - p)
    assert f(Fr(2, 3) - Fr(1, 300)) < Fr(20, 27)                        # the bound is tight at 2/3
    # Dirac's Blackboard: the same polynomial as Chapter 18's failure, read the other way
    Fail = lambda p: 3 * p ** 2 - 2 * p ** 3
    for p in grid + [Fr(k, 50) for k in range(51)]:
        assert f(p) == 1 - Fail(1 - p) and Fail(p) + Fail(1 - p) == 1
        assert f(p) - p == p * (1 - p) * (2 * p - 1)
        if Fr(1, 2) < p < 1:
            assert f(p) > p                                              # each vote raises it
    assert f(Fr(20, 27)) == Fr(16400, 19683)                            # a vote of votes
    assert [p for p in (Fr(k, 1000) for k in range(1001)) if f(p) == p] == [0, Fr(1, 2), 1]

    pictures()


# ------------------------------------------------------------- the prop pictures (box and lute)

_HEAD = " | cols=11 rows=7 y=0..6 near=false folded=false; corner=; "
FOURTH_DUMPS = (
    "laid knots=2" + _HEAD + "1:end green 0 @0,0; 2:ring green 0 @1,0; 3:end red 0 @0,1; 4:ring red 0 @1,1; "
    "5:ring green 0 @2,1; 6:end red 0 @0,2; 7:ring red 0 @2,2; 8:ring green 0 @6,3; 9:end red 0 @0,4; "
    "10:ring red 0 @2,4; 11:end red 0 @0,5; 12:ring red 0 @1,5; 13:ring green 0 @2,5; 14:end green 0 @0,6; "
    "15:ring green 0 @1,6; tie 10-13 []; tie 12-15 []; tie 2-4 []; tie 5-7 []; cord 1-8 [0,0 5,0 6,3]; "
    "cord 3-11 [0,1 4,1 4,5 0,5]; cord 6-9 [0,2 3,2 3,4 0,4]; cord 8-14 [6,3 5,6 0,6]; cord loose-8 [10.4,3 6,3]",
    "reds,together knots=2" + _HEAD + "1:ring green 0 @0,0; 2:ring green 0 @6,3; 3:ring green 0 @0,6; "
    "cord 1-2 [0,0 5,0 6,3]; cord 1-3 [0,0 0,2 3,2 3,4 0,4 0,6]; cord 1-3 [0,0 1,1 4,1 4,5 1,5 0,6]; "
    "cord 2-3 [6,3 5,6 0,6]; cord loose-2 [10.4,3 6,3]",
    "reds,together,loops knots=0" + _HEAD + "1:end green 0 @0,6; cord loose-1 [10.4,3 6,3 5,6 0,6]",
    "top | top=sounding pairs=still courses=1+9")
FOURTH_FRAMES = (
    ("Pages and looks", True, None, FOURTH_DUMPS[0]),
    ("Red points slid in", False, "red points slid in",
     "reds knots=2" + _HEAD + "1:end green 0 @0,0; 2:ring green 0 @1,0; 3:ring green 0 @2,1; 4:ring green 0 @6,3; "
     "5:ring green 0 @2,5; 6:end green 0 @0,6; 7:ring green 0 @1,6; cord 1-4 [0,0 5,0 6,3]; "
     "cord 2-7 [1,0 1,1 4,1 4,5 1,5 1,6]; cord 3-5 [2,1 2,2 3,2 3,4 2,4 2,5]; cord 4-6 [6,3 5,6 0,6]; "
     "cord loose-4 [10.4,3 6,3]"),
    ("His look along", False, "his look's ring slid along",
     "reds,part1 knots=2" + _HEAD + "1:ring green 0 @0,0; 2:ring green 0 @2,1; 3:ring green 0 @6,3; "
     "4:ring green 0 @2,5; 5:ring green 0 @0,6; cord 1-3 [0,0 5,0 6,3]; cord 1-5 [0,0 1,1 4,1 4,5 1,5 0,6]; "
     "cord 2-4 [2,1 2,2 3,2 3,4 2,4 2,5]; cord 3-5 [6,3 5,6 0,6]; cord loose-3 [10.4,3 6,3]"),
    ("Her look follows", False, "her look's ring slid along", FOURTH_DUMPS[1]),
    ("Loops lifted off", False, "loops off", FOURTH_DUMPS[2]),
    ("The lute, quiet", True, None, "still | top=still pairs=still courses=1+9"),
    ("Top note alone", False, "top string sounding", FOURTH_DUMPS[3]))
BOX_STEPS = ("red points slid in", "slid together", "loops off")
READ_AT, ANSWER_END = (6.0, 3.0), (10.4, 3.0)


def copy_of(things, layer):
    """D16, the copy as the Tortoise lays it: every thing of the upper layer with its row in the
    other order (y -> 6 - y), left and right kept, each bead turned the other way round
    (t -> -t); `layer` maps a place to its image."""
    return sorted((kind, c, (24 - t) % 24) + layer(x, y) for kind, c, t, x, y in things)


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


# the closed box's marks, step by step (the Lua's comment, pinned): ghosted grid places, and the
# arcs as (from place, to place); "his" and "hers" are the part-steps of slid together, in the
# text's order (E4, E5)
BOX_MARKS = {
    "reds": ([(0, 1), (0, 5), (1, 1), (1, 5), (0, 2), (0, 4), (2, 2), (2, 4)],          # points and rings
             [((0, 1), (1, 1)), ((0, 2), (2, 2)), ((0, 5), (1, 5)), ((0, 4), (2, 4))]),   # each point into its ring
    "his": ([(1, 0), (1, 6)], [((1, 0), (0, 0)), ((1, 6), (0, 6))]),                     # to the Cat's point, both layers
    "hers": ([(2, 1), (2, 5)], [((2, 1), (0, 0)), ((2, 5), (0, 6))]),
    "loops": ([(6, 3), (0, 0)], [((6, 3), (0, 6))])}                                     # the read ring in along the copy's cord


def box_marks(R, key):
    S0, S1 = PC.box_scene2(R.before), PC.box_scene2(R.after)
    g = PC.G(S1, PC.BOPTS, PC.BWHO)
    mark_bounds(R, g)
    assert R.marked
    places, arcs = BOX_MARKS[key]
    old = {(t["x"], t["y"]) for t in S0["things"]}
    new = {(t["x"], t["y"]) for t in S1["things"]}
    assert sorted(tuple(round(v, 6) for v in q[:2]) for q in R.ghosts) == sorted(tuple(round(v, 6) for v in g.XY(p)) for p in places)
    assert set(places) <= old and not (set(places) & new)                  # a ghost where a piece left
    got = []
    for a in R.arcs:
        (d0, c0) = nearest(a.frm, [g.XY(p) for p in old])
        (d1, c1) = nearest(a.to, [g.XY(p) for p in new | set(places)])     # into a piece after, or one this step ghosts
        assert a.kind == "slide" and d0 < 0.25 and d1 < 0.25, a.what
        at = {tuple(g.XY(p)): p for p in old | new}
        got.append((at[c0], at[c1]))
    assert sorted(got) == sorted(arcs), (key, got)
    if key in ("his", "hers"):                     # one move a layer, each the other's mirror: alike
        a, b = R.arcs
        assert np.isclose(a.len, b.len) and np.isclose(a.sag, b.sag)


def at_place(g, p):
    (v,) = [v for v in g.vertices() if (g.row(v), g.qubit(v)) == p]
    return v


def fused(dump, pairs):
    """graph_of a box dump with each (meeting place, ring) fused there by PyZX's own spider fusion
    (simp.check_fuse / simp.fuse); the ring that is left lies at the meeting place."""
    g = PC.graph_of(dump)
    for dst, src in pairs:
        a, b = at_place(g, src), at_place(g, dst)
        assert simp.check_fuse(g, a, b) and simp.fuse(g, a, b)
        (k,) = {a, b} & set(g.vertices())
        g.set_row(k, dst[0])
        g.set_qubit(k, dst[1])
    return g


def shape(g):
    """A box graph's structure: its spiders (place, colour, phase, legs) and its edges by the
    places they join (a ring at rest spliced into a second strand, off the grid, by its legs)."""
    from collections import Counter
    on = lambda v: float(g.row(v)).is_integer() and float(g.qubit(v)).is_integer()
    lab = lambda v: (g.row(v), g.qubit(v)) if on(v) else ("at rest", g.vertex_degree(v))
    return (Counter((lab(v), g.type(v), g.phase(v), g.vertex_degree(v)) for v in g.vertices()),
            Counter((frozenset((lab(a), lab(b))), g.edge_type((a, b))) for a, b in g.edges()))


def pictures():
    """The Fourth Challenge's prop pictures (identity card, sections 4 and 3e), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch19.py, and held against
    this file's closed_box (the master, The Closed Box) and green_point.  Round 4 draws the red
    points slid in, the looks' rings slid together move by move (his, then hers: E4, E5) and the
    lute still before its top string sounds."""
    calls = DIALOGUE.check_layout("Fourth", sizes=[1] * 5)
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("boxcloth", True, "", (), "Pages and looks"),
        ("boxcloth", False, "; step=red points slid in", (), "Red points slid in"),
        ("boxclothsteps", False, "after=red points slid in; from=off", ("his look's ring slid along; her look's ring slid along",),
         "His look along; Her look follows"),
        ("boxcloth", False, "red points slid in, slid together; step=loops off", (), "Loops lifted off"),
        ("crablutesteps", True, "", ("top string sounding",), "The lute, quiet; Top note alone")]
    assert PC.check_captions(DIALOGUE, "Fourth", calls) == 7
    blocks = PC.check_section(DIALOGUE, "Fourth", calls, [1, 1, 2, 1, 2])
    assert PC.check_anchors(DIALOGUE, "Fourth") == (8, 20)             # next-starts, phrases before (E4, E5 among them)
    for f_ in (DIALOGUE.dialog, DIALOGUE.raw):
        a42 = DIALOGUE.anchors("Fourth", f_)[2]
        assert PC.E_ITEMS["E4"] in a42 and PC.E_ITEMS["E5"] in a42
        assert a42.index(PC.E_ITEMS["E5"]) < a42.index("The same on the copy.")     # the upper layer's, then the copy
    for e in ("E4", "E5"):
        assert "a cord that re-routes, given by the text, " + e in PC.comment_prose("the slide together move by move: the PART-STEPS.")
    frames = [R for b in blocks for R, _, _ in b]
    assert tuple((cap, k, R.step, R.dump) for b in blocks for R, cap, k in b) == FOURTH_FRAMES      # the readings
    # round 3's four pictures are their blocks' last frames, unchanged
    pics = [frames[i] for i in (0, 3, 4, 6)]
    p41, p42, p43, p44 = pics
    assert tuple(p.dump for p in pics) == FOURTH_DUMPS
    assert tuple(R.picture.dump for R in (p41, p42, p43)) == FOURTH_DUMPS[:3]
    box_frames, lute_frames = frames[:5], frames[5:]
    # the chain, frame to frame: each step applied to the frame before; the part-steps compose to
    # round 3's slid together, and the frame between them is the box part-way
    for k, R in enumerate(box_frames[1:], 1):
        assert PC.box_dump2(R.before) == box_frames[k - 1].dump, k
    assert [R.step for R in frames] == [None, "red points slid in", "his look's ring slid along", "her look's ring slid along",
                                        "loops off", None, "top string sounding"]
    assert box_frames[2].part_way and not any(R.part_way for R in frames if R is not box_frames[2])
    assert repr(box_frames[3].after) == repr(PC.box_state2("red points slid in, slid together"))
    whole = PC.run_frame("box", "red points slid in; step=slid together", "", False)
    assert PC.TEX_ERRORS == [] and whole.dump == p42.dump
    # the marks of every step
    for R, key in zip(box_frames[1:], ("reds", "his", "hers", "loops")):
        box_marks(R, key)
    assert [(len(R.ghosts), len(R.arcs)) for R in box_frames] == [(0, 0), (8, 4), (2, 2), (2, 2), (2, 1)]
    # every one or none: the whole slide drawn at once gets NO arc -- her look's runs do not fit
    # beside his -- though his look's moves fit alone (the part-step draws them); its ghosts stay
    assert whole.arcs == [] and len(whole.ghosts) == 4 and any(n.startswith("no arcs for the looks' rings slid together")
                                                               for n in whole.notes)
    assert len(box_frames[2].arcs) == 2
    # E4, E5: the two page cords that re-route, as the text gives them -- his page's cord now leaves
    # the Cat's point down and to the right, then hers straight down the cloth's left-hand side
    c1 = PC.parse_dump(box_frames[2].dump)[3]
    assert [pts[:2] for a, b, pts in c1 if pts[0] == (0, 0) and len(pts) > 3] == [[(0, 0), (1, 1)]]
    c2 = PC.parse_dump(box_frames[3].dump)[3]
    assert sorted(pts[:2] for a, b, pts in c2 if pts[0] == (0, 0) and len(pts) > 3) == [[(0, 0), (0, 2)], [(0, 0), (1, 1)]]
    # L2: the part-steps are spider fusions, in the text's order: his look's ring fused into the
    # Cat's point (both layers) is the part-way picture; hers then is round 3's slid together.  Her
    # look's ring cannot go first: it is not yet next to the Cat's point
    reds, part1, together = (R.dump for R in box_frames[1:4])
    his = [((0, 0), (1, 0)), ((0, 6), (1, 6))]
    hers = [((0, 0), (2, 1)), ((0, 6), (2, 5))]
    assert shape(fused(reds, his)) == shape(PC.graph_of(part1))
    assert shape(fused(part1, hers)) == shape(PC.graph_of(together))
    g = PC.graph_of(reds)
    assert not simp.check_fuse(g, at_place(g, (2, 1)), at_place(g, (0, 0)))
    assert shape(fused(reds, his)) != shape(PC.graph_of(together))
    # the lute: no marks, ever (its strings are the wires; a counted frame stays unmarked), not
    # even when asked for; the top string alone only from still
    assert not any(R.marked or R.arcs or R.ghosts or R.mark_tikz for R in lute_frames)
    asked = PC.run_frame("lute", "still; step=top string sounding; marks=on", "", False)
    assert PC.TEX_ERRORS == [] and not (asked.marked or asked.mark_tikz) and asked.dump == p44.dump
    assert PC.faults(PC.run_frame, "lute", "pairs sounding; step=top string sounding", "", False) == ["lute-still-first"]
    assert lute_frames[0].dump == PC.crablute("").dump and repr(lute_frames[1].before) == repr(lute_frames[0].drawn)
    # no prediction frame on the box or the lute: expects= refused with the Lua's reasons, which
    # quote the dialogue
    assert PC.faults(PC.run_frame, "box", "; expects=a red point", "", False) == ["box-expects"]
    assert PC.faults(PC.run_frame, "lute", "; expects=the pairs", "", False) == ["lute-expects"]
    for msg in (PC.BEXPECTS, PC.LEXPECTS):
        for q in re.findall(r'"([^"]+)"', msg):
            assert all(q in DIALOGUE.read(f) for f in (DIALOGUE.dialog, DIALOGUE.raw)), q
    # the part-steps in turn: hers before his is refused, and so is a step before hers is taken
    assert PC.faults(PC.run_frame, "box", "red points slid in; step=her look's ring slid along", "", False) == ["box-part-turn"]
    assert PC.faults(PC.run_frame, "box", "red points slid in, his look's ring slid along; step=loops off", "", False) == [
        "box-part-next"]
    PC.TEX_ERRORS.clear()
    states = [""] + [", ".join(BOX_STEPS[:k]) for k in range(1, 4)]
    assert {PC.boxcloth(st).frame for st in states} == {p41.frame} == {R.frame for R in box_frames}    # one frame
    assert {PC.crablute(st).frame for st in ("", "pairs sounding", "top string sounding")} == {p44.frame} == {
        R.frame for R in lute_frames}

    # P4.1, P4.2, P4.3 and the unpictured red points slid in: graph_of each (a structural reading
    # of the cloth, not a second proof) is The Closed Box, and is the Cat's own green point at
    # rest, value (1, 1), exactly
    box, point = closed_box(), green_point()
    # L2: every frame of the box, the part-way one included, is the Cat's own green point at rest
    for R in box_frames:
        g = PC.graph_of(R.dump)
        assert PC.slips_turn(R.dump) == 0 and same_picture(g, point) and np.allclose(matrix_of(g).ravel(), [1, 1])
    assert same_picture(PC.graph_of(box_frames[2].dump), box)
    for st in states:
        g = PC.graph_of(PC.boxcloth(st).dump)
        assert PC.slips_turn(PC.boxcloth(st).dump) == 0
        assert same_picture(g, box) and same_picture(g, point), st
        assert np.allclose(matrix_of(g).ravel(), [1, 1]), st
    # each let-go cord one edge, knot and all, on the cloth; closed_box draws each let-go as a
    # cap with two legs: 17 against 19 (a structural fact, never said)
    g41 = PC.graph_of(p41.dump)
    assert g41.num_edges() == 17 and box.num_edges() == 19
    assert PC.parse_dump(p41.dump)[5] == PC.parse_dump(p42.dump)[5] == 2 and PC.parse_dump(p43.dump)[5] == 0
    # NEGATIVE: the Cat's point red (surely asleep) on both layers is not the Cat's own point
    asleep = p41.dump.replace("1:end green 0 @0,0", "1:end red 0 @0,0").replace("14:end green 0 @0,6", "14:end red 0 @0,6")
    assert same_picture(PC.graph_of(asleep), closed_box(cat=VT.X))
    assert not same_picture(PC.graph_of(asleep), point)

    # D16: the copy lies below, seen through but not as a card is held to the lamp: left and
    # right kept, the rows in the other order, every bead turned the other way round -- the
    # conjugate, not the mirror
    nodes, tubes, ties, cords, corner, knots = PC.parse_dump(p41.dump)
    upper = [v for v in nodes.values() if v[4] < 3]
    lower = sorted(v[:3] + (v[3], v[4]) for v in nodes.values() if v[4] > 3)
    assert len(upper) == len(lower) == 7
    assert copy_of(upper, lambda x, y: (x, 6 - y)) == lower                        # rows reversed
    assert copy_of(upper, lambda x, y: (10 - x, y + 4)) != lower                   # the lamp's mirror: no
    place = {k: (v[3], v[4]) for k, v in nodes.items()}
    flip = lambda k: (place[k][0], 6 - place[k][1])
    assert len(ties) == 4
    assert sorted(sorted([flip(a), flip(b)]) for kind, a, b, n in ties if place[a][1] < 3) == \
        sorted(sorted([place[a], place[b]]) for kind, a, b, n in ties if place[a][1] > 3)     # the ties
    paths = sorted(tuple(pts) for a, b, pts in cords)
    assert sorted(tuple(sorted([(x, 6 - y) for x, y in pts])) for pts in paths) == \
        sorted(tuple(sorted(pts)) for pts in paths)                                # the cords, too
    # her page's copy nearest the middle, Achilles's below it, the Cat's lowest; the read ring
    # in the clear row between the layers
    ends = {(c, y) for kind, c, t, x, y in nodes.values() if kind == "end"}
    assert ends == {("green", 0), ("red", 1), ("red", 2), ("red", 4), ("red", 5), ("green", 6)}
    assert [(kind, c, t, x, y) for kind, c, t, x, y in nodes.values() if y == 3] == [("ring", "green", 0) + READ_AT]
    # the turns are load-bearing: a ring of a page turned a generic step on the picture and the
    # other way round on its copy leaves the Cat's point; turned the same way on both, it does not
    for up, down in (("4:ring red 0 @1,1", "12:ring red 0 @1,5"), ("2:ring green 0 @1,0", "15:ring green 0 @1,6"),
                     ("7:ring red 0 @2,2", "10:ring red 0 @2,4")):
        for t in (5, 7):
            turned = lambda tu, td: p41.dump.replace(up, up.replace(" 0 @", " %d @" % tu)).replace(
                down, down.replace(" 0 @", " %d @" % td))
            assert same_picture(PC.graph_of(turned(t, 24 - t)), point), (up, t)
            assert not same_picture(PC.graph_of(turned(t, t)), point), (up, t)

    # the knots where each page's cord is tied to its copy's, where the cord turns down (D16):
    # each on the rounded turn at a page cord's corner, her page's turn inside Achilles's; gone
    # with the loops
    scene = PC.box_scene(p41.state)
    turns = {pts[1] for a, b, pts in cords if a != "loose" and b != "loose" and len(pts) == 4}
    for (name, row), (kx, ky) in zip(PC.KNOT_AT, p41.shown.knots):
        col = {"BEND_TOR": PC.BEND_TOR, "BEND_ACH": PC.BEND_ACH}[name]
        assert (col, row) in turns
        cx, cy = PC.C.cloth_xy(scene, PC.BOPTS, col, row, PC.BWHO)
        assert 0 < cx - kx < 0.1 and np.isclose(cx - kx, cy - ky)
    assert sorted(PC.KNOT_AT) == [("BEND_ACH", 1), ("BEND_TOR", 2)] and PC.BEND_TOR < PC.BEND_ACH
    assert len(p41.shown.knots) == len(p42.shown.knots) == 2 and p43.shown.knots == ()

    # the captions, true of their dumps: laid; slid together (one ring a layer and the read ring);
    # the loops lifted off (one ring, at rest, where the Cat's point lay on the copy; no loop)
    assert p41.dump.startswith("laid ") and p42.dump.startswith("reds,together ")
    n42 = PC.parse_dump(p42.dump)[0]
    assert sorted((c, t, x, y) for kind, c, t, x, y in n42.values()) == [
        ("green", 0, 0.0, 0.0), ("green", 0, 0.0, 6.0), ("green", 0) + READ_AT]
    n43, _, _, c43, _, _ = PC.parse_dump(p43.dump)
    assert list(n43.values()) == [("end", "green", 0, 0.0, 6.0)]
    assert all(a != b for a, b, pts in c43) and len(c43) == 1
    # fixed parts: the read ring in the clear row until the loops come off; the answer's cord's
    # loose end at the right-hand edge in every state
    for st in states:
        nodes, tubes, ties, cords, corner, knots = PC.parse_dump(PC.boxcloth(st).dump)
        assert [pts[0] for a, b, pts in cords if a == "loose"] == [ANSWER_END], st
        assert (("ring", "green", 0) + READ_AT in nodes.values()) == (st != ", ".join(BOX_STEPS)), st
    for k, R in enumerate(box_frames):                                       # ...and frame to frame
        nodes, tubes, ties, cords, corner, knots = PC.parse_dump(R.dump)
        assert [pts[0] for a, b, pts in cords if a == "loose"] == [ANSWER_END], k
        assert (("ring", "green", 0) + READ_AT in nodes.values()) == (k < 4), k

    # P4.4, the Crab's lute, its top string plucked alone (R7-Ch19): the lute draws its strings,
    # the top string alone and the pairs, 1 + 2 x 9 = 19 -- the master's line count, closed_box()'s
    # edges.  The picture does not carry the count: kinds only, asserted here and never said.
    v = p44.shown
    assert p44.dump == "top | top=sounding pairs=still courses=1+9"
    strings = v.strings
    assert len(strings) == 1 + 2 * 9 == box.num_edges() == 19
    assert [s_[0] for s_ in strings].count(0) == 1
    assert all([s_[0] for s_ in strings].count(k) == 2 for k in range(1, 10))
    assert [s_[0] for s_ in strings if s_[5]] == [0]                         # "Plucked alone"
    for st, sounding in (("", []), ("pairs sounding", list(range(1, 10)) * 2)):
        assert sorted(s_[0] for s_ in PC.crablute(st).shown.strings if s_[5]) == sorted(sounding)
        assert len(PC.crablute(st).shown.strings) == 19
    # (the card's ruling) a sounding string's swing -- its two edges, the one string at the ends
    # of its swing -- and the motion arcs are marks, not wires: the count stays 1 + 9 pairs.
    # Each swing edge runs from its own string's nut end to its bridge end and is that string;
    # no motion arc reaches the nut or the bridge.
    for st in ("", "top string sounding", "pairs sounding"):
        w = PC.crablute(st).shown
        ends = {(k, sd): (a, b) for k, sd, wd, a, b, so in w.strings}
        assert len(w.strings) == 19 and len(w.swings) == sum(1 for x in w.strings if x[5])
        for k, sd, wd, up, dn in w.swings:
            assert (up[0], up[3]) == (dn[0], dn[3]) == ends[(k, sd)]
        assert all(PC.LNUT < m[0][0] and m[3][0] < PC.LBRIDGE for m in w.marks)
        assert len(w.marks) == (0 if st == "" else 2)
    assert [(k, sd) for k, sd, *_ in v.swings] == [(0, 0)]               # the top string swings alone
    w = lute_frames[0].shown                                              # the still lute, before
    assert len(w.strings) == 19 and not any(x[5] for x in w.strings) and w.swings == w.marks == ()

    # the keyed words: read ring (P4.1) and top string and pairs (P4.4), set by the Lua, at their
    # compiled boxes and measured clear; the kit is not keyed on the box cloth; the rest unkeyed
    assert p41.labels == ("read ring",) and lute_frames[0].labels == ("top string", "pairs")     # the lute keyed still
    assert all(R.labels == () for R in box_frames[1:]) and p44.labels == ()
    for pic in (p41, lute_frames[0].picture, PC.crablute("top string sounding", True), PC.crablute("pairs sounding", True)):
        PC.check_compiled_boxes(pic)
        assert min(PC.check_keyed_words(pic).values()) >= PC.LETTER_FLOOR

    # the Lua's refusals: an unknown step; a step twice; a step before the one it follows; an
    # unknown state of the lute
    PC.TEX_ERRORS.clear()
    PC.box_state("pages folded")
    PC.box_state("red points slid in, red points slid in")
    PC.box_state("slid together")
    PC.lute_state("humming")
    assert [k for k, _ in PC.TEX_ERRORS] == ["step-unknown", "step-done", "step-after", "lute"]
    PC.TEX_ERRORS.clear()


# ----------------------------------------------------------------------------- the sketchbook

def closed_box(cat=VT.Z, other_basis=False):
    """The Closed Box, doubled and unfolded, built by hand (no ground vertex: pyzx 0.10.6
    refuses them).  Every turn is 0, so the mirror copy is drawn the same as the picture.
      On each copy (the picture above, the mirror copy below, in reflected order):
        C: the Cat's point -- g1 (Achilles's look) -- R (the read-spider)
        A: Achilles's page, a red point -- r1 (his look) -- g2 (the Tortoise's look) -- cap
        T: the Tortoise's page, a red point -- r2 (her look) -- cap
        ties: g1 -- r1, g2 -- r2
      The nine undoubled edges: C point-g1, g1-R; A point-r1, r1-g2, g2-cap; T point-r2, r2-cap;
      the ties g1-r1, g2-r2.  Doubled, each edge and its mirror copy (a cap's two legs among
      them), and the one classical wire, R -- out.
    With other_basis, Achilles's look asks the Cat another question: a red spider on the Cat's
    wire, tied by a yellow box to a red spider on his page (it copies |+>/|->)."""
    g = zx.Graph()
    R = g.add_vertex(VT.Z, 0, 14)
    out = g.add_vertex(VT.BOUNDARY, 0, 17)
    g.add_edge((R, out))                                   # the classical answer
    cap_a, cap_t = g.add_vertex(VT.Z, 0, 9), g.add_vertex(VT.Z, 0, 7)   # letting go: caps
    for side in (1, -1):                                   # the picture, then its mirror copy
        c_pt = g.add_vertex(cat, 5 * side, 0)
        g1 = g.add_vertex(VT.X if other_basis else VT.Z, 5 * side, 3)
        a_pt, r1 = g.add_vertex(VT.X, 3 * side, 0), g.add_vertex(VT.X, 3 * side, 3)
        g2 = g.add_vertex(VT.Z, 3 * side, 6)
        t_pt, r2 = g.add_vertex(VT.X, 1 * side, 0), g.add_vertex(VT.X, 1 * side, 6)
        for e in ((c_pt, g1), (g1, R), (a_pt, r1), (r1, g2), (g2, cap_a), (t_pt, r2), (r2, cap_t),
                  (g2, r2)):
            g.add_edge(e)
        g.add_edge((g1, r1), ET.HADAMARD if other_basis else ET.SIMPLE)
    g.set_inputs(())
    g.set_outputs((out,))
    return g


def green_point():
    g = zx.Graph()
    v, o = g.add_vertex(VT.Z, 0, 0), g.add_vertex(VT.BOUNDARY, 0, 2)
    g.add_edge((v, o))
    g.set_outputs((o,))
    assert g.num_edges() == 1
    return g


def pure_part():
    """The picture above the fold, undoubled: three points, two looks, three wires open."""
    g = zx.Graph()
    outs = [g.add_vertex(VT.BOUNDARY, r, 9) for r in range(3)]
    c_pt, g1 = g.add_vertex(VT.Z, 0, 0), g.add_vertex(VT.Z, 0, 3)
    a_pt, r1, g2 = g.add_vertex(VT.X, 1, 0), g.add_vertex(VT.X, 1, 3), g.add_vertex(VT.Z, 1, 6)
    t_pt, r2 = g.add_vertex(VT.X, 2, 0), g.add_vertex(VT.X, 2, 6)
    for e in ((c_pt, g1), (g1, outs[0]), (a_pt, r1), (r1, g2), (g2, outs[1]), (t_pt, r2),
              (r2, outs[2]), (g1, r1), (g2, r2)):
        g.add_edge(e)
    g.set_outputs(tuple(outs))
    assert g.num_edges() == 9
    return g


def odds(g):
    v = matrix_of(g).ravel()
    assert np.allclose(v.imag, 0)                        # a classical vector: real
    return v.real / v.real.sum()


def sketchbook():
    # --- The Closed Box, built as drawn
    box = closed_box()
    assert box.num_edges() == 19          # the master's lines, doubled and classical (Section 19.8)
    assert np.allclose(matrix_of(pure_part()).ravel(), ket(0, 0, 0) + ket(1, 1, 1))
    value = matrix_of(box).ravel()
    assert np.allclose(value, [1, 1])                                   # the Cat's own point
    assert same_picture(box, green_point())
    # DERIVATION: PyZX's full_reduce leaves one green spider at rest on the answer wire, number 1
    h = box.copy()
    simp.full_reduce(h)
    inner = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
    assert len(inner) == 1 and h.type(inner[0]) == VT.Z and h.phase(inner[0]) == 0
    assert h.num_edges() == 1 and h.edge_type(next(iter(h.edges()))) == ET.SIMPLE
    assert np.isclose(h.scalar.to_number(), 1)
    assert same_picture(h, green_point())
    # the same reduction by named rules only: fusion, the identity card, and loops coming off
    h = box.copy()
    while simp.spider_simp(h) or simp.id_simp(h):
        pass
    inner = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
    assert len(inner) == 1 and h.phase(inner[0]) == 0 and same_picture(h, green_point())

    # --- whoever watched, in pictures: the Cat surely asleep (a red point at rest in its place)
    asleep = closed_box(cat=VT.X)
    assert asleep.num_edges() == 19
    assert np.allclose(odds(asleep), [1, 0])                            # still surely asleep
    # NEGATIVE: a look that asks the Cat another question changes its odds
    other = closed_box(cat=VT.X, other_basis=True)
    assert other.num_edges() == 19
    assert np.allclose(odds(other), [0.5, 0.5]) and not np.allclose(odds(other), [1, 0])


if __name__ == "__main__":
    verdict("OneAnswerOut", blackboard, sketchbook)
