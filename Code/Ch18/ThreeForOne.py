"""Challenge I: Three for One  (noise; the vote; the encoder)
============================================================
"About one hollow in ten read false."  -- Crab
"...Oh. You can't copy a turn; you can only spread it."  -- Achilles

Claim: a vote of three fails with probability F(p) = 3p^2 - 2p^3 whichever way the bit was
set (Result 18.2), and the quantum bit-flip code turns independent flips into a logical flip
of exactly that probability (Result 18.11).  The encoder Enc = |000><0| + |111><1| is one
green spider, 1 -> 3, with no number (Result 18.3); its mirror undoes it; a dark or lit point
copies through it with the number 1/2, while a turned point fuses into it and comes out
entangled, never as three copies (Result 18.4).  The bit-flip channel, doubled, is (1-p) bare
+ p red halves on the wire and its mirror (Result 18.1).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction
from itertools import product

import numpy as np
import pyzx.simplify as simp
from pyzx.utils import EdgeType as ET, VertexType as VT

from dfp import I2, Picture, ket, kron, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch18 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
Y = 1j * X @ Z                                             # Gottesman's Table 1: Y = iXZ

ENC = np.outer(ket(0, 0, 0), ket(0)) + np.outer(ket(1, 1, 1), ket(1))   # 8 x 2, top wire first

# the door dial's six marks, read as states
MARKS = {
    "dark": ket(0), "lit": ket(1),
    "at rest": (ket(0) + ket(1)) / np.sqrt(2), "a half": (ket(0) - ket(1)) / np.sqrt(2),
    "a quarter round": (ket(0) + 1j * ket(1)) / np.sqrt(2),
    "a quarter back": (ket(0) - 1j * ket(1)) / np.sqrt(2),
}


def F(p):
    return 3 * p**2 - 2 * p**3


def on(k, A):
    """A on wire k (0 = top) of three."""
    ops = [I2, I2, I2]
    ops[k] = A
    return kron(*ops)


def X_pattern(e):
    return kron(*[X if b else I2 for b in e])


# the checks and the mending table of Section 18.2 (defined here too: files stand alone)
Z1Z2, Z2Z3 = on(0, Z) @ on(1, Z), on(1, Z) @ on(2, Z)
I8 = np.eye(8)


def P(s):
    s1, s2 = s
    return (I8 + (-1) ** s1 * Z1Z2) @ (I8 + (-1) ** s2 * Z2Z3) / 4


CORR = {(0, 0): (0, 0, 0), (1, 0): (1, 0, 0), (1, 1): (0, 1, 0), (0, 1): (0, 0, 1)}


def superop(kraus):
    """Row-major doubling: vec(rho) = sum rho_ij |i>|j>, and A acts as A (x) conj(A)."""
    return sum(np.kron(K, K.conj()) for K in kraus)


def random_density(rng, n=2):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def blackboard():
    rng = np.random.default_rng(1801)

    # --- the two flips on the six marks (Section 18.1)
    assert np.allclose(X @ MARKS["at rest"], MARKS["at rest"])                       # X|+> = |+>
    assert np.allclose(X @ MARKS["a quarter round"], 1j * MARKS["a quarter back"])   # up to i
    fid = lambda a, b: abs(np.vdot(a, b)) ** 2
    harmless_X = {m for m, v in MARKS.items() if np.isclose(fid(v, X @ v), 1)}
    harmless_Z = {m for m, v in MARKS.items() if np.isclose(fid(v, Z @ v), 1)}
    assert harmless_X == {"at rest", "a half"}          # N^X never harms at rest (nor a half)
    assert harmless_Z == {"dark", "lit"}                # N^Z never harms dark or lit
    assert harmless_X != set(MARKS) and harmless_Z != set(MARKS)   # neither harmless for all six

    # --- the three channels in Kraus form are trace preserving, for every p
    for p in (0.1, 0.37, rng.uniform()):
        chans = {
            "X": [np.sqrt(1 - p) * I2, np.sqrt(p) * X],
            "Z": [np.sqrt(1 - p) * I2, np.sqrt(p) * Z],
            "dep": [np.sqrt(1 - p) * I2] + [np.sqrt(p / 3) * Q for Q in (X, Y, Z)],
        }
        for ks in chans.values():
            assert np.allclose(sum(K.conj().T @ K for K in ks), I2)

    # --- Result 18.2: the vote of three, enumerated over the eight flip patterns,
    #     for a bit set dark AND a bit set lit (the tins' damp goes both ways)
    for p in (0.1, 0.37, rng.uniform()):
        for setting in (0, 1):
            wrong = 0.0
            for flips in product((0, 1), repeat=3):
                prob = np.prod([p if f else 1 - p for f in flips])
                read = [setting ^ f for f in flips]
                majority = int(sum(read) >= 2)
                wrong += prob * (majority != setting)
            assert np.isclose(wrong, F(p))
    assert np.isclose(F(0.1), 0.028)                                  # three in a hundred
    assert np.isclose(F(F(0.1)), 0.002308096, atol=1e-12)             # the ledger's vote of votes
    assert not np.isclose(F(0.1), 0.1)                                # the vote does buy something

    # --- the encoder: Enc(a|0> + b|1>) = a|000> + b|111>, NOT the clone |psi>^(x)3
    psi = random_state(rng)                                           # a generic, non-real message
    assert not np.allclose(psi, psi.real)
    assert np.allclose(ENC @ psi, psi[0] * ket(0, 0, 0) + psi[1] * ket(1, 1, 1))
    assert not np.allclose(ENC @ psi, kron(psi, psi, psi).ravel())
    for a in (0, 1):                                                  # dark and lit: a true copy
        assert np.allclose(ENC @ ket(a), kron(ket(a), ket(a), ket(a)).ravel())
    q = MARKS["a quarter round"]                                      # the dialogue's turned message
    assert np.allclose(ENC @ q, (ket(0, 0, 0) + 1j * ket(1, 1, 1)) / np.sqrt(2))
    assert not np.allclose(ENC @ q, kron(q, q, q).ravel())
    assert np.allclose(ENC.conj().T @ ENC, I2)                        # the encoder loses nothing
    # Enc is real, so its mirror (conjugate transpose) is just its transpose
    assert np.allclose(ENC, ENC.real)

    # --- Result 18.4 on the blackboard: copy (with 1/2) and spread (entangled)
    for a in (0, 1):
        red_point = np.sqrt(2) * ket(a)                               # bare red point, a*pi
        three = kron(red_point, red_point, red_point).ravel()         # (sqrt2)^3 |aaa>
        assert np.allclose(ENC @ red_point, three / 2)                # the number 1/2
        assert not np.allclose(ENC @ red_point, three)
    for phi in (0.5 * np.pi, rng.uniform(0, 2 * np.pi)):              # a quarter round; a random turn
        g = ket(0) + np.exp(1j * phi) * ket(1)                        # a green point, bare
        out = ENC @ g
        assert np.allclose(out, ket(0, 0, 0) + np.exp(1j * phi) * ket(1, 1, 1))
        assert np.linalg.matrix_rank(out.reshape(2, 4)) == 2          # entangled across the cut
        pts = kron(g, g, g).ravel()
        assert np.linalg.matrix_rank(pts.reshape(2, 4)) == 1          # three points: rank 1

    # --- Result 18.11: encode, independent flips, recover, good-branch decode
    #     = the bit-flip channel with p_L = F(p), as 4 x 4 superoperators
    for p in (0.1, 0.37, rng.uniform()):
        full = np.zeros((4, 4), dtype=complex)
        for e in product((0, 1), repeat=3):
            w = sum(e)
            Ke = np.sqrt(p**w * (1 - p) ** (3 - w)) * X_pattern(e)
            for s, c in CORR.items():
                K = ENC.conj().T @ X_pattern(c) @ P(s) @ Ke @ ENC
                full += np.kron(K, K.conj())
        pL = F(p)
        assert np.allclose(full, superop([np.sqrt(1 - pL) * I2, np.sqrt(pL) * X]))
        assert not np.allclose(full, superop([np.sqrt(1 - p) * I2, np.sqrt(p) * X]))  # not p itself

    pictures()


# ------------------------------------------------------------------------------ prop pictures

FIRST_DUMPS = (
    "first=dark second=dark third=lit",
    "dark,pushed aside=1 | cols=7 rows=3 y=0..2 near=false folded=false; 1:end red 0 @4,0; 2:end red 0 @4,1; "
    "3:end red 0 @4,2; cord loose-1 [6.3,0 4,0]; cord loose-2 [6.3,1 4,1]; cord loose-3 [6.3,2 4,2]",
    "turned,pushed aside=0 | cols=7 rows=3 y=0..2 near=false folded=false; 1:ring green 6 @3,1; "
    "cord loose-1 [6.3,0 4,0 3,1]; cord loose-1 [6.3,1 3,1]; cord loose-1 [6.3,2 4,2 3,1]",
    "mend=0,0,0 tell=0,0,0 clip=none glass=reset",
)


FRAME_DUMPS = (                                        # the log's dump, frame by frame (card 3e.6, item 6)
    'first=down second=down third=down',
    'first=dark second=down third=down',
    'first=dark second=dark third=down',
    'first=dark second=dark third=lit',
    'dark aside=0 | cols=7 rows=3 y=0..2 near=false folded=false; 1:end red 0 @0,1; 2:ring green 0 @3,1; cord '
    'loose-1 [1,1 0,1]; cord loose-2 [6.3,0 4,0 3,1]; cord loose-2 [2,1 3,1]; cord loose-2 [6.3,1 3,1]; cord '
    'loose-2 [6.3,2 4,2 3,1]',
    'dark,pushed aside=1 | cols=7 rows=3 y=0..2 near=false folded=false; 1:end red 0 @4,0; 2:end red 0 @4,1; '
    '3:end red 0 @4,2; cord loose-1 [6.3,0 4,0]; cord loose-2 [6.3,1 4,1]; cord loose-3 [6.3,2 4,2]',
    'turned aside=0 | cols=7 rows=3 y=0..2 near=false folded=false; 1:end green 6 @0,1; 2:ring green 0 @3,1; cord '
    'loose-1 [1,1 0,1]; cord loose-2 [6.3,0 4,0 3,1]; cord loose-2 [2,1 3,1]; cord loose-2 [6.3,1 3,1]; cord '
    'loose-2 [6.3,2 4,2 3,1]',
    'turned aside=0 | cols=7 rows=3 y=0..2 near=false folded=false; 1:end green 6 @0,1; 2:ring green 0 @3,1; cord '
    'loose-1 [1,1 0,1]; cord loose-2 [6.3,0 4,0 3,1]; cord loose-2 [2,1 3,1]; cord loose-2 [6.3,1 3,1]; cord '
    'loose-2 [6.3,2 4,2 3,1]',
    'turned,pushed aside=0 | cols=7 rows=3 y=0..2 near=false folded=false; 1:ring green 6 @3,1; cord loose-1 '
    '[6.3,0 4,0 3,1]; cord loose-1 [6.3,1 3,1]; cord loose-1 [6.3,2 4,2 3,1]',
    'mend=0,0,0 tell=0,0,0 clip=none glass=reset',
)


DP, TP = "the dark point", "the turned point"


def knot(g):
    """The point's cord knotted to the lead (the step the Tortoise takes before she pushes): the
    picture's one input, the lead's loose end, joined to its output at the point's loose end (on
    the middle row, the leftmost), both loose ends gone."""
    h = g.copy()
    (i,) = h.inputs()
    o = min((v for v in h.outputs() if h.qubit(v) == 1), key=h.row)
    (a,), (b,) = h.neighbors(i), h.neighbors(o)
    assert h.edge_type(h.edge(i, a)) == h.edge_type(h.edge(o, b)) == ET.SIMPLE
    outs = tuple(v for v in h.outputs() if v != o)
    h.remove_vertices([i, o])
    h.add_edge((a, b))
    h.set_inputs(())
    h.set_outputs(outs)
    assert h.num_edges() == g.num_edges() - 1
    return h


# an arc ends beside its ring: a hop HOPR from its centre, a route ROUTER, the third strand's 0.16 (its end
# set back past the parting ring's ties); every one within NEAR
NEAR = 0.17
GRID = re.compile(r"no ghost of the (\w+) (\w+) \(grid ([-\d.]+),([-\d.]+)\): (.+)")


def check_cloth_marks(R, scene_of, opts, who, extra=()):
    """A cloth step's marks, on the drawn extent (card 3e.6): every piece of the state before that
    left its place has a ghost there, the size of a ring, or the log says why it has none; every
    ghost stands on such a piece (or on one of `extra`, a place that is no piece of the cloth's
    table); every arc runs from beside a piece of the state before to beside a piece of the step's
    state (or to one of this step's ghosts: a ring slid into one that goes with it), another place, bowed at least the Lua's floor and clear of the net by its clearance;
    pieces gone and no arc drawn: the log says why."""
    S0, S1 = scene_of(R.before), scene_of(R.after)
    g = PC.G(S1, opts, who)
    at = lambda S: {(t["x"], t["y"]): (t["kind"], t["c"]) for t in S["things"]}
    old, new = at(S0), at(S1)
    gone = {p for p in old if p not in new}
    hidden = {p for p in old if p in new and new[p] != old[p]}
    said = {(float(x), float(y)): why for c, kw, x, y, why in (m.groups() for m in map(GRID.fullmatch, R.notes) if m)}
    for p in hidden:
        assert said.get(p) == "the piece lying on its place now hides it", p
    ghosted = set()
    for q in R.ghosts:
        assert np.isclose(q[2], g.RO), q.what
        here = [p for p in gone if np.allclose(q[:2], g.XY(p), atol=1e-3)]
        assert here or any(np.allclose(q[:2], e, atol=1e-3) for e in extra), q.what
        ghosted |= set(here)
    assert gone - ghosted <= set(said), (gone, ghosted, said)          # a piece gone: ghosted, or why not
    assert not (ghosted & set(said)), ghosted & set(said)
    near = lambda pt, places: min(((np.hypot(pt[0] - c[0], pt[1] - c[1]), c) for c in places), key=lambda u: u[0])
    for a in R.arcs:
        assert a.kind == "slide" and a.sag >= PC.MINSAG - 1e-9 and a.fit[0] >= PC.MINCLEAR - 1e-9, (a.what, a.sag, a.fit)
        d0, c0 = near(a.frm, [g.XY(p) for p in old] + list(extra))
        d1, c1 = near(a.to, [g.XY(p) for p in new] + [tuple(q[:2]) for q in R.ghosts])   # or this step's ghost: both gone
        assert d0 < NEAR and d1 < NEAR and c0 != c1, (a.what, d0, d1)
    if gone and not R.arcs:
        assert any(n.startswith("no arc") for n in R.notes), R.step
    assert len(R.mark_tikz) >= len(R.arcs) + len(R.ghosts)


def pictures():
    """The First Challenge's prop pictures (identity card, sections 4 and 3e), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch18.py, and held against
    this file's blackboard and pictures (graph_of: a structural reading of each cloth dump, not a
    proof).  Round 4 draws the process: the tin opened lid by lid, each point laid before it is
    pushed, and Achilles's prediction (three copies of the turned point) drawn pale -- three turned
    points, a product state, which is not what the parting ring makes of a turn."""
    calls = DIALOGUE.check_layout("First", sizes=[1, 1, 2, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("plaintinsteps", True, "", ("first dark; second dark; third lit",),
         "The tin as set; Left lid off; Centre lid up; A tin, spent"),
        ("partclothsteps", True, "", (DP, "pushed in"), "The dark point; Out at the ends"),
        ("partcloth", False, "expects=three copies", (TP,), "as Achilles expects"),
        ("partclothsteps", False, "", (TP, "pushed in"), "The turned point; Taken in"),
        ("stormline", True, "", (), "Strung for the storm")]
    PC.check_captions(DIALOGUE, "First", calls)
    PC.check_section(DIALOGUE, "First", calls)
    assert PC.check_anchors(DIALOGUE, "First", PC.BLOCKS["First"]) == (3, 5)   # next-starts, phrases before
    frames = [PC.frame_chain(c) for c in calls]
    assert [len(f) for f in frames] == [4, 2, 1, 2, 1]                     # card 3e.6: 4 blocks, 10 frames
    assert tuple(R.head for f in frames for R in f) == FRAME_DUMPS          # the log's dump, frame by frame
    assert all(R.head == R.dump for f in frames for R in f if R.wrong is None)   # the Lua's dump is round 3's
    tin, dark, (pale,), turned, (p14,) = frames
    p11, p12, p13 = tin[-1], dark[-1], turned[-1]
    pics = [p11, p12, p13, p14]
    # every strip's last frame is round 3's picture of its block (round 3's assertions carry over)
    assert tuple(p.dump for p in pics) == FIRST_DUMPS
    assert p11.dump == PC.plaintin("first dark, second dark, third lit").dump
    assert p12.dump == PC.partcloth("pushed in", DP).dump and p13.dump == PC.partcloth("pushed in", TP).dump

    # one frame per macro, in every state (the counter band drawn whether or not it holds one)
    assert {PC.plaintin(s).frame for s in ("", "first lit", "first dark, second lit, third dark")} == {p11.frame}
    assert {PC.partcloth(s, st).frame for s in ("", "pushed in")
            for st in (DP, "the lit point", TP)} == {p12.frame} == {p13.frame} == {pale.frame}
    assert {PC.stormline(s).frame for s in ("", "clip-bell on, glass cloudy", "middle mending ring at a half")} == {p14.frame}
    assert {R.frame for f in frames[:4] for R in f} == {p11.frame} | {p12.frame}
    assert len({R.frame for R in [pale] + turned}) == 1                    # P1.3's block: one height

    # P1.1, lid by lid (card 3e.6, item 7): the tin as set, every lid down; then the readings in
    # order -- dark, dark, then lit -- each step lifting one lid, the lids it does not lift kept
    # as they were; two dark readings already decide the vote, and the third (lit, false) leaves it
    want = ("first dark", "second dark", "third lit")
    assert tin[0].step is None and not tin[0].marked and tin[0].keyed is True
    assert not any(L["up"] for L in tin[0].state.lids)
    for k, R in enumerate(tin[1:]):
        assert R.step == want[k] and R.marked and not R.refused
        for j, (L0, L1) in enumerate(zip(tin[k].state.lids, R.state.lids)):
            assert L1 == L0 if j != k else (not L0["up"] and L1["up"] and L1["r"] == want[k].split()[1]), (k, j)
    shown = lambda R: [{"dark": 0, "lit": 1}[L["r"]] for L in R.state.lids if L["up"]]
    assert [shown(R) for R in tin] == [[], [0], [0, 0], [0, 0, 1]]
    for third in (0, 1):                                          # after "Centre lid up" the vote is cast
        assert int(sum(shown(tin[2]) + [third]) >= 2) == 0
    # the marks: one lift arc per lid lifted, from beside its hollow to beside it laid in front,
    # bowed at least the floor and clear of the other lids; no ghost (the open hollow shows it)
    for k, R in enumerate(tin[1:]):
        (arc,) = R.arcs
        assert arc.kind == "swing" and R.ghosts == [] and arc.what == "lift arc for the %s lid" % PC.LIDS[k]
        assert np.isclose(np.hypot(arc.frm[0] - PC.LIDX[k], arc.frm[1]), PC.LIFTOUT, atol=1e-3)
        assert np.isclose(np.hypot(arc.to[0] - PC.LIDX[k], arc.to[1] - PC.LY), PC.LIFTOUT, atol=1e-3)
        assert arc.frm[1] > arc.to[1] and arc.sag >= PC.MINSAG - 1e-9 and arc.fit[0] > 0
        assert any(n.startswith("no ghost of the %s lid" % PC.LIDS[k]) for n in R.notes)
        assert len(R.mark_tikz) == 1

    # P1.2 and P1.3, the point laid and then pushed (item 7): the before-frame read as a picture,
    # the point's cord knotted to the lead, has the value of the frame after it times what the
    # counter set aside stands for -- 1/2 for the dark point, nothing for the turned one (green
    # meets green, the turns adding)
    for R, st, aside in ((dark[0], DP, 0), (turned[0], TP, 0)):
        assert R.step is None and not R.marked and R.state == PC.part_state("", st) and R.state.aside == aside
    assert dark[0].keyed is True and turned[0].keyed is False
    laid = PC.graph_of(dark[0].dump)
    assert laid.num_inputs() == 1 and laid.num_outputs() == 4                  # the lead in; the point's end
    k_dark = knot(laid)
    pushed = PC.graph_of(p12.dump)
    half = pushed.copy()
    half.scalar.add_power(-2)                                     # the counter: 1/2 = sqrt2^-2
    assert same_picture(k_dark, half) and not same_picture(k_dark, pushed)
    assert same_picture(k_dark, point_into_encoder(VT.X, 0))
    k_turned = knot(PC.graph_of(turned[0].dump))
    assert same_picture(k_turned, PC.graph_of(p13.dump))          # exactly: nothing set aside
    assert same_picture(k_turned, green_three_legs(Fraction(1, 2)))
    assert not same_picture(k_turned, point_into_encoder(VT.Z, 0))   # the turn is not lost
    # the marks of the push: the point's hop along the lead into the ring; for the dark point
    # its three arcs out on every cord (every arc or none), the ring's and the point's ghosts
    for R, n_arcs in ((dark[1], 4), (turned[1], 1)):
        check_cloth_marks(R, PC.part_scene, PC.PARTOPTS, PC.PWHO)
        assert len(R.arcs) == n_arcs and R.arcs[0].what == "slide arc for the point, pushed in along the lead into the ring"
    assert [a.what for a in dark[1].arcs[1:]] == ["arc for the point out on the %s cord" % w for w in ("first", "middle", "third")]
    assert sorted(q.what for q in dark[1].ghosts) == ["the green ring (grid 3,1)", "the red point (grid 0,1)"]
    assert [q.what for q in turned[1].ghosts] == ["the green point (grid 0,1)"]  # the ring stays, turned
    assert "no mark for the counter set aside: it comes from her work-bag" in dark[1].notes

    # the pale frame: Achilles's rule, "three copies", applied by this file to the point as laid --
    # whatever point goes in comes out on every cord of it, the same point on each
    Q = pale.before
    assert Q == PC.part_state("", TP) and Q.kind == "turned" and pale.step is None and not pale.marked
    rule = [("end", Q.c, 6 * Q.q, 4.0, float(r)) for r in range(3)]      # its colour and turn, at each near end
    nodes = PC.parse_dump(pale.dump)[0]
    assert sorted(nodes.values()) == rule and pale.dump.startswith("aside=0 | ")   # no counter set aside
    assert pale.head == turned[0].head                            # drawn from the frame of the point as laid
    guess = PC.graph_of(pale.dump)
    assert same_picture(guess, three_points(VT.Z, Fraction(1, 2)))  # three turned points ...
    true = PC.graph_of(p13.dump)
    for power in (-2, 0, 2):                                     # ... which "Taken in" is not, at any number
        g_ = guess.copy()
        g_.scalar.add_power(power)
        assert not same_picture(g_, true)
    assert np.linalg.matrix_rank(matrix_of(guess).reshape(2, 4)) == 1          # a product state
    assert np.linalg.matrix_rank(matrix_of(true).reshape(2, 4)) == 2           # spread, not copied
    # where the rule is right -- a dark or a lit point -- its picture is the true one up to the
    # counter, which is why the Lua refuses it there
    for st, frame_after in ((DP, p12), ("the lit point", PC.run_frame("part", "step=pushed in", "the lit point", False))):
        S, _ = PC.part_wrong(PC.part_state("", st))
        g_ = PC.graph_of("aside=0 | " + PC.kit_dump(S, PC.PWHO))
        assert same_picture(g_, PC.graph_of(frame_after.dump))
        assert PC.faults(PC.run_frame, "part", "expects=three copies", st, False) == ["part-expects-true"]
    PC.TEX_ERRORS.clear()
    # drawn: the cloth under a white veil, cut away round the three points, each at full strength
    # with a second edge; never keyed, never marked
    assert PC.PALE == 0.5 and pale.own[0] == "\\begin{scope}[even odd rule]" and pale.own[3] == "\\end{scope}"
    assert pale.own[2].startswith("\\fill[white, opacity=0.500] ")
    gp = PC.G(PC.part_wrong(Q)[0], PC.PARTOPTS, PC.PWHO)
    edges = pale.own[4:7]
    for r, e in enumerate(edges):
        m = re.fullmatch(r"\\draw\[propclothgreenedge, line width=([\d.]+)pt\] \((-?[\d.]+),(-?[\d.]+)\) circle \(([\d.]+)\);", e)
        assert m and float(m.group(1)) == PC.PALEEDGE
        assert np.allclose([float(m.group(2)), float(m.group(3))], gp.XY((4, r)), atol=1e-3)
        assert np.isclose(float(m.group(4)), gp.RO + PC.PALEOUT, atol=1e-3)
        assert 2 * float(m.group(4)) + PC.PALEEDGE * 2.54 / 72.27 >= 0.15     # legible across
    assert pale.arcs == pale.ghosts == pale.mark_tikz == [] and pale.keyed is False
    # the wrong rule's comment in the Lua, word for word as the card copies it (section 3e.1), and
    # every phrase it quotes in both dialogue files (planning comments left out)
    heading = PC.RULE_COMMENTS[0]
    assert heading == 'round 4: "three copies" AND ITS WRONG RULE.'
    card = (pathlib.Path(DIALOGUE.dialog).parent / "identity.md").read_text(encoding="utf-8")
    assert ("```\n" + PC.rule_comment(heading).rstrip("\n") + "\n```") in card
    quotes = PC.comment_quotes(heading)
    assert quotes[0] == "three copies" == PC.PEXPECTS and len(quotes) == 7
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        text = DIALOGUE.read(path)
        for q in quotes[1:]:
            assert q in text, (path, q)
    assert "whatever point is pushed into the green ring at rest comes out on every cord of it" in PC.comment_prose(heading)

    # P1.1: three hollows over one setting nobody in the hall knew; the ledger, in the paragraph
    # after the block, shows it was set dark.  For this tin one reading is false (the last), and
    # the vote of the three is the setting: a single false hollow is a pattern the vote survives
    # (the enumeration of Result 18.2 counts it a success), two would not be.
    after = DIALOGUE.following("First")[0]
    assert "The damp got the last one." in after and "Set dark. The vote has it." in after
    readings = [{"dark": 0, "lit": 1}[L["r"]] for L in p11.state.lids]
    setting = 0                                                   # "Set dark."
    flips = tuple(r ^ setting for r in readings)
    assert flips == (0, 0, 1) and int(sum(readings) >= 2) == setting
    assert sum(r ^ 1 for r in readings) == 2                      # had it been set lit: two false, the vote wrong
    assert all(L["up"] for L in p11.state.lids)                   # "A tin, spent": no lid down
    assert "down" in PC.plaintin("first dark, second dark").dump  # (a tin not spent would show one)

    # P1.2: the dark point pushed in is EXACTLY TWICE the dark point knotted into the parting
    # ring; so the counter set aside stands for 1/2, the number the blackboard asserts
    # (ENC @ red_point == three / 2)
    g12 = PC.graph_of(p12.dump)
    assert g12.num_inputs() == 0 and g12.num_outputs() == 3 and g12.num_edges() == 3
    knotted = point_into_encoder(VT.X, 0)
    twice = point_into_encoder(VT.X, 0)
    twice.scalar.add_power(2)                                     # sqrt2^2 = 2
    assert same_picture(g12, twice) and not same_picture(g12, knotted)
    pushed, kn = matrix_of(g12).ravel(), matrix_of(knotted).ravel()
    counter = np.vdot(pushed, kn) / np.vdot(pushed, pushed)       # what the counter stands for
    assert np.isclose(counter, 0.5) and np.allclose(counter * pushed, kn)
    red_point = np.sqrt(2) * ket(0)
    assert np.allclose(kn, ENC @ red_point) and np.allclose(pushed, kron(red_point, red_point, red_point).ravel())
    assert p12.state.aside == 1 and len(p12.shown.counters) == 1
    # the lit point (unpictured) the same, at a half: twice the knotted lit point
    lit = PC.graph_of(PC.partcloth("pushed in", "the lit point").dump)
    twice_lit = point_into_encoder(VT.X, 1)
    twice_lit.scalar.add_power(2)
    assert same_picture(lit, twice_lit)
    # "Out at the ends": pushed, three red points at rest at the near ends of the cords
    nodes = PC.parse_dump(p12.dump)[0]
    assert p12.state.pushed and sorted(nodes.values()) == [("end", "red", 0, 4.0, float(r)) for r in range(3)]
    # the counter lies off the linen, beyond its near edge and clear of its outline and its
    # shadow, by the right-hand end of the cloth, inside the band and the frame: measured on the
    # drawn extents (the counter's disc, edge and shadow; the linen's stroke and its shadow), as
    # word_gaps measures a word, with a positive gap
    left, bottom, right, top = p12.shown.linen
    box = PC.counter_box(*p12.shown.counters[0])
    s = PC.C._geometry(p12.cloth, p12.options)["P"] / 0.30
    sh = PC.LINEN_SHADOW * s
    linen = (PC.O.rect(left, bottom, right, top), PC._hw(0.3 * s))
    shadow = (PC.O.rect(left + sh, bottom - sh, right + sh, top - sh), 0.0)
    disc = (PC.O.rect(*box), 0.0)
    assert PC.O.gap(disc, linen) > 0 and PC.O.gap(disc, shadow) > 0
    assert box[3] < bottom - sh                                   # beyond the near edge, not beside it
    frame = (-1.25, p12.shown.fbot - PC.ASIDE, 1.25, PC.CLOTHLB)  # word_gaps's frame (the Lua's, guarded)
    assert min(box[0] - frame[0], box[1] - frame[1], frame[2] - box[2], frame[3] - box[3]) > 0
    assert box[1] > frame[1] and box[3] < p12.shown.fbot          # in the band below the cloth's frame
    assert (left + right) / 2 < box[0]                            # at its right-hand end
    # round 4 keys the dark point as laid (the frame before the push): point, the kit's own
    # placement (its no-room fault is the check), and ring, which the Lua sets itself (measured
    # below); after the push, the frame unkeyed, the Lua sets no word of its own
    assert dark[0].labels == ("point", "ring") and [w for w, *_ in PC.own_words(dark[0].picture)] == ["ring"]
    assert p12.labels == () and PC.own_words(p12.picture) == ()

    # P1.3: the turned point pushed in EQUALS the turned point knotted in, exactly: one green
    # ring a quarter round with its three cords (C1); nothing set aside
    g13 = PC.graph_of(p13.dump)
    assert same_picture(g13, point_into_encoder(VT.Z, Fraction(1, 2)))
    assert same_picture(g13, green_three_legs(Fraction(1, 2)))
    assert not same_picture(g13, point_into_encoder(VT.Z, Fraction(3, 2)))     # not a quarter back
    for power in (-2, 0, 2):                                      # and three turned points at no number
        assert not same_picture(g13, three_points(VT.Z, Fraction(1, 2), power))
    assert p13.state.aside == 0 and p13.shown.counters == ()
    nodes = PC.parse_dump(p13.dump)[0]                            # "Taken in": turned, pushed, no point
    assert p13.state.kind == "turned" and p13.state.pushed and [n[0] for n in nodes.values()] == ["ring"]

    # P1.4: the line as strung is the storm line itself: eighteen cords, the master's edge count
    assert PC.line_cords() == 18 == len(p14.shown.spans)
    v = p14.shown
    assert len(v.rings) == 13 and len(v.bells) == 2
    assert sum(1 for r in v.rings if r[3] == "green") == 5 and sum(1 for r in v.rings if r[3] == "red") == 8
    # "Strung for the storm": every bead at rest (at the top of its ring), no clip, glass reset
    assert all(q == 0 for *_, q in v.rings) and v.clip == () and v.glass == "reset"
    assert all(np.isclose(by, r[2] + PC.SRM) and np.isclose(bx, r[1]) for (bx, by), r in zip(v.beads, v.rings))
    # the strands: the first on top; the bells between them, a little towards the kitchen
    tell = [r for r in v.rings if r[0].startswith("t")]
    bells = [r for r in v.rings if r[0].startswith("h")]
    assert [r[2] for r in tell] == sorted((r[2] for r in tell), reverse=True)
    assert all(tell[k + 1][2] < b[2] < tell[k][2] and b[1] > tell[k][1] for k, b in enumerate(bells))

    # the keyed words the Lua sets itself: measured clear (letters 0.02 cm from every part and
    # word, boxes 0.025 cm inside the frame), at the compiled boxes
    # (round 4: the tin keyed as set, before any lid comes up; the dark point keyed as laid)
    for pic in (tin[0].picture, dark[0].picture, p14.picture):
        PC.check_compiled_boxes(pic)
        least = PC.check_keyed_words(pic)
        assert set(least) == {w for w, *_ in PC.own_words(pic)} and min(least.values()) >= PC.LETTER_FLOOR
    assert tin[0].labels == ("first", "second", "third") and all(R.labels == () for R in tin[1:] + turned + [pale])
    assert p14.labels == ("door", "kitchen", "weather ring", "mending ring", "tell ring")
    # P1.4 is a counted frame: the eighteen cords as strung, never marked
    assert p14.step is None and not p14.marked and p14.mark_tikz == p14.arcs == p14.ghosts == []

    # the Lua's refusals: lids out of order, a lid lifted twice, a weather ring turned
    PC.TEX_ERRORS.clear()
    PC.tin_state("second dark")
    PC.tin_state("first dark, first lit")
    PC.line_state("middle weather ring at a half")
    PC.part_state("pushed in, pushed in", "the dark point")
    assert [k for k, _ in PC.TEX_ERRORS] == ["tin-order", "tin-lifted", "line-weather", "part-twice"]
    PC.TEX_ERRORS.clear()


# ------------------------------------------------------------------------------ pictures

def encoder(kind=VT.Z):
    """The parting ring: one spider at rest, one leg in, three out (strand 1 on top)."""
    p = Picture()
    i = p.inp(1)
    E = p.g.add_vertex(kind, 1, 1)
    p.wire(i, E)
    for k in range(3):
        p.wire(E, p.out(k))
    return p.done()


def round_trip():
    """Parting ring, three strands, joining ring.  Each strand carries a red ring at rest
    (bare cord, row 57) so that PyZX keeps three separate strands: its simple graphs hold at
    most one edge between two vertices."""
    p = Picture()
    i, o = p.inp(1), p.out(1)
    E, D = p.z(1, 1), p.z(1, 5)
    p.wire(i, E)
    for k in range(3):
        p.wire(E, p.x(k, 3, 0), D)
    p.wire(D, o)
    return p.done()


def point_into_encoder(colour, phase):
    p = Picture()
    pt = p.g.add_vertex(colour, 1, 0, phase=Fraction(phase))
    E = p.z(1, 1)
    p.wire(pt, E)
    for k in range(3):
        p.wire(E, p.out(k))
    return p.done()


def three_points(colour, phase, power=0):
    p = Picture()
    for k in range(3):
        p.wire(p.g.add_vertex(colour, k, 0, phase=Fraction(phase)), p.out(k))
    g = p.done()
    g.scalar.add_power(power)                    # a factor sqrt2^power
    return g


def green_three_legs(phase):
    p = Picture()
    v = p.z(1, 1, phase)
    for k in range(3):
        p.wire(v, p.out(k))
    return p.done()


def doubled(ops_top, ops_mirror):
    """A hand-built doubled wire (Chapter 9, row 66): row 0 the wire, row 1 its mirror copy.
    No ground vertex: pyzx 0.10.6 refuses grounded graphs.  ops are ('z'|'x', phase) in the
    order the message meets them."""
    p = Picture()
    for row, ops in ((0, ops_top), (1, ops_mirror)):
        chain = [p.inp(row)]
        for col, (kind, ph) in enumerate(ops, start=1):
            chain.append((p.z if kind == "z" else p.x)(row, col, ph))
        chain.append(p.out(row))
        p.wire(*chain)
    return p.done()


def sketchbook():
    rng = np.random.default_rng(1802)

    # --- Result 18.3: the encoder is one green spider, exactly, with no number
    enc = encoder()
    assert np.allclose(matrix_of(enc), ENC)
    assert not same_picture(enc, encoder(VT.X))            # not the red one (that is Enc', 18.13)

    # DERIVED: the parting ring followed by its mirror is a bare wire, number 1
    rt = round_trip()
    assert rt.num_edges() == 8                              # lead in, 3 x 2 strand lengths, lead out
    assert np.allclose(matrix_of(rt), I2)
    h = rt.copy()
    simp.full_reduce(h)
    assert h.num_vertices() == 2 and h.num_edges() == 1     # the two ends and one bare wire
    assert np.isclose(h.scalar.to_number(), 1)
    assert same_picture(h, rt)

    # --- Result 18.4 (C5): a red point, dark or lit, copies with the number 1/2
    for a in (0, 1):
        g = point_into_encoder(VT.X, a)
        ref = three_points(VT.X, a, power=-2)               # 1/2 x three red points (sqrt2^-2)
        assert same_picture(g, ref)
        assert not same_picture(g, three_points(VT.X, a))   # the 1/2 is not optional
        # DERIVED by full_reduce: three separate points, nothing joining them, number 1/2
        h = g.copy()
        simp.full_reduce(h)
        spiders = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
        assert len(spiders) == 3
        assert all(h.type(u) == VT.BOUNDARY for v in spiders for u in h.neighbors(v))
        assert {next(iter(h.neighbors(v))) for v in spiders} == set(h.outputs())
        assert np.isclose(h.scalar.to_number(), 0.5)
        assert same_picture(h, ref)

    # --- Result 18.4: a green point turned phi fuses in; it is not three points of any kind
    for phi in (Fraction(1, 2), Fraction(int(rng.integers(1, 97)), 97)):
        g = point_into_encoder(VT.Z, phi)
        h = g.copy()
        assert simp.spider_simp(h)                          # DERIVED: fusion (row 57)
        spiders = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
        assert len(spiders) == 1 and h.phase(spiders[0]) == phi
        assert len(list(h.neighbors(spiders[0]))) == 3
        assert same_picture(h, g) and same_picture(g, green_three_legs(phi))
        m = matrix_of(g).reshape(2, 4)                      # top wire | the other two
        assert np.linalg.matrix_rank(m) == 2                # entangled...
        pts = matrix_of(three_points(VT.Z, phi)).reshape(2, 4)
        assert np.linalg.matrix_rank(pts) == 1              # ...three points never are, at any number

    # --- (T1) translation check, not a derivation: a weighted sum of pictures is not one
    # diagram, so nothing rewrites it.  The doubled channel is evaluated against Kraus form.
    bare = doubled([], [])
    halves = doubled([("x", 1)], [("x", 1)])                # X on the wire, conj(X) = X on the mirror
    assert np.allclose(matrix_of(bare), np.eye(4))
    assert np.allclose(matrix_of(halves), np.kron(X, X.conj()))
    greens = doubled([("z", 1)], [("z", 1)])                # the phase flip: green halves
    assert np.allclose(matrix_of(greens), np.kron(Z, Z.conj()))
    # Y = iXZ: the message meets Z, then X.  On the mirror conj(Y) = -iXZ; the numbers i and -i
    # cancel, so the doubled Y is the plain picture Z-then-X on both copies, with no number.
    ys = doubled([("z", 1), ("x", 1)], [("z", 1), ("x", 1)])
    assert np.allclose(matrix_of(ys), np.kron(Y, Y.conj()))
    assert not np.allclose(matrix_of(ys), np.kron(Y, Y))    # the mirror conjugates
    for p in (0.1, 0.37, rng.uniform()):
        rho = random_density(rng)
        assert not np.allclose(rho, rho.real)               # a non-real density matrix
        vec = rho.reshape(4)                                # row-major doubling
        NX = (1 - p) * matrix_of(bare) + p * matrix_of(halves)
        assert np.allclose(NX @ vec, ((1 - p) * rho + p * X @ rho @ X).reshape(4))
        NZ = (1 - p) * matrix_of(bare) + p * matrix_of(greens)
        assert np.allclose(NZ @ vec, ((1 - p) * rho + p * Z @ rho @ Z).reshape(4))
        Nd = (1 - p) * matrix_of(bare) + p / 3 * (matrix_of(halves) + matrix_of(ys) + matrix_of(greens))
        dep = (1 - p) * rho + p / 3 * (X @ rho @ X + Y @ rho @ Y.conj().T + Z @ rho @ Z)
        assert np.allclose(Nd @ vec, dep.reshape(4))
        # the plausible slip: a half on the wire only is X rho, not X rho X
        wrong = (1 - p) * matrix_of(bare) + p * matrix_of(doubled([("x", 1)], []))
        assert not np.allclose(wrong @ vec, NX @ vec)


if __name__ == "__main__":
    verdict("ThreeForOne", blackboard, sketchbook)
