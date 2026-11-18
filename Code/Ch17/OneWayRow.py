"""Challenge III: The One-Way Row  (patterns; one step; feed-forward)
===================================================================
"These are the one thing I own that works only because you do."  -- Crab
"A lit reading sends a half forward: the next end turns the other way, and the one after gets a half."  -- Tortoise

Claim: reading one wire of CZ(|psi> (x) |+>) at Danos-Kashefi angle alpha with outcome s
leaves (1/sqrt2) X^s H Z(-alpha) |psi>, each outcome with probability 1/2 (Result 17.8).  In
the three-wire row with end turns pi/4, 0, -pi/4, read by the rule (second setting the other
way and a half on the third if the first is lit; third setting the other way if the second
is lit) the last reading is dark with probability 1 in all four branches; without the rule it
is lit half the time (Result 17.9); kept unread, wire 3 holds X^{s2} Z^{s1} |mu>, |mu> = T|+>.
In pictures: a red half against an end turned phi is e^{i phi} times the end turned -phi
(Result 17.10); the closed row is exactly 4 times the amplitude; and a lit half travels
forward (Result 17.11).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from itertools import product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import I2, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch17 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.diag([1, np.exp(1j * np.pi / 4)])
CZ = np.diag([1, 1, 1, -1]).astype(complex)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MU = T @ PLUS                                          # Chapter 16's magic state
PHI = (np.pi / 4, 0.0, -np.pi / 4)                     # the preset: the ends' turns (picture angles)


def Zr(a):
    return np.diag([1, np.exp(1j * a)])


def k3(a, b, c):
    return np.kron(np.kron(a, b), c)


def end(theta):
    """The effect (1/sqrt2) times a green end turned theta: (<0| + e^{i theta}<1|)/sqrt2, as a row."""
    return np.array([1, np.exp(1j * theta)]) / np.sqrt(2)


def dk_effect(alpha, s):
    """Danos-Kashefi's <+-_alpha|: the conjugate of (|0> +- e^{i alpha}|1>)/sqrt2."""
    return np.array([1, (-1) ** s * np.exp(1j * alpha)]).conj() / np.sqrt(2)


ROW = np.kron(CZ, I2) @ np.kron(I2, CZ) @ k3(PLUS, PLUS, PLUS)   # wire 1 on top; CZ_12 CZ_23 |+++>


def read_first_two(t1, t2):
    """Unnormalized state of wire 3 after the first two readings at end turns t1, t2."""
    return np.kron(np.kron(end(t1), end(t2)), I2) @ ROW


def blackboard():
    rng = np.random.default_rng(1704)
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj() * (psi[0] / psi[0].conj()))   # genuinely non-real
    # --- Result 17.8: one step, for three angles and both outcomes
    for alpha in rng.uniform(-np.pi, np.pi, 3):
        for s in (0, 1):
            eff = dk_effect(alpha, s)
            assert np.allclose(eff, end(-alpha + s * np.pi))            # the end turned -alpha + s pi
            branch = np.kron(eff.reshape(1, 2), I2) @ CZ @ np.kron(psi, PLUS)
            want = np.linalg.matrix_power(X, s) @ H @ Zr(-alpha) @ psi / np.sqrt(2)
            assert np.allclose(branch, want)
            assert np.isclose(np.vdot(branch, branch).real, 0.5)       # probability 1/2
            # the mix-ups: the end turned +alpha (the survey's label read as a turn), or no X^s
            wrong = np.kron(end(alpha + s * np.pi).reshape(1, 2), I2) @ CZ @ np.kron(psi, PLUS)
            assert not np.allclose(wrong, want)
            if s == 1:
                assert not np.allclose(branch, H @ Zr(-alpha) @ psi / np.sqrt(2))

    # --- the row: its three-wire state is the net (graph state) of the chain
    assert np.isclose(np.linalg.norm(ROW), 1)
    # read as preset, the first two dark: wire 3 holds (1/2) |mu>, and the last reads dark surely
    assert np.allclose(read_first_two(PHI[0], PHI[1]), MU / 2)
    assert np.allclose(H @ Zr(PHI[1]) @ H @ Zr(PHI[0]) @ PLUS, MU)

    # --- Result 17.9: the four branches, with and without the rule
    def tin(second, third):
        """{(s1, s2): (P(s1, s2), P(last lit | s1, s2), last setting)}"""
        out = {}
        for s1, s2 in product((0, 1), repeat=2):
            st = read_first_two(PHI[0] + s1 * np.pi, second(s1) + s2 * np.pi)
            p12 = np.vdot(st, st).real
            t3 = third(s1, s2)
            lit = abs(end(t3 + np.pi) @ st) ** 2 / p12
            dark = abs(end(t3) @ st) ** 2 / p12
            assert np.isclose(lit + dark, 1)
            out[(s1, s2)] = (p12, lit, t3)
        return out

    def overall(res):
        return sum(p * lit for p, lit, _ in res.values())

    rule = tin(lambda s1: (-1) ** s1 * PHI[1], lambda s1, s2: (-1) ** s2 * PHI[2] + s1 * np.pi)
    for (s1, s2), (p, lit, t3) in rule.items():
        assert np.isclose(p, 0.25)                                        # each pattern: a quarter
        assert np.isclose(lit, 0)                                         # the last: dark, surely
    settings = {k: np.exp(1j * v[2]) for k, v in rule.items()}
    want = {(0, 0): -np.pi / 4, (0, 1): np.pi / 4, (1, 0): 3 * np.pi / 4, (1, 1): 5 * np.pi / 4}
    assert all(np.isclose(settings[k], np.exp(1j * a)) for k, a in want.items())
    # without the rule (always -pi/4): lit half the time; never, even, always, even
    none = tin(lambda s1: PHI[1], lambda s1, s2: PHI[2])
    assert np.isclose(overall(none), 0.5)
    assert np.allclose([none[k][1] for k in sorted(none)], [0, 0.5, 1, 0.5])
    # the Crab's tins: Tin 1 (lit, dark, as preset) comes up lit with certainty
    assert np.isclose(none[(1, 0)][1], 1)
    # half a rule is no rule
    other_only = tin(lambda s1: (-1) ** s1 * PHI[1], lambda s1, s2: (-1) ** s2 * PHI[2])
    half_only = tin(lambda s1: PHI[1], lambda s1, s2: PHI[2] + s1 * np.pi)
    assert np.isclose(overall(other_only), 0.5) and np.isclose(overall(half_only), 0.25)

    # --- kept unread: wire 3 holds X^{s2} Z^{s1} |mu>, exactly (the branch over its length)
    for s1, s2 in product((0, 1), repeat=2):
        st = read_first_two(PHI[0] + s1 * np.pi, (-1) ** s1 * PHI[1] + s2 * np.pi)
        kept = st / np.linalg.norm(st)
        pauli = np.linalg.matrix_power(X, s2) @ np.linalg.matrix_power(Z, s1)
        assert np.allclose(kept, pauli @ MU)
        overlap = abs(np.vdot(MU, kept)) ** 2
        assert np.isclose(overlap, {(0, 0): 1, (0, 1): 0.5, (1, 0): 0, (1, 1): 0.5}[(s1, s2)])
        # the named Paulis turn it back into |mu>, up to a sign
        back = pauli @ kept
        assert np.allclose(back, MU) or np.allclose(back, -MU)
        assert np.allclose(np.linalg.matrix_power(Z, s1) @ np.linalg.matrix_power(X, s2) @ kept, MU)

    # --- Result 17.12 on the row (Danos-Kashefi Theorem 1, f(1) = 2, f(2) = 3), generic angles:
    # read 1, correct X_2 Z_3; read 2, correct X_3: every branch leaves the same state.
    for _ in range(3):
        a1, a2 = rng.uniform(-np.pi, np.pi, 2)                          # D-K angles
        outs = []
        for s1, s2 in product((0, 1), repeat=2):
            st = np.kron(dk_effect(a1, s1).reshape(1, 2), np.eye(4)) @ ROW          # wires 2, 3
            st = np.kron(np.linalg.matrix_power(X, s1), np.linalg.matrix_power(Z, s1)) @ st
            st = np.kron(dk_effect(a2, s2).reshape(1, 2), I2) @ st                  # wire 3
            st = np.linalg.matrix_power(X, s2) @ st
            assert np.isclose(np.vdot(st, st).real, 0.25)
            outs.append(st.ravel())
        for o in outs[1:]:
            assert np.allclose(o, outs[0])                                # the same, exactly
        assert np.allclose(outs[0], H @ Zr(-a2) @ H @ Zr(-a1) @ PLUS / 2)
        # without the corrections the branches differ
        st = np.kron(dk_effect(a1, 1).reshape(1, 2), np.eye(4)) @ ROW
        st = (np.kron(dk_effect(a2, 0).reshape(1, 2), I2) @ st).ravel()
        assert not np.allclose(st, outs[0])

    pictures(rule, none)


# every frame's dump, in the dialogue's order (the log's, card 3e.6 item 6: r17/build/def/dumps_md.txt)
FRAME_DUMPS = (
    # Third block 1 frame 1: The tin as set
    'first=1:down second=0:down last=7:down',
    # Third block 1 frame 2: Left lid off, lit
    'first=1:lit second=0:down last=7:down',
    # Third block 1 frame 3: Centre lid up, dark
    'first=1:lit second=0:dark last=7:down',
    # Third block 1 frame 4: Every lid lifted
    'first=1:lit second=0:dark last=7:lit',
    # Third block 2 frame 1: Pointer turned past rest
    'first=1:dark second=0:lit last=1:down',
    # Third block 3 frame 1: Pointer swung round
    'first=1:lit second=0:down last=3:down',
    # Third block 3 frame 2: Centre lid up, lit
    'first=1:lit second=0:lit last=3:down',
    # Third block 3 frame 3: Pointer back past rest
    'first=1:lit second=0:lit last=5:down',
    # Third block 3 frame 4: Tin read by the rules
    'first=1:lit second=0:lit last=5:dark',
    # Third block 4 frame 1: Ends waiting
    'ends=5,0,3 | cols=5 rows=2 y=0..1 near=false folded=false; 1:ring green 0 @0,0; 2:ring green 0 @2,0; 3:ring '
    'green 0 @4,0; 4:end green 15 @0,1; 5:end green 0 @2,1; 6:end green 9 @4,1; tie 1-2 tubes1 []; tie 2-3 tubes1'
    ' []; cord 1-4 [0,0 0,1]; cord 2-5 [2,0 2,1]; cord 3-6 [4,0 4,1]',
    # Third block 5 frame 1: A turn sent off the left end
    'ends=1,0,3 halves=tie1:near:green | cols=5 rows=2 y=0..1 near=false folded=false; 1:ring green 0 @0,0; '
    '2:ring green 0 @2,0; 3:ring green 0 @4,0; 4:end green 3 @0,1; 5:end green 0 @2,1; 6:end green 9 @4,1; tie '
    '1-2 tubes1 []; tie 2-3 tubes1 []; cord 1-4 [0,0 0,1]; cord 2-5 [2,0 2,1]; cord 3-6 [4,0 4,1]',
    # Third block 5 frame 2: Red past the tube
    'ends=1,0,3 halves=tie1:far:red | cols=5 rows=2 y=0..1 near=false folded=false; 1:ring green 0 @0,0; 2:ring '
    'green 0 @2,0; 3:ring green 0 @4,0; 4:end green 3 @0,1; 5:end green 0 @2,1; 6:end green 9 @4,1; tie 1-2 '
    'tubes1 []; tie 2-3 tubes1 []; cord 1-4 [0,0 0,1]; cord 2-5 [2,0 2,1]; cord 3-6 [4,0 4,1]',
    # Third block 5 frame 3: Out by the middle
    'ends=1,0,3 halves=loose2:red,tie2:near:red | cols=5 rows=2 y=0..1 near=false folded=false; 1:ring green 0 '
    '@0,0; 2:ring green 0 @2,0; 3:ring green 0 @4,0; 4:end green 3 @0,1; 5:end green 0 @2,1; 6:end green 9 @4,1; '
    'tie 1-2 tubes1 []; tie 2-3 tubes1 []; cord 1-4 [0,0 0,1]; cord 2-5 [2,0 2,1]; cord 3-6 [4,0 4,1]',
    # Third block 5 frame 4: Like a tin read dark
    'ends=1,0,7 | cols=5 rows=2 y=0..1 near=false folded=false; 1:ring green 0 @0,0; 2:ring green 0 @2,0; 3:ring '
    'green 0 @4,0; 4:end green 3 @0,1; 5:end green 0 @2,1; 6:end green 21 @4,1; tie 1-2 tubes1 []; tie 2-3 tubes1'
    ' []; cord 1-4 [0,0 0,1]; cord 2-5 [2,0 2,1]; cord 3-6 [4,0 4,1]',
)
THIRD_DUMPS = (
    "first=1:lit second=0:dark last=7:lit",
    "first=1:lit second=0:lit last=5:dark",
    "ends=5,0,3 | cols=5 rows=2 y=0..1 near=false folded=false; 1:ring green 0 @0,0; 2:ring green 0 @2,0; 3:ring green 0 @4,0; "
    "4:end green 15 @0,1; 5:end green 0 @2,1; 6:end green 9 @4,1; tie 1-2 tubes1 []; tie 2-3 tubes1 []; cord 1-4 [0,0 0,1]; "
    "cord 2-5 [2,0 2,1]; cord 3-6 [4,0 4,1]",
    "ends=1,0,7 | cols=5 rows=2 y=0..1 near=false folded=false; 1:ring green 0 @0,0; 2:ring green 0 @2,0; 3:ring green 0 @4,0; "
    "4:end green 3 @0,1; 5:end green 0 @2,1; 6:end green 21 @4,1; tie 1-2 tubes1 []; tie 2-3 tubes1 []; cord 1-4 [0,0 0,1]; "
    "cord 2-5 [2,0 2,1]; cord 3-6 [4,0 4,1]",
)


def same_angle(a, b):
    return np.isclose(np.exp(1j * a), np.exp(1j * b))


def rule_steps(s1, s2):
    """The Crab's rules, as the steps a tin is read by: the dials turned before each lid."""
    steps = ["first " + ("lit" if s1 else "dark")]
    if s1:
        steps += ["second dial the other way", "a half on the last dial"]
    steps += ["second " + ("lit" if s2 else "dark")]
    if s2:
        steps += ["last dial the other way"]
    return steps


def reading_chance(T, k):
    """The chance of lid k's reading (k = 0, 1, 2) given the readings before it, from the row's
    state and the dials as they stand in tin state T (a dial of s eighths an end turned s pi/4,
    a lit reading a half further): P(first k + 1 readings) / P(first k)."""
    t = [np.pi / 4 * L["s"] for L in T.lids]
    r = [int(L["r"] == "lit") for L in T.lids]

    def p(n):
        tot = 0
        for rest in product((0, 1), repeat=3 - n):
            rr = r[:n] + list(rest)
            tot += abs(k3(*(end(t[i] + rr[i] * np.pi) for i in range(3))) @ ROW) ** 2
        return tot
    return p(k + 1) / p(k)


def row_graph(R):
    """A tin-row frame as a PyZX graph, the halves on their way included (graph_of reads the dump's
    rings, ends and ties, and leaves the halves out): a half on a tie by its near ring a green half
    between the ring and the tie's tube, by its far ring a red half past the tube, on a loose cord a
    red half on the cord."""
    t = [Fraction(v[2], 12) for v in PC.parse_dump(R.dump)[0].values() if v[0] == "end"]
    g, rings, ends = closed_row(t)
    for h in PC.tinrow_halves(R.drawn):
        k = h["k"] - 1
        if h["on"] == "loose":
            g.remove_edge(g.edge(rings[k], ends[k]))
            v = g.add_vertex(VT.X if h["c"] == "red" else VT.Z, 0.5, 2 * k, phase=Fraction(1))
            g.add_edge((rings[k], v))
            g.add_edge((v, ends[k]))
            continue
        g.remove_edge(g.edge(rings[k], rings[k + 1]))
        v = g.add_vertex(VT.X if h["c"] == "red" else VT.Z, 0, 2 * k + 1, phase=Fraction(1))
        if h["at"] == "near":
            g.add_edge((rings[k], v))
            g.add_edge((v, rings[k + 1]), ET.HADAMARD)
        else:
            g.add_edge((rings[k], v), ET.HADAMARD)
            g.add_edge((v, rings[k + 1]))
    assert g.num_edges() == 5 + sum(1 for _ in PC.tinrow_halves(R.drawn))
    return g


def bead_ghost_ok(q, before_dump, g):
    """A bead's ghost stands on the bead as it was: on the ring (or end) nearest it in the frame
    before, at the bead's radius round that ring's centre, at the ring's turn then (in
    twenty-fourths; an eighth is three)."""
    nodes = [v for v in PC.parse_dump(before_dump)[0].values()]
    d, (kind, col, t, x, y) = min(((np.hypot(g.XY((v[3], v[4]))[0] - q[0], g.XY((v[3], v[4]))[1] - q[1]), v) for v in nodes),
                                  key=lambda u: u[0])
    cx, cy = g.XY((x, y))
    a = np.radians(90 - 45 * t / 3)
    return d < g.RO and np.hypot(q[0] - cx - g.RM * np.cos(a), q[1] - cy - g.RM * np.sin(a)) < 1e-3


def gap(p, c):
    """From a point to a circle's edge (x, y, r)."""
    return np.hypot(p[0] - c[0], p[1] - c[1]) - c[2]


def pictures(rule, none):
    """The Third Challenge's prop pictures (identity card, sections 4 and 3e), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch17.py, and held against
    this file's blackboard (`rule`, `none`: the four branches with and without the Crab's rules) and
    pictures (graph_of: a structural reading of each cloth dump, not a proof).  Round 4 reads the
    tins lid by lid and turns the dials as the rules say; and sends the half on step by step, every
    frame the same value as the row before it."""
    calls = DIALOGUE.check_layout("Third", sizes=[1, 1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("onewaytinsteps", True, "", ("first lit; second dark; last lit",),
         "The tin as set; Left lid off, lit; Centre lid up, dark; Every lid lifted"),
        ("onewaytin", False, "first dark, second lit; step=last dial the other way", (), "Pointer turned past rest"),
        ("onewaytinsteps", False, "after=first lit, second dial the other way; from=off",
         ("a half on the last dial; second lit; last dial the other way; last dark",),
         "Pointer swung round; Centre lid up, lit; Pointer back past rest; Tin read by the rules"),
        ("tincloth", True, "", ("lit, dark, lit",), "Ends waiting"),
        ("tinclothsteps", False, "from=off",
         ("lit, dark, lit", "the first half off its end; through the tube; out on every other cord; on to the ends"),
         "A turn sent off the left end; Red past the tube; Out by the middle; Like a tin read dark")]
    PC.check_captions(DIALOGUE, "Third", calls)
    PC.check_section(DIALOGUE, "Third", calls)
    assert PC.check_anchors(DIALOGUE, "Third") == (8, 12)                 # P3.1, B3.1a, P3.2, P3.4
    frames = [PC.frame_chain(c) for c in calls]
    assert [len(f) for f in frames] == [4, 1, 4, 1, 4]                    # card 3e.6: 5 blocks, 14 frames
    assert tuple(R.head for f in frames for R in f) == FRAME_DUMPS
    b31, (b31a,), b32, (p33,), b34 = frames
    p31, p32, p34 = b31[-1], b32[-1], b34[-1]
    pics = [p31, p32, p33, p34]
    # every strip's last frame is round 3's picture of its block (round 3's assertions carry over)
    assert p31.dump == PC.onewaytin("first lit, second dark, last lit", False).dump
    r3 = ", ".join(rule_steps(1, 1) + ["last dark"])                     # round 3's P3.2 bracket
    assert PC.tin_state(p32.drawn_bracket) == PC.tin_state(r3) and p32.dump == PC.onewaytin(r3, False).dump
    assert p34.dump == PC.tincloth("first half sent on", "lit, dark, lit", False).dump
    assert tuple(p.dump for p in pics) == THIRD_DUMPS
    assert [p.frame for p in pics] == [1.56, 1.56, 1.0, 1.0]
    assert {R.frame for f in frames[:3] for R in f} == {1.56} and {R.frame for R in b34} == {1.0}

    # the tins: a dial setting of k eighths is an end turned k pi/4, clockwise from the top
    # (the pointer's tip there); a reading lit is the end turned a half further
    setting = lambda p: [np.pi / 4 * L["s"] for L in p.state.lids]
    reads = lambda p: [int(L["r"] == "lit") for L in p.state.lids]
    for p in (p31, p32):
        for (x, y, k), path in zip(p.shown.lids, p.shown.pointers):
            tip = PC.pointer_path(x, y, k)[1][2:4]
            ang = np.arctan2(np.mean([q[1] for q in tip]) - y, np.mean([q[0] for q in tip]) - x)
            assert same_angle(ang, np.pi / 2 - np.pi / 4 * k)
        assert all(y == PC.LY for x, y, k in p.shown.lids)               # every lid lifted, lying in front
    # P3.1: Tin 1, as preset, no rule: the ends turned pi/4, 0, -pi/4, read (1, 0) and then the
    # last -- lit, as it is certain to be in that branch without the rules
    assert all(same_angle(a, b) for a, b in zip(setting(p31), PHI))
    assert reads(p31) == [1, 0, 1]
    assert np.isclose(none[(1, 0)][1], 1)
    # P3.2: Tin 3, read by the rules in the branch (1, 1): the last dial at 5 pi/4, and the last
    # hollow dark with probability 1 there
    s1, s2 = reads(p32)[:2]
    assert (s1, s2) == (1, 1) and reads(p32)[2] == 0
    assert same_angle(setting(p32)[2], rule[(1, 1)][2]) and same_angle(setting(p32)[2], 5 * np.pi / 4)
    assert np.isclose(rule[(1, 1)][1], 0)
    # the mix-ups: the same tin read without the rule leaves the last dial at the preset (and
    # its last reading a coin toss); a dial turned after its lid is lifted is refused
    plain = PC.tin_state("first lit, second lit")
    assert not same_angle(np.pi / 4 * plain.lids[2]["s"], rule[(1, 1)][2])
    assert np.isclose(none[(1, 1)][1], 0.5)
    PC.TEX_ERRORS.clear()
    PC.tin_state("first lit, second lit, second dial the other way")
    PC.tin_state("second lit")
    assert [k for k, _ in PC.TEX_ERRORS] == ["tin-spent", "tin-order"]
    PC.TEX_ERRORS.clear()
    # and every branch of the rules, read through the transcription, ends dark, surely
    for (b1, b2), (p, lit, t3) in rule.items():
        T_ = PC.tin_state(", ".join(rule_steps(b1, b2)))
        assert same_angle(np.pi / 4 * T_.lids[2]["s"], t3) and np.isclose(lit, 0)

    # B3.1, Tin 1 lid by lid (card 3e.6, item 7): the tin as set (the preset dials, every lid down),
    # then each lid lifted in order, its reading's chance from the row as the dials stand: the first
    # two a coin each, the last (no rule turned) lit surely
    assert [R.step for R in b31] == [None, "first lit", "second dark", "last lit"]
    assert b31[0].drawn.lids == PC.tin_state("").lids and not b31[0].marked and b31[0].keyed
    assert [[L["up"] for L in R.drawn.lids] for R in b31] == [[False] * 3, [True, False, False], [True, True, False], [True] * 3]
    assert [np.round(reading_chance(R.drawn, k), 12) for k, R in enumerate(b31[1:])] == [0.5, 0.5, 1.0]
    assert np.isclose(reading_chance(b31[2].drawn, 1), none[(1, 0)][0] / 0.5)            # P(1, 0) / P(first lit)
    # B3.1a, Tin 2 (first dark, second lit): its last dial turned the other way, back through rest
    # and on -- from seven eighths to one, the rule's setting for (0, 1); its last lid then dark surely
    assert b31a.step == "last dial other way" and b31a.before.lids[2]["s"] == 7 and b31a.after.lids[2]["s"] == 1
    assert same_angle(np.pi / 4 * b31a.after.lids[2]["s"], rule[(0, 1)][2])
    assert not same_angle(np.pi / 4 * b31a.before.lids[2]["s"], rule[(0, 1)][2])        # the preset: not it
    dark = PC.tin_state("first dark, second lit, last dial the other way, last dark")
    assert np.isclose(reading_chance(dark, 2), 1)
    assert np.isclose(reading_chance(PC.tin_state("first dark, second lit, last dark"), 2), 0.5)   # unturned: a coin
    # B3.2, Tin 3 by the rules (first lit: the second dial the other way, at rest still, and a half on
    # the last; second lit: the last dial the other way): frame by frame the dials the rule names,
    # the second lid a coin, the last dark surely
    assert [R.step for R in b32] == ["a half on last dial", "second lit", "last dial other way", "last dark"]
    assert b32[0].before == PC.tin_state("first lit, second dial the other way")
    assert [R.drawn.lids[2]["s"] for R in b32] == [3, 3, 5, 5]
    assert np.isclose(reading_chance(b32[1].drawn, 1), 0.5) and np.isclose(reading_chance(b32[3].drawn, 2), 1)
    assert same_angle(np.pi / 4 * b32[2].drawn.lids[2]["s"], rule[(1, 1)][2])
    assert not same_angle(np.pi / 4 * b32[0].drawn.lids[2]["s"], rule[(1, 1)][2])        # half-way: not yet

    # their marks: a lid lifted, a lift arc beside it, passing over the tin's near edge, bowed at
    # least the floor (no ghost: its open hollow lies where it lay); a dial turned, the pointer's ghost
    # where it stood and a turn arc round inside the lid the way it went (the other way through rest),
    # clear of the knob, the dots and the pointers on the drawn extent
    hw = 0.6 * 2.54 / 72.27 / 2                                           # the swing mark's half width (0.6pt)
    knobr = 0.05 + 0.25 * 2.54 / 72.27 / 2                                # the knob as drawn, its 0.25pt edge included
    assert '" circle (0.05);")' in PC.LUA.read_text(encoding="utf-8")     # (ch17.lua's lid: the brass knob)
    turns = []
    for R in b31[1:] + [b31a] + b32:
        assert R.marked and not R.keyed and len(R.arcs) == 1, R.step
        (a,) = R.arcs
        assert a.kind == "swing", R.step                              # (MINSAG is the slide arcs' floor)
        k = next(i for i in range(3) if R.before.lids[i] != R.after.lids[i])
        cx = PC.LIDX[k]
        if R.after.lids[k]["up"]:                                         # a lid lifted
            assert R.ghosts == [] and a.what == "lift arc for the " + PC.LIDS[k] + " lid"
            assert np.isclose(a.frm[0], a.to[0]) and a.frm[1] > a.to[1] and np.isclose(a.sag, 0.064, atol=5e-4)
            assert min(p[0] for p in a.body) > cx                         # bowed to the right of the lid
            for y in (0, PC.LY):                                          # clear of the lid where it lay and laid
                assert min(np.hypot(p[0] - cx, p[1] - y) for p in a.pts) - PC.LR - hw > 0.03
            continue
        (q,) = R.ghosts
        # a turn arc is a piece of the circle of radius TURNR round the knob: curved by its circle
        assert all(np.isclose(np.hypot(p[0] - cx, p[1]), PC.TURNR, atol=2e-3) for p in a.body), R.step
        _, d, s0, s1, a0, a1 = a.turn
        assert (s0, s1) == (R.before.lids[k]["s"], R.after.lids[k]["s"]) and q.pointer[0] == s0
        assert q.pointer[1] == PC.pointer_poly(cx, 0, s0)                 # the ghost where the pointer stood
        assert (s0 + d) % 8 == s1 and np.sign(a1 - a0) == -np.sign(d)     # clockwise for d > 0
        if R.step.endswith("other way"):
            assert s1 == (-s0) % 8 and abs(d) == (2 * s0 if s0 <= 4 else 2 * (8 - s0))      # back through rest
            rest = (a0 - 90) % 360 if d > 0 else (90 - a0) % 360       # how far round it meets the top
            assert rest < abs(a1 - a0)                                    # the arc sweeps across rest (the top)
        else:
            assert d == 4
        knob = min(np.hypot(p[0] - cx, p[1]) for p in a.body) - knobr - hw
        dots = min(gap(p, (cx + PC.DOTR * np.cos(np.radians(90 - 45 * j)), PC.DOTR * np.sin(np.radians(90 - 45 * j)), PC.DOTS))
                   for p in a.body for j in range(8)) - hw
        poly = lambda p, P: min(PC.seg_dist(p, P[i], P[(i + 1) % len(P)]) for i in range(len(P)))
        ptr = min(poly(p, PC.pointer_poly(cx, 0, s1)) for p in a.body) - hw
        ghost = min(poly(p, q.pointer[1]) for p in a.body) - hw
        head = a.pts[len(a.body):]
        hdots = min(gap(p, (cx + PC.DOTR * np.cos(np.radians(90 - 45 * j)), PC.DOTR * np.sin(np.radians(90 - 45 * j)), PC.DOTS))
                    for p in head for j in range(8))
        assert min(knob, dots, ptr, ghost, hdots) > 0, (R.step, knob, dots, ptr, ghost, hdots)
        turns.append((R.step, round(a.len, 3), round(dots, 3), round(knob, 3), round(ptr, 3)))
    # L7: B3.1a's arc 0.170 cm long; its body clear of the dots by 0.032, of the knob's drawn edge by
    # 0.097, of the pointer by 0.028 (its head's corner comes 0.010 from a dot)
    assert turns[0] == ("last dial other way", 0.17, 0.032, 0.097, 0.028), turns[0]
    assert [t[1] for t in turns] == [0.17, 0.424, 0.679]

    # the tin row: the ends are the (R) reading of lit, dark, lit -- each end turned as its dial
    # and a half more where lit; P3.4 sends the first half on (Result 17.11)
    e33 = [t for k, c, t, x, y in PC.parse_dump(p33.dump)[0].values() if k == "end"]
    e34 = [t for k, c, t, x, y in PC.parse_dump(p34.dump)[0].values() if k == "end"]
    assert e33 == [15, 0, 9] and e34 == [3, 0, 21]                      # twenty-fourths
    assert p33.state.ends == [(PC.PRESET[i] + 4 * r) % 8 for i, r in enumerate((1, 0, 1))]
    assert p34.state.ends == list(PC.PRESET)                             # a tin read dark, dark, dark
    t33 = [Fraction(t, 12) for t in e33]
    t34 = [Fraction(t, 12) for t in e34]
    g33, g34 = PC.graph_of(p33.dump), PC.graph_of(p34.dump)
    assert same_picture(g33, closed_row(t33)[0]) and same_picture(g34, closed_row(t34)[0])
    amp = lambda t: k3(*(end(np.pi * float(x)) for x in t)) @ ROW
    assert np.isclose(value(g33), 4 * amp(t33)) and np.isclose(value(g34), 4 * amp(t34))
    # the two closed rows are exactly equal: the travelling half's number e^{i phi_2} is 1,
    # the second end being at rest
    assert t33[1] == 0 and same_picture(g33, g34)
    assert np.isclose(value(g33), np.exp(1j * np.pi * float(t33[1])) * value(g34))
    # the half sent on the wrong way (the second end left alone, the third not turned) is not it
    wrong = closed_row([t33[0] - 1, t33[1], t33[2]])[0]
    assert not same_picture(g33, wrong)
    # the row's number is four times the branch's amplitude: Tin 1's branch (1, 0, lit) has
    # probability 1/4 x 1 (the blackboard's, without the rules)
    assert np.isclose(abs(value(g33)), 4 * np.sqrt(none[(1, 0)][0] * none[(1, 0)][1]))

    # B3.4, the half sent on step by step (card 3e.6, item 7): off the first end and through its ring
    # onto the tie (green, by the ring), past the tube (red), out of the second ring on every other
    # cord (onto its loose cord and the tie onward), on to the ends.  Every frame, its halves on
    # their way read in, has exactly P3.3's value, scalar included; read without its halves (as the
    # dump's rings and ends alone) it has not
    assert [R.step for R in b34] == ["first half off its end", "through tube", "out on every other cord", "on to ends"]
    assert b34[0].before == p33.state
    assert [R.drawn.ends for R in b34] == [[1, 0, 3]] * 3 + [[1, 0, 7]]
    assert [[(h["on"], h["k"], h.get("at"), h["c"]) for h in PC.tinrow_halves(R.drawn)] for R in b34] == [
        [("tie", 1, "near", "green")], [("tie", 1, "far", "red")], [("loose", 2, None, "red"), ("tie", 2, "near", "red")], []]
    for R in b34:
        g = row_graph(R)
        assert same_picture(g, g33), R.step
        if PC.tinrow_halves(R.drawn):
            assert not np.isclose(value(PC.graph_of(R.dump)), value(g33)), R.step          # graph_of leaves the halves out
            # the plausible mix-up: a half the other colour (a red half before the tube, a green past it)
            other = row_graph_with(R, [dict(h, c=("green" if h["c"] == "red" else "red")) for h in PC.tinrow_halves(R.drawn)])
            assert not same_picture(other, g33), R.step
    assert same_picture(row_graph(b34[-1]), g34) and not b34[-1].halves
    # the halves drawn: at 0.57 of a ring, each on the cord the state names, clear of its ring
    gg = PC.G(PC.tinrow_scene(b34[0].drawn), PC.ROWOPTS, PC.RWHO)
    assert PC.TK == 0.57
    for R in b34:
        hv = PC.tinrow_halves(R.drawn)
        assert len(R.halves) == len(hv)
        for (x, y, r, col), h in zip(R.halves, hv):
            assert col == h["c"] and np.isclose(r, PC.TK * gg.RO)
            ring = gg.XY((PC.ROWX[h["k"] - 1], 0))
            if h["on"] == "loose":
                assert np.isclose(x, ring[0]) and gg.Y(1) < y < ring[1] - gg.RO    # on its loose cord, below its ring
            else:
                near, far = ring, gg.XY((PC.ROWX[h["k"]], 0))
                assert np.isclose(y, ring[1]) and near[0] < x < far[0]          # on its tie
                by = near if h["at"] == "near" else far
                assert np.isclose(abs(x - by[0]) - gg.RO - r, PC.HALFGAP) and PC.HALFGAP > 0
    # its marks: each half's ghost where it lay the frame before (the first end's bead where it
    # stood, the last end's likewise); an arc from each part that moved to where it went -- every
    # one drawn (L6: both of "out on every other cord"), bowed at least the floor, clear of the net
    assert [len(R.arcs) for R in b34] == [1, 1, 2, 2]
    for P, R in zip(b34, b34[1:]):
        hg = [q for q in R.ghosts if q.half]
        assert sorted((round(q[0], 3), round(q[1], 3)) for q in hg) == sorted((round(x, 3), round(y, 3)) for x, y, r, c in P.halves)
    ends_cm = [gg.XY((x, 1)) for x in PC.ROWX]
    for R in b34:
        for a in R.arcs:
            assert a.kind == "slide" and a.sag >= PC.MINSAG - 1e-9 and a.fit[0] >= PC.MINCLEAR - 1e-9, a.what
            if R.halves:                                                  # it ends at a half it brought
                assert min(gap(a.to, h) for h in R.halves) < 0.05, a.what
            else:                                                         # or at the end it went into
                assert min(range(3), key=lambda i: np.hypot(a.to[0] - ends_cm[i][0], a.to[1] - ends_cm[i][1])) in (1, 2)
            hg = [q for q in R.ghosts if q.half]
            if hg:                                                        # it starts at the half's ghost
                assert min(gap(a.frm, q) for q in hg) < 0.05, a.what
            else:                                                         # or at the first end
                assert min(range(3), key=lambda i: np.hypot(a.frm[0] - ends_cm[i][0], a.frm[1] - ends_cm[i][1])) == 0
    assert [q.what for q in b34[0].ghosts] == ["the first end"] and b34[0].ghosts[0].bead
    for P, R in zip([p33] + b34, b34):                                     # each bead's ghost on the bead as it was
        for q in (q for q in R.ghosts if q.bead):
            assert bead_ghost_ok(q, P.dump, gg), (R.step, q.what)
    assert [q.what for q in b34[3].ghosts if q.bead] == ["the last end"]
    assert any("no ghost of the second end's bead" in n for n in b34[3].notes)   # rest the other way is rest
    # L6: "out on every other cord", both arcs drawn: 0.268 cm (bowed 0.068) and 0.152 (bowed 0.047)
    assert sorted((round(a.len, 3), round(a.sag, 3)) for a in b34[2].arcs) == [(0.152, 0.047), (0.268, 0.068)]
    assert sorted(round(a.fit[0], 3) for a in b34[2].arcs) == [0.023, 0.026]

    # the captions' truth: every lid lifted (lit or dark); the rules' steps; ends, not rings; the
    # ends at the preset, as a tin read dark, dark and dark would leave them; the first lid off lit,
    # the centre up dark, then lit; the pointer past rest; a turn sent off the left end (the first),
    # red past the tube
    assert all(L["up"] and L["r"] in ("lit", "dark") for L in p31.state.lids)
    assert b31[1].drawn.lids[0]["r"] == "lit" and b31[2].drawn.lids[1]["r"] == "dark" and b32[1].drawn.lids[1]["r"] == "lit"
    assert p33.dump.startswith("ends=") and not p33.state.slid
    assert p34.state.ends == PC.tinrow_state("", "dark, dark, dark").ends
    assert b34[0].drawn.ends[0] == p33.state.ends[0] - 4 and b34[1].halves[0][3] == "red"
    # fixed parts, frame to frame: the tin row's three rings at 0, 2, 4 in every frame, and the
    # ends the half has not reached where they were; the tins' lids not touched by a step as they lay
    rings = lambda p: sorted(x for k, c, t, x, y in PC.parse_dump(p.dump)[0].values() if k == "ring")
    for p in [p33] + b34:
        assert rings(p) == [0, 2, 4]
        assert [v for v in PC.parse_dump(p.dump)[0].values() if v[0] == "end"][1] == ("end", "green", 0, 2, 1)
    for P, R in zip(b31 + b32, b31[1:] + b32[1:]):
        if R.before == P.drawn:
            k = [i for i in range(3) if P.drawn.lids[i] != R.drawn.lids[i]]
            assert len(k) == 1, R.step                                     # one lid or dial at a time
    assert [L for L in b31a.before.lids[:2]] == [L for L in b31a.after.lids[:2]]

    # the keyed words: the lids' names (P3.1's strip, on its before-frame) over the tin, and on the
    # row the kit's ring (placed by the kit, not re-measured here; card section 4: at least 0.028 cm)
    # and the Lua's own end, in the margin at the linen's left, measured
    assert b31[0].labels == ("first", "second", "last") and all(R.labels == () for R in b31[1:] + [b31a] + b32)
    assert p33.labels == ("ring", "end") and all(R.labels == () for R in b34)
    for p in (b31[0], p33):
        PC.check_compiled_boxes(p.picture)
        assert min(PC.check_keyed_words(p.picture).values()) >= 0.02
    # a marked step frame is never keyed; a frame with a half on its way is never keyed
    assert PC.faults(PC.run_frame, "tin", "step=first lit", "", True) == ["marks-keyed"]
    assert PC.faults(PC.run_frame, "row", "step=first half off its end", "lit, dark, lit", True) == ["row-halves-keyed"]
    PC.TEX_ERRORS.clear()


def row_graph_with(R, halves):
    """row_graph with other halves in place of the state's (a mix-up)."""
    saved = PC.tinrow_halves
    PC.tinrow_halves = lambda _R: halves
    try:
        return row_graph(R)
    finally:
        PC.tinrow_halves = saved


# ------------------------------------------------------------------ pictures
# PyZX draws the rings and ends as green (Z) spiders, the half as a red (X) spider;
# a Hadamard edge is the yellow box on a tie.
def closed_row(t):
    """Three green rings at rest, tied 1=>2=>3 by Hadamard edges, an end turned t[k] on each."""
    g = zx.Graph()
    rings = [g.add_vertex(VT.Z, 0, 2 * k) for k in range(3)]
    ends = [g.add_vertex(VT.Z, 1, 2 * k, phase=Fraction(t[k])) for k in range(3)]
    g.add_edge((rings[0], rings[1]), ET.HADAMARD)
    g.add_edge((rings[1], rings[2]), ET.HADAMARD)
    for r, e in zip(rings, ends):
        g.add_edge((r, e))
    return g, rings, ends


def value(g):
    return complex(np.asarray(matrix_of(g)).ravel()[0])


def end_on_cord(phi, half=True):
    """An end turned phi (one input leg), with a red pi on its cord if half."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    e = g.add_vertex(VT.Z, 0, 2, phase=Fraction(phi))
    if half:
        r = g.add_vertex(VT.X, 0, 1, phase=Fraction(1))
        g.add_edge((i, r))
        g.add_edge((r, e))
    else:
        g.add_edge((i, e))
    g.set_inputs((i,))
    g.set_outputs(())
    return g, e


def one_step(theta):
    """Input - ring (an end turned theta hanging) => ring at rest - output."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    r1 = g.add_vertex(VT.Z, 0, 1)
    e = g.add_vertex(VT.Z, 1, 1, phase=Fraction(theta))
    r2 = g.add_vertex(VT.Z, 0, 2)
    o = g.add_vertex(VT.BOUNDARY, 0, 3)
    g.add_edge((i, r1))
    g.add_edge((r1, e))
    g.add_edge((r1, r2), ET.HADAMARD)
    g.add_edge((r2, o))
    g.set_inputs((i,))
    g.set_outputs((o,))
    return g


def sketchbook():
    rng = np.random.default_rng(1705)
    rand = lambda: Fraction(int(rng.integers(1, 194)), 97)      # generic angles, in units of pi
    ex = lambda f: np.exp(1j * np.pi * float(f))

    # --- the closed row: exactly 4 times the amplitude, for all eight readings
    for base in ((Fraction(1, 4), Fraction(0), Fraction(-1, 4)), (rand(), rand(), rand())):
        for s in product((0, 1), repeat=3):
            t = [base[k] + s[k] for k in range(3)]
            g, _, _ = closed_row(t)
            amp = k3(end(np.pi * float(t[0])), end(np.pi * float(t[1])), end(np.pi * float(t[2]))) @ ROW
            assert np.isclose(value(g), 4 * amp)
            assert not np.isclose(value(g), amp) or np.isclose(amp, 0)
            # DERIVED: a reading is an end slid into its ring (fusion, row 57); nothing changes
            f = g.copy()
            assert simp.spider_simp(f)
            assert sorted(Fraction(f.phase(v)) % 2 for v in f.vertices()) == sorted(x % 2 for x in t)
            assert np.isclose(value(f), value(g))

    # --- the one step as a picture: exactly X^s H Z(phi), i.e. sqrt2 times the blackboard's branch
    for phi in (Fraction(1, 4), rand()):
        for s in (0, 1):
            g = one_step(phi + s)
            want = np.linalg.matrix_power(X, s) @ H @ Zr(np.pi * float(phi))
            assert np.allclose(matrix_of(g), want)
            assert not np.allclose(matrix_of(g), want / np.sqrt(2))
            # DERIVED: slide the end in (fusion), and the ring at rest with two legs is wire
            d = g.copy()
            assert simp.spider_simp(d)
            simp.id_simp(d)
            sp = [v for v in d.vertices() if d.type(v) != VT.BOUNDARY]
            assert len(sp) == 1 and Fraction(d.phase(sp[0])) == (phi + s) % 2
            assert same_picture(d, g)

    # --- Result 17.10: a red half against an end turned phi = e^{i phi} x the end turned -phi
    for phi in (Fraction(0), Fraction(1), Fraction(1, 4), rand()):
        lhs, e = end_on_cord(phi)
        rhs, _ = end_on_cord(-phi, half=False)
        assert np.allclose(matrix_of(lhs), ex(phi) * matrix_of(rhs))
        rhs_n = rhs.copy()
        rhs_n.scalar.add_phase(phi)                                       # the number e^{i phi}
        assert same_picture(lhs, rhs_n)
        # DERIVED: pi-commutation (row 60) pushes the half through the end; PyZX leaves the
        # number unwritten, so we write it
        d = lhs.copy()
        assert simp.pi_commute(d, e)
        simp.id_simp(d)
        assert all(d.type(v) != VT.X for v in d.vertices())
        assert Fraction(d.phase(e)) % 2 == (-phi) % 2
        d.scalar.add_phase(phi)
        assert same_picture(d, lhs)
        if phi not in (0, 1):
            plain, _ = end_on_cord(phi, half=False)
            assert not same_picture(lhs, plain)                            # the half is not free
    # by case: at rest, unchanged (1); a half, a half (-1); an eighth, an eighth back
    assert np.allclose(matrix_of(end_on_cord(0)[0]), matrix_of(end_on_cord(0, False)[0]))
    assert np.allclose(matrix_of(end_on_cord(1)[0]), -matrix_of(end_on_cord(1, False)[0]))
    assert np.allclose(matrix_of(end_on_cord(Fraction(1, 4))[0]),
                       ex(Fraction(1, 4)) * matrix_of(end_on_cord(Fraction(-1, 4), False)[0]))

    # --- Result 17.11: the travelling half, both sides built and compared as numbers
    P = lambda t: value(closed_row(t)[0])
    for p1, p2, p3 in [(Fraction(1, 4), Fraction(0), Fraction(-1, 4))] + [(rand(), rand(), rand()) for _ in range(4)]:
        assert np.isclose(P([p1 + 1, p2, p3]), ex(p2) * P([p1, -p2, p3 + 1]))
        assert np.isclose(P([p1, p2 + 1, p3]), ex(p3) * P([p1, p2, -p3]))
        if p2 not in (0, 1):                                              # the half does not stay put,
            assert not np.isclose(P([p1 + 1, p2, p3]), ex(p2) * P([p1, p2, p3 + 1]))  # nor skip the turn
            assert not np.isclose(P([p1 + 1, p2, p3]), P([p1, p2, p3]))
        # DERIVED: put a red at rest on the second end's cord (a red at rest with two legs is
        # wire, row 57); pi-commutation at the second ring (at rest, row 120) sends a half out
        # on every leg: onto rings 1 and 3 (through their boxes, where it lands green) and onto
        # the red on the cord; pi-commutation at the second end takes it off again, turning
        # the end the other way (Result 17.10, the number e^{i p2} written by hand).
        # (pyzx 0.10.6's pi_commute inserts a red at rest, not a red half, on a leg whose
        # neighbour cannot absorb the half; the red at rest put on the cord first avoids that
        # branch, and the value is checked after each step.)
        g, rings, ends = closed_row([p1 + 1, p2, p3])
        d = g.copy()
        d.remove_edge(d.edge(rings[1], ends[1]))
        red = d.add_vertex(VT.X, 0.5, 2)
        d.add_edge((rings[1], red))
        d.add_edge((red, ends[1]))
        assert np.isclose(value(d), value(g))
        assert simp.pi_commute(d, rings[1])
        assert d.phase(red) == 1 and np.isclose(value(d), value(g))
        assert simp.pi_commute(d, ends[1])
        simp.id_simp(d)
        assert all(d.type(v) == VT.Z for v in d.vertices())
        simp.spider_simp(d)
        assert not np.isclose(value(d), value(g)) or p2 in (0, 1) or np.isclose(value(g), 0)
        d.scalar.add_phase(p2)
        assert np.isclose(value(d), value(g))
        want, _, _ = closed_row([p1, -p2, p3 + 1])
        w = want.copy()
        simp.spider_simp(w)
        assert sorted(Fraction(d.phase(v)) % 2 for v in d.vertices()) == \
            sorted(Fraction(w.phase(v)) % 2 for v in w.vertices())


if __name__ == "__main__":
    verdict("OneWayRow", blackboard, sketchbook)
