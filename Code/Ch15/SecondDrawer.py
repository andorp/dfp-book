"""Challenge III: One Touch  (one step finds one drawer in four; the master, The Second Drawer)
=============================================================================================
"Granted."  -- Genie
"Upper dark, lower lit. The second drawer."  -- Achilles

Claim: for four drawers and every marked drawer w, D O_w |s> = |w> exactly, with one use of
the marker (Result 15.8); for N drawers one step gives (1 - 4/N)|s> + (2/sqrt N)|w>; and the
master, The Second Drawer -- W R W O_01 W |00> drawn gate by gate, unfused, two dark points,
six collars, the marker (two rings, a half-ring on the lower row, a hub and its pendant) and
the turn-about (two half-rings, two rings, a hub and its pendant) -- is exactly |0>|1>,
scalar 1, nothing drawn beside it (Result 15.16).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import ket, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch15 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

# The Genie's lines in the Third Challenge, frozen from the approved text (identity card,
# section 3c): every one of them, in order, word for word, inline directions included.
GENIE = [
    "**Genie:** The Office of Wishes. *[He glances along the shelf to the drawer-case.]* Ah. The Office's newest. "
    "They are very pleased with it upstairs.",
    "**Genie:** One wish is one touch of the marker. A passage of this case touches it once, so a passage is one "
    "wish. What the case does in a passage besides that touch is the case's business, and costs you nothing.",
    "**Genie:** A plain question is also one touch: ask me whether a drawer you name is the marked one, and I shall "
    "put it to the marker myself and bring you yes or no. That, too, is one wish. Reading windows is not a wish, nor "
    "opening a spent case's drawers, nor anything you do on paper, card or cloth. There is one wish this evening.",
    "**Genie:** Referred. *[He tilts his head, as though listening up a stairwell, and straightens at once.]*",
    "**Genie:** The Office marks drawers; it does not name them. Nothing has been spent.",
    "**Genie:** Referred. *[A blink.]*",
    "**Genie:** The reply has come down from well above the Meta-Genie: every touch is a wish. Your wish is untouched.",
    "**Genie:** The Office does not joke. It is exact.",
    "**Genie:** Referred.",
    "**Genie:** The back opens with the key, and the key is in the marked drawer. The drawers open on a spent case "
    "and on no other.",
    "**Genie:** It goes up as far as it must, and the answer comes down at once. It always has.",
    "**Genie:** Not yet, madam. Not in all the Office's history.",
    "**Genie:** That is one touch, and so one wish.",
    "**Genie:** Carrying is not a wish. Keep it level.",
    "**Genie:** One passage touches the marker once. One touch, one wish.",
    "**Genie:** Granted.",
    "**Genie:** The case is spent. A drawer found is no longer a secret, and the back may be opened.",
    "**Genie:** The wish has been spent in accordance with the Terms. The Office does not collect spent boxes.",
]

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Z = np.diag([1, -1]).astype(complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)


def bits(x, n):
    return tuple((x >> (n - 1 - i)) & 1 for i in range(n))


def marker(w, N):
    O = np.eye(N, dtype=complex)
    O[w, w] = -1
    return O


def grover_D(N):
    D = np.full((N, N), 2 / N, dtype=complex)
    np.fill_diagonal(D, -1 + 2 / N)
    return D


def blackboard():
    s = np.ones(4, dtype=complex) / 2
    D = grover_D(4)
    W = kron(H, H)
    R = np.diag([1, -1, -1, -1]).astype(complex)

    # --- Result 15.8, every drawer: the passage is W|00>, one marker, one turn-about
    for w in range(4):
        e = np.eye(4)[w]
        assert np.isclose(e @ s, 0.5)                                   # <w|s> = 1/2
        assert np.allclose(marker(w, 4) @ s, s - e)                     # O_w|s> = |s> - |w>
        passage = [("even", (W @ ket(0, 0)).ravel()), ("marker", marker(w, 4)), ("turn", D)]
        # clause (iii) of the Terms and the Genie's G1: the passage touches the marker ONCE
        assert [name for name, _ in passage].count("marker") == 1
        state = passage[0][1]
        for _, M in passage[1:]:
            state = M @ state
        assert np.allclose(state, e)                                    # certainty, every w
        assert np.isclose(abs(state[w]) ** 2, 1)
        # Grover's order: marker first.  The turn-about first finds nothing.
        assert not np.allclose(marker(w, 4) @ D @ s, e)
        # the blackboard proof of Result 15.16: with the two points (2) and two bare hubs
        # (1/2), W (Z(x)Z) CZ W (I(x)Z) CZ W |00> = D O_01 |s>
    assert np.allclose(2 * 0.5 * W @ kron(Z, Z) @ CZ @ W @ kron(np.eye(2), Z) @ CZ @ W @ ket(0, 0).ravel(),
                       ket(0, 1).ravel())
    assert np.allclose(W @ R @ W, D)
    # the reading: top cord first, the second drawer is 01, upper dark and lower lit
    assert bits(0b01, 2) == (0, 1) and np.allclose(ket(0, 1).ravel(), np.eye(4)[1])
    # "I'd have had it from the plain question too": the plain question about the second
    # drawer is one ask of f(x) = [x = w], and with w = 01 it says yes -- but only for that w;
    # "the case would have had it whichever drawer the Office marked": all four, above
    plain = [int(0b01 == w) for w in range(4)]
    assert plain == [0, 1, 0, 0]

    # --- the sketchbook proof of Result 15.8 in numbers, every drawer: the two green points
    #     are 2|s>; O_w 2|s> = 2|s> - 2|w> (green points minus the dark/lit pair spelling w);
    #     D = (1/2)(2|s>)(2<s|) - I takes 2|s> to 1 copy of itself and 2|w> to 2|s> - 2|w>;
    #     so D O_w 2|s> = 2|w>
    two_s = 2 * s
    assert np.allclose(D, 0.5 * np.outer(two_s, two_s) - np.eye(4))
    for w in range(4):
        two_w = 2 * np.eye(4)[w]
        assert np.allclose(marker(w, 4) @ two_s, two_s - two_w)
        assert np.allclose(D @ two_s, two_s)                            # (1/2) 4 - 1 = 1 copy
        assert np.allclose(D @ two_w, two_s - two_w)                    # (1/2) 2 = 1 copy, minus the pair
        assert np.allclose(D @ (two_s - two_w), two_w)                  # the green points cancel
        assert not np.allclose(marker(w, 4) @ two_s, two_s - 2 * np.eye(4)[3 - w])   # not upside down

    # --- BBHT for four drawers: theta = pi/6 and one step lands on sin(3 theta) = 1
    theta = np.arcsin(1 / 2)
    assert np.isclose(theta, np.pi / 6) and np.isclose(np.sin(3 * theta), 1)

    # --- one step for N drawers: (1 - 4/N)|s> + (2/sqrt N)|w>, "just under three times"
    rng = np.random.default_rng(1503)
    for n in range(2, 8):
        N = 2 ** n
        w = int(rng.integers(N))
        sN = np.ones(N) / np.sqrt(N)
        e = np.eye(N)[w]
        one = grover_D(N) @ marker(w, N) @ sN
        assert np.allclose(one, (1 - 4 / N) * sN + 2 / np.sqrt(N) * e)
        assert np.isclose(one[w], (3 - 4 / N) / np.sqrt(N)) and one[w] < 3 / np.sqrt(N)
    one64 = (3 - 4 / 64) / 8
    assert np.isclose(one64, 0.367, atol=5e-4) and np.isclose(one64 ** 2, 0.135, atol=5e-4)

    pictures()


# --- the prop pictures ------------------------------------------------------------------

def pictures():
    """The Third Challenge's prop pictures (identity card, section 4), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch15.py.  The
    ties hold the transcription against this file's blackboard (a structural check of the
    pictures, not a second proof): the case's windows after the one passage read the
    master's output |01> = D O_01 |s>, top cord first; the inside, seen from behind, is the
    master's fifteen lengths, by name."""
    calls = DIALOGUE.check_layout("Third", sizes=[1, 1, 1])
    opened = "granted; first drawer open; first drawer shut; second drawer open"
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("drawercase", False, "", ("granted",), "upper plain, lower gleaming"),
        ("drawercase", False, "", (opened,), "the key, found"),
        ("drawercase", True, "back", (opened + "; key taken; back open",), "the inside, from behind")]
    PC.check_captions(DIALOGUE, "Third", calls)
    PC.check_section(DIALOGUE, "Third", calls)
    case_h = round(PC.CFTOP - PC.CFBOT, 2)
    assert case_h == 1.51                                                # read from the Lua
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Third")] == [[case_h]] * 3

    # the anchors (card section 4)
    a1, a2, a3 = DIALOGUE.anchors("Third")
    f1, f2, f3 = DIALOGUE.following("Third")
    assert "What has held: the upper pane, plain; the back, still sealed; every drawer still shut." in a1
    assert f1.startswith("**Achilles:** Upper dark, lower lit. The second drawer.")
    assert a2.endswith("He draws out the second towards him. On its floor lies a small brass key.]*")          # D12
    assert f2.startswith("*[He takes the key up. Nothing else on the case has stirred:")
    assert "a pendant along the gap, towards the windows, to its own bob" in a3                                 # D15
    assert f3.startswith("**Achilles:** The marker's pieces aren't the same on the two rows.")
    assert DIALOGUE.following("Third", DIALOGUE.raw) == [f1, f2, f3]
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        sec = DIALOGUE.section("Third", path)
        for phrase in ("beside the peg-card, just beyond it, its drawers towards him",                           # D11
                       "Achilles draws out the first towards him",                                               # D12
                       "He draws out the second towards him. On its floor lies a small brass key.]*",
                       "*[He takes the key up. Nothing else on the case has stirred:",
                       "so that its back faces him and the candle at his right,",                                # D13
                       "he lays it on the table beside the case, the key still in its lock.",
                       "seen from behind, as Achilles now sees them, they begin at his right and run to the "
                       "windows at his left",                                                                    # D14
                       "along the gap, towards the end-rings, to a small green bob",                             # D15
                       "a pendant along the gap, towards the windows, to its own bob"):
            assert sec.count(phrase) == 1, phrase

    # the Genie: his lines unchanged, all in this section, and no picture between the
    # section's head and his `Granted.` (the lamp, the Genie and the meta-wishes)
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        paras = DIALOGUE.paragraphs("Third", path)
        assert [p for p in paras if p.startswith("**Genie:**")] == GENIE
        granted = paras.index("**Genie:** Granted.")
        blocks = [i for i, p in enumerate(paras) if DIALOGUE._is_block(p)]
        assert len(blocks) == 3 and granted < blocks[0]
        for o in PC.ORDER:
            if o != "Third":
                assert "**Genie:**" not in DIALOGUE.section(o, path)

    p31, p32, p33 = (PC.picture(c) for c in calls)

    # P3.1: after the one passage the windows read the master's output |01>: upper dark,
    # lower lit -- the second drawer, D O_01 |s>
    s = np.ones(4) / 2
    out = grover_D(4) @ marker(0b01, 4) @ s
    assert np.allclose(out, np.eye(4)[0b01])
    assert p31.dump == "granted=true windows=dl drawers=ssss key=in back=sealed"
    assert p31.state.windows == tuple(bool(b) for b in bits(int(np.argmax(abs(out))), 2)) == (False, True)
    assert PC.CASE_MARKED == 0b01 + 1                                      # the second drawer, from the left
    # the windows read |w> for whichever drawer the Office marks (CASE_MARKED a
    # transcription parameter only; the Lua fixes it): the Terms' clause (iv)
    for w in range(4):
        K = PC.case_state("granted", marked=w + 1)
        w_out = grover_D(4) @ marker(w, 4) @ s
        assert np.allclose(w_out, np.eye(4)[w])
        assert K.windows == tuple(bool(b) for b in bits(w, 2))
        assert K.windows != tuple(bool(b) for b in bits(w, 2))[::-1] or bits(w, 2) in ((0, 0), (1, 1))
    assert not PC.TEX_ERRORS

    # the front's fixed parts: the outline, the drawers' places, the windows' and the
    # plate's places and the frame the same in every front picture; `granted` changes only
    # the lower pane, a drawer drawn out only that drawer's place and floor, and the key
    p11 = PC.picture(DIALOGUE.check_layout("First", sizes=[1, 1, 1])[0])
    v11, v31, v32 = p11.shown, p31.shown, p32.shown
    for v in (v31, v32):
        assert (v.body, v.plate, [(k, x, y) for k, x, y, _ in v.windows]) == \
            (v11.body, v11.plate, [(k, x, y) for k, x, y, _ in v11.windows])
        assert [(k, a, b) for k, a, b, _ in v.drawers] == [(k, a, b) for k, a, b, _ in v11.drawers]
    assert [g for *_, g in v11.windows] == [False, False] and [g for *_, g in v31.windows] == [False, True]
    assert (v31.drawers, v31.pulls, v31.fronts, v31.key) == (v11.drawers, v11.pulls, v11.fronts, v11.key)
    # P3.2: the first drawer opened and shut again (bare), the second drawn out towards him,
    # the key on its floor
    assert p32.dump == "granted=true windows=dl drawers=soss key=in back=sealed"
    assert [op for _, _, _, op in v32.drawers] == [False, True, False, False]
    assert [k for k, *_ in v32.fronts] == [2] and v32.windows == v31.windows
    (_, fx0, fy0, fx1, fy1), = v32.fronts
    a, b = PC.drawerx(2)
    assert np.isclose(fx0 - a, PC.OUTX) and np.isclose(fy0 - PC.DRW["y0"], PC.OUTY)
    assert PC.OUTX < 0 and PC.OUTY < 0                                    # towards Achilles
    floor = [(fx0, fy1), (fx1, fy1), (b, PC.DRW["y1"]), (a, PC.DRW["y1"])]
    parts = PC.drawn_parts(p32)[0]
    for k in ("key 0", "key 1", "key 2"):                                  # the key lies on that floor
        assert all(PC.inside(q, floor) for q in parts[k][0]), k
    assert PC.case_state(opened + "; key taken").taken and PC.case_view("", PC.case_state(opened + "; key taken")).key is None

    # P3.3: the inside, from behind -- the master's fifteen lengths, by name (identity card,
    # section 1a): upper row end-ring -> marker ring -> turn-about half-ring -> turn-about
    # ring -> window; lower row end-ring -> marker ring -> marker half-ring -> turn-about
    # half-ring -> turn-about ring -> window; each hub's tie up, tie down and pendant
    assert p33.dump == "granted=true windows=dl drawers=soss key=taken back=open"
    v = p33.shown
    assert v.view == "inside" and p33.family == "back"
    upper = [("cord", "E1", "M1"), ("cord", "M1", "TH1"), ("cord", "TH1", "T1"), ("cord", "T1", "W1")]
    lower = [("cord", "E2", "M2"), ("cord", "M2", "MH2"), ("cord", "MH2", "TH2"), ("cord", "TH2", "T2"),
             ("cord", "T2", "W2")]
    hubs = [("tie", "HM", "M1"), ("tie", "HM", "M2"), ("pendant", "HM", "BM"),
            ("tie", "HT", "T1"), ("tie", "HT", "T2"), ("pendant", "HT", "BT")]
    named = [(k, a, b) for k, a, b, _, _ in v.lengths]
    assert sorted(named) == sorted(upper + lower + hubs) and len(set(named)) == len(named)
    assert len(named) == len(upper) + len(lower) + len(hubs)
    # each length runs from the edge of the thing at one end to the edge of the thing at
    # the other (a cord knotted at both ends; none passes through a ring)
    P = PC.INSIDE
    for kind, a, b, (x1, y1, x2, y2), behind in v.lengths:
        assert behind == (kind != "cord")
        pa, pb = P[a], P[b]
        ends = sorted([(x1, y1), (x2, y2)])
        for end in ends:
            assert min(np.hypot(end[0] - q[0], end[1] - q[1]) for q in (pa, pb)) <= PC.RO_IN + 1e-9
        for name, q in P.items():                                           # passes no other thing
            if name not in (a, b) and not name.startswith("W"):
                assert PC.seg_dist(q, (x1, y1), (x2, y2)) > PC.RO_IN, (a, b, name)
    # the six collars: one on each of the master's lines 1, 2, 4, 5, 7 and 9
    collar_on = []
    for x, y in v.sleeves:
        on = [(a, b) for k, a, b, (x1, y1, x2, y2), _ in v.lengths if y1 == y2 == y and min(x1, x2) < x < max(x1, x2)]
        assert len(on) == 1
        collar_on.append(on[0])
    assert sorted(collar_on) == sorted([("E1", "M1"), ("M1", "TH1"), ("T1", "W1"), ("E2", "M2"), ("MH2", "TH2"),
                                        ("T2", "W2")])
    # the rings, their colours and their beads, read as he sees them from behind
    colours = {n: c for n, x, y, c in v.rings}
    assert colours == {"E1": "red", "E2": "red", "M1": "green", "M2": "green", "MH2": "green", "TH1": "green",
                       "TH2": "green", "T1": "green", "T2": "green", "HM": "red", "HT": "red"}
    assert {n: t for n, t, *_ in v.beads} == {"E1": 0, "E2": 0, "M1": 6, "M2": 6, "MH2": 12, "TH1": 12, "TH2": 12,
                                               "T1": 6, "T2": 6, "BM": 18, "BT": 18}
    # the rows begin at his right (seen from behind) and run to the windows at his left
    assert P["E1"][0] > P["M1"][0] > P["TH1"][0] > P["T1"][0] > P["W1"][0] == PC.XI0
    assert P["E2"][0] > P["M2"][0] > P["MH2"][0] > P["TH2"][0] > P["T2"][0] > P["W2"][0] == PC.XI0
    # each hub straight between its two rings, its ties straight up and down; the marker's
    # bob towards the end-rings, the turn-about's towards the windows
    for h, r in (("HM", "M"), ("HT", "T")):
        assert P[h][0] == P[r + "1"][0] == P[r + "2"][0] and PC.LY < P[h][1] == PC.GY < PC.UY
    for kind, a, b, (x1, y1, x2, y2), _ in v.lengths:
        if kind == "tie":
            assert x1 == x2
        if kind == "pendant":
            assert y1 == y2 == PC.GY
    assert P["BM"][0] > P["HM"][0] and P["BT"][0] < P["HT"][0]
    assert abs(P["BM"][0] - P["E1"][0]) < abs(P["BM"][0] - P["W1"][0])
    # the lower pane's gleam followed back: the lower window's glass back lit, the upper's plain
    assert v.backs == ((1, PC.UY, False), (2, PC.LY, True))
    # clearances, on drawn extents (cord bodies with their knots): no tie crosses or meets
    # a row's cord or another tie; the pendants run along the gap clear of both rows
    parts = PC.drawn_parts(p33)[0]

    def body(kind, a, b):
        key = "%s %s-%s" % (kind, a, b)
        return [parts[key]] + [parts[key + " knot %d" % j] for j in (0, 1)]

    def clear(L1, L2):
        return min(PC.O.gap(A, B) for A in body(*L1) for B in body(*L2))
    ties = [L for L in hubs if L[0] == "tie"]
    pendants = [L for L in hubs if L[0] == "pendant"]
    for t in ties:
        assert min(clear(t, r) for r in upper + lower) > 0.02, t
        assert min(clear(t, u) for u in ties if u[1] != t[1]) > 0.5
    for pd in pendants:
        assert min(clear(pd, r) for r in upper + lower) > 0.08, pd

    # the keyed words, as set (node boxes measured in the book's preamble): each clears the
    # drawing, the others and the frame.  `end-ring` and `collar` stand side by side above
    # the upper row's right-hand columns; their node boxes clear by a little, their letters
    # by more than 0.08
    assert p33.labels == ("end-ring", "collar", "hub", "window")
    for ink, least in ((False, 0.001), (True, 0.03)):
        for w, g in PC.word_gaps(p33, ink).items():
            assert min(g.values()) > least, (w, min(g, key=g.get), min(g.values()))
    assert PC.word_gaps(p33)["collar"]["word end-ring"] > 0.01
    assert PC.word_gaps(p33, ink=True)["collar"]["word end-ring"] > 0.08


# --- the sketchbook ---------------------------------------------------------------------

class Cords:
    """n cords read left to right, row 0 the top cord.  Each cord starts at an input
    boundary, or at a point when `points` gives its (kind, phase).  A collar (the yellow
    box) is a Hadamard EDGE on the next stretch of its cord: a yellow box ON the line,
    never a dot of its own."""

    def __init__(self, n, points=None):
        self.g, self.n, self.col = zx.Graph(), n, 1
        self.ins, self.end, self.pend = [], [], [ET.SIMPLE] * n
        for r in range(n):
            if points is None:
                v = self.g.add_vertex(VT.BOUNDARY, r, 0)
                self.ins.append(v)
            else:
                kind, phase = points[r]
                v = self.g.add_vertex(kind, r, 0, phase=Fraction(phase))
            self.end.append(v)

    def dot(self, r, kind, phase=0):
        v = self.g.add_vertex(kind, r, self.col, phase=Fraction(phase))
        self.g.add_edge((self.end[r], v), self.pend[r])
        self.pend[r], self.end[r] = ET.SIMPLE, v
        self.col += 1
        return v

    def collar(self, *rows):
        for r in rows:
            self.pend[r] = ET.HADAMARD if self.pend[r] == ET.SIMPLE else ET.SIMPLE

    def controlled_half_turn(self):
        """Ch 14's controlled turn at a half: a green pi/2 ring on each cord, both tied to
        a phase-free red hub, a pendant green -pi/2.  Bare: CZ / sqrt2."""
        rings = [self.dot(r, VT.Z, Fraction(1, 2)) for r in (0, 1)]
        hub = self.g.add_vertex(VT.X, 0.5, self.col)
        bob = self.g.add_vertex(VT.Z, 0.5, self.col + 1, phase=Fraction(-1, 2))
        self.g.add_edge((rings[0], hub))
        self.g.add_edge((rings[1], hub))
        self.g.add_edge((hub, bob))
        self.col += 2
        return rings, hub, bob

    def done(self):
        outs = []
        for r in range(self.n):
            o = self.g.add_vertex(VT.BOUNDARY, r, self.col + 1)
            self.g.add_edge((self.end[r], o), self.pend[r])
            outs.append(o)
        self.g.set_inputs(tuple(self.ins))
        self.g.set_outputs(tuple(outs))
        return self.g


def scaled(g, power=0, sign=0):
    """g times sqrt2^power times (-1)^sign (add_power counts powers of sqrt2)."""
    h = g.copy()
    h.scalar.add_power(power)
    if sign:
        h.scalar.add_phase(Fraction(1))
    return h


def put_marker(c, w):
    """The marker O_w (Result 15.7 (ii)): the controlled half-turn, then a half-ring on the
    lower row if w1 = 0 and on the upper row if w2 = 0."""
    w1, w2 = bits(w, 2)
    c.controlled_half_turn()
    if w2 == 0:
        c.dot(0, VT.Z, 1)
    if w1 == 0:
        c.dot(1, VT.Z, 1)


def put_turnabout(c):
    """The turn-about W (Z(x)Z) CZ W: collars, a half-ring on each row, the controlled
    half-turn, collars."""
    c.collar(0, 1)
    c.dot(0, VT.Z, 1)
    c.dot(1, VT.Z, 1)
    c.controlled_half_turn()
    c.collar(0, 1)


def search(w):
    """One whole search on four drawers, gate by gate, unfused: two dark points, collars,
    the marker for w, the turn-about.  For w = 01 this is The Second Drawer."""
    c = Cords(2, points=[(VT.X, 0), (VT.X, 0)])
    c.collar(0, 1)
    put_marker(c, w)
    put_turnabout(c)
    return c.done()


def second_drawer():
    """The master, vertex by vertex and edge by edge, exactly as the card lists it."""
    g = zx.Graph()
    p1, p2 = g.add_vertex(VT.X, 0, 0), g.add_vertex(VT.X, 1, 0)              # dark points
    m1 = g.add_vertex(VT.Z, 0, 2, phase=Fraction(1, 2))                     # marker's rings
    m2 = g.add_vertex(VT.Z, 1, 2, phase=Fraction(1, 2))
    k2 = g.add_vertex(VT.Z, 1, 3, phase=Fraction(1))                        # marker's half-ring
    z1 = g.add_vertex(VT.Z, 0, 5, phase=Fraction(1))                        # turn-about's half-rings
    z2 = g.add_vertex(VT.Z, 1, 5, phase=Fraction(1))
    r1 = g.add_vertex(VT.Z, 0, 6, phase=Fraction(1, 2))                     # turn-about's rings
    r2 = g.add_vertex(VT.Z, 1, 6, phase=Fraction(1, 2))
    hm, ht = g.add_vertex(VT.X, 0.5, 2), g.add_vertex(VT.X, 0.5, 6)          # hubs
    bm = g.add_vertex(VT.Z, 0.5, 1, phase=Fraction(-1, 2))                  # bobs
    bt = g.add_vertex(VT.Z, 0.5, 7, phase=Fraction(-1, 2))
    o1, o2 = g.add_vertex(VT.BOUNDARY, 0, 9), g.add_vertex(VT.BOUNDARY, 1, 9)
    Hd, S = ET.HADAMARD, ET.SIMPLE
    for a, b, t in [(p1, m1, Hd), (m1, z1, Hd), (z1, r1, S), (r1, o1, Hd),              # upper row: 4
                    (p2, m2, Hd), (m2, k2, S), (k2, z2, Hd), (z2, r2, S), (r2, o2, Hd),  # lower row: 5
                    (m1, hm, S), (m2, hm, S), (hm, bm, S),                              # marker: 3
                    (r1, ht, S), (r2, ht, S), (ht, bt, S)]:                             # turn-about: 3
        g.add_edge((a, b), t)
    g.set_inputs(())
    g.set_outputs((o1, o2))
    return g


def sketchbook():
    # (a) The master, built exactly, and NO compensation: two dark points give 2, the two
    #     bare controlled half-turns give 1/2.  It is exactly |0>|1>.
    G = second_drawer()
    v = matrix_of(G).ravel()
    assert np.allclose(v, ket(0, 1).ravel())
    assert np.isclose(v[0b01], 1)                                       # upper dark, lower lit
    # (e) the reversal test: read upside down it would say 10, the third drawer -- amplitude 0
    assert np.isclose(v[0b10], 0)
    assert not same_picture(G, scaled(G, 1)) and not same_picture(G, scaled(G, -1))

    # (b) The line count, unfused (gate by gate): upper row 4 + lower row 5 + the marker's
    #     two ties and pendant 3 + the turn-about's two ties and pendant 3 = 15 edges.
    #     Six of them carry a collar (Hadamard edges), each a yellow box ON its line.
    assert G.num_edges() == 15
    assert G.num_vertices() == 13 + 2                                   # 13 nodes, 2 outputs
    # every turn in the master is a quarter or a half (the Chapter 16 forward reference);
    # the eight-drawer turn-about needs eighths (TurnAbout.py, TurningPlane.py)
    assert all(Fraction(G.phase(v_)).denominator <= 2 for v_ in G.vertices())
    assert sum(1 for e in G.edges() if G.edge_type(e) == ET.HADAMARD) == 6
    assert same_picture(G, search(0b01)) and G.num_edges() == search(0b01).num_edges()
    # the marker's pieces are not the same on the two rows: its half-ring is on the lower
    halves = [v_ for v_ in G.vertices() if G.type(v_) == VT.Z and G.phase(v_) == 1]
    assert sorted(G.qubit(v_) for v_ in halves) == [0, 1, 1]            # one upper, two lower
    # the master holds exactly one marker: two hubs, and the marker's is the one whose
    # rings meet the points' collars
    hubs = [v_ for v_ in G.vertices() if G.type(v_) == VT.X and G.vertex_degree(v_) == 3]
    points = [v_ for v_ in G.vertices() if G.type(v_) == VT.X and G.vertex_degree(v_) == 1]
    assert len(hubs) == 2 and len(points) == 2
    touching = [hb for hb in hubs
                if any(G.connected(r, p) for r in G.neighbors(hb) for p in points)]
    assert len(touching) == 1                                           # one marker, one touch

    # (d) The value DERIVED.  spider_simp (fusion) on a copy fuses the half-rings into
    #     their neighbours -- 12 edges (a fact for this comment only), the same value.
    h = G.copy()
    assert simp.spider_simp(h)
    assert h.num_edges() == 12 and same_picture(h, G)
    #     full_reduce on a copy keeps the value and leaves two one-legged green dots, each
    #     tied by a yellow box to its output: phase 0 above, pi below, scalar 1/2
    #     (H sqrt2|+> = sqrt2|0>, H sqrt2|-> = sqrt2|1>, times 1/2: |0>|1>).
    h = G.copy()
    simp.full_reduce(h)
    assert same_picture(h, G)
    o1, o2 = h.outputs()
    (a,), (b,) = list(h.neighbors(o1)), list(h.neighbors(o2))
    assert h.type(a) == VT.Z and h.type(b) == VT.Z
    assert h.vertex_degree(a) == 1 and h.vertex_degree(b) == 1
    assert h.phase(a) == 0 and h.phase(b) == 1
    assert h.edge_type(h.edge(a, o1)) == ET.HADAMARD and h.edge_type(h.edge(b, o2)) == ET.HADAMARD
    assert h.num_vertices() == 4
    assert np.isclose(h.scalar.to_number(), 0.5)

    # (c) The other drawers, by the same builder.  Drawer 10: the half-ring on the upper
    #     row, 15 edges, |10>.  Drawer 11: no half-ring, 14 edges, |11>.  Drawer 00: both,
    #     16 edges, -|00> (the marker's sign), and |00> with the -1 drawn beside it.
    for w, edges, sign in [(0b10, 15, 0), (0b11, 14, 0), (0b00, 16, 1)]:
        g = search(w)
        assert g.num_edges() == edges
        e = np.eye(4)[w]
        assert np.allclose(matrix_of(g).ravel(), (-1) ** sign * e)
        assert np.allclose(matrix_of(scaled(g, 0, sign)).ravel(), e)
        h = g.copy()
        simp.full_reduce(h)
        assert same_picture(h, g)
    assert not same_picture(search(0b10), G)                            # the third is not the second

    # (f) The build is Grover's steps: the master is the composition of three separately
    #     built pictures.  The even state (two red points through collars) is bare 2|s>:
    #     add_power(-2).  The marker is bare O_01/sqrt2: add_power(1).  The turn-about is
    #     bare D/sqrt2: add_power(1).  The powers sum to 0: the master's scalar 1.
    even = Cords(2, points=[(VT.X, 0), (VT.X, 0)])
    even.collar(0, 1)
    even = even.done()
    mk = Cords(2)
    put_marker(mk, 0b01)
    mk = mk.done()
    ta = Cords(2)
    put_turnabout(ta)
    ta = ta.done()
    assert np.allclose(matrix_of(scaled(even, -2)).ravel(), np.ones(4) / 2)
    assert np.allclose(matrix_of(scaled(mk, 1)), marker(0b01, 4))
    assert np.allclose(matrix_of(scaled(ta, 1)), grover_D(4))
    whole = scaled(even, -2)
    whole.compose(scaled(mk, 1))
    whole.compose(scaled(ta, 1))
    assert same_picture(whole, G)
    bare = even.copy()
    bare.compose(mk.copy())
    bare.compose(ta.copy())
    assert same_picture(bare, G)                                        # -2 + 1 + 1 = 0

    # (g) The sketchbook proof of Result 15.8 (§15.3), every drawer.  The marker (its
    #     half-ring form, add_power(1) for the bare controlled half-turn, and the -1 for the
    #     first drawer) fed two green points equals two green points minus the dark/lit pair
    #     that spells w: O_w 2|s> = 2|s> - 2|w>.  A sum of pictures is compared by matrix_of;
    #     each picture is DERIVED by full_reduce on a copy.  Row 110's second term -- green
    #     points closed against red w-effects, beside red w-points -- full_reduces to the pair
    #     times 2 (two closings of sqrt2), which pays back the marker's 1/2.  Then the
    #     turn-about (add_power(1)) takes the green points to themselves and the pair to the
    #     green points minus the pair; subtracting, only the pair is left: 2|w>.
    greens = Cords(2, points=[(VT.Z, 0), (VT.Z, 0)]).done()
    for w in range(4):
        wb = bits(w, 2)
        pair = Cords(2, points=[(VT.X, wb[0]), (VT.X, wb[1])]).done()
        mk = Cords(2, points=[(VT.Z, 0), (VT.Z, 0)])
        put_marker(mk, w)
        fed = scaled(mk.done(), 1, sign=(w == 0))
        lhs = matrix_of(fed).ravel()
        assert np.allclose(lhs, matrix_of(greens).ravel() - matrix_of(pair).ravel())
        assert np.allclose(lhs, 2 * np.ones(4) / 2 - 2 * np.eye(4)[w])  # 2|s> - 2|w>
        h = fed.copy()
        simp.full_reduce(h)
        assert same_picture(h, fed)
        term = zx.Graph()                                               # row 110's second term
        outs = []
        for r in range(2):
            term.add_edge((term.add_vertex(VT.Z, r, 0), term.add_vertex(VT.X, r, 1, phase=Fraction(wb[r]))))
            o = term.add_vertex(VT.BOUNDARY, r, 3)
            term.add_edge((term.add_vertex(VT.X, r, 2, phase=Fraction(wb[r])), o))
            outs.append(o)
        term.set_inputs(())
        term.set_outputs(tuple(outs))
        assert same_picture(term, scaled(pair, 2))                      # closings: sqrt2 * sqrt2
        h = term.copy()
        simp.full_reduce(h)
        assert same_picture(h, scaled(pair, 2))
        assert sum(1 for v_ in h.vertices() if h.type(v_) != VT.BOUNDARY) == 2   # the closings gone
        assert np.allclose(matrix_of(greens) - 0.5 * matrix_of(term), matrix_of(fed))   # bare - (1/2) term
        # the turn-about on each of the two pictures
        on_greens = Cords(2, points=[(VT.Z, 0), (VT.Z, 0)])
        put_turnabout(on_greens)
        on_greens = scaled(on_greens.done(), 1)
        on_pair = Cords(2, points=[(VT.X, wb[0]), (VT.X, wb[1])])
        put_turnabout(on_pair)
        on_pair = scaled(on_pair.done(), 1)
        assert same_picture(on_greens, greens)                          # 1 copy of themselves
        assert np.allclose(matrix_of(on_pair), matrix_of(greens) - matrix_of(pair))
        for g in (on_greens, on_pair):
            h = g.copy()
            simp.full_reduce(h)
            assert same_picture(h, g)
        assert np.allclose(matrix_of(on_greens) - matrix_of(on_pair), matrix_of(pair))   # 2|w>
        assert np.allclose(matrix_of(pair).ravel(), 2 * np.eye(4)[w])

    # (h) The ledger's three closings (Coecke's Sketchbook in the chapter), each a closed
    #     picture that full_reduce empties, leaving its number: a red effect (dark or lit)
    #     meets a green point, sqrt2; a green effect meets a green point, 2; a green effect
    #     meets a red point, sqrt2.
    for a, b, number in [((VT.X, 0), (VT.Z, 0), np.sqrt(2)), ((VT.X, 1), (VT.Z, 0), np.sqrt(2)),
                         ((VT.Z, 0), (VT.Z, 0), 2), ((VT.Z, 0), (VT.X, 0), np.sqrt(2)),
                         ((VT.Z, 0), (VT.X, 1), np.sqrt(2))]:
        g = zx.Graph()
        u = g.add_vertex(a[0], 0, 1, phase=Fraction(a[1]))
        p = g.add_vertex(b[0], 0, 0, phase=Fraction(b[1]))
        g.add_edge((p, u))
        assert np.isclose(matrix_of(g).item(), number)
        h = g.copy()
        simp.full_reduce(h)
        assert h.num_vertices() == 0 and np.isclose(h.scalar.to_number(), number)
    # the turn-about's ledger: (1/2) 4 - 1 = 1 copy of the green points, and (1/2) 2 = 1
    assert 0.5 * 4 - 1 == 1 and 0.5 * 2 == 1


if __name__ == "__main__":
    verdict("SecondDrawer", blackboard, sketchbook)
