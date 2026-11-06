"""Challenge V: No Faster Hand  (the fair share; the optimality sketch)
=====================================================================
"Four touches, one hand: not enough. Five, one and a quarter hands: still short. Six, one and a half: enough. That's the six we found."  -- Achilles

Claim: for every method with T touches, changing the answer for one drawer w moves the final
state by at most twice the sum of that drawer's amplitudes at the touches,
|| phi^w - phi^0 || <= 2 sum_i alpha_{i,w} (Result 15.13, BBHT's Lemma 6(1)), also when a
touch is the query box with its flag in any state; the squares of each touch add to one, so
some drawer gets at most its fair share T/N and sum_i alpha_{i,x} <= T/sqrt N; a method
that finds every w with probability 1/2 needs T >= sqrt N/(4 sqrt2) - 1/2 (Result 15.14:
181 for 2^20 drawers); and for Grover's search on sixty-four drawers the bound is T/4
against the distance 1.323 from |s> to |w>, so five touches cannot reach and six can.
BBBV's Theorem 3.5 (Result 13.14) and BBHT's Theorem 7 (Result 15.15) are stated and cited
in the chapter, not proved; Theorem 7's figure is checked here as documentation only.

Translation exception (recorded on the identity card, deviation (p)): no picture proves a
bound.  As spec/book.md puts it, a diagram rewrite "does not prove complexity bounds,
optimality, or most security claims."  So sketchbook() computes the two processes -- The
Second Drawer (one touch, drawer 01 marked) and the same picture with no drawer marked --
by the rules and as tensors, and compares their distance with the blackboard's bound, which
the four-drawer search attains exactly.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch15 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

# each section's picture blocks and their sizes (identity card, section 4)
SIZES = {"First": [1, 1, 1], "Second": [1, 1, 1, 1], "Third": [1, 1, 1], "Fourth": [1, 1, 1, 1], "Fifth": [1]}

# the Old fragments the card's directions replaced: gone from both whole files (R7-Ch14)
NEGATIVE_ANCHORS = (
    "two bare cords, each ending at a green end-ring at rest",                     # D7
    "a small tick at half a notch beside the slits",                               # D4
    "ties both down to a bare red hub in the gap between the cords",               # D6
    "so that its back faces the candle,",                                          # D13
    "a half is set aside, off the cloth, by the candlestick",                      # D9
    "*[She takes a fresh sheet, draws the unmarked line",                          # D17
    "*[She lays the whole build again",                                            # D8
    "*[She draws a line across the sheet and a line straight up it",               # D16
    "He draws out the second. On its floor",                                       # D12
    "a pendant along the gap to its own bob",                                      # D15
    "from the hub a pendant lies out along the gap to a small green bob",          # D15
)


def haar(rng, d):
    """A Haar-random unitary: QR of a complex Gaussian matrix, phases fixed."""
    q, r = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def touch_marker(N, w, extra):
    """O_w on the n drawer cords, nothing on the `extra`-dimensional workspace."""
    O = np.eye(N, dtype=complex)
    O[w, w] = -1
    return np.kron(O, np.eye(extra))


def touch_query_box(N, w, extra):
    """The query box U_f, f(x) = [x = w], on the drawer cords and a flag, the flag being
    the first workspace cord (so extra is even): U_f - I = |w><w| (x) (X - I) (x) I."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    P = np.zeros((N, N)); P[w, w] = 1
    return np.eye(N * extra, dtype=complex) + np.kron(P, np.kron(X - np.eye(2), np.eye(extra // 2)))


def run(Vs, touch, start):
    """V_T touch V_{T-1} ... V_1 touch V_0 |start>; returns the final state and the states
    just before each touch."""
    psi = Vs[0] @ start
    before = []
    for V in Vs[1:]:
        before.append(psi)
        psi = V @ (touch @ psi)
    return psi, before


def grover_D(N):
    D = np.full((N, N), 2 / N, dtype=complex)
    np.fill_diagonal(D, -1 + 2 / N)
    return D


def blackboard():
    rng = np.random.default_rng(1505)

    # --- with no drawer marked the hand never moves: G^k |s> = |s>, every peg an eighth
    s64 = np.ones(64) / 8
    v = s64.copy()
    for _ in range(12):
        v = grover_D(64) @ v                                            # the marker is I
        assert np.allclose(v, s64)

    # --- Result 15.13 (BBHT Lemma 6(1)) and the fair share, on 200 random methods: Haar
    #     unitaries between the touches, N in {4, 8, 16}, T <= 5, a workspace beside the
    #     drawer cords; every w.  Half of them touch with the marker, half with the query
    #     box whose flag is in whatever state the random unitaries leave it.
    count = 0
    for trial in range(200):
        N = [4, 8, 16][trial % 3]
        T = 1 + trial % 5
        extra = 2
        d = N * extra
        Vs = [haar(rng, d) for _ in range(T + 1)]
        start = np.zeros(d, dtype=complex); start[0] = 1
        phi0, before = run(Vs, np.eye(d), start)                        # the unmarked run
        # alpha_{i,x}: the length of the part of psi_i in which the drawer cords hold x
        alpha = np.array([[np.linalg.norm(p.reshape(N, extra)[x]) for x in range(N)] for p in before])
        assert np.isclose(np.sum(alpha ** 2), T)                        # one whole per touch
        S = np.sum(alpha ** 2, axis=0)
        for w in range(N):
            touch = touch_marker(N, w, extra) if trial % 2 else touch_query_box(N, w, extra)
            phiw, _ = run(Vs, touch, start)
            dist = np.linalg.norm(phiw - phi0)
            assert dist <= 2 * np.sum(alpha[:, w]) + 1e-9               # Lemma 6(1)
            assert np.sum(alpha[:, w]) <= np.sqrt(T) * np.sqrt(S[w]) + 1e-12   # Cauchy-Schwarz
            count += 1
        least = int(np.argmin(S))
        assert S[least] <= T / N + 1e-12                                # the fair share
        assert np.sum(alpha[:, least]) <= T / np.sqrt(N) + 1e-12
        # the counting in Result 15.14's proof, for this method: few drawers asked much,
        # few drawers read much in the unmarked run, so at least N/2 drawers are neither
        read0 = np.array([np.linalg.norm(phi0.reshape(N, extra)[x]) ** 2 for x in range(N)])
        heavy = {x for x in range(N) if S[x] > 4 * T / N}
        loud = {x for x in range(N) if read0[x] >= 4 / N}
        assert len(heavy) <= N / 4 and len(loud) <= N / 4
        quiet = [x for x in range(N) if x not in heavy and x not in loud]
        assert len(quiet) >= N / 2
        for w in quiet:
            touch = touch_marker(N, w, extra) if trial % 2 else touch_query_box(N, w, extra)
            phiw, _ = run(Vs, touch, start)
            Pw = lambda phi: np.linalg.norm(phi.reshape(N, extra)[w])  # the reading names w
            dist = np.linalg.norm(phiw - phi0)
            assert dist >= Pw(phiw) - Pw(phi0) - 1e-12                  # reverse triangle
            assert Pw(phi0) < 2 / np.sqrt(N)
            assert dist <= 4 * T / np.sqrt(N) + 1e-9
    assert count == sum([4, 8, 16][t % 3] for t in range(200))
    # with the query box, U_f - I = |w><w| (x) (X - I) (x) I, and X - I at most doubles a length
    Xm = np.array([[0, 1], [1, 0]]) - np.eye(2)
    assert np.isclose(np.linalg.norm(Xm, 2), 2)

    # --- Result 15.14's bound, and Grover's search against it
    bound = lambda N: np.sqrt(N) / (4 * np.sqrt(2)) - 0.5
    assert int(np.ceil(bound(2 ** 20))) == 181                          # "at least 181"
    for n in range(2, 31):                                              # Grover meets it
        N = 2 ** n
        th = np.arcsin(1 / np.sqrt(N))
        half = int(np.ceil((np.pi / (4 * th) - 1) / 2))                 # smallest k for odds 1/2
        assert (2 * half + 1) * th >= np.pi / 4 > (2 * half - 1) * th
        assert half >= bound(N)
        assert np.sin((2 * half + 1) * th) ** 2 >= 0.5
    th = np.arcsin(2 ** -10)
    assert min(k for k in range(2000) if (2 * k + 1) * th >= np.pi / 4) == 402
    assert int(np.floor(np.pi / (4 * th))) == 804
    assert 100 <= 181 < 804                                            # "hundreds, not a handful"

    # --- Grover on sixty-four drawers: each touch spends 1/8 on the marked drawer
    N, th = 64, np.arcsin(1 / 8)
    s = np.ones(N) / 8
    w = 37
    e = np.eye(N)[w]
    gap = np.linalg.norm(s - e)
    assert np.isclose(gap, np.sqrt(2 - 2 * np.sin(th))) and np.isclose(gap, np.sqrt(7 / 4))
    assert np.isclose(gap, 1.323, atol=5e-4)
    v = s.copy()
    for T in range(1, 8):
        O = np.eye(N); O[w, w] = -1
        v = grover_D(N) @ O @ v
        dist = np.linalg.norm(v - s)                                    # the unmarked run stays at s
        assert np.isclose(dist, 2 * np.sin(T * th))
        assert dist <= 2 * T / 8 + 1e-12                                # the bound, T/4
    assert np.isclose(2 * np.sin(2 * th), 0.496, atol=5e-4) and np.isclose(2 * 2 / 8, 0.5)
    assert np.isclose(2 * np.sin(6 * th), 1.366, atol=5e-4) and np.isclose(2 * 6 / 8, 1.5)
    assert 4 * 0.25 < gap and 5 * 0.25 < gap <= 6 * 0.25               # five short, six enough
    # the rule laid tip to tip: ten and a half thumbs and a little over, a little under
    # one and a third hands (the hand is eight thumbs)
    assert 10.5 < 8 * gap < 8 * 4 / 3 and gap < 4 / 3
    # the fair share after two touches: some drawer has had at most 2/64; split evenly that
    # is 1/8 of height at each touch and half a hand at most; split unevenly, less
    assert np.isclose(2 * 2 * np.sqrt(1 / 64), 0.5)
    for _ in range(50):
        a1 = rng.uniform(0, np.sqrt(2 / 64))
        a2 = np.sqrt(2 / 64 - a1 ** 2)
        assert 2 * (a1 + a2) <= 0.5 + 1e-12

    # --- documentation only (Result 15.15, BBHT Theorem 7; stated and cited, not proved):
    #     Grover's steps to success 1/2 against floor(sin(pi/8) sqrt N); the ratio tends to
    #     (pi/8)/sin(pi/8) = 1.026, "a few percent"
    assert np.isclose((np.pi / 8) / np.sin(np.pi / 8), 1.026, atol=5e-4)
    ratios = []
    for n in (10, 20, 40):
        N = 2 ** n
        th = np.arcsin(1 / np.sqrt(N))
        half = int(np.ceil((np.pi / (4 * th) - 1) / 2))                 # smallest k: (2k+1)theta >= pi/4
        assert (2 * half + 1) * th >= np.pi / 4 > (2 * half - 1) * th
        assert half >= np.floor(np.sin(np.pi / 8) * np.sqrt(N))         # Grover above the bound
        ratios.append(half / np.floor(np.sin(np.pi / 8) * np.sqrt(N)))
    assert ratios[0] > ratios[1] > ratios[2]                            # 1.083, 1.028, 1.026
    assert np.isclose(ratios[2], (np.pi / 8) / np.sin(np.pi / 8), atol=1e-4)

    pictures()


# --- the prop pictures ------------------------------------------------------------------

def spoken(p):
    """A paragraph's speaker and words, inline directions stripped (None if not spoken)."""
    m = re.match(r"^\*\*(\w+):\*\*(.*)$", p, re.S)
    return (m.group(1), re.sub(r"\*\[.*?\]\*", "", m.group(2)).strip()) if m else None


def pictures():
    """The Fifth Challenge's prop picture (identity card, section 4) and the whole
    dialogue's picture rules, read through tools/dfp_props.py and the transcription
    tools/props_ch15.py.  The tie holds the transcription against this file's blackboard
    (a structural check, not a second proof): with no drawer marked the marker turns
    nothing over, and the turn-about fixes the even state, D|s> = |s>, so the hand on the
    even line does not move."""
    calls = DIALOGUE.check_layout("Fifth", sizes=SIZES["Fifth"])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("handsheet", False, "seven steps, back to the first dot, no drawer marked, the turn-about's fold",
         ("sixty-four drawers",), "unmarked, the hand stays")]
    PC.check_captions(DIALOGUE, "Fifth", calls)
    PC.check_section(DIALOGUE, "Fifth", calls)
    a1, = DIALOGUE.anchors("Fifth")
    f1, = DIALOGUE.following("Fifth")
    assert "Unfolded, there is no new hole: the prick is the dot." in a1
    assert f1.startswith("**Achilles:** It hasn't moved.")
    assert DIALOGUE.following("Fifth", DIALOGUE.raw) == [f1]
    p51, = (PC.picture(c) for c in calls)

    # P5.1: back at the first dot, no drawer marked, the turn-about's fold: the hand stays
    N = 64
    s = np.ones(N) / np.sqrt(N)
    assert np.allclose(grover_D(N) @ s, s)
    assert np.allclose(grover_D(N) @ np.eye(N) @ s, s)                    # the marker, nothing marked: I
    th = np.arcsin(1 / np.sqrt(N))
    assert p51.dump == ("drawers=64 marked=false angle=7.18 tip=1.00 "
                        "holes=7/-7/22/-22/36/-36/50/-50/65/-65/79/-79/93/-93/108")
    assert np.isclose(p51.state.a, th) and np.isclose(p51.state.a, p51.state.tilt)
    assert "tip=%.2f" % (8 * np.sin(th)) == "tip=1.00"
    # its holes are P4.4's, none added: the prick is the dot
    p44 = PC.picture(DIALOGUE.check_layout("Fourth", sizes=SIZES["Fourth"])[3])
    assert p51.state.holes == p44.state.holes and p51.shown.lines == p44.shown.lines
    assert any(np.isclose(h, th) for h in p51.state.holes)
    # and the marker, with no drawer marked, makes no fold: the Lua refuses one
    PC.TEX_ERRORS.clear()
    PC.sheet_state("no drawer marked, the marker's fold", "sixty-four drawers")
    assert [k for k, _ in PC.TEX_ERRORS] == ["sheet-nomark"]
    PC.TEX_ERRORS.clear()

    # --- the whole dialogue ---------------------------------------------------------------
    # every picture inside a challenge section, the five in order: as many as the
    # sections' blocks hold, none before the First Challenge, none after P5.1
    total = DIALOGUE.check_whole(PC.ORDER)
    assert total == sum(sum(v) for v in SIZES.values())
    every = []
    for o in PC.ORDER:
        cs = DIALOGUE.check_layout(o, sizes=SIZES[o])
        for c in cs:
            assert PC.place_ok(o, c), (o, c.text)                          # the forbidden places
            PC.picture(c)                                                  # no fault anywhere
        every += [(o, i, c) for i, c in enumerate(cs)]
    text = DIALOGUE.read(DIALOGUE.dialog)
    assert text.rindex(every[-1][2].text) > text.index("## Fifth Challenge")
    assert every[-1][2].text == calls[0].text
    # the keyed rule: the first picture of each kind keyed, and only those -- P1.1, P1.2,
    # P2.2, P3.3 and P4.1
    DIALOGUE.keyed_first(PC.ORDER, kind=PC.keyed_kind)
    assert [(o, i + 1) for o, i, c in every if c.keyed] == [("First", 1), ("First", 2), ("Second", 2),
                                                            ("Third", 3), ("Fourth", 1)]
    # a place a picture may not take: the case's front after the wish, in the First
    assert not PC.place_ok("First", every[7][2]) and not PC.place_ok("Fifth", every[0][2])

    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        whole = DIALOGUE.read(path)                                        # planning comment stripped
        for old in NEGATIVE_ANCHORS:
            assert old not in whole, old
        # the toccata (identity card, section 1c): the figure once in the Second and six
        # times in the Fourth, one Granted., the Genie's, in the Third; every Genie line in
        # the Third; three Referred. lines, the Genie's, each standing alone
        ok, figure, counts, granted, referred, outside, replies = PC.form_check(path.read_text(encoding="utf-8"))
        assert ok and figure == (("Achilles", "The marker turns it over."), ("Tortoise", "And we turn it about."))
        assert {c: n for c, n in counts.items() if n} == {"Second Challenge": 1, "Fourth Challenge": 6}
        assert granted == "Third Challenge" and outside == [] and replies == []
        assert referred == [("Genie", "Referred.")] * 3
        lines = [spoken(p) for p in re.split(r"\n[ \t]*\n", whole) if spoken(p)]
        assert [x for x in lines if x[1] == "Referred."] == [("Genie", "Referred.")] * 3
        # no picture between a figure's touch line and its turn line
        for o in ("Second", "Fourth"):
            paras = DIALOGUE.paragraphs(o, path)
            for i, p in enumerate(paras):
                if spoken(p) == ("Achilles", "The marker turns it over."):
                    j = next(k for k in range(i + 1, len(paras)) if spoken(paras[k]))
                    assert spoken(paras[j]) == ("Tortoise", "And we turn it about.")
                    assert not any(DIALOGUE._is_block(q) for q in paras[i:j])
        # no "Let Bob draw" this chapter, and the signal grep finds nothing (comments included)
        raw = path.read_text(encoding="utf-8")
        assert raw.count("Let Bob draw") == 0
        assert not re.search(r"(?i)(?<![\w])(fifteen|fifteenth|15)(?![\w])", raw)



# --- the sketchbook ---------------------------------------------------------------------

class Cords:
    """Two cords read left to right, row 0 the top cord, each starting at a dark point.
    A collar (the yellow box) is a Hadamard EDGE on the next stretch of its cord; two
    collars with nothing between them meet and cancel, and are drawn as one plain edge."""

    def __init__(self, points):
        self.g, self.col = zx.Graph(), 1
        self.end, self.pend = [], [ET.SIMPLE, ET.SIMPLE]
        for r, (kind, phase) in enumerate(points):
            self.end.append(self.g.add_vertex(kind, r, 0, phase=Fraction(phase)))

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
        """Green pi/2 rings on both cords, a phase-free red hub, a green -pi/2 pendant.
        Bare: CZ / sqrt2."""
        rings = [self.dot(r, VT.Z, Fraction(1, 2)) for r in (0, 1)]
        hub = self.g.add_vertex(VT.X, 0.5, self.col)
        bob = self.g.add_vertex(VT.Z, 0.5, self.col + 1, phase=Fraction(-1, 2))
        self.g.add_edges([(rings[0], hub), (rings[1], hub), (hub, bob)])
        self.col += 2

    def done(self):
        outs = []
        for r in (0, 1):
            o = self.g.add_vertex(VT.BOUNDARY, r, self.col + 1)
            self.g.add_edge((self.end[r], o), self.pend[r])
            outs.append(o)
        self.g.set_inputs(())
        self.g.set_outputs(tuple(outs))
        return self.g


def scaled(g, power):
    """g times sqrt2^power (add_power counts powers of sqrt2)."""
    h = g.copy()
    h.scalar.add_power(power)
    return h


def four_drawers(marked):
    """Two dark points, collars, the marker for drawer 01 if `marked` (the controlled
    half-turn, a half-ring on the lower cord), then the turn-about (collars, half-rings,
    the controlled half-turn, collars).  Marked: The Second Drawer.  Unmarked: the collars
    either side of the missing marker meet, one plain edge on each cord."""
    c = Cords([(VT.X, 0), (VT.X, 0)])
    c.collar(0, 1)
    if marked:
        c.controlled_half_turn()
        c.dot(1, VT.Z, 1)
    c.collar(0, 1)
    c.dot(0, VT.Z, 1)
    c.dot(1, VT.Z, 1)
    c.controlled_half_turn()
    c.collar(0, 1)
    return c.done()


def sketchbook():
    marked, unmarked = four_drawers(True), four_drawers(False)
    assert marked.num_edges() == 15                                     # The Second Drawer
    greens = Cords([(VT.Z, 0), (VT.Z, 0)]).done()                        # two green points: 2|s>

    # The unmarked run is The Second Drawer with the marker taken out: its two rings, its
    # half-ring, its hub and its bob go (five dots, six edges), and the collars either side
    # meet and cancel.
    assert unmarked.num_vertices() == marked.num_vertices() - 5
    assert unmarked.num_edges() == marked.num_edges() - 6
    # The turn-about's middle (a half-ring on each cord, the controlled half-turn) is
    # R/sqrt2, and R = (1/2) dark effects beside dark points - bare cords (row 112).  A dark
    # effect closes on a dark point to 2, so two dark points come through as (1/2) 4 - 1 = 1
    # copy of themselves: the middle fed two dark points is 2^{-1/2} times them (DERIVED by
    # full_reduce); the last collars turn them green (colour change, named rule).
    R = np.diag([1, -1, -1, -1]).astype(complex)
    mid = zx.Graph()
    ins = [mid.add_vertex(VT.BOUNDARY, r, 0) for r in (0, 1)]
    outs = [mid.add_vertex(VT.BOUNDARY, r, 9) for r in (0, 1)]
    halves = [mid.add_vertex(VT.Z, r, 1, phase=Fraction(1)) for r in (0, 1)]
    rings = [mid.add_vertex(VT.Z, r, 2, phase=Fraction(1, 2)) for r in (0, 1)]
    hub, bob = mid.add_vertex(VT.X, 0.5, 2), mid.add_vertex(VT.Z, 0.5, 3, phase=Fraction(-1, 2))
    mid.add_edges([(ins[r], halves[r]) for r in (0, 1)] + [(halves[r], rings[r]) for r in (0, 1)]
                  + [(rings[r], outs[r]) for r in (0, 1)] + [(rings[0], hub), (rings[1], hub), (hub, bob)])
    mid.set_inputs(tuple(ins)); mid.set_outputs(tuple(outs))
    assert np.allclose(matrix_of(mid), R / np.sqrt(2))
    dk = zx.Graph()                                                     # dark effects beside dark points
    ins = [dk.add_vertex(VT.BOUNDARY, r, 0) for r in (0, 1)]
    outs = [dk.add_vertex(VT.BOUNDARY, r, 3) for r in (0, 1)]
    for r in (0, 1):
        dk.add_edge((ins[r], dk.add_vertex(VT.X, r, 1)))
        dk.add_edge((dk.add_vertex(VT.X, r, 2), outs[r]))
    dk.set_inputs(tuple(ins)); dk.set_outputs(tuple(outs))
    assert np.allclose(0.5 * matrix_of(dk) - np.eye(4), R)
    closing = zx.Graph()
    closing.add_edge((closing.add_vertex(VT.X, 0, 0), closing.add_vertex(VT.X, 0, 1)))
    h = closing.copy()
    simp.full_reduce(h)
    assert np.isclose(matrix_of(closing).item(), 2)
    assert h.num_vertices() == 0 and np.isclose(h.scalar.to_number(), 2)
    darks = Cords([(VT.X, 0), (VT.X, 0)]).done()
    fed = darks.copy()
    fed.compose(mid)
    assert same_picture(fed, scaled(darks, -1))                         # 1 copy, times 2^{-1/2}
    h = fed.copy()
    simp.full_reduce(h)
    assert same_picture(h, scaled(darks, -1))
    boxed = Cords([(VT.X, 0), (VT.X, 0)])
    boxed.collar(0, 1)
    boxed = boxed.done()
    for v in [v for v in boxed.vertices() if boxed.type(v) == VT.X]:
        assert simp.check_color_change(boxed, v) and simp.color_change(boxed, v)
    assert same_picture(boxed, greens)                                  # the last collars: green
    assert same_picture(unmarked, scaled(greens, -1))                   # (1/sqrt2) two green points

    # The unmarked picture: two points (2) and one bare hub (2^{-1/2}) -- bare it is
    # exactly sqrt2 |s>.  So (unmarked, add_power(-1)) = (two green points, add_power(-2)):
    # both sides are exactly |s>.  The hand never leaves the even line.
    s = np.ones(4) / 2
    assert np.allclose(matrix_of(unmarked).ravel(), np.sqrt(2) * s)
    assert same_picture(scaled(unmarked, -1), scaled(greens, -2))
    assert not same_picture(unmarked, scaled(greens, -2))               # the sqrt2 is drawn
    collars = [e for e in unmarked.edges() if unmarked.edge_type(e) == ET.HADAMARD]
    assert len(collars) == 2                                            # only the last two remain:
    points = [p for p in unmarked.vertices() if unmarked.type(p) == VT.X and unmarked.vertex_degree(p) == 1]
    for p in points:                                                    # the met ones: plain edges
        (nb,) = list(unmarked.neighbors(p))
        assert unmarked.edge_type(unmarked.edge(p, nb)) == ET.SIMPLE and unmarked.phase(nb) == 1
    # DERIVED: full_reduce on a copy leaves two green points and the value
    h = scaled(unmarked, -1)
    simp.full_reduce(h)
    assert same_picture(h, scaled(greens, -2))
    ends = [next(iter(h.neighbors(o))) for o in h.outputs()]
    assert all(h.type(v) == VT.Z and h.phase(v) == 0 and h.vertex_degree(v) == 1 for v in ends)
    assert all(h.edge_type(h.edge(v, o)) == ET.SIMPLE for v, o in zip(ends, h.outputs()))
    # The marked picture is |01> exactly; DERIVED by full_reduce on a copy
    h = marked.copy()
    simp.full_reduce(h)
    assert same_picture(h, marked)
    phi_w = matrix_of(marked).ravel()
    phi_0 = matrix_of(scaled(unmarked, -1)).ravel()
    assert np.allclose(phi_w, [0, 1, 0, 0]) and np.allclose(phi_0, s)

    # The two processes differ by exactly 1, and BBHT's bound for one touch on four drawers
    # is 2 * alpha, alpha = 1/2 the marked drawer's height in the unmarked run at the touch
    # (the even state).  The bound is attained.
    assert np.allclose(phi_w - phi_0, 0.5 * np.array([-1, 1, -1, -1]))  # the difference
    dist = np.linalg.norm(phi_w - phi_0)
    alpha = abs(s[0b01])
    assert np.isclose(dist, 1) and np.isclose(2 * alpha, 1)
    assert np.isclose(dist, 2 * alpha)
    # and by Result 15.14's count, one touch can never do this for many drawers:
    # 2 * 1/sqrt N < 1 once N > 4
    assert all(2 / np.sqrt(N) < 1 for N in (8, 16, 64))


if __name__ == "__main__":
    verdict("NoFasterHand", blackboard, sketchbook)
