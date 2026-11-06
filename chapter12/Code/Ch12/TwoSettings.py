"""Challenge I: Same Setting, Same Answer  (conjugate bases; BB84 correctness)
============================================================================
"...Oh. There's no crosswise answer in an upright glint, waiting to be read out."  -- Achilles

Claim: Alice sends her bit x at setting r as H^r|x> (r = 0 plain, 1 cross); Bob
reads at setting t and gets y.  Then P(y | x, r, t) = [y = x] when r = t and 1/2
when r != t, where [y = x] is 1 if y = x and 0 otherwise.  As doubled processes on
the classical wires, with the sender E_r = (H^r (x) H^r) delta and the reader
R_t = delta^dagger (H^t (x) H^t), exactly R_t E_r = I (r = t) and 1/2 J (r != t),
J the all-ones 2x2 matrix: matching settings fuse into a wire, mismatched settings
disconnect into noise.  The plain and cross bases are conjugate, and BB84's
circular basis c1 = (1, i)/sqrt2, c2 = (i, 1)/sqrt2 is conjugate to both.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import BELL, CUP, I2, kron, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch12 as PC


# --- the chapter's matrices, written out (kept local: the mathematics stays here) ---
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
DELTA = np.outer(ket(0, 0), ket(0)) + np.outer(ket(1, 1), ket(1))   # the copier, 4x2
J = np.ones((2, 2))
PLUS, MINUS = H @ ket(0), H @ ket(1)
C1 = np.array([1, 1j]) / np.sqrt(2)                   # BB84's circular basis, as printed
C2 = np.array([1j, 1]) / np.sqrt(2)


def Hp(r):
    """H^r for a setting bit r: H^0 = I, H^1 = H."""
    return H if r else I2


def sender(r):
    """E_r = (H^r (x) H^r) delta: one classical wire in, the doubled pair (q, qbar) out."""
    return kron(Hp(r), Hp(r)) @ DELTA


def reader(t):
    """R_t = delta^dagger (H^t (x) H^t): Chapter 9's read-spider, the doubled pair in."""
    return DELTA.conj().T @ kron(Hp(t), Hp(t))


def eff(k, s):
    """The reading effect <k| H^s at setting s, as a 1x2 row."""
    return ket(k).reshape(1, 2) @ Hp(s)


def iv(b):
    """The Iverson bracket [b]: 1 if b is true, else 0."""
    return 1.0 if b else 0.0


def blackboard():
    rng = np.random.default_rng(1201)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))      # cup unnormalized
    assert np.allclose(BELL, CUP / np.sqrt(2))
    assert np.allclose(H, H.T) and np.allclose(H, H.conj())  # H real, symmetric ...
    assert np.allclose(H @ H, I2)                           # ... and its own inverse

    # all sixteen amplitudes <y| H^t H^r |x>, and their squares
    for r in (0, 1):
        for t in (0, 1):
            for x in (0, 1):
                for y in (0, 1):
                    amp = ket(y) @ Hp(t) @ Hp(r) @ ket(x)
                    if r == t:
                        assert np.isclose(amp, iv(y == x))
                    else:
                        assert np.isclose(amp, (-1) ** (x * y) / np.sqrt(2))
                    P = abs(amp) ** 2
                    assert np.isclose(P, iv(y == x) if r == t else 0.5)
    # the sign the blackboard keeps and then squares away: only (x, y) = (1, 1) is negative
    assert (ket(1) @ H @ ket(1)).real < 0 and (ket(0) @ H @ ket(1)).real > 0

    # Result 12.1: conjugate bases.  |<i|H|j>|^2 = 1/2 for all four pairs ...
    for i in (0, 1):
        for j in (0, 1):
            assert np.isclose(abs(ket(i) @ H @ ket(j)) ** 2, 0.5)
    # ... and the circular basis, NON-REAL, orthonormal, conjugate to plain and to cross
    assert not np.allclose(C1, C1.conj()) and not np.allclose(C2, C2.conj())
    assert np.isclose(np.vdot(C1, C1), 1) and np.isclose(np.vdot(C1, C2), 0)
    for c in (C1, C2):
        for b in (ket(0), ket(1), PLUS, MINUS):
            assert np.isclose(abs(np.vdot(b, c)) ** 2, 0.5)
    # the four cross overlaps exactly as the proof prints them
    assert np.isclose(np.vdot(PLUS, C1), (1 + 1j) / 2) and np.isclose(np.vdot(MINUS, C1), (1 - 1j) / 2)
    assert np.isclose(np.vdot(PLUS, C2), (1 + 1j) / 2) and np.isclose(np.vdot(MINUS, C2), (1j - 1) / 2)
    # Dirac's Blackboard: NO real unit vector (cos th, sin th) is conjugate to both bases;
    # its distance from conjugacy is (|cos 2th| + |sin 2th|)/2 >= 1/2 for every th
    th = np.linspace(0, 2 * np.pi, 20001)
    miss = abs(np.cos(th) ** 2 - 0.5) + abs(0.5 * (np.cos(th) + np.sin(th)) ** 2 - 0.5)
    assert miss.min() > 0.49

    # Result 12.2, second form: the doubled round, exactly
    for r in (0, 1):
        for t in (0, 1):
            RE = reader(t) @ sender(r)
            assert np.allclose(RE, I2 if r == t else J / 2)
            assert not np.allclose(RE, J / 2 if r == t else I2)
    # the reader turns a doubled pure state psi (x) psi* into the vector of probabilities;
    # tested on a NON-REAL psi, where forgetting the mirror's conjugation is wrong
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())
    for t in (0, 1):
        probs = np.array([abs(ket(y) @ Hp(t) @ psi) ** 2 for y in (0, 1)])
        assert np.allclose((reader(t) @ np.kron(psi, psi.conj())).ravel(), probs)
        assert not np.allclose((reader(t) @ np.kron(psi, psi)).ravel(), probs)

    # over uniform random x, r, t: matched fraction 1/2, raw error 1/4
    matched = sum(0.125 for x in (0, 1) for r in (0, 1) for t in (0, 1) if r == t)
    err = sum(0.125 * (reader(t) @ sender(r))[1 - x, x] for x in (0, 1) for r in (0, 1) for t in (0, 1))
    assert np.isclose(matched, 0.5) and np.isclose(err, 0.25)

    # --- the table: the lamp's twin glint, read at both ends (the E91 form, IV) ---
    def twin(rA, rT):
        """p[kA, kB]: Phi+ read by Achilles at rA and the Tortoise at rT."""
        return np.array([[abs(kron(ket(a).reshape(1, 2) @ Hp(rA), ket(b).reshape(1, 2) @ Hp(rT))
                              @ BELL)[0] ** 2 for b in (0, 1)] for a in (0, 1)])
    for rA in (0, 1):
        for rT in (0, 1):
            p = twin(rA, rT)
            assert np.allclose(p, np.eye(2) / 2 if rA == rT else np.full((2, 2), 0.25))
    # Achilles's guess: an upright yes asked crosswise "leans towards yes", three times in
    # four.  It does not: at mixed settings the calls agree exactly half the time
    agree_mixed = np.trace(twin(0, 1))
    assert np.isclose(agree_mixed, 0.5) and not np.isclose(agree_mixed, 0.75)

    # the ledger of I (yes = 0 = white, no = 1 = black): every round was possible, and
    # at a matching setting the calls sounded in unison
    I_rounds = [(0, 0, 0, 0), (0, 0, 1, 1), (0, 0, 0, 0),   # both upright
                (1, 1, 1, 1), (1, 1, 0, 0),                 # both crosswise
                (0, 1, 0, 1), (0, 1, 0, 0), (0, 1, 1, 0)]   # Achilles upright, Tortoise cross
    for rA, rT, a, b in I_rounds:
        assert twin(rA, rT)[a, b] > 0
        if rA == rT:
            assert a == b
    # the rows after the mixed rounds are struck: white, black, white, black, white, both
    rowA = [a for rA, rT, a, b in I_rounds if rA == rT]
    rowT = [b for rA, rT, a, b in I_rounds if rA == rT]
    assert rowA == rowT == [0, 1, 0, 1, 0]

    # Alice's lock-book (the trials, before the relay house): 200 glints, 96 matched,
    # every one as sent (the model: P(y != x | matched) = 0, exactly); 104 mismatched,
    # yes on 53 and no on 51.  The counts are only plausible draws from the model's fair
    # coins (within 3 standard deviations of binomial(n, 1/2)); nothing more is claimed.
    assert 96 + 104 == 200 and 53 + 51 == 104
    for k, n in ((96, 200), (53, 104)):
        assert abs(k - n / 2) <= 3 * np.sqrt(n / 4)
    assert all(np.isclose((reader(r) @ sender(r))[1 - x, x], 0) for r in (0, 1) for x in (0, 1))
    pictures()


# --- the prop pictures (dialog.md: \glinttray, \keycloth; tools/props_ch12.py) -----------

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = "...Oh. There's no crosswise answer in an upright glint, waiting to be read out."
COLOUR = {"green": 0, "red": 1}          # the cloth's rings: green upright (plain), red crosswise
PIECE = {"yes": 0, "no": 1}              # a window, and the piece laid for it: white yes, black no


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    calls = DIALOGUE.check_layout("First", sizes=[1, 2, 1])
    p11, p12, p13, p14 = calls
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("glinttray", True, "flash yes yes", ("as set out",), "a window lit at each end"),
        ("keycloth", True, "", ("green to green",), "sending upright, reading upright"),
        ("keycloth", False, "slide together, lift off", ("green to green",), "slid together and lifted off"),
        ("keycloth", False, "draw apart", ("green to red",), "green and red drawn apart")]
    PC.check_captions(DIALOGUE, "First", calls)
    PC.check_section(DIALOGUE, "First", calls)
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("First")] == [[PC.TRAY_FRAME], [1.24, 1.24], [1.24]]

    # the anchors (card section 4), in dialog.md; the raw file's blocks are the same
    a1, a2, a3 = DIALOGUE.anchors("First")
    f1, f2, f3 = DIALOGUE.following("First")
    assert "one window flashes — once — and is dark again" in a1 and f1.startswith("**Achilles:** Yes.")
    assert "Alice's short cord running straight on into Bob's" in a2
    assert "beneath it and running the same way" in a2                          # D4
    assert f2.startswith("**Achilles:** Straight through.")
    assert "only it is joined to nothing Alice put in" in a3 and "gathers the cord into her pouch" in a3   # D5
    assert f3.startswith("**Achilles:** Tied to nothing she put in.")
    assert DIALOGUE.following("First", DIALOGUE.raw) == [f1, f2, f3]
    section = DIALOGUE.section("First")
    assert "His row will run from there along the edge towards the middle" in section   # D3
    assert "behind the lamp" in section                                                  # D2

    t11, c12, c13, c14 = (PC.picture(c) for c in calls)
    # P1.1: the tray as set out, a window lit at each end (the reading, card section 3b)
    assert t11.dump == ("sleeve closed; tell's key in the box; side rod in the box; his []; hers []; "
                        "tell's []; flash his yes hers yes; bell still")
    assert t11.labels == ("lamp", "key", "rod", "reflection", "sleeve", "his row", "her row")
    shown = {t.name: t for t in t11.shown.black}
    assert set(shown) == {"his key", "her key", "lamp", "sleeve"}
    assert (shown["his key"].lit, shown["her key"].lit) == ("yes", "yes")
    assert all(t.reflected for t in shown.values())            # each lit window's reflection lit with it
    assert (shown["his key"].x, shown["lamp"].x, shown["sleeve"].x, shown["her key"].x) == \
        (PC.XHIS, PC.XLAMP, PC.XSLEEVE, PC.XHERS)
    assert t11.shown.rows == {"his": (), "hers": (), "tell": ()} and t11.shown.bell == (PC.XBELL, PC.YBELL, 0.0)
    assert PC.tray_state("flash yes yes; ring the bell", "as set out").flash is None   # a flash is a moment

    # P1.2 | P1.3: the round, then slid together and lifted off -- one cord edge to edge
    assert c12.dump == ("scene=rounds round1=green-green joined fork=false | cols=7 rows=2 y=0..1 near=false "
                        "folded=false; 1:ring green 0 @0.6,0; 2:ring green 0 @5.4,0; cord 1-2 [0.6,0 5.4,0]; "
                        "cord 1-2 [0.6,0 1.1,1 4.9,1 5.4,0]; cord loose-1 [-0.45,0 0.6,0]; "
                        "cord loose-2 [6.45,0 5.4,0]")
    assert c12.labels == ("Alice's", "Bob's", "glint's", "mirror")
    assert [(w, side) for w, side, *_ in c12.own] == [("Alice's", "top"), ("Bob's", "top"), ("glint's", "top"),
                                                      ("mirror", "bottom")]  # off the linen, in the band
    PC.check_band_words(c12)
    glint, mirror = [d["pts"] for d in c12.cloth["cords"] if d["a"] == 1 and d["b"] == 2]
    assert all(y == 0 for _, y in glint) and max(y for _, y in mirror) == 1   # the mirror just below ...
    assert glint[0][0] < glint[-1][0] and mirror[0][0] < mirror[-1][0]      # ... running the same way
    assert c13.dump == ("scene=rounds round1=green-green lifted fork=false | cols=7 rows=2 y=0..1 near=false "
                        "folded=false; cord loose-loose [-0.45,0 6.45,0]")
    assert c14.dump == ("scene=rounds round1=green-red apart fork=false | cols=7 rows=2 y=0..1 near=false "
                        "folded=false; 1:ring green 0 @0.1,0; 2:ring red 0 @5.9,0; cord loose-1 [-0.45,0 0.1,0]; "
                        "cord loose-2 [6.45,0 5.9,0]; cord loose-loose [1,-0.12 1.9,0.1 2.9,-0.16 3.9,0.06 4.9,-0.1]; "
                        "cord loose-loose [4.9,0.9 4.1,0.74 3.1,0.98 2.1,0.7 1.1,0.92]")
    # the fixed parts: under draw apart the rings MOVE out (0.6 -> 0.1, 5.4 -> 5.9), the motion
    joined = PC.keycloth("", "green to red")
    xs = lambda pic: [r["x"] for r in pic.cloth["things"]]
    assert xs(joined) == [PC.AX, PC.BX] and xs(c14) == [0.1, 5.9]
    assert xs(c14)[0] < xs(joined)[0] and xs(c14)[1] > xs(joined)[1]
    # the clearance, on drawn extents: the slack cords touch neither ring
    rings, cords = PC.cloth_drawn(c14)
    slack = [(pts, reach) for kind, a, b, pts, reach in cords if (a, b) == ("loose", "loose")]
    assert len(slack) == 2
    for _, _, X, Y, rr in rings:
        for pts, reach in slack:
            assert PC.point_to_polyline((X, Y), pts) - rr - reach > 0

    # the ties to the blackboard (a structural check, stated as such: the cloth's joins
    # against the doubled round's matrix; the pictures prove nothing by themselves)
    # P1.1 lights the same window at both ends; the pair read at matching settings never
    # disagrees (plain and cross), so a lit pair (yes, yes) is a possible round
    for r in (0, 1):
        p = np.array([[abs((kron(eff(a, r), eff(b, r)) @ BELL)[0]) ** 2 for b in (0, 1)] for a in (0, 1)])
        assert np.isclose(p[0, 1] + p[1, 0], 0)
    lit = t11.state.flash
    assert lit["his"] == lit["hers"] and abs((kron(eff(PIECE[lit["his"]], 0), eff(PIECE[lit["hers"]], 0)) @ BELL)[0]) > 0
    # a lifted round joins the left edge to the right edge: the doubled round is a wire, R_r E_r = I
    for c in ("green", "red"):
        lifted = PC.keycloth("slide together, lift off", f"{c} to {c}")
        assert any({("left", 0), ("right", 0)} <= comp for comp in PC.cloth_links(lifted.cloth))
        r = COLOUR[c]
        assert np.allclose(reader(r) @ sender(r), I2) and np.linalg.matrix_rank(reader(r) @ sender(r)) == 2
    # drawn apart: each ring holds only its own short cord, nothing joins the two ends, and
    # R_1 E_0 = 1/2 J of rank 1 -- Bob's reading is independent of what Alice sent
    comps = PC.cloth_links(c14.cloth)
    assert not any({("left", 0), ("right", 0)} <= comp for comp in comps)
    assert {1, ("left", 0)} in comps and {2, ("right", 0)} in comps
    RE = reader(COLOUR[c14.state["rounds"][0]["b"]]) @ sender(COLOUR[c14.state["rounds"][0]["a"]])
    assert np.allclose(RE, J / 2) and np.linalg.matrix_rank(RE) == 1
    # and before the slide, a round's ends are joined through both rings
    assert any({1, 2, ("left", 0), ("right", 0)} <= comp for comp in PC.cloth_links(c12.cloth))


# --- pictures (hand-built doubled diagrams; pyzx 0.10.6 refuses ground vertices) -----

def round_picture(r, t, rings=False):
    """Alice's sender at setting r, then Bob's reader at setting t, on the doubled line.
    Vertices: input x (classical), the sender (green, one leg in, two out), two phase-0
    green identity dots on the lines q and qbar (PyZX keeps one edge per pair of dots, so
    the two parallel lines between sender and reader go through them), the reader (Ch 9's
    read-spider), output y (classical).  A cross setting puts a Hadamard EDGE (exact H,
    one line with a box on it) on both quantum legs: sender-side at r = 1, reader-side at
    t = 1.  With rings=True a cross sender or reader is drawn as the Tortoise's cloth draws
    it: a RED ring, its Hadamards moved by colour change onto its classical leg.
    No scalar compensation: every dot here is exact."""
    g = zx.Graph()
    V = g.add_vertex
    x = V(VT.BOUNDARY, 0, 0)
    s = V(VT.X if (rings and r) else VT.Z, 0, 1)
    q, qb = V(VT.Z, -0.5, 2), V(VT.Z, 0.5, 2)
    rd = V(VT.X if (rings and t) else VT.Z, 0, 3)
    y = V(VT.BOUNDARY, 0, 4)
    hr = ET.HADAMARD if (r and not rings) else ET.SIMPLE
    ht = ET.HADAMARD if (t and not rings) else ET.SIMPLE
    g.add_edge((x, s), ET.HADAMARD if (rings and r) else ET.SIMPLE)
    g.add_edge((s, q), hr); g.add_edge((s, qb), hr)
    g.add_edge((q, rd), ht); g.add_edge((qb, rd), ht)
    g.add_edge((rd, y), ET.HADAMARD if (rings and t) else ET.SIMPLE)
    g.set_inputs((x,)); g.set_outputs((y,))
    return g


def noise_picture():
    """1/2 times a green effect on x beside a green point on y: 1/2 J.
    Each one-legged phase-free green dot is |0>+|1> (or <0|+<1|): add_power(-2) is the 1/2."""
    g = zx.Graph()
    x, e = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.Z, 0, 1)
    p, y = g.add_vertex(VT.Z, 0, 2), g.add_vertex(VT.BOUNDARY, 0, 3)
    g.add_edge((x, e)); g.add_edge((p, y))
    g.set_inputs((x,)); g.set_outputs((y,))
    g.scalar.add_power(-2)
    return g


def wire_picture():
    g = zx.Graph()
    x, y = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 0, 1)
    g.add_edge((x, y))
    g.set_inputs((x,)); g.set_outputs((y,))
    return g


def connected(g, u, v):
    seen, todo = {u}, [u]
    while todo:
        w = todo.pop()
        for n in g.neighbors(w):
            if n not in seen:
                seen.add(n); todo.append(n)
    return v in seen


def sketchbook():
    # convention: a Hadamard edge is exactly H (a phase-free two-legged green dot is a wire)
    g = zx.Graph()
    a, z, b = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.Z, 0, 1), g.add_vertex(VT.BOUNDARY, 0, 2)
    g.add_edge((a, z)); g.add_edge((z, b), ET.HADAMARD)
    g.set_inputs((a,)); g.set_outputs((b,))
    assert np.allclose(matrix_of(g), H)

    # Result 12.1 as a picture: 1/2 x (red point a.pi joined to green effect b.pi) is
    # (-1)^(ab)/sqrt2 -- the same modulus for all four choices, only the sign moves
    for a in (0, 1):
        for b in (0, 1):
            g = zx.Graph()
            u, v = g.add_vertex(VT.X, 0, 0, phase=Fraction(a)), g.add_vertex(VT.Z, 0, 1, phase=Fraction(b))
            g.add_edge((u, v))
            g.set_inputs(()); g.set_outputs(())
            g.scalar.add_power(-2)                              # the 1/sqrt2 . 1/sqrt2 in front
            assert np.isclose(matrix_of(g)[0, 0], (-1) ** (a * b) / np.sqrt(2))
            h = g.copy(); simp.full_reduce(h)                   # derived: a closed picture is its number
            assert h.num_vertices() == 0
            assert np.isclose(h.scalar.to_number(), (-1) ** (a * b) / np.sqrt(2))

    for r in (0, 1):
        for t in (0, 1):
            g = round_picture(r, t)
            # (a) the VALUE, with no compensation: I or 1/2 J, exactly as on the blackboard
            want = I2 if r == t else J / 2
            assert np.allclose(matrix_of(g), want)
            assert np.allclose(matrix_of(g), reader(t) @ sender(r))
            assert same_picture(g, wire_picture() if r == t else noise_picture())
            assert not same_picture(g, noise_picture() if r == t else wire_picture())
            # the cloth's rings (a cross ring drawn red) are the same picture
            assert same_picture(round_picture(r, t, rings=True), g)
            # (b) DERIVATION: full_reduce, scalar kept
            h = g.copy(); simp.full_reduce(h)
            assert same_picture(h, g)
            (x,), (y,) = h.inputs(), h.outputs()
            if r == t:
                # matching settings fuse into a wire: the boxes cancel in pairs, the two
                # dots fuse along two lines, the self-loop goes -- a bare classical wire
                assert h.num_vertices() == 2 and h.num_edges() == 1 and connected(h, x, y)
            else:
                # mismatched settings disconnect into noise: colour change, then Hopf;
                # a green effect on x, a green point on y, and a cut through empty paper
                assert h.num_vertices() == 4 and h.num_edges() == 2 and not connected(h, x, y)
                assert same_picture(h, noise_picture())


if __name__ == "__main__":
    verdict("TwoSettings", blackboard, sketchbook)
