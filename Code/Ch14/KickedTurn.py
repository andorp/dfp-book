"""Challenge II: The Round Turned Whole  (eigenrounds; kickback; the controlled turn; one listener's blur)
=====================================================================================================
"...Oh. The round hasn't gone anywhere. Every ring has moved, and the round lies where it lay, only turned, all of it at once, a quarter round."  -- Achilles

Claim: the rounds |psi_k> = r^{-1/2} sum_s e^{-2 pi i k s / r} |a^s mod N> are orthonormal and
U_a |psi_k> = e^{2 pi i k / r} |psi_k> (Result 14.3); a controlled box kicks an eigenphase
back onto its control, Lambda(U)(|chi>|u>) = (Z(theta)|chi>)|u> (Result 14.4); the bare
controlled turn (green alpha/2 on each wire, both tied to a phase-free red hub, a green
-alpha/2 pendant) is exactly CP(alpha)/sqrt2 and symmetric in its wires (Result 14.5); a
basis point passes through it leaving its turn on the other wire, the bare left side being
2^{-1/2} times the right (Result 14.6); and one listener reads dark with probability
(1 + cos 2 pi phi)/2, so a quarter and three quarters give the same odds (Result 14.7,
Kitaev's Eq. (17)).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from math import gcd

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import I2, ket, kron, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch14 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def Zph(theta):
    """The one-wire phase box Z(theta) = diag(1, e^{i theta})."""
    return np.diag([1, np.exp(1j * theta)])


def CP(alpha):
    """The controlled turn CP(alpha) = diag(1, 1, 1, e^{i alpha})."""
    return np.diag([1, 1, 1, np.exp(1j * alpha)])


def controlled(U):
    """Lambda(U) = |0><0| (x) I + |1><1| (x) U, the control on the TOP wire."""
    d = U.shape[0]
    return np.kron(np.diag([1, 0]), np.eye(d)) + np.kron(np.diag([0, 1]), U)


# --- the blackboard ---------------------------------------------------------------------

def tick_box(a, N, wires):
    d = 2 ** wires
    M = np.zeros((d, d), dtype=complex)
    for x in range(d):
        M[(a * x) % N if x < N else x, x] = 1
    return M


def rounds(a, N, wires, r):
    """|psi_k> for k = 0..r-1 on the register, normalized."""
    orbit = [pow(a, s, N) for s in range(r)]
    out = []
    for k in range(r):
        v = np.zeros(2 ** wires, dtype=complex)
        for s, h in enumerate(orbit):
            v[h] = np.exp(-2j * np.pi * k * s / r) / np.sqrt(r)
        out.append(v)
    return out, orbit


def unitary_with_eigenphase(rng, phi, d=3):
    """A random complex, non-diagonal unitary with the random complex eigenvector u
    and U u = e^{2 pi i phi} u."""
    V, _ = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    others = rng.random(d - 1)
    U = V @ np.diag(np.exp(2j * np.pi * np.r_[phi, others])) @ V.conj().T
    return U, V[:, 0]


def listener_odds(U, u):
    """Kitaev's measuring operator, his Eq. (16): a Hadamard gate on each side of
    Lambda(U), the control starting at |0>.  Returns P(read 0)."""
    d = U.shape[0]
    Hc = np.kron(H, np.eye(d))
    out = Hc @ controlled(U) @ Hc @ np.kron(ket(0), u)
    return float(np.linalg.norm(out[:d]) ** 2)


def blackboard():
    rng = np.random.default_rng(1402)

    # --- Result 14.3 on the dial: N = 15, a = 7, r = 4, on four wires
    N, a, r = 15, 7, 4
    U7 = tick_box(a, N, 4)
    psis, orbit = rounds(a, N, 4, r)
    assert orbit == [1, 7, 4, 13]
    G = np.array([[p.conj() @ q for q in psis] for p in psis])
    assert np.allclose(G, np.eye(r))                                    # orthonormal
    for k, psi in enumerate(psis):
        assert np.allclose(U7 @ psi, np.exp(2j * np.pi * k / r) * psi)
        if k in (1, 3):                                                 # the sign is real
            assert not np.allclose(U7 @ psi, np.exp(-2j * np.pi * k / r) * psi)
    # the chapter's worked round: |psi_1> = (|1> - i|7> - |4> + i|13>)/2, and U_7 psi_1 = i psi_1
    psi1 = (ket(0, 0, 0, 1) - 1j * ket(0, 1, 1, 1) - ket(0, 1, 0, 0) + 1j * ket(1, 1, 0, 1)) / 2
    assert np.allclose(psis[1], psi1) and np.allclose(U7 @ psi1, 1j * psi1)
    # a turn is invisible by itself: psi and i psi give the same odds for every reading
    assert np.allclose(np.abs(U7 @ psi1) ** 2, np.abs(psi1) ** 2)

    # --- the round ticked, as the dialogue lays it: beads at rest at one, a quarter back at
    #     seven, a half at four, three quarters back at thirteen.  Every ring is carried to
    #     the hand's next hour with its bead; then at each hour the bead stands a quarter ON
    #     from the bead that stood there before.  Four ticks and every bead is back.
    bead = {h: np.exp(-2j * np.pi * s / 4) for s, h in enumerate(orbit)}
    assert np.isclose(bead[1], 1) and np.isclose(bead[7], -1j)
    assert np.isclose(bead[4], -1) and np.isclose(bead[13], 1j)
    ticked = {(a * h) % N: b for h, b in bead.items()}                   # ring carried on
    assert set(ticked) == set(bead)                                     # no hour empty
    for h in bead:
        assert np.isclose(ticked[h], 1j * bead[h])                      # a quarter on
    assert np.allclose(np.linalg.matrix_power(U7, 4) @ psis[1], psis[1])

    # Result 14.3 in general: random (N, a) pairs
    for N2, a2 in [(21, 2), (33, 5), (35, 3), (39, 7)]:
        assert gcd(N2, a2) == 1
        r2 = next(t for t in range(1, N2) if pow(a2, t, N2) == 1)
        w = int(np.ceil(np.log2(N2)))
        Ua = tick_box(a2, N2, w)
        ps, _ = rounds(a2, N2, w, r2)
        assert np.allclose(np.array([[p.conj() @ q for q in ps] for p in ps]), np.eye(r2))
        for k, p in enumerate(ps):
            assert np.allclose(Ua @ p, np.exp(2j * np.pi * k / r2) * p)

    # --- Result 14.4, kickback of an eigenphase, with a random complex non-diagonal U
    for _ in range(3):
        phi = rng.random()
        U, u = unitary_with_eigenphase(rng, phi)
        assert not np.allclose(U, np.diag(np.diag(U)))                  # not diagonal
        assert not np.allclose(u, u.conj())                             # non-real eigenvector
        assert np.allclose(U @ u, np.exp(2j * np.pi * phi) * u)
        theta = 2 * np.pi * phi
        L = controlled(U)
        for c in (0, 1):
            assert np.allclose(L @ np.kron(ket(c), u), np.exp(1j * c * theta) * np.kron(ket(c), u))
        chi = random_state(rng)
        assert not np.allclose(chi, chi.conj())
        assert np.allclose(L @ np.kron(chi, u), np.kron(Zph(theta) @ chi, u))
        assert not np.allclose(L @ np.kron(chi, u), np.kron(Zph(-theta) @ chi, u))   # not -phi
        # the register is unchanged; the CONTROL is what turns
        assert not np.allclose(L @ np.kron(chi, u), np.kron(chi, u))
        # Lambda(U^{2^m}) kicks back e^{2 pi i c 2^m phi}
        for m in range(4):
            Lm = controlled(np.linalg.matrix_power(U, 2 ** m))
            assert np.allclose(Lm @ np.kron(chi, u), np.kron(Zph(2 ** m * theta) @ chi, u))

    # --- the controlled turn on the blackboard: Lambda(Z(alpha)) = CP(alpha), symmetric
    SW = kron(I2, I2)[[0, 2, 1, 3]]                                     # the crossing
    for alpha in (np.pi / 2, 3 * np.pi / 7, rng.random() * 2 * np.pi):
        assert np.allclose(controlled(Zph(alpha)), CP(alpha))
        assert np.allclose(SW @ CP(alpha) @ SW, CP(alpha))
        # the basis-point lemma, Result 14.6, on either wire, psi random complex
        psi = random_state(rng)
        for c in (0, 1):
            assert np.allclose(CP(alpha) @ np.kron(ket(c), psi), np.kron(ket(c), Zph(c * alpha) @ psi))
            assert np.allclose(CP(alpha) @ np.kron(psi, ket(c)), np.kron(Zph(c * alpha) @ psi, ket(c)))
        # a controlled box is not symmetric in general
        U, _ = unitary_with_eigenphase(rng, 0.3, d=2)
        assert not np.allclose(SW @ controlled(U) @ SW, controlled(U))

    # --- Result 14.7, one listener (Kitaev's Eq. (17))
    odds = {}
    for phi in (0, 0.25, 0.5, 0.75):
        U, u = unitary_with_eigenphase(rng, phi)
        odds[phi] = listener_odds(U, u)
        assert np.isclose(odds[phi], (1 + np.cos(2 * np.pi * phi)) / 2)
    assert np.isclose(odds[0], 1) and np.isclose(odds[0.5], 0)
    assert np.isclose(odds[0.25], 0.5) and np.isclose(odds[0.75], 0.5)     # the trap
    # the chapter's worked amplitudes for a quarter: (|0> + i|1>)/sqrt2 before the last
    # Hadamard, ((1+i)|0> + (1-i)|1>)/2 after it
    assert np.allclose(Zph(np.pi / 2) @ H @ ket(0), (ket(0) + 1j * ket(1)) / np.sqrt(2))
    assert np.allclose(H @ Zph(np.pi / 2) @ H @ ket(0), ((1 + 1j) * ket(0) + (1 - 1j) * ket(1)) / 2)
    for _ in range(5):
        phi = rng.random()
        U, u = unitary_with_eigenphase(rng, phi)
        p = listener_odds(U, u)
        assert np.isclose(p, (1 + np.cos(2 * np.pi * phi)) / 2)
        U2, u2 = unitary_with_eigenphase(rng, 1 - phi)
        assert np.isclose(listener_odds(U2, u2), p)                     # phi and 1-phi alike
        # Kitaev's remedy: iU in place of U turns the cosine into minus the sine
        assert np.isclose(listener_odds(1j * U, u), (1 - np.sin(2 * np.pi * phi)) / 2)
        if not np.isclose(np.sin(2 * np.pi * phi), 0):
            assert not np.isclose(listener_odds(1j * U, u), listener_odds(1j * U2, u2))

    pictures()


# --- the pictures in the dialogue ----------------------------------------------------------

def bead_phase(t):
    """A bead turned t twenty-fourths clockwise, as a phase."""
    return np.exp(2j * np.pi * t / 24)


def cloth_clearances(p):
    """On the cloth as drawn (ring edges and cord strokes, the soft shadows aside): no tie
    or pendant meets a ring but its own two, and no hub or bob meets a cord it is not
    on.  Returns the least gap, in cm."""
    rings, cords = PC.cloth_drawn(p)
    least = 1.0
    for kind, a, b, pts, reach in cords:
        for i, rk, x, y, r, rreach in rings:
            if kind != "cord" and i not in (a, b):
                least = min(least, PC.point_to_polyline((x, y), pts) - rreach - reach)
            if kind == "cord" or i not in (a, b):
                if rk in ("hub", "bob"):
                    least = min(least, PC.point_to_polyline((x, y), pts) - rreach - reach)
    return least


def pictures():
    """The Second Challenge's prop pictures (identity card, section 4), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch14.py.  The
    ties below hold the transcription against this file's blackboard: a structural
    check of the pictures, not a second proof."""
    calls = DIALOGUE.check_layout("Second", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("tickdial", True, "", ("first round",), "the round as laid"),
        ("tickdial", False, "", ("first round ticked once",), "the round, ticked"),
        ("tickcloth", True, "", ("one listener; wheel lit",), "a controlled turn, the wheel lit"),
        ("tickcloth", False, "push the wheel", ("one listener; wheel lit",), "the turn landed")]
    PC.check_captions(DIALOGUE, "Second", calls)
    PC.check_section(DIALOGUE, "Second", calls)
    dial_h = round(PC.DIAL_FTOP - PC.DIAL_FBOT, 2)
    assert dial_h == 2.42                                             # read from the Lua
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Second")] == [[dial_h], [dial_h], [1.45], [1.45]]

    # the anchors (card section 4), in dialog.md; the raw file's following paragraphs agree
    a1, a2, a3, a4 = DIALOGUE.anchors("Second")
    f1, f2, f3, f4 = DIALOGUE.following("Second")
    assert "A bead at rest stands at the top of its ring as Achilles sees it" in a1               # D6
    assert "crossing itself once" in a1                                                           # D6
    assert f1.startswith("**Tortoise:** Tick it.")
    assert "He does not touch a bead" in a2
    assert f2.startswith("**Achilles:** Every ring has gone to a new hour")
    assert "The picture as it stands" in a3
    assert f3.startswith("*[First she slides the eighth off the wheel's green ring")
    assert "the ring back along the cord into the end-ring, the stretch between swallowed" in a4  # D11
    assert "The wheel's end-ring lies just past where its green ring stood" in a4                 # D11
    assert f4.startswith("**Achilles:** So what you're saying is that the wheel gave its quarter to the listener.")
    assert DIALOGUE.following("Second", DIALOGUE.raw) == [f1, f2, f3, f4]
    section = DIALOGUE.section("Second")
    for phrase in (
            "a loop of pale twine threaded with small rings of plain wood",                         # D5
            "setting her little box of rings and the packet of tubes on the table beside the cloth",  # D7
            "the part of the cloth nearest Achilles",                                               # D7
            "laying a second cord above it, further from Achilles",                                 # D8
            "each from its cord's loose far end", "set to their right, midway between the cords",   # D9
            "a tie,", "running on to the right",                                                    # D9
            "the stretch of wheel cord it ran along swallowed with it",                             # D10
            "the tie swallowed behind it",                                                          # D10
            "She gathers the cloth clear, the rings into her little box of rings and the string "
            "into her work-bag, and lays it all again",                                             # D12
            "and lays them side by side on the cloth.",                                             # D13
            "She lays them back where they lay."):                                                  # D14
        assert phrase in section, phrase
    assert "The wheel cord runs from end to end as it did before" not in section                    # D11's old
    assert "set off to one side" not in section                                                     # D9's old

    p21, p22, p23, p24 = (PC.picture(c) for c in calls)

    # P2.1: the round as laid IS |psi_1> (the 1/2 aside, which the beads do not carry): at
    # the i-th hour of the hand's way the bead stands i quarters back
    N, a, r = 15, 7, 4
    U7 = tick_box(a, N, 4)
    psis, orb = rounds(a, N, 4, r)
    assert PC.dial_orbit(15) == orb == [1, 7, 4, 13]
    assert p21.dump == "hours=15 hand=1 rings=1:1:0,7:1:18,4:1:12,13:1:6"
    assert p21.labels == ("ring", "bead")
    # the keyed words, as set (measured node boxes, strokes included): "ring" in the sheet's
    # open left margin beside the ring at thirteen, "bead" just below the bead at seven --
    # each inside the sheet and touching nothing drawn
    # (the node box, padding included, clears everything; the letters by over 0.03 -- the
    # tightest are "ring" to the sheet's left edge and "bead" to the pencil mark at eight)
    for word in ("ring", "bead"):
        gaps = PC.dial_word_gaps(p21.shown, word)
        assert min(gaps.values()) > 0, (word, gaps)
        assert min(PC.dial_word_gaps(p21.shown, word, ink=True).values()) > 0.03, word
    (_, bx, by, banc), = [l for l in p21.shown.labels if l[0] == "bead"]
    seven = [r_ for r_ in p21.shown.rings if r_[0] == 7][0]
    assert banc == "north east" and (bx, by) == (seven[5] - PC.BEAD_DX, seven[6] - PC.RBR - PC.BEAD_DY)
    laid = {ring["hour"]: bead_phase(ring["t"]) for ring in p21.state.rings}
    assert sorted(laid) == sorted(orb)
    v = np.zeros(16, dtype=complex)
    for h, z in laid.items():
        v[h] = z / 2
    assert np.allclose(v, psis[1]) and not np.allclose(v, psis[3])   # a quarter BACK, not on
    # P2.2: ticked once, every ring carried to the next hour of the hand's way, its bead
    # untouched -- and the round is U_7 |psi_1> = i |psi_1>: every hour's bead a quarter on
    assert p22.dump == "hours=15 hand=1 rings=1:1:6,7:1:0,4:1:18,13:1:12"
    after = {ring["hour"]: ring["t"] for ring in p22.state.rings}
    before = {ring["hour"]: ring["t"] for ring in p21.state.rings}
    assert all(after[h] == (before[h] + 6) % 24 for h in orb)
    w = np.zeros(16, dtype=complex)
    for h, t in after.items():
        w[h] = bead_phase(t) / 2
    assert np.allclose(w, U7 @ psis[1]) and np.allclose(w, 1j * psis[1])
    # each ring carried with its bead: the ring that stood at hour h now stands at 7h mod 15
    for k, ring in enumerate(p21.state.rings):
        moved = p22.state.rings[k]
        assert moved["hour"] == (7 * ring["hour"]) % 15 and moved["t"] == ring["t"]
    # what is drawn: one ring at each of the hand's hours, on its mark, the loop running
    # ring to ring in the hand's order and back, crossing itself once (one-seven over
    # four-thirteen); every bead on its ring's band, at rest at the top
    for p in (p21, p22):
        loop = p.shown.loops[1]
        assert loop == tuple(PC.hourxy(15, h) for h in orb)
        segs = list(zip(loop, loop[1:] + loop[:1]))
        crossings = [(i, j) for i in range(4) for j in range(i + 2, 4) if not (i == 0 and j == 3)
                     and PC.crosses(segs[i], segs[j])]
        assert crossings == [(0, 2)]                                   # 1-7 crosses 4-13, once
        for hour, k, t, x, y, bx, by in p.shown.rings:
            assert np.isclose(np.hypot(bx - x, by - y), PC.RRM)
            assert np.isclose((90 - np.degrees(np.arctan2(by - y, bx - x))) % 360, 15 * t)
    rest = [ring for ring in p21.shown.rings if ring[0] == 1][0]
    assert np.isclose(rest[5], rest[3]) and rest[6] > rest[4]          # at rest: at the top

    # P2.3: the controlled turn for a quarter, the wheel lit -- each green ring half the
    # turn (an eighth), the bob half the turn back, the hub bare: the gadget of the
    # sketchbook, green alpha/2, green alpha/2, red hub, green -alpha/2, at alpha = pi/2
    S3 = p23.scene                                                      # the scene as the Lua builds it
    assert p23.dump == (
        "scene=one wheel=lit listener=green:0 ct=listener-wheel:6 crossed=false | cols=8 rows=2 y=0..1 "
        "near=false folded=false; gaps=0.5; 1:end green 0 @0,0; 2:ring green 3 @2,0; 3:hub red 0 @3,0.5; "
        "4:bob green 21 @4,0.5; 5:end red 12 @0,1; 6:ring green 3 @2,1; tie 2-3 []; tie 3-6 []; pendant 3-4 []; "
        "cord loose-1 [7,0 0,0]; cord loose-5 [7,1 0,1]")
    assert p23.labels == ("listener", "wheel cord", "hub", "bob", "tie")
    turn = p23.state["cts"][0]["turn"]
    assert np.isclose(bead_phase(turn), 1j)                            # a quarter: alpha = pi/2
    halves = [x["t"] for x in S3["things"] if x["kind"] == "ring"]
    bob = [x["t"] for x in S3["things"] if x["kind"] == "bob"]
    assert np.allclose([bead_phase(t) for t in halves], np.exp(1j * np.pi / 4))
    assert np.isclose(bead_phase(bob[0]), np.exp(-1j * np.pi / 4))
    # the hub "set to their right, midway between the cords"; the pendant on to the right
    hub = [x for x in S3["things"] if x["kind"] == "hub"][0]
    ring_x = {x["x"] for x in S3["things"] if x["kind"] == "ring"}
    bob_x = [x["x"] for x in S3["things"] if x["kind"] == "bob"]
    assert hub["y"] == 0.5 and hub["x"] > max(ring_x) and bob_x[0] > hub["x"]
    least = cloth_clearances(p23)
    assert least > 0.03, least

    # P2.4: the wheel pushed, lit: the quarter lands on the listener (its end-ring at a
    # quarter), as CP(pi/2) (|+> (x) |1>) = (|0> + i|1>)/sqrt2 (x) |1> on the blackboard;
    # with the wheel dark (not pictured, l.455) the listener stays at rest, as on |0>
    assert p24.dump == (
        "scene=one wheel=lit listener=green:6 crossed=false | cols=8 rows=2 y=0..1 near=false folded=false; "
        "gaps=0.5; 1:end green 6 @0,0; 2:end red 12 @2.5,1; cord loose-1 [7,0 0,0]; cord loose-2 [7,1 2.5,1]")
    plus = (ket(0) + ket(1)) / np.sqrt(2)
    for scene, wheel in (("one listener; wheel lit", 1), ("one listener; wheel dark", 0)):
        M = PC.cloth_state("push the wheel", scene)
        out = CP(np.pi / 2) @ np.kron(plus, ket(wheel))
        listener = out.reshape(2, 2)[:, wheel]
        assert np.isclose(listener[1] / listener[0], bead_phase(M["cords"]["listener"]["t"]))
    assert not PC.TEX_ERRORS
    # the fixed parts, P2.3 -> P2.4: the listener's end-ring keeps its place; the wheel's
    # runs to just past the last ring it went into (x + 0.5), its cord's drawn start with
    # it (the swallowed length); the turn's rings, hub, bob and ties are gone; nothing added
    ends3 = {x["y"]: x for x in S3["things"] if x["kind"] == "end"}
    S4 = p24.scene
    ends4 = {x["y"]: x for x in S4["things"] if x["kind"] == "end"}
    assert ends3[0]["x"] == ends4[0]["x"] == 0
    assert ends4[1]["x"] == max(ring_x) + 0.5 == 2.5
    assert [d["pts"] for d in S3["cords"] if d["kind"] == "cord"] == [[[0, 0], [7, 0]], [[0, 1], [7, 1]]]
    assert [d["pts"] for d in S4["cords"]] == [[[0, 0], [7, 0]], [[2.5, 1], [7, 1]]]
    assert len(S4["things"]) == 2


# --- the sketchbook ---------------------------------------------------------------------

def gadget(alpha, swap=False):
    """The bare controlled turn, alpha in units of pi.  Green alpha/2 on each wire, both
    tied to one phase-free red hub, a green -alpha/2 pendant point on the hub.  With
    swap=True the same shape is built with its two wires exchanged."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in (0, 1)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 10) for r in (0, 1)]
    rings = [g.add_vertex(VT.Z, r, 2, phase=alpha / 2) for r in (0, 1)]
    hub = g.add_vertex(VT.X, 0.5, 3)
    pend = g.add_vertex(VT.Z, 0.5, 4, phase=-alpha / 2)
    for r in (0, 1):
        g.add_edge((ins[r], rings[r])); g.add_edge((rings[r], outs[r]))
        g.add_edge((rings[r], hub))                                      # the ties
    g.add_edge((hub, pend))                                              # the pendant
    if swap:
        ins, outs = ins[::-1], outs[::-1]
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    return g


def scaled(g, power):
    """g times sqrt2^power (PyZX's add_power counts powers of sqrt2)."""
    h = g.copy()
    h.scalar.add_power(power)
    return h


def lemma_left(c, alpha):
    """A red point c*pi fed into the top wire of the bare controlled turn."""
    g = zx.Graph()
    p = g.add_vertex(VT.X, 0, 0, phase=Fraction(c))
    i1 = g.add_vertex(VT.BOUNDARY, 1, 0)
    o0, o1 = g.add_vertex(VT.BOUNDARY, 0, 10), g.add_vertex(VT.BOUNDARY, 1, 10)
    t, b = g.add_vertex(VT.Z, 0, 2, phase=alpha / 2), g.add_vertex(VT.Z, 1, 2, phase=alpha / 2)
    hub, pend = g.add_vertex(VT.X, 0.5, 3), g.add_vertex(VT.Z, 0.5, 4, phase=-alpha / 2)
    g.add_edges([(p, t), (t, o0), (i1, b), (b, o1), (t, hub), (b, hub), (hub, pend)])
    g.set_inputs((i1,)); g.set_outputs((o0, o1))
    return g


def lemma_right(c, alpha):
    """The point passed straight through, and a green c*alpha on the other wire."""
    g = zx.Graph()
    p = g.add_vertex(VT.X, 0, 0, phase=Fraction(c))
    i1 = g.add_vertex(VT.BOUNDARY, 1, 0)
    o0, o1 = g.add_vertex(VT.BOUNDARY, 0, 10), g.add_vertex(VT.BOUNDARY, 1, 10)
    z = g.add_vertex(VT.Z, 1, 5, phase=c * alpha)
    g.add_edges([(p, o0), (i1, z), (z, o1)])
    g.set_inputs((i1,)); g.set_outputs((o0, o1))
    return g


def sketchbook():
    alphas = (Fraction(1, 2), Fraction(3, 7))                           # pi/2 and 3pi/7

    # (a) Result 14.5: the bare controlled turn is CP(alpha)/sqrt2, so add_power(1) (one
    #     factor sqrt2) makes it exactly CP(alpha); and it is symmetric in its two wires
    for alpha in alphas + (Fraction(1), Fraction(5, 3), Fraction(-2, 9)):
        g = gadget(alpha)
        assert np.allclose(matrix_of(scaled(g, 1)), CP(float(alpha) * np.pi))
        assert np.allclose(matrix_of(g), CP(float(alpha) * np.pi) / np.sqrt(2))   # scalar drawn
        assert same_picture(g, gadget(alpha, swap=True))                # wires exchanged
        if alpha % 1:                                                   # (CP(pi) = CP(-pi))
            assert not np.allclose(matrix_of(scaled(g, 1)), CP(-float(alpha) * np.pi))  # sign
    # the hub reads the parity: a phase-free red dot, two ties in, one out, is
    # |x>|y> -> 2^{-1/2} |x xor y>
    hg = zx.Graph()
    i0, i1 = hg.add_vertex(VT.BOUNDARY, 0, 0), hg.add_vertex(VT.BOUNDARY, 1, 0)
    hub, o = hg.add_vertex(VT.X, 0.5, 1), hg.add_vertex(VT.BOUNDARY, 0.5, 2)
    hg.add_edges([(i0, hub), (i1, hub), (hub, o)])
    hg.set_inputs((i0, i1)); hg.set_outputs((o,))
    for x in (0, 1):
        for y in (0, 1):
            assert np.allclose(matrix_of(hg) @ ket(x, y), ket(x ^ y) / np.sqrt(2))

    # (b) Result 14.6, the basis-point lemma: bare left side = 2^{-1/2} x right side, so the
    #     left side gets add_power(1).  DERIVED: full_reduce on a copy of the left side
    #     removes the whole gadget and leaves exactly the right side's shape -- one point on
    #     the top wire carrying c*pi, and on the other wire a single dot with turn alpha when
    #     the point is lit, a bare wire when it is dark.
    for alpha in alphas:
        for c in (0, 1):
            L, R = scaled(lemma_left(c, alpha), 1), lemma_right(c, alpha)
            assert same_picture(L, R)
            assert not same_picture(lemma_left(c, alpha), R)            # the 2^{-1/2} is real
            assert not same_picture(L, lemma_right(1 - c, alpha))       # dark != lit
            h = L.copy()
            simp.full_reduce(h)
            assert same_picture(h, R)                                   # value kept
            inner = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
            o0, o1 = h.outputs()
            (pt,) = h.neighbors(o0)
            assert h.vertex_degree(pt) == 1 and h.phase(pt) == c         # the point, c*pi
            turned = [v for v in inner if v != pt]
            if c:
                (z,) = turned                                           # lit: one dot, alpha
                assert h.phase(z) == alpha and h.vertex_degree(z) == 2
                assert h.connected(z, o1) and h.connected(z, h.inputs()[0])
            else:
                assert turned == [] and h.connected(h.inputs()[0], o1)  # dark: a bare wire
            # the reduced right side agrees too (the non-Clifford route: both reduced,
            # compared as tensors)
            hr = R.copy()
            simp.full_reduce(hr)
            assert same_picture(h, hr)

    # (b') the derivation's scalars, one step at a time
    for alpha in alphas:
        a_rad = float(alpha) * np.pi
        for c in (0, 1):
            # copy: a red point c into a phase-free green with three open legs = 1/2 x three
            # red points c  (add_power(-2) is the factor 1/2)
            g = zx.Graph()
            p, z = g.add_vertex(VT.X, 0, 0, phase=Fraction(c)), g.add_vertex(VT.Z, 1, 1)
            outs = [g.add_vertex(VT.BOUNDARY, r, 3) for r in range(3)]
            g.add_edges([(p, z)] + [(z, o) for o in outs])
            g.set_inputs(()); g.set_outputs(tuple(outs))
            three = zx.Graph()
            outs = [three.add_vertex(VT.BOUNDARY, r, 3) for r in range(3)]
            for r in range(3):
                three.add_edge((three.add_vertex(VT.X, r, 0, phase=Fraction(c)), outs[r]))
            three.set_inputs(()); three.set_outputs(tuple(outs))
            assert same_picture(g, scaled(three, -2))
            assert not same_picture(g, three)
            # closing: a red point c joined to a green alpha/2 point is sqrt2 e^{i c alpha/2}
            g = zx.Graph()
            p, z = g.add_vertex(VT.X, 0, 0, phase=Fraction(c)), g.add_vertex(VT.Z, 0, 1, phase=alpha / 2)
            g.add_edge((p, z))
            closing = matrix_of(g).item()
            assert np.isclose(closing, np.sqrt(2) * np.exp(1j * c * a_rad / 2))
            # pi-commutation: a two-legged red pi on a green -alpha/2 point equals
            # e^{-i alpha/2} x a green +alpha/2 point
            g = zx.Graph()
            z = g.add_vertex(VT.Z, 0, 0, phase=-alpha / 2)
            x = g.add_vertex(VT.X, 0, 1, phase=Fraction(1))
            o = g.add_vertex(VT.BOUNDARY, 0, 2)
            g.add_edges([(z, x), (x, o)]); g.set_inputs(()); g.set_outputs((o,))
            f = zx.Graph()
            z = f.add_vertex(VT.Z, 0, 0, phase=alpha / 2)
            o = f.add_vertex(VT.BOUNDARY, 0, 2)
            f.add_edge((z, o)); f.set_inputs(()); f.set_outputs((o,))
            flip = np.exp(-1j * a_rad / 2)
            assert np.allclose(matrix_of(g), flip * matrix_of(f))
            assert not np.allclose(matrix_of(g), matrix_of(f))          # the scalar matters
            # the ledger: 1/2 from the copy, the closing, the flip when lit -> 2^{-1/2}
            total = 0.5 * closing * (flip if c else 1)
            assert np.isclose(total, 2 ** -0.5)

    # (c) one listener, in pictures.  A green point on the listener, a lit wheel, the
    #     controlled turn for a quarter (alpha = pi/2) or three quarters (3pi/2), then the
    #     listener read through a yellow box (a Hadamard edge).  The listener's two readings
    #     come out at even odds for both turns: the trap.
    def listener(alpha):
        g = zx.Graph()
        gp = g.add_vertex(VT.Z, 0, 0)                                   # green point: |0>+|1>
        wp = g.add_vertex(VT.X, 1, 0, phase=Fraction(1))                # lit wheel: sqrt2|1>
        o0, o1 = g.add_vertex(VT.BOUNDARY, 0, 10), g.add_vertex(VT.BOUNDARY, 1, 10)
        t, b = g.add_vertex(VT.Z, 0, 2, phase=alpha / 2), g.add_vertex(VT.Z, 1, 2, phase=alpha / 2)
        hub, pend = g.add_vertex(VT.X, 0.5, 3), g.add_vertex(VT.Z, 0.5, 4, phase=-alpha / 2)
        g.add_edges([(gp, t), (wp, b), (b, o1), (t, hub), (b, hub), (hub, pend)])
        g.add_edge((t, o0), ET.HADAMARD)                                # the yellow box
        g.set_inputs(()); g.set_outputs((o0, o1))
        return g
    amps = {}
    for alpha in (Fraction(1, 2), Fraction(3, 2)):
        v = matrix_of(listener(alpha)).ravel()
        # green point sqrt2|+>, lit point sqrt2|1>, bare gadget CP/sqrt2: the state is
        # sqrt2 (H Z(alpha)|+>) (x) |1>
        a_rad = float(alpha) * np.pi
        assert np.allclose(v, np.sqrt(2) * np.kron(H @ Zph(a_rad) @ H @ ket(0), ket(1)))
        p = np.abs(v) ** 2 / np.sum(np.abs(v) ** 2)
        assert np.isclose(p[0b01], 0.5) and np.isclose(p[0b11], 0.5)    # even odds
        amps[alpha] = v[0b01]
    assert not np.isclose(amps[Fraction(1, 2)], amps[Fraction(3, 2)])    # different numbers,
    assert np.isclose(abs(amps[Fraction(1, 2)]), abs(amps[Fraction(3, 2)]))   # the same size

    # the one-line picture: a green theta point, a yellow box, the dark point = a lone red
    # theta dot = 1 + e^{i theta}.  DERIVED for a quarter and three quarters (Clifford):
    # full_reduce empties the closed picture and leaves exactly that number.
    for theta in (Fraction(1, 2), Fraction(3, 2), Fraction(3, 7)):
        g = zx.Graph()
        z, x = g.add_vertex(VT.Z, 0, 0, phase=theta), g.add_vertex(VT.X, 0, 2)
        g.add_edge((z, x), ET.HADAMARD)
        lone = zx.Graph()
        lone.add_vertex(VT.X, 0, 0, phase=theta)
        number = 1 + np.exp(1j * float(theta) * np.pi)
        assert np.isclose(matrix_of(g).item(), number)
        assert np.isclose(matrix_of(lone).item(), number)
        if theta.denominator <= 2:
            h = g.copy()
            simp.full_reduce(h)
            assert h.num_vertices() == 0 and np.isclose(h.scalar.to_number(), number)
    # the dark reading's amplitude is (1 + e^{i theta})/2: the two points carry 2
    for theta in (np.pi / 2, 3 * np.pi / 2):
        assert np.isclose((ket(0) @ H @ Zph(theta) @ H @ ket(0)), (1 + np.exp(1j * theta)) / 2)


if __name__ == "__main__":
    verdict("KickedTurn", blackboard, sketchbook)
