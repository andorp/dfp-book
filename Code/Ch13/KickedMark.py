"""Challenge II: The Mark on the Cords  (kickback as a copy; Deutsch-Jozsa read off the web)
=========================================================================================
"...Oh. It's your token, wedged up on its edge on the porch in the spring. The turn had nothing on the flag to spend itself on, so it jumped back up the tie onto the cord that asked."  -- Achilles

Claim: with the flag at |->, a tie leaves a Z on its rod and the flag untouched,
CNOT (|psi>|->) = (Z|psi>)|->; in pictures a green pi point fed into the red end of a
tie is COPIED up it (copy rule, then fusion), with the factor 1/sqrt2 drawn; a
phase-free green point (the flag at |+>) copies as nothing, and for EVERY f
(H^n (x) I) U_f (H^n (x) I) |0^n>|+> = |0^n>|+> (Result 13.4).  For every c,
sum_x (-1)^{c.x} = 2^n [c = 0] (Result 13.5), so for f = (a.x) xor b the all-dark
amplitude of Chapter 5's test is (-1)^b [a = 0]: no ties, every rod dark; any tie,
some rod lit (Result 13.6).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from functools import reduce
from itertools import product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import I2, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch13 as PC
import props_cloth as CL

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Z = np.diag([1, -1]).astype(complex)
PLUS, MINUS = H @ ket(0), H @ ket(1)          # normalized |+>, |->
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def strings(n):
    return list(product((0, 1), repeat=n))


def dot(a, x):
    return sum(ai * xi for ai, xi in zip(a, x)) % 2


def index(bits):
    return int("".join(map(str, bits)) or "0", 2)


def U(f, n):
    """U_f |x>|y> = |x>|y xor f(x)>, rods first, the flag last."""
    M = np.zeros((2 ** (n + 1), 2 ** (n + 1)), dtype=complex)
    for x in strings(n):
        for y in (0, 1):
            M[index(x + (y ^ f(x),)), index(x + (y,))] = 1
    return M


def Hn(n):
    return reduce(np.kron, [H] * n, np.eye(1))


def signs(f, n):
    """The diagonal (-1)^{f(x)} on the rods."""
    return np.diag([(-1.0) ** f(x) for x in strings(n)]).astype(complex)


def balanced_table(rng, n):
    t = np.array([0] * 2 ** (n - 1) + [1] * 2 ** (n - 1))
    rng.shuffle(t)
    return t


def blackboard():
    rng = np.random.default_rng(1302)
    assert np.isclose(np.linalg.norm(MINUS), 1) and np.isclose(np.linalg.norm(PLUS), 1)  # normalized

    # --- kickback (row 37) on a random NON-LINEAR f, n = 4: every basis x, and a random
    #     complex rod state
    n = 4
    table = rng.integers(0, 2, 2 ** n)
    f = lambda x: int(table[index(x)])
    assert not any(all(f(x) == dot(a, x) ^ b for x in strings(n)) for a in strings(n) for b in (0, 1))
    Uf = U(f, n)
    for x in strings(n):
        ex = np.eye(2 ** n)[index(x)]
        assert np.allclose(Uf @ np.kron(ex, MINUS), (-1) ** f(x) * np.kron(ex, MINUS))
    psi = random_state(rng, 2 ** n)
    assert not np.allclose(psi, psi.conj())
    assert np.allclose(Uf @ np.kron(psi, MINUS), np.kron(signs(f, n) @ psi, MINUS))
    assert not np.allclose(Uf @ np.kron(psi, MINUS), np.kron(psi, MINUS))   # the sign did land
    # the flag window, crosswise and lit, "comes back lit every time, marks or none":
    # whatever the box, the flag leaves as |-> (read crosswise: lit, with certainty)
    out = (np.kron(np.eye(2 ** n), H) @ Uf @ np.kron(psi, MINUS)).reshape(2 ** n, 2)
    assert np.allclose(out[:, 0], 0) and np.isclose(np.linalg.norm(out[:, 1]), 1)

    # --- Result 13.4 (i): CNOT (|psi>|->) = (Z|psi>)|->, psi a random complex rod state
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj()) and not np.allclose(Z @ psi, psi) and not np.allclose(Z @ psi, -psi)
    assert np.allclose(CNOT @ np.kron(psi, MINUS), np.kron(Z @ psi, MINUS))
    assert not np.allclose(CNOT @ np.kron(psi, MINUS), np.kron(psi, MINUS))       # not nothing
    assert not np.allclose(CNOT @ np.kron(psi, MINUS), np.kron(psi, -MINUS))      # not a global sign
    # (ii) the trap: CNOT (|psi>|+>) = |psi>|+>
    assert np.allclose(CNOT @ np.kron(psi, PLUS), np.kron(psi, PLUS))
    # ... and for EVERY f: U_f acts on (anything)|+> as the identity, so the network returns
    # |0^n>|+>.  Exhaustive over all f for n = 1, 2, 3; every linear f and a random
    # non-linear f for n = 4; on a random complex rod state as well
    def trap(Uf, n):
        return np.kron(Hn(n), I2) @ Uf @ np.kron(Hn(n), I2) @ np.kron(ket(*[0] * n), PLUS)
    for n in (1, 2, 3):
        for tab in product((0, 1), repeat=2 ** n):
            assert np.allclose(trap(U(lambda x: tab[index(x)], n), n), np.kron(ket(*[0] * n), PLUS))
    n = 4
    boxes = [lambda x, a=a, b=b: dot(a, x) ^ b for a in strings(n) for b in (0, 1)] + [f]
    for g in boxes:
        Ug = U(g, n)
        assert np.allclose(trap(Ug, n), np.kron(ket(*[0] * n), PLUS))
        Psi = random_state(rng, 2 ** n)
        assert np.allclose(Ug @ np.kron(Psi, PLUS), np.kron(Psi, PLUS))

    # --- the cloth of Challenge II: Achilles's slip, the first and the fifth squares marked
    slip = (1, 0, 0, 0, 1, 0)
    n = 6
    Us = U(lambda x: dot(slip, x), n)
    rods_crosswise_dark = reduce(np.kron, [PLUS] * n)
    read_crosswise = Hn(n + 1)
    # the trap: every knob crosswise and dark, the flag crosswise and dark.  "Every
    # wish-cord ends as it began", the flag too; read crosswise, every window dark.
    start = np.kron(rods_crosswise_dark, PLUS)
    assert np.allclose(Us @ start, start)
    assert np.allclose(read_crosswise @ Us @ start, ket(*[0] * (n + 1)))
    # "Then the web is empty. There are no marks." -- it is not: Achilles's mistake
    assert not np.allclose(Us, np.eye(2 ** (n + 1)))
    # the bead set halfway: the flag crosswise and lit.  The half-turns land on EXACTLY the
    # first and fifth cords; the flag keeps its own.
    start = np.kron(rods_crosswise_dark, MINUS)
    cloth = reduce(np.kron, [MINUS if m else PLUS for m in slip] + [MINUS])
    assert np.allclose(Us @ start, cloth)
    # read crosswise: the first and the fifth windows gleam, and the flag's
    assert np.allclose(read_crosswise @ Us @ start, ket(*slip, 1))
    assert not np.allclose(read_crosswise @ Us @ start, ket(*slip[::-1], 1))    # top wire first
    # a box with no marks at all: nothing comes back; the flag still gleams
    U0 = U(lambda x: 0, n)
    assert np.allclose(read_crosswise @ U0 @ start, ket(*[0] * n, 1))

    # --- Result 13.5, the sign sum, every c for n = 1..6
    for n in range(1, 7):
        for c in strings(n):
            assert sum((-1) ** dot(c, x) for x in strings(n)) == 2 ** n * (not any(c))

    # --- Chapter 5's test (row 38, reused): the all-dark amplitude is 2^{-n} sum (-1)^{f(x)};
    #     constant f: 1 in probability; a random balanced f (tally checked first): 0.
    for n in (4, 5, 6):
        def amp0(g):
            state = np.kron(Hn(n), I2) @ U(g, n) @ np.kron(Hn(n), I2) @ np.kron(ket(*[0] * n), MINUS)
            return state.reshape(2 ** n, 2) @ MINUS.conj()           # rods at 0^n, flag |->
        for b in (0, 1):
            assert np.isclose(abs(amp0(lambda x, b=b: b)[0]) ** 2, 1)
        tab = balanced_table(rng, n)
        assert tab.sum() == 2 ** (n - 1) and (1 - tab).sum() == 2 ** (n - 1)
        bal = lambda x: int(tab[index(x)])
        assert np.isclose(amp0(bal)[0], sum((-1) ** bal(x) for x in strings(n)) / 2 ** n)
        assert np.isclose(abs(amp0(bal)[0]) ** 2, 0)
    # Result 13.6: for f = (a.x) xor b the amplitude is (-1)^b [a = 0], every a, b at n = 4;
    # a != 0 makes f balanced; "If every wish-window stays dark, the box has no marks. If
    # any wish-window gleams, it has."
    n = 4
    for a in strings(n):
        for b in (0, 1):
            g = lambda x: dot(a, x) ^ b
            state = np.kron(Hn(n), I2) @ U(g, n) @ np.kron(Hn(n), I2) @ np.kron(ket(*[0] * n), MINUS)
            rods = state.reshape(2 ** n, 2) @ MINUS.conj()
            assert np.isclose(rods[0], (-1) ** b * (not any(a)))
            assert np.isclose(abs(rods[0]) ** 2, 1.0 if not any(a) else 0.0)
            if any(a):
                assert sum(g(x) for x in strings(n)) == 2 ** (n - 1)            # balanced
    # the BPP remark: k random queries to a balanced f all agree with probability 2^{1-k}
    # (exact, over every k-tuple of strings, n = 3)
    n = 3
    tab = balanced_table(rng, n)
    for k in (1, 2, 3, 4):
        agree = sum(len({int(tab[index(x)]) for x in xs}) == 1
                    for xs in product(strings(n), repeat=k))
        assert np.isclose(agree / 2 ** (n * k), 2.0 ** (1 - k))

    pictures()


def cloth_reading(M):
    """The web on the cloth read at its end-rings (card section 3, l.436, l.562): each
    wish-cord's end-ring with any ring slid along its cord (their turns add), and the
    flag's end-ring; a half turn is lit.  Returns (the colours, the bits, wish-cords then
    the flag)."""
    rows = [M["wish"][k] for k in range(1, 7)] + [M["flag"]]
    bits, colours = [], set()
    for c in rows:
        t = c["endt"] + (c["ring"]["t"] if c.get("ring") else 0)
        colours.add(c["endc"])
        bits.append(int(t % 24 == 12))
    return colours, tuple(bits)


def turned_web(a, start):
    """H on every wire, U_f, H on every wire, from the basis state `start` (rods, flag)."""
    n = len(a)
    return Hn(n + 1) @ U(lambda x: dot(a, x), n) @ Hn(n + 1) @ ket(*start)


def pictures():
    """The Second Challenge's prop pictures (identity card, section 4), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch13.py.  The
    ties below hold the transcription against this file's blackboard: a structural
    check of the pictures, not a second proof."""
    calls = DIALOGUE.check_layout("Second", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("wishcloth", True, "tubes out", ("marks first and fifth; wish-knobs dark; flag dark",),
         "every end-ring turned green"),
        ("wishcloth", False, "tubes out, push the flag, slide together",
         ("marks first and fifth; wish-knobs dark; flag dark",), "every bead still at rest"),
        ("wishcloth", False, "push the flag", ("marks first and fifth; wish-knobs crosswise dark; flag crosswise lit",),
         "turned at the marks, the flag as it began"),
        ("wishcloth", False, "push the flag", ("no marks; wish-knobs crosswise dark; flag crosswise lit",),
         "run clear to the far end")]
    PC.check_captions(DIALOGUE, "Second", calls)
    PC.check_section(DIALOGUE, "Second", calls)
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Second")] == [[2.74]] * 4

    # the anchors (card section 4), in dialog.md; the raw file's following paragraphs agree
    a1, a2, a3, a4 = DIALOGUE.anchors("Second")
    f1, f2, f3, f4 = DIALOGUE.following("Second")
    assert "Last she does the same to the flag's own end-ring: green, at rest." in a1
    assert "clips a tube from the packet onto the cord" in a1                             # D9
    assert f1.startswith("**Tortoise:** A window reads the far end of its cord by the same code.")
    assert "not a tie anywhere, and not a red ring" in a2 and "the ring the tie came up to" in a2   # D11
    assert f2.startswith("**Achilles:** Then the web is empty.")
    assert "It has given its turn away twice and still has it." in a3 and "the tie swallowed behind it" in a3  # D13
    assert f3.startswith("**Achilles:** The flag's turn has gone to the first and the fifth")
    assert "every bead on the cloth is where it was before she pushed" in a4
    assert "the flag cord it ran along swallowed behind it" in a4                          # D15
    assert f4.startswith("**Tortoise:** If every wish-window stays dark")
    assert DIALOGUE.following("Second", DIALOGUE.raw) == [f1, f2, f3, f4]
    section = DIALOGUE.section("Second")
    for phrase in ("What is left of the handful she drops back into the box.",                         # D7
                   "slit along its length, so that it clips onto a cord anywhere",                       # D8
                   "the stretch of flag cord it ran along is swallowed with it",                         # D10
                   "where the flag cord now begins",
                   "Then she slides that ring out along its cord to the green end-ring at the cord's start",
                   "She gathers the cloth clear, the rings into her little box of rings and the string "     # D12
                   "into her work-bag, and lays the web afresh as it was:",
                   "She gathers the cloth clear again, the rings into their box and the string into her work-bag"):  # D14
        assert phrase in section, phrase
    assert "She unties both ties" not in section and "shaking a new packet out onto the cloth" not in section

    p21, p22, p23, p24 = (PC.picture(c) for c in calls)
    p14 = PC.wishcloth("", "marks first and fifth")                   # the web as laid (P1.4)
    a = (1, 0, 0, 0, 1, 0)                                            # Achilles's slip
    assert tuple(int(k in p14.state["marks"]) for k in range(1, 7)) == a

    # P2.1: the convention -- every end-ring turned green at rest, the rings and ties as laid
    assert p21.dump == (
        "scene=web marks=[first fifth] first=green0+green0@2tied second=green0 third=green0 fourth=green0 "
        "fifth=green0+green0@4tied sixth=green0 flag=green0@0[red0@2 red0@4] threaded=false pushed=false "
        "far=false | cols=8 rows=7 y=0..6 near=false folded=false; gaps=5.5; 1:end green 0 @0,0; "
        "2:ring green 0 @2,0; 3:end green 0 @0,1; 4:end green 0 @0,2; 5:end green 0 @0,3; 6:end green 0 @0,4; "
        "7:ring green 0 @4,4; 8:end green 0 @0,5; 9:end green 0 @0,6; 10:ring red 0 @2,6; 11:ring red 0 @4,6; "
        "tie 2-10 []; tie 7-11 []; cord loose-1 [6.2,0 0,0]; cord loose-3 [6.2,1 0,1]; cord loose-4 [6.2,2 0,2]; "
        "cord loose-5 [6.2,3 0,3]; cord loose-6 [6.2,4 0,4]; cord loose-8 [6.2,5 0,5]; cord loose-9 [7,6 0,6]")
    assert p21.labels == ("end-ring", "bead")
    ends = [r for r in p21.cloth["things"] if r["kind"] == "end"]
    assert len(ends) == 7 and all((r["c"], r["t"], r["x"]) == ("green", 0, 0) for r in ends)
    rings = lambda pic: sorted((r["c"], r["x"], r["y"]) for r in pic.cloth["things"] if r["kind"] == "ring")
    ties = lambda pic: sorted((pic.cloth["things"][d["a"] - 1]["x"], pic.cloth["things"][d["b"] - 1]["x"])
                              for d in pic.cloth["cords"] if d["kind"] == "tie")
    assert rings(p21) == rings(p14) and ties(p21) == ties(p14) == [(2, 2), (4, 4)]   # ties straight down

    # P2.2: the trap (the flag crosswise dark): nothing comes back -- every end-ring green
    # at rest, and H on every wire, U_f, H on every wire takes |0000000> to |0000000>
    assert p22.dump == (
        "scene=web marks=[first fifth] first=green0 second=green0 third=green0 fourth=green0 fifth=green0 "
        "sixth=green0 flag=green0@4.5[] threaded=false pushed=true far=false | cols=8 rows=7 y=0..6 near=false "
        "folded=false; gaps=5.5; 1:end green 0 @0,0; 2:end green 0 @0,1; 3:end green 0 @0,2; 4:end green 0 @0,3; "
        "5:end green 0 @0,4; 6:end green 0 @0,5; 7:end green 0 @4.5,6; cord loose-1 [6.2,0 0,0]; "
        "cord loose-2 [6.2,1 0,1]; cord loose-3 [6.2,2 0,2]; cord loose-4 [6.2,3 0,3]; cord loose-5 [6.2,4 0,4]; "
        "cord loose-6 [6.2,5 0,5]; cord loose-7 [7,6 4.5,6]")
    colours, bits = cloth_reading(p22.state)
    assert colours == {"green"} and bits == (0,) * 7
    assert np.allclose(turned_web(a, (0,) * 7), ket(*bits))
    assert not any(d["kind"] == "tie" for d in p22.cloth["cords"]) and not any(r["c"] == "red" for r in p22.cloth["things"])

    # P2.3: the flag crosswise lit -- its half turn copied up the ties onto exactly the marked
    # cords (turns add, 0 + 12), the flag keeping its own; H U_f H |0000001> = |1000101>
    assert p23.dump.startswith(
        "scene=web marks=[first fifth] first=green0+green12@2 second=green0 third=green0 fourth=green0 "
        "fifth=green0+green12@4 sixth=green0 flag=green12@4.5[] threaded=false pushed=true far=false | ")
    colours, bits = cloth_reading(p23.state)
    assert colours == {"green"} and bits == (1, 0, 0, 0, 1, 0, 1) == a + (1,)
    assert np.allclose(turned_web(a, (0,) * 6 + (1,)), ket(*bits))          # every entry exact
    assert not np.allclose(turned_web(a, (0,) * 6 + (1,)), ket(*(a + (0,))))   # the flag keeps its turn
    assert [(r["x"], r["t"]) for r in p23.cloth["things"] if r["kind"] == "ring"] == [(2, 12), (4, 12)]

    # P2.4: no marks -- the flag runs clear to the far end; every wish-cord reads dark;
    # with no tie the blackboard's turned web is bare: |0000001> stays |0000001>
    assert p24.dump == (
        "scene=web marks=[] first=green0 second=green0 third=green0 fourth=green0 fifth=green0 sixth=green0 "
        "flag=green12@7.0[] threaded=false pushed=true far=true | cols=8 rows=7 y=0..6 near=false folded=false; "
        "gaps=5.5; 1:end green 0 @0,0; 2:end green 0 @0,1; 3:end green 0 @0,2; 4:end green 0 @0,3; "
        "5:end green 0 @0,4; 6:end green 0 @0,5; 7:end green 12 @7,6; cord loose-1 [6.2,0 0,0]; "
        "cord loose-2 [6.2,1 0,1]; cord loose-3 [6.2,2 0,2]; cord loose-4 [6.2,3 0,3]; cord loose-5 [6.2,4 0,4]; "
        "cord loose-6 [6.2,5 0,5]; cord loose-7 [7,6 7,6]")
    colours, bits = cloth_reading(p24.state)
    assert colours == {"green"} and bits == (0,) * 6 + (1,)
    assert np.allclose(turned_web((0,) * 6, bits), ket(*bits))
    # after the push to the far end no flag cord shows beyond the end-ring: its record is
    # zero-length at the ring's x, and nothing (no cord, no whipped tip) is drawn for it
    flag = [d for d in p24.cloth["cords"] if d["kind"] == "cord" and d["pts"][0][1] == PC.FLAGROW]
    assert len(flag) == 1 and flag[0]["pts"] == [[7, 6], [7, 6]] and CL.zero_length(flag[0])
    assert CL.drawn(p24.cloth) == {"paths": 6, "tips": 6, "lashings": 6}       # the six wish-cords only
    assert CL.drawn(p22.cloth) == {"paths": 7, "tips": 7, "lashings": 7}       # a flag cord left at 4.5

    # the fixed parts: the end-rings not pushed keep x = 0; the flag's end-ring moves right
    # (0 -> 4.5 past the last ring, or -> 7.0 with no ring) and the flag cord's drawn start
    # moves with it; a slid ring lands at the end-ring's place, x = 0, and no ring is added
    endx = lambda pic: [r["x"] for r in pic.cloth["things"] if r["kind"] == "end" and r["y"] == PC.FLAGROW]
    assert endx(p21) == [0] and endx(p22) == endx(p23) == [4.5] and endx(p24) == [PC.FAR]
    for pic in (p21, p22, p23, p24):
        assert [r["x"] for r in pic.cloth["things"] if r["kind"] == "end" and r["y"] < PC.FLAGROW] == [0] * 6
        start = [min(x for x, _ in d["pts"]) for d in pic.cloth["cords"] if d["kind"] == "cord" and d["pts"][0][1] == PC.FLAGROW]
        assert start == endx(pic)
    assert len(p22.cloth["things"]) == 7 < len(p21.cloth["things"]) == 11
    assert not PC.TEX_ERRORS


# --- pictures ---------------------------------------------------------------------------

def tie_with_flag_point(phase):
    """An open rod through a green control, tied to a red target on the flag; the flag
    enters as a green point of the given phase (pi: sqrt2 |->, 0: sqrt2 |+>).
    Input: the rod.  Outputs: rod, flag.  Bare: no scalar added."""
    g = zx.Graph()
    i, o, fo = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 0, 3), g.add_vertex(VT.BOUNDARY, 1, 3)
    c = g.add_vertex(VT.Z, 0, 1.5)
    t = g.add_vertex(VT.X, 1, 1.5)
    p = g.add_vertex(VT.Z, 1, 0, phase=Fraction(phase))
    g.add_edges([(i, c), (c, o), (c, t), (p, t), (t, fo)])
    g.set_inputs((i,)); g.set_outputs((o, fo))
    return g


def rod_and_point(rod_phase, point_phase):
    """A green dot of rod_phase on the rod (pi: Z; 0: a plain wire) beside a green point of
    point_phase on the flag.  times 1/sqrt2 (add_power(-1)): the point is sqrt2 |+->."""
    g = zx.Graph()
    i, o, fo = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 0, 3), g.add_vertex(VT.BOUNDARY, 1, 3)
    d = g.add_vertex(VT.Z, 0, 1.5, phase=Fraction(rod_phase))
    p = g.add_vertex(VT.Z, 1, 2, phase=Fraction(point_phase))
    g.add_edges([(i, d), (d, o), (p, fo)])
    g.set_inputs((i,)); g.set_outputs((o, fo))
    g.scalar.add_power(-1)
    return g


def cloth(a, flag_phase, read_crosswise):
    """The Tortoise's cloth: every wish-cord begins at a green end-ring (a crosswise dark
    knob, sqrt2 |+>), the flag at a green end-ring of flag_phase; the web of a (a green
    ring on each marked cord, tied to its own red ring on the flag cord); if
    read_crosswise, a yellow box (Hadamard edge) before every window.  No scalar added."""
    n = len(a)
    g = zx.Graph()
    outs = [g.add_vertex(VT.BOUNDARY, r, 9) for r in range(n + 1)]
    last = [g.add_vertex(VT.Z, r, 0) for r in range(n)]
    flag = g.add_vertex(VT.Z, n, 0, phase=Fraction(flag_phase))
    for i in range(n):
        if a[i]:
            c = g.add_vertex(VT.Z, i, 2 + i)
            t = g.add_vertex(VT.X, n, 2 + i)
            g.add_edges([(last[i], c), (flag, t), (c, t)])
            last[i], flag = c, t
    et = ET.HADAMARD if read_crosswise else ET.SIMPLE
    for i in range(n):
        g.add_edge((last[i], outs[i]), et)
    g.add_edge((flag, outs[n]), et)
    g.set_inputs(()); g.set_outputs(tuple(outs))
    return g


def read_points(h):
    """After full_reduce: every output hangs on its own one-legged dot.  Return, per wire,
    (phase, edge type to the output)."""
    got = []
    for o in h.outputs():
        (v,) = h.neighbors(o)
        assert h.vertex_degree(v) == 1 and h.type(v) != VT.BOUNDARY
        got.append((h.phase(v), h.edge_type(h.edge(v, o))))
    return got


def sketchbook():
    minus_map = np.kron(Z, MINUS.reshape(2, 1))              # Z (x) |->, rod in, (rod, flag) out
    plus_map = np.kron(I2, PLUS.reshape(2, 1))                # I (x) |+>

    # (a) kickback as a copy.  The bare left side is exactly Z (x) |->: the tie is CNOT/sqrt2
    #     and the green pi point is sqrt2 |->.  The right side: Z on the rod and the green pi
    #     point on the flag, times 1/sqrt2 (add_power(-1), the copy's price).
    kick = tie_with_flag_point(1)
    assert np.allclose(matrix_of(kick), minus_map)
    assert same_picture(kick, rod_and_point(1, 1))
    assert not same_picture(kick, rod_and_point(0, 1))       # the rod is NOT left untouched
    # DERIVED with the named rules, in the chapter's order: copy, then fusion
    h = kick.copy()
    (p,) = [v for v in h.vertices() if h.type(v) == VT.Z and h.vertex_degree(v) == 1]
    assert simp.check_copy(h, p) and simp.copy(h, p)          # the red dot copies the green pi
    assert simp.spider_simp(h)                                # the copy fuses into the control
    assert same_picture(h, rod_and_point(1, 1))
    assert [h.phase(v) for v in h.vertices() if h.type(v) == VT.Z and h.vertex_degree(v) == 2] == [1]
    h = kick.copy(); simp.full_reduce(h); assert same_picture(h, kick)

    # (b) the trap: a phase-free green point copies as nothing.  Bare left side is I (x) |+>
    #     exactly; the right side (a plain rod beside a green point) needs add_power(-1)
    #     (determined here: the card had not run it) -- the same 1/sqrt2 as in (a).
    trap = tie_with_flag_point(0)
    assert np.allclose(matrix_of(trap), plus_map)
    assert same_picture(trap, rod_and_point(0, 0))
    assert not same_picture(trap, rod_and_point(1, 0))        # no Z kicked back
    assert not same_picture(trap, kick)
    h = trap.copy()
    (p,) = [v for v in h.vertices() if h.type(v) == VT.Z and h.vertex_degree(v) == 1]
    assert simp.copy(h, p) and simp.spider_simp(h)
    assert same_picture(h, rod_and_point(0, 0))

    # (c) the cloth of Challenge II, Achilles's slip 100010 (w = 2, n = 6).  Seven green
    #     points (2^{7/2}) through a web of w bare ties (2^{-w/2}): the value is
    #     2^{(7-w)/2} U_f |+..+>|+->.
    slip, n, w = (1, 0, 0, 0, 1, 0), 6, 2
    Us = U(lambda x: dot(slip, x), n)
    rods = reduce(np.kron, [PLUS] * n)
    for flag_phase, flag_state in ((0, PLUS), (1, MINUS)):
        g = cloth(slip, flag_phase, read_crosswise=False)
        assert np.allclose(matrix_of(g).ravel(), 2 ** ((n + 1 - w) / 2) * (Us @ np.kron(rods, flag_state)))
        # derived: full_reduce leaves one green point per cord, pi exactly where the
        # half-turn landed (nowhere, in the trap)
        h = g.copy(); simp.full_reduce(h); assert same_picture(h, g)
        pts = read_points(h)
        want = [flag_phase * m for m in slip] + [flag_phase]
        assert [ph for ph, _ in pts] == want
        assert all(et == ET.SIMPLE for _, et in pts)            # green points: still crosswise
    # read crosswise (a yellow box before every window): the trap gives every window dark,
    # the lit flag the first, the fifth and the flag window.  Value times 2^{(w-n-1)/2}
    # (add_power(w - n - 1)) is the reading itself.
    for flag_phase, reading in ((0, (0,) * 7), (1, (1, 0, 0, 0, 1, 0, 1))):
        g = cloth(slip, flag_phase, read_crosswise=True)
        g.scalar.add_power(w - n - 1)
        v = matrix_of(g).ravel()
        assert np.allclose(v, ket(*reading))
        if flag_phase:                                          # top wire first: not 010001
            assert not np.allclose(v, ket(*reading[:6][::-1], reading[6]))
        h = g.copy(); simp.full_reduce(h); assert same_picture(h, g)
        pts = read_points(h)
        # one point per wire; sqrt2|1> is a green pi point behind a yellow box, sqrt2|0> a
        # green point behind one: pi exactly on the lit windows
        assert [ph for ph, _ in pts] == list(reading) and all(et == ET.HADAMARD for _, et in pts)
    # a box with no marks: nothing climbs anywhere; the flag keeps its turn
    g = cloth((0,) * 6, 1, read_crosswise=True)
    g.scalar.add_power(-7)
    assert np.allclose(matrix_of(g).ravel(), ket(0, 0, 0, 0, 0, 0, 1))


if __name__ == "__main__":
    verdict("KickedMark", blackboard, sketchbook)
