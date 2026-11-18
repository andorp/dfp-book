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


DARK = "marks first and fifth; wish-knobs dark; flag dark"
LIT = "marks first and fifth; wish-knobs crosswise dark; flag crosswise lit"
BARE = "no marks; wish-knobs crosswise dark; flag crosswise lit"
GHOST_FLAG = "ghost the flag's end-ring in dotted outline at {} (grid {},6): slid along the flag cord"
GHOST_RED = ("ghost the red ring for the {} square in dotted outline at {} (grid {},6), under the ring now lying "
             "over part of its place: pushed into: gone, its copy up the tie to the {} square's cord")
GHOST_OUT = ("ghost the green ring on the {} square's cord in dotted outline at {} (grid {}): slid out along its "
             "cord into the end-ring")
NO_TIE_ARC = "no arc up a tie: it would cross every cord between (a cut); the ring at the top shows what it got"
FLAG_ARC = ("slide arc the flag's end-ring along its cord, above the row, {} -> {}: on the linen (clear of its edges "
            "and its near hem or fold) by {}; clear of every cord and tie by 0.115 (centre line 0.148), of every "
            "ring by 0.071 (centre line 0.148){} (edge to edge)")
OUT_ARC = ("slide arc the {} square's ring out to its end-ring, beneath the row, {} -> {}: on the linen (clear of "
           "its edges and its near hem or fold) by {}; clear of every cord and tie by {}, of every ring by 0.071 "
           "(centre line 0.148), of the ghosts it leaves by 0.046 (edge to edge)")
GHOSTS = ", of the ghosts it leaves by 0.046"
# the marks each frame draws, as the Lua logs them (card 3e.6, r13/BA/dumps_md.txt)
MARKS = {
    "F12": [GHOST_FLAG.format("(-0.770,-2.266)", 0), GHOST_RED.format("first", "(-0.330,-2.266)", 2, "first"),
            FLAG_ARC.format("(-0.750,-2.145)", "(-0.306,-2.145)", "0.199 (centre line 0.210)", GHOSTS), NO_TIE_ARC],
    "F13": [GHOST_OUT.format("first", "(-0.330,-0.132)", "2,0"),
            OUT_ARC.format("first", "(-0.350,-0.253)", "(-0.684,-0.253)", "0.242 (centre line 0.253)",
                           "0.087 (centre line 0.121)")],
    "F14": [GHOST_FLAG.format("(-0.220,-2.266)", 2.5), GHOST_RED.format("fifth", "(0.110,-2.266)", 4, "fifth"),
            FLAG_ARC.format("(-0.200,-2.145)", "(0.134,-2.145)", "0.207 (centre line 0.218)", GHOSTS), NO_TIE_ARC],
    "F15": [GHOST_OUT.format("fifth", "(0.110,-1.452)", "4,4"),
            OUT_ARC.format("fifth", "(0.090,-1.573)", "(-0.684,-1.573)", "0.299 (centre line 0.310)",
                           "0.068 (centre line 0.102)")],
    "F19": [GHOST_FLAG.format("(-0.770,-2.266)", 0),
            FLAG_ARC.format("(-0.750,-2.145)", "(0.684,-2.145)", "0.180 (centre line 0.191)", GHOSTS)],
}
MARKS["F17"], MARKS["F18"] = MARKS["F12"], MARKS["F14"]


def pictures():
    """The Second Challenge's prop pictures (identity card, sections 4 and 3e), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch13.py: four blocks,
    three strips and a single call, eleven frames (F09-F19).  The ties below hold the
    transcription against this file's blackboard: a structural check, not a second proof."""
    calls = DIALOGUE.check_layout("Second", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("wishclothsteps", True, "", (DARK, "clip the tubes on; tubes out"),
         "every knob dark; a tube by every end-ring; every end-ring turned green"),
        ("wishclothsteps", False, "after=tubes out; from=off",
         (DARK, "push into the next ring; slide together; push into the next ring; slide together"),
         "into a red ring; slid out to its end-ring; on into the other; every bead still at rest"),
        ("wishclothsteps", False, "", (LIT, "push into the next ring; push into the next ring"),
         "laid afresh, lit at the start; climbed to a mark, a turn; turned at the marks, still lit at the end"),
        ("wishcloth", False, "step=push the flag", (BARE,), "run clear to the far end")]
    PC.check_captions(DIALOGUE, "Second", calls)
    PC.check_section(DIALOGUE, "Second", calls)
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Second")] == [[2.74]] * 4
    frames = [(c, fc, p, cap) for c in calls for fc, (p, cap) in zip(PC.frame_calls(c), PC.pictures(c))]
    names = ["F%02d" % i for i in range(9, 20)]
    assert len(frames) == len(names) == 11
    F = dict(zip(names, [p for _, _, p, _ in frames]))
    FC = dict(zip(names, [fc for _, fc, _, _ in frames]))
    for c, fc, p, cap in frames:                                     # a strip frame IS its single call
        assert PC.drawn(PC.picture(fc)) == PC.drawn(p) and PC.picture(fc).log == p.log, fc.text
    # the strips' frames, each the single call with the steps before it as operations
    assert [(fc.keyed, fc.opt) for fc in FC.values()] == [
        (True, ""), ("new", "step=clip the tubes on"), ("new", "clip the tubes on; step=tubes out"),
        (False, "tubes out; step=push into the next ring"),
        (False, "tubes out, push into the next ring; step=slide together"),
        (False, "tubes out, push into the next ring, slide together; step=push into the next ring"),
        (False, "tubes out, push into the next ring, slide together, push into the next ring; step=slide together"),
        (False, ""), (False, "step=push into the next ring"),
        (False, "push into the next ring; step=push into the next ring"), (False, "step=push the flag")]

    # the anchors (card sections 4 and 3e.4), in dialog.md; the raw file's following agree
    a1, a2, a3, a4 = DIALOGUE.anchors("Second")
    f1, f2, f3, f4 = DIALOGUE.following("Second")
    assert "Last she does the same to the flag's own end-ring: green, at rest." in a1
    assert "clips a tube from the packet onto the cord" in a1                             # D9
    assert "till a tube sits by every end-ring" in a1                                     # T1: every tube clipped first
    assert a1.index("till a tube sits by every end-ring") < a1.index("Then she slides the top tube outward")
    assert f1.startswith("**Tortoise:** A window reads the far end of its cord by the same code.")
    assert "not a tie anywhere, and not a red ring" in a2 and "the ring the tie came up to" in a2   # D11
    assert f2.startswith("**Achilles:** Then the web is empty.")
    assert "It has given its turn away twice and still has it." in a3 and "the tie swallowed behind it" in a3  # D13
    assert f3.startswith("**Achilles:** The flag's turn has gone to the first and the fifth")
    assert "every bead on the cloth is where it was before she pushed" in a4
    assert "the flag cord it ran along swallowed behind it" in a4                          # D15
    assert f4.startswith("**Tortoise:** If every wish-window stays dark")
    assert DIALOGUE.following("Second", DIALOGUE.raw) == [f1, f2, f3, f4]
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        assert "till a tube sits by every end-ring" in DIALOGUE.section("Second", path)
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
    # the keyed words named before their pictures (card 3e.6): end-ring, bead, tube
    text = DIALOGUE.read(DIALOGUE.dialog)
    k3 = text.index(calls[0].text)
    for word in ("end-ring", "bead", "tube"):
        assert word in text[:k3], word

    laid = PC.run_frame("cloth", "", "marks first and fifth", False)  # the web as laid (P1.4)
    a = (1, 0, 0, 0, 1, 0)                                            # Achilles's slip
    assert tuple(int(k in laid.after["marks"]) for k in range(1, 7)) == a
    rings = lambda pic: sorted((r["c"], r["x"], r["y"]) for r in pic.canon["things"] if r["kind"] == "ring")
    ties = lambda pic: sorted((pic.canon["things"][d["a"] - 1]["x"], pic.canon["things"][d["b"] - 1]["x"])
                              for d in pic.canon["cords"] if d["kind"] == "tie")

    # K3 (F09-F11): the end-rings red at rest (F09, keyed end-ring and bead); every tube clipped
    # on by its end-ring before any is slid (F10, keyed new: tube -- T1); every end-ring green
    # (F11).  No marks: a tube slides less than a head and its gap
    assert F["F09"].words == ("end-ring", "bead") and F["F09"].notes == []
    ends = [r for r in F["F09"].canon["things"] if r["kind"] == "end"]
    assert len(ends) == 7 and all((r["c"], r["t"], r["x"]) == ("red", 0, 0) for r in ends)
    assert rings(F["F09"]) == rings(laid) and ties(F["F09"]) == ties(laid) == [(2, 2), (4, 4)]
    h9, h10 = (F[n].head.replace("far=false clipped=true |", "far=false |").split("; ") for n in ("F09", "F10"))
    assert "clipped=true" in F["F10"].head and "clipped" not in F["F09"].head
    assert [q for q in h10 if q not in h9] == ["tube @0.62,%d a0" % y for y in range(7)]   # a tube by every end-ring
    assert [q for q in h9 if q not in h10] == []                                          # nothing else changed
    assert F["F10"].after["clipped"] and F["F10"].words == ("tube",)        # keyed new: the part it brought
    assert F["F11"].words == ()                                              # tubes out brings no new part
    for n, op in (("F10", "clip the tubes on"), ("F11", "tubes out")):
        p = F[n]
        assert p.notes == ["step " + op + " -> " + p.head,
                           "no marks: " + op + " shows what it changed (the tubes, the rings' colours, the folded band)"]
        assert p.mark_tikz == [] and p.arcs == [] and p.ghosts == []
    assert not any("keyed new" in l for p in F.values() for l in p.log)    # the Lua draws the word, logs none
    # F11: the convention -- every end-ring turned green at rest, the rings and ties as laid
    p21 = F["F11"]
    assert p21.head == (
        "scene=web marks=[first fifth] first=green0+green0@2tied second=green0 third=green0 fourth=green0 "
        "fifth=green0+green0@4tied sixth=green0 flag=green0@0[red0@2 red0@4] threaded=false pushed=false "
        "far=false | cols=8 rows=7 y=0..6 near=false folded=false; gaps=5.5; 1:end green 0 @0,0; "
        "2:ring green 0 @2,0; 3:end green 0 @0,1; 4:end green 0 @0,2; 5:end green 0 @0,3; 6:end green 0 @0,4; "
        "7:ring green 0 @4,4; 8:end green 0 @0,5; 9:end green 0 @0,6; 10:ring red 0 @2,6; 11:ring red 0 @4,6; "
        "tie 2-10 []; tie 7-11 []; cord loose-1 [6.2,0 0,0]; cord loose-3 [6.2,1 0,1]; cord loose-4 [6.2,2 0,2]; "
        "cord loose-5 [6.2,3 0,3]; cord loose-6 [6.2,4 0,4]; cord loose-8 [6.2,5 0,5]; cord loose-9 [7,6 0,6]")
    ends = [r for r in p21.canon["things"] if r["kind"] == "end"]
    assert len(ends) == 7 and all((r["c"], r["t"], r["x"]) == ("green", 0, 0) for r in ends)
    assert rings(p21) == rings(laid) and ties(p21) == ties(laid) == [(2, 2), (4, 4)]   # ties straight down

    # the moves' marks (card 3e.6, the readings): every ghost's place, every arc's ends and
    # side, every logged gap, and "no arc up a tie" where a copy climbs
    marked = [n for n in names if F[n].mark_tikz]
    assert marked == ["F12", "F13", "F14", "F15", "F17", "F18", "F19"]
    for n in marked:
        p = F[n]
        step = PC.options(FC[n])["step"]
        assert p.notes[0] == "step " + step + " -> " + p.head and p.notes[1:] == MARKS[n], n
        S, so = PC.cloth_scene(p.drawn)
        G = PC.ClothGeom(S, so)
        M0, M = p.before, p.after
        (arc,) = p.arcs
        if step == "slide together":            # the ring on a marked cord slid out into its end-ring
            (k,) = [k for k in range(1, 7) if M0["wish"][k].get("ring") and not M["wish"][k].get("ring")]
            assert [grid for _, grid, _ in PC.ghost_places(p)] == [(M0["wish"][k]["ring"]["x"], k - 1)]
            frm, to, row, side = M0["wish"][k]["ring"]["x"], 0, k - 1, -1          # beneath the row
        else:                                    # the flag's end-ring pushed along its cord
            gone = [r["x"] for r in M0["flag"]["rings"] if r not in M["flag"]["rings"]]
            assert [grid for _, grid, _ in PC.ghost_places(p)] == [(M0["flag"]["endx"], PC.FLAGROW)] + [
                (x, PC.FLAGROW) for x in gone]
            frm, to, row, side = M0["flag"]["endx"], M["flag"]["endx"], PC.FLAGROW, 1   # above the row
            # the ring it went into is gone from the flag, its copy at the top of its tie
            for x in gone:
                (k,) = [k for k in range(1, 7) if M0["wish"][k].get("ring") and M0["wish"][k]["ring"]["x"] == x]
                assert M0["wish"][k]["ring"]["tied"] and not M["wish"][k]["ring"]["tied"]
                assert M["wish"][k]["ring"]["t"] == M0["wish"][k]["ring"]["t"] + M0["flag"]["endt"]   # turns add
        # the arc's sense: it starts at the ghost and heads to where the ring went, on its side
        assert abs(arc.p0[0] - G.X(frm)) < G.RO and (arc.p1[0] - arc.p0[0]) * (G.X(to) - G.X(frm)) > 0, n
        assert abs(arc.p1[0] - G.X(to)) < 1.5 * G.RO, n
        assert np.isclose(arc.p0[1], G.Y(row) + side * G.D) and np.isclose(arc.p1[1], arc.p0[1]), n
        assert (arc.c1[1] - arc.p0[1]) * side > 0, n                           # bowed away from the row
        assert PC.mark_clearance(p) > 0.01, n                                    # inside the linen
        edge, centre = PC.cord_clearance(p)
        assert edge > 0.02 and centre > 0.1, (n, edge, centre)                   # off every cord and tie
        off = PC.twin(FC[n])                                                     # marks=off: the same frame
        assert PC.drawn(off) == PC.drawn(p) and off.mark_tikz == [] and off.ghosts == [] and off.arcs == [], n
        assert off.notes == ["step " + step + " -> " + p.head + " (no marks)"], n
    assert min(PC.cord_clearance(F[n])[0] for n in marked) > 0.068 - 0.001     # F15's, the least here

    # K4 (F12-F15): the push into a red ring, its copy up the tie, the ring slid out into its
    # end-ring; twice.  The frames are the strip's steps in the order of l.613
    assert F["F12"].after["partial"] and "partial=true" in F["F12"].head
    assert [F[n].after["flag"]["endx"] for n in ("F12", "F13", "F14", "F15")] == [2.5, 2.5, 4.5, 4.5]
    assert [k for k in range(1, 7) if F["F13"].after["wish"][k].get("ring")] == [5]
    # F15 (= P2.2): the trap (the flag crosswise dark): nothing comes back -- every end-ring
    # green at rest, and H on every wire, U_f, H on every wire takes |0000000> to |0000000>
    p22 = F["F15"]
    assert p22.head == (
        "scene=web marks=[first fifth] first=green0 second=green0 third=green0 fourth=green0 fifth=green0 "
        "sixth=green0 flag=green0@4.5[] threaded=false pushed=true far=false | cols=8 rows=7 y=0..6 near=false "
        "folded=false; gaps=5.5; 1:end green 0 @0,0; 2:end green 0 @0,1; 3:end green 0 @0,2; 4:end green 0 @0,3; "
        "5:end green 0 @0,4; 6:end green 0 @0,5; 7:end green 0 @4.5,6; cord loose-1 [6.2,0 0,0]; "
        "cord loose-2 [6.2,1 0,1]; cord loose-3 [6.2,2 0,2]; cord loose-4 [6.2,3 0,3]; cord loose-5 [6.2,4 0,4]; "
        "cord loose-6 [6.2,5 0,5]; cord loose-7 [7,6 4.5,6]")
    colours, bits = cloth_reading(p22.after)
    assert colours == {"green"} and bits == (0,) * 7
    assert np.allclose(turned_web(a, (0,) * 7), ket(*bits))
    assert not any(d["kind"] == "tie" for d in p22.canon["cords"]) and not any(r["c"] == "red" for r in p22.canon["things"])

    # K5 (F16-F18): laid afresh, the flag lit (its end-ring set halfway); each push carries the
    # half turn up a tie to a mark (turns add, 0 + 12), the flag keeping its own
    assert "flag=green12@0[red0@2 red0@4]" in F["F16"].head and F["F16"].words == ()
    assert "first=green0+green12@2 " in F["F17"].head and "fifth=green0+green0@4tied" in F["F17"].head
    # F18 (= P2.3): H U_f H |0000001> = |1000101>, read crosswise
    p23 = F["F18"]
    assert p23.head.startswith(
        "scene=web marks=[first fifth] first=green0+green12@2 second=green0 third=green0 fourth=green0 "
        "fifth=green0+green12@4 sixth=green0 flag=green12@4.5[] threaded=false pushed=true far=false | ")
    colours, bits = cloth_reading(p23.after)
    assert colours == {"green"} and bits == (1, 0, 0, 0, 1, 0, 1) == a + (1,)
    assert np.allclose(turned_web(a, (0,) * 6 + (1,)), ket(*bits))          # every entry exact
    assert not np.allclose(turned_web(a, (0,) * 6 + (1,)), ket(*(a + (0,))))   # the flag keeps its turn
    assert [(r["x"], r["t"]) for r in p23.canon["things"] if r["kind"] == "ring"] == [(2, 12), (4, 12)]
    # the frame between: after one push the first mark has its turn, the fifth not yet
    colours, bits = cloth_reading(F["F17"].after)
    assert bits == (1, 0, 0, 0, 0, 0, 1)

    # the cloth's wrong rule `gave nothing` (card 3e.1: not used in the dialogue, tested): the
    # push as it is, but no turn added to the ring a copy climbs into.  On K5's two pushes the
    # pale cloth is drawn with that ring again at full strength, its bead at rest, while the
    # flag gave it its half turn (12); the Lua refuses it where it would agree (a flag at rest)
    rule = PC.rule_comment("THE CLOTH'S STEP, ITS MARKS AND ITS WRONG RULE")
    assert "Then the flag gave nothing to anyone." in rule and "only where it differs" in rule
    for br, k in (("step=push into the next ring", 1), ("push into the next ring; step=push into the next ring", 5)):
        PC.TEX_ERRORS.clear()
        p = PC.run_frame("cloth", br + "; expects=gave nothing", LIT, False)
        assert not PC.TEX_ERRORS and p.wrong["telling"] == [k]
        assert p.wrong["wish"][k]["ring"]["t"] == 0 and p.after["wish"][k]["ring"]["t"] == 12
        assert p.notes[0] == "expects gave nothing (the cloth paler, the rings it names at full strength)"
        assert p.notes[2:] == ["the " + PC.ORDNAME[k - 1] + " square's ring at full strength over the pale cloth, "
                               "its bead at rest (the flag gave it nothing; it has 12)"]
        assert p.own[:2] == ["\\begin{scope}[transparency group, opacity=0.500]", "\\end{scope}"]
        assert p.words == () and p.mark_tikz == [] and p.arcs == []
        true = PC.run_frame("cloth", br, LIT, False)
        x = str(2 if k == 1 else 4)                                     # only that ring's bead differs
        assert p.head.split(" | ")[0] == true.head.split(" | ")[0].replace("+green12@" + x, "+green0@" + x)
    PC.TEX_ERRORS.clear()
    PC.run_frame("cloth", "tubes out; step=push into the next ring; expects=gave nothing", DARK, False)
    assert [s for s, _ in PC.TEX_ERRORS] == ["cloth-rule-rest"]
    PC.TEX_ERRORS.clear()

    # F19 (P2.4): no marks -- the flag runs clear to the far end; every wish-cord reads dark;
    # with no tie the blackboard's turned web is bare: |0000001> stays |0000001>
    p24 = F["F19"]
    assert p24.head == (
        "scene=web marks=[] first=green0 second=green0 third=green0 fourth=green0 fifth=green0 sixth=green0 "
        "flag=green12@7.0[] threaded=false pushed=true far=true | cols=8 rows=7 y=0..6 near=false folded=false; "
        "gaps=5.5; 1:end green 0 @0,0; 2:end green 0 @0,1; 3:end green 0 @0,2; 4:end green 0 @0,3; "
        "5:end green 0 @0,4; 6:end green 0 @0,5; 7:end green 12 @7,6; cord loose-1 [6.2,0 0,0]; "
        "cord loose-2 [6.2,1 0,1]; cord loose-3 [6.2,2 0,2]; cord loose-4 [6.2,3 0,3]; cord loose-5 [6.2,4 0,4]; "
        "cord loose-6 [6.2,5 0,5]; cord loose-7 [7,6 7,6]")
    colours, bits = cloth_reading(p24.after)
    assert colours == {"green"} and bits == (0,) * 6 + (1,)
    assert np.allclose(turned_web((0,) * 6, bits), ket(*bits))
    # after the push to the far end no flag cord shows beyond the end-ring: its record is
    # zero-length at the ring's x, and nothing (no cord, no whipped tip) is drawn for it
    flag = [d for d in p24.canon["cords"] if d["kind"] == "cord" and d["pts"][0][1] == PC.FLAGROW]
    assert len(flag) == 1 and flag[0]["pts"] == [[7, 6], [7, 6]] and CL.zero_length(flag[0])
    assert CL.drawn(p24.canon) == {"paths": 6, "tips": 6, "lashings": 6}       # the six wish-cords only
    assert CL.drawn(p22.canon) == {"paths": 7, "tips": 7, "lashings": 7}       # a flag cord left at 4.5

    # each marked frame with marks=off is its round-3 call (P2.1-P2.4), up to keying and words
    for n, (br, st, k) in (("F11", ("tubes out", DARK, True)),
                           ("F15", ("tubes out, push the flag, slide together", DARK, False)),
                           ("F18", ("push the flag", LIT, False)), ("F19", ("push the flag", BARE, False))):
        off = PC.twin(FC[n]) if "step" in PC.options(FC[n]) else F[n]
        assert PC.drawn(off) == PC.drawn(PC.run_frame("cloth", br, st, k)), n
    assert PC.drawn(PC.twin(FC["F19"])) == PC.drawn(PC.run_frame("cloth", "push the flag", BARE, False))

    # the fixed parts: the end-rings not pushed keep x = 0; the flag's end-ring moves right
    # and the flag cord's drawn start moves with it; no ring is added
    endx = lambda pic: [r["x"] for r in pic.canon["things"] if r["kind"] == "end" and r["y"] == PC.FLAGROW]
    assert [endx(F[n]) for n in names] == [[0], [0], [0], [2.5], [2.5], [4.5], [4.5], [0], [2.5], [4.5], [PC.FAR]]
    for pic in F.values():
        assert [r["x"] for r in pic.canon["things"] if r["kind"] == "end" and r["y"] < PC.FLAGROW] == [0] * 6
        start = [min(x for x, _ in d["pts"]) for d in pic.canon["cords"] if d["kind"] == "cord" and d["pts"][0][1] == PC.FLAGROW]
        assert start == endx(pic)
    counts = [len(F[n].canon["things"]) for n in names]
    assert counts == sorted(counts[:7], reverse=True) + counts[7:] and counts[0] == 11 and counts[6] == 7
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
