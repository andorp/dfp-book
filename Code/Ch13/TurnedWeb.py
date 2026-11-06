"""Challenge III: The Turned Web  (the trio; Bernstein-Vazirani; The Theorem, Twice; the master)
===========================================================================================
"The second and the fourth, and the flag. Not my marks at all. The casket's own."  -- Achilles

Claim: for every n, a and b, with f(x) = (a.x) xor b and |-> normalized,
(H^n (x) I) U_f (H^n (x) I) |0^n>|-> = (-1)^b |a>|->, exactly (Result 13.8, in Cleve
et al.'s one-query form).  The Hadamards turn the web:
(H (x) H) CNOT_{1->2} (H (x) H) = CNOT_{2->1}, and H^{n+1} U_f H^{n+1} =
Z_flag^b prod_{a_i=1} CNOT_{flag->i}, a fan-out, with no scalar in pictures
(Result 13.7).  The thirteen-line master, The Turned Web (a = 010100, b = 0, symmetric
flag form, the two red targets unfused), is exactly 2^{5/2} |010100>|1>, read top
wire first (Result 13.13).  The Genie's rules G1 (one passage is one application of
U_f, whatever it is set to) and G2 (a plain wish returns one definite reading) hold.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
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
import props_office as OF

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Z = np.diag([1, -1]).astype(complex)
PLUS, MINUS = H @ ket(0), H @ ket(1)

CASKET = (0, 1, 0, 1, 0, 0)     # the Genie's secret: rods 2 and 4 marked (not a palindrome)
SLIP = (1, 0, 0, 0, 1, 0)       # Achilles's slip, the rehearsal box


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


def linear(a, b=0):
    return lambda x: dot(a, x) ^ b


def Hn(n):
    return reduce(np.kron, [H] * n, np.eye(1))


def flag_to(i, n):
    """CNOT with control the flag, target rod i (0-based)."""
    M = np.zeros((2 ** (n + 1), 2 ** (n + 1)), dtype=complex)
    for x in strings(n):
        for y in (0, 1):
            x2 = tuple(xj ^ (y if j == i else 0) for j, xj in enumerate(x))
            M[index(x2 + (y,)), index(x + (y,))] = 1
    return M


def headline(a, b):
    """(H^n (x) I) U_f (H^n (x) I) |0^n>|->."""
    n = len(a)
    layer = np.kron(Hn(n), I2)
    return layer @ U(linear(a, b), n) @ layer @ np.kron(ket(*[0] * n), MINUS)


def blackboard():
    rng = np.random.default_rng(1303)
    assert np.isclose(np.linalg.norm(MINUS), 1)                              # |-> normalized

    # --- Result 13.8, the headline: every a and both b at n = 4, the casket's secret and
    #     random secrets at n = 6.  Exact, the sign (-1)^b included.
    cases = [(a, b) for a in strings(4) for b in (0, 1)]
    cases += [(CASKET, 0), (CASKET, 1), (SLIP, 0)]
    cases += [(tuple(int(t) for t in rng.integers(0, 2, 6)), int(rng.integers(0, 2))) for _ in range(6)]
    for a, b in cases:
        out = headline(a, b)
        assert np.allclose(out, (-1) ** b * np.kron(ket(*a), MINUS))
        if b:
            assert not np.allclose(out, np.kron(ket(*a), MINUS))              # the sign is real
        if a != a[::-1]:
            assert not np.allclose(abs(out), abs(np.kron(ket(*a[::-1]), MINUS)))  # top wire first
    # section 13.3's worked case, a = 101: U_f |+++>|-> = |-+->|->, and the rods read 101
    U101 = U(linear((1, 0, 1)), 3)
    assert np.allclose(U101 @ reduce(np.kron, [PLUS, PLUS, PLUS, MINUS]),
                       reduce(np.kron, [MINUS, PLUS, MINUS, MINUS]))
    assert np.allclose(np.kron(Hn(3), I2) @ reduce(np.kron, [MINUS, PLUS, MINUS, MINUS]),
                       np.kron(ket(1, 0, 1), MINUS))

    # Result 13.5 at n = 4 (the Dirac's Blackboard sum), and its bit-by-bit factoring
    for c in strings(4):
        s = sum((-1) ** dot(c, x) for x in strings(4))
        assert s == 16 * (not any(c)) == np.prod([1 + (-1) ** ci for ci in c])

    # --- Result 13.7, the turned tie and the turned web
    CNOT12 = U(lambda x: x[0], 1)                                             # control top
    CNOT21 = flag_to(0, 1)                                                    # control bottom
    HH = np.kron(H, H)
    assert np.allclose(HH @ CNOT12 @ HH, CNOT21)
    assert not np.allclose(HH @ CNOT12 @ HH, CNOT12)                         # it has turned
    # the section-13.3 feed-throughs: |01> -> |11>, |10> -> |10>
    assert np.allclose(HH @ CNOT12 @ HH @ ket(0, 1), ket(1, 1))
    assert np.allclose(HH @ CNOT12 @ HH @ ket(1, 0), ket(1, 0))
    for a, b in [(a, b) for a in strings(4) for b in (0, 1)] + [(CASKET, 0), (CASKET, 1), (SLIP, 0)]:
        n = len(a)
        turned = Hn(n + 1) @ U(linear(a, b), n) @ Hn(n + 1)
        fan = np.linalg.matrix_power(np.kron(np.eye(2 ** n), Z), b)
        for i in range(n):
            if a[i]:
                fan = fan @ flag_to(i, n)
        assert np.allclose(turned, fan)
        if any(a):
            assert not np.allclose(turned, U(linear(a, b), n))                # not the untouched web
            # Hadamards on the rods only do NOT turn it
            assert not np.allclose(np.kron(Hn(n), I2) @ U(linear(a, b), n) @ np.kron(Hn(n), I2), fan)

    # --- Result 13.13, the master in the symmetric flag form
    Uc = U(linear(CASKET), 6)
    out = Hn(7) @ Uc @ Hn(7) @ ket(0, 0, 0, 0, 0, 0, 1)
    assert np.allclose(out, ket(0, 1, 0, 1, 0, 0, 1))
    assert np.isclose(out[0b0101001], 1) and np.isclose(out[0b0010101], 0)   # not read upside down
    assert np.allclose(Hn(7) @ Uc @ Hn(7), flag_to(1, 6) @ flag_to(3, 6))
    # the cloth in III: the slip's web, turned, the flag pushed through: first, fifth, flag
    Us = U(linear(SLIP), 6)
    assert np.allclose(Hn(7) @ Us @ Hn(7) @ ket(0, 0, 0, 0, 0, 0, 1), ket(1, 0, 0, 0, 1, 0, 1))
    # "Not my marks at all. The casket's own."
    assert not np.allclose(out, ket(1, 0, 0, 0, 1, 0, 1))

    # --- the Genie's rules
    # G1: one passage is one application of U_f, whatever the knobs and collars are set to:
    # on any input state -- here a random complex one, all collars crosswise or not -- the
    # passage is the single linear map U_f (a superposed query is one query)
    psi = random_state(rng, 128)
    assert not np.allclose(psi, psi.conj())
    for collars in (np.eye(128), Hn(7)):
        passage = collars @ Uc @ collars
        assert np.allclose(passage @ psi, sum(psi[k] * passage[:, k] for k in range(128)))
    # G2: a plain wish (every collar upright) returns the knobs in the wish-windows and ONE
    # reading in the flag window: y xor a.x, a definite basis state, for every setting
    for x in strings(6):
        for y in (0, 1):
            o = Uc @ ket(*x, y)
            assert np.allclose(o, ket(*x, y ^ dot(CASKET, x)))
    # the crosswise wish itself: the windows read a with probability 1
    probs = np.abs(out) ** 2
    assert np.isclose(probs[index(CASKET + (1,))], 1)

    # --- the hazard (section 13.3, "Why this definition"): the rods crosswise and the flag
    #     upright.  For every a != 0 at n = 4: with the flag at |0>,
    #     1/2(|0^n> + |a>)|0> + 1/2(|0^n> - |a>)|1>; at |1> the branches change places.
    #     Rods 0^n or a at 1/2 each, the flag an independent fair coin.  NOT a claim of the
    #     dialogue (the Tortoise says only "That would gamble the only wish"); asserted to
    #     document why the wish is not set that way.
    n = 4
    for a in strings(n):
        if not any(a):
            continue
        z, A = ket(*[0] * n), ket(*a)
        net = lambda y: np.kron(Hn(n), I2) @ U(linear(a), n) @ np.kron(Hn(n), I2) @ np.kron(z, ket(y))
        assert np.allclose(net(0), 0.5 * np.kron(z + A, ket(0)) + 0.5 * np.kron(z - A, ket(1)))
        assert np.allclose(net(1), 0.5 * np.kron(z - A, ket(0)) + 0.5 * np.kron(z + A, ket(1)))
        for y in (0, 1):
            p = (np.abs(net(y)) ** 2).reshape(2 ** n, 2)
            joint = np.array([[p[index((0,) * n), fy], p[index(a), fy]] for fy in (0, 1)])
            assert np.allclose(joint, 0.25)                           # four outcomes, 1/4 each
            # the dialogue's case: the flag KNOB upright, its window crosswise -- still a
            # gamble: the rods read a only half the time
            q = (np.abs(np.kron(np.eye(2 ** n), H) @ net(y)) ** 2).reshape(2 ** n, 2)
            assert np.isclose(q[index(a)].sum(), 0.5) and np.isclose(q[0].sum(), 0.5)

    pictures()


# --- the prop pictures --------------------------------------------------------------------

GENIE_LINES = (                 # the approved text (commit a98c028), word for word
    'The Office of Wishes. The lamp has been rubbed once, which is correct.',
    'One wish is one passage of the casket, whatever its knobs and collars are set to. A wish with every collar crosswise is still one wish.',
    "The Office anticipates. A plain wish, every collar upright, returns the knobs' settings in the wish-windows and one reading in the flag window: the flag knob's setting, turned once for each marked knob set lit. That is one answer.",
    'Is not a wish. Setting is free; the Office charges for passage. There is one wish.',
    "Referred. *[He inclines his head as though listening up a stairwell, and straightens at once.]* The Meta-Genie regrets. The marks are the casket's to tell, and the casket tells only by passage. Nothing has been spent.",
    'Any request other than one passage of the casket as it is now set is a meta-wish, and it goes up. The Meta-Genie referred yours to the one above, who referred it higher, and so on all the way; the answer came down all the way at once. The Office has no stairs.',
    'Referred. *[The same tilt of the head, no longer than a blink.]* The reply has come down: there is one wish. The Meta-Genie adds that a wish for wishes is a wish about wishes, and is not charged.',
    'Referred. *[A blink.]* The lid lifts on a spent casket and on no other. Your wish is untouched.',
    'The Office is in no hurry. The Office is never in a hurry.',
    'That is not a wish. That is furniture. Carry it level.',
    'Every collar crosswise, every knob as set: one passage. One wish.',
    'Granted.',
    'The casket is spent. A secret read is no longer a secret, and the lid may be lifted.',
    'The wish has been spent in accordance with the Terms. The Office does not collect spent boxes.',
)

# Section 1a's thirteen lengths inside the casket, top to bottom: (what, one end, the other)
THIRTEEN = (("cord", "knob 1", "window 1"), ("cord", "knob 2", "green 2"), ("cord", "green 2", "window 2"),
            ("tie", "green 2", "red 1"), ("cord", "knob 3", "window 3"), ("cord", "knob 4", "green 4"),
            ("cord", "green 4", "window 4"), ("tie", "green 4", "red 2"), ("cord", "knob 5", "window 5"),
            ("cord", "knob 6", "window 6"), ("cord", "knob 7", "red 1"), ("cord", "red 1", "red 2"),
            ("cord", "red 2", "window 7"))


def layer(rows):
    """H on the wires whose bit is set, I on the rest (top wire first)."""
    return reduce(np.kron, [H if r else I2 for r in rows], np.eye(1))


def casket_model(knobs, kc, wc):
    """A casket model with every row set by hand (the macro's text sets collars in four
    groups only; the passage itself takes any setting)."""
    rows = range(1, PC.NROW + 1)
    return PC.Casket({k: bool(knobs[k - 1]) for k in rows}, {k: bool(kc[k - 1]) for k in rows},
                     {k: bool(wc[k - 1]) for k in rows})


def bits_of(windows):
    return tuple(int(windows[k]) for k in range(1, PC.NROW + 1))


def cloth_reading(M):
    """The web on the cloth read at its end-rings: each wish-cord's end-ring with any ring
    slid along its cord (turns add), then the flag's; a half turn is lit."""
    rows = [M["wish"][k] for k in range(1, 7)] + [M["flag"]]
    colours = {c["endc"] for c in rows}
    bits = tuple(int((c["endt"] + (c["ring"]["t"] if c.get("ring") else 0)) % 24 == 12) for c in rows)
    return colours, bits


def pictures():
    """The Third Challenge's prop pictures (identity card, section 4), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch13.py.  The
    ties below hold the transcription (the Lua's own passage, simulated on its 128
    entries) against this file's blackboard: a structural check, not a second proof."""
    calls = DIALOGUE.check_layout("Third", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("wishcloth", False, "thread the tubes, slide the tubes in", ("marks first and fifth; wish-knobs dark; flag lit",),
         "colours swapped, the end-rings unchanged"),
        ("wishcloth", False, "thread the tubes, slide the tubes in, push the flag, slide together, on to the far end",
         ("marks first and fifth; wish-knobs dark; flag lit",), "carried up to the marks, and on"),
        ("wishcasket", True, "windows", ("collars crosswise; flag knob lit; granted",), "the gleams come up, the seal parted"),
        ("wishcasket", True, "inside", ("collars crosswise; flag knob lit; granted; lid lifted",), "inside, from the top down")]
    PC.check_captions(DIALOGUE, "Third", calls)
    PC.check_section(DIALOGUE, "Third", calls)
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Third")] == [[2.74], [2.74], [2.82], [2.82]]

    # the anchors (card section 4); P3.4 is followed by an ambient paragraph in dialog.md,
    # so the raw file's following paragraph is checked too
    a1, a2, a3, a4 = DIALOGUE.anchors("Third")
    f1, f2, f3, f4 = DIALOGUE.following("Third")
    assert "and now the green is on the flag" in a1
    assert "its tubes go into it and are gone, as at the end-rings" in a1                          # D18
    assert f1.startswith("**Achilles:** The ties run the other way.")
    assert "red and at rest from end to end" in a2 and "into one red end-ring at the cord's start" in a2   # D19
    assert f2.startswith("**Tortoise:** No tubes left, so read them plainly")
    assert "What has held: every setting, every collar, and the lid, still down." in a3
    assert f3.startswith("**Achilles:** The second and the fourth, and the flag.")
    assert "every gleam can now be followed back, length by length, to the knob it came from" in a4
    assert "below the upper and further across" in a4                                              # D21
    assert f4.startswith("*[A smell comes up out of the open casket")
    r1, r2, r3, r4 = DIALOGUE.following("Third", DIALOGUE.raw)
    assert [r1, r2, r3] == [f1, f2, f3] and r4.startswith("**Achilles:** Green rings at the second and the fourth knobs")
    section = DIALOGUE.section("Third")
    for phrase in ("gathering the cloth clear, the rings into their box and the string into her work-bag, and "   # D16
                   "laying the slip's web out afresh on the cloth",
                   "She clips tubes from the packet onto the web.", "On each tie she clips a pair of tubes,",     # D17
                   "a quarter-turn clockwise as he faces it, its nib going from the top round to the side"):     # D20
        assert phrase in section, phrase
    assert "threads tubes onto the web" not in section and "beneath the upper." not in section
    # the Genie's lines unchanged, in order, in both files
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        said = re.findall(r"^\*\*Genie:\*\* (.*)$", DIALOGUE.section("Third", path), re.M)
        assert tuple(said) == GENIE_LINES, path
    # the places: no picture among the Genie's arrival lines or at the Escher moment, and
    # the window face only after "Granted."
    paras = DIALOGUE.paragraphs("Third")
    at = lambda text: [i for i, p in enumerate(paras) if text in p]
    blocks = [i for i, p in enumerate(paras) if p.startswith("\\begin{center}")]
    (arrive,), (furniture,), (granted,) = at("The Office of Wishes. The lamp has been rubbed once"), \
        at("Carry it level."), at("**Genie:** Granted.")
    (escher,) = at("\\textup{Belvedere}")
    assert len(blocks) == 4 and not any(arrive <= i <= furniture for i in blocks)
    assert blocks[0] < escher and blocks[1] > escher + 2 and blocks[2] > granted

    p31, p32, p33, p34 = (PC.picture(c) for c in calls)

    # P3.1: the turned web -- the tubes swap every ring's colour (the rings on the marked
    # cords red, those on the flag cord green), the end-rings as set
    assert p31.dump.startswith(
        "scene=web marks=[first fifth] first=red0+red0@2tied second=red0 third=red0 fourth=red0 "
        "fifth=red0+red0@4tied sixth=red0 flag=red12@0[green0@2 green0@4] threaded=false pushed=false far=false | ")
    rings = sorted((r["c"], r["x"], r["y"]) for r in p31.cloth["things"] if r["kind"] == "ring")
    assert rings == [("green", 2, 6), ("green", 4, 6), ("red", 2, 0), ("red", 4, 4)]
    laid = PC.wishcloth("", "marks first and fifth")                       # P1.4: the same places
    assert sorted((r["x"], r["y"]) for r in laid.cloth["things"]) == sorted((x, y) for _, x, y in rings)

    # P3.2: the push read plainly -- red at rest dark, red a half lit: the marked cords and
    # the flag lit, which is H U_f H |0000001> = |1000101> on the blackboard, amplitude +1
    assert p32.dump.startswith(
        "scene=web marks=[first fifth] first=red12 second=red0 third=red0 fourth=red0 fifth=red12 sixth=red0 "
        "flag=red12@7.0[] threaded=false pushed=true far=true | ")
    colours, bits = cloth_reading(p32.state)
    assert colours == {"red"} and bits == SLIP + (1,)                       # P1.2's inked squares, and the flag
    out = Hn(7) @ U(linear(SLIP), 6) @ Hn(7) @ ket(*[0] * 6, 1)
    assert np.allclose(out, ket(*bits))
    assert not any(d["kind"] == "tie" for d in p32.cloth["cords"])
    flag = [d for d in p32.cloth["cords"] if d["kind"] == "cord" and d["pts"][0][1] == PC.FLAGROW]
    assert len(flag) == 1 and CL.zero_length(flag[0])                      # swallowed whole ...
    assert CL.drawn(p32.cloth) == {"paths": 6, "tips": 6, "lashings": 6}   # ... and nothing drawn for it

    # the passage IS the blackboard's (H on the window-collared rows) U_f (H on the knob-
    # collared rows), knob side first: all 128 knob settings, every collar upright and every
    # collar crosswise, through the macro's own words
    Uf = U(linear(CASKET), 6)
    assert PC.MARKS == (2, 4) and tuple(int(k in PC.MARKS) for k in range(1, 7)) == CASKET
    for cw in (0, 1):
        for knobs in strings(7):
            lit = [PC.ORDNAME[i] for i in range(6) if knobs[i]]
            text = "; ".join(["collars " + ("crosswise" if cw else "upright"),
                              "wish-knobs " + (" and ".join(lit) + " lit" if lit else "dark"),
                              "flag knob " + ("lit" if knobs[6] else "dark"), "granted"])
            K = PC.casket_state("windows", text)
            assert not PC.TEX_ERRORS and bits_of(K.knobs) == knobs
            v = layer([cw] * 7) @ Uf @ layer([cw] * 7) @ ket(*knobs)
            j = int(np.argmax(np.abs(v)))
            assert np.isclose(abs(v[j]), 1) and bits_of(K.windows) == strings(7)[j]
    # P3.3: the casket's own marks come up -- windows=dldlddl is headline(a, b) at a = 010100
    # with the flag window crosswise too: |0101001>, amplitude +1
    assert p33.dump == ("knobs=ddddddl kcollars=ccccccc wcollars=ccccccc granted=true windows=dldlddl "
                        "seal=parted lid=down")
    assert p33.labels == ("window", "collar", "flag window", "lid", "seal")
    assert bits_of(p33.state.windows) == CASKET + (1,)
    assert np.allclose(np.kron(np.eye(64), H) @ headline(CASKET, 0), ket(*CASKET, 1))
    assert [g for *_, g in p33.shown.windows] == [bool(b) for b in CASKET + (1,)]
    # a plain wish (not pictured): every collar upright, the second and third knobs lit
    K = PC.casket_state("", "wish-knobs second and third lit; granted")
    assert PC.casket_dump(K).split()[4] == "windows=dlldddl"
    assert np.allclose(Uf @ ket(0, 1, 1, 0, 0, 0, 0), ket(*bits_of(K.windows)))
    # the gamble: the passage has no one reading EXACTLY when the blackboard's state is not
    # a basis state -- every collar pattern (2^14), at two knob settings
    count = 0
    for knobs in ((0, 0, 0, 0, 0, 0, 1), (0, 1, 1, 0, 0, 0, 0)):
        after = np.array([Uf @ layer(kc) @ ket(*knobs) for kc in strings(7)]).T    # column per knob pattern
        for wc in strings(7):
            final = np.abs(layer(wc) @ after)
            basis = np.isclose(final.max(axis=0), 1)                    # per knob pattern
            top = final.argmax(axis=0)
            for c, kc in enumerate(strings(7)):
                w = PC.casket_passage(casket_model(knobs, kc, wc))
                assert (w is not None) == basis[c]
                if basis[c]:
                    count += 1
                    assert bits_of(w) == strings(7)[top[c]]
    assert count == 160
    rng = np.random.default_rng(13033)
    for _ in range(200):
        knobs, kc, wc = (tuple(int(b) for b in rng.integers(0, 2, 7)) for _ in range(3))
        v = layer(wc) @ Uf @ layer(kc) @ ket(*knobs)
        assert (PC.casket_passage(casket_model(knobs, kc, wc)) is not None) == np.isclose(np.max(np.abs(v)), 1)
    # l.570, the flag knob's collar left upright: the macro refuses it, and on the blackboard
    # the likeliest reading has probability 1/2
    PC.casket_state("windows", "collars crosswise; flag knob's collar upright; flag knob lit; granted")
    assert [s for s, _ in PC.TEX_ERRORS] == ["casket-gamble"]
    PC.TEX_ERRORS.clear()
    v = layer([1] * 7) @ Uf @ layer([1] * 6 + [0]) @ ket(*[0] * 6, 1)
    assert np.isclose(np.max(np.abs(v) ** 2), 0.5)

    # P3.4: the inside -- section 1a's thirteen lengths by name, every one knotted at both
    # ends, the ties behind; the lower red ring below the upper and further across; every
    # gleam followed back to its window
    assert p34.dump == p33.dump.replace("lid=down", "lid=lifted")
    assert p34.labels == ("knob", "window", "ring", "tie")
    w = p34.shown
    assert tuple((kind, a, b) for kind, a, b, *_ in w.lengths) == tuple(sorted(THIRTEEN, key=lambda t: (t[0] != "tie",)))
    assert sorted((kind, a, b) for kind, a, b, *_ in w.lengths) == sorted(THIRTEEN)
    assert all(behind == (kind == "tie") for kind, _, _, *_, behind in w.lengths)
    place = {name: (x, y) for name, _, x, y in w.rings}
    assert sorted(place) == ["green 2", "green 4", "red 1", "red 2"]
    assert place["red 2"][1] < place["red 1"][1] and place["red 2"][0] > place["red 1"][0]   # D21
    assert [place["green %d" % k][1] for k in PC.MARKS] == [PC.ROWY[k - 1] for k in PC.MARKS]
    assert [g for _, g in w.backs] == [g for *_, g in p33.shown.windows]
    # the words "ring" and "tie" in the wide gap between the last wish row and the flag row,
    # either side of the first tie: each word's box clear of every length (its body and its
    # knots) and of every ring, by a positive gap
    yg = (PC.ROWY[PC.NROW - 2] + PC.ROWY[PC.FLAG - 1]) / 2
    assert np.isclose(yg, 0.515)
    assert [(word, y, anchor) for word, _, y, anchor in w.labels[2:]] == [("ring", yg, "east"), ("tie", yg, "west")]
    # (and with the 1pt inner sep, which nothing draws, taken off: by at least 0.02 cm)
    least, least_ink = 1.0, 1.0
    e = PC.INNER_SEP * PC.PT
    for word, x, y, anchor in w.labels[2:]:
        node = PC.label_box(word, x, y, anchor)
        (x0, y0), (x1_, y1_) = node[0], node[2]
        for box, ink in (((node, 0.0), False),
                         (([(x0 + e, y0 + e), (x1_ - e, y0 + e), (x1_ - e, y1_ - e), (x0 + e, y1_ - e)], 0.0), True)):
            here = 1.0
            for kind, a, b, x1, y1, x2, y2, behind in w.lengths:
                cord = OF.cord(x1, y1, x2, y2, behind)
                here = min([here, OF.gap(box, cord["body"])] + [OF.gap(box, k) for k in cord["knots"]])
            for _, _, x, y in w.rings:
                here = min(here, OF.gap(box, OF.ring(x, y)["ring"]))
            if ink:
                least_ink = min(least_ink, here)
            else:
                least = min(least, here)
    assert least > 0, least
    assert least_ink >= 0.02, least_ink
    # no length passes through a ring: each ends at a ring's rim, never crossing one
    for kind, a, b, x1, y1, x2, y2, _ in w.lengths:
        for name, (x, y) in place.items():
            d = OF._seg_dist((x, y), (x1, y1), (x2, y2))
            assert d > PC.RO_IN - 1e-9 if name in (a, b) else d > PC.RO_IN + 0.02, (a, b, name)

    # the fixed parts: in every view the rows are the same, the column at the face's middle;
    # "granted" changes only the windows and the seal, "lid lifted" only the lid
    before = PC.casket_view(PC.casket_state("windows", "collars crosswise; flag knob lit"), True)
    after_ = p33.shown
    assert before.collars == after_.collars and before.labels == after_.labels and before.plate == after_.plate
    assert before.windows != after_.windows and len(before.seal) == 1 and len(after_.seal) == 2
    lifted = PC.casket_view(PC.casket_state("windows", "collars crosswise; flag knob lit; granted; lid lifted"))
    assert lifted.collars == after_.collars and lifted.windows == after_.windows
    assert (lifted.lid, after_.lid) == ("lifted", "down") and lifted.plate is None
    assert [y for _, _, y, _ in after_.collars] == list(PC.ROWY)
    assert {x for _, x, _, _ in after_.collars} == {sum(PC.WINDOWS_FACE) / 2}
    # the clearances on the window face: every nib clears its pane and the neighbour collars;
    # the parted seal clears the plate
    for k, x, y, cw in after_.collars:
        nib = OF.collar(x, y, cw)["nib"]
        assert OF.gap(nib, OF.window(x, y, True)["pane"]) > 0.02
        for j in (k - 1, k + 1):
            if 1 <= j <= PC.NROW:
                assert OF.gap(nib, OF.collar(x, PC.ROWY[j - 1], cw)["ring"]) > 0.004
    for q in after_.seal:
        assert OF.gap((list(q), OF.half("quad")), OF.plate(*after_.plate)) > 0.05
    assert not PC.TEX_ERRORS


# --- pictures ---------------------------------------------------------------------------

def master(a, b=0, cleve=False):
    """One instance of the headline as a state.  Symmetric form (cleve=False): a red point
    on each rod, a red pi point on the flag, a yellow box (Hadamard edge) on every wire on
    both sides of the web.  Cleve's form: the flag enters as a green pi point with no
    yellow boxes.  The web: a green control on every marked rod, each tied to its OWN red
    target on the flag (unfused).  An unmarked rod is ONE plain edge, point to output: the
    drawing shows two yellow boxes on it, which cancel (HH = I), and a line through two
    collars is still one line.  No scalar added."""
    n = len(a)
    g = zx.Graph()
    outs = [g.add_vertex(VT.BOUNDARY, r, 10) for r in range(n + 1)]
    for i in range(n):
        p = g.add_vertex(VT.X, i, 0)                                   # red point: sqrt2 |0>
        if a[i]:
            c = g.add_vertex(VT.Z, i, 2 + i)
            g.add_edge((p, c), ET.HADAMARD)
            g.add_edge((c, outs[i]), ET.HADAMARD)
        else:
            g.add_edge((p, outs[i]))
    if cleve:
        chain = [g.add_vertex(VT.Z, n, 0, phase=Fraction(1))]          # green pi point: sqrt2 |->
    else:
        chain = [g.add_vertex(VT.X, n, 0, phase=Fraction(1))]          # red pi point: sqrt2 |1>
    ctrls = [v for v in g.vertices() if g.type(v) == VT.Z and g.qubit(v) < n]
    for c in sorted(ctrls, key=g.qubit):
        t = g.add_vertex(VT.X, n, 2 + g.qubit(c))
        g.add_edge((c, t))                                             # the tie
        chain.append(t)
    if b:
        chain.append(g.add_vertex(VT.X, n, 9, phase=Fraction(1)))      # red pi dot: X
    chain.append(outs[n])
    edges = list(zip(chain, chain[1:]))
    for k, e in enumerate(edges):
        boxed = not cleve and (k == 0 or k == len(edges) - 1)
        g.add_edge(e, ET.HADAMARD if boxed else ET.SIMPLE)
    if not cleve and len(edges) == 1:                                  # no dot on the flag:
        g.set_edge_type(g.edge(*edges[0]), ET.SIMPLE)                  # its two boxes cancel
    g.set_inputs(()); g.set_outputs(tuple(outs))
    return g


def outward(g):
    """The tubes slid OUTWARD onto the end-rings: the named colour-change rule on every
    input point (a red point behind a yellow box is a green point, H sqrt2|0> = sqrt2|+>)."""
    h = g.copy()
    for v in list(h.vertices()):
        if h.type(v) == VT.X and h.vertex_degree(v) == 1 \
                and h.edge_type(list(h.incident_edges(v))[0]) == ET.HADAMARD:
            assert simp.check_color_change(h, v) and simp.color_change(h, v)
    return h


def web_between_boxes(a, b=0, boxed=True):
    """The web as an operator (inputs and outputs rod 1..rod n, flag), with a yellow box on
    all 2(n+1) outer legs if boxed.  Unmarked rods pass through a phase-free green dot, so
    that each yellow box has an edge to sit on."""
    n = len(a)
    et = ET.HADAMARD if boxed else ET.SIMPLE
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in range(n + 1)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 10) for r in range(n + 1)]
    chain = [ins[n]]
    for i in range(n):
        c = g.add_vertex(VT.Z, i, 2 + i)
        g.add_edge((ins[i], c), et); g.add_edge((c, outs[i]), et)
        if a[i]:
            t = g.add_vertex(VT.X, n, 2 + i)
            g.add_edge((c, t))
            chain.append(t)
    if b:
        chain.append(g.add_vertex(VT.X, n, 9, phase=Fraction(1)))
    chain.append(outs[n])
    if len(chain) == 2:
        chain.insert(1, g.add_vertex(VT.Z, n, 5))
    for k, e in enumerate(zip(chain, chain[1:])):
        g.add_edge(e, et if k in (0, len(chain) - 2) else ET.SIMPLE)
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    return g


def fan_out(a, b=0):
    """The turned web: the flag's green dots tied to a red dot on each marked rod; a green
    pi dot (Z) on the flag if b = 1.  No yellow boxes, no scalar."""
    n = len(a)
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in range(n + 1)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 10) for r in range(n + 1)]
    chain = [ins[n]]
    for i in range(n):
        if a[i]:
            r = g.add_vertex(VT.X, i, 2 + i)
            s = g.add_vertex(VT.Z, n, 2 + i)
            g.add_edges([(ins[i], r), (r, outs[i]), (r, s)])
            chain.append(s)
        else:
            g.add_edge((ins[i], outs[i]))
    if b:
        chain.append(g.add_vertex(VT.Z, n, 9, phase=Fraction(1)))
    chain.append(outs[n])
    g.add_edges(list(zip(chain, chain[1:])))
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    return g


def ties(g):
    """The ties of a web: edges between two dots on different wires, as pairs of wires
    (PyZX calls the wire coordinate `qubit`)."""
    out = set()
    for e in g.edges():
        u, x = g.edge_st(e)
        if VT.BOUNDARY not in (g.type(u), g.type(x)) and g.qubit(u) != g.qubit(x):
            out.add(frozenset((g.qubit(u), g.qubit(x))))
    return out


def scaled(g, power):
    h = g.copy()
    h.scalar.add_power(power)
    return h


def points_after_reduce(g):
    """full_reduce a copy; check that each output hangs on its own one-legged dot and
    return (phase, edge type) per wire, top wire first."""
    h = g.copy()
    simp.full_reduce(h)
    assert same_picture(h, g)                                      # the value is kept
    got = []
    for o in h.outputs():
        (v,) = h.neighbors(o)
        assert h.type(v) != VT.BOUNDARY and h.vertex_degree(v) == 1
        got.append((h.phase(v), h.edge_type(h.edge(v, o))))
    assert h.num_vertices() == 2 * len(h.outputs())
    return got


def sketchbook():
    # conventions: a Hadamard edge (the yellow box) is exactly H, and two collars cancel
    e = zx.Graph()
    i, o = e.add_vertex(VT.BOUNDARY, 0, 0), e.add_vertex(VT.BOUNDARY, 0, 1)
    e.add_edge((i, o), ET.HADAMARD)
    e.set_inputs((i,)); e.set_outputs((o,))
    assert np.allclose(matrix_of(e), H) and np.allclose(H @ H, I2)

    # (a) The Turned Web, built: 18 vertices, 13 edges (n + 3w + 1 = 6 + 6 + 1), the two red
    #     targets unfused, one per CNOT.  Scalar: seven points give 2^{7/2}, two bare ties
    #     2^{-1}, so add_power(-5) (2^{-5/2}) leaves exactly |010100>|1>.
    g = master(CASKET)
    assert g.num_vertices() == 18 and g.num_edges() == 13 == 6 + 3 * 2 + 1
    assert len([v for v in g.vertices() if g.type(v) == VT.X and g.qubit(v) == 6 and g.vertex_degree(v) == 3]) == 2
    v = matrix_of(scaled(g, -5)).ravel()
    assert np.allclose(v, ket(0, 1, 0, 1, 0, 0, 1))
    assert np.isclose(v[0b0101001], 1) and np.isclose(v[0b0010101], 0)      # top wire first
    assert np.count_nonzero(~np.isclose(v, 0)) == 1
    assert not np.allclose(matrix_of(g).ravel(), v)                           # the scalar is drawn
    assert not np.allclose(matrix_of(scaled(g, -4)).ravel(), v)
    assert np.allclose(matrix_of(g).ravel(), 2 ** 2.5 * ket(0, 1, 0, 1, 0, 0, 1))

    # (c) DERIVED: full_reduce keeps the value and leaves one one-legged dot per wire, a
    #     pi exactly on rods 2 and 4 and the flag.  (sqrt2|1> in reduced form is a green pi
    #     point behind a yellow box; sqrt2|0> a phase-free green point behind one.)
    pts = points_after_reduce(scaled(g, -5))
    assert [ph for ph, _ in pts] == [0, 1, 0, 1, 0, 0, 1]
    assert all(et == ET.HADAMARD for _, et in pts)
    # the rehearsal on the cloth: the slip's web turned and the flag pushed through
    pts = points_after_reduce(scaled(master(SLIP), -5))
    assert [ph for ph, _ in pts] == [1, 0, 0, 0, 1, 0, 1]

    # "Outward or inward, it's the same collar, and the two ways agree": the tubes slid
    # outward onto the end-rings (the named colour-change rule on each point) give the
    # same picture as the tubes left for the web
    h = outward(g)
    green_points = [u for u in h.vertices() if h.type(u) == VT.Z and h.vertex_degree(u) == 1]
    assert len(green_points) == 3                     # rods 2 and 4 and the flag: crosswise knobs
    assert same_picture(h, g)

    # (d) The turned web, as an operator: yellow boxes on all fourteen outer legs of the web
    #     = the fan-out from the flag, with NO scalar.  Every a at n = 4, both b, and the
    #     casket.  DERIVED with named rules: id_simp takes out the two-box stretches of the
    #     unmarked rods (HH = I), then colour change on every dot of the web -- a tie's
    #     cancelling pair of boxes is split, one absorbed at each end, so the tie stays plain.
    for a, b in [(a, b) for a in strings(4) for b in (0, 1)] + [(CASKET, 0), (SLIP, 0)]:
        n = len(a)
        lhs, rhs = web_between_boxes(a, b), fan_out(a, b)
        turned = Hn(n + 1) @ U(linear(a, b), n) @ Hn(n + 1)
        w = sum(a)
        assert np.allclose(matrix_of(scaled(web_between_boxes(a, b, boxed=False), w)), U(linear(a, b), n))
        assert same_picture(lhs, rhs)                                   # no scalar between them
        # both sides carry the same 2^{-w/2} (every bare green-red pair is CNOT/sqrt2, either
        # way up), so each is 2^{-w/2} times the matrix H^{n+1} U_f H^{n+1}
        assert np.allclose(matrix_of(scaled(rhs, w)), turned)
        if w:
            assert not same_picture(web_between_boxes(a, b, boxed=False), rhs)   # boxes needed
        h = lhs.copy()
        ties_before = ties(h)
        simp.id_simp(h)
        dots = [u for u in h.vertices() if h.type(u) != VT.BOUNDARY]
        for u in dots:
            assert simp.check_color_change(h, u) and simp.color_change(h, u)
        assert all(h.edge_type(e) == ET.SIMPLE for e in h.edges())      # every box is gone
        assert same_picture(h, rhs)
        # "Nothing has moved. Every tie joins the cords it always joined; only the rings have
        # turned": the same ties, now red on the rods and green on the flag
        assert ties(h) == ties_before == {frozenset((i, n)) for i in range(n) if a[i]}
        for u in dots:
            if h.qubit(u) < n:
                assert h.type(u) == VT.X                                 # rod rings now red
            elif h.phase(u) == 0:
                assert h.type(u) == VT.Z                                 # flag rings now green

    # (e) Cleve's form for random secrets at n = 6: rods as in the master, the flag a green pi
    #     point with no yellow boxes.  Scalar add_power(w - n - 1).  Value (-1)^b |a>|->,
    #     DERIVED by full_reduce: a red pi point on exactly the marked rods, the flag |->.
    rng = np.random.default_rng(13031)
    secrets = [(CASKET, 1), ((0,) * 6, 1), ((1,) * 6, 0)]
    secrets += [(tuple(int(t) for t in rng.integers(0, 2, 6)), int(rng.integers(0, 2))) for _ in range(5)]
    for a, b in secrets:
        n, w = len(a), sum(a)
        c = scaled(master(a, b, cleve=True), w - n - 1)
        vec = matrix_of(c).ravel()
        assert np.allclose(vec, (-1) ** b * np.kron(ket(*a), MINUS))
        if b:
            assert not np.allclose(vec, np.kron(ket(*a), MINUS))
        pts = points_after_reduce(c)
        assert [ph for ph, _ in pts[:n]] == list(a)
        assert all(et == ET.HADAMARD for _, et in pts[:n])               # red points: bits a_i
        assert pts[n] == (1, ET.SIMPLE)                                  # the flag: green pi, |->


if __name__ == "__main__":
    verdict("TurnedWeb", blackboard, sketchbook)
