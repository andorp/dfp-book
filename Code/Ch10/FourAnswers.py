"""Challenge I: Four Ways to Answer  (the Bell basis and the Bell reading)
========================================================================
"It has two cups side by side under one lens, and it reads two stones together, never one."  -- Crab

Claim: the four Bell states B_{m1m2} = CNOT (H (x) I)|m1 m2> are the pair
|Phi+> = cup/sqrt2 with a Pauli on one leg, (Z^m1 X^m2 (x) I)|Phi+> =
(I (x) X^m2 Z^m1)|Phi+>, an orthonormal, complete basis; the reading's effects
<B_{m1m2}| = <m1 m2|(H (x) I) CNOT = (1/sqrt2) cap (X^m2 Z^m1 (x) I) are a cap
carrying two bit-labelled rings; one half-turn ring slides round a bend
unchanged, and a chain of two rings arrives in reversed order.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import BELL, CAP, CUP, I2, Picture, kron, ket, matrix_of, random_matrix, \
    random_state, same_picture, same_picture_up_to_scalar, verdict
from dfp_props import Dialogue, words_only
import props_ch10 as PC


# --- the chapter's matrices, written out (kept local: the mathematics stays here) ---
X = np.array([[0, 1], [1, 0]], dtype=complex)             # the flip
Z = np.array([[1, 0], [0, -1]], dtype=complex)            # the spin (phase-flip)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
KS = [(0, 0), (0, 1), (1, 0), (1, 1)]                     # the four answers (m1, m2)


def P(M, m):
    """M^m for a bit m: do it if m = 1."""
    return np.linalg.matrix_power(M, m)


def bell(m1, m2):
    """|B_{m1m2}> = CNOT (H (x) I)|m1 m2>, the circuit of Chapter 6 fed all four inputs."""
    return (CNOT @ kron(H, I2) @ ket(m1, m2).reshape(4, 1)).ravel()


def blackboard():
    rng = np.random.default_rng(1001)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))      # cup = |00>+|11>, UNNORMALIZED
    assert np.allclose(BELL, CUP / np.sqrt(2))               # |Phi+> = cup / sqrt2

    # the two Paulis: self-inverse, real, symmetric -- and they anticommute
    assert np.allclose(X @ X, I2) and np.allclose(Z @ Z, I2)
    assert np.allclose(X, X.T) and np.allclose(Z, Z.T)
    assert np.allclose(X @ Z, -Z @ X)
    assert not np.allclose(X @ Z, Z @ X)

    # the table of Section 10.1, written out (top wire first)
    s = 1 / np.sqrt(2)
    assert np.allclose(bell(0, 0), s * (ket(0, 0) + ket(1, 1)))      # Phi+
    assert np.allclose(bell(0, 1), s * (ket(0, 1) + ket(1, 0)))      # Psi+
    assert np.allclose(bell(1, 0), s * (ket(0, 0) - ket(1, 1)))      # Phi-
    assert np.allclose(bell(1, 1), s * (ket(0, 1) - ket(1, 0)))      # Psi-
    assert np.allclose(bell(0, 0), BELL)                             # B00 is Chapter 6's pair

    # Result 10.1: each answer is the plain pair with a flip, a spin or both on ONE leg
    for m1, m2 in KS:
        B = bell(m1, m2)
        top = (kron(P(Z, m1) @ P(X, m2), I2) @ BELL).ravel()         # on the top leg
        bottom = (kron(I2, P(X, m2) @ P(Z, m1)) @ BELL).ravel()      # slid to the bottom leg
        assert np.allclose(B, top) and np.allclose(B, bottom)
    # the product slid to the other leg comes out REVERSED: same order on the bottom is wrong
    assert np.allclose((kron(I2, Z @ X) @ BELL).ravel(), -bell(1, 1))
    assert not np.allclose((kron(I2, Z @ X) @ BELL).ravel(), bell(1, 1))

    # orthonormal and complete: four answers, no room for a fifth
    Bs = np.array([bell(*k) for k in KS])                    # rows are the four states
    assert np.allclose(Bs.conj() @ Bs.T, np.eye(4))           # <B_j|B_k> = delta_jk
    assert np.allclose(sum(np.outer(b, b.conj()) for b in Bs), np.eye(4))   # sum |B_k><B_k| = I

    # the reading's effects are the MIRRORS (daggers) of the states, and equal the
    # circuit run backwards, then a plain look: <m1 m2|(H (x) I) CNOT
    for m1, m2 in KS:
        eff = bell(m1, m2).conj().reshape(1, 4)
        ringed_cap = BELL.conj().reshape(1, 4) @ kron(P(X, m2) @ P(Z, m1), I2)
        circuit = ket(m1, m2).reshape(1, 4) @ kron(H, I2) @ CNOT
        assert np.allclose(eff, ringed_cap) and np.allclose(eff, circuit)
        assert np.allclose(eff, CAP @ kron(P(X, m2) @ P(Z, m1), I2) / np.sqrt(2))
    assert np.allclose(CAP, np.sqrt(2) * bell(0, 0).conj())    # Chapter 6's cap = sqrt2 <B00|

    # the ledger's quarters: read a message against half a pair, and each of the four
    # answers comes a quarter of the time -- for a NON-REAL message, whatever it is
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())
    whole = kron(psi, BELL).ravel()                           # wires 1, a, c (top first)
    probs = []
    for m1, m2 in KS:
        branch = kron(bell(m1, m2).conj().reshape(1, 4), I2) @ whole
        probs.append(np.linalg.norm(branch) ** 2)
    assert np.allclose(probs, 0.25) and np.isclose(sum(probs), 1.0)

    # sliding round a bend TRANSPOSES (Chapter 6) -- on a generic, non-symmetric,
    # non-real box -- so a single half-turn (its own transpose) comes round unchanged,
    # while a product comes round reversed: (BA)^T = A^T B^T
    A, B = random_matrix(rng), random_matrix(rng)
    assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())
    assert np.allclose(kron(A, I2) @ CUP, kron(I2, A.T) @ CUP)
    assert not np.allclose(kron(A, I2) @ CUP, kron(I2, A.conj().T) @ CUP)   # not the dagger
    assert np.allclose(kron(B @ A, I2) @ CUP, kron(I2, A.T @ B.T) @ CUP)     # reversed
    assert not np.allclose(kron(B @ A, I2) @ CUP, kron(I2, B.T @ A.T) @ CUP) # not same order
    for R in (X, Z):
        assert np.allclose(kron(R, I2) @ CUP, kron(I2, R) @ CUP)             # one ring: unchanged

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the pairing-glass as it is unwrapped
# (keyed: its tokens blank in their trays, brass left, copper right), the two
# tokens close to in the dish (keyed) as one is turned -- round 4, "Show the
# process": the state after the step, a swing arc over the brass token and a pin
# round the copper, which held still -- and the Tortoise's four bends, two above
# two (keyed), then the flip's ring slid round its bend (its ghost where it lay,
# the slide arc round the outside of the bend),
#   \pairingglass{state}{caption}, \twotokens[place; step=...]{pattern}{caption},
#   \ringstring[operations; step=...]{scene}{caption}      (pdf/props/ch10.tex)
# tools/props_ch10.py transcribes the Lua (proved against texlua; check_lua
# holds it to the Lua and the .tex it was proved on; the string's layout read
# by the shared cloth kit, tools/props_cloth.py); the calls are read out of
# this challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3): a pattern of the tokens is an
# answer (m1, m2) -- copper up the spin m1, brass up the flip m2; a red ring is
# X, a green ring Z; on a leg the ring nearest the bend is applied first.  Each
# bend with its rings is sqrt2 times the Bell state its answer names, and a ring
# slid round the bend comes round TRANSPOSED: one ring unchanged, two rings the
# other way about (L1).  A structural check against this file's blackboard, not
# a second proof.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = "It has two cups side by side under one lens, and it reads two stones together, never one."
RING = {"red": X, "green": Z}
# round 4, T1: the token's side, named (card section 3e.3)
EDITS = [("and turns the brass token over: it rocks onto its back",
          "and turns the brass token over towards the copper: it rocks onto its back")]


def leg_operator(rings):
    """The operator a leg's rings make, listed from the bend outward: the ring nearest
    the bend is applied first, so R = r_k ... r_1."""
    R = I2
    for c in rings:
        R = RING[c] @ R
    return R


def head(c):
    """A call's bracket as its log line writes it (the Lua's trim of what TeX hands it):
    the items in the order the dialogue writes them, which the picture never depends on."""
    return PC.trim(PC.bracket(c) or "")


def r3(x):
    """A mark's numbers to the Lua's three places (its log's)."""
    if isinstance(x, tuple):
        return tuple(r3(y) for y in x)
    return round(x, 3) if isinstance(x, float) else x


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    PC.check_lua()                                                # the Lua the transcription was proved on
    PC.check_sites()                                              # its 73 fault sites, one for one
    found = PC.check_section(DIALOGUE, "First", sizes=[1, 1, 1, 1], frames=[1, 1, 1, 1])
    PC.check_edits(DIALOGUE, "First", EDITS)
    calls = [c for b in found for c, _ in b]
    p11, p12, p13, p14 = calls
    assert [c.macro for c in calls] == ["pairingglass", "twotokens", "ringstring", "ringstring"]
    assert [c.keyed for c in calls] == [True, True, True, False]  # P1.1, P1.2, P1.3 keyed
    assert not any(PC.is_prediction(c) or PC.is_strip(c) for c in calls)
    assert [PC.options(c) for c in calls] == [
        {"bare": ""}, {"step": "turn brass", "bare": ""}, {"bare": ""},
        {"step": "slide the flip round", "bare": ""}]
    assert [c.args[0] for c in calls] == ["tokens in the trays", "both down", "the four bends", "the four bends"]
    assert [c.caption for c in calls] == [
        "blank side up, in the trays", "the brass turned, the copper still",
        "every way a reading ends", "round the bend, unchanged"]
    g11, t12, s13, s14 = (PC.picture(c) for c in calls)
    a1, a2, a3, a4 = DIALOGUE.anchors("First")
    f1, f2, f3, f4 = DIALOGUE.following("First")
    assert QUOTE in DIALOGUE.section("First")                     # the docstring's line
    # the Lua's round-4 comments quote the First's lines, verbatim, in both files
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        here = [(n, q) for n, q, o in PC.quote_places(DIALOGUE, path) if o == "First"]
        assert sorted({n for n, _ in here}) == ["THE STRING'S STEPS", "THE TOKENS' STEP"] and len(here) == 4
    # the log the four frames write, as the card's dump list has it (F01-F04)
    assert [l for c in calls for l in PC.describe(c)] == [
        "pairingglass: [] {tokens in the trays} | cups empty; tokens in the trays; trays level",
        "twotokens: [" + head(p12) + "] {both down} | dish; brass up",
        "twotokens: + step turn brass -> brass up",
        "twotokens: + swing arc over the brass token, turned over where it lies, (-0.710,-0.520) -> "
        "(-0.090,-0.520) (no ghost: it covers its own place)",
        "twotokens: + pin round the copper token, which held still, at (0.400,-0.530)",
        "ringstring: [] {the four bends} | " + s13.dump,
        "ringstring: [" + head(p14) + "] {the four bends} | " + s14.dump,
        "ringstring: + step slide the flip round -> " + s14.dump,
        "ringstring: + ghost flip red ring in dotted outline at (0.440,-0.132) (grid 6,0): it went to grid 6,1",
        "ringstring: + slide arc flip red round the outside of the flip bend, (0.420,-0.031) -> "
        "(0.420,-0.629) (the turn at x 0.128)"]

    # P1.1: the glass unwrapped, keyed -- cups empty, both tokens blank in their trays,
    # the trays level (nothing has chimed); brass tray left, copper right, as said
    assert (g11.wrapped, g11.cups, g11.tokens, g11.pattern, g11.tipped) == \
        (False, "empty", "trays", None, False)
    assert g11.labels == ("lens", "cup", "trays", "brass", "copper", "dish") and g11.marks == ()
    assert g11.frame == PC.GLASS_FRAME == 1.58
    assert "A brass dish waits beneath the trays' lips" in a1
    assert "In the left tray lies a brass token, in the right a copper one, each face-down" in a1
    assert f1.startswith("**Crab:** When both cups are full")

    # P1.2: the tokens close to in the dish, keyed, as a step -- both lay blank (written,
    # not pictured), the brass is turned over: brass up (its stamp), copper down
    assert (t12.before, t12.step, t12.turned) == (PC.Pattern(False, False), "turn brass", ("brass",))
    assert (t12.place, t12.pattern, t12.left, t12.right) == \
        ("dish", PC.Pattern(brass=True, copper=False), "brass", "copper")
    assert t12.labels == ("stamp", "blank", "brass", "copper", "dish")
    assert t12.frame == PC.TOKENS_FRAME == 1.10
    assert "the other has held still" in a2
    assert "and turns the brass token over towards the copper: it rocks onto its back" in a2
    assert "while the copper token lies exactly as it was" in a2
    assert f2.startswith("**Achilles:** Brass up, copper down.")
    # its marks: one swing arc over the brass, from its left rim over to its right, its
    # head towards the copper (T1); no ghost (the token covers its own place); a pin
    # round the copper, which held still
    swing, pin = t12.marks
    assert swing.kind == "swing" and swing.name == "brass" and t12.mark("ghost") == []
    (x0, y0), _, _, (x1, y1) = swing.pts
    assert r3((x0, y0, x1, y1)) == (-0.71, -0.52, -0.09, -0.52) and y0 == y1 == PC.TOKEN_Y == -0.52
    assert PC.TOKEN_X == {"brass": -0.4, "copper": 0.4}
    assert x0 < PC.TOKEN_X["brass"] < x1 < PC.TOKEN_X["copper"]   # over the brass, towards the copper
    assert all(y > y0 for _, y in swing.pts[1:3])                  # bowed upward, over it
    assert (pin.kind, pin.name, r3(pin.at), pin.r) == ("pin", "copper", (0.4, -0.53), (0.33, 0.2))
    # the marks-off twin is round 3's P1.2 to the byte: \twotokens*{brass up}
    t = PC.twin(p12)
    assert t.marks == () and PC.frame_of(t) == PC.frame_of(PC.twotokens(None, "brass up", True))
    # with pin=off the arc stays and the pin goes; turning both turns both, and pins none
    assert [m.kind for m in PC.twotokens("step=turn brass; pin=off", "both down").marks] == ["swing"]
    both = PC.twotokens("step=turn both", "both down")
    assert both.pattern == PC.Pattern(True, True) and [m.kind for m in both.marks] == ["swing", "swing"]
    # the pattern is an answer: brass up is m2 = 1, the flip -- the one red ring of the
    # flip bend, and the Bell state with X on a leg
    assert PC.bits(t12.pattern) == (0, 1) and PC.legrings(t12.pattern) == ["red"]
    assert np.allclose((kron(X, I2) @ BELL).ravel(), bell(0, 1))
    # one token turned changes one bit: the step is the flip (x) nothing, on the answer
    for p in PC.PATTERNS:
        for which, flips in (("turn brass", (0, 1)), ("turn copper", (1, 0)), ("turn both", (1, 1))):
            q = PC.twotokens("step=" + which, PC.patname(p)).pattern
            assert tuple(a ^ b for a, b in zip(PC.bits(p), flips)) == PC.bits(q)
    # the ledger's quarters: the four patterns are the four answers, each a quarter
    # of the time on a NON-REAL message read against half a pair
    assert sorted(PC.bits(p) for p in PC.PATTERNS) == KS
    rng = np.random.default_rng(1011)
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())
    for p in PC.PATTERNS:
        read = bell(*PC.bits(p)).conj().reshape(1, 4)
        assert np.isclose(np.linalg.norm(kron(read, I2) @ kron(psi, BELL)) ** 2, 0.25)

    # P1.3: the four bends, keyed -- two above two (a gap row between the pairs): plain,
    # a red close in, a green, a red close in and a green further out
    assert s13.dump == ("four bends; plain: upper [] lower []; flip: upper [red] lower []; "
                        "spin: upper [green] lower []; both: upper [red green] lower []")
    assert s13.labels == ("bend", "ring", "bead", "string") and s13.marks == ()
    assert PC.bends_of(s13.cloth) == 4 and s13.cloth["gaps"] and s13.frame == 1.95
    assert "two above two" in a3 and "The first she leaves plain." in a3
    assert "a red ring close in by the bend and a green ring further out on the same leg" in a3
    assert f3.startswith("**Achilles:** A plain bend, a flip")
    # each bend is sqrt2 times the Bell state its answer names: (R (x) I) cup
    names = [name for name, _, _ in s13.state.bends]
    assert names == [name for name, _, _ in PC.BENDS] == ["plain", "flip", "spin", "both"]
    for (name, up, lo), (_, brass, copper) in zip(s13.state.bends, PC.BENDS):
        m = PC.bits(PC.Pattern(brass, copper))
        assert up == PC.legrings(PC.Pattern(brass, copper)) and lo == []
        assert np.allclose(kron(leg_operator(up), I2) @ CUP, np.sqrt(2) * bell(*m))

    # P1.4: the flip's red slid round its bend, as a step from P1.3's state -- the same
    # distance out on the lower leg, the other three bends unchanged; the frame kept
    assert s14.before.bends == s13.state.bends and s14.step == "slide the flip round"
    assert s14.dump == ("four bends; plain: upper [] lower []; flip: upper [] lower [red]; "
                        "spin: upper [green] lower []; both: upper [red green] lower []")
    assert [s14.state.bends[i] for i in (0, 2, 3)] == [s13.state.bends[i] for i in (0, 2, 3)]
    assert s14.frame == s13.frame and s14.labels == ()
    assert "Nothing about it has changed but the leg it sits on." in a4
    assert "into the bend, round it, and out along the other leg" in a4
    assert f4.startswith("**Achilles:** It came round exactly")
    # its marks: the red's ghost in outline where it lay on the upper leg, and one slide
    # arc round the OUTSIDE of the bend to the same distance out on the lower leg
    ghost, slide = s14.marks
    assert (ghost.kind, ghost.name, ghost.how, r3(ghost.at), ghost.grid) == \
        ("ghost", "flip red", "outline", (0.44, -0.132), (6, 0))
    assert slide.kind == "slide" and slide.name == "flip red" and s14.mark("pin") == []
    (sx, sy), (ex, ey) = slide.pts
    assert r3((sx, sy, ex, ey)) == (0.42, -0.031, 0.42, -0.629)
    o14 = PC.C.options(PC.options_text(s14.state), "ringstring")
    ring_x, ring_y = PC.C.cloth_xy(s14.cloth, o14, 6, 1, "ringstring")
    turn_x, _ = PC.C.cloth_xy(s14.cloth, o14, 5 - 0.42, 0, "ringstring")
    assert r3(turn_x) == 0.128 and turn_x < ring_x                # the turn, outside the bend's places
    assert np.isclose(ghost.at[0], ring_x)                         # the same distance from the bend
    assert sy > ghost.at[1] > ring_y > ey                          # from above the ghost to below the ring
    # the marks-off twin is round 3's P1.4 to the byte
    t = PC.twin(p14)
    assert t.marks == () and PC.frame_of(t) == PC.frame_of(PC.ringstring("slide the flip round", "the four bends"))
    # one ring round a bend: (I (x) R) cup is EXACTLY (R (x) I) cup (a half-turn is its
    # own transpose) -- the flip as drawn, and the spin the same way
    for op, i in (("slide the flip round", 1), ("slide the spin round", 2)):
        M = PC.apply_string(s13.state, op)
        (_, up0, _), (_, up, lo) = s13.state.bends[i], M.bends[i]
        assert up == [] and lo == up0
        assert np.allclose(kron(I2, leg_operator(lo)) @ CUP, kron(leg_operator(up0), I2) @ CUP)
    # two rings round a bend (L1, modelled; the dialogue never draws it): they cannot pass
    # each other, so the lower leg reads green, red from the bend, and (I (x) XZ) cup =
    # (ZX (x) I) cup -- the transpose of a product reverses it.  The same order as the
    # upper leg (the Lua before L1) is off by a sign.
    both = PC.apply_string(s13.state, "slide both round").bends[3]
    assert both == ("both", [], ["green", "red"])
    ZX = leg_operator(["red", "green"])
    assert np.allclose(ZX, Z @ X) and np.allclose(leg_operator(both[2]), X @ Z)
    assert np.allclose(kron(I2, leg_operator(both[2])) @ CUP, kron(ZX, I2) @ CUP)
    assert np.allclose(kron(I2, ZX) @ CUP, -kron(ZX, I2) @ CUP)
    assert not np.allclose(kron(I2, ZX) @ CUP, kron(ZX, I2) @ CUP)
    assert "Two together come round the other way about" in DIALOGUE.section("First")
    assert PC.ringstring("slide both round", "the four bends").frame == s13.frame
    # marked, the two come round in their own lanes, each its ghost and its arc
    m2 = PC.ringstring("step=slide both round", "the four bends")
    assert [(m.kind, m.name) for m in m2.marks] == [("ghost", "both green"), ("ghost", "both red"),
                                                    ("slide", "both green"), ("slide", "both red")]
    # the Lua refuses a second slide of a bend already slid, and a pull on the bends
    assert PC.faults(PC.ringstring, "slide the flip round, slide the flip round", "the four bends") == \
        ["bends-round"]
    assert PC.faults(PC.ringstring, "pull", "the four bends") == ["bends-op"]
    # the tokens' step: only a turn, and pin= only with one
    assert PC.faults(PC.twotokens, "step=flip brass", "both down") == ["turn-word"]
    assert PC.faults(PC.twotokens, "pin=off", "both down") == ["pin-step"]


# --- pictures -----------------------------------------------------------------

def one_ring(colour):
    """A single one-in/one-out ring of phase pi on a bare wire."""
    p = Picture()
    ring = p.z(0, 1, 1) if colour == "Z" else p.x(0, 1, 1)
    p.wire(p.inp(0), ring, p.out(0))
    return p.done()


def ringed_cup(m1, m2):
    """(1/sqrt2) cup with, outward from the bend on its top leg, a red ring m2*pi and
    then a green ring m1*pi: the state |B_{m1m2}> (Rosetta row 73).
    Scalar: PyZX's phase-free bend is exactly the unnormalized cup; add_power(-1)
    supplies the 1/sqrt2 of |Phi+>."""
    p = Picture()
    bend = p.bend(0.5, 1)
    p.wire(bend, p.x(0, 2, m2), p.z(0, 3, m1), p.out(0))
    p.wire(bend, p.out(1))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def ringed_cap(m1, m2):
    """(1/sqrt2) cap with, on its incoming top leg, a green ring m1*pi and then a red
    ring m2*pi: the effect <B_{m1m2}| = (1/sqrt2) cap (X^m2 Z^m1 (x) I) (row 74).
    add_power(-1): the fused Bell effect's 1/sqrt2."""
    p = Picture()
    bend = p.bend(0.5, 4)
    p.wire(p.inp(0), p.z(0, 1, m1), p.x(0, 2, m2), bend)
    p.wire(p.inp(1), bend)
    g = p.done()
    g.scalar.add_power(-1)
    return g


def circuit_reading(m1, m2, as_state=False):
    """The reading as built: CNOT (a green dot on the top wire corded to a red dot on
    the bottom), a Hadamard EDGE on the top wire (not an H-box: full_reduce in pyzx
    0.10.6 rejects H-boxes), then the two one-legged red effects of phase m1*pi and
    m2*pi.  As an effect (two inputs) it is <m1 m2|(H (x) I) CNOT; with as_state=True
    the same picture has its two ends turned to outputs (the effect's transpose --
    here also its mirror, since every entry is real).
    Scalar: the bare picture is sqrt2 x the circuit (CNOT picture = CNOT/sqrt2; each
    one-legged red effect = sqrt2 <m|; the Hadamard edge is exactly H), so
    add_power(-1)."""
    g = zx.Graph()
    b1 = g.add_vertex(VT.BOUNDARY, 0, 0)
    b2 = g.add_vertex(VT.BOUNDARY, 1, 0)
    c = g.add_vertex(VT.Z, 0, 1)                             # CNOT control (green)
    t = g.add_vertex(VT.X, 1, 1)                             # CNOT target (red)
    e1 = g.add_vertex(VT.X, 0, 3, phase=Fraction(m1))       # <m1| on the Hadamarded top wire
    e2 = g.add_vertex(VT.X, 1, 3, phase=Fraction(m2))       # <m2| on the bottom wire
    g.add_edge((b1, c)); g.add_edge((b2, t)); g.add_edge((c, t))
    g.add_edge((c, e1), ET.HADAMARD)                         # H on the top wire, as an edge
    g.add_edge((t, e2))
    if as_state:
        g.set_inputs(()); g.set_outputs((b1, b2))
    else:
        g.set_inputs((b1, b2)); g.set_outputs(())
    g.scalar.add_power(-1)
    return g


def chain_on_cup(row, phases):
    """A cup with a two-ring chain on leg `row`, listed outward from the bend."""
    p = Picture()
    bend = p.bend(0.5, 1)
    (c1, a), (c2, b) = phases
    r1 = p.z(row, 2, a) if c1 == "Z" else p.x(row, 2, a)
    r2 = p.z(row, 3, b) if c2 == "Z" else p.x(row, 3, b)
    p.wire(bend, r1, r2, p.out(row))
    p.wire(bend, p.out(1 - row))
    return p.done()


def sketchbook():
    # conventions: the one-in/one-out half-turn rings are EXACTLY X and Z (no scalar)
    assert np.allclose(matrix_of(one_ring("X")), X)
    assert np.allclose(matrix_of(one_ring("Z")), Z)

    for m1, m2 in KS:
        eff = bell(m1, m2).conj().reshape(1, 4)
        # the ringed cup is |B_k>, the ringed cap is <B_k|, exactly
        assert np.allclose(matrix_of(ringed_cup(m1, m2)).ravel(), bell(m1, m2))
        assert np.allclose(matrix_of(ringed_cap(m1, m2)), eff)
        # Result 10.2, tensor check: the circuit-form reading IS the ringed cap, exactly
        assert same_picture(circuit_reading(m1, m2), ringed_cap(m1, m2))

        # Result 10.2, DERIVED: full_reduce (fusion, colour change, identity removal)
        # boils the circuit down, scalar kept.  The reduced B00 effect is a bare
        # boundary-to-boundary edge that matrix_of cannot evaluate as an effect, so the
        # derivation is run on the state form (ends turned to outputs) for all four.
        g = circuit_reading(m1, m2, as_state=True)
        before = g.num_vertices()
        simp.full_reduce(g)
        assert g.num_vertices() < before                     # the rules did the work
        assert not any(g.type(v) == VT.X for v in g.vertices())   # no red dot survives
        assert np.allclose(matrix_of(g).ravel(), bell(m1, m2).conj())
        assert same_picture(g, ringed_cup(m1, m2))           # = the ringed bend, exactly

    # the rings are in a fixed order: red nearer the bend on the cup's leg.  The other
    # order is right only up to a sign, on the one answer with both rings
    wrong = Picture(); bend = wrong.bend(0.5, 1)
    wrong.wire(bend, wrong.z(0, 2, 1), wrong.x(0, 3, 1), wrong.out(0)); wrong.wire(bend, wrong.out(1))
    wrong = wrong.done(); wrong.scalar.add_power(-1)
    assert not same_picture(wrong, ringed_cup(1, 1))
    assert same_picture_up_to_scalar(wrong, ringed_cup(1, 1))

    # one ring slides round a bend UNCHANGED (a half-turn is its own transpose)
    for colour in ("Z", "X"):
        one = Fraction(1)
        assert same_picture(chain_on_cup(0, [(colour, one), ("Z", 0)]),
                            chain_on_cup(1, [(colour, one), ("Z", 0)]))
    # ... but a chain of two, with generic phases, arrives REVERSED -- not in the same order
    a, b = Fraction(1, 3), Fraction(1, 4)
    top = chain_on_cup(0, [("Z", a), ("X", b)])
    assert same_picture(top, chain_on_cup(1, [("X", b), ("Z", a)]))      # reversed: yes
    assert not same_picture(top, chain_on_cup(1, [("Z", a), ("X", b)]))  # same order: no


if __name__ == "__main__":
    verdict("FourAnswers", blackboard, sketchbook)
