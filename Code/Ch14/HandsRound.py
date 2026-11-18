"""Challenge I: The Fifteen-Hour Dial  (the order: a hidden period, found by ticking)
===================================================================================
"Four ticks. Seven thirteens are ninety-one; less six fifteens, one. Home."  -- Achilles

Claim: the order of 7 mod 15 is 4, found by ticking the hand 1, 7, 4, 13, 1; for every
(N, a) with gcd(a, N) = 1 the tick box U_a |x> = |a x mod N> (x < N; x >= N left alone) is a
permutation, hence unitary, and U_a^r = I (Result 14.1); and for N = 15, a = 7 the
four-wire picture C_7 -- the bottom wire climbing to the top over the other three, then a
red pi on every wire -- is exactly x -> 7x mod 15 on the residues 1 <= x < N, sends 0 to
15 (not a residue), and C_7^4 = I on all sixteen basis states (Result 14.2).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction
from math import gcd

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import ket, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch14 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)


# --- the blackboard ---------------------------------------------------------------------

def orbit(a, N):
    """Tick the hand from one until it is home; return the hours it visits."""
    hours, x = [1], a % N
    while x != 1:
        hours.append(x)
        x = (a * x) % N
    return hours


def order_by_power(a, N):
    """The least r >= 1 with a^r = 1 mod N (Shor section 5)."""
    return next(r for r in range(1, N + 1) if pow(a, r, N) == 1)


def tick_box(a, N, wires):
    """U_a on `wires` wires, top wire the most significant bit:
    |x> -> |a x mod N> for 0 <= x < N, and |x> left alone for x >= N."""
    d = 2 ** wires
    M = np.zeros((d, d), dtype=complex)
    for x in range(d):
        M[(a * x) % N if x < N else x, x] = 1
    return M


def blackboard():
    # --- the dial of the First Challenge: one, seven, four, thirteen, home
    N, a = 15, 7
    assert orbit(a, N) == [1, 7, 4, 13]
    r = len(orbit(a, N))
    assert r == 4 == order_by_power(a, N)
    # Achilles's arithmetic, tick by tick
    assert 7 * 1 == 7 and 7 * 7 == 3 * 15 + 4 and 7 * 4 == 15 + 13 and 7 * 13 == 6 * 15 + 1
    # the other two dials of the chapter
    assert order_by_power(2, 21) == 6 == len(orbit(2, 21))
    assert order_by_power(5, 33) == 10 == len(orbit(5, 33))

    # --- Result 14.1 for the dial: U_7 on four wires is a permutation matrix, unitary,
    #     and its fourth power is the identity -- but no smaller power is
    U7 = tick_box(a, N, 4)
    assert np.allclose(U7.conj().T @ U7, np.eye(16))
    assert (np.abs(U7).sum(axis=0) == 1).all() and (np.abs(U7).sum(axis=1) == 1).all()
    assert np.allclose(np.linalg.matrix_power(U7, r), np.eye(16))
    for s in range(1, r):
        assert not np.allclose(np.linalg.matrix_power(U7, s), np.eye(16))
    # on the residues {x : gcd(x, N) = 1} it is a permutation of the residues
    units = [x for x in range(1, N) if gcd(x, N) == 1]
    assert sorted((a * x) % N for x in units) == units
    # top wire first: U_7 |0001> = |0111>
    assert np.allclose(U7 @ ket(0, 0, 0, 1), ket(0, 1, 1, 1))

    # --- Result 14.1 in general: random (N, a) with gcd(a, N) = 1.  The order found by
    #     ticking is the least r with a^r = 1; x -> ax mod N permutes {0, ..., N-1}; U_a is
    #     unitary on the register and U_a^r = I.
    rng = np.random.default_rng(1401)
    tried = 0
    while tried < 25:
        N = int(rng.integers(3, 60))
        a = int(rng.integers(2, N))
        if gcd(a, N) != 1:
            # the hypothesis is needed: a shared factor makes the tick non-invertible
            assert len({(a * x) % N for x in range(N)}) < N
            continue
        tried += 1
        r = order_by_power(a, N)
        assert len(orbit(a, N)) == r
        assert sorted((a * x) % N for x in range(N)) == list(range(N))
        w = int(np.ceil(np.log2(N)))
        Ua = tick_box(a, N, w)
        I = np.eye(2 ** w)
        assert np.allclose(Ua.conj().T @ Ua, I)
        assert np.allclose(np.linalg.matrix_power(Ua, r), I)
        if r > 1:
            assert not np.allclose(np.linalg.matrix_power(Ua, r - 1), I)

    # --- "It can be nearly as large as N": for an odd prime N some base has order N - 1
    for N in (5, 7, 11, 13, 17, 19, 23, 29, 31):
        assert max(order_by_power(a, N) for a in range(2, N)) == N - 1

    pictures()


# --- the pictures in the dialogue ----------------------------------------------------------

COUNT = ("none", "once", "twice", "three times", "four times", "five times", "six times", "seven times",
         "eight times")


def clockwise(p):
    """The angle of a point on the dial, clockwise from the top mark as Achilles sees it, in degrees."""
    return (90 - np.degrees(np.arctan2(p[1] - PC.DCY, p[0] - PC.DCX))) % 360


def numbers(note):
    """The figures of a logged clearance line, by name ('the hand 0.104, ...')."""
    return {k.strip(): float(v) for k, v in re.findall(r"([a-z' ]+?) (-?\d+\.\d+)", note.split(": ", 1)[1])}


def pictures():
    """The First Challenge's prop pictures (identity card, sections 4 and 3e), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch14.py: three blocks,
    two single calls and a strip of three, five frames (F01-F05).  The ties below hold the
    transcription against this file's blackboard: a structural check of the pictures, not a
    second proof."""
    calls = DIALOGUE.check_layout("First", sizes=[1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("tickdial", True, "", ("hand at one",), "the paper dial"),
        ("tickdial", False, "step=tick", ("hand at one",), "after a tick"),
        ("tickdialsteps", False, "from=off", ("hand ticked once", "tick; tick; tick"),
         "on past the top; round again; home again")]
    PC.check_captions(DIALOGUE, "First", calls)
    PC.check_section(DIALOGUE, "First", calls)
    dial_h = round(PC.DIAL_FTOP - PC.DIAL_FBOT, 2)
    assert dial_h == 2.42                                             # read from the Lua
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("First")] == [[dial_h]] * 3
    frames = [(fc, p, cap) for c in calls for fc, (p, cap) in zip(PC.frame_calls(c), PC.pictures(c))]
    names = ["F%02d" % i for i in range(1, 6)]
    assert len(frames) == len(names) == 5
    F = dict(zip(names, [p for _, p, _ in frames]))
    FC = dict(zip(names, [fc for fc, _, _ in frames]))
    for fc, p, cap in frames:                                         # a strip frame IS its single call
        assert PC.drawn(PC.picture(fc)) == PC.drawn(p) and PC.picture(fc).log == p.log, fc.text
    assert [(fc.keyed, fc.opt, fc.args[0]) for fc in FC.values()] == [
        (True, "", "hand at one"), (False, "step=tick", "hand at one"),
        (False, "step=tick", "hand ticked once"), (False, "step=tick", "fifteen hours; hand ticked twice"),
        (False, "step=tick", "fifteen hours; hand ticked three times")]

    # the anchors (card sections 4 and 3e.4): the paragraph before each block and the one after,
    # in dialog.md; B03's next paragraph is ambient there, so the raw file's is checked too
    a1, a2, a3 = DIALOGUE.anchors("First")
    f1, f2, f3 = DIALOGUE.following("First")
    assert "on the side of the sheet furthest from Achilles" in a1 and "with a brass pin" in a1     # D3
    assert "clockwise as Achilles sees it" in a1                                                  # D3
    assert f1.startswith("**Tortoise:** It starts at one.")
    assert "carries it round clockwise, past" in a2 and "One has become seven." in a2            # D4
    assert f2.startswith("**Achilles:** Two ticks.")
    assert "now it points at one, as though it had never left" in a3                              # B03
    assert "It has pointed at seven, at four and at thirteen on the way" in a3
    assert f3.startswith("*[Up the lane, after all their tumbling")
    r1, r2, r3 = DIALOGUE.following("First", DIALOGUE.raw)
    assert (r1, r2) == (f1, f2) and r3.startswith("**Achilles:** Four ticks to come round.")
    # D1 and D2 lie in the Opening: asserted on the whole of both files, with D2's old
    # bracket gone (the spoken "...with the Office's label on its side." is another line)
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        whole = DIALOGUE.read(path)
        assert "her little box of rings and the packet of tubes]*" in whole                       # D1
        assert "with the Office's label on its left side." in whole                               # D2
        assert "with the Office's label on its side.]*" not in whole

    # the hand's hours ARE the blackboard's orbit: one tick multiplies by seven, counted
    # round the fifteen; the orbit's length is the order r = 4
    assert PC.dial_orbit(15) == orbit(7, 15) == [1, 7, 4, 13]
    assert len(PC.dial_orbit(15)) == order_by_power(7, 15) == 4
    for m, w in enumerate(COUNT):
        assert PC.dial_state("hand ticked " + w).hand == pow(7, m, 15)
    assert PC.dial_state("hand at one").hand == 1

    # F01-F05: one, then 7^m mod 15 for m = 1..4 -- seven, four, thirteen, home; no round on the sheet
    hands = [F[n].drawn.hand for n in names]
    assert hands == [pow(7, m, 15) for m in range(5)] == [1, 7, 4, 13, 1]
    for n in names:
        assert F[n].head == "hours=15 hand=%d rings=none" % F[n].drawn.hand
        assert F[n].view.rings == () and F[n].view.pairs == ()
    assert F["F01"].words == ("top", "hand") and F["F01"].notes == [] and F["F01"].mark_tikz == []
    assert all(F[n].words == () for n in names[1:])                   # keyed once, on F01

    # every tick is marked (card 3e.6, the readings): the hand's ghost at the hour it left and a
    # swing arc round the pin, clockwise through the marks it passed -- the clockwise way from
    # h to 7h mod 15: 6, 12, 9, 3 marks, which sum to 30, two whole turns of the dial
    ways = []
    for n, h0, h1 in zip(names[1:], hands, hands[1:]):
        p = F[n]
        k = (h1 - h0) % 15
        ways.append(k)
        assert p.before.hand == h0 and p.after.hand == h1 == (7 * h0) % 15
        assert p.notes[0] == "step tick -> " + p.head
        assert p.notes[1] == "ghost of the hand in dotted outline at hour %d, under the hand and the pin" % h0
        assert re.fullmatch(r"swing arc round the pin, clockwise, %d marks \(%.3f degrees\), radius 0\.700 -> 0\.\d{3}"
                            % (k, 24 * k), p.notes[2]), p.notes[2]
        assert len(p.notes) == 4 and p.notes[3].startswith("clearances (edge to edge): ")
        gaps = numbers(p.notes[3])
        assert gaps.pop("inside the sheet by") > 0.01                 # every mark on the sheet
        assert min(gaps.values()) > 0.01, gaps                        # touching nothing drawn
        assert "the rings none, the twine none" in p.notes[3]
        (arc,) = p.arcs
        assert arc.kind == "swing" and not arc.headless and len(p.mark_tikz) == 2   # the ghost, the arc
        # drawn: the arc starts just clockwise of the hand it left, ends just short of the
        # hand it reached, and between them runs clockwise the hand's way
        assert 0 < (clockwise(arc.p0) - 24 * h0) % 360 < 12
        assert 0 < (24 * h1 - clockwise(arc.p1)) % 360 < 12
        assert ((24 * h1 - 24 * h0) % 360) == 24 * k
        assert np.hypot(arc.p1[0] - PC.DCX, arc.p1[1] - PC.DCY) < np.hypot(arc.p0[0] - PC.DCX, arc.p0[1] - PC.DCY)
        # the wrong sense (counter-clockwise, 15 - k marks) is never what is logged
        assert k != 15 - k and "counter-clockwise" not in p.notes[2]
        # the identities (card 3e.6): marks=off draws the same picture, the marks left out
        assert PC.drawn(PC.twin(FC[n])) == PC.drawn(p) and PC.twin(FC[n]).mark_tikz == []
    assert ways == [6, 12, 9, 3] and sum(ways) == 30
    # F02 and F05 with marks=off are round 3's P1.2 and P1.3
    p12 = PC.run_frame("dial", "", "hand ticked once", False)
    p13 = PC.run_frame("dial", "", "hand ticked four times", False)
    assert not PC.TEX_ERRORS
    assert PC.drawn(PC.twin(FC["F02"])) == PC.drawn(p12) and PC.drawn(PC.twin(FC["F05"])) == PC.drawn(p13)

    # what is drawn: the hand's tip points at its hour, counted clockwise from the top mark
    # (y up, the top of the picture the far side of the sheet); the tip stays inside the
    # pencilled circle; the "top" word stands above the top mark, clear of it
    for n in names:
        p = F[n]
        tip = p.view.hand_outline[2]
        assert np.isclose(clockwise(tip), 360 * p.drawn.hand / 15)
        assert 0 < PC.DR - np.hypot(tip[0] - PC.DCX, tip[1] - PC.DCY)
    assert F["F02"].view.hand_outline != F["F01"].view.hand_outline == F["F05"].view.hand_outline
    p11 = F["F01"]
    # the keyed words, as set (measured boxes, strokes included): "top" just right of the top
    # mark's outer end and above the circle, its whole node box inside the sheet and
    # touching nothing drawn; "hand" beside the hand, its letters clear of it
    (word, x, y, anchor), = [l for l in p11.labels if l[0] == "top"]
    assert anchor == "south west" and (x, y) == (PC.DCX + PC.TOP_DX, PC.DCY + PC.DR + PC.TOP_DY)
    top = PC.label_box("top", x, y, anchor)
    assert top[0] > PC.DCX and top[1] > PC.DCY + PC.DR and top[1] < PC.DCY + PC.DR + PC.MARK
    gaps = PC.dial_word_gaps(p11.view, "top")
    assert min(gaps.values()) > 0.005 and gaps["sheet"] > 0.01, gaps
    assert min(PC.dial_word_gaps(p11.view, "top", ink=True).values()) > 0.03
    # "hand" is held on its letters: its node box's 1pt inner sep is padding nothing draws
    # (the label has no fill and no border), and that padding reaches the hand's edge
    # (ch14.lua sets the word 0.07 off the hand's middle line), so the box is not the
    # measure of whether the word touches the hand; the letters clear it by over 0.03
    hand = PC.dial_word_gaps(p11.view, "hand", ink=True)
    assert min(hand.values()) > 0.03, hand
    assert PC.dial_word_gaps(p11.view, "hand")["hand"] < 0      # the padding does reach it
    # F02: the first tick carried the hand clockwise as Achilles sees it, six hours on
    # (one to seven): 6 x 24 = 144 degrees clockwise, not 216
    t11, t12 = p11.view.hand_outline[2], F["F02"].view.hand_outline[2]
    assert np.isclose((clockwise(t12) - clockwise(t11)) % 360, 144)
    assert not PC.TEX_ERRORS


# --- the sketchbook ---------------------------------------------------------------------

def dial_picture(shift=1, flips=True):
    """C_7, drawn: input row i runs to output row (i + shift) mod 4 -- with shift = 1 the
    bottom wire climbs to the top and the other three step down one row (three left
    rotations of the bits, that is x -> 8x mod 15) -- then a red pi dot (NOT) on every wire.
    A crossing is only connectivity, so it is built as plain edges between rows."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, i, 0) for i in range(4)]
    outs = [g.add_vertex(VT.BOUNDARY, i, 10) for i in range(4)]
    for i in range(4):
        row = (i + shift) % 4
        if flips:
            nd = g.add_vertex(VT.X, row, 6, phase=Fraction(1))     # red pi: X, exactly
            g.add_edge((ins[i], nd)); g.add_edge((nd, outs[row]))
        else:
            g.add_edge((ins[i], outs[row]))
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    return g


def dial_circuit():
    """The same C_7 as a gate list: three SWAPs (the bottom wire carried up to the top),
    then a NOT on each wire.  PyZX lowers each SWAP to three CNOTs, whose scalars it
    carries itself, so no compensation is added."""
    c = zx.Circuit(4)
    for top in (2, 1, 0):
        c.add_gate("SWAP", top, top + 1)
    for q in range(4):
        c.add_gate("NOT", q)
    return c.to_graph()


def bare_wires(n):
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, i, 0) for i in range(n)]
    outs = [g.add_vertex(VT.BOUNDARY, i, 10) for i in range(n)]
    for i, o in zip(ins, outs):
        g.add_edge((i, o))
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    return g


def power(g, k):
    h = g.copy()
    for _ in range(k - 1):
        h.compose(g.copy())                                       # compose mutates h
    return h


def sketchbook():
    N = 15
    C7 = dial_picture()
    M = matrix_of(C7)

    # (a) C_7 is the permutation x -> 15 - rot(x) of all sixteen basis states, exactly
    def rot3(x):                                                  # three left rotations
        for _ in range(3):
            x = ((x << 1) & 15) | (x >> 3)
        return x
    P = np.zeros((16, 16))
    for x in range(16):
        P[15 - rot3(x), x] = 1
    assert np.allclose(M, P)
    # ... which is x -> 7x mod 15 on the residues 1 <= x < N only
    for x in range(1, N):
        assert np.allclose(M[:, x], ket(*[int(b) for b in f"{(7 * x) % N:04b}"]))
    # 0 goes to 15, which is not an hour of the dial (the orbit of one never goes there)
    assert np.allclose(M @ ket(0, 0, 0, 0), ket(1, 1, 1, 1))
    assert not np.allclose(M[:, 0], ket(0, 0, 0, 0))
    # top wire first, and the rotation's direction: 0001 -> 0111 (1 -> 7).  The other
    # direction (one left rotation, x -> 2x, then the flips: x -> -2x = 13x) sends 1 to 13.
    assert np.allclose(M @ ket(0, 0, 0, 1), ket(0, 1, 1, 1))
    wrong = dial_picture(shift=3)
    assert np.allclose(matrix_of(wrong) @ ket(0, 0, 0, 1), ket(1, 1, 0, 1))
    assert not same_picture(C7, wrong)
    assert not same_picture(C7, dial_picture(flips=False))        # the flips are needed
    # the gate list of three SWAPs and four NOTs is the same picture, scalar and all
    assert same_picture(dial_circuit(), C7)

    # (b) C_7^4 = I, DERIVED: full_reduce on a copy (fusion of the four red pi dots on each
    #     wire to a phase-free dot, then the plain two-legged dots removed; the crossings
    #     compose to none) leaves four bare wires and no dot at all
    C4 = power(C7, 4)
    assert same_picture(C4, bare_wires(4))
    h = C4.copy()
    simp.full_reduce(h)
    assert h.num_vertices() == 8 and h.num_edges() == 4           # only the boundaries
    for i, o in zip(h.inputs(), h.outputs()):                     # each input straight across
        assert h.connected(i, o)
    assert np.isclose(h.scalar.to_number(), 1)                    # no scalar anywhere
    assert same_picture(h, bare_wires(4))
    # and no smaller power is the identity: the dial's period, in pictures
    for k in (1, 2, 3):
        assert not same_picture(power(C7, k), bare_wires(4))
    # the gate list, four times, reduces the same way
    h = power(dial_circuit(), 4)
    simp.full_reduce(h)
    assert same_picture(h, bare_wires(4))
    # NOTE: the picture is special to N = 15, a = 7 (15 = 16 - 1 and 7 = -8 mod 15, so the
    # multiplication is a rotation of four bits and a flip).  A general multiplication mod N
    # is a reversible arithmetic circuit (Shor section 3) with no such picture.


if __name__ == "__main__":
    verdict("HandsRound", blackboard, sketchbook)
