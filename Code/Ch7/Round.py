"""Challenge 5: The Round
==========================
"Odd. One black. Won."  -- Achilles

Claim (the master diagram, The Round).  For each of the four dealt question
triples (q_A,q_B,q_C) in {000,011,101,110}, with the tally slip r = q_A or
q_B or q_C and the LOSING effect <1|, the whole seven-wire picture is exactly
the zero vector: no losing outcome has any amplitude.  With the WINNING effect
<0| it equals the same picture with the tally wire and its three CNOTs erased.
The two-wire box P = diag(H, HS*), its upper wire pinned between |q> and <q|,
is exactly its q-th block.  And no player learns anything: each player's own
answer is 1/2, 1/2 for all eight triples -- nothing travels between the rooms.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
from fractions import Fraction

import numpy as np
from pyzx.utils import EdgeType

from dfp import I2, Picture, ket, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch07 as props      # not "P": P is the two-wire box below

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Sd = np.array([[1, 0], [0, -1j]], dtype=complex)
HSd = H @ Sd
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)
P = kron(P0, H) + kron(P1, HSd)              # two-wire box, question on top
N = ket(0, 0, 0) + ket(1, 1, 1)


def bra(bit):
    e = np.zeros((1, 2), dtype=complex)
    e[0, bit] = 1
    return e


def block(q):
    return H if q == 0 else HSd


def round_state(qA, qB, qC, r, s):
    """The seven-wire diagram of section 1a, evaluated to a state on (A,B,C).
    r is the tally slip's word, s the tally effect (0 winning, 1 losing)."""
    # legs A,B,C carry the node N; each q wire is a state |q>, the tally a state |r>
    inp = ket(qA, 0, qB, 0, qC, 0, r) + ket(qA, 1, qB, 1, qC, 1, r)
    st = kron(P, P, P, I2) @ inp                        # each player's box P
    # pin every question wire between |q> and <q>: P collapses to its block
    st = kron(bra(qA), I2, bra(qB), I2, bra(qC), I2, I2) @ st     # -> (A,B,C,tally)
    # the tally is the target of CNOT with leg C, then B, then A (with crossings)
    st = kron(I2, I2, CNOT) @ st                        # CNOT (C -> tally)
    st = kron(I2, I2, SWAP) @ st                        # cross: (A,B,tally,C)
    st = kron(I2, CNOT, I2) @ st                        # CNOT (B -> tally)
    st = kron(I2, SWAP, I2) @ st                        # cross: (A,tally,B,C)
    st = kron(CNOT, I2, I2) @ st                        # CNOT (A -> tally)
    st = kron(I2, bra(s), I2, I2) @ st                  # the slip's effect -> (A,B,C)
    return st.ravel()


def tally_less(qA, qB, qC):
    """The round with the tally wire and its three CNOTs erased."""
    inp = ket(qA, 0, qB, 0, qC, 0) + ket(qA, 1, qB, 1, qC, 1)
    st = kron(P, P, P) @ inp
    st = kron(bra(qA), I2, bra(qB), I2, bra(qC), I2) @ st
    return st.ravel()


def blackboard():
    # the node is unnormalized, and kets are written top wire first
    assert np.isclose(np.linalg.norm(N), np.sqrt(2)) and np.isclose(np.vdot(N, N), 2)
    assert ket(1, 0, 0).ravel()[4] == 1                     # |100> is index 4

    # --- P pinned between |q> and <q> is exactly its block -------------------
    for q in (0, 1):
        ket_q = kron(ket(q), I2)                        # |q> (x) I  (4x2)
        bra_q = kron(bra(q), I2)                        # <q| (x) I  (2x4)
        assert np.allclose(bra_q @ P @ ket_q, block(q))         # its block
        other = kron(bra(1 - q), I2)
        assert np.allclose(other @ P @ ket_q, 0)                # the other: nothing

    deals = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]        # r = OR of questions
    for (qA, qB, qC) in deals:
        r = qA | qB | qC
        # losing slip: the zero vector, exactly
        assert np.allclose(round_state(qA, qB, qC, r, s=1), 0)
        # winning slip: the tally-less picture, exactly
        assert np.allclose(round_state(qA, qB, qC, r, s=0), tally_less(qA, qB, qC))
        # the wrong slip word does NOT vanish
        assert not np.allclose(round_state(qA, qB, qC, 1 - r, s=1), 0)

    # an undealt triple (r = OR = 1) loses either way: both slips are non-zero
    for s in (0, 1):
        assert not np.allclose(round_state(1, 0, 0, r=1, s=s), 0)

    # the dialogue's deal 011 with notes (dark, bright, bright) = |100>: amp 1/sqrt(2)
    out = tally_less(0, 1, 1)
    assert np.isclose(abs(out[4]), 1 / np.sqrt(2))     # |100> is index 4, top wire first
    assert np.isclose(np.vdot(out, out), 2)            # squared norm 2 = <N|N>

    # --- nothing travels: each player's own answer is 1/2, 1/2 every triple --
    for qA in (0, 1):
        for qB in (0, 1):
            for qC in (0, 1):
                state = (kron(block(qA), block(qB), block(qC)) @ (N / np.sqrt(2)))
                t = state.reshape(2, 2, 2)
                pA0 = np.sum(np.abs(t[0, :, :]) ** 2)
                pB0 = np.sum(np.abs(t[:, 0, :]) ** 2)
                pC0 = np.sum(np.abs(t[:, :, 0]) ** 2)
                assert np.isclose(pA0, 1 / 2)
                assert np.isclose(pB0, 1 / 2)
                assert np.isclose(pC0, 1 / 2)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws each player's box close up in his own room
# (\triobox[envelope]{way}{caption}) and, on the return, the hall table
# (\halltable{state}{caption}).  The Lua COMPUTES the slip's word from the
# deal (even for all doors, odd otherwise), each spent box's open way from the
# deal, and refuses counters that break the slip's word ("the workshop's ledger
# shows no trio ever breaking its guarantee").  tools/props_ch07.py transcribes
# the Lua; the calls are read out of dialog.md, never copied.  This file also
# checks the whole dialogue's pictures: their number, their places, the keyed
# rule, the captions and the forbidden views (identity card, section 4).
#
# The state model (identity card, section 3): left to right on the table
# Achilles's, the Tortoise's, the Anteater's -- box, envelope and counter --
# which are legs A, B, C and question wires q_A, q_B, q_C; the slip is the
# tally wire, its word r = q_A or q_B or q_C (door 0, drawer 1; even 0, odd 1);
# white = bright = answer 0, black = dark = answer 1.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=props.MACROS)
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth")
SIZES = {"First": [1, 1, 1, 1], "Second": [1, 1, 1, 1], "Third": [2, 1, 1],
         "Fourth": [1, 1, 1], "Fifth": [1, 1, 1, 1]}
DRAWS = {"First": {"notecard"}, "Second": {"scorebook", "halltable"},
         "Third": {"halltable", "triobox", "notecard"}, "Fourth": {"notecard"},
         "Fifth": {"triobox", "halltable"}}
WAY = {0: "door", 1: "drawer"}
BIT = {"white": 0, "black": 1}
DEALT = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]


def check_transcription():
    """The Lua's slip against the Round's tally wire (a structural check, not a
    second proof): the slip's word is r = q_A or q_B or q_C, and for every deal
    the counters the macro accepts are exactly the answers with a non-zero
    amplitude in the node under that deal's blocks (the Round with the winning
    slip); those it refuses have amplitude exactly 0 -- which is why the Round
    with the losing slip is the zero vector."""
    assert [" ".join(WAY[b] for b in q) for q in DEALT] == list(props.DEALS)
    for q in DEALT:
        deal = " ".join(WAY[b] for b in q)
        r = q[0] | q[1] | q[2]
        assert props.slip_word(deal) == ("odd" if r else "even")
        amps = tally_less(*q)
        assert np.allclose(round_state(*q, r, s=1), 0)
        for faces in itertools.product(props.FACES, repeat=3):
            a, b, c = (BIT[f] for f in faces)
            live = not np.isclose(amps[4 * a + 2 * b + c], 0)          # top wire first
            state = f"boxes out; spent; deal {deal}; counters {' '.join(faces)}; slip open"
            assert (props.faults(props.halltable, "", state) == []) == live
            t = props.halltable("", state)
            assert t.cloth == t.deal == tuple(deal.split()) and t.slip_shows == props.slip_word(deal)
    # an undealt triple is refused; the macro's other refusals
    assert props.faults(props.halltable, "", "deal door door drawer") != []
    assert props.faults(props.halltable, "", "case open; boxes out") != []                # the trio twice
    assert props.faults(props.halltable, "", "envelopes open; envelopes sealed; deal door door door") != []
    assert props.faults(props.halltable, "", "boxes out; spent; deal door door door; counters white white") != []
    assert props.faults(props.halltable, "", "boxes out; spent; deal door door door; counters white grey white") != []
    assert props.faults(props.halltable, "", "boxes on the floor") != []                  # an unknown item
    assert props.faults(props.halltable, "", "envelopes open") != []                      # needs the deal
    assert props.faults(props.halltable, "", "slip open") != []
    assert props.faults(props.halltable, "", "boxes out; spent") != []
    assert props.faults(props.halltable, "", "case open; spent; deal door door door") != []   # spent, not out
    assert props.faults(props.halltable, "", "boxes out; deal door door door; counters white white white") != []
    assert props.faults(props.halltable, "", "boxes out; spent; deal door drawer drawer; "
                                     "counters white white white") != []          # breaks the slip
    assert props.faults(props.halltable, "", "") == [] and props.halltable("", "").trio == "case shut"
    # the handwriting keeps apart what a reader must tell apart on the table
    assert props.OUTLINES["door"] != props.OUTLINES["drawer"] and props.OUTLINES["even"] != props.OUTLINES["odd"]
    assert len({props.OUTLINES[w] for w in props.WHO}) == 3


def check_whole_dialogue():
    assert DIALOGUE.check_whole(ORDINALS) == 19                  # 4 + 4 + 4 + 3 + 4
    raw = DIALOGUE.read(DIALOGUE.raw)
    assert len(DIALOGUE.call_re.findall(raw)) == 19             # dialog.raw.md, the same
    DIALOGUE.keyed_first(ORDINALS, kind=props.keyed_kind)
    placed = [(o, i, c) for o in ORDINALS
              for i, c in enumerate(c for b in DIALOGUE.blocks(o) for c in b)]
    assert [(o, i) for o, i, c in placed if c.keyed] == [
        ("First", 0), ("First", 1), ("Second", 0), ("Second", 3),
        ("Third", 1), ("Third", 2), ("Fourth", 0)]
    for o in ORDINALS:
        assert [len(b) for b in DIALOGUE.blocks(o)] == SIZES[o], o
        assert {c.macro for oo, _, c in placed if oo == o} == DRAWS[o], o
    for o, i, c in placed:
        pic = props.picture(c)                                        # no macro ever faults
        assert words_only(c.caption, forbidden=props.NOT_IN_CAPTIONS), c.caption
        assert "round" not in c.caption.lower()                  # never the master's name
        if c.macro in props.DRAWS_WIRE_OBJECTS:                      # no count of the wire-objects
            assert not props.quantity_words(c.caption), c.caption
        if c.macro == "notecard":
            assert not (pic.side == "back" and pic.view == "card")        # P3.3 is the first back
            assert pic.view != "under" or pic.cup is not None             # never a lifted cup
        if c.macro == "halltable":
            assert pic.trio not in ("case shut", "boxes away") and pic.envelopes != "away"


def pictures():
    check_transcription()
    check_whole_dialogue()
    p51, p52, p53, p54 = DIALOGUE.check_layout("Fifth", sizes=[1, 1, 1, 1])
    assert [c.macro for c in (p51, p52, p53, p54)] == ["triobox"] * 3 + ["halltable"]
    for c in (p51, p52, p53, p54):
        assert words_only(c.caption, forbidden=props.NOT_IN_CAPTIONS), c.caption
        assert not props.quantity_words(c.caption), c.caption
    x51, x52, x53, t54 = (props.picture(c) for c in (p51, p52, p53, p54))
    sec = DIALOGUE.section("Fifth")

    # each solo's picture directly after that solo's speech, captioned by its room
    a1, a2, a3, a4 = props.anchors(DIALOGUE, "Fifth")
    assert a1.startswith("**Achilles:** You can see all three of us")
    assert a1.endswith("It might as well have been a coin that sent me up here.")
    assert a2.startswith("**Tortoise:** Mine says drawer.") and a2.endswith("hold on to them.")
    assert a3.startswith("**Anteater:** My envelope says drawer as well.")
    assert a3.endswith("only you are standing where that can be seen.")
    assert a4.startswith("**Achilles:** Now the counters.") and a4.endswith("Odd. One black. Won.")
    paras = props.paragraphs(DIALOGUE, "Fifth")
    solos = [i for i, q in enumerate(paras) if "\\triobox" in q]
    rooms = [("Achilles", "the attic"), ("Tortoise", "the cellar"), ("Anteater", "the garden room")]
    for i, (who, room), c in zip(solos, rooms, (p51, p52, p53)):
        assert paras[i - 2] == f"*[{'The ' if who != 'Achilles' else ''}{who}, alone in {room}.]*"
        assert paras[i - 1].startswith(f"**{who}:**") and c.caption == f"in {room}"
    # no picture before the solos begin; nothing drawn after the slip is read
    assert sec.index("*[Achilles, alone in the attic.]*") < sec.index("\\begin{center}")
    assert not any("\\" + m in sec.split(p54.text)[1] for m in props.MACROS)

    # P5.1-P5.3: the way each opened, and the envelope beside it naming that way
    assert [(x.way, x.envelope) for x in (x51, x52, x53)] == [
        ("door", "door"), ("drawer", "drawer"), ("drawer", "drawer")]
    assert not any(x.keyed for x in (x51, x52, x53))
    assert '"Door." So I swing the door.' in a1 and "Mine says drawer." in a2
    assert "says drawer as well" in a3
    notes = ["dark" if "Dark." in a1 else "bright", "bright" if "Bright." in a2 else "dark",
             "bright" if "Bright." in a3 else "dark"]
    assert notes == ["dark", "bright", "bright"]                     # heard in the rooms

    # P5.4: back on the cloth where they stood, left to right Achilles, Tortoise,
    # Anteater, each still open the way it was opened; the envelopes before them;
    # a counter by each for its note; the slip unfolded, its word computed
    assert "Achilles's on the left, the Tortoise's in the middle, the Anteater's on the right" in sec
    assert t54.trio == "boxes out" and t54.spent and t54.nests == (None,) * 3
    assert t54.deal == ("door", "drawer", "drawer")                  # the card's fixed deal
    assert t54.cloth == tuple(x.way for x in (x51, x52, x53)) == t54.deal
    assert t54.envelopes == "open" and t54.envelope_faces == t54.deal
    assert t54.by_boxes == ("black", "white", "white") == tuple(
        {"dark": "black", "bright": "white"}[n] for n in notes)
    assert sum(f == "black" for f in t54.by_boxes) == 1              # "One black."
    assert t54.slip == "open" and t54.slip_shows == "odd" and t54.glass == "set aside"
    assert "Odd. One black. Won." in a4
    # the same moment on the blackboard: deal 011, r = 1; the outcome |100> lives
    q = tuple({"door": 0, "drawer": 1}[w] for w in t54.deal)
    assert q == (0, 1, 1) and (q[0] | q[1] | q[2]) == 1
    a = tuple(BIT[f] for f in t54.by_boxes)
    assert np.isclose(abs(tally_less(*q)[4 * a[0] + 2 * a[1] + a[2]]), 1 / np.sqrt(2))
    assert np.allclose(round_state(*q, 1, s=1), 0)                   # the losing slip: nothing


# --------------------------------------------------------------------------
# The sketchbook.  PyZX has no controlled box, so we draw the FOUR dealt
# instances with P replaced by its block per player: H = a Hadamard edge,
# HS* = a phase node (-1/2) then a Hadamard edge.  A CNOT is a control node on
# the leg and a target node on the tally, copied from chapter04/Code/Ch4/
# Yoke.py (add_power(1) each).  The two crossings need no node -- connectivity
# is all that counts, so the tally simply joins each leg's control (bending
# wires is free).  A picture equal to zero is asserted with np.allclose(
# matrix_of(g), 0): same_picture against a picture is not available for zero.
# Closed comparisons go through matrix_of / same_picture ("naive" strategy).
# --------------------------------------------------------------------------
def block_on_leg(p, dot, row, q, ctrl_col):
    """From the node, the player's block, then the CNOT control node on the leg."""
    if q == 1:                                          # HS* : phase node then H edge
        a = p.z(row, 2, Fraction(-1, 2))
        b = p.z(row, 3, 0)
        p.wire(dot, a)                                  # PyZX draws this with the spiders you meet in Chapter 8.
        p.g.add_edge((a, b), EdgeType.HADAMARD)
    else:                                               # H : a Hadamard edge
        b = p.z(row, 3, 0)   # PyZX draws this with the spiders you meet in Chapter 8.
        p.g.add_edge((dot, b), EdgeType.HADAMARD)
    ctrl = p.z(row, ctrl_col, 0)
    p.wire(b, ctrl, p.out(row))
    return ctrl


def the_round(qA, qB, qC, s):
    """The dealt round as one connected picture, with slip effect s."""
    p = Picture()
    dot = p.z(1, 0)   # the node: PyZX draws this with the spiders you meet in Chapter 8.
    controls = [block_on_leg(p, dot, r, q, ctrl_col=6 + r)
                for r, q in enumerate((qA, qB, qC))]
    r = qA | qB | qC
    tstate = p.x(3, 0, Fraction(r))                     # tally state sqrt(2)|r>
    targets = [p.x(3, 2 + 2 * i, 0) for i in range(3)]  # three CNOT targets
    for t, ctrl in zip(targets, controls[::-1]):        # C first, then B, then A
        p.wire(t, ctrl)                                 # the CNOT edge
    teff = p.x(3, 9, Fraction(s))                       # the slip's effect sqrt(2)<s|
    p.wire(tstate, targets[0], targets[1], targets[2], teff)
    g = p.done()
    g.scalar.add_power(1)          # +3 for the three CNOTs, -2 for tally state and effect
    return g


def node_with_blocks(qA, qB, qC):
    """The tally wire and its CNOTs erased: node + each player's block, open."""
    p = Picture()
    dot = p.z(1, 0)   # the node: PyZX draws this with the spiders you meet in Chapter 8.
    for r, q in enumerate((qA, qB, qC)):
        if q == 1:
            a = p.z(r, 2, Fraction(-1, 2))
            b = p.z(r, 3, 0)
            p.wire(dot, a)
            p.g.add_edge((a, b), EdgeType.HADAMARD)
        else:
            b = p.z(r, 3, 0)
            p.g.add_edge((dot, b), EdgeType.HADAMARD)
        p.wire(b, p.out(r))
    return p.done()


def lone_x_state(r):
    """A one-legged X state (before its sqrt(2)): stands for sqrt(2)|r>."""
    p = Picture()
    s = p.x(0, 1, Fraction(r))   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(s, p.out(0))
    return p.done()


def lone_x_effect(s):
    """A one-legged X effect (before its sqrt(2)): stands for sqrt(2)<s|."""
    p = Picture()
    e = p.x(0, 1, Fraction(s))   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(p.inp(0), e)
    return p.done()


def hadamard_edge_box():
    """A bare Hadamard edge from input to output: the block H (scalar included)."""
    p = Picture()
    p.g.add_edge((p.inp(0), p.out(0)), EdgeType.HADAMARD)   # PyZX draws this with the spiders you meet in Chapter 8.
    return p.done()


def hsdagger_box():
    """A phase node (-1/2) then a Hadamard edge: the block HS*."""
    p = Picture()
    a = p.z(0, 1, Fraction(-1, 2))   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(p.inp(0), a)
    p.g.add_edge((a, p.out(0)), EdgeType.HADAMARD)
    return p.done()


def sketchbook():
    # the pieces, literally, against NumPy.  The tally's X state/effect each cost
    # a sqrt(2); the two blocks (a Hadamard edge, and a phase node then that edge)
    # carry their 1/sqrt(2) inside the edge, so they need no scalar.
    for bit in (0, 1):
        assert np.allclose(matrix_of(lone_x_state(bit)).ravel(), np.sqrt(2) * ket(bit))
        assert np.allclose(matrix_of(lone_x_effect(bit)), np.sqrt(2) * bra(bit))
    assert np.allclose(matrix_of(hadamard_edge_box()), H)        # block for the door
    assert np.allclose(matrix_of(hsdagger_box()), HSd)           # block for the drawer

    deals = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    for (qA, qB, qC) in deals:
        # the losing slip: the zero picture
        assert np.allclose(matrix_of(the_round(qA, qB, qC, s=1)), 0)
        # the winning slip: the tally-less picture, exactly (scalar included)
        assert same_picture(the_round(qA, qB, qC, s=0), node_with_blocks(qA, qB, qC))
        # and the winning picture is not itself zero
        assert not np.allclose(matrix_of(the_round(qA, qB, qC, s=0)), 0)

    # the four winning amplitudes are 1/sqrt(2) each: squared norm 2 = <N|N>
    win = matrix_of(node_with_blocks(0, 1, 1)).ravel()
    assert np.isclose(abs(win[4]), 1 / np.sqrt(2))     # deal 011 -> |100>
    assert np.isclose(np.vdot(win, win), 2)


if __name__ == "__main__":
    verdict("Round", blackboard, sketchbook)
