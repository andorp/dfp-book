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
# (\triobox[envelope]{way}{caption}, and a step strip of it, \triosteps
# [opts]{state}{steps}{captions}: the box shut, then opened the named way) and,
# on the return, the hall table (a strip, \hallsteps: the counters laid, then
# the glass slid off and the slip unfolded).  The Lua COMPUTES the slip's word
# from the deal (even for all doors, odd otherwise), each spent box's open way
# from the deal, and refuses counters that break the slip's word ("the
# workshop's ledger shows no trio ever breaking its guarantee"); and, round 4,
# the state after each step and its marks (a door's ghost, in outline, and a
# swing arc, a drawer's ghost, in outline, and a slide arc clear of the drawer,
# the pin ring of the steadying hand; the glass's ghost and slide arc).  tools/props_ch07.py transcribes the Lua (proved
# against texlua; P.check_lua holds the Lua and the .tex to the text it was
# proved on); the calls are read out of dialog.md and dialog.raw.md, never
# copied.  This file also checks the whole dialogue's pictures: their number,
# their places, the keyed rule, the predictions, the captions, the forbidden
# views, the log of the whole dialogue, and every refusal the Lua can raise
# (identity card, sections 4 and 3e).
#
# The state model (identity card, section 3): left to right on the table
# Achilles's, the Tortoise's, the Anteater's -- box, envelope and counter --
# which are legs A, B, C and question wires q_A, q_B, q_C; the slip is the
# tally wire, its word r = q_A or q_B or q_C (door 0, drawer 1; even 0, odd 1);
# white = bright = answer 0, black = dark = answer 1.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=props.MACROS)
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth")
SIZES = {"First": [1, 1, 2, 1], "Second": [1, 1, 1, 1], "Third": [2, 1, 1, 1],
         "Fourth": [1, 1, 2], "Fifth": [1, 1, 1, 1]}                   # calls, block by block
FRAMES = {"First": [1, 4, 2, 1], "Second": [1, 1, 1, 2], "Third": [4, 1, 1, 2],
          "Fourth": [1, 1, 2], "Fifth": [2, 2, 1, 2]}                  # frames, block by block
DRAWS = {"First": {"notecard", "cardsteps"}, "Second": {"scorebook", "hallsteps"},
         "Third": {"hallsteps", "triobox", "notecard", "cardsteps"}, "Fourth": {"notecard"},
         "Fifth": {"triosteps", "triobox", "hallsteps"}}
WAY = {0: "door", 1: "drawer"}
BIT = {"white": 0, "black": 1}
DEALT = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
FW = ("front; all white",)
# One call for each refusal ch07.lua can raise, in the order the sites stand in the Lua
# (props.TEX_SITES): each must raise its own (and no call may stop the Lua itself).
SITE_CALLS = (
    ("triobox", "lid=x", ("shut",), "c"),
    ("triobox", "open=door; open=door", ("shut",), "c"),
    ("triobox", "door; drawer", ("shut",), "c"),
    ("triobox", "open=door; marks=maybe", ("shut",), "c"),
    ("notecard", "masks; turn=left lid; lay=next mask", FW, "c"),
    ("triobox", "keyroom=off", ("shut",), "c"),
    ("notecard", "card", ("side; all white",), "c"),
    ("notecard", "card", ("front; all white; all black",), "c"),
    ("notecard", "card", ("front; middle lid white",), "c"),
    ("notecard", "card", ("front; left lid grey",), "c"),
    ("notecard", "under two drawers", FW, "c"),
    ("notecard", "sideways", FW, "c"),
    ("notecard", "masks, button on two lids", FW, "c"),
    ("notecard", "card, cup on left lid", FW, "c"),
    ("notecard", "under two lids, cup on left lid, cup on right lid", FW, "c"),
    ("notecard", "masks", ("front; left lid white",), "c"),
    ("notecard", "under two lids, cup on left door", FW, "c"),
    ("notecard", "under two lids, cup on left lid", ("front; left lid white",), "c"),
    ("notecard", "under two lids, cup on left lid", ("front; left lid white, right lid black",), "c"),
    ("notecard", "masks; turn=middle lid", FW, "c"),
    ("notecard", "card; turn=left lid", ("front; right lid white",), "c"),
    ("notecard", "under two lids, cup on left lid; turn=left lid", FW, "c"),
    ("notecard", "card; turn=left lid and left lid", FW, "c"),
    ("notecard", "card; turn=", FW, "c"),
    ("notecard", "card; lay=all", FW, "c"),
    ("notecard", "masks; lay=next mask", FW, "c"),
    ("notecard", "card; lift=cup", FW, "c"),
    ("notecard", "card; lift=mask", FW, "c"),
    ("notecard", "under two lids, cup on left lid; lift=mask", FW, "c"),
    ("notecard", "card; lift=lid", FW, "c"),
    ("notecard", "card; cup=left lid", FW, "c"),
    ("notecard", "under two lids; cup=left door", FW, "c"),
    ("notecard", "under two lids, cup on left lid; cup=left lid", FW, "c"),
    ("notecard", "masks; expects=mended", FW, "c"),
    ("notecard", "masks; marks=off", FW, "c"),
    ("notecard", "masks; turn=left lid; expects=gaps", FW, "c"),
    ("notecard", "three masks; lay=next mask; expects=mended", FW, "c"),
    ("triobox", "", ("open",), "c"),
    ("triobox", "lid", ("shut",), "c"),
    ("triobox", "door", ("drawer out",), "c"),
    ("triobox", "open=lid", ("shut",), "c"),
    ("triobox", "open=door", ("door open",), "c"),
    ("halltable", "", ("case open; case shut",), "c"),
    ("halltable", "", ("envelopes open; envelopes sealed; deal door door door",), "c"),
    ("halltable", "", ("deal door door drawer",), "c"),
    ("halltable", "", ("boxes out; spent; deal door door door; counters white white",), "c"),
    ("halltable", "", ("boxes out; spent; deal door door door; counters white grey white",), "c"),
    ("halltable", "", ("boxes on the floor",), "c"),
    ("halltable", "", ("envelopes open",), "c"),
    ("halltable", "", ("case open; spent; deal door door door",), "c"),
    ("halltable", "", ("boxes out; deal door door door; counters white white white",), "c"),
    ("halltable", "", ("boxes out; spent; deal door drawer drawer; counters white white white",), "c"),
    ("halltable", "open=case", ("case open",), "c"),
    ("halltable", "open=slip", ("slip open; deal door door door",), "c"),
    ("halltable", "open=slip", ("case shut",), "c"),
    ("halltable", "open=lid", ("case shut",), "c"),
    ("halltable", "lift=boxes", ("case shut",), "c"),
    ("halltable", "lift=next box", ("boxes out",), "c"),
    ("halltable", "lift=lid", ("case open",), "c"),
    ("halltable", "case open", ("case shut",), "c"),
    ("scorebook", "sideways", (), "c"),
    ("triosteps", "", ("shut", "open=lid"), "a; b"),
    ("triosteps", "bare", ("shut", "open=door"), "a; b"),
    ("triosteps", "", ("shut", "lift=door; open=door"), "a; b"),
    ("triosteps", "", ("shut", ""), "a"),
    ("hallsteps", "", ("case shut", "open=case; lift=next box; lift=next box; lift=next box"), "a; b; c; d; e"),
    ("triosteps", "", ("shut", "open=door"), "a"),
)


def check_transcription():
    """The Lua's slip against the Round's tally wire (a structural check, not a
    second proof): the slip's word is r = q_A or q_B or q_C, and for every deal
    the counters the macro accepts are exactly the answers with a non-zero
    amplitude in the node under that deal's blocks (the Round with the winning
    slip); those it refuses have amplitude exactly 0 -- which is why the Round
    with the losing slip is the zero vector."""
    props.check_lua()
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
            if live:                                                     # and read by a step
                u = props.clean(props.halltable, "open=slip", state.replace("slip open", "slip sealed"))
                assert u.slip_shows == t.slip_shows and u.spec == t.spec
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
    # round 4: the slip read twice; a box opened that stands open
    assert props.faults(props.halltable, "open=slip", "slip open; deal door door door") != []
    assert props.faults(props.triobox, "door; open=door", "door open") != []
    # the handwriting keeps apart what a reader must tell apart on the table
    assert props.OUTLINES["door"] != props.OUTLINES["drawer"] and props.OUTLINES["even"] != props.OUTLINES["odd"]
    assert len({props.OUTLINES[w] for w in props.WHO}) == 3


def check_every_site():
    """Every refusal of ch07.lua, reached by one call, in the Lua's order (and TEX_SITES is
    the Lua's own list of them)."""
    props.check_sites()
    assert len(SITE_CALLS) == len(props.TEX_SITES)
    for i, (macro, opt, args, captions) in enumerate(SITE_CALLS):
        props.TEX_ERRORS.clear()
        props.run_call(macro, False, opt, args, captions)
        assert i in {j for e in props.TEX_ERRORS for j in props.sites_of(e)}, (i, props.TEX_SITES[i])
    # the strips' own: more than four frames, frames and captions of different counts
    assert props.faults(props.strip, "cardsteps", False, "front; all white", "from=off",
                        "lay=next mask; " * 5, "a; b; c; d; e") != []
    assert props.faults(props.strip, "hallsteps", False, "case shut", "", "open=case", "a") != []


def check_whole_dialogue():
    """The whole dialogue's pictures (both files): 22 calls, 32 frames, 7 strips, 2
    predictions, the keyed rule, the forbidden views, and the 88 lines of its log."""
    total, frames, starred, log = props.check_dialogue(DIALOGUE, ORDINALS)
    assert total == 22                                           # 5 + 4 + 5 + 4 + 4
    raw = DIALOGUE.read(DIALOGUE.raw)
    assert len(DIALOGUE.call_re.findall(raw)) == 22             # dialog.raw.md, the same
    assert starred == [("First", 0), ("First", 1), ("Second", 0), ("Second", 3),
                       ("Third", 1), ("Third", 2), ("Fourth", 0)]
    calls = [(o, c) for o in ORDINALS for b in DIALOGUE.blocks(o) for c in b]
    assert len(frames) == 32
    assert sum(props.is_strip(c) for _, c in calls) == 7
    assert [(o, cap) for o, c, p, cap in frames if p.expects] == [
        ("First", "as Achilles expects"), ("Fourth", "as Achilles expects")]
    assert len(log) == 88 and sum(": + " not in l and "strip of" not in l for l in log) == 32
    assert sum("strip of" in l for l in log) == 7
    for path in (DIALOGUE.dialog, DIALOGUE.raw):                 # the same pictures in both
        assert [c.text for o in ORDINALS for b in DIALOGUE.blocks(o, path) for c in b] == \
            [c.text for _, c in calls]
    for o in ORDINALS:
        assert [len(b) for b in DIALOGUE.blocks(o)] == SIZES[o], o
        assert [sum(len(props.pictures(c)) for c in b) for b in DIALOGUE.blocks(o)] == FRAMES[o], o
        assert {c.macro for oo, c in calls if oo == o} == DRAWS[o], o
    for o, c, pic, cap in frames:
        if props.BASE[c.macro] == "notecard":
            assert not (pic.side == "back" and pic.view == "card")        # P3.3 is the first back
    # the egg-cup off the card: only after the Anteater lifts it, until he stands it again
    assert [cap for o, c, p, cap in frames
            if props.BASE[c.macro] == "notecard" and p.view == "under" and p.cup is None] == \
        ["the cup lifted", "set back black"]
    # the case shut once, before it is opened; the nests never empty; the envelopes never away
    tables = [(cap, p) for o, c, p, cap in frames if props.BASE[c.macro] == "halltable"]
    assert [cap for cap, p in tables if p.trio == "case shut"] == ["lid down"]
    assert not any(p.trio == "boxes away" or p.envelopes == "away" for _, p in tables)
    # every marked frame carries its step; no prediction and no before-frame is marked
    for o, c, p, cap in frames:
        assert not p.marked or (p.step is not None and p.expects is None), (o, cap)
    # the card's step frames: seventeen, all marked but F21 (the cup stood from the hand)
    stepped = [cap for o, c, p, cap in frames if p.step is not None and p.expects is None]
    marked = [cap for o, c, p, cap in frames if p.marked]
    assert len(stepped) == 17 and [cap for cap in stepped if cap not in marked] == ["a different window"]
    assert len(marked) == 16 and "the counters laid" not in marked and "shut and full" not in marked


def pictures():
    check_transcription()
    check_every_site()
    check_whole_dialogue()
    found = props.check_section(DIALOGUE, "Fifth", sizes=[1, 1, 1, 1], frames=[2, 2, 1, 2])
    (p51, [(x26, cap26), (x51, cap51)]), = found[0]
    (p52, [(x28, cap28), (x52, cap52)]), = found[1]
    (p53, [(x53, cap53)]), = found[2]
    (p54, [(t31, cap31), (t54, cap54)]), = found[3]
    assert [c.macro for c in (p51, p52, p53, p54)] == ["triosteps", "triosteps", "triobox", "hallsteps"]
    assert (cap26, cap28, cap31, cap54) == ("shut and full", "shut and full", "the counters laid", "the slip read")
    sec = DIALOGUE.section("Fifth")

    # each solo's picture directly after that solo's speech, captioned by its room
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a1, a2, a3, a4 = DIALOGUE.anchors("Fifth", path)
        assert a1.startswith("**Achilles:** You can see all three of us")
        assert a1.endswith("It might as well have been a coin that sent me up here.")
        assert a2.startswith("**Tortoise:** Mine says drawer.") and a2.endswith("hold on to them.")
        assert a3.startswith("**Anteater:** My envelope says drawer as well.")
        assert a3.endswith("only you are standing where that can be seen.")
        assert a4.startswith("**Achilles:** Now the counters.") and a4.endswith("Odd. One black. Won.")
        paras = DIALOGUE.paragraphs("Fifth", path)
        solos = [i for i, q in enumerate(paras) if "\\triobox" in q or "\\triosteps" in q]
        rooms = [("Achilles", "the attic"), ("Tortoise", "the cellar"), ("Anteater", "the garden room")]
        for i, (who, room) in zip(solos, rooms):
            assert paras[i - 2] == f"*[{'The ' if who != 'Achilles' else ''}{who}, alone in {room}.]*"
            assert paras[i - 1].startswith(f"**{who}:**")
    # the room's name alone under the frame that shows the box opened there (C2-C4)
    assert (cap51, cap52, cap53) == ("attic", "cellar", "garden room")
    for room, cap in zip(("the attic", "the cellar", "the garden room"), (cap51, cap52, cap53)):
        assert cap == room[len("the "):] and f"alone in {room}" in sec
    # the direction and the captions edited this round (D6, C2-C4), in both files
    props.check_edits(DIALOGUE, "Fifth", [
        ("*[He slides the glass paperweight off the slip and unfolds it.]*",
         "*[He slides the glass paperweight off the slip, towards the case, and unfolds it.]*"),
        ("}{in the attic}", "; attic}"), ("}{in the cellar}", "; cellar}"),
        ("}{in the garden room}", "}{garden room}")])
    # no picture before the solos begin; nothing drawn after the slip is read
    assert sec.index("*[Achilles, alone in the attic.]*") < sec.index("\\begin{center}")
    assert not any("\\" + m in sec.split(p54.text)[1] for m in props.MACROS)

    # Blocks 5.1, 5.2: each box shut and full beside its opened envelope (the before-frame:
    # the close-up's last picture, P3.2's box, shut), then opened the named way
    (_, x32), = [(c, props.pictures(c)[0][0]) for c in DIALOGUE.blocks("Third")[0] if c.macro == "triobox"]
    for p, x0_, x1_, way in ((p51, x26, x51, "door"), (p52, x28, x52, "drawer")):
        assert [(s.opts, s.state) for s in props.frames_of(p)] == [(way, "shut"), (way + "; open=" + way, "shut")]
        assert x0_.way == x32.way == "shut" and x0_.envelope == way and not x0_.marked and x0_.step is None
        assert x1_.way == way and x1_.envelope == way and x1_.step == ("open", way)
        assert not p.keyed and not x0_.labels and not x1_.labels
        # the hand that steadies it: one pin ring, on the box's plain top, touching no edge
        g = props.box_geometry(x1_.x0, props.BOXY, props.BOXSC)
        (pin,) = x1_.pins
        corner = np.array((x1_.x0, props.BOXY + g["H"]))
        assert np.allclose((pin.x, pin.y), corner + (np.array((g["W"], 0)) + (g["dx"], g["dy"])) / 2)
        d = np.array((g["dx"], g["dy"])) / np.hypot(g["dx"], g["dy"])
        c = np.array((pin.x, pin.y)) - corner
        assert abs(d[0] * c[1] - d[1] * c[0]) > pin.r and g["dy"] / 2 > pin.r      # inside the top
    assert "the box itself does not stir" in sec and "the box holds its place" in sec
    # each ghost is the part's outline only, dotted, where it stood shut: one stroke in the
    # ghost's own style -- no fill, no knob, no pale copy of the part (which would paint over
    # the part moved)
    for x, what in ((x51, "door"), (x52, "drawer")):
        (gh,) = x.ghosts
        assert gh.what == what and gh.outline
        drawn = [t for scope, t in x.tikz if "propchsevenghost" in t or scope == "ghost"]
        assert len(drawn) == 1 and all(scope == "" for scope, _ in x.tikz)
        assert drawn[0].startswith("\\draw[propchsevenghost") and "fill" not in drawn[0]
        assert "circle" not in drawn[0]
    # P5.1's marks: the door's ghost shut in the side face, and a swing arc round its hinge,
    # out from the box
    g = props.box_geometry(x51.x0, props.BOXY, props.BOXSC)
    sc = props.BOXSC
    assert [(gh.what, gh.at) for gh in x51.ghosts] == [("door", g["a"])]
    assert 0 < g["ta"] < g["tb"] < 1 and 0 < g["za"] < g["zb"] < 1               # in the side face
    corners = " -- ".join(props.pt(*g[k]) for k in "abce") + " -- cycle;"
    assert [t for _, t in x51.tikz][0].endswith(corners)                        # the whole doorway
    (swing,) = x51.arcs
    assert swing.kind == "swing" and swing.tag == "door round the hinge"
    # the hinge is the door's front edge (corner a); from the ghost's free top corner (c),
    # over the open door's top edge, to the open door's free top corner, out to the left
    dw = (g["tb"] - g["ta"]) * g["RD"]
    assert np.allclose(swing.start, (g["c"][0] - 0.01 * sc, g["c"][1] + 0.02 * sc))
    assert np.allclose(swing.end, (g["a"][0] - dw - 0.01 * sc, g["e"][1] + 0.03 * sc))
    assert swing.end[0] < swing.start[0] < g["a"][0]
    assert "catches the little hinged door in its side and swings it open" in sec
    # P5.2's marks: the drawer's ghost shut at the foot, and a slide arc out to the front,
    # parallel to the pull, beside the drawer and clear of it
    dxa, dxb, dza, dzb = props.box_geometry(x52.x0, props.BOXY, props.BOXSC)["drawer"]
    assert [(gh.what, gh.at) for gh in x52.ghosts] == [("drawer", (dxa, dza))]
    assert [t for _, t in x52.tikz][0].endswith(props.pt(dxa, dza) + " rectangle " + props.pt(dxb, dzb) + ";")
    (slide,) = x52.arcs
    assert slide.kind == "slide" and slide.tag == "drawer towards the company"
    assert slide.end[0] > slide.start[0] and slide.end[1] < slide.start[1]     # out of the front, down
    # the drawer as the Lua draws it pulled out (its shut place, open top, left side and front,
    # moved by the pull (sx, sy)) lies inside the hull of its shut and drawn-out corners; every
    # point of the arc is outside that hull, by more than a stroke
    g52 = props.box_geometry(x52.x0, props.BOXY, props.BOXSC)
    sx, sy = -0.42 * g52["dx"], -0.42 * g52["dy"]
    hull = [(dxa + sx, dza + sy), (dxb + sx, dza + sy), (dxb + sx, dzb + sy), (dxb, dzb), (dxa, dzb), (dxa, dza)]
    edges = list(zip(hull, hull[1:] + hull[:1]))

    def side(q, a, b):
        return (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])

    def seg_dist(q, a, b):
        v = np.subtract(b, a)
        t = min(1, max(0, np.dot(np.subtract(q, a), v) / np.dot(v, v)))
        return float(np.hypot(*(np.subtract(q, a) - t * v)))
    centre = tuple(np.mean(hull, axis=0))
    assert all(side(centre, a, b) > 0 for a, b in edges)                        # convex, counter-clockwise
    pts = slide.points()
    assert not any(all(side(q, a, b) > 0 for a, b in edges) for q in pts)
    assert min(seg_dist(q, a, b) for q in pts for a, b in edges) > 0.1          # clear of the drawer
    ux, uy = np.array((sx, sy)) / np.hypot(sx, sy)
    vx, vy = np.subtract(slide.end, slide.start)
    assert np.isclose(vx * uy - vy * ux, 0, atol=1e-9)                     # along the pull
    assert "draws out the little drawer from underneath" in sec
    # P5.3: the Anteater's, a repeat of the drawer drawn out: unmarked
    assert x53.way == "drawer" and x53.envelope == "drawer" and not x53.marked and x53.step is None

    # P5.1-P5.3: the way each opened, and the envelope beside it naming that way
    assert [(x.way, x.envelope) for x in (x51, x52, x53)] == [
        ("door", "door"), ("drawer", "drawer"), ("drawer", "drawer")]
    assert not any(x.keyed for x in (x51, x52, x53))
    assert '"Door." So I swing the door.' in a1 and "Mine says drawer." in a2
    assert "says drawer as well" in a3
    notes = ["dark" if "Dark." in a1 else "bright", "bright" if "Bright." in a2 else "dark",
             "bright" if "Bright." in a3 else "dark"]
    assert notes == ["dark", "bright", "bright"]                     # heard in the rooms

    # Block 5.4: back on the cloth where they stood, left to right Achilles, Tortoise,
    # Anteater, each still open the way it was opened; the envelopes before them; a counter
    # by each for its note (F31, the before-frame: the table's last picture, block 3.1's
    # F16, with what l.510-l.512 lay on it); then the slip unfolded, its word computed
    t16 = props.pictures(DIALOGUE.blocks("Third")[0][0])[-1][0]
    assert t31.trio == t16.trio == "boxes out" and t31.nests == t16.nests == (None,) * 3
    assert t31.slip == t16.slip == "sealed" and t31.glass == t16.glass == props.GLASS0
    assert t16.envelopes == "sealed" and t31.envelopes == "open" and t31.spent and not t16.spent
    assert t31.by_boxes == ("black", "white", "white") and not t31.marked and t31.step is None
    assert [(s.opts, s.state) for s in props.frames_of(p54)] == [("", p54.args[0]), ("open=slip", p54.args[0])]
    assert "Achilles's on the left, the Tortoise's in the middle, the Anteater's on the right" in sec
    assert t54.trio == "boxes out" and t54.spent and t54.nests == (None,) * 3
    assert t54.deal == ("door", "drawer", "drawer")                  # the card's fixed deal
    assert t54.cloth == tuple(x.way for x in (x51, x52, x53)) == t54.deal
    assert t54.envelopes == "open" and t54.envelope_faces == t54.deal
    assert t54.by_boxes == ("black", "white", "white") == tuple(
        {"dark": "black", "bright": "white"}[n] for n in notes)
    assert sum(f == "black" for f in t54.by_boxes) == 1              # "One black."
    assert t54.slip == "open" and t54.slip_shows == "odd" and t54.glass == props.GLASS1
    assert "Odd. One black. Won." in a4
    # its marks (D6): the glass's ghost on the folded slip and a slide arc from there,
    # towards the case, to the glass set aside -- one glass, drawn at one size; it stands
    # between the case and the slip, its body clear of both (the strokes meet them by a
    # hairline, as ruled: identity card 3e.7 Q2, "touching both")
    assert t54.step == ("open", "slip") and t54.pins == []
    assert [(g.what, g.at) for g in t54.ghosts] == [("glass", props.GLASS0)]
    (glide,) = t54.arcs
    assert glide.kind == "slide" and glide.tag == "glass to beside the slip"
    (x0, y0, r0), (x1, y1, r1) = props.GLASS0, props.GLASS1
    assert r0 == r1                                                  # the same glass, the same size
    assert np.allclose(glide.start, (x0, y0 - r0 - 0.02)) and np.allclose(glide.end, (x1 + 0.04, y1 - r1 - 0.02))
    assert x1 < x0 and glide.end[0] < glide.start[0]                 # towards the case
    assert props.CX1 < x1 - r1 and x1 + r1 < props.SLIPX[0]          # between the case and the slip
    assert props.SLIPX[0] <= x0 - r0 and x0 + r0 <= props.SLIPX[1]   # on the slip, before
    # the same moment on the blackboard: deal 011, r = 1; the outcome |100> lives
    q = tuple({"door": 0, "drawer": 1}[w] for w in t54.deal)
    assert q == (0, 1, 1) and (q[0] | q[1] | q[2]) == 1
    a = tuple(BIT[f] for f in t54.by_boxes)
    assert np.isclose(abs(tally_less(*q)[4 * a[0] + 2 * a[1] + a[2]]), 1 / np.sqrt(2))
    assert np.allclose(round_state(*q, 1, s=1), 0)                   # the losing slip: nothing

    # the log the pictures write, line for line as the build writes it
    st = "boxes out; spent; deal door drawer drawer; envelopes open; counters black white white"
    assert props.section_dumps(found) == [
        "triosteps: strip of 2 frames",
        "triobox: [door] {shut} | shut, envelope door",
        "triobox: [door; open=door] {shut} | door, envelope door",
        "triobox: + step open=door -> [door] {door open}",
        "triobox: + ghost door shut in the side face, in outline; swing arc round the hinge",
        "triobox: + pin on the box (it does not stir)",
        "triosteps: strip of 2 frames",
        "triobox: [drawer] {shut} | shut, envelope drawer",
        "triobox: [drawer; open=drawer] {shut} | drawer, envelope drawer",
        "triobox: + step open=drawer -> [drawer] {drawer out}",
        "triobox: + ghost drawer shut, in outline; slide arc towards the company",
        "triobox: + pin on the box (it does not stir)",
        "triobox: [drawer] {drawer out} | drawer, envelope drawer",
        "hallsteps: strip of 2 frames",
        "halltable: {" + st + "} | trio boxes out, envelopes open, slip sealed, ways door drawer drawer",
        "halltable: [open=slip] {" + st + "} | trio boxes out, envelopes open, slip open reading odd, "
        "ways door drawer drawer",
        "halltable: + step open=slip -> {" + st + "; slip open}",
        "halltable: + ghost glass on the slip; slide arc to beside it"]


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
