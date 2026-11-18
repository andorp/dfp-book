"""Challenge 3: Three from One Bench
====================================
"Foretold without touching it: then it was there."  -- Achilles

Claim: the three-legged node is N = |000> + |111> (unnormalized, <N|N> = 2).
On any guaranteed deal two answers fix the third with certainty (the wrong-
parity amplitudes are exactly 0).  Diagonal boxes slide through the node from
leg to leg and two on a leg merge into one; a generic box does not.  A bright
door-effect on one leg leaves (1/sqrt(2)) cup; a dark one leaves the same with
a Z box on one leg.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
from fractions import Fraction

import numpy as np
from dfp import BELL, CUP, I2, Picture, ket, kron, matrix_of, random_matrix, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch07 as P

Z = np.array([[1, 0], [0, -1]], dtype=complex)
Sd = np.array([[1, 0], [0, -1j]], dtype=complex)   # S-dagger = diag(1, -i)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
N = ket(0, 0, 0) + ket(1, 1, 1)


def bra(bit):
    e = np.zeros((1, 2), dtype=complex)
    e[0, bit] = 1
    return e


def door_eff(a):     # the X question: <a|H
    return bra(a) @ H


def drawer_eff(a):   # the Y question: <a|H S-dagger  ( = <+/-i| )
    return bra(a) @ H @ Sd


def blackboard():
    rng = np.random.default_rng(73)

    # --- the node, and its (unnormalized) norm ------------------------------
    assert np.isclose(np.linalg.norm(N), np.sqrt(2))
    assert np.isclose(np.vdot(N, N), 2)
    assert np.allclose(BELL, CUP / np.sqrt(2))

    # --- guaranteed deals: two answers fix the third (wrong parity is 0) ------
    #   all doors  -> darks even ;  one door, two drawers -> darks odd
    deals = {
        ("door", "door", "door"): 0,       # even
        ("door", "drawer", "drawer"): 1,   # odd
        ("drawer", "door", "drawer"): 1,
        ("drawer", "drawer", "door"): 1,
    }
    eff = {"door": door_eff, "drawer": drawer_eff}
    for (sA, sB, sC), parity in deals.items():
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    amp = kron(eff[sA](a), eff[sB](b), eff[sC](c)) @ N
                    amp = complex(amp.ravel()[0])
                    if (a + b + c) % 2 != parity:
                        assert np.isclose(amp, 0)          # wrong parity: never
                    else:
                        assert not np.isclose(amp, 0)      # right parity: always

    # --- diagonal boxes slide through the node and merge ---------------------
    for _ in range(20):
        d = rng.normal(size=2) + 1j * rng.normal(size=2)
        e = rng.normal(size=2) + 1j * rng.normal(size=2)
        D, Dp = np.diag(d), np.diag(e)
        assert np.allclose(kron(D, I2, I2) @ N, kron(I2, D, I2) @ N)      # slide
        assert np.allclose(kron(D, I2, I2) @ N, kron(I2, I2, D) @ N)
        assert np.allclose(kron(D, Dp, I2) @ N, kron(I2, I2, D @ Dp) @ N)  # merge
    # a generic (non-diagonal) box does NOT slide
    for _ in range(20):
        B = random_matrix(rng)
        assert not np.allclose(kron(B, I2, I2) @ N, kron(I2, B, I2) @ N)
    # the case the theorem needs: two S-daggers merge into a Z, on any leg
    assert np.allclose(kron(I2, Sd, Sd) @ N, kron(I2, I2, Z) @ N)
    assert np.allclose(kron(I2, I2, Z) @ N, kron(Z, I2, I2) @ N)

    # --- closing one leg (whole and parts) ----------------------------------
    plus, minus = bra(0) @ H, bra(1) @ H
    assert np.allclose((kron(plus, I2, I2) @ N).ravel(), BELL)             # bright -> (1/sqrt2)cup
    assert np.allclose((kron(bra(0), I2, I2) @ N).ravel(), ket(0, 0))      # <0| -> |00>
    dark = (kron(minus, I2, I2) @ N).ravel()
    assert np.allclose(dark, (kron(Z, I2) @ CUP).ravel() / np.sqrt(2))     # dark -> Z on one leg
    assert not np.allclose(dark, BELL)

    # --- the node is not a product across any cut (rank test) ----------------
    assert np.linalg.matrix_rank(N.reshape(2, 4)) == 2      # A | BC
    assert np.linalg.matrix_rank(N.reshape(4, 2)) == 2      # AB | C
    middle = N.reshape(2, 2, 2).transpose(1, 0, 2).reshape(2, 4)
    assert np.linalg.matrix_rank(middle) == 2               # B | AC

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the trio lifted out onto the hall table
# (a step strip, \hallsteps[opts]{state}{steps}{captions}, each frame a
# \halltable), one box close up (\triobox[envelope]{way}{caption}), and the
# back of Achilles's card under a mask with the egg-cup (\notecard[under <mask>,
# cup on <square>; step]{state}{caption}, and a strip of it, \cardsteps).  The
# Lua refuses an egg-cup over counters that break the mask's word -- the
# Anteater sets them obeying it -- so the face it hides is fixed by the two it
# shows; and, round 4, it computes the state after each step (a box lifted from
# its nest, the cup lifted or stood, counters turned) with the step's marks.
# tools/props_ch07.py transcribes the Lua (proved against texlua; P.check_lua
# holds the Lua and the .tex to the text it was proved on); the calls are read
# out of this challenge's section of dialog.md and dialog.raw.md, never copied.
#
# The state model (identity card, section 3): on the back a column for each box
# as they stand on the cloth (left, middle, right = Achilles's, the Tortoise's,
# the Anteater's = legs A, B, C), the door row above the drawer row; a mask's
# windows are one deal (door = the X question, drawer = the Y question); white =
# bright = answer 0, black = dark = answer 1.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
COL = {"left": 0, "middle": 1, "right": 2}
BIT = {"white": 0, "black": 1}
EFF = {"door": door_eff, "drawer": drawer_eff}


def deal_of(m):
    """The way each box is asked under a back mask, legs A, B, C."""
    ways = [None] * 3
    for s in m.sq:
        col, way = s.split(" ")
        ways[COL[col]] = way
    return tuple(ways)


def amplitude(m, faces):
    """The node under the deal of mask m, answered with these faces (square -> face)."""
    ans = [None] * 3
    for s, f in faces.items():
        ans[COL[s.split(" ")[0]]] = BIT[f]
    e = kron(*(EFF[w](a) for w, a in zip(deal_of(m), ans)))
    return complex((e @ N).ravel()[0])


def turned(before, after):
    assert before.keys() == after.keys()
    return sorted(s for s in before if before[s] != after[s])


def check_transcription():
    """The egg-cup against the node's certainty (a structural check, not a
    second proof): for every mask, every window under the cup and every two
    faces shown, the macro accepts exactly one face under the cup -- and it is
    the one answer the node gives a non-zero amplitude, the wrong-parity
    amplitudes being exactly 0 (the blackboard above)."""
    P.check_lua()
    deals = {("door", "door", "door"): "even", ("door", "drawer", "drawer"): "odd",
             ("drawer", "door", "drawer"): "odd", ("drawer", "drawer", "door"): "odd"}
    assert {deal_of(m): m.rule for m in P.MASKS["back"]} == deals
    assert {" ".join(d): w for d, w in deals.items()} == P.DEALS
    for m in P.MASKS["back"]:
        for cup in m.sq:
            others = [s for s in m.sq if s != cup]
            for faces in itertools.product(P.FACES, repeat=2):
                cs = dict(zip(others, faces))
                keep = P.keeping_faces(m, cs, cup)
                assert len(keep) == 1                         # foretold: one face only
                live = tuple(f for f in P.FACES if not np.isclose(amplitude(m, {**cs, cup: f}), 0))
                assert live == keep                           # the node's one answer
                for f in P.FACES:                             # the macro takes that face alone
                    spec = "back; " + ", ".join(f"{s} {g}" for s, g in {**cs, cup: f}.items())
                    ok = P.faults(P.notecard, f"under {m.name}, cup on {cup}", spec) == []
                    assert ok == (f in keep)
    # the macro's own refusals: the egg-cup, a front mask on the back, the box
    three = "back; left door white, middle door white, right door white"
    assert P.faults(P.notecard, "under even, cup on left drawer", three) != []   # off the windows
    assert P.faults(P.notecard, "under even, cup on left door",
                    "back; left door white, middle door white") != []           # an empty window
    assert P.faults(P.notecard, "under even, cup on left door, cup on right door", three) != []
    assert P.faults(P.notecard, "card, cup on left door", three) != []           # no mask laid
    assert P.faults(P.notecard, "under two lids", three) != []                   # a front mask
    assert P.faults(P.notecard, "under unlike", three) != []
    assert P.faults(P.notecard, "under even",                                     # no cup, no check
                    "back; left door black, middle door white, right door white") == []
    assert P.faults(P.triobox, "", "door ajar") != []
    assert P.faults(P.triobox, "lid", "door open") != []
    assert P.faults(P.triobox, "door", "drawer out") != []                      # it disagrees
    assert P.faults(P.triobox, "drawer", "shut") == [] and P.faults(P.triobox, "door", "shut") == []
    # round 4's refusals on this section's steps: the cup, the counters, the boxes
    assert P.faults(P.notecard, "under even; lift=cup", three) != []             # no cup to lift
    assert P.faults(P.notecard, "under even, cup on right door; lift=mask", three) != []   # cup first
    assert P.faults(P.notecard, "under even; cup=left drawer", three) != []      # off the windows
    assert P.faults(P.notecard, "under even, cup on left door; cup=left door", three) != []   # there already
    assert P.faults(P.notecard, "card; cup=left door", three) != []              # no mask laid
    assert P.faults(P.notecard, "under even, cup on left door; turn=left door", three) != []   # hidden
    assert P.faults(P.notecard, "under even; turn=left door and left door", three) != []       # twice
    assert P.faults(P.halltable, "lift=next box", "boxes out") != []             # none left in the case
    assert P.faults(P.halltable, "lift=boxes", "case shut") != []                # the lid first
    assert P.faults(P.halltable, "lift=lid", "case open") != []


def pictures():
    check_transcription()
    found = P.check_section(DIALOGUE, "Third", sizes=[2, 1, 1, 1], frames=[4, 1, 1, 2])
    (p31, [(t14, cap14), (t15, cap15), (t16, cap16)]), (p32, [(x32, cap32)]) = found[0]
    (p33, [(c33, cap33)]), = found[1]
    (p33l, [(c19, cap19)]), = found[2]
    (p34, [(c20, cap20), (c34, cap34)]), = found[3]
    assert [c.macro for c in (p31, p32, p33, p33l, p34)] == \
        ["hallsteps", "triobox", "notecard", "notecard", "cardsteps"]
    assert (cap14, cap15, cap16, cap32, cap33, cap19, cap20, cap34) == \
        ("lifted out", "then the next", "out on the cloth", "a door and a drawer", "one hidden",
         "the cup lifted", "set back black", "a different window")
    sec = DIALOGUE.section("Third")

    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a1, a2, a3, a4 = DIALOGUE.anchors("Third", path)
        assert a1.startswith("**Tortoise:** The wedge is a pair. This is a trio.")
        assert "stands them on the cloth, shut" in a1
        assert "two ways in only: a little door in the side, and a drawer underneath" in a1
        assert a2.startswith("*[Achilles turns to the window. The Tortoise lays a single trio-mask over the empty card")
        assert a2.endswith("What is fixed and hidden now is the one note under the cup.]*")
        assert a3 == "*[The Anteater lifts the egg-cup. The counter beneath shows white.]*"
        assert a4.startswith("*[Achilles turns away. The same EVEN mask stays where it is, and the middle counter with it, white.")
        assert a4.endswith("then stands the egg-cup over the left window — not the right one, this time.]*")
        # block 3.3 stands before the Anteater's next speech
        (after,) = [a for b, a in zip(DIALOGUE.blocks("Third", path), DIALOGUE.following("Third", path))
                    if "lift" in P.options(b[0])]
        assert after == "**Anteater:** Bright. Again — turn round."
    # the direction and the caption edited this round (D4, C1), in both files
    P.check_edits(DIALOGUE, "Third", [
        ("one, and one, and one, and stands them on the cloth, shut.]*",
         "She lifts the three boxes from their nests, one, and one, and one, from the left, and "
         "stands them on the cloth, shut, in a row between the case and the envelopes.]*"),
        ("}{one under the cup}", "}{one hidden}")])

    # Block 3.1: the boxes lifted from their nests one at a time, from the left (D4), and
    # stood shut on the cloth.  Its before-frame is block 2.4's F13: the case open, the
    # three boxes in their nests (nothing on the table has moved since)
    (p24,) = DIALOGUE.blocks("Second")[3]
    t13 = P.pictures(p24)[-1][0]
    assert P.options(p31)["from"] == "off" and t13.trio == "case open"
    assert P.hall_spec(P.hall_parse(p31.args[0])) == t13.spec
    assert [(s.opts, s.state) for s in P.frames_of(p31)] == [
        ("lift=next box", "case open"),
        ("lift=next box", "one box out; envelopes sealed; slip sealed"),
        ("lift=next box", "two boxes out; envelopes sealed; slip sealed")]
    lifts = (t14, t15, t16)
    assert [t.trio for t in lifts] == ["one box out", "two boxes out", "boxes out"]
    assert [t.nests for t in lifts] == [(None, "box", "box"), (None, None, "box"), (None, None, None)]
    assert [t.cloth for t in lifts] == [("shut",), ("shut",) * 2, ("shut",) * 3]
    assert not p31.keyed and not any(t.keyed or t.phantom or t.labels for t in lifts)
    box_top = P.ROWBOX[0] + 0.78 * P.ROWBOX[1] + 0.22 * P.ROWBOX[1]
    env_top = P.ENVY + P.ENVSC * P.ENVH / 2
    for k, t in enumerate(lifts):
        assert t.step == ("lift", "next box") and t.pins == []
        assert [(g.what, g.name) for g in t.ghosts] == [("box", P.WHO[k])]   # from the left
        assert t.ghosts[0].at == (P.NESTX[k], P.CY0)                          # in its nest
        (arc,) = t.arcs
        assert arc.kind == "swing" and arc.tag == "box " + P.WHO[k]
        assert np.allclose(arc.start, (P.NESTX[k], P.CY0 + 0.15))
        assert np.isclose(arc.end[0], P.ROWX[k] - 0.064)                      # to its place on the cloth
        assert box_top < arc.end[1] < P.CASEFOOT                              # just over the box stood
        assert t.case_lid == "up" and t.envelopes == "sealed" and t.envelope_faces == P.WHO
        assert t.slip == "sealed" and t.glass == P.GLASS0 and t.deal == ()
    assert list(P.NESTX) == sorted(P.NESTX) and list(P.ROWX) == sorted(P.ROWX)   # left to right
    # "in a row between the case and the envelopes": the boxes on the cloth stand below the
    # case's foot and above the envelopes' top, side by side
    assert env_top < P.ROWBOX[0] and box_top < P.CASEFOOT
    # P3.1's last frame: the boxes out on the cloth, shut; the case open and empty
    assert t16.trio == "boxes out" and t16.cloth == ("shut",) * 3 and t16.nests == (None,) * 3
    # P3.2: one of them close up, keyed door and drawer -- the same box, shut
    assert p32.keyed and x32.labels == ("door", "drawer")
    assert x32.way == "shut" and x32.envelope == "none" and x32.way in t16.cloth
    assert not x32.marked and x32.step is None

    # P3.3: the back, keyed; the EVEN mask on the empty card, three whites in the
    # door row, the egg-cup over the third window (the right door)
    assert p33.keyed and c33.side == "back" and c33.view == "under"
    assert c33.labels == ("door", "drawer", "left", "middle", "right")
    assert "one per box in the order they stand on the cloth, and two rows, the door above the drawer" in sec
    assert P.place("back", "left door") == (0, 0) and P.place("back", "right drawer") == (1, 2)
    assert c33.mask.name == "even" and deal_of(c33.mask) == ("door", "door", "door")
    assert c33.counters == {"left door": "white", "middle door": "white", "right door": "white"}
    assert c33.cup == "right door" and c33.mask.sq.index(c33.cup) == 2       # "the third window"
    assert c33.shown == {"left door": "white", "middle door": "white"}      # two brights showing
    assert P.keeping_faces(c33.mask, c33.shown, c33.cup) == ("white",) == (c33.counters[c33.cup],)
    assert "So: bright." in sec and "The counter beneath shows white." in sec
    assert not c33.marked and c33.step is None

    # Block 3.3: the egg-cup lifted (F19); its before-frame is P3.3, the paragraph before
    # last: the same mask, the same counters, the same cup
    assert P.options(p33l)["bare"] == P.options(p33)["bare"] and p33l.args == p33.args
    assert not p33l.keyed and c19.step == ("lift", "cup")
    assert c19.mask == c33.mask and c19.counters == c33.counters and c19.cup is None
    assert c19.shown == c33.counters                                         # the third shows: white
    assert c19.shown["right door"] == "white"
    # its mark: the cup's ghost in outline where it stood, and no arc (the text names no
    # place it goes: it is in the Anteater's hand)
    assert [(g.what, g.name, g.outline) for g in c19.ghosts] == [("cup", "right door", True)]
    assert c19.arcs == [] and c19.pins == []

    # Block 3.4: the left and right counters set back black in their own windows, then the
    # cup stood over the left window.  Its before-frame is F19: the cup off, three white
    assert P.options(p34)["from"] == "off" and P.options(p34)["view"] == "under even"
    side, cs = P.parse_card(p34.args[0])
    assert side == "back" and cs == c19.counters
    assert [(s.opts, s.state) for s in P.frames_of(p34)] == [
        ("under even; turn=left door and right door", p34.args[0]),
        ("under even; cup=left door", "back; left door black, middle door white, right door black")]
    # F20: two counters turned in place, the middle one with them untouched; no cup
    assert not p34.keyed and c20.step == ("turn", "left door and right door")
    assert turned(c19.counters, c20.counters) == ["left door", "right door"]
    assert c20.counters["middle door"] == "white" and c20.cup is None and c20.ghosts == []
    assert [a.tag for a in c20.arcs] == ["flip left door", "flip right door"] and c20.pins == []
    for a, sq in zip(c20.arcs, ("left door", "right door")):
        x, y = P.sqpos("back", sq)
        assert abs(a.start[0] - x) < 0.13 and abs(a.start[1] - y) < 0.15     # over its own window
    # F21: the same mask; the cup over the left window -- from the hand, no mark
    assert c34.step == ("cup", "left door") and not c34.marked
    assert c34.mask == c33.mask and c34.side == "back"
    assert c34.counters == c20.counters
    assert turned(c33.counters, c34.counters) == ["left door", "right door"]
    assert c34.counters["middle door"] == "white"
    assert c34.cup == "left door" and c34.cup != c33.cup
    assert c34.shown == {"middle door": "white", "right door": "black"}     # one bright, one dark
    assert "One bright, one dark shown." in sec
    assert P.keeping_faces(c34.mask, c34.shown, c34.cup) == ("black",) == (c34.counters[c34.cup],)
    assert "Dark." in sec and "the counter is black" in sec
    # the node agrees: under all doors, with those two shown, only a dark has amplitude
    assert not np.isclose(amplitude(c34.mask, c34.counters), 0)
    assert np.isclose(amplitude(c34.mask, {**c34.counters, "left door": "white"}), 0)

    # the log the pictures write, line for line as the build writes it
    three = "back; left door white, middle door white, right door white"
    set_back = "back; left door black, middle door white, right door black"
    tail = "; envelopes sealed; slip sealed"
    assert P.section_dumps(found) == [
        "hallsteps: strip of 3 frames",
        "halltable: [lift=next box] {case open} | trio one box out, envelopes sealed, slip sealed",
        "halltable: + step lift=next box -> {one box out" + tail + "}",
        "halltable: + ghost box achilles in its nest; arc to the cloth",
        "halltable: [lift=next box] {one box out" + tail + "} | trio two boxes out, envelopes sealed, slip sealed",
        "halltable: + step lift=next box -> {two boxes out" + tail + "}",
        "halltable: + ghost box tortoise in its nest; arc to the cloth",
        "halltable: [lift=next box] {two boxes out" + tail + "} | trio boxes out, envelopes sealed, slip sealed",
        "halltable: + step lift=next box -> {boxes out" + tail + "}",
        "halltable: + ghost box anteater in its nest; arc to the cloth",
        "triobox: [none] {shut} | shut, envelope none",
        "notecard: [under even, cup on right door] {" + three + "} | " + three + " | under even, cup right door",
        "notecard: [under even, cup on right door; lift=cup] {" + three + "} | " + three + " | under even, cup none",
        "notecard: + step lift=cup -> [under even] {" + three + "}",
        "notecard: + ghost cup on right door; shown right door white",
        "cardsteps: strip of 2 frames",
        "notecard: [under even; turn=left door and right door] {" + three + "} | " + set_back
        + " | under even, cup none",
        "notecard: + step turn=left door and right door -> [under even] {" + set_back + "}",
        "notecard: + turned left door white to black",
        "notecard: + turned right door white to black",
        "notecard: [under even; cup=left door] {" + set_back + "} | " + set_back + " | under even, cup left door",
        "notecard: + step cup=left door -> [under even, cup on left door] {" + set_back + "}",
        "notecard: + cup stood on left door (from the hand: no marks)"]


# --------------------------------------------------------------------------
# The sketchbook.  The node is one unlabelled dot where three wires meet --
# nothing else.  A diagonal box on a leg is a phase node (diag(1, e^{i pi t}),
# no scalar), so sliding and merging it is exact.  A one-legged effect costs a
# sqrt(2), paid back with add_power(-1).
# --------------------------------------------------------------------------
def node(leg_chains, outs=(0, 1, 2)):
    """The three-legged node, with a chain of nodes on each leg then an output.
    leg_chains[r] is a list of (kind, phase) applied along leg r before out(r)."""
    p = Picture()
    dot = p.z(1, 1)   # the node: PyZX draws this with the spiders you meet in Chapter 8.
    for r in range(3):
        col = 3
        chain = []
        for kind, ph in leg_chains[r]:
            chain.append(p.z(r, col, ph) if kind == "z" else p.x(r, col, ph))
            col += 1
        p.wire(dot, *chain, p.out(r))
    return p.done()


def bare_node():
    return node([[], [], []])


def node_effect_on_leg0(eff_phase, box_phase=None):
    """The node with a door-effect (Z-effect phase eff_phase) on leg 0, and an
    optional Z box (phase box_phase) between the dot and the effect."""
    p = Picture()
    dot = p.z(1, 1)   # the node: PyZX draws this with the spiders you meet in Chapter 8.
    chain = []
    if box_phase is not None:
        chain.append(p.z(0, 3, box_phase))
    e = p.z(0, 5, eff_phase)                 # door-effect <a|H is a Z-effect phase a; add_power(-1) below
    p.wire(dot, *chain, e)
    p.wire(dot, p.out(1))
    p.wire(dot, p.out(2))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def cup_with_box(box_phase=None):
    """(1/sqrt(2)) cup, with an optional Z box on the top leg."""
    p = Picture()
    bend = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    if box_phase is None:
        p.wire(bend, p.out(0))
    else:
        p.wire(bend, p.z(0, 3, box_phase), p.out(0))
    p.wire(bend, p.out(1))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def lone_door_effect(ans):
    """A single door-effect on an input wire (before paying its sqrt(2))."""
    p = Picture()
    e = p.z(0, 1, ans)   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(p.inp(0), e)
    return p.done()


def lone_z_box():
    """A single Z box (phase 1) on a wire."""
    p = Picture()
    b = p.z(0, 1, Fraction(1))   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(p.inp(0), b, p.out(0))
    return p.done()


def sketchbook():
    # the pieces, literally, against NumPy: a door-effect node is sqrt(2)<a|H,
    # and a phase-1 node is the Z box (both exact -- no scalar on Z).
    for ans in (0, 1):
        assert np.allclose(matrix_of(lone_door_effect(ans)), np.sqrt(2) * (bra(ans) @ H))
    assert np.allclose(matrix_of(lone_z_box()), Z)

    # the node is exactly N -- one dot, no scalar
    assert np.allclose(matrix_of(bare_node()).ravel(), N)

    # a diagonal box slides from leg to leg, and two on a leg merge into one
    phi, psi = Fraction(1, 3), Fraction(2, 5)
    on_A = node([[("z", phi)], [], []])
    on_B = node([[], [("z", phi)], []])
    assert same_picture(on_A, on_B)                                  # slide
    two = node([[("z", phi)], [("z", psi)], []])
    merged = node([[], [], [("z", phi + psi)]])
    assert same_picture(two, merged)                                # merge

    # a generic (non-diagonal) box does not slide
    gen = [("z", Fraction(1, 4)), ("x", Fraction(1, 3)), ("z", Fraction(3, 5))]
    assert not same_picture(node([gen, [], []]), node([[], gen, []]))

    # a bright door-effect on one leg leaves (1/sqrt(2)) cup
    assert same_picture(node_effect_on_leg0(0), cup_with_box())      # <+| : yes
    assert np.allclose(matrix_of(cup_with_box()).ravel(), BELL)
    # a dark one leaves the same with a Z box on one leg
    assert same_picture(node_effect_on_leg0(1), cup_with_box(Fraction(1)))
    assert not same_picture(node_effect_on_leg0(1), cup_with_box())

    # the 1/sqrt(2) matters: the node with <+| is not the bare (unnormalized) cup
    bare_cup = Picture()
    bc = bare_cup.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    bare_cup.wire(bc, bare_cup.out(0))
    bare_cup.wire(bc, bare_cup.out(1))
    assert not same_picture(node_effect_on_leg0(0), bare_cup.done())


if __name__ == "__main__":
    verdict("Trio", blackboard, sketchbook)
