"""Challenge 6: The Relay
=======================
"They never shared a bench."  -- Anteater

Claim: on six wires, three cups (wires 1-2, 3-4, 5-6) followed by two caps
(wires 2-3, 4-5) are exactly one cup on wires 1 and 6:
(I (x) cap (x) cap (x) I)(cup (x) cup (x) cup) = cup.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re

import numpy as np
from dfp import BELL, CAP, CUP, I2, Picture, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch06 as P


def blackboard():
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))      # unnormalized cup
    assert np.allclose(CAP, CUP.conj().reshape(1, 4))       # ... and cap
    assert np.allclose(BELL, CUP / np.sqrt(2))

    # one junction: two pairs, one cap across the middle
    one_junction = kron(I2, CAP, I2) @ kron(CUP, CUP)
    assert np.allclose(one_junction.ravel(), CUP)

    # the chain: six wires in, wires 1 and 6 out
    three_cups = kron(CUP, CUP, CUP)
    assert three_cups.shape == (2 ** 6, 1)                  # six wires
    chain = kron(I2, CAP, CAP, I2) @ three_cups
    assert np.allclose(chain.ravel(), CUP)                  # exact: no stray number

    # what the pull costs, with normalized pairs and the normalized outcome
    bell_bra = BELL.conj().reshape(1, 4)
    relay = (kron(I2, bell_bra, bell_bra, I2) @ kron(BELL, BELL, BELL)).ravel()
    assert np.allclose(relay, BELL / 4)                     # 1/2 per junction
    assert not np.allclose(relay, BELL)
    assert np.isclose(np.linalg.norm(relay) ** 2, 1 / 16)   # = 1/4 * 1/4

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the six boxes in a row with the string
# laid before them, and tonight's page of the ledger, with
#   \musicboxes[opts]{boxes}{caption}   \crabledger[opts]{lines}{caption}
# and their step strips \boxsteps and \ledgersteps (pdf/props/ch06.tex).  The
# Lua COMPUTES the string laid pair by pair (lay=next pair), the pulled string
# -- bends of opposite kinds cancel, far bends minus near turns, one wide bend
# from the first leg to the last (pull=ends) -- Achilles's rule for it (expects=
# gaps: a separate bend at each pair, cut at each station), the ledger's
# right-hand column, and every step's marks.  tools/props_ch06.py transcribes
# it; the calls are read out of dialog.md, never copied.  This file also checks
# the dialogue's pictures as a whole, and that the transcription refuses every
# input the Lua refuses.
#
# The state model (identity card, sections 3 and 3e): in the row the boxes stand
# in wire order, left pearwood = 1 ... right ebony = 6; the string's leg k runs
# from the near side up to box k; a far bend at each pair (at the boxes), a near
# turn between pairs (at the stations), the free ends at the near edge, under
# boxes 1 and 6.
# --------------------------------------------------------------------------
MACROS = ("musicboxes", "walnutstring", "beadstring", "crabledger",
          "boxsteps", "walnutsteps", "beadsteps", "ledgersteps")
DIALOGUE = Dialogue(__file__, macros=MACROS)
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")
DRAWS = {"First": {"musicboxes", "crabledger"}, "Second": {"crabledger"},
         "Third": {"walnutstring"}, "Fourth": {"beadstring"}, "Fifth": {"walnutstring"},
         "Sixth": {"musicboxes", "crabledger"}}


def string_map(s, early):
    """A laid string as a map, read with time running away from the `early`
    side of the table: a bend on the early side is a cup, on the late side a
    cap; a free end on the early side is an input, on the late side an output;
    the legs, left to right, are the wires, top first.  In the row the boxes
    are at the far side and the free ends at the near edge: early = far, so a
    far bend at a pair is a cup and a near turn at a station is a cap."""
    late = {"far": "near", "near": "far"}[early]
    order = s.names

    def layer(side, joined):
        out, i = [], 0
        while i < len(order):
            bend = [b for b in s.bends if b.side == side and order[i] in b.legs]
            if bend:
                assert bend[0].legs == (order[i], order[i + 1])
                out.append(joined)
                i += 2
            else:
                out.append(I2)
                i += 1
        return kron(*out)

    return layer(late, CAP) @ layer(early, CUP)


def check_transcription():
    """The Lua's pull rule against the blackboard (a structural check, not a
    second proof): three far bends and two near turns pull to 3 - 2 = 1 far
    bend from leg 1 to leg 6, and the map is the Chain's, before and after."""
    P.check_sites()
    P.check_lua()
    laid = P.row_string("three bends")
    assert P.continuous(laid) and laid.names == (1, 2, 3, 4, 5, 6)
    assert [(b.side, b.legs) for b in laid.bends] == \
        [("far", (1, 2)), ("near", (2, 3)), ("far", (3, 4)), ("near", (4, 5)), ("far", (5, 6))]
    assert laid.ends == ("near", "near")
    assert np.allclose(string_map(laid, "far"), kron(I2, CAP, CAP, I2) @ kron(CUP, CUP, CUP))
    assert P.faults(P.row_string, "three bends pulled") == []
    one = P.row_string("three bends pulled")
    assert P.continuous(one) and [(b.side, b.legs) for b in one.bends] == [("far", (1, 6))]
    assert one.ends == laid.ends
    assert np.allclose(string_map(one, "far").ravel(), CUP)
    assert np.allclose(string_map(laid, "far"), string_map(one, "far"))
    # the partial strings, laid pair by pair: k bends join box 1 to box 2k, the rest slack
    for k, word in ((1, "one bend"), (2, "two bends")):
        part = P.row_string(word)
        assert part.ends == ("near", "slack") and part.names == tuple(range(1, 2 * k + 1))
        assert np.allclose(string_map(part, "far").ravel(), CUP)
    assert P.faults(P.row_string, "two bends pulled") == [
        "musicboxes: the string is pulled only when all three bends are laid (three bends pulled)"]
    assert P.faults(P.row_string, "four bends") == [
        "musicboxes: the string is none, bare, one bend, two bends or three bends, optionally "
        "pulled (got 'four bends')"]
    # the gaps rule, as the Lua's comment states it: three pieces, each one far bend at
    # its pair, cut at the near side; the pull rule on each piece leaves its one bend
    _, pieces = P.gap_strands()
    assert [[(b.side, b.legs) for b in p.bends] for p in pieces] == \
        [[("far", (1, 2))], [("far", (3, 4))], [("far", (5, 6))]]
    for p in pieces:
        assert P.continuous(p) and P.pulled(p, 1).bends == p.bends
    # the steps on the row: the next pair laid only on a string bare, one bend or two;
    # the ends pulled only from three bends laid; a prediction refuses any other string
    row = "row; left pearwood spent"
    assert P.faults(P.musicboxes, "three bends; lay=next pair", row) == [
        "musicboxes: lay=next pair: the string is bare, one bend or two bends before it (not "
        "'three bends')"]
    assert P.faults(P.musicboxes, "bare; lay=all", row) == [
        "musicboxes: lay=next pair (not 'all')"]
    assert P.faults(P.musicboxes, "lay=next pair", "tray") == [
        "musicboxes: lay= lays the string before the row: the row"]
    assert P.faults(P.musicboxes, "two bends; pull=ends", row) == [
        "musicboxes: pull= pulls the string laid before the row with three bends"]
    assert P.faults(P.musicboxes, "three bends; pull=hard", row) == [
        "musicboxes: pull=ends or pull=ends, walnut bend held (not 'hard')"]
    assert P.faults(P.musicboxes, "two bends; expects=gaps", row) == [
        "musicboxes: expects=gaps reads the string laid before the row with three bends"]
    assert P.faults(P.musicboxes, "three bends; expects=joins", row) == [
        "musicboxes: unknown rule 'joins' (gaps)"]
    assert P.faults(P.musicboxes, "three bends; expects=gaps", row, True) == [
        "musicboxes: an expects picture is not keyed"]
    assert P.faults(P.musicboxes, "three bends; pull=ends; expects=gaps", row) == [
        "musicboxes: an expects picture carries no step"]
    assert P.faults(P.musicboxes, "bare; one bend", row) == [
        "musicboxes: one bracket word only ('bare', 'one bend')"]
    assert P.faults(P.musicboxes, "pull=ends; pull=ends", row) == [
        "musicboxes: option 'pull' given twice",
        "musicboxes: pull= pulls the string laid before the row with three bends"]


def check_every_site():
    """Every tex.error site of the Lua has an input the transcription refuses with that
    site's words: the aimed calls below reach them all.  One site cannot be reached, in
    the Lua or here: "slide=back: the bead is on the far leg already" stands after
    "the bead is there already", which already refuses back from the far leg (BPOS
    gives far leg and back the same place).  The four "unknown step" sites and the two
    "internal" guards are unreachable through a call (the steps are read only under the
    keys allowed, and the Lua's own pull arithmetic is fixed): the transcription's step
    functions and pull rule are called on them directly."""
    POSE = "point to the bend, notch to the window"
    st = P.parse_boxes("row")[1]
    aimed = [
        (P.musicboxes, ("shut=left pearwood", "tray")), (P.musicboxes, ("open=left pearwood; open=left pearwood", "tray")),
        (P.musicboxes, ("bare; one bend", "row")), (P.musicboxes, ("open=left pearwood; marks=dim", "tray")),
        (P.musicboxes, ("open=left pearwood; close=left pearwood", "tray")), (P.musicboxes, ("", "")),
        (P.musicboxes, ("", "table")), (P.musicboxes, ("", "tray; middle pearwood open")),
        (P.musicboxes, ("", "row; right pearwood kept")), (P.musicboxes, ("", "tray; left pearwood broken")),
        (P.musicboxes, ("", "tray; left pearwood open; left pearwood spent")),
        (P.musicboxes, ("", "tray; left pearwood spent; left pearwood kept")),
        (P.musicboxes, ("", "tray; left pearwood kept; right pearwood kept")),
        (P.musicboxes, ("four bends", "row")), (P.musicboxes, ("two bends pulled", "row")),
        (P.pulled, (P.walnut_strand("S")[0], 1)),
        (P.musicboxes, ("open=middle pearwood", "tray")), (P.musicboxes, ("open=left pearwood", "tray; left pearwood spent")),
        (P.musicboxes, ("close=left pearwood", "tray")), (P.musicboxes, ("keep=left pearwood", "row")),
        (P.musicboxes, ("keep=left pearwood", "tray; left pearwood spent")),
        (P.musicboxes, ("keep=left pearwood", "tray; right walnut kept")),
        (P.musicboxes, ("bare; lay=all", "row")), (P.musicboxes, ("lay=next pair", "tray")),
        (P.musicboxes, ("three bends; lay=next pair", "row")), (P.musicboxes, ("three bends; pull=hard", "row")),
        (P.musicboxes, ("two bends; pull=ends", "row")), (P.box_step, ("row", st, "bare", "turn", "x", "musicboxes")),
        (P.musicboxes, ("two bends; expects=gaps", "row")), (P.musicboxes, ("marks=off", "tray")),
        (P.musicboxes, ("three bends; expects=joins", "row")), (P.musicboxes, ("three bends; expects=gaps", "row", True)),
        (P.musicboxes, ("three bends; pull=ends; expects=gaps", "row")), (P.musicboxes, ("bare", "tray")),
        (P.walnutstring, ("", "out back out, the other way about")), (P.walnutstring, ("", "hairpin, knotted")),
        (P.walnutstring, ("", "key")), (P.walnutstring, ("", "out back out, key")),
        (P.pulled, (P.walnut_strand("hairpin")[0], 0)),
        (P.walnutstring, ("pulling; bend=hairpin", "out back out")), (P.walnutstring, ("bend=hairpin", "hairpin")),
        (P.walnutstring, ("bend=straight", "straight")), (P.walnutstring, ("bend=knot", "straight")),
        (P.walnutstring, ("push=sideways", "hairpin")), (P.walnutstring, ("push=out", "straight")),
        (P.walnutstring, ("push=out", "hairpin, key")), (P.walnutstring, ("push=close", "hairpin close")),
        (P.walnutstring, ("pull=half", "hairpin")), (P.walnutstring, ("pulling; pull=half", "out back out")),
        (P.walnutstring, ("pulled; pull=ends", "out back out")), (P.walnutstring, ("pull=all", "out back out")),
        (P.walnut_step, ("laid", "S", False, "twist", "x", "walnutstring")),
        (P.walnutstring, ("keyroom=off", "hairpin")), (P.walnutstring, ("sideways", "hairpin")), (P.walnutstring, ("pulled", "hairpin")),
        (P.walnutstring, ("pulling", "hairpin")), (P.walnutstring, ("marks=off", "hairpin")),
        (P.beadstring, ("", "point up, notch to the window")), (P.beadstring, ("", "point to the bend")),
        (P.bead_step, ("far leg", "push", "x", "beadstring")), (P.beadstring, ("napkin; slide=near leg", POSE)),
        (P.beadstring, ("slide=napkin", POSE)), (P.beadstring, ("near leg; slide=near leg", POSE)),
        (P.beadstring, ("on the box", POSE)), (P.beadstring, ("near leg; marks=off", POSE)),
        (P.beadstring, ("near leg; expects=turned", POSE)), (P.beadstring, ("far leg; expects=unchanged", POSE)),
        (P.beadstring, ("near leg; expects=unchanged", POSE, True)),
        (P.beadstring, ("far leg; slide=near leg; expects=mirrored", POSE)),
        (P.crabledger, ("", "high, grey")), (P.crabledger, ("", "low completed")),
        (P.crabledger, ("", ", ".join(["high"] * 9))), (P.crabledger, ("", "tonight high, low")),
        (P.crabledger, ("blotter left; slide=across", "high")), (P.crabledger, ("slide=blotter left", "high")),
        (P.crabledger, ("blotter left; lift=up", "high")), (P.crabledger, ("lift=off", "high")),
        (P.crabledger, ("lay=blotter", "high")), (P.crabledger, ("blotter left; lay=blotter left", "high")),
        (P.crabledger, ("write=left", "high")), (P.crabledger, ("write=right", "high")),
        (P.ledger_step, ("open", [], "scrub", "x", "crabledger")), (P.crabledger, ("blotter top", "high")),
        (P.crabledger, ("marks=off", "high")),
        (P.strip, ("beadsteps", False, POSE, "bare", "slide=in the bend", "a; b")),
        (P.strip, ("beadsteps", False, POSE, "", "slide=far leg; lift=off", "a; b")),
        (P.strip, ("ledgersteps", False, "high", "", "", "a")),
        (P.strip, ("ledgersteps", False, "high", "", "lay=blotter left; slide=blotter right; lift=off; "
                   "lay=blotter right", "a; b; c; d; e")),
        (P.strip, ("ledgersteps", False, "high", "", "lay=blotter left", "a")),
    ]
    hit = set()
    for fn, args in aimed:
        errors = P.faults(fn, *args)
        assert errors, (fn.__name__, args)
        for e in errors:
            assert P.sites_of(e), e                         # every refusal is one of the Lua's
            hit.update(P.sites_of(e))
    dead = P.TEX_SITES.index("*: slide=back: the bead is on the far leg already")
    assert hit == set(range(len(P.TEX_SITES))) - {dead}, sorted(set(range(len(P.TEX_SITES))) - hit)


FORBIDDEN = {"walnutstring": ("nudged", "hairpin out", "hairpin close", "hairpin aside")}


def check_whole_dialogue():
    """The dialogue's pictures as a whole (card 3e.2's counts): 18 blocks, 22 calls, 32
    frames, 7 strips, 17 step frames (all marked but the word written), 2 predictions,
    the starred form at round 3's five places; the log the build writes, 158 lines."""
    total, frames, starred, log = P.check_dialogue(DIALOGUE, ORDINALS)
    assert total == 22                                        # 3 + 3 + 4 + 3 + 2 + 7 calls
    assert [len(DIALOGUE.blocks(o)) for o in ORDINALS] == [3, 3, 4, 3, 2, 3]   # moments, <= 4
    assert len(frames) == 32
    calls = [c for o in ORDINALS for b in DIALOGUE.blocks(o) for c in b]
    assert [c.macro for c in calls if P.is_strip(c)] == \
        ["boxsteps", "walnutsteps", "walnutsteps", "beadsteps", "walnutsteps", "boxsteps", "ledgersteps"]
    stepped = [(o, cap) for o, c, p, cap in frames if p.step is not None]
    assert len(stepped) == 17
    assert [cap for o, c, p, cap in frames if p.step is not None and not p.marked] == \
        ["the same word twice"]                               # a word written moves nothing
    assert [(o, p.expects) for o, c, p, cap in frames if p.expects] == \
        [("Fourth", "unchanged"), ("Sixth", "gaps")]
    assert starred == [("First", 0), ("First", 2), ("Third", 0), ("Fourth", 0), ("Sixth", 0)]
    # the phantom labels: exactly the frames after the first of a starred strip -- each with
    # its strip's keyed labels, so that the frames of a strip stand level
    strips = [c for c in calls if P.is_strip(c)]
    phantoms = [(o, cap) for o, c, p, cap in frames if p.phantom]
    assert phantoms == [(o, cap) for o, c, p, cap in frames
                        if P.is_strip(c) and c.keyed and not p.keyed]
    assert [cap for _, cap in phantoms] == ["the lid up", "spent", "a pair", "first pair",
                                            "next pair", "last pair"]
    for o, c, p, cap in frames:
        key = [q for oo, cc, q, _ in frames if cc is c and q.keyed]
        assert bool(p.labels) == (p.keyed or p.phantom)
        if p.phantom:
            assert p.labels == key[0].labels and "keyroom=on" in P.split(
                [s.opts for s in P.frames_of(c) if s.caption == cap][0], ";")
    assert len(strips) == 7
    assert len(log) == 158 and sum(":" in l and ": +" not in l and "strip of" not in l
                                   for l in log) == 32
    assert sum("strip of" in l for l in log) == 7
    for o in ORDINALS:
        assert {P.BASE[c.macro] for c in calls if c in [cc for b in DIALOGUE.blocks(o) for cc in b]} \
            == DRAWS[o], o
        for b in DIALOGUE.blocks(o):                              # the walnut pictures stand alone
            bases = {P.BASE[c.macro] for c in b}
            assert len(bases) == 1 or "walnutstring" not in bases
    for o, c, pic, caption in frames:
        assert "chain" not in caption.lower()                    # never the master's name
        if P.BASE[c.macro] == "walnutstring":
            assert not pic.nudged and pic.shape not in ("hairpin out", "hairpin close",
                                                         "hairpin aside")
        if P.BASE[c.macro] == "beadstring":
            assert pic.place not in ("back", "napkin")
        if P.BASE[c.macro] == "musicboxes":
            assert (pic.layout == "row") == (o == "Sixth")         # the tray before, the row after
            assert all(b.state != "kept" for b in pic.boxes.values()) or o == "First"
    # no picture in the relay's opening (the carriers), and none after the last note
    six = DIALOGUE.section("Sixth")
    first_block = six.index("\\begin{center}")
    assert six.index("Even so — the boxes in between?") < first_block
    assert six.index("Let Bob draw on to the second station; the last pair.") < first_block
    last = DIALOGUE.blocks("Sixth")[-1][-1]
    assert not any("\\" + m in six.split(last.text)[1] for m in MACROS)
    # D9: the rule after the planning comments is gone (outside every section, so read the
    # files themselves, comments and all)
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        text = path.read_text(encoding="utf-8")
        assert text.count("-->\n\n*[Achilles and the Tortoise have come") == 1, path.name
        assert "-->\n\n---\n\n*[Achilles and the Tortoise have come" not in text, path.name


def pictures():
    check_transcription()
    check_every_site()
    check_whole_dialogue()
    found = P.check_section(DIALOGUE, "Sixth", sizes=[1, 2, 2], frames=[4, 2, 3])
    (p61, laying), = found[0]
    (p62e, [(r62e, cap62e)]), (p62, [(r62, cap62)]) = found[1]
    (p63, [(r63, cap63)]), (p64, [(t64a, cap64a), (t64, cap64)]) = found[2]
    assert [c.macro for c in (p61, p62e, p62, p63, p64)] == \
        ["boxsteps", "musicboxes", "musicboxes", "musicboxes", "ledgersteps"]
    (r60, c60), (r61a, c61a), (r61b, c61b), (r61, c61) = laying
    assert (c60, c61a, c61b, c61, cap62e, cap62, cap63, cap64a, cap64) == \
        ("loose", "first pair", "next pair", "last pair", "as Achilles expects",
         "both ends drawn in", "the last note", "one word waiting", "the same word twice")
    six = DIALOGUE.section("Sixth")

    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a1, a2, a3 = DIALOGUE.anchors("Sixth", path)
        assert "one below the far-left box and one below the far-right" in a1
        assert "joined now by one unbroken bend across the whole front of the row" in a2
        assert "The Crab turns the ledger to tonight's page" in a3
        assert "a single high tone, the same as the first" in a3
    # the directions edited this round (D8, D10, D11), in both files
    P.check_edits(DIALOGUE, "Sixth", [
        ("the freed string lies loose and bare in front of them, nothing threaded on it",
         "the Tortoise lays the freed string loose and bare in front of them, from one end of "
         "the row to the other, nothing threaded on it"),
        ("joined now by one unbroken bend across the whole front of the row.]*",
         "joined now by one unbroken bend across the whole front of the row, and its two free "
         "ends set down again at the near edge.]*"),
        ("laying the string on the cloth in front of the row as he speaks,",
         "laying the string on the cloth in front of the row as she speaks,")])

    # 6.1, the strip: the row, keyed; boxes in wire order; box 1 spent; the string laid
    # pair by pair from the bare string -- each frame the state after its step
    rows = (r60, r61a, r61b, r61, r62e, r62, r63)
    for r in rows:
        assert r.layout == "row" and r.order == (1, 2, 3, 4, 5, 6)
        assert r.state(1) == "spent" and all(r.state(w) == "shut" for w in range(3, 7))
    assert p61.keyed and r60.keyed and not any(r.keyed for r in rows[1:])
    assert r60.labels == ("pearwood", "walnut", "ebony") and r60.labels_at == "above the pairs"
    assert [(s.opts, s.state) for s in P.frames_of(p61)] == \
        [("bare", "row; left pearwood spent")] + \
        [(w + "; lay=next pair; keyroom=on", "row; left pearwood spent")
         for w in ("bare", "one bend", "two bends")]
    # the keyed strip's other frames keep the key's room: the labels as vertical phantoms
    assert not r60.phantom and all(r.phantom and (r.labels, r.labels_at) == (r60.labels, r60.labels_at)
                                   for r in (r61a, r61b, r61))
    assert not any(r.phantom or r.labels for r in (r62e, r62, r63))
    # before: loose and bare along the near side, from one end of the row to the other (D8),
    # nothing bent and nothing threaded
    assert r60.string_word == "bare" and r60.string.bends == () and not r60.marked
    (bare_path, bare_ends), = P.row_strands("bare")[0]
    for (x, _, _, _), box in zip(bare_ends, (P.ROWX[0], P.ROWX[-1])):  # end under box 1, box 6
        assert abs(x - box) < P.RW / 2
    assert max(y for _, y in P.sample_path(bare_path)) < P.LEGTOP
    # each lay: the next pair's bend, the string before it left as a ghost; the laid part
    # joins box 1 to box 2k -- one bend, then a turn and a bend: one pair, cup_{1,2k}
    for k, (before, after) in enumerate(((r60, r61a), (r61a, r61b), (r61b, r61)), start=1):
        assert after.step == ("lay", "next pair") and after.arcs == [] and after.pins == []
        assert [(g.what, g.state) for g in after.ghosts] == [("string", before.string)]
        assert after.string.names == tuple(range(1, 2 * k + 1))
        assert after.string.count("far") == k and after.string.count("near") == k - 1
        assert np.allclose(string_map(after.string, "far").ravel(), CUP)
    assert r61.string == P.row_string("three bends") and r61.string.ends == ("near", "near")
    assert {b.legs for b in r61.string.bends if b.side == "far"} == \
        {(P.WIRE[("left", p)], P.WIRE[("right", p)]) for p in P.PAIRS}   # a bend at each pair
    assert np.allclose(string_map(r61.string, "far"), kron(I2, CAP, CAP, I2) @ kron(CUP, CUP, CUP))

    # 6.2, as Achilles expects (the gaps rule, on the pulled row): three pieces, a bend at
    # each pair, cut at each station, paler -- six free ends, three pairs side by side
    assert P.options(p62e) == {"expects": "gaps", "bare": "three bends pulled"} and r62e.expects == "gaps"
    assert not p62e.keyed and not r62e.marked and r62e.step is None
    assert r62e.string_word == "three bends pulled" and r62e.string is None
    assert {sc for sc, _ in r62e.drawn} == {sc for sc, _ in r62e.strands} == {"pale"}
    gaps = kron(*[string_map(p, "far") for p in r62e.pieces])
    assert np.allclose(gaps, kron(CUP, CUP, CUP)) and gaps.shape == (2 ** 6, 1)
    assert np.allclose(kron(I2, CAP, CAP, I2) @ gaps, string_map(r61.string, "far"))  # the turns join
    # drawn: the laid string cut at each station -- its corners, piece by piece, the two near
    # turns (between legs 2-3 and 4-5) left out; the outer ends free at the near edge
    corners = lambda path: [tuple(map(float, c)) for c in re.findall(r"\((-?[\d.]+),(-?[\d.]+)\)", path)]
    drawn = [corners(t) for sc, t in r62e.strands]
    (laid_path, _), = P.row_strands("three bends")[0]
    assert sum(drawn, []) == corners(laid_path) and [len(c) for c in drawn] == [4, 4, 4]
    for k, c in enumerate(drawn):
        assert [x for x, _ in c] == [P.ROWX[2 * k]] * 2 + [P.ROWX[2 * k + 1]] * 2
        assert (c[0][1], c[-1][1]) == ((P.ENDY if k == 0 else P.LEGBOT), (P.ENDY if k == 2 else P.LEGBOT))
    # 6.2, the pull: both ends drawn in, the walnut bend held -- nothing else moved; one far
    # bend, box 1 to box 6: the Chain
    assert P.options(p62) == {"pull": "ends, walnut bend held", "bare": "three bends"}
    assert p62.args[0] == p62e.args[0] == p61.args[0]                 # the same row
    assert {w: r62.state(w) for w in range(1, 7)} == {w: r61.state(w) for w in range(1, 7)}
    assert [(b.side, b.legs) for b in r62.string.bends] == [("far", (1, 6))]
    assert np.allclose(string_map(r62.string, "far").ravel(), CUP)
    assert "One bend. From Alice's box to Bob's." in six
    # the prediction differs from the truth beside it: three pieces against one string
    assert len(r62e.pieces) == 3 and len(r62e.strands) == 3 and len([s for s in r62.strands
                                                                     if s[0] == ""]) == 1
    # the marks: the laid row's ghost, the two stations' turns running up into the bend,
    # each between the ghost's legs it does not pass and clear of every box; a ring on the
    # bend in front of the walnut pair, under Achilles's finger; the ends at the near edge,
    # under boxes 1 and 6 (D10)
    assert r62.step == ("pull", "ends, walnut bend held")
    (ghost,) = r62.ghosts
    assert (ghost.what, ghost.state) == ("string", r61.string)
    (pin,) = r62.pins
    assert pin.name == "walnut bend" and (pin.x, pin.y) == ((P.ROWX[2] + P.ROWX[3]) / 2, P.LEGTOP)
    assert pin.y < P.ROWY - P.RD / 2 - P.RH                        # in front of the walnut pair
    (path, ends), = P.row_strands("three bends pulled")[0]
    assert [e[:2] for e in ends] == [(P.ROWX[0], P.ENDY), (P.ROWX[5], P.ENDY)]
    assert [e[:2] for e in P.row_strands("three bends")[0][0][1]] == [e[:2] for e in ends]
    ghost_line = P.sample_path(ghost.strands[0][0])
    for arc, x in zip(r62.arcs, ((P.ROWX[1] + P.ROWX[2]) / 2, (P.ROWX[3] + P.ROWX[4]) / 2)):
        assert arc.name in ("turn first station into the bend", "turn second station into the bend")
        # the two ghost legs either side of the station (the turn joins them; it passes neither)
        legs = [p for p in ghost_line if np.isclose(abs(p[0] - x), (P.ROWX[1] - P.ROWX[0]) / 2)]
        assert len({round(p[0], 6) for p in legs}) == 2 and len(legs) > 20
        assert np.isclose(arc.pts[0][0], x) and arc.pts[0][1] < arc.pts[-1][1] < P.LEGTOP
        assert P.gap_points(arc.points, legs) - P.SWING_W / 2 - P.STRING_W / 2 > 0.1
        assert min(P.gap_rect(arc.points, *d.body) for _, d in r62.drawn) - P.SWING_W / 2 > 0
        (x0, y0), *_, (x1, y1) = arc.points
        assert any(abs((xx - x0) * (y1 - y0) - (yy - y0) * (x1 - x0)) > 1e-3 for xx, yy in arc.points)
    # "the hairpin you laid for the walnut boxes, only wider": the Third's pair, the same map
    hairpin = P.pictures(DIALOGUE.blocks("Third")[0][0])[1][0]
    assert hairpin.shape == "hairpin"
    assert np.allclose(string_map(hairpin.string, "near").ravel(), string_map(r62.string, "far").ravel())

    # 6.3: the last note -- box 2's lid up, the string's one bend unchanged, nothing marked
    assert r63.string == r62.string and r63.state(2) == "open" and r63.boxes[2].lid_up
    assert not r63.marked and r63.step is None
    # the first note and the last are the two halves of one pair (the First's strip, its
    # second frame: the lid up)
    b11 = P.pictures(DIALOGUE.blocks("First")[0][0])[1][0]
    assert [w for w in range(1, 7) if b11.boxes[w].lid_up] == [1]
    assert [w for w in range(1, 7) if r63.boxes[w].lid_up] == [2]
    assert (b11.boxes[1].wood, b11.boxes[1].side, r63.boxes[2].wood, r63.boxes[2].side) == \
        ("pearwood", "left", "pearwood", "right")

    # 6.3, the ledger strip: tonight's page (the First's, P1.3), then its right-hand word
    # written: the same word twice
    t13 = P.picture(DIALOGUE.blocks("First")[1][1])
    assert [(s.opts, s.state) for s in P.frames_of(p64)] == \
        [("open", "low, high, high, low, tonight high"),
         ("open; write=right", "low, high, high, low, tonight high")]
    assert t64a.lines == t13.lines and not t64a.marked and not t64a.keyed
    assert t64.step == ("write", "right") and not t64.marked
    assert t13.written[:-1] == t64.written[:-1]
    assert t13.lines[:-1] == t64.lines[:-1]
    assert (t64.lines[-1].left, t64.lines[-1].right, t64.lines[-1].ink) == ("high", "high", "fresh")
    assert t13.lines[-1].right is None
    assert "**Achilles:** High." in six

    # the log the pictures write, line for line as the build writes it
    head = "musicboxes: row; left pearwood spent | string "
    page = ["low low", "high high", "high high", "low low"]
    assert P.section_dumps(found) == [
        "boxsteps: strip of 4 frames",
        head + "bare | keyed", "musicboxes: + string bare",
        head + "one bend", "musicboxes: + step lay next pair", "musicboxes: + ghost string bare",
        "musicboxes: + string one bend",
        head + "two bends", "musicboxes: + step lay next pair", "musicboxes: + ghost string one bend",
        "musicboxes: + string two bends",
        head + "three bends", "musicboxes: + step lay next pair",
        "musicboxes: + ghost string two bends", "musicboxes: + string three bends",
        head + "three bends pulled",
        "musicboxes: + expects gaps: three pieces, a far bend at each pair, cut at each station; "
        "pulled, each piece keeps its one bend (one far, none near)",
        "musicboxes: + string three bends pulled (drawn by the gaps rule)",
        head + "three bends pulled", "musicboxes: + step pull ends, walnut bend held",
        "musicboxes: + ghost string three bends", "musicboxes: + string three bends pulled",
        "musicboxes: + arc turn first station into the bend",
        "musicboxes: + arc turn second station into the bend",
        "musicboxes: + pin walnut bend",
        "musicboxes: row; left pearwood spent; right pearwood open | string three bends pulled",
        "musicboxes: + string three bends pulled",
        "ledgersteps: strip of 2 frames",
        "crabledger: open | low, high, high, low, tonight high",
        *["crabledger: + line " + w + " old" for w in page], "crabledger: + line high blank fresh",
        "crabledger: open | low, high, high, low, tonight high completed",
        "crabledger: + step write right",
        *["crabledger: + line " + w + " old" for w in page], "crabledger: + line high high fresh"]


def the_chain():
    """Six rows (row k-1 is wire k).  Bends sit between the rows they join:
    cups on the left at 0.5, 2.5, 4.5; caps on the right at 1.5, 3.5."""
    p = Picture()
    cup12 = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    cup34, cup56 = p.bend(2.5, 1), p.bend(4.5, 1)
    cap23, cap45 = p.bend(1.5, 3), p.bend(3.5, 3)
    # the string: out round 1-2, back between 2 and 3, out round 3-4, back, out round 5-6
    p.wire(p.out(0), cup12, cap23, cup34, cap45, cup56, p.out(5))
    return p.done()


def one_bend():
    p = Picture()
    bend = p.bend(0.5, 1)
    p.wire(p.out(0), bend, p.out(1))
    return p.done()


def sketchbook():
    chain = the_chain()
    assert chain.num_edges() == 6                           # one stretch of string per wire
    assert len(chain.inputs()) == 0 and len(chain.outputs()) == 2
    top, bottom = chain.outputs()                           # outputs sorted top wire first;
    assert (chain.qubit(top), chain.qubit(bottom)) == (0, 5)    # PyZX calls our rows "qubits"
    assert np.allclose(matrix_of(one_bend()).ravel(), CUP)

    assert same_picture(chain, one_bend())                  # pull: one bend, wire 1 to wire 6

    # the normalized version: 1/sqrt(2) beside each of the five bends
    relay = the_chain()
    relay.scalar.add_power(-5)
    assert np.allclose(matrix_of(relay).ravel(), BELL / 4)
    assert not same_picture(relay, one_bend())


if __name__ == "__main__":
    verdict("Chain", blackboard, sketchbook)
