"""Challenge 1: Two Recipe Cards
==============================
"Two cards won't do it. It needs one card for the pair."  -- Achilles

Claim: the Bell state is not a product of two one-wire states: a two-wire
state c00|00> + c01|01> + c10|10> + c11|11> is a product exactly when
c00*c11 - c01*c10 = 0, and for the Bell state that number is 1/2.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction

import numpy as np
import pyzx
from dfp import BELL, CUP, I2, Picture, ket, kron, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch06 as P


def det_of(state):
    """c00*c11 - c01*c10.  Top wire first: the TOP wire is the row index."""
    c = np.asarray(state).reshape(2, 2)
    return c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0]


def blackboard():
    rng = np.random.default_rng(61)

    # conventions: top wire first; the cup is unnormalized; Bell = cup / sqrt(2)
    assert ket(0, 1)[1] == 1 and ket(1, 0)[2] == 1
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))
    assert np.allclose(BELL, CUP / np.sqrt(2))

    # two recipe cards: a product state always has determinant 0
    alpha, beta = random_state(rng), random_state(rng)
    two_cards = np.kron(alpha, beta)
    assert np.isclose(det_of(two_cards), 0)

    # the pair: determinant 1/2 (and 1 for the unnormalized cup) -- no two cards
    assert np.isclose(det_of(BELL), 1 / 2)
    assert np.isclose(det_of(CUP), 1)

    # the converse, on a sample: determinant 0 means rank one, and a rank-one
    # table of amplitudes IS a product (column times row)
    rank = lambda state: np.linalg.matrix_rank(np.asarray(state).reshape(2, 2))
    assert rank(two_cards) == 1 and rank(BELL) == 2
    generic = random_state(rng, 4)
    assert not np.isclose(det_of(generic), 0) and rank(generic) == 2

    # the Bell state is nothing exotic: H on the top wire, then CNOT, on |00>
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert np.allclose(CNOT @ kron(H, I2) @ ket(0, 0), BELL)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the Crab's boxes and his ledger with
#   \musicboxes[opts]{boxes}{caption}   \crabledger[opts]{lines}{caption}
# and, to show a move as it happens, a step strip of either,
#   \boxsteps[opts]{boxes}{steps}{captions}   \ledgersteps[opts]{lines}{steps}{captions}
# (pdf/props/ch06.tex).  A picture's bracket carries at most one step (open=,
# close=, keep= on the boxes): the state WRITTEN is the state before it, and the
# Lua computes the state after and draws the step's marks -- a ghost of the part
# as it was, a swing arc for the way it went.  A strip is a before-frame and one
# frame per step, each frame's state computed from the one before.  The Lua also
# COMPUTES the ledger's right-hand column (a pair opened by the same door plays
# one word: the left word again) and draws a spent box shut; tools/props_ch06.py
# transcribes it all.  The calls are read out of this challenge's section of
# dialog.md (tools/dfp_props.py), never copied, so an edit to the dialogue that
# breaks a picture breaks here.
#
# The state model (identity card, sections 3 and 3e): wire map left pearwood = 1
# ... right ebony = 6; box 1 open at the first note and spent from then on; box 2
# lifted apart in the Crab's claw.  A ledger line is one pair's joint outcome,
# the left box's word then the right box's; the lid's words (high, low) are one
# pair of outcomes, the side door's (bright, dark) another.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("musicboxes", "walnutstring", "beadstring", "crabledger",
                                      "boxsteps", "walnutsteps", "beadsteps", "ledgersteps"))


def two_cards_disagree():
    """Achilles's two cards: two fair coins tossed separately.  How often
    would a ledger line built from them disagree?"""
    coin = np.array([1 / 2, 1 / 2])
    joint = np.outer(coin, coin)                                  # the two cards, side by side
    return joint[0, 1] + joint[1, 0]


def the_pair_disagrees():
    """The same, from the one card for the pair (the Bell state), by the lid."""
    joint = [[abs((kron(ket(i), ket(j)).conj().T @ BELL).item()) ** 2 for j in (0, 1)]
             for i in (0, 1)]                                     # top wire = left box
    return joint[0][1] + joint[1][0]


def tray_moments():
    """The tray at each moment of the First, as the text has it (card, section 3e.2's trace):
    the six shut and full; box 1's lid up at the first note ("only the little hinged lid
    swings up"); box 1 spent ("the lid settles back on a box with nothing left inside");
    box 2 lifted off the tray and held apart, shut and full."""
    full = {w: "shut" for w in range(1, 7)}
    return [full, {**full, 1: "open"}, {**full, 1: "spent"}, {**full, 1: "spent", 2: "kept"}]


def check_transcription():
    """The macros' own rules (the Lua's tex.error, word for word), as the transcription has
    them; the Lua itself the one the transcription was proved against."""
    P.check_sites()
    P.check_lua()
    assert P.faults(P.musicboxes, "", "table; left pearwood open") == [
        "musicboxes: the first item is the layout, tray or row (got 'table')"]
    assert P.faults(P.musicboxes, "", "tray; left pearwood open; left pearwood spent") == [
        "musicboxes: a box is open or spent, not both"]
    assert P.faults(P.musicboxes, "", "tray; left pearwood kept; right pearwood kept") == [
        "musicboxes: only one box is kept off the tray"]
    assert P.faults(P.musicboxes, "", "tray; left pearwood spent; left pearwood kept") == [
        "musicboxes: a kept box is shut and full"]
    assert P.faults(P.musicboxes, "three bends", "tray") == [
        "musicboxes: the string is laid before the row, not the tray"]
    assert P.faults(P.musicboxes, "", "row; right pearwood kept") == [
        "musicboxes: a box is kept off the tray only when the boxes are on the tray"]
    assert P.faults(P.crabledger, "", "high, tonight low, high") == [
        "crabledger: tonight's line is the last written"]
    assert P.faults(P.crabledger, "", "low completed") == [
        "crabledger: only tonight's line waits to be completed"]
    assert P.faults(P.crabledger, "", ", ".join(["high"] * 9)) == [
        "crabledger: a page holds eight lines"]
    assert P.faults(P.crabledger, "blotter across", "high") == [
        "crabledger: the blotter is open (none laid), blotter right or blotter left (got "
        "'blotter across')"]
    # the steps on the tray: a lid lifts only off a shut, full box, settles only on an open
    # one; a box is kept only off the tray, shut and full, one at a time
    assert P.faults(P.musicboxes, "open=left pearwood", "tray; left pearwood spent") == [
        "musicboxes: open=left pearwood: only a shut, full box is opened (it is spent)"]
    assert P.faults(P.musicboxes, "close=left pearwood", "tray") == [
        "musicboxes: close=left pearwood: only an open box's lid settles back (it is shut)"]
    assert P.faults(P.musicboxes, "keep=right pearwood", "row") == [
        "musicboxes: keep= lifts a box off the tray: the tray"]
    assert P.faults(P.musicboxes, "keep=left pearwood", "tray; left pearwood spent") == [
        "musicboxes: keep=left pearwood: only a shut, full box is kept (it is spent)"]
    assert P.faults(P.musicboxes, "keep=right pearwood", "tray; left walnut kept") == [
        "musicboxes: keep=: a box is kept already"]
    assert P.faults(P.musicboxes, "open=middle pearwood", "tray") == [
        "musicboxes: open=<left|right> <pearwood|walnut|ebony> (not 'middle pearwood')"]
    assert P.faults(P.musicboxes, "open=left pearwood; close=left pearwood", "tray") == [
        "musicboxes: one step to a picture (open= and close=)"]
    assert P.faults(P.musicboxes, "marks=off", "tray") == ["musicboxes: marks= go with a step"]
    assert P.faults(P.musicboxes, "open=left pearwood; marks=dim", "tray") == [
        "musicboxes: marks is on or off (not 'dim')"]
    assert P.faults(P.musicboxes, "shut=left pearwood", "tray") == [
        "musicboxes: unknown option 'shut'"]
    # a strip: steps of its own macro only, at most four frames, a caption to each frame
    assert P.faults(P.strip, "boxsteps", True, "tray", "", "lift=left pearwood", "a") == [
        "boxsteps: a step is one of close=, keep=, lay=, open=, pull= (not 'lift=left pearwood')",
        "boxsteps: a strip takes at least one step"]
    assert P.faults(P.strip, "boxsteps", True, "tray", "", "open=left pearwood", "a; b; c") == [
        "boxsteps: 2 frames but 3 captions"]
    assert P.faults(P.strip, "boxsteps", True, "tray", "", "; ".join(
        ["open=left pearwood", "close=left pearwood", "open=right pearwood", "close=right pearwood"]),
        "a; b; c; d; e") == ["boxsteps: 5 frames; at most four fit a line (from=off)"]
    # a spent box is drawn shut: only the lamp's glint on its lid is missing
    b = P.musicboxes("", "tray; left pearwood spent")
    assert b.boxes[1].looks_shut and not b.boxes[1].glint and b.boxes[2].glint
    # the keep's lift is drawn for the pearwood only (card 3e.7, ruling (b)): kept off a lower
    # row, its arc would cross a box the lifted box does not pass
    low = P.clean(P.musicboxes, "keep=right ebony", "tray")
    walnut = next(d for _, d in low.drawn if (d.wood, d.x, d.y) == ("walnut", P.TRAYCOL["right"],
                                                                     P.TRAYROW["walnut"]))
    assert P.gap_rect(low.arcs[0].points, *walnut.body) < 0


def clear_of(arc, drawn, passed=()):
    """The least gap from a swing arc's drawn line to every box drawn but those the moving
    piece passes (its ghost, its own box, where it lands): positive, all of them."""
    gaps = [P.gap_rect(arc.points, *d.body) - P.SWING_W / 2 for sc, d in drawn
            if (d.x, d.y) not in passed]
    assert gaps and min(gaps) > 0, (arc.name, gaps)
    return min(gaps)


def pictures():
    check_transcription()
    found = P.check_section(DIALOGUE, "First", sizes=[1, 2, 1], frames=[3, 2, 1])
    (p11, strip), = found[0]
    (p12, [(f04, cap04)]), (p13, [(t13, cap13)]) = found[1]
    (p14, [(c14, cap14)]), = found[2]
    assert [c.macro for c in (p11, p12, p13, p14)] == \
        ["boxsteps", "musicboxes", "crabledger", "crabledger"]
    (f01, cap01), (f02, cap02), (f03, cap03) = strip
    assert (cap01, cap02, cap03, cap04) == ("shut and full", "the lid up", "spent", "held apart")

    # anchors: each block sits after the moment it shows (both files).  The first note's
    # direction is whole again (D0): the strip follows it, the note sounded and gone.
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a1, a2, a3 = DIALOGUE.anchors("First", path)
        assert "a single high tone, clear as a struck glass. Then it is gone" in a1
        assert a1.endswith("a small pearwood box that will never sing again.]*")
        assert "out over its right-hand rim" in a2 and "you have had high" in a2
        assert a2.endswith("And nobody touches this one.")
        assert a3.endswith("I was told so very firmly.")
        after = DIALOGUE.section("First", path).split(p14.text)[1]
        assert after.split("**Achilles:**")[1].lstrip().startswith("They never disagree.")
    # the directions edited this round (D0, D1), in both files
    P.check_edits(DIALOGUE, "First", [
        ("clear as a struck glass.]*", "clear as a struck glass. Then it is gone,"),
        ("then he lifts the right pearwood box off the tray and folds it against his shell",
         "then he lifts the right pearwood box off the tray, out over its right-hand rim, and "
         "folds it against his shell")])

    # the tray, frame by frame, against the moments (the strip, then the keep)
    frames = (f01, f02, f03, f04)
    for b, moment in zip(frames, tray_moments()):
        assert b.layout == "tray" and b.string is None
        assert {w: b.state(w) for w in range(1, 7)} == moment
        assert all(b.boxes[w].glint == (moment[w] in ("shut", "kept")) for w in moment)
    assert p11.keyed and f01.keyed and not any(b.keyed for b in frames[1:])
    assert f01.labels == ("pearwood", "walnut", "ebony") and f01.labels_at == "beside the rows"
    # the keyed strip's other frames keep the key's room: its labels as vertical phantoms
    # (keyroom=on), so the three frames stand level; the picture after the strip has none
    assert not f01.phantom and f01.label_nodes == f01.labels
    for b in (f02, f03):
        assert b.phantom and (b.labels, b.labels_at) == (f01.labels, f01.labels_at)
        assert b.label_nodes == tuple("\\vphantom{" + t + "}" for t in f01.labels)
    assert not f04.phantom and f04.labels == ()
    assert f02.boxes[1].lid_up and f03.boxes[1].looks_shut and not f03.boxes[1].glint
    # each step frame's state is the state after its step, on the frame before it
    steps = P.frames_of(p11)
    assert [s.opts for s in steps] == ["none", "none; open=left pearwood; keyroom=on",
                                       "none; close=left pearwood; keyroom=on"]
    assert [s.state for s in steps] == ["tray", "tray", "tray; left pearwood open"]
    assert P.options(p12)["keep"] == "right pearwood" and p12.args == (f03.spec,)

    # the marks: nothing moves in the before-frame; the lid swings up, then settles back
    # with its raised self left as a ghost; the kept box leaves its ghost in its tray place
    assert not f01.marked and f01.step is None
    assert f02.step == ("open", "left pearwood") and f02.ghosts == [] and f02.pins == []
    assert [a.name for a in f02.arcs] == ["lid left pearwood up"]
    assert f03.step == ("close", "left pearwood") and f03.pins == []
    assert [(g.what, g.state) for g in f03.ghosts] == [("lid", "open")]
    assert f03.ghosts[0].place == (P.TRAYCOL["left"], P.TRAYROW["pearwood"])   # its own box
    assert [a.name for a in f03.arcs] == ["lid left pearwood down"]
    assert f03.arcs[0].pts == tuple(reversed(f02.arcs[0].pts))                 # the same swing, back
    assert f04.step == ("keep", "right pearwood") and f04.pins == []
    (ghost,) = f04.ghosts
    assert (ghost.what, ghost.state) == ("box", f03.state(2))                  # shut and full, as it was
    assert ghost.place == (P.TRAYCOL["right"], P.TRAYROW["pearwood"])
    kept = (P.KEPTX, P.TRAYROW["pearwood"] + P.KEPTDY)
    assert [(d.x, d.y) for sc, d in f04.drawn if d.raised] == [kept]           # beside the tray
    assert [a.name for a in f04.arcs] == ["lift right pearwood over the rim"]
    # every arc is a curve, and crosses no box the moving piece does not pass: the lid's
    # arc bows into the pair's gap (clear of the right pearwood box), the lift goes over
    # the tray's right rim (D1: the side it passes) to where the box is held
    for a in f02.arcs + f03.arcs + f04.arcs:
        (x0, y0), *_, (x1, y1) = a.points
        assert any(abs((x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)) > 1e-3 for x, y in a.points)
    box1 = (P.TRAYCOL["left"], P.TRAYROW["pearwood"])
    for a in f02.arcs + f03.arcs:
        assert clear_of(a, f02.drawn, passed=(box1,)) > 0.1
        right = P.TRAYCOL["left"] + P.TW / 2, P.TRAYCOL["right"] - P.TW / 2
        assert all(right[0] <= x < right[1] for x, _ in a.points)             # in the gap
    (lift,) = f04.arcs
    assert clear_of(lift, f04.drawn, passed=(ghost.place, kept)) > 0.4
    xs = [x for x, _ in lift.points]
    assert min(xs) < P.TRAY[2] < max(xs)                                       # over the right rim
    assert min(y for x, y in lift.points if abs(x - P.TRAY[2]) < 0.01) > P.TRAY[3]

    # page T, keyed -- four finished lines, then tonight's lone word in fresh ink
    assert cap13 == "one word waiting" and p13.keyed and t13.labels == ("left boxes", "right boxes")
    assert t13.written == ("low", "high", "high", "low", "tonight high") and not t13.marked
    assert all(l.right == l.left and l.ink == "old" for l in t13.lines[:4])
    tonight = t13.lines[-1]
    assert (tonight.left, tonight.right, tonight.ink) == ("high", None, "fresh")
    assert "a single high tone" in a1                     # the word is the note just heard

    # P1.4: page C, eight finished lines, each the same word twice, door words and lid words
    assert cap14 == "read across" and not c14.keyed and len(c14.lines) == 8
    assert c14.blotter == "open" and not c14.marked
    assert all(l.right == l.left for l in c14.lines)
    assert {l.left for l in c14.lines} >= {"bright", "dark", "high", "low"}
    # the leaf after page A (the old page, pictured in the Second): Achilles's
    # spoken reading of A and of C's first line fits the drawn pages
    page_a = P.picture(DIALOGUE.blocks("Second")[0][0])
    reading = re.search(r"\*\*Achilles:\*\* ((?:(?:High|Low), (?:high|low)\. ?)+)",
                        DIALOGUE.section("First")).group(1)
    pairs = re.findall(r"(High|Low), (high|low)\.", reading)
    assert [(a.lower(), b) for a, b in pairs] == \
        [(l.left, l.right) for l in page_a.lines[:len(pairs)]] and len(pairs) == 5
    assert "Here's a \"bright, bright\"" in DIALOGUE.section("First")
    assert (c14.lines[0].left, c14.lines[0].right) == ("bright", "bright")
    assert c14.written != page_a.written != t13.written

    # the log the pictures write, line for line as the build writes it
    assert P.section_dumps(found) == [
        "boxsteps: strip of 3 frames",
        "musicboxes: tray | string none | keyed",
        "musicboxes: tray; left pearwood open | string none",
        "musicboxes: + step open left pearwood",
        "musicboxes: + arc lid left pearwood up",
        "musicboxes: tray; left pearwood spent | string none",
        "musicboxes: + step close left pearwood",
        "musicboxes: + ghost lid left pearwood raised",
        "musicboxes: + arc lid left pearwood down",
        "musicboxes: tray; left pearwood spent; right pearwood kept | string none",
        "musicboxes: + step keep right pearwood",
        "musicboxes: + ghost box right pearwood in its tray place",
        "musicboxes: + arc lift right pearwood over the rim",
        "crabledger: open | low, high, high, low, tonight high | keyed",
        "crabledger: + line low low old",
        "crabledger: + line high high old",
        "crabledger: + line high high old",
        "crabledger: + line low low old",
        "crabledger: + line high blank fresh",
        "crabledger: open | bright, dark, dark, bright, dark, bright, high, low",
        "crabledger: + line bright bright old",
        "crabledger: + line dark dark old",
        "crabledger: + line dark dark old",
        "crabledger: + line bright bright old",
        "crabledger: + line dark dark old",
        "crabledger: + line bright bright old",
        "crabledger: + line high high old",
        "crabledger: + line low low old"]

    # the surprise, against the blackboard: two cards (two separate coins)
    # would write a disagreeing line half the time; the pair, never -- and no
    # drawn page has one
    assert np.isclose(two_cards_disagree(), 1 / 2)
    assert np.isclose(the_pair_disagrees(), 0)
    for page in (t13, c14):
        assert not any(l.right is not None and l.right != l.left for l in page.lines)


def bent_wire():
    """The pair as ONE connected picture: a wire bent round, both ends out."""
    p = Picture()
    bend = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(bend, p.out(0))
    p.wire(bend, p.out(1))
    return p.done()


def two_separate_states(a, b):
    """Two recipe cards: a DISCONNECTED picture, one one-wire state per row."""
    p = Picture()
    p.wire(p.z(0, 1, a), p.out(0))
    p.wire(p.z(1, 1, b), p.out(1))
    return p.done()


def prepared_pair():
    """|00>, then H on the top wire, then CNOT -- as a picture."""
    p = Picture()
    p.wire(p.x(0, 1), p.out(0))          # PyZX's phase-free one-legged x node
    p.wire(p.x(1, 1), p.out(1))          # is sqrt(2)|0>, so ...
    g = p.done()
    g.scalar.add_power(-2)               # ... divide by sqrt(2), twice: now |00>
    assert np.allclose(matrix_of(g).ravel(), ket(0, 0))
    circuit = pyzx.Circuit(2)
    circuit.add_gate("HAD", 0)           # row 0 is the top wire
    circuit.add_gate("CNOT", 0, 1)
    g.compose(circuit.to_graph())        # first the state, then the circuit
    return g


def sketchbook():
    # (i) a disconnected picture is a product, and is not the bent wire
    cards = two_separate_states(Fraction(1, 4), Fraction(1, 3))
    assert np.isclose(det_of(matrix_of(cards)), 0)
    assert np.isclose(det_of(matrix_of(bent_wire())), 1)
    assert not same_picture(cards, bent_wire())

    # (ii) the preparation circuit draws 1/sqrt(2) times the bent wire -- exactly
    bell = bent_wire()
    bell.scalar.add_power(-1)            # the number 1/sqrt(2) written beside it
    assert np.allclose(matrix_of(bell).ravel(), BELL)
    assert same_picture(prepared_pair(), bell)
    assert not same_picture(prepared_pair(), bent_wire())   # the 1/sqrt(2) matters


if __name__ == "__main__":
    verdict("NotAProduct", blackboard, sketchbook)
