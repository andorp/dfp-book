"""Challenge 2: The Ledger
========================
"At which level are you asking?"  -- Anteater

Claim: each half of a Bell pair, asked alone, is a fair coin in every basis:
(<b| (x) I) Bell = |b*> / sqrt(2), so every outcome has probability 1/2 and
the other half is left in |b*>; in every real basis the two halves agree.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction

import numpy as np
from dfp import BELL, CUP, I2, Picture, ket, kron, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch06 as P


def bra(v):
    """The effect <v|: conjugate the amplitudes and lay them in a row."""
    return np.asarray(v).conj().reshape(1, -1)


def joint_table(basis):
    """Probabilities of the four joint outcomes when both wires are asked in `basis`."""
    return [abs((kron(bra(top), bra(bottom)) @ BELL).item()) ** 2
            for top in basis for bottom in basis]          # top wire first


def blackboard():
    rng = np.random.default_rng(62)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))     # unnormalized cup
    assert np.allclose(BELL, CUP / np.sqrt(2))

    b = random_state(rng)
    assert not np.allclose(b, b.conj())                    # generic: not a real vector

    # one part alone (the blotter over the other column): a fair coin
    left_behind = (kron(bra(b), I2) @ BELL).ravel()        # ask the TOP wire for b
    assert np.allclose(left_behind, b.conj() / np.sqrt(2))       # |b*> / sqrt(2)
    assert not np.allclose(left_behind, b / np.sqrt(2))          # not |b> / sqrt(2)
    assert np.isclose(np.linalg.norm(left_behind) ** 2, 1 / 2)   # probability 1/2

    # the same from the other side
    left_behind = (kron(I2, bra(b)) @ BELL).ravel()        # ask the BOTTOM wire for b
    assert np.allclose(left_behind, b.conj() / np.sqrt(2))
    assert np.isclose(np.linalg.norm(left_behind) ** 2, 1 / 2)

    # the whole (the blotter lifted): in a real basis the columns are identical
    lid = [ket(0), ket(1)]
    side_door = [(ket(0) + ket(1)) / np.sqrt(2), (ket(0) - ket(1)) / np.sqrt(2)]
    assert np.allclose(joint_table(lid), [1 / 2, 0, 0, 1 / 2])
    assert np.allclose(joint_table(side_door), [1 / 2, 0, 0, 1 / 2])

    # ... and only in a real basis: here b* is the OTHER basis vector
    not_real = [(ket(0) + 1j * ket(1)) / np.sqrt(2), (ket(0) - 1j * ket(1)) / np.sqrt(2)]
    assert np.allclose(joint_table(not_real), [0, 1 / 2, 1 / 2, 0])

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the Crab's ledger with
#   \crabledger[opts]{lines}{caption}      (pdf/props/ch06.tex)
# The Lua COMPUTES the right-hand column (the left word again) and lays the
# blotter over one column; a bracket may carry one step on the blotter (slide=,
# lift=, lay=) -- the state written is the state before it, and the Lua draws the
# blotter where the step leaves it, its ghost where it lay, and, for a slide, the
# arc it went by.  tools/props_ch06.py transcribes it.  The calls are read out of
# this challenge's section of dialog.md, never copied.
#
# The state model (identity card, sections 3 and 3e): one page, A, under the
# blotter on the right, then slid across onto the left, then lifted clean off; a
# line is one pair's joint outcome, left box first (the top wire).  The lid asks
# in one basis (high, low), the side door in another (bright, dark).
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("musicboxes", "walnutstring", "beadstring", "crabledger",
                                      "boxsteps", "walnutsteps", "beadsteps", "ledgersteps"))
DOORS = {"lid": (("high", "low"), [ket(0), ket(1)]),
         "side door": (("bright", "dark"),
                       [(ket(0) + ket(1)) / np.sqrt(2), (ket(0) - ket(1)) / np.sqrt(2)])}


def blotter_moments():
    """Where the blotter lies at each picture of the Second, as the text has it: down the
    right-hand column ("edge to the centre rule"); slid across to bury the left column,
    uncovering the right; lifted clean off.  And so which columns the lamp reads."""
    return [("blotter right", ("left",)), ("blotter left", ("right",)), ("open", ("left", "right"))]


def check_transcription():
    """The Lua's ledger against the blackboard (a structural check, not a
    second proof): the rule 'right = left on every finished line, by either
    door' is the pair's joint table having nothing off its diagonal, in the
    lid's basis and in the door's; and either column alone is a fair coin."""
    P.check_sites()                                          # every tex.error of ch06.lua
    P.check_lua()                                            # the Lua as it was proved
    for words, basis in DOORS.values():
        table = np.array(joint_table(basis)).reshape(2, 2)       # [left word][right word]
        possible = {(words[i], words[j]) for i in range(2) for j in range(2)
                    if not np.isclose(table[i, j], 0)}
        page = P.crabledger("", ", ".join(words))
        assert possible == {(l.left, l.right) for l in page.lines}   # what the Lua writes
        assert np.allclose(table.sum(axis=1), [1 / 2, 1 / 2])        # the left boxes alone
        assert np.allclose(table.sum(axis=0), [1 / 2, 1 / 2])        # the right boxes alone
    assert P.faults(P.crabledger, "blotter middle", "high") == [
        "crabledger: the blotter is open (none laid), blotter right or blotter left (got "
        "'blotter middle')"]
    assert P.crabledger("blotter right", "high").visible == ("left",)
    assert P.crabledger("blotter left", "high").visible == ("right",)
    assert P.crabledger("", "high").visible == ("left", "right")
    # the blotter's steps: a slide only from the other column, a lift only off a laid
    # blotter, a lay only on an open page; and the words of each refusal (the Lua's)
    assert P.faults(P.crabledger, "blotter left; slide=blotter left", "high") == [
        "crabledger: slide=blotter left: the blotter lies over the other column before it "
        "(blotter right)"]
    assert P.faults(P.crabledger, "slide=blotter right", "high") == [
        "crabledger: slide=blotter right: the blotter lies over the other column before it "
        "(blotter left)"]
    assert P.faults(P.crabledger, "blotter left; slide=across", "high") == [
        "crabledger: slide=blotter left or blotter right (not 'across')"]
    assert P.faults(P.crabledger, "lift=off", "high") == ["crabledger: lift=off: no blotter is laid"]
    assert P.faults(P.crabledger, "blotter left; lift=up", "high") == [
        "crabledger: lift=off (not 'up')"]
    assert P.faults(P.crabledger, "blotter left; lay=blotter right", "high") == [
        "crabledger: lay=: a blotter is laid already"]
    assert P.faults(P.crabledger, "lay=blotter", "high") == [
        "crabledger: lay=blotter left or blotter right (not 'blotter')"]
    assert P.faults(P.crabledger, "blotter left; marks=off", "high") == [
        "crabledger: marks= go with a step"]
    assert P.faults(P.crabledger, "blotter left; lift=off; slide=blotter right", "high") == [
        "crabledger: one step to a picture (slide= and lift=)"]
    assert P.faults(P.crabledger, "blotter left; write=left", "high") == [
        "crabledger: write=right (not 'left')"]
    assert P.faults(P.crabledger, "write=right", "high, tonight low completed") == [
        "crabledger: write=right answers tonight's lone word: the last line is tonight <word>"]
    # a lay comes from by the inkwell, which no picture draws: no ghost, no arc
    lay = P.clean(P.crabledger, "lay=blotter right", "high")
    assert lay.blotter == "blotter right" and lay.step and not lay.marked
    # marks=off: the step taken, nothing drawn of it
    quiet = P.clean(P.crabledger, "blotter right; slide=blotter left; marks=off", "high")
    assert quiet.blotter == "blotter left" and not quiet.marked and quiet.moved is None


def pictures():
    check_transcription()
    found = P.check_section(DIALOGUE, "Second", sizes=[1, 1, 1], frames=[1, 1, 1])
    ((p21, [(a21, _)]),), ((p22, [(a22, _)]),), ((p23, [(a23, _)]),) = found
    for c in (p21, p22, p23):
        assert c.macro == "crabledger" and not c.keyed

    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a1, a2, a3 = DIALOGUE.anchors("Second", path)
        assert "lays the sheet of blotting paper down the right-hand column" in a1
        assert a2.endswith("And the right boxes.") and "bury the left column" in a2
        assert a3.endswith("And now.") and "lifting the blotter clean off" in a3

    # one page, A, in all three: the blotter right, then left, then lifted
    assert a21.written == a22.written == a23.written
    assert p21.args == p22.args == p23.args
    for page, (blotter, visible) in zip((a21, a22, a23), blotter_moments()):
        assert (page.blotter, page.visible) == (blotter, visible)
    # each step is written on the state before it: the slide on the blotter right, the
    # lift on the blotter left -- the picture before
    assert P.options(p22)["bare"] == a21.blotter and P.options(p22)["slide"] == "blotter left"
    assert P.options(p23)["bare"] == a22.blotter and P.options(p23)["lift"] == "off"

    # the marks.  P2.1 is the lay, with no marks (it comes from by the inkwell, which no
    # picture draws).  The slide leaves the blotter's ghost over the right column and an
    # arc from that column to the left one, just above the page, crossing nothing; the
    # lift leaves its ghost over the left column and no arc (the text names no place).
    assert not a21.marked and a21.step is None
    assert a22.step == ("slide", "blotter left") and a22.pins == []
    (g22,) = a22.ghosts
    assert (g22.what, g22.state, g22.place) == ("blotter", a21.blotter, P.blotter_rect(a21.blotter))
    (slide,) = a22.arcs
    assert slide.name == "slide blotter right to blotter left over the top of the page"
    pts = slide.points
    centre = lambda which: sum(P.blotter_x(which)) / 2
    assert np.isclose(pts[0][0], centre(a21.blotter)) and np.isclose(pts[-1][0], centre(a22.blotter))
    ghost_top = P.blotter_rect(a21.blotter)[3] + 0.012          # the ghost's dotted outline
    assert min(y for _, y in pts) - P.SWING_W / 2 > ghost_top > P.PAGE[3]
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    assert any(abs((x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)) > 1e-3 for x, y in pts)  # a curve
    assert a23.step == ("lift", "off") and a23.arcs == [] and a23.pins == []
    (g23,) = a23.ghosts
    assert (g23.what, g23.state, g23.place) == ("blotter", a22.blotter, P.blotter_rect(a22.blotter))
    # a slide and a lift stay distinguishable: the blotter's slide is a flat bow along its
    # surface, the keep's lift in the First a tall hop
    def rise_over_span(arc):
        ps = arc.points
        xs, ys = [x for x, _ in ps], [y for _, y in ps]
        return (max(ys) - min(ps[0][1], ps[-1][1])) / (max(xs) - min(xs))
    (lift,) = [a for b in DIALOGUE.blocks("First") for c in b if c.macro == "musicboxes"
               for a in P.picture(c).arcs]
    assert rise_over_span(slide) < 0.15 and rise_over_span(lift) > 0.45
    # page A is neither tonight's page nor the leaf after it (both pictured in the First)
    first = [P.picture(c) for b in DIALOGUE.blocks("First") for c in b
             if c.macro == "crabledger"]
    assert len(first) == 2 and all(a21.written != q.written for q in first)
    assert first[0].written != first[1].written

    # P2.1: the left boxes alone -- Achilles's tally is the drawn column
    left = a21.column("left")
    tally = re.search(r"(\w+) highs and (\w+) lows on this page", DIALOGUE.section("Second"))
    numbers = {w: n for n, w in enumerate("no one two three four five six seven eight".split())}
    assert (left.count("high"), left.count("low")) == \
        (numbers[tally.group(1).lower()], numbers[tally.group(2)]) == (5, 3)
    # ... and the leaf he peeks at is page C of the First: "bright, dark, dark, bright"
    assert "bright, dark, dark, bright" in DIALOGUE.section("Second")
    assert first[1].column("left")[:4] == ("bright", "dark", "dark", "bright")

    # P2.2: the right boxes alone -- the same run as P2.1's left column, computed
    assert a22.column("right") == left

    # P2.3: the blotter lifted -- the same word twice on every line
    assert all(l.left == l.right for l in a23.lines)
    assert "It's the same column twice." in DIALOGUE.section("Second")

    # the log the pictures write, line for line as the build writes it
    page = ["high high", "low low", "low low", "high high", "low low", "high high",
            "high high", "high high"]
    lines = ["crabledger: + line " + w + " old" for w in page]
    assert P.section_dumps(found) == [
        "crabledger: blotter right | high, low, low, high, low, high, high, high", *lines,
        "crabledger: blotter left | high, low, low, high, low, high, high, high",
        "crabledger: + step slide blotter left",
        "crabledger: + ghost blotter blotter right", *lines,
        "crabledger: + arc slide blotter right to blotter left over the top of the page",
        "crabledger: open | high, low, low, high, low, high, high, high",
        "crabledger: + step lift off",
        "crabledger: + ghost blotter blotter left", *lines]


def effect_on_top_leg(a):
    """A one-legged node plugged into the top leg of a bent wire."""
    p = Picture()
    bend = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(bend, p.z(0, 2, a))
    p.wire(bend, p.out(1))
    return p.done()


def state_on_bottom_wire(a):
    """The same one-legged node, as a state on the bottom wire."""
    p = Picture()
    p.wire(p.z(1, 2, a), p.out(1))
    return p.done()


def sketchbook():
    a = Fraction(1, 3)
    s = np.array([1, np.exp(1j * np.pi * a)])              # the node, read as a state
    assert np.allclose(matrix_of(state_on_bottom_wire(a)).ravel(), s)
    assert not np.allclose(s, s.conj())                    # generic: not real

    # the effect slides round the bend and comes out as a state -- exactly
    assert same_picture(effect_on_top_leg(a), state_on_bottom_wire(a))
    assert not same_picture(effect_on_top_leg(a), state_on_bottom_wire(-a))

    # PyZX does not conjugate a node when it turns it from a state into an
    # effect: the plugged-in node is the row (1, e^{ia}), which is <s*|, not <s|.
    # So the picture says "ask for s*, find s" -- the blackboard's b -> b*.
    assert np.allclose((kron(bra(s.conj()), I2) @ CUP).ravel(), s)
    assert np.allclose(matrix_of(effect_on_top_leg(a)).ravel(), s)


if __name__ == "__main__":
    verdict("Marginals", blackboard, sketchbook)
