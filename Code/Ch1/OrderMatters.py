"""Challenge 3: Which Tag First?
=================================
"Same tags. Walk it again."  -- Tortoise

Claim: a row of boxes read left to right is the matrix product read right to
left, and the order matters: NOT then SET-0 is SET-0, while SET-0 then NOT is
SET-1.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools

import numpy as np
from dfp import I2, Picture, ket, matrix_of, random_matrix, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch01 as P

X = np.array([[0, 1], [1, 0]], dtype=complex)      # NOT
S0 = np.array([[1, 1], [0, 0]], dtype=complex)     # SET-0: "heads up, whatever it was"
S1 = np.array([[0, 0], [1, 1]], dtype=complex)     # SET-1
FOUR = {"I": I2, "X": X, "S0": S0, "S1": S1}


def blackboard():
    # a row "A then B" is the product B A -- the picture left to right, the product
    # right to left.  With two different boxes the convention becomes visible:
    assert np.allclose(S0 @ X, S0)                 # NOT then SET-0
    assert np.allclose(X @ S0, S1)                 # SET-0 then NOT
    assert not np.allclose(S0 @ X, X @ S0)         # same tags, other face

    # the same coin, the same start, the two tags in the two orders
    assert np.allclose(S0 @ X @ ket(0), ket(0))
    assert np.allclose(X @ S0 @ ket(0), ket(1))

    # one of the twins comes off, and never both: SET-0 twice does what it does once
    assert np.allclose(S0 @ S0, S0)
    assert not np.allclose(S0 @ S0, I2)
    # after SET-0 nothing on the string can tell how the coin started ...
    assert np.allclose(S0 @ ket(0), S0 @ ket(1))
    assert np.allclose(X @ S0 @ ket(0), X @ S0 @ ket(1))   # "both starts arrive alike"
    # ... and nothing hung further along would bring it back
    assert np.isclose(np.linalg.det(S0), 0)
    rng = np.random.default_rng(3)
    B = random_matrix(rng)
    assert np.allclose((B @ S0)[:, 0], (B @ S0)[:, 1])     # the two columns collapse
    assert not np.allclose(B @ S0, I2)
    # while the turn-over tag never loses anything
    assert not np.allclose(X @ ket(0), X @ ket(1))
    assert not np.isclose(np.linalg.det(X), 0)

    # the four one-bit boxes are closed under rows: all sixteen products
    for A in FOUR.values():
        for C in FOUR.values():
            assert any(np.allclose(C @ A, D) for D in FOUR.values())

    # left to right versus right to left, on a generic non-commuting pair
    A, C = random_matrix(rng), random_matrix(rng)
    assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())
    assert not np.allclose(A @ C, C @ A)
    v = random_state(rng)
    assert np.allclose(C @ (A @ v), (C @ A) @ v)           # first A, then C
    assert not np.allclose(C @ (A @ v), (A @ C) @ v)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the Tortoise's string with
# \tagstring[place;opts]{start}{tags}{caption} and a step strip of one walk
# along it with \tagwalk[opts]{start}{tags}{caption; caption; ...} (pdf/props/ch01.tex;
# the rules are in pdf/props/ch01.lua).  The macro COMPUTES the traveller at the
# place, the ghost of the step just made (none where the tag changed nothing),
# and with expects=<rule> the traveller a named wrong rule predicts, drawn pale.
# tools/props_ch01.py transcribes the Lua, all but the drawing; the calls are
# read out of this challenge's section of dialog.md (tools/dfp_props.py), never
# copied.  Every frame is checked against the model below, which is the
# blackboard's: heads the first entry, "over" NOT, "heads" SET-0.  The two
# predictions are checked against this file's own models of what Achilles
# says, and against the true frames beside them.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("tagstring", "tagwalk", "arrowsquare"))
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth")
QUOTE = __doc__.split('"')[1]

HEADS = np.array([1, 0], dtype=complex)
TAILS = np.array([0, 1], dtype=complex)
FACE = {"heads": HEADS, "tails": TAILS}
TAG = {"over": X, "heads": S0}
PLACES = ("start", "first", "second", "third", "fourth")


def carry(v, tags):
    """The tags met, in the order met: the string read left to right."""
    for t in tags:
        v = TAG[t] @ v
    return v


def face_of(v):
    names = [f for f, u in FACE.items() if np.allclose(v, u)]
    assert len(names) == 1, v
    return names[0]


def check_coin(pic, marks):
    """A coin picture against the model: the face drawn is the model's after the
    tags passed; with marks, the ghost is the model's after one tag fewer, and
    there is none exactly when that tag left the coin as it was."""
    assert pic.start["kind"] == "coin" and pic.expects is None
    start, tags = FACE[pic.start["face"]], list(pic.tags)
    now = carry(start, tags[:pic.passed])
    assert pic.shown["face"] == face_of(now) and pic.marks == marks, pic.dump
    if marks:
        before = carry(start, tags[:pic.passed - 1])
        assert (pic.ghost is None) == np.allclose(before, now), pic.dump
        assert (pic.ghost is None) == pic.dump.endswith(" unchanged"), pic.dump
        assert pic.ghost is None or pic.ghost["face"] == face_of(before), pic.dump
    else:
        assert pic.ghost is None, pic.dump
    return now


def lift_twins(tags):
    """Achilles: "Twins. Off they come." -- as the turn-over pair came off.  The
    Tortoise's way of lifting: find the leftmost two of the same tag side by side,
    lift them, and look again (a pair gone may bring two more side by side).
    Returns the pairs lifted (1-based places on the string as hung)."""
    hanging, pairs = list(range(1, len(tags) + 1)), []
    while True:
        for a, b in zip(hanging, hanging[1:]):
            if tags[a - 1] == tags[b - 1]:
                pairs.append((a, b))
                hanging.remove(a)
                hanging.remove(b)
                break
        else:
            return pairs


def twins_expected(start, tags, passed):
    """Where he expects the coin, just past the place-th tag as hung: it has walked
    only the tags still hanging before that place.  None if the place falls
    between the two tags of a pair that comes off (the rule cannot say)."""
    pairs = lift_twins(tags)
    if any(a <= passed < b for a, b in pairs):
        return None, pairs
    gone = {i for p in pairs for i in p}
    return carry(start, [t for i, t in enumerate(tags[:passed], 1) if i not in gone]), pairs


def any_order(start, tags, passed):
    """Achilles: "the order along the string can't matter" -- so walking the same
    tags in the other order (the string read from its right-hand end) does the
    same."""
    return carry(start, list(tags)[::-1][:passed])


def check_coin_rule():
    """The macro's coin against the model, and the rules `twins` and `anyorder`
    against this file's models of what he says: every start, every string of up
    to four tags, every place."""
    differ = 0
    for start in FACE:
        for n in range(5):
            for tags in itertools.product(("over", "heads"), repeat=n):
                spec = ",".join(tags)
                for place in PLACES[:n + 1] + ("end",):
                    check_coin(P.tagstring(place, start, spec), False)
                    if place in PLACES[1:]:
                        check_coin(P.tagstring(place + ";marks=on", start, spec), True)
                    opt = place + ";expects=twins"
                    errors = P.faults(P.tagstring, opt, start, spec)
                    pic = P.tagstring(opt, start, spec)
                    guess, pairs = twins_expected(FACE[start], tags, pic.passed)
                    gone = {i for p in pairs for i in p}
                    assert pic.lifted == gone and pic.hanging == tuple(
                        i for i in range(1, n + 1) if i not in gone)
                    if guess is None:
                        inside = sorted((p for p in pairs if p[0] <= pic.passed < p[1]),
                                        key=lambda p: p[1])     # one fault per pair, as closed
                        assert errors == ["tagstring: expects=twins: the place falls inside a "
                                          "pair that comes off (past tag %d, not past tag %d)" % p
                                          for p in inside]
                    else:
                        assert errors == [] and pic.shown["face"] == face_of(guess)
                    if n >= 2:
                        opt = place + ";expects=anyorder"
                        assert P.faults(P.tagstring, opt, start, spec) == []
                        pic = P.tagstring(opt, start, spec)
                        assert pic.expects == "anyorder" and pic.lifted is None
                        assert pic.shown["face"] == face_of(any_order(FACE[start], tags, pic.passed))
                        if place == "end":
                            truth = P.tagstring("end", start, spec).shown
                            differ += pic.shown != truth
                    else:
                        assert P.faults(P.tagstring, place + ";expects=anyorder", start, spec)
    # the string read from the right parts from the string read from the left on
    # 24 of the 56 walks (the 28 strings of two to four tags, from either start):
    # the order is part of what the string says
    assert differ == 24, differ


def pictures():
    P.check_lua()                                  # the Lua and .tex the transcription was proved on
    check_coin_rule()
    P.check_dialogue(DIALOGUE, ORDINALS)
    assert QUOTE in DIALOGUE.section("Third")

    # Four moments.  Block 3.1: the prediction, pale -- the twins lifted off --
    # beside the step strip of the first walk past "heads up, whatever it was"
    # tags.  Block 3.2: "turn it over" then "heads up", as hung.  Block 3.3: the
    # same two tags swapped, the prediction "the order can't matter" beside the
    # true frame.  Block 3.4: swapped, from tails.
    found = P.check_section(DIALOGUE, "Third", sizes=[2, 1, 2, 1], frames=[3, 1, 2, 1])
    (c_twins, c_walk), _, (c_any, c_swapped), _ = DIALOGUE.blocks("Third")
    assert c_twins.macro == "tagstring" and c_walk.macro == "tagwalk"
    (_, [(twins, caption)]), (_, walk) = found[0]
    (first, _), (second, _) = walk
    assert caption == "as Achilles expects"
    assert [p.where for p, _ in walk] == ["first", "second"]
    assert twins.tags == first.tags == second.tags == ("heads", "heads")
    assert twins.start == first.start == {"kind": "coin", "face": "tails"}

    # the true walk: laid heads up at the first (ghost tails); at the second it
    # is heads up already -- the tag changes nothing, and no marks are drawn
    p1 = check_coin(second, True)
    assert np.allclose(check_coin(first, True), S0 @ TAILS) and first.ghost["face"] == "tails"
    assert second.marks and second.ghost is None and np.allclose(p1, S0 @ S0 @ TAILS)
    assert np.allclose(S0 @ (S0 @ TAILS), S0 @ TAILS)                # the step did nothing
    # the prediction: both twins lifted, and nothing else -- the string bare where
    # they hung -- and the coin, read at the same place, arrives tails, "along a
    # bare string"; the true frame beside it is heads
    assert twins.expects == "twins" and not twins.marks
    assert twins.lifted == {1, 2} and twins.hanging == ()
    assert twins.where == second.where and twins.passed == second.passed == 2
    guess, pairs = twins_expected(TAILS, twins.tags, twins.passed)
    assert pairs == [(1, 2)] and np.allclose(guess, TAILS) and np.allclose(guess, carry(TAILS, []))
    assert twins.shown["face"] == face_of(guess) == "tails" != second.shown["face"]
    # the place is read on the string as hung; one that falls inside the pair is a fault
    assert P.faults(P.tagstring, "first;expects=twins", "tails", "heads,heads") == [
        "tagstring: expects=twins: the place falls inside a pair that comes off "
        "(past tag 1, not past tag 2)"]
    assert P.faults(P.tagstring, "end;expects=twins", "tails", "heads,heads") == []
    # "along a bare string it would have arrived tails!": the twins can't come off,
    # and one of them lifted, the other does what the pair did
    assert not np.allclose(p1, TAILS) and np.allclose(carry(TAILS, ["heads"]), p1)

    (_, [(as_hung, _)]), = found[1]
    (_, [(anyorder, caption)]), (_, [(swapped, _)]) = found[2]
    (_, [(from_tails, _)]), = found[3]
    assert caption == "as Achilles expects"
    p2, p3, p4 = (check_coin(p, False) for p in (as_hung, swapped, from_tails))
    assert as_hung.tags == ("over", "heads") and swapped.tags == from_tails.tags == ("heads", "over")
    assert as_hung.start == swapped.start == {"kind": "coin", "face": "heads"}
    assert from_tails.start == {"kind": "coin", "face": "tails"}
    # same tags, same start, the other face
    assert sorted(as_hung.tags) == sorted(swapped.tags) and as_hung.tags == swapped.tags[::-1]
    assert np.allclose(p2, HEADS) and np.allclose(p3, TAILS)
    assert np.allclose(p2, S0 @ X @ HEADS) and np.allclose(p3, X @ S0 @ HEADS)
    # "Both starts arrive alike!"
    assert np.allclose(p3, p4)
    # the prediction: the order can't matter -- the swapped tags walked in the
    # order they were first hung, which is block 3.2's state; the true frame
    # beside it, at the same place, is the other face.  The mix-up the
    # convention rules out is exactly this: the string read from the right.
    assert anyorder.expects == "anyorder" and not anyorder.marks
    assert anyorder.where == swapped.where == "end" and anyorder.tags == swapped.tags
    guess = any_order(HEADS, swapped.tags, anyorder.passed)
    assert anyorder.shown["face"] == face_of(guess) == face_of(p2) == "heads"
    assert anyorder.shown != swapped.shown
    assert [p.dump for b in found for _, fr in b for p, _ in fr] == [
        "tagstring: second coin-tails -> coin-tails expects twins lifted 1,2",
        "tagstring: first coin-tails -> coin-heads ghost coin-tails",
        "tagstring: second coin-tails -> coin-heads unchanged",
        "tagstring: end coin-heads -> coin-heads",
        "tagstring: end coin-heads -> coin-heads expects anyorder",
        "tagstring: end coin-heads -> coin-tails",
        "tagstring: end coin-tails -> coin-tails"]

    # where they stand: the unmarked second frame follows "and he leaves it"
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a_twins, a_hung, a_swapped, a_tails = DIALOGUE.anchors("Third", path)
        assert "at the second it is heads up already, and he leaves it" in a_twins
        assert a_twins.endswith("The pair can't come off.")
        assert a_hung.endswith("It arrives heads.")
        assert a_swapped.endswith("one order leaves you dry, the other wet.")
        assert a_tails.startswith("**Achilles:** *[with \"tails up\" tied on")
        assert "}%\n\\tagwalk[from=off]" in P.block_texts(DIALOGUE, "Third", path)[0]


def row_of(*tags):
    """A row of boxes on one wire, read left to right.  Tags: "X", "S0", "S1".

    NOT is one two-legged node of phase pi.  SET-0 and SET-1 are drawn in the book
    as ordinary labelled boxes; here each is a one-legged node closing the incoming
    wire and a one-legged node opening the outgoing one, which together denote
    exactly [[1,1],[0,0]] and [[0,0],[1,1]].
    """
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    prev, col, power = p.inp(0), 2, 0
    for tag in tags:
        if tag == "X":
            node = p.x(0, col, 1)
            p.wire(prev, node)
            prev = node
        else:
            p.wire(prev, p.z(0, col, 0))
            prev = p.x(0, col + 1, 0 if tag == "S0" else 1)
            power -= 1
        col += 2
    p.wire(prev, p.out(0))
    g = p.done()
    g.scalar.add_power(power)
    return g


def sketchbook():
    # the boxes denote exactly what the blackboard writes
    assert np.allclose(matrix_of(row_of("X")), X)
    assert np.allclose(matrix_of(row_of("S0")), S0)
    assert np.allclose(matrix_of(row_of("S1")), S1)
    assert np.allclose(matrix_of(row_of()), I2)

    # picture against picture: the order along the wire is part of what it says
    assert same_picture(row_of("X", "S0"), row_of("S0"))
    assert same_picture(row_of("S0", "X"), row_of("S1"))
    assert not same_picture(row_of("X", "S0"), row_of("S0", "X"))

    # one of the twins comes off, and never both
    assert same_picture(row_of("S0", "S0"), row_of("S0"))
    assert not same_picture(row_of("S0", "S0"), row_of())

    # a row read left to right is the product read right to left
    for tags in (("X", "S0"), ("S0", "X"), ("S0", "S0"), ("X", "X")):
        first, then = (matrix_of(row_of(t)) for t in tags)
        assert np.allclose(matrix_of(row_of(*tags)), then @ first)


if __name__ == "__main__":
    verdict("OrderMatters", blackboard, sketchbook)
