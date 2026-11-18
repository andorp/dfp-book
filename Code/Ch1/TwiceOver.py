"""Challenge 2: There and Back
==============================
"...Oh. I needn't have walked it."  -- Achilles

Claim (the headline theorem): two NOT boxes in a row on one wire are exactly the
bare wire -- X X = I -- whatever the wire carries.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
from dfp import I2, Picture, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch01 as P

X = np.array([[0, 1], [1, 0]], dtype=complex)      # NOT
S0 = np.array([[1, 1], [0, 0]], dtype=complex)     # SET-0: "heads up, whatever it was"


def blackboard():
    # the theorem, entry by entry
    assert np.allclose(X @ X, I2)
    # Achilles's guess -- two tags do "twice as much" -- is not this
    assert not np.allclose(X @ X, X)

    rng = np.random.default_rng(1)
    # a random stochastic vector: two non-negative entries summing to one
    q = rng.random()
    p = np.array([q, 1 - q], dtype=complex)
    assert np.isclose(p.sum(), 1) and not np.isclose(q, 0.5)
    # a random complex unit vector, asserted non-real with unequal entries
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())
    assert not np.isclose(psi[0], psi[1])

    for v in (ket(0), ket(1), p, psi):
        assert np.allclose(X @ (X @ v), v)         # it arrives as it left
    # and one tag alone does move the two generic passengers
    assert not np.allclose(X @ p, p)
    assert not np.allclose(X @ psi, psi)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the Tortoise's string with
# \tagstring[place;opts]{start}{tags}{caption} and a step strip of one walk
# along it with \tagwalk[opts]{start}{tags}{caption; caption; ...} (pdf/props/ch01.tex;
# the rules are in pdf/props/ch01.lua).  The macro COMPUTES the traveller at the
# place, the ghost of the step just made, and with expects=<rule> the traveller
# a named wrong rule predicts, drawn pale.  tools/props_ch01.py transcribes the
# Lua, all but the drawing; the calls are read out of this challenge's section
# of dialog.md (tools/dfp_props.py), never copied.  Every frame is checked
# against the model below, which is the blackboard's: heads the first entry,
# "over" NOT, "heads" SET-0.  The prediction is checked against this file's own
# model of what Achilles says, and against the true frame beside it.
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
    tags passed; with marks, the ghost is the model's after one tag fewer (none
    when that tag left the coin as it was)."""
    assert pic.start["kind"] == "coin" and pic.expects is None
    start, tags = FACE[pic.start["face"]], list(pic.tags)
    now = carry(start, tags[:pic.passed])
    assert pic.shown["face"] == face_of(now) and pic.marks == marks, pic.dump
    if marks:
        before = carry(start, tags[:pic.passed - 1])
        assert (pic.ghost is None) == np.allclose(before, now), pic.dump
        assert pic.ghost is None or pic.ghost["face"] == face_of(before), pic.dump
    else:
        assert pic.ghost is None, pic.dump
    return now


def twice_as_much(start, passed):
    """Achilles's reasoning: "One tag turns the coin over.  Two tags must do twice
    as much" -- turned over, only more so: once it has been turned at all, the
    coin lies the other way up from its start, however many tags it passes."""
    return X @ start if passed > 0 else start


def check_twice_rule():
    """The named rule `twice`, on every start and every row of up to four
    turn-over tags at every place, is this file's model of what he says; and
    from the second tag on it parts from the truth."""
    for start in FACE:
        for n in range(1, 5):
            spec = ",".join(["over"] * n)
            for place in PLACES[:n + 1] + ("end",):
                pic = P.tagstring(place + ";expects=twice", start, spec)
                assert P.faults(P.tagstring, place + ";expects=twice", start, spec) == []
                assert pic.expects == "twice" and not pic.marks and pic.ghost is None
                guess = twice_as_much(FACE[start], pic.passed)
                truth = carry(FACE[start], ["over"] * pic.passed)
                assert pic.shown["face"] == face_of(guess)
                assert np.allclose(guess, truth) == (pic.passed % 2 == 1 or pic.passed == 0)
    # the rule is about turn-overs on a coin, and draws no marks
    assert P.faults(P.tagstring, "first;expects=twice", "heads", "heads")
    assert P.faults(P.tagstring, "first;expects=twice", "strip-five", "over")
    assert P.faults(P.tagstring, "first;expects=twice;marks=on", "heads", "over")


def pictures():
    P.check_lua()                                  # the Lua and .tex the transcription was proved on
    check_twice_rule()
    P.check_dialogue(DIALOGUE, ORDINALS)
    assert QUOTE in DIALOGUE.section("Second")

    # Three moments.  Block 2.1: the prediction, pale -- "Very... thoroughly
    # tails" -- beside the step strip of the first two-tag walk (no start frame:
    # the start is the frame just above), each tag marked.  Block 2.2: the other
    # start.  Block 2.3: the pair slid together, then lifted off.
    found = P.check_section(DIALOGUE, "Second", sizes=[2, 1, 2], frames=[3, 1, 2])
    (c_guess, c_walk), (c_tails,), (c_slid, c_lifted) = DIALOGUE.blocks("Second")
    assert c_guess.macro == "tagstring" and c_walk.macro == "tagwalk"
    (_, [(guess, caption)]), (_, walk) = found[0]
    (first, _), (second, _) = walk
    assert caption == "as Achilles expects"
    assert [p.where for p, _ in walk] == ["first", "second"]       # from=off: no start frame
    two = ("over", "over")
    assert guess.tags == first.tags == second.tags == two
    assert guess.start == first.start == {"kind": "coin", "face": "heads"}
    # the true walk: turned at the first (ghost heads), turned back at the second
    assert np.allclose(check_coin(first, True), X @ HEADS) and first.ghost["face"] == "heads"
    assert np.allclose(check_coin(second, True), X @ X @ HEADS) and second.ghost["face"] == "tails"
    # the prediction: the named rule, his reasoning, at the place of the true
    # frame beside it -- and not that frame
    assert guess.expects == "twice" and not guess.marks and guess.ghost is None
    assert guess.where == second.where and guess.passed == second.passed == 2
    assert guess.shown["face"] == face_of(twice_as_much(HEADS, guess.passed)) == "tails"
    assert guess.shown != second.shown
    assert not np.allclose(twice_as_much(HEADS, 2), X @ X @ HEADS)

    (_, [(from_tails, _)]), = found[1]
    (_, [(slid, _)]), (_, [(lifted, _)]) = found[2]
    for pic in (from_tails, slid, lifted):
        check_coin(pic, False)
        assert pic.where == "end" and pic.start == {"kind": "coin", "face": "tails"}
    assert from_tails.groups == (("over",), ("over",))
    assert slid.groups == (("over", "over"),) and lifted.groups == ()
    # it arrives as it left, from both starts -- and a coin has only the two
    assert np.allclose(carry(HEADS, second.tags), HEADS)
    assert np.allclose(carry(TAILS, from_tails.tags), TAILS)
    # not "twice as much": one tag alone would have turned it
    assert not np.allclose(carry(HEADS, ["over"]), HEADS)
    # slid together it is the same pair; lifted off, a bare string at the end
    # reads as its start; and the coin lies where the walk from tails left it
    assert slid.shown == lifted.shown == from_tails.shown == {"kind": "coin", "face": "tails"}
    assert [p.dump for _, fr in found[0] + found[1] + found[2] for p, _ in fr] == [
        "tagstring: second coin-heads -> coin-tails expects twice",
        "tagstring: first coin-heads -> coin-tails ghost coin-heads",
        "tagstring: second coin-heads -> coin-heads ghost coin-tails",
        "tagstring: end coin-tails -> coin-tails",
        "tagstring: end coin-tails -> coin-tails",
        "tagstring: end coin-tails -> coin-tails"]

    # where they stand: the two marked frames follow "over at the first tag, over
    # at the second"; the prediction shares their line by a bare % line end
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a_walk, a_tails, a_slid = DIALOGUE.anchors("Second", path)
        assert a_walk.startswith("**Achilles:** Very... thoroughly tails. No — I had better walk it.")
        assert "over at the first tag, over at the second" in a_walk
        assert a_walk.endswith("Tails, then heads. It arrives as it left.")
        assert a_tails.startswith("**Achilles:** *[The end tag is changed to \"tails up\"")
        assert a_slid.endswith("The string is bare but for its end tag.]*")
        assert "}%\n\\tagwalk[from=off]" in P.block_texts(DIALOGUE, "Second", path)[0]
    assert DIALOGUE.following("Second")[0].startswith("*[As he says it, the gust falls away")


def row_of_nots(k):
    """k NOT boxes in a row on one wire, read left to right."""
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    nodes = [p.x(0, 2 + 2 * i, 1) for i in range(k)]
    p.wire(p.inp(0), *nodes, p.out(0))
    return p.done()


def state_then_nots(bit, k):
    """The state box `bit`, then k NOT boxes, in a row."""
    p = Picture()
    s = p.x(0, 1, bit)
    nodes = [p.x(0, 3 + 2 * i, 1) for i in range(k)]
    p.wire(s, *nodes, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def sketchbook():
    bare = row_of_nots(0)                          # the empty row: a bare wire
    assert np.allclose(matrix_of(bare), I2)

    # (i) the equation, exactly, scalar included
    assert same_picture(row_of_nots(2), bare)
    assert not same_picture(row_of_nots(1), bare)
    assert same_picture(row_of_nots(3), row_of_nots(1))

    # (ii) the sketchbook proof of the chapter: run both state boxes through the row
    for bit in (0, 1):
        assert same_picture(state_then_nots(bit, 2), state_then_nots(bit, 0))
        assert not same_picture(state_then_nots(bit, 1), state_then_nots(bit, 0))

    # the principle that closes that proof -- a box is fixed by what it does to the
    # states 0 and 1 -- as matrices: those two pictures are the row's two columns
    two = matrix_of(row_of_nots(2))
    for bit in (0, 1):
        assert np.allclose(two[:, bit], matrix_of(state_then_nots(bit, 2)).ravel())
    assert np.allclose(two, I2)


if __name__ == "__main__":
    verdict("TwiceOver", blackboard, sketchbook)
