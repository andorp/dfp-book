"""Challenge 1: A Coin on a String
==================================
"Read it from the left. The end tag is where it starts."  -- Tortoise

Claim: NOT turns the state 0 into the state 1 and the state 1 into the state 0
-- X e0 = e1 and X e1 = e0 -- and NOT is not the identity.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools

import numpy as np
from dfp import I2, Picture, ket, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch01 as P

X = np.array([[0, 1], [1, 0]], dtype=complex)      # NOT
S0 = np.array([[1, 1], [0, 0]], dtype=complex)     # SET-0: "heads up, whatever it was"


def blackboard():
    # convention: the first (top) entry belongs to 0, the second to 1
    e0, e1 = ket(0), ket(1)
    assert np.allclose(e0, [1, 0]) and np.allclose(e1, [0, 1])

    # one tag on the string: "turn it over"
    assert np.allclose(X @ e0, e1)                 # heads in, tails out
    assert np.allclose(X @ e1, e0)                 # and the other way about
    assert not np.allclose(X @ e0, e0)             # it really moved
    assert not np.allclose(X, I2)                  # the tag is not a bare string

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the Tortoise's string with
# \tagstring[place;opts]{start}{tags}{caption} and a step strip of one walk
# along it with \tagwalk[opts]{start}{tags}{caption; caption; ...} (pdf/props/ch01.tex;
# the rules are in pdf/props/ch01.lua).  start is the end tag as tied; tags are
# hung left to right, "over" for "turn it over" and "heads" for "heads up,
# whatever it was"; place is how far the traveller has got.  The macro COMPUTES
# what the traveller looks like there, and with marks=on the ghost (the
# traveller one tag earlier); a step strip is one such frame per tag.  The
# dialogue never writes a turned result.  tools/props_ch01.py transcribes the
# Lua, all but the drawing; the calls are read out of this challenge's section
# of dialog.md (tools/dfp_props.py), never copied, so an edit to the dialogue
# that breaks a picture breaks here.  Every frame is checked against the model
# below, which is the blackboard's: heads the first entry, "over" NOT, "heads"
# SET-0.
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
    tags passed.  With marks, the ghost is the model's after one tag fewer, and
    there is no ghost exactly when that tag left the coin as it was."""
    assert pic.start["kind"] == "coin" and pic.expects is None and pic.lifted is None
    start, tags = FACE[pic.start["face"]], list(pic.tags)
    now = carry(start, tags[:pic.passed])
    assert pic.shown["face"] == face_of(now), pic.dump
    assert pic.marks == marks, pic.dump
    if marks:
        before = carry(start, tags[:pic.passed - 1])
        if np.allclose(before, now):
            assert pic.ghost is None and pic.dump.endswith(" unchanged"), pic.dump
        else:
            assert pic.ghost["face"] == face_of(before), pic.dump
    else:
        assert pic.ghost is None, pic.dump
    return now


def check_coin_rule():
    """The macro's coin against the model: every start, every string of up to
    four tags, every place the coin can have got to, with and without marks;
    and every step strip of such a walk, frame by frame."""
    for start in FACE:
        for n in range(5):
            for tags in itertools.product(("over", "heads"), repeat=n):
                spec = ",".join(tags)
                for place in PLACES[:n + 1] + ("end",):
                    check_coin(P.tagstring(place, start, spec), False)
                    if place in PLACES[1:]:
                        check_coin(P.tagstring(place + ";marks=on", start, spec), True)
                if n == 0:
                    assert P.faults(P.tagwalk, start, spec, "", "a")    # nothing to walk past
                    continue
                for opt, first, last in (("", 0, n), ("from=off", 1, n), ("to=end", 0, n + 1),
                                         ("from=off;to=end", 1, n + 1)):
                    k = n + 1 - first + (last > n)
                    captions = "; ".join(["a"] * k)
                    assert P.faults(P.tagwalk, start, spec, opt, captions) == []
                    frames = P.tagwalk(start, spec, opt, captions)
                    assert len(frames) == k
                    for i, f in zip(range(first, last + 1), frames):
                        assert f.picture.passed == min(i, n)
                        check_coin(f.picture, 1 <= i <= n)       # marks on every tag frame
                    assert frames[-1].picture.where == ("end" if last > n else PLACES[n])
                    # a caption short is a fault; so is one too many
                    assert P.faults(P.tagwalk, start, spec, opt, captions + "; a")
                    if k > 1:
                        assert P.faults(P.tagwalk, start, spec, opt, "; ".join(["a"] * (k - 1)))
                # marks=off: the same frames, no marks
                for f in P.tagwalk(start, spec, "marks=off", "; ".join(["a"] * (n + 1))):
                    check_coin(f.picture, False)


def pictures():
    P.check_lua()                                  # the Lua and .tex the transcription was proved on
    check_coin_rule()
    P.check_dialogue(DIALOGUE, ORDINALS)
    assert QUOTE in DIALOGUE.section("First")       # the docstring's line is spoken there

    # Two moments.  Block 1.1: the evening's first walk, a step strip of three
    # frames, keyed -- from "heads up", just past the one "turn it over" tag,
    # and arrived.  Block 1.2: the same walk from "tails up", one picture.
    (walk,), (other,) = [[fr for _, fr in b] for b in
                         P.check_section(DIALOGUE, "First", sizes=[1, 1], frames=[3, 1])]
    (c_walk,), (c_other,) = DIALOGUE.blocks("First")
    assert c_walk.macro == "tagwalk" and c_walk.keyed and c_other.macro == "tagstring"
    (start, _), (turned, _), (arrived, _) = walk
    assert [p.where for p, _ in walk] == ["start", "first", "end"]
    assert start.tags == turned.tags == arrived.tags == ("over",)
    assert np.allclose(check_coin(start, False), HEADS)            # from heads up
    assert np.allclose(check_coin(turned, True), X @ HEADS)       # turned at the tag ...
    assert turned.ghost["face"] == "heads"                       # ... from heads, the ghost
    assert np.allclose(check_coin(arrived, False), TAILS)          # carried on: tails
    # the key is on the first frame only, and the frames after it keep its depth
    assert start.keyed and start.labels == ("the end tag", "a tag") and start.deep is None
    assert not turned.keyed and not arrived.keyed and turned.labels == arrived.labels == ()
    assert turned.deep is True and arrived.deep is True
    (pic, _), = other
    assert np.allclose(check_coin(pic, False), X @ TAILS) and pic.where == "end"
    assert not np.allclose(X @ HEADS, HEADS)                      # the tag did something
    # no prediction here: Achilles is right ("Well, of course it's tails up")
    assert all(p.expects is None for p, _ in walk + other)
    # the render the card records (identity.md section 3e.2)
    assert [p.dump for p, _ in walk + other] == [
        "tagstring: start coin-heads -> coin-heads",
        "tagstring: first coin-heads -> coin-tails ghost coin-heads",
        "tagstring: end coin-heads -> coin-tails",
        "tagstring: end coin-tails -> coin-heads"]

    # where they stand: block 1.1 follows the walk, and the ghost and its hop are
    # carried by the words that describe the same motion; then the draught
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a_walk, a_other = DIALOGUE.anchors("First", path)
        assert a_walk.endswith("Tails up. Well, of course it's tails up.")
        for phrase in ("slides it rightward along the table beside the string, on their side of it",
                       "where the tag hangs he stops and turns the coin over in his fingers",
                       "it is the same coin, only the other way up",
                       "Then he carries it on.]*"):
            assert phrase in a_walk, phrase
        assert a_other.startswith("**Achilles:** *[setting the coin tails up at the left end")
    assert DIALOGUE.following("First")[0].startswith("*[Under the door the draught comes again")


def not_box():
    p = Picture()
    n = p.x(0, 2, 1)                 # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(p.inp(0), n, p.out(0))    # input boundary on the left: read left to right
    return p.done()


def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def state(bit):
    """The state box labelled 0 or 1: a one-legged node, and the stray sqrt(2) removed."""
    p = Picture()
    s = p.x(0, 1, bit)
    p.wire(s, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def state_then_not(bit):
    """A state box, then a NOT box, in a row on one wire, read left to right."""
    p = Picture()
    s = p.x(0, 1, bit)
    n = p.x(0, 3, 1)
    p.wire(s, n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def sketchbook():
    # the boxes denote exactly what the blackboard writes, scalar included
    assert np.allclose(matrix_of(not_box()), X)
    assert np.allclose(matrix_of(bare_wire()), I2)
    assert np.allclose(matrix_of(state(0)).ravel(), ket(0))
    assert np.allclose(matrix_of(state(1)).ravel(), ket(1))
    assert np.isclose(np.linalg.norm(matrix_of(state(0))), 1)
    assert np.isclose(np.linalg.norm(matrix_of(state(1))), 1)

    # picture against picture: "state 0, then NOT" IS the picture "state 1"
    assert same_picture(state_then_not(0), state(1))
    assert same_picture(state_then_not(1), state(0))
    assert not same_picture(state_then_not(0), state(0))
    assert not same_picture(state_then_not(1), state(1))

    # and the tag is not a bare string
    assert not same_picture(not_box(), bare_wire())


if __name__ == "__main__":
    verdict("Flip", blackboard, sketchbook)
