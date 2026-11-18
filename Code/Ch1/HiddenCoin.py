"""Challenge 4: The Covered Coin
=================================
"My strip couldn't show it: turned end for end, half-and-half is half-and-half."  -- Achilles

Claim: a start may be a division of one whole into a heads share and a tails
share; NOT exchanges the two shares and keeps them summing to one; and the fair
half-and-half start is left where it is by NOT, although NOT is not the bare
wire -- one start is not enough to tell a tag from no tag.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
from dfp import I2, Picture, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch01 as P

X = np.array([[0, 1], [1, 0]], dtype=complex)      # NOT
S0 = np.array([[1, 1], [0, 0]], dtype=complex)     # SET-0
S1 = np.array([[0, 0], [1, 1]], dtype=complex)     # SET-1

# convention: the first entry is the heads (0) share, the second the tails share
FIVE_ONE = np.array([5 / 6, 1 / 6], dtype=complex)   # Tortoise's strip: the die
FAIR = np.array([1 / 2, 1 / 2], dtype=complex)       # Achilles's strip: a tossed coin


def blackboard():
    assert np.isclose(FIVE_ONE.sum(), 1) and np.all(FIVE_ONE.real >= 0)

    # the strip turned end for end: the division moves from five parts out to one
    assert np.allclose(X @ FIVE_ONE, [1 / 6, 5 / 6])
    assert np.isclose((X @ FIVE_ONE).sum(), 1)          # still one strip
    assert np.allclose(X @ (X @ FIVE_ONE), FIVE_ONE)    # and back: home

    # a random division, seeded; kept away from half-and-half so the move shows
    rng = np.random.default_rng(4)
    q = rng.uniform(0.05, 0.45)
    p = np.array([q, 1 - q], dtype=complex)
    assert np.isclose(p.sum(), 1) and not np.isclose(q, 0.5)
    assert np.allclose(X @ p, p[::-1]) and not np.allclose(X @ p, p)
    assert np.isclose((X @ p).sum(), 1)
    assert np.allclose(X @ (X @ p), p)

    # every one-bit box is column-stochastic, which is exactly why the two shares
    # still make up the whole after the walk
    for M in (I2, X, S0, S1):
        assert np.allclose(M.imag, 0) and np.all(M.real >= 0)
        assert np.allclose(M.sum(axis=0), [1, 1])
        for v in (FIVE_ONE, FAIR, p):
            assert np.isclose((M @ v).sum(), 1)

    # Achilles's trap: half-and-half reads the same from either end ...
    assert np.allclose(X @ FAIR, FAIR)
    # ... and yet the tag is at work -- it is not the bare wire
    assert not np.allclose(X, I2)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the Tortoise's string with
# \tagstring[place;opts]{start}{tags}{caption} and a step strip of one walk
# along it with \tagwalk[opts]{start}{tags}{caption; caption; ...} (pdf/props/ch01.tex;
# the rules are in pdf/props/ch01.lua).  Here the traveller is the strip: the
# macro COMPUTES which end of it is at the string and so heads' share, the ghost
# of the step just made, and with expects=idle the strip Achilles expects,
# drawn pale.  tools/props_ch01.py transcribes the Lua, all but the drawing; the
# calls are read out of this challenge's section of dialog.md
# (tools/dfp_props.py), never copied.  Every frame is checked against the model
# below.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("tagstring", "tagwalk", "arrowsquare"))
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth")
QUOTE = __doc__.split('"')[1]
PLACES = ("start", "first", "second", "third", "fourth")

# --- the strip, in the chapter's state model ---------------------------------
# strip-k is a strip of six panels shaded k from one end, tied on by that dark
# end: the division (k/6, (6-k)/6), heads' share first -- from the knot out to the
# change of shade.  "turn it over" is NOT, the matrix blackboard() uses; on the
# strip it is the turn end for end: the other end is held to the string, and
# heads' share h becomes six minus h.  So the strip as a thing is two facts --
# how many panels are shaded, and which end is held -- and its division is
# what they make together.


def strip_start(dark):
    return np.array([dark / 6, (6 - dark) / 6], dtype=complex)


def carry(v, tags):
    """The tags met, in the order met: the string read left to right."""
    for t in tags:
        assert t == "over"
        v = X @ v
    return v


def held_end(tags):
    """Which end of the strip is held to the string after these tags: each turn
    end for end brings the other end to the knot."""
    return "white" if len(tags) % 2 else "dark"


def share_by_eye(panels):
    """Heads' share read off the drawing: panels from the knot to the change of
    shade (for a strip that has a change)."""
    k = 0
    while k < 6 and panels[k] == panels[0]:
        k += 1
    return k


def check_strip(pic, marks):
    """A strip picture against the model: heads' share drawn is six times the
    model's heads entry after the tags passed, and the end at the string is the
    model's; with marks the ghost is the strip one tag earlier, there exactly when
    that tag moved the strip -- and every turn end for end moves it, even when
    the division reads the same."""
    assert pic.start["kind"] == "strip" and pic.expects is None
    dark, tags = pic.start["dark"], list(pic.tags)
    v = carry(strip_start(dark), tags[:pic.passed])
    end = held_end(tags[:pic.passed])
    assert np.isclose(6 * v[0], P.heads_share(pic.shown)), pic.dump
    assert np.isclose(v.sum(), 1)                            # one strip, one length
    assert pic.shown["turned"] == (end == "white"), pic.dump
    panels = P.strip_panels(dark, pic.shown["turned"])
    assert sum(panels) == dark                               # turning shades nothing
    if 0 < dark < 6:
        assert share_by_eye(panels) == P.heads_share(pic.shown)   # as drawn
    assert pic.marks == marks, pic.dump
    if marks:
        before = tags[:pic.passed - 1]
        assert held_end(before) != end and pic.ghost is not None, pic.dump
        assert pic.ghost["turned"] == (held_end(before) == "white"), pic.dump
        assert np.isclose(6 * carry(strip_start(dark), before)[0], P.heads_share(pic.ghost))
    else:
        assert pic.ghost is None, pic.dump
    return v, end


def check_strip_rule():
    """The macro's strip against the model: every strip from none to six shaded
    panels, up to four turn-over tags, every place, with and without marks; every
    step strip of such a walk; and the rule `idle` against this file's model of
    what Achilles says ("for a coin nobody has seen, this tag does nothing")."""
    for w, k in P.WORDNUM.items():                        # strip-five is strip-5
        assert P.parse_start("strip-" + w) == P.parse_start("strip-%d" % k)
    for dark in range(7):
        for n in range(5):
            tags = ["over"] * n
            spec = ",".join(tags)
            for place in PLACES[:n + 1] + ("end",):
                check_strip(P.tagstring(place, "strip-%d" % dark, spec), False)
                if place in PLACES[1:]:
                    check_strip(P.tagstring(place + ";marks=on", "strip-%d" % dark, spec), True)
                opt = place + ";expects=idle"
                assert P.faults(P.tagstring, opt, "strip-%d" % dark, spec) == []
                guess = P.tagstring(opt, "strip-%d" % dark, spec)
                assert guess.expects == "idle" and not guess.marks and guess.ghost is None
                assert guess.shown == {"kind": "strip", "dark": dark, "turned": False}  # as tied
                assert np.isclose(6 * (I2 @ strip_start(dark))[0], P.heads_share(guess.shown))
                truth = P.tagstring(place, "strip-%d" % dark, spec)
                assert (guess.shown == truth.shown) == (truth.passed % 2 == 0)
            if n:
                captions = "; ".join(["a"] * (n + 1))
                frames = P.tagwalk("strip-%d" % dark, spec, "", captions)
                assert P.faults(P.tagwalk, "strip-%d" % dark, spec, "", captions) == []
                assert [f.picture.passed for f in frames] == list(range(n + 1))
                for i, f in enumerate(frames):
                    check_strip(f.picture, i > 0)
                # a strip walk is never keyed: key the strip alone first
                assert P.faults(P.tagwalk, "strip-%d" % dark, spec, "", captions, True)
    # the rule is for a strip; the strip takes no "heads up" tag; a keyed
    # strip takes its labels from the strip drawn
    assert P.faults(P.tagstring, "first;expects=idle", "heads", "over")
    assert P.faults(P.tagstring, "first", "strip-five", "heads")
    for dark in range(7):
        pic = P.tagstring("start", "strip-%d" % dark, "", True)
        assert pic.labels == tuple(l for l, there in (("heads' share", dark > 0),
                                                     ("tails' share", dark < 6)) if there)


def pictures():
    P.check_lua()                                  # the Lua and .tex the transcription was proved on
    check_strip_rule()
    P.check_dialogue(DIALOGUE, ORDINALS)
    assert QUOTE in DIALOGUE.section("Fourth")
    assert np.allclose(strip_start(3), FAIR) and np.allclose(strip_start(5), FIVE_ONE)

    # Four moments.  Block 4.1, the keyed strip picture: Achilles's tossed coin
    # tied on, the string still bare.  Block 4.2: carried to the tag and turned
    # there, marked (its start is block 4.1, a few speeches up).  Block 4.3: the
    # prediction, pale -- "this tag does nothing" -- beside the step strip of the
    # Tortoise's strip: tied on, then turned at the tag.  Block 4.4: tied on
    # again, the pair lifted off.
    found = P.check_section(DIALOGUE, "Fourth", sizes=[1, 1, 2, 1], frames=[1, 1, 3, 1])
    (c_keyed,), _, (c_idle, c_walk), _ = DIALOGUE.blocks("Fourth")
    assert c_keyed.macro == "tagstring" and c_keyed.keyed
    assert c_idle.macro == "tagstring" and c_walk.macro == "tagwalk" and not c_walk.keyed
    (_, [(tossed_pic, _)]), = found[0]
    (_, [(turned_pic, _)]), = found[1]
    (_, [(idle, caption)]), (_, walk) = found[2]
    (_, [(home_pic, _)]), = found[3]
    (die_pic, _), (die_turned_pic, _) = walk
    assert caption == "as Achilles expects"

    # every strip as tied: by its dark end
    for pic in (tossed_pic, turned_pic, idle, die_pic, die_turned_pic, home_pic):
        assert pic.start["kind"] == "strip" and not pic.start["turned"]
    assert tossed_pic.start["dark"] == turned_pic.start["dark"] == 3
    assert idle.start["dark"] == die_pic.start["dark"] == home_pic.start["dark"] == 5

    tossed, _ = check_strip(tossed_pic, False)
    assert tossed_pic.tags == () and tossed_pic.where == "start"
    assert tossed_pic.labels == ("heads' share", "tails' share")   # the key
    tossed_turned, end = check_strip(turned_pic, True)
    assert turned_pic.where == "first" and turned_pic.tags == ("over",) and end == "white"
    # the trap: half-and-half turned end for end reads as it did -- and yet the
    # ghost is drawn, because the strip did move: its other end is at the string
    assert np.allclose(tossed_turned, tossed) and np.allclose(X @ FAIR, FAIR)
    assert turned_pic.ghost == {"kind": "strip", "dark": 3, "turned": False}

    die, end0 = check_strip(die_pic, False)
    die_turned, end1 = check_strip(die_turned_pic, True)
    assert [p.where for p, _ in walk] == ["start", "first"] and (end0, end1) == ("dark", "white")
    assert die_turned_pic.ghost == die_pic.shown                 # the ghost is the strip as tied
    # "It moved!": five-and-one does not read as it did
    assert not np.allclose(die_turned, die) and np.allclose(die_turned, X @ FIVE_ONE)
    # the shading only marks the place: dark does not mean heads
    turned_panels = P.strip_panels(5, die_turned_pic.shown["turned"])
    assert sum(turned_panels) == 5 and not np.isclose(6 * die_turned[0], 5)
    # the prediction: the named rule, his reasoning ("this tag does nothing": the
    # bare wire), at the place of the true frame beside it -- and not that frame
    assert idle.expects == "idle" and not idle.marks and idle.ghost is None
    assert idle.where == die_turned_pic.where and idle.passed == die_turned_pic.passed == 1
    assert idle.tags == die_turned_pic.tags == ("over",)
    assert np.isclose(6 * (I2 @ FIVE_ONE)[0], P.heads_share(idle.shown)) and not idle.shown["turned"]
    assert idle.shown != die_turned_pic.shown
    assert not np.allclose(I2 @ FIVE_ONE, X @ FIVE_ONE)

    home, _ = check_strip(home_pic, False)
    assert home_pic.tags == () and home_pic.where == "start"
    # the pair lifted off, and the strip is as it began: the round trip
    assert np.allclose(home, die) and np.allclose(carry(die, ["over", "over"]), die)
    assert [p.dump for b in found for _, fr in b for p, _ in fr] == [
        "tagstring: start strip-3-dark -> strip-3-dark",
        "tagstring: first strip-3-dark -> strip-3-white ghost strip-3-dark",
        "tagstring: first strip-5-dark -> strip-5-dark expects idle",
        "tagstring: start strip-5-dark -> strip-5-dark",
        "tagstring: first strip-5-dark -> strip-1-white ghost strip-5-dark",
        "tagstring: start strip-5-dark -> strip-5-dark"]
    # a keyed strip picture carries one tag at most
    for o in ORDINALS:
        for b in DIALOGUE.blocks(o):
            for c in b:
                if c.keyed and P.kind(c) == "strip":
                    assert c.macro == "tagstring" and len(P.picture(c).tags) <= 1, c.text

    # where they stand
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        paras = DIALOGUE.paragraphs("Fourth", path)
        a_keyed, a_turned, a_walk, a_home = DIALOGUE.anchors("Fourth", path)
        assert a_keyed.endswith("A tossed coin.")
        # block 4.2's ghost and hop: carried along the string as far as the tag;
        # which end is held moved, the change of shade did not
        assert ("What has moved is which end is held to the string; what has not moved "
                "is the change of shade") in a_turned
        carried = [i for i, q in enumerate(paras)
                   if "carrying it along the string as far as the tag" in q]
        assert len(carried) == 1 and carried[0] < paras.index(a_turned)
        # block 4.3: the Tortoise's strip tied on and walked; the frame that once
        # stood after "That's your die." is folded into its start frame
        assert ("He unties the strip, carries it along to the tag, turns it end for end, "
                "and holds its white end to the string.") in a_walk
        assert "The strip is the same length it was." in a_walk
        assert a_walk.endswith("And it is all one sheet of paper.")
        assert not any("That's your die." in a for a in (a_keyed, a_turned, a_walk, a_home))
        assert a_home.endswith("as she first tied it.]*")
        assert "}%\n\\tagwalk{strip-five}" in P.block_texts(DIALOGUE, "Fourth", path)[2]


def fair_coin():
    """The fair start IS a picture: one one-legged node, its two stray sqrt(2)s removed."""
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    n = p.z(0, 1, 0)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-2)
    return g


def fair_then_nots(k):
    """The fair start, then k NOT boxes, in a row on one wire."""
    p = Picture()
    n = p.z(0, 1, 0)
    nodes = [p.x(0, 3 + 2 * i, 1) for i in range(k)]
    p.wire(n, *nodes, p.out(0))
    g = p.done()
    g.scalar.add_power(-2)
    return g


def not_box():
    p = Picture()
    n = p.x(0, 2, 1)
    p.wire(p.inp(0), n, p.out(0))
    return p.done()


def set0_box():
    p = Picture()
    p.wire(p.inp(0), p.z(0, 2, 0))
    p.wire(p.x(0, 3, 0), p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def two_not_boxes():
    p = Picture()
    a, b = p.x(0, 2, 1), p.x(0, 4, 1)
    p.wire(p.inp(0), a, b, p.out(0))
    return p.done()


def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def sketchbook():
    assert np.allclose(matrix_of(fair_coin()).ravel(), FAIR)

    # the trap, as pictures: the fair start is where it was, and yet the tag is
    # not the bare wire.  Exact, scalar included.
    assert same_picture(fair_then_nots(1), fair_coin())
    assert not same_picture(not_box(), bare_wire())
    assert same_picture(fair_then_nots(2), fair_coin())

    # A general division such as five-and-one has no exact picture built from these
    # nodes, so for it we check the boxes instead: what the pictures denote is
    # column-stochastic, and acts on the strip as claimed.
    xm, s0m = matrix_of(not_box()), matrix_of(set0_box())
    for M in (xm, s0m):
        assert np.allclose(M.imag, 0) and np.all(M.real >= -1e-12)
        assert np.allclose(M.sum(axis=0), [1, 1])
    rng = np.random.default_rng(44)
    q = rng.uniform(0.05, 0.45)
    p = np.array([q, 1 - q], dtype=complex)
    for v in (FIVE_ONE, p):
        assert np.allclose(xm @ v, v[::-1]) and not np.allclose(xm @ v, v)
        assert np.isclose((xm @ v).sum(), 1)
        assert np.allclose(matrix_of(two_not_boxes()) @ v, v)


if __name__ == "__main__":
    verdict("HiddenCoin", blackboard, sketchbook)
