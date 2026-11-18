"""Challenge 5: A Key in the Post
===============================
"...Oh. Every key is a pair."  -- Achilles

Claim: bending a box's input wire round turns the box A into the two-wire
state (A (x) I) cup, whose four amplitudes are A's four entries; bending the
wire back with a cap, and pulling straight, gives the box again.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import BELL, CAP, CUP, I2, ket, Picture, kron, matrix_of, random_matrix, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch06 as P

D13 = ("*[The Crab sets both walnut boxes back where they stood, side by side as before, the "
       "window behind them both.]*")


def blackboard():
    rng = np.random.default_rng(65)
    A = random_matrix(rng)
    assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())   # generic
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))                # unnormalized cup
    assert np.allclose(CAP, CUP.conj().reshape(1, 4))                 # ... and cap

    # the key becomes a pair: A on the TOP leg
    psi = kron(A, I2) @ CUP
    assert np.allclose(psi.reshape(2, 2), A)          # amplitude of |ik> is the entry A_ik
    assert not np.allclose(psi.reshape(2, 2), A.T)

    # the pitfall: with A on the BOTTOM leg the amplitudes are the transpose's entries
    psi_bottom = kron(I2, A) @ CUP
    assert np.allclose(psi_bottom.reshape(2, 2), A.T)
    assert not np.allclose(psi_bottom.reshape(2, 2), A)

    # the pair becomes the key again: bend back with a cap (and pull straight)
    assert np.allclose(kron(I2, CAP) @ kron(psi, I2), A)

    # nothing is lost either way: every two-wire state is some box, bent
    phi = random_state(rng, 4)
    assert np.allclose(kron(phi.reshape(2, 2), I2) @ CUP, phi)

    # the identity box is the cup itself
    assert np.allclose(kron(I2, I2) @ CUP, CUP)

    # product states are exactly the rank-one boxes (Challenge 1's test again)
    alpha, beta = random_state(rng), random_state(rng)
    rank_one = np.outer(alpha, beta)
    assert np.isclose(np.linalg.det(rank_one), 0)
    assert np.allclose(kron(rank_one, I2) @ CUP, np.kron(alpha, beta))
    assert not np.isclose(np.linalg.det(A), 0)        # a generic box: not a product

    # The dialogue's claim about the pair "with the key already turned in one of them"
    # (Achilles: "On a lid evening it would read high-low, low-high ... never agreeing").
    # Take the key to be the bit-flip.  By the lid (0/1) the pair now never agrees;
    # by the side door (+/-) it still always agrees -- hence "on a lid evening".
    flip = np.array([[0, 1], [1, 0]], dtype=complex)
    turned = (kron(flip, I2) @ BELL).ravel()
    plus, minus = np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)
    agree = lambda basis: sum(abs(np.kron(b, b).conj() @ turned) ** 2 for b in basis)
    assert np.isclose(agree([ket(0), ket(1)]), 0.0)   # lid: high-low, low-high
    assert np.isclose(agree([plus, minus]), 1.0)      # side door: still agreeing

    pictures(A)


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the key on the string at the walnut pair with
#   \walnutstring[opts]{laid}{caption}   and its step strip
#   \walnutsteps[opts]{laid}{steps}{captions}         (pdf/props/ch06.tex)
# The Lua draws the key at one fixed spot (it "never moves"): riding the middle
# of the straight string, and on the leg that ends at the left walnut box when
# the string is a hairpin.  A step (bend=hairpin, bend=straight) is written on
# the string as it lies before it; the Lua draws the string after it, its ghost
# before, the two ends' arcs, and a ring on the key, held fast.
# tools/props_ch06.py transcribes it; the calls are read out of this
# challenge's section of dialog.md, never copied.
#
# The state model (identity card, sections 3 and 3e): the walnut pair is wires 3
# (left) and 4 (right), the left one the top wire of the pair; the hairpin's
# ends rest at the boxes and its bend lies towards the company.  The picture
# does not show which way round the key sits; the words do: "a note comes in
# along the string at the left end" (the key's input side faces the string's
# left end), and "She carries the right end across and up to the left walnut
# box, and swings the left end down, round and up to the right one".  Bent
# back: "bending the left end back the way it came and laying the right end
# back down".
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("musicboxes", "walnutstring", "beadstring", "crabledger",
                                      "boxsteps", "walnutsteps", "beadsteps", "ledgersteps"))


def string_map(s, early, keys=None):
    """A laid string as a map, read with time running away from the `early`
    side of the table: a bend on the early side is a cup, on the late side a
    cap; a free end on the early side is an input, on the late side an output;
    the legs, left to right, are the wires, top first.  `keys` puts a box A on
    a leg, {leg: (A, facing)}, its input side facing the drawn path's "start" or
    "finish" end: where time meets the box input side first the leg carries A,
    where it meets the output side first it carries A turned end for end, A^T."""
    keys = keys or {}
    late = {"far": "near", "near": "far"}[early]
    order = s.names
    heading = {n: {"up": "far", "down": "near"}[d] for n, d in s.legs}

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

    def on(leg):
        if leg not in keys:
            return I2
        A, facing = keys[leg]
        with_path = heading[leg] == late          # the path runs with time on this leg
        return A if with_path == (facing == "start") else A.T

    return layer(late, CAP) @ kron(*[on(n) for n in order]) @ layer(early, CUP)


def check_transcription():
    """The key's spot, as the Lua draws it: the same in both pictures."""
    P.check_sites()
    P.check_lua()
    straight = P.walnutstring("", "straight, key")
    hairpin = P.walnutstring("", "hairpin, key")
    assert straight.key_spot == hairpin.key_spot == (P.KEYX, P.KEYY)
    assert straight.key_leg == "line" and np.isclose(straight.key_at, 1 / 2)   # the middle
    assert hairpin.key_leg == "left box" and 0 < hairpin.key_at < 1           # before the bend
    assert np.isclose(P.BOXX["left"], P.KEYX)                                  # under the left box
    # the macro's own rule: the key rides a straight string or a hairpin, nothing else
    assert P.faults(P.walnutstring, "", "out back out, key") == [
        "walnutstring: the key rides a straight string or a hairpin"]
    assert P.faults(P.walnutstring, "", "the other way about, key") == [
        "walnutstring: the key rides a straight string or a hairpin"]
    assert P.faults(P.walnutstring, "", "hairpin, key") == []
    # the key never moves, so a step that would move it is refused: the key's hairpin is
    # not pushed about (the wander is the bare hairpin's, the Third's)
    assert P.faults(P.walnutstring, "push=out", "hairpin, key") == [
        "walnutstring: push=: the key's hairpin is not pushed about"]
    assert P.faults(P.walnutstring, "push=out", "straight") == [
        "walnutstring: push= moves a hairpin's bend"]
    assert P.faults(P.walnutstring, "push=sideways", "hairpin") == [
        "walnutstring: push=out, close or aside (not 'sideways')"]
    assert P.faults(P.walnutstring, "push=close", "hairpin close") == [
        "walnutstring: push=close: the bend lies so already"]
    # the wander, drawn by the Lua though the dialogue gives it no frame (card 3e.1, K7):
    # both ends held (a ring on each), the bend moved
    wander = P.clean(P.walnutstring, "push=out", "hairpin")
    assert [p.name for p in wander.pins] == ["end left box", "end right box"]
    assert wander.shape == "hairpin out" and [a.name for a in wander.arcs] == [
        "bend hairpin to hairpin out"]
    assert P.walnut_strand(wander.shape)[0].bends == P.walnut_strand("hairpin")[0].bends
    # a bend with the key: the key ringed, held fast; without it, no ring
    assert [p.name for p in P.clean(P.walnutstring, "bend=hairpin", "straight, key").pins] == ["key"]
    assert P.clean(P.walnutstring, "bend=hairpin", "straight").pins == []


def pictures(A):
    check_transcription()
    found = P.check_section(DIALOGUE, "Fifth", sizes=[1, 1], frames=[2, 1])
    (p51, [(w51, cap51), (w52, cap52)]), = found[0]
    (p52, [(w53, cap53)]), = found[1]
    assert (p51.macro, p52.macro) == ("walnutsteps", "walnutstring")
    assert (cap51, cap52, cap53) == ("the key at work", "a pair now", "at work again")
    for c in (p51, p52):
        assert not c.keyed and "nudged" not in P.split(c.args[0], ",")
    assert not any(w.keyed or w.nudged for w in (w51, w52, w53))

    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a1, a2 = DIALOGUE.anchors("Fifth", path)
        assert "a note comes in along the string at the left end, passes through the key, " \
               "and leaves at the right end" in a1
        assert "She carries the right end across and up to the left walnut box, and swings " \
               "the left end down, round and up to the right one" in a1
        assert "The key has not moved a hair" in a1
        assert "slips the bead off the string" in a1              # no bead in these pictures
        assert "in front of the left walnut box" in a1            # D6: where the line lies
        assert a2.endswith("The key again. Nothing lost on the way.")
    # the directions edited this round (D6, D7), in both files
    P.check_edits(DIALOGUE, "Fifth", [
        ("draws it out straight on the cloth in front of the walnut pair,",
         "draws it out straight on the cloth in front of the left walnut box,"),
        ("there are now two legs lying side by side — one running to the left walnut box with "
         "the key still upon it, the other running bare to the right.",
         "there are now two legs lying side by side."),
        (None, D13)])
    # D13: the walnut pair set back for the key, in words, before any key picture -- a
    # direction of its own between the Crab's "looked at sternly." and "May I thread it?"
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        paras = DIALOGUE.paragraphs("Fifth", path)
        i = paras.index(D13)
        assert paras[i - 1].endswith("The key does not survive being looked at sternly.")
        assert paras[i + 1].startswith("**Tortoise:** May I thread it?")
        assert i < min(j for j, q in enumerate(paras) if DIALOGUE.is_block(q))   # before F21
    # ... and every key frame (F21-F23) draws the pair where D13 leaves it: back where the
    # Third's pictures stood them, side by side, level -- not where the bead's (D12) left them
    third = {tuple((d.x, d.y) for _, d in p.drawn)
             for b in DIALOGUE.blocks("Third") for c in b for p, _ in P.pictures(c)}
    assert third == {((-P.WX, P.WY), (P.WX, P.WY))}
    bead = {tuple((d.x, d.y) for _, d in p.drawn)
            for b in DIALOGUE.blocks("Fourth") for c in b for p, _ in P.pictures(c) if p.drawn}
    assert bead == {P.BEADBOXES}
    for w in (w51, w52, w53):
        places = tuple((d.x, d.y) for _, d in w.drawn)
        assert places == ((-0.45, 1.3), (0.45, 1.3)) and {places} == third
        assert places[0][1] == places[1][1] and places[0][0] < places[1][0]   # side by side
        assert set(places).isdisjoint(P.BEADBOXES)

    # each frame the state after its step, on the frame before it: the strip bends the key's
    # straight string into the pair; the single picture bends the pair back
    assert [(s.opts, s.state) for s in P.frames_of(p51)] == \
        [("laid", "straight, key"), ("laid; bend=hairpin", "straight, key")]
    assert P.options(p52)["bend"] == "straight" and p52.args == ("hairpin, key",)
    assert P.walnut_spec(w52.shape, w52.key, w52.nudged) == p52.args[0]

    # 5.1, before: the key at work -- the straight string, the key at the middle of the line,
    # the line drawn out in front of the left walnut box (D6)
    assert w51.shape == "straight" and w51.key and w51.key_leg == "line" and not w51.marked
    assert np.isclose(w51.key_at, 1 / 2) and np.isclose(w51.key_spot[0], P.BOXX["left"])
    assert w51.key_spot[1] < P.WY - P.WD / 2 - P.WH                  # nearer the company
    assert w51.string.ends == ("left", "right") and w51.string.bends == ()
    # in at the left end (the drawn line's start), through the key, out at the right: A itself
    # 5.1, after: a pair now -- the hairpin; the key at the SAME spot, on the leg to the left box
    assert w52.shape == "hairpin" and w52.key and w52.key_spot == w51.key_spot
    assert w52.key_leg == "left box" and w52.ends_at == ("left box", "right box")
    assert w52.string.names[0] == "left box"                  # the drawn path starts at box 3,
    # where the words put the string's RIGHT end: its left end (the key's input
    # side) is the path's finish, at box 4
    assert P.WIRE[("left", "walnut")] == 3 < P.WIRE[("right", "walnut")] == 4   # the top wire
    # its marks: the straight string's ghost, the two ends' arcs (the right end across to
    # box 3, the left end down, round and up to box 4), and a ring on the key: it is held
    # fast, "The key has not moved a hair" -- and neither arc crosses it
    assert w52.step == ("bend", "hairpin")
    assert [(g.what, g.state) for g in w52.ghosts] == [("string", (w51.view, w51.shape))]
    across, round_ = w52.arcs
    assert (across.name, round_.name) == ("right end across and up to the left box",
                                          "left end down, round and up to the right box")
    (pin,) = w52.pins
    assert (pin.name, pin.x, pin.y) == ("key", *w52.key_spot) and pin.r > P.KEY_BOW_R
    a, b = P.key_shank()
    shank = [(a[0] + (b[0] - a[0]) * t / 50, a[1] + (b[1] - a[1]) * t / 50) for t in range(51)]
    for arc in (across, round_):
        bow = min(np.hypot(x - P.KEYX, y - P.KEYY) for x, y in arc.points) - P.KEY_BOW_R
        assert bow - P.SWING_W / 2 > 0.2 and P.gap_points(arc.points, shank) - P.SWING_W / 2 > 0.2

    # what the pair is, from the picture and the words (a reading, checked here):
    # the key's output side faces box 3, and the bend lies on its input side --
    # it is the key's INPUT wire that is bent round to box 4: the top-leg state.
    psi = string_map(w52.string, "near", keys={w52.key_leg: (A, "finish")}).ravel()
    assert np.allclose(psi, kron(A, I2) @ CUP)                # A on the top leg, as in 6.5
    assert np.allclose(psi.reshape(2, 2), A)                  # box 3's word first: A's entries
    assert not np.allclose(psi.reshape(2, 2), A.T)            # not the transposed one
    assert not np.allclose(psi, kron(I2, A) @ CUP)            # not A on the bottom leg
    assert not np.isclose(np.linalg.det(psi.reshape(2, 2)), 0)   # a pair, not two cards
    # had the left end gone to box 3 instead, the pair would be the transposed one
    wrong = string_map(w52.string, "near", keys={w52.key_leg: (A, "start")}).ravel()
    assert np.allclose(wrong.reshape(2, 2), A.T) and not np.allclose(wrong, psi)

    # 5.2: the key again -- bend the left end (box 4, the input) back and lay the right end
    # down: the straight string, the key at the middle exactly as before; the yank gives A
    assert "bending the left end back the way it came and laying the right end back down" \
        in DIALOGUE.section("Fifth")
    assert np.allclose(kron(I2, CAP) @ kron(psi, I2), A)
    assert w53.step == ("bend", "straight") and w53.shape == w51.shape == "straight"
    assert (w53.key_spot, w53.key_leg, w53.key_at) == (w51.key_spot, w51.key_leg, w51.key_at)
    assert w53.string == w51.string
    assert [(g.what, g.state) for g in w53.ghosts] == [("string", (w52.view, w52.shape))]
    back_round, back_down = w53.arcs                          # the same two arcs, run back
    assert back_round.name == "left end back round from the right box"
    assert back_down.name == "right end back down from the left box"
    assert back_round.pts == tuple(reversed(round_.pts))
    assert back_down.pts == tuple(reversed(across.pts))
    assert [(p.name, p.x, p.y, p.r) for p in w53.pins] == [(pin.name, pin.x, pin.y, pin.r)]
    # the bare hairpin (the Third's pair) is the identity key, bent
    assert np.allclose(string_map(w52.string, "near").ravel(), CUP)
    assert "The key again." in DIALOGUE.section("Fifth")
    # the Crab's key, the flip: the pair that never agrees by the lid
    flip = np.array([[0, 1], [1, 0]], dtype=complex)
    assert np.allclose(string_map(w52.string, "near", keys={w52.key_leg: (flip, "finish")}).ravel(),
                       kron(flip, I2) @ CUP)

    # the log the pictures write, line for line as the build writes it
    straight, pair = ("walnutstring: laid | {} | {} | key on the left leg".format(s, w)
                      for s, w in (("straight, key", "straight"), ("hairpin, key", "hairpin")))
    assert P.section_dumps(found) == [
        "walnutsteps: strip of 2 frames",
        straight, pair,
        "walnutstring: + step bend hairpin",
        "walnutstring: + ghost string laid straight",
        "walnutstring: + arc right end across and up to the left box",
        "walnutstring: + arc left end down, round and up to the right box",
        "walnutstring: + pin key",
        straight,
        "walnutstring: + step bend straight",
        "walnutstring: + ghost string laid hairpin",
        "walnutstring: + arc left end back round from the right box",
        "walnutstring: + arc right end back down from the left box",
        "walnutstring: + pin key"]


KEY = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))   # a generic box, input side first


def thread(p, row, box):
    a, b, c = box
    return [p.z(row, 3, a), p.x(row, 4, b), p.z(row, 5, c)]


def key():
    """The box alone: one wire in, one wire out."""
    p = Picture()
    p.wire(p.inp(0), *thread(p, 0, KEY), p.out(0))
    return p.done()


def key_as_pair():
    """The input wire bent round: no inputs, two outputs."""
    p = Picture()
    bend = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(bend, *thread(p, 0, KEY), p.out(0))
    p.wire(bend, p.out(1))
    return p.done()


def pair_bent_back():
    """Three rows: the pair on rows 0-1, then a cap joining row 1 to a fresh
    input on row 2.  in -- cap -- cup -- box -- out."""
    p = Picture()
    cup, cap = p.bend(0.5, 1), p.bend(1.5, 7)
    p.wire(p.inp(2), cap, cup, *thread(p, 0, KEY), p.out(0))
    return p.done()


def sketchbook():
    M = matrix_of(key())
    assert not np.allclose(M, M.T) and not np.allclose(M, M.conj())   # generic
    # the pair carries the key's four numbers, top wire first
    assert np.allclose(matrix_of(key_as_pair()).reshape(2, 2), M)
    assert not np.allclose(matrix_of(key_as_pair()).reshape(2, 2), M.T)
    # bend back and pull: the key again, exactly
    assert same_picture(pair_bent_back(), key())


if __name__ == "__main__":
    verdict("Bending", blackboard, sketchbook)
