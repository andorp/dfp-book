"""Challenge 3: The String
========================
"Take that end. I have this one. Pull."  -- Tortoise

Claim: a wire bent out, back and out again is a straight wire, whichever way
it is bent, with no stray number: (cap (x) I)(I (x) cup) = I = (I (x) cap)(cup (x) I).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
from dfp import BELL, CAP, CUP, I2, Picture, ket, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch06 as P


def blackboard():
    # conventions: cup = |00> + |11> and cap = <00| + <11|, both UNNORMALIZED
    assert np.allclose(CUP, ket(0, 0) + ket(1, 1))
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))
    assert np.allclose(CAP, CUP.conj().reshape(1, 4))       # the cap is the cup's bra
    assert np.allclose(BELL, CUP / np.sqrt(2))

    # the Z-bend: in on the top row, cup below, cap above, out on the bottom row
    assert np.allclose(kron(CAP, I2) @ kron(I2, CUP), I2)
    # the S-bend: the same with the rows in the other order
    assert np.allclose(kron(I2, CAP) @ kron(CUP, I2), I2)

    # why the cup carries no 1/sqrt(2): with the Bell state and its bra,
    # a wire with a kink in it would be HALF a wire
    bell_bra = BELL.conj().reshape(1, 4)
    kinked = kron(bell_bra, I2) @ kron(I2, BELL)
    assert np.allclose(kinked, I2 / 2)
    assert not np.allclose(kinked, I2)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the string at the walnut pair with
#   \walnutstring[opts]{laid}{caption}   and its step strip
#   \walnutsteps[opts]{laid}{steps}{captions}        (pdf/props/ch06.tex)
# The Lua COMPUTES the pulled string: bends of opposite kinds run into one
# another and cancel, so a pulled string keeps (far bends) - (near bends) of
# them.  A step (bend=hairpin, pull=half, pull=ends) is written on the string
# as it lies before it; the Lua draws the string after it, the string before it
# as a ghost, and the arcs the moving parts take.  tools/props_ch06.py
# transcribes it, as legs and bends in the order the string runs; the calls are
# read out of this challenge's section of dialog.md, never copied.
#
# The state model (identity card, sections 3 and 3e): the window at the top
# (far), the company at the bottom (near); the loose string drawn out straight in
# front of the left walnut box, then laid in a hairpin, its ends at the two
# boxes; the S (out, back, out) is a far bend behind the left box, then a near
# bend in front of the right box; pulled, its bends run in halfway and then
# through, its ends drawn away from one another; the Z the same with far and
# near exchanged.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("musicboxes", "walnutstring", "beadstring", "crabledger",
                                      "boxsteps", "walnutsteps", "beadsteps", "ledgersteps"))


def string_map(s, early):
    """A laid string as a map, read with time running away from the `early`
    side of the table: a bend on the early side is a cup (it makes a pair), a
    bend on the late side is a cap; a free end on the early side is an input,
    on the late side an output.  The legs, left to right, are the wires, top
    first.  (This is our reading of the Lua's words, not the Lua's own.)"""
    late = {"far": "near", "near": "far"}[early]
    order = s.names                           # every string drawn here runs left to right

    def layer(side, joined):
        out, i = [], 0
        while i < len(order):
            bend = [b for b in s.bends if b.side == side and order[i] in b.legs]
            if bend:
                assert bend[0].legs == (order[i], order[i + 1])   # neighbours only
                out.append(joined)
                i += 2
            else:
                out.append(I2)                    # a free end: a wire passing through
                i += 1
        return kron(*out)

    return layer(late, CAP) @ layer(early, CUP)


def check_transcription():
    """The Lua's pull rule against the blackboard (a structural check, not a
    second proof): the out-back-out string and the other-way-about string are
    the two bent wires of the yanking equation, one for each chirality; the pull
    leaves far - near = 0 bends, and the pulled straight string is the same map."""
    P.check_sites()
    P.check_lua()
    s_laid, _ = P.walnut_strand("S")
    z_laid, _ = P.walnut_strand("Z")
    for laid in (s_laid, z_laid):
        assert P.continuous(laid) and len(laid.legs) == 3 and len(laid.bends) == 2
        assert laid.count("far") == laid.count("near") == 1
        assert P.faults(P.pulled, laid, 0) == []              # the Lua's own arithmetic
        straight = P.pulled(laid, drawn=0)
        assert P.continuous(straight)
        assert straight.bends == () and straight.ends == laid.ends   # the ends stay put
        for early in ("near", "far"):             # whichever way time is read
            assert np.allclose(string_map(laid, early), I2)
            assert np.allclose(string_map(straight, early), I2)
    # the Lua's guard: a pull at the walnut pair must leave no bend (a hairpin would leave one)
    assert P.faults(P.pulled, P.walnut_strand("hairpin")[0], 0) == [
        "walnutstring: internal: the pull leaves a bend"]
    # read with time running up the page (the company's side first), the S is
    # (cap (x) I)(I (x) cup) and the Z is (I (x) cap)(cup (x) I)
    assert np.allclose(string_map(s_laid, "near"), kron(CAP, I2) @ kron(I2, CUP))
    assert np.allclose(string_map(z_laid, "near"), kron(I2, CAP) @ kron(CUP, I2))
    # the hairpin: one near bend, its ends at the boxes -- a pair, the cup
    hairpin, ends_at = P.walnut_strand("hairpin")
    assert ends_at == ("left box", "right box") and P.continuous(hairpin)
    assert np.allclose(string_map(hairpin, "near").ravel(), CUP)
    # the straight string: one line, no bend -- a bare wire
    line, _ = P.walnut_strand("straight")
    assert P.continuous(line) and line.bends == () and np.allclose(string_map(line, "near"), I2)
    # the macro's own rules, in the Lua's words
    assert P.faults(P.walnutstring, "pulled", "hairpin") == [
        "walnutstring: only the out-back-out string and the other-way-about string are pulled"]
    assert P.faults(P.walnutstring, "pulling", "straight") == [
        "walnutstring: only the out-back-out string and the other-way-about string are pulling"]
    assert P.faults(P.walnutstring, "", "out back out, the other way about") == [
        "walnutstring: one shape only"]
    assert P.faults(P.walnutstring, "", "key") == ["walnutstring: name the string's shape"]
    assert P.faults(P.walnutstring, "", "out back out, key") == [
        "walnutstring: the key rides a straight string or a hairpin"]
    assert P.faults(P.walnutstring, "sideways", "hairpin") == [
        "walnutstring: the view is laid or pulled (got 'sideways')"]
    assert P.faults(P.walnutstring, "", "hairpin, knotted") == [
        "walnutstring: unknown word 'knotted' (straight, hairpin, hairpin out, hairpin close, "
        "hairpin aside, out back out, the other way about, key, nudged)"]
    # the steps: the bend from a laid straight string, the pulls on the S and the Z only,
    # half from the string laid, to the ends from laid or half-way
    assert P.faults(P.walnutstring, "laid; bend=hairpin", "hairpin") == [
        "walnutstring: bend=hairpin bends the straight string"]
    assert P.faults(P.walnutstring, "laid; bend=straight", "straight") == [
        "walnutstring: bend=straight opens the hairpin out"]
    assert P.faults(P.walnutstring, "pulling; bend=hairpin", "out back out") == [
        "walnutstring: bend= bends a string laid, not pulling"]
    assert P.faults(P.walnutstring, "bend=knot", "straight") == [
        "walnutstring: bend=hairpin or bend=straight (not 'knot')"]
    assert P.faults(P.walnutstring, "pull=half", "hairpin") == [
        "walnutstring: pull= draws apart the out-back-out string or the other-way-about string"]
    assert P.faults(P.walnutstring, "pulling; pull=half", "out back out") == [
        "walnutstring: pull=half starts from the string laid"]
    assert P.faults(P.walnutstring, "pulled; pull=ends", "out back out") == [
        "walnutstring: pull=ends: the string is pulled already"]
    assert P.faults(P.walnutstring, "pull=all", "out back out") == [
        "walnutstring: pull=half or pull=ends (not 'all')"]
    assert P.faults(P.walnutstring, "pull=half; bend=hairpin", "out back out") == [
        "walnutstring: one step to a picture (bend= and pull=)",
        "walnutstring: bend=hairpin bends the straight string"]     # the first step tried, and refused
    assert P.faults(P.walnutstring, "marks=off", "out back out") == [
        "walnutstring: marks= go with a step"]
    assert P.faults(P.strip, "walnutsteps", False, "out back out", "view=laid; bare", "pull=half",
                    "a; b") == ["walnutsteps: no bare word in a strip's options (view=, from=, marks=)"]
    # the bare [pulled] view keeps round 3's line, its ends inside the laid S's; only a
    # pull=ends step draws the ends apart (SZAPART), and the Z is the S reflected
    bare = P.clean(P.walnutstring, "pulled", "out back out")
    assert not bare.apart and bare.line == P.ROUND3_PULLED["S"]
    assert bare.reading == "straight between the boxes"
    for shape, laid in (("S", "out back out"), ("Z", "the other way about")):
        for before in ("laid", "pulling"):
            yanked = P.clean(P.walnutstring, before + "; pull=ends", laid)
            assert yanked.apart and yanked.line == P.SZAPART[shape]
            assert yanked.reading == "straight between the boxes, its ends drawn apart"
    (s0, s1), (z0, z1) = P.SZAPART["S"], P.SZAPART["Z"]
    assert (z0, z1) == ((-s0[0], s0[1]), (-s1[0], s1[1]))
    assert np.isclose(P.meetx("Z"), -P.meetx("S"))


FORBIDDEN_WORDS = ("nudged", "hairpin out", "hairpin close", "hairpin aside")


def is_curve(arc):
    (x0, y0), *_, (x1, y1) = arc.points
    return any(abs((x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)) > 1e-3 for x, y in arc.points)


def box_gaps(arc, w):
    """The gap from an arc's drawn line to each walnut box's body (left, right)."""
    return [P.gap_rect(arc.points, *d.body) - P.SWING_W / 2 for _, d in w.drawn]


def pictures():
    check_transcription()
    found = P.check_section(DIALOGUE, "Third", sizes=[1, 1, 1, 1], frames=[2, 1, 2, 1])
    (p31, [(w30, c30), (w31, c31)]), = found[0]
    (p32, [(w32, c32)]), = found[1]
    (p33, [(w33h, c33h), (w33, c33)]), = found[2]
    (p34, [(w34, c34)]), = found[3]
    assert [c.macro for c in (p31, p32, p33, p34)] == \
        ["walnutsteps", "walnutstring", "walnutsteps", "walnutstring"]
    assert (c30, c31, c32, c33h, c33, c34) == \
        ("straight", "a pair", "out, back, out", "drawn apart", "pulled straight", "the bends swapped")
    frames = (w30, w31, w32, w33h, w33, w34)
    for c in (p31, p32, p33, p34):
        assert not any(w in P.split(c.args[0], ",") for w in FORBIDDEN_WORDS)   # never nudged
        assert "push" not in P.options(c) and not any(s.startswith("push=") for s in c.args[1:])
    assert not any(w.key or w.nudged for w in frames)

    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a1, a2, a3, a4 = DIALOGUE.anchors("Third", path)
        assert a1.endswith("tell me what the other is the end of.")
        assert "draws the loose string out straight on the cloth in front of the left walnut box" in a1
        assert "round in a first bend behind it, towards the window" in a2
        assert "in front of the right one" in a2 and "and still one string" in a2
        assert "Each takes an end and draws it away from the other" in a3
        assert "The kinks are gone" in a3
        assert a4.endswith("And bent like this?")
        assert "the first bend now facing down where it faced up" in a4
    # the directions and the caption edited this round (D2, D3, D4, C1), in both files
    P.check_edits(DIALOGUE, "Third", [
        ("*[she takes the loose straight string and lays it on the cloth in a single hairpin —",
         "*[she draws the loose string out straight on the cloth in front of the left walnut box, "
         "then lays it in a single hairpin —"),
        ("What was three legs and two bends is one taut line from Achilles's hand to the "
         "Tortoise's, and the two walnut boxes it bent around a moment ago sit untouched to "
         "either side.",
         "The two walnut boxes it bent around a moment ago sit untouched to either side."),
        ("*[laying it out again the other way about,",
         "*[taking it up and laying it out again the other way about,"),
        ("}{the other way about}", "}{the bends swapped}")])

    # each strip frame is the state after its step, on the frame before it; the yank
    # strip starts from P3.2's S (its before-frame is P3.2 itself, one speech above)
    assert [(s.opts, s.state) for s in P.frames_of(p31)] == \
        [("laid", "straight"), ("laid; bend=hairpin; keyroom=on", "straight")]
    assert P.options(p33)["from"] == "off" and p33.args[0] == p32.args[0]
    assert [(s.opts, s.state) for s in P.frames_of(p33)] == \
        [("laid; pull=half", "out back out"), ("pulling; pull=ends", "out back out")]
    assert [(w.view, w.shape) for w in frames] == \
        [("laid", "straight"), ("laid", "hairpin"), ("laid", "S"), ("pulling", "S"),
         ("pulled", "S"), ("laid", "Z")]

    # 3.1, before: the loose string drawn out straight in front of the left walnut box (D2):
    # one line, centred on that box, nearer the company than the box's front -- a bare wire
    assert p31.keyed and w30.keyed and not w31.keyed
    assert w30.labels == ("the window", "the company")
    # the keyed strip's second frame keeps the key's room: the labels as vertical phantoms
    assert not w30.phantom and w31.phantom and w31.labels == w30.labels
    assert w31.label_nodes == ("\\vphantom{the window}", "\\vphantom{the company}")
    assert not any(w.phantom or w.labels for w in frames[2:])     # the rest: no key, no room
    (ends,) = [e for _, e in P.walnut_strands("laid", "straight")]
    assert np.isclose((ends[0][0] + ends[1][0]) / 2, P.BOXX["left"])
    assert ends[0][1] == ends[1][1] < P.WY - P.WD / 2 - P.WH
    assert w30.string.bends == () and np.allclose(string_map(w30.string, "near"), I2)
    assert not w30.marked
    # 3.1, after the bend: the hairpin, the ends at the two boxes, one bend towards the company
    assert w31.shape == "hairpin" and w31.ends_at == ("left box", "right box")
    assert [b.side for b in w31.string.bends] == ["near"]
    assert np.allclose(string_map(w31.string, "near").ravel(), CUP)      # "There is a pair."
    # its marks: the straight string's ghost; the right end carried across and up to the
    # left box, the left end swung down, round and up to the right box (l.394)
    assert w31.step == ("bend", "hairpin") and w31.pins == []
    assert [(g.what, g.state) for g in w31.ghosts] == [("string", (w30.view, w30.shape))]
    across, round_ = w31.arcs
    assert across.name == "right end across and up to the left box"
    assert round_.name == "left end down, round and up to the right box"
    assert all(is_curve(a) for a in w31.arcs)
    hairpin = P.sample_path(P.walnut_strands("laid", "hairpin")[0][0])
    right_leg = [p for p in hairpin if np.isclose(p[0], P.WX)]
    the_bend = [p for p in hairpin if p[1] < P.HAIRPIN_Y["hairpin"] - 1e-9]
    straight_ends = ends
    assert np.allclose(across.pts[0], np.add(straight_ends[1][:2], (0.04, 0.04)))  # from the right end
    assert np.allclose(round_.pts[0], np.add(straight_ends[0][:2], (-0.04, -0.04)))  # from the left
    left_gap, right_gap = box_gaps(across, w31)
    assert right_gap > 0.1 and left_gap > -P.SWING_W                # it ends on its goal, the left box
    assert P.gap_points(across.points, right_leg) - P.SWING_W / 2 - P.STRING_W / 2 > 0
    left_gap, right_gap = box_gaps(round_, w31)
    assert left_gap > 0.6 and right_gap > 0                         # round under the left box
    assert P.gap_points(round_.points, the_bend) - P.SWING_W / 2 - P.STRING_W / 2 > 0
    assert min(y for _, y in round_.points) < min(y for _, y in hairpin)   # round, towards the company

    # P3.2: out, back, out -- a far bend behind the left box, then a near one in front of the right
    assert not p32.keyed and w32.view == "laid" and w32.shape == "S" and not w32.marked
    assert [(b.side, b.around) for b in w32.string.bends] == [("far", "left"), ("near", "right")]
    assert np.allclose(string_map(w32.string, "near"), I2)

    # 3.3, half-way: the same S, its two bends run in towards one another
    assert w33h.step == ("pull", "half") and w33h.pins == []
    assert [(g.what, g.state) for g in w33h.ghosts] == [("string", (w32.view, w32.shape))]
    assert [a.name for a in w33h.arcs] == ["far bend halfway in", "near bend halfway in"]
    (far0, near0), (far1, near1) = P.sz_crowns("laid", "S"), P.sz_crowns("pulling", "S")
    assert far1[1] < far0[1] and near1[1] > near0[1]                # both towards the middle
    assert w33h.string.bends == w32.string.bends
    assert np.allclose(string_map(w33h.string, "near"), I2)
    for a in w33h.arcs:
        assert is_curve(a) and min(box_gaps(a, w33h)) > 0.5
    # 3.3, pulled: far - near = 0 bends, one straight line (the yank), a box to either side
    assert w33.view == "pulled" and w33.step == ("pull", "ends") and w33.pins == []
    assert [(g.what, g.state) for g in w33.ghosts] == [("string", (w33h.view, w33h.shape))]
    assert w33.string.bends == () and len(w33.string.legs) == 1
    assert w33.string.ends == w32.string.ends                      # the same two ends
    assert np.allclose(string_map(w33.string, "near"), string_map(w32.string, "near"))
    assert np.allclose(kron(CAP, I2) @ kron(I2, CUP), I2)
    # ... its ends drawn AWAY from one another ("Each takes an end and draws it away from
    # the other"): each end at or beyond where it lay in the S, in x and in y and along the
    # line; the line clear of both walnut boxes; F14 the only picture drawn so
    assert w33.apart and w33.line == P.SZAPART["S"]
    assert w33.reading == "straight between the boxes, its ends drawn apart"
    (lo, hi) = w33.line
    (laid_lo, laid_hi) = [e[:2] for e in P.walnut_strands("laid", "S")[0][1]]
    assert lo[0] <= laid_lo[0] and lo[1] <= laid_lo[1] and hi[0] >= laid_hi[0] and hi[1] >= laid_hi[1]
    u = np.subtract(hi, lo) / np.linalg.norm(np.subtract(hi, lo))
    assert np.dot(np.subtract(laid_lo, lo), u) > 0.19 and np.dot(np.subtract(hi, laid_hi), u) > 0.05
    line = P.sample_path(w33.strands[-1][1])
    assert np.allclose(line[0], lo) and np.allclose(line[-1], hi)
    assert min(P.gap_rect(line, *d.body) for _, d in w33.drawn) > 0.11     # its centre (ch06.lua)
    xs, ys = [x for x, _ in line], [y for _, y in line]
    assert np.isclose(np.interp(P.MEETY, ys, xs), P.meetx("S"), atol=1e-3)
    # ... its arcs from the two crowns to the meeting between the boxes, each in its own
    # bend's crook, each on its own side of the pulled line, never crossing it
    far, near = w33.arcs
    assert far.name == "far bend to the meeting between the boxes"
    assert near.name == "near bend to the meeting between the boxes"
    mx = P.meetx("S")
    assert np.allclose(far.pts[-1], (mx - 0.04, P.MEETY + 0.03))
    assert np.allclose(near.pts[-1], (mx + 0.04, P.MEETY - 0.03))
    side = lambda p: np.sign(u[0] * (p[1] - lo[1]) - u[1] * (p[0] - lo[0]))
    assert {side(p) for p in far.points} == {1} and {side(p) for p in near.points} == {-1}
    for a in w33.arcs:
        assert is_curve(a) and min(box_gaps(a, w33)) > 0
    others = [w for o in ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")
              for b in DIALOGUE.blocks(o) for c in b for w, _ in P.pictures(c)]
    assert [w for w in others if getattr(w, "apart", False)] == [w33]

    # P3.4: the other way about -- the first bend now near, the second far;
    # its pull (spoken, not drawn) leaves 0 bends too: the other chirality
    assert w34.view == "laid" and w34.shape == "Z" and not w34.marked
    assert [(b.side, b.around) for b in w34.string.bends] == [("near", "left"), ("far", "right")]
    assert w32.string.bends[0].side == "far" and w34.string.bends[0].side == "near"   # up, down
    assert w34.string != w32.string and w34.string.ends == ("far", "near")
    assert P.faults(P.pulled, w34.string, 0) == [] and P.pulled(w34.string, 0).bends == ()
    assert np.allclose(string_map(w34.string, "near"), I2)
    assert "Straight again." in DIALOGUE.section("Third")

    # the log the pictures write, line for line as the build writes it
    assert P.section_dumps(found) == [
        "walnutsteps: strip of 2 frames",
        "walnutstring: laid | straight | straight | keyed",
        "walnutstring: laid | hairpin | hairpin",
        "walnutstring: + step bend hairpin",
        "walnutstring: + ghost string laid straight",
        "walnutstring: + arc right end across and up to the left box",
        "walnutstring: + arc left end down, round and up to the right box",
        "walnutstring: laid | out back out | out back out",
        "walnutsteps: strip of 2 frames",
        "walnutstring: pulling | out back out | out back out",
        "walnutstring: + step pull half",
        "walnutstring: + ghost string laid out back out",
        "walnutstring: + arc far bend halfway in",
        "walnutstring: + arc near bend halfway in",
        "walnutstring: pulled | out back out | straight between the boxes, its ends drawn apart",
        "walnutstring: + step pull ends",
        "walnutstring: + ghost string pulling out back out",
        "walnutstring: + ends drawn away from one another: (-1.090,-0.310) (0.900,2.520), "
        "each at or beyond where it lay",
        "walnutstring: + arc far bend to the meeting between the boxes",
        "walnutstring: + arc near bend to the meeting between the boxes",
        "walnutstring: laid | the other way about | the other way about"]


def bent(row_in, row_out):
    """Three rows.  The wire comes in on `row_in`, goes right, doubles back along
    the middle row, and leaves on `row_out`:  in -- cap -- cup -- out."""
    p = Picture()
    cup = p.bend((1 + row_out) / 2, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    cap = p.bend((row_in + 1) / 2, 3)    # same kind of node; wired the other way round
    p.wire(p.inp(row_in), cap, cup, p.out(row_out))
    return p.done()


def straight():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def sketchbook():
    z_bend = bent(row_in=0, row_out=2)
    s_bend = bent(row_in=2, row_out=0)
    assert np.allclose(matrix_of(straight()), I2)
    assert same_picture(z_bend, straight())                 # pull: exact,
    assert same_picture(s_bend, straight())                 # both ways round

    # the normalized version: write 1/sqrt(2) beside the cup and beside the cap
    kinked = bent(row_in=0, row_out=2)
    kinked.scalar.add_power(-2)
    assert np.allclose(matrix_of(kinked), I2 / 2)
    assert not same_picture(kinked, straight())


if __name__ == "__main__":
    verdict("Yanking", blackboard, sketchbook)
