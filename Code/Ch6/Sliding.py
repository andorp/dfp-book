"""Challenge 4: The Bead
======================
"It went in point first. Look how it comes out."  -- Tortoise

Claim: a box slid round a bend arrives turned half round, which for matrices
is the transpose: (A (x) I) cup = (I (x) A^T) cup for every 2x2 matrix A --
the transpose, not the dagger, not the conjugate, and not A itself.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from itertools import product

import numpy as np
from dfp import CAP, CUP, I2, Picture, kron, matrix_of, random_matrix, same_picture, verdict
from dfp_props import Dialogue
import props_ch06 as P

D12 = ("*[The Crab, with the string about to come near his boxes again, draws the right walnut box "
       "back towards the window, clear of it, and eases its partner a little aside to the left, "
       "towards the company.]*")


def blackboard():
    rng = np.random.default_rng(64)
    A = random_matrix(rng)                       # any matrix at all: not even unitary
    assert not np.allclose(A, A.T)               # generic: not symmetric,
    assert not np.allclose(A, A.conj())          #          not real,
    assert not np.allclose(A @ A.conj().T, I2)   #          not unitary
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))      # unnormalized cup

    # the theorem: A on the TOP leg = A^T on the BOTTOM leg
    lhs = kron(A, I2) @ CUP
    assert np.allclose(lhs, kron(I2, A.T) @ CUP)            # transpose: yes
    assert not np.allclose(lhs, kron(I2, A.conj().T) @ CUP) # dagger (the mirror): no
    assert not np.allclose(lhs, kron(I2, A.conj()) @ CUP)   # conjugate: no
    assert not np.allclose(lhs, kron(I2, A) @ CUP)          # unchanged: no

    # the transpose as a picture: A on the middle row with both ends bent round
    # (in on the top row, cap on the right, cup on the left, out on the bottom row)
    AT_picture = kron(CAP, I2) @ kron(I2, A, I2) @ kron(I2, CUP)
    assert np.allclose(AT_picture, A.T)
    assert not np.allclose(AT_picture, A.conj().T)
    assert not np.allclose(AT_picture, A)

    # the sketchbook proof, as matrices: put that picture on the bottom leg of a
    # cup (four rows); one yank later it is A on the top leg
    assert np.allclose(kron(I2, AT_picture) @ CUP, kron(A, I2) @ CUP)

    pictures(A)


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the bead on the string bent round the left
# walnut box, with the napkin and the Crab's glass, with
#   \beadstring[opts]{pose}{caption}   and its step strip
#   \beadsteps[opts]{pose}{steps}{captions}           (pdf/props/ch06.tex)
# The Lua COMPUTES the bead's picture at each place from the string's
# direction of travel there (its point along or against the travel, its notch
# on the string's left or right, both fixed when it was threaded), the napkin
# drawing (the bead as threaded), and the looking-glass bead behind the glass
# (point changed side, notch not).  A step (slide=) is written on the place the
# bead has before it; the Lua draws the bead after it, its ghost where it was,
# and an arc along the string.  A prediction (expects=unchanged) draws the bead
# by Achilles's rule, paler.  tools/props_ch06.py transcribes it; the calls are
# read out of this challenge's section of dialog.md, never copied.
#
# A pose is (axis, notch): unit vectors on the page, x to the right, y up
# (towards the window).  The dialogue's pose, as threaded on the far leg:
# point to the bend (left), notch to the window (up).
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("musicboxes", "walnutstring", "beadstring", "crabledger",
                                      "boxsteps", "walnutsteps", "beadsteps", "ledgersteps"))
LEFT, RIGHT, UP, DOWN = (-1, 0), (1, 0), (0, 1), (0, -1)
THREADED = ((-1, 0), (0, 1))                       # point to the bend, notch to the window
POSE = "point to the bend, notch to the window"


def slide_table():
    """The slide round the bend, as the Lua computes it: the bead whose far-leg
    picture is `pose` has this picture on the near leg.  A map on the four
    poses (the Lua gives it for every way of threading the bead)."""
    table = {}
    for alpha, side in product((1, -1), (1, -1)):
        table[P.pose_at("far leg", alpha, side)] = P.pose_at("near leg", alpha, side)
    assert sorted(table) == sorted(table.values())      # a permutation of the four poses
    return table


def quarter_turn(pose):
    """The bead turned a quarter round on the cloth, as the string turns under it going
    into the bend (anticlockwise: the far leg runs left, the bend's foot runs down)."""
    (ax, ay), (nx, ny) = pose
    return (-ay, ax), (-ny, nx)


def check_transcription(A):
    """The Lua's bead against the blackboard (a structural check, not a second
    proof).  On the four poses the two moves -- slide round the bend, look in
    the glass -- generate the group {as threaded, half-turn, mirror, flip over
    the long axis}; under slide <-> transpose and glass <-> dagger it is the
    group {A, A^T, A^dagger, conj(A)} of the generic A: two words give the same
    pose exactly when they give the same matrix."""
    P.check_sites()
    P.check_lua()
    slide = slide_table()
    for pose, after in slide.items():                   # the slide is a half-turn:
        assert after == (P.scale(-1, pose[0]), P.scale(-1, pose[1]))   # point and notch round together
        assert P.handedness(after) == P.handedness(pose)                # a left shoe stays left
        assert P.handedness(P.glass_image(pose)) == -P.handedness(pose)  # the glass: a right one
    moves = {"slide": (slide.__getitem__, lambda M: M.T),
             "glass": (P.glass_image, lambda M: M.conj().T)}
    words = [w for n in range(4) for w in product(moves, repeat=n)]

    def run(word, M=A):
        pose = THREADED
        for m in word:
            pose, M = moves[m][0](pose), moves[m][1](M)
        return pose, M

    poses = {run(w)[0] for w in words}
    assert len(poses) == 4                              # four poses, no more
    for w1, w2 in product(words, repeat=2):
        (q1, M1), (q2, M2) = run(w1), run(w2)
        assert (q1 == q2) == bool(np.allclose(M1, M2)), (w1, w2)
    for w in words:                                     # handedness kept <-> no conjugation
        q, _ = run(w)
        linear = np.allclose(run(w, 1j * A)[1], 1j * run(w)[1])
        assert (P.handedness(q) == P.handedness(THREADED)) == bool(linear), w
    # slide, then glass: point kept, notch turned -- "lay it down on its face" -- the conjugate
    flip, M = run(("slide", "glass"))
    assert flip == (THREADED[0], P.scale(-1, THREADED[1])) and np.allclose(M, A.conj())
    # the slide in two steps: a quarter-turn into the bend, a quarter-turn out of it
    for alpha, side in product((1, -1), (1, -1)):
        far, bend, near = (P.pose_at(p, alpha, side) for p in ("far leg", "in the bend", "near leg"))
        assert quarter_turn(far) == bend and quarter_turn(bend) == near == slide[far]
    # the wrong rules, as the Lua's comment states them: `unchanged` draws the near-leg bead
    # with the far-leg bead's point and notch; `mirrored` as its own image in the glass
    for alpha, side in product((1, -1), (1, -1)):
        far = P.pose_at("far leg", alpha, side)
        assert P.rule_unchanged(alpha, side) == far
        assert P.rule_mirrored(alpha, side) == P.glass_image(far)
    # the macro's own rules, in the Lua's words
    assert P.faults(P.beadstring, "", "point to the bend") == [
        "beadstring: give the point and the notch as threaded"]             # the notch missing
    assert P.faults(P.beadstring, "", "notch to the window") == [
        "beadstring: give the point and the notch as threaded"]             # the point missing
    assert P.faults(P.beadstring, "", "point up, notch to the window") == [
        "beadstring: the bead as threaded is 'point to the bend' or 'point away from the bend', "
        "and 'notch to the window' or 'notch to the company' (got 'point up')",
        "beadstring: give the point and the notch as threaded"]
    assert P.faults(P.beadstring, "on the box", POSE) == [
        "beadstring: the place is far leg, in the bend, near leg, back, napkin or glass (got "
        "'on the box')"]
    # the steps: a slide along the string only, never to where the bead is, never back
    # from the far leg; a prediction only of the near leg, unkeyed, with no step
    assert P.faults(P.beadstring, "napkin; slide=near leg", POSE) == [
        "beadstring: slide= moves the bead on the string, not on the napkin"]
    assert P.faults(P.beadstring, "slide=napkin", POSE) == [
        "beadstring: slide=in the bend, near leg or back (not 'napkin')"]
    assert P.faults(P.beadstring, "near leg; slide=near leg", POSE) == [
        "beadstring: slide=near leg: the bead is there already"]
    assert P.faults(P.beadstring, "slide=back", POSE) == [
        "beadstring: slide=back: the bead is there already"]    # (the far leg IS back)
    assert P.faults(P.beadstring, "far leg; expects=unchanged", POSE) == [
        "beadstring: expects=unchanged predicts how the bead arrives: the near leg"]
    assert P.faults(P.beadstring, "near leg; expects=turned", POSE) == [
        "beadstring: unknown rule 'turned' (unchanged, mirrored)"]
    assert P.faults(P.beadstring, "near leg; expects=unchanged", POSE, True) == [
        "beadstring: an expects picture is not keyed"]
    assert P.faults(P.beadstring, "far leg; slide=near leg; expects=mirrored", POSE) == [
        "beadstring: an expects picture carries no step"]
    assert P.faults(P.beadstring, "near leg; marks=off", POSE) == [
        "beadstring: marks= go with a step"]
    # back again: a slide from the near leg to the far one ("I slid it round", undone)
    back = P.clean(P.beadstring, "near leg; slide=back", POSE)
    assert back.place == "back" and back.bead == THREADED
    # `mirrored` is not drawn (card 3e.7, Q4: at its refutation the glass still stands on
    # the napkin, which the full view cannot show); its frame would differ from the true
    # near-leg bead in the notch alone
    m = P.clean(P.beadstring, "near leg; expects=mirrored", POSE)
    true = P.clean(P.beadstring, "near leg", POSE)
    assert m.bead == (RIGHT, UP) and true.bead == (RIGHT, DOWN)
    assert m.bead[0] == true.bead[0] and m.bead[1] != true.bead[1]


def pictures(A):
    check_transcription(A)
    found = P.check_section(DIALOGUE, "Fourth", sizes=[1, 2, 1], frames=[1, 3, 1])
    (p41, [(b41, cap41)]), = found[0]
    (p42e, [(b42e, cap42e)]), (p42, [(b42b, cap42b), (b42, cap42)]) = found[1]
    (p43, [(b43, cap43)]), = found[2]
    assert [c.macro for c in (p41, p42e, p42, p43)] == \
        ["beadstring", "beadstring", "beadsteps", "beadstring"]
    assert (cap41, cap42e, cap42b, cap42, cap43) == \
        ("going in", "as Achilles expects", "round the bend", "coming out", "point to point")
    for c in (p41, p42e, p42, p43):
        assert c.args[0] == POSE                                       # one bead, one threading
        assert c.args == ((POSE,) if c.macro == "beadstring" else (POSE, c.args[1]))
    frames = (b41, b42e, b42b, b42, b43)
    assert not any(b.place in ("back", "napkin") for b in frames)     # not drawn (card, 3b)
    # Q4, ruled: no frame for `mirrored`
    assert not any(P.options(c).get("expects") == "mirrored" for c in (p41, p42e, p42, p43))

    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a1, a2, a3 = DIALOGUE.anchors("Fourth", path)
        assert "Point towards the bend. Notch towards the window." in a1
        assert "a point at its left-hand end, a bite out of the flank on the window side" in a1
        assert a2.endswith("Look how it comes out.")
        assert a3.endswith("Read it to me.") and "point to point behind the glass" in a3
    section = DIALOGUE.section("Fourth")
    # the direction and the caption edited this round (D5, C2), in both files
    P.check_edits(DIALOGUE, "Fourth", [
        ("She bends the string once round the left walnut box, so that",
         "She takes the string up and bends it once round the left walnut box, so that"),
        ("}{in the glass}", "}{point to point}"),
        (None, D12)])
    # D12: the walnut pair moved for the bead, in words, before any bead picture -- a direction
    # of its own between Achilles's "Lopsided twice over." and the Tortoise's "Twice over"
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        paras = DIALOGUE.paragraphs("Fourth", path)
        i = paras.index(D12)
        assert paras[i - 1].endswith("Lopsided twice over.")
        assert paras[i + 1].startswith("**Tortoise:** Twice over is what we shall need.")
        assert i < min(j for j, q in enumerate(paras) if DIALOGUE.is_block(q))   # before F16
    # ... and every bead frame on the cloth (F16-F19) draws the pair where D12 leaves it: the
    # right box drawn back towards the window, the left eased to the left, towards the
    # company -- each away from where the Third's pictures stood them -- the string bent round
    # the left box, and both boxes clear of it (drawn extents, string widths subtracted)
    third = [p for b in DIALOGUE.blocks("Third") for c in b for p, _ in P.pictures(c)]
    assert {tuple((d.x, d.y) for _, d in p.drawn) for p in third} == {((-P.WX, P.WY), (P.WX, P.WY))}
    tl, ty = -P.WX, P.WY                                                 # the Third's left box
    for b in (b41, b42e, b42b, b42):
        (right, left), = {tuple((d.x, d.y) for _, d in b.drawn)}
        assert (right, left) == P.BEADBOXES == ((0.30, 2.10), (-0.64, 1.08))
        assert right[1] > ty and left[0] < tl and left[1] < ty          # window; left; company
        assert np.hypot(left[0] - P.BCX, left[1] - P.BCY) < P.BR        # inside the bend
        string = P.sample_path(b.strands[0][1])
        for _, d in b.drawn:
            assert P.gap_rect(string, *d.body) - P.STRING_W / 2 > 0.05  # clear of the string
    assert b43.closeup and b43.drawn == []                              # the glass: no boxes

    # P4.1: going in, keyed -- on the far leg, point left (to the bend), notch up (the window);
    # the napkin's drawing records exactly that
    assert p41.keyed and b41.place == "far leg" and not b41.marked
    assert b41.labels == ("the window", "far leg", "near leg", "the company")
    assert b41.bead == (LEFT, UP) == THREADED and b41.drawing == b41.bead
    assert "Point to the left, notch to the window." in section    # Achilles's prediction ...

    # 4.2, as Achilles expects: the near-leg bead drawn by the `unchanged` rule -- the
    # far-leg bead's point and notch, paler, unkeyed, nothing moving
    assert P.options(p42e)["expects"] == "unchanged" and b42e.rule == "unchanged"
    assert not p42e.keyed and not b42e.marked and b42e.step is None
    assert b42e.place == "near leg" and b42e.bead == P.rule_unchanged(1, -1) == b41.bead
    assert {sc for sc, _ in b42e.drawn} == {sc for sc, _ in b42e.strands} == {"pale"}
    # 4.2, the strip: it starts where P4.1 left the bead (the far leg) -- the slide round the
    # bend, then out along the near leg, each frame the state after its step
    assert P.options(p42)["from"] == "off" and not p42.keyed
    assert [(s.opts, s.state) for s in P.frames_of(p42)] == \
        [("far leg; slide=in the bend", POSE), ("in the bend; slide=near leg", POSE)]
    assert (b42b.place, b42.place) == ("in the bend", "near leg")
    # in the bend: turned a quarter round (point to the company, notch to the left, outside
    # the bend) -- half-way to the transpose
    assert b42b.bead == (DOWN, LEFT) == quarter_turn(b41.bead)
    # coming out, on the near leg -- point right, notch down (the company): a half-turn
    assert b42.bead == (RIGHT, DOWN) == quarter_turn(b42b.bead)
    assert b42.bead == slide_table()[b41.bead]
    assert b42.bead != b41.bead                                      # ... is wrong
    assert b42e.bead[0] != b42.bead[0] and b42e.bead[1] != b42.bead[1]   # the prediction, refuted
    assert b42e.bead_at == b42.bead_at                               # drawn where it arrives
    assert P.handedness(b42.bead) == P.handedness(b42b.bead) == P.handedness(b41.bead)
    assert "pointing to the *right*" in section                      # the axis, read aloud
    assert np.allclose(kron(A, I2) @ CUP, kron(I2, A.T) @ CUP)       # the slide is the transpose
    # the marks: the bead's ghost where it was, as it was; an arc along the string, at a
    # fixed distance outside it, round the bend behind the box, clear of the napkin and of
    # both walnut boxes
    string = P.sample_path(P.bead_path())
    for b, before, name in ((b42b, b41, "slide far leg to in the bend, along the string outside the bend"),
                            (b42, b42b, "slide in the bend to near leg, along the string outside the bend")):
        assert b.step[0] == "slide" and b.pins == []
        (g,) = b.ghosts
        assert (g.what, g.state, g.place) == ("bead", before.bead, before.bead_at)
        (arc,) = b.arcs
        assert arc.name == name
        pts = arc.points
        off = [min(np.hypot(x - u, y - v) for u, v in string) for x, y in pts[::4]]
        assert np.allclose(off, P.BOFF, atol=1e-3)                    # keeps to the string
        assert all(np.hypot(x - P.BCX, y - P.BCY) > P.BR for x, y in pts if x < P.BCX)  # outside it
        assert P.gap_rect(pts, *P.NAPKIN_FULL) - P.SWING_W / 2 > 0
        assert min(P.gap_rect(pts, *d.body) for _, d in b.drawn) - P.SWING_W / 2 > 0
    assert b42b.ghosts[0].place[1] > P.BCY and b42.bead_at[1] < P.BCY  # from the far leg to the near

    # P4.3: in the glass -- the drawing point left, notch up; the image point right, notch STILL up
    assert b43.closeup and b43.drawing == (LEFT, UP) and not b43.marked
    assert b43.image == (RIGHT, UP) and b43.point_to_point
    assert "In the glass the point is turned about: pointing right." in section
    assert "And the notch is... towards the window." in section
    # the real bead's notch is towards the company ("towards *me*"): not the image's
    assert "But the real one has its notch towards *me*. Away from the window." in section
    assert b43.image != b42.bead and b43.image[0] == b42.bead[0] and b43.image[1] != b42.bead[1]
    assert P.handedness(b43.image) == -P.handedness(b41.bead)        # handedness changed
    assert not np.allclose(A.conj().T, A.T)                           # mirror is not the slide

    # the log the pictures write, line for line as the build writes it
    head = "beadstring: {} | point to the bend, notch to the window"
    drawing = "beadstring: + drawing point left notch window"
    assert P.section_dumps(found) == [
        head.format("far leg") + " | keyed", drawing,
        "beadstring: + bead far leg point left notch window",
        head.format("near leg") + " | expects unchanged", drawing,
        "beadstring: + bead near leg point left notch window (by the unchanged rule)",
        "beadsteps: strip of 2 frames",
        head.format("in the bend"),
        "beadstring: + step slide far leg to in the bend", drawing,
        "beadstring: + ghost bead far leg point left notch window",
        "beadstring: + bead in the bend point company notch left",
        "beadstring: + arc slide far leg to in the bend, along the string outside the bend",
        head.format("near leg"),
        "beadstring: + step slide in the bend to near leg", drawing,
        "beadstring: + ghost bead in the bend point company notch left",
        "beadstring: + bead near leg point right notch company",
        "beadstring: + arc slide in the bend to near leg, along the string outside the bend",
        head.format("glass"), drawing,
        "beadstring: + image point right notch window"]


# A generic one-wire box for the sketchbook: a chain of three phase nodes,
# listed from the box's INPUT side to its OUTPUT side.  Each single node is a
# symmetric matrix, so reading the chain backwards is the transpose: the box
# turned end for end.  Negating the phases is the complex conjugate.
BEAD = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))     # phases in units of pi


def turned(box):        # half-turn: the transpose
    return box[::-1]


def conjugated(box):
    return tuple(-phase for phase in box)


def mirrored(box):      # mirror image: the dagger
    return conjugated(turned(box))


def thread(p, row, box):
    """Put the box on a row; returns its nodes in order, input side first."""
    a, b, c = box
    return [p.z(row, 3, a), p.x(row, 4, b), p.z(row, 5, c)]


def box_alone(box):
    p = Picture()
    p.wire(p.inp(0), *thread(p, 0, box), p.out(0))
    return p.done()


def box_on_bend(row, box):
    """A bent wire with the box on one leg (row 0 = top leg, row 1 = bottom leg).
    The box's input side faces the bend."""
    p = Picture()
    bend = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(bend, *thread(p, row, box), p.out(row))
    p.wire(bend, p.out(1 - row))
    return p.done()


def both_ends_bent_round(box):
    """Three rows: in on the top row, a cap on the right, the box on the middle
    row, a cup on the left, out on the bottom row.  On the middle row the wire
    runs right to left, so coming from the cap we meet the box's OUTPUT side first."""
    p = Picture()
    cap, cup = p.bend(0.5, 7), p.bend(1.5, 1)
    p.wire(p.inp(0), cap, *thread(p, 1, box)[::-1], cup, p.out(2))
    return p.done()


def substituted(box):
    """Four rows: the three-row picture above, sitting on the bottom leg of an outer bend."""
    p = Picture()
    outer, cap, cup = p.bend(0.5, 0.5), p.bend(1.5, 7), p.bend(2.5, 1)
    p.wire(p.out(0), outer, cap, *thread(p, 2, box)[::-1], cup, p.out(3))
    return p.done()


def sketchbook():
    M = matrix_of(box_alone(BEAD))
    assert not np.allclose(M, M.T) and not np.allclose(M, M.conj())   # a lopsided bead
    assert np.allclose(matrix_of(box_alone(turned(BEAD))), M.T)
    assert np.allclose(matrix_of(box_alone(mirrored(BEAD))), M.conj().T)
    assert np.allclose(matrix_of(box_alone(conjugated(BEAD))), M.conj())

    # slide the bead from the top leg round to the bottom leg
    on_top = box_on_bend(0, BEAD)
    assert np.allclose(matrix_of(on_top).ravel(), kron(M, I2) @ CUP)  # top wire first
    assert same_picture(on_top, box_on_bend(1, turned(BEAD)))         # turned end for end: yes
    assert not same_picture(on_top, box_on_bend(1, mirrored(BEAD)))   # mirrored: no
    assert not same_picture(on_top, box_on_bend(1, conjugated(BEAD))) # conjugated: no
    assert not same_picture(on_top, box_on_bend(1, BEAD))             # unchanged: no

    # the box with both ends bent round IS the turned box ...
    assert same_picture(both_ends_bent_round(BEAD), box_alone(turned(BEAD)))
    assert not same_picture(both_ends_bent_round(BEAD), box_alone(mirrored(BEAD)))
    # ... and substituting it on the bottom leg, then yanking, is the slide
    assert same_picture(substituted(BEAD), on_top)


if __name__ == "__main__":
    verdict("Sliding", blackboard, sketchbook)
