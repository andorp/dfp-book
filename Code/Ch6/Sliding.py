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
from dfp_props import Dialogue, words_only
import props_ch06 as P


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
#   \beadstring[place]{pose}{caption}         (pdf/props/ch06.tex)
# The Lua COMPUTES the bead's picture at each place from the string's
# direction of travel there (its point along or against the travel, its notch
# on the string's left or right, both fixed when it was threaded), the napkin
# drawing (the bead as threaded), and the looking-glass bead behind the glass
# (point changed side, notch not).  tools/props_ch06.py transcribes it; the
# calls are read out of this challenge's section of dialog.md, never copied.
#
# A pose is (axis, notch): unit vectors on the page, x to the right, y up
# (towards the window).  The dialogue's pose, as threaded on the far leg:
# point to the bend (left), notch to the window (up).
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("musicboxes", "walnutstring", "beadstring", "crabledger"))
LEFT, RIGHT, UP, DOWN = (-1, 0), (1, 0), (0, 1), (0, -1)
THREADED = ((-1, 0), (0, 1))                       # point to the bend, notch to the window


def slide_table():
    """The slide round the bend, as the Lua computes it: the bead whose far-leg
    picture is `pose` has this picture on the near leg.  A map on the four
    poses (the Lua gives it for every way of threading the bead)."""
    table = {}
    for alpha, side in product((1, -1), (1, -1)):
        table[P.pose_at("far leg", alpha, side)] = P.pose_at("near leg", alpha, side)
    assert sorted(table) == sorted(table.values())      # a permutation of the four poses
    return table


def check_transcription(A):
    """The Lua's bead against the blackboard (a structural check, not a second
    proof).  On the four poses the two moves -- slide round the bend, look in
    the glass -- generate the group {as threaded, half-turn, mirror, flip over
    the long axis}; under slide <-> transpose and glass <-> dagger it is the
    group {A, A^T, A^dagger, conj(A)} of the generic A: two words give the same
    pose exactly when they give the same matrix."""
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
    # the macro's own rules
    assert P.faults(P.beadstring, "", "point to the bend") != []            # the notch missing
    assert P.faults(P.beadstring, "", "notch to the window") != []          # the point missing
    assert P.faults(P.beadstring, "", "point up, notch to the window") != []
    assert P.faults(P.beadstring, "on the box", "point to the bend, notch to the window") != []


def pictures(A):
    check_transcription(A)
    p41, p42, p43 = DIALOGUE.check_layout("Fourth", sizes=[1, 1, 1])
    for c in (p41, p42, p43):
        assert c.macro == "beadstring"
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS), c.caption
        assert "chain" not in c.caption.lower()
        assert c.opt not in ("in the bend", "back", "napkin")          # not drawn (card, 3b)
        assert c.args == ("point to the bend, notch to the window",)   # one bead, one threading
    b41, b42, b43 = (P.picture(c) for c in (p41, p42, p43))

    a1, a2, a3 = P.anchors(DIALOGUE, "Fourth")
    assert "Point towards the bend. Notch towards the window." in a1
    assert "a point at its left-hand end, a bite out of the flank on the window side" in a1
    assert a2.endswith("Look how it comes out.")
    assert a3.endswith("Read it to me.") and "point to point behind the glass" in a3
    section = DIALOGUE.section("Fourth")

    # P4.1: going in, keyed -- on the far leg, point left (to the bend), notch up (the window);
    # the napkin's drawing records exactly that
    assert p41.keyed and b41.place == "far leg"
    assert b41.labels == ("the window", "far leg", "near leg", "the company")
    assert b41.bead == (LEFT, UP) == THREADED and b41.drawing == b41.bead
    assert "Point to the left, notch to the window." in section    # Achilles's prediction ...

    # P4.2: coming out, on the near leg -- point right, notch down (the company): a half-turn
    assert not p42.keyed and b42.place == "near leg"
    assert b42.bead == (RIGHT, DOWN) and b42.bead == slide_table()[b41.bead]
    assert b42.bead != b41.bead                                      # ... is wrong
    assert P.handedness(b42.bead) == P.handedness(b41.bead)          # turned, not mirrored
    assert "pointing to the *right*" in section                      # the axis, read aloud
    assert np.allclose(kron(A, I2) @ CUP, kron(I2, A.T) @ CUP)       # the slide is the transpose

    # P4.3: in the glass -- the drawing point left, notch up; the image point right, notch STILL up
    assert b43.closeup and b43.drawing == (LEFT, UP)
    assert b43.image == (RIGHT, UP) and b43.point_to_point
    assert "In the glass the point is turned about: pointing right." in section
    assert "And the notch is... towards the window." in section
    # the real bead's notch is towards the company ("towards *me*"): not the image's
    assert "But the real one has its notch towards *me*. Away from the window." in section
    assert b43.image != b42.bead and b43.image[0] == b42.bead[0] and b43.image[1] != b42.bead[1]
    assert P.handedness(b43.image) == -P.handedness(b41.bead)        # handedness changed
    assert not np.allclose(A.conj().T, A.T)                           # mirror is not the slide


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
