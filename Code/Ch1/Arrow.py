"""Challenge 5: The Arrow
==========================
"Five. Not seven."  -- Achilles

Claim: a qubit state is normalized by its squared moduli and not by its entries;
NOT exchanges the two reaches and keeps the length; and two NOT boxes in a row
return the arrow exactly -- the third passenger of The Round Trip.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
from fractions import Fraction

import numpy as np
from dfp import I2, Picture, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch01 as P

X = np.array([[0, 1], [1, 0]], dtype=complex)      # NOT
S0 = np.array([[1, 1], [0, 0]], dtype=complex)     # SET-0

ARROW = np.array([3 / 5, 4 / 5], dtype=complex)    # three matches along, four up


def blackboard():
    # nine and sixteen make twenty-five: the squares on the reaches make the whole
    assert np.isclose(abs(ARROW[0]) ** 2 + abs(ARROW[1]) ** 2, 1)
    assert np.isclose(np.linalg.norm(ARROW), 1)
    # three and four overshoot: as a wager it is nonsense
    assert np.isclose(ARROW.sum().real, 7 / 5) and not np.isclose(ARROW.sum().real, 1)

    # turned over: the reaches change places, and it is still one arrow
    assert np.allclose(X @ ARROW, [4 / 5, 3 / 5])
    assert np.isclose(np.linalg.norm(X @ ARROW), 1)
    assert not np.allclose(X @ ARROW, ARROW)
    assert np.allclose(X @ (X @ ARROW), ARROW)              # home

    # a generic passenger: a complex unit vector, asserted non-real with a
    # non-real ratio, so nothing here can hide behind a symmetry
    rng = np.random.default_rng(5)
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())
    assert not np.isclose(psi[0], 0) and not np.isclose(psi[1], 0)
    assert not np.isclose((psi[1] / psi[0]).imag, 0)
    assert np.isclose(np.linalg.norm(psi), 1)
    assert np.allclose(X @ psi, psi[::-1])
    assert np.isclose(np.linalg.norm(X @ psi), 1)
    assert not np.allclose(X @ psi, psi)
    assert np.allclose(X @ (X @ psi), psi)                  # The Round Trip

    # the two definite starts are legal under all three rules
    for e in (ket(0), ket(1)):
        assert np.isclose(e.sum().real, 1)                  # a bit, and a wager
        assert np.isclose(np.linalg.norm(e), 1)             # and an arrow

    # not every classical box may be hung on a wire carrying an arrow (chapter.md,
    # end of 1.5): SET-0 keeps sums but not sums of squares
    assert np.isclose((S0 @ ARROW).sum(), ARROW.sum())      # the sum survives
    assert np.allclose(S0 @ ARROW, [7 / 5, 0])
    assert not np.isclose(np.linalg.norm(S0 @ ARROW), 1)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the square of tracing paper close up with
# \arrowsquare[view;opts]{matches}{caption}, and the square tied on to the
# Tortoise's string with \tagstring{square}{tags}{caption} (pdf/props/ch01.tex;
# the rules are in pdf/props/ch01.lua).  The macro COMPUTES the reaches from the
# view and how many matches lie in each row; with marks=on, the arrow where it
# lay before the turn; with expects=sum, the arrow's row as Achilles counts it,
# drawn pale.  tools/props_ch01.py transcribes the Lua, all but the drawing; the
# calls are read out of this challenge's section of dialog.md
# (tools/dfp_props.py), never copied.  Every picture is checked against the
# model below.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("tagstring", "tagwalk", "arrowsquare"))
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth")
QUOTE = __doc__.split('"')[1]


# --- the arrow, in the chapter's state model ---------------------------------
# heads-ward first; with the matches for a ruler its reaches are five times its
# entries and its length five times its norm.  "turn it over" is NOT, the
# matrix blackboard() uses: the square turned over about the line from the
# arrow's corner to the far corner, the two reaches changing places.
def in_matches(v):
    """The reaches heads-ward and tails-ward, and the length, in matches."""
    return (5 * v[0].real, 5 * v[1].real, 5 * np.linalg.norm(v))


def carry_arrow(v, tags):
    """The tags met, in the order met: the string read left to right."""
    for t in tags:
        assert t == "over"
        v = X @ v
    return v


def as_achilles_counts(v):
    """Achilles: "Three parts heads-ward and four tails-ward: seven parts in all.
    Your arrow is seven matches long" -- the reaches added, as for a strip."""
    along, up, _ = in_matches(v)
    return along + up


def check_close_up(sq, turns, rows):
    """A close-up against the model: the arrow turned over `turns` times, the
    reaches drawn and the matches in each laid row."""
    v = np.linalg.matrix_power(X, turns) @ ARROW
    along, up, length = in_matches(v)
    assert sq.back == (turns % 2 == 1)                       # seen through the paper
    assert np.allclose(sq.reaches, (along, up)) and sq.rows == frozenset(rows)
    assert sq.count[:2] == (round(along) if "along" in rows else 0,
                            round(up) if "up" in rows else 0)
    if not sq.sum:
        assert sq.count[2] == sq.arrow_row == (round(length) if "arrow" in rows else 0)
    if sq.marks:
        before = np.linalg.matrix_power(X, turns - 1) @ ARROW    # the arrow before the turn
        assert np.allclose(sq.ghost, in_matches(before)[:2])
    else:
        assert sq.ghost is None
    return v


def check_square_rule():
    """The macro's square against the model, on every view and every choice of
    rows, with and without the turn's marks; the rule `sum` against Achilles's
    count; and the square tied on under up to four turn-over tags."""
    for view, turns in (("drawn", 0), ("", 0), ("home", 2), ("twice", 2), ("over", 1)):
        for k in range(4):
            for rows in itertools.combinations(("along", "up", "arrow"), k):
                for spec in (",".join(rows),) + (("none",) if k == 0 else ()):
                    check_close_up(P.arrowsquare(view, spec), turns, rows)
                    mopt = view + ";marks=on"
                    if turns:
                        assert P.faults(P.arrowsquare, mopt, spec) == []
                        sq = P.arrowsquare(mopt, spec)
                        assert sq.marks and check_close_up(sq, turns, rows) is not None
                    else:
                        assert P.faults(P.arrowsquare, mopt, spec)   # no turn, no marks
                    sopt = view + ";expects=sum"
                    if turns % 2 == 0 and "arrow" in rows:     # as drawn (home, twice too)
                        assert P.faults(P.arrowsquare, sopt, spec) == []
                        sq = P.arrowsquare(sopt, spec)
                        assert sq.sum and not sq.marks
                        check_close_up(sq, turns, rows)
                        assert sq.count[2] == sq.arrow_row == as_achilles_counts(ARROW) == 7
                    else:
                        assert P.faults(P.arrowsquare, sopt, spec)
    along, up, length = in_matches(ARROW)
    assert np.isclose(along ** 2 + up ** 2, 25) and np.isclose(length, 5)
    assert not np.isclose(along + up, length)          # three and four are not seven
    assert P.arrowsquare("", "edges") == P.arrowsquare("drawn", "along,up")
    assert P.arrowsquare("", "all") == P.arrowsquare("drawn", "along,up,arrow")
    assert P.faults(P.arrowsquare, "expects=sum;marks=on", "all")
    for n in range(5):
        spec = ",".join(["over"] * n)
        t = P.tagstring("start", "square", spec)
        assert P.faults(P.tagstring, "start", "square", spec) == []
        assert t.shown == {"kind": "square", "turned": False}        # drawn tied on, as tied
        whole = P.travel(t.start, list(t.tags), n)                   # what the tags would do
        reaches = (4, 3) if whole["turned"] else (3, 4)
        assert np.allclose(reaches, in_matches(np.linalg.matrix_power(X, n) @ ARROW)[:2])
        # the square is never walked, never marked, never predicted
        if n:
            for opt in ("first", "end", "first;marks=on", "end;expects=anyorder"):
                assert P.faults(P.tagstring, opt, "square", spec)


def pictures():
    P.check_lua()                                  # the Lua and .tex the transcription was proved on
    check_square_rule()
    P.check_dialogue(DIALOGUE, ORDINALS)
    assert QUOTE in DIALOGUE.section("Fifth")

    # Four moments.  Block 5.1, the keyed close-up: the matches along the two
    # edges.  Block 5.2: the prediction, pale -- seven matches -- beside the
    # arrow measured, five.  Block 5.3: the turn, marked (the square cleared of
    # matches), then measured once more.  Block 5.4: the square tied on under two
    # turn-over tags, and with them lifted off.
    found = P.check_section(DIALOGUE, "Fifth", sizes=[1, 2, 2, 2], frames=[1, 2, 2, 2])
    (_, [(start, _)]), = found[0]
    (_, [(seven, caption)]), (_, [(five, _)]) = found[1]
    (_, [(turn, _)]), (_, [(again, _)]) = found[2]
    (c_tied, [(tied, _)]), (c_lifted, [(lifted, _)]) = found[3]
    assert caption == "as Achilles expects"

    check_close_up(start, 0, {"along", "up"})
    assert start.keyed and start.labels == ("heads-ward", "tails-ward") and start.count == (3, 4, 0)
    # the measure: five, not seven, and nine and sixteen make twenty-five
    v = check_close_up(five, 0, {"along", "up", "arrow"})
    assert five.count == (3, 4, 5) and not five.sum
    assert five.count[0] + five.count[1] == 7 != five.count[2]
    assert five.count[0] ** 2 + five.count[1] ** 2 == five.count[2] ** 2
    assert round(in_matches(v)[2]) == five.count[2]
    # the prediction: the named rule, his count, on the same square with the same
    # rows laid -- seven along the arrow -- and not the five beside it.  The row
    # of seven runs past the tip and off the paper, by a clear gap.
    assert seven.sum and not seven.marks and seven.ghost is None
    assert (seven.view, seven.rows, seven.reaches) == (five.view, five.rows, five.reaches)
    check_close_up(seven, 0, {"along", "up", "arrow"})
    assert seven.count == (3, 4, 7) and seven.count[2] == as_achilles_counts(ARROW)
    assert seven.count[2] != five.count[2] and not np.isclose(as_achilles_counts(ARROW),
                                                              in_matches(ARROW)[2])
    assert seven.far_end - P.SIDE > 0.1                     # off the paper
    assert five.arrow_row * 4 / 5 < P.SIDE                  # the five stay on it
    # the turn: no matches (she cleared them), the arrow where it lay -- three
    # and four -- marked, the reaches now four and three; then measured: still five
    w = check_close_up(turn, 1, set())
    assert turn.marks and turn.count == (0, 0, 0)
    assert turn.ghost == (3, 4) and turn.reaches == (4, 3)
    assert np.allclose(turn.ghost, in_matches(ARROW)[:2]) and np.allclose(w, X @ ARROW)
    check_close_up(again, 1, {"along", "up", "arrow"})
    assert again.count == (4, 3, 5) and not again.marks and again.back
    # tied on, nothing walked: drawn as tied, under two tags and with them lifted
    assert c_tied.macro == c_lifted.macro == "tagstring"
    assert tied.tags == ("over", "over") and lifted.tags == ()
    for t in (tied, lifted):
        assert t.where == "start" and t.passed == 0 and not t.marks and t.expects is None
        assert t.shown == {"kind": "square", "turned": False}
        assert np.allclose((4, 3) if t.shown["turned"] else (3, 4), in_matches(ARROW)[:2])
        # had it been walked past the tags, it would arrive as it left
        whole = P.travel(t.start, list(t.tags), len(t.tags))
        assert whole == t.start
        assert np.allclose(carry_arrow(ARROW, t.tags), ARROW)
    assert [p.dump for b in found for _, fr in b for p, _ in fr] == [
        "arrowsquare: drawn reaches 3,4 matches 3,4,0",
        "arrowsquare: drawn reaches 3,4 matches 3,4,7 expects sum",
        "arrowsquare: drawn reaches 3,4 matches 3,4,5",
        "arrowsquare: over reaches 4,3 matches 0,0,0 ghost 3,4",
        "arrowsquare: over reaches 4,3 matches 4,3,5",
        "tagstring: start square-34 -> square-34",
        "tagstring: start square-34 -> square-34"]

    # where they stand
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        paras = DIALOGUE.paragraphs("Fifth", path)
        a_start, a_five, a_again, a_tied = DIALOGUE.anchors("Fifth", path)
        assert a_start.endswith("A start.")
        assert a_five.endswith("As a wager it's nonsense.")
        # block 5.3 follows the measuring; the turn it marks is the paragraph before
        assert a_again.endswith("turned over, it is still one arrow.")
        turned = paras[paras.index(a_again) - 1]
        for phrase in ("She clears the matches away and turns the square face-down",
                       "The corner stays pinned where it was",
                       "what swings across is the rest of the square",
                       "through the paper the arrow shows, as long as ever"):
            assert phrase in turned, phrase
        assert a_tied.endswith("He slides the two tags together and lifts them off.]*")


PHASES = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))


def arrow_then_nots(k):
    """A generic arrow -- the state box 0 followed by three phase nodes -- then k
    NOT boxes, in a row on one wire, read left to right."""
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    s = p.x(0, 1, 0)
    a, b, c = PHASES
    chain = [p.z(0, 2, a), p.x(0, 3, b), p.z(0, 4, c)]
    nots = [p.x(0, 6 + 2 * i, 1) for i in range(k)]
    p.wire(s, *chain, *nots, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def not_box():
    p = Picture()
    n = p.x(0, 2, 1)
    p.wire(p.inp(0), n, p.out(0))
    return p.done()


def two_not_boxes():
    p = Picture()
    a, b = p.x(0, 2, 1), p.x(0, 4, 1)
    p.wire(p.inp(0), a, b, p.out(0))
    return p.done()


def sketchbook():
    psi = matrix_of(arrow_then_nots(0)).ravel()
    assert np.isclose(np.linalg.norm(psi), 1)               # exact after the scalar fix
    assert not np.isclose(psi[0], 0) and not np.isclose(psi[1], 0)
    assert not np.isclose((psi[1] / psi[0]).imag, 0)        # a genuine arrow, not a wager

    # one turn-over: the two reaches change places
    turned = matrix_of(arrow_then_nots(1)).ravel()
    assert np.allclose(turned, psi[::-1])
    assert np.allclose(turned, X @ psi)
    assert not same_picture(arrow_then_nots(1), arrow_then_nots(0))

    # two turn-overs: the same picture again, exactly -- The Round Trip, read with
    # its third passenger
    assert same_picture(arrow_then_nots(2), arrow_then_nots(0))

    # The three-four-five arrow itself has no exact picture built from these nodes,
    # so it is checked on the blackboard above and against the box here.
    assert np.allclose(matrix_of(not_box()) @ ARROW, [4 / 5, 3 / 5])
    assert np.allclose(matrix_of(two_not_boxes()) @ ARROW, ARROW)
    assert np.allclose(matrix_of(two_not_boxes()), I2)


if __name__ == "__main__":
    verdict("Arrow", blackboard, sketchbook)
