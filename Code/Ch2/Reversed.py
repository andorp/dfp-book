"""Challenge 5: The Arrow Reversed
====================================
"A different arrow. Find me the question that knows it."  -- Tortoise

Claim: multiplying the whole column by a phase gives a different column and a
different picture, and no question whatever tells it from the original; a phase
on ONE entry does not have that property.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import math
from fractions import Fraction

import numpy as np
from dfp import Picture, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch02 as P

PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)
T = np.array([3 / 5, 4 / 5], dtype=complex)
PHASES = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))
THETA = 2 * np.pi / 7          # the picture's phase, as an angle


def born(state, b):
    return (np.vdot(state, b) * np.vdot(b, state)).real


def blackboard():
    # the arrow reversed: every amplitude changes sign, every chance stays
    rev = -T
    assert not np.allclose(rev, T) and np.isclose(np.vdot(rev, rev), 1)
    assert np.isclose(np.vdot(ket(0), rev), -3 / 5)
    assert np.isclose(np.vdot(ket(1), rev), -4 / 5)
    assert np.isclose(np.vdot(PLUS, rev), -7 / (5 * np.sqrt(2)))
    assert np.isclose(np.vdot(MINUS, rev), +1 / (5 * np.sqrt(2)))
    for b, want in ((ket(0), 9 / 25), (ket(1), 16 / 25),
                    (PLUS, 49 / 50), (MINUS, 1 / 50)):
        assert np.isclose(born(T, b), want) and np.isclose(born(rev, b), want)

    # the general statement, with a generic complex state
    rng = np.random.default_rng(25)
    psi = random_state(rng)
    assert not np.allclose(psi.imag, 0) and not np.isclose(psi @ psi, 1)

    for theta in (THETA, np.pi):
        primed = np.exp(1j * theta) * psi
        assert np.isclose(np.vdot(primed, primed), 1)
        assert not np.allclose(primed, psi)                    # a different column
        for b in (ket(0), ket(1), PLUS, MINUS):
            # the call gains the phase, the echo gives it back
            assert np.isclose(np.vdot(b, primed), np.exp(1j * theta) * np.vdot(b, psi))
            assert np.isclose(np.vdot(primed, b), np.exp(-1j * theta) * np.vdot(psi, b))
            assert np.isclose(born(primed, b), born(psi, b))
        # and for effects drawn at random, a hundred of them
        for _ in range(100):
            b = random_state(rng)
            assert np.isclose(born(primed, b), born(psi, b))
    assert np.allclose(np.exp(1j * np.pi) * psi, -psi)

    # the contrast: a sign on ONE entry is caught by the turned question
    flipped = np.array([psi[0], -psi[1]])
    # the two Born values differ by 2*Re(conj(alpha)*beta), which is not zero here
    assert not np.isclose((psi[0].conj() * psi[1]).real, 0)
    assert np.isclose(born(flipped, ket(0)), born(psi, ket(0)))
    assert np.isclose(born(flipped, ket(1)), born(psi, ket(1)))
    assert not np.isclose(born(flipped, PLUS), born(psi, PLUS))

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws its props with six macros
# (pdf/props/ch02.tex; the rules are in pdf/props/ch02.lua):
#   \echostrings[strip]{pale}{dark}{caption}           the two strings and their tags
#   \crosssheet[opts]{arrow}{position}{read}{caption}  the pale string's reading, close up
#   \crosssteps[opts]{arrow}{position}{read}{steps}{caption; ...}  a step strip of it
#   \tailontail[show]{start}{closing}{caption}         the dark string's reading
#   \sharestrip[opts]{share}{caption}                  a strip, k parts in n shaded
#   \fivesquare[opts]{laid}{caption}                   the five-by-five outline
# The Lua COMPUTES what a picture shows -- the dark string's tags under "echo",
# where a tag hangs, every reach with its sign, where the card's foot falls, how
# many matches and snapped pieces, where the little squares lie, the cut and the
# halves closed round the slant, how many parts of a strip are shaded, how the
# outline fills -- and, for round 4, every frame of a step strip from the frame
# before, the step just made (the swung sheet's old lines and its arcs, the
# square turned end for end, the halves that moved, the squares taken from their
# blocks), and, pale, what a named wrong rule predicts (expects=).  The dialogue
# never writes a reading by hand.  tools/props_ch02.py transcribes the Lua, all
# but the drawing; the calls are read out of this challenge's section of
# dialog.md (tools/dfp_props.py), never copied, so an edit to the dialogue that
# breaks a picture breaks here.  Every picture is checked against the model below.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("echostrings", "crosssheet", "crosssteps", "tailontail",
                                      "sharestrip", "fivesquare"))
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")
# the starred pictures of identity.md section 3e.2: (challenge, block, call)
KEYED = [("First", 0, 0), ("First", 1, 0), ("Second", 1, 0), ("Second", 3, 1),
         ("Fourth", 0, 0), ("Sixth", 2, 0)]


def check_whole_dialogue():
    """The six challenge sections, in order, every picture in one of them; 21
    blocks, 27 calls, 31 frames; and the keyed rule -- a picture is starred
    exactly when it is the first of its prop or of its labelling, predictions
    and the sheet with no line names left out -- on the six of the card only."""
    total, frames, starred = P.check_dialogue(DIALOGUE, ORDINALS)
    assert (total, frames) == (27, 31), (total, frames)
    assert [len(DIALOGUE.blocks(o)) for o in ORDINALS] == [2, 4, 3, 4, 4, 4]
    assert starred == KEYED, starred


# --- the state model: the blackboard's own vectors, in matches ----------------
# An arrow is a real state times its drawn length: the three-and-four arrow is
# five times T, heads-ward and tails-ward five times ket(0) and ket(1), the
# slant and the back-slant root two times PLUS and MINUS.  A line of the sheet
# is a question's answer: the square sheet asks ket(0), ket(1); the turned
# sheet MINUS (line A, where heads-ward swung to) and PLUS (line B).  A reach is
# the amplitude times the drawn length, sign included; its square over the
# length squared is the Born value.  Turning a square end for end is a -> -a.
MODEL = {"three-and-four": 5 * T, "heads-ward": 5 * ket(0), "tails-ward": 5 * ket(1),
         "slant": np.sqrt(2) * PLUS, "back-slant": np.sqrt(2) * MINUS}
# the sheet's positions, as turns of the square sheet (degrees, anticlockwise
# seen from above): turned is a half right angle the other way, clockwise
ANGLE = {"square": 0, "turned": -45, "own": 20}


def rotated(v, degrees):
    t = math.radians(degrees)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]) @ v


def model_lines(position, arrow=None):
    """The two answers a position of the sheet asks, as unit columns."""
    if position == "square":
        return ket(0), ket(1)
    if position == "turned":
        return MINUS, PLUS
    if position == "own":
        return rotated(ket(0).real, 20).astype(complex), rotated(ket(1).real, 20).astype(complex)
    u = arrow / np.linalg.norm(arrow)                  # along: the arrow's own line
    return u, rotated(u.real, 90).astype(complex)


def amplitude_reach(arrow, line):
    """The reach, from the blackboard: <line|arrow>, the arrow at its length."""
    r = np.vdot(line, arrow)
    assert np.isclose(r.imag, 0)                       # real arrows: no conjugate to see
    return r.real


def same_arrow(a, vec):
    """A parsed arrow is the model's column, length and direction."""
    return np.allclose(a.v, vec.real) and np.allclose(vec.imag, 0)


def model_vec(word):
    """The model's column for an arrow word, end for end as often as it says."""
    s, flips = P.norm(word), 0
    while s.endswith(" end-for-end"):
        s, flips = s[:-len(" end-for-end")], flips + 1
    if s == "reversed":
        s, flips = "three-and-four", flips + 1
    return (-1) ** flips * MODEL[s]


WORDS = list(P.BASE) + ["reversed"] + [w + " end-for-end" for w in P.BASE] + ["reversed end-for-end"]


def check_rules():
    """The transcription against the model, on every arrow, turned end for end
    or not, under every position of the sheet and every legal reading."""
    for w in WORDS:
        a = P.clean(P.parse_arrow, w, "test")
        vec = model_vec(w)
        flips = w.count("end-for-end") + w.startswith("reversed")
        assert same_arrow(a, vec), w
        assert np.isclose(a.L, np.linalg.norm(vec))
        # end for end: the square turned half round about its centre, so the
        # tail goes to the opposite corner and the arrow points the other way
        S, m = 1.0, 0.1
        t0 = np.array(P.tail_on_square(P.BASE[a.name][2], S, m))
        t1 = np.array(P.tail_on_square(a.corner, S, m))
        assert np.allclose(t1, t0 if flips % 2 == 0 else np.array([S, S]) - t0), w
        for pos in ("square", "turned", "own", "along"):
            lines = model_lines(pos, vec)
            words_here = list(P.LINE[pos])
            reads = ["both", "none"] + words_here + [",".join(words_here[::-1])]
            for rd in reads:
                for extra in ("", ",squares", ",halves", ",cut", ",both-ways", ",no-matches", ",arrow"):
                    spec = rd + extra
                    cs, errors = P.run(P.crosssheet, w, pos, spec, True, "")
                    reaches = [amplitude_reach(vec, ln) for ln in lines]
                    off_whole = [l for l, r in zip("AB", reaches)
                                 if l in cs.read and abs(r - round(r)) > 1e-6]
                    want = []
                    if "squares" in extra:
                        want += ["crosssheet: squares only on a whole reach of matches"] * len(off_whole)
                    if extra in (",halves", ",cut") and not all(np.isclose(abs(r), 1) for r in reaches):
                        want.append("crosssheet: the halves (and the cut) are for an arrow whose "
                                    "two reaches are one match each")
                    assert errors == want, (w, pos, spec, errors)
                    if want:                                   # the fault is not drawn, and stops the build
                        assert not any(l in cs.squares for l in off_whole)
                        assert cs.halves is None and cs.cut is None
                        continue
                    for l, ln in zip("AB", lines):
                        assert np.allclose(cs.e[l], ln.real), (w, pos)
                        assert np.isclose(cs.reach[l], amplitude_reach(vec, ln)), (w, pos, l)
                        # the reach squared, over the length squared, is the Born value
                        assert np.isclose(cs.reach[l] ** 2 / cs.arrow.L ** 2,
                                          abs(np.vdot(ln, vec / np.linalg.norm(vec))) ** 2)
                    # the squares on the two reaches make the square on the arrow
                    assert np.isclose(cs.reach["A"] ** 2 + cs.reach["B"] ** 2, a.L ** 2)
                    card = cs.card
                    if card is not None:
                        e = cs.e[card.line]
                        assert np.isclose((a.v - card.foot) @ e, 0)          # dropped square
                        assert (card.out < 0) == (cs.reach[card.line] < -1e-6)
                        assert card.line == cs.order[-1]                     # at the line read last
                    for l, row in cs.rows.items():
                        r = cs.reach[l]
                        assert np.isclose(sum(b - f for f, b, _ in row), r)  # matches make the reach
                        assert sum(not s for _, _, s in row) == math.floor(abs(r) + 1e-6)
                        assert all((b - f) * r > 0 for f, b, _ in row)      # laid the reach's way
                        if l in cs.back_rows:                               # both-ways: the other side
                            back = cs.back_rows[l]
                            assert len(back) == len(row)
                            assert all(np.isclose(f2, -f) and np.isclose(b2, -b) and s2 == s
                                       for (f, b, s), (f2, b2, s2) in zip(row, back))
                    for l, sq in cs.squares.items():
                        r = cs.reach[l]
                        assert len(sq) == round(r) ** 2                     # side |reach|, count reach^2
                        c_tip = cross(cs.e[l], a.v)
                        tip_side = 0 if abs(c_tip) < 1e-9 else np.sign(c_tip)
                        for c in sq:
                            assert np.sign(c @ cs.e[l]) == np.sign(r)       # behind, if backwards
                            if tip_side != 0:
                                assert np.sign(cross(cs.e[l], c)) == -tip_side   # away from the arrow
                    if cs.halves is not None:
                        pieces = cs.halves.pieces
                        area = sum(abs(cross(p - c0, q - c0)) / 2 for c0, p, q in pieces)
                        assert np.isclose(area, a.L ** 2)                   # nothing over
                        assert np.isclose(area, cs.reach["A"] ** 2 + cs.reach["B"] ** 2)
                        assert all(np.isclose(abs(cross(p - c0, q - c0)) / 2, 0.5) for c0, p, q in pieces)
                    if cs.cut is not None:                                  # the cut, where they lay
                        assert len(cs.cut) == 4
                        assert all(np.allclose(tri[0], 0) for tri in cs.cut)  # each through the crossing
                        assert np.isclose(sum(abs(cross(q - p, s - p)) / 2 for p, q, s in cs.cut), 2)
        # the strings: an echo is the pale pair with its ends exchanged, and the
        # dark string's number is the pale string's, sign included
        checked = []
        for w2 in WORDS:
            b = P.clean(P.parse_arrow, w2, "test")
            Lp = P.pair_length(a, b)
            assert any(np.isclose(Lp, x.L) for x in (a, b))         # one of the two's own lengths
            if np.isclose(a.L, b.L):
                assert np.isclose(Lp, a.L)
            if not np.isclose(Lp, a.L):
                continue                                   # a question is as long as its start
            checked.append(w2)
            es = P.clean(P.echostrings, "", w + "/" + w2, "echo", False)
            ds, dc = es.tags["dark start"].arrow, es.tags["dark closing"].arrow
            assert ds == b and dc == a
            pale = float(a.v @ b.u)                        # the start's reach along the closing line
            dark = P.clean(P.tailontail, w2, w, False, "").number   # the dark start laid over its closing
            assert np.isclose(pale, dark), (w, w2)
            assert np.isclose(pale, amplitude_reach(vec, model_vec(w2) / np.linalg.norm(model_vec(w2))))
        assert w in checked                                # every arrow asked about itself
    # every other fault the macros know, each raised once and only there
    for fn, args in ((P.parse_arrow, ("three-and-five", "test")),
                     (P.crosssheet, ("slant", "sideways", "both")),
                     (P.crosssheet, ("slant", "square", "slant")),
                     (P.crosssheet, ("slant", "turned", "heads-ward")),
                     (P.crosssheet, ("slant", "square", "both,squares,halves")),
                     (P.crosssheet, ("slant", "off", "both")),
                     (P.echostrings, ("", "three-and-four/", "echo")),
                     (P.echostrings, ("one in two", "bare", "bare")),
                     (P.sharestrip, ("nine in thirty",)),
                     (P.sharestrip, ("ten in nine",)),
                     (P.fivesquare, ("twenty-six",))):
        assert len(P.faults(fn, *args)) == 1, (fn.__name__, args)
    for k in range(26):
        f = P.clean(P.fivesquare, str(k), False, "")
        assert len(f.laid) + f.left9 + f.left16 == 25                 # none over, none short
        assert len(set(f.laid)) == k and all(0 <= x < 5 and 0 <= y < 5 for x, y in f.laid)
        assert f.left9 == 9 - min(k, 9)                                # the three-by-three first
    for n in range(1, 26):
        for k in range(n + 1):
            s = P.clean(P.sharestrip, "%d in %d" % (k, n), False, "")
            assert sum(s.shaded) == k and list(s.shaded) == sorted(s.shaded, reverse=True)


def cross(e, v):
    return e[0] * v[1] - e[1] * v[0]


def laid(row):
    """A row of matches, read as the dialogue reads it: how many whole matches,
    how long the snapped piece (0 if none), and which way (+1 forward, -1
    backwards, 0 for no row at all)."""
    row = row or []
    whole = sum(not s for _, _, s in row)
    piece = sum(abs(b - f) for f, b, s in row if s)
    way = int(np.sign(row[0][1] - row[0][0])) if row else 0
    return whole, round(piece, 2), way


def section_dumps(found):
    """The log lines a section's pictures write, in order."""
    return [line for b in found for _, fr in b for pic, _ in fr for line in pic.dump]


QUOTE = __doc__.split('"')[1]


def end_for_end(vec):
    """The model's end-for-end turn: the square half round about its centre, a -> -a."""
    return -vec


def check_turned_square(pic, vec):
    """A square just turned end for end, against the model: the arrow inked as it
    now lies is the turn of the arrow it was, drawn dotted where it lay; the
    half-turn about the square's centre carries the old tail onto the new tail
    and the old tip onto the new tip; the turn is clockwise seen from above."""
    g = pic.ghost
    assert g is not None and pic.read == frozenset() and pic.rows == {} and pic.card is None
    assert same_arrow(pic.arrow, vec)
    was = np.array(g.arrow, dtype=complex)
    assert np.allclose(end_for_end(was), vec)                          # a -> -a
    assert g.corner == P.OPP[pic.arrow.corner]                         # its tail at the other corner
    AL = P.close_len(pic.arrow.L)
    assert np.allclose(2 * g.centre - g.tail, 0)                       # old tail -> new tail
    assert np.allclose(2 * g.centre - g.tip, AL * pic.arrow.u)         # old tip -> new tip
    assert g.sweep == -180                                             # half a turn, clockwise
    return was


def check_round4_rules():
    """The end-for-end turn on every five-long arrow, with the sheet lifted or laid,
    alone or as a step of a strip; flip_arrow is the model's a -> -a; a slant's
    square is not drawn turned; a turned square has nothing read on it yet."""
    for w in WORDS:
        vec = model_vec(w)
        assert np.allclose(model_vec(P.flip_arrow(w)), end_for_end(vec)), w
        assert P.norm(P.flip_arrow(P.flip_arrow(w))) == P.norm(w) or \
            np.allclose(model_vec(P.flip_arrow(P.flip_arrow(w))), vec)
        a = P.clean(P.parse_arrow, w, "test")
        if a.L < 2:
            assert P.faults(P.crosssheet, w, "off", "none", False, "endforend=on")
            continue
        for pos in ("off", "square", "turned"):
            pic = P.clean(P.crosssheet, w, pos, "none", False, "endforend=on")
            was = check_turned_square(pic, vec)
            assert np.allclose(was, model_vec(P.flip_arrow(w)))
            assert pic.shifted == (pos == "off")                       # no sheet: the view stays put
        frames = P.clean(P.crosssteps, P.flip_arrow(w), "off", "none", "end-for-end, lay square, read both",
                         "", "a; b; c; d", False)
        start, turned, put, read_ = (f.picture for f in frames)
        assert same_arrow(start.arrow, end_for_end(vec)) and start.ghost is None
        check_turned_square(turned, vec)
        assert put.position == "square" and same_arrow(put.arrow, vec) and put.ghost is None
        assert np.allclose([read_.reach[l] for l in "AB"],
                           [amplitude_reach(vec, ln) for ln in model_lines("square")])
    assert P.faults(P.crosssheet, "reversed", "off", "both", False, "endforend=on")   # nothing read yet
    assert P.faults(P.crosssheet, "reversed", "off", "both", False, "")              # no sheet, no reading
    assert P.faults(P.crosssteps, "reversed", "square", "none", "lay square", "", "a; b")


def pictures():
    P.check_lua()                     # ch02.lua and ch02.tex as the transcription was proved
    check_whole_dialogue()
    check_rules()
    check_round4_rules()
    # Four moments.  Block 5.0: the square just turned end for end, the sheet
    # lifted aside, the arrow as it lay dotted.  P5.1-P5.3: the reversed arrow
    # read under the square sheet, the turned sheet and Achilles's own.
    found = P.check_section(DIALOGUE, "Fifth", sizes=[1, 1, 1, 1], frames=[1, 1, 1, 1])
    (c_turn, [(turn, _)]), = found[0]
    calls = [c for b in found for c, _ in b]
    sq, tn, ow = (fr[0][0] for b in found[1:] for _, fr in b)
    REV = -5 * T
    for c in calls:
        assert c.macro == "crosssheet" and not c.keyed and not P.is_prediction(c), c.text

    # Block 5.0: the same square as P4.4's (last evening's arrow), turned end for
    # end in place: the reversed arrow inked, the old one dotted where it lay.
    assert turn.position == "off" and not turn.sheet_on and turn.shifted
    before = P.picture([c for blk in DIALOGUE.blocks("Fourth") for c in blk][-1])
    assert same_arrow(before.arrow, 5 * T)                             # P4.4: last evening's arrow
    was = check_turned_square(turn, REV)
    assert np.allclose(was, before.arrow.v) and np.allclose(end_for_end(before.arrow.v), turn.arrow.v)
    assert P.norm(c_turn.args[0]) == P.flip_arrow(before.arrow.name)   # flip_arrow of the square it was
    assert turn.arrow.corner == "fr" and turn.ghost.corner == before.arrow.corner == "nl"
    assert np.isclose(turn.arrow.L, before.arrow.L)                    # its length unchanged

    for c in calls[1:]:
        a = P.clean(P.parse_arrow, c.args[0], "test")
        # the square turned end for end: the arrow points the other way, its
        # tail at what had been the far corner, the square tied on by it
        assert same_arrow(a, REV) and a.corner == "fr" and a == turn.arrow   # the next pictures: this square
        assert not same_arrow(a, 5 * T)

    def as_drawn(position, read):
        return P.clean(P.crosssheet, "three-and-four", position, read, False, "")

    # P5.1: the square sheet: three backwards and four backwards, both feet
    # behind the crossing; the nine and the sixteen built again, each behind its
    # backwards reach -- the same blocks, only their side has changed.
    assert sq.position == "square"
    assert np.isclose(sq.reach["A"], amplitude_reach(REV, ket(0))) and np.isclose(sq.reach["A"], -3)
    assert np.isclose(sq.reach["B"], amplitude_reach(REV, ket(1))) and np.isclose(sq.reach["B"], -4)
    assert laid(sq.rows["A"]) == (3, 0, -1) and laid(sq.rows["B"]) == (4, 0, -1)
    assert sq.card is None
    assert (len(sq.squares["A"]), len(sq.squares["B"])) == (9, 16)
    for l in "AB":
        assert all(c @ sq.e[l] < 0 for c in sq.squares[l])           # the far side of the crossing
    t = as_drawn("square", "both,squares")
    for l in "AB":
        assert np.isclose(sq.reach[l], -t.reach[l])                   # every reach backwards
        assert {tuple(np.round(-c, 6)) for c in t.squares[l]} == \
               {tuple(np.round(c, 6)) for c in sq.squares[l]}        # the same block, turned round

    # P5.2: the turned sheet: just short of five, backwards, along the slant's
    # line, and the sliver forwards now.  The same shares as before.
    assert tn.position == "turned"
    rA, rB = tn.reach["A"], tn.reach["B"]
    assert np.isclose(rB, amplitude_reach(REV, PLUS)) and round(rB, 2) == -4.95   # identity 3b
    assert np.isclose(rA, amplitude_reach(REV, MINUS)) and round(rA, 2) == 0.71
    assert laid(tn.rows["B"]) == (4, 0.95, -1) and laid(tn.rows["A"]) == (0, 0.71, 1)
    assert tn.card.line == "B" and tn.card.out == -1
    t = as_drawn("turned", "both")
    for l in "AB":
        assert np.isclose(tn.reach[l], -t.reach[l])
        assert np.isclose(tn.reach[l] ** 2, t.reach[l] ** 2)          # all but a sliver, as before

    # P5.3: Achilles's own position -- the square sheet swung a little way
    # round, the heads-ward line tipped a little away from them -- and the
    # both-ways row for the square turned back: the same lengths, every one on
    # the other side of the crossing.
    assert ow.position == "own" and 0 < ow.theta < 45 and ow.e["A"][1] > 0
    lines = model_lines("own")
    rA, rB = ow.reach["A"], ow.reach["B"]
    assert np.isclose(rA, amplitude_reach(REV, lines[0])) and round(rA, 2) == -4.19   # identity 3b
    assert np.isclose(rB, amplitude_reach(REV, lines[1])) and round(rB, 2) == -2.73
    assert laid(ow.rows["A"]) == (4, 0.19, -1) and laid(ow.rows["B"]) == (2, 0.73, -1)
    assert laid(ow.back_rows["A"]) == (4, 0.19, 1) and laid(ow.back_rows["B"]) == (2, 0.73, 1)
    t = as_drawn("own", "both")
    for l in "AB":
        assert np.isclose(ow.reach[l], -t.reach[l])                   # the row for the square turned back
        assert np.isclose(sum(b - f for f, b, _ in ow.back_rows[l]), t.reach[l])
        assert np.isclose(abs(np.vdot(lines["AB".index(l)], REV)) ** 2,
                          abs(np.vdot(lines["AB".index(l)], 5 * T)) ** 2)   # I can't catch it
    assert np.isclose(rA ** 2 + rB ** 2, 25)
    assert ow.card.line == "B" and ow.card.out == -1

    assert section_dumps(found) == [
        "crosssheet: -3,-4 off reach -3.00,-4.00 read none card none",
        "crosssheet: + endforend ghost 3,4",
        "crosssheet: -3,-4 square reach -3.00,-4.00 read both,squares card none",
        "crosssheet: -3,-4 turned reach 0.71,-4.95 read both card B",
        "crosssheet: -3,-4 own reach -4.19,-2.73 read both,both-ways card B"]

    # where they stand, in both dialogue files
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a_turn, a_sq, a_tn, a_ow = DIALOGUE.anchors("Fifth", path)
        f_turn = DIALOGUE.following("Fifth", path)[0]
        assert a_turn.endswith("Every reach backwards at once.")
        for phrase in ("She clears the matches away and lifts the ruled sheet aside",
                       "turns the square end for end on the table, clockwise",
                       "the tip swung down toward where the tail was",
                       "the ink not redrawn and its length unchanged"):
            assert phrase in a_turn, phrase
        assert f_turn.startswith("**Achilles:** Then it will be caught at once.")
        assert "then he lays the sheet square" in f_turn                  # laid again, after the frame
        assert a_sq.endswith("Nine and sixteen. My first strip.")
        assert a_tn.endswith("All but a sliver, as before.")
        assert a_ow.endswith("I can't catch it!")
        assert DIALOGUE.following("Fifth", path)[2] == "**Tortoise:** " + QUOTE


def basis_state(b):
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    n = p.x(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def turned_state(b):
    p = Picture()
    n = p.z(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def generic_state():
    p = Picture()
    n = p.x(0, 1, 0)
    a, b, c = PHASES
    chain = [p.z(0, 2, a), p.x(0, 3, b), p.z(0, 4, c)]
    p.wire(n, *chain, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def with_phase(g, half_turns):
    """The state picture multiplied by a number: e^{i pi * half_turns}."""
    h = g.copy()
    h.scalar.add_phase(Fraction(half_turns))
    return h


def echo(state, b):
    return state + b.adjoint() + b + state.adjoint()


def sketchbook():
    psi = generic_state()
    col = matrix_of(psi).ravel()

    for half_turns, theta in ((Fraction(2, 7), THETA), (Fraction(1), np.pi)):
        primed = with_phase(psi, half_turns)
        factor = np.exp(1j * theta)
        assert np.allclose(matrix_of(primed).ravel(), factor * col)

        # two different pictures -- the comparison keeps every number
        assert not same_picture(primed, psi)

        for b in (basis_state(0), basis_state(1), turned_state(0), turned_state(1)):
            left = matrix_of(primed + b.adjoint())[0, 0]
            right = matrix_of(b + primed.adjoint())[0, 0]
            assert np.isclose(left, factor * matrix_of(psi + b.adjoint())[0, 0])
            assert np.isclose(right, np.conj(factor) * matrix_of(b + psi.adjoint())[0, 0])
            # and the two Echo pictures are the same number, exactly
            assert same_picture(echo(primed, b), echo(psi, b))
            assert np.isclose(matrix_of(echo(primed, b))[0, 0], left * right)

    # half a turn is the arrow reversed: exactly minus the column
    assert np.allclose(matrix_of(with_phase(psi, Fraction(1))).ravel(), -col)


if __name__ == "__main__":
    verdict("Reversed", blackboard, sketchbook)
