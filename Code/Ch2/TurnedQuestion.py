"""Challenge 4: The Turned Question
=====================================
"...Oh. They were never the same start."  -- Achilles

Claim: {+, -} is an orthonormal basis -- a question like any other -- and asked
that question + and - answer with certainty, oppositely.  The sign that one
question could not see is the whole of the difference for another.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import math
from fractions import Fraction

import numpy as np
import pyzx as zx
from dfp import Picture, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch02 as P

PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)
T = np.array([3 / 5, 4 / 5], dtype=complex)
CHI = np.array([3 / 5, 4j / 5], dtype=complex)     # blackboard only
PHASES = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))


def born(state, b):
    return (np.vdot(state, b) * np.vdot(b, state)).real


def blackboard():
    # an orthonormal basis: the Gram matrix of the pair is the identity
    basis = [PLUS, MINUS]
    gram = np.array([[np.vdot(bi, bj) for bj in basis] for bi in basis])
    assert np.allclose(gram, np.eye(2))
    # the standard pair is one too
    std = [ket(0), ket(1)]
    assert np.allclose([[np.vdot(bi, bj) for bj in std] for bi in std], np.eye(2))

    # certainty, and opposite certainties
    assert np.isclose(born(PLUS, PLUS), 1) and np.isclose(born(PLUS, MINUS), 0)
    assert np.isclose(born(MINUS, PLUS), 0) and np.isclose(born(MINUS, MINUS), 1)

    # the expansion: every state is its two amplitudes times the two members
    rng = np.random.default_rng(24)
    psi = random_state(rng)
    assert not np.allclose(psi.imag, 0) and not np.isclose(psi @ psi, 1)
    rebuilt = np.vdot(PLUS, psi) * PLUS + np.vdot(MINUS, psi) * MINUS
    assert np.allclose(rebuilt, psi)
    assert np.isclose(born(psi, PLUS) + born(psi, MINUS), 1)

    # the arrow under the turned question: all but a sliver
    assert np.isclose(np.vdot(PLUS, T), 7 / (5 * np.sqrt(2)))
    assert np.isclose(np.vdot(MINUS, T), -1 / (5 * np.sqrt(2)))   # backwards
    assert np.isclose(born(T, PLUS), 49 / 50) and np.isclose(born(T, MINUS), 1 / 50)
    assert np.isclose(born(T, PLUS) + born(T, MINUS), 1)

    # the dark string says the same, sign included, because T is real
    assert np.isclose(np.vdot(MINUS, T), np.vdot(T, MINUS))

    # the conjugate has its turned question too: CHI matches T under the
    # standard question and parts from it under the turned one.  CHI is not
    # exactly representable as a picture here, so it stays on the blackboard.
    assert np.isclose(born(CHI, ket(0)), 9 / 25) and np.isclose(born(CHI, ket(1)), 16 / 25)
    assert np.isclose(born(T, ket(0)), 9 / 25) and np.isclose(born(T, ket(1)), 16 / 25)
    assert np.isclose(np.vdot(PLUS, CHI), (3 + 4j) / (5 * np.sqrt(2)))
    assert np.isclose(born(CHI, PLUS), 1 / 2)
    assert not np.isclose(born(CHI, PLUS), born(T, PLUS))

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


def angle_of(position, vec):
    """A position's turn of the square sheet, in degrees (along: the arrow's own)."""
    if position == "along":
        return math.degrees(math.atan2(vec.real[1], vec.real[0]))
    return ANGLE[position]


def model_swing(old, new, vec):
    """The model's swing about the crossing, the short way round: the signed angle
    from the old position to the new, in (-180, 180] (negative: clockwise seen
    from above); the ruled lines as they lay are the old position's answers."""
    d = (angle_of(new, vec) - angle_of(old, vec) + 180) % 360 - 180
    return (180.0 if np.isclose(d, -180) else d), model_lines(old, vec)


def as_bookkeeping(vec):
    """Achilles: "So the backwards reach is bookkeeping! ... Same strip, same
    start." -- every backwards reach counted forwards: the arrow read as its
    reaches' sizes, (|x|, |y|).  The back-slant is read as the slant."""
    return np.abs(vec.real).astype(complex)


def check_round4_rules():
    """The swing on every arrow between every two positions of the sheet, and
    the bookkeeping rule on every arrow with a backwards reach, under every
    position, along each line, against the model; and their faults."""
    for w in WORDS:
        vec = model_vec(w)
        for old in ("square", "turned", "own", "along"):
            for new in ("square", "turned", "own", "along"):
                if old == new:
                    assert P.faults(P.crosssheet, w, new, "none", False, "swung=" + old)
                    continue
                pic = P.clean(P.crosssheet, w, new, "none", False, "swung=" + old)
                d, ghosts = model_swing(old, new, vec)
                assert pic.swung == old and np.isclose(pic.swing, d)
                for l, ln in zip("AB", ghosts):                  # the old lines, where they lay
                    assert np.allclose(P.line_dir(pic.swung_theta, l), ln.real)
                for l, ln in zip("AB", model_lines(new, vec)):   # the new, where the arcs end
                    assert np.allclose(rotated(P.line_dir(pic.swung_theta, l), pic.swing), ln.real)
                assert pic.read == frozenset() and pic.rows == {} and pic.card is None
        backwards = (vec.real < -1e-9).any()
        for pos in ("square", "turned", "own", "along"):
            lines = model_lines(pos, vec)
            for word, line in P.LINE[pos].items():
                pic, errors = P.run(P.crosssheet, w, pos, word, False, "expects=bookkeeping")
                assert (errors == []) == backwards, (w, pos, word)
                want = amplitude_reach(as_bookkeeping(vec), lines["AB".index(line)])
                assert np.isclose(pic.laid[line], want) and pic.card is None and pic.expects == "bookkeeping"
    assert P.faults(P.crosssheet, "back-slant", "off", "none", False, "swung=square")
    assert P.faults(P.crosssheet, "back-slant", "turned", "none", False, "swung=off")
    assert P.faults(P.crosssheet, "back-slant", "turned", "slant", True, "expects=bookkeeping")
    assert P.faults(P.crosssheet, "back-slant", "turned", "slant", False, "expects=bookkeeping;swung=square")


def pictures():
    P.check_lua()                     # ch02.lua and ch02.tex as the transcription was proved
    check_whole_dialogue()
    check_rules()
    check_round4_rules()
    # Four moments.  Block 4.0, keyed: the sheet just swung, its old lines and
    # their arcs.  Block 4.1: the back-slant asked along the slant's line as
    # Achilles expects it, pale, beside the true reading.  Block 4.2-4.3: the
    # two strips.  P4.4: last evening's arrow under the turned sheet.
    found = P.check_section(DIALOGUE, "Fourth", sizes=[1, 2, 2, 1], frames=[1, 2, 2, 1])
    (c_swung, [(swung, _)]), = found[0]
    (c_expects, [(expected, caption)]), (c_true, [(cs, _)]) = found[1]
    (c_dark, [(dark, _)]), (c_white, [(white, _)]) = found[2]
    (c_sliver, [(t, _)]), = found[3]
    SLANT, BACK = np.sqrt(2) * PLUS, np.sqrt(2) * MINUS

    # Block 4.0, keyed: the ruled sheet swung about the crossing from where it
    # last lay -- square, under P3.3 -- to the turned position, nothing on it.
    # The heads-ward line comes round toward the two of them (clockwise from
    # above), onto the back-slant; the slant's line is the old tails-ward line
    # swung the same way.  The old lines are drawn where they lay.
    assert c_swung.macro == "crosssheet" and c_swung.keyed, c_swung.text
    before = [c for blk in DIALOGUE.blocks("Third") for c in blk][-1]
    assert before.macro == "crosssheet" and P.picture(before).arrow == swung.arrow
    assert swung.swung == P.norm(before.args[1]) == "square"           # swung from where it lay
    assert swung.position == "turned" and same_arrow(swung.arrow, BACK)
    d, ghosts = model_swing("square", "turned", BACK)
    assert np.isclose(swung.swing, d) and np.isclose(d, -45)          # an eighth, clockwise
    assert np.isclose(swung.theta, -45) and swung.e["A"][1] < 0        # heads-ward comes toward them
    for l, ln in zip("AB", ghosts):
        assert np.allclose(P.line_dir(swung.swung_theta, l), ln.real)  # heads-ward, tails-ward as they lay
    assert np.allclose(swung.e["A"], MINUS.real) and np.allclose(swung.e["B"], PLUS.real)
    assert swung.read == frozenset() and swung.rows == {} and swung.card is None   # cleared
    assert swung.labels == ("back-slant", "slant")
    assert np.allclose(swung.e["A"], P.clean(P.parse_arrow, "back-slant", "test").u)   # named truly
    assert np.allclose(swung.e["B"], P.clean(P.parse_arrow, "slant", "test").u)
    # swung the other way the back-slant's whole arrow would lie BACKWARDS along
    # its line -- the dialogue's signs fix the sense of the turn
    other = rotated(np.array([0.0, 1.0]), 45)
    assert np.isclose(abs(BACK.real @ other), swung.arrow.L) and BACK.real @ other < 0

    # Block 4.1: the prediction, pale -- the back-slant read as the slant, every
    # backwards reach counted forwards: along the slant's line the whole arrow,
    # one match and a nearly whole piece, no card -- beside the truth: no reach,
    # the card's foot on the crossing, no matches.
    assert P.is_prediction(c_expects) and not c_expects.keyed and caption == "as Achilles expects"
    assert c_true.macro == "crosssheet" and not c_true.keyed and c_true.opt == "", c_true.text
    assert (expected.arrow, expected.position, expected.read) == (cs.arrow, cs.position, cs.read)
    assert expected.expects == "bookkeeping" and expected.card is None and set(expected.rows) == {"B"}
    want = amplitude_reach(as_bookkeeping(BACK), PLUS)
    assert np.isclose(expected.laid["B"], want) and np.isclose(want, np.linalg.norm(BACK))  # the whole arrow
    assert np.isclose(expected.laid["B"], amplitude_reach(SLANT, PLUS))  # as for the slant
    assert np.isclose(expected.laid["B"], P.rule_bookkeeping(cs.arrow, cs.theta, {"B": True})["B"])
    assert laid(expected.rows["B"]) == (1, 0.41, 1)
    assert cs.read == {"B", "no-matches"} and cs.rows == {}           # card only
    assert np.isclose(cs.reach["B"], amplitude_reach(BACK, PLUS)) and np.isclose(cs.reach["B"], 0)
    assert np.allclose(cs.card.foot, 0) and cs.card.line == "B"        # on the crossing: the card
    assert not np.isclose(expected.laid["B"], cs.reach["B"])           # the prediction is refuted
    assert np.isclose(cs.reach["A"], cs.arrow.L)                       # it lies along the whole arrow
    assert np.isclose(cs.reach["A"], amplitude_reach(BACK, MINUS))     # (read aloud next: "The whole arrow")
    assert cs.labels == () and cs.swung is None                        # the key went to block 4.0

    # Block 4.2-4.3: the two strips.  Under the turned sheet the share from the
    # knot belongs to line A, where heads-ward swung to: the back-slant's strip
    # all dark, the slant's all white.
    shares = []
    for c, ss, word, vec in ((c_dark, dark, "back-slant", BACK), (c_white, white, "slant", SLANT)):
        assert c.macro == "sharestrip" and not c.keyed, c.text        # heads' share would be wrong here
        assert P.norm(c.caption) == "the " + word + "'s strip"
        on_line_a = P.clean(P.crosssheet, word, "turned", "both", False, "")
        share = on_line_a.reach["A"] ** 2 / on_line_a.arrow.L ** 2
        assert np.isclose(ss.k / ss.n, share)
        assert np.isclose(share, abs(np.vdot(MINUS, vec / np.linalg.norm(vec))) ** 2)
        shares.append(ss.k / ss.n)
    assert shares == [1, 0]                                           # all dark, all white
    assert dark.shaded == (True, True) and white.shaded == (False, False)
    # under the square sheet the two had one strip between them: one in two
    for word in ("back-slant", "slant"):
        sq = P.clean(P.crosssheet, word, "square", "both", False, "")
        assert np.isclose(sq.reach["A"] ** 2 / sq.arrow.L ** 2, 1 / 2)

    # P4.4: last evening's arrow under the turned sheet: just short of five
    # along the slant's line, and short of one match, backwards, along the
    # back-slant's.  All but a sliver -- and the sliver from a backwards reach.
    assert c_sliver.macro == "crosssheet" and not c_sliver.keyed, c_sliver.text
    assert t.position == "turned" and same_arrow(t.arrow, 5 * T)
    rA, rB = t.reach["A"], t.reach["B"]
    assert np.isclose(rB, amplitude_reach(5 * T, PLUS)) and round(rB, 2) == 4.95   # identity 3b
    assert np.isclose(rA, amplitude_reach(5 * T, MINUS)) and round(rA, 2) == -0.71
    assert 4.9 < rB < 5 and -1 < rA < 0                                # just short; short of one
    assert laid(t.rows["B"]) == (4, 0.95, 1)                           # four and a nearly whole piece
    assert laid(t.rows["A"]) == (0, 0.71, -1)                          # a piece behind the crossing
    assert t.card.line == "B" and np.allclose(t.card.foot, rB * PLUS.real)
    assert np.isclose(rB ** 2 / 25, 49 / 50) and np.isclose(rA ** 2 / 25, 1 / 50)

    assert section_dumps(found) == [
        "crosssheet: 1,-1 turned reach 1.41,0.00 read none card none",
        "crosssheet: + swung square",
        "crosssheet: 1,-1 turned reach 1.41,0.00 read slant,no-matches card none",
        "crosssheet: + expects bookkeeping laid -,1.41",
        "crosssheet: 1,-1 turned reach 1.41,0.00 read slant,no-matches card B",
        "sharestrip: 2/2",
        "sharestrip: 0/2",
        "crosssheet: 3,4 turned reach -0.71,4.95 read both card B"]

    # where they stand, in both dialogue files
    trap = "So the backwards reach is bookkeeping! A mark for the paper, and nothing to the world."
    assert sum(trap in p for p in DIALOGUE.paragraphs("Third")) == 1      # what he expected
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a_swung, a_expects, a_strips, a_sliver = DIALOGUE.anchors("Fourth", path)
        assert a_swung == ("*[The ruled sheet has come to rest a half-right-angle round, "
                           "the crossing still pinned to the arrow's tail.]*")
        # the swing itself closes the Third: the sheet cleared, then swung, the
        # crossing held fast and the heads-ward line coming round toward them
        third = [p for p in DIALOGUE.paragraphs("Third", path) if p != "---"]
        last = third[-1]
        assert last.startswith("*[She clears the matches away and the two little squares back into the dish;")
        for phrase in ("sets a fingertip on the crossing of the ruled sheet, holding it fast to the arrow's tail",
                       "the heads-ward line coming round toward the two of them",
                       "The crossing does not stir"):
            assert phrase in last, phrase
        assert max(i for i, p in enumerate(third) if DIALOGUE.is_block(p)) < len(third) - 1  # after P3.3
        assert a_expects.endswith("No matches at all. It has no reach along that line.")
        assert "Its foot falls on the crossing." in a_expects
        assert a_strips.endswith("All dark. All white.")
        assert "He shades the back-slant's strip" in a_strips and "the slant's strip white" in a_strips
        assert a_sliver.endswith("Along the back-slant's: short of one match, and backwards.")
        assert sum(QUOTE in p for p in DIALOGUE.paragraphs("Fourth", path)) == 1   # and named right


def basis_state(b):
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    n = p.x(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def turned_state(b):
    """The states + and -.  They are states, drawn like any other; nothing in
    this file builds a box that turns one question into the other."""
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


def echo(state, b):
    return state + b.adjoint() + b + state.adjoint()


def plain_wires(g):
    """No wire in the picture carries a mark of its own: every edge is plain."""
    return all(g.edge_type(e) == zx.EdgeType.SIMPLE for e in g.edges())


def sketchbook():
    plus, minus = turned_state(0), turned_state(1)
    psi = generic_state()

    # + and - are states; the box that would turn one question into the other
    # belongs to Chapter 3, and is built nowhere here
    for g in (plus, minus, psi, echo(psi, plus)):
        assert plain_wires(g)

    # the turned question, put to the two starts of the trap
    for state, want in ((plus, (1, 0)), (minus, (0, 1))):
        for b, expected in zip((plus, minus), want):
            value = matrix_of(echo(state, b))[0, 0]
            if expected == 0:
                assert np.allclose(value, 0)                  # zeros go to NumPy
            else:
                assert np.isclose(value, 1)
    # the non-zero partner for those zeros, as pictures
    assert not same_picture(echo(plus, minus), echo(plus, plus))

    # a generic state under the turned question: the picture agrees with the
    # blackboard, and the two numbers make one whole
    col = matrix_of(psi).ravel()
    total = 0
    for b, vec in ((plus, PLUS), (minus, MINUS)):
        value = matrix_of(echo(psi, b))[0, 0]
        assert np.isclose(value, abs(np.vdot(vec, col)) ** 2)
        assert np.isclose(value.imag, 0) and value.real > 0
        total += value.real
    assert np.isclose(total, 1)                               # adding numbers, not pictures

    # and the standard question gives a different pair of numbers for it
    std_total = sum(matrix_of(echo(psi, basis_state(k)))[0, 0].real for k in (0, 1))
    assert np.isclose(std_total, 1)
    assert not np.isclose(matrix_of(echo(psi, plus))[0, 0],
                          matrix_of(echo(psi, basis_state(0)))[0, 0])


if __name__ == "__main__":
    verdict("TurnedQuestion", blackboard, sketchbook)
