"""Challenge 6: The Whole Square
==================================
"None over, none short. ...Oh. A string tagged at both ends is a number, and
the number may be backwards. The number with its twin from the dark string is a
square of paper, and that never is."  -- Achilles

Claim: a state meets its own mirror image in the number 1 -- and therefore the
chances it gives make up one whole for every question there is.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import math
from fractions import Fraction

import numpy as np
import pyzx as zx
from dfp import (Picture, ket, matrix_of, random_matrix, random_state,
                 same_picture, verdict)
from dfp_props import Dialogue
import props_ch02 as P

PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)
T = np.array([3 / 5, 4 / 5], dtype=complex)
CHI = np.array([3 / 5, 4j / 5], dtype=complex)      # blackboard only
PHASES = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))


def born(state, b):
    return (np.vdot(state, b) * np.vdot(b, state)).real


def blackboard():
    rng = np.random.default_rng(26)
    psi = random_state(rng)
    assert not np.allclose(psi.imag, 0) and not np.allclose(psi, psi.conj())

    # the theorem: the conjugated row meets its column in one
    assert np.isclose(np.vdot(psi, psi), 1)
    # the plain row does not: the unconjugated mirror fails
    assert not np.isclose(psi @ psi, 1)
    assert np.isclose(np.vdot(CHI, CHI), 1) and np.isclose(CHI @ CHI, -7 / 25)
    assert np.isclose(np.vdot(T, T), 1)

    # the arrow asked about itself: five in five, and the square on it is
    # made of the nine and the sixteen
    assert 9 + 16 == 25
    assert np.isclose(born(T, ket(0)) + born(T, ket(1)), 1)
    # the dark string repeats the pale string's number, sign included, because
    # T is real -- here for the answer that reaches backwards
    assert np.isclose(np.vdot(MINUS, T), np.vdot(T, MINUS))
    assert np.isclose(np.vdot(MINUS, T), -1 / (5 * np.sqrt(2)))
    # for a state with a non-real entry the two halves are conjugates, not equal
    assert np.isclose(np.vdot(PLUS, CHI), np.conj(np.vdot(CHI, PLUS)))
    assert not np.isclose(np.vdot(PLUS, CHI), np.vdot(CHI, PLUS))

    # the corollary: the totals make one whole for every question.
    # the standard basis, the turned basis, and a basis drawn at random
    random_q, _ = np.linalg.qr(random_matrix(rng))
    assert np.allclose(random_q.conj().T @ random_q, np.eye(2))
    assert not np.allclose(random_q.imag, 0)                   # a genuinely complex basis
    for basis in ([ket(0), ket(1)], [PLUS, MINUS],
                  [random_q[:, 0], random_q[:, 1]]):
        gram = np.array([[np.vdot(bi, bj) for bj in basis] for bi in basis])
        assert np.allclose(gram, np.eye(2))
        for state in (psi, T, CHI):
            total = sum(born(state, b) for b in basis)
            assert np.isclose(total, np.vdot(state, state).real)
            assert np.isclose(total, 1)
    # and the expansion the corollary rests on
    for basis in ([PLUS, MINUS], [random_q[:, 0], random_q[:, 1]]):
        assert np.allclose(sum(np.vdot(b, psi) * b for b in basis), psi)

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


def check_mirror_gesture(es, string):
    """The mirror gesture, as the macro computes it: a start tag knotted by its
    near left-hand corner, standing off beyond the string; the closing tag the
    same kind of square carried AS IT LIES -- its arrow neither turned round nor
    turned over -- knotted by the far right-hand corner, so it hangs on the
    near side.  Turn the string's picture half round about its midpoint and
    the two squares change places, while the arrows do not turn."""
    (x0, x1) = es.strings[0 if string == "pale" else 1]
    start, closing = es.tags[string + " start"], es.tags[string + " closing"]
    Lp = es.lengths[0 if string == "pale" else 1]
    assert np.isclose(Lp, start.arrow.L)                    # a question is as long as its start
    assert np.isclose(start.drawn, closing.drawn)           # both drawn as long as each other
    assert (start.square[:, 1] > 0).all()                   # beyond the string
    assert (closing.square[:, 1] < 0).all()                 # the near side, toward the two of them
    assert np.allclose(start.knot, [x0 + 0.07, 0]) and np.allclose(closing.knot, [x1 - 0.07, 0])
    assert np.allclose(start.knot, start.square[0] - [0, 0.012])        # near left-hand corner
    assert np.allclose(closing.knot, closing.square[2] + [0, 0.012])    # far right-hand corner
    # the two squares are each other's half-turn about the string's midpoint
    half = {tuple(np.round([x0 + x1 - x, -y], 9)) for x, y in start.square}
    assert half == {tuple(np.round(p, 9)) for p in closing.square}
    # ... but each arrow keeps its place in its own square and its heading
    for tag in (start, closing):
        S = tag.square[2][0] - tag.square[0][0]
        assert np.allclose(tag.tail - tag.square[0], P.tail_on_square(tag.arrow.corner, S, 0.09 * S))
        assert np.allclose(tag.head - tag.tail, tag.drawn * tag.arrow.u)
    # the half-turn would have turned the closing arrow round: it did not
    turned = -(start.head - start.tail)
    if start.arrow.name == closing.arrow.name:
        assert not np.allclose(closing.head - closing.tail, turned)
        assert np.allclose(closing.head - closing.tail, -turned)


def section_dumps(found):
    """The log lines a section's pictures write, in order."""
    return [line for b in found for _, fr in b for pic, _ in fr for line in pic.dump]


QUOTE = " ".join(__doc__.split('"')[1].split())


def model_relay(k0, k):
    """The model's re-laying, from k0 squares in the outline to k: the squares go
    in row by row from the near left-hand corner (the j-th laid in cell
    (j mod 5, j div 5)); the three-by-three is broken up first, then the
    four-by-four, each block emptied from its last square.  Returns the cells
    filled, what is left of each block, and the squares taken since k0 as
    (block, index in it)."""
    cells = [(j % 5, j // 5) for j in range(k)]
    left = (9 - min(k, 9), 16 - max(0, k - 9))
    taken = {(9, 8 - j) if j < 9 else (16, 15 - (j - 9)) for j in range(k0, k)}
    return cells, left, taken


def block_cells(which):
    """Where each block's squares stand beside the outline: the three-by-three
    nearer, the four-by-four beyond it, both on the near edge."""
    side, x0 = (3, 6.2) if which == 9 else (4, 9.8)
    return {(x0 + c, r) for c in range(side) for r in range(side)}


def check_outline(f, k, k0=None):
    cells, left, taken = model_relay(k0 or 0, k)
    assert list(f.laid) == cells and (f.left9, f.left16) == left
    assert len(f.laid) + f.left9 + f.left16 == 25                     # none over, none short
    if k0 is None:
        assert f.since is None and f.ghosts == () and f.arc is None
        return
    assert f.since == k0 and len(f.ghosts) == k - k0
    assert {(b, i) for b, i, _ in f.ghosts} == taken                  # the squares taken, ghosted
    for b, i, place in f.ghosts:
        assert tuple(np.round(place, 9)) in {tuple(np.round(c, 9)) for c in block_cells(b)}
    start, end = f.arc
    assert start[0] > 5 > end[0]                                      # from the blocks to the outline


def check_round4_rules():
    """Every re-laying step of the outline, from any count to any greater, against
    the model; from= must be fewer than now."""
    for k in range(26):
        check_outline(P.clean(P.fivesquare, str(k), False, ""), k)
        for k0 in range(k):
            check_outline(P.clean(P.fivesquare, str(k), False, "from=%d" % k0), k, k0)
        assert P.faults(P.fivesquare, str(k), False, "from=%d" % k)
    for which in (9, 16):                                             # a whole block, ghosted
        n = int(math.isqrt(which))
        assert {tuple(np.round(P.block_place(i, which), 9)) for i in range(which)} == \
               {tuple(np.round(c, 9)) for c in block_cells(which)} and n * n == which


def pictures():
    P.check_lua()                     # ch02.lua and ch02.tex as the transcription was proved
    check_whole_dialogue()
    check_rules()
    check_round4_rules()
    assert QUOTE in " ".join(" ".join(DIALOGUE.paragraphs("Sixth")).split())
    # Four moments.  P6.1: the sheet laid along the arrow.  P6.2: the strings.
    # P6.3, keyed: the empty outline.  Block 6.4, a step pair: the nine re-laid
    # first, then the sixteen.
    found = P.check_section(DIALOGUE, "Sixth", sizes=[1, 1, 1, 2], frames=[1, 1, 1, 2])
    (c_along, [(cs, _)]), = found[0]
    (c_strings, [(es, _)]), = found[1]
    (c_empty, [(f0, _)]), = found[2]
    (c_nine, [(f1, _)]), (c_filled, [(f2, _)]) = found[3]

    # P6.1: the sheet turned until one line lies along the arrow itself: the
    # card's foot falls on the arrow's own tip -- five in five -- and along the
    # other line, no reach at all.
    assert c_along.macro == "crosssheet" and not c_along.keyed, c_along.text
    assert cs.position == "along" and same_arrow(cs.arrow, 5 * T)
    assert np.allclose(cs.e["A"], T.real)                              # the arrow's own line
    assert cs.order[-1] == "A" and cs.card.line == "A"                 # the card at the line read last
    assert np.isclose(cs.reach["A"], amplitude_reach(5 * T, T)) and np.isclose(cs.reach["A"], 5)
    assert np.isclose(cs.reach["B"], 0)
    assert np.allclose(cs.card.foot, cs.arrow.v)                       # on the arrow's own tip
    assert laid(cs.rows["A"]) == (5, 0, 1) and laid(cs.rows.get("B")) == (0, 0, 0)
    assert np.isclose(cs.reach["A"] / cs.arrow.L, np.vdot(T, T).real)  # five parts in five

    # P6.2: the arrow at both ends of the pale string, and -- once Achilles has
    # tied the dark string's copies -- the echo of that pair on the dark.
    assert c_strings.macro == "echostrings" and not c_strings.keyed, c_strings.text
    assert P.norm(c_strings.args[1]) == "echo"
    assert set(es.tags) == {"pale start", "pale closing", "dark start", "dark closing"}
    assert all(same_arrow(t.arrow, 5 * T) for t in es.tags.values())
    for s in ("pale", "dark"):
        check_mirror_gesture(es, s)
    pale = float(es.tags["pale start"].arrow.v @ es.tags["pale closing"].arrow.u)
    dark = P.clean(P.tailontail, es.tags["dark start"].arrow.name,
                   es.tags["dark closing"].arrow.name, False, "").number
    assert np.isclose(pale, 5) and np.isclose(dark, pale) and np.isclose(pale, cs.reach["A"])

    # P6.3, keyed: the empty five-by-five, and beside it the three-by-three and
    # the four-by-four -- the very blocks laid in the Second Challenge.
    assert c_empty.macro == "fivesquare" and c_empty.keyed, c_empty.text
    check_outline(f0, 0)
    assert (f0.left9, f0.left16) == (9, 16) and f0.labels == ("the outline",)
    laid_then = [pic for b in DIALOGUE.blocks("Second") for c in b for pic, _ in P.pictures(c)
                 if c.macro in ("crosssheet", "crosssteps") and "squares" in pic.read]
    assert len(laid_then) == 1
    then = laid_then[0]
    assert (len(then.squares["A"]), len(then.squares["B"])) == (f0.left9, f0.left16)

    # Block 6.4: the blocks broken up and re-laid, row filling into row from the
    # near left-hand corner -- the nine first, the three-by-three ghosted where it
    # stood; then the sixteen, the four-by-four ghosted -- until the outline is a
    # full field and nothing is left of either block.  None over, none short.
    for c in (c_nine, c_filled):
        assert c.macro == "fivesquare" and not c.keyed, c.text
    assert P.options(c_nine)["from"] == P.norm(c_empty.args[0])        # from P6.3's state
    assert f1.since == len(f0.laid) == 0 and f2.since == len(f1.laid) == 9   # each from the one before
    check_outline(f1, 9, 0)
    check_outline(f2, 25, 9)
    assert set(f1.laid) == {(x, 0) for x in range(5)} | {(x, 1) for x in range(4)}   # the near row and four
    assert {b for b, _, _ in f1.ghosts} == {9} and len(f1.ghosts) == 9 and (f1.left9, f1.left16) == (0, 16)
    assert {b for b, _, _ in f2.ghosts} == {16} and len(f2.ghosts) == 16 and (f2.left9, f2.left16) == (0, 0)
    assert sorted(f2.laid) == [(x, y) for x in range(5) for y in range(5)]
    assert len(f2.laid) == f0.left9 + f0.left16 == 25
    assert np.isclose(len(f2.laid), cs.reach["A"] ** 2)                # the square on the arrow
    assert np.isclose(len(f2.laid), 25 * np.vdot(T, T).real)           # asked about itself: one whole
    assert f1.deep is None and f2.deep is None                         # the keyed outline stands alone

    assert section_dumps(found) == [
        "crosssheet: 3,4 along reach 5.00,0.00 read across,along card A",
        "echostrings: pale 3,4/3,4 dark 3,4/3,4 strip none",
        "fivesquare: laid 0 left 9,16",
        "fivesquare: laid 9 left 0,16 from 0 ghosts 9",
        "fivesquare: laid 25 left 0,0 from 9 ghosts 16"]

    # where they stand, in both dialogue files
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a_along, a_strings, a_empty, a_relay = DIALOGUE.anchors("Sixth", path)
        assert a_along.endswith("And along the other line, no reach at all.")
        assert a_strings.endswith("Five in five. What else could it say?")
        assert a_empty.endswith("The square on it. Fill it in.")
        assert a_relay.endswith("and the slices still come back to the one loaf.")
        for phrase in ("He breaks the three-by-three block into its nine loose squares",
                       "the nine first, from its near left-hand corner, row filling into row",
                       "nothing is left of either block"):
            assert phrase in a_relay, phrase
        assert DIALOGUE.following("Sixth", path)[3] == "**Tortoise:** Exactly?"


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


def unconjugated_mirror():
    """The generic state's chain reversed but its phases left alone: the plain
    transpose, which is not the mirror image."""
    a, b, c = PHASES
    p = Picture()
    chain = [p.z(0, 1, c), p.x(0, 2, b), p.z(0, 3, a)]
    n = p.x(0, 4, 0)
    p.wire(p.inp(0), *chain, n)
    g = p.done()
    g.scalar.add_power(-1)
    return g


def echo(state, b):
    return state + b.adjoint() + b + state.adjoint()


def sketchbook():
    psi = generic_state()
    col = matrix_of(psi).ravel()
    assert np.isclose(np.linalg.norm(col), 1)          # a state, exactly

    # the theorem as a picture: state psi, wire, effect psi
    closed = psi + psi.adjoint()
    assert closed.inputs() == () and closed.outputs() == ()
    assert np.allclose(matrix_of(closed), [[1]])
    # a picture with no vertices at all, used here only as the number 1
    number_one = zx.Graph()
    assert same_picture(closed, number_one)

    # the unconjugated mirror closes the same state on a different number
    wrong = psi + unconjugated_mirror()
    assert not np.isclose(matrix_of(wrong)[0, 0], 1)
    assert not same_picture(wrong, number_one)                # the partner for the test above

    # The Echo with b = psi is the theorem twice over
    assert np.isclose(matrix_of(echo(psi, psi))[0, 0], 1)

    # the corollary, as arithmetic on evaluated pictures: the two Echo numbers
    # of a question add to the number of the closed diagram.  Adding numbers,
    # not pictures.
    whole = matrix_of(closed)[0, 0]
    for question in ((basis_state(0), basis_state(1)),
                     (turned_state(0), turned_state(1))):
        total = sum(matrix_of(echo(psi, b))[0, 0] for b in question)
        assert np.isclose(total, whole) and np.isclose(total, 1)

    # the dark string, once more: for a real state both halves agree, sign and
    # all; for the generic state they are conjugates and differ
    minus, one_state = turned_state(1), basis_state(1)
    left = matrix_of(minus + one_state.adjoint())[0, 0]
    right = matrix_of(one_state + minus.adjoint())[0, 0]
    assert np.isclose(left, -1 / np.sqrt(2)) and np.isclose(right, left)
    gl = matrix_of(psi + one_state.adjoint())[0, 0]
    gr = matrix_of(one_state + psi.adjoint())[0, 0]
    assert np.isclose(gr, np.conj(gl)) and not np.isclose(gr, gl)


if __name__ == "__main__":
    verdict("WholeSquare", blackboard, sketchbook)
