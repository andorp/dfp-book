"""Challenge 3: A Reach Backwards
===================================
"So the backwards reach is bookkeeping! A mark for the paper, and nothing to
the world. Same strip, same start."  -- Achilles

Claim: the states + and - give the very same wager under the standard question,
and are nevertheless perpendicular -- as different as two states can be.  One
question is not enough to tell two states apart.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
import math
from fractions import Fraction

import numpy as np
from dfp import Picture, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch02 as P

PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)
T = np.array([3 / 5, 4 / 5], dtype=complex)        # last evening's arrow
PHASES = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))


def born(state, b):
    amp = np.vdot(b, state)
    return (np.vdot(state, b) * amp).real


def blackboard():
    # both are states
    assert np.isclose(np.vdot(PLUS, PLUS), 1) and np.isclose(np.vdot(MINUS, MINUS), 1)

    # the same half-and-half strip from both, under the standard question
    for s in (PLUS, MINUS):
        assert np.isclose(born(s, ket(0)), 1 / 2)
        assert np.isclose(born(s, ket(1)), 1 / 2)
    assert np.isclose(np.vdot(ket(1), MINUS), -1 / np.sqrt(2))   # a reach backwards

    # and yet they are different columns, and perpendicular
    assert not np.allclose(PLUS, MINUS)
    assert np.isclose(np.vdot(PLUS, MINUS), 0)
    assert np.isclose(np.vdot(PLUS, PLUS), 1)                    # the non-zero partner
    assert np.isclose(np.vdot(ket(0), ket(1)), 0)                # as far apart as 0 and 1

    # in general: alpha|0> + beta|1> and alpha|0> - beta|1>, for a generic
    # complex state, share every standard Born value and are different columns
    rng = np.random.default_rng(23)
    psi = random_state(rng)
    assert not np.allclose(psi.imag, 0) and not np.isclose(psi @ psi, 1)
    flipped = np.array([psi[0], -psi[1]])
    assert not np.allclose(psi, flipped)
    for b in (ket(0), ket(1)):
        assert np.isclose(born(psi, b), born(flipped, b))
    # the sign is in the amplitude all along; only the square eats it
    assert np.isclose(np.vdot(ket(1), flipped), -np.vdot(ket(1), psi))
    assert not np.isclose(np.vdot(ket(1), flipped), np.vdot(ket(1), psi))

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


QUOTE = " ".join(__doc__.split('"')[1].split())


def centroid(tri):
    return sum(np.asarray(p, dtype=float) for p in tri) / 3


def model_halves(vec):
    """The model's cut, for an arrow one match each way: a little square on each
    reach, on the side away from the arrow (behind the crossing if the reach is
    backwards); each cut along its diagonal from the crossing to the opposite
    corner, where it lies; the four halves closed into the square on the arrow,
    on its left (tails-ward, for the slant), each half to the place that makes
    the least travel in all -- a unique least, so no tie is broken.  Returns the
    squares, the halves as cut (square A's two, then B's), the four places, and
    the place each half goes to."""
    v = vec.real
    squares, halves = [], []
    for ln in model_lines("square"):
        e, r = ln.real, amplitude_reach(vec, ln)
        assert np.isclose(abs(r), 1)
        away = -np.sign(cross(e, v)) * np.array([-e[1], e[0]])
        c = [np.zeros(2), r * e, r * e + away, away]
        squares.append(c)
        halves += [(c[0], c[1], c[2]), (c[0], c[2], c[3])]
    n = rotated(v, 90)
    corners = [np.zeros(2), v, v + n, n]
    centre = (v + n) / 2
    places = [(corners[i], corners[(i + 1) % 4], centre) for i in range(4)]
    cost = {perm: sum(np.linalg.norm(centroid(halves[i]) - centroid(places[perm[i]])) for i in range(4))
            for perm in itertools.permutations(range(4))}
    best = min(cost, key=cost.get)
    assert all(cost[p] > cost[best] + 1e-9 for p in cost if p != best)      # the shortest way, unique
    return squares, halves, corners, places, best


def same_tri(t1, t2):
    return all(np.allclose(p, q) for p, q in zip(t1, t2))


def same_place(t1, t2):
    """Two triangles covering the same ground (their corners in any order)."""
    return all(any(np.allclose(p, q) for q in t2) for p in t1)


def check_cut_strip(frames, vec):
    """A cut strip's three frames against the model: the little squares; the cut,
    where they lay; the halves closed round the arrow, with their marks."""
    squares, halves, corners, places, best = model_halves(vec)
    (sq, _), (cut, _), (closed, _) = frames
    assert sq.read == {"A", "B", "squares"} and cut.read == {"A", "B", "cut"}
    assert closed.read == {"A", "B", "halves"}
    for pic in (sq, cut, closed):
        assert same_arrow(pic.arrow, vec) and pic.card is None and pic.position == "square"
        assert pic.rows == sq.rows                                      # the matches stay
    # one little square on each reach, where the model has it
    for l, c in zip("AB", squares):
        (centre,) = sq.squares[l]
        assert np.allclose(centre, sum(c) / 4)
    # the cut: the same squares, each in two halves through the crossing, lying where they lay
    assert cut.halves is None and not cut.squares and len(cut.cut) == 4
    assert all(same_tri(t, h) for t, h in zip(cut.cut, halves))
    assert all(np.allclose(t[0], 0) for t in cut.cut)                  # corner to corner from the crossing
    # closed: the square on the arrow, nothing over
    assert closed.cut is None and not closed.squares
    assert all(np.allclose(p, q) for p, q in zip(closed.halves.corners, corners))
    area = sum(abs(cross(p - c, q - c)) / 2 for c, p, q in closed.halves.pieces)
    assert np.isclose(area, 2) and np.isclose(area, np.linalg.norm(vec) ** 2)   # (one and a bit)^2
    # the marks: the squares ghosted where they lay, one arc per half that moved
    assert closed.marks and len(closed.moves) == 4
    assert all(np.allclose(p, q) for u, c in zip(closed.unit_squares, squares) for p, q in zip(u, c))
    for i, m in enumerate(closed.moves):
        assert same_tri(m.frm, halves[i]) and same_tri(m.to, places[best[i]])
        assert m.square == "AB"[i // 2]
        assert m.moved == (not np.allclose(centroid(halves[i]), centroid(places[best[i]])))
    return closed


def check_round4_rules():
    """The cut and the closing on every arrow one match each way (the slant, the
    back-slant, and each turned end for end) against the model; the steps out of
    order are faults, and so is a cut on a reach of more than one match."""
    for w in ("slant", "back-slant", "slant end-for-end", "back-slant end-for-end"):
        frames = [(f.picture, f.caption) for f in
                  P.clean(P.crosssteps, w, "square", "both,squares", "cut, close", "", "a; b; c", False)]
        closed = check_cut_strip(frames, model_vec(w))
        assert closed.moved == sum(m.moved for m in closed.moves) and 0 < closed.moved < 4
    assert P.faults(P.crosssteps, "slant", "square", "both", "close", "", "a; b")       # nothing cut
    assert P.faults(P.crosssteps, "slant", "square", "both", "cut", "", "a; b")         # no squares
    assert P.faults(P.crosssheet, "three-and-four", "square", "both,cut", False, "")    # not one match
    assert P.faults(P.crosssheet, "slant", "square", "both,cut", False, "marks=on")     # marks on the close only


def pictures():
    P.check_lua()                     # ch02.lua and ch02.tex as the transcription was proved
    check_whole_dialogue()
    check_rules()
    check_round4_rules()
    # Three moments.  Block 3.1, a step strip: a little square on each reach;
    # the two cut corner to corner; the halves closed round the slant.  P3.2:
    # the back-slant, a reach backwards.  P3.3: a little square on each reach.
    found = P.check_section(DIALOGUE, "Third", sizes=[1, 1, 1], frames=[3, 1, 1])
    (c_cut, cut_frames), = found[0]
    (c_back, [(b, _)]), = found[1]
    (c_squares, [(s, _)]), = found[2]
    assert c_cut.macro == "crosssteps" and not c_cut.keyed, c_cut.text
    for c in (c_back, c_squares):
        assert c.macro == "crosssheet" and not c.keyed and c.opt == "", c.text
    assert not any(P.is_prediction(c) for blk in DIALOGUE.blocks("Third") for c in blk)
    SLANT, BACK = np.sqrt(2) * PLUS, np.sqrt(2) * MINUS

    # Block 3.1: the slant under the square sheet, one match each way; a little
    # square on each reach; each cut from the crossing to its far corner, where
    # it lies; the four halves closed round the slant into the square on it, on
    # its tails-ward side, nothing over -- one half stays, three move.
    cs = check_cut_strip(cut_frames, SLANT)
    sq = cut_frames[0][0]
    assert np.isclose(cs.reach["A"], amplitude_reach(SLANT, ket(0))) and np.isclose(cs.reach["A"], 1)
    assert np.isclose(cs.reach["B"], amplitude_reach(SLANT, ket(1))) and np.isclose(cs.reach["B"], 1)
    assert laid(cs.rows["A"]) == (1, 0, 1) and laid(cs.rows["B"]) == (1, 0, 1)
    assert {l: len(v) for l, v in sq.squares.items()} == {"A": 1, "B": 1}   # a square on each
    area = sum(abs(cross(p - c, q - c)) / 2 for c, p, q in cs.halves.pieces)
    assert np.isclose(area, cs.reach["A"] ** 2 + cs.reach["B"] ** 2)       # made of those two
    assert np.isclose(cs.arrow.L, np.sqrt(2)) and not np.isclose(cs.arrow.L, 2)   # not two
    assert cs.halves.centre[1] > cs.halves.centre[0]                       # the tails-ward side
    assert cross(cs.arrow.u, cs.halves.centre) > 0                         # away from the heads-ward line
    # one half stays: the tails-ward square's half already in its place
    assert cs.moved == 3 and [m.moved for m in cs.moves] == [True, True, False, True]
    (stay,) = [m for m in cs.moves if not m.moved]
    assert stay.square == "B" and same_place(stay.frm, stay.to)     # it lies in its place
    _, halves, corners, places, best = model_halves(SLANT)
    assert same_tri(stay.frm, halves[2]) and same_tri(stay.to, places[best[2]])   # the model's assignment
    assert sum(any(np.allclose(p, q) for q in corners) for p in stay.frm) == 2   # an edge of the square on it
    # it lies against the tails-ward reach -- its edge from the crossing to the
    # reach's end is the reach itself -- and not against the slant
    def has(tri, pt):
        return any(np.allclose(p, pt) for p in tri)
    tails_end = cs.reach["B"] * cs.e["B"]
    assert has(stay.frm, np.zeros(2)) and has(stay.frm, tails_end)
    assert not (has(stay.frm, np.zeros(2)) and has(stay.frm, cs.arrow.v))   # no edge along the slant
    assert all(m.moved for m in cs.moves if m.square == "A")      # the heads-ward halves both move

    # P3.2: the back-slant, drawn from the far left-hand corner: heads-ward one,
    # tails-ward one BACKWARDS -- the card's foot falls behind the crossing.
    assert b.position == "square" and same_arrow(b.arrow, BACK)
    assert b.arrow.corner == "fl"                                          # its tail at the far left
    assert np.isclose(b.reach["A"], amplitude_reach(BACK, ket(0))) and np.isclose(b.reach["A"], 1)
    assert np.isclose(b.reach["B"], amplitude_reach(BACK, ket(1))) and np.isclose(b.reach["B"], -1)
    assert b.card.line == "B" and b.card.out == -1                         # the card at the tails-ward line
    assert np.allclose(b.card.foot, [0, -1])                               # behind the crossing
    assert laid(b.rows["A"]) == (1, 0, 1) and laid(b.rows["B"]) == (1, 0, -1)

    # P3.3: the same arrow, a little square on each reach, the tails-ward one on
    # the far side of the crossing: a square like any other.
    assert s.arrow == b.arrow and same_arrow(s.arrow, BACK)
    assert np.allclose([s.reach["A"], s.reach["B"]], [1, -1]) and s.card is None
    (fore,), (behind,) = s.squares["A"], s.squares["B"]
    assert fore @ s.e["A"] > 0 and behind @ s.e["B"] < 0                  # behind the crossing
    shares = [len(s.squares[l]) / s.arrow.L ** 2 for l in "AB"]
    assert np.allclose(shares, [1 / 2, 1 / 2])                             # one in two
    assert np.allclose(shares, [abs(np.vdot(ket(k), MINUS)) ** 2 for k in (0, 1)])
    # the trap's evidence: the slant and the back-slant differ in a reach ...
    assert not np.isclose(cs.reach["B"], s.reach["B"])
    assert np.isclose(cs.reach["B"], -s.reach["B"])
    # ... and give the very same squares, so the very same strip
    assert np.isclose(cs.reach["A"] ** 2, s.reach["A"] ** 2)
    assert np.isclose(cs.reach["B"] ** 2, s.reach["B"] ** 2)
    assert np.allclose(shares, [abs(np.vdot(ket(k), PLUS)) ** 2 for k in (0, 1)])

    assert section_dumps(found) == [
        "crosssheet: 1,1 square reach 1.00,1.00 read both,squares card none",
        "crosssheet: 1,1 square reach 1.00,1.00 read both,cut card none",
        "crosssheet: 1,1 square reach 1.00,1.00 read both,halves card none",
        "crosssheet: + halves moved 3 stay 1",
        "crosssheet: 1,-1 square reach 1.00,-1.00 read both card B",
        "crosssheet: 1,-1 square reach 1.00,-1.00 read both,squares card none"]

    # where they stand, in both dialogue files
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        paras = DIALOGUE.paragraphs("Third", path)
        a_cut, a_back, a_squares = DIALOGUE.anchors("Third", path)
        assert a_cut.endswith("Made of those two, and nothing over.")
        for phrase in ("She cuts the two little squares where they lie", "corner to corner from the crossing",
                       "One half already lies in its place against the tails-ward reach", "each the shortest way",
                       "on its tails-ward side, away from the heads-ward line"):
            assert phrase in a_cut, phrase
        # the sheet bare but for the slant's squares (D7), and cleared before P3.2 (D8)
        assert "against the slant" not in a_cut
        i_cut, i_back = paras.index(a_cut), paras.index(a_back)
        # the trace closed before the strip's first frame: the slant's matches taken
        # back, a little square set out on each reach (no matches on the arrow drawn)
        before_cut = [p for p in paras[:i_cut] if p.strip()][-1]
        assert before_cut.startswith("**Achilles:** *[He takes the matches back off the slant's own arrow "
                                     "and sets a little square out on each reach.]*")
        assert sum("The two blocks are lifted off the ruled sheet whole and set beside the dish" in p
                   for p in paras[:i_cut]) == 1
        assert sum("The square on the slant goes back into the dish in its four halves" in p
                   for p in paras[i_cut:i_back]) == 1
        assert a_back.endswith("One match, backwards.")
        assert QUOTE in " ".join(a_squares.split())                       # the trap, then its picture


def basis_state(b):
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    n = p.x(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def turned_state(b):
    """The state + (b=0) and the state - (b=1)."""
    p = Picture()
    n = p.z(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def echo(state, b):
    return state + b.adjoint() + b + state.adjoint()


def sketchbook():
    plus, minus = turned_state(0), turned_state(1)
    assert np.allclose(matrix_of(plus).ravel(), PLUS)
    assert np.allclose(matrix_of(minus).ravel(), MINUS)

    # the two state pictures are not the same picture
    assert not same_picture(plus, minus)

    # but their Echo pictures are, for both answers of the standard question,
    # and each is one half
    for k in (0, 1):
        b = basis_state(k)
        assert same_picture(echo(plus, b), echo(minus, b))
        assert np.isclose(matrix_of(echo(plus, b))[0, 0], 1 / 2)
        # in the second picture each half is negative and the echo repeats it
        if k == 1:
            half = matrix_of(minus + b.adjoint())[0, 0]
            assert np.isclose(half, -1 / np.sqrt(2))
            assert np.isclose(matrix_of(b + minus.adjoint())[0, 0], half)

    # one closed diagram whose number is zero.  A zero is checked with NumPy,
    # never by comparing two pictures -- two zero pictures agree vacuously.
    zero_diagram = minus + plus.adjoint()
    assert np.allclose(matrix_of(zero_diagram), 0)
    one_diagram = plus + plus.adjoint()
    assert np.isclose(matrix_of(one_diagram)[0, 0], 1)           # the non-zero partner
    assert not same_picture(zero_diagram, one_diagram)


if __name__ == "__main__":
    verdict("HiddenSign", blackboard, sketchbook)
