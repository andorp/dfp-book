"""Challenge 2: Why Squares
============================
"Three by three on the heads-ward reach: nine. Four by four on the tails-ward:
sixteen. Nine and sixteen. Twenty-five in all."  -- Achilles

Claim: a chance is an amplitude times its mirror twin.  The Echo -- a closed
diagram followed by its mirror image -- is |<b|psi>|^2: real, never negative,
and the two values of a question make one whole.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import math
from fractions import Fraction

import numpy as np
from dfp import Picture, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch02 as P

T = np.array([3 / 5, 4 / 5], dtype=complex)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)
PHASES = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))


def blackboard():
    # the reaches are not shares: three in five and four in five overshoot
    reaches = np.array([np.vdot(ket(0), T), np.vdot(ket(1), T)])
    assert np.allclose(reaches, [3 / 5, 4 / 5])
    assert np.isclose(reaches.sum(), 7 / 5) and not np.isclose(reaches.sum(), 1)

    # the number with its twin: nine and sixteen, twenty-five in all
    born = np.array([np.conj(a) * a for a in reaches]).real
    assert np.allclose(born, [9 / 25, 16 / 25])
    assert np.isclose(born.sum(), 1)
    assert np.isclose(9 + 16, 25)

    # a generic complex state: the Born values are a covered coin
    rng = np.random.default_rng(22)
    psi = random_state(rng)
    assert not np.allclose(psi.imag, 0) and not np.isclose(psi @ psi, 1)
    values = []
    for b in (ket(0), ket(1)):
        amp = np.vdot(b, psi)                 # <b|psi>
        twin = np.vdot(psi, b)                # <psi|b>, the mirror twin
        assert np.isclose(twin, np.conj(amp))
        p = twin * amp
        assert np.isclose(p, abs(amp) ** 2)
        assert np.isclose(p.imag, 0) and p.real >= -1e-12
        values.append(p.real)
    assert np.isclose(sum(values), 1)         # a stochastic vector

    # the plain square, without the conjugate, is not even real here
    amp1 = np.vdot(ket(1), psi)
    assert not np.isclose((amp1 ** 2).imag, 0)

    # the dark string repeats the pale string's number, sign included, for a
    # real state; for a state with non-real entries the two halves are
    # conjugates and in general not equal
    for b in (ket(0), ket(1)):
        assert np.isclose(np.vdot(b, T), np.vdot(T, b))          # real: equal
    assert np.isclose(np.vdot(ket(1), psi), np.conj(np.vdot(psi, ket(1))))
    assert not np.isclose(np.vdot(ket(1), psi), np.vdot(psi, ket(1)))

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


PREDICTED = "Heads three parts in five, tails four. Which makes seven parts in — five."


def model_tail_on_tail(start, closing):
    """The model's dark reading, step by step: the closing square alone; the start
    square laid over it, tail on tail; the card dropped from the start's tip onto
    the closing arrow's line, its foot at <closing|start> (the mirror twin of the
    pale string's <closing|start> -- for a real arrow, the same number); the
    matches laid to the foot.  Each state: (laid, foot or None, reach laid or None)."""
    u = closing / np.linalg.norm(closing)
    r = amplitude_reach(start, u)
    return {"closing": (False, None, None), "laid": (True, None, None),
            "card": (True, r * u.real, None), "matches": (True, r * u.real, r)}


def check_dark_frame(tt, start, closing, show):
    lay, foot, row = model_tail_on_tail(start, closing)[show]
    assert same_arrow(tt.start, start) and same_arrow(tt.closing, closing)
    assert tt.show == show and tt.laid == lay
    if foot is None:
        assert tt.card is None
    else:
        assert np.allclose(tt.card.foot, foot)
        assert np.isclose((tt.tip - tt.card.foot) @ tt.closing.u, 0)    # dropped square
    if row is None:
        assert tt.row is None
    else:
        assert np.isclose(sum(b - f for f, b, _ in tt.row), row)


def achilles_shares(vec, lines):
    """Achilles, at the end of the First: "Then those are the shares! Heads three
    parts in five, tails four. Which makes seven parts in -- five."  Each reach
    taken as its own share of the arrow's length: the strip folded in the arrow's
    length, the heads' reach shaded from the knot, the tails' reach running on
    from the change of shade -- past the paper's end by what overshoots."""
    L = np.linalg.norm(vec)
    heads, tails = (amplitude_reach(vec, ln) for ln in lines)
    return dict(folded=L, heads=heads, tails=tails, parts=heads + tails, past_end=heads + tails - L)


def check_round4_rules():
    """The new options of the Second against the model, beyond the dialogue's own
    pictures: every tail-on-tail stage of every pair; every squares strip on a
    whole-reach arrow; every strip Achilles's rule can fold; deep and its fault."""
    for w in WORDS:
        for w2 in WORDS:
            a, b = (P.clean(P.parse_arrow, x, "test") for x in (w, w2))
            if not np.isclose(P.pair_length(a, b), a.L):
                continue
            start = model_vec(w)
            for show in ("closing", "laid", "card", "matches"):
                tt = P.clean(P.tailontail, w, w2, False, show)
                check_dark_frame(tt, start, model_vec(w2), show)
            assert P.clean(P.tailontail, w, w2, False, "").show == "matches"   # the default
        vec = model_vec(w)
        lines = model_lines("square")
        reaches = [amplitude_reach(vec, ln) for ln in lines]
        if all(np.isclose(r, round(r)) for r in reaches):
            (bare, _), (blocks, _) = [(f.picture, f.caption)
                                      for f in P.clean(P.crosssteps, w, "square", "both", "squares", "",
                                                       "a; b", False)]
            assert bare.squares == {} and bare.card.line == "B"
            assert blocks.card is None and blocks.rows == bare.rows              # the matches stay
            assert {l: len(s) for l, s in blocks.squares.items()} == \
                   {l: round(r) ** 2 for l, r in zip("AB", reaches)}
    assert P.faults(P.tailontail, "slant", "slant", False, "bogus")
    assert P.faults(P.crosssteps, "slant", "square", "none", "squares", "", "a; b")   # nothing read
    for n in range(1, 26):
        for k in range(1, n):
            ok = P.isqrt(n) is not None and P.isqrt(k) is not None and P.isqrt(n - k) is not None
            s, errors = P.run(P.sharestrip, "%d in %d" % (k, n), False, "expects=reaches")
            assert (errors == []) == ok, (k, n)
            if ok:
                # the arrow root n long, its reaches root k and root (n - k)
                vec = np.array([math.isqrt(k), math.isqrt(n - k)], dtype=complex)
                want = achilles_shares(vec, model_lines("square"))
                p = s.predicted
                assert (p.folded, p.heads, p.tails, p.parts, p.past_end) == tuple(
                    round(want[x]) for x in ("folded", "heads", "tails", "parts", "past_end"))
                assert p.past_end > 0                      # the reaches overshoot the arrow, always
                assert not np.isclose(p.heads / p.folded, k / n)   # and their share is not the squares'
    for opt in ("", "deep=on", "expects=reaches", "expects=reaches;deep=on"):
        s = P.clean(P.sharestrip, "nine in twenty-five", False, opt)
        assert s.dump[0].endswith(" deep") == ("deep=on" in opt)
        assert (s.deep is True) == ("deep=on" in opt)
    assert P.faults(P.sharestrip, "nine in twenty-five", True, "deep=on")          # never on the key
    assert P.faults(P.sharestrip, "nine in twenty-five", True, "expects=reaches")  # nor a prediction
    assert P.faults(P.sharestrip, "all dark", False, "expects=reaches")


def pictures():
    P.check_lua()                     # ch02.lua and ch02.tex as the transcription was proved
    check_whole_dialogue()
    check_rules()
    check_round4_rules()
    # Four moments.  P2.1: the strings.  Block 2.2, a step strip of the dark
    # string's reading (keyed on its first frame).  P2.3: the little squares
    # built on the reaches (one picture: the card never stands on this sheet).
    # Block 2.4: the strip as Achilles expects it, pale, beside the strip as it
    # is (keyed).
    found = P.check_section(DIALOGUE, "Second", sizes=[1, 3, 1, 2], frames=[1, 3, 1, 2])
    (c_strings, [(es, _)]), = found[0]
    dark_calls = found[1]
    (c_squares, [(cs, _)]), = found[2]
    (c_expects, [(expected, caption)]), (c_strip, [(ss, _)]) = found[3]

    # P2.1: the heads-ward square back on as the pale closing tag (the re-tie),
    # and the dark string tagged with tracings of the same two, each at the
    # other end.  "echo" is computed by the macro, never written out.
    assert c_strings.macro == "echostrings" and not c_strings.keyed, c_strings.text
    assert P.norm(c_strings.args[1]) == "echo"
    tags = {k: v.arrow for k, v in es.tags.items()}
    assert set(tags) == {"pale start", "pale closing", "dark start", "dark closing"}
    assert same_arrow(tags["pale start"], 5 * T) and same_arrow(tags["pale closing"], 5 * ket(0))
    assert same_arrow(tags["dark start"], 5 * ket(0)) and same_arrow(tags["dark closing"], 5 * T)
    for s in ("pale", "dark"):
        check_mirror_gesture(es, s)
    assert es.labels == ()
    pale = float(tags["pale start"].v @ tags["pale closing"].u)
    assert np.isclose(pale, 3)

    # Block 2.2: the dark string's reading, step by step -- its heads-ward start
    # laid over the three-and-four closing, tail on tail; the card dropped from
    # the heads-ward tip onto the three-and-four line; three matches along it.
    # Each frame is the model's stage; the key moves onto the first.
    shows = []
    for c, [(tt, _)] in dark_calls:
        assert c.macro == "tailontail", c.text
        assert tt.start == tags["dark start"] and tt.closing == tags["dark closing"]   # its own tags
        check_dark_frame(tt, 5 * ket(0), 5 * T, tt.show)
        shows.append(tt.show)
    assert shows == ["laid", "card", "matches"]                         # in the order she did it
    first, dropped, read_ = (tt for _, [(tt, _)] in dark_calls)
    assert [c.keyed for c, _ in dark_calls] == [True, False, False]
    assert first.labels == ("start over closing",) and dropped.labels == read_.labels == ()
    assert np.isclose(read_.length, 5)                                 # drawn five matches long
    assert np.isclose(read_.number, 3) and np.isclose(read_.number, pale)   # identity 3b: 3
    assert laid(read_.row) == (3, 0, 1) and read_.card.out == 1
    assert np.allclose(read_.card.foot, 3 * T.real) and np.allclose(dropped.card.foot, read_.card.foot)
    # the model: the dark string reads the mirror twin <T|0>, the pale string
    # the call <0|T>; for this real arrow they are one number, sign included
    assert np.isclose(read_.number, 5 * np.vdot(T, ket(0)))
    assert np.isclose(np.vdot(T, ket(0)), np.vdot(ket(0), T))

    # P2.3: the square sheet, reaches three and four, and the little paper
    # squares laid on them, nine and sixteen, each block on the side of its
    # reach away from the arrow.  No card on this sheet: the matches lie bare,
    # just as the matches of the First's reading lay (only paper added).
    assert c_squares.macro == "crosssheet" and not c_squares.keyed and c_squares.opt == "", c_squares.text
    assert cs.position == "square" and same_arrow(cs.arrow, 5 * T)
    assert np.isclose(cs.reach["A"], amplitude_reach(5 * T, ket(0))) and np.isclose(cs.reach["A"], 3)
    assert np.isclose(cs.reach["B"], amplitude_reach(5 * T, ket(1))) and np.isclose(cs.reach["B"], 4)
    assert laid(cs.rows["A"]) == (3, 0, 1) and laid(cs.rows["B"]) == (4, 0, 1)
    assert cs.read == {"A", "B", "squares"} and cs.card is None        # no card stands here
    bare = P.clean(P.crosssheet, "three-and-four", "square", "both", False, "")
    assert bare.rows == cs.rows                                        # the matches do not move
    nine, sixteen = cs.squares["A"], cs.squares["B"]
    assert (len(nine), len(sixteen)) == (9, 16) and len(nine) + len(sixteen) == 25
    assert (len(nine), len(sixteen)) == tuple(round(amplitude_reach(5 * T, b)) ** 2 for b in (ket(0), ket(1)))
    assert np.isclose(len(nine) + len(sixteen), cs.arrow.L ** 2)
    assert all(c[1] < 0 for c in nine) and all(c[0] < 0 for c in sixteen)   # away from the arrow
    assert not {tuple(np.round(c, 6)) for c in nine} & {tuple(np.round(c, 6)) for c in sixteen}

    # Block 2.4: the prediction, pale -- each reach its own share: the strip
    # folded in five, three parts shaded, the tails' four running on past the
    # paper's end -- beside the true strip, keyed: nine in twenty-five.
    assert P.is_prediction(c_expects) and not c_expects.keyed and caption == "as Achilles expects"
    want = achilles_shares(5 * T, model_lines("square"))
    p = expected.predicted
    assert expected.expects == "reaches" and (expected.k, expected.n) == (ss.k, ss.n)
    assert (p.folded, p.heads, p.tails, p.parts, p.past_end) == (5, 3, 4, 7, 2)
    assert np.allclose([p.folded, p.heads, p.tails, p.parts, p.past_end],
                       [want[x] for x in ("folded", "heads", "tails", "parts", "past_end")])
    assert np.isclose(want["parts"] / want["folded"], 7 / 5)          # seven parts in five: over one
    assert expected.deep is True and ss.deep is None                  # level beside the key
    assert expected.pad == ss.pad                                     # the keyed strip's room
    assert c_strip.macro == "sharestrip" and c_strip.keyed, c_strip.text
    assert (ss.k, ss.n) == (9, 25) and ss.shaded == (True,) * 9 + (False,) * 16
    heads = Fraction(ss.k, ss.n)
    assert heads == Fraction(len(nine), len(nine) + len(sixteen))     # the squares, not the reaches
    assert np.isclose(float(heads), abs(np.vdot(ket(0), T)) ** 2)     # the Born value, 9/25
    assert np.isclose(float(1 - heads), abs(np.vdot(ket(1), T)) ** 2) # tails' share, 16/25
    assert not np.isclose(float(heads), cs.reach["A"] / 5)            # not three in five
    assert Fraction(p.heads, p.folded) != heads and p.parts > p.folded  # the prediction is refuted
    assert ss.labels == ("heads' share", "tails' share")
    assert c_strip.caption.endswith(P.norm(c_strip.args[0]))          # the caption reads it

    assert section_dumps(found) == [
        "echostrings: pale 3,4/5,0 dark 5,0/3,4 strip none",
        "tailontail: 5,0 over 3,4 reach 3.00 show laid",
        "tailontail: 5,0 over 3,4 reach 3.00 show card",
        "tailontail: 5,0 over 3,4 reach 3.00 show matches",
        "crosssheet: 3,4 square reach 3.00,4.00 read both,squares card none",
        "sharestrip: 9/25 expects reaches 3+4/5 deep",
        "sharestrip: 9/25"]

    # where they stand, in both dialogue files
    assert sum(PREDICTED in p for p in DIALOGUE.paragraphs("First")) == 1   # what he expected
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a_strings, a_dark, a_squares, a_strip = DIALOGUE.anchors("Second", path)
        assert a_strings.endswith("the arrow that asked now waits, and the arrow that waited now asks.]*")
        assert a_dark.endswith("Three parts in five — the pale string's number, arriving from the other end.")
        for phrase in ("He slides the heads-ward start square", "until tail sits on tail",
                       "the card drops from the heads-ward tip", "he lays the matches"):
            assert phrase in a_dark, phrase
        assert a_squares.endswith("Nine and sixteen. Twenty-five in all.")
        assert a_squares.startswith("**Achilles:** *[The heads-ward reach lies bare along its line, "
                                    "three matches from the crossing.")      # no card to lift
        assert "card" not in a_squares
        for phrase in ("The matches do not move; only paper has been added.",
                       "on the side away from the arrow"):
            assert phrase in a_squares, phrase
        assert a_strip.endswith("There's no foot here, only an arrow and a question.")
        assert "And those *do* make one whole." in a_strip            # the refutation, spoken


def basis_state(b):
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    n = p.x(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def turned_state(b):
    """The states + (b=0) and - (b=1): real, with one backwards reach."""
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
    """The Echo: state psi, wire, effect b -- a gap -- state b, wire, effect psi."""
    return state + b.adjoint() + b + state.adjoint()


def sketchbook():
    psi = generic_state()
    col = matrix_of(psi).ravel()
    total = 0
    for k in (0, 1):
        b = basis_state(k)
        e = echo(psi, b)
        assert e.inputs() == () and e.outputs() == ()
        value = matrix_of(e)[0, 0]

        left = matrix_of(psi + b.adjoint())[0, 0]       # the call
        right = matrix_of(b + psi.adjoint())[0, 0]      # its echo
        assert np.isclose(right, np.conj(left))
        assert np.isclose(value, left * right)          # closed diagrams in a row multiply
        assert np.isclose(value, abs(np.vdot(matrix_of(b).ravel(), col)) ** 2)
        assert np.isclose(value.imag, 0) and value.real >= -1e-12
        total += value.real

        # the right half IS the left half mirrored
        assert same_picture((psi + b.adjoint()).adjoint(), b + psi.adjoint())

    # adding numbers, not pictures: the two Echo numbers make one whole
    assert np.isclose(total, 1)

    # the dark string, as pictures: for real states the two halves agree, sign
    # and all -- both are minus one over root two
    minus, one = turned_state(1), basis_state(1)
    left = matrix_of(minus + one.adjoint())[0, 0]
    right = matrix_of(one + minus.adjoint())[0, 0]
    assert np.isclose(left, -1 / np.sqrt(2)) and np.isclose(right, left)
    assert np.isclose(left * right, 1 / 2)
    # for the generic state they are conjugates and differ
    gl = matrix_of(psi + one.adjoint())[0, 0]
    gr = matrix_of(one + psi.adjoint())[0, 0]
    assert np.isclose(gr, np.conj(gl)) and not np.isclose(gr, gl)

    # (3/5, 4/5) has no exact picture built from these nodes; its nine and
    # sixteen are checked on the blackboard only.


if __name__ == "__main__":
    verdict("Squares", blackboard, sketchbook)
