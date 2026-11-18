"""Challenge 1: The Right-Hand End
==================================
"Three. And the arrow is five. Three parts in five."  -- Achilles

Claim: a bra is the ket laid on its side with every entry CONJUGATED, and a
state at the left end of a wire with an effect at the right end is a closed
diagram, that is, one number: <0|psi> = alpha.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import math
from fractions import Fraction

import numpy as np
from dfp import Picture, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch02 as P

T = np.array([3 / 5, 4 / 5], dtype=complex)        # last evening's arrow
CHI = np.array([3 / 5, 4j / 5], dtype=complex)     # the state with a non-real entry
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)

# The generic state of the sketchbook: the state 0 followed by three phases.
# It has no tidy fractions, which is the point -- it is non-real and its two
# entries differ, so a plain transpose cannot pass for the conjugate one.
PHASES = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))


def blackboard():
    # a ket is a name for a column; the first (top) entry belongs to 0
    assert np.allclose(ket(0), [1, 0]) and np.allclose(ket(1), [0, 1])
    alpha, beta = T
    assert np.allclose(alpha * ket(0) + beta * ket(1), T)

    # the two amplitudes of the arrow: three parts in five, four parts in five
    assert np.isclose(np.vdot(ket(0), T), 3 / 5)
    assert np.isclose(np.vdot(ket(1), T), 4 / 5)

    # a generic state: complex, non-real, and not its own conjugate
    rng = np.random.default_rng(21)
    psi = random_state(rng)
    phi = random_state(rng)
    assert not np.allclose(psi.imag, 0) and not np.allclose(psi, psi.conj())
    assert np.isclose(np.linalg.norm(psi), 1)

    # the bra: the column laid on its side, every entry conjugated
    bra = psi.conj()
    assert not np.allclose(bra, psi)

    # <0|psi> = alpha, <1|psi> = beta -- the entries of a column are amplitudes
    assert np.isclose(ket(0).conj() @ psi, psi[0])
    assert np.isclose(ket(1).conj() @ psi, psi[1])

    # exchange the labels and the bracket is conjugated
    assert np.isclose(np.vdot(psi, phi), np.conj(np.vdot(phi, psi)))
    assert not np.isclose(np.vdot(psi, phi), np.vdot(phi, psi))   # they really differ

    # why the conjugate, in the chapter's own example
    assert np.isclose(np.vdot(CHI, CHI), 1)          # conjugated row: one
    assert np.isclose(CHI @ CHI, -7 / 25)            # plain row: minus seven in twenty-five
    assert not np.isclose(CHI @ CHI, 1)
    # for the real arrow the conjugation is invisible, which is why it hides
    assert np.isclose(np.vdot(T, T), 1) and np.isclose(T @ T, 1)

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


QUOTE = __doc__.split('"')[1]


def model_reading(vec, line, steps):
    """The model's reading, step by step: the sheet laid and nothing yet; "card":
    the card dropped from the tip onto the line asked, its foot the reach along
    that line (<line|arrow>, sign included); "matches": the matches laid from the
    crossing to the foot.  Each state is (foot or None, reach laid or None)."""
    r = amplitude_reach(vec, line)
    states, foot, row = [(None, None)], None, None
    for s in steps:
        if s == "card":
            foot = r * line.real
        elif s == "matches":
            assert foot is not None                    # the card first, then the matches
            row = r
        states.append((foot, row))
    return states


def as_a_coin(vec):
    """Achilles: "the string hands over a coin. Heads, I expect." -- all or nothing:
    the whole arrow handed to the one line read, forwards."""
    return np.linalg.norm(vec)


def check_frame(pic, vec, lines, line, state):
    """A frame of a reading strip against the model's state."""
    foot, row = state
    assert same_arrow(pic.arrow, vec) and pic.sheet_on
    for l, ln in zip("AB", lines):
        assert np.allclose(pic.e[l], ln.real)
    if foot is None:
        assert pic.card is None
    else:
        assert pic.card.line == line and np.allclose(pic.card.foot, foot)
    if row is None:
        assert pic.rows == {}                                   # no matches yet
    else:
        assert set(pic.rows) == {line} and np.isclose(sum(b - f for f, b, _ in pic.rows[line]), row)
    # a reading carries no marks and predicts nothing
    assert (pic.swung, pic.ghost, pic.moves, pic.expects) == (None, None, None, None)


def check_reading_strips():
    """Every step strip of a first reading -- the sheet laid, the card dropped,
    the matches laid -- on every arrow, under every sheet with named lines,
    along each line, against the model; the steps out of order are faults; and
    the coin rule, which the dialogue does not draw, against Achilles's coin."""
    for w in WORDS:
        vec = model_vec(w)
        for pos in ("square", "turned", "along"):
            lines = model_lines(pos, vec)
            for word, line in P.LINE[pos].items():
                frames = P.clean(P.crosssteps, w, pos, "none", "card %s, matches" % word, "",
                                 "a; b; c", True)
                states = model_reading(vec, lines["AB".index(line)], ["card", "matches"])
                assert len(frames) == len(states) == 3
                for f, s in zip(frames, states):
                    check_frame(f.picture, vec, lines, line, s)
                assert [f.keyed for f in frames] == [True, False, False]   # the key on the first
                assert [f.caption for f in frames] == ["a", "b", "c"]
                # the last frame is the plain reading of that line
                assert frames[2].picture.dump == P.clean(P.crosssheet, w, pos, word, False, "").dump
                # the coin: the whole arrow to the line read, forwards, no card
                coin = P.clean(P.crosssheet, w, pos, word, False, "expects=coin")
                assert coin.expects == "coin" and coin.card is None
                assert np.isclose(coin.laid[line], as_a_coin(vec)) and set(coin.rows) == {line}
                if not np.isclose(abs(amplitude_reach(vec, lines["AB".index(line)])), np.linalg.norm(vec)):
                    assert not np.isclose(coin.laid[line], frames[2].picture.reach[line])
    # out of order: matches with no card standing, a card after the matches
    assert P.faults(P.crosssteps, "slant", "square", "none", "matches", "", "a; b")
    assert P.faults(P.crosssteps, "slant", "square", "heads-ward", "card tails-ward", "", "a; b")
    assert P.faults(P.crosssteps, "slant", "square", "none", "card heads-ward", "", "a")  # a caption short
    assert P.faults(P.crosssheet, "slant", "square", "both", False, "expects=coin")     # a coin reads one line


def pictures():
    P.check_lua()                     # ch02.lua and ch02.tex as the transcription was proved
    check_whole_dialogue()
    check_rules()
    check_reading_strips()
    assert QUOTE in DIALOGUE.section("First")
    # Two moments.  P1.1, keyed: the strings.  Block 1.2, a keyed step strip of
    # the evening's first reading: the sheet laid; the card dropped; the matches.
    found = P.check_section(DIALOGUE, "First", sizes=[1, 1], frames=[1, 3])
    (c_strings, [(es, _)]), = found[0]
    (c_steps, steps), = found[1]

    # P1.1, keyed: last evening's arrow tied on as the start, and the heads-ward
    # square carried to the right-hand end as it lies and knotted the other
    # way -- a closing tag.  The dark string waits, bare ("Later.").
    assert c_strings.macro == "echostrings" and c_strings.keyed, c_strings.text
    assert set(es.tags) == {"pale start", "pale closing"}             # the dark string bare
    start, closing = es.tags["pale start"].arrow, es.tags["pale closing"].arrow
    assert same_arrow(start, 5 * T)                                   # three and four, five long
    assert same_arrow(closing, 5 * ket(0))                            # heads-ward, five long
    check_mirror_gesture(es, "pale")
    assert es.labels == ("start tag", "closing tag", "pale", "dark")
    # the string's one number: the start's reach along the closing arrow's line
    number = float(start.v @ closing.u)
    assert np.isclose(number, amplitude_reach(5 * T, ket(0))) and np.isclose(number, 3)

    # Block 1.2: the cross-sheet laid square over the start, its heads-ward line
    # along the closing tag's arrow; then the card dropped from the tip, its foot
    # at three; then three matches from the crossing to the foot.  Every frame
    # is the model after that many steps.
    assert c_steps.macro == "crosssteps" and c_steps.keyed, c_steps.text
    (laid_, _), (dropped, _), (read_, _) = steps
    lines = model_lines("square")
    states = model_reading(5 * T, ket(0), ["card", "matches"])
    for (pic, _), state in zip(steps, states):
        assert pic.position == "square" and pic.arrow.name == start.name
        check_frame(pic, 5 * T, lines, "A", state)
    assert np.allclose(dropped.e["A"], closing.u)                     # the line the question asks
    assert [p.keyed for p, _ in steps] == [True, False, False]        # the key on the first frame
    assert laid_.labels == ("heads-ward", "tails-ward") and dropped.labels == read_.labels == ()
    assert np.allclose(laid_.e["A"], P.clean(P.parse_arrow, "heads-ward", "test").u)  # named truly
    assert np.allclose(laid_.e["B"], P.clean(P.parse_arrow, "tails-ward", "test").u)
    assert laid_.read == frozenset() and laid_.card is None and laid_.rows == {}
    assert dropped.read == {"A", "no-matches"} and dropped.rows == {}
    assert np.allclose(dropped.card.foot, [3, 0]) and dropped.card.out == 1
    reach = read_.reach["A"]
    assert np.isclose(reach, number) and np.isclose(reach, 3)          # identity 3b: 3
    assert laid(read_.rows["A"]) == (3, 0, 1) and set(read_.rows) == {"A"}   # three whole, forward
    assert np.allclose(read_.card.foot, dropped.card.foot)             # the card stays where it fell
    assert np.isclose(read_.arrow.L, 5)                                # and the arrow is five
    assert np.isclose(reach / read_.arrow.L, np.vdot(ket(0), T).real)  # three parts in five
    # the last frame is the round-3 picture of the reading, exactly
    old = P.clean(P.crosssheet, "three-and-four", "square", "heads-ward", False, "")
    assert (old.dump, old.rows, old.card.line) == (read_.dump, read_.rows, read_.card.line)
    # the other line would have said four, not three (spoken, not drawn)
    assert np.isclose(read_.reach["B"], 4) and not np.isclose(read_.reach["B"], reach)
    # three and four in five are reaches, not shares: they overshoot the whole
    assert np.isclose((read_.reach["A"] + read_.reach["B"]) / 5, 7 / 5)
    # no prediction is drawn (the coin rule is not used, identity.md 3e.7 (a)):
    # what the string hands over is a row of three matches, not a coin of five
    assert not any(P.is_prediction(c) for b in DIALOGUE.blocks("First") for c in b)
    coin = P.clean(P.crosssheet, "three-and-four", "square", "heads-ward", False, "expects=coin")
    assert np.isclose(coin.laid["A"], as_a_coin(5 * T)) and not np.isclose(coin.laid["A"], reach)

    assert section_dumps(found) == [
        "echostrings: pale 3,4/5,0 dark -/- strip none",
        "crosssheet: 3,4 square reach 3.00,4.00 read none card none",
        "crosssheet: 3,4 square reach 3.00,4.00 read heads-ward,no-matches card A",
        "crosssheet: 3,4 square reach 3.00,4.00 read heads-ward card A"]

    # where they stand, in both dialogue files
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a_strings, a_steps = DIALOGUE.anchors("First", path)
        _, f_steps = DIALOGUE.following("First", path)
        assert a_strings.endswith("It waits for what comes. A closing tag.")
        assert a_steps.endswith(QUOTE)
        # the strip's three frames: the sheet laid (before), the card, the matches
        assert "He stands the square-cornered card on its true corner at the arrow's tip" in a_steps
        assert "he lays the matches end to end" in a_steps
        paras = DIALOGUE.paragraphs("First", path)
        before = paras[:paras.index(a_steps)]
        assert sum("lets it rest on the start tag with the crossing on the arrow's tail" in p
                   for p in before) == 1
        assert f_steps.startswith("**Tortoise:** *[She unties the closing tag,")


def basis_state(b):
    """The state 0 or the state 1, as a picture."""
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    n = p.x(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)          # remove PyZX's stray factor: exactly (1,0) / (0,1)
    return g


def generic_state():
    """A state with non-real entries, both of them non-zero."""
    p = Picture()
    n = p.x(0, 1, 0)
    a, b, c = PHASES
    chain = [p.z(0, 2, a), p.x(0, 3, b), p.z(0, 4, c)]
    p.wire(n, *chain, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def effect_by_hand(conjugate=True):
    """The mirror image of the generic state, drawn by hand.

    The mirror reverses the row of phases; conjugating them means negating them.
    With conjugate=False the phases are left alone: that is the plain transpose,
    the wrong question of the chapter's sidebar.
    """
    s = -1 if conjugate else 1
    a, b, c = PHASES
    p = Picture()
    chain = [p.z(0, 1, s * c), p.x(0, 2, s * b), p.z(0, 3, s * a)]
    n = p.x(0, 4, 0)
    p.wire(p.inp(0), *chain, n)
    g = p.done()
    g.scalar.add_power(-1)
    return g


def sketchbook():
    psi = generic_state()
    col = matrix_of(psi).ravel()
    assert np.isclose(np.linalg.norm(col), 1) and not np.allclose(col.imag, 0)

    # the effect: PyZX's adjoint() is its name for the mirror image of a picture
    mirror = psi.adjoint()
    row = matrix_of(mirror)
    assert row.shape == (1, 2)
    assert np.allclose(row, col.conj().reshape(1, 2))
    assert not np.allclose(row, col.reshape(1, 2))        # not the plain transpose

    # the same effect drawn by hand, and the un-conjugated counterfeit
    assert same_picture(effect_by_hand(True), mirror)
    assert not same_picture(effect_by_hand(False), mirror)
    assert np.allclose(matrix_of(effect_by_hand(False)), col.reshape(1, 2))

    # a closed diagram: state, wire, effect.  No open end, and one number.
    zero = basis_state(0)
    closed = psi + zero.adjoint()
    assert closed.inputs() == () and closed.outputs() == ()
    assert matrix_of(closed).shape == (1, 1)
    assert np.isclose(matrix_of(closed)[0, 0], col[0])    # <0|psi> = alpha

    # the effect is on the RIGHT of the picture, the bra on the LEFT of the bracket
    e = zero.adjoint()
    assert np.allclose(matrix_of(psi + e), matrix_of(e) @ matrix_of(psi))

    # mirror the whole closed diagram and the number is conjugated
    mirrored = zero + psi.adjoint()
    assert np.isclose(matrix_of(mirrored)[0, 0], np.conj(col[0]))
    assert not np.isclose(matrix_of(mirrored)[0, 0], col[0])

    # and closing the state with the counterfeit effect gives the wrong number
    counterfeit = matrix_of(psi + effect_by_hand(False))[0, 0]
    assert not np.isclose(counterfeit, 1)
    assert np.isclose(matrix_of(psi + psi.adjoint())[0, 0], 1)

    # (3/5, 4/5) and (3/5, 4i/5) are not exactly representable with these
    # phases, so they are checked on the blackboard only.


if __name__ == "__main__":
    verdict("Closing", blackboard, sketchbook)
