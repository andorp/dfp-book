"""Challenge 1: One Turn and Its Mirror
=======================================
"I turn it like a page. Seen through the paper, the last arrow would come
first, and every arrow runs the other way round. Now I play what I see."
 -- Tortoise

Claim: the mirror image of a box is its conjugate transpose -- neither the
transpose nor the conjugate alone -- and a unitary box meeting its mirror, in
either order, leaves a bare wire.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction

import numpy as np
import pyzx as zx
from dfp import Picture, ket, matrix_of, random_matrix, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch03 as P


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def random_unitary(rng):
    """A generic unitary: the Q of a QR factorisation of a generic matrix.
    NumPy only -- no SciPy anywhere in this chapter."""
    q, r = np.linalg.qr(random_matrix(rng))
    # fix the phases of the columns so that Q is not accidentally real
    return q * (np.diag(r) / np.abs(np.diag(r)))


def blackboard():
    rng = np.random.default_rng(31)

    # --- the mirror of a single box: three different operations -------------
    A = random_matrix(rng)
    assert not np.allclose(A, A.T)                      # non-symmetric
    assert not np.allclose(A, A.conj())                 # non-real
    dag = A.conj().T
    assert not np.allclose(dag, A.T)                    # dagger is not transpose
    assert not np.allclose(dag, A.conj())               # dagger is not conjugate
    assert not np.allclose(dag, A)
    assert np.allclose(dag.conj().T, A)                 # twice over is nothing
    assert np.allclose(I2.conj().T, I2)

    # a box on a closed wire is an entry, and the mirror swaps i, j and conjugates
    for i in (0, 1):
        for j in (0, 1):
            assert np.isclose(np.vdot(ket(i), A @ ket(j)), A[i, j])
            assert np.isclose(np.vdot(ket(j), dag @ ket(i)), np.conj(A[i, j]))

    # --- boxes that keep the promise ---------------------------------------
    U = random_unitary(rng)
    assert not np.allclose(U, U.T) and not np.allclose(U, U.conj())
    assert np.allclose(U.conj().T @ U, I2)              # the mirror undoes it
    assert np.allclose(U @ U.conj().T, I2)              # in either order
    assert not np.allclose(U.T @ U, I2)                 # the transpose does not
    assert not np.allclose(U.conj() @ U, I2)            # nor the conjugate
    # the columns are an orthonormal basis (Theorem 3.2)
    assert np.allclose([[np.vdot(U[:, i], U[:, j]) for j in (0, 1)] for i in (0, 1)], I2)

    # the chapter's test piece, U = (HSH)S: unitary, non-real, non-symmetric
    V = H @ S @ H
    test = V @ S
    assert np.allclose(test, 0.5 * np.array([[1 + 1j, 1 + 1j], [1 - 1j, -1 + 1j]]))
    assert not np.allclose(test, test.T) and not np.allclose(test, test.conj())
    assert np.allclose(test.conj().T @ test, I2)
    assert np.allclose(test.T @ test, np.array([[0, 1j], [1j, 0]]))    # not the bare wire
    assert not np.allclose(test.conj() @ test, I2)

    # --- the six boxes ------------------------------------------------------
    for P in (X, Y, Z, H):
        assert np.allclose(P.conj().T, P)               # its own mirror image
        assert np.allclose(P @ P, I2)                   # hence its own undoing
    # Y is the interesting one: neither real nor symmetric, and still self-mirror
    assert not np.allclose(Y, Y.conj()) and not np.allclose(Y, Y.T)
    assert np.allclose(Y.T, -Y) and np.allclose(Y.conj(), -Y)
    assert np.allclose(Y, 1j * X @ Z)                   # Y = iXZ
    assert np.allclose(X @ Z, -(Z @ X))
    for P in (S, T):
        assert not np.allclose(P.conj().T, P)           # not their own mirrors
        assert np.allclose(P.conj().T @ P, I2)          # still unitary
    assert np.allclose(S @ S, Z) and np.allclose(T @ T, S)
    assert not np.allclose(S @ S, I2)
    # the trap of Section 3.4: wrapping in S is not wrapping in S-mirrored
    assert np.allclose(S @ X @ S.conj().T, Y)
    assert np.allclose(S @ X @ S, 1j * X)

    beads()
    strip()


# --------------------------------------------------------------------------
# The ball.  A position is a rotation R of ordinary space: the bead of a
# needle whose home direction is b now points along R @ b.  A twirl is made
# about a needle AS IT LIES, so its axis is R @ b, and R gains the turn on the
# right.  "With the clock" is the sense seen from that needle's own bead, which
# is a negative turn about the direction the bead points.
# --------------------------------------------------------------------------
UP = np.array([0.0, 0.0, 1.0])
COMPANY = np.array([0.0, 1.0, 0.0])
STOVE = np.array([1.0, 0.0, 0.0])            # the company's left; dresser is -STOVE
HOME = {"glass": UP, "steel": COMPANY, "brass": (UP + COMPANY) / np.sqrt(2)}
PLACES = {"up": UP, "down": -UP, "company": COMPANY, "away": -COMPANY,
          "stove": STOVE, "dresser": -STOVE}


def turn(axis, angle):
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def play(score, sense=+1):
    """Play a score from home.  Each item is (needle, clicks, +1 with the
    clock / -1 against); sense=-1 reads the whole score the other way round."""
    R = np.eye(3)
    for needle, clicks, way in score:
        R = R @ turn(HOME[needle], -sense * way * clicks * np.pi / 2)
    return R


def reading(R, needle):
    v = R @ HOME[needle]
    for name, u in PLACES.items():
        if np.allclose(v, u):
            return name
    return tuple(np.round(v, 6))


def at_home(R):
    return np.allclose(R, np.eye(3))


def page(score):
    """The strip turned like a page: order reversed AND every arrow reversed."""
    return [(n, c, -w) for n, c, w in reversed(score)]


def round_on_table(score):
    """Order reversed, senses as drawn."""
    return list(reversed(score))


def top_over_bottom(score):
    """Order kept, every sense reversed."""
    return [(n, c, -w) for n, c, w in score]


def beads():
    cw = +1
    # one click of the glass needle, with the clock
    one = [("glass", 1, cw)]
    assert reading(play(one), "glass") == "up"
    assert reading(play(one), "steel") == "stove"
    assert not at_home(play(one))
    # the page brings it home -- and so, for a single arrow, does the flip top
    # over bottom, which is why one note settles nothing.  (Round on the table
    # it is the same arrow again: a half-turn, not home.)
    assert at_home(play(one + page(one)))
    assert top_over_bottom(one) == page(one)
    assert not at_home(play(one + round_on_table(one)))

    # Achilles's two-headed score: the trap.  With half-turns the order does
    # not matter and all three ways of reading the strip backwards work.
    trap = [("glass", 2, cw), ("steel", 2, cw)]
    swapped = [("steel", 2, cw), ("glass", 2, cw)]
    assert np.allclose(play(trap), play(swapped))
    assert reading(play(trap), "glass") == "down"
    assert reading(play(trap), "steel") == "away"
    for undo in (page, round_on_table, top_over_bottom):
        assert at_home(play(trap + undo(trap)))
    # the same holds read the other way round throughout
    for s in (+1, -1):
        assert np.allclose(play(trap, s), play(swapped, s))
        for undo in (page, round_on_table, top_over_bottom):
            assert at_home(play(trap + undo(trap), s))
    # and against the clock a single click sends the steel bead the other way
    assert reading(play(one, -1), "steel") == "dresser"


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the ball with \crabball[opts]{glass}{steel}{..},
# a step strip of it with \crabplay[opts]{glass}{steel}{score}{cap; cap; ..}, and
# the Tortoise's strip with \scorestrip[view]{score}{caption} (pdf/props/ch03.tex).
# pdf/props/ch03.lua COMPUTES what each shows -- the ball after a score is played,
# a step strip's frames with their marks, the state a wrong rule predicts, the
# strip as it now lies -- and tools/props_ch03.py transcribes it.  The calls are
# read out of this challenge's section of dialog.md (tools/dfp_props.py), never
# copied, so an edit to the dialogue that breaks a picture breaks here.  Every
# state is checked against the model above.  The pictures use the room's frame
# (x towards the dresser, y up, z towards the company); ours is x towards the
# stove, y towards the company, z up, and ROOM carries one into the other.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("crabball", "crabplay", "scorestrip"))
ROOM = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def check_frame():
    # the change of frame is a turn, not a reflection -- else every "with the
    # clock" would silently become "against"
    assert np.allclose(ROOM.T @ ROOM, np.eye(3)) and np.isclose(np.linalg.det(ROOM), 1)
    for name, u in PLACES.items():                      # all six names agree
        assert np.allclose(ROOM @ u, P.vec(name)), name


def beads_of(R):
    """Where the model puts the three beads, in the room's frame."""
    return {n: ROOM @ R @ HOME[n] for n in HOME}


def drawn_at(state, R):
    """A drawn state (glass, steel, and the brass the macro puts midway) is the
    model's position R."""
    return all(np.allclose(state.bead(n), v) for n, v in beads_of(R).items())


def triples(arrows):
    """The transcription's arrows as the model's (needle, clicks, way)."""
    return [(a.needle, a.clicks, a.sign) for a in arrows]


def marked(ball, before, needle, after):
    """The marks of one step from position `before` to `after`, twirling
    `needle`: a ghost where each bead that moved stood before, and the hold
    ring round the bead of the needle held fast -- which did not stir."""
    m = ball.marks
    assert m is not None and m.arrow.needle == needle
    b0, b1 = beads_of(before), beads_of(after)
    assert drawn_at(m.before, before)
    assert np.allclose(m.hold, b1[needle]) and np.allclose(m.hold, b0[needle])
    moved = {n for n in HOME if not np.allclose(b0[n], b1[n])}
    assert set(m.ghosts) == moved and needle not in moved
    assert all(np.allclose(m.ghosts[n], b0[n]) for n in moved)


def check_views():
    """What the four views mean as turns, on generic strips, not only the
    dialogue's own; and the transcription turns the ball as the model does."""
    rng = np.random.default_rng(3)
    needles = ("glass", "steel", "brass")
    home = P.State(P.vec("up"), P.vec("company"))
    # ch03.lua curls a two-click arrow the other way in the page and flip
    # views; a half-turn either way round is the same turn, so nothing is lost
    for n in needles:
        assert np.allclose(play([(n, 2, +1)]), play([(n, 2, -1)]))
        assert not np.allclose(play([(n, 1, +1)]), play([(n, 1, -1)]))

    def random_spec(clicks, names):
        toks = []
        for _ in range(int(rng.integers(2, 8))):
            if toks and toks[-1] != P.SEAM and rng.random() < 0.2:
                toks.append(P.SEAM)
            toks.append("%s%s%d" % (rng.choice(names), rng.choice(["+", "-"]), rng.choice(clicks)))
        return ",".join(toks)

    def seen(items, view):
        return P.view_score(items, view)[0]

    def pieces(items):
        out = [[]]
        for it in items:
            if it == P.SEAM:
                out.append([])
            else:
                out[-1].append(it)
        return out

    # any strip at all: the page read after the strip brings the ball home,
    # read either way round throughout; round and flip do not in general
    fails = {"round": 0, "flip": 0}
    for _ in range(200):
        items = P.parse_score(random_spec((1, 2), needles))
        drawn_ = triples(P.arrows_of(items))
        paged = triples(P.arrows_of(seen(items, "page")))
        # the Lua's twirl about the needle as it lies = the model's twirl
        assert drawn_at(P.play(home, P.arrows_of(items)), play(drawn_))
        assert pieces(seen(items, "page")) == [seen(p, "page") for p in pieces(items)[::-1]]
        for s in (+1, -1):
            assert at_home(play(drawn_ + paged, s))
            assert np.allclose(play(paged, s), play(drawn_, s).T)
        for v in ("round", "flip"):
            fails[v] += not at_home(play(drawn_ + triples(P.arrows_of(seen(items, v)))))
    assert fails["round"] > 100 and fails["flip"] > 100      # astray more often than not
    # one arrow of one click: flip is the page, round is not
    one = P.parse_score("glass+1")
    assert seen(one, "flip") == seen(one, "page")
    assert not at_home(play(triples(one + seen(one, "round"))))

    # half-turns only -- the trap of the First Challenge.  Round always works
    # (every half-turn is its own undoing) and plays exactly as the page does.
    for _ in range(100):
        items = P.arrows_of(P.parse_score(random_spec((2,), needles)))
        assert at_home(play(triples(items + seen(items, "round"))))
        assert np.allclose(play(triples(seen(items, "round"))), play(triples(seen(items, "page"))))
    # flip works too while the needles are glass and steel, whose half-turns
    # do not mind their order ...
    for _ in range(100):
        items = P.arrows_of(P.parse_score(random_spec((2,), ("glass", "steel"))))
        assert at_home(play(triples(items + seen(items, "flip"))))
    # ... but not once the brass needle joins them: order matters again
    bg = P.parse_score("brass+2,glass+2")
    assert not at_home(play(triples(bg + seen(bg, "flip"))))
    assert at_home(play(triples(bg + seen(bg, "round"))))


def check_faults():
    """Every fault the macros raise, raised by the transcription (the build
    would stop): what the dialogue may not write."""
    ball, play_, strip_ = P.crabball, P.crabplay, P.scorestrip
    for fn, *args in (
            (ball, "up", "sideways", ""),                            # no such direction
            (ball, "up", "up", ""),                                  # not at right angles
            (ball, "up", "company", "play=glass+3"),                 # no three-click arrow
            (ball, "up", "company", "play=wood+1"),                  # no such needle
            (ball, "up", "company", "spin=glass+1"),                 # no such option
            (ball, "up", "company", "view=page"),                    # a view with nothing played
            (ball, "up", "company", "play=glass+1;then=sideways"),   # no such view
            (ball, "up", "company", "play=glass+1;marks=maybe"),
            (ball, "up", "company", "play=glass+1,steel+1;marks=on"),  # marks need one arrow
            (ball, "up", "company", "play=glass+1;expects=hope"),    # no such rule
            (ball, "up", "company", "play=glass+1;expects=anyend"),  # nothing read backwards
            (ball, "up", "company", "play=glass+1;then=round;expects=anyend;marks=on"),
            (ball, "up", "company", "play=glass+1;expects=oneturn"),   # first= needed
            (ball, "up", "company", "play=glass+1;first=steel+1"),     # first= is oneturn's
            (play_, "up", "company", "glass+1", "by=step", "one"),
            (play_, "up", "company", "glass+1", "marks=maybe", "one"),
            (play_, "up", "company", "glass+1", "play=glass+1", "one"),  # not a ball option
            (play_, "up", "company", "glass+1", "expects=anyorder", "one"),
            (play_, "up", "company", "", "", "nothing"),             # nothing to play
            (play_, "up", "company", "glass+1,steel+1", "", "one"),  # a caption a frame
            (strip_, "sideways", "glass+1")):
        assert P.faults(fn, *args), args
    # and what it may: one arrow is marked by default, an expects frame never
    assert P.faults(ball, "up", "company", "play=glass+1;marks=on") == []
    assert P.crabball("up", "company", "play=glass+1").marks is not None
    assert P.crabball("up", "company", "play=glass+1;then=round;expects=anyend").marks is None
    assert P.faults(ball, "up", "company", "play=glass+1;expects=oneturn;first=steel+1") == []
    assert P.faults(play_, "up", "company", "glass+1,steel+1", "", "one; two") == []


def place_named(paragraph, pattern):
    """A room place the text names, read out of a direction by a pattern."""
    m = re.search(pattern, paragraph)
    assert m and m.group(1) in PLACES, pattern
    return m.group(1)


def strip():
    P.check_lua()                                  # the Lua and .tex the transcription was proved on
    check_frame()
    check_views()
    check_faults()
    cw = +1
    one = [("glass", 1, cw)]
    trap = [("glass", 2, cw), ("steel", 2, cw)]
    swapped = [("steel", 2, cw), ("glass", 2, cw)]

    calls = DIALOGUE.check_layout("First", sizes=[2, 2, 1, 2])    # four moments
    P.check_rows(DIALOGUE, "First")
    home_pic, one_pic, her, her_page, steps, his, his_round = calls
    assert [c.macro for c in calls] == ["crabball", "crabball", "scorestrip", "scorestrip",
                                        "crabplay", "scorestrip", "scorestrip"]
    paras = DIALOGUE.paragraphs("First")
    a_one, a_strip, a_steps, a_round = DIALOGUE.anchors("First")

    # --- home, keyed; then the first twirl, with its marks -----------------
    ball = P.picture(home_pic)
    assert home_pic.keyed and ball.keyed and drawn_at(ball.after, np.eye(3))
    assert ball.marks is None and ball.expects is None
    ball = P.picture(one_pic)
    assert not one_pic.keyed and drawn_at(ball.start, np.eye(3))
    assert triples(ball.arrows) == one                  # glass, one click, with the clock
    assert drawn_at(ball.after, play(one))
    assert ball.dump == "crabball: glass=up steel=stove marks"
    marked(ball, np.eye(3), "glass", play(one))         # pin on the glass; steel, brass ghosted
    # the direction before it names each mark in words, at its first use
    assert a_one == "**Achilles:** The steel bead lies level and points at the stove."
    twirl = paras[paras.index(a_one) - 1]
    for phrase in ("The glass bead is the pin the turn is made about: it does not stir",
                   "the steel bead, that had pointed at the company",
                   "swings a quarter of the way round"):
        assert phrase in twirl, phrase

    # --- the Tortoise's strip, as drawn and turned like a page --------------
    s1, s2 = P.picture(her), P.picture(her_page)
    assert s1.view == "drawn" and s2.view == "page" and s2.back and not s1.back
    assert s2.drawn == s1.drawn                         # the same strip, turned over
    assert triples(s1.shown) == one
    assert triples(s2.shown) == [("glass", 1, -cw)] == page(one)   # one click against
    for s in (+1, -1):                                  # played from where it stands: home
        assert at_home(play(triples(s1.shown + s2.shown), s))

    # --- Achilles's score, click by click: the step strip -----------------
    frames = P.picture(steps)
    assert dict(P.parse_opts(steps.opt, P.PLAYOPTS, "test")) == {"by": "click"}
    clicks = [(n, 1, w) for n, c, w in trap for _ in range(c)]
    assert len(frames) == len(clicks) == 4              # four frames, one line
    assert [f.caption for f in frames] == [c.strip() for c in steps.caption.split(";")]
    assert all(words_only(f.caption) for f in frames)
    for k, f in enumerate(frames, 1):                   # frame k = the model after k clicks
        before, after = play(clicks[:k - 1]), play(clicks[:k])
        assert triples([f.step]) == [clicks[k - 1]]
        assert drawn_at(f.ball.start, before) and drawn_at(f.ball.after, after)
        marked(f.ball, before, clicks[k - 1][0], after)
    assert drawn_at(frames[-1].ball.after, play(trap))
    assert reading(play(trap), "glass") == "down" and reading(play(trap), "steel") == "away"
    # mid-score the beads leave their home lines, as the direction says: the
    # steel round by the stove, then the glass round by the stove
    assert a_steps.startswith("**Achilles:** Glass bead straight down")
    assert DIALOGUE.following("First")[2] == "**Tortoise:** And to undo it?"
    walk = paras[paras.index(a_steps) - 1]
    steel_by = place_named(walk, r"off the company and round by the (\w+)\.")
    glass_by = place_named(walk, r"from the top and round by the (\w+)\.")
    assert np.allclose(frames[0].ball.after.s, P.vec(steel_by))
    assert reading(play(clicks[:1]), "steel") == steel_by
    assert np.allclose(frames[2].ball.after.g, P.vec(glass_by))
    assert reading(play(clicks[:3]), "glass") == glass_by
    # the trap: the other order lands at the very place the strip's last frame shows
    assert drawn_at(frames[-1].ball.after, play(swapped))
    assert any("the very place the first order left it" in p for p in paras)
    # while with one click each the order would matter
    assert not np.allclose(play([("glass", 1, cw), ("steel", 1, cw)]),
                           play([("steel", 1, cw), ("glass", 1, cw)]))

    # --- his strip, as drawn and round on the table: the trap springs -----
    s3, s4 = P.picture(his), P.picture(his_round)
    assert s3.view == "drawn" and s4.view == "round" and not s4.back
    assert s4.drawn == s3.drawn
    assert triples(s3.shown) == trap                    # glass two, then steel two
    assert triples(s4.shown) == swapped == round_on_table(trap)
    for s in (+1, -1):                                  # home -- the trap springs
        assert at_home(play(triples(s3.shown + s4.shown), s))
    assert a_strip.startswith("*[She holds the strip by its short upright edge")
    assert a_round.startswith("*[Achilles lays his strip flat")


# --------------------------------------------------------------------------
# The sketchbook.  PyZX draws this with the spiders you meet in Chapter 8.
# --------------------------------------------------------------------------
def chain(*nodes):
    """A box built as a row of phase nodes on one wire."""
    p = Picture()
    a, b = p.inp(0), p.out(0)
    vs = [(p.z if kind == "z" else p.x)(0, col + 1, ph)
          for col, (kind, ph) in enumerate(nodes)]
    p.wire(a, *vs, b)
    return p.done()


def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def hadamard():
    """The box H, drawn as a marked edge: exactly (1/sqrt 2)[[1,1],[1,-1]]."""
    p = Picture()
    a, b = p.inp(0), p.out(0)
    p.g.add_edge((a, b), zx.EdgeType.HADAMARD)
    return p.done()


def sketchbook():
    wire = bare_wire()
    zbox, xbox = chain(("z", 1)), chain(("x", 1))
    sbox, tbox = chain(("z", Fraction(1, 2))), chain(("z", Fraction(1, 4)))
    hbox = hadamard()
    assert np.allclose(matrix_of(hbox), H)              # exact, scalar included

    # the test piece as a row: S first, then V = HSH, which PyZX draws x(1/2)
    vbox = chain(("x", Fraction(1, 2)))
    assert np.allclose(matrix_of(vbox), H @ S @ H)
    u = sbox + vbox                                     # a row: g1 then g2
    um = matrix_of(u)
    assert np.allclose(um, matrix_of(vbox) @ matrix_of(sbox))   # row order
    assert not np.allclose(um, um.T) and not np.allclose(um, um.conj())

    # the mirror image, and the two cheaper reflections
    assert np.allclose(matrix_of(u.adjoint()), um.conj().T)
    assert same_picture(u + u.adjoint(), wire)          # a box meets its mirror
    assert same_picture(u.adjoint() + u, wire)          # in either order
    kept_order = sbox.adjoint() + vbox.adjoint()        # pieces mirrored, order kept
    kept_pieces = vbox + sbox                           # order reversed, pieces kept
    assert np.allclose(matrix_of(kept_order), um.conj())
    assert np.allclose(matrix_of(kept_pieces), um.T)
    assert not same_picture(u + kept_order, wire)
    assert not same_picture(u + kept_pieces, wire)
    assert not same_picture(kept_order, u.adjoint())
    assert not same_picture(kept_pieces, u.adjoint())

    # a hand-built mirror of a generic box: the row reversed with every phase
    # negated.  (Reversal alone gives the transpose, negation alone the
    # conjugate; both fail against the wire.)
    generic = chain(("z", Fraction(1, 4)), ("x", Fraction(1, 3)), ("z", Fraction(3, 4)))
    gm = matrix_of(generic)
    assert not np.allclose(gm, gm.T) and not np.allclose(gm, gm.conj())
    by_hand = chain(("z", Fraction(-3, 4)), ("x", Fraction(-1, 3)), ("z", Fraction(-1, 4)))
    assert same_picture(by_hand, generic.adjoint())
    assert same_picture(generic + by_hand, wire)
    reversed_only = chain(("z", Fraction(3, 4)), ("x", Fraction(1, 3)), ("z", Fraction(1, 4)))
    negated_only = chain(("z", Fraction(-1, 4)), ("x", Fraction(-1, 3)), ("z", Fraction(-3, 4)))
    assert np.allclose(matrix_of(reversed_only), gm.T)
    assert np.allclose(matrix_of(negated_only), gm.conj())
    assert not same_picture(generic + reversed_only, wire)
    assert not same_picture(generic + negated_only, wire)

    # four boxes that are their own mirror images, hence their own undoings
    ybox = zbox + xbox                          # the row Z then X, whose matrix is XZ
    ybox.scalar.add_phase(Fraction(1, 2))       # Y = iXZ: the phase is +pi/2
    assert np.allclose(matrix_of(ybox), Y)
    for g in (xbox, ybox, zbox, hbox):
        assert same_picture(g.adjoint(), g)
        assert same_picture(g + g, wire)
    # ... and two that are not
    for g in (sbox, tbox):
        assert not same_picture(g.adjoint(), g)
        assert same_picture(g + g.adjoint(), wire)
    assert not same_picture(sbox + sbox, wire)
    assert same_picture(sbox + sbox, zbox)              # the ladder
    assert same_picture(tbox + tbox, sbox)


if __name__ == "__main__":
    verdict("Mirror", blackboard, sketchbook)
