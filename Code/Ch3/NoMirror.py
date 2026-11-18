"""Challenge 5: What Has No Mirror
===================================
"The page would have to lead to two places at once. ...Oh. It can't be drawn."
  -- Achilles

Claim: SET-0 does not keep the promise -- it meets its mirror image in
something that is not a bare wire, and it has no undoing at all -- while every
row of turns, however long, is undone by its page.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import Picture, ket, matrix_of, random_matrix, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch03 as P


X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
I2 = np.eye(2, dtype=complex)
SET0 = np.array([[1, 1], [0, 0]], dtype=complex)     # both answers go to 0
D = np.diag([1, 2]).astype(complex)                  # invertible, still no good
ARROW = np.array([3 / 5, 4 / 5], dtype=complex)      # blackboard only, as in Chapter 2


def random_unitary(rng):
    q, r = np.linalg.qr(random_matrix(rng))
    return q @ np.diag(np.diag(r) / np.abs(np.diag(r)))


def blackboard():
    rng = np.random.default_rng(35)

    # --- SET-0 meets its mirror in something that is not a bare wire -------
    assert np.allclose(SET0 @ ket(0), ket(0)) and np.allclose(SET0 @ ket(1), ket(0))
    assert np.allclose(SET0.conj().T @ SET0, np.ones((2, 2)))
    assert not np.allclose(SET0.conj().T @ SET0, I2)
    assert not np.allclose(SET0 @ SET0.conj().T, I2)
    assert np.allclose(SET0 @ ARROW, np.array([7 / 5, 0]))
    assert np.isclose(np.vdot(SET0 @ ARROW, SET0 @ ARROW), 49 / 25)
    assert np.isclose(np.vdot(ARROW, ARROW), 1)         # it was a state before

    # --- and no undoing at all: it loses the question ----------------------
    assert np.isclose(np.linalg.det(SET0), 0)
    assert not np.allclose(ket(0), ket(1))
    assert np.allclose(SET0 @ ket(0), SET0 @ ket(1))    # two places, one image
    for _ in range(20):
        psi = random_state(rng)
        other = psi.copy()
        other[1] = -other[1]                            # a different state ...
        assert not np.allclose(psi, other)
        assert np.allclose(SET0 @ (psi + other), SET0 @ psi + SET0 @ other)
    # so there is no B with B SET-0 = I: the page would lead to two places
    for _ in range(20):
        B = random_matrix(rng)
        assert not np.allclose(B @ SET0, I2)

    # --- what keeping the promise buys: every number between states stays --
    for _ in range(50):
        U = random_unitary(rng)
        phi, psi = random_state(rng), random_state(rng)
        before = np.vdot(phi, psi)
        assert not np.isclose(before.imag, 0)           # a number with a corner
        assert np.isclose(np.vdot(U @ phi, U @ psi), before)
        assert not np.isclose(np.vdot(SET0 @ phi, SET0 @ psi), before)

    # --- invertible is not enough --------------------------------------------
    assert not np.isclose(np.linalg.det(D), 0)
    assert np.allclose(np.linalg.inv(D), np.diag([1, 0.5]))
    assert np.isclose(np.vdot(D @ ARROW, D @ ARROW), 73 / 25)
    assert not np.isclose(np.vdot(D @ ARROW, D @ ARROW), 1)
    assert not np.allclose(D.conj().T @ D, I2)

    beads()
    strip()


# --------------------------------------------------------------------------
# The ball.  A position is a rotation R of ordinary space: the bead of a
# needle whose home direction is b now points along R @ b.  A twirl is made
# about a needle AS IT LIES, so its axis is R @ b and R gains the turn on the
# right.  "With the clock" is the sense seen from that needle's own bead.
# --------------------------------------------------------------------------
UP = np.array([0.0, 0.0, 1.0])
COMPANY = np.array([0.0, 1.0, 0.0])
STOVE = np.array([1.0, 0.0, 0.0])            # the company's left; dresser is -STOVE
HOME = {"glass": UP, "steel": COMPANY, "brass": (UP + COMPANY) / np.sqrt(2)}
PLACES = {"up": UP, "down": -UP, "company": COMPANY, "away": -COMPANY,
          "stove": STOVE, "dresser": -STOVE}


# the evening's strips, gummed end to end, all with the clock
LONG_SCORE = ([("glass", 1, +1)]                        # the Tortoise's one arrow
              + [("glass", 1, +1), ("steel", 1, +1)]    # Achilles's strip
              + [("brass", 2, +1), ("glass", 2, +1), ("brass", 2, +1)])


def turn(axis, angle):
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def play(score, sense=+1):
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
    return [(n, c, -w) for n, c, w in reversed(score)]


def round_on_table(score):
    return list(reversed(score))


def top_over_bottom(score):
    return [(n, c, -w) for n, c, w in score]


def beads():
    cw = +1

    # --- the catch is not a turn: it takes two places to the same one ------
    first = play([("steel", 1, cw)])
    assert reading(first, "glass") == "dresser"
    assert reading(first, "steel") == "company"
    second = play([("glass", 1, -cw), ("steel", 2, cw)])
    assert reading(second, "glass") == "down"
    assert reading(second, "steel") == "dresser"
    assert not np.allclose(first, second)
    # from the first place, the catch did what steel one click against the
    # clock would have done
    assert np.allclose(first @ play([("steel", 1, -cw)]), np.eye(3))
    # and the only turn that brings a position home is that position undone --
    # so one turn cannot serve both places
    assert np.allclose(first.T @ first, np.eye(3))
    assert np.allclose(second.T @ second, np.eye(3))
    assert not np.allclose(first.T, second.T)

    # --- the evening's strips, gummed end to end ---------------------------
    long_score = LONG_SCORE
    R = play(long_score)
    assert reading(R, "glass") == "dresser"                # level, sideways ...
    assert np.allclose(PLACES["dresser"], -PLACES["stove"])  # ... the far side from
    assert reading(R, "steel") == "away"                     # where Challenge 2 sent
    assert np.allclose(R @ HOME["brass"], (-COMPANY - STOVE) / np.sqrt(2))
    assert not at_home(R)

    # read off the back of the strip, it is played exactly as Achilles reads it
    assert page(long_score) == [("brass", 2, -cw), ("glass", 2, -cw),
                                ("brass", 2, -cw), ("steel", 1, -cw),
                                ("glass", 1, -cw), ("glass", 1, -cw)]
    assert at_home(play(long_score + page(long_score)))
    # the two cheaper reflections leave it astray
    assert not at_home(play(long_score + round_on_table(long_score)))
    assert not at_home(play(long_score + top_over_bottom(long_score)))
    assert not np.allclose(play(round_on_table(long_score)), play(page(long_score)))
    assert not np.allclose(play(top_over_bottom(long_score)), play(page(long_score)))
    for s in (+1, -1):
        assert at_home(play(long_score + page(long_score), s))
        assert not at_home(play(long_score + round_on_table(long_score), s))
        assert not at_home(play(long_score + top_over_bottom(long_score), s))


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


def check_turning(seed):
    """The transcription turns the ball as the model does, on generic scores:
    the Lua's twirl about the held needle as it lies is the model's."""
    rng = np.random.default_rng(seed)
    home = P.State(P.vec("up"), P.vec("company"))
    for _ in range(100):
        score = [(str(rng.choice(["glass", "steel", "brass"])), int(rng.integers(1, 3)),
                  int(rng.choice([-1, 1]))) for _ in range(int(rng.integers(1, 7)))]
        assert drawn_at(P.play(home, [P.Arrow(n, w, c) for n, c, w in score]), play(score))


def catch(R):
    """The reset catch: from wherever the ball stands, home.  Not a turn."""
    return np.eye(3)


ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth")


def check_whole_dialogue():
    """Every picture of the dialogue stands in a challenge's section; at most
    four moments a challenge (a step strip is one), at most four pictures to a
    row (a step strip counts its frames); the keyed ball comes first."""
    total = DIALOGUE.check_whole(ORDINALS)
    blocks = {o: P.check_rows(DIALOGUE, o) for o in ORDINALS}
    assert [len(blocks[o]) for o in ORDINALS] == [4, 4, 3, 1, 4]       # moments
    calls = [c for o in ORDINALS for b in blocks[o] for c in b]
    assert total == len(calls) == 33
    # thirteen strips, none in the Fourth, whose scores are only spoken
    strips = [sum(c.macro == "scorestrip" for b in blocks[o] for c in b) for o in ORDINALS]
    assert strips == [4, 4, 2, 0, 3] and sum(strips) == 13
    # three step strips: the First's, and the Second's walk out and walk back,
    # and the Third's
    assert [o for o in ORDINALS for b in blocks[o] for c in b if c.macro == "crabplay"] == \
        ["First", "Second", "Second", "Third"]
    # the strip is never starred; the ball (a picture or a step strip) is keyed
    # once, on the first picture of it
    DIALOGUE.keyed_first(ORDINALS, kind=lambda c: "ball", select=lambda c: c.macro != "scorestrip")
    keyed = [c for c in calls if c.keyed]
    assert len(keyed) == 1 and keyed[0] == blocks["First"][0][0] and keyed[0].caption == "home"
    # every caption words only -- each frame of a step strip its own -- and no
    # picture faults
    for c in calls:
        out = P.picture(c)
        for cap in ([f.caption for f in out] if c.macro == "crabplay" else [c.caption]):
            assert cap and words_only(cap), cap
    # two pictures are predictions, each a named wrong rule; no oneturn tonight
    rules = [P.picture(c).expects for c in calls if c.macro == "crabball"]
    assert [r for r in rules if r] == ["anyend", "anyorder"]


def strip():
    P.check_lua()                                  # the Lua and .tex the transcription was proved on
    check_frame()
    check_turning(53)
    check_whole_dialogue()
    cw = +1
    first = play([("steel", 1, cw)])
    second = play([("glass", 1, -cw), ("steel", 2, cw)])
    assert np.allclose(catch(first), catch(second))     # two places, one image

    calls = DIALOGUE.check_layout("Fifth", sizes=[3, 1, 2, 2])     # four moments
    P.check_rows(DIALOGUE, "Fifth")
    one_pic, another_pic, catch_pic, blank_pic, gummed_pic, gummed_page, \
        score_pic, page_pic = calls
    assert [c.macro for c in calls] == ["crabball"] * 3 + ["scorestrip"] * 3 + ["crabball"] * 2
    assert [c.row for c in (gummed_pic, gummed_page)] == [0, 1]
    a_catch, a_blank, a_long, a_home = DIALOGUE.anchors("Fifth")

    # --- two places, and the catch from either ------------------------------
    ball = P.picture(one_pic)
    assert dict(P.parse_opts(one_pic.opt, P.BALLOPTS, "test")) == {"play": "steel+1", "marks": "off"}
    assert triples(ball.arrows) == [("steel", 1, cw)] and drawn_at(ball.start, np.eye(3))
    assert drawn_at(ball.after, first) and ball.marks is None
    assert P.crabball("up", "company", "play=steel+1").marks is not None   # marks=off bites
    ball = P.picture(another_pic)
    assert triples(ball.arrows) == [("glass", 1, -cw), ("steel", 2, cw)]
    assert drawn_at(ball.after, second) and ball.marks is None          # two arrows: no marks
    assert not np.allclose(first, second)
    ball = P.picture(catch_pic)                 # hand-given: the catch travels no path
    assert catch_pic.opt == "" and ball.arrows == () and ball.marks is None
    assert drawn_at(ball.after, catch(first)) and drawn_at(ball.after, catch(second))
    assert a_catch.startswith("**Crab:** *[He presses the catch, and from here too")

    # --- "Draw the catch." The strip stays blank, whichever way it lies, and a
    # blank strip is no turn at all: from the first place it leaves the ball
    # where it was, not home.  The catch is not on the strip.
    blank = P.picture(blank_pic)
    assert blank.drawn == blank.shown == ()
    assert all(P.view_score([], v)[0] == [] for v in P.VIEWS)
    assert np.allclose(first @ play(triples(blank.shown)), first)
    assert not np.allclose(first @ play(triples(blank.shown)), catch(first))
    # and a ball cannot be made to play it: a step strip of nothing is a fault
    assert P.faults(P.crabplay, "up", "company", "", "", "the catch")

    # --- the evening's strips, gummed end to end in the order they were drawn:
    # her one blue arrow, Achilles's strip with the heads rubbed out, brass,
    # glass, brass -- each read out of its own challenge, not copied here
    gummed, again = P.picture(gummed_pic), P.picture(gummed_page)
    assert gummed.view == "drawn" and again.view == "page" and again.drawn == gummed.drawn
    first_strips = [next(P.picture(c) for b in DIALOGUE.blocks(o) for c in b
                         if c.macro == "scorestrip") for o in ("First", "Second", "Third")]

    def pieces(items):
        out = [[]]
        for it in items:
            if it == P.SEAM:
                out.append([])
            else:
                out[-1].append(it)
        return out

    assert pieces(gummed.drawn) == [list(s.drawn) for s in first_strips]
    assert triples(P.arrows_of(gummed.drawn)) == LONG_SCORE
    # turned like a page: the pieces in the reverse order, each one paged
    assert pieces(again.shown) == [P.view_score(q, "page")[0] for q in pieces(gummed.drawn)[::-1]]
    # which is what Achilles reads: brass, glass, brass; then steel, glass and
    # glass, one click each, every one against the clock
    as_drawn, as_page = triples(P.arrows_of(gummed.shown)), triples(P.arrows_of(again.shown))
    assert as_page == [("brass", 2, -cw), ("glass", 2, -cw), ("brass", 2, -cw),
                       ("steel", 1, -cw), ("glass", 1, -cw), ("glass", 1, -cw)]
    assert as_page == page(LONG_SCORE)
    for s in (+1, -1):
        assert not at_home(play(as_drawn, s))
        assert at_home(play(as_drawn + as_page, s))
    assert a_long.startswith("**Tortoise:** *[She turns the whole long strip over like a page.]*")

    # --- the long score played, and its page: home ------------------------
    for pic, then, R in ((score_pic, None, play(LONG_SCORE)),
                         (page_pic, "page", play(LONG_SCORE + page(LONG_SCORE)))):
        ball = P.picture(pic)
        o = P.parse_opts(pic.opt, P.BALLOPTS, "test")
        assert o.get("play") == gummed_pic.args[0] and o.get("then") == then  # the strip drawn
        assert drawn_at(ball.start, np.eye(3)) and drawn_at(ball.after, R)
        assert ball.marks is None and ball.expects is None
    assert reading(play(LONG_SCORE), "glass") == "dresser" and reading(play(LONG_SCORE), "steel") == "away"
    assert at_home(play(LONG_SCORE + page(LONG_SCORE)))
    assert a_home == "**Achilles:** Home — by turns alone, to the last click."


# --------------------------------------------------------------------------
# The sketchbook.  PyZX draws this with the spiders you meet in Chapter 8.
# --------------------------------------------------------------------------
def chain(*nodes):
    p = Picture()
    a, b = p.inp(0), p.out(0)
    vs = [(p.z if kind == "z" else p.x)(0, col + 1, ph)
          for col, (kind, ph) in enumerate(nodes)]
    p.wire(a, *vs, b)
    return p.done()


def set_zero():
    """SET-0 as a picture: the input is closed off, a fresh 0 is opened."""
    p = Picture()
    a, b = p.inp(0), p.out(0)
    p.wire(a, p.z(0, 1, 0))          # the wire ends here
    p.wire(p.x(0, 2, 0), b)          # and a new one begins
    g = p.done()
    g.scalar.add_power(-1)
    return g


def basis_state(b):
    p = Picture()
    n = p.x(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def generic_state():
    p = Picture()
    n = p.x(0, 1, 0)
    vs = [p.z(0, 2, Fraction(1, 4)), p.x(0, 3, Fraction(1, 3)), p.z(0, 4, Fraction(3, 4))]
    p.wire(n, *vs, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def value(g):
    return matrix_of(g)[0, 0]


def sketchbook():
    wire = chain(("z", 0))
    s0 = set_zero()
    assert np.allclose(matrix_of(s0), SET0)             # the picture is SET-0
    assert np.allclose(matrix_of(wire), I2)

    # a box that does keep the promise, and is neither real nor symmetric
    u = chain(("z", Fraction(1, 2))) + chain(("x", Fraction(1, 2)))
    um = matrix_of(u)
    assert not np.allclose(um, um.T) and not np.allclose(um, um.conj())
    assert same_picture(u + u.adjoint(), wire)
    assert same_picture(u.adjoint() + u, wire)

    # --- SET-0 meets its mirror in something that is not a bare wire -------
    assert not same_picture(s0 + s0.adjoint(), wire)
    assert not same_picture(s0.adjoint() + s0, wire)
    # SET-0 then its mirror is the box of all ones, not a bare wire
    assert np.allclose(matrix_of(s0 + s0.adjoint()), SET0.conj().T @ SET0)
    assert np.allclose(matrix_of(s0 + s0.adjoint()), np.ones((2, 2)))

    # --- a closed picture that should have been a certainty ----------------
    psi = generic_state()
    assert np.isclose(value(psi + psi.adjoint()), 1)
    assert np.isclose(value(psi + u + u.adjoint() + psi.adjoint()), 1)
    assert not np.isclose(value(psi + s0 + s0.adjoint() + psi.adjoint()), 1)

    # --- the Echo, with a box in it ----------------------------------------
    # Chapter 2's echo: what leaves by every answer comes back to 1 -- but
    # only when the box between keeps the promise.
    for box, total in ((u, 1), (s0, None)):
        echo = 0
        for b in (0, 1):
            out = value(psi + box + basis_state(b).adjoint())
            back = value(basis_state(b) + box.adjoint() + psi.adjoint())
            assert np.isclose(back, np.conj(out))
            echo += out * back
        assert np.isclose(echo.imag, 0)
        if total is not None:
            assert np.isclose(echo, 1)
        else:
            assert not np.isclose(echo, 1)
            # it is the squared length of what SET-0 left, and nothing else
            state = matrix_of(psi).ravel()
            assert np.isclose(echo, np.vdot(SET0 @ state, SET0 @ state))

    # --- and the row of turns, however long, is undone by its page ---------
    row = (chain(("z", Fraction(1, 4))) + chain(("x", Fraction(1, 3)))
           + chain(("z", Fraction(3, 5))) + u)
    assert same_picture(row + row.adjoint(), wire)
    assert same_picture(row.adjoint() + row, wire)
    assert not same_picture(row + row, wire)            # a row is not its own page
    with_catch = row + s0
    assert not same_picture(with_catch + with_catch.adjoint(), wire)


if __name__ == "__main__":
    verdict("NoMirror", blackboard, sketchbook)
