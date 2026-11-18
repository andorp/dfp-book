"""Challenge 2: The Score Read Backwards
=========================================
"...Oh. Both at once. Last arrow first *and* every arrow the other way
round, and neither half will do alone."  -- Achilles

Claim: the mirror image of a row is the mirrored pieces in the reverse order,
(BA)^dagger = A^dagger B^dagger; reversing the order alone, or mirroring the
pieces alone, is not an undoing -- not even as a turn of the sphere.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pyzx as zx
from dfp import Picture, ket, matrix_of, random_matrix, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch03 as P


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
I2 = np.eye(2, dtype=complex)
PAULI = (X, Y, Z)
ARROW = np.array([3 / 5, 4 / 5], dtype=complex)      # blackboard only
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)


def sphere(M):
    """The turn of the sphere that the box M performs: the real 3x3 matrix R
    with new_j = sum_k R[j,k] old_k for the three closed pictures of Sec 3.5."""
    return np.array([[0.5 * np.trace(Pk @ M.conj().T @ Pj @ M).real
                      for Pk in PAULI] for Pj in PAULI])


def point(psi):
    return np.array([np.vdot(psi, P @ psi).real for P in PAULI])


def blackboard():
    rng = np.random.default_rng(32)

    # --- the mirror of a row, on generic matrices --------------------------
    A, B = random_matrix(rng), random_matrix(rng)
    for M in (A, B):
        assert not np.allclose(M, M.T) and not np.allclose(M, M.conj())
    row = B @ A                                         # the row "A, then B"
    assert np.allclose(row.conj().T, A.conj().T @ B.conj().T)
    assert not np.allclose(row.conj().T, B.conj().T @ A.conj().T)   # order matters
    assert not np.allclose(row.conj().T, (B @ A).T)                 # so does the bar

    # the bra of a turned state, and the conjugate of a closed picture
    psi, phi = random_state(rng), random_state(rng)
    assert not np.allclose(psi.imag, 0) and not np.allclose(phi.imag, 0)
    assert np.allclose((A @ psi).conj(), psi.conj() @ A.conj().T)
    value = np.vdot(phi, A @ psi)
    assert not np.isclose(value.imag, 0)                # else the bar is invisible
    assert np.isclose(np.conj(value), np.vdot(psi, A.conj().T @ phi))
    assert not np.isclose(np.conj(value), np.vdot(psi, A.T @ phi))
    assert not np.isclose(np.conj(value), np.vdot(psi, A.conj() @ phi))

    # --- the chapter's test piece: the row S, then V = HSH ------------------
    V = H @ S @ H
    U = V @ S
    assert not np.allclose(U, U.T) and not np.allclose(U, U.conj())
    assert np.allclose(U.conj().T @ U, I2)
    assert np.allclose(U.conj().T, S.conj().T @ V.conj().T)         # mirror of the row
    # the two cheaper recipes, exactly as Sec 3.2 computes them
    mirrored_only = V.conj().T @ S.conj().T            # "S-dagger, then V-dagger"
    reversed_only = S @ V                              # "V, then S"
    assert np.allclose(mirrored_only, U.conj())
    assert np.allclose(reversed_only, U.T)
    assert not np.allclose(mirrored_only, U.conj().T)
    assert not np.allclose(reversed_only, U.conj().T)
    assert not np.allclose(mirrored_only @ U, I2)
    assert not np.allclose(reversed_only @ U, I2)
    assert np.allclose(reversed_only @ U, 1j * X)      # Sec 3.4's "nowhere near"

    # and they fail even as turns of the sphere, where a global phase is invisible
    R = sphere(U)
    assert np.allclose(R @ R.T, np.eye(3)) and np.allclose(R.imag, 0)
    for _ in range(50):
        chi = random_state(rng)
        assert np.allclose(point(U @ chi), R @ point(chi))
    assert np.allclose(sphere(U.conj().T) @ R, np.eye(3))           # the mirror does undo
    assert not np.allclose(sphere(mirrored_only) @ R, np.eye(3))
    assert not np.allclose(sphere(reversed_only) @ R, np.eye(3))

    # the worked instance of Sec 3.2, with a number that is not real
    assert np.isclose(np.vdot(PLUS, U @ ARROW), (3 + 4j) / (5 * np.sqrt(2)))
    assert np.isclose(np.vdot(ARROW, U.conj().T @ PLUS), (3 - 4j) / (5 * np.sqrt(2)))
    assert not np.isclose(np.vdot(ARROW, U.T @ PLUS), (3 - 4j) / (5 * np.sqrt(2)))
    assert not np.isclose(np.vdot(ARROW, U.conj() @ PLUS), (3 - 4j) / (5 * np.sqrt(2)))

    # the sloppy rule of Challenge 1 dies here too: HZ and ZH differ by no phase
    assert not np.allclose(H @ Z, Z @ H)
    assert np.isclose((H @ Z)[0, 0], (Z @ H)[0, 0]) and not np.isclose((H @ Z)[0, 1], (Z @ H)[0, 1])
    assert np.allclose(np.vdot(ket(0), Y @ ket(0)), 0)

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
    score = [("glass", 1, cw), ("steel", 1, cw)]
    assert reading(play(score), "glass") == "company"
    assert reading(play(score), "steel") == "stove"
    assert not at_home(play(score))

    # Achilles's first way: the strip round on the table.
    wrong_a = play(score + round_on_table(score))
    assert reading(wrong_a, "glass") == "down"
    assert reading(wrong_a, "steel") == "company"       # steel looks at home
    assert not at_home(wrong_a)

    # his second: the strip flipped top over bottom.  A different failure.
    wrong_b = play(score + top_over_bottom(score))
    assert reading(wrong_b, "glass") == "stove"
    assert reading(wrong_b, "steel") == "down"
    assert not at_home(wrong_b)
    assert not np.allclose(wrong_a, wrong_b)

    # the page: home.
    assert at_home(play(score + page(score)))

    # In the scene the Crab does not reset the ball between the two failures:
    # he backs out Achilles's clicks last first and each the other way, which
    # is the page of what Achilles just played, and lands on the same position
    # the score itself reaches.  So playing afresh from home is the same test.
    for wrong in (round_on_table(score), top_over_bottom(score)):
        assert np.allclose(play(score + wrong + page(wrong)), play(score))

    # every yes/no answer above is the same read the other way round throughout
    for s in (+1, -1):
        assert not at_home(play(score, s))
        assert at_home(play(score + page(score), s))
        assert not at_home(play(score + round_on_table(score), s))
        assert not at_home(play(score + top_over_bottom(score), s))
    # and sideways changes hands
    assert reading(play(score, -1), "steel") == "dresser"
    assert reading(play(score, -1), "glass") == "company"


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


def any_end(score, undo):
    """Achilles's rule, "play it again from the other end -- any other end":
    whichever backwards reading `undo` he plays, he expects what the page does."""
    assert undo in (round_on_table, top_over_bottom, page)
    return score + page(score)


def strip():
    P.check_lua()                                  # the Lua and .tex the transcription was proved on
    check_frame()
    check_views()
    cw = +1
    score = [("glass", 1, cw), ("steel", 1, cw)]
    rot, flip = round_on_table(score), top_over_bottom(score)

    calls = DIALOGUE.check_layout("Second", sizes=[1, 6, 1, 1])   # four moments
    P.check_rows(DIALOGUE, "Second")
    walk_out, drawn_pic, round_pic, flip_pic, expect_pic, round_ball, flip_ball, \
        page_pic, walk_back = calls
    assert [c.macro for c in calls] == ["crabplay"] + ["scorestrip"] * 3 + ["crabball"] * 3 \
        + ["scorestrip", "crabplay"]
    assert [c.row for c in calls[1:7]] == [0, 0, 0, 1, 1, 1]     # strips over balls
    paras = DIALOGUE.paragraphs("Second")
    a_out, a_rules, a_page, a_back = DIALOGUE.anchors("Second")
    f_out, f_rules, f_page, f_back = DIALOGUE.following("Second")

    # --- the score played, one arrow a frame: the walk out --------------------
    frames = P.picture(walk_out)
    assert walk_out.opt == "" and len(frames) == len(score) == 2
    assert [f.caption for f in frames] == ["first the glass", "then the steel"]
    assert all(words_only(f.caption) for f in frames)
    for k, f in enumerate(frames, 1):
        before, after = play(score[:k - 1]), play(score[:k])
        assert triples([f.step]) == [score[k - 1]]
        assert drawn_at(f.ball.start, before) and drawn_at(f.ball.after, after)
        marked(f.ball, before, score[k - 1][0], after)
    # the second twirl is about the steel needle where the first left it, at
    # the stove: the ring sits there, and the glass bead leaves the top
    assert reading(play(score[:1]), "steel") == "stove"
    assert np.allclose(frames[1].ball.marks.hold, P.vec("stove"))
    assert np.allclose(frames[1].ball.marks.ghosts["glass"], P.vec("up"))
    assert "Then the steel needle, held where it now lies: it keeps its place" in a_out
    assert f_out.startswith("**Achilles:** Now, my rule.")
    assert drawn_at(frames[-1].ball.after, play(score))
    assert reading(play(score), "glass") == "company" and reading(play(score), "steel") == "stove"

    # --- the strips: as drawn, round on the table, top over bottom ------------
    shown = [P.picture(c) for c in (drawn_pic, round_pic, flip_pic, page_pic)]
    assert [s.view for s in shown] == ["drawn", "round", "flip", "page"]
    assert all(triples(s.drawn) == score for s in shown)          # one strip, four ways of lying
    as_drawn, as_round, as_flip, as_page = [triples(s.shown) for s in shown]
    assert as_drawn == score
    assert as_round == [("steel", 1, cw), ("glass", 1, cw)] == rot     # his first rule
    assert as_flip == [("glass", 1, -cw), ("steel", 1, -cw)] == flip   # his second
    assert as_page == [("steel", 1, -cw), ("glass", 1, -cw)] == page(score)
    for s in (+1, -1):
        assert not at_home(play(as_drawn + as_round, s))
        assert not at_home(play(as_drawn + as_flip, s))
        assert at_home(play(as_drawn + as_page, s))     # both at once
    assert not np.allclose(play(as_drawn + as_round), play(as_drawn + as_flip))

    # --- what each rule did, and what Achilles expected of it -----------------
    # In the scene the Crab backs each failure out, last click first and each
    # the other way, before the next is tried -- so each rule is in effect
    # played from where the score left the ball: "played, then undone that way".
    after_rot = as_drawn + as_round
    after_flip = after_rot + page(as_round) + as_flip
    assert np.allclose(play(after_flip), play(as_drawn + as_flip))
    for ball_pic, undo, R in ((round_ball, "round", play(after_rot)),
                              (flip_ball, "flip", play(after_flip))):
        ball = P.picture(ball_pic)
        assert dict(P.parse_opts(ball_pic.opt, P.BALLOPTS, "test")) == \
            {"play": "glass+1,steel+1", "then": undo}
        assert drawn_at(ball.start, np.eye(3)) and drawn_at(ball.after, R)
        assert ball.marks is None and ball.expects is None      # two arrows and more: no marks
    assert reading(play(after_rot), "glass") == "down" and reading(play(after_rot), "steel") == "company"
    assert reading(play(after_flip), "glass") == "stove" and reading(play(after_flip), "steel") == "down"

    # the prediction: pale, no marks, computed by the rule he named -- home
    expected = P.picture(expect_pic)
    assert expect_pic.caption == "as Achilles expects"
    assert expected.expects == "anyend" and expected.marks is None
    assert drawn_at(expected.start, np.eye(3))
    assert np.allclose(play(any_end(score, round_on_table)), np.eye(3))
    assert drawn_at(expected.after, play(any_end(score, round_on_table)))
    # one prediction frame serves both rules: read top over bottom, he expects the same
    as_flipped = replace(expect_pic, opt=expect_pic.opt.replace("then=round", "then=flip"))
    assert as_flipped.opt != expect_pic.opt
    assert drawn_at(P.picture(as_flipped).after, play(any_end(score, top_over_bottom)))
    # and it is not what the ball did, beside it, either way
    for ball_pic in (round_ball, flip_ball):
        assert not drawn_at(expected.after, play(triples(P.picture(ball_pic).arrows)))
    assert not np.allclose(P.picture(round_ball).after.g, expected.after.g)
    # the rule needs a backwards reading to be wrong about
    assert P.faults(P.crabball, "up", "company", "play=glass+1,steel+1;then=page;expects=anyend")
    assert P.faults(P.crabball, "up", "company", "play=glass+1,steel+1;then=round;expects=anyend;marks=on")
    assert a_rules.startswith("**Crab:** Allow me.") and f_rules == "**Achilles:** But both rules worked. Every time!"

    # --- the page, played from where the score left the ball: the walk back --
    assert a_page.startswith("**Tortoise:** On arrows with two heads.")
    frames = P.picture(walk_back)
    assert dict(P.parse_opts(walk_back.opt, P.PLAYOPTS, "test")) == {"view": "page"}
    assert walk_back.args[2] == walk_out.args[2]                 # the same score, as drawn
    # it starts where the walk out ended
    start = P.picture(walk_out)[-1].ball.after
    assert walk_back.args[:2] == (P.name(start.g), P.name(start.s))
    assert drawn_at(frames[0].ball.start, play(score))
    assert len(frames) == 2
    assert [f.caption for f in frames] == ["the page: first the steel", "then the glass: home"]
    for k, f in enumerate(frames, 1):
        before, after = play(score + as_page[:k - 1]), play(score + as_page[:k])
        assert triples([f.step]) == [as_page[k - 1]]
        assert drawn_at(f.ball.start, before) and drawn_at(f.ball.after, after)
        marked(f.ball, before, as_page[k - 1][0], after)
    assert at_home(play(score + as_page)) and drawn_at(frames[-1].ball.after, np.eye(3))
    # bead for bead, the walk out backwards: each frame of the walk back is the
    # frame before it in the walk out, and each ring sits where it sat there
    out_ = P.picture(walk_out)
    assert drawn_at(frames[0].ball.after, play(score[:1]))
    assert np.allclose(frames[0].ball.marks.hold, out_[1].ball.marks.hold)
    assert np.allclose(frames[1].ball.marks.hold, out_[0].ball.marks.hold)
    assert "neither half will do alone." in a_back and a_back.startswith("**Achilles:** Steel first")
    assert f_back.startswith("*[On the fender the mittens")


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


def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def hadamard():
    p = Picture()
    a, b = p.inp(0), p.out(0)
    p.g.add_edge((a, b), zx.EdgeType.HADAMARD)
    return p.done()


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
    vs = [p.z(0, 2, Fraction(1, 4)), p.x(0, 3, Fraction(1, 3)), p.z(0, 4, Fraction(3, 4))]
    p.wire(n, *vs, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def sketchbook():
    wire = bare_wire()
    hbox = hadamard()
    sbox = chain(("z", Fraction(1, 2)))
    vbox = chain(("x", Fraction(1, 2)))

    # the mirror of a row of two: reversed and reflected, on a row whose two
    # pieces are NOT each other's mirrors
    hs = hbox + sbox                                    # H, then S
    assert same_picture(hs.adjoint(), sbox.adjoint() + hbox)
    assert not same_picture(hs.adjoint(), hbox + sbox.adjoint())

    # the chapter's test piece: the row S, then V
    u = sbox + vbox
    um = matrix_of(u)
    assert not np.allclose(um, um.T) and not np.allclose(um, um.conj())
    both = vbox.adjoint() + sbox.adjoint()              # mirrored AND reversed
    assert same_picture(both, u.adjoint())
    assert same_picture(u + both, wire)
    reversed_only = vbox + sbox                         # order reversed only
    mirrored_only = sbox.adjoint() + vbox.adjoint()     # pieces mirrored only
    assert not same_picture(u + reversed_only, wire)
    assert not same_picture(u + mirrored_only, wire)
    assert np.allclose(matrix_of(reversed_only), um.T)
    assert np.allclose(matrix_of(mirrored_only), um.conj())

    # a closed picture and its mirror image are conjugate numbers
    psi, plus = generic_state(), turned_state(0)
    closed = psi + u + plus.adjoint()                   # state psi, box U, effect +
    mirrored = plus + u.adjoint() + psi.adjoint()       # every piece mirrored, reversed
    v1 = matrix_of(closed)[0, 0]
    v2 = matrix_of(mirrored)[0, 0]
    assert not np.isclose(v1.imag, 0)                   # else the bar is invisible
    assert np.isclose(v2, np.conj(v1))
    assert not same_picture(closed, mirrored)           # different pictures
    # the cheaper reflections give the wrong number
    assert not np.isclose(matrix_of(plus + reversed_only + psi.adjoint())[0, 0], np.conj(v1))
    assert not np.isclose(matrix_of(plus + mirrored_only + psi.adjoint())[0, 0], np.conj(v1))


if __name__ == "__main__":
    verdict("Backwards", blackboard, sketchbook)
