"""Challenge 3: The Sandwich
=============================
"One turn of the steel — and there they all are again."  -- Crab

Claim: HZH = X, exactly and scalar included; on the sphere H exchanges the x-
and z-axes and reverses y, so a half-turn about z, seen through that exchange,
is a half-turn about x.  The Turn in Three Steps is 1 in all three readings.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction

import numpy as np
import pyzx as zx
from dfp import Picture, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch03 as P


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
I2 = np.eye(2, dtype=complex)
PAULI = (X, Y, Z)
ARROW = np.array([3 / 5, 4 / 5], dtype=complex)      # blackboard only, as in Chapter 2


def point(psi):
    """The three numbers of Sec 3.5, each a closed picture with one box."""
    return np.array([np.vdot(psi, P @ psi) for P in PAULI])


def sphere(M):
    return np.array([[0.5 * np.trace(Pk @ M.conj().T @ Pj @ M).real
                      for Pk in PAULI] for Pj in PAULI])


def rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])


def rx(theta):
    return H @ rz(theta) @ H


def blackboard():
    rng = np.random.default_rng(33)

    # --- the headline, and its two companions ------------------------------
    assert np.allclose(H @ Z @ H, X)                    # exact, scalar included
    assert np.allclose(H @ X @ H, Z)
    assert np.allclose(H @ Y @ H, -Y)
    assert not np.allclose(H @ Z @ H, Z)                # it really is a different box
    assert np.allclose(H @ H, I2)

    # a sign on a whole box: different boxes, the same turn of the sphere
    assert np.allclose(X @ Z, -(Z @ X))
    assert not np.allclose(X @ Z, Z @ X)
    assert np.allclose(sphere(X @ Z), sphere(Z @ X))
    assert np.allclose(sphere(X @ Z), np.diag([-1.0, 1.0, -1.0]))    # the y half-turn
    for phase in (np.exp(1j * np.pi / 4), -1):
        assert np.allclose(sphere(phase * H), sphere(H))

    # --- what H does to the sphere -----------------------------------------
    RH, RZ, RX = sphere(H), sphere(Z), sphere(X)
    assert np.allclose(RH, np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]))   # x<->z, y->-y
    assert np.allclose(RH @ RZ @ RH, RX)                # the sandwich, on the sphere
    assert not np.allclose(RZ, RX)

    # --- Proposition 3.3, on states drawn at random ------------------------
    for _ in range(100):
        psi = random_state(rng)
        p = point(psi)
        assert np.allclose(p.imag, 0)                   # each number is real
        p = p.real
        assert np.isclose(p @ p, 1)                     # and the point is on the sphere
        theta = rng.uniform(0, 2 * np.pi)
        assert np.allclose(point(np.exp(1j * theta) * psi).real, p)      # no phase seen
        assert np.allclose(point(H @ psi).real, (p[2], -p[1], p[0]))     # Prop 3.6 (1)
        for angle in (np.pi / 3, -0.7):
            c, s = np.cos(angle), np.sin(angle)
            assert np.allclose(point(rz(angle) @ psi).real,
                               (p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]))
            assert np.allclose(point(rx(angle) @ psi).real,
                               (p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c))

    # a generic state is not real and not its own conjugate: the test is not vacuous
    psi = random_state(rng)
    assert not np.allclose(psi.imag, 0)
    assert not np.isclose(point(psi)[1].real, 0)        # y is off the real circle

    # --- the named points ---------------------------------------------------
    omega = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    for state, want in ((np.array([1, 0], dtype=complex), (0, 0, 1)),
                        (np.array([0, 1], dtype=complex), (0, 0, -1)),
                        (np.array([1, 1], dtype=complex) / np.sqrt(2), (1, 0, 0)),
                        (np.array([1, -1], dtype=complex) / np.sqrt(2), (-1, 0, 0)),
                        (omega, (0, 1, 0))):
        assert np.allclose(point(state).real, want)
    assert np.allclose(point(ARROW).real, (24 / 25, 0, -7 / 25))
    assert np.isclose(np.vdot(ARROW, X @ ARROW), 24 / 25)      # the number of Sec 3.1
    assert np.allclose(point(H @ ARROW).real, (-7 / 25, 0, 24 / 25))

    # --- the third reading needs a box that is neither real nor symmetric ---
    V = H @ S @ H
    U = V @ S
    assert not np.allclose(U, U.T) and not np.allclose(U, U.conj())
    assert np.allclose(U.conj().T @ U, I2)
    for A in (S, V, np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)):
        assert np.allclose(A, A.T)                      # S, HSH and T are symmetric ...
    assert np.allclose(U.conj().T @ U, I2)              # ... so only U tests the dagger
    # with the transpose in the place of the mirror the first factor is not 1
    assert np.isclose(np.vdot(ARROW, S.T @ S @ ARROW), -7 / 25)
    assert np.allclose(S.T @ S, Z)

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


def beads():
    cw = +1
    # the brass half-turn exchanges the glass and steel needles, bead for bead
    brass = play([("brass", 2, cw)])
    assert np.allclose(brass @ HOME["glass"], HOME["steel"])
    assert np.allclose(brass @ HOME["steel"], HOME["glass"])
    assert reading(brass, "glass") == "company" and reading(brass, "steel") == "up"

    # brass, glass, brass -- two clicks each -- and one half-turn of the steel
    score = [("brass", 2, cw), ("glass", 2, cw), ("brass", 2, cw)]
    steel = [("steel", 2, cw)]
    assert np.allclose(play(score), play(steel))
    assert reading(play(score), "glass") == "down"
    assert reading(play(score), "steel") == "company"
    # the brass bead itself ends down and towards the company
    assert np.allclose(play(score) @ HOME["brass"], (COMPANY - UP) / np.sqrt(2))
    assert not at_home(play(score))
    # Achilles's guess -- "the brass arrows cancel, what is left is the glass
    # half-turn" -- is exactly the other thing
    guess = play([("glass", 2, cw)])
    assert reading(guess, "glass") == "up" and reading(guess, "steel") == "away"
    assert not np.allclose(play(score), guess)

    # the score reads the same either side of the paper, and undoes itself
    # every click here is a half-turn, so the sense does not matter: the score
    # reads the same either side of the paper, and undoes itself
    assert np.allclose(play(score), play(page(score)))
    assert at_home(play(score + page(score)))
    for s in (+1, -1):
        assert np.allclose(play(score, s), play(steel, s))
        assert at_home(play(score + page(score), s))


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


def any_order(score):
    """Achilles's rule, "the order never mattered anyway": each needle's clicks
    gathered, in the order the needles first come, and added (a needle's two
    half-turns cancel)."""
    net = {}
    for n, c, w in score:
        net[n] = net.get(n, 0) + w * c
    out = []
    for n, q in net.items():
        out += {0: [], 1: [(n, 1, +1)], 2: [(n, 2, +1)], 3: [(n, 1, -1)]}[q % 4]
    return out


def half_way(before, needle, angle, bead):
    """Where a bead is half-way along a twirl, in the room's frame."""
    return ROOM @ before @ turn(HOME[needle], angle / 2) @ HOME[bead]


def side_named(paragraph, pattern):
    """The sides of the room a direction names, read out of it by a pattern."""
    m = re.search(pattern, paragraph)
    assert m and all(g in ("dresser", "stove") for g in m.groups()), pattern
    return m.groups()


def strip():
    P.check_lua()                                  # the Lua and .tex the transcription was proved on
    check_frame()
    check_turning(33)
    cw = +1
    score = [("brass", 2, cw), ("glass", 2, cw), ("brass", 2, cw)]

    calls = DIALOGUE.check_layout("Third", sizes=[2, 2, 1])       # three moments
    P.check_rows(DIALOGUE, "Third")
    drawn_pic, page_pic, expect_pic, steps, steel_pic = calls
    assert [c.macro for c in calls] == ["scorestrip", "scorestrip", "crabball", "crabplay",
                                        "crabball"]
    assert sum(P.frames(c) for c in (expect_pic, steps)) == 4      # four frames, one line
    paras = DIALOGUE.paragraphs("Third")
    a_strips, a_sandwich, a_steel = DIALOGUE.anchors("Third")
    f_strips, f_sandwich, f_steel = DIALOGUE.following("Third")

    # --- the strip, as drawn and turned like a page -----------------------
    s1, s2 = P.picture(drawn_pic), P.picture(page_pic)
    assert s1.view == "drawn" and s2.view == "page" and s2.drawn == s1.drawn
    as_drawn, as_page = triples(s1.shown), triples(s2.shown)
    assert as_drawn == score                        # brass, glass, brass, two clicks each
    # through the paper every arrow curls the other way ...
    assert as_page == [("brass", 2, -cw), ("glass", 2, -cw), ("brass", 2, -cw)]
    assert as_page == page(score) and as_page != as_drawn
    # ... but each is a half-turn, and "the same score, either side of the paper"
    for s in (+1, -1):
        assert np.allclose(play(as_page, s), play(as_drawn, s))
        assert at_home(play(as_drawn + as_page, s))   # the Crab plays the page: home

    # --- the prediction: Achilles's rule, drawn pale ------------------------
    expected = P.picture(expect_pic)
    assert expect_pic.caption == "as Achilles expects"
    assert expected.expects == "anyorder" and expected.marks is None
    assert expect_pic.opt.startswith("play=brass+2,glass+2,brass+2;")
    assert any_order(score) == [("glass", 2, cw)]   # the brass clicks add to a whole turn
    assert drawn_at(expected.start, np.eye(3))
    assert drawn_at(expected.after, play(any_order(score)))
    # which is what he said, in so many words
    said = [p for p in paras if "Brass twice over is no turn at all" in p]
    assert len(said) == 1
    glass_said, steel_said = re.search(r"Glass bead (\w+), steel bead (\w+)\.", said[0]).groups()
    assert (reading(play(any_order(score)), "glass"), reading(play(any_order(score)), "steel")) \
        == (glass_said, steel_said) == ("up", "away")
    # and it is not what the ball did, beside it
    assert not drawn_at(expected.after, play(score))
    assert reading(play(score), "glass") == "down" and reading(play(score), "steel") == "company"
    assert "it is exactly the other thing" in a_sandwich
    assert f_sandwich.startswith("**Tortoise:** *[She turns the strip like a page once more.]*")

    # --- the sandwich, one turn a frame ------------------------------------
    frames = P.picture(steps)
    assert steps.opt == "" and steps.args[2] == "brass+2,glass+2,brass+2"
    assert len(frames) == 3
    assert [f.caption for f in frames] == ["first the brass", "then the glass",
                                           "then the brass again"]
    assert all(words_only(f.caption) for f in frames)
    for k, f in enumerate(frames, 1):
        before, after = play(score[:k - 1]), play(score[:k])
        assert triples([f.step]) == [score[k - 1]]
        assert drawn_at(f.ball.start, before) and drawn_at(f.ball.after, after)
        marked(f.ball, before, score[k - 1][0], after)
    assert drawn_at(frames[-1].ball.after, play(score))
    # the brass half-turn swaps glass and steel, bead for bead (frame one)
    assert reading(play(score[:1]), "glass") == "company" and reading(play(score[:1]), "steel") == "up"
    # the pin: the brass bead, then the glass where it now lies, then the brass
    brass_para = [p for p in paras if "The brass bead is the pin now" in p]
    glass_para = [p for p in paras if "The glass needle, as it now lies" in p]
    assert len(brass_para) == len(glass_para) == 1
    brass_para, glass_para = brass_para[0], glass_para[0]
    assert np.allclose(frames[0].ball.marks.hold, P.vec("0,1,1"))         # brass at home
    assert np.allclose(frames[1].ball.marks.hold, P.vec("company"))       # the glass, level
    # the sides each half-turn swings past -- the place each moving bead
    # passes half-way, one click in -- are the sides the directions name
    glass_by, steel_by = side_named(
        brass_para, r"the glass bead round by the (\w+) and the steel round by the (\w+)\.")
    (steel_by_2,) = side_named(glass_para, r"the steel swings half-way round, by the (\w+)\.")
    assert "by the same sides" in glass_para
    sides = [{"glass": glass_by, "steel": steel_by}, {"steel": steel_by_2},
             {"glass": glass_by, "steel": steel_by}]
    for k, (f, named) in enumerate(zip(frames, sides), 1):
        needle, clicks, way = score[k - 1]
        assert clicks == 2                                                # a half-turn
        assert set(f.ball.marks.midway) == set(named) | ({"brass"} if needle != "brass" else set())
        for bead, side in named.items():
            mid = half_way(play(score[:k - 1]), needle, -way * clicks * np.pi / 2, bead)
            assert np.allclose(f.ball.marks.midway[bead], mid)
            assert mid @ P.vec(side) > 0.5                                # clearly that side
            other = "stove" if side == "dresser" else "dresser"
            assert mid @ P.vec(other) < -0.5
            # and it is the bead's place after the turn's first click
            assert np.allclose(mid, ROOM @ play(score[:k - 1] + [(needle, 1, way)]) @ HOME[bead])

    # --- one steel half-turn, from home: bead for bead the same -------------
    ball = P.picture(steel_pic)
    assert a_steel == "**Achilles:** Bead for bead the same!"
    assert f_steel == "**Tortoise:** And your numbers?"
    assert triples(ball.arrows) == [("steel", 2, cw)] and drawn_at(ball.start, np.eye(3))
    assert drawn_at(ball.after, play([("steel", 2, cw)])) and drawn_at(ball.after, play(score))
    assert ball.marks is None and ball.dump == "crabball: glass=down steel=company"
    # marks=off is what keeps the marks off this one-arrow picture
    assert P.crabball("up", "company", "play=steel+2").marks is not None


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


def hadamard():
    p = Picture()
    a, b = p.inp(0), p.out(0)
    p.g.add_edge((a, b), zx.EdgeType.HADAMARD)
    return p.done()


def basis_state(b):
    p = Picture()
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
    vs = [p.z(0, 2, Fraction(1, 4)), p.x(0, 3, Fraction(1, 3)), p.z(0, 4, Fraction(3, 4))]
    p.wire(n, *vs, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def value(g):
    return matrix_of(g)[0, 0]


def sketchbook():
    hbox, zbox, xbox = hadamard(), chain(("z", 1)), chain(("x", 1))
    assert np.allclose(matrix_of(hbox), H)              # exact, scalar included

    # --- the theorem, as a picture equation ---------------------------------
    assert same_picture(hbox + zbox + hbox, xbox)       # HZH = X
    assert same_picture(hbox + xbox + hbox, zbox)
    assert not same_picture(hbox + zbox + hbox, zbox)
    ybox = zbox + xbox                                  # the row Z then X: XZ
    ybox.scalar.add_phase(Fraction(1, 2))               # Y = iXZ: the phase is +pi/2
    assert np.allclose(matrix_of(ybox), Y)
    minus_y = ybox.copy()
    minus_y.scalar.add_phase(Fraction(1))
    assert same_picture(hbox + ybox + hbox, minus_y)    # HYH = -Y
    assert not same_picture(hbox + ybox + hbox, ybox)

    # --- The Turn in Three Steps, first reading: as drawn -------------------
    zero, one = basis_state(0), basis_state(1)
    plus, minus = turned_state(0), turned_state(1)
    line1 = zero + hbox + plus.adjoint()                # state 0, H, effect +
    line2 = plus + zbox + minus.adjoint()               # state +, Z, effect -
    line3 = minus + hbox + one.adjoint()                # state -, H, effect 1
    for line in (line1, line2, line3):
        assert np.isclose(value(line), 1)
    row = line1 + line2 + line3                         # closed pictures multiply
    assert np.isclose(value(row), 1)
    # each line really is a certainty and not an accident of the arithmetic:
    # the same wire with the other effect is 0
    assert np.isclose(value(zero + hbox + minus.adjoint()), 0)
    assert np.isclose(value(plus + zbox + plus.adjoint()), 0)

    # --- second reading: in the mirror --------------------------------------
    mirror = one + hbox + minus.adjoint()               # state 1, H, effect -
    mirror = mirror + (minus + zbox + plus.adjoint())   # state -, Z, effect +
    mirror = mirror + (plus + hbox + zero.adjoint())    # state +, H, effect 0
    assert same_picture(row.adjoint(), mirror)          # every piece mirrored, reversed
    assert np.isclose(value(mirror), 1)                 # the conjugate of 1
    assert same_picture(row, mirror)                    # both are the number 1

    # --- third reading: any three boxes that keep the promise ---------------
    # A = S, B = (HSH)S, C = T.  S, HSH and T are symmetric, so B is the only
    # one of the three that can test a dagger at all.
    sbox, vbox, tbox = chain(("z", Fraction(1, 2))), chain(("x", Fraction(1, 2))), chain(("z", Fraction(1, 4)))
    A, B, C = sbox, sbox + vbox, tbox
    bm = matrix_of(B)
    assert not np.allclose(bm, bm.T) and not np.allclose(bm, bm.conj())
    psi = generic_state()
    st1 = psi + A
    st2 = st1 + B
    st3 = st2 + C
    for state, box, after in ((psi, A, st1), (st1, B, st2), (st2, C, st3)):
        assert np.isclose(value(state + box + after.adjoint()), 1)
    assert np.isclose(value((psi + A + st1.adjoint())
                            + (st1 + B + st2.adjoint())
                            + (st2 + C + st3.adjoint())), 1)
    # with the cheaper reflection in the place of the mirror, the line is not 1
    wrong = vbox + sbox                                 # order reversed, pieces kept
    assert not np.isclose(value(st1 + B + wrong + st1.adjoint()), 1)
    assert np.isclose(value(st1 + B + B.adjoint() + st1.adjoint()), 1)

    # --- the closed pictures of Sec 3.5, and why they are real --------------
    xpic = psi + xbox + psi.adjoint()
    assert same_picture(xpic, xpic.adjoint())           # its own mirror image
    assert np.isclose(value(xpic).imag, 0)              # hence a real number
    zpic = psi + zbox + psi.adjoint()
    assert not np.isclose(value(zpic), value(xpic))     # different numbers
    spic = psi + sbox + psi.adjoint()                   # S is not its own mirror
    assert not same_picture(spic, spic.adjoint())
    assert not np.isclose(value(spic).imag, 0)
    assert np.isclose(value(spic.adjoint()), np.conj(value(spic)))

    # the theorem inside a closed picture, for a state with no name
    assert same_picture(psi + hbox + zbox + hbox + psi.adjoint(), xpic)
    assert same_picture(psi + hbox + xbox + hbox + psi.adjoint(), zpic)
    assert not np.isclose(value(xpic), 0) and not np.isclose(value(zpic), 0)
    assert not same_picture(xpic, zpic)


if __name__ == "__main__":
    verdict("Sandwich", blackboard, sketchbook)
