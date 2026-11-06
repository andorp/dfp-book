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
# The pictures.  The dialogue draws the ball with \crabball{glass}{steel}{...}
# (pdf/props/ch03.tex), in the room's frame: x towards the dresser, y up, z towards
# the company.  Ours above is x towards the stove, y towards the company, z up;
# ROOM carries a direction of ours into the room's.  The pictures are read out
# of this challenge's section of dialog.md, never copied, so an edit to the
# dialogue that breaks a picture breaks here.
# --------------------------------------------------------------------------
DIALOG = pathlib.Path(__file__).resolve().parents[2] / "dialog.md"
ROOM = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
ROOM_PLACES = {"up": (0, 1, 0), "down": (0, -1, 0), "company": (0, 0, 1),
               "away": (0, 0, -1), "dresser": (1, 0, 0), "stove": (-1, 0, 0)}


def room_direction(arg):
    """A picture argument read as pdf/props/ch03.lua reads it: a name, or a
    triple x,y,z scaled to length one."""
    arg = re.sub(r"\s", "", arg)
    if arg in ROOM_PLACES:
        return np.array(ROOM_PLACES[arg], dtype=float)
    v = np.array([float(t) for t in arg.split(",")])
    assert v.shape == (3,), arg
    return v / np.linalg.norm(v)


def section(ordinal):
    """The dialogue's '## <ordinal> Challenge' section, up to the next heading."""
    text = DIALOG.read_text(encoding="utf-8")
    heads = list(re.finditer(r"^## " + ordinal + r" Challenge\b.*$", text, re.M))
    assert len(heads) == 1, ordinal
    rest = text[heads[0].end():]
    after = re.search(r"^## ", rest, re.M)
    return rest[:after.start()] if after else rest


def drawn(ordinal):
    """Every picture of the ball in the dialogue's '## <ordinal> Challenge'
    section, in order: (keyed, glass, steel, caption)."""
    macro = r"\\crabball(\*?)\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}"
    return [(star == "*", room_direction(g), room_direction(s), caption)
            for star, g, s, caption in re.findall(macro, section(ordinal))]


def check_frame():
    # the change of frame is a turn, not a reflection -- else every "with the
    # clock" would silently become "against"
    assert np.allclose(ROOM.T @ ROOM, np.eye(3)) and np.isclose(np.linalg.det(ROOM), 1)
    for name, u in PLACES.items():                      # all six names agree
        assert np.allclose(ROOM @ u, room_direction(name)), name
    assert np.allclose(ROOM @ HOME["glass"], room_direction("up"))       # home
    assert np.allclose(ROOM @ HOME["steel"], room_direction("company"))


def check_strip(ordinal, moments, keyed_first=False):
    """The pictures of one challenge against the positions the file computes
    for those moments, one position per picture, in order."""
    check_frame()
    pictures = drawn(ordinal)
    assert len(pictures) == len(moments), (ordinal, len(pictures))
    for k, ((keyed, glass, steel, caption), R) in enumerate(zip(pictures, moments)):
        assert keyed == (keyed_first and k == 0), caption
        assert np.isclose(glass @ steel, 0), caption    # at right angles
        assert np.allclose(glass, ROOM @ R @ HOME["glass"]), caption
        assert np.allclose(steel, ROOM @ R @ HOME["steel"]), caption
        # the macro draws the brass needle midway; so does the ball
        assert np.allclose((glass + steel) / np.sqrt(2), ROOM @ R @ HOME["brass"]), caption


# --------------------------------------------------------------------------
# The Tortoise's strip.  The dialogue draws it with
# \scorestrip[view]{score}{caption} (pdf/props/ch03.tex).  The score is the strip
# AS DRAWN, left to right: "glass+1" is glass, one click, with the clock;
# "steel-2" steel, two clicks, against; a bare "|" a gummed join; nothing at
# all, a blank strip.  The view says how the strip now lies -- drawn, page
# (order reversed, every arrow the other way round), round (order reversed,
# arrows as drawn), flip (order kept, every arrow the other way round) -- and
# pdf/props/ch03.lua draws what is then seen.  parse_score and view_score are a
# line-for-line transcription of ch03.lua, so what is asserted here is what
# the printed page shows.  An arrow is (needle, clicks, +1 with the clock /
# -1 against), the very triple play() takes.
# --------------------------------------------------------------------------
VIEWS = ("drawn", "page", "round", "flip")
SEAM = "|"


def parse_score(spec):
    items = []
    for tok in spec.split(","):
        tok = re.sub(r"\s", "", tok)
        if tok == SEAM:
            items.append(SEAM)
        elif tok != "":
            m = re.fullmatch(r"([A-Za-z]+)([+-])([12])", tok)
            assert m and m.group(1) in ("glass", "steel", "brass"), tok
            items.append((m.group(1), int(m.group(3)), +1 if m.group(2) == "+" else -1))
    return items


def view_score(items, view):
    assert view in VIEWS, view
    reverse = view in ("page", "round")
    mirror = view in ("page", "flip")
    seq = items[::-1] if reverse else list(items)
    return [it if it == SEAM else (it[0], it[1], -it[2] if mirror else it[2])
            for it in seq]


def arrows(items):
    """What is played: the arrows, left to right; a gummed join is not a turn."""
    return [it for it in items if it != SEAM]


def pieces(items):
    """The strips a gummed strip was made from, left to right."""
    out = [[]]
    for it in items:
        if it == SEAM:
            out.append([])
        else:
            out[-1].append(it)
    return out


def strips(ordinal):
    """Every strip in the dialogue's '## <ordinal> Challenge' section, in
    order: (view, the strip as drawn, the strip as shown, caption)."""
    text = section(ordinal)
    macro = r"\\scorestrip(?:\[([^\]]*)\])?\{([^}]*)\}\{([^}]*)\}"
    found = [(view or "drawn", parse_score(spec), view_score(parse_score(spec), view or "drawn"),
              caption) for view, spec, caption in re.findall(macro, text)]
    assert text.count("\\scorestrip") == len(found), ordinal   # none malformed, none skipped
    return found


def check_views():
    """What the four views mean as turns, on generic strips, not only the
    dialogue's own."""
    rng = np.random.default_rng(3)
    needles = ("glass", "steel", "brass")
    # ch03.lua curls a two-click arrow the other way in the page and flip
    # views; a half-turn either way round is the same turn, so nothing is lost
    for n in needles:
        assert np.allclose(play([(n, 2, +1)]), play([(n, 2, -1)]))
        assert not np.allclose(play([(n, 1, +1)]), play([(n, 1, -1)]))

    def random_spec(clicks, names):
        toks = []
        for _ in range(int(rng.integers(2, 8))):
            if toks and toks[-1] != SEAM and rng.random() < 0.2:
                toks.append(SEAM)
            toks.append("%s%s%d" % (rng.choice(names), rng.choice(["+", "-"]), rng.choice(clicks)))
        return ",".join(toks)

    # any strip at all: the page read after the strip brings the ball home,
    # read either way round throughout; round and flip do not in general
    fails = {"round": 0, "flip": 0}
    for _ in range(200):
        items = parse_score(random_spec((1, 2), needles))
        drawn_ = arrows(items)
        assert pieces(view_score(items, "page")) == \
            [view_score(p, "page") for p in pieces(items)[::-1]]      # seams travel too
        for s in (+1, -1):
            assert at_home(play(drawn_ + arrows(view_score(items, "page")), s))
            assert np.allclose(play(arrows(view_score(items, "page")), s), play(drawn_, s).T)
        for v in ("round", "flip"):
            fails[v] += not at_home(play(drawn_ + arrows(view_score(items, v))))
    assert fails["round"] > 100 and fails["flip"] > 100      # astray more often than not
    generic = parse_score("glass+1,steel-2,|,brass+1,glass-1")
    for v in ("round", "flip"):
        assert not at_home(play(arrows(generic) + arrows(view_score(generic, v))))
    # one arrow of one click: flip is the page, round is not
    one = parse_score("glass+1")
    assert view_score(one, "flip") == view_score(one, "page")
    assert not at_home(play(one + view_score(one, "round")))

    # half-turns only -- the trap of the First Challenge.  Round always works
    # (every half-turn is its own undoing) and plays exactly as the page does.
    for _ in range(100):
        items = arrows(parse_score(random_spec((2,), needles)))
        assert at_home(play(items + view_score(items, "round")))
        assert np.allclose(play(view_score(items, "round")), play(view_score(items, "page")))
    # flip works too while the needles are glass and steel, whose half-turns
    # do not mind their order ...
    for _ in range(100):
        items = arrows(parse_score(random_spec((2,), ("glass", "steel"))))
        assert at_home(play(items + view_score(items, "flip")))
    # ... but not once the brass needle joins them: order matters again
    bg = parse_score("brass+2,glass+2")
    assert not at_home(play(bg + view_score(bg, "flip")))
    assert at_home(play(bg + view_score(bg, "round")))


def strip():
    check_views()
    cw = +1
    score = [("brass", 2, cw), ("glass", 2, cw), ("brass", 2, cw)]

    # the strip, as drawn and turned like a page
    shown = strips("Third")
    assert len(shown) == 2
    (_, drawn_, as_drawn, _), (_, again, as_page, _) = shown
    assert as_drawn == drawn_ == score              # brass, glass, brass, two clicks each
    assert again == drawn_
    # through the paper every arrow curls the other way ...
    assert as_page == [("brass", 2, -cw), ("glass", 2, -cw), ("brass", 2, -cw)]
    assert as_page == page(score) and as_page != as_drawn
    # ... but each is a half-turn, and "the same score, either side of the paper"
    for s in (+1, -1):
        assert np.allclose(play(as_page, s), play(as_drawn, s))
        # the Crab plays the page from where the score left the ball: home
        assert at_home(play(as_drawn + as_page, s))

    check_strip("Third", [play(as_drawn[:1]),             # after one brass turn
                          play(as_drawn),                 # brass, glass, brass
                          play([("steel", 2, cw)])])      # one steel turn, from home

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
