"""Challenge 4: Three Turns Make Any Turn
==========================================
"Glass, steel, glass. It is always enough."  -- Tortoise

Claim: S (HSH) S = e^{i pi/4} H, scalar and all; and every one-wire box that
keeps the promise is a phase times a Z-turn, an X-turn and a Z-turn.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction

import numpy as np
import pyzx as zx
from dfp import Picture, matrix_of, random_matrix, same_picture, verdict



X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
I2 = np.eye(2, dtype=complex)
PAULI = (X, Y, Z)


def rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])


def rx(theta):
    return H @ rz(theta) @ H


def sphere(M):
    return np.array([[0.5 * np.trace(Pk @ M.conj().T @ Pj @ M).real
                      for Pk in PAULI] for Pj in PAULI])


def random_unitary(rng):
    """A box that keeps the promise, drawn at random (NumPy only)."""
    q, r = np.linalg.qr(random_matrix(rng))
    return q @ np.diag(np.diag(r) / np.abs(np.diag(r)))


def euler(U):
    """The four numbers of Theorem 3.7, read off the four entries of U.

    U = e^{i delta} Rz(alpha) Rx(beta) Rz(gamma).  Comparing entries:
    the two angle differences below are alpha+gamma and alpha-gamma, each
    known only modulo 2 pi, so alpha and gamma are fixed only up to a common
    half-turn -- which is why beta is then read off the ratio U01/U00 with
    its sign, and the phase off one entry at the end.
    """
    plus = np.angle(U[1, 1]) - np.angle(U[0, 0])        # alpha + gamma
    minus = np.angle(U[1, 0]) - np.angle(U[0, 1])       # alpha - gamma
    alpha, gamma = (plus + minus) / 2, (plus - minus) / 2
    t = 1j * (U[0, 1] / U[0, 0]) * np.exp(-1j * gamma)  # = tan(beta/2)
    assert abs(t.imag) < 1e-9                           # real: U keeps the promise
    beta = 2 * np.arctan(t.real)
    W = rz(alpha) @ rx(beta) @ rz(gamma)
    delta = np.angle(U[0, 0] / W[0, 0])
    return alpha, beta, gamma, delta


def blackboard():
    rng = np.random.default_rng(34)
    V = H @ S @ H

    # --- the instance, with its scalar ------------------------------------
    assert np.allclose(S @ V @ S, np.exp(1j * np.pi / 4) * H)
    assert not np.allclose(S @ V @ S, H)                # the phase is part of it
    # every arrow mirrored: the same turn of the sphere, the opposite phase
    assert np.allclose(H.conj().T, H)
    assert np.allclose(S.conj().T @ V.conj().T @ S.conj().T,
                       np.exp(-1j * np.pi / 4) * H)
    assert np.allclose(sphere(S @ V @ S), sphere(H))
    assert np.allclose(sphere(S @ V @ S), sphere(S.conj().T @ V.conj().T @ S.conj().T))
    assert np.allclose(V, rx(np.pi / 2) * np.exp(1j * np.pi / 4))   # HSH is an x-turn

    # --- three turns are always enough -------------------------------------
    worst = 0.0
    for _ in range(100):
        U = random_unitary(rng)
        assert np.allclose(U.conj().T @ U, I2)
        alpha, beta, gamma, delta = euler(U)
        rebuilt = np.exp(1j * delta) * rz(alpha) @ rx(beta) @ rz(gamma)
        worst = max(worst, np.abs(rebuilt - U).max())
    assert worst < 1e-10

    # and the boxes used are generic: not real, not symmetric
    U = random_unitary(rng)
    assert not np.allclose(U, U.conj()) and not np.allclose(U, U.T)
    alpha, beta, gamma, delta = euler(U)
    assert np.allclose(np.exp(1j * delta) * rz(alpha) @ rx(beta) @ rz(gamma), U)
    assert not np.isclose(beta % (2 * np.pi), 0)        # the middle turn is needed

    # --- one needle is never enough: a Z-turn does not move the pole -------
    pole = np.array([0.0, 0.0, 1.0])
    for _ in range(20):
        theta, phi = rng.uniform(0, 2 * np.pi, 2)
        assert np.allclose(sphere(rz(theta)) @ pole, pole)
        assert np.allclose(rz(theta) @ rz(phi), rz(theta + phi))     # still one turn
        assert not np.allclose(sphere(rx(theta)) @ pole, pole) or np.isclose(
            np.sin(theta), 0)
    assert not np.allclose(sphere(H) @ pole, pole)      # H does move it

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
    rng = np.random.default_rng(43)
    cw = +1
    score = [("glass", 1, cw), ("steel", 1, cw), ("glass", 1, cw)]

    # glass, steel, glass -- one click each -- is the brass half-turn
    R = play(score)
    assert reading(R, "glass") == "company" and reading(R, "steel") == "up"
    assert np.allclose(R, play([("brass", 2, cw)]))
    assert not at_home(R)

    # the score reads the same from either end, so its page is the same three
    # needles with every arrow reversed
    assert [n for n, _, _ in score] == [n for n, _, _ in reversed(score)]
    assert page(score) == [("glass", 1, -cw), ("steel", 1, -cw), ("glass", 1, -cw)]
    assert at_home(play(score + page(score)))           # it undoes the score ...
    assert np.allclose(play(page(score)), R)            # ... and from home, same place
    assert at_home(play(score + [("brass", 2, cw)]))    # the brass takes it back
    for s in (+1, -1):
        assert at_home(play(score + page(score), s))
        assert np.allclose(play(page(score), s), play(score, s))

    # --- the tumbled ball, brought home in three ---------------------------
    # glass, then steel, then glass: the same three turns as the theorem, with
    # the glass needle where the Z-turn is and the steel where the X-turn is.
    for _ in range(100):
        M = tumble(rng)
        a, b, c = home_in_three(M)
        assert at_home(M @ turn(HOME["glass"], a)
                       @ turn(HOME["steel"], b)
                       @ turn(HOME["glass"], c))
        # the glass bead does not stir while the glass needle is held
        assert np.allclose(M @ turn(HOME["glass"], a) @ HOME["glass"],
                           M @ HOME["glass"])
        # and it is the Tortoise's recipe: the first glass turn lays the
        # steel level, the steel turn then stands the glass on top
        assert np.isclose(M @ turn(HOME["glass"], a) @ HOME["steel"] @ UP, 0)
        assert np.allclose(M @ turn(HOME["glass"], a) @ turn(HOME["steel"], b)
                           @ HOME["glass"], UP)
    # one needle alone leaves the ball nowhere near home
    M = tumble(rng)
    for angle in rng.uniform(0, 2 * np.pi, 10):
        assert not at_home(M @ turn(HOME["glass"], angle))


def tumble(rng):
    """The Crab, eyes shut: a rotation with no name."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q if np.linalg.det(q) > 0 else q @ np.diag([1.0, 1.0, -1.0])


def home_in_three(M):
    """Angles for glass, steel, glass that carry the position M home."""
    N = M.T                                  # what the three turns must add up to
    b = np.arccos(np.clip(N[2, 2], -1, 1))   # how far the steel must swing
    a = np.arctan2(N[1, 2], N[0, 2])
    c = np.arctan2(N[2, 1], -N[2, 0])
    return a, b, c


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
# The tumble the dialogue pictures.  The Crab's eyes were shut, so this one
# position is given, not computed: the glass bead along 12,-15,16 and the
# steel bead along -8,16,21, in the room's frame (x towards the dresser,
# y up, z towards the company).  Everything after it is computed.
# --------------------------------------------------------------------------
TUMBLED_GLASS = np.array([12.0, -15.0, 16.0]) / 25
TUMBLED_STEEL = np.array([-8.0, 16.0, 21.0]) / np.sqrt(761)


def swing(axis, start, stop):
    """The angle of the twirl about `axis` that carries `start` to `stop`,
    both at right angles to it.  Positive is against the clock as seen from
    the bead at the end of `axis`."""
    assert np.isclose(axis @ start, 0) and np.isclose(axis @ stop, 0)
    return np.arctan2(axis @ np.cross(start, stop), start @ stop)


def twirl(R, needle, angle):
    """Twirl the ball about a needle AS IT LIES.  The turn about the needle's
    present direction, made after R, is the turn about its home direction
    made before R: the same position either way, as the model says."""
    after = turn(R @ HOME[needle], angle) @ R
    assert np.allclose(after, R @ turn(HOME[needle], angle))
    return after


def pictured_tumble():
    """Glass until the steel lies level; steel until the glass is on top;
    glass until the steel is at the company.  The four positions."""
    g, s = ROOM.T @ TUMBLED_GLASS, ROOM.T @ TUMBLED_STEEL      # into our frame
    assert np.isclose(g @ s, 0)
    R0 = np.column_stack([np.cross(s, g), s, g])   # stove, company, up at home go here
    assert np.allclose(R0.T @ R0, np.eye(3)) and np.isclose(np.linalg.det(R0), 1)
    assert np.allclose(R0 @ HOME["glass"], g) and np.allclose(R0 @ HOME["steel"], s)

    # Achilles's reading: "The glass low, leaning towards us and the dresser.
    # The steel high, leaning towards us and a little to the stove.  Nowhere
    # with a name."
    gx, gy, gz = TUMBLED_GLASS
    sx, sy, sz = TUMBLED_STEEL
    assert gy < 0 and gx > 0 and gz > 0                 # low; us; dresser
    assert sy > 0 and sy > abs(sx)                      # high, clearly
    assert sz > 0 and sx < 0                            # us; a little to the stove
    for bead in (g, s):
        assert not any(np.allclose(bead, u) for u in PLACES.values())

    # Why the glass first: a steel twirl swings the glass round the circle at
    # right angles to the steel, and that circle passes the top only when the
    # steel lies level.  Tumbled, it does not lie level.
    assert not np.isclose(s @ UP, 0)

    # 1. glass.  Two level places are at right angles to the glass; the twirl
    #    stops at the nearer.
    level = [v / np.linalg.norm(v) for v in (np.cross(g, UP), np.cross(UP, g))]
    a = min((swing(g, s, v) for v in level), key=abs)
    R1 = twirl(R0, "glass", a)
    s1 = R1 @ HOME["steel"]
    assert np.allclose(R1 @ HOME["glass"], g)           # the glass bead does not stir
    assert np.isclose(s1 @ UP, 0)                       # the steel lies level

    # 2. steel, as it now lies, until the glass is on top
    b = swing(s1, g, UP)
    R2 = twirl(R1, "steel", b)
    assert np.allclose(R2 @ HOME["steel"], s1)          # the steel bead does not stir
    assert reading(R2, "glass") == "up"
    # the glass on top: the ball is now wrong only by a spin about the glass
    c = swing(UP, s1, COMPANY)
    assert np.allclose(R2, turn(UP, -c))

    # 3. glass again, until the steel is at the company
    R3 = twirl(R2, "glass", c)
    assert at_home(R3)

    # the three twirls, solved; none is a whole number of clicks
    assert np.allclose([np.cos(a), np.sin(a)], np.array([19, 20]) / np.sqrt(761))
    assert np.allclose([np.cos(b), np.sin(b)], [-3 / 5, 4 / 5])
    assert np.allclose([np.cos(c), np.sin(c)], [3 / 5, 4 / 5])
    for angle in (a, b, c):
        assert not np.isclose(np.sin(2 * angle), 0)
    # and one needle alone would not have done: no glass twirl brings it home
    for angle in np.linspace(0, 2 * np.pi, 73):
        assert not at_home(R0 @ turn(HOME["glass"], angle))
    return [R0, R1, R2, R3]


def strip():
    # the Tortoise's strip does not appear here: the Fourth's scores are
    # spoken, never drawn, so its section holds no \scorestrip at all
    assert section("Fourth").count("\\scorestrip") == 0
    check_strip("Fourth", pictured_tumble())      # tumbled; glass; steel; glass: home

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


def euler_picture(alpha, beta, gamma, delta):
    """The row of three turns as a picture, right down to its scalar.

    A z node of phase a is diag(1, e^{i pi a}) = e^{i pi a / 2} Rz(pi a), and
    likewise for x, so the three nodes together carry a phase of their own
    which the picture's scalar must pay back.  The angles are drawn at random
    and are not exact fractions of pi, so the phases are the nearest fraction
    with a denominator under a billion and the comparison is np.allclose.
    """
    def frac(x):
        return Fraction(x / np.pi).limit_denominator(10 ** 9)

    g = chain(("z", frac(gamma)), ("x", frac(beta)), ("z", frac(alpha)))
    g.scalar.add_phase(frac(delta - (alpha + beta + gamma) / 2))
    return g


def sketchbook():
    rng = np.random.default_rng(344)
    hbox = hadamard()
    sbox = chain(("z", Fraction(1, 2)))
    vbox = chain(("x", Fraction(1, 2)))
    assert np.allclose(matrix_of(hbox), H)
    assert np.allclose(matrix_of(sbox), S)
    assert np.allclose(matrix_of(vbox), H @ S @ H)      # x(1/2) is HSH, exactly
    assert not np.allclose(matrix_of(vbox), (I2 + 1j * X) / np.sqrt(2))

    # --- glass, steel, glass: S, HSH, S -----------------------------------
    row = chain(("z", Fraction(1, 2)), ("x", Fraction(1, 2)), ("z", Fraction(1, 2)))
    assert same_picture(row, sbox + vbox + sbox)
    assert np.allclose(matrix_of(row), np.exp(1j * np.pi / 4) * H)
    assert not same_picture(row, hbox)                  # the scalar is part of it
    row.scalar.add_phase(-Fraction(1, 4))
    assert same_picture(row, hbox)                      # now, on the nose

    # every arrow mirrored: the same three needles, the other way round
    mirror = chain(("z", -Fraction(1, 2)), ("x", -Fraction(1, 2)), ("z", -Fraction(1, 2)))
    assert same_picture(mirror, (sbox + vbox + sbox).adjoint())
    assert np.allclose(matrix_of(mirror), np.exp(-1j * np.pi / 4) * H)
    mirror.scalar.add_phase(Fraction(1, 4))
    assert same_picture(mirror, hbox)
    assert same_picture(row + mirror, chain(("z", 0)))  # H met its own reflection

    # --- any box at all, in three turns -------------------------------------
    for _ in range(20):
        U = random_unitary(rng)
        alpha, beta, gamma, delta = euler(U)
        g = euler_picture(alpha, beta, gamma, delta)
        assert np.allclose(matrix_of(g), U, atol=1e-7)
    # the scalar is not decoration: without it the picture is a different box
    U = random_unitary(rng)
    assert not np.allclose(U, U.conj()) and not np.allclose(U, U.T)
    alpha, beta, gamma, delta = euler(U)
    assert np.allclose(matrix_of(euler_picture(alpha, beta, gamma, delta)), U, atol=1e-7)
    assert not np.allclose(matrix_of(euler_picture(alpha, beta, gamma, 0)), U, atol=1e-7)
    # and the middle turn is not decoration either
    assert not np.allclose(matrix_of(euler_picture(alpha, 0, gamma, delta)), U, atol=1e-7)

    # one needle alone: a row of z nodes is always a diagonal box
    only_z = chain(("z", Fraction(1, 3)), ("z", Fraction(3, 5)))
    m = matrix_of(only_z)
    assert np.isclose(m[0, 1], 0) and np.isclose(m[1, 0], 0)
    assert same_picture(only_z, chain(("z", Fraction(1, 3) + Fraction(3, 5))))
    assert not same_picture(only_z, hbox)


if __name__ == "__main__":
    verdict("ThreeTurns", blackboard, sketchbook)
