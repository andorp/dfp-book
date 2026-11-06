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

import re
from fractions import Fraction

import numpy as np
import pyzx as zx
from dfp import Picture, ket, matrix_of, random_matrix, random_state, same_picture, verdict


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
    score = [("glass", 1, cw), ("steel", 1, cw)]
    rot, flip = round_on_table(score), top_over_bottom(score)

    # the strips: Achilles's as drawn, round on the table, top over bottom --
    # all three above the balls -- and, after, the Tortoise's page.  Each
    # shows what is then played, from where the ball then stands.
    shown = strips("Second")
    assert len(shown) == 4
    assert [view for view, _, _, _ in shown] == ["drawn", "round", "flip", "page"]
    assert all(d == score for _, d, _, _ in shown)  # one strip, four ways of lying
    as_drawn, as_round, as_flip, as_page = [s for _, _, s, _ in shown]
    assert as_drawn == score                        # glass one, steel one, with the clock
    assert as_round == [("steel", 1, cw), ("glass", 1, cw)] == rot     # his first rule
    assert as_flip == [("glass", 1, -cw), ("steel", 1, -cw)] == flip   # his second
    assert as_page == [("steel", 1, -cw), ("glass", 1, -cw)] == page(score)
    for s in (+1, -1):
        assert not at_home(play(as_drawn + as_round, s))
        assert not at_home(play(as_drawn + as_flip, s))
        assert at_home(play(as_drawn + as_page, s))     # both at once
    assert not np.allclose(play(as_drawn + as_round), play(as_drawn + as_flip))

    # in the order of the scene: the Crab backs each failure out, last click
    # first and each the other way, before the next is tried -- so each rule
    # is in effect played from where the score left the ball
    after_rot = as_drawn + as_round
    after_flip = after_rot + page(as_round) + as_flip
    after_page = after_flip + page(as_flip) + as_page
    assert np.allclose(play(after_flip), play(as_drawn + as_flip))
    assert np.allclose(play(after_page), play(as_drawn + as_page))
    check_strip("Second", [play(as_drawn),        # as played
                           play(after_rot),       # round on the table
                           play(after_flip),      # top over bottom
                           play(after_page)])     # the page: home

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
