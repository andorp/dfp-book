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

import re
from fractions import Fraction

import numpy as np
from dfp import Picture, ket, matrix_of, random_matrix, random_state, same_picture, verdict



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


def catch(R):
    """The reset catch: from wherever the ball stands, home.  Not a turn."""
    return np.eye(3)


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
    first = play([("steel", 1, cw)])
    second = play([("glass", 1, -cw), ("steel", 2, cw)])
    assert np.allclose(catch(first), catch(second))     # two places, one image

    # every strip in the dialogue stands in some challenge's section:
    # thirteen in all, and none in the Fourth, whose scores are only spoken
    counts = [len(strips(o)) for o in ("First", "Second", "Third", "Fourth", "Fifth")]
    assert counts == [4, 4, 2, 0, 3]
    assert DIALOG.read_text(encoding="utf-8").count("\\scorestrip") == sum(counts) == 13

    shown = strips("Fifth")
    (_, blank, blank_drawn, _), (_, gummed, as_drawn, _), (_, again, as_page, _) = shown

    # "Draw the catch." The strip stays blank, whichever way it lies, and a
    # blank strip is no turn at all: from the first place it leaves the ball
    # where it was, not home.  The catch is not on the strip.
    assert blank == blank_drawn == []
    assert all(view_score(blank, v) == [] for v in VIEWS)
    assert np.allclose(first @ play(arrows(blank)), first)
    assert not np.allclose(first @ play(arrows(blank)), catch(first))

    # the evening's strips, gummed end to end in the order they were drawn:
    # her one blue arrow, Achilles's strip with the heads rubbed out, brass,
    # glass, brass -- each read out of its own challenge, not copied here
    assert again == gummed
    assert as_drawn == gummed
    assert pieces(gummed) == [strips("First")[0][1],
                              strips("Second")[0][1],
                              strips("Third")[0][1]]
    assert arrows(gummed) == LONG_SCORE
    # turned like a page: the pieces in the reverse order, each one paged
    assert pieces(as_page) == [view_score(q, "page") for q in pieces(gummed)[::-1]]
    # which is what Achilles reads: brass, glass, brass; then steel, glass and
    # glass, one click each, every one against the clock
    assert arrows(as_page) == [("brass", 2, -cw), ("glass", 2, -cw), ("brass", 2, -cw),
                               ("steel", 1, -cw), ("glass", 1, -cw), ("glass", 1, -cw)]
    assert arrows(as_page) == page(LONG_SCORE)
    for s in (+1, -1):
        assert not at_home(play(arrows(as_drawn), s))
        assert at_home(play(arrows(as_drawn) + arrows(as_page), s))

    check_strip("Fifth", [first,                                  # one place
                          second,                                 # another place
                          catch(second),                          # the catch, from either: home
                          play(arrows(as_drawn)),                 # the evening's score
                          play(arrows(as_drawn) + arrows(as_page))])   # its page: home

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
