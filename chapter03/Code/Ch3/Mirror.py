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
    one = [("glass", 1, cw)]
    trap = [("glass", 2, cw), ("steel", 2, cw)]
    swapped = [("steel", 2, cw), ("glass", 2, cw)]

    # the strips: the Tortoise's, as drawn and turned like a page; Achilles's,
    # as drawn and turned round on the table.  Each shows what is then played.
    shown = strips("First")
    assert len(shown) == 4
    (_, her, her_drawn, _), (_, her_again, her_page, _), \
        (_, his, his_drawn, _), (_, his_again, his_round, _) = shown
    assert her_drawn == her == one                  # glass, one click, with the clock
    assert her_again == her                         # the same strip, turned over
    assert her_page == [("glass", 1, -cw)]          # seen through it: one click against
    assert her_page == page(one)
    for s in (+1, -1):                              # played from where it stands: home
        assert at_home(play(her_drawn + her_page, s))
    assert his_drawn == his == trap                 # glass two, then steel two
    assert his_again == his
    assert his_round == swapped                     # steel two, then glass two, as drawn
    assert his_round == round_on_table(trap)
    for s in (+1, -1):                              # home -- the trap springs
        assert at_home(play(his_drawn + his_round, s))

    check_strip("First", [np.eye(3),                # home (the keyed picture)
                          play(her_drawn),          # glass, one click with the clock
                          play(his_drawn),          # glass, then steel, two clicks each
                          play(his_round)],         # steel, then glass
                keyed_first=True)

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
