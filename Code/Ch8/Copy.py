"""Challenge III: Each Copies the Other's Points (copy + pi-commutation)
=======================================================================
"...Oh."  -- Achilles

Claim: a phase-free spider of one colour COPIES the basis points of the other
colour -- up to sqrt2:  Z(1,2,0)|r> = (1/sqrt2)|r>(x)|r> for red points |r>.
And a half-turn (pi) pushed through the other colour reverses that colour's turn
-- up to a scalar, exactly only at phase 0:  X(pi)Z(a) = Z(-a)X(pi).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import string
from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import ket, kron, matrix_of, same_picture, same_picture_up_to_scalar, verdict
from dfp_props import Dialogue, words_only
import props_ch08 as P


# --- spiders as matrices: the chapter's own maths, pinned in every file ---
KET_PLUS = (ket(0) + ket(1)) / np.sqrt(2)
KET_MINUS = (ket(0) - ket(1)) / np.sqrt(2)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def zspider(m, k, alpha):
    """Z spider (green): |0..0><0..0| + e^{i alpha}|1..1><1..1|, m legs in / k out."""
    M = np.zeros((2 ** k, 2 ** m), dtype=complex)
    M[0, 0] = 1.0
    M[-1, -1] = np.exp(1j * alpha)
    return M


def _pow(v, n):
    out = np.array([1.0 + 0j])
    for _ in range(n):
        out = np.kron(out, v)
    return out


def xspider(m, k, beta):
    """X spider (red), built INTRINSICALLY from normalized |+>,|-> --
    NOT as H^(x)k Z H^(x)m, so colour change stays a theorem, not a definition."""
    pk, pm = _pow(KET_PLUS, k), _pow(KET_PLUS, m)
    mk, mm = _pow(KET_MINUS, k), _pow(KET_MINUS, m)
    return np.outer(pk, pm.conj()) + np.exp(1j * beta) * np.outer(mk, mm.conj())


def pin_spiders():
    assert np.allclose(zspider(1, 1, np.pi), np.diag([1, -1]))
    assert np.allclose(xspider(1, 1, np.pi), np.array([[0, 1], [1, 0]]))
    assert np.allclose(H @ H, np.eye(2))                               # H^2 = I
    assert np.allclose(KET_PLUS + KET_MINUS, np.sqrt(2) * ket(0))


def _connected(g, start, target):
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for w in g.neighbors(u):
            if w not in seen:
                seen.add(w); stack.append(w)
    return target in seen


def blackboard():
    pin_spiders()

    # a green copier duplicates its OWN basis points exactly -- one in, two out
    assert np.allclose(zspider(1, 2, 0) @ ket(0), ket(0, 0))
    assert np.allclose(zspider(1, 2, 0) @ ket(1), ket(1, 1))

    # the red one-leg basis points, each carrying the cup's sqrt2 (unnormalized)
    r0 = xspider(0, 1, 0).ravel()
    r1 = xspider(0, 1, np.pi).ravel()
    assert np.allclose(r0, np.sqrt(2) * ket(0)) and np.allclose(r1, np.sqrt(2) * ket(1))

    # green copies a red point -- UP TO THE SCALAR sqrt2:  Z(1,2,0)|r> = (1/sqrt2)|r>(x)|r>
    assert np.allclose(zspider(1, 2, 0) @ r0, (1 / np.sqrt(2)) * kron(r0, r0).ravel())
    assert np.allclose(zspider(1, 2, 0) @ r1, (1 / np.sqrt(2)) * kron(r1, r1).ravel())
    # symmetric: a red copier copies a green point, likewise up to sqrt2
    g0 = zspider(0, 1, 0).ravel()
    assert np.allclose(xspider(1, 2, 0) @ g0, (1 / np.sqrt(2)) * kron(g0, g0).ravel())

    # BUT only BASIS points copy: a red point with a phase is not copied, even up to scalar
    r_phase = xspider(0, 1, 0.9).ravel()
    lhs = zspider(1, 2, 0) @ r_phase
    assert not _proportional(lhs, kron(r_phase, r_phase).ravel())

    # pi-commutation: a red half-turn through a green dot flips the green turn --
    # exact ONLY at a=0, up to a scalar otherwise (so we test a != 0, not just 0)
    assert np.allclose(xspider(1, 1, np.pi) @ zspider(1, 1, 0),
                       zspider(1, 1, 0) @ xspider(1, 1, np.pi))              # a=0: exact
    for a in [0.7, 1.3]:
        lhs = xspider(1, 1, np.pi) @ zspider(1, 1, a)
        rhs = zspider(1, 1, -a) @ xspider(1, 1, np.pi)
        assert _proportional(lhs, rhs)                                      # up to scalar
        assert not np.allclose(lhs, rhs)                                    # but NOT exact at a != 0

    pictures()


def _proportional(A, B):
    A, B = np.asarray(A).ravel(), np.asarray(B).ravel()
    k = np.argmax(np.abs(A))
    if np.isclose(A[k], 0):
        return bool(np.allclose(A, 0) and np.allclose(B, 0))
    return bool(np.allclose(A / A[k], B / B[k]))


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the Crab's cords and curtain-rings
#   \ringcords[view]{tangle}{caption}           (pdf/props/ch08.tex)
# and the Crab's press, \koekpan[view]{caption}.  The Lua COMPUTES the views:
# slid, flipped, pushed, pressed, and the ring come clear of the flip-slot.
# Round 4 ("Show the process") adds the marks (marks=on: a moved part's ghost,
# its arc, a pin on what is held), the prediction frames (expects=: Achilles's
# refuted rule, drawn paler) and \ringsteps[opts]{tangle}{steps}{captions}, a
# strip whose every frame starts from the state the one before left.
# tools/props_ch08.py transcribes the Lua (proved against texlua; P.check_lua
# holds the Lua and the .tex to the text it was proved on); the calls are read
# out of this challenge's section of dialog.md and dialog.raw.md, never copied.
#
# The state model (identity card, section 3): a green ring is a Z spider, a red
# ring an X spider; a ring's turn t (twenty-fourths, clockwise from the top as
# the company sees it) is the phase 2 pi t / 24; cords in on the left are
# inputs, out on the right outputs, the upper line first.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
QUOTE = "...Oh."


def phase(t):
    """A turn in twenty-fourths as a phase in radians."""
    return 2 * np.pi * t / P.WHOLE


def tangle_map(T):
    """The map a tangle of rings and cords denotes, contracted in NumPy: each
    ring a spider of its colour with one leg per cord segment at it; ends on
    the left are inputs, on the right outputs, the upper line first (top wire
    first).  Leg order and in/out do not matter: these Z and X spiders are
    symmetric in all their legs (|0>,|1>,|+>,|-> are real)."""
    letters = iter(string.ascii_letters)
    legs = {n: [] for n in range(1, len(T.rings) + 1)}
    ins, outs, ops, subs = [], [], [], []
    for a, b in P.segments(T):
        if isinstance(a, P.End) and isinstance(b, P.End):     # a bare cord, hook to hook
            la, lb = next(letters), next(letters)
            ops.append(np.eye(2)); subs.append(la + lb)
            pairs = ((a, la), (b, lb))
        else:
            l = next(letters)
            pairs = ((a, l), (b, l))
        for e, l in pairs:
            if isinstance(e, P.End):
                (ins if e.side == "L" else outs).append((-e.y, l))
            else:
                legs[e].append(l)
    for n, r in enumerate(T.rings, 1):
        k = len(legs[n])
        spider = zspider if r.c == "green" else xspider
        ops.append(spider(0, k, phase(r.t)).reshape([2] * k)); subs.append("".join(legs[n]))
    out = "".join(l for _, l in sorted(outs)); inn = "".join(l for _, l in sorted(ins))
    M = np.einsum(",".join(subs) + "->" + out + inn, *ops)
    return M.reshape(2 ** len(outs), 2 ** len(ins))


def ratio(A, B):
    """The scalar c with A = c B (A, B proportional and non-zero)."""
    A, B = np.asarray(A).ravel(), np.asarray(B).ravel()
    k = np.argmax(np.abs(B))
    return A[k] / B[k]


def check_transcription():
    """The Lua's pressed point-and-fork and its pushed view against copy and
    pi-commutation (a structural check, not a second proof)."""
    P.check_lua()
    P.check_sites()
    # pressed point + fork <-> copy, both colourings: the fork is gone and each right hook
    # carries a point of the point's colour -- UP TO the scalar: the tangle is 1/sqrt2
    # times the two points.  The tangle grammar gives point and fork no turn, so the rule's
    # phase-free precondition is built in.
    for c, point, fork in (("red", xspider, zspider), ("green", zspider, xspider)):
        T = P.parse_tangle(f"{c} point, {P.OTHER[c]} fork")
        assert T.kind == "copy" and [r.t for r in T.rings] == [0, 0]
        assert np.allclose(tangle_map(T).ravel(), (fork(1, 2, 0) @ point(0, 1, 0)).ravel())
        D = P.view_pressed(T)
        assert D.kind == "points" and [(r.c, r.t) for r in D.rings] == [(c, 0), (c, 0)]
        assert sorted(P.segments(D), key=str) == sorted(((1, P.End("R", P.ROW)), (2, P.End("R", -P.ROW))), key=str)
        pt = point(0, 1, 0).ravel()
        assert np.allclose(tangle_map(D).ravel(), kron(pt, pt).ravel())
        assert np.allclose(tangle_map(T).ravel(), tangle_map(D).ravel() / np.sqrt(2))
        # the pan never changes a colour or a turn
        assert {r.c for r in D.rings} <= {r.c for r in T.rings} and all(r.t == 0 for r in D.rings)

    # pushed <-> pi-commutation, every turn, both colourings.  Read left to right, the
    # tangle is the traveller then the ring, Z(t) X(pi); pushed, it is the mirrored ring
    # then the traveller, X(pi) Z(24 - t) -- the card's X(pi) Z(t) ~ Z(24 - t) X(pi) with t
    # renamed 24 - t.  UP TO a non-zero scalar, and exactly equal only at turn 0.
    assert np.isclose(phase(12), np.pi)                                   # the half-turn is pi
    for c, trav, ring in (("red", xspider, zspider), ("green", zspider, xspider)):
        exact = []
        for t in range(P.WHOLE):
            T = P.half_turn(c, t)
            U = P.view_pushed(T)
            m, tr = U.rings
            assert (m.c, m.t, m.x) == (P.OTHER[c], (P.WHOLE - t) % P.WHOLE, T.rings[1].x)
            assert (tr.c, tr.t) == (c, 12) and tr.x > m.x                 # still a half-turn, beyond
            assert np.allclose(tangle_map(T), ring(1, 1, phase(t)) @ trav(1, 1, np.pi))
            assert np.allclose(tangle_map(U), trav(1, 1, np.pi) @ ring(1, 1, phase(m.t)))
            k = ratio(tangle_map(T), tangle_map(U))
            assert not np.isclose(k, 0) and np.allclose(tangle_map(T), k * tangle_map(U))
            if np.isclose(k, 1):
                exact.append(t)
            # the ring NOT mirrored would be wrong, except where t is its own mirror
            if t not in (0, 12):
                assert not _proportional(tangle_map(T), trav(1, 1, np.pi) @ ring(1, 1, phase(t)))
        assert exact == [0]
    k8 = ratio(tangle_map(P.half_turn("red", 8)), tangle_map(P.view_pushed(P.half_turn("red", 8))))
    assert np.isclose(k8, np.exp(2j * np.pi / 3))                         # the pictured turn's scalar

    # the macro's refusals on this challenge's tangles
    assert P.faults(P.parse_tangle, "red point, red fork") == ["point-fork"]
    assert P.faults(P.parse_tangle, "red half-turn, red ring") == ["half-turn"]
    assert P.faults(P.parse_tangle, "red half-turn, green ring, loose") == ["half-turn-item"]
    assert P.faults(P.ringcords, "pushed", "green ring, turn a third") == ["pushed"]


def bezier(p0, p1, p2, p3, n=4001):
    """Points along a cubic Bezier, as TikZ draws "p0 .. controls p1 and p2 .. p3"."""
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.array(p, dtype=float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


SWING_HALF = 0.6 / 72.27 * 2.54 / 2      # half the slide arc's 0.6pt stroke, in the picture's cm


def passes(T):
    """Achilles's rule, in the challenge's own model: the fork has nothing to say to a point of
    the other colour, so the point "passes straight through and comes out on one leg,
    unchanged" -- the map is the point alone, one cord out."""
    (point, fork) = T.rings
    assert point.c != fork.c
    return (xspider if point.c == "red" else zspider)(0, 1, phase(point.t))


def pictures():
    assert f'"{QUOTE}"' in __doc__                              # quoted byte for byte
    check_transcription()
    found = P.check_section(DIALOGUE, "Third", sizes=[1, 2, 2], frames=[1, 2, 2])
    calls = tuple(c for b in found for c, _ in b)
    p31, p32, p33, p34, p35 = calls
    b31, b32, b33, b34, b35 = (fr[0][0] for b in found for _, fr in b)
    assert [c.macro for c in calls] == ["ringcords"] * 5
    for c in calls:
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS) and P.caption_ok(c), c.caption
    assert [P.view_of(c) for c in calls] == ["", "pressed", "pressed", "", "pushed"]
    assert [P.options(c) for c in calls] == [
        {"bare": ""}, {"bare": "pressed", "expects": "passes"}, {"bare": "pressed"}, {"bare": ""},
        {"bare": "pushed", "marks": "on"}]
    assert [P.is_prediction(c) for c in calls] == [False, True, False, False, False]
    assert [c.keyed for c in calls] == [False, False, True, False, False]   # the first TRUE press
    assert [c.caption for c in calls] == ["point and fork", "as Achilles expects", "off the second hooks",
                                          "before the push", "gone through"]
    sec = DIALOGUE.section("Third")
    a1, a2, a3 = DIALOGUE.anchors("Third")
    f1, f2, f3 = DIALOGUE.following("Third")
    # the direction and captions edited this round (N5; C1, C2), in both files
    P.check_edits(DIALOGUE, "Third", [
        ("to the mirror side, the same turn the other way about.",
         "to the mirror side, round beneath the cord, the same turn the other way about."),
        ("{a point and a fork}", "{point and fork}"),
        ("{pushed through}", "{gone through}")])

    # P3.1: the red point and the green fork, the state Achilles predicts on
    assert p31.args == p32.args == p33.args == ("red point, green fork",)
    assert "This is a red point — a red ring with just the one cord" in a1
    assert "a green ring made to fork: one cord in, two cords out, its bead at the top, no turn." in a1
    assert b31.setting == "loose" and b31.drawn.kind == "copy" and b31.marks == ()
    pt, fk = b31.drawn.rings
    assert (pt.c, fk.c) == ("red", "green") and pt.x < fk.x and pt.t == fk.t == P.parse_turn("no turn")
    assert f1.startswith("**Achilles:** Then I can tell you what happens without your pan.")
    assert "It passes straight through and comes out on one leg, unchanged." in f1   # the prediction

    # B3.2, the prediction frame: Achilles's rule "passes", drawn paler, never keyed, beside the
    # true press; the macro's wrong rule is the challenge's model of what he said, and both
    # are refuted by the press: one point against two
    for q in P.rule_quotes()[:2]:
        assert q in sec, q                                     # the Lua's WRONG quotes, spoken here
    assert P.rule_quotes()[0] in f1 and P.rule_quotes()[1] in a2
    assert b32.expects == "passes" and b32.pale and not b32.keyed and b32.labels == ()
    assert b32.marks == () and b32.mark_tikz == ("\\begin{scope}[transparency group, opacity=0.500]",)
    assert b32.made == b31.drawn                                            # from the tangle as made
    assert b32.setting == "hooks" and [(r.c, r.t) for r in b32.drawn.rings] == [("red", 0)]
    assert all(e == P.End("R", P.ROW) for s in P.segments(b32.drawn) for e in s if isinstance(e, P.End))
    assert np.allclose(tangle_map(b32.drawn).ravel(), passes(b31.drawn).ravel())   # the rule drawn
    assert tangle_map(b32.drawn).shape == (2, 1)                                   # one cord out
    assert P.describe(p32)[1:] == ["ringcords: + expects passes (the whole picture paler)"]
    assert P.faults(P.ringcords, "pressed; expects=passes", "green and red, two cords") == ["wrong-passes"]
    assert P.faults(P.ringcords, "expects=passes", "red point, green fork") == ["expects-pressed"]
    assert P.faults(P.ringcords, "pressed; expects=passes", "red point, green fork", True) == ["expects-keyed"]

    # B3.2, the true press: off the second hooks, two red points, the green ring gone (keyed)
    assert "There is no green ring left at all" in a2 and "each is a red point" in a2
    assert b33.setting == "hooks" and b33.drawn.kind == "points" and not b33.pale and b33.expects is None
    assert b33.labels == ("second hooks", "red", "red") and b33.marks == ()
    assert [(r.c, r.t) for r in b33.drawn.rings] == [("red", 0), ("red", 0)]
    assert all(e.side == "R" for s in P.segments(b33.drawn) for e in s if isinstance(e, P.End))
    assert np.allclose(tangle_map(b31.drawn).ravel(), tangle_map(b33.drawn).ravel() / np.sqrt(2))
    # the prediction differs from what came off: one point against two, one cord out against two
    assert len(b32.drawn.rings) == 1 and len(b33.drawn.rings) == 2
    assert tangle_map(b32.drawn).shape != tangle_map(b33.drawn).shape
    assert b32.drawn.rings[0] in b33.drawn.rings                           # the upper one kept
    assert f2.startswith("**Achilles:** *[staring]* Two.") and QUOTE in f2   # the docstring's line

    # B3.3: the push -- the traveller the same, the ring it passed through mirrored
    assert p34.args == p35.args == ("red half-turn, green ring, turn a third",)
    assert "a red ring with its bead set square at the halfway mark" in sec
    assert "The green ring sits with its bead a third of the way round" in a3
    assert "still a red half-turn, its bead square at the mark" in a3
    assert "the same turn the other way about" in a3
    assert b34.marks == ()
    tr, g = b34.drawn.rings
    assert (tr.c, tr.t) == ("red", P.parse_turn("the halfway mark")) and tr.x < g.x
    assert (g.c, g.t) == ("green", P.parse_turn("a third of the way round"))      # four o'clock
    g2, tr2 = b35.drawn.rings
    assert (g2.c, g2.t, g2.x) == ("green", 16, g.x)                                # eight o'clock
    assert (tr2.c, tr2.t) == ("red", 12) and tr2.x > g2.x
    k = ratio(tangle_map(b34.drawn), tangle_map(b35.drawn))
    assert np.allclose(tangle_map(b34.drawn), k * tangle_map(b35.drawn)) and not np.isclose(k, 1)
    assert b35.made == b34.drawn                                                   # the same moment
    # its marks (marks=on): the traveller where it waited, its way "through the green ring to the
    # far side" (over it), the green bead where it was and its swing "round beneath the cord"
    assert "pushes the red half-turn along the cord and through the green ring to the far side" in a3
    assert [m.kind for m in b35.marks] == ["ghost", "slide", "bead", "bead-arc"]
    ghost, slide, bead, arc = b35.marks
    assert ghost.ring == tr and ghost.at == (tr.x, tr.y)                           # where it waited
    assert all(np.hypot(ghost.at[0] - q.x, ghost.at[1] - q.y) > 2 * P.RO for q in b35.drawn.rings)
    (x0, y0), c1, c2, (x1, y1) = slide.pts
    assert (x0, x1) == (tr.x, tr2.x)
    path = bezier((x0, y0), c1, c2, (x1, y1))
    over = path[np.argmin(abs(path[:, 0] - g2.x))]
    assert over[1] - SWING_HALF - (g2.y + P.RO) > 0                                # clear over the ring
    assert bead.ring == g and np.allclose(bead.at, P.bead_xy(g))                   # the bead as it was
    (cx, cy), (rad, s0, s1) = arc.pts
    assert (cx, cy) == (g2.x, g2.y) and arc.dir == 1
    bottom = -90                                                                   # beneath the cord
    assert s0 > bottom > s1 and P.turn_angle(g.t) > s0 and s1 > P.turn_angle(g2.t)  # 8 -> 16 by the bottom
    assert P.describe(p35)[1:] == ["ringcords: + ghost red ring 12 at (-0.500,0.000)",
                                   "ringcords: + slide arc -0.500 -> 0.660 above the cord, over the ring",
                                   "ringcords: + ghost bead 8 on the green ring",
                                   "ringcords: + bead arc 8 -> 16 clockwise through the bottom"]
    # the mirror is the cord's: a turn on the other side of the cord would swing over the top
    assert P.mirror_dir(8) == 1 and P.mirror_dir(16) == -1 and P.mirror_dir(0) == P.mirror_dir(12) == 0
    assert P.ringcords("pushed", p35.args[0]).marks == ()                          # round 3's call
    assert f3 == ("**Achilles:** The green ring's turn has flipped to the far side, the mirror of what it "
                  "was. A half-turn of one colour, pushed through the other, turns the other's turn around.")


# --- pictures: phases below are Fractions in units of pi ---
def red_into_green():
    """A red point fed into a phase-free green 1->2 dot."""
    g = zx.Graph()
    x = g.add_vertex(VT.X, 0, 0)                 # red point (a red 0->1 state)
    z = g.add_vertex(VT.Z, 0, 1)                 # phase-free green 1->2 copier
    o1 = g.add_vertex(VT.BOUNDARY, 0, 2)
    o2 = g.add_vertex(VT.BOUNDARY, 1, 2)
    g.add_edge((x, z)); g.add_edge((z, o1)); g.add_edge((z, o2))
    g.set_inputs(()); g.set_outputs((o1, o2))
    return g


def two_red_points():
    """The same red point on two separate wires."""
    g = zx.Graph()
    x1 = g.add_vertex(VT.X, 0, 0); o1 = g.add_vertex(VT.BOUNDARY, 0, 1)
    x2 = g.add_vertex(VT.X, 1, 0); o2 = g.add_vertex(VT.BOUNDARY, 1, 1)
    g.add_edge((x1, o1)); g.add_edge((x2, o2))
    g.set_inputs(()); g.set_outputs((o1, o2))
    return g


def red_pi_then_green(a):
    """A red pi-dot, then a green a-dot, on one wire."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    x = g.add_vertex(VT.X, 0, 1, phase=Fraction(1))          # pi
    z = g.add_vertex(VT.Z, 0, 2, phase=Fraction(a))
    o = g.add_vertex(VT.BOUNDARY, 0, 3)
    g.add_edge((i, x)); g.add_edge((x, z)); g.add_edge((z, o))
    g.set_inputs((i,)); g.set_outputs((o,))
    return g


def green_negpi_then_red(a):
    """A green (-a)-dot, then a red pi-dot -- the turn reversed, the half-turn passed."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    z = g.add_vertex(VT.Z, 0, 1, phase=Fraction(-a))
    x = g.add_vertex(VT.X, 0, 2, phase=Fraction(1))          # pi
    o = g.add_vertex(VT.BOUNDARY, 0, 3)
    g.add_edge((i, z)); g.add_edge((z, x)); g.add_edge((x, o))
    g.set_inputs((i,)); g.set_outputs((o,))
    return g


def sketchbook():
    # copy: one red point into a green fork = two red points, UP TO the scalar sqrt2
    assert same_picture_up_to_scalar(red_into_green(), two_red_points())
    # the exact scalar, stated: the copy comes out 1/sqrt2 times the clean pair
    assert np.allclose(matrix_of(red_into_green()), matrix_of(two_red_points()) / np.sqrt(2))

    # derive the copy with the rewrite rules -- and pin the STRUCTURE the rule produces,
    # not just the tensor (which already agreed).  Before: one green fork (degree 3) joins
    # both outputs.  After full_reduce: the fork is gone, each output carries its own
    # degree-1 point, and the two outputs sit in different components -- the point was
    # genuinely copied onto two independent wires.
    before = red_into_green()
    assert max(len(list(before.neighbors(v))) for v in before.vertices()) == 3   # the fork
    g = red_into_green()
    simp.full_reduce(g)
    assert same_picture_up_to_scalar(g, two_red_points())
    assert all(len(list(g.neighbors(v))) == 1                                    # fork gone
               for v in g.vertices() if g.type(v) != VT.BOUNDARY)
    assert not _connected(g, g.outputs()[0], g.outputs()[1])                     # two independent wires

    # pi-commutation: a half-turn pushed through flips the other colour's turn, up to scalar
    a = Fraction(7, 10)
    assert same_picture_up_to_scalar(red_pi_then_green(a), green_negpi_then_red(a))
    # exact ONLY at phase 0 -- so it is a genuine up-to-scalar equality, not a hidden exact one
    assert same_picture(red_pi_then_green(Fraction(0)), green_negpi_then_red(Fraction(0)))
    assert not same_picture(red_pi_then_green(a), green_negpi_then_red(a))


if __name__ == "__main__":
    verdict("Copy", blackboard, sketchbook)
