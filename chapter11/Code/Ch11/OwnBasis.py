"""Challenge I: The Press Copies Its Own  (no-cloning from linearity; the copier copies its own basis)
===================================================================================================
"...Oh. It doesn't copy the crosswise stone. It makes a *pair* of it."  -- Achilles

Claim: the only linear map copying |0> and |1> is the copier
delta = |00><0| + |11><1|, and it sends psi = a|0> + b|1> to a|00> + b|11>, which is
psi (x) psi only if ab = 0; any linear machine with an apparatus that copies |0> and |1>
leaves no |01> part where a clone has ab|A_psi>; delta|+-> = Phi+- (a pair, not two
copies).  In pictures: red points go through the green copier as two red points, and
a green point fuses with it into one cup (times 1/sqrt2).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import BELL, CUP, I2, kron, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch11 as PC
import props_cloth as CLOTH


DELTA = np.outer(ket(0, 0), ket(0)) + np.outer(ket(1, 1), ket(1))    # 4x2, one wire in, two out
PLUS = (ket(0) + ket(1)) / np.sqrt(2)
MINUS = (ket(0) - ket(1)) / np.sqrt(2)
PHI_MINUS = (ket(0, 0) - ket(1, 1)) / np.sqrt(2)


def cnot(n, c, t):
    """CNOT on n wires, control c, target t (wire 0 is the TOP wire, written first)."""
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for k in range(2 ** n):
        bits = [(k >> (n - 1 - i)) & 1 for i in range(n)]
        if bits[c]:
            bits[t] ^= 1
        M[int("".join(map(str, bits)), 2), k] = 1
    return M


def readings(state, basis):
    """p(i, j) for reading both struck stones in `basis` (a pair of unit kets)."""
    return np.array([[abs(np.vdot(kron(basis[i], basis[j]).ravel(), state)) ** 2
                      for j in (0, 1)] for i in (0, 1)])


def blackboard():
    rng = np.random.default_rng(11)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))       # cup UNNORMALIZED; Phi+ = cup/sqrt2
    assert np.allclose(ket(0, 1), kron(ket(0), ket(1)).ravel())   # top wire first

    # the copier copies the two plain states, exactly (normalized kets, no sqrt2)
    assert np.allclose(DELTA @ ket(0), ket(0, 0)) and np.allclose(DELTA @ ket(1), ket(1, 1))
    # ... and a linear map from one wire to two is fixed by those two columns: it IS delta
    L = np.column_stack([ket(0, 0), ket(1, 1)])
    assert np.allclose(L, DELTA)

    # a generic, NON-REAL stone: delta psi = a|00> + b|11>, not psi (x) psi
    psi = random_state(rng)
    a, b = psi
    assert not np.allclose(psi, psi.conj()) and not np.isclose(a * b, 0)
    out = DELTA @ psi
    assert np.allclose(out, a * ket(0, 0) + b * ket(1, 1))
    assert not np.allclose(out, kron(psi, psi).ravel())
    # product test (Theorem 6.1): the 2x2 of coefficients has det ab, rank 2 -- entangled
    assert np.isclose(np.linalg.det(out.reshape(2, 2)), a * b)
    assert np.linalg.matrix_rank(out.reshape(2, 2)) == 2
    assert np.linalg.matrix_rank(kron(psi, psi).reshape(2, 2)) == 1

    # the crosswise stones: delta|+> = Phi+, delta|-> = Phi- (a pair, not two copies)
    assert np.allclose(DELTA @ PLUS, BELL) and np.allclose(DELTA @ MINUS, PHI_MINUS)
    assert not np.allclose(DELTA @ PLUS, kron(PLUS, PLUS).ravel())

    # the blank is inside the copier: CNOT (I (x) |0>) = delta, exactly
    assert np.allclose(cnot(2, 0, 1) @ kron(I2, ket(0)), DELTA)

    # Result 11.1 with an apparatus.  The basis cloner V = CNOT_{0->2} CNOT_{0->1} on
    # (stone, blank, apparatus) clones |0> and |1> (apparatus ends in |0>, |1>) ...
    V = cnot(3, 0, 2) @ cnot(3, 0, 1)
    assert np.allclose(V @ ket(0, 0, 0), ket(0, 0, 0)) and np.allclose(V @ ket(1, 0, 0), ket(1, 1, 1))
    got = V @ kron(psi, ket(0), ket(0)).ravel()
    assert np.allclose(got, a * ket(0, 0, 0) + b * ket(1, 1, 1))
    part01 = kron(ket(0, 1).reshape(1, 4), I2)               # (<01| (x) I): the |01> part
    assert np.allclose(part01 @ got, 0)                      # no |01> part at all
    A_psi = random_state(rng)                                # any unit apparatus state
    clone = kron(psi, psi, A_psi).ravel()
    assert np.allclose(part01 @ clone, a * b * A_psi)        # a clone's |01> part is ab A_psi
    assert not np.allclose(part01 @ clone, 0)
    # ... and so does EVERY linear machine that copies |0>, |1>: a random non-real linear L
    # on (stone, blank, apparatus of two wires), forced only on the two plain inputs
    A = random_state(rng, 4)
    A0, A1 = random_state(rng, 4), random_state(rng, 4)
    x0, x1 = kron(ket(0), ket(0), A).ravel(), kron(ket(1), ket(0), A).ravel()
    t0, t1 = kron(ket(0, 0), A0).ravel(), kron(ket(1, 1), A1).ravel()
    X = np.column_stack([x0, x1])                            # orthonormal inputs
    M = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    Lm = M @ (np.eye(16) - X @ X.conj().T) + np.column_stack([t0, t1]) @ X.conj().T
    assert np.allclose(Lm @ x0, t0) and np.allclose(Lm @ x1, t1)
    got = Lm @ kron(psi, ket(0), A).ravel()
    assert np.allclose(got, a * t0 + b * t1)                 # branch by branch
    part01 = kron(ket(0, 1).reshape(1, 4), np.eye(4))
    assert np.allclose(part01 @ got, 0)                      # still no |01> part

    # a red point with any other turn is not copied (Coecke's Sketchbook, Section 11.1):
    # (1/sqrt2) X(0,1,theta) has ab = (1 - e^{2i theta}) / 4, zero only at theta = 0, pi
    for theta in (0, np.pi, 1 / 3 * np.pi, 0.7, 2.1):
        red =(PLUS + np.exp(1j * theta) * MINUS) / np.sqrt(2)
        assert np.isclose(np.linalg.norm(red), 1)
        assert np.isclose(red[0] * red[1], (1 - np.exp(2j * theta)) / 4)
        copied = np.allclose(DELTA @ red, kron(red, red).ravel())
        assert copied == (np.isclose(theta, 0) or np.isclose(theta, np.pi))

    # --- the proving-book (every settling the scene allows) --------------------
    plain, cross = (ket(0), ket(1)), (PLUS, MINUS)
    # settled plain-low / plain-high: both struck stones read the same, every line
    assert np.allclose(readings(DELTA @ ket(0), plain), [[1, 0], [0, 0]])
    assert np.allclose(readings(DELTA @ ket(1), plain), [[0, 0], [0, 1]])
    # "A crosswise stone is half high and half low, near enough"
    assert np.isclose(abs(np.vdot(ket(1), PLUS)) ** 2, 0.5)
    # a crosswise stone asked the crosswise question gives the same answer every time
    assert np.isclose(abs(np.vdot(PLUS, PLUS)) ** 2, 1) and np.isclose(abs(np.vdot(MINUS, PLUS)) ** 2, 0)
    # settled crosswise one way, read at cross: each a coin toss, yet ALWAYS agree
    p = readings(DELTA @ PLUS, cross)
    assert np.allclose(p, [[0.5, 0], [0, 0.5]])
    assert np.allclose(p.sum(axis=1), 0.5) and np.allclose(p.sum(axis=0), 0.5)
    # ... settled the other way: ALWAYS disagree
    p = readings(DELTA @ MINUS, cross)
    assert np.allclose(p, [[0, 0.5], [0.5, 0]])
    assert np.allclose(p.sum(axis=1), 0.5) and np.allclose(p.sum(axis=0), 0.5)
    # Achilles's guess -- two crosswise stones, each saying its original's answer every
    # time -- would read (+,+) always: NOT what the book shows
    assert np.allclose(readings(kron(PLUS, PLUS).ravel(), cross), [[1, 0], [0, 0]])
    assert not np.allclose(readings(DELTA @ PLUS, cross), [[1, 0], [0, 0]])
    # settled crosswise, EITHER way, read plain: both high or both low, half the time
    # each, never one of each
    for s in cross:
        assert np.allclose(readings(DELTA @ s, plain).ravel(), [0.5, 0, 0, 0.5])
    # two honest crosswise stones read plain would be two separate tosses: one of each
    # (high-and-low or low-and-high) as often as not
    p = readings(kron(PLUS, PLUS).ravel(), plain)
    assert np.allclose(p, 0.25) and np.isclose(p[0, 1] + p[1, 0], 0.5)

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the black slab from the side nearer the
# fire as the press is set down (keyed: the near stone, the far stone, the press,
# the reflection, the counters, the viewer), then, on the Tortoise's cloth, the
# fork with a red point before and after it is threaded (keyed: point, fork), and
# the green point threaded and the ring lifted off, a bend,
#   \presstable[operations]{scene}{caption}, \copycloth[operations]{scene}{caption}
#                                                         (pdf/props/ch11.tex)
# tools/props_ch11.py transcribes the Lua (with L1-L5); the calls are read out of
# dialog.md, never copied.
#
# The ties (identity card, section 4), a structural check against this file's
# blackboard, not a second proof: a red point is a stone settled plain, and the
# copier sends it to two copies -- a product, Schmidt rank 1 -- which the cloth
# draws as two separate points, each on a cord of its own; a green point is a stone
# settled crosswise, and the copier sends it to cup/sqrt2 exactly -- Schmidt rank 2
# -- which the cloth draws as one bend, both ends on one piece of cord.  The bend
# is the cup (unnormalized; the 1/sqrt2 stays in chapter.md).  On the table every
# stone of the fresh pair is half of Phi+: 1/2 I alone, its reflection slack.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = "...Oh. It doesn't copy the crosswise stone. It makes a *pair* of it."
F = "the fresh pair"


def half(state, n, i):
    """The one-stone state of stone i of an n-stone pure state (top wire first)."""
    m = np.moveaxis(state.reshape([2] * n), i, 0).reshape(2, -1)
    return m @ m.conj().T


def pieces(S):
    """The separate pieces of cord on a cloth scene: each a set of its rings (1-based)
    and a count of its loose ends.  A ring joins every cord that ends at it."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for i, d in enumerate(S["cords"]):
        parent.setdefault(("c", i), ("c", i))
        for e in (d.get("a"), d.get("b")):
            if isinstance(e, int):
                parent[find(("c", i))] = find(("r", e))
    groups = {}
    for i, d in enumerate(S["cords"]):
        root = find(("c", i))
        g = groups.setdefault(root, {"rings": set(), "loose": 0})
        for e in (d.get("a"), d.get("b")):
            if isinstance(e, int):
                g["rings"].add(e)
            elif e == "loose":
                g["loose"] += 1
    return list(groups.values())


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    calls = DIALOGUE.check_layout("First", sizes=[1, 2, 1])
    p11, p12, p13, p14 = calls
    text, raw = DIALOGUE.read(DIALOGUE.dialog), DIALOGUE.read(DIALOGUE.raw)
    spoken = PC.spoken_sentences(text) | PC.spoken_sentences(raw)
    for c in calls:
        assert words_only(c.caption, forbidden=PC.CAPTION_FORBIDDEN), c.caption
        assert c.macro == "presstable" or CLOTH.caption_ok(c.caption), c.caption
        assert PC.normal_sentence(c.caption) not in spoken, c.caption   # the gigue: never said twice
        assert PC.place_ok("First", c), c.text
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("First")] == [[1.66], [1.78, 1.78], [1.78]]
    assert [c.macro for c in calls] == ["presstable", "copycloth", "copycloth", "copycloth"]
    assert [c.keyed for c in calls] == [True, True, False, False]
    assert [(c.opt, c.args[0]) for c in calls] == [
        ("", F), ("", "the fork, red point"), ("thread", "the fork, red point"),
        ("thread, lift off", "the fork, green point")]
    assert [c.caption for c in calls] == [
        "set down beside the far stone", "a green ring made to fork", "slid through, the green ring gone",
        "bent round on itself"]
    t11, c12, c13, c14 = (PC.picture(c) for c in calls)
    a1, a2, a3 = DIALOGUE.anchors("First")
    f1, f2, f3 = DIALOGUE.following("First")
    section = DIALOGUE.section("First")
    assert QUOTE in section                                       # the docstring's line

    # P1.1: the press set down on the black at the far end, keyed; the fresh pair
    # unread, each stone restless with its reflection slack; the counters in the tray;
    # the viewer by the tray, collar upright, needle in the middle
    assert t11.dump == "near restless slack; far restless slack; strikes 0; tray 3; viewer tray upright middle"
    assert t11.labels == ("near stone", "far stone", "press", "reflection", "counters", "viewer")
    v = t11.shown
    things = {t.name: t for t in v.black}
    assert set(things) == {"press", "near", "far"}
    assert things["press"] == PC.Thing("press", "press", PC.XPRESS, None, "tight")
    assert things["near"] == PC.Thing("near", "stone", PC.XNEAR, "restless", "slack")
    assert things["far"] == PC.Thing("far", "stone", PC.XFAR, "restless", "slack")
    assert (v.tray, v.counters, v.viewer) == (3, (), ("tray", PC.XVIEW["tray"], "upright", "middle"))
    assert (v.frame, v.ends) == (PC.TABLE_FRAME, PC.SLAB_ENDS)
    assert "beside the far stone" in a1 and "as a half of a pair's always does" in a1      # D2
    assert f1.startswith("**Crab:** The copying press.")
    # every keyed word is named in the dialogue before the picture shows it
    before = text[:text.index(p11.text)].lower()
    for w in t11.labels:
        assert w in before, w
    assert "on the side nearer the fire" in before and "spread and bare" in before       # D1
    # L1 (a rule, not coordinates; the x-values are the Lua's): the far stone in the
    # slab's right half, the press beside it at the far end; the gap from the near stone
    # to the far stone the widest on the black, at least twice the far stone's to the
    # press; the tray at the middle of the edge; no two reflections touch
    xs = sorted((t.x, t.name) for t in v.black)
    assert [n for _, n in xs] == ["near", "far", "press"]
    assert 0 < PC.XFAR < PC.XPRESS < PC.SX1 and PC.XNEAR < 0 and PC.XTRAY == (PC.SX0 + PC.SX1) / 2
    gaps = [xs[i + 1][0] - xs[i][0] for i in range(2)]
    assert gaps[0] == max(gaps) and round(gaps[0], 2) >= round(2 * gaps[1], 2)
    # the drawn extents (card 1c): every stone, standing and reflected, a positive gap
    # from the press's legs, body and chutes and from the other stone; every reflection
    # a positive gap inside the slab's ends
    for a, b in ((things["near"], things["far"]), (things["near"], things["press"]),
                 (things["far"], things["press"])):
        for view in (PC.standing, PC.reflection):
            assert min(PC.gap(r, q) for r in view(a) for q in view(b)) > 0, (a.name, b.name)
    for t in things.values():
        for x0, y0, x1, y1 in PC.standing(t) + PC.reflection(t):
            assert PC.SX0 < x0 and x1 < PC.SX1                    # not equality: a gap
    # the tie: either half of the fresh pair (Phi+, near stone the top wire) is 1/2 I
    for i, name in enumerate(("near", "far")):
        rho = half(BELL, 2, i)
        assert np.allclose(rho, I2 / 2) and np.isclose(np.trace(rho @ rho).real, 0.5)
        assert things[name].refl == "slack"

    # P1.2 | P1.3: the fork, keyed, a red point at the in-cord's left end; threaded, the
    # fork and its in-cord gone and a red point resting at the head of each out-cord,
    # where the fork stood -- on the fork's row, either side of its place (FORKHEAD) --
    # its cord running on by its own slanted leg and out to the right (D3)
    assert {PC.height(c) for c in (p12, p13)} == {1.78}
    assert c12.labels == ("point", "fork")
    assert c12.dump.endswith("1:end red 0 @0,1; 2:ring green 0 @2,1; cord 1-2 [0,1 2,1]; "
                             "cord loose-2 [6,0 3,0 2,1]; cord loose-2 [6,2 3,2 2,1]")
    assert c13.dump == ("scene=fork point=red threaded=true lifted=false | cols=7 rows=3 y=0..2 "
                        "near=false folded=false; 1:end red 0 @1.55,1; 2:end red 0 @2.45,1; "
                        "cord loose-1 [6,0 3,0 1.55,1]; cord loose-2 [6,2 3,2 2.45,1]")
    (fork,) = [(r["x"], r["y"]) for r in c12.cloth["things"] if r["kind"] == "ring"]
    ends = [(r["kind"], r["c"], r["x"], r["y"]) for r in c13.cloth["things"]]
    assert ends == [("end", "red", fork[0] - PC.FORKHEAD, fork[1]), ("end", "red", fork[0] + PC.FORKHEAD, fork[1])]
    assert 0 < PC.FORKHEAD < 0.5                                  # either side, on the fork's own slot
    outs = [d["pts"] for d in c12.cloth["cords"] if "loose" in (d["a"], d["b"])]
    heads = [d["pts"] for d in c13.cloth["cords"]]
    assert [p[:2] for p in outs] == [[[6, 0], [3, 0]], [[6, 2], [3, 2]]]      # the out-cords' courses ...
    assert [p[:2] for p in heads] == [p[:2] for p in outs]                    # ... kept, running right
    # "no longer and no shorter" (D3): each out-cord, measured on the picture (columns
    # and rows are spaced differently, props_cloth.cloth_xy), within a twentieth of its
    # length before -- the Lua reports +5% and -3.5%; the slanted legs alone change by
    # more (+12%, -9%), so the whole out-cord is what the direction's words measure
    def length(pic, pts):
        xy = [CLOTH.cloth_xy(pic.cloth, pic.options, x, y) for x, y in pts]
        return sum(np.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(xy, xy[1:]))
    change = [length(c13, h) / length(c12, o) - 1 for h, o in zip(heads, outs)]
    assert all(abs(c) <= 1 / 20 + 1e-9 for c in change), change
    assert change[0] > 0 > change[1]                              # the upper a little longer, the lower shorter
    assert "where the fork stood" in a2 and "only the green ring and its in-cord have gone" in a2   # D3
    assert f2.startswith("**Achilles:** Two, and both red.")
    # P1.4: the green point threaded and the bare ring lifted off: one cord bent round on
    # itself, the turn at the left, both ends running out to the right
    assert c14.dump.endswith("near=false folded=false; cord loose-loose [6,0 3,0 2,1 3,2 6,2]")
    (bend,) = c14.cloth["cords"]
    assert not c14.cloth["things"] and bend["a"] == bend["b"] == "loose"
    assert min(p[0] for p in bend["pts"]) == 2 < bend["pts"][0][0] == bend["pts"][-1][0] == 6
    assert c14.frame == c13.frame == 1.78                         # a ring the same size throughout
    assert "gathers the red points into her shell" in a3 and "back in her shell" in a3    # D4
    assert f3.startswith("**Tortoise:** Red goes through as two.")
    # the green point threaded (not drawn: the direction gives it in words): one green
    # ring with just the two out-cords; and red has no bare ring to lift off
    green = PC.copycloth("thread", "the fork, green point")
    assert [(r["kind"], r["c"]) for r in green.cloth["things"]] == [("ring", "green")]
    assert len(green.cloth["cords"]) == 2
    assert PC.faults(PC.copycloth, "thread, lift off", "the fork, red point") == ["fork-lift-red"]
    assert PC.faults(PC.copycloth, "lift off", "the fork, green point") == ["fork-lift-early"]

    # the ties.  Red: two separate points, one loose end each -- the copier's output on
    # a plain stone is a product (Schmidt rank 1), each factor the stone again.
    red = pieces(c13.cloth)
    assert len(red) == 2 and all(len(g["rings"]) == 1 and g["loose"] == 1 for g in red)
    assert all(r["c"] == "red" and r["kind"] == "end" for r in c13.cloth["things"])
    for a in (0, 1):
        out = DELTA @ ket(a)
        assert np.linalg.matrix_rank(out.reshape(2, 2)) == 1
        assert np.allclose(out, kron(ket(a), ket(a)).ravel())    # each the very twin
    # Green: one piece of cord with both ends loose -- the copier's output on a crosswise
    # stone is the cup over sqrt2, exactly (Schmidt rank 2): a bend, not two points
    (piece,) = pieces(c14.cloth)
    assert piece == {"rings": set(), "loose": 2}
    assert len(pieces(green.cloth)) == 1                          # before the lift: one piece too
    assert np.allclose(DELTA @ PLUS, CUP / np.sqrt(2))
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))            # the bend is the UNNORMALIZED cup
    assert np.linalg.matrix_rank((DELTA @ PLUS).reshape(2, 2)) == 2
    assert not np.allclose(DELTA @ PLUS, kron(PLUS, PLUS).ravel())   # not two green points


# --- pictures -----------------------------------------------------------------
# A one-legged dot is sqrt2 times a unit state (red phase a*pi: sqrt2|a>; green phase 0:
# |0>+|1> = sqrt2|+>), Chapter 8's unnormalized points.  Every point below carries
# add_power(-1), the 1/sqrt2 the chapter writes in front of it.

def fed_copier(colour, phase):
    """A normalized point (red 'X' or green 'Z', phase in units of pi) on the copier's
    in-leg; outputs b1 (top), b2.  The copier is the phase-free Z(1->2): no scalar."""
    g = zx.Graph()
    pt = g.add_vertex(VT.X if colour == "red" else VT.Z, 0.5, 0, phase=Fraction(phase))
    c = g.add_vertex(VT.Z, 0.5, 1)
    o1, o2 = g.add_vertex(VT.BOUNDARY, 0, 2), g.add_vertex(VT.BOUNDARY, 1, 2)
    g.add_edge((pt, c)); g.add_edge((c, o1)); g.add_edge((c, o2))
    g.set_inputs(()); g.set_outputs((o1, o2))
    g.scalar.add_power(-1)                                   # the point's 1/sqrt2
    return g


def two_points(phase):
    """Two separate normalized red points, phase phase*pi each: |a> (x) |a>."""
    g = zx.Graph()
    p1, p2 = g.add_vertex(VT.X, 0, 0, phase=Fraction(phase)), g.add_vertex(VT.X, 1, 0, phase=Fraction(phase))
    o1, o2 = g.add_vertex(VT.BOUNDARY, 0, 2), g.add_vertex(VT.BOUNDARY, 1, 2)
    g.add_edge((p1, o1)); g.add_edge((p2, o2))
    g.set_inputs(()); g.set_outputs((o1, o2))
    g.scalar.add_power(-2)                                   # 1/sqrt2 for each point
    return g


def bell_picture(phase):
    """1/sqrt2 cup with a green phase-(phase*pi) dot on the top leg: Phi+ or Phi-.
    cup = |00> + |11> (unnormalized), so the 1/sqrt2 is add_power(-1)."""
    g = zx.Graph()
    bend = g.add_vertex(VT.Z, 0.5, 0)
    turn = g.add_vertex(VT.Z, 0, 1, phase=Fraction(phase))
    o1, o2 = g.add_vertex(VT.BOUNDARY, 0, 2), g.add_vertex(VT.BOUNDARY, 1, 2)
    g.add_edge((bend, turn)); g.add_edge((turn, o1)); g.add_edge((bend, o2))
    g.set_inputs(()); g.set_outputs((o1, o2))
    g.scalar.add_power(-1)
    return g


def copier():
    """The green copier Z(1->2), one leg in, two out."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0.5, 0)
    c = g.add_vertex(VT.Z, 0.5, 1)
    o1, o2 = g.add_vertex(VT.BOUNDARY, 0, 2), g.add_vertex(VT.BOUNDARY, 1, 2)
    g.add_edge((i, c)); g.add_edge((c, o1)); g.add_edge((c, o2))
    g.set_inputs((i,)); g.set_outputs((o1, o2))
    return g


def cnot_with_blank():
    """CNOT (green control on the top wire, red target below) with the blank -- a red
    point -- plugged into the target.  The bare CNOT picture is CNOT/sqrt2 (row 64) and
    the red point is sqrt2|0>: the two cancel, so no compensation."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    blank = g.add_vertex(VT.X, 1, 0)
    ctl, tgt = g.add_vertex(VT.Z, 0, 1), g.add_vertex(VT.X, 1, 1)
    o1, o2 = g.add_vertex(VT.BOUNDARY, 0, 2), g.add_vertex(VT.BOUNDARY, 1, 2)
    g.add_edge((i, ctl)); g.add_edge((ctl, o1)); g.add_edge((ctl, tgt))
    g.add_edge((blank, tgt)); g.add_edge((tgt, o2))
    g.set_inputs((i,)); g.set_outputs((o1, o2))
    return g


def sketchbook():
    # (a) the PyZX green dot with one leg in and two out is EXACTLY delta (no scalar)
    assert np.allclose(matrix_of(copier()), DELTA)
    # the blank inside the node: CNOT with its blank loaded equals the copier, and the
    # rules DERIVE it (the blank fuses into the red target, which is then a bare wire)
    g = cnot_with_blank()
    assert same_picture(g, copier())
    h = g.copy(); simp.full_reduce(h)
    assert same_picture(h, copier())
    assert h.num_vertices() == 4                             # one green dot, three ends

    # (b) red points copy: a red point (phase 0 or pi) through the copier is two red
    # points, exactly; and full_reduce derives it (two separate points left)
    for a in (0, 1):
        g = fed_copier("red", a)
        assert np.allclose(matrix_of(g).ravel(), ket(a, a))
        assert same_picture(g, two_points(a))
        h = g.copy(); simp.full_reduce(h)
        assert same_picture(h, two_points(a))
        assert h.num_vertices() == 4 and h.num_edges() == 2  # two points, two ends
    # a red point with another turn is NOT copied
    assert not same_picture(fed_copier("red", Fraction(1, 3)), two_points(Fraction(1, 3)))

    # (c) green points fuse: a green point into the copier is 1/sqrt2 cup = Phi+
    # (phase pi: Phi-), exactly; the drawn 1/sqrt2 does real work
    g = fed_copier("green", 0)
    assert np.allclose(matrix_of(g).ravel(), BELL)
    assert same_picture(g, bell_picture(0))
    assert not same_picture(g, two_points(0))                # not two copies
    h = g.copy(); simp.full_reduce(h)                        # DERIVED: fusion leaves the bend
    assert h.num_vertices() == 2 and h.num_edges() == 1      # a bare boundary-boundary edge
    assert np.allclose(matrix_of(h).ravel(), BELL)           # a state: evaluates directly
    assert same_picture(h, bell_picture(0))
    g = fed_copier("green", 1)
    assert np.allclose(matrix_of(g).ravel(), PHI_MINUS) and same_picture(g, bell_picture(1))
    h = g.copy(); simp.full_reduce(h)
    assert same_picture(h, bell_picture(1))

    # (d) a green point with any other phase also comes out entangled: Schmidt rank 2
    g = fed_copier("green", Fraction(1, 3))
    T = matrix_of(g).reshape(2, 2)
    assert np.linalg.matrix_rank(T) == 2                     # rank 1 would be a product
    h = g.copy(); simp.full_reduce(h)
    assert np.allclose(matrix_of(h), matrix_of(g))


if __name__ == "__main__":
    verdict("OwnBasis", blackboard, sketchbook)
