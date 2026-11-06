"""Challenge IV: The Cradle Comes Undone (bialgebra + Hopf)
===========================================================
"Fewer cords for the same job."  -- Achilles

Claim (bialgebra): two greens crossed to two reds by a complete cross of four
wires equal one green-red pair -- up to sqrt2.
Claim (Hopf): a green and a red joined by two distinct wires equal the two
disconnected -- up to 2.  The two facts are one (complementarity <=> Hopf).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
import string
from dataclasses import replace

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import ket, kron, matrix_of, same_picture_up_to_scalar, verdict
from dfp_props import Dialogue, words_only
import props_ch08 as P


# --- spiders as matrices: the chapter's own maths, pinned in every file ---
KET_PLUS = (ket(0) + ket(1)) / np.sqrt(2)
KET_MINUS = (ket(0) - ket(1)) / np.sqrt(2)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I2 = np.eye(2, dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


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


def _proportional(A, B):
    A, B = np.asarray(A).ravel(), np.asarray(B).ravel()
    k = np.argmax(np.abs(A))
    if np.isclose(A[k], 0):
        return bool(np.allclose(A, 0) and np.allclose(B, 0))
    return bool(np.allclose(A / A[k], B / B[k]))


def blackboard():
    pin_spiders()

    # --- BIALGEBRA (all spiders phase-free, as the rule requires) ---
    # LHS: two greens (each copy 1->2) into a complete cross into two reds (each 2->1).
    # the cross swaps the two MIDDLE of the four wires, so each red meets each green.
    Z12, X21 = zspider(1, 2, 0), xspider(2, 1, 0)
    cross = kron(I2, SWAP, I2)                       # I (x) SWAP (x) I on 4 wires = the K_2,2
    lhs = kron(X21, X21) @ cross @ kron(Z12, Z12)
    rhs = Z12 @ X21                                  # one red gathers, one green issues
    # the cradle reads the parity of its two inputs and writes it onto both outputs
    parity = 0.5 * (np.outer(ket(0, 0), (ket(0, 0) + ket(1, 1)).conj())
                    + np.outer(ket(1, 1), (ket(0, 1) + ket(1, 0)).conj()))
    assert np.allclose(lhs, parity)
    # bialgebra holds UP TO sqrt2: the cradle comes out 1/sqrt2 times the single pair
    assert _proportional(lhs, rhs)
    assert np.allclose(lhs, rhs / np.sqrt(2))       # exact scalar: cradle = pair / sqrt2
    # NEGATIVE: swapping which colour gathers and which issues is a different map
    assert not _proportional(xspider(1, 2, 0) @ zspider(2, 1, 0), lhs)

    # --- HOPF (the twin) ---
    # green-(two wires)-red = green(1->2) then red(2->1) sharing BOTH legs
    hopf = X21 @ Z12
    assert np.allclose(hopf, np.outer(ket(0), KET_PLUS.conj()))          # = |0><+|, rank one
    # the disconnected picture: a red state sqrt2|0> beside a green effect <0|+<1|
    disc = np.outer((np.sqrt(2) * ket(0)), (ket(0) + ket(1)).conj())
    assert _proportional(hopf, disc)
    assert np.allclose(hopf, disc / 2)                                   # exact scalar: link = disc / 2
    # NEGATIVE: a SINGLE wire between the colours is not disconnected -- it is a bare wire
    single = xspider(1, 1, 0) @ zspider(1, 1, 0)
    assert np.allclose(single, I2)
    assert not _proportional(single, disc)

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the Crab's cords and curtain-rings
#   \ringcords[view]{tangle}{caption}           (pdf/props/ch08.tex)
# and the Crab's press, \koekpan[view]{caption}.  The Lua COMPUTES the views:
# slid, flipped, pushed, pressed, and the ring come clear of the flip-slot.
# tools/props_ch08.py transcribes the Lua; the calls are read out of this
# challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3): a green ring is a Z spider, a red
# ring an X spider; a ring's turn t (twenty-fourths, clockwise from the top as
# the company sees it) is the phase 2 pi t / 24; cords in on the left are
# inputs, out on the right outputs, the upper line first.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
QUOTE = "Fewer cords for the same job."
# the rings of the frame by the dialogue's names (the Lua's ring numbers)
FRAME_RINGS = {"top green": 1, "bottom green": 2, "top red": 3, "bottom red": 4}


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


def cradle_maps():
    """The file's own cradle (lhs), pair (rhs) and Hopf link, as in blackboard()."""
    Z12, X21 = zspider(1, 2, 0), xspider(2, 1, 0)
    lhs = kron(X21, X21) @ kron(I2, SWAP, I2) @ kron(Z12, Z12)
    return lhs, Z12 @ X21, X21 @ Z12


def check_transcription():
    """The Lua's frame and its pressed views against the bialgebra and Hopf
    laws (a structural check, not a second proof)."""
    lhs, rhs, hopf = cradle_maps()
    # pressed cradle <-> bialgebra: the transcription's cradle IS the file's lhs, and what
    # the pan presses it to (red on the left gathering both in-hooks, one cord, green on the
    # right issuing both out-hooks) IS the file's rhs; lhs = rhs / sqrt2
    T = P.parse_tangle("cradle")
    assert P.connections(T.cords) == sorted(f"{a}-{b}" for a, b in P.CRADLE_CORDS)
    assert np.allclose(tangle_map(T), lhs)
    D = P.view_pressed(T)
    red, green = D.rings
    assert (red.c, green.c) == ("red", "green") and red.x < green.x and red.t == green.t == 0
    assert P.connections(D.cords) == sorted(["L1-1", "L2-1", "1-2", "2-R1", "2-R2"])
    assert np.allclose(tangle_map(D), rhs)
    assert np.allclose(tangle_map(T), tangle_map(D) / np.sqrt(2))
    # the colours changing sides is the rule, not a drawing choice: the pair the other way
    # round (green gathering, red issuing) is not the cradle, even up to a scalar
    swapped = P.Tangle("pair", tuple(replace(r, c=P.OTHER[r.c]) for r in D.rings), D.cords)
    assert not _proportional(tangle_map(swapped), lhs)
    # pressed twice-tied pair <-> Hopf: the red point on its out-cord, the green point-effect
    # on its in-cord, nothing between -- the link is half of it
    H2 = P.parse_tangle("green and red, two cords")
    assert sorted(P.segments(H2), key=str).count((1, 2)) == 2          # two distinct cords
    assert np.allclose(tangle_map(H2), hopf)
    U = P.view_pressed(H2)
    assert not any(isinstance(a, int) and isinstance(b, int) for a, b in P.segments(U))
    r_pt, g_pt = xspider(0, 1, 0).ravel(), zspider(0, 1, 0).ravel()
    assert np.allclose(tangle_map(U), np.outer(r_pt, g_pt.conj()))
    assert np.allclose(tangle_map(H2), tangle_map(U) / 2)
    # a single cord between them is not undone: it is a bare cord
    one = P.Tangle("hopf", H2.rings, (H2.cords[0], H2.cords[1], H2.cords[3]))
    assert np.allclose(tangle_map(one), I2)
    # the pan never changes a colour or a turn
    for T0, D0 in ((T, D), (H2, U)):
        assert {r.c for r in D0.rings} <= {r.c for r in T0.rings} and all(r.t == 0 for r in D0.rings)
    # the macro's refusals on this challenge's tangles
    assert P.faults(P.ringcords, "pressed", "gate") == ["pressed"]
    assert P.faults(P.ringcords, "pressed", "green ring, four cords") == ["pressed"]
    assert P.faults(P.parse_tangle, "green and red, three cords") == ["green-and-red"]
    assert P.faults(P.parse_tangle, "green and red") == ["green-and-red"]


def spoken_cross(text):
    """The cross cords as the direction names them ("top green to top red; ...")."""
    names = "|".join(FRAME_RINGS)
    pairs = re.findall(r"(" + names + r")(?: [a-z]+)*? (?:across )?to (" + names + r")", text)
    return sorted(f"{FRAME_RINGS[a]}-{FRAME_RINGS[b]}" for a, b in pairs)


def pictures():
    assert f'"{QUOTE}"' in __doc__                              # quoted byte for byte
    check_transcription()
    lhs, rhs, hopf = cradle_maps()
    p41, p42, p43, p44 = DIALOGUE.check_layout("Fourth", sizes=[1, 1, 2])
    calls = (p41, p42, p43, p44)
    assert [c.macro for c in calls] == ["ringcords"] * 4
    for c in calls:
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS) and P.caption_ok(c), c.caption
    assert not P.quantity_words(p41.caption)                           # a drawn copy of the frame
    assert [c.opt for c in calls] == ["", "pressed", "", "pressed"]
    assert [c.keyed for c in calls] == [True, False, False, False]     # the frame keyed once
    b41, b42, b43, b44 = (P.picture(c) for c in calls)
    sec = DIALOGUE.section("Fourth")
    a1, a2, a3 = P.anchors(DIALOGUE, "Fourth")
    f1, f2, f3 = P.following(DIALOGUE, "Fourth")

    # P4.1: the cradle on the beam, keyed -- its cords the ones the scene strings
    assert p41.args == p42.args == ("cradle",)
    assert b41.setting == "beam" and b41.drawn.frame and b41.labels == ("in", "green", "red", "out")
    assert "the two green rings on the left, the two red on the right, feed-cords in, draw-cords out" in sec
    assert a1.startswith("*[He runs the four cords of the cross, one after another:")
    assert a1.endswith("The cradle stands strung on the beam.]*")
    cross = spoken_cross(a1)
    assert len(cross) == 4 and len(set(cross)) == 4
    assert P.connections(b41.drawn.cords) == sorted(cross + ["L1-1", "L2-2", "3-R1", "4-R2"])
    g_top, g_bot, r_top, r_bot = b41.drawn.rings
    assert (g_top.c, g_bot.c, r_top.c, r_bot.c) == ("green", "green", "red", "red")
    assert g_top.x < r_top.x and g_top.y > g_bot.y and r_top.y > r_bot.y   # greens left, reds right
    assert all(r.t == 0 for r in b41.drawn.rings)
    assert np.allclose(tangle_map(b41.drawn), lhs)

    # P4.2: off the pan, the pair -- red gathering on the left, green issuing on the right
    assert "one knot-pair, doing what the whole cross did" in a2
    assert "the cradle on the beam is untouched" in a2
    assert b42.setting == "hooks" and b42.labels == ()
    red, green = b42.drawn.rings
    assert (red.c, green.c) == ("red", "green") and red.x < green.x
    assert "comes off as one red ring, one cord, one green ring" in f2 and QUOTE in f2
    assert np.allclose(tangle_map(b42.drawn), rhs)
    assert np.allclose(tangle_map(b41.drawn), tangle_map(b42.drawn) / np.sqrt(2))   # bialgebra

    # P4.3, P4.4: twice tied, then nothing between
    assert p43.args == p44.args == ("green and red, two cords",)
    assert "a cord running in to the green and one running out of the red" in sec
    assert "The green ring and the red ring hang joined by two distinct cords." in a3
    assert "but with nothing between them at all" in a3
    assert b43.setting == "loose" and b44.setting == "hooks"
    assert sorted(P.segments(b43.drawn), key=str).count((1, 2)) == 2
    assert not any(isinstance(a, int) and isinstance(b, int) for a, b in P.segments(b44.drawn))
    assert [r.c for r in b44.drawn.rings] == ["green", "red"]
    assert np.allclose(tangle_map(b43.drawn), hopf)
    assert np.allclose(tangle_map(b43.drawn), tangle_map(b44.drawn) / 2)          # Hopf
    assert f3.startswith("**Achilles:** They fell apart.") and "Twice tied is untied." in f3


# --- pictures: all spiders phase-free (bialgebra / Hopf need it) ---
def cradle():
    """Two green copiers, two red adders, a complete cross of four wires between."""
    g = zx.Graph()
    i1 = g.add_vertex(VT.BOUNDARY, 0, 0); i2 = g.add_vertex(VT.BOUNDARY, 1, 0)
    gt = g.add_vertex(VT.Z, 0, 1); gb = g.add_vertex(VT.Z, 1, 1)
    rt = g.add_vertex(VT.X, 0, 2); rb = g.add_vertex(VT.X, 1, 2)
    o1 = g.add_vertex(VT.BOUNDARY, 0, 3); o2 = g.add_vertex(VT.BOUNDARY, 1, 3)
    g.add_edge((i1, gt)); g.add_edge((i2, gb))
    g.add_edge((gt, rt)); g.add_edge((gt, rb))          # top green to BOTH reds
    g.add_edge((gb, rt)); g.add_edge((gb, rb))          # bottom green to BOTH reds
    g.add_edge((rt, o1)); g.add_edge((rb, o2))
    g.set_inputs((i1, i2)); g.set_outputs((o1, o2))
    return g


def pair():
    """The single green-red pair: both inputs into one red spider, one wire on to
    one green spider, both outputs from the green -- what the pan presses the
    cradle to (Z(1,2) X(2,1))."""
    g = zx.Graph()
    i1 = g.add_vertex(VT.BOUNDARY, 0, 0); i2 = g.add_vertex(VT.BOUNDARY, 1, 0)
    r = g.add_vertex(VT.X, 0, 1); z = g.add_vertex(VT.Z, 0, 2)
    o1 = g.add_vertex(VT.BOUNDARY, 0, 3); o2 = g.add_vertex(VT.BOUNDARY, 1, 3)
    g.add_edge((i1, r)); g.add_edge((i2, r)); g.add_edge((r, z))
    g.add_edge((z, o1)); g.add_edge((z, o2))
    g.set_inputs((i1, i2)); g.set_outputs((o1, o2))
    return g


def hopf_link():
    """A green and a red joined by TWO distinct wires.  A phase-0 identity Z spider
    sits on one of the two paths so the two edges are genuinely distinct (see the trap
    demonstrated in the sketchbook: add_edge twice does NOT make a parallel wire)."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    z = g.add_vertex(VT.Z, 0, 1)
    idz = g.add_vertex(VT.Z, 1, 2)                       # phase-0 identity spider on one path
    x = g.add_vertex(VT.X, 0, 3)
    o = g.add_vertex(VT.BOUNDARY, 0, 4)
    g.add_edge((i, z))
    g.add_edge((z, x))                                   # first wire (direct)
    g.add_edge((z, idz)); g.add_edge((idz, x))          # second wire (via the identity spider)
    g.add_edge((x, o))
    g.set_inputs((i,)); g.set_outputs((o,))
    return g


def disconnected():
    """A green effect (<0|+<1|) beside a red state (sqrt2|0>) -- nothing between them."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    z = g.add_vertex(VT.Z, 0, 1)                         # green with only the input leg
    x = g.add_vertex(VT.X, 0, 3)                         # red with only the output leg
    o = g.add_vertex(VT.BOUNDARY, 0, 4)
    g.add_edge((i, z)); g.add_edge((x, o))
    g.set_inputs((i,)); g.set_outputs((o,))
    return g


def sketchbook():
    # bialgebra: the four-cross cradle = the single pair, UP TO sqrt2
    c = cradle()
    assert c.num_edges() == 8                                   # 2 in + 4 cross + 2 out
    assert all(len(list(c.neighbors(v))) == 3                   # K_2,2: each interior dot has degree 3
               for v in c.vertices() if c.type(v) in (VT.Z, VT.X))
    assert same_picture_up_to_scalar(c, pair())
    assert np.allclose(matrix_of(c), matrix_of(pair()) / np.sqrt(2))   # cradle = pair / sqrt2

    # TRAP (documented in code): adding the same simple edge twice does NOT make two
    # parallel wires.  In PyZX 0.10.6 the second add_edge cancels the first.
    trap = zx.Graph()
    tz = trap.add_vertex(VT.Z, 0, 0); tx = trap.add_vertex(VT.X, 0, 1)
    trap.add_edge((tz, tx))
    assert trap.num_edges() == 1
    trap.add_edge((tz, tx))
    assert trap.num_edges() == 0                                # cancelled -- not a double link
    assert hopf_link().num_edges() == 5                        # the genuine link is two DISTINCT paths

    # Hopf: two wires between the colours = none, UP TO 2
    assert same_picture_up_to_scalar(hopf_link(), disconnected())
    assert np.allclose(matrix_of(hopf_link()), matrix_of(disconnected()) / 2)   # link = disc / 2

    # derive the disconnection with the rewrite rules: after full_reduce the input
    # boundary and the output boundary are in different components -- the wires fell away
    g = hopf_link()
    simp.full_reduce(g)
    assert same_picture_up_to_scalar(g, disconnected())
    assert not _connected(g, g.inputs()[0], g.outputs()[0])


def _connected(g, start, target):
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for w in g.neighbors(u):
            if w not in seen:
                seen.add(w); stack.append(w)
    return target in seen


if __name__ == "__main__":
    verdict("Complementarity", blackboard, sketchbook)
