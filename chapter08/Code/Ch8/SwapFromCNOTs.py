"""Challenge V: Three Crossings, One Swap (the theorem, twice)
=============================================================
"Three of them, and they just... swap."  -- Achilles

Claim: three CNOTs, the middle one reversed, make a SWAP.
  Blackboard:  CNOT . CNOT' . CNOT = SWAP,  exact.
  Sketchbook:  three spider-CNOTs reduce to a bare crossing, up to 1/(2 sqrt2).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
import string

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
I2 = np.eye(2, dtype=complex)


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


# the three gates as explicit permutation matrices, basis |00>,|01>,|10>,|11>
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)   # control top
CNOTp = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)  # control bottom
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def blackboard():
    pin_spiders()

    # the headline, exact -- three sparse matrices multiply to the swap, no scalar
    assert np.allclose(CNOT @ CNOTp @ CNOT, SWAP)
    # NEGATIVE: the middle gate must be reversed.  Three same-orientation CNOTs give
    # CNOT^3 = CNOT, not the swap.
    assert np.allclose(CNOT @ CNOT @ CNOT, CNOT)
    assert not np.allclose(CNOT @ CNOT @ CNOT, SWAP)

    # a bare spider-CNOT (green 1->2 control, red 2->1 target, one shared leg) is CNOT/sqrt2:
    # the missing factor is the cup convention.  So the scalar is documented, not hidden.
    spider_cnot = kron(I2, xspider(2, 1, 0)) @ kron(zspider(1, 2, 0), I2)
    assert np.allclose(spider_cnot, CNOT / np.sqrt(2))

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
QUOTE = "Three of them, and they just... swap."


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


def gates_of(T):
    """The gates of a tangle of gates, left to right: (top colour, bottom colour)."""
    cols = sorted({r.x for r in T.rings})
    return [tuple(r.c for r in sorted((r for r in T.rings if r.x == x), key=lambda r: -r.y))
            for x in cols]


def check_transcription():
    """The Lua's gate, three gates and their pressed crossing against the
    three-CNOT theorem (a structural check, not a second proof)."""
    # a gate (green on the top line over red on the bottom, one cord) is the spider CNOT,
    # CNOT / sqrt2, the control on the green's line; the gate the other way up is CNOT'
    G = P.parse_tangle("gate")
    assert gates_of(G) == [("green", "red")]
    assert np.allclose(tangle_map(G), CNOT / np.sqrt(2))
    Gr = P.Tangle("gate", tuple(P.Ring(P.OTHER[r.c], r.t, r.x, r.y) for r in G.rings), G.cords)
    assert np.allclose(tangle_map(Gr), CNOTp / np.sqrt(2))
    # left to right along the lines is the order of application: two gates, the plain one
    # first, are CNOT' . CNOT -- not CNOT . CNOT' (they differ)
    T3 = P.parse_tangle("three gates")
    two = P.Tangle("three", T3.rings[:4], (P.Cord(P.End("L", P.ROW), 1), P.Cord(1, 3),
                   P.Cord(3, P.End("R", P.ROW)), P.Cord(P.End("L", -P.ROW), 2), P.Cord(2, 4),
                   P.Cord(4, P.End("R", -P.ROW)), P.Cord(1, 2), P.Cord(3, 4)))
    assert np.allclose(tangle_map(two), CNOTp @ CNOT / 2)
    assert not np.allclose(CNOTp @ CNOT, CNOT @ CNOTp)
    # three gates, the middle reversed: CNOT . CNOT' . CNOT / (2 sqrt2) = SWAP / (2 sqrt2)
    assert gates_of(T3) == [("green", "red"), ("red", "green"), ("green", "red")]
    assert np.allclose(tangle_map(T3), CNOT @ CNOTp @ CNOT / (2 * np.sqrt(2)))
    # pressed: the bare crossing, upper left to lower right and lower left to upper right --
    # SWAP, exactly (the pan drops the scalar, as the Tortoise says)
    D = P.view_pressed(T3)
    assert D.kind == "crossing" and D.rings == ()
    assert P.connections(D.cords) == ["L1-R2", "L2-R1"]
    assert np.allclose(tangle_map(D), SWAP)
    assert np.allclose(tangle_map(T3), tangle_map(D) / (2 * np.sqrt(2)))
    # the macro's refusals: a tangle of gates never hangs on the beam; no unknown tangle
    assert P.faults(P.ringcords, "beam", "three gates") == ["beam"]
    assert P.faults(P.ringcords, "beam", "gate") == ["beam"]
    assert P.faults(P.parse_tangle, "four gates") == ["tangle"]
    assert P.faults(P.parse_tangle, "gate, twice") == ["tangle"]


def pictures():
    assert f'"{QUOTE}"' in __doc__                              # quoted byte for byte
    check_transcription()
    p51, p52 = DIALOGUE.check_layout("Fifth", sizes=[1, 1])
    calls = (p51, p52)
    assert [c.macro for c in calls] == ["ringcords"] * 2
    for c in calls:
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS) and P.caption_ok(c), c.caption
    assert [c.opt for c in calls] == ["", "pressed"] and not any(c.keyed for c in calls)
    assert p51.args == p52.args == ("three gates",)
    b51, b52 = (P.picture(c) for c in calls)
    sec = DIALOGUE.section("Fifth")
    a1, a2 = P.anchors(DIALOGUE, "Fifth")
    f1, f2 = P.following(DIALOGUE, "Fifth")

    # P5.1: the three gates as the Crab strings them -- the colours top and bottom in the
    # order he names them, green up on the top line of a plain gate
    assert "green up on the top line, red down on the bottom" in sec
    spoken = re.search(r"green-red, then red-green, then green-red — the middle one reversed", a1)
    assert spoken and a1.endswith("Three crossings' worth of gate.")
    named = re.findall(r"(green|red)-(green|red)", spoken.group(0))
    assert b51.setting == "loose" and gates_of(b51.drawn) == named
    assert all(r.t == 0 for r in b51.drawn.rings)
    ring_cords = [s for s in P.segments(b51.drawn) if all(isinstance(e, int) for e in s)]
    rings = b51.drawn.rings
    assert sum(rings[a - 1].x == rings[b - 1].x for a, b in ring_cords) == 3   # one cord a gate
    assert f1.startswith("**Achilles:** *[quick, tracing it in the air]*") and "a swap, plain and exact" in f1

    # P5.2: what came off -- no ring, the two lines crossed
    assert "Off the second hooks comes no tangle at all" in a2 and "a bare crossing" in a2
    assert b52.setting == "hooks" and b52.drawn.rings == ()
    assert P.connections(b52.drawn.cords) == ["L1-R2", "L2-R1"]
    assert np.allclose(tangle_map(b52.drawn), SWAP)
    assert np.allclose(tangle_map(b51.drawn), tangle_map(b52.drawn) / (2 * np.sqrt(2)))
    assert f2 == "**Achilles:** A bare crossing. The same swap I traced — the two lines simply changed places."
    assert QUOTE in sec                                                 # the docstring's line


# --- pictures: phase-free spiders throughout ---
def three_spider_cnots():
    """Three spider-CNOTs in a row: top green,red,green; bottom red,green,red;
    three vertical cords.  The middle CNOT is reversed (control on the bottom)."""
    g = zx.Graph()
    it = g.add_vertex(VT.BOUNDARY, 0, 0); ib = g.add_vertex(VT.BOUNDARY, 1, 0)
    g1 = g.add_vertex(VT.Z, 0, 1); r1 = g.add_vertex(VT.X, 1, 1)     # CNOT  (control top)
    r2 = g.add_vertex(VT.X, 0, 2); g2 = g.add_vertex(VT.Z, 1, 2)     # CNOT' (control bottom)
    g3 = g.add_vertex(VT.Z, 0, 3); r3 = g.add_vertex(VT.X, 1, 3)     # CNOT  (control top)
    ot = g.add_vertex(VT.BOUNDARY, 0, 4); ob = g.add_vertex(VT.BOUNDARY, 1, 4)
    g.add_edge((it, g1)); g.add_edge((g1, r2)); g.add_edge((r2, g3)); g.add_edge((g3, ot))  # top wire
    g.add_edge((ib, r1)); g.add_edge((r1, g2)); g.add_edge((g2, r3)); g.add_edge((r3, ob))  # bottom wire
    g.add_edge((g1, r1)); g.add_edge((r2, g2)); g.add_edge((g3, r3))                        # vertical cords
    g.set_inputs((it, ib)); g.set_outputs((ot, ob))
    return g


def spider_cnot():
    """One spider-CNOT: green control on top, red target on bottom, one shared cord."""
    g = zx.Graph()
    it = g.add_vertex(VT.BOUNDARY, 0, 0); ib = g.add_vertex(VT.BOUNDARY, 1, 0)
    z = g.add_vertex(VT.Z, 0, 1); x = g.add_vertex(VT.X, 1, 1)
    ot = g.add_vertex(VT.BOUNDARY, 0, 2); ob = g.add_vertex(VT.BOUNDARY, 1, 2)
    g.add_edge((it, z)); g.add_edge((z, ot)); g.add_edge((z, x))
    g.add_edge((ib, x)); g.add_edge((x, ob))
    g.set_inputs((it, ib)); g.set_outputs((ot, ob))
    return g


def bare_swap():
    """A bare crossing: top in to bottom out, bottom in to top out."""
    g = zx.Graph()
    it = g.add_vertex(VT.BOUNDARY, 0, 0); ib = g.add_vertex(VT.BOUNDARY, 1, 0)
    ot = g.add_vertex(VT.BOUNDARY, 0, 1); ob = g.add_vertex(VT.BOUNDARY, 1, 1)
    g.add_edge((it, ob)); g.add_edge((ib, ot))
    g.set_inputs((it, ib)); g.set_outputs((ot, ob))
    return g


def sketchbook():
    # one spider-CNOT is CNOT/sqrt2 -- the honest per-gate scalar
    assert np.allclose(matrix_of(spider_cnot()), CNOT / np.sqrt(2))

    # the three spider-CNOTs = a bare crossing, UP TO the scalar 1/(2 sqrt2) = (1/sqrt2)^3
    assert same_picture_up_to_scalar(three_spider_cnots(), bare_swap())
    assert not same_picture(three_spider_cnots(), bare_swap())          # scalar differs
    assert np.allclose(matrix_of(three_spider_cnots()),
                       matrix_of(bare_swap()) / (2 * np.sqrt(2)))       # exact: swap / (2 sqrt2)

    # THE RULES DERIVE IT: full_reduce cooks the tangle down to a bare crossing --
    # 4 vertices left, all of them boundaries (no dots), i.e. a plain swap of wires
    g = three_spider_cnots()
    simp.full_reduce(g)
    assert g.num_vertices() == 4
    assert all(g.type(v) == VT.BOUNDARY for v in g.vertices())
    assert same_picture_up_to_scalar(g, bare_swap())


if __name__ == "__main__":
    verdict("SwapFromCNOTs", blackboard, sketchbook)
