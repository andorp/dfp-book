"""Challenge VI: Octet (the count, the revelation)
==================================================
"Eight cords. Eight, and not one of them spare — every one carrying its part, and the whole thing meaning nothing if you pull a single one out."  -- Achilles

Claim: the Octet (the bialgebra cradle) is a genuine 8-wire diagram that
evaluates, up to sqrt2, to a single green-red pair -- and only its topology
matters (deform the drawing, the tensor does not move).
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


def blackboard():
    pin_spiders()

    # the Octet as a 2->2 map: two green copiers into a complete cross into two red
    # adders (all phase-free) -- the same cradle Complementarity.py built for bialgebra
    Z12, X21 = zspider(1, 2, 0), xspider(2, 1, 0)
    cross = kron(I2, SWAP, I2)                         # I (x) SWAP (x) I on 4 wires = the K_2,2
    octet_m = kron(X21, X21) @ cross @ kron(Z12, Z12)
    pair_m = Z12 @ X21                                 # one red gathers, one green issues

    # it reads the parity of its two inputs and writes that parity on BOTH outputs
    parity = 0.5 * (np.outer(ket(0, 0), (ket(0, 0) + ket(1, 1)).conj())
                    + np.outer(ket(1, 1), (ket(0, 1) + ket(1, 0)).conj()))
    assert np.allclose(octet_m, parity)
    # the master equality: Octet = a single green-red pair, UP TO the scalar sqrt2
    assert np.allclose(octet_m, pair_m / np.sqrt(2))  # exact scalar: cradle = pair / sqrt2
    # NEGATIVE: without the full cross (only the two straight cords) it is NOT the pair
    straight = kron(X21, X21) @ kron(Z12, Z12)
    assert not np.allclose(straight, pair_m / np.sqrt(2))

    pictures(octet_m, pair_m)


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
QUOTE = ("Eight cords. Eight, and not one of them spare — every one carrying its part, "
         "and the whole thing meaning nothing if you pull a single one out.")
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")
SIZES = {"First": [1, 2], "Second": [1, 1, 2], "Third": [1, 1, 2], "Fourth": [1, 1, 2],
         "Fifth": [1, 1], "Sixth": [1, 1, 1]}


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


def unordered(conns):
    """A multiset of connections "a-b", each read either way round."""
    return sorted("-".join(sorted(c.split("-"))) for c in conns)


def check_faults():
    """Every one of the Lua's twenty tex.error sites, each raised by a call of its
    own and by nothing else."""
    cradle = P.parse_tangle("cradle")
    bad = [
        ("turn", P.ringcords, "", "green ring, turn a fifth"),
        ("count", P.ringcords, "", "green ring, seven cords"),
        ("ring-item", P.ringcords, "", "green ring, sideways"),
        ("no-cords", P.ringcords, "", "green ring, none in, none out"),
        ("two-turns", P.ringcords, "", "two green rings"),
        ("two-extra", P.ringcords, "", "two green rings, turns top and half, loose"),
        ("point-fork", P.ringcords, "", "red point, red fork"),
        ("half-turn", P.ringcords, "", "red half-turn, red ring"),
        ("half-turn-item", P.ringcords, "", "red half-turn, green ring, loose"),
        ("green-and-red", P.ringcords, "", "green and red, three cords"),
        ("reshaped", P.check_connections, cradle.cords[:-1]),          # a cord lost
        ("tangle", P.ringcords, "", "blue ring"),
        ("slid", P.ringcords, "slid", "green ring"),
        ("flipped", P.ringcords, "flipped", "three gates"),
        ("pushed", P.ringcords, "pushed", "gate"),
        ("pressed", P.ringcords, "pressed", "gate"),
        ("view", P.ringcords, "sideways", "gate"),
        ("frame-loose", P.ringcords, "loose", "cradle"),
        ("beam", P.ringcords, "beam", "gate"),
        ("koekpan-view", P.koekpan, "open"),
    ]
    for site, fn, *args in bad:
        assert P.faults(fn, *args) == [site], (site, P.faults(fn, *args))
    assert [b[0] for b in bad] == list(P.ERROR_SITES) and len(set(P.ERROR_SITES)) == 20
    # a re-joined cord is caught as well as a lost one; the reshape as drawn loses none
    moved = cradle.cords[:4] + (P.Cord(1, 2),) + cradle.cords[5:]
    assert P.faults(P.check_connections, moved) == ["reshaped"]
    assert P.faults(P.parse_tangle, "reshaped cradle") == []


def check_whole_dialogue():
    """All twenty pictures: their number and places, the keyed rule, the
    captions, the views each challenge may use, and no fault anywhere."""
    assert DIALOGUE.check_whole(ORDINALS) == 20                       # 3 + 4 + 4 + 4 + 2 + 3
    raw = DIALOGUE.read(DIALOGUE.raw)
    assert len(DIALOGUE.call_re.findall(raw)) == 20                   # dialog.raw.md, the same
    first = DIALOGUE.read(DIALOGUE.dialog).index("## First Challenge")
    assert not DIALOGUE.call_re.findall(DIALOGUE.read(DIALOGUE.dialog)[:first])   # none before it
    DIALOGUE.keyed_first(ORDINALS, kind=P.keyed_kind)
    placed = [(o, i, c) for o in ORDINALS
              for i, c in enumerate(c for b in DIALOGUE.blocks(o) for c in b)]
    assert [(o, i) for o, i, c in placed if c.keyed] == [
        ("First", 0), ("Second", 0), ("Third", 1), ("Fourth", 0)]      # P1.1, P2.1, P3.2, P4.1
    for o in ORDINALS:
        assert [len(b) for b in DIALOGUE.blocks(o)] == SIZES[o], o
        assert all(len(b) <= 4 for b in DIALOGUE.blocks(o)) and sum(SIZES[o]) <= 4, o
    for o, i, c in placed:
        P.picture(c)                                                   # the macro never faults
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS) and P.caption_ok(c), c.caption
        view = c.opt.lower()
        tangle = c.args[0].lower() if c.args else ""
        if c.macro == "koekpan":
            assert o == "Second", c.text
            continue
        assert (view != "slid" or o == "First") and (view != "flipped" or o == "Second")
        assert (view != "pushed" or o == "Third")
        assert view != "pressed" or o in ("Third", "Fourth", "Fifth", "Sixth"), c.text
        assert tangle not in ("greens", "greens and reds"), c.text
        assert tangle not in P.FRAME or o in ("Fourth", "Sixth"), c.text


def pictures(octet_m, pair_m):
    assert f'"{QUOTE}"' in __doc__                              # quoted byte for byte
    check_faults()
    check_whole_dialogue()
    p61, p62, p63 = DIALOGUE.check_layout("Sixth", sizes=[1, 1, 1])
    calls = (p61, p62, p63)
    assert [c.macro for c in calls] == ["ringcords"] * 3 and not any(c.keyed for c in calls)
    assert [(c.opt, c.args) for c in calls] == [("", ("cradle",)), ("", ("reshaped cradle",)),
                                              ("pressed", ("reshaped cradle",))]
    for c in (p61, p62):
        assert not P.quantity_words(c.caption), c.caption              # drawn copies of the frame
    b61, b62, b63 = (P.picture(c) for c in calls)
    sec = DIALOGUE.section("Sixth")
    paras = P.paragraphs(DIALOGUE, "Sixth")
    a1, a2, a3 = P.anchors(DIALOGUE, "Sixth")
    f1, f2, f3 = P.following(DIALOGUE, "Sixth")

    # P6.1: the cradle alone on the beam, just before the count; the count is its cords
    assert a1.endswith("and nothing else strung anywhere.]*")
    assert ("the two feed-cords in from the left, the four cords of the cross, "
            "the two draw-cords out to the right") in a1
    assert f1.startswith("**Achilles:** *[reaching up, touching each cord as he counts]*")
    assert QUOTE in f1                                                 # the docstring's line
    assert "One... two — the two coming in on the left." in f1
    assert "Three, four, five, six — the whole cross of them, every green to every red." in f1
    assert "Seven... eight, going out." in f1
    segs = P.segments(b61.drawn)
    feeds = [s for s in segs if isinstance(s[0], P.End)]
    cross = [s for s in segs if all(isinstance(e, int) for e in s)]
    draws = [s for s in segs if isinstance(s[1], P.End)]
    assert (len(feeds), len(cross), len(draws), len(segs)) == (2, 4, 2, 8)   # counted as spoken
    assert b61.setting == "beam" and b61.labels == ()
    assert np.allclose(tangle_map(b61.drawn), octet_m)

    # P6.2: the reshape -- every ring moved, every connection and hook kept
    assert a2.endswith("has changed.]*")
    assert "every cord still runs from the ring it left to the ring it reached" in a2
    assert f2.startswith("*[The snow-hush deepens")                    # the lit print, after
    glance = [i for i, q in enumerate(paras) if "Metamorphosis II" in q]
    assert len(glance) == 1 and paras[glance[0] - 1] == f2               # no block before the glance
    assert b62.setting == "beam" and b62.drawn.frame
    assert P.connections(b62.drawn.cords) == P.connections(b61.drawn.cords)
    assert [r.c for r in b62.drawn.rings] == [r.c for r in b61.drawn.rings]
    assert all((r.x, r.y) != (q.x, q.y) for r, q in zip(b62.drawn.rings, b61.drawn.rings))
    ends = lambda T: sorted({e for s in P.segments(T) for e in s if isinstance(e, P.End)}, key=str)
    assert ends(b62.drawn) == ends(b61.drawn)                          # the hooks where they were
    assert np.allclose(tangle_map(b62.drawn), octet_m)

    # P6.3: pressed from the crooked one -- the very pair the neat cradle gave (P4.2)
    (p42,) = [c for c in DIALOGUE.blocks("Fourth")[1]]
    b42 = P.picture(p42)
    assert "Off the second hooks comes the very same knot-pair the neat cradle gave." in a3
    assert b63.setting == "hooks" and b63.drawn.rings == b42.drawn.rings
    assert b63.drawn.cords == b42.drawn.cords
    assert np.allclose(tangle_map(b63.drawn), pair_m)
    assert np.allclose(tangle_map(b62.drawn), tangle_map(b63.drawn) / np.sqrt(2))
    assert f3.startswith("*[Beyond the pane")
    last = max(i for i, q in enumerate(paras) if "\\ringcords" in q)
    assert paras[last].count("\\ringcords") == 1 and p63.text in paras[last]   # the section's last call
    assert "The same little pair, out of that mare's nest. It didn't matter how you drew it." in sec


# --- pictures: all spiders phase-free (the bialgebra rule needs it) ---
def octet():
    """The Octet: two greens, two reds, a complete cross (K_2,2) between,
    two input cords and two output cords.  Eight edges = eight wires."""
    g = zx.Graph()
    i1 = g.add_vertex(VT.BOUNDARY, 0, 0); i2 = g.add_vertex(VT.BOUNDARY, 1, 0)
    gt = g.add_vertex(VT.Z, 0, 1); gb = g.add_vertex(VT.Z, 1, 1)
    rt = g.add_vertex(VT.X, 0, 2); rb = g.add_vertex(VT.X, 1, 2)
    o1 = g.add_vertex(VT.BOUNDARY, 0, 3); o2 = g.add_vertex(VT.BOUNDARY, 1, 3)
    g.add_edge((i1, gt)); g.add_edge((i2, gb))         # (1) top feed, (2) bottom feed
    g.add_edge((gt, rt)); g.add_edge((gt, rb))         # (3),(4) top green to both reds
    g.add_edge((gb, rt)); g.add_edge((gb, rb))         # (5),(6) bottom green to both reds
    g.add_edge((rt, o1)); g.add_edge((rb, o2))         # (7) top draw, (8) bottom draw
    g.set_inputs((i1, i2)); g.set_outputs((o1, o2))
    return g


def pair():
    """A single green-red pair: both inputs into one RED spider, one cord to one
    GREEN spider, both outputs from the green.  (= Complementarity.py's bialgebra RHS.)"""
    g = zx.Graph()
    i1 = g.add_vertex(VT.BOUNDARY, 0, 0); i2 = g.add_vertex(VT.BOUNDARY, 1, 0)
    r = g.add_vertex(VT.X, 0, 1); z = g.add_vertex(VT.Z, 0, 2)
    o1 = g.add_vertex(VT.BOUNDARY, 0, 3); o2 = g.add_vertex(VT.BOUNDARY, 1, 3)
    g.add_edge((i1, r)); g.add_edge((i2, r)); g.add_edge((r, z))
    g.add_edge((z, o1)); g.add_edge((z, o2))
    g.set_inputs((i1, i2)); g.set_outputs((o1, o2))
    return g


def octet_deformed():
    """The SAME Octet, drawn differently: spiders placed at other coordinates and
    the cords added in a different order.  Same connectivity, so it must have the
    same tensor -- only topology matters."""
    g = zx.Graph()
    # the four boundary endpoints keep their top/bottom identity (that IS which wire is
    # which); the interior rings are placed at scrambled coordinates and the cords are
    # strung in a different order -- a genuine redrawing of the same connectivity
    rb = g.add_vertex(VT.X, 5, 9); gt = g.add_vertex(VT.Z, 2, 1)
    o2 = g.add_vertex(VT.BOUNDARY, 1, 4); i1 = g.add_vertex(VT.BOUNDARY, 0, 0)
    gb = g.add_vertex(VT.Z, 3, 8); rt = g.add_vertex(VT.X, 6, 0)
    o1 = g.add_vertex(VT.BOUNDARY, 0, 4); i2 = g.add_vertex(VT.BOUNDARY, 1, 0)
    g.add_edge((gb, rb)); g.add_edge((gt, rt)); g.add_edge((i2, gb))
    g.add_edge((rt, o1)); g.add_edge((gt, rb)); g.add_edge((i1, gt))
    g.add_edge((gb, rt)); g.add_edge((rb, o2))
    g.set_inputs((i1, i2)); g.set_outputs((o1, o2))
    return g


def sketchbook():
    # the wire count, checked in code: eight edges = eight wires
    o = octet()
    assert o.num_edges() == 8
    # each interior ring has degree 3 (one boundary cord + two cross cords): the K_2,2
    assert all(len(list(o.neighbors(v))) == 3
               for v in o.vertices() if o.type(v) in (VT.Z, VT.X))

    # the master equality: Octet = single green-red pair, UP TO sqrt2
    assert same_picture_up_to_scalar(o, pair())
    assert np.allclose(matrix_of(o), matrix_of(pair()) / np.sqrt(2))   # cradle = pair / sqrt2

    # DERIVE it with the NAMED bialgebra rule, in the clean unfold direction pair -> cradle.
    # Locate the two interior spiders by colour (the rule renumbers vertices, so we must
    # not trust saved handles).
    g = pair()
    v_red = [v for v in g.vertices() if g.type(v) == VT.X][0]
    v_green = [v for v in g.vertices() if g.type(v) == VT.Z][0]
    assert simp.check_bialgebra(g, v_red, v_green)
    assert simp.bialgebra(g, v_red, v_green)                           # unfold
    assert g.num_edges() == 8                                          # the count is PRODUCED
    assert same_picture_up_to_scalar(g, octet())                       # and it is the Octet

    # ONLY TOPOLOGY MATTERS: the same connectivity drawn differently is the same tensor,
    # scalar included (a genuine deformation, not a rewrite -- no rule was applied)
    assert same_picture(octet_deformed(), octet())

    frame_is_the_octet()


# --------------------------------------------------------------------------
# The frame on the beam is the Octet.  The transcription's cradle, strung as a
# PyZX graph ring by ring and cord by cord, is octet() -- the same eight edges
# under the wire map of the identity card (left hook top <-> i_top, top green
# <-> g_top, ...), and the same picture, exactly; the reshaped cradle too; and
# the pressed pair is pair().  (A structural tie, not a second proof.)
# --------------------------------------------------------------------------
def graph_of(T):
    """A tangle as a PyZX graph: a green ring a Z spider, a red ring an X spider,
    its turn t the phase t/12 (in units of pi); every cord segment one edge; the
    hooks on the left the inputs, on the right the outputs, the upper line first."""
    g = zx.Graph()
    v = {}
    segs = P.segments(T)
    hooks = sorted({e for s in segs for e in s if isinstance(e, P.End)}, key=lambda e: -e.y)
    for e in hooks:
        v[e] = g.add_vertex(VT.BOUNDARY, 0 if e.y > 0 else 1, 0 if e.side == "L" else 3)
    for n, r in enumerate(T.rings, 1):
        v[n] = g.add_vertex(VT.Z if r.c == "green" else VT.X, 0 if r.y > 0 else 1,
                            1 if r.c == "green" else 2, phase=Fraction(r.t, 12))
    for a, b in segs:
        g.add_edge((v[a], v[b]))
    assert g.num_edges() == len(segs)                    # no strand collapsed
    g.set_inputs(tuple(v[e] for e in hooks if e.side == "L"))
    g.set_outputs(tuple(v[e] for e in hooks if e.side == "R"))
    return g


def roles(g):
    """octet()'s edges as the transcription names their ends: L1/L2 the input
    hooks, 1/2 the top/bottom green, 3/4 the top/bottom red, R1/R2 the outputs."""
    ring = {(VT.Z, 0): "1", (VT.Z, 1): "2", (VT.X, 0): "3", (VT.X, 1): "4"}

    def role(u):
        q = int(g.qubit(u))
        if u in g.inputs():
            return f"L{q + 1}"
        if u in g.outputs():
            return f"R{q + 1}"
        return ring[(g.type(u), q)]
    return unordered(f"{role(a)}-{role(b)}" for a, b in (g.edge_st(e) for e in g.edges()))


def frame_is_the_octet():
    cradle = P.parse_tangle("cradle")
    reshaped = P.parse_tangle("reshaped cradle")
    o = octet()
    assert roles(o) == unordered(P.connections(cradle.cords))          # the eight edges, by name
    assert unordered(P.connections(reshaped.cords)) == roles(o)
    for T in (cradle, reshaped):
        g = graph_of(T)
        assert g.num_edges() == 8 == len(P.CRADLE_CORDS)                # eight cords, eight wires
        assert same_picture(g, o) and same_picture(g, octet_deformed())
    assert same_picture(graph_of(P.view_pressed(reshaped)), pair())
    assert same_picture(graph_of(P.view_pressed(cradle)), pair())
    # a turned ring: the graph's phase (t/12, in units of pi) is the NumPy turn 2 pi t/24
    for spec in ("green ring, turn a third", "red ring, five cords, turn seven twelfths"):
        T = P.parse_tangle(spec)
        assert np.allclose(matrix_of(graph_of(T)), tangle_map(T))
    assert not np.allclose(matrix_of(graph_of(P.parse_tangle("green ring, turn a third"))),
                           tangle_map(P.parse_tangle("green ring, turn a quarter")))
    # the frame before the cross is not the Octet
    assert graph_of(P.parse_tangle("greens and reds")).num_edges() == 4


if __name__ == "__main__":
    verdict("Octet", blackboard, sketchbook)
