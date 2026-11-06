"""Challenge II: The Other Colour (colour change)
=================================================
"A red ring and a green ring see different things."  -- Tortoise

Claim: a Hadamard on EVERY leg of a Z spider turns it into an X spider of the
same phase -- exactly:  H^(x)k Z(m,k,a) H^(x)m = X(m,k,a).  Because the red
spider is built intrinsically from |+>,|->, this is a theorem, not a definition.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import string
from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import ket, kron, same_picture, same_picture_up_to_scalar, verdict
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


def blackboard():
    pin_spiders()

    # colour change is EXACT for every phase -- the red spider was built from |+-|,
    # so this equality is a genuine fact, proved, not assumed
    for alpha in [0.0, 0.7, 1.3, 2.9]:
        assert np.allclose(H @ zspider(1, 1, alpha) @ H, xspider(1, 1, alpha))
    # multi-leg: a Hadamard on each of the three legs of a 1->2 green makes a 1->2 red
    for alpha in [0.7, 1.3]:
        assert np.allclose(kron(H, H) @ zspider(1, 2, alpha) @ H, xspider(1, 2, alpha))

    # genericity: green and red differ (they must, or the theorem would be empty) --
    # this needs a NON-zero phase, since Z(1,1,0) = X(1,1,0) = I
    assert not np.allclose(zspider(1, 1, 1.3), xspider(1, 1, 1.3))
    # a Hadamard on only ONE leg does NOT change the colour
    assert not np.allclose(H @ zspider(1, 1, 1.3), xspider(1, 1, 1.3))

    # conventions: the intrinsic red spider is |+..+><+..+| + e^{ib}|-..-><-..-|
    assert np.allclose(xspider(1, 1, 0.7),
                       np.outer(KET_PLUS, KET_PLUS.conj())
                       + np.exp(1j * 0.7) * np.outer(KET_MINUS, KET_MINUS.conj()))
    # and the red one-leg state is sqrt2|0>
    assert np.allclose(xspider(0, 1, 0).ravel(), np.sqrt(2) * ket(0))

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
QUOTE = "A red ring and a green ring see different things."
I2 = np.eye(2, dtype=complex)


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


def check_transcription():
    """The Lua's flipped view and the flip-slot against colour change (a
    structural check, not a second proof)."""
    # flipped <-> colour change, every turn, on the two shapes the Second draws: the ring
    # at the end of a cord through the slot (one cord in) and the four-cord ring (two in,
    # two out).  Every leg dressed in a flip is the other colour, the same turn, EXACTLY.
    for c in ("green", "red"):
        for t in range(P.WHOLE):
            for nin, nout in ((1, 0), (2, 2)):
                T = P.single(c, t, nin, nout)
                F = P.view_flipped(T)
                (r,) = F.rings
                assert r.c == P.OTHER[c] and r.t == t and P.segments(F) == P.segments(T)
                Hin, Hout = kron(*[H] * nin), (kron(*[H] * nout) if nout else np.eye(1))
                assert np.allclose(Hout @ tangle_map(T) @ Hin, tangle_map(F))
            # a flip on one leg alone is NOT the other colour (at any turn)
            T22 = tangle_map(P.single(c, t, 2, 2))
            assert not np.allclose(kron(H, I2) @ T22, tangle_map(P.view_flipped(P.single(c, t, 2, 2))))
    assert np.allclose(tangle_map(P.single("green", 8, 2, 2)), zspider(2, 2, phase(8)))
    assert np.allclose(tangle_map(P.single("red", 8, 2, 2)), xspider(2, 2, phase(8)))
    # the close-up of the slot: the ring comes clear in the other colour
    assert P.koekpan("slot green").ring == "red" and P.koekpan("slot red").ring == "green"
    assert P.koekpan("").view == "shut" and P.koekpan("").ring is None
    # the macro's refusals on this challenge's props
    assert P.faults(P.ringcords, "flipped", "two green rings, turns top and top") == ["flipped"]
    assert P.faults(P.ringcords, "flipped", "green ring, four cords") == []
    assert P.faults(P.koekpan, "open") == ["koekpan-view"]
    assert P.faults(P.koekpan, "slot blue") == ["koekpan-view"]


def pictures():
    assert f'"{QUOTE}"' in __doc__                              # quoted byte for byte
    check_transcription()
    p21, p22, p23, p24 = DIALOGUE.check_layout("Second", sizes=[1, 1, 2])
    calls = (p21, p22, p23, p24)
    assert [c.macro for c in calls] == ["koekpan", "koekpan", "ringcords", "ringcords"]
    for c in calls:
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS) and P.caption_ok(c), c.caption
    assert [c.keyed for c in calls] == [True, False, False, False]     # the pan keyed once
    assert [c.opt for c in calls] == ["", "slot green", "", "flipped"]
    assert p21.args == p22.args == ()                                    # the pan: a caption only
    b21, b22, b23, b24 = (P.picture(c) for c in calls)
    sec = DIALOGUE.section("Second")
    a1, a2, a3 = P.anchors(DIALOGUE, "Second")
    f1, f2, f3 = P.following(DIALOGUE, "Second")

    # P2.1: the pan shut, its parts named in the speech it follows
    assert b21.view == "shut" and b21.labels == ("rim", "flip-slot", "handle", "second hooks")
    assert ("a griddle-like press with a single slot in its rim and, standing beside it, "
            "a little wooden rack with two empty hooks up each post") in a1
    assert "this **flip-slot**" in a1 and "this second set of hooks" in a1
    assert a1.endswith("and leaves the first cords be.")
    assert f1.startswith("*[It is a homely thing")                   # the ambient after the block

    # P2.2: the close-up: the cord of a GREEN ring through the slot, the ring clear in RED,
    # the same cord and the same turn
    assert "He takes a loose cord feeding a green ring and passes the cord through the flip-slot." in sec
    assert b22.view == "slot green" and b22.ring == "red" and b22.labels == ()
    assert "is red where it was green" in a2 and "same cord, same turn" in a2

    # P2.3, P2.4: the four-cord ring, before and after every cord went through
    assert p23.args == p24.args
    assert "takes a green ring with four cords" in sec
    assert "The green ring hangs with its four cords." in a3
    assert "The ring's four cords still run where they ran; only green has become red." in a3
    (g,), (r,) = b23.drawn.rings, b24.drawn.rings
    assert b23.setting == b24.setting == "loose"
    ends = [e for s in P.segments(b23.drawn) for e in s if isinstance(e, P.End)]
    assert (sum(e.side == "L" for e in ends), sum(e.side == "R" for e in ends)) == (2, 2)
    assert (g.c, r.c) == ("green", "red") and g.t == r.t == 0
    assert P.segments(b23.drawn) == P.segments(b24.drawn)             # the same cords
    assert np.allclose(kron(H, H) @ tangle_map(b23.drawn) @ kron(H, H), tangle_map(b24.drawn))
    assert np.allclose(tangle_map(b24.drawn), xspider(2, 2, 0))
    assert f3.startswith("**Crab:** *[handing the flipped-round ring to Achilles]*")
    assert QUOTE in sec                                                # the docstring's line


# --- pictures: phases below are Fractions in units of pi ---
def green_all_H(a):
    """A green spider with a Hadamard-EDGE on every leg (in and out)."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    z = g.add_vertex(VT.Z, 0, 2, phase=Fraction(a))
    o = g.add_vertex(VT.BOUNDARY, 0, 4)
    # a Hadamard-edge is 'a plain wire with a yellow box on it'; it carries the H scalar
    g.add_edge((i, z), edgetype=ET.HADAMARD)
    g.add_edge((z, o), edgetype=ET.HADAMARD)
    g.set_inputs((i,)); g.set_outputs((o,))
    return g


def plain_green(a):
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    z = g.add_vertex(VT.Z, 0, 1, phase=Fraction(a))
    o = g.add_vertex(VT.BOUNDARY, 0, 2)
    g.add_edge((i, z)); g.add_edge((z, o))
    g.set_inputs((i,)); g.set_outputs((o,))
    return g


def green_one_H(a):
    """Only the input leg dressed in a Hadamard-edge -- not enough to change colour."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    z = g.add_vertex(VT.Z, 0, 2, phase=Fraction(a))
    o = g.add_vertex(VT.BOUNDARY, 0, 4)
    g.add_edge((i, z), edgetype=ET.HADAMARD)
    g.add_edge((z, o))
    g.set_inputs((i,)); g.set_outputs((o,))
    return g


def red(a):
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    x = g.add_vertex(VT.X, 0, 1, phase=Fraction(a))
    o = g.add_vertex(VT.BOUNDARY, 0, 2)
    g.add_edge((i, x)); g.add_edge((x, o))
    g.set_inputs((i,)); g.set_outputs((o,))
    return g


def sketchbook():
    a = Fraction(1, 3)
    # colour change is EXACT (scalar included): the Hadamard-edge carries the H scalar
    assert same_picture(green_all_H(a), red(a))
    assert not same_picture(plain_green(a), red(a))            # a bare green is not red (a != 0)
    assert not same_picture(green_one_H(a), red(a))            # one leg is not enough

    # derive with the NAMED colour-change rule (not full_reduce: that starts with to_gh,
    # which IS the colour-change move, so it would read circular)
    g = green_all_H(a)
    zv = [v for v in g.vertices() if g.type(v) == VT.Z][0]
    assert simp.check_color_change(g, zv)
    simp.color_change(g, zv)
    assert g.type(zv) == VT.X                                  # the dot is now red
    assert all(g.edge_type(e) == ET.SIMPLE for e in g.edges())  # the Hadamards are gone
    # the rule drops the global scalar, so compare up to scalar
    assert same_picture_up_to_scalar(g, red(a))


if __name__ == "__main__":
    verdict("ColourChange", blackboard, sketchbook)
