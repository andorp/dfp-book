"""Challenge 1: Two Roads
=======================
"Do the amplitudes add — and can they cancel?"  -- Achilles

Claim: two mirrors undo each other, HH = I (the two roads to |0> pile up, the
walker comes home); a half-turn of phase between them is a flip, HZH = X (the
roads to |0> cancel, the walker is turned away).  Both exact, scalar included.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
from dfp import I2, Picture, ket, matrix_of, same_picture, verdict
from dfp_props import Dialogue, words_only
from pyzx.utils import EdgeType
import props_ch05 as P

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def blackboard():
    # two mirrors undo each other: no phase => the walker comes home
    assert np.allclose(H @ H, I2)                       # HH = I, exact, no stray scalar
    assert not np.allclose(H @ H, X)                    # not a flip
    assert np.allclose(H.conj().T, H) and np.allclose(H @ H, I2)   # H is self-inverse

    # a half-turn of phase between the mirrors is a flip
    assert np.allclose(H @ Z @ H, X)                    # HZH = X, exact
    assert not np.allclose(H @ Z @ H, I2)               # not the identity
    assert not np.allclose(H @ Z @ H, Z)               # the phase did not survive as itself

    # read the two arrival amplitudes on |0>.  With no phase the two roads to
    # |0> have amplitude +1/2 each and PILE UP to 1; the roads to |1> are
    # +1/2 and -1/2 and CANCEL to 0 -- the walker comes home.
    home = H @ H @ ket(0)
    assert np.allclose(home, ket(0))                    # amplitude of |0> = 1, of |1> = 0
    # a half-turn between the mirrors swaps the fortunes: |0> cancels, |1> fills
    away = H @ Z @ H @ ket(0)
    assert np.allclose(away, ket(1))                    # turned away to |1>

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the Tortoise's conduit-board with
#   \conduitboard[view]{setting}{caption}     (pdf/props/ch05.tex)
# The Lua COMPUTES the roads, the tokens' faces, the deal and the settling;
# tools/props_ch05.py transcribes it.  The calls are read out of this
# challenge's section of dialog.md (tools/dfp_props.py), never copied, so an
# edit to the dialogue that breaks a picture breaks here.
#
# The state model (identity card, section 3): a road token's face is its
# road's sign, silver = +1, slate = -1; the pocket it lies in is its value,
# home = <0|, away = <1|; each spreader is H.  A settled pocket's net (silver
# count minus slate count) is the amplitude times sqrt2 per spreader walked
# through: a one-lane walk with both spreaders, net = 2 x amplitude.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("conduitboard", "roadtiles"))
NOT_IN_CAPTIONS = ("wire", "wires", "diagram", "diagrams", "drawing", "drawings", "picture",
                   "pictures", "box", "boxes", "dot", "dots", "node", "nodes", "lanes",
                   "strings", "five")


def heap_net(heap):
    """A drawn heap (face, how many) as a signed count; a bare cup is 0."""
    return 0 if heap is None else P.SIGN[heap[0]] * heap[1]


def one_lane_model(M):
    """What a one-lane walk with both spreaders and M between them must show:
    the tokens just past M (sqrt2 x M H|0>, one per road) and the two settled
    pockets (2 x <y| H M H |0>)."""
    past = np.sqrt(2) * (M @ H @ ket(0)).ravel()
    arrive = 2 * (H @ M @ H @ ket(0)).ravel()
    return past, arrive


def check_transcription():
    """The Lua's walk, deal and settling against the blackboard: the peg is Z
    on the lane, the foot spreader's deal is the second H."""
    for peg, M in ((False, I2), (True, Z)):
        setting = "willow peg" if peg else "willow"
        past, arrive = one_lane_model(M)
        b = P.conduitboard("past", setting)
        assert [v for v, _, _ in b.walking] == [0, 1]                  # low road, then high
        assert np.allclose([P.SIGN[f] for _, f, _ in b.walking], past)
        assert [ahead for _, _, ahead in b.walking] == [False, True]    # the high road ahead
        assert all(f == "silver" for _, f, _ in P.conduitboard("before", setting).walking)
        e = P.conduitboard("end", setting).heaps["willow"]
        assert np.allclose([heap_net(e["home"]), heap_net(e["away"])], arrive)
        for view in ("pockets", "pockets settled"):
            c = P.conduitboard(view, setting)
            assert np.allclose([P.net(c.cups["home"]), P.net(c.cups["away"])], arrive)
        c = P.conduitboard("pockets settled", setting)
        assert [sum(P.SIGN[f] * k for f, k in c.heaps[p]) for p in ("home", "away")] \
            == [heap_net(e["home"]), heap_net(e["away"])]
    # the macro's own rules: a peg never shares the board with the bar; no
    # arrival with a foot spreader lifted; past on one lane only
    assert P.faults(P.conduitboard, "set", "willow peg; reed two-faced; bar never") != []
    assert P.faults(P.conduitboard, "end", "willow no foot") != []
    assert P.faults(P.conduitboard, "past", "willow; alder") != []
    assert P.faults(P.conduitboard, "set", "willow; willow") != []
    assert P.faults(P.conduitboard, "sideways", "willow") != []


def pictures():
    check_transcription()
    p11, p12, p13, p14 = DIALOGUE.check_layout("First", sizes=[1, 1, 2])
    for c in (p11, p12, p13, p14):
        assert c.macro == "conduitboard"
        assert words_only(c.caption, forbidden=NOT_IN_CAPTIONS), c.caption
        assert "sixteen roads" not in c.caption.lower()
    b11, b12, b13, b14 = (P.picture(c) for c in (p11, p12, p13, p14))
    for b in (b11, b12, b13, b14):
        assert isinstance(b, P.Board) and b.grooves == P.GROOVES        # the whole board
        assert b.live == ("willow",) and b.spreaders == {"willow": (True, True)}
        assert b.bar is None and b.reed is None                         # no bar, no reed token

    # anchors: each block sits after the speech or walk it shows
    a1, a2, a3 = P.anchors(DIALOGUE, "First")
    assert a1.endswith("without the channel ever splitting.")
    assert "Both come up silver, the bright face." in a2 and "peg" not in a2
    assert "as the high road passes the peg, its token turns over" in a3

    # P1.1: the board as set, keyed; the willow alone in play
    assert p11.keyed and b11.view == "set"
    assert b11.labels == ("willow", "alder", "hazel", "rowan", "reed", "home", "away")
    assert b11.pegs == () and b11.walking == () and b11.heaps == {}

    # P1.2: the first walk, no peg -- home heaped silver, away bare: 2 HH|0>
    assert not p12.keyed and b12.view == "end" and b12.pegs == ()
    _, arrive = one_lane_model(I2)
    assert b12.heaps["willow"] == {"home": ("silver", 2), "away": None}
    assert np.allclose([heap_net(b12.heaps["willow"][p]) for p in ("home", "away")], arrive)
    assert np.allclose(arrive, 2 * ket(0).ravel())                      # HH = I: home, net +2

    # P1.3: past the peg, the low road silver, the high road slate: Z on the roads
    assert b13.view == "past" and b13.pegs == ("willow",)
    assert b13.walking == ((0, "silver", False), (1, "slate", True))
    past, arrive = one_lane_model(Z)
    assert np.allclose([P.SIGN[f] for _, f, _ in b13.walking], past)
    assert np.allclose(past, np.diag(Z))                                 # +1 low, -1 high

    # P1.4: the same walk settled -- home bare, away heaped: 2 HZH|0> = 2|1>
    assert b14.view == "end" and b14.pegs == ("willow",)
    assert b14.heaps["willow"] == {"home": None, "away": ("silver", 2)}
    assert np.allclose([heap_net(b14.heaps["willow"][p]) for p in ("home", "away")], arrive)
    assert np.allclose(arrive, 2 * (X @ ket(0)).ravel())
    # Achilles's prediction (the chances only add) would have left home heaped
    assert heap_net(b14.heaps["willow"]["home"]) != heap_net(b12.heaps["willow"]["home"])


def hedge(p, a, b):
    """A Hadamard edge -- the box H, drawn the exact Stage-A way (see the card):
    NOT an Euler phase-node chain, which would carry a stray e^{i pi/4}."""
    p.g.add_edge((a, b), EdgeType.HADAMARD)


def two_mirrors(middle_phase):
    """One wire: H, then a Z-phase of `middle_phase` half-turns, then H."""
    p = Picture()
    i = p.inp(0)
    mid = p.z(0, 2, middle_phase)   # PyZX draws this with the spiders you meet in Chapter 8.
    o = p.out(0)
    hedge(p, i, mid)                # first mirror
    hedge(p, mid, o)               # second mirror
    return p.done()


def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def gate(kind, phase):
    p = Picture()
    node = p.z(0, 2, phase) if kind == "Z" else p.x(0, 2, phase)
    p.wire(p.inp(0), node, p.out(0))
    return p.done()


def sketchbook():
    # no phase between the mirrors: H-H is a bare wire, exactly
    assert np.allclose(matrix_of(two_mirrors(0)), I2)
    assert same_picture(two_mirrors(0), bare_wire())            # HH = I
    assert not same_picture(two_mirrors(0), gate("X", 1))       # not a flip

    # a half-turn (Z, one half-turn = phase pi) between the mirrors is a flip
    assert same_picture(two_mirrors(1), gate("X", 1))           # HZH = X
    assert not same_picture(two_mirrors(1), bare_wire())        # not the identity
    assert not same_picture(two_mirrors(1), gate("Z", 1))       # not a bare Z


if __name__ == "__main__":
    verdict("TwoRoads", blackboard, sketchbook)
