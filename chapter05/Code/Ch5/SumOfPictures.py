"""Challenge 2: A Sum of Pictures
================================
"Where did the other road go — can you draw the both-at-once?"  -- Achilles

Claim: a box on a state is a SUM of whole one-road pictures, one per output
value, each with an unbroken wire and a number in front:
    A|x> = sum_y A_yx |y>,   BA|x> = sum_{y,z} B_zy A_yx |z> = the roads.
No wire is ever forked; "both at once" is paid for in + signs, not new wires.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import I2, Picture, ket, matrix_of, random_matrix, same_picture, verdict
from dfp_props import Dialogue, words_only
from pyzx.utils import EdgeType
import props_ch05 as P

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def blackboard():
    rng = np.random.default_rng(2)
    A = random_matrix(rng)                       # generic complex box: so a
    B = random_matrix(rng)                       # transpose/dagger slip is caught
    assert not np.allclose(A, A.T)               # non-symmetric
    assert not np.allclose(A, A.conj())          # non-real

    # A|x> is the weighted list of output values -- the column x of A, entry by
    # entry, each entry the amplitude of one road from x to y.
    for x in (0, 1):
        road_sum = sum(A[y, x] * ket(y) for y in (0, 1))
        assert np.allclose(A @ ket(x), road_sum)
        # the road from x to y carries A_yx, NOT A_xy: a transpose would swap them
        assert not np.allclose(A @ ket(x), sum(A[x, y] * ket(y) for y in (0, 1)))

    # two boxes in a row = a sum over the MIDDLE value y = the roads (the sum
    # over paths).  BA|x> = sum_{y,z} B_zy A_yx |z>.
    for x in (0, 1):
        roads = sum(B[z, y] * A[y, x] * ket(z) for y in (0, 1) for z in (0, 1))
        assert np.allclose((B @ A) @ ket(x), roads)
        assert np.allclose((B @ A) @ ket(x), sum((B @ A)[z, x] * ket(z) for z in (0, 1)))
        assert not np.allclose((B @ A) @ ket(x), (A @ B) @ ket(x))   # order of the road matters

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the Tortoise's two one-road tiles with
#   \roadtiles[view]{low face, high face}{caption}     (pdf/props/ch05.tex)
# and, in the First Challenge, the conduit-board; tools/props_ch05.py
# transcribes what the Lua computes.  The calls are read out of dialog.md
# (tools/dfp_props.py), never copied.
#
# The state model (identity card, section 3): a tile's face is the sign of
# its road, silver = +1, slate = -1; `read` shows the net of the two faces,
# the two roads of H|0> carried to the home pocket by the second spreader:
#   net = s_low + s_high = 2 <0| H diag(s_low, s_high) H |0>.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("conduitboard", "roadtiles"))
NOT_IN_CAPTIONS = ("wire", "wires", "diagram", "diagrams", "drawing", "drawings", "picture",
                   "pictures", "box", "boxes", "dot", "dots", "node", "nodes", "lanes",
                   "strings", "five")
PAIRS = [(a, b) for a in P.FACES for b in P.FACES]


def tile_net(t):
    """The net of the tiles a picture draws (a bare pocket is 0)."""
    return sum(P.SIGN[f] for f in t.drawn)


def home_net(board):
    heap = board.heaps["willow"]["home"]
    return 0 if heap is None else P.SIGN[heap[0]] * heap[1]


def check_transcription():
    for low, high in PAIRS:
        s = np.array([P.SIGN[low], P.SIGN[high]])
        t = P.roadtiles("read", f"{low}, {high}")
        assert tile_net(t) == s.sum()                                   # s_low + s_high
        assert np.isclose(tile_net(t), 2 * (ket(0).conj().T @ H @ np.diag(s) @ H @ ket(0)).item())
        assert t.drawn == ((low, high) if low == high else ())         # a stack, or bare
        laid = P.roadtiles("laid", f"{low}, {high}")
        assert laid.where == "pocket" and laid.drawn == (low, high)   # the high tile on top
        assert P.roadtiles("apart", f"{low}, {high}").where == "table"
    # row 33: the tiles of H|0> are silver, silver; of H|1>, silver, slate
    for x, faces in ((0, ("silver", "silver")), (1, ("silver", "slate"))):
        assert np.allclose(np.sqrt(2) * (H @ ket(x)).ravel(), [P.SIGN[f] for f in faces])
    # the macro's own rules
    assert P.faults(P.roadtiles, "read", "silver") != []                 # two faces
    assert P.faults(P.roadtiles, "read", "silver, slate, silver") != []
    assert P.faults(P.roadtiles, "read", "silver, gold") != []
    assert P.faults(P.roadtiles, "stacked", "silver, slate") != []


def pictures():
    check_transcription()
    p21, p22, p23, p24 = DIALOGUE.check_layout("Second", sizes=[1, 3])
    for c in (p21, p22, p23, p24):
        assert c.macro == "roadtiles"
        assert words_only(c.caption, forbidden=NOT_IN_CAPTIONS), c.caption
    t21, t22, t23, t24 = (P.picture(c) for c in (p21, p22, p23, p24))

    a1, a2 = P.anchors(DIALOGUE, "Second")
    assert "blank but for its face" in a1
    assert "silver over silver" in a2 and "silver beneath slate" in a2

    # P2.1: two blank tiles on the table, both silver, keyed low road / high road
    assert p21.keyed and t21.view == "apart" and t21.where == "table"
    assert t21.faces == ("silver", "silver") and t21.labels == ("low road", "high road")

    # P2.2: silver over silver, read -- a doubled stack, net +2
    assert not p22.keyed and t22.view == "read" and t22.faces == ("silver", "silver")
    assert t22.drawn == ("silver", "silver") and tile_net(t22) == 2

    # P2.3: laid in the home pocket, the high tile (slate) over the low (silver)
    assert t23.view == "laid" and t23.where == "pocket"
    assert t23.drawn == ("silver", "slate")                  # drawn bottom first: silver beneath

    # P2.4: silver and slate, read -- lifted out, the pocket bare, net 0
    assert t24.view == "read" and t24.faces == ("silver", "slate")
    assert t24.drawn == () and tile_net(t24) == 0
    assert t23.faces == t24.faces                            # the pair that was laid is the pair read

    # nothing walked, the same pockets as the Cat's walks (P1.2's home, P1.4's home)
    first = [c for b in DIALOGUE.blocks("First") for c in b]
    walk1, walk2 = P.picture(first[1]), P.picture(first[3])
    assert walk1.view == walk2.view == "end" and walk1.pegs == () and walk2.pegs == ("willow",)
    assert tile_net(t22) == home_net(walk1) == 2
    assert tile_net(t24) == home_net(walk2) == 0
    # and the tiles' faces are the faces the roads carry (P1.3, just past the peg)
    assert tuple(f for _, f, _ in P.picture(first[2]).walking) == t24.faces


# A generic one-wire box for the sketchbook: a chain of three phase nodes.
# PyZX draws this with the spiders you meet in Chapter 8.
BOX_A = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 5))
BOX_B = (Fraction(2, 5), Fraction(1, 6), Fraction(4, 7))


def box(p, row, col0, phases):
    a, b, c = phases
    ns = [p.z(row, col0, a), p.x(row, col0 + 1, b), p.z(row, col0 + 2, c)]
    return ns


def box_alone(phases):
    p = Picture()
    ns = box(p, 0, 1, phases)
    p.wire(p.inp(0), *ns, p.out(0))
    return p.done()


def two_in_a_row(pa, pb):
    """Box A then box B on one wire: A's input side first, then B."""
    p = Picture()
    na = box(p, 0, 1, pa)
    nb = box(p, 0, 4, pb)
    p.wire(p.inp(0), *na, *nb, p.out(0))
    return p.done()


def hedge(p, a, b):
    p.g.add_edge((a, b), EdgeType.HADAMARD)


def h_on(bit):
    """The state |bit> sent through the box H, drawn as one picture."""
    p = Picture()
    st = p.x(0, 0, bit)               # |0> = X(0)/sqrt2, |1> = X(pi)/sqrt2
    o = p.out(0)
    hedge(p, st, o)
    g = p.done()
    g.scalar.add_power(-1)            # the 1/sqrt2 of the state
    return g


def basis(bit):
    p = Picture()
    st = p.x(0, 0, bit)
    p.wire(st, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def superposed(phase):
    """|+> = (|0>+|1>)/sqrt2 for phase 0, |-> = (|0>-|1>)/sqrt2 for phase 1,
    drawn as one picture."""
    p = Picture()
    st = p.z(0, 0, phase)            # Z(0) -> |0>+|1>, Z(pi) -> |0>-|1>
    p.wire(st, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def sketchbook():
    # --- the road-sum, as a translation check (no Stage-A rewrite derives it) ---
    # two boxes on a wire denote the matrix product = the sum over the middle
    # value.  A generic complex box has no picture-shorthand for its product, so
    # we confirm the picture evaluates to the road-sum matrix and stop there.
    Amat = matrix_of(box_alone(BOX_A))
    Bmat = matrix_of(box_alone(BOX_B))
    assert not np.allclose(Amat, Amat.T) and not np.allclose(Amat, Amat.conj())  # generic
    P = two_in_a_row(BOX_A, BOX_B)
    assert np.allclose(matrix_of(P), Bmat @ Amat)          # composition IS the road-sum
    assert not np.allclose(matrix_of(P), Amat @ Bmat)      # order of the road matters
    assert not np.allclose(matrix_of(P), (Bmat @ Amat).T)  # not the transpose

    # --- the sum of pictures itself (row 33), as two-graph picture equalities ---
    # H|0> = a |0>-road plus a |1>-road, BOTH signed +  (= |+>)
    assert same_picture(h_on(0), superposed(0))            # exact, as pictures
    # H|1> = the |0>-road plus the SIGNED |1>-road  (= |->)
    assert same_picture(h_on(1), superposed(1))
    assert not same_picture(h_on(1), superposed(0))        # the |1>-road really is signed minus
    # and the literal "a picture equals a sum of whole pictures": the one-road
    # tiles |0> and |1> summed, each with its number in front.
    k0, k1 = matrix_of(basis(0)), matrix_of(basis(1))
    assert np.allclose(matrix_of(h_on(0)), (1 / np.sqrt(2)) * (k0 + k1))
    assert np.allclose(matrix_of(h_on(1)), (1 / np.sqrt(2)) * (k0 - k1))


if __name__ == "__main__":
    verdict("SumOfPictures", blackboard, sketchbook)
