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
#   \roadtiles[view; opts]{low face, high face}{caption}     (pdf/props/ch05.tex)
# and, in the First Challenge, the conduit-board; tools/props_ch05.py
# transcribes what the Lua computes.  The calls are read out of dialog.md
# (tools/dfp_props.py), never copied.
#
# The state model (identity card, section 3): a tile's face is the sign of
# its road, silver = +1, slate = -1; `read` shows the net of the two faces,
# the two roads of H|0> carried to the home pocket by the second spreader:
#   net = s_low + s_high = 2 <0| H diag(s_low, s_high) H |0>.
# The tiles laid for the plain walk are sqrt2 H|0> (silver, silver); turning
# the high-road tile is Z on the high road: sqrt2 Z H|0> (silver, slate).
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("conduitboard", "conduitwalk", "roadtiles"))
NOT_IN_CAPTIONS = ("wire", "wires", "diagram", "diagrams", "drawing", "drawings", "picture",
                   "pictures", "box", "boxes", "dot", "dots", "node", "nodes", "lanes",
                   "strings", "five")
PAIRS = [(a, b) for a in P.FACES for b in P.FACES]
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PILE_PLACES = tuple((P.CX + dx, P.CY + dy, rot) for dx, dy, rot in P.PILE)
LAID_PLACES = tuple((P.CX + dx, P.CY + dy, rot) for dx, dy, rot in P.LAID)


def tile_net(t):
    """The net of the tiles a picture draws (a bare pocket is 0)."""
    return sum(P.SIGN[f] for f in t.drawn)


def home_net(board):
    heap = board.heaps["willow"]["home"]
    return 0 if heap is None else P.SIGN[heap[0]] * heap[1]


def faces(signs):
    assert np.allclose(np.abs(signs), 1) and np.allclose(np.imag(signs), 0)
    return tuple(P.face(float(np.real(s))) for s in signs)


def check_transcription():
    for low, high in PAIRS:
        s = np.array([P.SIGN[low], P.SIGN[high]])
        t = P.roadtiles("read", f"{low}, {high}")
        assert tile_net(t) == s.sum()                                   # s_low + s_high
        assert np.isclose(tile_net(t), 2 * (ket(0).conj().T @ H @ np.diag(s) @ H @ ket(0)).item())
        assert t.drawn == ((low, high) if low == high else ())         # a stack, or bare
        laid = P.roadtiles("laid", f"{low}, {high}")
        assert laid.where == "pocket" and laid.drawn == (low, high)   # the high tile on top
        assert laid.places == LAID_PLACES
        assert P.roadtiles("apart", f"{low}, {high}").where == "table"
        # a turn: that tile's sign changes (Z on its road), and it is laid back where it
        # lay -- at the pile when the faces before were alike, else where `laid` spreads them
        for v, which in ((0, "low"), (1, "high")):
            tt = P.clean(P.roadtiles, f"laid; turn={which}", f"{low}, {high}")
            flip = np.diag([-1 if u == v else 1 for u in (0, 1)])
            assert tt.before == (low, high) and tt.turned == which
            assert tt.faces == faces(flip @ s)
            assert tt.places == (PILE_PLACES if low == high else LAID_PLACES)
            assert tt.halo[:3] == tt.places[v] and tt.flip is not None
            (x0, y0), _, _, (x1, _) = tt.flip
            assert np.isclose((x0 + x1) / 2, tt.places[v][0]) and y0 > tt.places[v][1]
            assert [d.split(" at ")[1] for d in tt.dump if d.startswith("turned")] == \
                ["pile" if low == high else "laid"]
        # a read with marks lifts an unlike pair out of the pile
        if low != high:
            r = P.clean(P.roadtiles, "read; marks=on", f"{low}, {high}")
            assert r.ghosts == tuple((f,) + pl for f, pl in zip((low, high), PILE_PLACES))
            assert r.drawn == () and r.marked
    # row 33: the tiles of H|0> are silver, silver; of H|1>, silver, slate
    for x, fs in ((0, ("silver", "silver")), (1, ("silver", "slate"))):
        assert np.allclose(np.sqrt(2) * (H @ ket(x)).ravel(), [P.SIGN[f] for f in fs])
    # the macro's own rules
    assert P.faults(P.roadtiles, "read", "silver") != []                 # two faces
    assert P.faults(P.roadtiles, "read", "silver, slate, silver") != []
    assert P.faults(P.roadtiles, "read", "silver, gold") != []
    assert P.faults(P.roadtiles, "stacked", "silver, slate") != []
    assert P.faults(P.roadtiles, "read; turn=high", "silver, silver") == [
        "roadtiles: turn= turns a tile laid in the pocket: the laid view"]
    assert P.faults(P.roadtiles, "read; marks=on", "silver, silver") == [
        "roadtiles: marks=on: like faces lie together; nothing lifts out"]


def pictures():
    P.check_lua()                                  # the Lua and .tex the transcription was proved on
    check_transcription()
    found = P.check_section(DIALOGUE, "Second", sizes=[1, 3], frames=[1, 3],
                            forbidden=NOT_IN_CAPTIONS)
    (p21, [(t21, _)]), = found[0]
    (p22, [(t22, _)]), (p23, [(t23, _)]), (p24, [(t24, _)]) = found[1]
    for c in (p21, p22, p23, p24):
        assert c.macro == "roadtiles"

    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a1, a2 = DIALOGUE.anchors("Second", path)
        assert "blank but for its face" in a1
        assert "silver over silver" in a2 and "silver beneath slate" in a2
        assert "lifts the high-road tile, turns it to its slate face, and lays it back" in a2

    # P2.1: two blank tiles on the table, both silver, keyed low road / high road
    assert p21.keyed and t21.view == "apart" and t21.where == "table" and not t21.marked
    assert t21.faces == ("silver", "silver") and t21.labels == ("low road", "high road")

    # the model, moment by moment: laid for the plain walk, the high tile turned, read
    s = np.sqrt(2) * (H @ ket(0)).ravel()                    # (+1, +1): silver, silver
    turned = Z @ s                                           # (+1, -1): silver, slate

    # F10: silver over silver, read -- a doubled stack at the pile's places, net +2
    assert not p22.keyed and P.options(p22) == {"view": "read"} and not t22.marked
    assert t22.faces == faces(s) and t22.drawn == faces(s) and tile_net(t22) == 2
    assert np.isclose(tile_net(t22), 2 * (ket(0).conj().T @ H @ np.diag(s) @ H @ ket(0)).item())
    assert t22.places == PILE_PLACES

    # F11: the high tile lifted, turned to slate and laid back where it lay -- both
    # tiles still at the pile (the low one never moved): a halo and a flip on it
    assert P.options(p23) == {"view": "laid", "turn": "high"} and t23.marked
    assert t23.before == t22.faces                           # the faces before the turn: F10's
    assert t23.turned == "high" and t23.faces == faces(turned)
    assert t23.drawn == ("silver", "slate")                  # drawn bottom first: silver beneath
    assert t23.places == t22.places == PILE_PLACES           # laid back: nothing else moved
    assert t23.halo == t23.places[1] + (0.014,)              # its ghost, at its own place
    (x0, y0), _, _, (x1, _) = t23.flip
    assert np.isclose((x0 + x1) / 2, t23.places[1][0])        # over the turned tile
    assert y0 > t23.places[1][1] + P.TILE / 2
    assert t23.dump[0] == "turned high silver to slate at pile"

    # F12: silver and slate, read -- the pair lifts out from the pile, the pocket bare
    assert P.options(p24) == {"view": "read", "marks": "on"} and t24.marked
    assert t24.faces == t23.faces and t24.drawn == () and tile_net(t24) == 0
    assert t24.ghosts == tuple((f,) + pl for f, pl in zip(t23.faces, t23.places))  # where they lay
    assert np.isclose(tile_net(t24), 2 * (ket(0).conj().T @ H @ np.diag(turned) @ H @ ket(0)).item())
    # the tiles are turned once and lifted out once
    assert [t.turned for t in (t21, t22, t23, t24)] == [None, None, "high", None]
    assert [bool(t.ghosts) for t in (t21, t22, t23, t24)] == [False, False, False, True]

    # nothing walked, the same pockets as the Cat's walks (walk 1's end, the peg walk's)
    first = DIALOGUE.blocks("First")
    walk1 = [f for f, _ in P.pictures(first[1][0])]
    expects, walk2 = P.picture(first[2][0]), [f for f, _ in P.pictures(first[2][1])]
    assert walk1[-1].view == walk2[-1].view == "end" and expects.expects == "add"
    assert walk1[-1].pegs == () and walk2[-1].pegs == ("willow",)
    assert tile_net(t22) == home_net(walk1[-1]) == 2
    assert tile_net(t24) == home_net(walk2[-1]) == 0
    # and the tiles' faces are the faces the roads carry (just past the peg)
    assert tuple(t["face"] for t in walk2[0].tokens) == t23.faces == t24.faces
    # and the piles agree: two like faces pile in the tiles' pocket and in walk 1's
    assert walk1[1].cups["home"][:2] == (list(t22.faces), "piled")

    # the block replaced this round: the turn drawn on the faces before it
    P.check_edits(DIALOGUE, "Second", [
        ("\\roadtiles[laid]{silver, slate}{silver beneath slate}",
         "\\roadtiles[laid; turn=high]{silver, silver}{silver beneath slate}"),
        ("\\roadtiles[read]{silver, slate}{the pocket bare}",
         "\\roadtiles[read; marks=on]{silver, slate}{the pocket bare}")])

    assert P.section_dumps(found) == [
        "roadtiles: apart | silver,silver",
        "roadtiles: read | silver,silver",
        "roadtiles: + read stack silver 2",
        "roadtiles: laid turn high | silver,slate",
        "roadtiles: + turned high silver to slate at pile",
        "roadtiles: + halo high",
        "roadtiles: + flip high",
        "roadtiles: read marks | silver,slate",
        "roadtiles: + lifted silver,slate from pile",
        "roadtiles: + read bare"]


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
