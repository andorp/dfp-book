"""Challenge II: Safe Touches  (the headline; The Theorem, Twice; the master, The Touched Bow Tie)
===============================================================================================
"It did something to every ring. It did nothing to the tray."  -- Tortoise
"And here is the thing the old man told me, the first day. If you know every touch a tray will not notice, you know the tray. Only one tray sits still for all of them."  -- Crab

Claim: for every graph G and every vertex a, K_a |G> = |G> exactly, K_a = X_a prod_{b in N_a} Z_b
(Result 16.6); the K_a generate an abelian group of 2^n Paulis that fixes |G> and nothing
else, with the basis Z^b|G> and the projector 2^-n sum sigma (Result 16.7).  In pictures:
the bare net of v vertices and e edges is exactly 2^((v-e)/2)|G> (Result 16.4); a red pi on
one leg of a phase-free green spider is a red pi on each other leg, scalar 1 (Result 16.5);
and the master, The Touched Bow Tie (red pi on the knot's clip ring, green pi on the four
wing rings), equals the untouched net exactly, 2^(-1/2)|G_bowtie> (Result 16.18), derived by
the moves of the Sketchbook proof (row 120, colour change, fusion), checked after each move.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import I2, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch16 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)
CZ = np.diag([1, 1, 1, -1]).astype(complex)

# The bow tie, bells in ket order (top wire first): 0 left-back, 1 left-front, 2 the knot,
# 3 right-back, 4 right-front.  Ties 12, 13, 23, 34, 35, 45 of the chapter, 0-indexed.
BOWTIE = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
KNOT = 2


# --- the blackboard's plumbing -----------------------------------------------------------

def on(n, ops):
    """Tensor product on n wires with ops = {wire: 2x2 matrix}, top wire first."""
    return kron(*[ops.get(w, I2) for w in range(n)])


def nbrs(E, a):
    return sorted({v for e in E for v in e if a in e and v != a})


def U_G(n, E):
    diag = np.ones(2 ** n, dtype=complex)
    for x in range(2 ** n):
        bits = [(x >> (n - 1 - w)) & 1 for w in range(n)]
        diag[x] = (-1) ** sum(bits[b] * bits[c] for b, c in E)
    return np.diag(diag)


def graph_state(n, E):
    return U_G(n, E) @ kron(*[PLUS] * n).ravel()


def K(n, E, a):
    ops = {b: Z for b in nbrs(E, a)}
    ops[a] = X
    return on(n, ops)


def random_graph(rng, n):
    return [e for e in combinations(range(n), 2) if rng.random() < 0.5]


def blackboard():
    rng = np.random.default_rng(1602)

    # --- the opening example: two wires, one tie
    G2 = CZ @ kron(PLUS, PLUS).ravel()
    assert np.allclose(G2, np.array([1, 1, 1, -1]) / 2)
    assert not np.allclose(kron(X, I2) @ G2, G2)                       # a flip alone: noticed
    assert np.allclose(kron(X, Z) @ G2, G2)                            # with a sign on its partner: not
    assert np.allclose(kron(Z, X) @ G2, G2)
    assert np.allclose(K(2, [(0, 1)], 0), kron(X, Z)) and np.allclose(K(2, [(0, 1)], 1), kron(Z, X))

    # --- one tie at a time: CZ X_a CZ = X_a Z_b; CZ commutes with Z
    assert np.allclose(CZ @ kron(X, I2) @ CZ, kron(X, Z))
    assert not np.allclose(CZ @ kron(X, I2) @ CZ, kron(X, I2))
    assert np.allclose(CZ @ kron(Z, I2), kron(Z, I2) @ CZ)

    # --- the bow tie: K_3 = Z Z X Z Z, K_1 = X Z Z I I (top wire first)
    n = 5
    G = graph_state(n, BOWTIE)
    assert np.allclose(K(n, BOWTIE, KNOT), kron(Z, Z, X, Z, Z))
    assert np.allclose(K(n, BOWTIE, 0), kron(X, Z, Z, I2, I2))
    assert not np.allclose(K(n, BOWTIE, 0), kron(I2, I2, Z, Z, X))     # not read bottom-up

    # --- the headline, and the conjugation, on the bow tie and on random graphs
    cases = [(5, BOWTIE)] + [(m, random_graph(rng, m)) for m in rng.integers(1, 7, size=30)]
    for m, E in cases:
        Gs = graph_state(m, E)
        # the written-out form: amplitude 2^{-n/2} (-1)^{E(x)}
        amps = [(-1) ** sum(((x >> (m - 1 - b)) & 1) * ((x >> (m - 1 - c)) & 1) for b, c in E)
                for x in range(2 ** m)]
        assert np.allclose(Gs, np.array(amps) / 2 ** (m / 2))
        UG = U_G(m, E)
        for a in range(m):
            Ka = K(m, E, a)
            assert np.allclose(Ka @ Gs, Gs)                             # K_a |G> = |G>, scalar 1
            assert np.allclose(UG @ on(m, {a: X}) @ UG.conj().T, Ka)    # U_G X_a U_G^dag = K_a
            if nbrs(E, a):
                assert not np.allclose(on(m, {a: X}) @ Gs, Gs)          # X_a alone is noticed
    # the colours swapped at the knot (Z on the knot, X on the wings) is noticed
    swapped = on(5, {**{b: X for b in nbrs(BOWTIE, KNOT)}, KNOT: Z})
    assert not np.allclose(swapped @ G, G)

    # --- Result 16.7 on the bow tie and on random graphs
    for m, E in [(5, BOWTIE)] + [(m, random_graph(rng, m)) for m in rng.integers(1, 6, size=12)]:
        Gs = graph_state(m, E)
        Ks = [K(m, E, a) for a in range(m)]
        # (i) they commute; 2^n different products, none -I, each fixing |G>
        for A_, B_ in combinations(Ks, 2):
            assert np.allclose(A_ @ B_, B_ @ A_)
        group = []
        for W in product((0, 1), repeat=m):
            P = np.eye(2 ** m, dtype=complex)
            for a in range(m):
                if W[a]:
                    P = P @ Ks[a]
            assert np.allclose(P @ Gs, Gs)
            assert not np.allclose(P, -np.eye(2 ** m))
            group.append(P)
        assert len(group) == 2 ** m
        assert all(not np.allclose(group[i], group[j])
                   for i, j in combinations(range(len(group)), 2))
        # (ii) the basis Z^b |G>: orthonormal, K_a Z^b|G> = (-1)^{b_a} Z^b|G>
        basis = []
        for bvec in product((0, 1), repeat=m):
            Zb = on(m, {a: Z for a in range(m) if bvec[a]})
            state = Zb @ Gs
            for a in range(m):
                assert np.allclose(Ks[a] @ state, (-1) ** bvec[a] * state)
            basis.append(state)
        B = np.array(basis)
        assert np.allclose(B.conj() @ B.T, np.eye(2 ** m))
        # (iii) the projector
        assert np.allclose(np.outer(Gs, Gs.conj()), sum(group) / 2 ** m)
        # (iv) the only state fixed by every K_a: the joint +1 space is one-dimensional
        proj = np.eye(2 ** m, dtype=complex)
        for Ka in Ks:
            proj = proj @ (np.eye(2 ** m) + Ka) / 2
        assert np.linalg.matrix_rank(proj) == 1
        if E == BOWTIE:
            assert len(group) == 32                                     # the bow tie's S_G

    # --- two touches at once: left-back and right-back.  K_1 K_4 = X_1 Z_2 X_4 Z_5 (the
    #     knot's two Z's cancel), and it fixes the tray
    K14 = K(5, BOWTIE, 0) @ K(5, BOWTIE, 3)
    assert np.allclose(K14, kron(X, Z, I2, X, Z))
    assert np.allclose(K14 @ G, G)

    # --- Result 16.5 on the blackboard: X^{(x)k} fixes |0..0> + |1..1>, so an X on one
    #     wire moves to the other k-1 wires
    for k in range(2, 7):
        ghz = np.zeros(2 ** k, dtype=complex)
        ghz[0] = ghz[-1] = 1
        assert np.allclose(kron(*[X] * k) @ ghz, ghz)
        assert np.allclose(on(k, {0: X}) @ ghz, on(k, {w: X for w in range(1, k)}) @ ghz)
        # a turned spider does not: |0..0> + i|1..1>
        turned = ghz.copy()
        turned[-1] = 1j
        assert not np.allclose(on(k, {0: X}) @ turned, on(k, {w: X for w in range(1, k)}) @ turned)


    pictures()


# --- the prop pictures --------------------------------------------------------------------

def clip_op(colour, turn):
    """A clip (or a piece threaded on a loose cord) as an operator on its bell's wire: a
    two-legged spider of its colour with phase turn/12 (pi at a half, 12; pi/2 at a quarter
    round, 6; 3pi/2 at a quarter back, 18).  Red: |+><+| + e^{i a}|-><-|; green: diag(1, e^{i a})."""
    a = np.exp(1j * np.pi * turn / 12)
    if colour == "green":
        return np.diag([1, a])
    return (np.outer(PLUS, PLUS) + a * np.outer(MINUS, MINUS)).astype(complex)


def tray_op(T):
    """The clips on the tray as one operator, ket order = the bells' order (top wire first)."""
    ops = []
    for b in PC.BELLS:
        M = I2
        for colour, turn in T.rings[b]:
            M = clip_op(colour, turn) @ M
        ops.append(M)
    return kron(*ops)


def bow_net(B):
    """The net a \\bowcloth state stands for: a green spider per ring on the cloth, phase
    (its bead's turn)/12; a Hadamard edge per tie; an output per loose cord (a capped cord
    has none: the cap's quarter is in the ring's turn), through its piece, a two-legged
    spider, if one is threaded on.  Outputs in the bells' order.  The counters by the
    teapot are numbers, not part of the net."""
    g = zx.Graph()
    v, outs = {}, []
    for w, b in enumerate(PC.BELLS):
        R = B.rings[b]
        if R["on"]:
            v[b] = g.add_vertex(VT.Z, w, 1, phase=Fraction(R["t"], 12))
    for key in sorted(B.ties):
        a, c = key.split("-", 1)
        if a in v and c in v:
            g.add_edge((v[a], v[c]), ET.HADAMARD)
    for w, b in enumerate(PC.BELLS):
        R = B.rings[b]
        if R["on"] and R["cord"]:
            last = v[b]
            if R["piece"]:
                colour, turn = R["piece"]
                p = g.add_vertex(VT.X if colour == "red" else VT.Z, w, 2, phase=Fraction(turn, 12))
                g.add_edge((last, p))
                last = p
            o = g.add_vertex(VT.BOUNDARY, w, 3)
            g.add_edge((last, o))
            outs.append(o)
    g.set_inputs(())
    g.set_outputs(tuple(outs))
    assert g.num_edges() == len([k for k in B.ties if all(x in v for x in k.split("-", 1))]) + \
        sum(1 + bool(B.rings[b]["piece"]) for b in v if B.rings[b]["cord"])
    return g


def check_bells():
    """Ket order is the bells' order: the transcription's bells and ties are this file's
    BOWTIE, wire for wire (0 left-back, 1 left-front, 2 knot, 3 right-back, 4 right-front)."""
    assert PC.BELLS == ("lb", "lf", "knot", "rb", "rf") and PC.BELLS.index("knot") == KNOT
    assert sorted(tuple(sorted((PC.BELLS.index(a), PC.BELLS.index(b)))) for a, b in PC.TIES) == BOWTIE


def rings_in_place(pic):
    """Every ring still on the cloth lies at its grid place (BOWAT), in the canonical scene."""
    for b in PC.BELLS:
        if pic.state.rings[b]["on"]:
            assert any((t["x"], t["y"]) == tuple(PC.BOWAT[b]) and t["c"] == "green" for t in pic.cloth["things"]), b


def pictures():
    """The Second Challenge's prop pictures (identity card, section 4), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch16.py.  The ties
    below hold the transcription against this file's blackboard: a structural check of the
    pictures, not a second proof."""
    check_bells()
    calls = DIALOGUE.check_layout("Second", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("belltray", True, "", ("the knot's touch",), "Every ring clipped"),
        ("bowcloth", True, "", ("bare",), "The net laid out"),
        ("bowcloth", False, "", ("the knot's touch",), "Threaded for the knot"),
        ("bowcloth", False, "push", ("the knot's touch",), "Every piece gone")]
    PC.check_captions(DIALOGUE, "Second", calls)
    PC.check_section(DIALOGUE, "Second", calls)
    PC.check_anchors(DIALOGUE, "Second", ["P2.1", "P2.2", "P2.3", "P2.4"])
    tray_h, cloth_h = round(PC.TFTOP - PC.TFBOT, 2), 2.32
    assert tray_h == 1.99                                               # read from the Lua
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Second")] == [[tray_h], [cloth_h], [cloth_h], [cloth_h]]
    p21, p22, p23, p24 = (PC.picture(c) for c in calls)
    n, G = 5, graph_state(5, BOWTIE)

    # P2.1: the knot's touch on the tray -- red at a half on the knot, green at a half on the
    # four wings: the clips are K_3 = Z Z X Z Z, and K_3 |G> = |G> exactly
    assert p21.dump == "lb=green:12,turn=12 lf=green:12,turn=12 knot=red:12,turn=12 rb=green:12,turn=12 rf=green:12,turn=12"
    K3 = tray_op(p21.state)
    assert np.allclose(K3, K(n, BOWTIE, KNOT)) and np.allclose(K3, kron(Z, Z, X, Z, Z))
    assert np.allclose(K3 @ G, G)
    assert all(p21.state.rings[b] for b in PC.BELLS)                    # "Every ring clipped"
    # the plausible mix-ups: the colours swapped, a red clip alone, the touch read bottom-up
    assert not np.allclose(tray_op(PC.tray_state("green a half on the knot; red a half on the left-back; "
                                                 "red a half on the left-front; red a half on the right-back; "
                                                 "red a half on the right-front")) @ G, G)
    assert not np.allclose(tray_op(PC.tray_state("red a half on the knot")) @ G, G)
    K1 = tray_op(PC.tray_state("the left-back's touch"))
    assert np.allclose(K1, K(n, BOWTIE, 0)) and np.allclose(K1 @ G, G)
    assert not np.allclose(kron(I2, I2, Z, Z, X), K1)                   # not the right-front's, bottom-up
    for b in PC.BELLS:                                                  # every bell's touch is safe
        assert np.allclose(tray_op(PC.tray_state("the %s's touch" % PC.BELLNAME[b])) @ G, G)
    # the tray's sixteen cords are the same with and without clips (only the clips change)
    bare = PC.belltray("bare").shown
    assert p21.shown.cords == bare.cords and len(p21.shown.cords) == 16
    assert sorted(k for k, _, _ in bare.cords) == ["inner"] * 5 + ["outer"] * 5 + ["tie"] * 6
    assert len({(k, name) for k, name, _ in bare.cords}) == 16                 # sixteen different cords
    assert len(p21.shown.clips) == 5 and bare.clips == ()
    # the keyed words: letters clear of the drawing and of each other, the frame by 0.025
    assert p21.labels == ("back", "front", "knot", "ring", "clip", "sleeve", "tassel")
    PC.check_keyed_words(p21)

    # P2.2: the net laid out -- five rings at rest, each with a loose cord, the six ties:
    # exactly 2^{(5-6)/2} |G> (each yellow tube a normalized H)
    assert p22.dump.split(" | ")[0] == ("lb=0+cord lf=0+cord knot=0+cord rb=0+cord rf=0+cord "
                                        "ties=knot-rb,knot-rf,lb-knot,lb-lf,lf-knot,rb-rf edge= aside=0")
    net22 = bow_net(p22.state)
    assert np.allclose(matrix_of(net22).ravel(), 2 ** ((5 - 6) / 2) * G)
    assert not np.allclose(matrix_of(net22).ravel(), G)                 # the tubes' number is there
    assert p22.state.aside == 0 and not any(p22.state.rings[b]["piece"] for b in PC.BELLS)
    assert p22.labels == ("tie", "tube", "ring", "loose cord")
    # (the cloth kit places its own keyed words -- not transcribed; measured from the Lua's
    # output after the kit's TOPRAISE, their letters clear the cloth's parts and the linen's
    # edge by at least 0.024 cm: "ring" to the knot's loose cord the least; "loose cord", raised
    # in the band above the linen, clears the linen's edge by 0.052)

    # P2.3: the touch threaded on the loose cords -- a red pi on the knot's, a green pi on each
    # wing's: the net with K_3 on its outputs, which is the same state
    assert p23.dump.split(" | ")[0] == ("lb=0+cord:green12 lf=0+cord:green12 knot=0+cord:red12 rb=0+cord:green12 "
                                        "rf=0+cord:green12 ties=knot-rb,knot-rf,lb-knot,lb-lf,lf-knot,rb-rf edge= aside=0")
    net23 = bow_net(p23.state)
    assert [p23.state.rings[b]["piece"] for b in PC.BELLS] == [("green", 12)] * 2 + [("red", 12)] + [("green", 12)] * 2
    assert np.allclose(matrix_of(net23).ravel(), K3 @ matrix_of(net22).ravel())
    # P2.4: pushed home, every piece gone -- the bare net again, nothing set aside
    assert p24.dump == p22.dump and p24.state.aside == 0
    assert not any(p24.state.rings[b]["piece"] for b in PC.BELLS)       # "Every piece gone"
    assert same_picture(net23, bow_net(p24.state)) and same_picture(net23, net22)
    # a wrong threading (red on a wing) changes the net
    wrong = PC.bow_state("", "the left-back's touch")
    wrong.rings["knot"]["piece"] = ("red", 12)
    assert not same_picture(bow_net(wrong), net22)
    # fixed parts: the five rings stay at their places on the cloth
    for p in (p22, p23, p24):
        rings_in_place(p)
    assert not PC.TEX_ERRORS


# --- the sketchbook ----------------------------------------------------------------------

def net(n, E, clips=None):
    """The net: a phase-free green spider per bell, a Hadamard edge (the yellow box) per tie,
    one output per bell.  With clips, a clip ring -- a two-legged spider (kind, phase) -- on
    every output, unfused: bell -- ring -- output.  Returns (graph, bells, rings)."""
    g = zx.Graph()
    bells = [g.add_vertex(VT.Z, w, 1) for w in range(n)]
    for b, c in E:
        g.add_edge((bells[b], bells[c]), ET.HADAMARD)
    rings, outs = [], []
    for w in range(n):
        last = bells[w]
        if clips is not None:
            kind, phase = clips[w]
            r = g.add_vertex(kind, w, 2, phase=Fraction(phase))
            g.add_edge((last, r))
            rings.append(r)
            last = r
        o = g.add_vertex(VT.BOUNDARY, w, 3)
        g.add_edge((last, o))
        outs.append(o)
    g.set_inputs(())
    g.set_outputs(tuple(outs))
    return g, bells, rings


AT_REST = (VT.Z, 0)
RED_HALF = (VT.X, 1)
GREEN_HALF = (VT.Z, 1)


def touch(n, E, a):
    """The safe touch at a as clip rings: red half on a, green half on each neighbour."""
    clips = [AT_REST] * n
    clips[a] = RED_HALF
    for b in nbrs(E, a):
        clips[b] = GREEN_HALF
    return clips


def push_picture(k, kind, other, phase=0, side="one"):
    """A spider of `kind` with phase `phase` and k legs: one input, k-1 outputs.  A pi dot of
    colour `other` on the input leg (side='one') or on each output leg (side='rest')."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    s = g.add_vertex(kind, 0, 2, phase=Fraction(phase))
    if side == "one":
        p = g.add_vertex(other, 0, 1, phase=Fraction(1))
        g.add_edge((i, p))
        g.add_edge((p, s))
    else:
        g.add_edge((i, s))
    outs = []
    for j in range(k - 1):
        o = g.add_vertex(VT.BOUNDARY, j, 4)
        if side == "rest":
            p = g.add_vertex(other, j, 3, phase=Fraction(1))
            g.add_edge((s, p))
            g.add_edge((p, o))
        else:
            g.add_edge((s, o))
        outs.append(o)
    g.set_inputs((i,))
    g.set_outputs(tuple(outs))
    return g


def sketchbook():
    rng = np.random.default_rng(1622)

    # --- Result 16.4: the bare net is exactly 2^{(v-e)/2} |G>, bow tie and random graphs
    bare, _, _ = net(5, BOWTIE)
    G = graph_state(5, BOWTIE)
    assert np.allclose(matrix_of(bare).ravel(), 2 ** -0.5 * G)
    assert not np.allclose(matrix_of(bare).ravel(), G)                  # the number is drawn
    two, _, _ = net(2, [(0, 1)])                                        # the smallest net
    assert np.allclose(matrix_of(two).ravel(), np.sqrt(2) * CZ @ kron(PLUS, PLUS).ravel())
    for m in rng.integers(1, 7, size=40):
        E = random_graph(rng, m)
        g, _, _ = net(m, E)
        assert np.allclose(matrix_of(g).ravel(), 2 ** ((m - len(E)) / 2) * graph_state(m, E))

    # --- Result 16.5 (row 120): a red pi on one leg of a phase-free green spider with k legs
    #     is a red pi on each of the other k-1 legs, scalar 1; colours exchanged likewise.
    for k in range(2, 7):
        for kind, other in [(VT.Z, VT.X), (VT.X, VT.Z)]:
            assert same_picture(push_picture(k, kind, other, side="one"),
                                push_picture(k, kind, other, side="rest"))
            # a turned spider does not copy: the rule needs the spider phase-free
            if k >= 3:
                assert not same_picture(push_picture(k, kind, other, Fraction(1, 2), "one"),
                                        push_picture(k, kind, other, Fraction(1, 2), "rest"))
    #     DERIVED, as in the chapter's sidebar: feed a red point c*pi into both sides.  Left:
    #     fuse first (spider_simp), then copy.  Right: copy first, then fuse.  Both end as
    #     k-1 red points with phase (c+1)*pi and the same number.
    for k in range(2, 7):
        for c in (0, 1):
            ends = []
            for side in ("one", "rest"):
                g = push_picture(k, VT.Z, VT.X, side=side)
                (i,) = g.inputs()
                g.set_type(i, VT.X)
                g.set_phase(i, Fraction(c))
                g.set_inputs(())
                before = g.copy()
                if side == "one":
                    simp.spider_simp(g)
                pts = [v for v in g.vertices() if g.type(v) == VT.X and g.vertex_degree(v) == 1]
                assert len(pts) == 1 and simp.copy(g, pts[0])
                simp.spider_simp(g)
                assert same_picture(g, before)
                reds = [v for v in g.vertices() if g.type(v) != VT.BOUNDARY]
                assert len(reds) == k - 1
                assert all(g.type(v) == VT.X and g.phase(v) == (c + 1) % 2 for v in reds)
                ends.append(g.scalar.to_number())
            assert np.isclose(ends[0], ends[1])                          # the same price

    # --- the master, The Touched Bow Tie: the bow tie's net with a clip ring on every
    #     output, red pi on the knot's ring, green pi on the four wing rings
    untouched, _, _ = net(5, BOWTIE, [AT_REST] * 5)
    master, bells, rings = net(5, BOWTIE, touch(5, BOWTIE, KNOT))
    for g in (untouched, master):
        # 6 ties (Hadamard edges) + 5 inner lengths (bell -- ring) + 5 outer lengths
        # (ring -- output) = 16 lines
        assert g.num_edges() == 16
        assert g.num_vertices() == 15                                   # 5 bells, 5 rings, 5 ends
        assert sum(1 for e in g.edges() if g.edge_type(e) == ET.HADAMARD) == 6
    assert [master.type(r) for r in rings] == [VT.Z, VT.Z, VT.X, VT.Z, VT.Z]
    assert all(master.phase(r) == 1 for r in rings)
    assert same_picture(master, untouched)                              # Result 16.18
    assert same_picture(untouched, bare)                                # rings at rest: plain wire
    assert np.allclose(matrix_of(master).ravel(), 2 ** -0.5 * G)       # exactly 2^{-1/2}|G>
    # negative control: the colours swapped (green on the knot, red on the wings)
    swapped = [(VT.X, 1)] * 5
    swapped[KNOT] = (VT.Z, 1)
    assert not same_picture(net(5, BOWTIE, swapped)[0], untouched)
    # half the touch -- the red half on the knot without the green halves -- is noticed
    lone = [AT_REST] * 5
    lone[KNOT] = RED_HALF
    assert not same_picture(net(5, BOWTIE, lone)[0], untouched)
    # the touch at every bell (the left-back one has 16 edges too), and two touches at once
    for a in range(5):
        g, _, _ = net(5, BOWTIE, touch(5, BOWTIE, a))
        assert g.num_edges() == 16 and same_picture(g, untouched)
    both = [RED_HALF, GREEN_HALF, AT_REST, RED_HALF, GREEN_HALF]         # knot: two greens, at rest
    assert same_picture(net(5, BOWTIE, both)[0], untouched)
    # Result 16.6 in pictures, for random graphs and every vertex
    for m in rng.integers(1, 7, size=20):
        E = random_graph(rng, m)
        plain, _, _ = net(m, E, [AT_REST] * m)
        for a in range(m):
            assert same_picture(net(m, E, touch(m, E, a))[0], plain)

    # --- the rewrite that derives the master: the moves of §16.7 at a = the knot, as edits
    #     of the picture, the value checked after each
    g = master.copy()
    knot, kring = bells[KNOT], rings[KNOT]
    wings = [bells[b] for b in nbrs(BOWTIE, KNOT)]
    # Step 1 (row 120): the red pi leaves the knot's output (its ring left at rest) and
    # appears on every tie, next to the knot: knot -- red pi -- yellow box -- wing bell
    g.set_type(kring, VT.Z)
    g.set_phase(kring, 0)
    halves = []
    for w in wings:
        g.remove_edge(g.edge(knot, w))
        h = g.add_vertex(VT.X, KNOT, 0.5, phase=Fraction(1))
        g.add_edge((knot, h))
        g.add_edge((h, w), ET.HADAMARD)
        halves.append(h)
    assert same_picture(g, master)
    # Step 2 (colour change, row 58): each half crosses its box and turns green
    for h in halves:
        assert simp.check_color_change(g, h) and simp.color_change(g, h)
        assert g.type(h) == VT.Z and g.phase(h) == 1
        assert g.edge_type(g.edge(knot, h)) == ET.HADAMARD
    assert same_picture(g, master)
    # Step 3 (fusion, row 57): each green half slides into its wing bell, and so does the
    # green half on that bell's ring: pi + pi = 0
    for h, w, b in zip(halves, wings, nbrs(BOWTIE, KNOT)):
        assert simp.check_fuse(g, w, h) and simp.fuse(g, w, h)
        assert g.phase(w) == 1
        assert same_picture(g, master)
        r = rings[b]
        assert simp.check_fuse(g, w, r) and simp.fuse(g, w, r)
        assert g.phase(w) == 0                                          # no turn
        assert same_picture(g, master)
    # Step 4: read off.  Every spider is phase-free, the ties are the bow tie's six yellow
    # boxes, and nothing was set aside: the picture is the untouched net, scalar 1
    assert all(g.phase(v) == 0 for v in g.vertices())
    assert sum(1 for e in g.edges() if g.edge_type(e) == ET.HADAMARD) == 6
    assert np.isclose(g.scalar.to_number(), 1)
    assert same_picture(g, untouched) and same_picture(g, bare)


if __name__ == "__main__":
    verdict("SafeTouches", blackboard, sketchbook)
