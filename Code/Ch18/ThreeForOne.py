"""Challenge I: Three for One  (noise; the vote; the encoder)
============================================================
"About one hollow in ten read false."  -- Crab
"...Oh. You can't copy a turn; you can only spread it."  -- Achilles

Claim: a vote of three fails with probability F(p) = 3p^2 - 2p^3 whichever way the bit was
set (Result 18.2), and the quantum bit-flip code turns independent flips into a logical flip
of exactly that probability (Result 18.11).  The encoder Enc = |000><0| + |111><1| is one
green spider, 1 -> 3, with no number (Result 18.3); its mirror undoes it; a dark or lit point
copies through it with the number 1/2, while a turned point fuses into it and comes out
entangled, never as three copies (Result 18.4).  The bit-flip channel, doubled, is (1-p) bare
+ p red halves on the wire and its mirror (Result 18.1).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from itertools import product

import numpy as np
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import I2, Picture, ket, kron, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch18 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
Y = 1j * X @ Z                                             # Gottesman's Table 1: Y = iXZ

ENC = np.outer(ket(0, 0, 0), ket(0)) + np.outer(ket(1, 1, 1), ket(1))   # 8 x 2, top wire first

# the door dial's six marks, read as states
MARKS = {
    "dark": ket(0), "lit": ket(1),
    "at rest": (ket(0) + ket(1)) / np.sqrt(2), "a half": (ket(0) - ket(1)) / np.sqrt(2),
    "a quarter round": (ket(0) + 1j * ket(1)) / np.sqrt(2),
    "a quarter back": (ket(0) - 1j * ket(1)) / np.sqrt(2),
}


def F(p):
    return 3 * p**2 - 2 * p**3


def on(k, A):
    """A on wire k (0 = top) of three."""
    ops = [I2, I2, I2]
    ops[k] = A
    return kron(*ops)


def X_pattern(e):
    return kron(*[X if b else I2 for b in e])


# the checks and the mending table of Section 18.2 (defined here too: files stand alone)
Z1Z2, Z2Z3 = on(0, Z) @ on(1, Z), on(1, Z) @ on(2, Z)
I8 = np.eye(8)


def P(s):
    s1, s2 = s
    return (I8 + (-1) ** s1 * Z1Z2) @ (I8 + (-1) ** s2 * Z2Z3) / 4


CORR = {(0, 0): (0, 0, 0), (1, 0): (1, 0, 0), (1, 1): (0, 1, 0), (0, 1): (0, 0, 1)}


def superop(kraus):
    """Row-major doubling: vec(rho) = sum rho_ij |i>|j>, and A acts as A (x) conj(A)."""
    return sum(np.kron(K, K.conj()) for K in kraus)


def random_density(rng, n=2):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def blackboard():
    rng = np.random.default_rng(1801)

    # --- the two flips on the six marks (Section 18.1)
    assert np.allclose(X @ MARKS["at rest"], MARKS["at rest"])                       # X|+> = |+>
    assert np.allclose(X @ MARKS["a quarter round"], 1j * MARKS["a quarter back"])   # up to i
    fid = lambda a, b: abs(np.vdot(a, b)) ** 2
    harmless_X = {m for m, v in MARKS.items() if np.isclose(fid(v, X @ v), 1)}
    harmless_Z = {m for m, v in MARKS.items() if np.isclose(fid(v, Z @ v), 1)}
    assert harmless_X == {"at rest", "a half"}          # N^X never harms at rest (nor a half)
    assert harmless_Z == {"dark", "lit"}                # N^Z never harms dark or lit
    assert harmless_X != set(MARKS) and harmless_Z != set(MARKS)   # neither harmless for all six

    # --- the three channels in Kraus form are trace preserving, for every p
    for p in (0.1, 0.37, rng.uniform()):
        chans = {
            "X": [np.sqrt(1 - p) * I2, np.sqrt(p) * X],
            "Z": [np.sqrt(1 - p) * I2, np.sqrt(p) * Z],
            "dep": [np.sqrt(1 - p) * I2] + [np.sqrt(p / 3) * Q for Q in (X, Y, Z)],
        }
        for ks in chans.values():
            assert np.allclose(sum(K.conj().T @ K for K in ks), I2)

    # --- Result 18.2: the vote of three, enumerated over the eight flip patterns,
    #     for a bit set dark AND a bit set lit (the tins' damp goes both ways)
    for p in (0.1, 0.37, rng.uniform()):
        for setting in (0, 1):
            wrong = 0.0
            for flips in product((0, 1), repeat=3):
                prob = np.prod([p if f else 1 - p for f in flips])
                read = [setting ^ f for f in flips]
                majority = int(sum(read) >= 2)
                wrong += prob * (majority != setting)
            assert np.isclose(wrong, F(p))
    assert np.isclose(F(0.1), 0.028)                                  # three in a hundred
    assert np.isclose(F(F(0.1)), 0.002308096, atol=1e-12)             # the ledger's vote of votes
    assert not np.isclose(F(0.1), 0.1)                                # the vote does buy something

    # --- the encoder: Enc(a|0> + b|1>) = a|000> + b|111>, NOT the clone |psi>^(x)3
    psi = random_state(rng)                                           # a generic, non-real message
    assert not np.allclose(psi, psi.real)
    assert np.allclose(ENC @ psi, psi[0] * ket(0, 0, 0) + psi[1] * ket(1, 1, 1))
    assert not np.allclose(ENC @ psi, kron(psi, psi, psi).ravel())
    for a in (0, 1):                                                  # dark and lit: a true copy
        assert np.allclose(ENC @ ket(a), kron(ket(a), ket(a), ket(a)).ravel())
    q = MARKS["a quarter round"]                                      # the dialogue's turned message
    assert np.allclose(ENC @ q, (ket(0, 0, 0) + 1j * ket(1, 1, 1)) / np.sqrt(2))
    assert not np.allclose(ENC @ q, kron(q, q, q).ravel())
    assert np.allclose(ENC.conj().T @ ENC, I2)                        # the encoder loses nothing
    # Enc is real, so its mirror (conjugate transpose) is just its transpose
    assert np.allclose(ENC, ENC.real)

    # --- Result 18.4 on the blackboard: copy (with 1/2) and spread (entangled)
    for a in (0, 1):
        red_point = np.sqrt(2) * ket(a)                               # bare red point, a*pi
        three = kron(red_point, red_point, red_point).ravel()         # (sqrt2)^3 |aaa>
        assert np.allclose(ENC @ red_point, three / 2)                # the number 1/2
        assert not np.allclose(ENC @ red_point, three)
    for phi in (0.5 * np.pi, rng.uniform(0, 2 * np.pi)):              # a quarter round; a random turn
        g = ket(0) + np.exp(1j * phi) * ket(1)                        # a green point, bare
        out = ENC @ g
        assert np.allclose(out, ket(0, 0, 0) + np.exp(1j * phi) * ket(1, 1, 1))
        assert np.linalg.matrix_rank(out.reshape(2, 4)) == 2          # entangled across the cut
        pts = kron(g, g, g).ravel()
        assert np.linalg.matrix_rank(pts.reshape(2, 4)) == 1          # three points: rank 1

    # --- Result 18.11: encode, independent flips, recover, good-branch decode
    #     = the bit-flip channel with p_L = F(p), as 4 x 4 superoperators
    for p in (0.1, 0.37, rng.uniform()):
        full = np.zeros((4, 4), dtype=complex)
        for e in product((0, 1), repeat=3):
            w = sum(e)
            Ke = np.sqrt(p**w * (1 - p) ** (3 - w)) * X_pattern(e)
            for s, c in CORR.items():
                K = ENC.conj().T @ X_pattern(c) @ P(s) @ Ke @ ENC
                full += np.kron(K, K.conj())
        pL = F(p)
        assert np.allclose(full, superop([np.sqrt(1 - pL) * I2, np.sqrt(pL) * X]))
        assert not np.allclose(full, superop([np.sqrt(1 - p) * I2, np.sqrt(p) * X]))  # not p itself

    pictures()


# ------------------------------------------------------------------------------ prop pictures

FIRST_DUMPS = (
    "first=dark second=dark third=lit",
    "dark,pushed aside=1 | cols=7 rows=3 y=0..2 near=false folded=false; 1:end red 0 @4,0; 2:end red 0 @4,1; "
    "3:end red 0 @4,2; cord loose-1 [6.3,0 4,0]; cord loose-2 [6.3,1 4,1]; cord loose-3 [6.3,2 4,2]",
    "turned,pushed aside=0 | cols=7 rows=3 y=0..2 near=false folded=false; 1:ring green 6 @3,1; "
    "cord loose-1 [6.3,0 4,0 3,1]; cord loose-1 [6.3,1 3,1]; cord loose-1 [6.3,2 4,2 3,1]",
    "mend=0,0,0 tell=0,0,0 clip=none glass=reset",
)


def pictures():
    """The First Challenge's prop pictures (identity card, section 4), read from the dialogue
    through tools/dfp_props.py and the transcription tools/props_ch18.py, and held against this
    file's blackboard and pictures (graph_of: a structural reading of each cloth dump, not a
    proof)."""
    calls = DIALOGUE.check_layout("First", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("plaintin", True, "first dark, second dark, third lit", (), "A tin, spent"),
        ("partcloth", True, "pushed in", ("the dark point",), "Out at the ends"),
        ("partcloth", False, "pushed in", ("the turned point",), "Taken in"),
        ("stormline", True, "", (), "Strung for the storm")]
    PC.check_captions(DIALOGUE, "First", calls)
    PC.check_section(DIALOGUE, "First", calls)
    PC.check_anchors(DIALOGUE, "First", ["P1.1", "P1.2", "P1.3", "P1.4"])
    pics = [PC.picture(c) for c in calls]
    p11, p12, p13, p14 = pics
    assert tuple(p.dump for p in pics) == FIRST_DUMPS

    # one frame per macro, in every state (the counter band drawn whether or not it holds one)
    assert {PC.plaintin(s).frame for s in ("", "first lit", "first dark, second lit, third dark")} == {p11.frame}
    assert {PC.partcloth(s, st).frame for s in ("", "pushed in")
            for st in ("the dark point", "the lit point", "the turned point")} == {p12.frame} == {p13.frame}
    assert {PC.stormline(s).frame for s in ("", "clip-bell on, glass cloudy", "middle mending ring at a half")} == {p14.frame}

    # P1.1: three hollows over one setting nobody in the hall knew; the ledger, in the paragraph
    # after the block, shows it was set dark.  For this tin one reading is false (the last), and
    # the vote of the three is the setting: a single false hollow is a pattern the vote survives
    # (the enumeration of Result 18.2 counts it a success), two would not be.
    after = DIALOGUE.following("First")[0]
    assert "The damp got the last one." in after and "Set dark. The vote has it." in after
    readings = [{"dark": 0, "lit": 1}[L["r"]] for L in p11.state.lids]
    setting = 0                                                   # "Set dark."
    flips = tuple(r ^ setting for r in readings)
    assert flips == (0, 0, 1) and int(sum(readings) >= 2) == setting
    assert sum(r ^ 1 for r in readings) == 2                      # had it been set lit: two false, the vote wrong
    assert all(L["up"] for L in p11.state.lids)                   # "A tin, spent": no lid down
    assert "down" in PC.plaintin("first dark, second dark").dump  # (a tin not spent would show one)

    # P1.2: the dark point pushed in is EXACTLY TWICE the dark point knotted into the parting
    # ring; so the counter set aside stands for 1/2, the number the blackboard asserts
    # (ENC @ red_point == three / 2)
    g12 = PC.graph_of(p12.dump)
    assert g12.num_inputs() == 0 and g12.num_outputs() == 3 and g12.num_edges() == 3
    knotted = point_into_encoder(VT.X, 0)
    twice = point_into_encoder(VT.X, 0)
    twice.scalar.add_power(2)                                     # sqrt2^2 = 2
    assert same_picture(g12, twice) and not same_picture(g12, knotted)
    pushed, kn = matrix_of(g12).ravel(), matrix_of(knotted).ravel()
    counter = np.vdot(pushed, kn) / np.vdot(pushed, pushed)       # what the counter stands for
    assert np.isclose(counter, 0.5) and np.allclose(counter * pushed, kn)
    red_point = np.sqrt(2) * ket(0)
    assert np.allclose(kn, ENC @ red_point) and np.allclose(pushed, kron(red_point, red_point, red_point).ravel())
    assert p12.state.aside == 1 and len(p12.shown.counters) == 1
    # the lit point (unpictured) the same, at a half: twice the knotted lit point
    lit = PC.graph_of(PC.partcloth("pushed in", "the lit point").dump)
    twice_lit = point_into_encoder(VT.X, 1)
    twice_lit.scalar.add_power(2)
    assert same_picture(lit, twice_lit)
    # "Out at the ends": pushed, three red points at rest at the near ends of the cords
    nodes = PC.parse_dump(p12.dump)[0]
    assert p12.state.pushed and sorted(nodes.values()) == [("end", "red", 0, 4.0, float(r)) for r in range(3)]
    # the counter lies off the linen, beyond its near edge and clear of its outline and its
    # shadow, by the right-hand end of the cloth, inside the band and the frame: measured on the
    # drawn extents (the counter's disc, edge and shadow; the linen's stroke and its shadow), as
    # word_gaps measures a word, with a positive gap
    left, bottom, right, top = p12.shown.linen
    box = PC.counter_box(*p12.shown.counters[0])
    s = PC.C._geometry(p12.cloth, p12.options)["P"] / 0.30
    sh = PC.LINEN_SHADOW * s
    linen = (PC.O.rect(left, bottom, right, top), PC._hw(0.3 * s))
    shadow = (PC.O.rect(left + sh, bottom - sh, right + sh, top - sh), 0.0)
    disc = (PC.O.rect(*box), 0.0)
    assert PC.O.gap(disc, linen) > 0 and PC.O.gap(disc, shadow) > 0
    assert box[3] < bottom - sh                                   # beyond the near edge, not beside it
    frame = (-1.25, p12.shown.fbot - PC.ASIDE, 1.25, PC.CLOTHLB)  # word_gaps's frame (the Lua's, guarded)
    assert min(box[0] - frame[0], box[1] - frame[1], frame[2] - box[2], frame[3] - box[3]) > 0
    assert box[1] > frame[1] and box[3] < p12.shown.fbot          # in the band below the cloth's frame
    assert (left + right) / 2 < box[0]                            # at its right-hand end
    # the keyed word point is the kit's own placement (its no-room fault is the check); the Lua
    # sets no word of its own after the push
    assert p12.labels == ("point",) and PC.own_words(p12) == ()

    # P1.3: the turned point pushed in EQUALS the turned point knotted in, exactly: one green
    # ring a quarter round with its three cords (C1); nothing set aside
    g13 = PC.graph_of(p13.dump)
    assert same_picture(g13, point_into_encoder(VT.Z, Fraction(1, 2)))
    assert same_picture(g13, green_three_legs(Fraction(1, 2)))
    assert not same_picture(g13, point_into_encoder(VT.Z, Fraction(3, 2)))     # not a quarter back
    for power in (-2, 0, 2):                                      # and three turned points at no number
        assert not same_picture(g13, three_points(VT.Z, Fraction(1, 2), power))
    assert p13.state.aside == 0 and p13.shown.counters == ()
    nodes = PC.parse_dump(p13.dump)[0]                            # "Taken in": turned, pushed, no point
    assert p13.state.kind == "turned" and p13.state.pushed and [n[0] for n in nodes.values()] == ["ring"]

    # P1.4: the line as strung is the storm line itself: eighteen cords, the master's edge count
    assert PC.line_cords() == 18 == len(p14.shown.spans)
    v = p14.shown
    assert len(v.rings) == 13 and len(v.bells) == 2
    assert sum(1 for r in v.rings if r[3] == "green") == 5 and sum(1 for r in v.rings if r[3] == "red") == 8
    # "Strung for the storm": every bead at rest (at the top of its ring), no clip, glass reset
    assert all(q == 0 for *_, q in v.rings) and v.clip == () and v.glass == "reset"
    assert all(np.isclose(by, r[2] + PC.SRM) and np.isclose(bx, r[1]) for (bx, by), r in zip(v.beads, v.rings))
    # the strands: the first on top; the bells between them, a little towards the kitchen
    tell = [r for r in v.rings if r[0].startswith("t")]
    bells = [r for r in v.rings if r[0].startswith("h")]
    assert [r[2] for r in tell] == sorted((r[2] for r in tell), reverse=True)
    assert all(tell[k + 1][2] < b[2] < tell[k][2] and b[1] > tell[k][1] for k, b in enumerate(bells))

    # the keyed words the Lua sets itself: measured clear (letters 0.02 cm from every part and
    # word, boxes 0.025 cm inside the frame), at the compiled boxes
    for pic in (p11, p14):
        PC.check_compiled_boxes(pic)
        least = PC.check_keyed_words(pic)
        assert set(least) == set(pic.labels) and min(least.values()) >= PC.LETTER_FLOOR
    assert p11.labels == ("first", "second", "third")
    assert p14.labels == ("door", "kitchen", "weather ring", "mending ring", "tell ring")

    # the Lua's refusals: lids out of order, a lid lifted twice, a weather ring turned
    PC.TEX_ERRORS.clear()
    PC.tin_state("second dark")
    PC.tin_state("first dark, first lit")
    PC.line_state("middle weather ring at a half")
    PC.part_state("pushed in, pushed in", "the dark point")
    assert [k for k, _ in PC.TEX_ERRORS] == ["tin-order", "tin-lifted", "line-weather", "part-twice"]
    PC.TEX_ERRORS.clear()


# ------------------------------------------------------------------------------ pictures

def encoder(kind=VT.Z):
    """The parting ring: one spider at rest, one leg in, three out (strand 1 on top)."""
    p = Picture()
    i = p.inp(1)
    E = p.g.add_vertex(kind, 1, 1)
    p.wire(i, E)
    for k in range(3):
        p.wire(E, p.out(k))
    return p.done()


def round_trip():
    """Parting ring, three strands, joining ring.  Each strand carries a red ring at rest
    (bare cord, row 57) so that PyZX keeps three separate strands: its simple graphs hold at
    most one edge between two vertices."""
    p = Picture()
    i, o = p.inp(1), p.out(1)
    E, D = p.z(1, 1), p.z(1, 5)
    p.wire(i, E)
    for k in range(3):
        p.wire(E, p.x(k, 3, 0), D)
    p.wire(D, o)
    return p.done()


def point_into_encoder(colour, phase):
    p = Picture()
    pt = p.g.add_vertex(colour, 1, 0, phase=Fraction(phase))
    E = p.z(1, 1)
    p.wire(pt, E)
    for k in range(3):
        p.wire(E, p.out(k))
    return p.done()


def three_points(colour, phase, power=0):
    p = Picture()
    for k in range(3):
        p.wire(p.g.add_vertex(colour, k, 0, phase=Fraction(phase)), p.out(k))
    g = p.done()
    g.scalar.add_power(power)                    # a factor sqrt2^power
    return g


def green_three_legs(phase):
    p = Picture()
    v = p.z(1, 1, phase)
    for k in range(3):
        p.wire(v, p.out(k))
    return p.done()


def doubled(ops_top, ops_mirror):
    """A hand-built doubled wire (Chapter 9, row 66): row 0 the wire, row 1 its mirror copy.
    No ground vertex: pyzx 0.10.6 refuses grounded graphs.  ops are ('z'|'x', phase) in the
    order the message meets them."""
    p = Picture()
    for row, ops in ((0, ops_top), (1, ops_mirror)):
        chain = [p.inp(row)]
        for col, (kind, ph) in enumerate(ops, start=1):
            chain.append((p.z if kind == "z" else p.x)(row, col, ph))
        chain.append(p.out(row))
        p.wire(*chain)
    return p.done()


def sketchbook():
    rng = np.random.default_rng(1802)

    # --- Result 18.3: the encoder is one green spider, exactly, with no number
    enc = encoder()
    assert np.allclose(matrix_of(enc), ENC)
    assert not same_picture(enc, encoder(VT.X))            # not the red one (that is Enc', 18.13)

    # DERIVED: the parting ring followed by its mirror is a bare wire, number 1
    rt = round_trip()
    assert rt.num_edges() == 8                              # lead in, 3 x 2 strand lengths, lead out
    assert np.allclose(matrix_of(rt), I2)
    h = rt.copy()
    simp.full_reduce(h)
    assert h.num_vertices() == 2 and h.num_edges() == 1     # the two ends and one bare wire
    assert np.isclose(h.scalar.to_number(), 1)
    assert same_picture(h, rt)

    # --- Result 18.4 (C5): a red point, dark or lit, copies with the number 1/2
    for a in (0, 1):
        g = point_into_encoder(VT.X, a)
        ref = three_points(VT.X, a, power=-2)               # 1/2 x three red points (sqrt2^-2)
        assert same_picture(g, ref)
        assert not same_picture(g, three_points(VT.X, a))   # the 1/2 is not optional
        # DERIVED by full_reduce: three separate points, nothing joining them, number 1/2
        h = g.copy()
        simp.full_reduce(h)
        spiders = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
        assert len(spiders) == 3
        assert all(h.type(u) == VT.BOUNDARY for v in spiders for u in h.neighbors(v))
        assert {next(iter(h.neighbors(v))) for v in spiders} == set(h.outputs())
        assert np.isclose(h.scalar.to_number(), 0.5)
        assert same_picture(h, ref)

    # --- Result 18.4: a green point turned phi fuses in; it is not three points of any kind
    for phi in (Fraction(1, 2), Fraction(int(rng.integers(1, 97)), 97)):
        g = point_into_encoder(VT.Z, phi)
        h = g.copy()
        assert simp.spider_simp(h)                          # DERIVED: fusion (row 57)
        spiders = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
        assert len(spiders) == 1 and h.phase(spiders[0]) == phi
        assert len(list(h.neighbors(spiders[0]))) == 3
        assert same_picture(h, g) and same_picture(g, green_three_legs(phi))
        m = matrix_of(g).reshape(2, 4)                      # top wire | the other two
        assert np.linalg.matrix_rank(m) == 2                # entangled...
        pts = matrix_of(three_points(VT.Z, phi)).reshape(2, 4)
        assert np.linalg.matrix_rank(pts) == 1              # ...three points never are, at any number

    # --- (T1) translation check, not a derivation: a weighted sum of pictures is not one
    # diagram, so nothing rewrites it.  The doubled channel is evaluated against Kraus form.
    bare = doubled([], [])
    halves = doubled([("x", 1)], [("x", 1)])                # X on the wire, conj(X) = X on the mirror
    assert np.allclose(matrix_of(bare), np.eye(4))
    assert np.allclose(matrix_of(halves), np.kron(X, X.conj()))
    greens = doubled([("z", 1)], [("z", 1)])                # the phase flip: green halves
    assert np.allclose(matrix_of(greens), np.kron(Z, Z.conj()))
    # Y = iXZ: the message meets Z, then X.  On the mirror conj(Y) = -iXZ; the numbers i and -i
    # cancel, so the doubled Y is the plain picture Z-then-X on both copies, with no number.
    ys = doubled([("z", 1), ("x", 1)], [("z", 1), ("x", 1)])
    assert np.allclose(matrix_of(ys), np.kron(Y, Y.conj()))
    assert not np.allclose(matrix_of(ys), np.kron(Y, Y))    # the mirror conjugates
    for p in (0.1, 0.37, rng.uniform()):
        rho = random_density(rng)
        assert not np.allclose(rho, rho.real)               # a non-real density matrix
        vec = rho.reshape(4)                                # row-major doubling
        NX = (1 - p) * matrix_of(bare) + p * matrix_of(halves)
        assert np.allclose(NX @ vec, ((1 - p) * rho + p * X @ rho @ X).reshape(4))
        NZ = (1 - p) * matrix_of(bare) + p * matrix_of(greens)
        assert np.allclose(NZ @ vec, ((1 - p) * rho + p * Z @ rho @ Z).reshape(4))
        Nd = (1 - p) * matrix_of(bare) + p / 3 * (matrix_of(halves) + matrix_of(ys) + matrix_of(greens))
        dep = (1 - p) * rho + p / 3 * (X @ rho @ X + Y @ rho @ Y.conj().T + Z @ rho @ Z)
        assert np.allclose(Nd @ vec, dep.reshape(4))
        # the plausible slip: a half on the wire only is X rho, not X rho X
        wrong = (1 - p) * matrix_of(bare) + p * matrix_of(doubled([("x", 1)], []))
        assert not np.allclose(wrong @ vec, NX @ vec)


if __name__ == "__main__":
    verdict("ThreeForOne", blackboard, sketchbook)
