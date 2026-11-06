"""Challenge V: An Eighth  (T is not a Clifford; magic; injection)
===============================================================
"An eighth."  -- Tortoise
"It won't sit. The front lug has an edge in it, between the dark red and the pale gold."  -- Achilles

Claim: T = diag(1, e^{i pi/4}) has T X T^dagger = (X + Y)/sqrt2, not a phase times a Pauli,
so T is not a Clifford; on the octahedron it turns an eighth about the up-down axis, the
front tip +X going to the edge midpoint (1,1,0)/sqrt2, and the front lug meeting the edge
between the dark red (+X) and pale gold (-Y) tips; T^2 = S (Result 16.15).  The bow tie
capped with eighth-turn effects is ((3 sqrt2 - 4) + (sqrt2 - 2) i)/8, and PyZX's
quarter-turn simplifier stops with all five eighths inside.  Injection: for every data state
psi, reading 0 leaves T psi / sqrt2 and reading 1, after S, leaves e^{i pi/4} T psi / sqrt2,
each with probability 1/2 (Result 16.16); in pictures each branch is e^{i a pi/4} T.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import I2, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch16 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
Y = 1j * X @ Z
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
PAULI = [X, Y, Z]
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
W8 = np.exp(1j * np.pi / 4)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)

BOWTIE = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]


def rotation(U):
    return np.array([[np.trace(PAULI[i] @ U @ PAULI[j] @ U.conj().T).real / 2
                      for j in range(3)] for i in range(3)])


def the_24():
    found, frontier = {}, [I2]
    k = lambda U: tuple(np.rint(rotation(U)).astype(int).ravel())
    found[k(I2)] = I2
    while frontier:
        nxt = []
        for U in frontier:
            for M in (H, S):
                V = M @ U
                if k(V) not in found:
                    found[k(V)] = V
                    nxt.append(V)
        frontier = nxt
    return list(found.values())


def bloch(psi):
    return np.array([np.vdot(psi, P @ psi).real for P in PAULI])


def orbit(vec, rots):
    return {tuple(np.round(R @ vec, 9)) for R in rots}


def blackboard():
    rng = np.random.default_rng(1605)

    # --- Result 16.15: T X T^dagger = (X + Y)/sqrt2, not a phase times a Pauli
    img = T @ X @ T.conj().T
    assert np.allclose(img, [[0, np.conj(W8)], [W8, 0]])
    assert np.allclose(img, (X + Y) / np.sqrt(2))
    for P in [I2] + PAULI:
        c = np.trace(P.conj().T @ img) / 2                             # the only candidate factor
        assert not np.allclose(img, c * P)
    comps = [abs(np.trace(P @ img) / 2) for P in [I2] + PAULI]
    assert sum(1 for c in comps if c > 1e-9) == 2                       # two Pauli components
    # so T is not among the 24: its rotation is not a signed permutation
    RT = rotation(T)
    assert not np.allclose(RT, np.rint(RT))
    rots24 = [np.rint(rotation(U)) for U in the_24()]
    assert len(rots24) == 24 and not any(np.allclose(RT, R) for R in rots24)
    # two eighths make a quarter; the "pi/8" name is a global phase, not the angle
    assert np.allclose(T @ T, S)
    assert np.allclose(T, np.exp(1j * np.pi / 8) * np.diag([np.exp(-1j * np.pi / 8), np.exp(1j * np.pi / 8)]))
    assert np.allclose(RT @ RT, rotation(S))
    # T turns the X-Y plane by pi/4 (an eighth of a whole turn), and fixes Z
    for phi in rng.uniform(0, 2 * np.pi, size=5):
        lhs = T @ (np.cos(phi) * X + np.sin(phi) * Y) @ T.conj().T
        assert np.allclose(lhs, np.cos(phi + np.pi / 4) * X + np.sin(phi + np.pi / 4) * Y)
    assert np.allclose(T @ Z @ T.conj().T, Z)
    # the model after one eighth: the dark red (+X) tip goes to (1,1,0)/sqrt2, the pale gold
    # (-Y) tip to (1,-1,0)/sqrt2; the front lug, at (1,0,0), meets the edge between them
    s2 = np.sqrt(2)
    assert np.allclose(RT @ [1, 0, 0], np.array([1, 1, 0]) / s2)
    assert np.allclose(RT @ [0, -1, 0], np.array([1, -1, 0]) / s2)
    mid = RT @ [1, 0, 0] + RT @ [0, -1, 0]
    assert np.allclose(mid / np.linalg.norm(mid), [1, 0, 0])
    assert not np.allclose(RT @ [1, 0, 0], [1, 0, 0])                  # not seated
    # the up and down tips stay; the four round the middle all land on edge midpoints
    assert np.allclose(RT @ [0, 0, 1], [0, 0, 1]) and np.allclose(RT @ [0, 0, -1], [0, 0, -1])
    for tip in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]):
        v = RT @ tip
        assert sorted(np.round(np.abs(v), 9)) == sorted(np.round([0, 1 / s2, 1 / s2], 9))
    # the eighth is the quarter's way round: after two, the front lug holds the pale gold,
    # as after one quarter
    assert np.allclose(RT @ RT @ [0, -1, 0], [1, 0, 0])
    assert np.allclose(rotation(S) @ [0, -1, 0], [1, 0, 0])

    # --- Bravyi-Kitaev's counts on the model: an edge midpoint has 12 images under the 24,
    #     a face centre 8.  |mu> = T|+> points at the edge midpoint between +X and +Y.
    edge, face = np.array([1, 1, 0]) / s2, np.array([1, 1, 1]) / np.sqrt(3)
    assert len(orbit(edge, rots24)) == 12 and len(orbit(face, rots24)) == 8
    mu = T @ np.array([1, 1]) / s2
    assert np.allclose(mu, np.array([1, W8]) / s2)
    assert np.allclose(bloch(mu), edge)
    # their T = e^{i pi/4} K H (K = S) is a Clifford: a third of a turn cycling the axes,
    # X -> Z -> Y -> X (their eq. (8)); its eigenstates (T-type) point at face centres
    Tbk = W8 * S @ H
    assert np.allclose(Tbk @ X @ Tbk.conj().T, Z) and np.allclose(Tbk @ Z @ Tbk.conj().T, Y)
    assert np.allclose(Tbk @ Y @ Tbk.conj().T, X)
    assert np.allclose(np.linalg.matrix_power(np.rint(rotation(Tbk)), 3), np.eye(3))
    for vec in np.linalg.eig(Tbk)[1].T:
        b = bloch(vec / np.linalg.norm(vec))
        assert np.allclose(np.abs(b), 1 / np.sqrt(3))                   # a face centre
    # their |H> (an eigenstate of H) is at an edge midpoint, and |mu> is in its orbit: H-type
    Hvec = np.linalg.eigh(H)[1][:, 1]
    assert np.allclose(np.abs(bloch(Hvec)), np.array([1, 0, 1]) / s2)
    assert tuple(np.round(edge, 9)) in orbit(bloch(Hvec), rots24)
    # their Lambda(e^{-i pi/4}) = diag(1, e^{-i pi/4}) is our T^dagger; |A_{pi/4}> is |mu>
    assert np.allclose(np.diag([1, np.exp(-1j * np.pi / 4)]), T.conj().T)
    # van de Wetering's magic state |0> + e^{i pi/4}|1> is sqrt2 |mu>
    assert np.allclose(np.array([1, W8]), s2 * mu)

    # --- the eighth-capped bow tie: 32 terms on the blackboard
    val = sum(W8 ** bin(x).count("1") * (-1) ** sum(((x >> (4 - b)) & 1) * ((x >> (4 - c)) & 1)
                                                    for b, c in BOWTIE) for x in range(32)) / 8
    assert np.isclose(val, (3 * s2 - 4) / 8 + (s2 - 2) / 8 * 1j)

    # --- Result 16.16, injection, on random non-real data states: control on the data wire
    #     (top), target on the magic wire; read the magic wire
    for _ in range(5):
        psi = random_state(rng)
        assert not np.allclose(psi, psi.conj())                         # non-real
        a_, b_ = psi
        out = CNOT @ np.kron(psi, mu)
        assert np.allclose(out, (np.kron(np.array([a_, W8 * b_]), ket(0))
                                 + np.kron(np.array([W8 * a_, b_]), ket(1))) / s2)
        dark = np.kron(I2, ket(0).reshape(1, 2)) @ out                  # project the magic wire on <0|
        lit = np.kron(I2, ket(1).reshape(1, 2)) @ out                   # on <1|
        assert np.allclose(dark, T @ psi / s2)
        assert np.allclose(lit, W8 * T.conj().T @ psi / s2)             # the eighth the wrong way
        assert np.allclose(S @ lit, W8 * T @ psi / s2)                  # a quarter mends it
        assert np.isclose(np.linalg.norm(dark) ** 2, 0.5) and np.isclose(np.linalg.norm(lit) ** 2, 0.5)
        # negatives: without the quarter, or with the quarter turned back, it is not T psi
        for wrong in (lit, S.conj().T @ lit):
            k = np.vdot(T @ psi, wrong)
            assert not np.allclose(wrong, k * T @ psi)
        # the magic state is not a basis state: with |+> in its place nothing lands
        plain = CNOT @ np.kron(psi, np.array([1, 1]) / s2)
        assert np.allclose(np.kron(I2, ket(0).reshape(1, 2)) @ plain, psi / s2)
    assert np.allclose(S @ T.conj().T, T)

    pictures()


# --- the sketchbook ----------------------------------------------------------------------

def green_on_wire(phase):
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    v = g.add_vertex(VT.Z, 0, 1, phase=Fraction(phase))
    o = g.add_vertex(VT.BOUNDARY, 0, 2)
    g.add_edge((i, v))
    g.add_edge((v, o))
    g.set_inputs((i,))
    g.set_outputs((o,))
    return g


def injection(a, mend=True):
    """One branch: the magic point (green pi/4, one leg) on the lower wire; the green-red pair
    (green on the data wire, red on the magic wire); the red effect a*pi on the magic wire;
    the green a*pi/2 on the data wire (if mend).  One input, one output: the data wire."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    ctrl = g.add_vertex(VT.Z, 0, 2)
    corr = g.add_vertex(VT.Z, 0, 3, phase=Fraction(a, 2) if mend else 0)
    o = g.add_vertex(VT.BOUNDARY, 0, 5)
    point = g.add_vertex(VT.Z, 1, 0, phase=Fraction(1, 4))
    tgt = g.add_vertex(VT.X, 1, 2)
    eff = g.add_vertex(VT.X, 1, 3, phase=Fraction(a))
    for e in [(i, ctrl), (ctrl, corr), (corr, o), (ctrl, tgt), (point, tgt), (tgt, eff)]:
        g.add_edge(e)
    g.set_inputs((i,))
    g.set_outputs((o,))
    return g, dict(ctrl=ctrl, corr=corr, point=point, tgt=tgt, eff=eff)


def capped_bowtie(phase):
    g = zx.Graph()
    bells = [g.add_vertex(VT.Z, w, 1) for w in range(5)]
    for b, c in BOWTIE:
        g.add_edge((bells[b], bells[c]), ET.HADAMARD)
    for w in range(5):
        c = g.add_vertex(VT.Z, w, 2, phase=Fraction(phase))
        g.add_edge((bells[w], c))
    g.set_inputs(())
    g.set_outputs(())
    return g


def sketchbook():
    s2 = np.sqrt(2)

    # --- T is a green pi/4 on a wire, exactly; two of them fuse into a quarter, S
    assert np.allclose(matrix_of(green_on_wire(Fraction(1, 4))), T)
    assert not np.allclose(matrix_of(green_on_wire(Fraction(1, 4))), S)
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    t1 = g.add_vertex(VT.Z, 0, 1, phase=Fraction(1, 4))
    t2 = g.add_vertex(VT.Z, 0, 2, phase=Fraction(1, 4))
    o = g.add_vertex(VT.BOUNDARY, 0, 3)
    for e in [(i, t1), (t1, t2), (t2, o)]:
        g.add_edge(e)
    g.set_inputs((i,))
    g.set_outputs((o,))
    assert simp.check_fuse(g, t1, t2) and simp.fuse(g, t1, t2)
    assert g.phase(t1) == Fraction(1, 2) and same_picture(g, green_on_wire(Fraction(1, 2)))
    assert np.allclose(matrix_of(g), S)
    # an eighth is not a quarter-turn picture: its turn is not a multiple of a quarter
    tg = green_on_wire(Fraction(1, 4))
    assert any(Fraction(tg.phase(u)).denominator > 2 for u in tg.vertices())

    # --- the eighth stops the shrink: the bow tie capped with green eighth-turn effects
    P = capped_bowtie(Fraction(1, 4))
    val = matrix_of(P).item()
    assert np.isclose(val, (3 * s2 - 4) / 8 + (s2 - 2) / 8 * 1j)
    h = P.copy()
    simp.clifford_simp(h)                                               # quarters and halves only
    left = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
    assert len(left) == 5                                               # all five eighths inside
    assert all(h.type(v) == VT.Z and h.phase(v) == Fraction(1, 4) for v in left)
    assert np.isclose(matrix_of(h).item(), val)                         # value kept
    # the same picture with quarter caps shrinks to nothing (CrabsLedger.py): 1 + i
    q = capped_bowtie(Fraction(1, 2))
    simp.clifford_simp(q)
    assert q.num_vertices() == 0 and np.isclose(q.scalar.to_number(), 1 + 1j)

    # --- random closed quarter-and-eighth circuits: <0000| C |0000>.  clifford_simp keeps
    #     the value; wherever it stops short of a number, an eighth is what it stopped at.
    #     (On some circuits it does reach a number: there the pull-outs remove every
    #     neighbour of each eighth, and an eighth left with no legs is simply a number,
    #     1 + e^{i pi/4}.  No eighth is ever pulled out.)
    stuck = 0
    for seed in range(1, 21):
        c = zx.generate.cliffordT(4, 40, p_t=0.1, seed=seed)
        g = c.copy()
        g.apply_state("0000")
        g.apply_effect("0000")
        amp = matrix_of(g).item()
        assert np.isclose(amp, matrix_of(c)[0, 0])
        h = g.copy()
        simp.clifford_simp(h)
        if h.num_vertices() == 0:
            assert np.isclose(h.scalar.to_number(), amp)
        else:
            assert np.isclose(matrix_of(h).item(), amp)
            assert any(Fraction(h.phase(v)).denominator == 4 for v in h.vertices())
            stuck += 1
    assert stuck >= 10

    # --- Result 16.16 in pictures, branch by branch: magic point, green-red pair, red
    #     effect a*pi, green a*pi/2 = e^{i a pi/4} T; that is sqrt2 times the blackboard's branch
    mu = T @ np.array([1, 1]) / s2
    for a in (0, 1):
        g, v = injection(a)
        assert np.allclose(matrix_of(g), W8 ** a * T)
        branch = np.linalg.matrix_power(S, a) @ np.kron(I2, ket(a).reshape(1, 2)) @ CNOT @ np.kron(I2, mu.reshape(2, 1))
        assert np.allclose(matrix_of(g), s2 * branch)
    # without the mending quarter the lit branch is e^{i pi/4} T^dagger, not T
    g, _ = injection(1, mend=False)
    assert np.allclose(matrix_of(g), W8 * T.conj().T)
    assert not np.allclose(matrix_of(g), W8 * T)

    # --- DERIVED, reading 0: spider_simp fuses the red effect into the red dot; id_simp
    #     removes the phase-free two-legged red dot (a plain wire); spider_simp fuses the
    #     magic point into the data wire.  One green pi/4 is left: T, no number.
    g, v = injection(0)
    h = g.copy()
    simp.spider_simp(h)
    assert same_picture(h, g)
    assert simp.id_simp(h)
    simp.spider_simp(h)
    assert same_picture(h, g) and same_picture(h, green_on_wire(Fraction(1, 4)))
    inner = [u for u in h.vertices() if h.type(u) != VT.BOUNDARY]
    assert len(inner) == 1 and h.phase(inner[0]) == Fraction(1, 4)

    #     reading 1: the red effect fuses into the red dot, a red pi with two legs (X).  Pushed
    #     into the magic point (pi-commutation with its number, Ch 14): the point turns to
    #     -pi/4 and e^{i pi/4} is set aside.  Then the point fuses into the data wire, and the
    #     green quarter on top: -pi/4 + pi/2 = pi/4.
    g, v = injection(1)
    h = g.copy()
    assert simp.check_fuse(h, v["tgt"], v["eff"]) and simp.fuse(h, v["tgt"], v["eff"])
    assert h.type(v["tgt"]) == VT.X and h.phase(v["tgt"]) == 1 and h.vertex_degree(v["tgt"]) == 2
    assert same_picture(h, g)
    h.remove_vertex(v["tgt"])                                           # X on the point...
    h.add_edge((v["point"], v["ctrl"]))
    h.set_phase(v["point"], Fraction(-1, 4))                            # ...turns it the other way
    h.scalar.add_phase(Fraction(1, 4))                                  # and sets aside e^{i pi/4}
    assert same_picture(h, g)
    assert simp.check_fuse(h, v["ctrl"], v["point"]) and simp.fuse(h, v["ctrl"], v["point"])
    assert h.phase(v["ctrl"]) == Fraction(-1, 4) % 2
    assert simp.check_fuse(h, v["ctrl"], v["corr"]) and simp.fuse(h, v["ctrl"], v["corr"])
    assert h.phase(v["ctrl"]) == Fraction(1, 4)
    assert same_picture(h, g)
    ref = green_on_wire(Fraction(1, 4))
    ref.scalar.add_phase(Fraction(1, 4))
    assert same_picture(h, ref)                                         # e^{i pi/4} T
    # the push without its number would be wrong
    bare = h.copy()
    bare.scalar.add_phase(Fraction(-1, 4))
    assert not same_picture(bare, g)



# --- the prop pictures --------------------------------------------------------------------

ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth")
TIPPAULI = {"dark red": X, "pale red": -X, "dark gold": Y, "pale gold": -Y, "dark green": Z, "pale green": -Z}
PLACEPAULI = {"front": X, "right": Y, "back": -X, "left": -Y, "up": Z, "socket": -Z}


def eighth_net(I):
    """The cords an \\eighthcloth state (of the eighth-end scene) stands for: the upper cord,
    input to output, through the green ring (phase turn/12) and any ring beside it; the lower
    cord from the eighth-end (a green point, phase turn/12) through the red joiner ring to its
    far end -- loose (an output), or closed by a red end-ring (a red effect) -- the joiner
    tied to the green ring by a plain length; once the joiner is lifted the eighth-end's cord
    runs straight to the green ring; once slid up, nothing is left below.  Outputs top first."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    up = g.add_vertex(VT.Z, 0, 3, phase=Fraction(I.green, 12))
    g.add_edge((i, up))
    last = up
    if I.beside is not None:
        bz = g.add_vertex(VT.Z, 0, 4, phase=Fraction(I.beside, 12))
        g.add_edge((last, bz))
        last = bz
    o = g.add_vertex(VT.BOUNDARY, 0, 8)
    g.add_edge((last, o))
    outs = [o]
    if I.lower != "gone":
        assert I.eighth is not None                             # the eighth-end scene only
        pt_ = g.add_vertex(VT.Z, 1, 0, phase=Fraction(I.eighth, 12))
        if I.lower == "joined":
            g.add_edge((pt_, up))
        else:
            red = g.add_vertex(VT.X, 1, 3, phase=Fraction(I.red, 12))
            g.add_edge((pt_, red))
            g.add_edge((up, red))
            if I.lower == "full":
                if I.far is not None:
                    far = g.add_vertex(VT.X, 1, 7, phase=Fraction(I.far, 12))
                    g.add_edge((red, far))
                else:
                    lo = g.add_vertex(VT.BOUNDARY, 1, 8)
                    g.add_edge((red, lo))
                    outs.append(lo)
    g.set_inputs((i,))
    g.set_outputs(tuple(outs))
    assert g.num_edges() == g.num_vertices() - 1                      # a tree: every strand once
    return g


def pictures():
    """The Fifth Challenge's prop pictures (identity card, section 4), read from the dialogue
    through tools/dfp_props.py and the transcription tools/props_ch16.py, and the checks that
    run over the whole dialogue.  The ties below hold the transcription against this file's
    blackboard: a structural check of the pictures, not a second proof."""
    import props_ch14
    s2 = np.sqrt(2)
    calls = DIALOGUE.check_layout("Fifth", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("cradlemodel", False, "an eighth", ("home",), "Rocking, crooked"),
        ("eighthcloth", True, "", ("the eighth-end",), "The joiner, ready"),
        ("eighthcloth", False, "the dark reading", ("the eighth-end",), "Carried up"),
        ("eighthcloth", False, "the lit reading", ("the eighth-end",), "Carried up again")]
    PC.check_captions(DIALOGUE, "Fifth", calls)
    PC.check_section(DIALOGUE, "Fifth", calls)
    PC.check_anchors(DIALOGUE, "Fifth", ["P5.1", "P5.2", "P5.3", "P5.4"])
    assert [[round(PC.height(c), 2) for c in b] for b in DIALOGUE.blocks("Fifth")] == [[2.07], [1.78], [1.78], [1.78]]
    p51, p52, p53, p54 = (PC.picture(c) for c in calls)

    # P5.1: an eighth from home -- U = T; the lug at P holds T^dagger P T: up and socket keep
    # their tips, every other lug lies between two tips, (P1 + P2)/sqrt2, as the dump names them
    assert p51.dump == ("unseated up=dark green socket=pale green front=between dark red and pale gold "
                        "right=between dark gold and dark red back=between pale red and dark gold "
                        "left=between pale gold and pale red")
    assert p51.state.eighth == 1 and not p51.shown.seated              # "Rocking, crooked"
    seen = dict(kv.split("=") for kv in re.split(r" (?=\w+=)", p51.dump.split(" ", 1)[1]))
    assert set(seen) == set(PLACEPAULI)
    for place, P_ in PLACEPAULI.items():
        held = T.conj().T @ P_ @ T
        if place in ("up", "socket"):
            assert np.allclose(held, TIPPAULI[seen[place]])
        else:
            a, b = seen[place][len("between "):].split(" and ")
            assert np.allclose(held, (TIPPAULI[a] + TIPPAULI[b]) / s2), place
            assert not any(np.allclose(held, s * Q) for s in (1, -1) for Q in PAULI)   # no tip in it
    # the conjugation the wrong way round (where the lug's tip goes) names other tips
    assert not np.allclose(T @ X @ T.conj().T, (X - Y) / s2)
    # two eighths are a quarter: seated again, as one quarter
    assert PC.model_dump(PC.model_state("an eighth, an eighth", "home")) == PC.model_dump(PC.model_state("quarter", "home"))
    assert p51.labels == ()

    # P5.2: the joiner, ready -- the eighth-end's bead at an eighth is the green pi/4 point
    # |0> + e^{i pi/4}|1>; the far end loose
    assert p52.dump.split(" | ")[0] == "green=0 red=0 lower=full eighth=3 far=none beside=none aside=0"
    point = zx.Graph()
    v = point.add_vertex(VT.Z, 0, 0, phase=Fraction(p52.state.eighth, 12))
    o = point.add_vertex(VT.BOUNDARY, 0, 1)
    point.add_edge((v, o))
    point.set_outputs((o,))
    assert np.allclose(matrix_of(point).ravel(), [1, W8]) and np.allclose(matrix_of(point).ravel(), s2 * T @ PLUS)
    assert not np.allclose(matrix_of(point).ravel(), [1, 1j])           # an eighth, not a quarter
    net52 = eighth_net(p52.state)
    assert len(net52.outputs()) == 2
    # the readings: the far end closed by a red end-ring at rest is the effect sqrt2 <0|, at a
    # half sqrt2 <1|
    M52 = matrix_of(net52)
    dark_I, lit_I = PC.eighth_state("close dark", "the eighth-end"), PC.eighth_state("close lit", "the eighth-end")
    assert np.allclose(np.kron(I2, s2 * ket(0).reshape(1, 2)) @ M52, matrix_of(eighth_net(dark_I)))
    assert np.allclose(np.kron(I2, s2 * ket(1).reshape(1, 2)) @ M52, matrix_of(eighth_net(lit_I)))

    # P5.3: the dark reading -- the eighth carried up onto the upper cord: exactly T psi
    assert p53.dump.split(" | ")[0] == "green=3 red=none lower=gone eighth=none far=none beside=none aside=0"
    rng = np.random.default_rng(1615)
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())                             # non-real
    assert np.allclose(matrix_of(eighth_net(p53.state)) @ psi, T @ psi)
    assert same_picture(eighth_net(p53.state), eighth_net(dark_I)) and same_picture(eighth_net(p53.state), injection(0)[0])
    # P5.4: the lit reading, the quarter beside it slid in -- e^{i pi/4} T psi: the counter's
    # number is e^{i pi/4}
    assert p54.dump.split(" | ")[0] == "green=3 red=none lower=gone eighth=none far=none beside=none aside=1"
    assert np.allclose(W8 * matrix_of(eighth_net(p54.state)) @ psi, S @ matrix_of(eighth_net(lit_I)) @ psi)
    assert np.allclose(W8 * matrix_of(eighth_net(p54.state)) @ psi, W8 * T @ psi)
    assert not np.allclose(matrix_of(eighth_net(lit_I)) @ psi, W8 * T @ psi)          # not without the quarter
    assert not np.allclose(matrix_of(eighth_net(p54.state)) @ psi, S @ matrix_of(eighth_net(lit_I)) @ psi)  # nor without the counter
    # every step of either reading keeps the value, with e^{i pi/4} for each counter (the
    # quarter beside it is the mending S, applied, not a rewrite)
    steps = {"dark": ["close dark", "slide the end-ring in", "lift the red ring", "slide the eighth-end up"],
             "lit": ["close lit", "slide the end-ring in", "push the half into the eighth-end", "slide the eighth-end up"]}
    for name, seq in steps.items():
        first = None
        for k in range(1, len(seq) + 1):
            I = PC.eighth_state(", ".join(seq[:k]), "the eighth-end")
            assert not PC.TEX_ERRORS
            val = W8 ** I.aside * matrix_of(eighth_net(I))
            first = val if first is None else first
            assert np.allclose(val, first), (name, seq[:k])
    for p in (p53, p54):                                                # "Carried up (again)"
        assert (p.state.green, p.state.lower, p.state.red, p.state.eighth) == (3, "gone", None, None)
    assert (p53.state.aside, p54.state.aside) == (0, 1)

    # the keyed words (P5.2): end-ring, tie and cord set by the cloth kit (not transcribed;
    # measured from the Lua's output after the kit's TOPRAISE, their letters clear the cloth's
    # parts and the linen's edge by at least 0.019 cm, "cord" to its own cord the least); "ring"
    # set by the Lua itself below the red joiner: its letters clear every ring and cord drawn,
    # and the linen's edges, by at least 0.02 cm
    assert p52.labels == ("end-ring", "tie", "cord", "ring") and p53.labels == p54.labels == ()
    (w, x, y), = p52.shown.own
    letters = PC.letter_boxes(w, x, y, "center")
    rings, cords = props_ch14.cloth_drawn(p52)
    joiner = [r for r in rings if p52.cloth["things"][r[0] - 1]["c"] == "red"]
    assert len(joiner) == 1 and abs(joiner[0][2] - x) < 1e-9 and joiner[0][3] > y        # straight below it
    # (measured: the red joiner 0.038, the lower cord 0.097, the linen's edge 0.157)
    assert all(min(PC.box_to_point(b, (X_, Y_)) for b in letters) >= reach + 0.02 for _, _, X_, Y_, _, reach in rings)
    assert all(min(PC.box_to_polyline(b, pts) for b in letters) >= reach + 0.02 for _, _, _, pts, reach in cords)
    left, bottom, right, top = p52.shown.linen
    assert min(min(b[0] - left, right - b[2], b[1] - bottom, top - b[3]) for b in letters) >= 0.02
    # the counter (P5.4): one, by the teapot, off the linen and inside the frame
    assert p53.shown.counters == () and len(p54.shown.counters) == 1
    box = PC.counter_box(*p54.shown.counters[0])
    assert box[3] < p54.shown.linen[1] - PC.LINEN_SHADOW and box[1] > p54.shown.fbot - PC.ASIDE
    # fixed parts: the upper cord runs along the first row in every picture, the lower cord
    # (while there) along the second
    for p in (p52, p53, p54):
        assert {"kind": "cord", "pts": [[0, 0], [7, 0]], "a": "loose", "b": "loose"} in eighth_scene_cords(p)
    assert any(c["kind"] == "cord" and c["pts"] == [[0, 1], [7, 1]] for c in eighth_scene_cords(p52))
    assert not PC.TEX_ERRORS

    # --- the whole dialogue -------------------------------------------------------------
    assert DIALOGUE.check_whole(ORDINALS) == 20
    DIALOGUE.keyed_first(ORDINALS)                                      # the starred form: first of each macro
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        raw = pathlib.Path(path).read_text(encoding="utf-8")
        text = DIALOGUE.read(path)                                      # comments stripped
        for phrase in PC.NEG:                                           # the negative anchors
            assert phrase not in text, (path.name, phrase)
        ok, found = PC.form_check(raw)                                  # card section 1c's one-liner
        assert ok and len(found["seatings"]) == 24 and found["eighth"] == 1, found
        assert not re.search(r"\b(sixteen|sixteenth|16)\b", raw, re.I)  # the signal grep
        assert raw.count("Let Bob draw") == 0
        assert not re.search(r"^\*\*Genie", raw, re.M)
    # (the spoken "and sets a little something aside" of the Tortoise stays: no anchor matches it)
    assert "and sets a little something aside" in DIALOGUE.read(DIALOGUE.dialog)


def eighth_scene_cords(pic):
    return pic.scene["cords"]


if __name__ == "__main__":
    verdict("AnEighth", blackboard, sketchbook)
