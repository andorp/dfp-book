"""Challenge III: The Turn the Bells Miss  (phase flips; colour change; Shor's nine)
====================================================================================
"Cloudy! ...Oh. The bells never heard a thing, and it's spoilt all the same."  -- Achilles
"Then my storm line is half deaf."  -- Crab

Claim: a phase flip passes the bit-flip code unheard -- Enc^dag X^c(s) P_s Z_k Enc = [s = 00] Z
(Result 18.12), and on The Storm Line a green half on a tell ring reduces to a green half on
a wire, number 1.  Colour change turns the green encoder red: the red 1 -> 3 spider is
H^(x)3 Enc H, the phase-flip code (Result 18.13).  Shor's encoder Enc9 = Enc^(x)3 H^(x)3 Enc is
green, yellow, green, exactly; its eight checks commute and fix it; the 28 single-wire
Paulis give 22 syndromes and are each corrected up to a global phase r with |r| = 1
(Result 18.14); and by linearity every error on one wire is corrected (Gottesman's Theorem 2).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction
from itertools import product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import I2, ket, kron, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch18 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
Y = 1j * X @ Z                                                         # Gottesman's Table 1
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}
I8 = np.eye(8)

ENC = np.outer(ket(0, 0, 0), ket(0)) + np.outer(ket(1, 1, 1), ket(1))
DEC = ENC.conj().T
H3 = kron(H, H, H)


def on(k, A, n=3):
    ops = [I2] * n
    ops[k] = A
    return kron(*ops)


def pattern(A, e):
    return kron(*[A if b else I2 for b in e])


Z1Z2, Z2Z3 = on(0, Z) @ on(1, Z), on(1, Z) @ on(2, Z)


def P(s):
    s1, s2 = s
    return (I8 + (-1) ** s1 * Z1Z2) @ (I8 + (-1) ** s2 * Z2Z3) / 4


def sigma(e):
    return (e[0] ^ e[1], e[1] ^ e[2])


CORR = {(0, 0): (0, 0, 0), (1, 0): (1, 0, 0), (1, 1): (0, 1, 0), (0, 1): (0, 0, 1)}
SINGLE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
OUTCOMES = list(product((0, 1), repeat=2))

MARKS = {
    "dark": ket(0), "lit": ket(1),
    "at rest": (ket(0) + ket(1)) / np.sqrt(2), "a half": (ket(0) - ket(1)) / np.sqrt(2),
    "a quarter round": (ket(0) + 1j * ket(1)) / np.sqrt(2),
    "a quarter back": (ket(0) - 1j * ket(1)) / np.sqrt(2),
}

# --- Shor's nine (Section 18.3); nine wires, wire 1 on top
ENC9 = kron(ENC, ENC, ENC) @ H3 @ ENC                                   # 512 x 2
TABLE2 = ["ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII", "IIIIIIZZI", "IIIIIIIZZ",
          "XXXXXXIII", "IIIXXXXXX"]                                     # Gottesman's Table 2


def apply(word, M):
    """Apply the Pauli word (one letter per wire, wire 1 first) to the columns of M."""
    n = len(word)
    T = M.reshape((2,) * n + (-1,))
    for k, letter in enumerate(word):
        if letter != "I":
            T = np.moveaxis(np.tensordot(PAULI[letter], T, axes=([1], [k])), 0, k)
    return T.reshape(2**n, -1)


def single(letter, k, n=9):
    return "".join(letter if j == k else "I" for j in range(n))


def symp(word):
    """Binary (x | z) form: x where X or Y, z where Y or Z."""
    return (np.array([c in "XY" for c in word], dtype=int), np.array([c in "YZ" for c in word], dtype=int))


def syndrome(word, gens=TABLE2):
    x, z = symp(word)
    return tuple(int((x @ gz + z @ gx) % 2) for gx, gz in map(symp, gens))


def blackboard():
    rng = np.random.default_rng(1821)
    # Z, X, H, Enc and the checks are real; the messages and the one-wire errors are the
    # non-real, generic inputs.

    # --- Result 18.12: a phase flip passes unheard, whichever wire, for every outcome
    for k in range(3):
        assert np.allclose(DEC @ on(k, Z) @ ENC, Z)
        for s in OUTCOMES:
            lhs = DEC @ pattern(X, CORR[s]) @ P(s) @ on(k, Z) @ ENC
            assert np.allclose(lhs, (s == (0, 0)) * Z)
        assert not np.allclose(DEC @ on(k, Z) @ ENC, I2)          # it is NOT mended
    # the Tortoise's trap on the storm line: a green half on the middle tell ring
    def glass(msg):
        hit = on(1, Z) @ ENC @ msg
        clear, rang = 0.0, 0.0
        for s in OUTCOMES:
            branch = pattern(X, CORR[s]) @ P(s) @ hit
            w = np.linalg.norm(branch) ** 2
            rang += w * (s != (0, 0))
            if w > 1e-12:
                clear += abs(np.vdot(msg, DEC @ branch)) ** 2
        return clear, rang
    for name, msg in MARKS.items():
        clear, rang = glass(msg)
        assert np.isclose(rang, 0)                                 # both bells silent, always
        assert np.isclose(clear, 1 if name in ("dark", "lit") else 0), name   # turns: cloudy

    # --- Result 18.13: the phase-flip code, H^(x)3 Enc H, checks X1X2 and X2X3
    ENCp = H3 @ ENC @ H
    Pp = {s: H3 @ P(s) @ H3 for s in OUTCOMES}
    X1X2, X2X3 = on(0, X) @ on(1, X), on(1, X) @ on(2, X)
    for s in OUTCOMES:
        s1, s2 = s
        assert np.allclose(Pp[s], (I8 + (-1) ** s1 * X1X2) @ (I8 + (-1) ** s2 * X2X3) / 4)
    for e in SINGLE:
        for s in OUTCOMES:
            lhs = ENCp.conj().T @ pattern(Z, CORR[s]) @ Pp[s] @ pattern(Z, e) @ ENCp
            assert np.allclose(lhs, (s == sigma(e)) * I2)
    plus, minus = MARKS["at rest"], MARKS["a half"]
    RED = H3 @ ENC @ H                                             # what the red spider will be
    assert np.allclose(RED @ plus, kron(plus, plus, plus).ravel())
    assert np.allclose(RED @ minus, kron(minus, minus, minus).ravel())
    for k in range(3):                                             # and it is blind to bit flips
        for s in OUTCOMES:
            lhs = ENCp.conj().T @ pattern(Z, CORR[s]) @ Pp[s] @ on(k, X) @ ENCp
            assert np.allclose(lhs, (s == (0, 0)) * X)
    # Knill and Laflamme's eq. (15) encodes |0>,|1> to |+++>,|--->; the red spider sends
    # |+>,|-> there: the two differ by the one yellow box on the input
    KL15 = np.column_stack([kron(plus, plus, plus).ravel(), kron(minus, minus, minus).ravel()])
    assert np.allclose(KL15, ENCp @ H) and not np.allclose(KL15, ENCp)

    # --- Shor's nine: Gottesman's eqs. (9)-(10), divided by 2 sqrt2
    ghz_p, ghz_m = ket(0, 0, 0) + ket(1, 1, 1), ket(0, 0, 0) - ket(1, 1, 1)
    zero_L = kron(ghz_p, ghz_p, ghz_p).ravel()                     # unnormalized, as printed
    one_L = kron(ghz_m, ghz_m, ghz_m).ravel()
    assert np.allclose(ENC9[:, 0], zero_L / (2 * np.sqrt(2)))
    assert np.allclose(ENC9[:, 1], one_L / (2 * np.sqrt(2)))
    assert np.allclose(ENC9.conj().T @ ENC9, I2)
    # the eight checks of Table 2 commute and fix every Enc9|psi>
    for a in TABLE2:
        for b in TABLE2:
            xa, za = symp(a)
            xb, zb = symp(b)
            assert (xa @ zb + za @ xb) % 2 == 0
        assert np.allclose(apply(a, ENC9), ENC9)

    # --- the 28 single-wire operators: 22 syndromes, 1 + 9 + 3 + 9
    ops = [("I", 0)] + [(L, k) for L in "XYZ" for k in range(9)]
    assert len(ops) == 28
    syn = {(L, k): syndrome(single(L, k)) for L, k in ops}
    cls = {L: {syn[(L, k)] for k in range(9)} for L in "XYZ"}
    assert len(set(syn.values())) == 22
    assert len(cls["X"]) == 9 and len(cls["Y"]) == 9 and len(cls["Z"]) == 3
    assert syndrome("IIIIIIIII") not in cls["X"] | cls["Y"] | cls["Z"]
    assert not (cls["X"] & cls["Y"]) and not (cls["X"] & cls["Z"]) and not (cls["Y"] & cls["Z"])
    assert syn[("Z", 0)] == syn[("Z", 1)] == syn[("Z", 2)]           # degenerate: one block, one answer
    # each operator lands wholly in its syndrome's eigenspace (numerically, not just by counting)
    for L, k in ops:
        EM = apply(single(L, k), ENC9)
        for g, bit in zip(TABLE2, syn[(L, k)]):
            assert np.allclose(apply(g, EM), (-1) ** bit * EM)

    # the mending, as Section 18.3 describes it: the six inner bits name the wire to flip
    # back with X; the two outer bits name the block, whose first wire gets a Z
    x_table = {syndrome(single("X", k))[:6]: k for k in range(9)}
    z_table = {syndrome(single("Z", 3 * b))[6:]: 3 * b for b in range(3)}

    def correction(sy):
        word = ["I"] * 9
        if any(sy[:6]):
            word[x_table[sy[:6]]] = "X"
        if any(sy[6:]):
            k = z_table[sy[6:]]
            word[k] = "Y" if word[k] == "X" else "Z"
        return word

    def mend(sy, M):
        """Apply the correction X-part first, then the Z-part (so Z X, not Y)."""
        word = correction(sy)
        xpart = "".join("X" if c in "XY" else "I" for c in word)
        zpart = "".join("Z" if c in "YZ" else "I" for c in word)
        return apply(zpart, apply(xpart, M))

    psi1, psi2 = random_state(rng), random_state(rng)
    rs = set()
    for L, k in ops:
        out = mend(syn[(L, k)], apply(single(L, k), ENC9))
        r = np.vdot(ENC9[:, 0], out[:, 0])
        assert np.isclose(abs(r), 1)                               # up to a global phase |r| = 1
        assert np.allclose(out, r * ENC9)                          # the same r on both codewords...
        for psi in (psi1, psi2):                                   # ...so the same for every psi
            assert np.allclose(out @ psi, r * ENC9 @ psi)
        rs.add(complex(np.round(r, 9)))
    assert rs <= {1, -1, 1j, -1j} and rs != {1}                    # the sign is real, not always 1

    # two wires hit: not corrected (X on wires 1 and 2 looks like X on wire 3)
    two = apply("XXIIIIIII", ENC9)
    out = mend(syndrome("XXIIIIIII"), two)
    r = np.vdot(ENC9[:, 0], out[:, 0])
    assert not np.allclose(out, r * ENC9)                          # a logical flip, not the message

    # --- Gottesman's Theorem 2: every error on any ONE of the nine wires, by linearity.
    # For each of the 2^8 outcomes, the branch mend(P_syn A_k Enc9) is a number times Enc9,
    # and the numbers carry the error's whole weight: sum |lambda|^2 = ||A||_F^2 / 2.
    def project(sy, M):
        for g, bit in zip(TABLE2, sy):
            M = (M + (-1) ** bit * apply(g, M)) / 2
        return M
    all_syn = list(product((0, 1), repeat=8))
    seen = set(syn.values())
    for k in range(9):
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        assert not np.allclose(A, A.real)
        word_A = lambda M: np.moveaxis(np.tensordot(A, M.reshape((2,) * 9 + (-1,)), axes=([1], [k])),
                                       0, k).reshape(512, -1)
        hit = word_A(ENC9)
        weight = 0.0
        for sy in all_syn:
            branch = project(sy, hit)
            if sy not in seen:
                assert np.allclose(branch, 0)                      # outcomes that never happen
                continue
            out = mend(sy, branch)
            lam = np.vdot(ENC9[:, 0], out[:, 0])
            assert np.allclose(out, lam * ENC9)                    # mended: a number times Enc9
            weight += abs(lam) ** 2
        assert np.isclose(weight, np.linalg.norm(A) ** 2 / 2)      # "corrected up to its norm"

    # the single-wire depolarizing channel on the fifth wire, undone exactly on a density matrix
    p = rng.uniform(0.05, 0.5)
    B = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho = B @ B.conj().T
    rho /= np.trace(rho)
    coded = ENC9 @ rho @ ENC9.conj().T
    kraus = [(np.sqrt(1 - p), "I")] + [(np.sqrt(p / 3), L) for L in "XYZ"]
    noisy = sum(abs(c) ** 2 * apply(single(L, 4), apply(single(L, 4), coded).conj().T).conj().T
                for c, L in kraus)
    recovered = np.zeros_like(noisy)
    for sy in seen:
        right = project(sy, noisy)                                 # P rho
        both = project(sy, right.conj().T).conj().T                # P rho P  (P is Hermitian)
        m1 = mend(sy, both)
        recovered += mend(sy, m1.conj().T).conj().T
    assert np.allclose(recovered, coded)
    assert not np.allclose(noisy, coded)

    pictures()


# ------------------------------------------------------------------------------ prop pictures

THIRD_DUMPS = (
    "mend=0,0,0 tell=0,2,0 clip=none glass=cloudy",
    "fresh,pairs,through | cols=8 rows=3 y=0..2 near=false folded=false; 1:ring red 0 @3,1; tube @6,0 a0; "
    "tube @1,1 a0; tube @6,1 a0; tube @6,2 a0; cord loose-1 [7.3,0 4,0 3,1]; cord loose-1 [-0.3,1 3,1]; "
    "cord loose-1 [7.3,1 3,1]; cord loose-1 [7.3,2 4,2 3,1]",
    "nine | cols=9 rows=9 y=0..8 near=false folded=false; 1:ring green 0 @5,1; 2:ring green 0 @1,4; "
    "3:ring green 0 @5,4; 4:ring green 0 @5,7; tube @4,1 a0; tube @4,4 a0; tube @4,7 a0; cord 1-2 [5,1 3,1 1,4]; "
    "cord 2-3 [1,4 5,4]; cord 2-4 [1,4 3,7 5,7]; cord loose-1 [8.3,0 6,0 5,1]; cord loose-1 [8.3,1 5,1]; "
    "cord loose-1 [8.3,2 6,2 5,1]; cord loose-2 [-0.3,4 1,4]; cord loose-3 [8.3,3 6,3 5,4]; cord loose-3 [8.3,4 5,4]; "
    "cord loose-3 [8.3,5 6,5 5,4]; cord loose-4 [8.3,6 6,6 5,7]; cord loose-4 [8.3,7 5,7]; cord loose-4 [8.3,8 6,8 5,7]",
)


def pictures():
    """The Third Challenge's prop pictures (identity card, section 4), read from the dialogue
    through tools/dfp_props.py and the transcription tools/props_ch18.py, and held against this
    file's storm_line, encoder and shor_encoder (graph_of: a structural reading of each cloth
    dump, not a proof)."""
    calls = DIALOGUE.check_layout("Third", sizes=[1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("stormline", False, "middle tell ring at a half, glass cloudy", (), "Turned in passing"),
        ("tubecloth", True, "pairs put on, drawn through", ("a fresh parting ring",), "Come out red"),
        ("tubecloth", False, "", ("nine strands",), "Spread out again")]
    PC.check_captions(DIALOGUE, "Third", calls)
    PC.check_section(DIALOGUE, "Third", calls)
    PC.check_anchors(DIALOGUE, "Third", ["P3.1", "P3.2", "P3.3"])
    pics = [PC.picture(c) for c in calls]
    p31, p32, p33 = pics
    assert tuple(p.dump for p in pics) == THIRD_DUMPS
    # one frame per start (the nine strands are taller), and one storm-line frame
    for start, pic in (("a fresh parting ring", p32), ("nine strands", p33)):
        assert {PC.tubecloth(s, start).frame for s in ("", "pairs put on", "pairs put on, drawn through")
                if start != "nine strands" or s == ""} == {pic.frame}
    assert p33.frame > p32.frame
    assert PC.stormline("").frame == p31.frame

    # P3.1: the turn the bells miss.  The line with the middle tell ring at a half is EXACTLY Z,
    # not the identity, though neither bell's picture changes; a message a quarter round reads
    # cloudy at its mark, dark and lit would have read clear ("had you sent dark or lit")
    L = p31.state
    assert L.tell == [0, 2, 0] and L.mend == [0, 0, 0] and L.glass == "cloudy"   # "Turned in passing"
    tell = tuple(q // 2 for q in L.tell)
    line = matrix_of(storm_line((0, 0, 0), (0, 0), (0, 0, 0), tell))
    assert np.allclose(line, Z) and not np.allclose(line, I2)
    assert np.allclose(line, matrix_of(storm_line((0, 0, 0), (0, 0), (0, 0, 0), (0, 0, 1))))   # any strand
    clear = {name: abs(np.vdot(m, line @ m)) ** 2 for name, m in MARKS.items()}
    assert np.isclose(clear["dark"], 1) and np.isclose(clear["lit"], 1)
    for name in ("at rest", "a half", "a quarter round", "a quarter back"):
        assert np.isclose(clear[name], 0)                         # cloudy, for certain
    # the bells silent: no syndrome, with certainty, whatever the message
    for name, m in MARKS.items():
        assert np.isclose(np.linalg.norm(P((0, 0)) @ on(1, Z) @ ENC @ m), 1)
    # fixed parts, P2.1 -> P3.1: the middle mending ring back at rest, the middle tell ring at a
    # half, the glass lifted again (cloudy); every other ring, bead, cord, bell the same, no clip
    p21 = PC.stormline("middle mending ring at a half, glass clear")
    va, vb = p21.shown, p31.shown
    assert [(a[0], a[4], b[4]) for a, b in zip(va.rings, vb.rings) if a != b] == [("t2", 0, 2), ("m2", 2, 0)]
    assert [a[:4] for a in va.rings] == [b[:4] for b in vb.rings]
    assert [k for k, (a, b) in enumerate(zip(va.beads, vb.beads)) if a != b] == [5, 10]
    assert va.spans == vb.spans and va.knots == vb.knots and va.bells == vb.bells
    assert va.clip == vb.clip == () and (va.glass, vb.glass) == ("clear", "cloudy")
    assert p31.labels == ()

    # P3.2: the fresh parting ring, with its pairs, and drawn through: all three EQUAL, exactly
    # (C3); the last a red ring at rest with a tube on its lead and on each strand
    states = [PC.tubecloth(s, "a fresh parting ring") for s in ("", "pairs put on", "pairs put on, drawn through")]
    assert states[2].dump == p32.dump
    graphs = [PC.graph_of(s.dump) for s in states]
    green, _ = encoder(VT.Z)
    for g in graphs:
        assert (g.num_vertices(), g.num_edges(), g.num_inputs(), g.num_outputs()) == (5, 4, 1, 3)
        assert same_picture(g, green)
    assert np.allclose(matrix_of(graphs[2]), ENC)
    nodes, tubes, ties, cords, extra = PC.parse_dump(p32.dump)
    assert [n[:3] for n in nodes.values()] == [("ring", "red", 0)]   # "Come out red": through, red, at rest
    assert "through" in p32.dump.split(" |")[0] and len(tubes) == 4
    # the ring drawn through, its tubes taken off: the red ring alone, H^(x)3 Enc H, not Enc
    bare = PC.graph_of(re.sub(r" tube @[^;]*;", "", p32.dump))
    assert np.allclose(matrix_of(bare), H3 @ ENC @ H) and not same_picture(bare, green)
    # a ring drawn through only some of its tubes is not the same ring
    half = PC.graph_of(p32.dump.replace(" tube @1,1 a0;", ""))
    assert not same_picture(half, green)
    # the keyed word tube is the cloth kit's own placement (its no-room fault is the check);
    # the Lua sets no word of its own here
    assert p32.labels == ("tube",) and PC.own_words(p32) == () and PC.check_keyed_words(p32) == {}

    # P3.3: the nine strands are Shor's encoder, exactly: one input, nine outputs, thirteen
    # cords; no bells yet (they are laid in the next direction, unpictured)
    g33 = PC.graph_of(p33.dump)
    assert (g33.num_inputs(), g33.num_outputs(), g33.num_edges()) == (1, 9, 13)
    assert same_picture(g33, shor_encoder()) and np.allclose(matrix_of(g33), ENC9)
    assert not same_picture(g33, shor_encoder(ET.SIMPLE))       # its tubes carry it
    assert p33.dump.startswith("nine |") and "hub" not in p33.dump   # "Spread out again"
    assert p33.labels == ()

    # the Lua's refusals: a tell ring at no mark of the dial; a bell drawn ringing (heard, never
    # seen); drawn through before the pairs; the nine strands given steps
    PC.TEX_ERRORS.clear()
    PC.line_state("middle tell ring at a third")
    PC.line_state("first bell ringing")
    PC.tube_state("drawn through", "a fresh parting ring")
    PC.tube_state("pairs put on", "nine strands")
    assert [k for k, _ in PC.TEX_ERRORS] == ["line-tell", "line-bell", "tube-first", "tube-nine"]
    PC.TEX_ERRORS.clear()


# ------------------------------------------------------------------------------ pictures

def storm_line(e, s, c, tell=(0, 0, 0)):
    """The Storm Line of BlindVote.py, rebuilt here (files stand alone)."""
    g = zx.Graph()
    lead_in, lead_out = g.add_vertex(VT.BOUNDARY, 2, 0), g.add_vertex(VT.BOUNDARY, 2, 12)
    parting, joining = g.add_vertex(VT.Z, 2, 1), g.add_vertex(VT.Z, 2, 11)
    weather = [g.add_vertex(VT.X, 2 * k, 3, phase=Fraction(e[k])) for k in range(3)]
    tellr = [g.add_vertex(VT.Z, 2 * k, 5, phase=Fraction(tell[k])) for k in range(3)]
    bells = [g.add_vertex(VT.X, 2 * j + 1, 6, phase=Fraction(s[j])) for j in range(2)]
    mending = [g.add_vertex(VT.X, 2 * k, 9, phase=Fraction(c[k])) for k in range(3)]
    g.add_edge((lead_in, parting))
    for k in range(3):
        g.add_edge((parting, weather[k]))
        g.add_edge((weather[k], tellr[k]))
        g.add_edge((tellr[k], mending[k]))
        g.add_edge((mending[k], joining))
    for t, b in ((0, 0), (1, 0), (1, 1), (2, 1)):
        g.add_edge((tellr[t], bells[b]))
    g.add_edge((joining, lead_out))
    g.set_inputs((lead_in,))
    g.set_outputs((lead_out,))
    return g


def swap_colours(g):
    h = g.copy()
    for v in h.vertices():
        if h.type(v) == VT.Z:
            h.set_type(v, VT.X)
        elif h.type(v) == VT.X:
            h.set_type(v, VT.Z)
    return h


def encoder(kind):
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 1, 0)
    v = g.add_vertex(kind, 1, 1)
    outs = [g.add_vertex(VT.BOUNDARY, k, 3) for k in range(3)]
    g.add_edge((i, v))
    for o in outs:
        g.add_edge((v, o))
    g.set_inputs((i,))
    g.set_outputs(tuple(outs))
    return g, v


def shor_encoder(middle=ET.HADAMARD):
    """Green 1 -> 3, a yellow box (Hadamard edge) on each middle cord, three green 1 -> 3."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 4, 0)
    top = g.add_vertex(VT.Z, 4, 1)
    g.add_edge((i, top))
    outs = []
    for b in range(3):
        inner = g.add_vertex(VT.Z, 3 * b + 1, 3)
        g.add_edge((top, inner), middle)
        for k in range(3):
            o = g.add_vertex(VT.BOUNDARY, 3 * b + k, 5)
            g.add_edge((inner, o))
            outs.append(o)
    g.set_inputs((i,))
    g.set_outputs(tuple(outs))
    return g


def shor_round_trip():
    """Shor's encoder then its mirror.  Each of the nine strands carries a green ring at rest
    (bare cord, row 57), so PyZX keeps nine separate strands."""
    g = zx.Graph()
    i, o = g.add_vertex(VT.BOUNDARY, 4, 0), g.add_vertex(VT.BOUNDARY, 4, 10)
    top, bot = g.add_vertex(VT.Z, 4, 1), g.add_vertex(VT.Z, 4, 9)
    g.add_edge((i, top))
    g.add_edge((bot, o))
    for b in range(3):
        inner, inner_m = g.add_vertex(VT.Z, 3 * b + 1, 3), g.add_vertex(VT.Z, 3 * b + 1, 7)
        g.add_edge((top, inner), ET.HADAMARD)
        g.add_edge((inner_m, bot), ET.HADAMARD)
        for k in range(3):
            ring = g.add_vertex(VT.Z, 3 * b + k, 5)
            g.add_edge((inner, ring))
            g.add_edge((ring, inner_m))
    g.set_inputs((i,))
    g.set_outputs((o,))
    return g


def sketchbook():
    # --- Result 18.12 as a picture, DERIVED: The Storm Line with a green half on a tell ring
    # (every other bit 0) reduces to one green spider turned pi on plain edges, number 1: Z
    for k in range(3):                                   # the chapter draws the middle; all three hold
        tell = tuple(int(j == k) for j in range(3))
        g = storm_line((0, 0, 0), (0, 0), (0, 0, 0), tell)
        # the master as drawn: the lead-in and the lead-out, twelve strand lengths, four ties
        assert g.num_edges() == 18
        assert np.allclose(matrix_of(g), Z)
        h = g.copy()
        simp.full_reduce(h)
        inner = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
        assert len(inner) == 1
        v = inner[0]
        assert h.type(v) == VT.Z and h.phase(v) == 1
        assert [h.edge_type(ed) for ed in h.incident_edges(v)] == [ET.SIMPLE] * 2
        assert np.isclose(h.scalar.to_number(), 1)
        assert not same_picture(h, storm_line((0, 0, 0), (0, 0), (0, 0, 0)))   # not mended

    # --- Result 18.13: colour change.  The red 1 -> 3 spider is H^(x)3 Enc H, exactly
    red, v = encoder(VT.X)
    green, _ = encoder(VT.Z)
    assert np.allclose(matrix_of(red), H3 @ ENC @ H)
    assert not same_picture(red, green)
    # DERIVED: PyZX's colour change turns it green with a Hadamard edge on every leg
    h = red.copy()
    simp.to_gh(h)
    assert h.type(v) == VT.Z
    edges = list(h.incident_edges(v))
    assert len(edges) == 4 and all(h.edge_type(ed) == ET.HADAMARD for ed in edges)
    assert same_picture(h, red)
    # the Sketchbook sidebar: the whole storm line in the other colour is the old one with a
    # yellow box before it and one after, for every labelling
    for e, s, c in product(product((0, 1), repeat=3), OUTCOMES, product((0, 1), repeat=3)):
        g = storm_line(e, s, c)
        assert np.allclose(matrix_of(swap_colours(g)), H @ matrix_of(g) @ H)

    # --- Shor's encoder: green, yellow, green, exactly Enc9, no number
    s9 = shor_encoder()
    assert np.allclose(matrix_of(s9), ENC9)
    assert not same_picture(s9, shor_encoder(ET.SIMPLE))   # without the yellow boxes: not Shor's
    # DERIVED: Shor's encoder followed by its mirror reduces to a bare wire, number 1
    rt = shor_round_trip()
    # lead in, three yellow cords, nine strands in two lengths each, three yellow cords, lead out
    assert rt.num_edges() == 1 + 3 + 9 * 2 + 3 + 1
    assert np.allclose(matrix_of(rt), I2)
    h = rt.copy()
    simp.full_reduce(h)
    assert h.num_vertices() == 2 and h.num_edges() == 1
    assert np.isclose(h.scalar.to_number(), 1)


if __name__ == "__main__":
    verdict("NineStrands", blackboard, sketchbook)
