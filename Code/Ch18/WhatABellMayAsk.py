"""Challenge IV: What a Bell May Ask  (Knill-Laflamme; stabilizers; the weave; the vote of votes)
===============================================================================================
"Ask the one, not the pair."  -- Crab
"A bell that can tell you the message spoils every turned one."  -- Tortoise

Claim: the Knill-Laflamme conditions hold for the single gusts E_X (C = I_4) and fail for a
question about the value (Enc^dag Z1 Enc = Z; the clip-bell's projector gives |0><0|); for
Shor's 28 single-wire Paulis they hold with C of rank 22, a degenerate code (Result 18.16).
The clip-bell leaves dark and lit clear and makes every turned message a fair coin.  A check
of weight w is 2^(1-w/2) times its projector (Result 18.18).  The bit-flip code is [[3,1,1]],
Shor's no logical of weight <= 2; the toric code at l = 2, 4, 5 has 2l^2 - 2 independent checks
and protects a space of dimension 4 (Result 18.19).  The vote of votes has a toy threshold 1/2.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from itertools import combinations, product
from math import comb

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import I2, ket, kron, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch18 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
Y = 1j * X @ Z
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

ENC = np.outer(ket(0, 0, 0), ket(0)) + np.outer(ket(1, 1, 1), ket(1))
DEC = ENC.conj().T
H3 = kron(H, H, H)
ENC9 = kron(ENC, ENC, ENC) @ H3 @ ENC
SINGLE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

MARKS = {
    "dark": ket(0), "lit": ket(1),
    "at rest": (ket(0) + ket(1)) / np.sqrt(2), "a half": (ket(0) - ket(1)) / np.sqrt(2),
    "a quarter round": (ket(0) + 1j * ket(1)) / np.sqrt(2),
    "a quarter back": (ket(0) - 1j * ket(1)) / np.sqrt(2),
}
OPPOSITE = {("dark", "lit"), ("at rest", "a half"), ("a quarter round", "a quarter back")}


def F(p):
    return 3 * p**2 - 2 * p**3


def on(k, A, n=3):
    ops = [I2] * n
    ops[k] = A
    return kron(*ops)


def X_pattern(e):
    return kron(*[X if b else I2 for b in e])


def apply(word, M):
    """Apply a Pauli word (wire 1 first) to the columns of M."""
    n = len(word)
    T = M.reshape((2,) * n + (-1,))
    for k, letter in enumerate(word):
        if letter != "I":
            T = np.moveaxis(np.tensordot(PAULI[letter], T, axes=([1], [k])), 0, k)
    return T.reshape(2**n, -1)


def is_number_times_I(M):
    return np.allclose(M, M[0, 0] * np.eye(M.shape[0]))


# ------------------------------------------------------------ GF(2): the stabilizer bookkeeping

def symp(word):
    return np.array([c in "XY" for c in word] + [c in "YZ" for c in word], dtype=np.uint8)


def gf2_rank(rows):
    M = np.array(rows, dtype=np.uint8) % 2
    r = 0
    for col in range(M.shape[1]):
        piv = next((i for i in range(r, M.shape[0]) if M[i, col]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, col]:
                M[i] ^= M[r]
        r += 1
    return r


def commutes(a, b):
    n = len(a) // 2
    return (a[:n] @ b[n:] + a[n:] @ b[:n]) % 2 == 0


def logicals_up_to(gens, wmax):
    """Pauli words of weight 1..wmax in N(S) \\ S (up to phase), by listing them."""
    n = len(gens[0])
    G = [symp(g) for g in gens]
    r = gf2_rank(G)
    found = []
    for w in range(1, wmax + 1):
        for sites in combinations(range(n), w):
            for letters in product("XYZ", repeat=w):
                word = ["I"] * n
                for k, L in zip(sites, letters):
                    word[k] = L
                v = symp(word)
                if all(commutes(v, g) for g in G) and gf2_rank(G + [v]) > r:
                    found.append("".join(word))
    return found


def toric_checks(l):
    """Kitaev's toric code on an l x l grid on the torus: stars A_v (X) and faces B_f (Z), as
    GF(2) rows (x | z).  Edge h(i,j) joins (i,j)-(i,j+1); edge v(i,j) joins (i,j)-(i+1,j)."""
    n = 2 * l * l
    h = lambda i, j: (i % l) * l + (j % l)
    v = lambda i, j: l * l + (i % l) * l + (j % l)
    rows = []
    for i, j in product(range(l), repeat=2):                          # stars
        r = np.zeros(2 * n, dtype=np.uint8)
        for q in (h(i, j), h(i, j - 1), v(i, j), v(i - 1, j)):
            r[q] = 1
        rows.append(r)
    for i, j in product(range(l), repeat=2):                          # faces
        r = np.zeros(2 * n, dtype=np.uint8)
        for q in (h(i, j), h(i + 1, j), v(i, j), v(i, j + 1)):
            r[n + q] = 1
        rows.append(r)
    return n, rows, h


def blackboard():
    rng = np.random.default_rng(1841)
    # Enc, the Paulis X, Z and the projectors are real; the messages are non-real.

    # --- Result 18.16 (a): Knill-Laflamme for E_X, C = I_4 (nondegenerate: full rank)
    C = np.zeros((4, 4), dtype=complex)
    for a, ea in enumerate(SINGLE):
        for b, eb in enumerate(SINGLE):
            M = DEC @ X_pattern(ea).conj().T @ X_pattern(eb) @ ENC
            assert is_number_times_I(M)
            C[a, b] = M[0, 0]
    assert np.allclose(C, np.eye(4)) and np.linalg.matrix_rank(C) == 4
    # Gottesman's (30) on the codewords: <psi_i|E_a^dag E_b|psi_j> = C_ab [i = j]
    for a, ea in enumerate(SINGLE):
        for b, eb in enumerate(SINGLE):
            for i, j in product((0, 1), repeat=2):
                val = ENC[:, i].conj() @ X_pattern(ea).conj().T @ X_pattern(eb) @ ENC[:, j]
                assert np.isclose(val, C[a, b] * (i == j))
    # Gottesman's (29) for the linear space span E_X: <psi|E^dag E|psi> is independent of psi
    coeffs = rng.normal(size=4) + 1j * rng.normal(size=4)
    E = sum(c * X_pattern(e) for c, e in zip(coeffs, SINGLE))
    vals = [np.vdot(ENC @ psi, E.conj().T @ E @ ENC @ psi) for psi in (random_state(rng) for _ in range(3))]
    assert np.allclose(vals, vals[0])

    # --- Result 18.16 (b): a question about the value fails the conditions
    assert np.allclose(DEC @ on(0, Z) @ ENC, Z) and not is_number_times_I(DEC @ on(0, Z) @ ENC)
    clip = (np.eye(8) + on(0, Z)) / 2                                  # the clip-bell: "is it dark?"
    assert np.allclose(DEC @ clip @ ENC, np.outer(ket(0), ket(0)))
    assert not is_number_times_I(DEC @ clip @ ENC)
    # the numbers the code sees differ on |000> and |111>: (29) fails for {I, Z1}
    E = np.eye(8) + on(0, Z)                                           # a combination of I and Z1
    v0, v1 = (np.vdot(ENC[:, i], E.conj().T @ E @ ENC[:, i]) for i in (0, 1))
    assert not np.isclose(v0, v1)

    # --- the clip-bell on the storm line: dark, lit, and the four turns
    Z1Z2, Z2Z3 = on(0, Z) @ on(1, Z), on(1, Z) @ on(2, Z)
    for name, msg in MARKS.items():
        joint = np.zeros((2, 2))                                        # [bell, clear]
        for bell in (0, 1):
            branch = (np.eye(8) + (-1) ** bell * on(0, Z)) / 2 @ ENC @ msg
            pb = np.linalg.norm(branch) ** 2
            if pb < 1e-12:
                continue
            # the tied bells stay silent: the branch is still in the code space
            assert np.allclose(Z1Z2 @ branch, branch) and np.allclose(Z2Z3 @ branch, branch)
            pc = abs(np.vdot(msg, DEC @ branch)) ** 2                   # joint prob. of (bell, clear)
            joint[bell] = (pb - pc, pc)[::-1]
        p_bell, p_clear = joint.sum(axis=1), joint.sum(axis=0)
        if name == "dark":
            assert np.allclose(joint, [[1, 0], [0, 0]])                # silent, clear
        elif name == "lit":
            assert np.allclose(joint, [[0, 0], [1, 0]])                # rings, clear
        elif name in ("a quarter round", "a quarter back"):
            assert np.allclose(p_bell, [0.5, 0.5]) and np.allclose(p_clear[0], 0.5)
            assert np.allclose(joint, np.outer(p_bell, p_clear))       # independent of the bell
        else:                                                          # at rest, a half: also a coin
            assert np.allclose(joint, np.full((2, 2), 0.25))
    # the kitchen glass: clear for the mark, cloudy for its opposite, a fair coin otherwise
    for (m1, v1), (m2, v2) in product(MARKS.items(), repeat=2):
        f = abs(np.vdot(v1, v2)) ** 2
        if m1 == m2:
            assert np.isclose(f, 1)
        elif (m1, m2) in OPPOSITE or (m2, m1) in OPPOSITE:
            assert np.isclose(f, 0)
        else:
            assert np.isclose(f, 0.5)

    # --- Result 18.16 (c): Shor's 28 single-wire Paulis: the conditions hold, rank C = 22
    ops = ["I" * 9] + ["".join(L if j == k else "I" for j in range(9)) for L in "XYZ" for k in range(9)]
    moved = [apply(w, ENC9) for w in ops]                              # E_b Enc9
    C = np.zeros((28, 28), dtype=complex)
    for a in range(28):
        for b in range(28):
            M = moved[a].conj().T @ moved[b]                           # Enc9^dag E_a^dag E_b Enc9
            assert is_number_times_I(M)
            C[a, b] = M[0, 0]
    assert np.allclose(C, C.conj().T)                                  # Hermitian
    assert np.linalg.matrix_rank(C) == 22                              # degenerate: not 28
    z = [ops.index("".join("Z" if j == k else "I" for j in range(9))) for k in range(3)]
    assert np.allclose(C[z[0]], C[z[1]]) and np.allclose(C[z[1]], C[z[2]])   # Z1, Z2, Z3 alike

    # --- the bounds of Section 18.4 (stated, the arithmetic checked)
    hamming = lambda n, k, d: sum(3**j * comb(n, j) for j in range((d - 1) // 2 + 1)) * 2**k <= 2**n
    assert [n for n in range(1, 12) if hamming(n, 1, 3)][0] == 5       # (1 + 3n) 2 <= 2^n from n = 5
    assert all((1 + 3 * n) * 2 == sum(3**j * comb(n, j) for j in range(2)) * 2 for n in range(1, 12))
    singleton = lambda n, k, d: n - k >= 2 * d - 2
    assert not singleton(4, 1, 3) and singleton(5, 1, 3)               # no [[4,1,3]]; [[5,1,3]] fits

    # --- stabilizers (Result 18.17's examples; distances by listing, on the blackboard)
    bitflip = ["ZZI", "IZZ"]
    group = {tuple(int(x) for x in (a * symp(bitflip[0]) + b * symp(bitflip[1])) % 2)
             for a, b in product((0, 1), repeat=2)}
    assert len(group) == 4                                             # {I, Z1Z2, Z2Z3, Z1Z3}
    assert 3 - gf2_rank([symp(g) for g in bitflip]) == 1              # k = 1
    assert "ZII" in logicals_up_to(bitflip, 1)                         # Z1: weight 1, so d = 1
    shor = ["ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII", "IIIIIIZZI", "IIIIIIIZZ",
            "XXXXXXIII", "IIIXXXXXX"]                                  # Gottesman's Table 2
    steane = ["ZZZZIII", "ZZIIZZI", "ZIZIZIZ", "XXXXIII", "XXIIXXI", "XIXIXIX"]   # Table 3
    five = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]                        # Table 4
    for gens, n_gens in ((shor, 8), (steane, 6), (five, 4)):
        G = [symp(g) for g in gens]
        assert all(commutes(a, b) for a in G for b in G)
        assert gf2_rank(G) == n_gens and len(gens[0]) - n_gens == 1    # k = 1
        assert logicals_up_to(gens, 2) == []                           # nothing of weight <= 2...
        assert logicals_up_to(gens, 3) != []                           # ...something of weight 3: d = 3

    # --- Result 18.19: the toric code over GF(2).  l = 2, 4, 5 only, never l = 3 (the index;
    # identity card Section 1b).
    for l in (2, 4, 5):
        n, rows, h = toric_checks(l)
        assert n == 2 * l * l and len(rows) == 2 * l * l
        assert all(commutes(a, b) for a in rows for b in rows)         # stars and faces commute
        stars, faces = rows[: l * l], rows[l * l:]
        assert not (sum(stars) % 2).any() and not (sum(faces) % 2).any()   # prod A_v = prod B_f = I
        rank = gf2_rank(rows)
        assert rank == 2 * l * l - 2                                   # two relations, no more
        assert 2 ** (n - rank) == 4                                    # a protected space of dim 4
        # an example (not Kitaev's bound): a loop of Z round the torus, weight l, is undetected
        # and not a product of checks
        loop = np.zeros(2 * n, dtype=np.uint8)
        for j in range(l):
            loop[n + h(0, j)] = 1
        assert all(commutes(loop, r) for r in rows) and gf2_rank(rows + [loop]) > rank
    # at l = 2 the dimension also on the blackboard proper: the product of the check projectors
    n, rows, _ = toric_checks(2)
    Pi = np.eye(2**n)
    for r in rows:
        word = "".join("X" if r[q] else ("Z" if r[n + q] else "I") for q in range(n))
        Pi = (Pi + apply(word, Pi)) / 2
    assert np.isclose(np.trace(Pi).real, 4)
    words2 = ["".join("X" if r[q] else ("Z" if r[n + q] else "I") for q in range(n)) for r in rows]
    assert logicals_up_to(words2, 1) == []                             # at l = 2: distance 2 = l

    # --- the vote of votes (a toy, not the theorem): the fixed point 1/2
    assert np.allclose(sorted(np.roots([-2, 3, -1, 0]).real), [0, 0.5, 1])   # F(p) = p
    for p in rng.uniform(0, 1, size=5):
        assert np.isclose(F(p) - p, p * (1 - p) * (2 * p - 1))
        assert 6 * p * (1 - p) >= 0                                     # F increasing on [0, 1]
    assert np.isclose(F(0.6), 0.648) and F(0.6) > 0.6                  # damp above a half: worse
    assert np.isclose(F(0.028), 0.002308096, atol=1e-12)
    p, q = 0.1, 0.6
    for _ in range(5):                                                 # the levels drive p below 1/2
        assert F(p) < p and F(q) > q                                   # down to 0, above 1/2 up to 1
        p, q = F(p), F(q)
    assert p < 1e-12 and q > 0.95

    pictures()


# ------------------------------------------------------------------------------ prop pictures

FOURTH_DUMPS = ("mend=0,0,0 tell=0,0,0 clip=first glass=clear",
                "mend=0,0,0 tell=0,0,0 clip=first glass=cloudy")


def pictures():
    """The Fourth Challenge's prop pictures (identity card, section 4), read from the dialogue
    through tools/dfp_props.py and the transcription tools/props_ch18.py, and held against this
    file's clip-bell (the Z1 question asked on the first strand alone)."""
    calls = DIALOGUE.check_layout("Fourth", sizes=[1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("stormline", False, "clip-bell on the first tell ring, glass clear", (), "Clipped on"),
        ("stormline", False, "clip-bell on, glass cloudy", (), "Lifted at her mark")]
    PC.check_captions(DIALOGUE, "Fourth", calls)
    PC.check_section(DIALOGUE, "Fourth", calls)
    PC.check_anchors(DIALOGUE, "Fourth", ["P4.1", "P4.2"])
    p41, p42 = pics = [PC.picture(c) for c in calls]
    assert tuple(p.dump for p in pics) == FOURTH_DUMPS
    assert p41.frame == p42.frame == PC.stormline("").frame
    for pic in pics:                                             # every bead at rest, nothing mended
        assert pic.state.mend == [0, 0, 0] and pic.state.tell == [0, 0, 0]
        assert all(r[4] == 0 for r in pic.shown.rings) and pic.labels == ()
        assert pic.state.clip == 1 and "clip=first" in pic.dump       # "Clipped on"
    assert (p41.state.glass, p42.state.glass) == ("clear", "cloudy")   # "Lifted at her mark": lifted, cloudy

    # the clip-bell on the first tell ring asks the first strand alone: its two branches
    silent = (np.eye(8) + on(0, Z)) / 2                           # "is it dark?": quiet
    rings = (np.eye(8) - on(0, Z)) / 2                            # rings when the strand is lit
    assert np.allclose(silent + rings, np.eye(8)) and np.allclose(silent @ rings, 0)
    assert np.allclose(DEC @ silent @ ENC, np.outer(ket(0), ket(0)))
    assert np.allclose(DEC @ rings @ ENC, np.outer(ket(1), ket(1)))
    Z1Z2, Z2Z3 = on(0, Z) @ on(1, Z), on(1, Z) @ on(2, Z)
    for br in (silent, rings):                                    # the tied bells stay silent
        assert np.allclose(Z1Z2 @ br @ ENC, br @ ENC) and np.allclose(Z2Z3 @ br @ ENC, br @ ENC)
    # P4.1, the dark run (l.631): silent, the branch leaves the encoded dark message as it was;
    # read at dark, clear
    dark = ENC @ MARKS["dark"]
    assert np.isclose(np.linalg.norm(silent @ dark), 1) and np.allclose(silent @ dark, dark)
    assert np.isclose(abs(np.vdot(MARKS["dark"], DEC @ silent @ dark)) ** 2, 1)    # "glass=clear"
    # P4.2, the Tortoise's run (l.641): it RANG, so the branch leaves the encoded lit message,
    # whatever was sent; her mark was a quarter round, which reads cloudy as often as clear
    for msg in (MARKS["a quarter round"], random_state(np.random.default_rng(1842))):
        out = rings @ ENC @ msg
        lit = (ENC @ ket(1)).ravel()
        assert np.isclose(abs(np.vdot(lit, out.ravel())), np.linalg.norm(out))   # lit, up to a phase
    q = MARKS["a quarter round"]
    out = rings @ ENC @ q
    assert np.isclose(np.linalg.norm(out) ** 2, 0.5)             # it rings half the time
    cloudy = 1 - abs(np.vdot(q, DEC @ out / np.linalg.norm(out))) ** 2
    assert np.isclose(cloudy, 0.5)                                # "cloudy as often as clear"
    # (silent, the dark one reads cloudy at a quarter round half the time too: no branch spares it)
    out0 = silent @ ENC @ q
    assert np.isclose(1 - abs(np.vdot(q, DEC @ out0 / np.linalg.norm(out0))) ** 2, 0.5)
    # the picture's outcome is a possible one; the clear outcome would have been as likely, and
    # without the clip-bell the line gives a quarter round back clear for certain
    assert np.isclose(abs(np.vdot(q, DEC @ ENC @ q)) ** 2, 1)

    # fixed parts, P3.1 -> P4.1: the middle tell ring back at rest (l.575), the clip-bell on
    # (l.625), the glass reset and lifted at dark: clear.  P4.1 -> P4.2: only the glass.
    p31 = PC.stormline("middle tell ring at a half, glass cloudy")
    va, vb, vc = p31.shown, p41.shown, p42.shown
    assert [(a[0], a[4], b[4]) for a, b in zip(va.rings, vb.rings) if a != b] == [("t2", 2, 0)]
    assert [a[:4] for a in va.rings] == [b[:4] for b in vb.rings]
    assert va.spans == vb.spans == vc.spans and va.knots == vb.knots == vc.knots
    assert va.bells == vb.bells == vc.bells and va.clip == () and vb.clip == vc.clip != ()
    assert (va.glass, vb.glass, vc.glass) == ("cloudy", "clear", "cloudy")
    assert vb.rings == vc.rings and vb.beads == vc.beads
    # the clip-bell rides on the first tell ring: its clip under that ring, nowhere else
    t1 = next(r for r in vb.rings if r[0] == "t1")
    assert PC.stormline("clip-bell on the first tell ring").dump == PC.stormline("clip-bell on").dump
    assert vb.clip == vc.clip == (t1[1], t1[2])
    third = PC.stormline("clip-bell on the third tell ring")
    assert third.dump.endswith("clip=third glass=reset") and third.shown.clip != vb.clip

    # the Lua's refusals: a bell drawn ringing; the dial drawn open
    PC.TEX_ERRORS.clear()
    PC.line_state("first bell ringing")
    PC.line_state("door dial at a half")
    assert [k for k, _ in PC.TEX_ERRORS] == ["line-bell", "line-dial"]
    PC.TEX_ERRORS.clear()


# ------------------------------------------------------------------------------ pictures

def kl_picture(ring_rows):
    """Parting ring, three strands, joining ring.  ring_rows[k] lists the rings the message
    meets on strand k as (colour, phase); every strand carries at least one ring, so PyZX
    keeps three separate strands (its simple graphs hold one edge per pair)."""
    g = zx.Graph()
    i, o = g.add_vertex(VT.BOUNDARY, 1, 0), g.add_vertex(VT.BOUNDARY, 1, 10)
    E, D = g.add_vertex(VT.Z, 1, 1), g.add_vertex(VT.Z, 1, 9)
    g.add_edge((i, E))
    g.add_edge((D, o))
    for k, rings in enumerate(ring_rows):
        prev = E
        for col, (colour, ph) in enumerate(rings, start=3):
            v = g.add_vertex(colour, k, col, phase=Fraction(ph))
            g.add_edge((prev, v))
            prev = v
        g.add_edge((prev, D))
    g.set_inputs((i,))
    g.set_outputs((o,))
    return g


def check(w, s, feet=VT.Z, hub=VT.X, number=True):
    """A check of weight w: a foot on each of w wires, all tied to one hub turned s*pi.
    With number=True the picture carries 2^(w/2 - 1), so it is exactly the projector."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in range(w)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 4) for r in range(w)]
    fs = [g.add_vertex(feet, r, 1) for r in range(w)]
    hb = g.add_vertex(hub, w, 2, phase=Fraction(s))
    for r in range(w):
        g.add_edge((ins[r], fs[r]))
        g.add_edge((fs[r], outs[r]))
        g.add_edge((fs[r], hb))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    if number:
        g.scalar.add_power(w - 2)                   # sqrt2^(w-2) = 2^(w/2 - 1)
    return g


def check_twice(w, s):
    """The bare check followed by itself."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in range(w)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 6) for r in range(w)]
    f1 = [g.add_vertex(VT.Z, r, 1) for r in range(w)]
    f2 = [g.add_vertex(VT.Z, r, 3) for r in range(w)]
    h1 = g.add_vertex(VT.X, w, 2, phase=Fraction(s))
    h2 = g.add_vertex(VT.X, w, 4, phase=Fraction(s))
    for r in range(w):
        g.add_edge((ins[r], f1[r]))
        g.add_edge((f1[r], f2[r]))
        g.add_edge((f2[r], outs[r]))
        g.add_edge((f1[r], h1))
        g.add_edge((f2[r], h2))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


def sketchbook():
    # --- Result 18.16 (a) as pictures, DERIVED: Enc^dag E_a^dag E_b Enc for the sixteen pairs
    # of E_X.  Strand k carries E_b's ring then E_a's (red, e_k pi): each is X^(e_k).
    for a, ea in enumerate(SINGLE):
        for b, eb in enumerate(SINGLE):
            g = kl_picture([[(VT.X, eb[k]), (VT.X, ea[k])] for k in range(3)])
            assert g.num_edges() == 1 + 3 * 3 + 1
            assert np.allclose(matrix_of(g), (a == b) * I2)
            h = g.copy()
            simp.full_reduce(h)
            if a == b:
                assert h.num_vertices() == 2 and h.num_edges() == 1       # a bare wire,
                assert np.isclose(h.scalar.to_number(), 1)                # number C_aa = 1
            else:
                assert h.scalar.is_zero                                   # C_ab = 0
    # (b): a green half on the first strand does not close to a number: one green pi is left
    g = kl_picture([[(VT.Z, 1)], [(VT.X, 0)], [(VT.X, 0)]])
    assert np.allclose(matrix_of(g), Z)
    h = g.copy()
    simp.full_reduce(h)
    inner = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
    assert len(inner) == 1 and h.type(inner[0]) == VT.Z and h.phase(inner[0]) == 1
    assert np.isclose(h.scalar.to_number(), 1)
    assert not same_picture(h, kl_picture([[(VT.X, 0)]] * 3))            # not a number times a wire
    # the clip-bell: a foot on the first strand tied to a one-legged red hub at rest, with its
    # number 2^(-1/2), is 1/2 (I + Z1); inside the code it is |0><0| -- the wire is cut
    g = zx.Graph()
    i, o = g.add_vertex(VT.BOUNDARY, 1, 0), g.add_vertex(VT.BOUNDARY, 1, 10)
    E, D = g.add_vertex(VT.Z, 1, 1), g.add_vertex(VT.Z, 1, 9)
    foot, hub = g.add_vertex(VT.Z, 0, 4), g.add_vertex(VT.X, -1, 5)
    rest = [g.add_vertex(VT.X, k, 4) for k in (1, 2)]
    for u, v in ((i, E), (E, foot), (foot, D), (foot, hub), (D, o),
                 (E, rest[0]), (rest[0], D), (E, rest[1]), (rest[1], D)):
        g.add_edge((u, v))
    g.set_inputs((i,))
    g.set_outputs((o,))
    g.scalar.add_power(-1)
    assert np.allclose(matrix_of(g), np.outer(ket(0), ket(0)))
    h = g.copy()
    simp.full_reduce(h)
    assert same_picture(h, g)
    i2, o2 = h.inputs()[0], h.outputs()[0]
    reach, todo = {i2}, [i2]                                             # no path from in to out
    while todo:
        for u in h.neighbors(todo.pop()):
            if u not in reach:
                reach.add(u)
                todo.append(u)
    assert o2 not in reach

    # --- (T2) translation check, not a derivation: the number of a weight-w check against
    # its projector matrix, w = 1..5, both s; and the colour-swapped X-check, same number
    for w in range(1, 6):
        Zw = kron(*[Z] * w)
        Xw = kron(*[X] * w)
        for s in (0, 1):
            proj = (np.eye(2**w) + (-1) ** s * Zw) / 2
            bare = matrix_of(check(w, s, number=False))
            assert np.allclose(bare, 2 ** (1 - w / 2) * proj)
            assert np.allclose(matrix_of(check(w, s)), proj)
            assert np.allclose(bare, proj) == (w == 2)                    # the number is 1 only at w = 2
            xproj = (np.eye(2**w) + (-1) ** s * Xw) / 2
            assert np.allclose(matrix_of(check(w, s, VT.X, VT.Z, number=False)), 2 ** (1 - w / 2) * xproj)

    # --- DERIVED: the check followed by itself is 2^(1-w/2) times the check (the projector
    # property, drawn); full_reduce does the rewriting
    for w in (2, 3, 4):
        for s in (0, 1):
            g = check_twice(w, s)
            h = g.copy()
            simp.full_reduce(h)
            ref = check(w, s, number=False)
            ref.scalar.add_power(2 - w)                                   # 2^(1 - w/2) = sqrt2^(2 - w)
            assert same_picture(h, ref)
            r = ref.copy()
            simp.full_reduce(r)
            assert same_picture(h, r)
            assert h.num_vertices() <= g.num_vertices() - w               # the feet have fused


if __name__ == "__main__":
    verdict("WhatABellMayAsk", blackboard, sketchbook)
