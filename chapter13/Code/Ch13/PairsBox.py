"""Challenge IV: The Pairs Box  (the da capo; one round of Simon's problem)
========================================================================
"So that's the evening. A plain wish buys one answer. A crosswise wish on a box of marks buys the whole secret, read straight off its ties. On a pairs box it buys one clue: a line through the secret, not the secret. And no turning of the collars buys what the box doesn't hold."  -- Achilles

Claim: for every f: {0,1}^n -> {0,1}^m with Simon's promise (f(x) = f(x') exactly when
x' in {x, x xor s}, s != 0), one round -- H^n, one query into answer wires at |0>, H^n,
the answer wires discarded -- leaves the rods in rho = 2^{-n}(I + Z^s) = 2^{1-n} P_s,
so the clue z is uniform on {z : z.s = 0} (Result 13.9); n-1 independent clues fix s,
and on average at most 2(n-1) rounds collect them (Result 13.10); the parity gadget of s
satisfies P_s = 2^{|s|/2-1} x gadget, so rho = 2^{|s|/2-n} x gadget (Result 13.11).  For
the pairs box (n = 3, M = [[1,1,0],[0,0,1]], s = 110) the clues 001 and 110 give s = 110.
Stated in the chapter and cited, NOT tested here: Simon's classical separation
(Result 13.12, via BBBV section 1) and BBBV's Theorem 3.5 (Result 13.14).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from functools import reduce
from itertools import combinations, product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import CUP, ket, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch13 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)

M_PAIRS = ((1, 1, 0), (0, 0, 1))                        # the pairs box on the cloth
S_PAIRS = (1, 1, 0)
M_W3 = ((1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 0, 1))      # a weight-3 secret, n = 4
S_W3 = (1, 1, 1, 0)


def strings(n):
    return list(product((0, 1), repeat=n))


def dot(a, x):
    return sum(ai * xi for ai, xi in zip(a, x)) % 2


def xor(a, b):
    return tuple(ai ^ bi for ai, bi in zip(a, b))


def index(bits):
    return int("".join(map(str, bits)) or "0", 2)


def Hn(n):
    return reduce(np.kron, [H] * n, np.eye(1))


def pauli_on(P, s):
    """P^s = P^{s_1} (x) ... (x) P^{s_n}."""
    return reduce(np.kron, [P if si else np.eye(2) for si in s], np.eye(1))


def P_s(s):
    return (np.eye(2 ** len(s)) + pauli_on(Z, s)) / 2


def linear_box(M):
    return lambda x: tuple(dot(row, x) for row in M)


def U(f, n, m):
    """U_f |x>|y> = |x>|y xor f(x)> on n rods and m answer wires, as a dense matrix."""
    N = 2 ** (n + m)
    Uf = np.zeros((N, N), dtype=complex)
    for x in strings(n):
        for y in strings(m):
            Uf[index(x + xor(y, f(x))), index(x + y)] = 1
    return Uf


def round_rho(f, n, m, dense=True):
    """One round: rods |0^n>, answer wires |0^m>; H^n on the rods; one query; H^n on the
    rods; the answer wires discarded (partial trace).  Returns rho on the rods."""
    layer = np.kron(Hn(n), np.eye(2 ** m))
    start = np.zeros(2 ** (n + m), dtype=complex); start[0] = 1
    v = layer @ start
    if dense:
        v = U(f, n, m) @ v
    else:                                         # the same permutation, applied index-wise
        w = np.zeros_like(v)
        for x in strings(n):
            for y in strings(m):
                w[index(x + xor(y, f(x)))] = v[index(x + y)]
        v = w
    v = layer @ v
    Psi = v.reshape(2 ** n, 2 ** m)
    return Psi @ Psi.conj().T                     # tr_ans |Psi><Psi|


def pairs_function(rng, s, m):
    """A random NON-LINEAR f with Simon's promise for s: each pair {x, x xor s} gets its own
    randomly chosen m-bit value."""
    n = len(s)
    values = [strings(m)[k] for k in rng.permutation(2 ** m)]
    table, k = {}, 0
    for x in strings(n):
        if x not in table:
            table[x] = table[xor(x, s)] = values[k]; k += 1
    return lambda x: table[x]


def promise_holds(f, s):
    n = len(s)
    return all((f(x) == f(y)) == (xor(x, y) in {(0,) * n, s}) for x in strings(n) for y in strings(n))


def solutions(clues, n):
    """The nonzero strings v with v.z = 0 for every clue z."""
    return [v for v in strings(n) if any(v) and all(dot(v, z) == 0 for z in clues)]


def rank2(vectors):
    """Rank over GF(2)."""
    rows = [int("".join(map(str, v)), 2) for v in vectors]
    r = 0
    for bit in reversed(range(max(len(v) for v in vectors))):
        piv = next((i for i in range(r, len(rows)) if rows[i] >> bit & 1), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i] >> bit & 1:
                rows[i] ^= rows[r]
        r += 1
    return r


def blackboard():
    rng = np.random.default_rng(1304)
    n, m, s = 3, 2, S_PAIRS
    f = linear_box(M_PAIRS)

    # --- the pairs box, as Achilles finds it by touching the ties
    assert f((1, 0, 0)) == f((0, 1, 0)) == (1, 0)          # first or second: the upper answer turns
    assert f((1, 1, 0)) == f((0, 0, 0)) == (0, 0)          # both: turned twice, as if neither
    assert all(f(x)[1] == x[2] for x in strings(3))        # only the third touches the lower answer
    assert f((0, 0, 1)) == f((1, 1, 1)) == (0, 1) and f((0, 1, 1)) == f((1, 0, 1)) == (1, 1)
    # "two settings answer alike exactly when they differ at the first two places together,
    # and nowhere else": Simon's promise with s = 110
    assert promise_holds(f, s)
    assert not promise_holds(f, (1, 0, 0)) and not promise_holds(f, (0, 0, 1))

    # --- Result 13.9 on the pairs box: rho = 2^{-n}(I + Z^s) = 2^{1-n} P_s
    rho = round_rho(f, n, m)
    assert np.allclose(rho, (np.eye(8) + pauli_on(Z, s)) / 8)
    assert np.allclose(rho, 2.0 ** (1 - n) * P_s(s))
    assert np.isclose(np.trace(rho), 1)
    assert not np.allclose(rho, (np.eye(8) + pauli_on(X, s)) / 8)          # the second layer matters
    # the proof's middle line: before the second layer, 2^{-n}(I + X^s)
    layer = np.kron(Hn(n), np.eye(4))
    Psi = (U(f, n, m) @ layer @ ket(0, 0, 0, 0, 0)).reshape(8, 4)
    assert np.allclose(Psi @ Psi.conj().T, (np.eye(8) + pauli_on(X, s)) / 8)
    # the clues: 000, 001, 110, 111, a quarter each, and nothing else
    probs = np.real(np.diag(rho))
    clues = [z for z in strings(n) if probs[index(z)] > 1e-12]
    assert clues == [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)] and np.allclose(probs[probs > 1e-12], 0.25)
    assert all(np.isclose(probs[index(z)], 2.0 ** (1 - n) * (dot(z, s) == 0)) for z in strings(n))
    assert np.linalg.matrix_rank(rho) == 4                 # not a single pure answer
    # "nothing we do to the answer wires after the query can change the statistics of the
    # rods" (causality, Result 9.4): a random complex unitary on the answer wires, then discard
    G = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    assert not np.allclose(G, G.conj()) and np.allclose(G.conj().T @ G, np.eye(4))
    v = np.kron(Hn(n), np.eye(4)) @ U(f, n, m) @ layer @ ket(0, 0, 0, 0, 0)
    w = np.kron(np.eye(8), G) @ v
    assert np.allclose(w.reshape(8, 4) @ w.reshape(8, 4).conj().T, rho)

    # --- "Linearity is not part of Simon's promise, though at this size it comes free: with
    #     three rods and two answer wires every box with the promise is a web, perhaps with
    #     some answer bits flipped."  EXHAUSTIVE: all 4^8 = 65536 functions {0,1}^3 -> {0,1}^2;
    #     every one with the promise for s = 110 is f(x) = Mx xor c, where M's columns are
    #     f(e_i) xor f(000) and c = f(000).  (The same for every nonzero s.)
    def affine(g, n_):
        return all(g(xor(x, y)) == xor(g(x), xor(g(y), g((0,) * n_)))
                   for x in strings(n_) for y in strings(n_))

    def as_web(g, n_):
        """Mx xor c read off g: M's columns g(e_i) xor g(0), c = g(0)."""
        c = g((0,) * n_)
        cols = [xor(g(tuple(int(j == i) for j in range(n_))), c) for i in range(n_)]
        M = tuple(tuple(col[r] for col in cols) for r in range(len(c)))
        return lambda x: xor(linear_box(M)(x), c)

    two_bit = strings(2)
    for s_ in [z for z in strings(3) if any(z)]:
        count = 0
        for tab in product(range(4), repeat=8):
            g = lambda x, tab=tab: two_bit[tab[index(x)]]
            if promise_holds(g, s_):
                count += 1
                assert affine(g, 3)
                assert all(g(x) == as_web(g, 3)(x) for x in strings(3))   # a web, bits flipped
        assert count == 4 * 3 * 2 * 1          # the 4 pairs sent to the 4 answers, one-to-one
    # "With more answer wires there are boxes with the promise that are not webs": at m = 3,
    # s = 110, one exhibited explicitly (and the random ones below)
    table = {(0, 0, 0): (0, 0, 0), (1, 1, 0): (0, 0, 0), (0, 0, 1): (1, 0, 0), (1, 1, 1): (1, 0, 0),
             (0, 1, 0): (0, 1, 0), (1, 0, 0): (0, 1, 0), (0, 1, 1): (0, 0, 1), (1, 0, 1): (0, 0, 1)}
    g = lambda x: table[x]
    assert promise_holds(g, S_PAIRS) and not affine(g, 3)
    assert not all(g(x) == as_web(g, 3)(x) for x in strings(3))

    # --- "for EVERY f with the promise": random NON-LINEAR pairs boxes (n = 3, s = 110 with
    #     three answer wires, since with two every box is a web, as just checked; the
    #     weight-3 secret at n = 4; the chapter's six-rod instance s = 100010; a random s at
    #     n = 5).  "no turning of the collars buys what the box doesn't hold": the round sees
    #     the box only through s.
    s5 = (0,) * 5
    while not any(s5):
        s5 = tuple(int(t) for t in rng.integers(0, 2, 5))
    for s_, m_, dense in [(S_PAIRS, 3, True), (S_W3, 3, True), ((1, 0, 0, 0, 1, 0), 5, False),
                          (s5, 4, False)]:
        n_ = len(s_)
        g = pairs_function(rng, s_, m_)
        while affine(g, n_):                     # redraw until the box is NOT a web
            g = pairs_function(rng, s_, m_)
        assert promise_holds(g, s_) and not affine(g, n_)
        r = round_rho(g, n_, m_, dense)
        assert np.allclose(r, (np.eye(2 ** n_) + pauli_on(Z, s_)) / 2 ** n_)
        assert np.allclose(r, 2.0 ** (1 - n_) * P_s(s_))
    # the linear weight-3 box too
    assert promise_holds(linear_box(M_W3), S_W3)
    assert np.allclose(round_rho(linear_box(M_W3), 4, 3), 2.0 ** (1 - 4) * P_s(S_W3))

    # --- the supposed clues: "only the third window" (001), then "the first two" (110)
    assert solutions([(0, 0, 1)], 3) == [(0, 1, 0), (1, 0, 0), (1, 1, 0)]   # third place: never
    assert solutions([(1, 1, 0)], 3) == [(0, 0, 1), (1, 1, 0), (1, 1, 1)]   # first two: both or neither
    assert solutions([(0, 0, 1), (1, 1, 0)], 3) == [S_PAIRS]               # "Both, then"
    # one clue never names s: at least three candidates are left
    assert all(len(solutions([z], 3)) >= 3 for z in clues)
    # "a wish that lit nothing at all ... would pass every check there is, and tell me nothing"
    assert all(dot(v, (0, 0, 0)) == 0 for v in strings(3))
    assert solutions([(0, 0, 0), (0, 0, 1)], 3) == solutions([(0, 0, 1)], 3)

    # --- Result 13.10: any n-1 independent clues leave exactly s.  Every independent pair for
    #     the pairs box; random independent sets of five for s = 100010 (n = 6); fewer than
    #     n-1 never suffice
    pairs = [c for c in combinations(clues, 2) if rank2(c) == 2]
    assert len(pairs) == 3
    for c in pairs:
        assert solutions(c, 3) == [S_PAIRS]
    s6 = (1, 0, 0, 0, 1, 0)
    space = [z for z in strings(6) if dot(z, s6) == 0]
    assert len(space) == 2 ** 5
    for _ in range(200):
        c = [space[k] for k in rng.choice(len(space), 5, replace=False)]
        if rank2(c) == 5:
            assert solutions(c, 6) == [s6]
        else:
            assert len(solutions(c, 6)) >= 3
    for c in combinations(space[1:], 4):
        if rank2(c) == 4:
            assert len(solutions(c, 6)) == 3                  # n - 2 clues: s and two others
            break
    # the rounds: with k independent clues in hand a new clue falls among their 2^k sums with
    # probability 2^k / 2^{n-1} <= 1/2 (checked by counting, n = 3 and 6); so the expected
    # number of rounds sum_k 1/(1 - 2^{k-(n-1)}) is at most 2(n-1)
    for s_, sp in ((S_PAIRS, clues), (s6, space)):
        n_ = len(s_)
        for k in range(n_ - 1):
            basis = []                                    # k independent clues, greedily
            for z in sp:
                if len(basis) < k and rank2(basis + [z]) == len(basis) + 1:
                    basis.append(z)
            span = {reduce(xor, [b for b, t in zip(basis, sel) if t], (0,) * n_)
                    for sel in product((0, 1), repeat=k)}
            assert len(basis) == k and len(span) == 2 ** k
            assert np.isclose(sum(z in span for z in sp) / len(sp), 2.0 ** k / 2 ** (n_ - 1))
    for n_ in range(2, 13):
        expected = sum(1 / (1 - 2.0 ** (k - (n_ - 1))) for k in range(n_ - 1))
        assert expected <= 2 * (n_ - 1)

    pictures()


def pictures():
    """The Fourth Challenge's prop pictures (identity card, section 4), and the rules on the
    dialogue's fifteen pictures as a whole, read through tools/dfp_props.py and the
    transcription tools/props_ch13.py.  The ties below hold the transcription against this
    file's blackboard: a structural check of the pictures, not a second proof."""
    calls = DIALOGUE.check_layout("Fourth", sizes=[1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("wishcloth", True, "", ("pairs box",), "laid out, nothing folded yet"),
        ("wishcloth", True, "fold", ("pairs box",), "the near edge folded over"),
        ("wishcloth", False, "fold, slide together", ("pairs box",), "run together above the fold")]
    PC.check_captions(DIALOGUE, "Fourth", calls)
    PC.check_section(DIALOGUE, "Fourth", calls)
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Fourth")] == [[2.56]] * 3

    a1, a2, a3 = DIALOGUE.anchors("Fourth")
    f1, f2, f3 = DIALOGUE.following("Fourth")
    assert "Across the cloth's far side from Achilles" in a1 and "Each tie she leaves a little slack." in a1
    assert "the lower turning down a little before the upper" in a1                                # D23
    assert f1.startswith("**Tortoise:** Which settings answer alike?")
    assert "stop where nobody looks" in a2
    assert "reaches across, takes the cloth's near edge on Achilles's side" in a2                  # D24
    assert f2.startswith("*[On the upper answer-cord")
    assert "What has held: the ties, the wish-cords, and the fold." in a3
    assert "into the second place's red ring" in a3                                               # D25
    assert f3.startswith("**Tortoise:** A red ring counts the ties")
    assert DIALOGUE.following("Fourth", DIALOGUE.raw) == [f1, f2, f3]
    assert ("She clears the web from the cloth, the rings into her little box of rings and the string "
            "into her work-bag.") in DIALOGUE.section("Fourth")                                      # D22

    # the whole dialogue: fifteen pictures, all inside the four challenges, none after P4.3;
    # each in its place; the keyed form exactly on the first picture of each kind
    assert DIALOGUE.check_whole(PC.ORDER) == 15
    text = DIALOGUE.read(DIALOGUE.dialog)
    assert not DIALOGUE.call_re.search(text[text.index(calls[-1].text) + len(calls[-1].text):])
    everything = []
    for o in PC.ORDER:
        for b in DIALOGUE.blocks(o):
            for c in b:
                assert PC.place_ok(o, c), (o, c.text)
                everything.append(c)
    DIALOGUE.keyed_first(PC.ORDER, kind=PC.keyed_kind)
    keyed = [i for i, c in enumerate(everything) if c.keyed]
    names = ["P1.1", "P1.2", "P1.3", "P1.4", "P2.1", "P2.2", "P2.3", "P2.4",
             "P3.1", "P3.2", "P3.3", "P3.4", "P4.1", "P4.2", "P4.3"]
    assert [names[i] for i in keyed] == ["P1.1", "P1.2", "P1.4", "P2.1", "P3.3", "P3.4", "P4.1", "P4.2"]
    # no call with the l.570 gamble setting; no call raises a fault
    for c in everything:
        PC.picture(c)
        assert not (c.macro == "wishcasket" and "flag knob's collar upright" in PC.norm(c.args[0]))
    # the minuet and the Genie after the revision, on both files (section 1c's one-liner)
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        t = DIALOGUE.read(path)
        ok, trio, granted, refrain = PC.form_check(t)
        assert ok and trio == "Third Challenge" and granted == 1 and refrain, path
        assert t.count("Let Bob draw") == 0
        spoken = re.findall(r"^\*\*([^*]+):\*\*(.*)$", t, re.M)
        referred = [who for who, line in spoken if re.sub(r"\*\[.*?\]\*", "", line).strip().startswith("Referred.")]
        assert referred == ["Genie"] * 3, referred

    p41, p42, p43 = (PC.picture(c) for c in calls)
    S1, S2, S3 = (p.cloth for p in (p41, p42, p43))
    # P4.1: the pairs box -- its ties ARE f(x) = (x1 xor x2, x3): the upper answer-cord tied to
    # the first two places, the lower to the third; Simon's promise holds for s = 110
    assert p41.dump == (
        "scene=pairs folded=false slid=false | cols=8 rows=5 y=0..4 near=true folded=false; gaps=2.5; "
        "1:end red 0 @0,0; 2:ring green 0 @2,0; 3:end red 0 @0,1; 4:ring green 0 @3,1; 5:end red 0 @0,2; "
        "6:ring green 0 @4,2; 7:end red 0 @0,3; 8:ring red 0 @2,3; 9:ring red 0 @3,3; 10:end red 0 @0,4; "
        "11:ring red 0 @4,4; tie 2-8 []; tie 4-9 []; tie 6-11 []; cord fold-10 [5,4 0,4]; cord fold-7 [6,3 0,3]; "
        "cord loose-1 [6.2,0 0,0]; cord loose-3 [6.2,1 0,1]; cord loose-5 [6.2,2 0,2]")
    assert p41.labels == ("answer-cords",)

    def box(S):
        """The answer each answer-cord gives: which wish places tie down to it."""
        y = lambda i: S["things"][i - 1]["y"]
        M = [[0, 0, 0], [0, 0, 0]]
        for d in S["cords"]:
            if d["kind"] == "tie":
                M[y(d["b"]) - 3][y(d["a"])] = 1
        return tuple(map(tuple, M))
    assert box(S1) == box(S2) == box(S3) == M_PAIRS
    f = linear_box(box(S1))
    assert [f(x) for x in strings(3)] == [(x[0] ^ x[1], x[2]) for x in strings(3)]
    assert promise_holds(f, S_PAIRS) and not promise_holds(f, (0, 1, 1))

    # P4.2: the fold -- nothing moves but the near edge; the fold band covers both
    # answer-cords' ends and no ring
    assert p42.dump == p41.dump.replace("folded=false", "folded=true")
    assert p42.labels == ("fold",)
    assert S2["things"] == S1["things"] and S2["cords"] == S1["cords"]
    for p in (p42, p43):
        lo, hi = PC.fold_band(p)
        rings, cords = PC.cloth_drawn(p)
        ends = [pts[-1] if b == "fold" else pts[0] for kind, a, b, pts, _ in cords if "fold" in (a, b)]
        assert len(ends) == 2 and all(lo < y < hi for _, y in ends)
        assert all(Y - rr > hi for *_, Y, rr in rings)

    # P4.3: run together above the fold -- on the upper answer-cord one ring at the second
    # place's column holding the first two places' ties, on the lower one at its column
    # holding the third's; no ring added.  One round, the answer wires discarded, gives the
    # clues 000, 001, 110, 111, a quarter each, every one with z.s = 0
    assert p43.dump == (
        "scene=pairs folded=true slid=true | cols=8 rows=5 y=0..4 near=true folded=true; gaps=2.5; "
        "1:end red 0 @0,0; 2:ring green 0 @2,0; 3:end red 0 @0,1; 4:ring green 0 @3,1; 5:end red 0 @0,2; "
        "6:ring green 0 @4,2; 7:ring red 0 @3,3; 8:ring red 0 @4,4; tie 2-7 []; tie 4-7 []; tie 6-8 []; "
        "cord fold-7 [6,3 3,3]; cord fold-8 [5,4 4,4]; cord loose-1 [6.2,0 0,0]; cord loose-3 [6.2,1 0,1]; "
        "cord loose-5 [6.2,2 0,2]")
    answer = {r["y"]: (r["x"], r["c"]) for r in S3["things"] if r["y"] >= 3}
    assert answer == {3: (3, "red"), 4: (4, "red")}
    before = {(r["x"], r["y"]) for r in S1["things"] if r["y"] >= 3 and r["kind"] == "ring"}
    assert {(x, y) for y, (x, _) in answer.items()} <= before                # landed on a ring that was there
    rho = round_rho(linear_box(box(S3)), 3, 2)
    probs = {z: rho[index(z), index(z)].real for z in strings(3) if rho[index(z), index(z)].real > 1e-12}
    assert sorted(probs) == [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)]
    assert np.allclose(list(probs.values()), 0.25) and all(dot(z, S_PAIRS) == 0 for z in probs)

    # the clearances, on drawn extents: the answer-cords turn down at columns 5 (the lower)
    # and 6 (the upper) and cross nothing; each turn-down is clear of the other's rings
    for p in (p41, p42, p43):
        rings, cords = PC.cloth_drawn(p)
        row = lambda i: p.cloth["things"][i - 1]["y"]
        turns = {}                                  # answer row -> (its turn-down, reach)
        for kind, a, b, pts, reach in cords:
            if "fold" in (a, b):
                path = pts[::-1] if a == "fold" else pts                # ring end first, fold end last
                turns[row(b if a == "fold" else a)] = (path, reach)
        assert sorted(turns) == [3, 4]
        (up, up_r), (low, low_r) = turns[3], turns[4]
        assert up[-1][0] == up[-2][0] and low[-1][0] == low[-2][0]          # straight down to the fold
        assert up[-2][0] > low[-2][0]                                         # the lower turns first
        assert PC.polyline_gap(up, low) - up_r - low_r > 0.05                 # neither crosses the other
        for kind, a, b, pts, reach in cords:
            if "fold" not in (a, b):
                for q, r in ((up[-2:], up_r), (low[-2:], low_r)):
                    assert PC.polyline_gap(q, pts) - r - reach > 0.02         # the turn-downs cross nothing
        for i, c, X, Y, rr in rings:
            for own, (q, r) in ((3, (up[-2:], up_r)), (4, (low[-2:], low_r))):
                if row(i) != own:
                    assert PC.point_to_polyline((X, Y), q) - rr - r > 0.05    # clear of the other's rings
    assert not PC.TEX_ERRORS


# --- pictures (hand-built doubled diagrams; pyzx 0.10.6 refuses ground vertices) ---------

def doubled_round(M, compensate=True):
    """One round drawn doubled (Chapter 9): the circuit, and below it its mirror copy -- the
    same circuit, since every point and gate here is real (the mirror conjugates, and
    conjugating changes nothing).  Each copy: a red point on every rod and answer wire, a
    yellow box (Hadamard edge) into and out of each rod's green control, a red target on the
    answer wire for every 1 of M, a plain tie from control to target.  The discard: each
    answer wire joined to its OWN mirror copy by a plain edge -- the (unnormalized) cap.
    Outputs: rods 1..n, then mirror rods 1..n (the chapter draws the mirrors bottom-up; here
    they are listed in the same order as the rods, so the vector reshapes to rho[z, z']).
    Scalar: 2(n+m) points give 2^{n+m}, 2|M| bare ties 2^{-|M|}; add_power(-2(n+m-|M|))
    (= -4 for both instances here) leaves rho itself."""
    m, n = len(M), len(M[0])
    g = zx.Graph()
    outs = {}
    last_answer = {}
    for copy in (0, 1):
        base = copy * (n + m + 1)
        ctrl = []
        for i in range(n):
            p = g.add_vertex(VT.X, base + i, 0)
            c = g.add_vertex(VT.Z, base + i, 2 + i)
            o = g.add_vertex(VT.BOUNDARY, base + i, 10)
            g.add_edge((p, c), ET.HADAMARD); g.add_edge((c, o), ET.HADAMARD)
            ctrl.append(c); outs[(copy, i)] = o
        for j in range(m):
            prev = g.add_vertex(VT.X, base + n + j, 0)
            for i in range(n):
                if M[j][i]:
                    t = g.add_vertex(VT.X, base + n + j, 2 + i)
                    g.add_edge((prev, t)); g.add_edge((ctrl[i], t))
                    prev = t
            last_answer[(copy, j)] = prev
    for j in range(m):
        g.add_edge((last_answer[(0, j)], last_answer[(1, j)]))      # the cap: the discard
    g.set_inputs(())
    g.set_outputs(tuple(outs[(0, i)] for i in range(n)) + tuple(outs[(1, i)] for i in range(n)))
    if compensate:
        g.scalar.add_power(-2 * (n + m - sum(map(sum, M))))
    return g


def as_operator(g, n):
    """A doubled state on (rods, mirror rods) read as the matrix rho[z, z']."""
    return matrix_of(g).reshape(2 ** n, 2 ** n)


def gadget(s):
    """The parity gadget of s: a phase-free green dot on each rod where s has a 1 (legs in,
    out, and a tie), all tied to ONE phase-free red dot with |s| legs; the other rods bare."""
    n = len(s)
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, i, 0) for i in range(n)]
    outs = [g.add_vertex(VT.BOUNDARY, i, 4) for i in range(n)]
    red = g.add_vertex(VT.X, n, 2)
    for i in range(n):
        if s[i]:
            c = g.add_vertex(VT.Z, i, 2)
            g.add_edges([(ins[i], c), (c, outs[i]), (c, red)])
        else:
            g.add_edge((ins[i], outs[i]))
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    return g


def reduced_4rho():
    """The chapter's end of the rewrite: a green dot with legs on 1, 2, 1bar, 2bar and a
    green dot with legs on 3, 3bar (a cup).  Outputs 1, 2, 3, 1bar, 2bar, 3bar."""
    g = zx.Graph()
    outs = [g.add_vertex(VT.BOUNDARY, r, 2) for r in range(6)]
    big, cup = g.add_vertex(VT.Z, 1, 1), g.add_vertex(VT.Z, 4, 1)
    g.add_edges([(big, outs[k]) for k in (0, 1, 3, 4)] + [(cup, outs[2]), (cup, outs[5])])
    g.set_inputs(()); g.set_outputs(tuple(outs))
    return g


def scaled(g, power):
    h = g.copy()
    h.scalar.add_power(power)
    return h


def sketchbook():
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))       # the cap/cup is unnormalized

    # (a) the doubled round with the answer wires capped onto their mirrors: exactly rho.
    #     The pairs box and the weight-3 instance.
    for M, s in ((M_PAIRS, S_PAIRS), (M_W3, S_W3)):
        n = len(s)
        g = doubled_round(M)
        rho = round_rho(linear_box(M), n, len(M))
        assert np.allclose(as_operator(g, n), rho)
        assert not np.allclose(as_operator(doubled_round(M, compensate=False), n), rho)
        # DERIVED: full_reduce keeps the value
        h = g.copy(); simp.full_reduce(h)
        assert same_picture(h, g)

    # the pairs box, rewrite by rewrite as the chapter (and the Tortoise's fold) does it:
    # the bare picture is 4 rho ...
    bare = doubled_round(M_PAIRS, compensate=False)
    rho = round_rho(linear_box(M_PAIRS), 3, 2)
    assert np.allclose(as_operator(bare, 3), 4 * rho)
    # ... fusion alone (spider_simp) slides each answer wire's red point and red targets
    # together, ACROSS the cap, into one red dot: one tied to rods 1, 2, 1bar, 2bar, one to
    # 3, 3bar ("they run into one red ring"; no ring is worked past the fold)
    h = bare.copy()
    assert simp.spider_simp(h)
    reds = [v for v in h.vertices() if h.type(v) == VT.X and h.vertex_degree(v) > 1]
    assert sorted(h.vertex_degree(v) for v in reds) == [2, 4]
    assert same_picture(h, bare)
    # ... and the end of the chapter's four rewrites: a green dot on 1, 2, 1bar, 2bar beside a
    # cup on 3, 3bar, no scalar lost
    assert same_picture(bare, reduced_4rho())
    h = bare.copy(); simp.full_reduce(h)
    assert same_picture(h, reduced_4rho())
    # the cup on 3, 3bar is the identity on rod 3, and the four-legged dot |00><00| + |11><11|
    assert np.allclose(matrix_of(reduced_4rho()).reshape(8, 8),
                       np.kron(np.diag([1, 0, 0, 1]), np.eye(2)))

    # (b) the parity gadget: P_s = 2^{|s|/2 - 1} x gadget, i.e. add_power(|s| - 2).  Exact
    #     with no compensation only when |s| = 2 (the pairs box, and s = 100010).
    for s in (S_PAIRS, (1, 0, 0, 0, 1, 0), S_W3, (1, 1, 1, 1), (1, 0, 1, 1, 0)):
        k = sum(s)
        assert np.allclose(matrix_of(scaled(gadget(s), k - 2)), P_s(s))
        if k != 2:
            assert not np.allclose(matrix_of(gadget(s)), P_s(s))      # the scalar matters
        # the gadget is diagonal, zero exactly on the odd-parity strings
        d = np.diag(matrix_of(gadget(s))).real
        assert all(np.isclose(d[index(z)], 0) == (dot(z, s) == 1) for z in strings(len(s)))
        h = gadget(s).copy(); simp.full_reduce(h)                     # DERIVED
        assert same_picture(h, gadget(s))
    # for |s| = 2 the red dot has two legs, so it is a plain wire, and the two green dots fuse
    # through it into one: derived by id_simp then spider_simp
    h = gadget(S_PAIRS).copy()
    assert simp.id_simp(h) and simp.spider_simp(h)
    assert len([v for v in h.vertices() if h.type(v) != VT.BOUNDARY]) == 1
    assert same_picture(h, gadget(S_PAIRS))

    # (c) the round is 2^{|s|/2 - n} x gadget -- 1/4 for the pairs box; 2^{-5/2} for the
    #     weight-3 secret, where 2^{1-n} x gadget would be wrong
    for M, s in ((M_PAIRS, S_PAIRS), (M_W3, S_W3)):
        n, k = len(s), sum(s)
        rho = as_operator(doubled_round(M), n)
        assert np.allclose(rho, 2.0 ** (k / 2 - n) * matrix_of(gadget(s)))
        if k != 2:
            assert not np.allclose(rho, 2.0 ** (1 - n) * matrix_of(gadget(s)))
    assert np.allclose(as_operator(doubled_round(M_PAIRS), 3), matrix_of(gadget(S_PAIRS)) / 4)


if __name__ == "__main__":
    verdict("PairsBox", blackboard, sketchbook)
