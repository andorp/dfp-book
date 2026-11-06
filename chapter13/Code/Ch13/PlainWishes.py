"""Challenge I: One Wish, One Answer  (the query model; the web; the classical baseline)
=====================================================================================
"Whatever I light, all it can ever say is whether the count came out odd or even."  -- Achilles

Claim: a plain (definite) query returns one bit, the flag y xor f(x); for every f the
oracle U_f |x>|y> = |x>|y xor f(x)> is a permutation, hence unitary, and U_f^2 = I
(Result 13.1); for f(x) = (a.x) xor b it is the web of CNOTs
U_f = X_flag^b prod_{a_i=1} CNOT_{i->flag}, and in pictures U_f = 2^{w/2} x (web), two
webs in a row = 2^{-w} x bare wires (Result 13.2); with b known, n definite queries
suffice and fewer never do, adaptively or not; with b unknown, n+1 (Result 13.3).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement, product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch13 as PC
import props_office as OF

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)


# --- plumbing: bit strings are tuples, top wire first; the flag is the last wire ---------

def strings(n):
    return list(product((0, 1), repeat=n))


def dot(a, x):
    """a.x, the dot product taken mod 2."""
    return sum(ai * xi for ai, xi in zip(a, x)) % 2


def index(bits):
    """Top wire first: the first bit is the most significant."""
    return int("".join(map(str, bits)) or "0", 2)


def U(f, n):
    """The oracle U_f |x>|y> = |x>|y xor f(x)> on n rods and one flag, as a matrix."""
    M = np.zeros((2 ** (n + 1), 2 ** (n + 1)), dtype=complex)
    for x in strings(n):
        for y in (0, 1):
            M[index(x + (y ^ f(x),)), index(x + (y,))] = 1
    return M


def cnot_to_flag(i, n):
    """CNOT with control rod i (0-based) and target the flag."""
    return U(lambda x: x[i], n)


def on_flag(A, n):
    return np.kron(np.eye(2 ** n), A)


def linear(a, b=0):
    return lambda x: dot(a, x) ^ b


# --- the web of ties, drawn -----------------------------------------------------------

def web(a, b=0, times=1):
    """`times` webs of a = (a_1..a_n), b in a row: on every marked rod a green dot, tied to
    its OWN red dot on the flag; unmarked rods straight; a red pi dot on the flag if b = 1.
    Inputs and outputs rod 1 .. rod n, flag (top wire first).  Bare, no scalar added."""
    n = len(a)
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in range(n + 1)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 10 * times + 2) for r in range(n + 1)]
    rod_last = list(ins[:n])
    flag_last = ins[n]
    col = 1
    for _ in range(times):
        for i in range(n):
            if a[i]:
                c = g.add_vertex(VT.Z, i, col)                     # the green control
                t = g.add_vertex(VT.X, n, col)                     # its own red target
                g.add_edge((rod_last[i], c)); rod_last[i] = c
                g.add_edge((flag_last, t)); flag_last = t
                g.add_edge((c, t))                                 # the tie
                col += 1
        if b:
            p = g.add_vertex(VT.X, n, col, phase=Fraction(1))      # red pi dot = X exactly
            g.add_edge((flag_last, p)); flag_last = p
            col += 1
    for i in range(n):
        g.add_edge((rod_last[i], outs[i]))
    g.add_edge((flag_last, outs[n]))
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    return g


def bare_wires(k):
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in range(k)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 1) for r in range(k)]
    for i, o in zip(ins, outs):
        g.add_edge((i, o))
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    return g


def scaled(g, power):
    """g times sqrt(2)^power (PyZX counts scalar powers in units of sqrt 2)."""
    h = g.copy()
    h.scalar.add_power(power)
    return h


# --- the classical baseline -------------------------------------------------------------

def min_queries(candidates, name):
    """The fewest definite queries of ANY adaptive method (a decision tree) that names
    name(c) with certainty for every candidate c = (a, b).  Exhaustive over all trees."""
    n = len(candidates[0][0])
    Q = strings(n)

    @lru_cache(maxsize=None)
    def solvable(S, depth):
        if len({name(c) for c in S}) == 1:
            return True
        if depth == 0:
            return False
        for q in Q:
            S0 = frozenset(c for c in S if dot(c[0], q) ^ c[1] == 0)
            S1 = S - S0
            if S0 and S1 and solvable(S0, depth - 1) and solvable(S1, depth - 1):
                return True
        return False

    d = 0
    while not solvable(frozenset(candidates), d):
        d += 1
    return d


def separates(queries, candidates, name):
    """Does this fixed list of definite queries name name(c) for every candidate?"""
    seen = {}
    for c in candidates:
        answers = tuple(dot(c[0], q) ^ c[1] for q in queries)
        if seen.setdefault(answers, name(c)) != name(c):
            return False
    return True


def blackboard():
    rng = np.random.default_rng(1301)

    # --- Result 13.1, for EVERY f: exhaustive for n = 1, 2, 3 (all 2^(2^n) functions),
    #     and a random non-linear f at n = 4 and n = 6
    for n in (1, 2, 3):
        for table in product((0, 1), repeat=2 ** n):
            Uf = U(lambda x: table[index(x)], n)
            assert np.allclose(Uf.sum(axis=0), 1) and np.allclose(Uf.sum(axis=1), 1)  # a permutation
            assert np.allclose(Uf.conj().T @ Uf, np.eye(2 ** (n + 1)))               # unitary
            assert np.allclose(Uf @ Uf, np.eye(2 ** (n + 1)))                        # U_f^2 = I
    for n in (4, 6):
        table = rng.integers(0, 2, 2 ** n)
        f = lambda x: int(table[index(x)])
        assert not any(all(f(x) == dot(a, x) ^ b for x in strings(n))                 # not a web
                       for a in strings(n) for b in (0, 1))
        Uf = U(f, n)
        assert np.allclose(Uf.conj().T @ Uf, np.eye(2 ** (n + 1)))
        assert np.allclose(Uf @ Uf, np.eye(2 ** (n + 1)))
        # "a superposed query is still one query": ONE application of U_f on a random
        # complex state is the superposition of the definite answers (linearity)
        psi = random_state(rng, 2 ** (n + 1))
        assert not np.allclose(psi, psi.conj())
        assert np.allclose(Uf @ psi, sum(psi[k] * Uf[:, k] for k in range(2 ** (n + 1))))

    # section 13.1's small box: n = 2, f(x1 x2) = x1
    Uf = U(lambda x: x[0], 2)
    e = lambda *bits: np.eye(8)[index(bits)]
    assert np.allclose(Uf @ e(1, 0, 0), e(1, 0, 1))      # the flag is flipped
    assert np.allclose(Uf @ e(1, 0, 1), e(1, 0, 0))      # ... and flipped back
    assert np.allclose(Uf @ e(0, 1, 0), e(0, 1, 0)) and np.allclose(Uf @ e(0, 1, 1), e(0, 1, 1))

    # --- Result 13.2, the web lemma: every a and b at n = 4, the casket's and the slip's
    #     secrets and random ones at n = 6
    cases = [(a, b) for a in strings(4) for b in (0, 1)]
    cases += [((0, 1, 0, 1, 0, 0), 0), ((1, 0, 0, 0, 1, 0), 0)]
    cases += [(tuple(rng.integers(0, 2, 6)), int(rng.integers(0, 2))) for _ in range(4)]
    for a, b in cases:
        n = len(a)
        W = np.linalg.matrix_power(on_flag(X, n), b)
        for i in range(n):
            if a[i]:
                W = W @ cnot_to_flag(i, n)
        assert np.allclose(U(linear(a, b), n), W)
    # the factors commute
    A, B = cnot_to_flag(1, 6), cnot_to_flag(3, 6)
    assert np.allclose(A @ B, B @ A) and np.allclose(A @ on_flag(X, 6), on_flag(X, 6) @ A)
    # "only linear f is a web": the webs give 2^(n+1) distinct functions of the 2^(2^n);
    # for n = 2, eight of the sixteen
    for n in (1, 2, 3, 4):
        tables = {tuple(linear(a, b)(x) for x in strings(n)) for a in strings(n) for b in (0, 1)}
        assert len(tables) == 2 ** (n + 1)
    assert len({tuple(linear(a, b)(x) for x in strings(2)) for a in strings(2) for b in (0, 1)}) == 8
    assert 2 ** (2 ** 2) == 16

    # --- the Terms (iii) and G2: with every collar upright, each wish-window shows its own
    #     knob and the flag window the flag knob turned once per marked lit knob: ONE
    #     definite reading.  The casket's secret, every one of the 128 settings.
    casket = (0, 1, 0, 1, 0, 0)
    Uf = U(linear(casket), 6)
    for x in strings(6):
        for y in (0, 1):
            out = Uf @ np.eye(128)[index(x + (y,))]
            assert np.isclose(out[index(x + (y ^ dot(casket, x),))], 1)   # a basis state:
            assert np.isclose(np.abs(out).sum(), 1)                        # one answer

    # --- the slip game (disc: black = dark = 0, gilt = lit = 1).  Achilles's slip: the first
    #     and the fifth squares inked.
    slip = (1, 0, 0, 0, 1, 0)
    disc = lambda wish, face, marks=slip: face ^ dot(marks, wish)    # turned once per inked lit place
    first_third = (1, 0, 1, 0, 0, 0)
    assert disc(first_third, 0) == 1                   # "Lit. One face, one answer."
    every = (1,) * 6
    assert disc(every, 0) == 0                         # "Black. As if I'd made no marks at all."
    # "Or two, or four": every slip with an even number of marks gives the same face
    assert all(disc(every, 0, m) == 0 for m in strings(6) if sum(m) % 2 == 0)
    # the disc says only odd or even: the 64 slips split into two classes, whatever is lit
    for wish in strings(6):
        assert len({disc(wish, 0, m) for m in strings(6)}) <= 2
    assert disc(first_third, 1) == 0                   # handed gilt, it comes back black
    assert disc(first_third, disc(first_third, 0)) == 0  # "The wish, asked twice, undoes itself."
    # a wish for every knob reads the slip: gilt at the first and the fifth only
    units = [tuple(int(j == i) for j in range(6)) for i in range(6)]
    assert [disc(u, 0) for u in units] == [1, 0, 0, 0, 1, 0]

    # the roles changed: the Tortoise's slip (second, sixth) and her third strip (first,
    # third, fourth, sixth) answer Achilles's four wishes alike: odd, even, odd, even
    hers, other = (0, 1, 0, 0, 0, 1), (1, 0, 1, 1, 0, 1)
    wishes = [(1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0), (0, 0, 0, 0, 1, 1), (1, 0, 1, 0, 1, 0)]
    assert [dot(hers, q) for q in wishes] == [1, 0, 1, 0]
    assert [dot(other, q) for q in wishes] == [1, 0, 1, 0] and hers != other
    # "Whatever fewer wishes you choose, I can ink you a second slip that answers them
    # all alike": the kernel argument, at n = 6, for random sets of up to five wishes
    for _ in range(200):
        k = int(rng.integers(1, 6))
        qs = [tuple(rng.integers(0, 2, 6)) for _ in range(k)]
        a = tuple(rng.integers(0, 2, 6))
        v = next(v for v in strings(6) if any(v) and all(dot(v, q) == 0 for q in qs))
        a2 = tuple(ai ^ vi for ai, vi in zip(a, v))
        assert a2 != a and [dot(a, q) for q in qs] == [dot(a2, q) for q in qs]

    # --- Result 13.3, exhaustively at n = 4 ---------------------------------------------
    n = 4
    known = [(a, 0) for a in strings(n)]                          # b known (the Terms: b = 0)
    unknown = [(a, b) for a in strings(n) for b in (0, 1)]        # b unknown
    secret = lambda c: c[0]
    # section 13.1's example: 110 and 011 cannot tell 100 from 011 = 100 xor 111
    assert [dot((1, 0, 0), q) for q in ((1, 1, 0), (0, 1, 1))] == [1, 0]
    assert [dot((0, 1, 1), q) for q in ((1, 1, 0), (0, 1, 1))] == [1, 0]
    unit = [tuple(int(j == i) for j in range(n)) for i in range(n)]
    # (i) b known: no multiset of fewer than n strings separates all 16 secrets (for fixed
    # lists this is also plain counting, 2^k < 2^n answer patterns); the n unit strings do
    for k in range(n):
        assert not any(separates(qs, known, secret)
                       for qs in combinations_with_replacement(strings(n), k))
    assert separates(unit, known, secret)
    # (ii) b unknown: no multiset of n strings determines a; the unit strings plus x = 0 do
    assert not any(separates(qs, unknown, secret)
                   for qs in combinations_with_replacement(strings(n), n))
    assert separates(unit + [(0,) * n], unknown, secret)
    assert not separates(unit, unknown, secret)                   # the unit strings alone fail
    # ADAPTIVE methods, exhaustively over every decision tree: the fewest queries that name
    # a with certainty are exactly n (b known) and n + 1 (b unknown)
    assert min_queries(known, secret) == n
    assert min_queries(unknown, secret) == n + 1
    # the kernel step, (ii): for k <= n queries a nonzero (v, c) with v != 0 exists
    for qs in combinations_with_replacement(strings(n), n):
        sols = [(v, c) for v in strings(n) for c in (0, 1)
                if (any(v) or c) and all(dot(v, q) ^ c == 0 for q in qs)]
        assert sols and all(any(v) for v, c in sols)

    pictures()


def pictures():
    """The First Challenge's prop pictures (identity card, section 4), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch13.py.  The
    ties below hold the transcription against this file's blackboard: a structural
    check of the pictures, not a second proof."""
    calls = DIALOGUE.check_layout("First", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("wishcasket", True, "", ("",), "knobs down, collars upright"),
        ("wishslip", True, "", ("first and fifth",), "his slip, unfolded"),
        ("wishslip", False, "", ("second and sixth / first, third, fourth and sixth",),
         "hers, and the new slip beside it"),
        ("wishcloth", True, "", ("marks first and fifth",), "the slip's web")]
    PC.check_captions(DIALOGUE, "First", calls)
    PC.check_section(DIALOGUE, "First", calls)
    casket_h, slip_h = round(PC.FTOP - PC.FBOT, 2), round(PC.SLIP_FTOP - PC.SLIP_FBOT, 2)
    assert (casket_h, slip_h) == (2.82, 0.66)                         # read from the Lua
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("First")] == [[casket_h], [slip_h], [slip_h], [2.74]]

    # the anchors (card section 4): the paragraph before each block, and the one after,
    # in dialog.md; the raw file's following paragraphs are the same
    a1, a2, a3, a4 = DIALOGUE.anchors("First")
    f1, f2, f3, f4 = DIALOGUE.following("First")
    assert "a small brass nib on its rim" in a1                                           # D2
    assert f1.startswith("**Achilles:** *[standing, and bending to the lid]*")
    assert "the first square inked, and the fifth" in a2
    assert f2.startswith("**Tortoise:** Could you do it with fewer?")
    assert "the second square inked, and the sixth" in a3
    assert "where it lies, to the left of the new one" in a3                              # D4
    assert f3.startswith("**Achilles:** *[sitting back]* What does one wish buy?")
    assert "the marks have become ties" in a4 and "running from left to right as he sees it" in a4
    assert "beneath the cords between, joined to none of them" in a4                      # D6
    assert f4.startswith("**Tortoise:** That is all a box of marks is.")
    assert DIALOGUE.following("First", DIALOGUE.raw) == [f1, f2, f3, f4]
    section = DIALOGUE.section("First")
    assert "a faded paper band, its seal, pasted across the front edge of its lid" in section   # D1
    assert "a small oval of enamel set in the lid's top" in section                              # D1
    assert "to the right of her folded one, as Achilles sees them" in section                    # D3
    assert "her two slips off the cloth" in section                                              # D5
    assert "runs down across the cords between" not in section                                   # D6's old words

    p11, p12, p13, p14 = (PC.picture(c) for c in calls)

    # P1.1: the convention, every knob down and every collar upright, sealed
    assert p11.dump == ("knobs=ddddddd kcollars=uuuuuuu wcollars=uuuuuuu granted=false windows=------- "
                        "seal=whole lid=down")
    assert p11.labels == ("knob", "collar", "flag knob", "lid", "seal")
    v = p11.shown
    assert v.view == "knobs" and v.lid == "down" and v.plate is not None and len(v.seal) == 1   # one band, whole
    assert [k for k, *_ in v.knobs] == list(range(1, PC.NROW + 1)) and not any(lit for *_, lit in v.knobs)
    assert not any(cw for *_, cw in v.collars)                        # every nib at the top
    assert [y for _, _, y, _ in v.collars] == list(PC.ROWY)           # the rows, from the top
    steps = [PC.ROWY[i] - PC.ROWY[i + 1] for i in range(PC.NROW - 1)]
    assert len({round(d, 9) for d in steps[:-1]}) == 1 and steps[-1] > steps[0]   # the flag set apart
    assert all(OF.pointer_tip(x, y, lit)[1] < y for _, x, y, lit in v.knobs)        # pointers down: dark
    # the casket's clearances, on drawn outlines (the soft shadows aside): in both stops and
    # with the knob either way, every nib clears its own knob (ball and pointer) and the
    # collars of the rows above and below it; the seal band clears the plate
    xc = v.collars[0][1]
    least = 1.0
    for k, y in enumerate(PC.ROWY, 1):
        for cw in (False, True):
            nib = OF.collar(xc, y, cw)["nib"]
            for lit in (False, True):
                knob = OF.knob(xc, y, lit)
                least = min(least, OF.gap(nib, knob["ball"]), OF.gap(nib, knob["pointer"]))
            for j in (k - 1, k + 1):
                if 1 <= j <= PC.NROW:
                    least = min(least, OF.gap(nib, OF.collar(xc, PC.ROWY[j - 1], False)["ring"]))
    assert least > 0.004, least
    for view in ("", "windows"):
        for state in ("", "granted"):
            w = PC.casket_view(PC.casket_state(view, state))
            for q in w.seal:
                assert OF.gap((list(q), OF.half("quad")), OF.plate(*w.plate)) > 0.05
    assert len(PC.casket_view(PC.casket_state("", "granted")).seal) == 2       # granted: parted
    assert not PC.TEX_ERRORS

    # P1.2: his slip, the first and the fifth inked -- and the unit wishes read it: U_f
    # with a = 100010, b = 0, on the wish for each knob alone, returns the flag a_i
    assert p12.dump == "slip1=open:[first fifth] disc=none" and p12.labels == ("slip", "inked")
    slip = tuple(int(k in p12.state.slips[0]["inked"]) for k in range(1, 7))
    assert slip == (1, 0, 0, 0, 1, 0)
    Uf = U(linear(slip), 6)
    units = [tuple(int(j == i) for j in range(6)) for i in range(6)]
    flags = [int(np.argmax(np.abs(Uf @ np.eye(128)[index(u + (0,))]))) % 2 for u in units]
    assert flags == list(slip)                                         # lit at the first and fifth only
    sv = p12.shown.slips[0]
    assert len(sv["inked"]) == 2 and len(sv["squares"]) == 6

    # P1.3: both slips open, hers on the left as Achilles sees them; both answer his four
    # wishes alike (odd, even, odd, even), and the disc rule is y xor a.x for both
    assert p13.dump == "slip1=open:[second sixth] slip2=open:[first third fourth sixth] disc=none"
    hers, other = (tuple(int(k in s["inked"]) for k in range(1, 7)) for s in p13.state.slips)
    assert (hers, other) == ((0, 1, 0, 0, 0, 1), (1, 0, 1, 1, 0, 1))
    wishes = [(1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0), (0, 0, 0, 0, 1, 1), (1, 0, 1, 0, 1, 0)]
    assert [dot(hers, q) for q in wishes] == [dot(other, q) for q in wishes] == [1, 0, 1, 0]
    face = {"black": 0, "gilt": 1}
    for marks in (hers, other):
        ink = " and ".join(PC.ORDNAME[i] for i in range(6) if marks[i])
        for x in strings(6):
            if not any(x):
                continue        # "wish" with no place lit cannot be written in the macro; its rule
                                # (no turn, the face unchanged) is the y here with x = 0: y xor 0
            places = ", ".join(PC.ORDNAME[i] for i in range(6) if x[i])
            for y, handed in ((0, "black"), (1, "gilt")):
                L = PC.slip_state("", ink + "; handed " + handed + ", wish " + places)
                assert not PC.TEX_ERRORS and face[L.disc] == y ^ dot(marks, x)
    left, right = p13.shown.slips
    assert left["x2"] < right["x1"]                                    # hers on the left
    stroke = 0.3 * PC.PT / 2
    assert right["x1"] - left["x2"] - 2 * stroke > 0.04                # the slips do not touch
    assert left["shadow"][2] < right["x1"] - stroke                    # nor her shadow his paper
    assert PC.SLIP_FRAME[1] - right["shadow"][2] > 0.02                  # his shadow inside the frame
    assert left["x1"] - PC.SLIP_FRAME[0] > 0.02                          # her paper too

    # P1.4: the slip's web -- a tie exactly at each inked square of P1.2's slip, which is
    # the a of this file's web(a): its green dots sit on exactly the marked rods
    assert p14.dump == (
        "scene=web marks=[first fifth] first=loose+green0@2tied second=loose third=loose fourth=loose "
        "fifth=loose+green0@4tied sixth=loose flag=loose[red0@2 red0@4] threaded=false pushed=false far=false | "
        "cols=8 rows=7 y=0..6 near=false folded=false; gaps=5.5; 1:ring green 0 @2,0; 2:ring green 0 @4,4; "
        "3:ring red 0 @2,6; 4:ring red 0 @4,6; tie 1-3 []; tie 2-4 []; cord loose-loose [0,0 6.2,0]; "
        "cord loose-loose [0,1 6.2,1]; cord loose-loose [0,2 6.2,2]; cord loose-loose [0,3 6.2,3]; "
        "cord loose-loose [0,4 6.2,4]; cord loose-loose [0,5 6.2,5]; cord loose-loose [0,6 7,6]")
    assert p14.labels == ("cord", "ring", "tie", "flag cord")
    things = p14.cloth["things"]
    tied_rows = sorted(things[d["a"] - 1]["y"] for d in p14.cloth["cords"] if d["kind"] == "tie")
    assert [r + 1 for r in tied_rows] == p12.state.slips[0]["inked"]
    g = web(slip)
    greens = [v for v in g.vertices() if g.type(v) == VT.Z]
    assert sorted(g.qubit(v) for v in greens) == tied_rows            # on the marked rods ...
    assert all(sum(g.type(u) == VT.X and g.qubit(u) == 6 for u in g.neighbors(v)) == 1 for v in greens)
                                                                       # ... each tied to its own red
    assert all(things[d["b"] - 1]["y"] == PC.FLAGROW for d in p14.cloth["cords"] if d["kind"] == "tie")
    flag = [d for d in p14.cloth["cords"] if d["kind"] == "cord" and d["pts"][0][1] == PC.FLAGROW][0]
    assert flag["pts"][-1][0] > PC.WEND                                # the flag cord the longer
    # the ties run down beneath the cords between, joined to none (a tie is the kit's
    # under-cord length, tied only to its two rings), and no tie comes near any other ring
    rings, cords = PC.cloth_drawn(p14)
    for kind, a, b, pts, reach in cords:
        if kind != "tie":
            continue
        between = [c for c in cords if c[0] == "cord" and pts[-1][1] < c[3][0][1] < pts[0][1]]
        assert len(between) == things[b - 1]["y"] - things[a - 1]["y"] - 1 > 0
        assert all(PC.polyline_gap(pts, c[3]) == 0 for c in between)   # it passes them all
        for i, _, X, Y, rr in rings:
            if i not in (a, b):
                assert PC.point_to_polyline((X, Y), pts) - rr - reach > 0.1
    assert not PC.TEX_ERRORS


def sketchbook():
    # (a) the web IS the oracle: bare, each tie is CNOT/sqrt2, so U_f = 2^{w/2} x web.
    #     Scalar: add_power(w) (sqrt2 per tie).  The red pi dot for b = 1 is X exactly.
    for a in [(0, 1, 0, 1, 0, 0), (1, 0, 0, 0, 1, 0)]:             # the casket; the slip
        for b in (0, 1):
            w = sum(a)
            Uf = U(linear(a, b), len(a))
            assert np.allclose(matrix_of(scaled(web(a, b), w)), Uf)
            assert not np.allclose(matrix_of(web(a, b)), Uf)        # the scalar is load-bearing
            assert not np.allclose(matrix_of(scaled(web(a, b), w)), U(linear(a, 1 - b), len(a)))
            # top wire first: the same web read upside down is a different box
            assert not np.allclose(matrix_of(scaled(web(a[::-1], b), w)), Uf)
    for a in strings(3):                                             # every web on three rods
        for b in (0, 1):
            assert np.allclose(matrix_of(scaled(web(a, b), sum(a))), U(linear(a, b), 3))

    # (b) asking twice: two webs in a row.  DERIVED by the rules (fusion, Hopf, the red pi
    #     dots fusing to 2 pi = no turn): full_reduce leaves bare wires and the scalar 2^{-w}.
    for a in [(0, 1, 0, 1, 0, 0), (1, 0, 0, 0, 1, 0), (1, 1, 1, 0, 1, 1)]:
        for b in (0, 1):
            w, n = sum(a), len(a)
            g = web(a, b, times=2)
            assert np.allclose(matrix_of(scaled(g, 2 * w)), np.eye(2 ** (n + 1)))
            h = g.copy()
            simp.full_reduce(h)
            assert h.num_vertices() == 2 * (n + 1)                   # boundaries only
            for i, o in zip(h.inputs(), h.outputs()):                # each input wired
                assert list(h.neighbors(i)) == [o]                   # straight to its output
                assert h.edge_type(h.edge(i, o)) == ET.SIMPLE
            assert np.isclose(h.scalar.to_number(), 2.0 ** (-w))     # Hopf's price, 1/2 per tie
            assert same_picture(h, g)
            assert same_picture(scaled(h, 2 * w), bare_wires(n + 1))
            assert not same_picture(h, bare_wires(n + 1))            # not without 2^{-w}
    # the picture form of unitarity: the web followed by its MIRROR image (PyZX's adjoint,
    # the picture reflected left-right and conjugated) is bare wires, times 2^{-w}.  The web
    # is real and the mirror of a tie is the same tie, so this is the asking-twice picture.
    g = web((0, 1, 0, 1, 0, 0))
    assert same_picture(g.adjoint(), g)                                 # its own mirror
    h = g.copy()
    h.compose(g.adjoint())                                              # (compose mutates h)
    simp.full_reduce(h)
    assert h.num_vertices() == 14 and same_picture(scaled(h, 4), bare_wires(7))


if __name__ == "__main__":
    verdict("PlainWishes", blackboard, sketchbook)
