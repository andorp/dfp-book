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
    """The First Challenge's prop pictures (identity card, sections 4 and 3e), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch13.py: six blocks,
    six calls, eight frames (F01-F08).  The ties below hold the transcription against this
    file's blackboard: a structural check of the pictures, not a second proof."""
    calls = DIALOGUE.check_layout("First", sizes=[1, 2, 1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("wishcasket", True, "", ("",), "knobs down, collars upright"),
        ("wishslip", True, "folded", ("first and fifth; disc black",), "the black face up"),
        ("wishslip", False, "folded; step=wish every place; expects=every mark", ("first and fifth; disc black",),
         "as Achilles expects"),
        ("wishslipsteps", False, "folded; from=off", ("first and fifth; disc black", "turn at the first; turn at the fifth"),
         "over to gilt; and over again to black"),
        ("wishslip", True, "", ("first and fifth",), "his slip, unfolded"),
        ("wishslip", False, "", ("second and sixth / first, third, fourth and sixth",),
         "hers, and the new slip beside it"),
        ("wishcloth", True, "", ("marks first and fifth",), "the slip's web")]
    PC.check_captions(DIALOGUE, "First", calls)
    PC.check_section(DIALOGUE, "First", calls)
    casket_h, slip_h = round(PC.FTOP - PC.FBOT, 2), round(PC.SLIP_FTOP - PC.SLIP_FBOT, 2)
    assert (casket_h, slip_h) == (2.82, 0.66)                         # read from the Lua
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("First")] == [
        [casket_h], [slip_h, slip_h], [slip_h], [slip_h], [slip_h], [2.74]]
    frames = [(c, fc, p, cap) for c in calls for fc, (p, cap) in zip(PC.frame_calls(c), PC.pictures(c))]
    assert len(frames) == 8
    F = dict(zip(["F%02d" % i for i in range(1, 9)], [p for _, _, p, _ in frames]))
    for c, fc, p, cap in frames:                                     # a strip frame IS its single call
        assert PC.drawn(PC.picture(fc)) == PC.drawn(p) and PC.picture(fc).log == p.log, fc.text

    # the anchors (card sections 4 and 3e.4): the paragraph before each block, and the one
    # after, in dialog.md; the raw file's following paragraphs are the same
    a1, a2, a3, a4, a5, a6 = DIALOGUE.anchors("First")
    f1, f2, f3, f4, f5, f6 = DIALOGUE.following("First")
    assert "a small brass nib on its rim" in a1                                           # D2
    assert f1.startswith("**Achilles:** *[standing, and bending to the lid]*")
    assert "Choose, then." in a2 and "*[She hands him the disc, black up.]*" in a2          # B1.2a
    assert f2.startswith("*[Every place lit. Achilles runs his thumb")
    assert "over to gilt, and then over again to black" in a3                              # B1.2
    assert f3.startswith("**Achilles:** Black. As if I'd made no marks at all.")
    assert "the first square inked, and the fifth" in a4
    assert f4.startswith("**Tortoise:** Could you do it with fewer?")
    assert "the second square inked, and the sixth" in a5
    assert "where it lies, to the left of the new one" in a5                              # D4
    assert f5.startswith("**Achilles:** *[sitting back]* What does one wish buy?")
    assert "the marks have become ties" in a6 and "running from left to right as he sees it" in a6
    assert "beneath the cords between, joined to none of them" in a6                      # D6
    assert f6.startswith("**Tortoise:** That is all a box of marks is.")
    assert DIALOGUE.following("First", DIALOGUE.raw) == [f1, f2, f3, f4, f5, f6]
    section = DIALOGUE.section("First")
    assert "a faded paper band, its seal, pasted across the front edge of its lid" in section   # D1
    assert "a small oval of enamel set in the lid's top" in section                              # D1
    assert "to the right of her folded one, as Achilles sees them" in section                    # D3
    assert "her two slips off the cloth" in section                                              # D5
    assert "runs down across the cords between" not in section                                   # D6's old words
    # the keyed words named before their pictures (card 3e.6): slip, disc
    assert section.index("slip") < section.index(calls[1].text) and section.index("disc") < section.index(calls[1].text)

    # the wrong rule `every mark` and every spoken reaction to it, in order (card 3e.1): the
    # prediction, "Choose, then." with the pale frame after it, the look (the two turns, drawn
    # by B1.2), the naming and the recanting, the Tortoise's agreement
    paras = DIALOGUE.paragraphs("First")
    at = lambda text: [i for i, q in enumerate(paras) if text in q]
    (said,), (choose,), (b12a,), (look,), (b12,), (black,), (one,) = (
        at("with more places lit, it has more to say."), at("**Tortoise:** Choose, then."), at(calls[1].text),
        at("over to gilt, and then over again to black"), at(calls[3].text),
        at("**Achilles:** Black. As if I'd made no marks at all."),
        at("**Tortoise:** One face, one answer, however many places you light."))
    assert said < choose < b12a < look < b12 < black < one and [choose, look, black, one] == [said + 1, said + 3,
                                                                                             said + 5, said + 6]
    assert "black up, exactly as she handed it to him" in paras[look]
    assert "Every mark turns it over, and a disc turned twice is back where it started." in paras[black]
    assert "whether the count came out odd or even" in paras[black]
    assert PC.comment_quotes("THE DISC'S STEP AND ITS WRONG RULE")             # its comment pinned (check_lua)
    rule = PC.rule_comment("THE DISC'S STEP AND ITS WRONG RULE")
    for q in ("Light every place at once, and the disc will have to account for every\n--                mark I made",
              "Black. As if I'd made no\n--                marks at all.", "(an even count of inked squares lit, more than none)"):
        assert q in rule, q

    # F01 (P1.1): the convention, every knob down and every collar upright, sealed
    p11 = F["F01"]
    assert p11.head == ("knobs=ddddddd kcollars=uuuuuuu wcollars=uuuuuuu granted=false windows=------- "
                        "seal=whole lid=down")
    assert p11.words == ("knob", "collar", "flag knob", "lid", "seal")
    v = p11.view
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
    PC.TEX_ERRORS.clear()
    for view in ("", "windows"):
        for state in ("", "granted"):
            w = PC.casket_view(PC.casket_state(view, state))
            for q in w.seal:
                assert OF.gap((list(q), OF.half("quad")), OF.plate(*w.plate)) > 0.05
    assert len(PC.casket_view(PC.casket_state("", "granted")).seal) == 2       # granted: parted
    assert not PC.TEX_ERRORS

    # F02-F05: the disc, his slip folded under his hand (Q-A).  The disc rule is the
    # blackboard's: handed y, the wish x, it comes back y xor a.x with a = 100010 (P1.2's slip)
    slip = (1, 0, 0, 0, 1, 0)
    face = {"black": 0, "gilt": 1}
    name = {0: "black", 1: "gilt"}
    every = (1, 1, 1, 1, 1, 1)
    assert F["F02"].head == "slip1=folded:[first fifth] disc=black" and F["F02"].words == ("slip", "disc")
    assert F["F02"].arcs == [] and F["F02"].notes == []
    # F03, the prediction: the folded slip pale, the disc at full strength GILT -- the
    # `every mark` reading (turned iff any inked square is lit) -- while its step line logs the
    # true face, black: y xor a.x = 0 xor (1 + 1 mod 2) = 0
    p03 = F["F03"]
    assert p03.head == "slip1=folded:[first fifth] disc=gilt" and p03.words == () and p03.arcs == []
    assert p03.notes == ["expects every mark (the slip paler, the disc at full strength)",
                         "step wish every place -> turned 2 time(s) -> slip1=folded:[first fifth] disc=black (no marks)"]
    assert p03.own[0] == "\\begin{scope}[transparency group, opacity=0.500]" and p03.own[1] == "\\end{scope}"
    lit_inked = [i for i in range(6) if slip[i] and every[i]]
    assert name[int(any(lit_inked))] == p03.wrong.disc == "gilt"            # the wrong rule's face
    assert name[0 ^ dot(slip, every)] == p03.after.disc == "black"           # the true one
    U_slip = U(linear(slip), 6)
    out = U_slip @ np.eye(128)[index(every + (0,))]
    assert name[int(np.argmax(np.abs(out))) % 2] == "black"                  # the oracle agrees
    # it differs only where the count of inked squares lit is even and more than none: the
    # Lua refuses the prediction on an odd count (it would agree) and with no step=wish
    PC.TEX_ERRORS.clear()
    PC.run_frame("slip", "folded; step=wish first and second; expects=every mark", "first and fifth; disc black", False)
    assert [s for s, _ in PC.TEX_ERRORS] == ["slip-rule-agrees"]
    PC.TEX_ERRORS.clear()
    for x in strings(6):
        if not any(x):
            continue
        places = ", ".join(PC.ORDNAME[i] for i in range(6) if x[i])
        for y in (0, 1):
            PC.TEX_ERRORS.clear()
            R = PC.run_frame("slip", "folded; step=wish " + places + "; expects=every mark",
                             "first and fifth; disc " + name[y], False)
            wrong = y ^ int(any(slip[i] and x[i] for i in range(6)))
            true = y ^ dot(slip, x)
            if wrong == true:
                assert [s for s, _ in PC.TEX_ERRORS] == ["slip-rule-agrees"]
            else:
                assert not PC.TEX_ERRORS and face[R.wrong.disc] == wrong and face[R.after.disc] == true
                assert sum(slip[i] and x[i] for i in range(6)) % 2 == 0          # an even count, more than none
    PC.TEX_ERRORS.clear()
    # F04, F05 (the strip B1.2, from=off): one turn at each inked square, one swing arc each,
    # over the disc, left to right -- the same arc for the same verb; the faces gilt, then black
    assert [F[n].head for n in ("F04", "F05")] == ["slip1=folded:[first fifth] disc=gilt",
                                                    "slip1=folded:[first fifth] disc=black"]
    assert [F[n].before.disc for n in ("F04", "F05")] == ["black", "gilt"]       # each from the frame before
    # the strip's frames as TeX runs them (the texlua comparison checked the strip's printed
    # frames byte for byte): one turn at each inked square, in the slip's order (l.487: his
    # thumb runs along the slip), each from the face the turn before left
    assert [(fc.keyed, fc.opt, fc.args) for c, fc, p, cap in frames[3:5]] == [
        (False, "folded; step=turn at the first", ("first and fifth; disc black",)),
        (False, "folded; step=turn at the fifth", ("first and fifth; disc gilt",))]
    assert PC.split(calls[3].args[1], ";") == ["turn at the " + PC.ORDNAME[i] for i in range(6) if slip[i]]
    arc = ("swing", (0.89, -0.11), (1.21, -0.11))
    for n in ("F04", "F05"):
        p = F[n]
        assert PC.arc_ends(p) == [arc] and p.ghosts == [] and p.words == ()
        assert p.notes[1] == ("swing arc, turn 1, over the disc, left to right, (0.890,-0.110) -> (1.210,-0.110): "
                              "inside the frame by 0.029 (centre line 0.040); clear of the disc's rim by 0.033 "
                              "(centre line 0.044) and of its shadow by 0.042 (edge to edge)")
        a = p.arcs[0]
        assert a.p1[0] > a.p0[0] and a.c1[1] > a.p0[1] and a.c2[1] > a.p1[1]    # left to right, bowed up: over
        assert all(q[1] > p.view.disc[1] for q in (a.p0, a.p1))                   # above the disc's centre
        assert PC.mark_clearance(p) > 0.01                                          # inside its frame
        c = p.clearances[0]
        assert c["rim"] > 0.02 and c["shadow"] > 0.02                             # off the disc and its shadow
        off = PC.twin(next(fc for _, fc, q, _ in frames if q is p))
        assert PC.drawn(off) == PC.drawn(p) and off.mark_tikz == [] and off.arcs == []
    # the telling part of the prediction, the disc's face, is over the legibility floor
    # (card 3e.5: the disc 0.279-0.313 cm across at 300 dpi; drawn 0.26 cm)
    assert 2 * PC.DISCR > 0.15
    # the disc is turned only at an inked square: no turn at the second (the Lua refuses it)
    PC.TEX_ERRORS.clear()
    PC.run_frame("slip", "folded; step=turn at the second", "first and fifth; disc black", False)
    assert [s for s, _ in PC.TEX_ERRORS] == ["slip-step-bare"]
    PC.TEX_ERRORS.clear()
    # the two turns ARE the wish of every place: black, as handed
    assert F["F05"].after.disc == p03.after.disc == "black"

    # F06 (P1.2): his slip, the first and the fifth inked -- and the unit wishes read it: U_f
    # with a = 100010, b = 0, on the wish for each knob alone, returns the flag a_i
    p12 = F["F06"]
    assert p12.head == "slip1=open:[first fifth] disc=none" and p12.words == ("slip", "inked")
    assert tuple(int(k in p12.after.slips[0]["inked"]) for k in range(1, 7)) == slip
    units = [tuple(int(j == i) for j in range(6)) for i in range(6)]
    flags = [int(np.argmax(np.abs(U_slip @ np.eye(128)[index(u + (0,))]))) % 2 for u in units]
    assert flags == list(slip)                                         # lit at the first and fifth only
    sv = p12.view.slips[0]
    assert len(sv["inked"]) == 2 and len(sv["squares"]) == 6

    # F07 (P1.3): both slips open, hers on the left as Achilles sees them; both answer his four
    # wishes alike (odd, even, odd, even), and the disc rule is y xor a.x for both
    p13 = F["F07"]
    assert p13.head == "slip1=open:[second sixth] slip2=open:[first third fourth sixth] disc=none"
    hers, other = (tuple(int(k in s["inked"]) for k in range(1, 7)) for s in p13.after.slips)
    assert (hers, other) == ((0, 1, 0, 0, 0, 1), (1, 0, 1, 1, 0, 1))
    wishes = [(1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0), (0, 0, 0, 0, 1, 1), (1, 0, 1, 0, 1, 0)]
    assert [dot(hers, q) for q in wishes] == [dot(other, q) for q in wishes] == [1, 0, 1, 0]
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
                # and the step=wish of a frame turns the disc the same way
                R = PC.run_frame("slip", "step=wish " + places, ink + "; disc " + handed, False)
                assert not PC.TEX_ERRORS and face[R.after.disc] == y ^ dot(marks, x)
    left, right = p13.view.slips
    assert left["x2"] < right["x1"]                                    # hers on the left
    stroke = 0.3 * PC.PT / 2
    assert right["x1"] - left["x2"] - 2 * stroke > 0.04                # the slips do not touch
    assert left["shadow"][2] < right["x1"] - stroke                    # nor her shadow his paper
    assert PC.SLIP_FRAME[1] - right["shadow"][2] > 0.02                  # his shadow inside the frame
    assert left["x1"] - PC.SLIP_FRAME[0] > 0.02                          # her paper too

    # F08 (P1.4): the slip's web -- a tie exactly at each inked square of P1.2's slip, which is
    # the a of this file's web(a): its green dots sit on exactly the marked rods
    p14 = F["F08"]
    assert p14.head == (
        "scene=web marks=[first fifth] first=loose+green0@2tied second=loose third=loose fourth=loose "
        "fifth=loose+green0@4tied sixth=loose flag=loose[red0@2 red0@4] threaded=false pushed=false far=false | "
        "cols=8 rows=7 y=0..6 near=false folded=false; gaps=5.5; 1:ring green 0 @2,0; 2:ring green 0 @4,4; "
        "3:ring red 0 @2,6; 4:ring red 0 @4,6; tie 1-3 []; tie 2-4 []; cord loose-loose [0,0 6.2,0]; "
        "cord loose-loose [0,1 6.2,1]; cord loose-loose [0,2 6.2,2]; cord loose-loose [0,3 6.2,3]; "
        "cord loose-loose [0,4 6.2,4]; cord loose-loose [0,5 6.2,5]; cord loose-loose [0,6 7,6]")
    assert p14.words == ("cord", "ring", "tie", "flag cord")
    things = p14.canon["things"]
    tied_rows = sorted(things[d["a"] - 1]["y"] for d in p14.canon["cords"] if d["kind"] == "tie")
    assert [r + 1 for r in tied_rows] == p12.after.slips[0]["inked"]
    g = web(slip)
    greens = [v for v in g.vertices() if g.type(v) == VT.Z]
    assert sorted(g.qubit(v) for v in greens) == tied_rows            # on the marked rods ...
    assert all(sum(g.type(u) == VT.X and g.qubit(u) == 6 for u in g.neighbors(v)) == 1 for v in greens)
                                                                       # ... each tied to its own red
    assert all(things[d["b"] - 1]["y"] == PC.FLAGROW for d in p14.canon["cords"] if d["kind"] == "tie")
    flag = [d for d in p14.canon["cords"] if d["kind"] == "cord" and d["pts"][0][1] == PC.FLAGROW][0]
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
