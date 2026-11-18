"""Challenge II: The Blind Vote  (the headline; The Theorem, Twice; the master)
===============================================================================
"Ask the pair, not the one."  -- Tortoise
"Both bells rang. So the middle strand disagrees with both its neighbours, and it's the middle I mend."  -- Crab

Claim: one gust, mended blind -- for every X^e in E_X = {I, X1, X2, X3} and every outcome s,
Enc^dag X^c(s) P_s X^e Enc = [s = sigma(e)] I, exactly, scalar 1 (Result 18.7).  The checks
hear the gust and never the message (Result 18.6); two gusts give X (Result 18.9); every
mixture of single gusts is undone (Result 18.10).  The master, The Storm Line, is built as
drawn and equals [s = sigma(e)] (I | X | 0) for all 2^8 labellings (Result 18.24); PyZX's
full_reduce derives the bare wire, the single X or zero.  A bell tied twice to one green
spider comes off with (1 + e^{i theta})/2 (Result 18.8); a bell asks a pair (Result 18.5).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction
import copy
import itertools
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
I8 = np.eye(8)

ENC = np.outer(ket(0, 0, 0), ket(0)) + np.outer(ket(1, 1, 1), ket(1))   # 8 x 2, top wire first
DEC = ENC.conj().T                                                     # the good-branch decoder


def on(k, A):
    ops = [I2, I2, I2]
    ops[k] = A
    return kron(*ops)


def X_pattern(e):
    return kron(*[X if b else I2 for b in e])


Z1Z2, Z2Z3 = on(0, Z) @ on(1, Z), on(1, Z) @ on(2, Z)


def P(s):
    s1, s2 = s
    return (I8 + (-1) ** s1 * Z1Z2) @ (I8 + (-1) ** s2 * Z2Z3) / 4


def sigma(e):
    return (e[0] ^ e[1], e[1] ^ e[2])


CORR = {(0, 0): (0, 0, 0), (1, 0): (1, 0, 0), (1, 1): (0, 1, 0), (0, 1): (0, 0, 1)}
SINGLE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]                  # E_X as patterns
OUTCOMES = list(product((0, 1), repeat=2))
BITS3 = list(product((0, 1), repeat=3))


def storm_value(e, s, c):
    """Result 18.24's formula: [s = sigma(e)] * (I if e+c = 000, X if 111, 0 otherwise)."""
    if tuple(s) != sigma(e):
        return np.zeros((2, 2))
    f = tuple(a ^ b for a, b in zip(e, c))
    return I2 if f == (0, 0, 0) else X if f == (1, 1, 1) else np.zeros((2, 2))


MARKS = {
    "dark": ket(0), "lit": ket(1),
    "at rest": (ket(0) + ket(1)) / np.sqrt(2), "a half": (ket(0) - ket(1)) / np.sqrt(2),
    "a quarter round": (ket(0) + 1j * ket(1)) / np.sqrt(2),
    "a quarter back": (ket(0) - 1j * ket(1)) / np.sqrt(2),
}


def blackboard():
    rng = np.random.default_rng(1811)
    # Z, X, the P_s and Enc are real, so no transpose/dagger mix-up can hide in them; the
    # messages and the Kraus weights below are the non-real, generic inputs.

    # --- the checks: projectors, orthogonal, complete
    assert np.allclose(sum(P(s) for s in OUTCOMES), I8)
    for s, t in product(OUTCOMES, repeat=2):
        assert np.allclose(P(s) @ P(t), P(s) if s == t else 0)
    assert np.allclose(P((0, 0)) @ ENC, ENC)                  # the code space is "agree, agree"

    # --- Result 18.6: the bells hear the gust, not the message (all 8 patterns, all 4 outcomes)
    psi = random_state(rng)
    assert not np.allclose(psi, psi.real)
    for e in BITS3:
        for s in OUTCOMES:
            lhs = P(s) @ X_pattern(e) @ ENC
            assert np.allclose(lhs, (s == sigma(e)) * X_pattern(e) @ ENC)
            prob = np.linalg.norm(lhs @ psi) ** 2
            assert np.isclose(prob, float(s == sigma(e)))     # certain, whatever the message

    # the dialogue's headline run: a quarter round, a gust on the middle strand, both bells
    q = MARKS["a quarter round"]
    state = X_pattern((0, 1, 0)) @ ENC @ q
    assert np.allclose(state, (ket(0, 1, 0) + 1j * ket(1, 0, 1)) / np.sqrt(2))
    assert np.isclose(np.linalg.norm(P((1, 1)) @ state), 1)   # both ring, with certainty
    assert np.allclose(X_pattern(CORR[(1, 1)]) @ state, ENC @ q)   # the middle mended: exact

    # --- Result 18.7, the headline: exact, scalar 1, branch by branch
    for e in SINGLE:
        for s in OUTCOMES:
            lhs = DEC @ X_pattern(CORR[s]) @ P(s) @ X_pattern(e) @ ENC
            assert np.allclose(lhs, (s == sigma(e)) * I2)
    assert sorted(sigma(e) for e in SINGLE) == sorted(OUTCOMES)      # sigma one-to-one on E_X
    assert all(CORR[sigma(e)] == e for e in SINGLE)                  # c(sigma(e)) = e
    # negative controls: no mending, or mending the wrong strand, does not give the message back
    for e in SINGLE[1:]:
        s = sigma(e)
        assert not np.allclose(DEC @ P(s) @ X_pattern(e) @ ENC, I2)
        other = next(f for f in SINGLE[1:] if f != e)
        assert not np.allclose(DEC @ X_pattern(other) @ P(s) @ X_pattern(e) @ ENC, I2)

    # --- Result 18.9: every pattern, [s = sigma(e)] X^[wt(e) >= 2]
    for e in BITS3:
        for s in OUTCOMES:
            lhs = DEC @ X_pattern(CORR[s]) @ P(s) @ X_pattern(e) @ ENC
            assert np.allclose(lhs, (s == sigma(e)) * (X if sum(e) >= 2 else I2))

    # --- Result 18.24 on the blackboard: the free-c formula for all 2^8 labellings
    for e, s, c in product(BITS3, OUTCOMES, BITS3):
        lhs = DEC @ X_pattern(c) @ P(s) @ X_pattern(e) @ ENC
        assert np.allclose(lhs, storm_value(e, s, c))

    # --- Result 18.10: every mixture of single gusts, E = sum_e a_e X^e with complex a_e
    for _ in range(3):
        a = rng.normal(size=4) + 1j * rng.normal(size=4)
        E = sum(ae * X_pattern(e) for ae, e in zip(a, SINGLE))
        for s in OUTCOMES:
            k = SINGLE.index(next(e for e in SINGLE if sigma(e) == s))   # sigma^{-1}(s)
            assert np.allclose(X_pattern(CORR[s]) @ P(s) @ E @ ENC, a[k] * ENC)

    # the channel: five random non-real Kraus operators in span E_X, trace preserving on the
    # code (sum_i Enc^dag K_i^dag K_i Enc = I, i.e. the 5 x 4 weight matrix has norm 1)
    A = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
    A /= np.linalg.norm(A)
    kraus = [sum(A[i, j] * X_pattern(e) for j, e in enumerate(SINGLE)) for i in range(5)]
    assert np.allclose(sum(DEC @ K.conj().T @ K @ ENC for K in kraus), I2)
    B = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho = B @ B.conj().T
    rho /= np.trace(rho)
    assert not np.allclose(rho, rho.real)
    coded = ENC @ rho @ DEC
    noisy = sum(K @ coded @ K.conj().T for K in kraus)
    recovered = sum(X_pattern(CORR[s]) @ P(s) @ noisy @ P(s) @ X_pattern(CORR[s]) for s in OUTCOMES)
    assert np.allclose(recovered, coded)                       # undone exactly
    assert not np.allclose(noisy, coded)                       # (the storm did do something)

    # --- the storm line's props: the kitchen glass lifted at the mark sent (|<psi|out>|^2)
    def glass(msg, e):
        """Send msg, gust e, the Crab mends by the bells; return the clear-probability."""
        coded = ENC @ msg
        hit = X_pattern(e) @ coded
        clear = 0.0
        for s in OUTCOMES:
            branch = X_pattern(CORR[s]) @ P(s) @ hit
            if np.linalg.norm(branch) > 1e-12:
                out = DEC @ branch                              # the joining ring
                clear += abs(np.vdot(msg, out)) ** 2
        return clear
    for name, msg in MARKS.items():
        for e in SINGLE:                                        # one gust on any strand, or none
            assert np.isclose(glass(msg, e), 1), name
    two = (1, 1, 0)                                             # two gusts at once
    for name in ("dark", "lit", "a quarter round", "a quarter back"):
        assert np.isclose(glass(MARKS[name], two), 0)           # cloudy: the vote went wrong
    for name in ("at rest", "a half"):
        assert np.isclose(glass(MARKS[name], two), 1)           # clear: X leaves them alone

    pictures()


# ------------------------------------------------------------------------------ prop pictures

SECOND_DUMPS = (
    "mend=0,2,0 tell=0,0,0 clip=none glass=clear",
    "drawn | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; 1:ring red 0 @2.5,0; 2:ring green 0 @3.9,0; "
    "3:ring red 0 @6.6,0; 4:hub red 0 @5,0.5; 5:ring green 0 @1,1; 6:ring red 0 @2.5,1; 7:ring green 0 @3.9,1; "
    "8:ring red 0 @6.6,1; 9:ring green 0 @8,1; 10:hub red 0 @5,1.5; 11:ring red 0 @2.5,2; 12:ring green 0 @3.9,2; "
    "13:ring red 0 @6.6,2; tie 10-12 []; tie 2-4 []; tie 4-7 []; tie 7-10 []; cord 11-12 [2.5,2 3.9,2]; "
    "cord 12-13 [3.9,2 6.6,2]; cord 1-2 [2.5,0 3.9,0]; cord 1-5 [2.5,0 1,1]; cord 2-3 [3.9,0 6.6,0]; cord 3-9 [6.6,0 8,1]; "
    "cord 5-11 [1,1 2.5,2]; cord 5-6 [1,1 2.5,1]; cord 6-7 [2.5,1 3.9,1]; cord 7-8 [3.9,1 6.6,1]; cord 8-9 [6.6,1 8,1]; "
    "cord 9-13 [8,1 6.6,2]; cord loose-5 [0,1 1,1]; cord loose-9 [9,1 8,1]",
    "halves,through tiehalves=2 | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; 1:ring red 0 @2.5,0; "
    "2:ring green 0 @3.9,0; 3:ring red 0 @6.6,0; 4:hub red 12 @5,0.5; 5:ring green 0 @1,1; 6:ring red 0 @2.5,1; "
    "7:ring green 0 @3.9,1; 8:ring red 12 @5,1; 9:ring red 12 @6.6,1; 10:ring green 0 @8,1; 11:hub red 12 @5,1.5; "
    "12:ring red 0 @2.5,2; 13:ring green 0 @3.9,2; 14:ring red 0 @6.6,2; tie 11-13 []; tie 2-4 []; tie 4-7 []; "
    "tie 7-11 []; cord 10-14 [8,1 6.6,2]; cord 12-13 [2.5,2 3.9,2]; cord 13-14 [3.9,2 6.6,2]; cord 1-2 [2.5,0 3.9,0]; "
    "cord 1-5 [2.5,0 1,1]; cord 2-3 [3.9,0 6.6,0]; cord 3-10 [6.6,0 8,1]; cord 5-12 [1,1 2.5,2]; cord 5-6 [1,1 2.5,1]; "
    "cord 6-7 [2.5,1 3.9,1]; cord 7-9 [3.9,1 6.6,1]; cord 9-10 [6.6,1 8,1]; cord loose-10 [9,1 8,1]; cord loose-5 [0,1 1,1]",
    "halves,through,slidin,together,off,straight | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; "
    "cord loose-loose [0,1 9,1]",
)


def signature(g):
    """Each vertex's type, degree and the types of its neighbours, as a multiset."""
    return sorted((g.type(v), len(list(g.neighbors(v))), tuple(sorted(g.type(u) for u in g.neighbors(v))))
                  for v in g.vertices())


FRAME_DUMPS = (                                        # the log's dump, frame by frame (card 3e.6, item 6)
    'mend=0,2,0 tell=0,0,0 clip=none glass=clear',
    'drawn | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; 1:ring red 0 @2.5,0; 2:ring green 0 '
    '@3.9,0; 3:ring red 0 @6.6,0; 4:hub red 0 @5,0.5; 5:ring green 0 @1,1; 6:ring red 0 @2.5,1; 7:ring green 0 '
    '@3.9,1; 8:ring red 0 @6.6,1; 9:ring green 0 @8,1; 10:hub red 0 @5,1.5; 11:ring red 0 @2.5,2; 12:ring green 0 '
    '@3.9,2; 13:ring red 0 @6.6,2; tie 10-12 []; tie 2-4 []; tie 4-7 []; tie 7-10 []; cord 11-12 [2.5,2 3.9,2]; '
    'cord 12-13 [3.9,2 6.6,2]; cord 1-2 [2.5,0 3.9,0]; cord 1-5 [2.5,0 1,1]; cord 2-3 [3.9,0 6.6,0]; cord 3-9 '
    '[6.6,0 8,1]; cord 5-11 [1,1 2.5,2]; cord 5-6 [1,1 2.5,1]; cord 6-7 [2.5,1 3.9,1]; cord 7-8 [3.9,1 6.6,1]; '
    'cord 8-9 [6.6,1 8,1]; cord 9-13 [8,1 6.6,2]; cord loose-5 [0,1 1,1]; cord loose-9 [9,1 8,1]',
    'halves,through tiehalves=2 | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; 1:ring red 0 '
    '@2.5,0; 2:ring green 0 @3.9,0; 3:ring red 0 @6.6,0; 4:hub red 12 @5,0.5; 5:ring green 0 @1,1; 6:ring red 0 '
    '@2.5,1; 7:ring green 0 @3.9,1; 8:ring red 12 @5,1; 9:ring red 12 @6.6,1; 10:ring green 0 @8,1; 11:hub red 12 '
    '@5,1.5; 12:ring red 0 @2.5,2; 13:ring green 0 @3.9,2; 14:ring red 0 @6.6,2; tie 11-13 []; tie 2-4 []; tie '
    '4-7 []; tie 7-11 []; cord 10-14 [8,1 6.6,2]; cord 12-13 [2.5,2 3.9,2]; cord 13-14 [3.9,2 6.6,2]; cord 1-2 '
    '[2.5,0 3.9,0]; cord 1-5 [2.5,0 1,1]; cord 2-3 [3.9,0 6.6,0]; cord 3-10 [6.6,0 8,1]; cord 5-12 [1,1 2.5,2]; '
    'cord 5-6 [1,1 2.5,1]; cord 6-7 [2.5,1 3.9,1]; cord 7-9 [3.9,1 6.6,1]; cord 9-10 [6.6,1 8,1]; cord loose-10 '
    '[9,1 8,1]; cord loose-5 [0,1 1,1]',
    'halves,through,slidin | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; 1:ring red 0 @2.5,0; '
    '2:ring green 0 @3.9,0; 3:ring red 0 @6.6,0; 4:hub red 0 @5,0.5; 5:ring green 0 @1,1; 6:ring red 0 @2.5,1; '
    '7:ring green 0 @3.9,1; 8:ring red 0 @6.6,1; 9:ring green 0 @8,1; 10:hub red 0 @5,1.5; 11:ring red 0 @2.5,2; '
    '12:ring green 0 @3.9,2; 13:ring red 0 @6.6,2; tie 10-12 []; tie 2-4 []; tie 4-7 []; tie 7-10 []; cord 11-12 '
    '[2.5,2 3.9,2]; cord 12-13 [3.9,2 6.6,2]; cord 1-2 [2.5,0 3.9,0]; cord 1-5 [2.5,0 1,1]; cord 2-3 [3.9,0 '
    '6.6,0]; cord 3-9 [6.6,0 8,1]; cord 5-11 [1,1 2.5,2]; cord 5-6 [1,1 2.5,1]; cord 6-7 [2.5,1 3.9,1]; cord 7-8 '
    '[3.9,1 6.6,1]; cord 8-9 [6.6,1 8,1]; cord 9-13 [8,1 6.6,2]; cord loose-5 [0,1 1,1]; cord loose-9 [9,1 8,1]',
    'halves,through,slidin,part1 | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; 1:ring red 0 '
    '@6.6,0; 2:hub red 0 @5,0.5; 3:ring green 0 @1,1; 4:ring red 0 @2.5,1; 5:ring green 0 @3.9,1; 6:ring red 0 '
    '@6.6,1; 7:ring green 0 @8,1; 8:hub red 0 @5,1.5; 9:ring red 0 @2.5,2; 10:ring green 0 @3.9,2; 11:ring red 0 '
    '@6.6,2; tie 2-3 []; tie 2-5 []; tie 5-8 []; tie 8-10 []; cord 10-11 [3.9,2 6.6,2]; cord 1-3 [6.6,0 3.9,0 '
    '2.5,0 1,1]; cord 1-7 [6.6,0 8,1]; cord 3-4 [1,1 2.5,1]; cord 3-9 [1,1 2.5,2]; cord 4-5 [2.5,1 3.9,1]; cord '
    '5-6 [3.9,1 6.6,1]; cord 6-7 [6.6,1 8,1]; cord 7-11 [8,1 6.6,2]; cord 9-10 [2.5,2 3.9,2]; cord loose-3 [0,1 '
    '1,1]; cord loose-7 [9,1 8,1]',
    'halves,through,slidin,part2 | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; 1:hub red 0 '
    '@5,0.5; 2:ring green 0 @1,1; 3:ring red 0 @2.5,1; 4:ring green 0 @3.9,1; 5:ring red 0 @6.6,1; 6:hub red 0 '
    '@5,1.5; 7:ring red 0 @2.5,2; 8:ring green 0 @3.9,2; 9:ring red 0 @6.6,2; tie 1-2 []; tie 1-4 []; tie 4-6 []; '
    'tie 6-8 []; cord 2-3 [1,1 2.5,1]; cord 2-5 [1,1 2.915,0.291 6.171,0.291 6.81,0.75 6.6,1]; cord 2-7 [1,1 '
    '2.5,2]; cord 2-9 [1,1 2.707,0.145 6.385,0.145 7.02,0.6 7.3,1 6.6,2]; cord 3-4 [2.5,1 3.9,1]; cord 4-5 [3.9,1 '
    '6.6,1]; cord 7-8 [2.5,2 3.9,2]; cord 8-9 [3.9,2 6.6,2]; cord loose-2 [0,1 1,1]; cord loose-2 [9,1 8,1 6.6,0 '
    '2.5,0 1,1]',
    'halves,through,slidin,part3 tiedtwice=1 | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; 1:hub '
    'red 0 @1,0.5; 2:ring green 0 @1,1; 3:hub red 0 @5,1.5; 4:ring red 0 @2.5,2; 5:ring green 0 @3.9,2; 6:ring '
    'red 0 @6.6,2; tie 2-3 []; tie 3-5 []; cord 2-4 [1,1 2.5,2]; cord 2-6 [1,1 2.707,0.145 6.385,0.145 7.02,0.6 '
    '7.3,1 6.6,2]; cord 4-5 [2.5,2 3.9,2]; cord 5-6 [3.9,2 6.6,2]; cord loose-2 [0,1 1,1]; cord loose-2 [9,1 8,1 '
    '6.6,0 2.5,0 1,1]',
    'halves,through,slidin,together tiedtwice=2 | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; '
    '1:hub red 0 @1,0.5; 2:ring green 0 @1,1; 3:hub red 0 @1,1.5; cord loose-2 [0,1 1,1]; cord loose-2 [9,1 1,1]',
    'halves,through,slidin,together,off | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; 1:ring '
    'green 0 @1,1; cord loose-1 [0,1 1,1]; cord loose-1 [9,1 1,1]',
    'halves,through,slidin,together,off,straight | cols=10 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; '
    'cord loose-loose [0,1 9,1]',
)


# an arc ends beside its ring: a hop HOPR from its centre, a route ROUTER, the third strand's 0.16 (its end
# set back past the parting ring's ties); every one within NEAR
NEAR = 0.17
GRID = re.compile(r"no ghost of the (\w+) (\w+) \(grid ([-\d.]+),([-\d.]+)\): (.+)")


def check_cloth_marks(R, scene_of, opts, who, extra=()):
    """A cloth step's marks, on the drawn extent (card 3e.6): every piece of the state before that
    left its place has a ghost there, the size of a ring, or the log says why it has none; every
    ghost stands on such a piece (or on one of `extra`, a place that is no piece of the cloth's
    table); every arc runs from beside a piece of the state before to beside a piece of the step's
    state (or to one of this step's ghosts: a ring slid into one that goes with it), another place, bowed at least the Lua's floor and clear of the net by its clearance;
    pieces gone and no arc drawn: the log says why."""
    S0, S1 = scene_of(R.before), scene_of(R.after)
    g = PC.G(S1, opts, who)
    at = lambda S: {(t["x"], t["y"]): (t["kind"], t["c"]) for t in S["things"]}
    old, new = at(S0), at(S1)
    gone = {p for p in old if p not in new}
    hidden = {p for p in old if p in new and new[p] != old[p]}
    said = {(float(x), float(y)): why for c, kw, x, y, why in (m.groups() for m in map(GRID.fullmatch, R.notes) if m)}
    for p in hidden:
        assert said.get(p) == "the piece lying on its place now hides it", p
    ghosted = set()
    for q in R.ghosts:
        assert np.isclose(q[2], g.RO), q.what
        here = [p for p in gone if np.allclose(q[:2], g.XY(p), atol=1e-3)]
        assert here or any(np.allclose(q[:2], e, atol=1e-3) for e in extra), q.what
        ghosted |= set(here)
    assert gone - ghosted <= set(said), (gone, ghosted, said)          # a piece gone: ghosted, or why not
    assert not (ghosted & set(said)), ghosted & set(said)
    near = lambda pt, places: min(((np.hypot(pt[0] - c[0], pt[1] - c[1]), c) for c in places), key=lambda u: u[0])
    for a in R.arcs:
        assert a.kind == "slide" and a.sag >= PC.MINSAG - 1e-9 and a.fit[0] >= PC.MINCLEAR - 1e-9, (a.what, a.sag, a.fit)
        d0, c0 = near(a.frm, [g.XY(p) for p in old] + list(extra))
        d1, c1 = near(a.to, [g.XY(p) for p in new] + [tuple(q[:2]) for q in R.ghosts])   # or this step's ghost: both gone
        assert d0 < NEAR and d1 < NEAR and c0 != c1, (a.what, d0, d1)
    if gone and not R.arcs:
        assert any(n.startswith("no arc") for n in R.notes), R.step
    assert len(R.mark_tikz) >= len(R.arcs) + len(R.ghosts)


STEPS = ["halves set", "gust's half sent through", "halves slid in", "green rings slid together", "bells off", "drawn straight"]
PARTS = ["first tell ring in", "joining ring in", "middle strand in", "third strand in"]
# fixed parts (card 3e.6, item 6): every piece of the line on the cloth, at its place, and the frames
# of P2.2 .. P2.4 (0 .. 8) it lies there in -- each green ring until its own part-step, each red ring at
# rest until a green ring passes it or its loop is lifted, each bell until its strand's part-step lays
# it by the parting ring, which stays at the door end until it is unclipped
LIFE = {
    ("ring", "green", 1.0, 1.0): range(0, 8), ("ring", "green", 3.9, 0.0): range(0, 3),
    ("ring", "green", 8.0, 1.0): range(0, 4), ("ring", "green", 3.9, 1.0): range(0, 5),
    ("ring", "green", 3.9, 2.0): range(0, 6),
    ("ring", "red", 2.5, 0.0): range(0, 3), ("ring", "red", 6.6, 0.0): range(0, 4),
    ("ring", "red", 2.5, 1.0): range(0, 5), ("ring", "red", 6.6, 1.0): range(0, 5),
    ("ring", "red", 2.5, 2.0): range(0, 6), ("ring", "red", 6.6, 2.0): range(0, 6),
    ("ring", "red", 5.0, 1.0): range(1, 2),                       # the half on the mending cord (P2.3)
    ("hub", "red", 5.0, 0.5): range(0, 5), ("hub", "red", 5.0, 1.5): range(0, 6),
    ("hub", "red", 1.0, 0.5): range(5, 7), ("hub", "red", 1.0, 1.5): range(6, 7),
}
ONLYCORD = ('a red ring at rest on a strand is only cord ("And a red ring at rest on a strand is only cord"): '
            'not a piece that moves')


def lc_scene(K):
    return PC.lc_scene2(K)[0]


def pictures():
    """The Second Challenge's prop pictures (identity card, sections 4 and 3e), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch18.py, and held against
    this file's storm_line (graph_of: a structural reading of each cloth dump, not a proof).  Round
    4 draws the mending on the cloth move by move: the gust's half sent through, the halves slid in,
    the green rings slid together one strand at a time, the bells lifted off, the cord drawn
    straight; every frame is the bare wire, exactly."""
    calls = DIALOGUE.check_layout("Second", sizes=[1, 1, 1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("stormline", False, "middle mending ring at a half, glass clear", (), "Lifted at the kitchen"),
        ("linecloth", True, "", (), "On the cloth, at rest"),
        ("linecloth", False, "halves set; step=gust's half sent through", (), "The gust gone through"),
        ("linecloth", False, "halves set, gust's half sent through; step=halves slid in", (), "Reds at rest again"),
        ("lineclothsteps", False, "after=halves set, gust's half sent through, halves slid in; from=off", ("; ".join(PARTS),),
         "Greens start to gather; The joining ring follows; The middle loop gone; Every green gathered"),
        ("lineclothsteps", False, "after=halves set, gust's half sent through, halves slid in, green rings slid together; "
                                  "from=off", ("bells off; drawn straight",), "Ring and leads; Drawn straight")]
    PC.check_captions(DIALOGUE, "Second", calls)
    PC.check_section(DIALOGUE, "Second", calls)
    assert PC.check_anchors(DIALOGUE, "Second", PC.BLOCKS["Second"]) == (5, 9)  # next-starts, phrases before
    frames = [PC.frame_chain(c) for c in calls]
    assert [len(f) for f in frames] == [1, 1, 1, 1, 4, 2]                  # card 3e.6: 6 blocks, 10 frames
    assert tuple(R.head for f in frames for R in f) == FRAME_DUMPS          # the log's dump, frame by frame
    assert all(R.head == R.dump for f in frames for R in f)                 # the Lua's dump is the picture's
    (p21,), (p22,), (p23,), (b23a,), b23b, b24 = frames
    p24 = b24[-1]
    pics = [p21, p22, p23, p24]
    assert tuple(p.dump for p in pics) == SECOND_DUMPS                   # round 3's pictures, as last frames
    lc = lambda s: PC.linecloth(s)
    assert p23.dump == lc("halves set, gust's half sent through").dump and p24.dump == lc(", ".join(STEPS)).dump
    states = [""] + [", ".join(STEPS[:k]) for k in range(1, 7)] + [", ".join(STEPS[:3] + PARTS[:k]) for k in (1, 2, 3)]
    assert {PC.linecloth2(s).frame for s in states} == {p22.frame}         # one frame in every state, part-way too
    assert {R.frame for f in frames[1:] for R in f} == {p22.frame}
    assert {PC.stormline(s).frame for s in ("", "glass cloudy")} == {p21.frame}
    # the strips take up where the block before stops: B2.3b from the halves slid in (B2.3a), P2.4
    # from B2.3b's last frame; P2.3 from the halves set, a state not pictured (it cannot be told from
    # P2.2 at print size), whose value is P2.2's
    line = [p22, p23, b23a] + b23b + b24
    for P, R in zip(line[2:], line[3:]):
        assert PC.lc_dump2(R.before) == PC.lc_dump2(P.drawn), R.step
    assert PC.lc_dump(p23.before) == PC.lc_dump(PC.lc_state("halves set")) != PC.lc_dump(p22.drawn)
    assert same_picture(PC.graph_of(lc("halves set").dump), PC.graph_of(p22.dump))
    # B2.3b's last frame is the whole step, green rings slid together: its state and its picture
    last = b23b[-1]
    assert PC.lc_dump2(last.drawn) == PC.lc_dump(PC.lc_state(", ".join(STEPS[:4])))
    assert last.dump == lc(", ".join(STEPS[:4])).dump and last.drawn.part == 0
    assert [R.drawn.part for R in b23b] == [1, 2, 3, 0] and [R.step for R in b23b] == PARTS
    # fixed parts, frame to frame: each piece where it lay until the step that moves it (LIFE)
    for i, R in enumerate(line):
        here = {(k, c, x, y) for k, c, t, x, y in PC.parse_dump(R.dump)[0].values()}
        assert here == {p for p, when in LIFE.items() if i in when}, (i, R.step)

    # every frame of the cloth line is the bare wire, scalar 1 (card 3e.6, item 7): the half sent
    # through (C4), each half slid into a red ring (the turns adding), each green ring slid along
    # bare cord into another (green meets green, every turn at rest; a red ring at rest is plain
    # cord; a loop lifted off with the scalar the picture gives), each bell tied twice coming off,
    # a ring with two leads plain cord; a bell's place is no part of the value.  DERIVED: each
    # frame's picture rewrites (full_reduce) to the bare wire with number 1.
    for R in line:
        g = PC.graph_of(R.dump)
        assert np.allclose(matrix_of(g), I2), R.step
        assert same_picture(g, PC.graph_of(b23a.dump)), R.step
        h = g.copy()
        simp.full_reduce(h)
        assert h.num_vertices() == 2 and h.num_edges() == 1 and np.isclose(h.scalar.to_number(), 1), R.step
    # the plausible mix-ups: a bell tied twice read as tied once, or not tied at all
    for R in b23b[2:]:
        n = PC.parse_dump(R.dump)[4]["tiedtwice"]
        bare = PC.graph_of(R.dump.replace(" tiedtwice=%d" % n, "", 1))
        assert not np.allclose(matrix_of(bare), I2), R.step
    # the twice-ties part-way are \linecloth's own drawing (the middle strand in: the first bell's);
    # once the green rings are all together, round 3's own picture draws them
    drawn = [[l for l in R.own if l not in R.mark_tikz] for R in b23b]
    assert [len(d) > 0 for d in drawn] == [False, False, True, False]

    # the marks, frame by frame: every piece that left its place ghosted there (or the log says
    # why: a red ring at rest is only cord), every arc from the piece that moves to where it goes
    for R in line[1:]:
        assert R.marked and not R.refused, R.step
        extra = PC.lcpos(PC.G(lc_scene(R.after), PC.LINEOPTS, PC.KWHO))["tiehalf"] if R.step == "halves slid in" else ()
        check_cloth_marks(R, lc_scene, PC.LINEOPTS, PC.KWHO, extra)
        for n in R.notes:
            if n.startswith("no ghost of the red ring") and "hides it" not in n:
                assert n.endswith(ONLYCORD), n
    # P2.3: the gust's half slid into the tell ring (a hop); the three arcs out of it refused
    # together, every arc or none (the ties and the bells box the cord on to the mending ring)
    (hop,) = p23.arcs
    assert hop.what == "slide arc for the gust's half, along its cord into the tell ring" and p23.ghosts == []
    assert any(n.startswith("no arcs out on every other cord (3 like moves, every arc or none)") for n in p23.notes)
    # B2.3a: the halves on the ties and on the mending cord, each into its red ring; their ghosts
    assert [a.what for a in b23a.arcs] == ["arc for the half on the tie, down into its bell"] * 2 + [
        "arc for the half on the mending cord, into the middle mending ring"]
    assert sorted(q.what for q in b23a.ghosts) == ["the half on the tie"] * 2 + ["the red ring (grid 5.0,1)"]
    # B2.3b: one slide each, the green ring's ghost where it lay; a bell's ghost where it hung, no
    # arc (it follows its slack ties); P2.4: the bells' and the ring's ghosts, no arcs
    assert [len(R.arcs) for R in b23b] == [1, 1, 1, 1] and [len(R.ghosts) for R in b23b] == [1, 1, 2, 2]
    assert [a.what.split(",")[0] for R in b23b for a in R.arcs] == [
        "slide arc for the first tell ring", "slide arc for the joining ring", "slide arc for the middle tell ring",
        "slide arc for the third tell ring"]
    assert "the red bell (grid 5.0,0.5)" in [q.what for q in b23b[2].ghosts]
    assert "the red bell (grid 5.0,1.5)" in [q.what for q in b23b[3].ghosts]
    assert [R.arcs for R in b24] == [[], []] and [len(R.ghosts) for R in b24] == [2, 1]
    # the whole step slid together at once draws no arcs (its after-state crowds every arc's end:
    # the twice-tied bells' ties run into the parting ring), which is why the strip goes ring by ring
    whole = PC.run_frame("lc", "halves set, gust's half sent through, halves slid in; step=green rings slid together", "", False)
    assert whole.arcs == [] and any(n.startswith("no arcs for the green rings slid together") for n in whole.notes)
    assert whole.dump == last.dump and not PC.TEX_ERRORS

    # P2.1: the Crab's mend for the gust's pattern.  The gust turned the middle weather ring and
    # fell back (never drawn); both bells rang, s = (1, 1); the dump's mending ring at a half is
    # c(s) = (0, 1, 0); and the glass reads clear because that line is the identity, exactly.
    L = p21.state
    c = tuple(q // 2 for q in L.mend)
    assert all(q % 2 == 0 for q in L.mend) and c == CORR[(1, 1)] == (0, 1, 0)
    e, s = (0, 1, 0), (1, 1)
    assert sigma(e) == s
    assert np.allclose(matrix_of(storm_line(e, s, c)), I2)       # "glass=clear"
    assert not np.allclose(matrix_of(storm_line(e, s, (0, 0, 0))), I2)   # unmended: not clear
    assert L.glass == "clear" and L.tell == [0, 0, 0] and L.clip is None   # "Lifted at the kitchen"
    # fixed parts, P1.4 -> P2.1: only the middle mending ring's bead (by a half) and the glass
    # (lifted, clear) differ; every other ring, bead and cord the same, the dial shut in both
    before = PC.stormline("", True)
    assert before.dump == "mend=0,0,0 tell=0,0,0 clip=none glass=reset"
    vb, va = before.shown, p21.shown
    assert vb.spans == va.spans and vb.knots == va.knots and vb.bells == va.bells
    assert [(a[0], a[4], b[4]) for a, b in zip(vb.rings, va.rings) if a != b] == [("m2", 0, 2)]
    assert [a[:4] for a in vb.rings] == [b[:4] for b in va.rings]    # every ring where it was
    assert [k for k, (a, b) in enumerate(zip(va.beads, vb.beads)) if a != b] == [10]
    assert va.clip == vb.clip == () and (vb.glass, va.glass) == ("reset", "clear")

    # P2.2: the line on the cloth, ring for ring: its graph has the master's eighteen edges and
    # fifteen vertices, the same vertex signature, and equals the master exactly
    g22 = PC.graph_of(p22.dump)
    master = storm_line((0, 0, 0), (0, 0), (0, 0, 0))
    assert g22.num_edges() == 18 == master.num_edges() and g22.num_vertices() == 15 == master.num_vertices()
    assert signature(g22) == signature(master)
    assert same_picture(g22, master)
    nodes = PC.parse_dump(p22.dump)[0]                            # "On the cloth, at rest": drawn, every turn 0
    assert p22.dump.startswith("drawn |") and all(t == 0 for kind, col, t, x, y in nodes.values())
    # the bells between the strands, a little towards the kitchen end from the tell rings
    hubs = sorted((x, y) for kind, col, t, x, y in nodes.values() if kind == "hub")
    assert hubs == [(PC.LX["hub"], 0.5), (PC.LX["hub"], 1.5)] and PC.LX["hub"] > PC.LX["tell"]

    # P2.3: halves set equals the line with the gust, the bells and the mend at a half; the half
    # sent through (C4) leaves it exactly as it was
    halves = PC.graph_of(lc("halves set").dump)
    gust = storm_line((0, 1, 0), (1, 1), (0, 1, 0))
    assert same_picture(halves, gust)
    g23 = PC.graph_of(p23.dump)
    # the 18, the half on the mending cord splitting it, and a half on each tie to a bell
    assert g23.num_edges() == 21 and g23.num_vertices() == 18
    assert same_picture(g23, halves)
    assert np.allclose(matrix_of(g23), I2)                         # and it is the identity
    no_tie_halves = PC.graph_of(p23.dump.replace(" tiehalves=2", ""))
    assert not same_picture(no_tie_halves, halves)                 # the halves on the ties carry it
    assert "through" in p23.dump.split(" |")[0]                    # "The gust gone through"
    # the steps between, now pictured (round 4), each the same line (C4, C1, C6): slid in (B2.3a),
    # slid together ring by ring (B2.3b; each bell tied twice to the one ring), the bells off
    for k in (3, 4, 5):
        assert same_picture(PC.graph_of(lc(", ".join(STEPS[:k])).dump), master)
    together = PC.graph_of(last.dump)                             # one ring, each bell tied twice
    assert (together.num_vertices(), together.num_edges()) == (7, 8)
    # fixed parts: the parting and joining rings at their columns until the green rings slide
    for pic in (p22, p23):
        g = [(x, y) for kind, col, t, x, y in PC.parse_dump(pic.dump)[0].values() if col == "green" and y == 1]
        assert (PC.LX["part"], 1) in g and (PC.LX["join"], 1) in g

    # P2.4: the line drawn straight is the plain cord, and equals the line at rest exactly
    g24 = PC.graph_of(p24.dump)
    assert same_picture(g24, g22) and np.allclose(matrix_of(g24), I2)
    nodes, tubes, ties, cords, extra = PC.parse_dump(p24.dump)
    assert nodes == {} and tubes == [] and ties == [] and len(cords) == 1   # "Drawn straight"
    assert p24.dump.split(" |")[0].endswith("straight")

    # the keyed word (P2.2): bell, set by the Lua, measured clear; the kit is never keyed here
    assert p22.labels == ("bell",) and all(R.labels == () for R in [p21] + line[1:])
    PC.check_compiled_boxes(p22.picture)
    assert min(PC.check_keyed_words(p22.picture).values()) >= PC.LETTER_FLOOR
    # P2.2 is a counted frame: the eighteen cords drawn, never marked
    assert p22.step is None and not p22.marked and p22.mark_tikz == p22.arcs == p22.ghosts == []

    # the Lua's refusals: the green rings slid together while a half is still out; a step
    # out of order; a mending ring turned a quarter
    PC.TEX_ERRORS.clear()
    PC.lc_state("halves set, green rings slid together")
    PC.lc_state("halves slid in")
    PC.lc_state("green rings slid together, halves set")          # a step that must come first, late
    PC.line_state("middle mending ring a quarter round")
    assert [k for k, _ in PC.TEX_ERRORS] == ["lc-together", "lc-after", "lc-after", "line-mend-quarter"]
    assert PC.TEX_ERRORS[2][1] == "linecloth: 'halves set' comes first, before every other step"
    PC.TEX_ERRORS.clear()
    check_faults()


# Every tex.error site of ch18.lua (PC.TEX_SITES, in the Lua's file order), each tripped by a call of
# its own: (site, kind, keyed, bracket, start[, the sites raised]) -- a strip's case is (site,
# "steps", keyed, kind, start, options, steps, captions[, the sites raised]).  Two sites are reached
# only with a second fault each, named: expects=...; marks=on raises marks=on's own fault too, and an
# unknown strip has no captions to count.  The unit is round 3's: a literal call of a fault wrapper
# outside its definition; the four bare re-raisings of collected messages in the round-4 helpers
# re-raise a message some site already made, and are no sites of their own (PC.RERAISE).
DP, TP, FR, NI = "the dark point", "the turned point", "a fresh parting ring", "nine strands"
NG = "nothing gets past the bells"
FAULT_CASES = [
    # \plaintin (terr)
    ("tin-lifted", "tin", False, "first dark, first lit", ""),
    ("tin-order", "tin", False, "second dark", ""),
    ("tin-step", "tin", False, "first grey", ""),
    # \stormline (lerr)
    ("line-mend", "line", False, "middle mending ring at a third", ""),
    ("line-mend-quarter", "line", False, "middle mending ring a quarter round", ""),
    ("line-tell", "line", False, "middle tell ring at a third", ""),
    ("line-weather", "line", False, "middle weather ring at a half", ""),
    ("line-bell", "line", False, "first bell ringing", ""),
    ("line-dial", "line", False, "door dial at a half", ""),
    ("line-item", "line", False, "kettle on", ""),
    # \partcloth (perr)
    ("part-start", "part", False, "", "a cube"),
    ("part-twice", "part", False, "pushed in, pushed in", DP),
    ("part-step", "part", False, "pulled out", DP),
    # \linecloth (kerr)
    ("lc-step", "lc", False, "halves frobbed", ""),
    ("lc-twice", "lc", False, "halves set, halves set", ""),
    ("lc-together", "lc", False, "halves set, green rings slid together", ""),
    ("lc-after", "lc", False, "halves slid in", ""),
    # \tubecloth (uerr)
    ("tube-start", "tube", False, "", "a cube"),
    ("tube-nine", "tube", False, "pairs put on", NI),
    ("tube-pairs", "tube", False, "pairs put on, pairs put on", FR),
    ("tube-first", "tube", False, "drawn through", FR),
    ("tube-through", "tube", False, "pairs put on, drawn through, drawn through", FR),
    ("tube-step", "tube", False, "pulled out", FR),
    # \splitcloth (jerr)
    ("split-step", "split", False, "drawn out, half on the penny", ""),
    ("split-order", "split", False, "half on the string", ""),
    # round 4 (xerr): the bracket and the step
    ("opt-unknown", "tin", False, "step=first dark; colour=red", ""),
    ("opt-twice", "tin", False, "step=first dark; step=first dark", ""),
    ("opt-bare-two", "tin", False, "first dark; first dark; marks=off", ""),
    ("opt-onoff", "tin", False, "step=first dark; marks=maybe", ""),
    ("marks-no-step", "tin", False, "first dark; marks=on", ""),
    ("expects-keyed", "part", True, "expects=three copies", TP),
    # expects-marked fires only past another fault: without a step, marks-no-step; with one, the
    # macro's own refusal of a step in a prediction (part-expects-step)
    ("expects-marked", "part", False, "expects=three copies; marks=on", TP, ("marks-no-step", "expects-marked")),
    ("marks-keyed", "tin", True, "step=first dark", ""),
    ("step-commas", "tin", False, "step=first dark, second dark", ""),
    ("step-empty", "tin", False, "first dark; step=", ""),
    # round 4 (xerr): the six macros
    ("tin-expects", "tin", False, "expects=coin", ""),
    ("line-same", "line", False, "glass clear; step=glass clear", ""),
    ("line-glass-once", "line", False, "glass clear; step=glass cloudy", ""),
    ("line-clip-rides", "line", False, "clip-bell on; step=clip-bell on the third tell ring", ""),
    ("line-marks-counted", "line", False, "glass clear; step=glass reset; marks=on", ""),
    ("line-marks-on", "line", False, "step=glass clear; marks=on", ""),
    ("line-expects-rule", "line", False, "middle tell ring at a half, glass cloudy; expects=the bells ring", ""),
    ("line-expects-step", "line", False, "middle tell ring at a half; expects=" + NG + "; step=glass cloudy", ""),
    ("line-expects-unlifted", "line", False, "middle tell ring at a half; expects=" + NG, ""),
    ("line-expects-clear", "line", False, "glass clear; expects=" + NG, ""),
    ("line-expects-clip", "line", False, "clip-bell on, glass cloudy; expects=" + NG, ""),
    ("part-expects-rule", "part", False, "expects=three dark copies", TP),
    ("part-expects-step", "part", False, "expects=three copies; step=pushed in", TP),
    ("part-expects-bare", "part", False, "pushed in; expects=three copies", TP),
    ("part-expects-true", "part", False, "expects=three copies", DP),
    ("lc-part-turn", "lc", False, "joining ring in", ""),
    ("lc-part-begin", "lc", False, "halves set, first tell ring in", ""),
    ("lc-part-next", "lc", False, "first tell ring in, halves set", ""),
    ("lc-expects", "lc", False, "expects=two bells", ""),
    ("tube-expects", "tube", False, "expects=green", FR),
    ("split-expects", "split", False, "expects=two halves", ""),
    # round 4 (xerr): the strips
    # strip-unknown fires only past strip-captions: an unknown strip has no frames, and any
    # captions (even none: one empty item) are one too many
    ("strip-unknown", "steps", False, "nope", "", "", "x", "a", ("strip-unknown", "strip-captions")),
    ("strip-bare", "steps", False, "tin", "", "loud", "first dark", "a; b"),
    ("strip-star-from", "steps", True, "tin", "", "from=off", "first dark", "a"),
    ("strip-none", "steps", False, "tin", "", "", "", "a"),
    ("strip-commas", "steps", False, "tin", "", "", "first dark, second dark", "a; b"),
    ("strip-long", "steps", False, "split", "", "", "drawn out; half on the string; slid into the split; slid apart",
     "a; b; c; d; e"),
    ("strip-captions", "steps", False, "tin", "", "", "first dark", "a"),
]


def check_faults():
    """Every fault site of the chapter's Lua raised by its own call (the strip faults by the strip
    alone, as TeX raises them before any frame runs), each with messages of its own; the 63 sites
    are the Lua's, all of them; a refused step draws no marks, nor does a call with a fault; no
    step on the storm line is marked; the ';' bracket is order-free, each order run."""
    seen = {}
    for case in FAULT_CASES:
        site = case[0]
        if case[1] == "steps":
            _, _, keyed, kind, scene, opt, stepspec, caps = case[:8]
            want = case[8] if len(case) > 8 else (site,)
            fn, args = PC.steps, (kind, keyed, scene, opt, stepspec, caps)
        else:
            _, kind, keyed, br, state = case[:5]
            want = case[5] if len(case) > 5 else (site,)
            fn, args = PC.run_frame, (kind, br, state, keyed)
        assert tuple(dict.fromkeys(PC.faults(fn, *args))) == tuple(want), (site, PC.TEX_ERRORS)
        msgs = {m for s_, m in zip(PC.faults(fn, *args), PC.messages(fn, *args)) if s_ == site}
        assert msgs, site
        for m in msgs:
            assert m not in seen, (site, seen.get(m))                    # no two sites share a message
            seen[m] = site
    # the mutual-exclusion checks are made on the whole parse: every order of a refused bracket's
    # items raises the same faults (and a strip's options, likewise)
    orders = 0
    for case in FAULT_CASES:
        if case[1] == "steps":
            _, _, keyed, kind, scene, opt, stepspec, caps = case[:8]
            items, fn = opt.split("; "), lambda br: PC.faults(PC.steps, kind, keyed, scene, br, stepspec, caps)
        else:
            _, kind, keyed, br, state = case[:5]
            items, fn = br.split("; "), lambda br: PC.faults(PC.run_frame, kind, br, state, keyed)
        if len(items) < 2:
            continue
        want = sorted(fn("; ".join(items)))
        for order in itertools.permutations(items):
            assert sorted(fn("; ".join(order))) == want, (case[0], order)
            orders += 1
    assert orders >= 40
    # a step the Lua refuses draws no marks, in each macro's step= form; the same kind of step
    # taken draws them
    refused = 0
    for kind, br, state, site in (("tin", "step=second dark", "", "tin-order"),
                                  ("tin", "first dark; step=first lit", "", "tin-lifted"),
                                  ("part", "pushed in; step=pushed in", DP, "part-twice"),
                                  ("lc", "step=halves slid in", "", "lc-after"),
                                  ("lc", "halves set; step=green rings slid together", "", "lc-together"),
                                  ("lc", "step=joining ring in", "", "lc-part-turn"),
                                  ("lc", "halves set; step=first tell ring in", "", "lc-part-begin"),
                                  ("lc", "first tell ring in; step=halves set", "", "lc-part-next"),
                                  ("tube", "step=drawn through", FR, "tube-first"),
                                  ("tube", "pairs put on, drawn through; step=drawn through", FR, "tube-through"),
                                  ("tube", "step=pairs put on", NI, "tube-nine"),
                                  ("split", "step=half on the string", "", "split-order"),
                                  ("line", "glass clear; step=glass clear", "", "line-same"),
                                  ("line", "glass clear; step=glass cloudy", "", "line-glass-once"),
                                  ("line", "clip-bell on; step=clip-bell on the third tell ring", "", "line-clip-rides"),
                                  ("line", "step=middle weather ring at a half", "", "line-weather")):
        PC.TEX_ERRORS.clear()
        p = PC.run_frame(kind, br, state, False)
        assert [s_ for s_, _ in PC.TEX_ERRORS] == [site] and p.refused, (kind, br, PC.TEX_ERRORS)
        assert p.mark_tikz == [] and p.ghosts == [] and p.arcs == [], (kind, br)
        assert "no marks: the step was refused" in p.notes, (kind, br)
        refused += 1
    assert refused == 16
    for kind, br, state in (("tin", "step=first dark", ""), ("part", "step=pushed in", DP),
                            ("lc", "halves set; step=gust's half sent through", ""),
                            ("lc", ", ".join(STEPS[:3]) + "; step=first tell ring in", ""),
                            ("tube", "pairs put on; step=drawn through", FR),
                            ("split", "drawn out, half on the string; step=slid into the split", "")):
        PC.TEX_ERRORS.clear()
        p = PC.run_frame(kind, br, state, False)
        assert p.mark_tikz and p.arcs and p.marked and not PC.TEX_ERRORS, kind
    # the arcs' floor: an arc bowed under the Lua's MINSAG is refused, with the reason, and the same arc
    # bowed just over it fits (no arc of the dialogue or of the sweeps is ever refused for its bow, so
    # the rule is tried here on a drawn arc, the near tube's on the lead in, its bow changed)
    R = PC.run_frame("tube", "pairs put on; step=drawn through", FR, False)
    S1 = PC.tube_scene(R.after)
    g = PC.G(S1, PC.TUBEOPTS, PC.UWHO)
    O_ = PC.c_obstacles(S1, g)
    assert PC.c_fits(R.arcs[0], O_, list(R.ghosts), g)[0]
    flat = copy.copy(R.arcs[0])
    flat.sag = PC.MINSAG - 0.001
    assert PC.c_fits(flat, O_, list(R.ghosts), g) == (False, "it bows only %s from its chord, under %s" % (
        PC.f(flat.sag), PC.f(PC.MINSAG)))
    flat.sag = PC.MINSAG + 0.0005
    assert PC.c_fits(flat, O_, list(R.ghosts), g)[0]
    # a call with a fault draws no marks either
    for case in FAULT_CASES:
        if case[1] != "steps" and "step=" in case[3]:
            PC.TEX_ERRORS.clear()
            p = PC.run_frame(case[1], case[3], case[4], case[2])
            if PC.TEX_ERRORS:
                assert p.mark_tikz == [] and p.arcs == [] and p.ghosts == [], case[0]
    # no step on the storm line is marked (a bead's turn, the glass, the clip-bell): the log says
    # why, and a frame back at the line as strung is the counted frame
    for br, why in (("step=middle tell ring at a half", "no marks: the middle tell ring's bead turned from at rest to at a half"),
                    ("glass reset; step=glass clear", "no marks: the open mouth shows where the lid sat"),
                    ("glass clear; step=glass reset", "no marks: the line as strung is the wires themselves"),
                    ("step=clip-bell on", "no marks: the clip-bell comes from the Crab's claw")):
        PC.TEX_ERRORS.clear()
        p = PC.run_frame("line", br, "", False)
        assert not PC.TEX_ERRORS and not p.marked and p.mark_tikz == p.own == [], br
        assert p.notes[1].startswith(why), (br, p.notes)
    assert PC.faults(PC.run_frame, "line", "step=middle tell ring at a half; marks=on", "", False) == ["line-marks-on"]
    assert PC.faults(PC.run_frame, "line", "glass clear; step=glass reset; marks=on", "", False) == ["line-marks-counted"]
    assert [c[0] for c in FAULT_CASES] == [k for k, _ in PC.TEX_SITES] and len(PC.ERROR_SITES) == 63
    assert {k.rstrip("("): v for k, v in PC.SITE_COUNTS.items()} == {"terr": 3, "lerr": 7, "perr": 3, "kerr": 4, "uerr": 6,
                                                                     "jerr": 2, "xerr": 38}
    assert sum(PC.SITE_COUNTS.values()) == 63 == 25 + 38
    assert PC.tex_error_calls() == 11 == 7 + len(PC.RERAISE)             # seven wrappers' bodies, four re-raisings
    PC.check_sites()                                                     # the wrappers' tex.error calls
    PC.check_lua()                                                       # every rule and place, in the Lua's words
    PC.check_pins()                                                      # every chunk as transcribed, pinned
    # the bracket is order-free: every order of a frame's items draws the same frame
    for kind, state, items in (("lc", "", ["halves set", "step=gust's half sent through", "marks=off"]),
                               ("tin", "", ["first dark", "step=second dark"]),
                               ("part", DP, ["step=pushed in", "marks=on"]),
                               ("split", "", ["drawn out", "step=half on the string"]),
                               ("line", "", ["middle tell ring at a half, glass cloudy", "expects=" + NG])):
        drawn = set()
        for order in itertools.permutations(items):
            PC.TEX_ERRORS.clear()
            p = PC.run_frame(kind, "; ".join(order), state, False)
            assert not PC.TEX_ERRORS, (kind, order)
            drawn.add((p.head, tuple(p.log[1:]), tuple(p.own)))        # the log, its bracket as written aside
        assert len(drawn) == 1, kind
    PC.TEX_ERRORS.clear()


# ------------------------------------------------------------------------------ pictures

def storm_line(e, s, c, tell=(0, 0, 0)):
    """The Storm Line, built by hand as drawn (Section 18.8).  Rows: strand k on row 2k,
    bells between strands.  Every ring is drawn whatever its bit (row 75)."""
    g = zx.Graph()
    lead_in = g.add_vertex(VT.BOUNDARY, 2, 0)
    lead_out = g.add_vertex(VT.BOUNDARY, 2, 12)
    parting = g.add_vertex(VT.Z, 2, 1)                                        # Enc
    joining = g.add_vertex(VT.Z, 2, 11)                                       # Enc^dag
    weather = [g.add_vertex(VT.X, 2 * k, 3, phase=Fraction(e[k])) for k in range(3)]
    tellr = [g.add_vertex(VT.Z, 2 * k, 5, phase=Fraction(tell[k])) for k in range(3)]
    bells = [g.add_vertex(VT.X, 2 * j + 1, 6, phase=Fraction(s[j])) for j in range(2)]
    mending = [g.add_vertex(VT.X, 2 * k, 9, phase=Fraction(c[k])) for k in range(3)]
    g.add_edge((lead_in, parting), ET.SIMPLE)                                 # the lead-in
    for k in range(3):                                                        # four lengths a strand
        g.add_edge((parting, weather[k]), ET.SIMPLE)
        g.add_edge((weather[k], tellr[k]), ET.SIMPLE)
        g.add_edge((tellr[k], mending[k]), ET.SIMPLE)
        g.add_edge((mending[k], joining), ET.SIMPLE)
    for t, b in ((0, 0), (1, 0), (1, 1), (2, 1)):                             # the ties
        g.add_edge((tellr[t], bells[b]), ET.SIMPLE)
    g.add_edge((joining, lead_out), ET.SIMPLE)                                # the lead-out
    g.set_inputs((lead_in,))
    g.set_outputs((lead_out,))
    return g


def double_tied_bell(theta, phi, legs):
    """A red bell theta tied twice to one green spider phi with `legs` other legs.  PyZX's
    simple graphs keep one edge per pair, so the second tie carries a green ring at rest
    (bare cord, row 57): the tell ring not yet slid in."""
    g = zx.Graph()
    green = g.add_vertex(VT.Z, 1, 2, phase=Fraction(phi))
    bell = g.add_vertex(VT.X, 3, 2, phase=Fraction(theta))
    spacer = g.add_vertex(VT.Z, 2, 3)
    g.add_edge((green, bell))
    g.add_edge((green, spacer))
    g.add_edge((spacer, bell))
    ins, outs = [], []
    if legs >= 2:
        ins.append(g.add_vertex(VT.BOUNDARY, 0, 0))
    for k in range(legs - len(ins)):
        outs.append(g.add_vertex(VT.BOUNDARY, k, 4))
    for b in ins + outs:
        g.add_edge((b, green))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


def green_alone(phi, legs, theta=None):
    g = zx.Graph()
    green = g.add_vertex(VT.Z, 1, 2, phase=Fraction(phi))
    ins = [g.add_vertex(VT.BOUNDARY, 0, 0)] if legs >= 2 else []
    outs = [g.add_vertex(VT.BOUNDARY, k, 4) for k in range(legs - len(ins))]
    for b in ins + outs:
        g.add_edge((b, green))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    if theta is not None:                  # the number (1 + e^{i theta})/2
        g.scalar.add_node(Fraction(theta))
        g.scalar.add_power(-2)
    return g


def pair_check(s):
    """Result 18.5: a green ring at rest on each of two wires, tied to one red bell s*pi."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in (0, 1)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 4) for r in (0, 1)]
    feet = [g.add_vertex(VT.Z, r, 1) for r in (0, 1)]
    bell = g.add_vertex(VT.X, 0.5, 2, phase=Fraction(s))
    for r in (0, 1):
        g.add_edge((ins[r], feet[r]))
        g.add_edge((feet[r], outs[r]))
        g.add_edge((feet[r], bell))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


def two_bells(s):
    """The middle of the storm line: tell rings on three wires, the two bells."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, 2 * r, 0) for r in range(3)]
    outs = [g.add_vertex(VT.BOUNDARY, 2 * r, 4) for r in range(3)]
    tellr = [g.add_vertex(VT.Z, 2 * r, 1) for r in range(3)]
    bells = [g.add_vertex(VT.X, 2 * j + 1, 2, phase=Fraction(s[j])) for j in range(2)]
    for r in range(3):
        g.add_edge((ins[r], tellr[r]))
        g.add_edge((tellr[r], outs[r]))
    for t, b in ((0, 0), (1, 0), (1, 1), (2, 1)):
        g.add_edge((tellr[t], bells[b]))
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    return g


def sketchbook():
    rng = np.random.default_rng(1812)
    ZZ = np.kron(Z, Z)

    # --- Result 18.5 / row 138: a bell asks a pair, exactly, no number (the weight-2 case
    # of WhatABellMayAsk.py's T2; the derivation in this file is the master's)
    for s in (0, 1):
        m = matrix_of(pair_check(s))
        assert np.allclose(m, (np.eye(4) + (-1) ** s * ZZ) / 2)
        assert not np.allclose(m, (np.eye(4) + (-1) ** (1 - s) * ZZ) / 2)
    for s in OUTCOMES:                                                  # two bells: the P_s
        assert np.allclose(matrix_of(two_bells(s)), P(s))

    # --- The Storm Line: built as drawn, counted
    g0 = storm_line((0, 0, 0), (0, 0), (0, 0, 0))
    # the lead-in and the lead-out, twelve strand lengths (four on each strand), four ties
    assert g0.num_edges() == 18
    assert g0.num_vertices() == 15
    assert sum(1 for v in g0.vertices() if g0.type(v) == VT.BOUNDARY) == 2
    assert sum(1 for v in g0.vertices() if g0.type(v) == VT.Z) == 5
    assert sum(1 for v in g0.vertices() if g0.type(v) == VT.X) == 8
    assert all(g0.edge_type(ed) == ET.SIMPLE for ed in g0.edges())    # no Hadamard edge

    # --- Result 18.24: its value, for every one of the 2^8 labellings, exactly, scalar 1
    for e, s, c in product(BITS3, OUTCOMES, BITS3):
        g = storm_line(e, s, c)
        assert np.allclose(matrix_of(g), storm_value(e, s, c))
        # step (i) of Section 18.7, exact for every labelling: the gust's halves moved into
        # the bells and the mending rings, the weather rings left at rest
        moved = storm_line((0, 0, 0),
                           tuple(a ^ b for a, b in zip(s, sigma(e))),
                           tuple(a ^ b for a, b in zip(c, e)))
        assert same_picture(g, moved)

        # DERIVED: full_reduce takes each labelling to a bare wire, a single X, or zero
        h = g.copy()
        simp.full_reduce(h)
        f = tuple(a ^ b for a, b in zip(e, c))
        if s == sigma(e) and f == (0, 0, 0):
            assert h.num_vertices() == 2 and h.num_edges() == 1       # a bare wire...
            assert np.isclose(h.scalar.to_number(), 1)                # ...number 1
        elif s == sigma(e) and f == (1, 1, 1):
            inner = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
            assert len(inner) == 1                                    # one spider:
            v = inner[0]
            # PyZX's graph-like form of a red half: green pi between two Hadamard edges
            assert h.type(v) == VT.Z and h.phase(v) == 1
            assert [h.edge_type(ed) for ed in h.incident_edges(v)] == [ET.HADAMARD] * 2
            assert np.isclose(h.scalar.to_number(), 1)
            assert np.allclose(matrix_of(h), X)
        else:
            assert h.scalar.is_zero                                   # the branch is 0
    # the headline, as the picture: c = c(s) on E_X
    for e in SINGLE:
        for s in OUTCOMES:
            m = matrix_of(storm_line(e, s, CORR[s]))
            assert np.allclose(m, (s == sigma(e)) * I2)

    # --- Result 18.8: the double-tied bell comes off with (1 + e^{i theta})/2,
    # for a green spider with 1, 2 or 3 other legs and a random turn phi
    for legs in (1, 2, 3):
        phi = Fraction(int(rng.integers(1, 97)), 97)
        rand_theta = Fraction(int(rng.integers(1, 97)), 97)
        for theta in (Fraction(0), Fraction(1), rand_theta):
            g = double_tied_bell(theta, phi, legs)
            assert g.num_edges() == 3 + legs                         # two ties (one via the spacer)
            ref = green_alone(phi, legs, theta)
            assert same_picture(g, ref)
            expect = (1 + np.exp(1j * np.pi * float(theta))) / 2
            assert np.allclose(matrix_of(g), expect * matrix_of(green_alone(phi, legs)))
        # DERIVED by full_reduce: at rest, the green spider alone, number 1; at a half, zero
        h = double_tied_bell(0, phi, legs).copy()
        simp.full_reduce(h)
        assert same_picture(h, green_alone(phi, legs))
        assert np.isclose(h.scalar.to_number(), 1)
        h = double_tied_bell(1, phi, legs).copy()
        simp.full_reduce(h)
        assert h.scalar.is_zero
        # at a random theta, reduce both sides and compare
        h = double_tied_bell(rand_theta, phi, legs).copy()
        simp.full_reduce(h)
        r = green_alone(phi, legs, rand_theta)
        simp.full_reduce(r)
        assert same_picture(h, r)
    # the warning: a bell tied to two DIFFERENT green spiders does not come off
    for s in (0, 1):
        assert not np.allclose(matrix_of(pair_check(s)), np.eye(4))


if __name__ == "__main__":
    verdict("BlindVote", blackboard, sketchbook)
