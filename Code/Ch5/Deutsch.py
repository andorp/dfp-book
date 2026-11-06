"""Challenge 4: The One Question
================================
"I never learned either value, only whether they agree."  -- Achilles

Claim (Deutsch's algorithm).  For f : {0,1} -> {0,1} given as the oracle
U_f : |x>|y> -> |x>|y XOR f(x)>,
    (H (x) I) U_f (H (x) I) |0>|-> = (-1)^{f(0)} |f(0) XOR f(1)> |->,
exactly, GLOBAL SIGN included.  One oracle call decides constant (f(0)=f(1))
versus balanced (f(0)!=f(1)) with certainty, where any classical strategy needs
two evaluations.

Attribution (a trap): this deterministic one-query algorithm is due to Cleve,
Ekert, Macchiavello & Mosca (1997, quant-ph/9708016), with Alain Tapp
independent (1997, private communication).  Deutsch's 1985 original
(doi:10.1098/rspa.1985.0070) solved the same problem only probabilistically.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import I2, Picture, ket, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue, words_only
from pyzx.utils import EdgeType
import props_ch05 as P

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
FOURS = {"zero": (0, 0), "one": (1, 1), "id": (0, 1), "not": (1, 0)}

k0, k1 = ket(0), ket(1)
KMINUS = (k0 - k1) / np.sqrt(2)


def oracle(f):
    U = np.zeros((4, 4), dtype=complex)
    for x in (0, 1):
        for y in (0, 1):
            U[:, 2 * x + y] = ket(x, y ^ f[x])
    return U


def blackboard():
    for name, f in FOURS.items():
        U = oracle(f)
        circuit = kron(H, I2) @ U @ kron(H, I2)
        out = circuit @ kron(ket(0), KMINUS)
        a = f[0] ^ f[1]                                  # 0 constant, 1 balanced
        sign = (-1) ** f[0]                              # the global phase
        # exact, global sign included -- Stage A demands it
        assert np.allclose(out, sign * kron(ket(a), KMINUS))
        # the answer register is still |-> and the query reads f(0) XOR f(1)
        assert np.allclose(out.reshape(2, 2)[a], sign * KMINUS)
        # dropping the global sign is a falsehood whenever f(0) = 1
        if f[0] == 1:
            assert not np.allclose(out, kron(ket(a), KMINUS))
        # the query is |0> iff constant, |1> iff balanced -- a certain verdict
        query_reads = a
        assert query_reads == (0 if f[0] == f[1] else 1)

    # Why one classical evaluation cannot do it: group the four f by f(0).  Each
    # value of f(0) is shared by one constant and one balanced function, so no
    # single evaluation of f fixes the class.  (Not a complexity claim -- just
    # the finite table.)
    for v in (0, 1):
        share = [f for f in FOURS.values() if f[0] == v]
        classes = {("constant" if f[0] == f[1] else "balanced") for f in share}
        assert classes == {"constant", "balanced"}

    # The quantum "peek" is no help either: after H then U_f with NO second H,
    # reading the query is a fair coin for every f -- it holds no bit of f.
    for f in FOURS.values():
        state = oracle(f) @ kron(H, I2) @ kron(ket(0), KMINUS)
        amp0 = state.reshape(2, 2)[0]
        assert np.isclose(np.vdot(amp0, amp0).real, 0.5)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws close-ups of the willow's home pocket with
#   \conduitboard[view]{setting}{caption}     (pdf/props/ch05.tex)
# and tools/props_ch05.py transcribes what the Lua computes: the roads, the
# kick, the deal into the pockets and the settling.  The calls are read out of
# this challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3): silver = +1, slate = -1; home =
# <0|, away = <1| on the willow; the two-faced reed token is |-> and is closed
# by <-|; each spreader is H; the bar is U_f, its rule read as f.  A settled
# pocket's net is the amplitude times sqrt2 per spreader walked through: both
# spreaders, net = 2 x amplitude; the head lifted and the road walked by hand,
# one token = sqrt2 x amplitude.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("conduitboard", "roadtiles"))
NOT_IN_CAPTIONS = ("wire", "wires", "diagram", "diagrams", "drawing", "drawings", "picture",
                   "pictures", "box", "boxes", "dot", "dots", "node", "nodes", "lanes",
                   "strings", "five")
ONE_LANE_RULES = ("never", "always", "willow high", "willow low", "odd", "even")
FORBIDDEN_WITH_BOTH = {"past", "end", "home", "away", "pockets", "away settled",
                       "pockets settled"}          # would show the low road's answer
BRA = {y: kron(ket(y), KMINUS).conj().ravel() for y in (0, 1)}   # <y|<-|


def f_of(board):
    """The bar's rule as (f(0), f(1)), read through the transcribed Lua."""
    fn = P.rule_function(board.setting.bar, board.live)
    return (fn((0,)), fn((1,)))


def check_transcription():
    for rule in ONE_LANE_RULES:
        spec = f"willow; reed two-faced; bar hidden {rule}"
        f = f_of(P.conduitboard("set", spec))
        U = oracle(f)
        # both spreaders: each pocket's net is 2 x (<y|<-|)(H(x)I) U_f (H(x)I)|0>|->
        out = kron(H, I2) @ U @ kron(H, I2) @ kron(ket(0), KMINUS)
        want = [2 * (BRA[y] @ out).item() for y in (0, 1)]
        e = P.conduitboard("end", spec).heaps["willow"]
        got = [0 if e[p] is None else P.SIGN[e[p][0]] * e[p][1] for p in ("home", "away")]
        assert np.allclose(got, want)
        c = P.conduitboard("pockets", spec)
        assert np.allclose([P.net(c.cups["home"]), P.net(c.cups["away"])], want)
        # the head lifted, road x by hand: one token = sqrt2 x (<y|<-|)(H(x)I) U_f |x>|->
        for x, road in ((0, "low"), (1, "high")):
            c = P.conduitboard("pockets", f"willow no head {road}; reed two-faced; bar {rule}")
            one = kron(H, I2) @ U @ kron(ket(x), KMINUS)
            for p, y in (("home", 0), ("away", 1)):
                assert len(c.cups[p]) == 1
                assert np.isclose(P.SIGN[c.cups[p][0]], np.sqrt(2) * (BRA[y] @ one).item())
    # the macro's own rules: with the foot spreader lifted the arrival is a toss
    assert P.faults(P.conduitboard, "home", "willow no foot; reed two-faced; bar never") != []
    assert P.faults(P.conduitboard, "end", "willow no foot; reed two-faced; bar never") != []
    assert P.faults(P.conduitboard, "home settled", "willow; reed two-faced; bar willow") != []


def pictures():
    check_transcription()
    p41, p42 = DIALOGUE.check_layout("Fourth", sizes=[1, 1])
    for c in (p41, p42):
        assert c.macro == "conduitboard"
        assert words_only(c.caption, forbidden=NOT_IN_CAPTIONS), c.caption
    b41, b42 = P.picture(p41), P.picture(p42)
    for b in (b41, b42):
        assert isinstance(b, P.Closeup) and b.which == "home"      # the away cup never pictured
        assert b.live == ("willow",) and b.setting.reed == "two-faced"
        assert b.setting.bar.hidden                                # her hand over the bar
        both = b.setting.lanes["willow"].head and b.setting.lanes["willow"].foot
        assert not (both and b.view in FORBIDDEN_WITH_BOTH), b.view

    a1, a2 = P.anchors(DIALOGUE, "Fourth")
    assert "willow's low road" in a1 and a1.endswith("reads the face it brings home.]*")
    assert a2.startswith("*[He sets the foot spreader back.") and "leaving it bare" in a2
    between = DIALOGUE.section("Fourth").split(p41.text)[1].split(p42.text)[0]
    assert "He lifts the foot spreader clear" in between         # the toss: no picture of it

    # the rules of the card: the plod's is `never`, the last walk's `willow low`;
    # different from each other and from the Third Challenge's
    third = {f_of(P.picture(c)) for b in DIALOGUE.blocks("Third") for c in b}
    assert f_of(b41) == FOURS["zero"] and f_of(b42) == FOURS["not"]
    assert third == {FOURS["id"]} and f_of(b41) != f_of(b42)
    assert not ({f_of(b41), f_of(b42)} & third)

    # P4.1: the plod -- the head lifted, the low road by hand, keyed "home"; one silver
    assert p41.keyed and b41.labels == ("home",) and not b41.settled
    assert b41.setting.lanes["willow"].head is False and b41.setting.lanes["willow"].hand == 0
    assert b41.cups == {"home": ["silver"]}
    one = kron(H, I2) @ oracle(f_of(b41)) @ kron(ket(0), KMINUS)
    assert np.isclose(np.sqrt(2) * (BRA[0] @ one).item(), P.SIGN[b41.cups["home"][0]])  # +1

    # P4.2: the one walk, both spreaders, settled -- the home pocket bare, net 0
    assert not p42.keyed and b42.settled and b42.view == "home settled"
    assert b42.setting.lanes["willow"].head and b42.setting.lanes["willow"].foot
    assert b42.heaps == {"home": []} and P.net(b42.cups["home"]) == 0
    f = f_of(b42)
    out = kron(H, I2) @ oracle(f) @ kron(H, I2) @ kron(ket(0), KMINUS)
    assert np.isclose(2 * (BRA[0] @ out).item(), P.net(b42.cups["home"]))
    assert np.allclose(out, -kron(ket(1), KMINUS))                 # Deutsch for `not`: -|1>|->
    # the plod's one answer was a full cup; the one question's, a bare one
    assert P.net(b41.cups["home"]) != 0


def hedge(p, a, b):
    p.g.add_edge((a, b), EdgeType.HADAMARD)


def circuit_out(f):
    """The whole circuit as an output state: |0> and |-> at the head, an H on the
    query before and after the oracle, the answer wire held in |->."""
    p = Picture()
    qs = p.x(0, 0, 0)                 # |0> = X(0)/sqrt2
    qmid = p.z(0, 3, 0)              # PyZX draws this with the spiders you meet in Chapter 8.
    qo = p.out(0)
    hedge(p, qs, qmid)               # first H on the query
    hedge(p, qmid, qo)             # second H on the query
    a_st = p.z(1, 0, 1)             # |-> = Z(pi)/sqrt2
    ao = p.out(1)
    power = -2                       # |0> and |->
    if f == (0, 0):                 # f = 0: nothing on the answer
        p.wire(a_st, ao)
    elif f == (1, 1):               # f = 1: flip the answer always
        xn = p.x(1, 3, 1)
        p.wire(a_st, xn, ao)
    else:                           # flip the answer where the query reads 1
        xt = p.x(1, 3, 0)
        p.g.add_edge((qmid, xt))
        power += 1                   # the control/target pair carries a factor sqrt2
        if f == (1, 0):             # f = 1 XOR x: one further flip
            xn = p.x(1, 4, 1)
            p.wire(a_st, xt, xn, ao)
        else:
            p.wire(a_st, xt, ao)
    g = p.done()
    g.scalar.add_power(power)
    return g


def target(bit, sign):
    """(-1)^sign_bit |bit> |->, drawn as one picture."""
    p = Picture()
    qv = p.x(0, 0, bit)             # |bit> = X-node/sqrt2
    mv = p.z(1, 0, 1)              # |-> = Z(pi)/sqrt2
    p.wire(qv, p.out(0))
    p.wire(mv, p.out(1))
    g = p.done()
    g.scalar.add_power(-2)
    if sign < 0:
        g.scalar.add_phase(Fraction(1))
    return g


def sketchbook():
    for name, f in FOURS.items():
        a = f[0] ^ f[1]
        sign = (-1) ** f[0]
        lhs = circuit_out(f)
        assert same_picture(lhs, target(a, sign))          # exact, global sign included
        # the certain verdict: constant reads |0>, balanced reads |1>
        assert same_picture(lhs, target(0 if f[0] == f[1] else 1, sign))
        # dropping the global sign fails whenever f(0) = 1
        if f[0] == 1:
            assert not same_picture(lhs, target(a, +1))


if __name__ == "__main__":
    verdict("Deutsch", blackboard, sketchbook)
