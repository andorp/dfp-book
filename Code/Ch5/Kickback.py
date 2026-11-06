"""Challenge 3: The Mark on the Other Wire
==========================================
"The mark lands on the road I asked from, not the road I asked about."  -- Achilles

Claim (phase kickback, Cleve-Ekert-Macchiavello-Mosca 1997, quant-ph/9708016 Eq. 2.2):
with the answer wire held in the difference state |-> = (|0>-|1>)/sqrt2,
    U_f |x>|-> = (-1)^{f(x)} |x>|->,
for all four f : {0,1} -> {0,1}.  The oracle spends itself as a SIGN on the
query (the wire asked FROM); the answer wire (the wire asked ABOUT) is unchanged.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import Picture, ket, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue, words_only
from pyzx.utils import EdgeType
import props_ch05 as P

X = np.array([[0, 1], [1, 0]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# the four functions f : {0,1} -> {0,1}, as (f(0), f(1))
FOURS = {"zero": (0, 0), "one": (1, 1), "id": (0, 1), "not": (1, 0)}

k0, k1 = ket(0), ket(1)
KMINUS = (k0 - k1) / np.sqrt(2)          # normalized difference state |->
KPLUS = (k0 + k1) / np.sqrt(2)           # |+>, the WRONG answer state


def oracle(f):
    """U_f : |x>|y> -> |x>|y XOR f(x)>, top wire = query, bottom = answer."""
    U = np.zeros((4, 4), dtype=complex)
    for x in (0, 1):
        for y in (0, 1):
            U[:, 2 * x + y] = ket(x, y ^ f[x])
    return U


def blackboard():
    for name, f in FOURS.items():
        U = oracle(f)
        assert np.allclose(U @ U, np.eye(4))                 # reversible (self-inverse here)
        for x in (0, 1):
            sign = (-1) ** f[x]
            out = U @ kron(ket(x), KMINUS)
            # the sign is kicked onto the query road; the pair is otherwise unchanged
            assert np.allclose(out, sign * kron(ket(x), KMINUS))
            # the answer register returns to |-> exactly (it is the eigenstate)
            assert np.allclose(out.reshape(2, 2)[x], sign * KMINUS)
            # NOT up to scalar: the wrong sign fails whenever f(x) = 1
            if f[x] == 1:
                assert not np.allclose(out, kron(ket(x), KMINUS))

        # Achilles' first expectation: a DEFINITE answer |0> makes the mark land
        # on the answer wire (|x>|f(x)>), leaving the query untouched -- no kickback.
        for x in (0, 1):
            assert np.allclose(U @ kron(ket(x), k0), kron(ket(x), ket(f[x])))
        # and |+> is the eigenstate with eigenvalue +1 always: no mark at all.
        for x in (0, 1):
            assert np.allclose(U @ kron(ket(x), KPLUS), kron(ket(x), KPLUS))

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the conduit-board with
#   \conduitboard[view]{setting}{caption}     (pdf/props/ch05.tex)
# and tools/props_ch05.py transcribes what the Lua computes: the road's token
# and the reed's token, short of the bar (before) and just past it (past).
# The calls are read out of this challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3): the willow's head spreader is
# lifted, so one road is walked by hand -- the query is the definite |x>.  A
# road token's face is its sign, silver = +1, slate = -1.  The reed's FLAT
# token has one cup, so its face is the reed's VALUE, silver = |0>, slate =
# |1>; the TWO-FACED token is |->.  The bar is U_f, its rule read as f.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("conduitboard", "roadtiles"))
NOT_IN_CAPTIONS = ("wire", "wires", "diagram", "diagrams", "drawing", "drawings", "picture",
                   "pictures", "box", "boxes", "dot", "dots", "node", "nodes", "lanes",
                   "strings", "five")
REED_VALUE = {"silver": 0, "slate": 1}                 # a flat reed token's face is its value
ONE_LANE_RULES = ("never", "always", "willow high", "willow low", "odd", "even")


def f_of(rule):
    """A rule, as the dialogue would write it after `bar`, on the willow alone,
    as (f(0), f(1)) -- read through the transcribed Lua, never retyped."""
    return f_of_bar(P.conduitboard("set", f"willow; reed two-faced; bar {rule}"))


def check_transcription():
    """The hand-walked road under the bar, against U_f, for every one-lane rule."""
    assert {f_of(r) for r in ONE_LANE_RULES} == set(FOURS.values())   # all four f met
    assert f_of("odd") == f_of("willow high") and f_of("even") == f_of("willow low")
    for rule in ONE_LANE_RULES:
        U = oracle(f_of(rule))
        for x, road in ((0, "low"), (1, "high")):
            q = ket(x)
            # the reed two-faced: past the bar the road's token is (-1)^{f(x)}
            b = P.conduitboard("past", f"willow no head {road}; reed two-faced; bar {rule}")
            ((v, fc, _),) = b.walking
            assert v == x and b.reed == "two-faced"
            assert np.allclose(U @ kron(q, KMINUS), P.SIGN[fc] * kron(q, KMINUS))
            # the reed flat: the road's token stays silver, the reed's face turns
            for laid in ("silver", "slate"):
                spec = f"willow no head {road}; reed {laid}; bar {rule}"
                before, past = P.conduitboard("before", spec), P.conduitboard("past", spec)
                assert before.walking == ((x, "silver", False),) and before.reed == laid
                assert past.walking == ((x, "silver", False),)
                assert np.allclose(U @ kron(q, ket(REED_VALUE[laid])),
                                   kron(q, ket(REED_VALUE[past.reed])))
    # the macro's own rules
    assert P.faults(P.conduitboard, "past", "willow no head high; bar willow high") != []  # no reed token
    assert P.faults(P.conduitboard, "past", "willow high; reed silver; bar never") != []   # hand, head on
    assert P.faults(P.conduitboard, "past", "willow no head; reed silver; bar never") != []  # which road?
    assert P.faults(P.conduitboard, "past", "willow; reed silver; bar willow high") != []  # two reed faces
    assert P.faults(P.conduitboard, "set", "willow; reed two-faced; bar alder high") != []  # alder idle
    assert P.faults(P.conduitboard, "set", "willow; reed gold") != []


def pictures():
    check_transcription()
    p31, p32, p33, p34 = DIALOGUE.check_layout("Third", sizes=[2, 2])
    for c in (p31, p32, p33, p34):
        assert c.macro == "conduitboard" and not c.keyed
        assert words_only(c.caption, forbidden=NOT_IN_CAPTIONS), c.caption
    b31, b32, b33, b34 = (P.picture(c) for c in (p31, p32, p33, p34))
    assert p31.args == p32.args and p33.args == p34.args                # two moments of one walk
    for b in (b31, b32, b33, b34):
        assert b.grooves == P.GROOVES and b.live == ("willow",)
        assert b.spreaders == {"willow": (False, True)}                 # head lifted, by hand
        assert b.setting.lanes["willow"].hand == 1                      # the high road
        assert b.bar.kind == "lane" and not b.bar.hidden                # told aloud (l.231)
        assert f_of_bar(b) == FOURS["id"]                               # the rule is `willow high`
        assert b.pegs == () and b.heaps == {}                           # no peg; no end, no close-up
    assert [b.view for b in (b31, b32, b33, b34)] == ["before", "past", "before", "past"]

    a1, a2 = P.anchors(DIALOGUE, "Third")
    assert "silver up and lying flat" in a1 and "willow's high road" in a1
    assert "stands it up on its edge" in a2 and "just short of the bar" in a2

    U = oracle(FOURS["id"])
    q = ket(1)
    # P3.1, P3.2: the flat reed token, silver -> slate; the willow's token silver throughout
    assert b31.reed == "silver" and b31.walking == ((1, "silver", False),)
    assert b32.reed == "slate" and b32.walking == ((1, "silver", False),)
    assert np.allclose(U @ kron(q, ket(REED_VALUE[b31.reed])), kron(q, ket(REED_VALUE[b32.reed])))
    assert np.allclose(U @ kron(q, k0), kron(q, k1))                    # U_f|1>|0> = |1>|1>

    # P3.3, P3.4: the reed two-faced and unchanged; the willow's token silver -> slate
    assert b33.reed == b34.reed == "two-faced"
    assert b33.walking == ((1, "silver", False),) and b34.walking == ((1, "slate", False),)
    sign = P.SIGN[b34.walking[0][1]]
    assert sign == -1
    assert np.allclose(U @ kron(q, KMINUS), sign * kron(q, KMINUS))    # U_f|1>|-> = -|1>|->
    # Achilles's expectation -- the mark on the reed -- is what the flat token gave
    assert b32.reed != b31.reed and b34.reed == b33.reed


def f_of_bar(board):
    fn = P.rule_function(board.bar, board.live)
    return (fn((0,)), fn((1,)))


def hedge(p, a, b):
    p.g.add_edge((a, b), EdgeType.HADAMARD)


def query_state(p, bit, col=0):
    v = p.x(0, col, bit)        # |0> = X(0)/sqrt2, |1> = X(pi)/sqrt2
    return v, -1                # returns node and its scalar power


def minus_state(p, col=0):
    v = p.z(1, col, 1)          # |-> = Z(pi)/sqrt2
    return v, -1


def insert_oracle(p, f, q_in, a_in, col):
    """Wire U_f between the query node q_in and answer node a_in; return the
    query-side node, the answer-side node, and the extra scalar power."""
    # PyZX draws this with the spiders you meet in Chapter 8.
    power = 0
    if f == (0, 0):                       # f = 0: identity on both wires
        return q_in, a_in, power
    if f == (1, 1):                       # f = 1: flip the answer always
        xn = p.x(1, col, 1)
        p.wire(a_in, xn)
        return q_in, xn, power
    # a control/target pair: flip the answer where the query reads 1
    zc = p.z(0, col, 0)
    xt = p.x(1, col, 0)
    p.wire(q_in, zc)
    p.wire(a_in, xt)
    p.g.add_edge((zc, xt))
    power += 1                            # the pair carries a factor sqrt2
    if f == (1, 0):                       # f = 1 XOR x: flip the answer once more
        xn = p.x(1, col + 1, 1)
        p.wire(xt, xn)
        xt = xn
    return zc, xt, power


def kicked(f, x, signed):
    """The picture U_f |x>|->, or (when signed) the target (-1)^{f(x)}|x>|->."""
    p = Picture()
    qs, pq = query_state(p, x, 0)
    ms, pm = minus_state(p, 0)
    power = pq + pm
    q_out, a_out = qs, ms
    if not signed:
        q_out, a_out, dp = insert_oracle(p, f, qs, ms, 3)
        power += dp
    p.wire(q_out, p.out(0))
    p.wire(a_out, p.out(1))
    g = p.done()
    g.scalar.add_power(power)
    if signed and (-1) ** f[x] == -1:
        g.scalar.add_phase(Fraction(1))          # the kicked -1, inside the picture
    return g


def sketchbook():
    for name, f in FOURS.items():
        for x in (0, 1):
            lhs = kicked(f, x, signed=False)      # the oracle on |x>|->
            rhs = kicked(f, x, signed=True)       # (-1)^{f(x)} |x>|->
            assert same_picture(lhs, rhs)         # exact, the sign included
            # the sign is NOT up-to-scalar: when f(x) = 1 the UNSIGNED target
            # +|x>|-> (kicked with f = 0) fails to match.
            if f[x] == 1:
                plain = kicked((0, 0), x, signed=True)
                assert not same_picture(lhs, plain)


if __name__ == "__main__":
    verdict("Kickback", blackboard, sketchbook)
