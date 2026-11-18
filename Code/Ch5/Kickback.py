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
#   \conduitboard[view; opts]{setting}{caption}     (pdf/props/ch05.tex)
# and tools/props_ch05.py transcribes what the Lua computes: the road's token
# and the reed's token, short of the bar (before) and just past it (past); a
# step drawn on the setting before it (a spreader lifted, the reed token
# wedged), with its piece's ghost; the marks of the walk past the bar; and the
# walk Achilles expects.  The calls are read out of this challenge's section
# of dialog.md, never copied.
#
# The state model (identity card, section 3): the willow's head spreader is
# lifted, so one road is walked by hand -- the query is the definite |x>.  A
# road token's face is its sign, silver = +1, slate = -1.  The reed's FLAT
# token has one cup, so its face is the reed's VALUE, silver = |0>, slate =
# |1>; the TWO-FACED token is |->.  The bar is U_f, its rule read as f.  The
# moments of the challenge, in order:
#   F13  the head lifted: H|0> (two roads) becomes the hand-walked |1>
#   F14  U_f |1>|0> = |1>|1>: the reed turned, the willow's sign kept
#   F15  the reed lifted from |1> (slate) and wedged: |->
#   F16  as Achilles expects (rule `reed`): the turn spent on the reed, sign +1
#   F17  U_f |1>|-> = -|1>|->: the sign kicked back onto the willow
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("conduitboard", "conduitwalk", "roadtiles"))
NOT_IN_CAPTIONS = ("wire", "wires", "diagram", "diagrams", "drawing", "drawings", "picture",
                   "pictures", "box", "boxes", "dot", "dots", "node", "nodes", "lanes",
                   "strings", "five")
REED_VALUE = {"silver": 0, "slate": 1}                 # a flat reed token's face is its value
ONE_LANE_RULES = ("never", "always", "willow high", "willow low", "odd", "even")
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def f_of(rule):
    """A rule, as the dialogue would write it after `bar`, on the willow alone,
    as (f(0), f(1)) -- read through the transcribed Lua, never retyped."""
    return f_of_bar(P.conduitboard("set", f"willow; reed two-faced; bar {rule}"))


def f_of_bar(board):
    fn = P.rule_function(board.bar, board.live)
    return (fn((0,)), fn((1,)))


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
            assert b.kicked == (U[0, 0] != 1 if x == 0 else bool(f_of(rule)[1]))
            # the rule `reed` (no kick): the road's token stays silver, the reed as it was
            spec = f"willow no head {road}; reed two-faced; bar {rule}"
            e, errs = P.run(P.conduitboard, "past; expects=reed", spec)
            if f_of(rule)[x]:
                assert errs == [] and e.walking == ((x, "silver", False),)
            else:
                assert errs == ["conduitboard: expects=reed: no road's token is kicked; the "
                                "rule agrees with the board"]
            # the reed flat: the road's token stays silver, the reed's face turns
            for laid in ("silver", "slate"):
                spec = f"willow no head {road}; reed {laid}; bar {rule}"
                before, past = P.conduitboard("before", spec), P.conduitboard("past", spec)
                assert before.walking == ((x, "silver", False),) and before.reed == laid
                assert past.walking == ((x, "silver", False),)
                assert np.allclose(U @ kron(q, ket(REED_VALUE[laid])),
                                   kron(q, ket(REED_VALUE[past.reed])))
                # the wedge: the flat token, whatever its face, stood on its edge: |->
                w = P.clean(P.conduitboard, "before; wedge=reed", spec)
                assert w.reed == "two-faced" and w.moved == P.Moved("wedge", face=laid)
    # a head spreader lifted names the one road then walked by hand
    for road, x in (("high", 1), ("low", 0)):
        b = P.clean(P.conduitboard, f"before; lift=willow head {road}", "willow; reed silver")
        assert b.setting.lanes["willow"].hand == x and not b.setting.lanes["willow"].head
        assert b.walking == ((x, "silver", False),)
        assert [(g["piece"], g["which"]) for g in b.ghosts] == [("spreader", "head")]
    # the macro's own rules
    assert P.faults(P.conduitboard, "past", "willow no head high; bar willow high") != []  # no reed token
    assert P.faults(P.conduitboard, "past", "willow high; reed silver; bar never") != []   # hand, head on
    assert P.faults(P.conduitboard, "past", "willow no head; reed silver; bar never") != []  # which road?
    assert P.faults(P.conduitboard, "past", "willow; reed silver; bar willow high") != []  # two reed faces
    assert P.faults(P.conduitboard, "set", "willow; reed two-faced; bar alder high") != []  # alder idle
    assert P.faults(P.conduitboard, "set", "willow; reed gold") != []
    assert P.faults(P.conduitboard, "before; wedge=reed", "willow; reed two-faced") == [
        "conduitboard: wedge=reed needs the reed's token lying flat in its cup, silver or slate"]
    assert P.faults(P.conduitboard, "before; lift=willow head", "willow; reed silver") == [
        "conduitboard: lift=willow head high|low: name the one road then walked by hand"]


def pictures():
    P.check_lua()                                  # the Lua and .tex the transcription was proved on
    check_transcription()
    found = P.check_section(DIALOGUE, "Third", sizes=[2, 3], frames=[2, 3],
                            forbidden=NOT_IN_CAPTIONS)
    (p31, [(b13, _)]), (p32, [(b14, _)]) = found[0]
    (p33, [(b15, _)]), (p34, [(b16, _)]), (p35, [(b17, _)]) = found[1]
    calls = (p31, p32, p33, p34, p35)
    boards = (b13, b14, b15, b16, b17)
    assert [P.options(c) for c in calls] == [
        {"view": "before", "lift": "willow head high"}, {"view": "past", "marks": "on"},
        {"view": "before", "wedge": "reed"}, {"view": "past", "expects": "reed"},
        {"view": "past", "marks": "on"}]
    for c, b in zip(calls, boards):
        assert c.macro == "conduitboard" and not c.keyed
        assert b.grooves == P.GROOVES and b.live == ("willow",)
        assert b.spreaders == {"willow": (False, True)}                 # head lifted, by hand
        assert b.setting.lanes["willow"].hand == 1                      # the high road
        assert b.bar.kind == "lane" and not b.bar.hidden                # told aloud
        assert f_of_bar(b) == FOURS["id"]                               # the rule is `willow high`
        assert b.pegs == () and b.heaps == {} and b.cups == {}          # no peg; no end, no deal
        assert b.arcs == [] or b is b15                                 # no arc but the wedge's
        ((v, _, _),) = b.walking
        (t,) = b.tokens
        assert v == 1 and t["x"] == (P.XBEFORE if b.view == "before" else P.XPAST)

    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a1, a2 = DIALOGUE.anchors("Third", path)
        assert "silver up and lying flat" in a1 and "willow's high road" in a1
        assert "stands it up on its edge" in a2 and "just short of the bar" in a2
        assert "carries it over the bar" in a2
        # the clearings before each re-lay of the willow's pockets (D3, D4)
        assert a1.startswith("*[She tips the willow's pockets empty, and lifts the head spreader")
        assert a2.startswith("*[She tips the willow's pockets empty again, lifts the reed's token")
    a1, a2 = DIALOGUE.anchors("Third")

    U = oracle(FOURS["id"])
    q = ket(1)
    # F13: the bar laid, the head spreader lifted off -- the setting given is the one
    # before the step (two roads); drawn after it, one road by hand, the spreader's ghost
    before13 = P.clean(P.conduitboard, "before", p31.args[0])
    assert before13.setting.lanes["willow"].head and len(before13.walking) == 2
    assert np.allclose(np.sqrt(2) * (H @ ket(0)).ravel(),
                       [P.SIGN[f] for _, f, _ in before13.walking])     # H|0>: two roads
    assert b13.moved == P.Moved("lift", lane="willow", piece="head")
    assert [(g["piece"], g["lane"], g["which"], g["x"], g["y"]) for g in b13.ghosts] == \
        [("spreader", "willow", "head", P.XHEAD, P.ROWY["willow"])]     # where it stood
    assert b13.walking == ((1, "silver", False),)                       # |1>, by hand
    assert b13.reed == "silver" and not b13.flips and not b13.halos and not b13.pins
    assert P.serialize(b13.setting) == P.serialize(P.parse_setting(p32.args[0]))   # = F14's

    # F14: past the bar, the flat reed token turned: U_f|1>|0> = |1>|1>; the willow's
    # token runs on unmarked (its ghost at the head, no flip on it)
    assert b14.reed == "slate" and b14.walking == ((1, "silver", False),)
    assert np.allclose(U @ kron(q, ket(REED_VALUE[b13.reed])), kron(q, ket(REED_VALUE[b14.reed])))
    assert np.allclose(U @ kron(q, k0), kron(q, k1))
    assert [(g["place"], g["face"], g["x"], g["y"]) for g in b14.ghosts] == \
        [("before", t["face"], t["x"], t["y"]) for t in b13.tokens]
    assert b14.halos == [("reed", b13.reed, P.XHOME, P.ROWY["reed"], P.TOKR + 0.028)]
    assert [f["on"] for f in b14.flips] == ["reed"]                     # only the reed turned
    assert b14.pins == [] and not b14.kicked
    (fl,) = b14.flips
    assert P.curve_gap_circle(fl["pts"], P.XHOME, P.ROWY["rowan"], P.CUP_EXTENT) > 0

    # F15: the reed token as it lay before the step (slate: F14 turned it), lifted from
    # its cup, carried over the bar and stood on its edge just short of it: |->
    assert P.parse_setting(p33.args[0]).reed == b14.reed == "slate"
    assert b15.moved == P.Moved("wedge", face="slate") and b15.reed == "two-faced"
    assert [(g["piece"], g["face"], g["place"], g["x"], g["y"]) for g in b15.ghosts] == \
        [("reed", "slate", "cup", P.XHOME, P.ROWY["reed"])]             # its ghost in the cup
    assert b15.walking == ((1, "silver", False),) and not b15.flips and not b15.pins
    ((kind, to, arc),) = b15.arcs
    assert (kind, to) == ("lift", "wedge")
    (x0, _), _, _, (x1, _) = arc
    assert x1 < P.XBAR0 < P.XBAR1 < x0 and np.hypot(x0 - P.XHOME, 0) < P.CUPR   # cup -> wedge
    xs = [x for x, _ in P.bezier(arc)]
    assert min(xs) < P.XBAR0 and max(xs) > P.XBAR1                    # over the bar it crosses
    assert P.curve_gap_rect(arc, P.X0, P.ROWY["rowan"] - P.BRASS_H,
                            P.X1, P.ROWY["rowan"] + P.BRASS_H) > 0      # under the rowan's brass
    assert P.curve_gap_circle(arc, P.XHOME, P.ROWY["rowan"], P.CUP_EXTENT) > 0
    # "just short of the bar": the wedged token, drawn, clear of the bar on the head side
    assert P.XHEAD < P.XWEDGE and P.XWEDGE + P.TWOFACED_RX < P.XBAR0
    # "a little way past the lanes' heads": the bar clear of the head spreader
    assert P.XHEAD + P.SPREADER_EXTENT < P.XBAR0

    # F16: as Achilles expects (rule `reed`, card l.331): the bar's turn spent on the reed,
    # nothing comes back -- the willow token past the bar silver, sign +1
    assert b16.expects == "reed" and b16.setting.nokick and not b16.marked
    assert P.serialize(b16.setting) == P.serialize(b17.setting)          # the same walk
    assert b16.walking == ((1, "silver", False),) and b16.reed == "two-faced"
    expected = +1 * kron(q, KMINUS)                                     # the rule: no sign
    # F17: the kick -- U_f|1>|-> = -|1>|->; the reed token held still (the pin), the
    # willow token turned (its ghost at the head, silver; a flip); no arc for the kick
    assert b17.walking == ((1, "slate", False),) and b17.reed == "two-faced" and b17.kicked
    sign = P.SIGN[b17.walking[0][1]]
    assert sign == -1
    truth = U @ kron(q, KMINUS)
    assert np.allclose(truth, sign * kron(q, KMINUS))                  # U_f|1>|-> = -|1>|->
    assert np.allclose(expected, P.SIGN[b16.walking[0][1]] * kron(q, KMINUS))
    assert not np.allclose(expected, truth)                             # refuted
    assert [(g["place"], g["face"], g["x"]) for g in b17.ghosts] == \
        [("before", t["face"], t["x"]) for t in b15.tokens]             # where it set out
    assert [f["on"] for f in b17.flips] == [("willow", "high")]
    assert b17.pins == [("reed", P.XWEDGE, P.ROWY["reed"], 0.06 * 1.3 + 0.04, 0.118 * 1.3 + 0.04)]
    assert b17.halos == [] and b17.arcs == []                           # the reed unmoved; no arc
    assert not [g for g in b17.ghosts if g["piece"] == "reed"]          # "exactly as it was wedged"
    (fl,) = b17.flips
    assert max(y for _, y in P.bezier(fl["pts"])) + P.SWING_W / 2 < P.TOP
    assert P.curve_gap_circle(fl["pts"], P.XFOOT, 0, P.SPREADER_EXTENT) > 0
    # Achilles's expectation -- the mark on the reed -- is what the flat token gave
    assert b14.reed != b13.reed and b17.reed == b15.reed

    # the directions and captions edited this round, in both files
    P.check_edits(DIALOGUE, "Third", [
        ("*[She lifts the head spreader off the willow lane.",
         "*[She tips the willow's pockets empty, and lifts the head spreader off the willow lane."),
        ("comes home unmarked, silver as it set out.", "comes home unmarked."),
        ("*[She lifts the reed's token out of its pocket and stands it up on its edge",
         "*[She tips the willow's pockets empty again, lifts the reed's token out of its pocket, "
         "carries it over the bar, and stands it up on its edge"),
        ("{the bar across}", "{bar laid, head lifted}"),
        ("{the mark on the reed}", "{reed token turned}"),
        ("{the mark on the willow}", "{jumped to the willow}")])

    assert P.section_dumps(found) == [
        "conduitboard: before lift willow head | willow no head high; reed silver; bar willow high",
        "conduitboard: + ghost spreader willow head",
        "conduitboard: + road willow=high silver",
        "conduitboard: + reed silver",
        "conduitboard: + token willow high silver at before",
        "conduitboard: past marks | willow no head high; reed silver; bar willow high",
        "conduitboard: + road willow=high silver",
        "conduitboard: + reed slate",
        "conduitboard: + ghost token willow high silver at before",
        "conduitboard: + token willow high silver at past",
        "conduitboard: + halo reed silver",
        "conduitboard: + flip reed",
        "conduitboard: before wedge reed | willow no head high; reed two-faced; bar willow high",
        "conduitboard: + road willow=high slate",
        "conduitboard: + ghost reed slate cup",
        "conduitboard: + reed two-faced",
        "conduitboard: + token willow high silver at before",
        "conduitboard: + lift reed cup to wedge",
        "conduitboard: past expects reed | willow no head high; reed two-faced; bar willow high",
        "conduitboard: + road willow=high silver",
        "conduitboard: + reed two-faced",
        "conduitboard: + token willow high silver at past",
        "conduitboard: past marks | willow no head high; reed two-faced; bar willow high",
        "conduitboard: + road willow=high slate",
        "conduitboard: + reed two-faced",
        "conduitboard: + ghost token willow high silver at before",
        "conduitboard: + token willow high slate at past",
        "conduitboard: + flip willow high",
        "conduitboard: + pin reed"]


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
