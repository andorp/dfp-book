"""Challenge 5: The Sixteen Roads
================================
"are the roads all of one kind, or split?"  -- Achilles

Claim (Deutsch-Jozsa, master diagram).  With the answer wire held in |-> at both
ends (its kickback absorbed),
    <0|^4 <-| (H^4 (x) I) U_f (H^4 (x) I) |0>^4 |-> = (1/16) sum_x (-1)^{f(x)},
= +-1 for a constant f (all sixteen roads pile up), = 0 for a balanced f (eight +
roads and eight - roads cancel).  The chapter's title, made literal.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from functools import reduce

import numpy as np
from dfp import Picture, ket, kron, matrix_of, verdict
from dfp_props import Dialogue, words_only
from pyzx.utils import EdgeType
import props_ch05 as P

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

k0, k1 = ket(0), ket(1)
KMINUS = (k0 - k1) / np.sqrt(2)

# three representative functions f : {0,1}^4 -> {0,1}, x written top wire first
# (the top query wire is the most significant bit).
FS = {
    "const0": lambda x: 0,
    "const1": lambda x: 1,
    "balanced": lambda x: (x >> 3) & 1,   # f(x) = the first (top) query bit
}
VALUE = {"const0": 1, "const1": -1, "balanced": 0}


def bits(x, n=4):
    return [(x >> (n - 1 - i)) & 1 for i in range(n)]


def oracle(f):
    """U_f on four query wires + one answer wire: |x>|y> -> |x>|y XOR f(x)>."""
    U = np.zeros((32, 32), dtype=complex)
    for x in range(16):
        for y in (0, 1):
            col = 2 * x + y
            U[:, col] = kron(ket(*bits(x)), ket(y ^ f(x))).ravel()
    return U


def blackboard():
    H4 = reduce(np.kron, [H] * 4)
    init = kron(ket(*bits(0)), KMINUS)                # |0000>|->
    bra = kron(ket(*bits(0)), KMINUS).conj().reshape(1, 32)   # <0000|<-|

    for name, f in FS.items():
        # the balanced example must be GENUINELY balanced before we trust the 0
        tally = [sum(1 for x in range(16) if f(x) == v) for v in (0, 1)]
        if name == "balanced":
            assert tally == [8, 8]                    # eight and eight, exactly
        else:
            assert tally in ([16, 0], [0, 16])        # constant

        U = oracle(f)
        closed = (bra @ kron(H4, I2) @ U @ kron(H4, I2) @ init).item()
        formula = sum((-1) ** f(x) for x in range(16)) / 16
        assert np.isclose(closed, formula)            # the value IS the road-sum formula
        assert np.isclose(closed, VALUE[name])

    # constant piles to +-1, balanced cancels to 0 -- and they are not confused
    c0 = (bra @ kron(H4, I2) @ oracle(FS["const0"]) @ kron(H4, I2) @ init).item()
    bal = (bra @ kron(H4, I2) @ oracle(FS["balanced"]) @ kron(H4, I2) @ init).item()
    assert not np.isclose(c0, 0) and np.isclose(bal, 0)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the whole board and its home pocket with
#   \conduitboard[view]{setting}{caption}     (pdf/props/ch05.tex)
# and tools/props_ch05.py transcribes what the Lua computes: the sixteen
# roads in order, each road's token after the bar's kick, the whole board's
# home pocket and its settling.  The calls are read out of this challenge's
# section of dialog.md, never copied.  This last file also checks the
# dialogue as a whole: seventeen pictures, and the keyed rule.
#
# The state model (identity card, section 3): silver = +1, slate = -1; the
# home pocket is <0| on every lane in play; the two-faced reed token is |->,
# closed by <-|; each spreader is H; the bar is U_f, its rule read as f.  With
# four lanes and both spreaders the home pocket's net is 16 x the closed
# master value.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("conduitboard", "roadtiles"))
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth")
NOT_IN_CAPTIONS = ("wire", "wires", "diagram", "diagrams", "drawing", "drawings", "picture",
                   "pictures", "box", "boxes", "dot", "dots", "node", "nodes", "lanes",
                   "strings", "five")
FOUR = "willow; alder; hazel; rowan; reed two-faced"
BOARD_RULES = ("willow high", "never", "always", "odd", "even", "alder low", "rowan high")


def kind(call):
    """What a keyed picture labels: a board (its lanes and pockets), a
    close-up (the name of its cup), the tiles."""
    return (call.macro, call.opt.split()[0] in ("home", "away", "pockets") if call.opt else False)


def f_of(board):
    """The bar's rule as a function of x (top lane the most significant bit),
    read through the transcribed Lua."""
    fn = P.rule_function(board.setting.bar, board.live)
    return lambda x: fn(tuple(bits(x, len(board.live))))


def after_the_feet(board):
    """The blackboard's state at the feet, for a board with the reed two-faced:
    each lane |0> through its head spreader H (or its hand-walked |x>), the bar
    U_f, then each foot spreader H; returned as the amplitudes of <y|<-|, y in
    road order, times sqrt2 per spreader walked through (the Lua's own scale)."""
    live, lanes = board.live, board.setting.lanes
    n = len(live)
    fn = P.rule_function(board.setting.bar, live)
    U = np.zeros((2 ** (n + 1), 2 ** (n + 1)), dtype=complex)
    for x in range(2 ** n):
        for y in (0, 1):
            U[:, 2 * x + y] = kron(ket(*bits(x, n)), ket(y ^ fn(tuple(bits(x, n))))).ravel()
    heads = reduce(np.kron, [H @ ket(0) if lanes[l].head else ket(lanes[l].hand) for l in live])
    feet = reduce(np.kron, [H if lanes[l].foot else I2 for l in live])
    out = np.kron(feet, I2) @ U @ np.kron(heads.ravel(), KMINUS.ravel())
    m = sum(lanes[l].head + lanes[l].foot for l in live)
    return [np.sqrt(2) ** m * (kron(ket(*bits(y, n)), KMINUS).conj().ravel() @ out)
            for y in range(2 ** n)]


def check_transcription():
    for rule in BOARD_RULES:
        spec = f"{FOUR}; bar hidden {rule}"
        c = P.conduitboard("home", spec)
        f = f_of(c)
        # sixteen roads, in binary counting with the willow the top digit
        assert [r.values for r in c.roads] == \
            [tuple(zip(P.QUERY, bits(i))) for i in range(16)]
        # one token per road, as it came: (-1)^{f(x)}
        assert c.cups["home"] == [P.face((-1) ** f(x)) for x in range(16)]
        # its net is 16 x the closed board
        U = oracle(f)
        H4 = reduce(np.kron, [H] * 4)
        closed = (kron(ket(*bits(0)), KMINUS).conj().reshape(1, 32) @ np.kron(H4, I2) @ U
                  @ np.kron(H4, I2) @ kron(ket(*bits(0)), KMINUS)).item()
        assert np.isclose(P.net(c.cups["home"]), 16 * closed)
        assert np.isclose(P.net(c.cups["home"]), after_the_feet(c)[0])
        # the whole board's `end`: drawn only when the walk comes out in one
        # definite pocket per lane -- the one y where the blackboard is not 0
        amps = after_the_feet(c)
        where = [y for y in range(16) if not np.isclose(amps[y], 0)]
        assert len(where) == 1
        e = P.conduitboard("end", spec)
        assert {l: ("away" if b else "home") for l, b in zip(P.QUERY, bits(where[0]))} == \
            {l: next(p for p, h in e.heaps[l].items() if h) for l in P.QUERY}
        assert all(h[0] == P.face(amps[where[0]].real) for l in P.QUERY
                   for h in e.heaps[l].values() if h)
    # a board whose walk does NOT come out in one pocket per lane: the macro
    # refuses `end`, and the blackboard agrees there is more than one pocket
    spec = "willow no head high; alder; reed two-faced; bar never"
    assert P.faults(P.conduitboard, "end", spec) != []
    b = P.conduitboard("set", spec)
    assert sum(not np.isclose(a, 0) for a in after_the_feet(b)) > 1
    # and a two-lane board with both spreaders on each: one pocket per lane, drawn
    assert P.faults(P.conduitboard, "end", "willow; alder; reed two-faced; bar never") == []
    # with more than one lane in play only the home pocket is read
    assert P.faults(P.conduitboard, "away", f"{FOUR}; bar never") != []
    assert P.faults(P.conduitboard, "pockets settled", f"{FOUR}; bar never") != []


def check_whole_dialogue():
    assert DIALOGUE.check_whole(ORDINALS) == 17                  # 4 + 4 + 4 + 2 + 3
    DIALOGUE.keyed_first(ORDINALS, kind=kind)
    placed = [(o, i, c) for o in ORDINALS
              for i, c in enumerate(c for b in DIALOGUE.blocks(o) for c in b)]
    assert [(o, i) for o, i, c in placed if c.keyed] == [("First", 0), ("Second", 0),
                                                         ("Fourth", 0)]
    for o, i, c in placed:
        P.picture(c)                                              # no macro ever faults
        assert words_only(c.caption, forbidden=NOT_IN_CAPTIONS), c.caption
        assert "sixteen roads" not in c.caption.lower()           # never the master's name
    # the rules fixed by the card (section 3b, trace 6), and only those, in order
    rules = {}
    for o, i, c in placed:
        b = P.picture(c)
        if c.macro == "conduitboard" and b.setting.bar is not None:
            rules.setdefault(o, []).append(rule_name(b.setting.bar))
    assert rules == {"Third": ["willow high"] * 4, "Fourth": ["never", "willow low"],
                     "Fifth": ["willow high"] * 3}, rules


def rule_name(rule):
    if rule.kind == "lane":
        return f"{rule.lane} {'high' if rule.v else 'low'}"
    return rule.kind


def pictures():
    check_transcription()
    check_whole_dialogue()
    p51, p52, p53 = DIALOGUE.check_layout("Fifth", sizes=[1, 2])
    b51, b52, b53 = (P.picture(c) for c in (p51, p52, p53))
    assert p51.args == p52.args == p53.args                      # one board, three moments
    assert all(c.macro == "conduitboard" and not c.keyed for c in (p51, p52, p53))
    assert all(c.opt.split()[:1] != ["end"] for c in (p51, p52, p53))   # no `end` in the Fifth

    a1, a2 = P.anchors(DIALOGUE, "Fifth")
    assert a1.endswith("Walk it, and read the home pocket alone.")
    assert "the pocket a walker reaches only by coming home on every lane at once" in a2
    # nothing after the last block is a picture: the Cat's measurement ends it
    rest = DIALOGUE.section("Fifth").split(p53.text)[1]
    assert "\\conduitboard" not in rest and "\\roadtiles" not in rest
    assert "**Cat:** Different — the roads cancel." in rest

    # P5.1: every lane in play, both spreaders; the reed two-faced; the bar hidden
    assert b51.view == "set" and b51.grooves == P.GROOVES and b51.labels == ()
    assert b51.live == P.QUERY
    assert b51.spreaders == {l: (True, True) for l in P.QUERY}
    assert b51.reed == "two-faced" and b51.bar.hidden and b51.pegs == ()
    assert b51.walking == () and b51.heaps == {}
    f = f_of(b51)
    assert [f(x) for x in range(16)] == [FS["balanced"](x) for x in range(16)]   # `willow high`
    assert sorted(f(x) for x in range(16)) == [0] * 8 + [1] * 8                # balanced

    # P5.2: the whole board's home pocket as they fell -- eight silver, then eight slate
    assert b52.view == "home" and not b52.settled and b52.which == "home"
    assert b52.cups["home"] == ["silver"] * 8 + ["slate"] * 8
    assert b52.cups["home"] == [P.face((-1) ** FS["balanced"](x)) for x in range(16)]

    # P5.3: settled -- not a token left, net 0 = 16 x the closed master value
    assert b53.view == "home settled" and b53.heaps == {"home": []}
    H4 = reduce(np.kron, [H] * 4)
    closed = (kron(ket(*bits(0)), KMINUS).conj().reshape(1, 32) @ np.kron(H4, I2)
              @ oracle(FS["balanced"]) @ np.kron(H4, I2) @ kron(ket(*bits(0)), KMINUS)).item()
    assert P.net(b53.cups["home"]) == 0 and np.isclose(16 * closed, 0)
    # Achilles's other case: a constant rule would have piled all sixteen
    const = P.conduitboard("home settled", f"{FOUR}; bar hidden always")
    assert const.heaps == {"home": [("slate", 8), ("slate", 8)]}


def hedge(p, a, b):
    p.g.add_edge((a, b), EdgeType.HADAMARD)


def closed_board(name, normalized=True):
    """The five-wire board drawn CLOSED, so the picture is a single number.
    Four query wires: |0> - H - U_f - H - <0|.  Answer wire: |-> - U_f - <-|."""
    p = Picture()
    power = 0
    qmid = []
    for r in range(4):                                # PyZX draws this with the spiders you meet in Chapter 8.
        st = p.x(r, 0, 0)                             # |0> = X(0)/sqrt2
        mid = p.z(r, 3, 0)                            # a phase-free node between the two H's
        ef = p.x(r, 6, 0)                            # <0| = X(0)/sqrt2
        hedge(p, st, mid)
        hedge(p, mid, ef)
        qmid.append(mid)
        power -= 2                                    # the state and the effect
    a_st = p.z(4, 0, 1)                              # |-> = Z(pi)/sqrt2
    a_ef = p.z(4, 6, 1)                              # <-| = Z(pi)/sqrt2
    if normalized:
        power -= 2                                    # keep |-> normalized at BOTH ends
    f = FS[name]
    if name == "const0":                             # f = 0: nothing on the answer
        p.wire(a_st, a_ef)
    elif name == "const1":                           # f = 1: flip the answer always
        xn = p.x(4, 3, 1)
        p.wire(a_st, xn, a_ef)
    else:                                             # balanced: control on the top query wire
        xt = p.x(4, 3, 0)
        p.wire(a_st, xt, a_ef)
        p.g.add_edge((qmid[0], xt))
        power += 1                                    # the control/target pair carries a factor sqrt2
    g = p.done()
    g.scalar.add_power(power)
    return g


def sketchbook():
    # the closed diagram is a number; evaluate ONLY through tools/dfp.py
    # (matrix_of forces strategy="naive"; pyzx's default is wrong on some
    # closed diagrams -- issues/chapter03/01).
    for name, want in VALUE.items():
        val = matrix_of(closed_board(name)).ravel()[0]
        assert np.isclose(val, want)                  # exact scalar: 1, -1, or 0
    # constant and balanced are not confused
    assert not np.isclose(matrix_of(closed_board("const0")).ravel()[0], 0)
    assert np.isclose(matrix_of(closed_board("balanced")).ravel()[0], 0)

    # why |-> must stay normalized at BOTH ends: an unnormalized answer wire
    # multiplies the constant value by <-|-> = 2 (the stray factor the chapter names).
    for name in ("const0", "const1"):
        stray = matrix_of(closed_board(name, normalized=False)).ravel()[0]
        assert np.isclose(stray, 2 * VALUE[name])
        assert not np.isclose(stray, VALUE[name])


if __name__ == "__main__":
    verdict("SixteenRoads", blackboard, sketchbook)
