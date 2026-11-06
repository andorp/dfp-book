"""Challenge II: Say Only the Setting  (the BB84 protocol; sifting)
=================================================================
"Anyone with an ear to the keyhole has heard "upright, upright" and learned that we kept the round — and not a whisper of which piece we laid."  -- Achilles

Claim: with x, r, t independent fair coins in every round, exactly: the settings
match with probability 1/2; on every kept round (r = t) Bob's reading is Alice's
bit; before sifting y != x with probability 1/4; and the announcement (r, t) tells a
listener nothing about the kept bit, P(x = 0 | r, t, kept) = 1/2.  In the twin (pair)
form a kept round is a shared fair coin, p(kA, kB) = 1/2 [kA = kB] -- a green bend
on the two records with 1/2 in front -- and a struck round is two independent coins.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import BELL, CUP, I2, kron, ket, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch12 as PC


H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
DELTA = np.outer(ket(0, 0), ket(0)) + np.outer(ket(1, 1), ket(1))   # the copier
PLAIN, CROSS = 0, 1


def Hp(r):
    return H if r else I2


def bb84(y, x, r, t):
    """P(y | x, r, t): Alice sends H^r|x>, Bob reads with the effects <y|H^t."""
    return abs(ket(y) @ Hp(t) @ Hp(r) @ ket(x)) ** 2


def twin(rA, rB):
    """p[kA, kB | rA, rB]: the pair Phi+ = cup/sqrt2, read at both ends."""
    return np.array([[abs((kron(ket(a).reshape(1, 2) @ Hp(rA), ket(b).reshape(1, 2) @ Hp(rB))
                           @ BELL)[0]) ** 2 for b in (0, 1)] for a in (0, 1)])


def blackboard():
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))      # cup unnormalized
    # the doubled pair: Phi+ (x) Phi+* on (a, b, abar, bbar) is 1/2 cup (x) cup -- the
    # 1/2 that every doubled pair picture below carries as add_power(-2)
    assert np.allclose(np.kron(BELL, BELL.conj()), 0.5 * np.kron(CUP, CUP))

    # Result 12.3, by exact enumeration of the eight equally likely triples (x, r, t)
    trip = [(x, r, t) for x in (0, 1) for r in (0, 1) for t in (0, 1)]
    kept = sum(1 / 8 for x, r, t in trip if r == t)
    kept_err = sum(1 / 8 * bb84(1 - x, x, r, t) for x, r, t in trip if r == t)
    raw_err = sum(1 / 8 * bb84(1 - x, x, r, t) for x, r, t in trip)
    assert np.isclose(kept, 0.5)                             # (i)
    assert np.isclose(kept_err, 0)                           # (ii) every kept round: y = x
    assert np.isclose(raw_err, 0.25)                         # (iii) the raw key: 25% errors
    # (iv) the announcement is independent of the kept bit -- for Alice's x and Bob's y
    for r in (0, 1):
        for t in (0, 1):
            joint = np.array([[1 / 8 * bb84(y, x, r, t) for y in (0, 1)] for x in (0, 1)])
            p_rt = joint.sum()
            assert np.isclose(joint.sum(axis=1)[0] / p_rt, 0.5)      # P(x = 0 | r, t) = 1/2
            assert np.isclose(joint.sum(axis=0)[0] / p_rt, 0.5)      # P(y = 0 | r, t) = 1/2

    # the twin form: kept rounds are a shared fair coin, struck rounds two independent coins
    for rA in (0, 1):
        for rB in (0, 1):
            p = twin(rA, rB)
            if rA == rB:
                assert np.allclose(p, np.eye(2) / 2)
            else:
                assert np.allclose(p, np.outer(p.sum(axis=1), p.sum(axis=0)))   # a product
                assert np.allclose(p, 0.25)
            assert np.allclose(p.sum(axis=1), 0.5)           # each record alone: a fair coin

    # a seeded run of rounds at the table: secret settings, silent pieces, the settings
    # said aloud, the mismatched rounds struck.  The two rows are equal on every kept
    # round, whatever the seed (the model forbids anything else)
    for seed in range(20):
        rng = np.random.default_rng(1202 + seed)
        n = 400
        rowA, rowB, nkept = [], [], 0
        for _ in range(n):
            rA, rB = rng.integers(2), rng.integers(2)
            k = rng.choice(4, p=twin(rA, rB).ravel())
            if rA == rB:
                nkept += 1
                rowA.append(k // 2); rowB.append(k % 2)
        assert rowA == rowB
        assert abs(nkept - n / 2) <= 5 * np.sqrt(n / 4)     # about half kept

    # the worked run of section 12.2 (x, Alice's r, Bob's t, Bob's y)
    run = [(0, PLAIN, PLAIN, 0), (1, PLAIN, CROSS, 0), (1, CROSS, CROSS, 1),
           (0, CROSS, PLAIN, 1), (0, PLAIN, PLAIN, 0), (1, CROSS, CROSS, 1)]
    for x, r, t, y in run:
        assert bb84(y, x, r, t) > 0                          # every reading was possible
    assert [i + 1 for i, (x, r, t, y) in enumerate(run) if r == t] == [1, 3, 5, 6]
    assert [y for x, r, t, y in run if r == t] == [x for x, r, t, y in run if r == t] == [0, 1, 0, 1]

    # the ledger of II (settings said aloud, Achilles's then the Tortoise's):
    # keep, strike, keep, strike, keep, strike; the kept pieces black, white, black at both ends
    said = [(PLAIN, PLAIN), (CROSS, PLAIN), (CROSS, CROSS), (PLAIN, CROSS), (PLAIN, PLAIN), (CROSS, PLAIN)]
    assert [rA == rB for rA, rB in said] == [True, False, True, False, True, False]
    kept_pieces = [1, 0, 1]                                  # black, white, black
    for (rA, rB), k in zip([s for s in said if s[0] == s[1]], kept_pieces):
        assert twin(rA, rB)[k, k] > 0                        # possible ...
        assert np.isclose(twin(rA, rB)[k, 1 - k] + twin(rA, rB)[1 - k, k], 0)   # ... and matched
    # what a listener learns from "upright, upright": that the round was kept, and not
    # which piece was laid -- each colour exactly half the time
    assert np.allclose(twin(PLAIN, PLAIN).sum(axis=1), 0.5)

    # Alice's page: of 200 trial glints, 96 matched -- "about half, near enough" (within
    # 3 standard deviations of binomial(200, 1/2)); "not one of those disagreed" is exact;
    # "each glint I send is half a letter of key" is the kept fraction 1/2
    assert abs(96 - 100) <= 3 * np.sqrt(50)
    assert np.isclose(kept, 0.5) and np.isclose(kept_err, 0)
    pictures()


# --- the prop pictures (dialog.md: \glinttray, \keycloth; tools/props_ch12.py) -----------

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = ("Anyone with an ear to the keyhole has heard \"upright, upright\" and learned that we kept "
         "the round — and not a whisper of which piece we laid.")
R8 = "white black white black white black white black"
PIECE = {"white": 0, "black": 1}         # a piece: white yes (0), black no (1)
COLOUR = {"green": PLAIN, "red": CROSS}  # the cloth's rings: green upright, red crosswise


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    calls = DIALOGUE.check_layout("Second", sizes=[1, 2])
    p21, p22, p23 = calls
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("glinttray", False, "", (f"his: {R8}; hers: {R8}",), "laid open, piece for piece"),
        ("keycloth", False, "", ("green to green, red to red",), "upright above, crosswise below"),
        ("keycloth", False, "slide together, lift off", ("green to green, red to red",), "no difference to be seen")]
    PC.check_captions(DIALOGUE, "Second", calls)
    PC.check_section(DIALOGUE, "Second", calls)
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Second")] == [[PC.TRAY_FRAME], [2.32, 2.32]]

    a1, a2 = DIALOGUE.anchors("Second")
    f1, f2 = DIALOGUE.following("Second")
    assert "laid open now, the two rows match piece for piece" in a1
    assert f1.startswith("**Achilles:** The same, right to the end.")
    assert "The rounds that mixed red with green she does not lay at all." in a2
    assert "the fallen cords into her pouch" in a2                              # D7
    assert f2.startswith("**Achilles:** Two straight cords.")
    assert DIALOGUE.following("Second", DIALOGUE.raw) == [f1, f2]
    section = DIALOGUE.section("Second")
    assert "sweeps Achilles's spare pieces back into the box" in section         # D6
    assert "back to the table's far side" in section                            # D6
    # G11: II's screened run is never drawn -- the Second's first tray picture is the
    # laid-open rows, after the screens come down
    paras = DIALOGUE.paragraphs("Second")
    assert paras.index(next(p for p in paras if "\\glinttray" in p)) == paras.index(a1) + 1
    assert paras.index(a1) > paras.index(next(p for p in paras if "back to the table's far side" in p))

    t21, c22, c23 = (PC.picture(c) for c in calls)
    assert t21.dump == (f"sleeve closed; tell's key in the box; side rod in the box; his [{R8}]; hers [{R8}]; "
                        "tell's []; bell still")
    rows = t21.shown.rows
    assert [c for c, _, _ in rows["his"]] == [c for c, _, _ in rows["hers"]]          # piece for piece
    assert all(np.isclose(a[1], -b[1]) for a, b in zip(rows["his"], rows["hers"]))  # each opposite its partner
    assert all(a[1] < b[1] for a, b in zip(rows["his"], rows["hers"]))              # from under each key, inwards
    assert rows["his"][0][1] == -PC.XROW0 and rows["hers"][0][1] == PC.XROW0
    # the rows are I's five (l.397's read-out) and II's kept three (the ledger above)
    five, kept = [0, 1, 0, 1, 0], [1, 0, 1]
    assert [PIECE[c] for c, _, _ in rows["his"]] == five + kept
    # the fixed parts: the things on the black at the Lua's places, the bell still on its corner
    assert {t.name: t.x for t in t21.shown.black} == {"his key": PC.XHIS, "her key": PC.XHERS,
                                                       "lamp": PC.XLAMP, "sleeve": PC.XSLEEVE}
    assert t21.shown.bell == (PC.XBELL, PC.YBELL, 0.0)

    # P2.2 | P2.3: two rounds, green above and red below; each slid together and lifted off
    assert c22.dump.startswith("scene=rounds round1=green-green joined round2=red-red joined fork=false")
    assert [(r["c"], r["y"]) for r in c22.cloth["things"]] == [("green", 0), ("green", 0), ("red", 2), ("red", 2)]
    assert c23.dump == ("scene=rounds round1=green-green lifted round2=red-red lifted fork=false | cols=7 rows=4 "
                        "y=0..3 near=false folded=false; cord loose-loose [-0.45,0 6.45,0]; "
                        "cord loose-loose [-0.45,2 6.45,2]")
    upper, lower = c23.cloth["cords"]
    assert [[x, y + 2] for x, y in upper["pts"]] == lower["pts"] and upper["a"] == lower["a"] and \
        upper["b"] == lower["b"]                                  # identical but for the row
    assert c23.cloth["things"] == []                              # no colour left on the cloth
    assert c22.frame == c23.frame == 2.32

    # the ties to the blackboard (structural): a kept round's classical channel is the
    # identity at either setting -- the lifted cord from edge to edge -- and a listener
    # told the colour learns nothing of the piece
    for comp_row, c in ((0, "green"), (2, "red")):
        r = COLOUR[c]
        assert any({("left", comp_row), ("right", comp_row)} <= comp for comp in PC.cloth_links(c23.cloth))
        channel = np.array([[bb84(y, x, r, r) for x in (0, 1)] for y in (0, 1)])
        assert np.allclose(channel, I2)
        # p(x | r, t, kept) = 1/2: x is a fair coin, and the round's announcement is (r, r)
        post = np.array([0.5 * sum(bb84(y, x, r, r) for y in (0, 1)) for x in (0, 1)])
        assert np.allclose(post / post.sum(), [0.5, 0.5])
    # the rows laid open: every kept round's error is exactly 0 (piece for piece)
    for k in kept:
        for r in (PLAIN, CROSS):
            assert np.isclose(bb84(1 - k, k, r, r), 0)


# --- pictures (hand-built doubled diagrams; pyzx 0.10.6 refuses ground vertices) -----

def pair_round(rA, rB):
    """The twin read at both ends, in unfolded doubled form.  Vertices: the cup on (a, b)
    and the mirror cup on (abar, bbar), phase-free green bends; Alice's read-spider on
    (a, abar); Bob's on (b, bbar); outputs kA, kB (classical).  A cross reader has a
    Hadamard edge on both its quantum legs.
    Scalar: the doubled pair is 1/2 cup (x) cup and the cups are unnormalized, the read-
    spiders exact: add_power(-2)."""
    g = zx.Graph()
    V = g.add_vertex
    cup, cupb = V(VT.Z, 1, 0), V(VT.Z, 2, 0)
    rdA, rdB = V(VT.Z, 0, 2), V(VT.Z, 3, 2)
    kA, kB = V(VT.BOUNDARY, 0, 3), V(VT.BOUNDARY, 3, 3)
    ha = ET.HADAMARD if rA else ET.SIMPLE
    hb = ET.HADAMARD if rB else ET.SIMPLE
    g.add_edge((cup, rdA), ha); g.add_edge((cupb, rdA), ha)      # a, abar
    g.add_edge((cup, rdB), hb); g.add_edge((cupb, rdB), hb)      # b, bbar
    g.add_edge((rdA, kA)); g.add_edge((rdB, kB))                 # the two records
    g.set_inputs(()); g.set_outputs((kA, kB))
    g.scalar.add_power(-2)
    return g


def coin_round(r, t):
    """Section 12.2's left-hand picture: a fair coin (1/2 times a green point), Alice's
    copy of it on kA (the green copier; a classical bit may be copied), her sender at r,
    two identity dots on the doubled line, Bob's reader at t, his record kB.
    add_power(-2): the fair coin's 1/2."""
    g = zx.Graph()
    V = g.add_vertex
    coin, cp, s = V(VT.Z, 1, 0), V(VT.Z, 1, 1), V(VT.Z, 2, 2)
    q, qb, rd = V(VT.Z, 1.5, 3), V(VT.Z, 2.5, 3), V(VT.Z, 2, 4)
    kA, kB = V(VT.BOUNDARY, 0, 5), V(VT.BOUNDARY, 2, 5)
    hr = ET.HADAMARD if r else ET.SIMPLE
    ht = ET.HADAMARD if t else ET.SIMPLE
    g.add_edge((coin, cp)); g.add_edge((cp, kA)); g.add_edge((cp, s))
    g.add_edge((s, q), hr); g.add_edge((s, qb), hr)
    g.add_edge((q, rd), ht); g.add_edge((qb, rd), ht)
    g.add_edge((rd, kB))
    g.set_inputs(()); g.set_outputs((kA, kB))
    g.scalar.add_power(-2)
    return g


def shared_coin():
    """1/2 times a green bend on (kA, kB): 1/2(|00> + |11>).  add_power(-2)."""
    g = zx.Graph()
    z = g.add_vertex(VT.Z, 1, 0)
    kA, kB = g.add_vertex(VT.BOUNDARY, 0, 1), g.add_vertex(VT.BOUNDARY, 2, 1)
    g.add_edge((z, kA)); g.add_edge((z, kB))
    g.set_inputs(()); g.set_outputs((kA, kB))
    g.scalar.add_power(-2)
    return g


def two_coins():
    """1/4 times two green points: independent fair coins.  add_power(-4)."""
    g = zx.Graph()
    z1, z2 = g.add_vertex(VT.Z, 0, 0), g.add_vertex(VT.Z, 2, 0)
    kA, kB = g.add_vertex(VT.BOUNDARY, 0, 1), g.add_vertex(VT.BOUNDARY, 2, 1)
    g.add_edge((z1, kA)); g.add_edge((z2, kB))
    g.set_inputs(()); g.set_outputs((kA, kB))
    g.scalar.add_power(-4)
    return g


def connected(g, u, v):
    seen, todo = {u}, [u]
    while todo:
        w = todo.pop()
        for n in g.neighbors(w):
            if n not in seen:
                seen.add(n); todo.append(n)
    return v in seen


def sketchbook():
    for rA in (0, 1):
        for rB in (0, 1):
            g = pair_round(rA, rB)
            # the VALUE: the twin's joint distribution, exactly
            assert np.allclose(matrix_of(g).ravel(), twin(rA, rB).ravel())
            h = g.copy(); simp.full_reduce(h)               # DERIVATION, scalar kept
            kA, kB = h.outputs()
            if rA == rB:
                # a kept round is a shared fair coin: 1/2 x a bend on the two records
                assert same_picture(g, shared_coin())
                assert not same_picture(g, two_coins())
                # full_reduce leaves a bare edge joining the two outputs (a state; it
                # evaluates as one) and the 1/2
                assert h.num_vertices() == 2 and h.num_edges() == 1 and connected(h, kA, kB)
                assert same_picture(h, shared_coin())
            else:
                # a struck round: the two records fall apart, two independent coins
                assert same_picture(g, two_coins())
                assert not connected(h, kA, kB)
                assert same_picture(h, two_coins())

    # section 12.2: a fair coin, Alice's copy and the kept round fuse into one green dot
    # with two classical legs -- the same shared coin the twin gives (section 12.4)
    for s in (0, 1):
        g = coin_round(s, s)
        assert same_picture(g, shared_coin())
        assert same_picture(g, pair_round(s, s))
        h = g.copy(); simp.full_reduce(h)
        assert same_picture(h, shared_coin()) and connected(h, *h.outputs())
    # and the prepare-and-measure rounds that are struck: the same two coins
    assert same_picture(coin_round(0, 1), two_coins()) and same_picture(coin_round(1, 0), two_coins())


if __name__ == "__main__":
    verdict("SiftedKey", blackboard, sketchbook)
