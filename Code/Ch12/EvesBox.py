"""Challenge III: The Tell-Glass  (Eve is a box on the wire; information-disturbance)
===================================================================================
"And the upright rounds came through untouched because the tell-glass copies its own kind, as the press did: an upright glint it copies and lets be, and a crosswise one it can only make a pair of."  -- Achilles

Claim: Eve's press is Chapter 11's copier delta on the line of a kept round; Bob
reads its through-leg, Eve her kept leg.  At the plain setting
p(kB, kE | x) = [kB = x][kE = x]: Bob is right and Eve holds the key bit.  At the
cross setting, with Eve reading plain, p(kB, kE | x) = 1/4 for all pairs; with Eve
reading cross, 1/2 [kB xor kE = x].  Either way Bob disagrees with Alice half the
time and Eve's record alone is independent of x.  Averaged over a fair setting, Eve
learns 1/2 bit and Bob disagrees on 1/4 of the kept rounds.  With her copy
discarded the press is the four-legged green dephasing dot, and a cross round with
it is 1/2 J -- a kept round turned into a struck one.
Bennett and Brassard's general bound (no more than 1/2 expected bit; b bits force a
disagreement of at least b/2) is stated in the chapter and cited, NOT tested here.
"""
import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import BELL, CUP, I2, kron, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch12 as PC


H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
DELTA = np.outer(ket(0, 0), ket(0)) + np.outer(ket(1, 1), ket(1))   # the copier (Eve's press)
PLAIN, CROSS = 0, 1


def Hp(r):
    return H if r else I2


def eff(k, s):
    """The reading effect <k| H^s at setting s, as a 1x2 row."""
    return ket(k).reshape(1, 2) @ Hp(s)


def press_round(x, s, sE, press=DELTA):
    """p[kB, kE | x]: Alice sends H^s|x>, the press acts, Bob reads its through-leg at s
    and Eve her kept leg at sE (after the announcement)."""
    out = press @ Hp(s) @ ket(x)
    return np.array([[abs((kron(eff(kb, s), eff(ke, sE)) @ out)[0]) ** 2 for ke in (0, 1)]
                     for kb in (0, 1)])


def info(joint):
    """Mutual information in bits of a joint distribution p[x, k]."""
    px, pk = joint.sum(axis=1), joint.sum(axis=0)
    return sum(joint[i, j] * np.log2(joint[i, j] / (px[i] * pk[j]))
               for i in (0, 1) for j in (0, 1) if joint[i, j] > 1e-12)


def table(rA, rB, rE):
    """p[kA, kB, kE]: the table form -- the lamp's twin, the tell-glass (delta) on the
    Tortoise's half, the three keys at settings rA (Achilles), rB (Tortoise), rE (tell)."""
    ghz = kron(I2, DELTA) @ BELL                            # (I (x) delta)|Phi+>, on (A, B, E)
    return np.array([[[abs((kron(eff(a, rA), eff(b, rB), eff(e, rE)) @ ghz)[0]) ** 2
                       for e in (0, 1)] for b in (0, 1)] for a in (0, 1)])


def blackboard():
    rng = np.random.default_rng(1203)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))      # cup unnormalized
    # the press is exact and REAL, so its mirror copy is the green copier again
    assert np.allclose(DELTA @ ket(0), ket(0, 0)) and np.allclose(DELTA @ ket(1), ket(1, 1))
    assert np.allclose(DELTA, DELTA.conj())
    # Result 11.2 again: delta|+-> is a Bell pair, not two copies
    assert np.allclose(DELTA @ (H @ ket(0)), BELL)
    assert not np.allclose(DELTA @ (H @ ket(0)), np.kron(H @ ket(0), H @ ket(0)))

    # Result 12.4 (i): plain, p(kB, kE | x) = [kB = x][kE = x]
    for x in (0, 1):
        want = np.zeros((2, 2)); want[x, x] = 1
        assert np.allclose(press_round(x, PLAIN, PLAIN), want)
    # (ii): cross; Eve plain -> 1/4 everywhere; Eve cross -> 1/2 [kB xor kE = x]
    for x in (0, 1):
        assert np.allclose(press_round(x, CROSS, PLAIN), 0.25)
        want = np.array([[0.5 * ((kb ^ ke) == x) for ke in (0, 1)] for kb in (0, 1)])
        assert np.allclose(press_round(x, CROSS, CROSS), want)
        for sE in (PLAIN, CROSS):
            p = press_round(x, CROSS, sE)
            assert np.isclose(p[1 - x, :].sum(), 0.5)       # Bob wrong half the time
            assert np.allclose(p.sum(axis=0), 0.5)          # Eve's record alone: a fair coin
    # (iii) information and disagreement, with x a fair coin
    def eve_joint(s, sE):
        return np.array([0.5 * press_round(x, s, sE).sum(axis=0) for x in (0, 1)])   # p[x, kE]
    I_plain, I_cross = info(eve_joint(PLAIN, PLAIN)), info(eve_joint(CROSS, PLAIN))
    assert np.isclose(I_plain, 1) and np.isclose(I_cross, 0)
    assert np.isclose(info(eve_joint(CROSS, CROSS)), 0)       # waiting for "cross" does no better
    dis = {s: sum(0.5 * press_round(x, s, PLAIN)[1 - x, :].sum() for x in (0, 1)) for s in (0, 1)}
    assert np.isclose(dis[PLAIN], 0) and np.isclose(dis[CROSS], 0.5)
    b, d = (I_plain + I_cross) / 2, (dis[PLAIN] + dis[CROSS]) / 2
    assert np.isclose(b, 0.5) and np.isclose(d, 0.25)
    assert np.isclose(d, b / 2)          # the press sits exactly on the cited bound d >= b/2
    # Gisin et al.'s arithmetic, quoted in section 12.3: intercept-resend on a tenth of the
    # rounds costs about 2.5% errors for about 5% information
    assert np.isclose(0.1 * 0.25, 0.025) and np.isclose(0.1 * 0.5, 0.05)

    # "a cleverer Eve would have a glass that copies the crosswise answer instead" --
    # "On which rounds?": a cross copier spoils the UPRIGHT rounds instead.  A copier in
    # the wrong basis for the round costs disagreement 1/2, whichever way it goes.
    DELTA_X = kron(H, H) @ DELTA @ H
    for s in (PLAIN, CROSS):
        for press, own in ((DELTA, PLAIN), (DELTA_X, CROSS)):
            p_dis = sum(0.5 * press_round(x, s, s, press)[1 - x, :].sum() for x in (0, 1))
            assert np.isclose(p_dis, 0 if s == own else 0.5)

    # with Eve's copy discarded the press is dephasing: rho -> diag(rho), tested on a
    # NON-REAL, non-diagonal rho
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())
    rho = np.outer(psi, psi.conj())
    big = DELTA @ rho @ DELTA.conj().T
    kept = sum(kron(I2, ket(e).reshape(1, 2)) @ big @ kron(I2, ket(e).reshape(2, 1)) for e in (0, 1))
    assert np.allclose(kept, np.diag(np.diag(rho))) and not np.allclose(kept, rho)
    # ... in doubled form the four-legged green dot, sum_i |ii><ii| on (q, qbar)
    D4 = sum(np.outer(ket(i, i), ket(i, i)) for i in (0, 1))
    assert np.allclose(D4 @ np.kron(psi, psi.conj()), sum(rho[i, i] * ket(i, i) for i in (0, 1)))
    assert not np.allclose(D4 @ np.kron(psi, psi.conj()), np.kron(psi, psi.conj()))
    # section 12.3: forget Eve's record instead -- a cross round through the dephasing is
    # 1/2 J although the settings matched; a plain one is untouched
    E = lambda r: kron(Hp(r), Hp(r)) @ DELTA                # the sender
    R = lambda t: DELTA.conj().T @ kron(Hp(t), Hp(t))       # the reader
    assert np.allclose(R(CROSS) @ D4 @ E(CROSS), np.ones((2, 2)) / 2)
    assert np.allclose(R(PLAIN) @ D4 @ E(PLAIN), I2)

    # --- the table (III): the tell's key left upright; the two readers choose in secret ---
    for rA in (0, 1):
        for rB in (0, 1):
            p = table(rA, rB, PLAIN)
            assert np.isclose(p.sum(), 1)
            if rA == rB == PLAIN:
                # kA = kB = kE always: no disagreement on an upright round, and the tell has it
                assert np.isclose(sum(p[k, k, k] for k in (0, 1)), 1)
            if rA == rB == CROSS:
                # every triple 1/8: Achilles and the Tortoise disagree half the time, and the
                # tell's piece is independent of Achilles's -- "no guide at all"
                assert np.allclose(p, 1 / 8)
                pAE = p.sum(axis=1)
                assert np.allclose(pAE, np.outer(pAE.sum(axis=1), pAE.sum(axis=0)))
            if rA == PLAIN and rB == CROSS:
                assert np.isclose(sum(p[k, :, k].sum() for k in (0, 1)), 1)   # tell = Achilles
            if rA == CROSS and rB == PLAIN:
                assert np.isclose(sum(p[:, k, k].sum() for k in (0, 1)), 1)   # tell = Tortoise
    # "every one of them on a crosswise round, never an upright one"
    assert np.isclose(table(PLAIN, PLAIN, PLAIN)[0, 1, :].sum() + table(PLAIN, PLAIN, PLAIN)[1, 0, :].sum(), 0)

    # the ledger of III (yes = 0 = white, no = 1 = black):
    #   (Achilles's setting, the Tortoise's, Achilles's call, hers, the tell's piece)
    ledger = [(PLAIN, PLAIN, 0, 0, 0),     # R1 kept
              (CROSS, PLAIN, 1, 0, 0),     # R2 struck: the tell is her upright answer
              (CROSS, CROSS, 0, 1, 1),     # R3 kept: the first wrong note, unrung
              (PLAIN, CROSS, 0, 0, 0),     # R4 struck: the tell is his upright answer
              (PLAIN, PLAIN, 1, 1, 1),     # R5 kept
              (CROSS, CROSS, 1, 1, 0)]     # R6 kept: the tell's piece no guide
    for rA, rB, a, b, e in ledger:
        assert table(rA, rB, PLAIN)[a, b, e] > 0
    assert [(rA == rB) for rA, rB, *_ in ledger] == [True, False, True, False, True, True]
    # the kept upright rounds: calls agreed and the tell matched them, white then black
    assert [(a, b, e) for rA, rB, a, b, e in ledger if rA == rB == PLAIN] == [(0, 0, 0), (1, 1, 1)]
    # Alice's later pages: about one checked round in four disagreed -- the model's 1/4
    assert np.isclose(d, 0.25)
    pictures(ledger)


# --- the prop pictures (dialog.md: \glinttray, \keycloth; tools/props_ch12.py) -----------

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = ("And the upright rounds came through untouched because the tell-glass copies its own kind, as "
         "the press did: an upright glint it copies and lets be, and a crosswise one it can only make a "
         "pair of.")
R8 = "white black white black white black white black"
IIIR = (f"with the tell-glass; his: {R8} then white black white white black black; hers: {R8} then white white "
        "black white black black; tell's: then white white black white black white")
PIECE = {"white": 0, "black": 1}
SETTING = {"upright": PLAIN, "crosswise": CROSS}


def pictures(ledger):
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    calls = DIALOGUE.check_layout("Third", sizes=[1, 2, 1])
    p31, p32, p33, p34 = calls
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("glinttray", True, "set the tell's key; set in the tell-glass; lay the side rod",
         (f"his: {R8}; hers: {R8}",), "the tell-glass in the sleeve's place"),
        ("glinttray", False, "", (IIIR,), "the run as laid"),
        ("glinttray", False, "strike the second and the fourth", (IIIR,), "struck, the gaps closed"),
        ("keycloth", True, "thread the fork", ("green to red",), "threaded on partway along")]
    PC.check_captions(DIALOGUE, "Third", calls)
    PC.check_section(DIALOGUE, "Third", calls)
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Third")] == \
        [[PC.TRAY_FRAME], [PC.TRAY_FRAME] * 2, [1.78]]

    a1, a2, a3 = DIALOGUE.anchors("Third")
    f1, f2, f3 = DIALOGUE.following("Third")
    r1, r2, r3 = DIALOGUE.following("Third", DIALOGUE.raw)
    assert "till it stops at the upright stroke" in a1
    assert f1.startswith("*[") and r1.startswith("*[Achilles, looking down at the tell-glass")  # ambient; raw: Escher
    assert "round opposite round" in a2 and f2.startswith("**Achilles:** The upright rounds, the first and the fifth")
    assert "which still runs through unbroken" in a3 and "a green one, laid off to one side" in a3   # D10
    assert f3.startswith("**Achilles:** A side road off his line")
    assert (r2, r3) == (f2, f3)
    section = DIALOGUE.section("Third")
    for phrase in ("drops it into her wooden box", "level with her piece of this round",      # D8, D9
                   "stays where it lies on the table's edge"):                              # D11
        assert phrase in section, phrase

    t31, t32, t33, c34 = (PC.picture(c) for c in calls)
    assert t31.dump == (f"sleeve off, tell-glass in; tell's key set; side rod at the side port; his [{R8}]; "
                        f"hers [{R8}]; tell's []; bell still")
    assert t31.labels == ("lamp", "key", "rod", "reflection", "tell-glass", "tell's key")   # no room for the rows'
    assert t32.dump == (f"sleeve off, tell-glass in; tell's key set; side rod at the side port; "
                        f"his [{R8} white black white white black black]; hers [{R8} white white black white black black]; "
                        "tell's [white white black white black white]; bell still")
    assert t33.dump == (f"sleeve off, tell-glass in; tell's key set; side rod at the side port; "
                        f"his [{R8} white white black black]; hers [{R8} white black black black]; "
                        "tell's [white black black white]; bell still")
    # the fixed parts: the things on the black keep their places; the glass in the sleeve's
    for t in (t31, t32, t33):
        assert {b.name: (b.x, b.depth) for b in t.shown.black} == {
            "side rod": (PC.XSLEEVE + 0.03, 0.05), "tell's key": (PC.XTELL, PC.DTELL), "his key": (PC.XHIS, 0.0),
            "her key": (PC.XHERS, 0.0), "lamp": (PC.XLAMP, 0.0), "tell-glass": (PC.XSLEEVE, 0.0)}
        assert t.shown.bell == (PC.XBELL, PC.YBELL, 0.0)
    # one pitch for the evening: in the block P3.2 | P3.3 each piece before the first struck
    # round keeps its x, and the gaps close: every piece at its row's k-th place after the strike
    v2, v3 = t32.shown, t33.shown
    assert v2.d == v3.d
    for w in ("his", "hers"):
        assert v3.rows[w][:9] == v2.rows[w][:9]
        struck = [p for i, p in enumerate(v2.rows[w]) if i not in (9, 11)]
        assert [c for c, _, _ in v3.rows[w]] == [c for c, _, _ in struck]
        assert [x for _, x, _ in v3.rows[w]] == [x for _, x, _ in v2.rows[w][:len(struck)]]
    # round opposite round: his k-th opposite her k-th; the tell's row level with her pieces
    # of the same rounds (the last ones), one line further back; its first piece level with
    # her piece of III's first round (D9)
    for v in (v2, v3):
        assert all(np.isclose(a[1], -b[1]) for a, b in zip(v.rows["his"], v.rows["hers"]))
        n, m = len(v.rows["hers"]), len(v.rows["tell"])
        assert [x for _, x, _ in v.rows["tell"]] == [x for _, x, _ in v.rows["hers"][n - m:]]
        assert all(y == PC.YTELLROW > PC.YROW for _, _, y in v.rows["tell"])
    assert v2.rows["tell"][0][1] == v2.rows["hers"][8][1]        # the ninth round, III's first

    # the run, from the dialogue's settings (said aloud after the run, l.575-577) and the
    # pieces as laid: the blackboard's ledger, round for round
    said = {}
    for who in ("Achilles", "Tortoise"):
        line = next(p for p in DIALOGUE.paragraphs("Third")
                    if p.startswith(f"**{who}:** Upright,") or p.startswith(f"**{who}:** Crosswise,"))
        said[who] = [SETTING[w] for w in re.findall(r"[Uu]pright|[Cc]rosswise", line.split(".")[0].lower())]
    rows = t32.state.rows
    run = [(said["Achilles"][k], said["Tortoise"][k], PIECE[rows["his"][8 + k]], PIECE[rows["hers"][8 + k]],
            PIECE[rows["tell"][k]]) for k in range(6)]
    assert run == ledger
    # every round's triple is possible under the table's configuration (I (x) delta)|Phi+>,
    # read at (his, hers, upright) -- the tell's key stood at the upright stroke
    probs = [table(rA, rB, PLAIN)[a, b, e] for rA, rB, a, b, e in run]
    assert np.allclose(probs, [1 / 2, 1 / 4, 1 / 8, 1 / 4, 1 / 2, 1 / 8])
    # the strike takes the rounds whose settings differed (the second and the fourth) ...
    assert [k + 1 for k, (rA, rB, *_) in enumerate(run) if rA != rB] == [2, 4]
    # ... and on each kept upright round the tell's piece is his: P(kE != kA | U, U) = 0
    kept = [r for r in run if r[0] == r[1]]
    after = [(PIECE[a], PIECE[e]) for a, e in zip(t33.state.rows["his"][8:], t33.state.rows["tell"])]
    assert [(a, e) for (rA, rB, a, b, e) in kept] == after
    assert all(a == e for (rA, rB, a, b, e), _ in zip(kept, after) if rA == PLAIN)
    p = table(PLAIN, PLAIN, PLAIN)
    assert np.isclose(sum(p[a, b, e] for a in (0, 1) for b in (0, 1) for e in (0, 1) if e != a), 0)

    # the clearance, on drawn extents: every part of the tell's key standing (its foot,
    # stand, crystal, collar and windows) is clear of every part of her key standing, and
    # likewise their reflections, each clear of the other's (strokes included)
    parts = PC.key_parts(1)
    at = lambda x, y, sc, flip: {k: (PC.placed(o, x, y, sc, flip), w) for k, (o, w) in parts.items()}
    tell, hers = at(PC.XTELL, PC.DTELL, PC.SC * PC.KS * PC.TELLSCALE, False), at(PC.XHERS, 0.0, PC.SC * PC.KS, False)
    tellr, hersr = at(PC.XTELL, PC.DTELL, PC.SC * PC.KS * PC.TELLSCALE, True), at(PC.XHERS, 0.0, PC.SC * PC.KS, True)
    gaps = {(a, b): PC.gap(tell[a], hers[b]) for a in ("frame", "yes", "no") for b in ("crystal", "collar", "face")}
    assert min(gaps.values()) > 0, gaps
    # the windows' frame's foot above her crystal's top: the Lua's own measure (a positive gap)
    frame_bottom = min(y for _, y in tell["frame"][0])
    crystal_top = max(y for _, y in hers["crystal"][0])
    assert frame_bottom - crystal_top > 0
    assert min(PC.gap(tell[a], hers[b]) for a in tell for b in hers) > 0            # foot to crystal least
    assert min(PC.gap(tellr[a], hersr[b]) for a in tellr for b in hersr) > 0
    # NOT a clearance: the tell's key's reflection, standing further back, runs down behind
    # her standing key (its window frame under her crystal). The Lua draws the tell's key
    # (with its reflection) before hers, so hers covers it: an occlusion, reported.
    lua = PC.LUA_TEXT
    assert lua.index("standing(XTELL, DTELL,") < lua.index("standing(XHERS, 0,")

    # P3.4: the fork threaded on partway along, its mirror fork just opposite, Eve's ring green
    assert c34.labels == ("Alice's", "Bob's", "mirror", "fork", "Eve's")
    PC.check_band_words(c34)
    things = {(r["x"], r["y"]): r["c"] for r in c34.cloth["things"]}
    assert things == {(PC.AX, 0): "green", (2.4, 0): "green", (PC.BX, 0): "red", (2.4, 1): "green", (4, 2): "green"}
    assert c34.dump.startswith("scene=rounds round1=green-red joined fork=true")
    assert "5:ring green 0 @4,2; tie 2-5 []; tie 4-5 []" in c34.dump and c34.dump.endswith("cord loose-5 [6.45,2 4,2]")
    # the clearance: Eve's short cord and the side cords cross only the mirror cord -- the
    # upper side cord once, passing under it (a tie is drawn under every cord)
    rings, cords = PC.cloth_drawn(c34)
    eve = next(i for i, c, X, Y, _ in rings if (X, Y) == PC.cloth_cm(c34, 4, 2))
    ties = [(a, b, pts) for kind, a, b, pts, _ in cords if kind == "tie"]
    short = next(pts for kind, a, b, pts, _ in cords if kind == "cord" and b == eve)
    others = [(a, b, pts) for kind, a, b, pts, _ in cords if kind == "cord" and b != eve]
    mirror = next(pts for a, b, pts in others if len(pts) == 4)
    for name, pts in [("upper side", ties[0][2]), ("lower side", ties[1][2]), ("Eve's short", short)]:
        hits = [(a, b, n) for a, b, o in others for n in [PC.crossings(pts, o)] if n]
        want = [(1, 3, 1)] if name == "upper side" else []
        assert hits == want, (name, hits)
        assert hits == [] or next(o for a, b, o in others if (a, b) == (1, 3) and len(o) == 4) is mirror
    assert PC.crossings(ties[0][2], ties[1][2]) == 0
    # the ties (structural): the forks are green and so is Eve's ring -- the copier at the
    # upright stroke -- and the mirror fork is the fork again: the copier is real
    assert np.allclose(DELTA, DELTA.conj())
    assert {things[(2.4, 0)], things[(2.4, 1)], things[(4, 2)]} == {"green"}
    comps = PC.cloth_links(c34.cloth)
    assert any({1, 2, 3, 4, 5, ("left", 0), ("right", 0), ("right", 2)} <= comp for comp in comps)


# --- pictures (hand-built doubled diagrams; pyzx 0.10.6 refuses ground vertices) -----

def eves_box(s, sE, forget_bob=False, forget_eve=False):
    """Alice's sender at s; the press (green copier) on q and its mirror (the green
    copier again: delta is real) on qbar; Bob's read-spider on the through-legs (b', b'bar)
    at s; Eve's read-spider on the kept legs (e, ebar) at sE.  Input x, outputs kB, kE.
    Nine edges, no scalar compensation (every dot is exact).
    forget_bob: Bob's record forgotten -- a one-legged phase-free green effect on kB
    (its boundary vertex removed; the naive contractor rejects a dangling boundary).
    forget_eve: likewise for Eve's record."""
    g = zx.Graph()
    V = g.add_vertex
    x, snd = V(VT.BOUNDARY, 1, 0), V(VT.Z, 1, 1)
    c, cb = V(VT.Z, 0.5, 2), V(VT.Z, 1.5, 2)
    rB, rE = V(VT.Z, 0, 3), V(VT.Z, 2, 3)
    hs = ET.HADAMARD if s else ET.SIMPLE
    he = ET.HADAMARD if sE else ET.SIMPLE
    g.add_edge((x, snd))                                     # x    : classical, Alice's bit
    g.add_edge((snd, c), hs); g.add_edge((snd, cb), hs)      # b, bbar  (box at cross)
    g.add_edge((c, rB), hs); g.add_edge((cb, rB), hs)        # b', b'bar to Bob (box at cross)
    g.add_edge((c, rE), he); g.add_edge((cb, rE), he)        # e, ebar to Eve
    outs = []
    for rd, forget, row in ((rB, forget_bob, 0), (rE, forget_eve, 2)):
        if forget:
            g.add_edge((rd, V(VT.Z, row, 4)))                # forgetting: a green effect
        else:
            o = V(VT.BOUNDARY, row, 5); g.add_edge((rd, o)); outs.append(o)
    g.set_inputs((x,)); g.set_outputs(tuple(outs))
    return g


def dephasing_round(s, t):
    """Sender at s; the press with Eve's leg capped onto its mirror (discarded); reader
    at t.  Built from the copier and its mirror, a cap (a phase-free two-legged green dot)
    joining their kept legs."""
    g = zx.Graph()
    V = g.add_vertex
    x, snd, c, cb, cap = V(VT.BOUNDARY, 1, 0), V(VT.Z, 1, 1), V(VT.Z, 0.5, 2), V(VT.Z, 1.5, 2), V(VT.Z, 1, 3)
    rd, y = V(VT.Z, 1, 4), V(VT.BOUNDARY, 1, 5)
    hs, ht = (ET.HADAMARD if s else ET.SIMPLE), (ET.HADAMARD if t else ET.SIMPLE)
    g.add_edge((x, snd)); g.add_edge((snd, c), hs); g.add_edge((snd, cb), hs)
    g.add_edge((c, cap)); g.add_edge((cb, cap))
    g.add_edge((c, rd), ht); g.add_edge((cb, rd), ht); g.add_edge((rd, y))
    g.set_inputs((x,)); g.set_outputs((y,))
    return g


def press_discarded():
    """The press and its mirror on (q, qbar), Eve's legs capped together: a map
    (q, qbar) -> (q, qbar)."""
    g = zx.Graph()
    V = g.add_vertex
    i1, i2 = V(VT.BOUNDARY, 0, 0), V(VT.BOUNDARY, 1, 0)
    c, cb, cap = V(VT.Z, 0, 1), V(VT.Z, 1, 1), V(VT.Z, 0.5, 2)
    o1, o2 = V(VT.BOUNDARY, 0, 3), V(VT.BOUNDARY, 1, 3)
    g.add_edge((i1, c)); g.add_edge((i2, cb)); g.add_edge((c, cap)); g.add_edge((cb, cap))
    g.add_edge((c, o1)); g.add_edge((cb, o2))
    g.set_inputs((i1, i2)); g.set_outputs((o1, o2))
    return g


def four_legged():
    """Chapter 11's four-legged green dot (row 85): sum_i |ii><ii|."""
    g = zx.Graph()
    V = g.add_vertex
    z = V(VT.Z, 0.5, 1)
    ends = [V(VT.BOUNDARY, r, c) for c in (0, 2) for r in (0, 1)]
    for e in ends:
        g.add_edge((z, e))
    g.set_inputs(tuple(ends[:2])); g.set_outputs(tuple(ends[2:]))
    return g


def noise():
    """1/2 J: a green effect on x beside a green point on y, add_power(-2) for the 1/2."""
    g = zx.Graph()
    x, e, p, y = (g.add_vertex(t, 0, c) for c, t in enumerate((VT.BOUNDARY, VT.Z, VT.Z, VT.BOUNDARY)))
    g.add_edge((x, e)); g.add_edge((p, y))
    g.set_inputs((x,)); g.set_outputs((y,))
    g.scalar.add_power(-2)
    return g


def table_picture(rA, rB, rE):
    """The III table configuration in doubled form: the twin (cup, mirror cup), Achilles's
    reader at rA on (a, abar), the tell-glass (copier and mirror copier) on (b, bbar), the
    Tortoise's reader at rB on the through-legs, the tell's reader at rE on the kept legs.
    Outputs kA, kB, kE.  Scalar: the doubled pair is 1/2 cup (x) cup, the cups
    unnormalized, everything else exact: add_power(-2) (determined here; the card had
    not run this build)."""
    g = zx.Graph()
    V = g.add_vertex
    cup, cupb = V(VT.Z, 1, 0), V(VT.Z, 2, 0)
    rdA, c, cb = V(VT.Z, 0, 2), V(VT.Z, 1, 2), V(VT.Z, 2, 2)
    rdB, rdE = V(VT.Z, 1, 4), V(VT.Z, 2, 4)
    kA, kB, kE = V(VT.BOUNDARY, 0, 5), V(VT.BOUNDARY, 1, 5), V(VT.BOUNDARY, 2, 5)
    h = lambda s: ET.HADAMARD if s else ET.SIMPLE
    g.add_edge((cup, rdA), h(rA)); g.add_edge((cupb, rdA), h(rA))
    g.add_edge((cup, c)); g.add_edge((cupb, cb))
    g.add_edge((c, rdB), h(rB)); g.add_edge((cb, rdB), h(rB))
    g.add_edge((c, rdE), h(rE)); g.add_edge((cb, rdE), h(rE))
    g.add_edge((rdA, kA)); g.add_edge((rdB, kB)); g.add_edge((rdE, kE))
    g.set_inputs(()); g.set_outputs((kA, kB, kE))
    g.scalar.add_power(-2)
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
    # Eve is a box on the wire: the value, both settings, Eve reading plain or cross
    for s in (PLAIN, CROSS):
        for sE in (PLAIN, CROSS):
            g = eves_box(s, sE)
            assert g.num_edges() == 9
            # "six lines join five dots, two more than a tree needs": two self-loops to go
            inner = [v for v in g.vertices() if g.type(v) != VT.BOUNDARY]
            inner_edges = [e for e in g.edges() if all(g.type(v) != VT.BOUNDARY for v in e)]
            assert len(inner) == 5 and len(inner_edges) == 6 == (len(inner) - 1) + 2
            M = matrix_of(g)                                  # rows (kB, kE), columns x
            for x in (0, 1):
                assert np.allclose(M[:, x].real, press_round(x, s, sE).ravel())
            assert np.allclose(M.imag, 0)
            h = g.copy(); simp.full_reduce(h)                 # DERIVATION, scalar kept
            assert same_picture(h, g)
            if s == sE == PLAIN:
                # fusion makes the whole picture one green dot with legs x, kB, kE
                assert h.num_vertices() == 4 and h.num_edges() == 3

    # forget Bob's record: at cross Alice's bit is DISCONNECTED from Eve's record (a cut
    # through empty paper); at plain they are joined
    for s in (PLAIN, CROSS):
        g = eves_box(s, PLAIN, forget_bob=True)
        M = matrix_of(g)
        assert np.allclose(M, I2 if s == PLAIN else np.ones((2, 2)) / 2)
        h = g.copy(); simp.full_reduce(h)
        assert same_picture(h, g)
        (x,), (kE,) = h.inputs(), h.outputs()
        assert connected(h, x, kE) == (s == PLAIN)
        if s == CROSS:
            assert same_picture(h, noise())

    # with Eve's copy discarded, the press is the four-legged green dot (row 85) ...
    assert same_picture(press_discarded(), four_legged())
    h = press_discarded().copy(); simp.full_reduce(h)
    assert same_picture(h, four_legged())
    # ... and a CROSS round through it is 1/2 J although the settings matched (two Hopfs):
    # Eve's look has turned a kept round into a struck one.  A plain round is untouched.
    assert same_picture(dephasing_round(CROSS, CROSS), noise())
    assert not same_picture(dephasing_round(PLAIN, PLAIN), noise())
    assert np.allclose(matrix_of(dephasing_round(PLAIN, PLAIN)), I2)
    h = dephasing_round(CROSS, CROSS).copy(); simp.full_reduce(h)
    assert same_picture(h, noise()) and not connected(h, *h.inputs(), *h.outputs())
    # the same with Eve's record forgotten after her reading (forget_eve): 1/2 J again
    assert same_picture(eves_box(CROSS, PLAIN, forget_eve=True), noise())

    # the III table configuration, all four secret-setting pairs, the tell upright
    for rA in (PLAIN, CROSS):
        for rB in (PLAIN, CROSS):
            g = table_picture(rA, rB, PLAIN)
            assert np.allclose(matrix_of(g).ravel(), table(rA, rB, PLAIN).ravel())
            h = g.copy(); simp.full_reduce(h)
            assert same_picture(h, g)
            kA, kB, kE = h.outputs()
            if rA == rB == CROSS:
                # every record falls apart from every other: 1/8 on each triple
                assert not connected(h, kA, kE) and not connected(h, kA, kB)
            if rA == rB == PLAIN:
                assert connected(h, kA, kE) and connected(h, kA, kB)


if __name__ == "__main__":
    verdict("EvesBox", blackboard, sketchbook)
