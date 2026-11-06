"""Challenge V: The Alarm  (checks; monogamy, extreme form; the master)
=====================================================================
"Then that's the whole of it. Two calls at the same setting sing the same; a listener has to look; a look at the wrong setting is a wrong note, and the bell hears it; and a pair that never sings a wrong note, upright or crosswise, has nothing to spare for anyone else."  -- Achilles

Claim: in a check round at the announced setting S, with Eve's press (the copier)
on Bob's half of the pair and Eve reading her copy at S after the announcement, the
twelve-line master The Alarm gives p(kE, w) = 1/2 [w = 0] at plain and
1/2 [kE = w] at cross, where w = kA xor kB rings the bell: at plain the bell never
rings and Eve holds the key; at cross Eve's record IS the alarm bit, for every kA.
The checks see exactly the flips: a plain check disagrees on the Bell states with a
red half-turn ring, a cross check on those with a green one (Shor and Preskill's two
projector identities).  The pass projector 1/4 (I + ZZ)(I + XX) = |Phi+><Phi+| =
1/2 cup cap has rank 1, so a state that passes both checks with certainty is
|Phi+><Phi+| (x) rho_Eve (monogamy, extreme form).
Result 12.9 (Shor and Preskill's security theorem, the 11% and 15% thresholds) is
stated and cited in the chapter, NOT proved or tested here; only the arithmetic of
the quoted numbers is checked.
"""
import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import BELL, CUP, I2, Picture, kron, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue
import props_ch12 as PC


H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
DELTA = np.outer(ket(0, 0), ket(0)) + np.outer(ket(1, 1), ket(1))   # Eve's press
PLUS, MINUS = H @ ket(0), H @ ket(1)
PLAIN, CROSS = 0, 1
PHI_P, PHI_M = BELL, (ket(0, 0) - ket(1, 1)) / np.sqrt(2)
PSI_P, PSI_M = (ket(0, 1) + ket(1, 0)) / np.sqrt(2), (ket(0, 1) - ket(1, 0)) / np.sqrt(2)


def Hp(r):
    return H if r else I2


def eff(k, s):
    return ket(k).reshape(1, 2) @ Hp(s)


def proj(v):
    return np.outer(v, v.conj())


def bell(m1, m2):
    """|B_{m1 m2}> = (Z^m1 X^m2 (x) I)|Phi+>: X nearer the bend, then Z (Chapter 10)."""
    return (kron(np.linalg.matrix_power(Z, m1) @ np.linalg.matrix_power(X, m2), I2) @ BELL).ravel()


def master_dist(S):
    """p[kA, kB, kE]: (I (x) delta)|Phi+> on (a, b', e), all three read at S."""
    ghz = (kron(I2, DELTA) @ BELL).ravel()
    return np.array([[[abs((kron(eff(a, S), eff(b, S), eff(e, S)) @ ghz)[0]) ** 2
                       for e in (0, 1)] for b in (0, 1)] for a in (0, 1)])


def alarm_dist(S):
    """p[kE, w] with w = kA xor kB."""
    p, out = master_dist(S), np.zeros((2, 2))
    for a in (0, 1):
        for b in (0, 1):
            for e in (0, 1):
                out[e, a ^ b] += p[a, b, e]
    return out


def h2(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def blackboard():
    rng = np.random.default_rng(1205)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))      # cup unnormalized
    assert np.allclose(BELL, CUP / np.sqrt(2))

    # --- Result 12.8, the Alarm, on the blackboard ---
    ghz = (kron(I2, DELTA) @ BELL).ravel()
    assert np.allclose(ghz, (ket(0, 0, 0) + ket(1, 1, 1)) / np.sqrt(2))
    assert np.allclose(alarm_dist(PLAIN), 0.5 * np.array([[1, 0], [1, 0]]))   # 1/2 [w = 0]
    assert np.allclose(alarm_dist(CROSS), 0.5 * np.eye(2))                     # 1/2 [kE = w]
    assert not np.allclose(alarm_dist(PLAIN), alarm_dist(CROSS))
    pc, pp = master_dist(CROSS), master_dist(PLAIN)
    # at cross: the four even-parity triples, 1/4 each; so kE = kA xor kB for EVERY kA ...
    for a in (0, 1):
        for b in (0, 1):
            for e in (0, 1):
                assert np.isclose(pc[a, b, e], 0.25 if (a ^ b ^ e) == 0 else 0)
    # ... and kA, kE independent fair coins: Eve learns whether she was caught, nothing else
    pAE = pc.sum(axis=1)
    assert np.allclose(pAE, 0.25)
    # at plain: 000 or 111, 1/2 each -- kA = kB = kE, w = 0, and Eve holds the key
    assert np.isclose(pp[0, 0, 0], 0.5) and np.isclose(pp[1, 1, 1], 0.5)
    # the averages over a fair setting: the bell rings on 1/4 of checked rounds; Eve holds
    # 1/2 bit per key bit (1 at plain, 0 at cross)
    ring = 0.5 * alarm_dist(PLAIN)[:, 1].sum() + 0.5 * alarm_dist(CROSS)[:, 1].sum()
    assert np.isclose(ring, 0.25)
    def info(j):
        px, pk = j.sum(axis=1), j.sum(axis=0)
        return sum(j[i, k] * np.log2(j[i, k] / (px[i] * pk[k])) for i in (0, 1) for k in (0, 1) if j[i, k] > 1e-12)
    bits = [info(master_dist(S).sum(axis=1)) for S in (PLAIN, CROSS)]
    assert np.allclose(bits, [1, 0]) and np.isclose(np.mean(bits), 0.5)

    # the ledger of V (yes = 0 = white, no = 1 = black; the bell rings when w = 1):
    #   (setting, Achilles's call, the Tortoise's, the tell's piece, the bell)
    ledger = [(CROSS, 0, 1, 1, True),      # the one rung wrong note: the tell's piece black
              (CROSS, 1, 1, 0, False),     # no bell, and the tell's piece white
              (PLAIN, 1, 1, 1, False)]     # silent again, and black like his
    for S, a, b, e, bell_rang in ledger:
        p = master_dist(S)
        assert p[a, b, e] > 0                               # possible ...
        assert np.isclose(p[a, b, e], p[a, b, :].sum())     # ... and the tell's piece forced
        assert bell_rang == (a != b)
    # "At crosswise it always goes so: her piece is black whenever the bell rings, and
    # white whenever it keeps still" -- kE = w, i.e. P(kE != w) = 0 at cross
    assert np.isclose(alarm_dist(CROSS)[0, 1] + alarm_dist(CROSS)[1, 0], 0)
    # "At upright she has the key, and the bell never hears her"
    assert np.isclose(alarm_dist(PLAIN)[:, 1].sum(), 0)
    assert np.isclose(sum(pp[k, :, k].sum() for k in (0, 1)), 1)

    # the red X(2 -> 1) spider, |+><++| + |-><--|, is 1/sqrt2 times the parity
    Xsp = np.outer(PLUS, np.kron(PLUS, PLUS)) + np.outer(MINUS, np.kron(MINUS, MINUS))
    parity = np.array([[1.0 if w == (a ^ b) else 0.0 for a in (0, 1) for b in (0, 1)] for w in (0, 1)])
    assert np.allclose(Xsp, parity / np.sqrt(2))

    # --- Result 12.6, what the checks see (Shor and Preskill's identities) ---
    assert np.allclose(bell(0, 0), PHI_P) and np.allclose(bell(1, 0), PHI_M)
    assert np.allclose(bell(0, 1), PSI_P) and np.allclose(bell(1, 1), PSI_M)
    dis_plain = proj(ket(0, 1)) + proj(ket(1, 0))
    dis_cross = proj(np.kron(PLUS, MINUS)) + proj(np.kron(MINUS, PLUS))
    assert np.allclose(proj(PSI_P) + proj(PSI_M), dis_plain)
    assert np.allclose(proj(PHI_M) + proj(PSI_M), dis_cross)
    assert not np.allclose(proj(PHI_M) + proj(PSI_M), dis_plain)      # the checks differ
    assert np.allclose(PHI_M, (np.kron(PLUS, MINUS) + np.kron(MINUS, PLUS)) / np.sqrt(2))
    assert np.allclose(PSI_M, (np.kron(MINUS, PLUS) - np.kron(PLUS, MINUS)) / np.sqrt(2))
    for m1 in (0, 1):
        for m2 in (0, 1):
            B = bell(m1, m2)
            assert np.isclose(np.vdot(B, dis_plain @ B), m2)    # plain check sees the flip
            assert np.isclose(np.vdot(B, dis_cross @ B), m1)    # cross check sees the sign

    # --- Result 12.7, monogamy in its extreme form ---
    P_plain = proj(ket(0, 0)) + proj(ket(1, 1))
    P_cross = proj(np.kron(PLUS, PLUS)) + proj(np.kron(MINUS, MINUS))
    ZZ, XX, I4 = kron(Z, Z), kron(X, X), np.eye(4)
    assert np.allclose(P_plain, (I4 + ZZ) / 2) and np.allclose(P_cross, (I4 + XX) / 2)
    assert np.allclose(ZZ @ XX, XX @ ZZ)                       # the two checks commute
    assert np.allclose(kron(Z @ X, Z @ X), kron(X @ Z, X @ Z))
    P = (I4 + ZZ) @ (I4 + XX) / 4
    assert np.allclose(P, proj(PHI_P)) and np.allclose(P, 0.5 * np.outer(CUP, CUP))
    assert np.linalg.matrix_rank(P) == 1
    assert np.allclose(P_plain @ PHI_M, PHI_M) and np.allclose(P_cross @ PSI_P, PSI_P)
    assert np.allclose(np.kron(PLUS, PLUS) + np.kron(MINUS, MINUS), np.sqrt(2) * PHI_P)
    assert np.allclose(np.kron(PLUS, PLUS) - np.kron(MINUS, MINUS), np.sqrt(2) * PSI_P)
    # random pure states of A, B and a two-qubit Eve: those that pass both checks (the
    # projected ones) are Phi+ (x) something -- rank 1 across the cut AB | Eve
    dE = 4
    for _ in range(20):
        psi = random_state(rng, 4 * dE)
        passed = np.kron(P, np.eye(dE)) @ psi
        passed = passed / np.linalg.norm(passed)
        for Pc in (P_plain, P_cross):                          # passes each check w.p. 1
            assert np.isclose(np.vdot(passed, np.kron(Pc, np.eye(dE)) @ passed).real, 1)
        M = passed.reshape(4, dE)
        assert np.linalg.matrix_rank(M, tol=1e-9) == 1
        eve = (np.kron(PHI_P.conj().reshape(1, 4), np.eye(dE)) @ passed).ravel()
        assert np.allclose(passed, np.kron(PHI_P, eve))
        assert np.linalg.matrix_rank(psi.reshape(4, dE), tol=1e-9) > 1   # the unprojected: not
    # mixed states too: rho passing both checks is |Phi+><Phi+| (x) rho_Eve
    for _ in range(5):
        vs = [random_state(rng, 4 * dE) for _ in range(3)]
        ps = rng.random(3); ps /= ps.sum()
        rho = sum(p * proj(v) for p, v in zip(ps, vs))
        PP = np.kron(P, np.eye(dE))
        rp = PP @ rho @ PP; rp /= np.trace(rp)
        rho_eve = np.einsum('ijik->jk', rp.reshape(4, dE, 4, dE))
        assert np.allclose(rp, np.kron(proj(PHI_P), rho_eve))
    # why BOTH checks: each alone can be fooled.  (I (x) delta)Phi+ passes the plain check
    # every time and gives Eve the key; its cross twin (H (x) H (x) H) of it passes the
    # cross check every time and gives her the cross key.  Neither is a product.
    for st, Pok, Pbad in ((ghz, P_plain, P_cross), (kron(H, H, H) @ ghz, P_cross, P_plain)):
        st = np.asarray(st).ravel()
        assert np.isclose(np.vdot(st, np.kron(Pok, I2) @ st).real, 1)
        assert np.isclose(np.vdot(st, np.kron(Pbad, I2) @ st).real, 0.5)
        assert np.linalg.matrix_rank(st.reshape(4, 2)) == 2

    # --- arithmetic of the numbers section 12.8 quotes (NOT the theorem, which is cited) ---
    lo, hi = 0.01, 0.4                                          # 1 - 2 h(D) reaches 0 near 11%
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 1 - 2 * h2(mid) > 0 else (lo, mid)
    assert round(100 * lo) == 11
    assert round(100 * 0.5 * (1 - 1 / np.sqrt(2))) == 15       # D0 = 1/2 (1 - 1/sqrt2)
    assert np.isclose(0.6 ** 2 + 0.4 ** 2, 0.52)                # privacy amplification's 52%
    pictures(ledger)


# --- the prop pictures (dialog.md: \glinttray, \keycloth; tools/props_ch12.py) ----------
# The chapter's whole run of pictures is checked here too: the order, the keyed rule, the
# places, the fixed parts and the clearances over all nineteen, every fault site of the
# transcription, and the drift guard that ties the transcription to pdf/props/ch12.lua.

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = ("Then that's the whole of it. Two calls at the same setting sing the same; a listener has "
         "to look; a look at the wrong setting is a wrong note, and the bell hears it; and a pair that "
         "never sings a wrong note, upright or crosswise, has nothing to spare for anyone else.")
R8 = "white black white black white black white black"
V0 = (f"the tell-glass lifted out; his: {R8} white white black black; hers: {R8} white black black black; "
      "tell's: white black black white")
PIECE = {"white": 0, "black": 1}                 # yes = white = 0, no = black = 1
FULLRUN = "set back the tell-glass; lay white black black; lay black black white; lay black black black"


def pictures(ledger):
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    calls = DIALOGUE.check_layout("Fifth", sizes=[1, 1, 1, 1])
    p51, p52, p53, p54 = calls
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("glinttray", False, "set back the tell-glass; ring the bell", (V0,), "lifted by its handle and rung"),
        ("glinttray", False, "set back the tell-glass; lay white black black", (V0,), "his white, hers black"),
        ("keycloth", True, "", ("the pair's bend, red then green",), "red close in, green further out"),
        ("keycloth", False, "slide off red round the turn, slide off green", ("the pair's bend, red then green",),
         "bare, the turn where it was")]
    PC.check_captions(DIALOGUE, "Fifth", calls)
    PC.check_section(DIALOGUE, "Fifth", calls)
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Fifth")] == \
        [[PC.TRAY_FRAME], [PC.TRAY_FRAME], [1.24], [1.24]]

    anchors = DIALOGUE.anchors("Fifth")
    nexts = DIALOGUE.following("Fifth")
    raws = DIALOGUE.following("Fifth", DIALOGUE.raw)
    assert "She sets it back where it stood." in anchors[0] and nexts[0].startswith("*[He lays white; she lays black;")
    assert "she lays the tell's piece: black." in anchors[1]
    assert nexts[1].startswith("*[Across the lane a dog") and raws[1].startswith("**Achilles:** And the tell's piece is black.")
    assert "each bead sitting halfway round" in anchors[2]                                  # D14
    assert "the straightened cords and their rings into her pouch" in anchors[2]
    assert nexts[2].startswith("**Tortoise:** An upright check that never disagrees")
    assert "The bend lies bare on the cloth." in anchors[3] and "the upper one Alice's and the lower one Bob's" in anchors[3]
    assert nexts[3].startswith("**Tortoise:** Let Bob draw empty paper")
    assert [raws[0], raws[2], raws[3]] == [nexts[0], nexts[2], nexts[3]]
    section = DIALOGUE.section("Fifth")
    for phrase in ("takes out the tell-glass and sets it in its place",                         # D13
                   "round it, and out along the lower leg", "by that leg's own end",          # D15
                   "between the bend and the rest of the cloth", "rolls her drawing-cloth up"):   # D17, D18
        assert phrase in section, phrase
    text = DIALOGUE.read(DIALOGUE.dialog)
    assert not DIALOGUE.call_re.search(text[text.index(p54.text) + len(p54.text):])    # none after P5.4

    t51, t52, c53, c54 = (PC.picture(c) for c in calls)
    his12, hers12 = f"{R8} white white black black", f"{R8} white black black black"
    assert t51.dump == (f"sleeve off, tell-glass in; tell's key set; side rod at the side port; his [{his12}]; "
                        f"hers [{hers12}]; tell's [white black black white]; bell rung")
    assert t52.dump == (f"sleeve off, tell-glass in; tell's key set; side rod at the side port; his [{his12} white]; "
                        f"hers [{hers12} black]; tell's [white black black white black]; bell still")
    assert t51.labels == t52.labels == ()
    # P5.1: the rows as P3.3 left them; only the bell has moved: lifted and swinging
    p33 = DIALOGUE.blocks("Third")[1][1]
    t33 = PC.picture(p33)
    assert t51.dump == t33.dump.replace("bell still", "bell rung")
    assert t51.shown.black == t33.shown.black and t51.shown.rows == t33.shown.rows
    assert t33.shown.bell == (PC.XBELL, PC.YBELL, 0.0) and t51.shown.bell != t33.shown.bell
    assert t51.shown.bell[2] != 0 and t51.shown.bell[1] > PC.YBELL
    # P5.2: the bell back on its corner; one piece more in each row, the others unmoved
    assert t52.shown.bell == t33.shown.bell and t52.shown.d == t51.shown.d
    for w in ("his", "hers"):
        assert t52.shown.rows[w][:-1] == t51.shown.rows[w]
    assert t52.shown.rows["tell"][:-1] == t51.shown.rows["tell"]
    n, m = len(t52.shown.rows["hers"]), len(t52.shown.rows["tell"])
    assert [x for _, x, _ in t52.shown.rows["tell"]] == [x for _, x, _ in t52.shown.rows["hers"][n - m:]]

    # the ties (structural): the rounds of V, read off the dialogue -- the calls spoken
    # (Yes. / No.) and the pieces laid -- are the blackboard's ledger; each is possible under
    # the master (I (x) delta)|Phi+> at the agreed setting, 1/4, 1/4, 1/2; at cross the
    # tell's piece is black exactly when the bell rings, kE = kA xor kB
    paras = DIALOGUE.paragraphs("Fifth")
    said = [p.split(":** ")[1] for p in paras if p in ("**Achilles:** Yes.", "**Achilles:** No.",
                                                       "**Tortoise:** Yes.", "**Tortoise:** No.")]
    assert len(said) == 6
    calls_said = [(int(said[2 * k] == "No."), int(said[2 * k + 1] == "No.")) for k in range(3)]
    for phrase in ("**Achilles:** Crosswise, then — all of them.", "**Tortoise:** Another, crosswise still.",
                   "**Tortoise:** Now upright, all of them, agreed aloud."):
        assert any(p.startswith(phrase) for p in paras), phrase
    settings = [CROSS, CROSS, PLAIN]
    full = PC.glinttray(FULLRUN, V0)
    assert PC.faults(PC.tray_state, FULLRUN, V0) == []
    laid = [tuple(PIECE[r[12 + k]] for r in (full.state.rows["his"], full.state.rows["hers"])) +
            (PIECE[full.state.rows["tell"][4 + k]],) for k in range(3)]
    assert laid[0] == tuple(PIECE[w] for w in "white black black".split())       # P5.2's lay
    assert [(a, b) for a, b, _ in laid] == calls_said
    assert [(S, a, b, e) for S, (a, b, e) in zip(settings, laid)] == [l[:4] for l in ledger]
    assert np.allclose([master_dist(S)[a, b, e] for S, (a, b, e) in zip(settings, laid)], [1 / 4, 1 / 4, 1 / 2])
    pc = master_dist(CROSS)
    assert np.isclose(sum(pc[a, b, e] for a in (0, 1) for b in (0, 1) for e in (0, 1) if e != a ^ b), 0)
    assert [bool(a ^ b) for a, b, _ in laid] == [l[4] for l in ledger] == [True, False, False]   # rung once
    assert (t51.state.rung, t52.state.rung) == (True, False)

    # P5.3 | P5.4: the pair's bend, red close in, green further out; slid off, bare
    turn = "cord loose-loose [6.45,0 -0.42,0 -0.42,1 6.45,1]"
    assert c53.dump == ("scene=pair rings from the turn=[red green] | cols=7 rows=2 y=0..1 near=false "
                        f"folded=false; 1:ring red 12 @2,0; 2:ring green 12 @3.5,0; {turn}")
    assert c54.dump == f"scene=pair rings from the turn=[] | cols=7 rows=2 y=0..1 near=false folded=false; {turn}"
    assert c53.labels == ("bend", "ring", "bead") and c54.labels == ()
    assert c53.frame == c54.frame == 1.24
    PC.check_band_words(c53)
    # L1's route, keyed on x (the dump renumbers the staying ring): red goes round the
    # turn and the green ring keeps its place; then green goes off its own end
    mid = PC.keycloth("slide off red round the turn", "the pair's bend, red then green")
    at = lambda pic: {r["x"]: (r["c"], r["y"]) for r in pic.cloth["things"]}
    assert at(c53) == {PC.PAIRX[0]: ("red", 0), PC.PAIRX[1]: ("green", 0)}
    assert at(mid) == {PC.PAIRX[1]: ("green", 0)} and at(c54) == {}
    assert PC.messages(PC.cloth_state, "slide off red", "the pair's bend, red then green") == [
        "keycloth: the green ring lies between the red ring and the end: rings on a cord cannot pass each other"]
    assert c53.cloth["cords"] == mid.cloth["cords"] == c54.cloth["cords"]      # the bend has not moved
    # the claw's line just beyond the turn (D17): between the linen's left edge and the
    # turn's drawn ink there is room, so a line from top to bottom there crosses nothing
    g = PC.C._geometry(c54.cloth, c54.options)
    _, cords = PC.cloth_drawn(c54)
    (kind, _, _, pts, reach), = cords
    room = (min(x for x, _ in pts) - reach) - (PC.C._X(g, 0) - g["EX"])
    assert room > 0
    # the ties (structural): red nearer the turn is |B_m1m2> ~ (Z^m1 X^m2 (x) I) cup, X
    # nearer the cup; the other order only by a sign; carried round the turn the red ring
    # arrives as its transpose, X^T = X, so the state is unchanged (L1)
    for m1 in (0, 1):
        for m2 in (0, 1):
            Zm, Xm = np.linalg.matrix_power(Z, m1), np.linalg.matrix_power(X, m2)
            assert np.allclose(bell(m1, m2), kron(Zm @ Xm, I2) @ CUP / np.sqrt(2))
    assert np.allclose(kron(X @ Z, I2) @ CUP, -kron(Z @ X, I2) @ CUP)
    assert not np.allclose(kron(X @ Z, I2) @ CUP, kron(Z @ X, I2) @ CUP)
    assert np.allclose(X.T, X) and np.allclose(Z.T, Z)
    assert np.allclose(kron(X, I2) @ CUP, kron(I2, X.T) @ CUP)
    assert np.allclose(kron(Z, X) @ CUP, kron(Z @ X, I2) @ CUP)
    # the bare bend is the pass projector, 1/2 cup cap, of rank 1
    ZZ, XX, I4 = kron(Z, Z), kron(X, X), np.eye(4)
    Pp = (I4 + ZZ) @ (I4 + XX) / 4
    assert np.allclose(Pp, 0.5 * np.outer(CUP, CUP)) and np.linalg.matrix_rank(Pp) == 1

    whole()


# -- the whole run of pictures, over the five challenges -----------------------------------------
def clear(A, B):
    """PC.gap(A, B), quickly: two parts whose bounding boxes (strokes added) lie more than
    0.05 cm apart are clear by at least that distance, a lower bound on the gap."""
    (pa, wa), (pb, wb) = A, B
    box = lambda p: (min(x for x, _ in p), min(y for _, y in p), max(x for x, _ in p), max(y for _, y in p))
    (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = box(pa), box(pb)
    d = np.hypot(max(bx0 - ax1, ax0 - bx1, 0), max(by0 - ay1, ay0 - by1, 0)) - wa - wb
    return d if d > 0.05 else PC.gap(A, B)


def whole():
    D = DIALOGUE
    assert D.check_whole(PC.ORDER) == 19
    text = D.read(D.dialog)
    first = text.index("## First Challenge")
    assert not D.call_re.search(text[:first])                          # none before the First
    D.keyed_first(PC.ORDER, kind=PC.keyed_kind)
    calls = [(o, c) for o in PC.ORDER for b in D.blocks(o) for c in b]
    names = [f"P{i + 1}.{k + 1}" for i, o in enumerate(PC.ORDER) for k in range(sum(map(len, D.blocks(o))))]
    assert [n for n, (o, c) in zip(names, calls) if c.keyed] == ["P1.1", "P1.2", "P3.1", "P3.4", "P4.1", "P4.3", "P5.3"]
    for o, c in calls:                                                  # the places, globally
        assert PC.place_ok(o, c), (o, c.text)
        PC.picture(c)                                                   # and no fault
    # G11: no tray in the Second before the block after the screens came down
    second = [c.macro for b in D.blocks("Second") for c in b]
    assert second == ["glinttray", "keycloth", "keycloth"]

    # the fixed parts, over every \glinttray of the dialogue, in order
    trays = [PC.picture(c) for o, c in calls if c.macro == "glinttray"]
    assert len(trays) == 7
    home = {"side rod": (PC.XSLEEVE + 0.03, 0.05), "tell's key": (PC.XTELL, PC.DTELL), "his key": (PC.XHIS, 0.0),
            "her key": (PC.XHERS, 0.0), "lamp": (PC.XLAMP, 0.0), "tell-glass": (PC.XSLEEVE, 0.0),
            "sleeve": (PC.XSLEEVE, 0.0)}
    for t in trays:
        assert all((b.x, b.depth) == home[b.name] for b in t.shown.black)
        dx, dy, rot = PC.BELLRUNG
        assert t.shown.bell == ((PC.XBELL + dx, PC.YBELL + dy, rot) if t.state.rung else (PC.XBELL, PC.YBELL, 0.0))
    # one pitch for the evening: each row's k-th piece keeps its x from call to call; only a
    # strike closes a gap (P3.3), and there every piece keeps the x of the k-th place
    assert {t.shown.d for t in trays} == {PC.PITCH} and all(t.shown.rx == 0.45 * PC.PITCH for t in trays)
    struck = {4}                                                        # P3.3 closes the gaps
    for i in range(1, len(trays)):
        for w in ("his", "hers", "tell"):
            a, b = trays[i - 1].shown.rows[w], trays[i].shown.rows[w]
            k = min(len(a), len(b))
            if i not in struck or w == "tell":
                assert [x for _, x, _ in a[:k]] == [x for _, x, _ in b[:k]], (i, w)
            else:                                                       # the k-th place's x
                assert [x for _, x, _ in b] == [x for _, x, _ in a[:len(b)]], (i, w)
    for t in trays:                                                     # the tell's row: level
        v = t.shown                                                     # with her last pieces
        n, m = len(v.rows["hers"]), len(v.rows["tell"])
        assert [x for _, x, _ in v.rows["tell"]] == [x for _, x, _ in v.rows["hers"][n - m:]]

    # the clearances, on drawn extents (strokes and shadows included), over every tray
    # state of the dialogue and the full run of V (the rows at fifteen)
    full = PC.glinttray(FULLRUN, V0)
    assert len(full.state.rows["his"]) == len(full.state.rows["hers"]) == PC.MAXROW == 15
    assert len(full.state.rows["tell"]) == 7
    states = trays + [full]
    bell_gap = rows_gap = None
    B = PC.bell_parts()
    for t in states:
        v = t.shown
        bx, by, rot = v.bell
        bell = {k: (PC.placed(o, bx, by, 1.0, rot=rot), w) for k, (o, w) in B.items()}
        for w in ("his", "hers", "tell"):
            for c, x, y in v.rows[w]:
                P = PC.piece_parts(x, y, v.rx)
                g = min(clear(bell[a], P[b]) for a in bell for b in P)
                bell_gap = g if bell_gap is None else min(bell_gap, g)
        if v.rows["his"] and v.rows["hers"]:
            (_, xh, yh), (_, xs, ys) = v.rows["his"][-1], v.rows["hers"][-1]
            Ph, Ps = PC.piece_parts(xh, yh, v.rx), PC.piece_parts(xs, ys, v.rx)
            g = min(clear(Ph[a], Ps[b]) for a in Ph for b in Ps)
            rows_gap = g if rows_gap is None else min(rows_gap, g)
    assert bell_gap > 0 and rows_gap > 0
    # the Prop Artist's figures, from the Lua's places (centre lines, before strokes and
    # shadows): the still bell's foot 0.0376 clear of her first piece's body, and the rows
    # at fifteen 0.04 apart; on the drawn extents above the least gaps are smaller but positive
    rx = PC.tray_geometry(trays[1].state)["rx"]
    assert round((PC.XBELL - 0.06) - (PC.XROW0 + rx), 4) >= 0.0376          # values to beat
    v = full.shown
    assert round(v.rows["hers"][-1][1] - v.rows["his"][-1][1] - 2 * v.rx, 2) >= 0.04

    duet()
    check_faults()
    check_lua()


def duet():
    """The duet and the carrier after the revision (the card's section 1c one-liners)."""
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        t = re.sub(r"<!--.*?-->", "", DIALOGUE.read(path), flags=re.S)
        L = [(s, re.sub(r"\*\[.*?\]\*", "", l)) for s, l in re.findall(r"^\*\*([^*]+):\*\*(.*)$", t, re.M)]
        assert sorted({s for s, _ in L}) == ["Achilles", "Tortoise"]
        eve = [l for _, l in L if re.search(
            r"\bEve(?:\x27s|’s)?\s+(?:\w+\s+){0,2}(?:says|said|say|wrote|writes|write|asks|asked|answers|"
            r"answered|tells|told|calls|called|sings|sang|replies|replied|whispers|whispered)\b", l)]
        assert eve == []
        assert len([s for s, l in L if l.strip() in ("Yes.", "No.")]) == 44
        assert "".join(m[-1] for m in re.findall(r"Let Bob draw [A-Za-z]", t)).upper() == "COECKE"
        assert DIALOGUE.read(path).count("Let Bob draw") == 6
    text = DIALOGUE.read(DIALOGUE.dialog)
    for phrase in ("along the table's far side", "towards the window", "her string and rings in her pouch",
                   "the table's near corner at the dresser end"):     # D1, the card's table
        assert phrase in text, phrase


# -- every fault site of the transcription, each tripped by its own call ---------------------------
R8S = f"his: {R8}; hers: {R8}"
FULL = "with the tell-glass; his: white; hers: white; tell's: white"
W16 = " ".join(["white"] * 16)
W15 = "white " * 15
FAULT_CASES = [
    ("tray-then", PC.tray_state, ("", "his: white then black then white; hers: white then black then white")),
    ("tray-colour", PC.tray_state, ("", "his: grey white; hers: white")),
    ("tray-clause", PC.tray_state, ("", "yours: white")),
    ("tray-words", PC.tray_state, ("", "all set")),
    ("tray-run-uneven", PC.tray_state, ("", "his: white then black white; hers: white black then white")),
    ("tray-uneven", PC.tray_state, ("", "his: white; hers: white black")),
    ("tray-tell-long", PC.tray_state, ("", "with the tell-glass; his: white; hers: white; tell's: white black")),
    ("tray-tell-early", PC.tray_state, ("", "his: white; hers: white; tell's: white")),
    ("tray-tell-then", PC.tray_state, ("", "with the tell-glass; his: white then white; hers: white then white; "
                                           "tell's: then white white")),
    ("tray-row-long", PC.tray_state, ("", f"his: {W16}; hers: {W16}")),
    ("tray-lay-colour", PC.tray_state, ("lay white grey", "as set out")),
    ("tray-lay-count", PC.tray_state, ("lay white white", FULL)),
    ("tray-ordinal", PC.tray_state, ("strike the umpteenth", R8S)),
    ("tray-strike-twice", PC.tray_state, ("strike the second and the second", R8S)),
    ("tray-strike-none", PC.tray_state, ("strike", R8S)),
    ("tray-strike-range", PC.tray_state, ("strike the ninth", R8S)),
    ("tray-flash-word", PC.tray_state, ("flash yes maybe", "as set out")),
    ("tray-flash-count", PC.tray_state, ("flash yes no", FULL)),
    ("tray-key-twice", PC.tray_state, ("set the tell's key", FULL)),
    ("tray-glass-nokey", PC.tray_state, ("set in the tell-glass", "as set out")),
    ("tray-glass-in", PC.tray_state, ("set in the tell-glass", FULL)),
    ("tray-rod-early", PC.tray_state, ("set the tell's key; lay the side rod", "as set out")),
    ("tray-rod-twice", PC.tray_state, ("lay the side rod", FULL)),
    ("tray-lift-none", PC.tray_state, ("lift out the tell-glass", "the tell-glass lifted out")),
    ("tray-back-state", PC.tray_state, ("set back the tell-glass", FULL)),
    ("tray-op", PC.tray_state, ("polish the tray", "as set out")),
    ("tray-lay-long", PC.tray_state, ("lay white white", f"his: {W15}; hers: {W15}")),
    ("key-stop", PC.key_stop, ("sideways",)),
    ("bend-colours", PC.cloth_state, ("", "the lamp's bend, green")),
    ("pair-colour", PC.cloth_state, ("", "the pair's bend, blue")),
    ("pair-count", PC.cloth_state, ("", "the pair's bend, red then green then red then green")),
    ("cloth-scene", PC.cloth_state, ("", "a tangle")),
    ("cloth-colour", PC.cloth_state, ("", "green to blue")),
    ("cloth-rounds", PC.cloth_state, ("", "green to green, red to red, green to red")),
    ("slide-mixed", PC.cloth_state, ("slide together", "green to red")),
    ("slide-twice", PC.cloth_state, ("slide together, slide together", "green to green")),
    ("slide-fork", PC.cloth_state, ("thread the fork, slide together", "green to green")),
    ("lift-early", PC.cloth_state, ("lift off", "green to green")),
    ("lift-twice", PC.cloth_state, ("slide together, lift off, lift off", "green to green")),
    ("apart-same", PC.cloth_state, ("draw apart", "green to green")),
    ("apart-twice", PC.cloth_state, ("draw apart, draw apart", "green to red")),
    ("apart-fork", PC.cloth_state, ("thread the fork, draw apart", "green to red")),
    ("fork-rounds", PC.cloth_state, ("thread the fork", "green to green, red to red")),
    ("fork-twice", PC.cloth_state, ("thread the fork, thread the fork", "green to green")),
    ("fork-late", PC.cloth_state, ("slide together, thread the fork", "green to green")),
    ("pull-twice", PC.cloth_state, ("pull straight, pull straight", "the lamp's bend")),
    ("slide-off-none", PC.cloth_state, ("slide off green", "the pair's bend, red")),
    ("slide-off-order", PC.cloth_state, ("slide off red", "the pair's bend, red then green")),
    ("slide-round-none", PC.cloth_state, ("slide off green round the turn", "the pair's bend, red")),
    ("slide-round-order", PC.cloth_state, ("slide off green round the turn", "the pair's bend, red then green")),
    ("cloth-op", PC.cloth_state, ("fold", "green to green")),
]
# calls the macros accept (the Prop Artist's test sheet), each raising nothing
GOOD_TRAY = [("flash yes yes", "as set out"), ("lay white white", "as set out; his: none; hers: none"),
             ("strike the sixth, seventh and eighth",
              "his: white black white black white white white black; hers: white black white black white black white white"),
             ("set the tell's key; set in the tell-glass; lay the side rod; flash yes no yes; lay white white white", R8S),
             ("set back the tell-glass; flash yes no no; ring the bell", V0), (FULLRUN, V0)]
GOOD_KEY = ["upright", "the stroke", "at the upright stroke", "crosswise", "the cross", "twist", "the curl", "at the curl"]
GOOD_CLOTH = [("", "green to green"), ("slide together, lift off", "green to green"), ("draw apart", "green to red"),
              ("slide together, lift off", "green to green, red to red"), ("thread the fork", "green to red"),
              ("pull straight", "the lamp's bend"), ("", "the lamp's bend, green and green"),
              ("slide off green, slide off red", "the pair's bend, red then green"),
              ("slide off red round the turn, slide off green", "the pair's bend, red then green"),
              ("slide off red round the turn, slide off green round the turn", "the pair's bend, red then green"),
              ("slide off green, slide off red round the turn", "the pair's bend, red then green")]


def check_faults():
    seen = {}
    for site, fn, args in FAULT_CASES:
        assert set(PC.faults(fn, *args)) == {site}, site
        msgs = set(PC.messages(fn, *args))
        assert len(msgs) == 1 and not (msgs & set(seen)), site           # one message, its own
        seen[msgs.pop()] = site
    assert sorted(s for s, _, _ in FAULT_CASES) == sorted(PC.ERROR_SITES) and len(PC.ERROR_SITES) == 51
    for ops, st in GOOD_TRAY:
        assert PC.faults(PC.tray_state, ops, st) == [], ops
    for s in GOOD_KEY:
        assert PC.faults(PC.key_stop, s) == [], s
    for ops, sc in GOOD_CLOTH:
        assert PC.faults(PC.keycloth, ops, sc) == [], (ops, sc)


# -- the drift guard: the transcription against pdf/props/ch12.lua as it stands ---------------------
def check_lua():
    lua = re.sub(r"--[^\n]*", "", PC.LUA_TEXT)
    # the fault sites: 27 of the tray's (terr), 1 of the key's, 23 of the cloth's (cerr)
    assert len(re.findall(r"\bterr\(", lua)) - 1 == 27 and len(re.findall(r"\bcerr\(", lua)) - 1 == 23
    assert len(re.findall(r"\btex\.error\(", lua)) == 3                 # terr, cerr, and the key's
    assert 27 + 1 + 23 == len(PC.ERROR_SITES)
    # the places and sizes the drawn shapes rely on, where the Lua writes them inline
    for pattern in (r"standing\(XTELL, DTELL, function\(\) key\(1, lit\.tell\) end, SC \* KS \* "
                    + re.escape(str(PC.TELLSCALE)) + r"\)",
                    r"shift=\{\" \.\. pt\(XBELL - " + re.escape(str(-PC.BELLRUNG[0])) + r", YBELL \+ "
                    + re.escape(str(PC.BELLRUNG[1])) + r"\) \.\. \"\}, rotate=" + str(int(PC.BELLRUNG[2])) + r"\]",
                    r"local top, bot = " + re.escape(str(PC.KEYTOP)) + ", " + re.escape(str(PC.KEYBOT)) + r"\n",
                    r"local SIDE0, SIDED0 = XSLEEVE \+ 0\.03, 0\.05",
                    r"local cy, cr = ROD / KS, 0\.055",
                    r"pt\(0, 0\.008\) \.\. \" ellipse \(0\.045 and 0\.011\);",
                    r"local ry, th = rx \* 0\.62, rx \* 0\.30",
                    r"pt\(0, 0\.006\) \.\. \" ellipse \(0\.06 and 0\.01\);"):
        assert re.search(pattern, lua), pattern
    assert "return { d = PITCH, rx = 0.45 * PITCH }" in lua and "maxlen" not in lua
    # the band words of \keycloth* (props_ch12.ownlabel): BANDOUT further out than the band's middle
    flat = " ".join(lua.split())
    assert ('if it.side == "top" then y = CLOTHBAND + BANDOUT else y = 2 * CLOTHLB - '
            'props.cloth.frame_height(S, props.cloth.options(o)) - CLOTHBAND - BANDOUT end') in flat


# --- pictures (hand-built doubled diagrams; pyzx 0.10.6 refuses ground vertices) -----

def the_alarm(S, bent=False):
    """The Alarm, in unfolded doubled form (chapter section 12.7).
    Vertices: the cup (a, b) and the mirror cup (abar, bbar); Alice's reader rA; the
    copier c on b and the mirror copier cbar on bbar (delta is real, so the mirror copier
    is the green fork again); Bob's reader rB on the through-legs; Eve's reader rE on the
    kept legs; the red parity dot par on the records kA, kB; outputs kE, w.
    q = a Hadamard edge at cross (one line with a box on it).
    Scalar: the doubled pair is 1/2 cup (x) cup and the cups are unnormalized (x 1/2);
    the red X(2 -> 1) is 1/sqrt2 x parity, so the parity itself is sqrt2 x the dot
    (x sqrt2): add_power(-1).
    bent=True: the prepare-and-measure form -- the cup and Alice's reader fused into her
    sender, fed a fair coin: ONE green dot with legs kA (to the parity), q and qbar
    (1/2 for the coin, sqrt2 for the parity: add_power(-1) again)."""
    g = zx.Graph()
    V = g.add_vertex
    q = ET.HADAMARD if S == CROSS else ET.SIMPLE
    c, cb = V(VT.Z, 2, 2), V(VT.Z, 4, 2)
    rB, rE = V(VT.Z, 2, 4), V(VT.Z, 4, 4)
    par = V(VT.X, 1, 5)
    kE, w = V(VT.BOUNDARY, 4, 6), V(VT.BOUNDARY, 1, 6)
    if bent:
        snd = V(VT.Z, 1, 0)
        g.add_edge((snd, c), q); g.add_edge((snd, cb), q)    # the sender's doubled legs
        g.add_edge((snd, par))                               # kA
    else:
        cup, cupb = V(VT.Z, 1, 0), V(VT.Z, 3, 0)
        rA = V(VT.Z, 0, 2)
        g.add_edge((cup, rA), q)       # 1  a     : Alice's half, ket copy     [H at cross]
        g.add_edge((cupb, rA), q)      # 2  abar  : its mirror copy            [H at cross]
        g.add_edge((cup, c))           # 3  b     : Bob's half -> Eve's copier
        g.add_edge((cupb, cb))         # 4  bbar  : its mirror -> mirror copier
        g.add_edge((rA, par))          # 9  kA    : CLASSICAL, Alice's record -> parity
    g.add_edge((c, rB), q)             # 5  b'    : through-leg to Bob         [H at cross]
    g.add_edge((cb, rB), q)            # 6  b'bar : its mirror                 [H at cross]
    g.add_edge((c, rE), q)             # 7  e     : Eve's kept leg             [H at cross]
    g.add_edge((cb, rE), q)            # 8  ebar  : its mirror                 [H at cross]
    g.add_edge((rB, par))              # 10 kB    : CLASSICAL, Bob's record -> parity
    g.add_edge((rE, kE))               # 11 kE    : CLASSICAL, Eve's record
    g.add_edge((par, w))               # 12 w     : CLASSICAL, the alarm (the bell)
    g.set_inputs(()); g.set_outputs((kE, w))
    g.scalar.add_power(-1)
    return g


def expected(S):
    """plain: 1/2 [w = 0] = 2^(-3/2) x (green point on kE, red point on w), since a green
    point is |0>+|1> and a red point sqrt2|0>: add_power(-3).
    cross: 1/2 [kE = w] = 1/2 x a green bend on (kE, w): add_power(-2)."""
    g = zx.Graph()
    kE, w = g.add_vertex(VT.BOUNDARY, 4, 2), g.add_vertex(VT.BOUNDARY, 1, 2)
    if S == PLAIN:
        z, x = g.add_vertex(VT.Z, 4, 1), g.add_vertex(VT.X, 1, 1)
        g.add_edge((z, kE)); g.add_edge((x, w))
        g.scalar.add_power(-3)
    else:
        z = g.add_vertex(VT.Z, 2.5, 1)
        g.add_edge((z, kE)); g.add_edge((z, w))
        g.scalar.add_power(-2)
    g.set_inputs(()); g.set_outputs((kE, w))
    return g


def pass_projector():
    """P = 1/2 cap then cup, on single wires (not doubled): add_power(-2) for the 1/2."""
    g = zx.Graph()
    V = g.add_vertex
    i0, i1, cap, cup, o0, o1 = V(VT.BOUNDARY, 0, 0), V(VT.BOUNDARY, 1, 0), V(VT.Z, 0.5, 1), \
        V(VT.Z, 0.5, 2), V(VT.BOUNDARY, 0, 3), V(VT.BOUNDARY, 1, 3)
    g.add_edge((i0, cap)); g.add_edge((i1, cap)); g.add_edge((cup, o0)); g.add_edge((cup, o1))
    g.set_inputs((i0, i1)); g.set_outputs((o0, o1))
    g.scalar.add_power(-2)
    return g


def ringed_cup(m1, m2):
    """(1/sqrt2) cup with, outward from the bend on its top leg, a red half-turn ring
    m2*pi and then a green one m1*pi: |B_{m1 m2}> (Chapter 10, row 73).  add_power(-1)."""
    p = Picture()
    b = p.bend(0.5, 1)
    p.wire(b, p.x(0, 2, m2), p.z(0, 3, m1), p.out(0))
    p.wire(b, p.out(1))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def disagree(S):
    """The check's disagreement projector as a picture.  plain: |01><01| + |10><10| --
    a green copier on each wire, their spare legs joined by a two-legged red half-turn
    (the effect <01| + <10|, exact).  cross: the colours swapped, |+-><+-| + |-+><-+|.
    No compensation (checked against the matrix below)."""
    cp, mid = (VT.Z, VT.X) if S == PLAIN else (VT.X, VT.Z)
    g = zx.Graph()
    V = g.add_vertex
    i0, i1 = V(VT.BOUNDARY, 0, 0), V(VT.BOUNDARY, 1, 0)
    c0, c1 = V(cp, 0, 1), V(cp, 1, 1)
    m = V(mid, 0.5, 2, phase=Fraction(1))
    o0, o1 = V(VT.BOUNDARY, 0, 3), V(VT.BOUNDARY, 1, 3)
    g.add_edge((i0, c0)); g.add_edge((c0, o0)); g.add_edge((c0, m))
    g.add_edge((i1, c1)); g.add_edge((c1, o1)); g.add_edge((c1, m))
    g.set_inputs((i0, i1)); g.set_outputs((o0, o1))
    return g


def then(g1, g2):
    h = g1.copy(); h.compose(g2)                               # compose mutates in place
    return h


def ringed_check(m1, m2, S):
    """The bell on a ringed pair: the doubled Bell state B_{m1m2} (x) B_{m1m2}* -- the
    ringed cup and its mirror, whose rings carry the NEGATED phases (the mirror conjugates;
    for a half-turn -pi is pi) -- read at S at both ends, the parity dot on the records.
    Output w.  Scalar: 1/sqrt2 . 1/sqrt2 for the doubled Bell state, sqrt2 for the parity:
    add_power(-1)."""
    g = zx.Graph()
    V = g.add_vertex
    q = ET.HADAMARD if S == CROSS else ET.SIMPLE
    cup, cupb = V(VT.Z, 1, 0), V(VT.Z, 3, 0)
    ra, ga = V(VT.X, 0, 1, phase=Fraction(m2)), V(VT.Z, 0, 2, phase=Fraction(m1))
    rab, gab = V(VT.X, 2, 1, phase=Fraction(-m2) % 2), V(VT.Z, 2, 2, phase=Fraction(-m1) % 2)
    rA, rB, par, w = V(VT.Z, 1, 3), V(VT.Z, 3, 3), V(VT.X, 2, 4), V(VT.BOUNDARY, 2, 5)
    g.add_edge((cup, ra)); g.add_edge((ra, ga)); g.add_edge((ga, rA), q)      # a
    g.add_edge((cupb, rab)); g.add_edge((rab, gab)); g.add_edge((gab, rA), q)  # abar
    g.add_edge((cup, rB), q); g.add_edge((cupb, rB), q)                        # b, bbar
    g.add_edge((rA, par)); g.add_edge((rB, par)); g.add_edge((par, w))
    g.set_inputs(()); g.set_outputs((w,))
    g.scalar.add_power(-1)
    return g


def as_state(g):
    """The same picture with its inputs turned round into outputs (inputs first).  A
    reduced picture in which a bare edge joins two INPUTS cannot be evaluated as a map by
    pyzx 0.10.6 ("axes don't match array"); as a state it can, and two maps are equal
    exactly when their states are."""
    h = g.copy()
    h.set_outputs(tuple(h.inputs()) + tuple(h.outputs())); h.set_inputs(())
    return h


def connected(g, u, v):
    seen, todo = {u}, [u]
    while todo:
        w = todo.pop()
        for n in g.neighbors(w):
            if n not in seen:
                seen.add(n); todo.append(n)
    return v in seen


def sketchbook():
    # the red X(2 -> 1) dot is 1/sqrt2 x parity, as PyZX draws it
    g = zx.Graph()
    a, b, x, w = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0), \
        g.add_vertex(VT.X, 0.5, 1), g.add_vertex(VT.BOUNDARY, 0.5, 2)
    g.add_edge((a, x)); g.add_edge((b, x)); g.add_edge((x, w))
    g.set_inputs((a, b)); g.set_outputs((w,))
    parity = np.array([[1.0 if w_ == (i ^ j) else 0.0 for i in (0, 1) for j in (0, 1)] for w_ in (0, 1)])
    assert np.allclose(matrix_of(g), parity / np.sqrt(2))

    for S in (PLAIN, CROSS):
        g = the_alarm(S)
        # (a) the VALUE: p(kE, w), exactly
        assert np.allclose(matrix_of(g).ravel().real, alarm_dist(S).ravel())
        assert np.allclose(matrix_of(g).imag, 0)
        assert same_picture(g, expected(S))
        assert not same_picture(g, expected(1 - S))
        # (b) the LINES: twelve edges -- eight doubled quantum lines (a, abar, b, bbar, b',
        # b'bar, e, ebar) and four classical (rA-par kA, rB-par kB, rE-out kE, par-out w).
        # A Hadamard edge is one line with a box on it.
        assert g.num_edges() == 12
        classical = [e for e in g.edges()
                     if VT.BOUNDARY in (g.type(e[0]), g.type(e[1])) or VT.X in (g.type(e[0]), g.type(e[1]))]
        assert len(classical) == 4
        if S == CROSS:
            assert sum(g.edge_type(e) == ET.HADAMARD for e in g.edges()) == 6   # a, abar, b', b'bar, e, ebar
        # (c) DERIVATION: full_reduce, scalar kept
        h = g.copy(); simp.full_reduce(h)
        assert same_picture(h, expected(S))
        kE, w = h.outputs()
        if S == PLAIN:
            # Eve's record and the alarm fall apart: a green point on kE, a red point on w
            assert not connected(h, kE, w)
            assert h.num_vertices() == 4 and h.num_edges() == 2
        else:
            # one bend joins Eve's record to the alarm (a bare output-output edge; it
            # evaluates as a state)
            assert connected(h, kE, w)
            assert h.num_vertices() == 2 and h.num_edges() == 1
        # The Alarm yanks into the prepare-and-measure form: the cup and Alice's reader
        # fuse into her sender, fed a fair coin
        assert same_picture(the_alarm(S, bent=True), g)
        hb = the_alarm(S, bent=True).copy(); simp.full_reduce(hb)
        assert same_picture(hb, expected(S))

    # (d) the pass projector: 1/2 cap then cup = 1/4 (I + ZZ)(I + XX), rank 1, and
    # DISCONNECTED in the middle -- no path from its inputs to its outputs
    P = (np.eye(4) + kron(Z, Z)) @ (np.eye(4) + kron(X, X)) / 4
    g = pass_projector()
    assert np.allclose(matrix_of(g), P)
    assert np.linalg.matrix_rank(matrix_of(g)) == 1
    h = g.copy(); simp.full_reduce(h)
    assert same_picture(as_state(h), as_state(g))          # the cap is now a bare in-in edge
    for i in h.inputs():
        for o in h.outputs():
            assert not connected(h, i, o)

    # (e) the checks as half-turn rings on the bend (row 94)
    dis = {PLAIN: kron(ket(0, 1).reshape(4, 1), ket(0, 1).reshape(1, 4)) +
                  kron(ket(1, 0).reshape(4, 1), ket(1, 0).reshape(1, 4)),
           CROSS: np.outer(np.kron(PLUS, MINUS), np.kron(PLUS, MINUS)) +
                  np.outer(np.kron(MINUS, PLUS), np.kron(MINUS, PLUS))}
    for S in (PLAIN, CROSS):
        assert np.allclose(matrix_of(disagree(S)), dis[S])      # exact, no compensation
    # translation check (not a second proof): the four ringed-bend projectors, summed as
    # matrices, give Result 12.6's two identities
    proj_of = {(m1, m2): np.outer(matrix_of(ringed_cup(m1, m2)).ravel(), matrix_of(ringed_cup(m1, m2)).ravel().conj())
               for m1 in (0, 1) for m2 in (0, 1)}
    assert np.allclose(proj_of[0, 1] + proj_of[1, 1], dis[PLAIN])
    assert np.allclose(proj_of[1, 0] + proj_of[1, 1], dis[CROSS])
    for m1 in (0, 1):
        for m2 in (0, 1):
            B = ringed_cup(m1, m2)
            assert np.allclose(matrix_of(B).ravel(), bell(m1, m2))
            # a plain check sees the RED ring, a cross check the GREEN one:
            #   P_dis(S) B = (m2 or m1) B
            for S, m in ((PLAIN, m2), (CROSS, m1)):
                g = then(B, disagree(S))
                if m:
                    assert same_picture(g, B)
                else:
                    assert np.allclose(matrix_of(g), 0)
                h = g.copy(); simp.full_reduce(h)               # derived, scalar kept
                assert np.allclose(matrix_of(h), matrix_of(g))
            # passing both checks takes both rings off and leaves the bare bend:
            #   P B_{m1m2} = [m1 = m2 = 0] B_00
            g = then(B, pass_projector())
            if (m1, m2) == (0, 0):
                assert same_picture(g, ringed_cup(0, 0))
            else:
                assert np.allclose(matrix_of(g), 0)
            h = g.copy(); simp.full_reduce(h)
            assert np.allclose(matrix_of(h), matrix_of(g))
            # the bell on a ringed pair: at plain it rings exactly on a red ring (w = m2),
            # at cross exactly on a green one (w = m1)
            for S, m in ((PLAIN, m2), (CROSS, m1)):
                g = ringed_check(m1, m2, S)
                want = np.array([1.0 - m, float(m)])
                assert np.allclose(matrix_of(g).ravel(), want)
                h = g.copy(); simp.full_reduce(h)
                assert same_picture(h, g)


if __name__ == "__main__":
    verdict("Alarm", blackboard, sketchbook)
