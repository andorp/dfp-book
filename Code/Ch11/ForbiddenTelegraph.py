"""Challenge V: The Forbidden Telegraph  (no signal through a copier; the master)
==============================================================================
"So Delft sees the same whatever I do here."  -- Achilles
"It copies its own kind and makes a pair of every other; and a press that copied every kind would beat the post to Delft."  -- Achilles

Claim: share Phi+ between a near stone a and a far stone b; read a at a setting
(plain or cross) to get m; copy b with delta and read both copies plain, z1 and z2.
The eleven-line doubled master evaluates exactly to p(m,z1,z2) = 1/4 d(z1,z2) at
cross and 1/2 d(m,z1) d(z1,z2) at plain; the far marginal is 1/2 d(z1,z2) at both
settings: no signal.  An ideal cloner psi -> psi (x) psi (not linear; no picture)
would give P(z1 = z2) = 1/2 at cross against 1 at plain: a telegraph.  A cut through
one wire has rank <= 2, the clones span 3 dimensions, and (I (x) delta) cup is the
three-legged green dot |000> + |111>.
"""
import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import BELL, CUP, I2, kron, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch11 as PC
import props_cloth as CLOTH


DELTA = np.outer(ket(0, 0), ket(0)) + np.outer(ket(1, 1), ket(1))
PLUS = (ket(0) + ket(1)) / np.sqrt(2)
MINUS = (ket(0) - ket(1)) / np.sqrt(2)
SETTINGS = {"plain": (ket(0), ket(1)), "cross": (PLUS, MINUS)}
KD = np.eye(2)                                               # Kronecker delta d(i, j)


def telegraph(setting, copier):
    """p[m, z1, z2]: read the near stone of Phi+ at `setting`, apply `copier` (a function
    on the far stone's normalized branch state) and read both copies plain."""
    p = np.zeros((2, 2, 2))
    for m, n in enumerate(SETTINGS[setting]):
        branch = kron(n.conj().reshape(1, 2), I2) @ BELL     # (<n_m| (x) I) Phi+, unnormalized
        pm = np.linalg.norm(branch) ** 2
        far = copier(branch.ravel() / np.sqrt(pm))
        for z1 in (0, 1):
            for z2 in (0, 1):
                p[m, z1, z2] = pm * abs(np.vdot(ket(z1, z2), far)) ** 2
    return p


def blackboard():
    rng = np.random.default_rng(1105)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))       # cup unnormalized
    assert np.allclose(BELL, CUP / np.sqrt(2))               # Phi+ = cup / sqrt2

    # the real press, delta, is linear: the branch-by-branch computation is exact
    real = {s: telegraph(s, lambda v: DELTA @ v) for s in SETTINGS}
    want_plain = 0.5 * np.einsum('mz,zw->mzw', KD, KD)       # 1/2 d(m,z1) d(z1,z2)
    want_cross = 0.25 * np.einsum('m,zw->mzw', np.ones(2), KD)   # 1/4 d(z1,z2)
    assert np.allclose(real["plain"], want_plain)
    assert np.allclose(real["cross"], want_cross)
    # the proving-book: the struck pair ALWAYS agreed, half high, half low, at either
    # setting and for either near reading -- page for page the same
    for s in SETTINGS:
        far = real[s].sum(axis=0)
        assert np.allclose(far, 0.5 * KD)
        for m in (0, 1):
            cond = real[s][m] / real[s][m].sum()
            assert np.isclose(cond[0, 1] + cond[1, 0], 0)            # never one of each
    assert np.allclose(real["plain"].sum(axis=0), real["cross"].sum(axis=0))   # no signal
    # "where the near half was read plain, the struck pair matched the near reading too"
    assert np.isclose(sum(real["plain"][m, m, m] for m in (0, 1)), 1)
    # ... while at cross the near record is independent of the far ones (a product)
    pc = real["cross"]
    assert np.allclose(pc, np.einsum('m,zw->mzw', pc.sum(axis=(1, 2)), pc.sum(axis=0)))
    assert not np.allclose(real["plain"], np.einsum('m,zw->mzw', real["plain"].sum(axis=(1, 2)),
                                                    real["plain"].sum(axis=0)))
    # the single run on the table -- near high at cross, both struck stones low -- was
    # possible: a quarter of the time, whichever mark the near needle took
    for m in (0, 1):
        assert np.isclose(pc[m, 0, 0], 0.25)

    # the counterfactual ideal cloner K psi = psi (x) psi on each branch.  This map is
    # not linear; it has no picture.
    K = lambda v: np.kron(v, v)
    u, w = random_state(rng), random_state(rng)
    assert not np.allclose(K(u + w), K(u) + K(w))            # not linear
    ideal = {s: telegraph(s, K) for s in SETTINGS}
    agree = {s: sum(ideal[s][m, z, z] for m in (0, 1) for z in (0, 1)) for s in SETTINGS}
    assert np.isclose(agree["plain"], 1.0) and np.isclose(agree["cross"], 0.5)   # a telegraph
    # why only a nonlinear machine reads the setting: the far stone is I/2 either way,
    # and the two settings are two mixtures of the same matrix
    mix = {s: sum(0.5 * np.outer(n, n.conj()) for n in SETTINGS[s]) for s in SETTINGS}
    assert np.allclose(mix["plain"], I2 / 2) and np.allclose(mix["cross"], I2 / 2)
    lin = {s: sum(0.5 * np.outer(DELTA @ n, (DELTA @ n).conj()) for n in SETTINGS[s]) for s in SETTINGS}
    assert np.allclose(lin["plain"], lin["cross"])           # delta sees only the matrix
    kk = {s: sum(0.5 * np.outer(K(n), K(n).conj()) for n in SETTINGS[s]) for s in SETTINGS}
    assert not np.allclose(kk["plain"], kk["cross"])         # K tells the mixtures apart

    # rank across the cut m | z1 z2: 2 at plain (joined), 1 at cross (fallen apart)
    assert np.linalg.matrix_rank(real["plain"].reshape(2, 4)) == 2
    assert np.linalg.matrix_rank(real["cross"].reshape(2, 4)) == 1

    # Result 11.8 (i): for EVERY one-in, two-out D, (I (x) D) cup has rank <= 2 across the
    # cut a | b1 b2 -- it is the image of D, spanned by D|0>, D|1>
    for _ in range(50):
        D = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
        state = (kron(I2, D) @ CUP).ravel()
        assert np.linalg.matrix_rank(state.reshape(2, 4)) <= 2
        assert np.linalg.matrix_rank(D) <= 2
    # (ii) the clones psi (x) psi span three dimensions (the symmetric part) ...
    clones = np.array([np.kron(v, v) for v in (random_state(rng) for _ in range(12))])
    assert np.linalg.matrix_rank(clones) == 3
    sym = [ket(0, 0), ket(1, 1), (ket(0, 1) + ket(1, 0)) / np.sqrt(2)]
    assert np.linalg.matrix_rank(np.vstack([clones, sym])) == 3
    assert np.allclose(np.kron(PLUS, PLUS) - ket(0, 0) / 2 - ket(1, 1) / 2, (ket(0, 1) + ket(1, 0)) / 2)
    # ... more than one wire allows: no linear process copies every state
    # (iii) on a wire of dimension r the clones span r(r+1)/2 dimensions > r unless r <= 1
    for r in (1, 2, 3, 4):
        cl = np.array([np.kron(v, v) for v in (random_state(rng, r) for _ in range(3 * r * r))])
        assert np.linalg.matrix_rank(cl) == r * (r + 1) // 2
        assert (r * (r + 1) // 2 <= r) == (r <= 1)
    one_cup = np.array([[1.0]])                              # a cup on a 1-dim wire: rank 1
    assert np.linalg.matrix_rank(one_cup) == 1
    # (iv) what the copier does to a cup: it grows a leg, |000> + |111>, still joined
    ghz = ket(0, 0, 0) + ket(1, 1, 1)
    assert np.allclose((kron(I2, DELTA) @ CUP).ravel(), ghz)
    for cut in ((2, 4), (4, 2)):
        assert np.linalg.matrix_rank(ghz.reshape(cut)) == 2
    assert np.linalg.matrix_rank(np.moveaxis(ghz.reshape(2, 2, 2), 1, 0).reshape(2, 4)) == 2

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the black slab once more, just after the
# one strike: the near stone read high at cross, spent, its counter laid face-up
# opposite it; the far stone gone into the press; two struck stones below the
# chutes, their reflections slack as the fresh pair's were.  Then, on the
# Tortoise's cloth, the reading rings: at plain, slid together into one green ring
# with every counter's cord hanging from it (keyed: ring, counter's cord); at cross,
# the near reading's red ring joined to the far green ring, and then drawn away,
#   \presstable[operations]{scene}{caption}, \copycloth[operations]{scene}{caption}
#                                                         (pdf/props/ch11.tex)
# tools/props_ch11.py transcribes the Lua (with L1-L5); the calls are read out of
# dialog.md, never copied.  This file also checks the whole dialogue's sixteen
# pictures, the Lua's thirty-five fault sites one by one, and the transcription's
# constants against pdf/props/ch11.lua itself (a drift guard).
#
# The ties (identity card, section 4), a structural check against this file's
# blackboard, not a second proof.  Slack and tight: a stone's reflection is slack
# when the stone, averaged over every record laid for any other stone, is 1/2 I, and
# tight when, given the counter laid opposite it, it is pure.  The far stone after
# the near reading is 1/2 I averaged over the near record at either setting; given
# that record it is pure, and a different pure state at the two settings -- a far
# reflection that tightened on a near reading would be the telegraph (state model
# 2c), so the Lua keeps it slack and the dialogue never pictures it.  The run the
# table shows (l.436) has probability 1/4.  The reading rings: the records whose
# cords hang from one green ring always agree, and records on separate pieces are
# independent (the distribution is the product) -- the sketchbook's fusion result,
# read off the picture.
# --------------------------------------------------------------------------
LUA = pathlib.Path(__file__).resolve().parents[3] / "pdf" / "props" / "ch11.lua"
DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = "So Delft sees the same whatever I do here."
F = "the fresh pair"
STRIKE = "read near crosswise high, far into the hopper, strike"
RUN = STRIKE + ", read first plain low, read second plain low"      # l.436; not pictured
SIZES = {"First": [1, 2, 1], "Second": [1, 1, 1], "Third": [3], "Fourth": [2], "Fifth": [1, 1, 2]}
SETTING = {"plain": "plain", "crosswise": "cross"}       # the viewer's collar
RESULT = {"high": 0, "low": 1}                           # the needle's mark: plain high = |0>
RECORD = {"m": 0, "z1": 1, "z2": 2}                      # the axes of telegraph's p[m, z1, z2]


def dm(v):
    v = np.asarray(v).ravel()
    return np.outer(v, v.conj())


def on(n, i, op):
    """op on stone i of n stones (top wire first)."""
    return kron(*[op if j == i else I2 for j in range(n)])


def reduced(rho, n, i):
    """The one-stone state of stone i of an n-stone density matrix."""
    a, b = 2 ** i, 2 ** (n - i - 1)
    return np.einsum('xiyxjy->ij', rho.reshape(a, 2, b, a, 2, b))


def purity(rho):
    return np.trace(rho @ rho).real


def reading_proj(r):
    """The projector of a reading: its setting, its mark."""
    return dm(SETTINGS[SETTING[r.setting]][RESULT[r.result]])


def joint(T):
    """The stones of table state T as one joint state, before any reading: the near and
    far stone, Phi+; after the strike the near stone and the two struck stones, the
    copier on the far half, (I (x) delta) Phi+."""
    if T.struck:
        return ("near", "first", "second"), dm(kron(I2, DELTA) @ BELL)
    return ("near", "far"), dm(BELL)


def readings(T):
    """name -> Reading, for every stone of T read so far."""
    stones = [("near", T.near)] + ([("first", T.struck[0]), ("second", T.struck[1])] if T.struck else [])
    return {name: s.reading for name, s in stones if s.reading}


def averaged(T, name):
    """Stone `name`, averaged over every record laid for any OTHER stone: each such
    reading applied, its mark forgotten."""
    names, rho = joint(T)
    n = len(names)
    for other, r in readings(T).items():
        if other != name:
            i = names.index(other)
            rho = sum(on(n, i, P) @ rho @ on(n, i, P)
                      for P in (reading_proj(PC.Reading(r.setting, k)) for k in RESULT))
    return reduced(rho, n, names.index(name))


def given(T, name, records):
    """Stone `name` given the listed records (name -> Reading), and their probability."""
    names, rho = joint(T)
    n = len(names)
    for other, r in records.items():
        P = on(n, names.index(other), reading_proj(r))
        rho = P @ rho @ P
    p = np.trace(rho).real
    return reduced(rho / p, n, names.index(name)), p


def check_reflections(T):
    """The reflection rule on every stone the table view of T shows."""
    for t in PC.visible(T).black:
        if t.kind != "stone":
            continue
        avg = averaged(T, t.name)
        if t.refl == "slack":
            assert t.gloss == "restless" and t.name not in readings(T)
            assert np.allclose(avg, I2 / 2) and np.isclose(purity(avg), 0.5)
        else:
            r = readings(T)[t.name]
            own, p = given(T, t.name, {t.name: r})
            assert t.refl == "tight" and t.gloss == "spent" and p > 0
            assert np.isclose(purity(own), 1) and np.allclose(own, reading_proj(r))
            assert np.isclose(purity(avg), 0.5)          # tight by its own counter, nothing else


def hanging(S):
    """The reading rings' cloth: ring -> (colour, x, the records whose cords hang from
    it), the records named by where their free ends lie, left to right (the near
    counter's, then the far two); and the closed cords, each the set of rings it links."""
    cords = [d for d in S["cords"] if not d.get("closed")]
    assert all(d["a"] == "fold" and isinstance(d["b"], int) for d in cords)
    names = dict(zip(sorted(d["pts"][0][0] for d in cords), ("m", "z1", "z2")))
    rings = {}
    for i, t in enumerate(S["things"], 1):
        assert t["kind"] == "ring"
        rings[i] = (t["c"], t["x"], set())
    for d in cords:
        c, x, held = rings[d["b"]]
        assert d["pts"][-1] == [x, 0]                    # the cord ends at its ring
        held.add(names[d["pts"][0][0]])
    links = [{i for i, t in enumerate(S["things"], 1) if CLOTH._onpath(d["pts"], t["x"], t["y"], True)}
             for d in S["cords"] if d.get("closed")]
    return rings, links


def pieces_hold(p, groups):
    """p[m, z1, z2] as the picture's pieces say: the records on one piece always agree,
    and the distribution is the product over the pieces."""
    idx = np.indices(p.shape)
    prod = np.ones((1, 1, 1))
    for g in groups:
        keep = sorted(RECORD[r] for r in g)
        prod = prod * p.sum(axis=tuple(a for a in range(3) if a not in keep), keepdims=True)
        same = np.all([idx[a] == idx[keep[0]] for a in keep], axis=0)
        if not np.isclose(p[same].sum(), 1):
            return False
    return bool(np.allclose(p, prod))


def check_clearances(T):
    """Every drawn extent of table state T (card 1c): each stone, standing and
    reflected, a positive gap from the press's legs, body and chutes (standing and
    reflected) and from every other stone; everything on the black a positive gap
    inside the slab's ends."""
    black = PC.visible(T).black
    for i, a in enumerate(black):
        for b in black[i + 1:]:
            for view in (PC.standing, PC.reflection):
                g = min(PC.gap(r, q) for r in view(a) for q in view(b))
                assert g > 0, (T, a.name, b.name, view.__name__, g)
        for x0, y0, x1, y1 in PC.standing(a) + PC.reflection(a):
            assert PC.SX0 < x0 and x1 < PC.SX1, (a.name, x0, x1)


def check_faults():
    """Every one of the Lua's thirty-five fault sites, each raised by a call of its own
    and by nothing else."""
    T, Cl, K = PC.table_state, PC.cloth_state, PC.knot_pull
    g, r = "green", "red"
    bad = [
        ("table-scene", T, "", "the slab"), ("table-op", T, "look at the press", F),
        ("table-op-read", T, "read the press plain high", F), ("table-setting", T, "read near high", F),
        ("table-result", T, "read near plain middle", F), ("table-result-none", T, "read near", F),
        ("table-read-twice", T, "read near plain high, read near crosswise low", F),
        ("table-read-far", T, "read far plain high", F), ("table-read-early", T, "read first plain low", F),
        ("table-struck-twice", T, "far into the hopper, strike, read second plain low, read second plain high", F),
        ("table-hopper-twice", T, "far into the hopper, into the hopper", F),
        ("table-hopper-late", T, "far into the hopper, strike, into the hopper", F),
        ("table-strike-empty", T, "strike", F), ("table-strike-twice", T, "far into the hopper, strike, strike", F),
        ("cloth-scene", Cl, "", "the bend"), ("cloth-op", Cl, "pull", "the fork, red point"),
        ("fork-thread-twice", Cl, "thread, thread", "the fork, red point"),
        ("fork-lift-red", Cl, "thread, lift off", "the fork, red point"),
        ("fork-lift-early", Cl, "lift off", "the fork, green point"),
        ("fork-lift-twice", Cl, "thread, lift off, lift off", "the fork, green point"),
        ("loop-slide-twice", Cl, "slide the cards together, slide the cards together", "the likeness loop"),
        ("loop-slide-pressed", Cl, "after a faithful pressing, slide the cards together", "the likeness loop"),
        ("loop-press-twice", Cl, "after a faithful pressing, after a faithful pressing", "the likeness loop"),
        ("mirror-fold-twice", Cl, "fold the second copy, fold the second copy", "the fork and its mirror"),
        ("mirror-fused-fold", Cl, "fold the second copy, slide the forks together, fold the second copy",
         "the fork and its mirror"),
        ("mirror-slide-early", Cl, "slide the forks together", "the fork and its mirror"),
        ("mirror-slide-twice", Cl, "fold the second copy, slide the forks together, slide the forks together",
         "the fork and its mirror"),
        ("knot-pull-twice", Cl, "pull, pull", "the knot"),
        ("knot-not-alternating", K, [{"upper": g, "lower": r}] * 3),
        ("knot-count", K, [{"upper": g, "lower": r}, {"upper": r, "lower": g}]),
        ("reading-slide-cross", Cl, "slide together", "the reading rings, crosswise"),
        ("reading-slide-twice", Cl, "slide together, slide together", "the reading rings, plain"),
        ("reading-apart-plain", Cl, "slide together, draw apart", "the reading rings, plain"),
        ("reading-apart-twice", Cl, "draw apart, draw apart", "the reading rings, crosswise"),
        ("reading-plain-unslid", Cl, "", "the reading rings, plain"),
    ]
    for site, fn, *args in bad:
        assert PC.faults(fn, *args) == [site], (site, PC.faults(fn, *args))
    assert sorted(site for site, *_ in bad) == sorted(PC.ERROR_SITES) and len(bad) == 35
    # and the legal calls raise none: every scene bare or with its dialogue operations
    assert PC.faults(T, RUN, F) == [] and PC.faults(K, list(PC.KNOT)) == []


def check_lua():
    """The transcription's constants are the Lua's (a drift guard: the places after L1
    and L5, the frame, the reflection geometry), and the Lua has as many fault sites
    as ERROR_SITES -- its terr and cerr calls, their definitions aside."""
    lua = LUA.read_text(encoding="utf-8")
    nums = lambda pat: tuple(float(x) for x in re.search(pat, lua).groups())
    assert nums(r"local FTOP, FBOT = (\S+), (\S+)\n") == (PC.FTOP, PC.FBOT)
    assert nums(r"local SX0, SX1, STOP, SBOT = (\S+), (\S+), \S+, \S+\n") == PC.SLAB_ENDS
    assert nums(r"local XNEAR, XFAR, XPRESS = (\S+), (\S+), (\S+)\n") == (PC.XNEAR, PC.XFAR, PC.XPRESS)
    assert nums(r"local STRUCKDX = (\S+) ") == (PC.STRUCKDX,)
    assert "local XFIRST, XSECOND = XPRESS - STRUCKDX, XPRESS + STRUCKDX\n" in lua
    assert nums(r"local XTRAY = (\S+)\n") == (PC.XTRAY,)
    assert nums(r"local XVIEW = \{ tray = (\S+), window = (\S+), far = XSECOND \}") == \
        (PC.XVIEW["tray"], PC.XVIEW["window"]) and PC.XVIEW["far"] == PC.XSECOND
    assert nums(r"local COUNTERGAP = (\S+) ") == (PC.COUNTERGAP,)
    assert "local XCOUNTER = { near = XNEAR, first = XFIRST, second = XFIRST + COUNTERGAP }" in lua
    assert nums(r"local YCOUNTER = (\S+)\n") == (PC.YCOUNTER,)
    assert nums(r"local VSCALE = (\S+) ") == (PC.VSCALE,)
    assert nums(r"local VFOOT = (\S+)\n") == (PC.VFOOT,)
    assert "local VHALF = 0.12 * VSCALE" in lua and np.isclose(PC.VHALF, 0.12 * PC.VSCALE)
    assert nums(r"local SRX, SRY = (\S+), (\S+)\n") == (PC.SRX, PC.SRY)
    assert nums(r"local SLACKDX = \{ (\S+), (\S+) \}") == PC.SLACKDX
    assert nums(r"local GHOSTDX = (\S+) ") == (PC.GHOSTDX,)
    assert "ipairs({ -GHOSTDX, GHOSTDX })" in lua and "ipairs(SLACKDX)" in lua
    assert nums(r"local CRX, CRY = (\S+), (\S+)\n") == (PC.CRX, PC.CRY)
    assert nums(r"local TRW, TRY = (\S+), (\S+)\n") == (PC.TRW, PC.TRY)
    # the press's drawn extents: its legs and body from the drawing, its chutes and the
    # whole from the DRAWN EXTENTS comment the L1 clearances rest on
    legs = nums(r'pt\(s \* (\S+) - (\S+), 0\) \.\. " rectangle " \.\.\s+pt\(s \* \S+ \+ \S+, (\S+)\)')
    assert (legs[0] + legs[1], 0.0, legs[2]) == PC.PRESS_PARTS["legs"]
    body = nums(r'local body = pt\(-(\S+), (\S+)\) \.\. " rectangle " \.\. pt\((\S+), (\S+)\)')
    assert (body[0], body[1], body[3]) == PC.PRESS_PARTS["body"] and body[0] == body[2]
    ext = re.search(r"-- DRAWN EXTENTS.*?\n(?!--)", lua, re.S).group(0)
    ext = " ".join(l.lstrip("- ") for l in ext.splitlines())
    assert re.search(r"below y = (\S+) only the legs \(\|x\| <= (\S+)\)", ext).groups() == \
        (str(PC.PRESS_PARTS["body"][1]), str(PC.PRESS_PARTS["legs"][0]))
    assert re.search(r"the body \|x\| <= (\S+) from y = (\S+);", ext).groups() == \
        tuple(str(v) for v in PC.PRESS_PARTS["body"][:2])
    chutes = re.search(r"the chutes reach \|x\| = (\S+), their lowest edge at y = (\S+);", ext).groups()
    assert tuple(map(float, chutes)) == PC.PRESS_PARTS["chutes"][:2]
    whole = re.search(r"The whole press: \|x\| <= (\S+), y (\S+) \.\. (\S+)\.", ext).groups()
    assert tuple(map(float, whole)) == PC.PRESS_WHOLE
    assert nums(r"local FORKHEAD = (\S+)\n") == (PC.FORKHEAD,)
    assert nums(r"local REDJOINED, REDAPART = (\S+), (\S+)\n") == (PC.REDJOINED, PC.REDAPART)
    assert nums(r'"red", rx, 0\)\)\n +local g = add\(ring\("ring", "green", (\S+), 0\)\)') == (PC.XGREEN,)
    assert nums(r'x = rx - (\S+), side') == (PC.NEARCORD,)
    code = re.sub(r"--[^\n]*", "", lua)                           # comments aside
    direct = len(re.findall(r"tex\.error\(", code))               # inside terr and cerr only
    table = len(re.findall(r"\bterr\(", code)) - 1                # terr's own definition
    cloth = len(re.findall(r"\bcerr\(", code)) - 1                # cerr's own definition
    assert (direct, table, cloth) == (2, 14, 21) and table + cloth == len(PC.ERROR_SITES) == 35


def check_whole_dialogue():
    """All sixteen pictures: their number and places, the keyed rule, the captions, the
    frames, no fault anywhere; and the hooks."""
    assert DIALOGUE.check_whole(PC.ORDER) == 16                  # 4 + 3 + 3 + 2 + 4
    text, raw = DIALOGUE.read(DIALOGUE.dialog), DIALOGUE.read(DIALOGUE.raw)
    assert len(DIALOGUE.call_re.findall(raw)) == 16              # dialog.raw.md too
    for t in (text, raw):
        assert not DIALOGUE.call_re.findall(t[:t.index("## First Challenge")])   # none before
        assert set(re.findall(r"\\begin\{(\w+)\}", t[t.index("## First Challenge"):])) == {"center"}
        assert t.count("Let Bob draw") == 0                      # no nested Bob this chapter
    DIALOGUE.keyed_first(PC.ORDER, kind=PC.keyed_kind)
    placed = [(o, i, c) for o in PC.ORDER
              for i, c in enumerate(c for b in DIALOGUE.blocks(o) for c in b)]
    assert [(o, i) for o, i, c in placed if c.keyed] == [
        ("First", 0), ("First", 1), ("Second", 0), ("Third", 0), ("Fourth", 0), ("Fifth", 1)]
    spoken = PC.spoken_sentences(text) | PC.spoken_sentences(raw)
    for o in PC.ORDER:
        assert [len(b) for b in DIALOGUE.blocks(o)] == SIZES[o], o
        for b in DIALOGUE.blocks(o):
            assert len({PC.height(c) for c in b}) == 1, (o, [c.text for c in b])
            assert len({c.macro for c in b}) == 1, (o, [c.text for c in b])   # never mixed
    for o, i, c in placed:
        PC.picture(c)                                            # the macro never faults
        assert words_only(c.caption, forbidden=PC.CAPTION_FORBIDDEN), c.caption
        assert CLOTH.caption_ok(c.caption), c.caption
        assert PC.normal_sentence(c.caption) not in spoken, c.caption
        assert PC.place_ok(o, c), (o, c.text)
        assert len(c.args) == 1 and c.args[0]                    # one scene argument
    # the forbidden places, as the card lists them, nowhere in the dialogue
    assert [o for o, i, c in placed if c.macro == "presstable"] == ["First", "Fifth"]
    for fam, where in (("fork", "First"), ("loop", "Second"), ("mirror", "Third"), ("knot", "Fourth"),
                       ("reading", "Fifth")):
        assert {o for o, i, c in placed if PC.family(c) == fam} == {where}, fam


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    check_lua()
    check_faults()
    check_whole_dialogue()
    calls = DIALOGUE.check_layout("Fifth", sizes=[1, 1, 2])
    p51, p52, p53, p54 = calls
    text, raw = DIALOGUE.read(DIALOGUE.dialog), DIALOGUE.read(DIALOGUE.raw)
    spoken = PC.spoken_sentences(text) | PC.spoken_sentences(raw)
    for c in calls:
        assert words_only(c.caption, forbidden=PC.CAPTION_FORBIDDEN), c.caption
        assert PC.normal_sentence(c.caption) not in spoken, c.caption   # the gigue: never said twice
        assert PC.place_ok("Fifth", c), c.text
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Fifth")] == [[1.66], [1.22], [1.22, 1.22]]
    assert [c.macro for c in calls] == ["presstable"] + ["copycloth"] * 3
    assert [c.keyed for c in calls] == [False, True, False, False]
    assert [(c.opt, c.args[0]) for c in calls] == [
        (STRIKE, F), ("slide together", "the reading rings, plain"),
        ("", "the reading rings, crosswise"), ("draw apart", "the reading rings, crosswise")]
    assert [c.caption for c in calls] == ["nothing about either says which", "slid together, all green",
                                          "joined by the stone's and its reflection's",
                                          "red drawn away from green"]
    t51, c52, c53, c54 = (PC.picture(c) for c in calls)
    a1, a2, a3 = DIALOGUE.anchors("Fifth")
    f1, f2, f3 = DIALOGUE.following("Fifth")
    r1, r2, r3 = DIALOGUE.following("Fifth", DIALOGUE.raw)
    section = DIALOGUE.section("Fifth")
    assert "its counter lying opposite to say which way" in section           # D9
    assert "slack as the fresh pair's were" in a1                            # D10
    assert f1.startswith("*[The patch of sun")                               # the ambient, then
    assert r1.startswith("*[The Crab turns back from the fire")              # the Escher (raw)
    assert "every counter's cord" in a2 and f2.startswith("*[Then she")      # D12
    assert "lays it out afresh for the crosswise reading" in a3              # D13
    assert f3.startswith("**Achilles:** At cross, my counter")
    assert (r2[:10], r3[:36]) == (f2[:10], f3[:36])

    # P5.1: just after the strike (dump as the card's section 3), unkeyed
    assert t51.dump == ("near spent tight read crosswise high; far pressed; strikes 1; first restless "
                        "slack; second restless slack; tray 2; counter near up; viewer window crosswise high")
    run = PC.presstable(RUN, F)                                  # l.436, the full run: not pictured
    assert run.dump == ("near spent tight read crosswise high; far pressed; strikes 1; first spent tight "
                        "read plain low; second spent tight read plain low; tray 0; counter near up; "
                        "counter first down; counter second down; viewer far upright low")
    fresh = PC.presstable("", F)
    trays = [PC.presstable(o, F).state.tray for o in
             ("", "read near crosswise high", STRIKE, STRIKE + ", read first plain low", RUN)]
    assert trays == [3, 2, 2, 1, 0]                              # one counter per reading
    # L1, the fixed parts (the x-values are the Lua's, the rules the card's)
    x = lambda view: {t.name: t.x for t in view.black}
    v0, v1, v2 = fresh.shown, t51.shown, run.shown
    assert x(v0)["near"] == x(v1)["near"] == PC.XNEAR and x(v0)["press"] == x(v1)["press"] == PC.XPRESS
    assert "far" not in x(v1) and "far" in x(v0)                 # a bare patch where it stood
    assert (x(v1)["first"], x(v1)["second"]) == (x(v2)["first"], x(v2)["second"])
    assert x(v1)["first"] < PC.XPRESS < x(v1)["second"]          # left chute, right chute
    far = next(t for t in v0.black if t.name == "far")
    first = next(t for t in v1.black if t.name == "first")
    # the drawn extents (card 1c; the Lua header's CLEARANCES): a positive gap, never
    # equality, from the first struck stone standing to the far stone's bare place ...
    bare = PC.place(far)
    (stand,) = PC.standing(first)
    assert stand[0] - bare[1] > 0
    # ... and, in every state the table can reach, from each stone standing or
    # reflected to the press's legs, body and chutes and to every other stone, and from
    # each to the slab's ends
    states = [PC.table_state(o, F) for o in (
        "", "read near crosswise high", "read near plain low", "read near crosswise high, far into the hopper",
        STRIKE, STRIKE + ", read first plain low", RUN, "far into the hopper, strike",
        "far into the hopper, strike, read second plain high")]
    for T in states:
        check_clearances(T)
    # the numbers the Lua header states, reproduced from the transcription's extents:
    # each place within 0.005 (the header's rounding), each gap never overstated
    lua = LUA.read_text(encoding="utf-8")
    head = " ".join(l.lstrip("- ") for l in lua[lua.index("CLEARANCES"):lua.index("On the table's edge")].splitlines())
    said = lambda pat: tuple(float(v) for v in re.search(pat, head).groups())

    def stated(pat, places, gaps):
        got = said(pat)
        assert len(got) == len(places) + len(gaps), got
        assert all(abs(g - v) <= 0.005 + 1e-9 for g, v in zip(got, places)), (got, places)
        assert all(-1e-9 <= v - g < 0.01 for g, v in zip(got[len(places):], gaps)), (got, gaps)

    press = next(t for t in v1.black if t.name == "press")
    second = next(t for t in v1.black if t.name == "second")
    (r1,), (r2,) = PC.reflection(first), PC.reflection(second)
    rlegs, rbody, rchutes = PC.reflection(press)
    got = said(r"first struck stone, (\S+) \.\. (\S+) standing, lands (\S+) clear of the far stone's "
               r"bare place \((\S+) \.\. (\S+)\)")
    assert all(abs(g - v) <= 0.005 + 1e-9 for g, v in zip(got, (stand[0], stand[2], stand[0] - bare[1], *bare)))
    stated(r"its reflection \((\S+) \.\. (\S+)\) ends (\S+) short of the press's reflected legs and "
           r"(\S+) above", (r1[0], r1[2]), (rlegs[0] - r1[2], min(PC.gap(r1, rbody), PC.gap(r1, rchutes))))
    stated(r"the second's reflection \((\S+) \.\. (\S+)\) is (\S+) clear of the legs, (\S+) above the "
           r"chutes and (\S+) inside the slab's edge", (r2[0], r2[2]),
           (PC.gap(r2, rlegs), PC.gap(r2, rchutes), PC.SX1 - r2[2]))
    assert PC.XFAR > sum(PC.SLAB_ENDS) / 2                       # the far stone in the right half
    assert PC.XFAR - PC.XNEAR >= 2 * (PC.XPRESS - PC.XFAR)
    (_, xn, fn), (_, xa, fa), (_, xb, fb) = v2.counters
    assert (fn, fa, fb) == ("up", "down", "down") and xn == PC.XNEAR
    assert xa + PC.CRX < xb - PC.CRX                             # the far two side by side ...
    edge = [(PC.XTRAY - PC.TRW, PC.XTRAY + PC.TRW), (v2.viewer[1] - PC.VHALF, v2.viewer[1] + PC.VHALF)]
    assert not any(lo < xb and hi > xa for lo, hi in edge)       # ... nothing between them
    assert v2.viewer[:2] == ("far", x(v2)["second"])             # the viewer ends at the second
    assert v2.viewer[1] - PC.VHALF > xb + PC.CRX                 # clear of the counters
    assert v1.viewer == ("window", PC.XVIEW["window"], "crosswise", "high")
    assert v1.counters == (("near", PC.XNEAR, "up"),) and v1.tray == 2

    # the reflection rule on every stone the pictures (and the full run) show
    for T in (fresh.state, t51.state, run.state):
        check_reflections(T)
    near_x = PC.Reading("crosswise", "high")
    # each struck stone at cross: 1/2 I on EACH near outcome, not only on average
    for k in RESULT:
        for s in ("first", "second"):
            rho, p = given(t51.state, s, {"near": PC.Reading("crosswise", k)})
            assert np.isclose(p, 0.5) and np.allclose(rho, I2 / 2)
    # at plain (not pictured): 1/2 I averaged over the near record, pure given it -- the
    # rule takes the average, and the struck stones are drawn slack
    plain = PC.presstable("read near plain high, far into the hopper, strike", F).state
    check_reflections(plain)
    for s in ("first", "second"):
        assert np.allclose(averaged(plain, s), I2 / 2)
        rho, _ = given(plain, s, {"near": PC.Reading("plain", "high")})
        assert np.isclose(purity(rho), 1) and np.allclose(rho, dm(ket(0)))
    # the far stone after the near reading (state model 2c): slack, and not pictured
    early = {s: PC.presstable(f"read near {s} high", F).state for s in SETTING}
    assert PC.table_dump(early["crosswise"]).split("; ")[1] == "far restless slack"
    cond = {}
    for s, T in early.items():
        assert np.allclose(averaged(T, "far"), I2 / 2)           # the same at both settings
        cond[s], _ = given(T, "far", readings(T))
        assert np.isclose(purity(cond[s]), 1)                    # pure given the near record ...
    assert not np.allclose(cond["plain"], cond["crosswise"])     # ... and setting-dependent
    assert np.allclose(cond["crosswise"], dm(PLUS)) and np.allclose(cond["plain"], dm(ket(0)))
    # the run: near high at cross, both struck stones low -- a quarter
    rho, p = given(run.state, "near", readings(run.state))
    assert np.isclose(p, 0.25)
    real = {s: telegraph(s, lambda v: DELTA @ v) for s in SETTINGS}
    m, z1, z2 = (RESULT[readings(run.state)[k].result] for k in ("near", "first", "second"))
    assert np.isclose(real[SETTING[near_x.setting]][m, z1, z2], p)
    for s in SETTINGS:                                           # the far counters never disagree
        assert np.isclose(real[s][:, 0, 1].sum() + real[s][:, 1, 0].sum(), 0)

    # P5.2: the plain reading rings, keyed; one green ring holding every counter's cord
    assert c52.labels == ("ring", "counter's cord")
    before = text[:text.index(p52.text)].lower()
    for w in c52.labels:
        assert w in before, w                                    # named before it is keyed
    assert c52.dump.startswith("scene=reading setting=plain slid=true apart=false")
    rings, links = hanging(c52.cloth)
    assert list(rings.values()) == [("green", 3, {"m", "z1", "z2"})] and links == []
    # P5.3|P5.4: at cross, joined, then drawn apart (L5: the red ring moves left)
    rings3, links3 = hanging(c53.cloth)
    rings4, links4 = hanging(c54.cloth)
    assert rings3 == {1: ("red", PC.REDJOINED, {"m"}), 2: ("green", PC.XGREEN, {"z1", "z2"})}
    assert links3 == [{1, 2}]                                    # the stone's cord and its reflection's
    assert rings4 == {1: ("red", PC.REDAPART, {"m"}), 2: ("green", PC.XGREEN, {"z1", "z2"})}
    assert links4 == [] and PC.REDAPART < PC.REDJOINED
    assert c54.dump.endswith("1:ring red 0 @0.2,0; 2:ring green 0 @4.8,0; cord fold-1 [-0.5,0.45 0.2,0]; "
                             "cord fold-2 [5.3,0.45 4.8,0]; cord fold-2 [5.9,0.45 4.8,0]")
    assert c52.frame == c53.frame == c54.frame
    # the ties: each picture's pieces against the blackboard's distribution
    pieces = lambda rs: [held for _, _, held in rs.values()]
    assert pieces_hold(real["plain"], pieces(rings)) and not pieces_hold(real["plain"], pieces(rings4))
    assert pieces_hold(real["cross"], pieces(rings4)) and not pieces_hold(real["cross"], pieces(rings))
    # the joined picture is the same distribution: the link between the colours falls
    # away (the sketchbook's spider_simp), it carries no correlation
    assert pieces_hold(real["cross"], pieces(rings3))


# --- pictures (hand-built doubled diagrams; pyzx 0.10.6 refuses ground vertices) -----

def master(setting, rings=False):
    """The Forbidden Telegraph in unfolded doubled form.
    Vertices: the cup (a, b) and the mirror cup (abar, bbar); the near read-spider r_m on
    a, abar; the copier c on b and the mirror copier cbar on bbar (delta is real, so its
    conjugate copy is the green fork again); the far read-spiders r1 on (b1, b1bar) and
    r2 on (b2, b2bar); output boundaries m, z1, z2.
    At cross, a and abar carry a Hadamard box each: a Hadamard EDGE is one line with a
    box on it.  With rings=True the cross reader is drawn as the Tortoise's cloth draws
    it, a RED dot, its Hadamards moved by colour change onto the record m.
    Scalar: the bare picture is 2 x the book's (the doubled pair is 1/2 cup (x) cup and
    the cups are unnormalized, the copiers and read-spiders exact): add_power(-2)."""
    cross = setting == "cross"
    g = zx.Graph()
    V = g.add_vertex
    cup, cupb = V(VT.Z, 1, 0), V(VT.Z, 2, 0)
    r_m = V(VT.X if (rings and cross) else VT.Z, 0.5, 2)
    c, cb = V(VT.Z, 1, 2), V(VT.Z, 2, 2)
    r1, r2 = V(VT.Z, 1, 4), V(VT.Z, 2, 4)
    om, oz1, oz2 = V(VT.BOUNDARY, 0, 6), V(VT.BOUNDARY, 1, 6), V(VT.BOUNDARY, 2, 6)
    q = ET.HADAMARD if (cross and not rings) else ET.SIMPLE
    k = ET.HADAMARD if (cross and rings) else ET.SIMPLE
    g.add_edge((cup, r_m), q)    # 1  a     : near stone, ket copy       [H at cross]
    g.add_edge((cupb, r_m), q)   # 2  abar  : near stone, mirror copy    [H at cross]
    g.add_edge((cup, c))         # 3  b     : far stone, ket copy -> copier
    g.add_edge((cupb, cb))       # 4  bbar  : far stone, mirror copy -> mirror copier
    g.add_edge((r_m, om), k)     # 5  m     : CLASSICAL, the near record
    g.add_edge((c, r1))          # 6  b1    : first struck stone
    g.add_edge((cb, r1))         # 7  b1bar : its reflection
    g.add_edge((c, r2))          # 8  b2    : second struck stone
    g.add_edge((cb, r2))         # 9  b2bar : its reflection
    g.add_edge((r1, oz1))        # 10 z1    : CLASSICAL, first far record
    g.add_edge((r2, oz2))        # 11 z2    : CLASSICAL, second far record
    g.set_inputs(()); g.set_outputs((om, oz1, oz2))
    g.scalar.add_power(-2)
    return g


def expected(setting):
    """The right-hand sides of Result 11.7, drawn.  plain: 1/2 times a green dot with the
    three legs m, z1, z2 (1/2 sum_i |iii>) -> add_power(-2).  cross: 1/4 times a green
    point on m (|0>+|1>) beside a bend on z1, z2 (|00>+|11>) -> add_power(-4)."""
    g = zx.Graph()
    om, oz1, oz2 = (g.add_vertex(VT.BOUNDARY, r, 2) for r in range(3))
    if setting == "plain":
        z = g.add_vertex(VT.Z, 1, 1)
        for o in (om, oz1, oz2):
            g.add_edge((z, o))
        g.scalar.add_power(-2)
    else:
        pt, bend = g.add_vertex(VT.Z, 0, 1), g.add_vertex(VT.Z, 1.5, 1)
        g.add_edge((pt, om)); g.add_edge((bend, oz1)); g.add_edge((bend, oz2))
        g.scalar.add_power(-4)
    g.set_inputs(()); g.set_outputs((om, oz1, oz2))
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
    want = {"plain": 0.5 * np.einsum('mz,zw->mzw', KD, KD),
            "cross": 0.25 * np.einsum('m,zw->mzw', np.ones(2), KD)}

    for s in ("plain", "cross"):
        g = master(s)
        # (a) the VALUE: exactly the joint distribution of the three records
        assert np.allclose(matrix_of(g).reshape(2, 2, 2).real, want[s])
        assert np.allclose(matrix_of(g).imag, 0)
        assert same_picture(g, expected(s))
        # (b) the LINES: eleven edges -- eight doubled quantum lines (a, abar, b, bbar,
        # b1, b1bar, b2, b2bar) and three classical (r_m-m, r1-z1, r2-z2)
        assert g.num_edges() == 11 and len(g.outputs()) == 3

    # the far marginal is the same picture at both settings: forget m (a one-legged
    # phase-free green effect on the record) and the two masters agree
    def forget_m(g):
        f = zx.Graph()
        i0, i1, i2 = (f.add_vertex(VT.BOUNDARY, r, 0) for r in range(3))
        dot = f.add_vertex(VT.Z, 0, 1)
        o1, o2 = f.add_vertex(VT.BOUNDARY, 1, 2), f.add_vertex(VT.BOUNDARY, 2, 2)
        f.add_edge((i0, dot)); f.add_edge((i1, o1)); f.add_edge((i2, o2))
        f.set_inputs((i0, i1, i2)); f.set_outputs((o1, o2))
        h = g.copy(); h.compose(f)
        return h
    far = {s: forget_m(master(s)) for s in ("plain", "cross")}
    assert np.allclose(matrix_of(far["plain"]).reshape(2, 2).real, 0.5 * KD)
    assert same_picture(far["plain"], far["cross"])          # no signal, as pictures
    h = far["cross"].copy(); simp.full_reduce(h)             # derived: 1/2 cup on z1, z2
    assert same_picture(h, far["plain"])

    # (c) DERIVATION: full_reduce, scalar kept.  At cross the near record m has NO path
    # to z1, z2 (a cut through empty paper); at plain all three hang on one green dot.
    for s in ("plain", "cross"):
        h = master(s).copy(); simp.full_reduce(h)
        assert same_picture(h, expected(s))
        m, z1, z2 = h.outputs()
        assert connected(h, z1, z2)                          # the struck pair always agree
        if s == "cross":
            assert not connected(h, m, z1) and not connected(h, m, z2)
            assert h.num_vertices() == 4 and h.num_edges() == 2   # a point on m; a bend on z1 z2
        else:
            assert connected(h, m, z1) and connected(h, m, z2)
            assert h.num_vertices() == 4 and h.num_edges() == 3   # one green dot, three legs

    # (c') the cloth's picture, the reading rings: the cross reader drawn RED (colour
    # change moves its Hadamards onto m) is the same picture ...
    assert same_picture(master("cross", rings=True), master("cross"))
    assert same_picture(master("plain", rings=True), master("plain"))
    # ... and fusion alone (spider_simp; PyZX removes the two parallel edges between a
    # red and a green dot by Hopf, scalar kept) lays out what the Tortoise showed
    for s in ("plain", "cross"):
        h = master(s, rings=True).copy()
        assert simp.spider_simp(h)
        assert same_picture(h, master(s))
        m, z1, z2 = h.outputs()
        inner = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
        if s == "plain":
            # green meets green: one green ring with every counter's cord
            assert len(inner) == 1 and h.type(inner[0]) == VT.Z
            assert set(h.neighbors(inner[0])) == {m, z1, z2}
        else:
            # the red ring lies apart with the near cord; the far green ring holds both
            assert len(inner) == 2
            red = next(v for v in inner if h.type(v) == VT.X)
            green = next(v for v in inner if h.type(v) == VT.Z)
            assert set(h.neighbors(red)) == {m} and set(h.neighbors(green)) == {z1, z2}
            assert not connected(h, m, z1) and not connected(h, m, z2)

    # (d) the copier on one half of a cup: the three-legged green dot, |000> + |111>
    g = zx.Graph()
    bend, cp = g.add_vertex(VT.Z, 0.5, 0), g.add_vertex(VT.Z, 1.5, 1)
    oa, o1, o2 = (g.add_vertex(VT.BOUNDARY, r, 2) for r in range(3))
    g.add_edge((bend, oa)); g.add_edge((bend, cp)); g.add_edge((cp, o1)); g.add_edge((cp, o2))
    g.set_inputs(()); g.set_outputs((oa, o1, o2))
    three = zx.Graph()
    z = three.add_vertex(VT.Z, 1, 1)
    ends = [three.add_vertex(VT.BOUNDARY, r, 2) for r in range(3)]
    for e in ends:
        three.add_edge((z, e))
    three.set_inputs(()); three.set_outputs(tuple(ends))
    assert np.allclose(matrix_of(g).ravel(), ket(0, 0, 0) + ket(1, 1, 1))
    assert same_picture(g, three)
    h = g.copy(); simp.full_reduce(h)                        # derived: fusion
    assert same_picture(h, three) and h.num_vertices() == 4


if __name__ == "__main__":
    verdict("ForbiddenTelegraph", blackboard, sketchbook)
