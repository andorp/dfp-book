"""Challenge V: The Mended Junction  (entanglement swapping, corrected; the master)
================================================================================
"Nothing crossed the room but a letter."  -- Achilles

Claim: send half of a pair by teleportation and, on EVERY one of the four answers,
the corrected far ends are exactly Phi+:
(I_d (x) <B_k|_{ab} (x) (Z^m1 X^m2)_c)(Phi+_{da} (x) Phi+_{bc}) = 1/2 Phi+_{dc}
(uncorrected: 1/2 B_k, the very Bell state the reading named).  In doubled form
with the record kept, the ten-line master evaluates exactly to
1/4 sum_k |m2 m1><m2 m1|_{xz} (x) |Phi+><Phi+|_{dc}; the record forgotten, to
|Phi+><Phi+|; uncorrected and forgotten, to 1/4 I -- two strangers' coins.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from dataclasses import replace

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import BELL, CUP, I2, Picture, kron, ket, matrix_of, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch10 as PC
import props_cloth as CLOTH


X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
KS = [(0, 0), (0, 1), (1, 0), (1, 1)]
I4 = np.eye(4, dtype=complex)


def P(M, m):
    return np.linalg.matrix_power(M, m)


def bell(m1, m2):
    """|B_{m1m2}> = CNOT (H (x) I)|m1 m2>."""
    return (CNOT @ kron(H, I2) @ ket(m1, m2).reshape(4, 1)).ravel()


def correction(m1, m2):
    """Flip first, then spin: Z^m1 X^m2."""
    return P(Z, m1) @ P(X, m2)


def record(m1, m2):
    """The kept record |m2 m1><m2 m1| on the classical wires x (flip bit), z (turn bit)."""
    r = ket(m2, m1)
    return np.outer(r, r)


def blackboard():
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))       # cup unnormalized; Phi+ = cup/sqrt2
    Phi = np.outer(BELL, BELL.conj())
    two_pairs = kron(BELL, BELL).ravel()                     # wires d, a, b, c (top first)

    # per branch: corrected -> 1/2 Phi+ on (d, c); uncorrected -> 1/2 B_k, the named one
    for m1, m2 in KS:
        read = bell(m1, m2).conj().reshape(1, 4)
        corrected = (kron(I2, read, correction(m1, m2)) @ two_pairs).ravel()
        uncorrected = (kron(I2, read, I2) @ two_pairs).ravel()
        assert np.allclose(corrected, 0.5 * BELL)
        assert np.isclose(np.linalg.norm(corrected) ** 2, 0.25)           # a quarter each
        assert np.allclose(uncorrected, 0.5 * bell(m1, m2))
        if (m1, m2) != (0, 0):
            assert not np.allclose(uncorrected, 0.5 * BELL)                # Chapter 6: 1 in 4

    # the doubled picture on the blackboard: sum over branches of K rho K^dagger, tagged
    # by the kept record
    R = np.outer(two_pairs, two_pairs.conj())
    full = np.zeros((16, 16), dtype=complex)                 # record (x, z) (x) pair (d, c)
    full_unc = np.zeros((16, 16), dtype=complex)
    for m1, m2 in KS:
        read = bell(m1, m2).conj().reshape(1, 4)
        K = kron(I2, read, correction(m1, m2))
        Ku = kron(I2, read, I2)
        full += np.kron(record(m1, m2), K @ R @ K.conj().T)
        full_unc += np.kron(record(m1, m2), Ku @ R @ Ku.conj().T)
    # Result 10.8: the record a fair four-way toss, the pair exactly Phi+, independent
    assert np.allclose(full, np.kron(I4 / 4, Phi))
    rec = np.einsum('xpyp->xy', full.reshape(4, 4, 4, 4))    # the record alone
    pair = np.einsum('kxky->xy', full.reshape(4, 4, 4, 4))   # the pair, record forgotten
    assert np.allclose(rec, I4 / 4) and np.allclose(pair, Phi)
    assert np.allclose(full, np.kron(rec, pair))             # independent: a product
    # uncorrected, record kept: each record value carries its own Bell pair -- not Phi+
    for m1, m2 in KS:
        blk = full_unc.reshape(4, 4, 4, 4)[2 * m2 + m1, :, 2 * m2 + m1, :]
        assert np.allclose(blk, 0.25 * np.outer(bell(m1, m2), bell(m1, m2).conj()))
    assert not np.allclose(full_unc, np.kron(I4 / 4, Phi))
    # uncorrected and forgotten: 1/4 I, two uncorrelated coins, no pair at all
    pair_unc = np.einsum('kxky->xy', full_unc.reshape(4, 4, 4, 4))
    assert np.allclose(pair_unc, I4 / 4)
    assert np.allclose(pair_unc, np.kron(I2 / 2, I2 / 2))

    # the headline, bent (map-state duality): the junction is teleportation of half a
    # pair.  Push the Phi+ on (d, a) through Result 10.5's channel on a, and the pair
    # that comes out on (d, c) is the channel's bent form -- here the identity's, Phi+.
    Phi_da = Phi
    out = np.zeros((4, 4), dtype=complex)
    for m1, m2 in KS:
        T = correction(m1, m2) @ kron(bell(m1, m2).conj().reshape(1, 4), I2)   # a,(b,c) -> c
        K = kron(I2, T)                                                        # d along for the ride
        out += K @ np.kron(Phi_da, Phi) @ K.conj().T
    assert np.allclose(out, Phi)

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws, in one motion, the mirror-table as it
# lay all morning (keyed: the keepsake under its small cloth, the message stone,
# Alice's, and far off Bob's, each with its reflection, the drawing-cloth along
# the front), the glass chimed with its trays tipped and both tokens up in the
# dish, and the table read (the message stone and Alice's gone into the glass)
# -- round 4: the glass and the table each computed as a step, marks off (the
# block holds the ten, and a counted frame carries no mark); then the letter
# obeyed as a strip of three -- flip and spin (an arc each; the stone looks
# exactly as before), the tokens tipped out before Bob's stone (the envelope's
# ghost) --
#   \mirrortable[operations; step=...; marks=off]{scene}{caption},
#   \pairingglass[step=...; marks=off]{state}{caption},
#   \tablesteps[after=...; from=off]{scene}{steps}{captions}   (pdf/props/ch10.tex)
# tools/props_ch10.py transcribes the Lua (with L1 and L2; proved against
# texlua); the calls are read out of dialog.md, never copied.  This file also
# checks the whole dialogue: the nineteen calls and twenty-three frames, the
# keyed rule (true frames only), the predictions, the marks, the log, the
# frames, the places, the Lua's seventy-three faults, that nothing is pictured
# after the letter is obeyed, and that the six "Let Bob draw" hooks still spell
# C-O-E-C-K-E.
#
# The state model (identity card, section 3, state model 3): every stone on the
# table is half of a pair, so its state alone is 1/2 I and its reflection is
# drawn slack -- at the opening (the keepsake and the message stone one pair, d
# and a; Alice's and Bob's the other, b and c), after the reading on each of the
# four answers (the keepsake and Bob's stone then the named Bell pair), and after
# the corrections (Phi+).  A flip or a spin changes the pair, never one stone's
# own state -- so the turned stone's picture is unchanged.  A structural check
# against this file's blackboard, not a second proof.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = "Nothing crossed the room but a letter."
SIZES = {"First": [1, 1, 1, 1], "Second": [1, 2], "Third": [1, 1, 3], "Fourth": [1, 1, 1],
         "Fifth": [3, 1]}                                        # calls per block (card section 4)
FRAMES = {"First": [1, 1, 1, 1], "Second": [1, 3], "Third": [1, 1, 4], "Fourth": [1, 1, 1],
          "Fifth": [3, 3]}                                       # frames per block (a strip: one per step)
# round 4, T2 and T3: the turned stone's sides, named (card section 3e.3)
EDITS = [("what lay uppermost goes down against the black, and what lay",
          "what lay uppermost goes down against the black, its crown tipping over towards the far end, and what lay"),
         ("the side that faced the window now faces the room, and its top",
          "the side that faced the window now faces the room, coming round on the company's side, and its top")]
LUA = pathlib.Path(__file__).resolve().parents[3] / "pdf" / "props" / "ch10.lua"
W, WC = "the last two pairs", "the last two pairs, with the cloth"


def halves(state, n):
    """The one-stone states of an n-stone pure state (top wire first)."""
    t = state.reshape([2] * n)
    out = []
    for i in range(n):
        m = np.moveaxis(t, i, 0).reshape(2, -1)
        out.append(m @ m.conj().T)
    return out


def check_faults():
    """Every one of the Lua's seventy-three fault sites, each raised by a call of its own
    and by nothing else."""
    R, T, G, K, S = PC.ringstring, PC.mirrortable, PC.pairingglass, PC.twotokens, PC.steps
    TL = "the tale, both up"
    bad = [
        # the bracket (round 4)
        ("opt-unknown", R, "colour=red", TL),
        ("opt-twice", R, "step=pull; step=pull", TL),
        ("opt-bare", R, "pull; corrections", TL),
        ("onoff", R, "step=pull; marks=maybe", TL),
        ("marks-step", R, "marks=on", TL),
        ("expects-keyed", R, "step=pull; expects=apart", TL, True),
        ("expects-marks", R, "step=pull; expects=apart; marks=on", TL),
        # the token pattern
        ("pattern-word", K, "", "brass up, copper sideways"),
        ("pattern-twice", K, "", "brass up, brass down"),
        ("pattern-none", K, "", ""),
        # the glass, and its step
        ("glass-cups-twice", G, "", "cups empty, stones in the cups, both up"),
        ("glass-left-twice", G, "", "cups empty, message stone in the left cup"),
        ("glass-tokens-twice", G, "", "tokens in the trays, tokens away"),
        ("glass-word", G, "", "cups full"),
        ("glass-pattern-twice", G, "", "tokens away, brass up"),
        ("glass-wrapped", G, "", "wrapped, cups empty"),
        ("glass-full", G, "", "stones in the cups"),
        ("chime-word", G, "step=ring both up", "message stone in the left cup"),
        ("chime-early", G, "step=chime both up", "tokens in the trays"),
        ("glass-bare", G, "dish", "tokens in the trays"),
        # the tokens, and their step
        ("tokens-place", K, "envelope", "both up"),
        ("pin-step", K, "pin=off", "both up"),
        ("turn-word", K, "step=flip brass", "both up"),
        # the table
        ("table-scene", T, "", "the slab"),
        ("glass-again", T, "into the glass, into the glass", WC),
        ("letter-early", T, "letter both up", WC),
        ("letter-cloth", T, "into the glass, letter both up", W),
        ("letter-twice", T, "into the glass, letter both up, letter both up", WC),
        ("turn-which", T, "into the glass, letter both up, flip the keepsake", WC),
        ("turn-early", T, "into the glass, flip", WC),
        ("turn-tipped", T, "into the glass, letter both down, tip out, flip", WC),
        ("flip-brass-down", T, "into the glass, letter copper up, flip", WC),
        ("flip-twice", T, "into the glass, letter both up, flip, flip", WC),
        ("flip-after-spin", T, "into the glass, letter both up, spin, flip", WC),
        ("spin-copper-down", T, "into the glass, letter brass up, spin", WC),
        ("spin-twice", T, "into the glass, letter both up, flip, spin, spin", WC),
        ("tip-none", T, "into the glass, tip out", WC),
        ("tip-twice", T, "into the glass, letter both down, tip out, tip out", WC),
        ("tip-unobeyed", T, "into the glass, letter both up, flip, tip out", WC),
        ("table-op", T, "look at bob's stone", WC),
        # the string
        ("scene-words", R, "", "the four bends, both up"),
        ("loop-twice", R, "", "the loop, writing both up, writing both down, answer both up"),
        ("loop-stray", R, "", "the loop, brass up, writing both up, answer both up"),
        ("loop-missing", R, "", "the loop, writing both up"),
        ("string-scene", R, "", "the knot"),
        ("bends-op", R, "pull", "the four bends"),
        ("bends-round", R, "slide the flip round, slide the flip round", "the four bends"),
        ("pull-twice", R, "pull, pull", TL),
        ("halfway-again", R, "pull; step=pull halfway", TL),
        ("corrections-early", R, "corrections", TL),
        ("corrections-twice", R, "pull, corrections, corrections", TL),
        ("meet-early", R, "kin to kin", TL),
        ("tale-op", R, "slide round", TL),
        ("slid-twice", R, "slide round, slide round", "the loop, writing copper up, answer brass up"),
        ("slide-count", R, "slide round", "the loop, writing copper up, answer both up"),
        ("slide-room", R, "slide round", "the loop, writing brass up, answer copper up"),
        ("slide-rest", R, "green meets green; step=slide round", "the loop, writing copper up, answer both up"),
        ("loop-op", R, "pull", "the loop, writing copper up, answer copper up"),
        ("end-op", R, "pull", "end to end"),
        ("lift-none", R, "pull; step=lift off", TL),
        ("meet-rest", R, "pull, corrections, red meets red; step=green meets green", TL),
        ("kin-none", R, "pull, kin to kin", "the tale, both down"),
        ("meet-few", R, "pull, corrections, red to red", "the tale, copper up"),
        ("meet-blocked", R, "pull, corrections spin first, red to red", TL),
        # the wrong rules
        ("wrong-apart", R, "step=pull halfway; expects=apart", TL),
        ("wrong-more-turned", R, "pull; step=corrections; expects=more turned", TL),
        ("wrong-either-way", R, "pull, corrections; step=kin to kin; expects=either way", TL),
        ("string-rule", R, "step=pull; expects=less turned", TL),
        ("rule-step", R, "expects=apart", TL),
        # the strips: (macro, keyed, scene, options, steps, captions)
        ("strip-bare", S, "ringstring", False, TL, "pull", "pull halfway", "a; b"),
        ("strip-none", S, "ringstring", False, TL, "", "", "a"),
        ("strip-four", S, "ringstring", False, TL, "", "pull halfway; pull; corrections; red meets red",
         "a; b; c; d; e"),
        ("strip-captions", S, "ringstring", False, TL, "", "pull", "a"),
    ]
    for site, fn, *args in bad:
        assert PC.faults(fn, *args) == [site], (site, PC.faults(fn, *args))
    assert [b[0] for b in bad] == list(PC.ERROR_SITES) and len(set(PC.ERROR_SITES)) == 73
    assert set(PC.ROUND3_SITES) < set(PC.ERROR_SITES) and len(PC.ROUND3_SITES) == 47   # round 3's keys kept
    # the two "given twice" messages are one text at two sites, told apart by their keys
    assert PC.messages(G, "", "cups empty, stones in the cups")[:1] == PC.messages(
        G, "", "cups empty, message stone in the left cup") == ["pairingglass: the cups are given twice"]
    # the cascades: a bad token word and no good one leaves nothing said; both cups words
    # leave the later one, full, with the tokens still in the trays
    assert PC.faults(G, "", "brass sideways") == ["pattern-word", "pattern-none"]
    assert PC.faults(G, "", "cups empty, stones in the cups") == ["glass-cups-twice", "glass-full"]
    assert PC.faults(T, "into the glass, letter", WC) == ["pattern-none"]
    # the Lua's quirks, transcribed: 'neither' lays both down; 'bobs stone' is Bob's
    assert PC.pattern("neither up", "x") == PC.pattern("neither down", "x") == PC.Pattern(False, False)
    assert PC.faults(T, "into the glass, letter both up, flip bobs stone, spin bob's stone, tip out", WC) == []
    # the bracket is order-free: any order of its items draws one picture (and a key's
    # case and the white space round '=' do not matter)
    import itertools
    for items, scene, fn in (
            (["pull, corrections", "step=kin to kin", "expects=more turned"], TL, R),
            (["pull", "step=corrections", "marks=off"], TL, R),
            (["step=turn brass", "marks=off", "pin=off"], "both down", K),
            (["into the glass, letter both up", "step=flip", "marks=off"], WC, T),
            (["step=chime both up", "marks=off"], "message stone in the left cup", G)):
        frames = {PC.frame_of(fn("; ".join(o), scene)) for o in itertools.permutations(items)}
        assert len(frames) == 1, items
    assert PC.frame_of(R(" Step = PULL ;Marks= Off", TL)) == PC.frame_of(R("step=pull; marks=off", TL))
    for opts in itertools.permutations(["after=pull, corrections", "from=off", "marks=off"]):
        assert PC.strip_frames("ringstring", "; ".join(opts), TL, "red meets red; lift off")[0] == [
            ("pull, corrections; step=red meets red; marks=off", TL),
            ("pull, corrections, red meets red; step=lift off; marks=off", TL)]


def check_lua():
    """The transcription's constants are the Lua's (a drift guard: the places after L2,
    the frames, the reflection geometry, round 4's marks); the Lua and the .tex are the
    ones the transcription was proved on (PC.check_lua: every chunk pinned); and the Lua
    has as many fault sites as ERROR_SITES -- its direct tex.error calls and its serr
    calls, the definition aside."""
    PC.check_lua()
    PC.check_sites()
    lua = LUA.read_text(encoding="utf-8")
    nums = lambda pat: tuple(float(x) for x in re.search(pat, lua).groups())
    assert nums(r"local XKEEP, XMSG, XALICE, XBOB = (\S+), (\S+), (\S+), (\S+)\n") == \
        (PC.XKEEP, PC.XMSG, PC.XALICE, PC.XBOB) == (-1.0, -0.6, -0.11, 0.88)
    assert nums(r"local FTOP, FBOT = (\S+), (\S+)\n") == (PC.FTOP, PC.FBOT)
    assert nums(r"local TTOP, TBOT = (\S+), (\S+)\n") == (PC.TTOP, PC.TBOT)
    assert nums(r"local SX0, SX1, STOP, SBOT = (\S+), (\S+), \S+, \S+\n") == PC.SLAB_ENDS == (-1.22, 1.22)
    assert nums(r"local SRX, SRY = (\S+), (\S+)\n") == (PC.SRX, PC.SRY)
    assert nums(r"local K = (\S+) ") == (PC.K,)
    assert nums(r"local SLACKDX = \{ (\S+), (\S+) \}") == PC.SLACKDX
    assert nums(r"local GHOST = (\S+)\n") == (PC.GHOST,) == (0.35,)          # a ghost's opacity
    assert nums(r"local PALE = (\S+)\n") == (PC.PALE,) == (0.5,)             # a prediction's
    assert nums(r"local CUPX = (\S+)\n") == (PC.CUPX,) and nums(r"local TRAYY = (\S+)\n") == (PC.TRAYY,)
    tex = LUA.with_name("ch10.tex").read_text(encoding="utf-8")
    styles = dict(re.findall(r"(propchten(?:ghost|swing|slide|pinmark))/\.style=\{([^}]*)\}", tex))
    assert styles == {"propchtenghost": "densely dotted, black!55, line width=0.45pt",
                      "propchtenswing": "densely dotted, black!50, line width=0.6pt",
                      "propchtenslide": "dotted, black!60, line width=0.5pt",           # Q8: finer than a swing
                      "propchtenpinmark": "densely dotted, black!55, line width=0.6pt"}
    # the in-frame reading widens each mark by its style's own line width, and its head by TIP's
    assert {k: float(re.search(r"line width=([\d.]+)pt", v).group(1)) for k, v in styles.items()} == PC.MARK_WIDTH
    assert PC.TIP == "-{Stealth[length=%spt,width=%spt]}" % (PC.TIP_LENGTH, PC.TIP_WIDTH)
    code = re.sub(r"--[^\n]*", "", lua)                           # comments aside
    direct = len(re.findall(r"tex\.error\(", code)) - 1           # the one inside serr
    via = len(re.findall(r"\bserr\(", code)) - 1                  # serr's own definition
    assert (direct, via) == (44, 29) and direct + via == len(PC.ERROR_SITES) == 73
    # the wrong rules' prose, as the card copies it verbatim (section 3e.1)
    card = (pathlib.Path(__file__).resolve().parents[2] / "identity.md").read_text(encoding="utf-8")
    block = re.search(r"The wrong rules, copied verbatim.*?```\n(.*?)\n```", card, re.S).group(1)
    assert block + "\n" == PC.rule_comment("THE STRING'S WRONG RULES")


def check_whole_dialogue():
    """All nineteen calls and twenty-three frames: their number and places, the keyed
    rule (true frames only), the captions, the frames, the marks, the predictions, the
    log, no fault anywhere, nothing after the letter is obeyed; and the hooks."""
    assert DIALOGUE.check_whole(PC.ORDER) == 19                  # 4 + 3 + 5 + 3 + 4
    text = DIALOGUE.read(DIALOGUE.dialog)
    assert len(DIALOGUE.call_re.findall(DIALOGUE.read(DIALOGUE.raw))) == 19   # dialog.raw.md too
    assert not DIALOGUE.call_re.findall(text[:text.index("## First Challenge")])   # none before
    DIALOGUE.keyed_first(PC.ORDER, kind=PC.keyed_kind, select=PC.judged)
    placed = [(o, i, c) for o in PC.ORDER
              for i, c in enumerate(c for b in DIALOGUE.blocks(o) for c in b)]
    assert [(o, i) for o, i, c in placed if c.keyed] == [
        ("First", 0), ("First", 1), ("First", 2), ("Fifth", 0)]  # P1.1 P1.2 P1.3 P5.1
    frames = []
    for o in PC.ORDER:
        found = PC.check_section(DIALOGUE, o, SIZES[o], FRAMES[o])    # frames, heights, captions, places
        frames += [(o, c, p, cap) for b in found for c, fr in b for p, cap in fr]
        for b in found:
            assert len(b) <= 4
    assert len(frames) == 23 and sum(PC.is_strip(c) for o, i, c in placed) == 3
    assert [(o, cap) for o, c, p, cap in frames if getattr(p, "pale", False)] == [
        ("Second", "as Achilles expects"), ("Third", "as Achilles expects")]     # apart, more turned
    assert [p.expects for o, c, p, cap in frames if getattr(p, "pale", False)] == ["apart", "more turned"]
    assert sum(PC.is_prediction(c) for o, i, c in placed) == 2
    for o, c, p, cap in frames:                                    # a prediction: pale, unkeyed, unmarked
        if getattr(p, "pale", False):
            assert p.marks == () and p.labels == () and not c.keyed
    marked = [(o, cap) for o, c, p, cap in frames if p.marks]
    assert [len([m for m in marked if m[0] == o]) for o in PC.ORDER] == [2, 2, 3, 1, 3] and len(marked) == 11
    # every marked frame's marks-off twin draws the same picture; the marks are its only addition
    for o, c, p, cap in frames:
        if p.marks:
            fc = [f for f in PC.frame_calls(c) if f.caption == cap]
            assert len(fc) == 1
            t = PC.twin(fc[0])
            assert t.marks == () and t.mark_tikz == () and PC.frame_of(t)[:3] == PC.frame_of(p)[:3]
            assert all(PC.MARK_LINE.search(l) for l in p.mark_tikz)          # every mark line is a mark
            # every ghost has a named destination: the place it went to, the ring it ran
            # into, her pouch, or Achilles's hand
            notes = [l for l in p.log if ": + ghost " in l]
            assert len(notes) == len(p.mark("ghost"))
            assert all(re.search(r": (it went to grid \d,\d|run together into the ring at rest|lifted off, "
                                 r"into her pouch|handed up to Achilles)$", l) for l in notes), notes
    # every mark lies inside its picture (drawn `overlay`, nothing else keeps it there): each
    # mark's extent -- its path, half its line width, its arrow's head -- clear of the linen's
    # edge on the cloth and of the frame's box elsewhere, by a positive gap
    clear = [PC.mark_clearance(p) for o, c, p, cap in frames if p.marks]
    assert len(clear) == 11 and min(clear) > 0.01, clear
    assert [PC.mark_clearance(p) for o, c, p, cap in frames if not p.marks] == [None] * 12
    # ... and on every step the Lua draws marks for, from the states round 4 draws and their
    # neighbours (the lower pair round its bend, the loop's short and long meetings, the tale
    # drawn in on every pattern)
    T, LP, swept, kinds = "the tale, ", "the loop, writing ", [], set()
    for fn, bracket, scene in (
            [(PC.ringstring, "step=" + st, "the four bends") for st in
             ("slide the flip round", "slide the spin round", "slide both round")] +
            [(PC.ringstring, "slide the flip round; step=slide both round", "the four bends")] +
            [(PC.ringstring, br, T + pat) for pat in ("both up", "brass up", "copper up")
             for br in ("step=pull halfway", "pull halfway; step=pull", "step=pull", "pull; step=corrections")] +
            [(PC.ringstring, "pull, corrections" + br, T + "both up") for br in
             ("; step=red to red", "; step=kin to kin", "; step=red meets red", ", red meets red; step=lift off",
              ", red to red; step=green to green", ", red meets red, lift off; step=green meets green")] +
            [(PC.ringstring, "step=" + st, LP + w + ", answer " + a_) for st in
             ("kin to kin", "green meets green", "red meets red", "red to red", "slide round")
             for w in ("copper up", "brass up", "both up") for a_ in ("copper up", "brass up", "both up")
             if PC.messages(PC.ringstring, "step=" + st, LP + w + ", answer " + a_) == []] +
            [(PC.mirrortable, "into the glass, letter both up" + br, "the last two pairs, with the cloth") for br in
             ("; step=flip", ", flip; step=spin", ", flip, spin; step=tip out")] +
            [(PC.mirrortable, "step=into the glass", sc) for sc in ("the last two pairs", W + ", with the cloth")] +
            [(PC.twotokens, br, pat) for br, pat in (("step=turn brass", "both down"), ("step=turn both", "both down"),
                                                     ("letter; step=turn both", "both down"),
                                                     ("cloth; step=turn copper", "brass up"))] +
            [(PC.pairingglass, "step=chime " + pat, "message stone in the left cup") for pat in
             ("both up", "brass up", "copper up", "both down")]):
        pic = fn(bracket, scene, False)
        swept.append(PC.mark_clearance(pic))                       # None: the step moved nothing
        assert swept[-1] is None or swept[-1] > 0.01, (bracket, scene, swept[-1])
        if "inside of the" in " ".join(pic.log):
            kinds.add("lower pair, inside its bend")
        if "the long way round" in " ".join(pic.log) and "slid to" in " ".join(pic.log):
            kinds.add("short and long meetings on the loop")
            assert "about (0.000,-0.695)" in " ".join(pic.log)        # the long arc kept lower
    assert len(swept) == 58 and len([c for c in swept if c is not None]) == 56
    assert kinds == {"lower pair, inside its bend", "short and long meetings on the loop"}
    # the log, as the card's dump list has it: 23 heads, 39 entries, 3 strip lines, 65 in
    # all, the same for both files
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        log = [l for o in PC.ORDER for b in DIALOGUE.blocks(o, path) for c in b for l in PC.describe(c)]
        assert len(log) == 65 and sum(": + " in l for l in log) == 39
        assert sum(l.endswith(" frames") and ": strip of " in l for l in log) == 3
        assert log == [l for o in PC.ORDER for b in DIALOGUE.blocks(o) for c in b for l in PC.describe(c)]
    # the Lua's round-4 comments quote the dialogue: 25 phrases, each in one challenge
    places = PC.quote_places(DIALOGUE)
    assert places == PC.quote_places(DIALOGUE, DIALOGUE.raw) and len(places) == 25
    assert [len([q for q in places if q[2] == o]) for o in PC.ORDER] == [4, 4, 9, 1, 7]
    # the header's PLACES paragraph quotes the opening direction for the keepsake's hidden
    # reflection: once in each dialogue file, before the First Challenge
    header = " ".join(LUA.read_text(encoding="utf-8").split("\n-- round 4")[0].replace("--", " ").split())
    hidden = "it stays there, hidden, its reflection hidden with it under the cloth's edge"
    assert 'is hidden with it under the cloth\'s edge ("' + hidden + '");' in header
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        whole = " ".join(path.read_text(encoding="utf-8").split())
        assert whole.count(hidden) == 1 and whole.index(hidden) < whole.index("## First Challenge")
    for o, i, c in placed:
        PC.picture(c)                                            # the macro never faults
        assert PC.place_ok(o, c), (o, c.text)
        assert len(c.args) == (2 if PC.is_strip(c) else 1) and c.args[0]   # one scene (and a strip's steps)
    # the forbidden places, as the card lists them, nowhere in the dialogue
    assert {c.macro for o, i, c in placed if o != "Fifth"} == {"pairingglass", "twotokens", "ringstring",
                                                                "stringsteps"}
    assert [o for o, i, c in placed if c.macro in ("mirrortable", "tablesteps")] == ["Fifth"] * 3
    assert {o for o, i, c in placed if c.macro == "pairingglass"} == {"First", "Fifth"}
    for bad in ("end to end", "slide both round", "corrections spin first",
                "\\twotokens[cloth]", "\\twotokens*[cloth]"):
        assert all(bad not in c.text for o, i, c in placed), bad
    assert all(not {"wrapped", "under its chamois"} & set(PC.split(PC.lower(c.args[0]), ","))
               for o, i, c in placed if c.macro == "pairingglass")        # never wrapped
    # nothing pictured after the letter is obeyed: B5.4 is the last call of the dialogue
    p54 = DIALOGUE.blocks("Fifth")[-1][-1]
    paras = [q.strip() for q in re.split(r"\n[ \t]*\n", text) if q.strip()]
    pics = [i for i, q in enumerate(paras) if DIALOGUE.call_re.search(q)]
    assert p54.text in paras[pics[-1]] and placed[-1][2] == p54
    assert not any(DIALOGUE.call_re.search(q) or "\\begin{tikzpicture}" in q for q in paras[pics[-1] + 1:])
    end_to_end = [i for i, q in enumerate(paras) if q.startswith("**Tortoise:** Let Bob draw end to end.")]
    assert len(end_to_end) == 1 and end_to_end[0] > pics[-1]     # the straight string: not drawn
    # the six hooks, in order: C-O-E-C-K-E
    hooks = re.findall(r"Let Bob draw (\w+)", text)
    assert "".join(h[0].upper() for h in hooks) == "COECKE" and len(hooks) == 6, hooks
    assert re.findall(r"Let Bob draw (\w+)", DIALOGUE.read(DIALOGUE.raw)) == hooks


def head(c):
    """A call's bracket as its log line writes it (the Lua's trim of what TeX hands it):
    the items in the order the dialogue writes them, which the picture never depends on."""
    return PC.trim(PC.bracket(c) or "")


def r3(x):
    """A mark's numbers to the Lua's three places (its log's)."""
    if isinstance(x, tuple):
        return tuple(r3(y) for y in x)
    return round(x, 3) if isinstance(x, float) else x


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    check_faults()
    check_lua()
    check_whole_dialogue()
    found = PC.check_section(DIALOGUE, "Fifth", sizes=[3, 1], frames=[3, 3])
    PC.check_edits(DIALOGUE, "Fifth", EDITS)
    calls = [c for b in found for c, _ in b]
    p51, p52, p53, st54 = calls
    first, second = DIALOGUE.blocks("Fifth")
    assert {PC.height(c) for c in first + second} == {PC.GLASS_FRAME} == {1.58}   # glass and table level
    assert [c.macro for c in calls] == ["mirrortable", "pairingglass", "mirrortable", "tablesteps"]
    assert [c.keyed for c in calls] == [True, False, False, False]
    assert not any(PC.is_prediction(c) for c in calls)
    assert [(PC.options(c), c.args[0]) for c in calls] == [
        ({"bare": ""}, WC), ({"step": "chime both up", "marks": "off", "bare": ""}, "message stone in the left cup"),
        ({"step": "into the glass", "marks": "off", "bare": ""}, WC),
        ({"after": "into the glass, letter both up", "from": "off", "bare": ""}, WC)]
    assert st54.args[1] == "flip; spin; tip out"
    assert [cap for b in found for _, fr in b for _, cap in fr] == [
        "as they lay all morning", "chimed, and tipped", "read, and Bob's untouched",
        "flipped, by feel", "spun, by feel", "tokens out, and unread"]
    b51, g52, b53 = (PC.picture(c) for c in calls[:3])
    strip = PC.picture(st54)
    (f21, _), (f22, _), (f23, _) = strip.pictures
    b54 = f23
    a1, a2 = DIALOGUE.anchors("Fifth")
    f1, f2 = DIALOGUE.following("Fifth")
    assert QUOTE in DIALOGUE.section("Fifth")                     # the docstring's line
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        here = [(n, q) for n, q, o in PC.quote_places(DIALOGUE, path) if o == "Fifth"]
        assert sorted({n for n, _ in here}) == ["THE GLASS'S STEP", "THE TABLE'S STEPS"] and len(here) == 7
        # the flip's and the spin's sides, as the arcs draw them, are the round's direction edits
        for _, new in EDITS:
            side = new.split(", ")[-2]
            assert ("THE TABLE'S STEPS", side) in here, side
    WCB = "{the last two pairs, with the cloth} | "
    assert [l for c in calls for l in PC.describe(c)] == [
        "mirrortable: [] " + WCB + PC.table_words(b51.state),
        "pairingglass: [" + head(p52) + "] {message stone in the left cup} | "
        "stones in the cups; in the dish both up; trays tipped",
        "pairingglass: + step chime both up -> stones in the cups; in the dish both up; trays tipped",
        "mirrortable: [" + head(p53) + "] " + WCB + PC.table_words(b53.state),
        "mirrortable: + step into the glass -> " + PC.table_words(b53.state),
        "tablesteps: strip of 3 frames",
        "mirrortable: [into the glass, letter both up; step=flip] " + WCB + PC.table_words(f21.state),
        "mirrortable: + step flip -> " + PC.table_words(f21.state),
        "mirrortable: + swing arc round Bob's stone, over its crown and down its right side, radius 0.25 about "
        "(0.880,0.086) (100 -> -8 degrees): flipped, top to bottom",
        "mirrortable: [into the glass, letter both up, flip; step=spin] " + WCB + PC.table_words(f22.state),
        "mirrortable: + step spin -> " + PC.table_words(f22.state),
        "mirrortable: + swing arc half round and more above Bob's stone, about (0.880,0.273) (160 -> 380 "
        "degrees, its front from the window side to the room's): spun about its upright",
        "mirrortable: [into the glass, letter both up, flip, spin; step=tip out] " + WCB + PC.table_words(f23.state),
        "mirrortable: + step tip out -> " + PC.table_words(f23.state),
        "mirrortable: + ghost envelope, open, in dotted outline at (0.880,-0.865): handed up to Achilles"]
    assert PC.table_words(b51.state) == (
        "on the black: keepsake under its cloth at -1.000 (its reflection hidden), message stone at -0.600 "
        "slack, Alice's stone at -0.110 slack, Bob's stone at 0.880 slack; drawing-cloth bare")

    # B5.1, the counted block (it holds the ten): no mark on any of its three frames, and
    # the lines each draws, counted as CLAUDE.md counts them -- 8 (four things on the
    # black, each with its reflection), 2 (the two tokens in the dish, classical, no
    # reflection; the spent stones' lines end at the glass), 4 (the keepsake and Bob's
    # stone, each with its reflection).  P5.1 and P5.2 together: the ten.
    assert [PC.lines_drawn(p) for p in (b51, g52, b53)] == [8, 2, 4]
    assert PC.lines_drawn(b51) + PC.lines_drawn(g52) == 10
    for p in (b51, g52, b53):
        assert p.marks == () and p.mark_tikz == ()                # no ghost, no arc, no pin
        assert len(p.log) == (1 if p is b51 else 2)               # the head, and the step: no mark entry
    # with its marks on, the same step would draw them: marks=off is what keeps them off
    assert [m.kind for m in PC.pairingglass("step=chime both up", "message stone in the left cup").marks] == \
        ["ghost", "ghost", "swing", "swing"]
    assert [m.kind for m in PC.mirrortable("step=into the glass", WC).marks] == \
        ["ghost", "ghost", "swing", "swing"]
    # the computed P5.2 and P5.3 are round 3's calls to the byte
    assert PC.frame_of(g52)[:3] == PC.frame_of(PC.pairingglass(None, "stones in the cups, both up"))[:3]
    assert PC.frame_of(b53) == PC.frame_of(PC.mirrortable("into the glass", WC))
    assert g52.before["cups"] == "left" and g52.before["tokens"] == "trays"   # the glass waiting
    assert PC.glass_words(g52.before) == "message stone in the left cup; tokens in the trays; trays level"
    assert b53.before.glass is False and b53.moved == "lift"

    # P5.1: the table as it lay all morning, keyed -- the keepsake under its small cloth
    # (its own reflection hidden; the cloth's, an object, tight), the message stone,
    # Alice's, and far off Bob's, every stone restless and its reflection slack; the
    # drawing-cloth bare along the front
    v51 = b51.shown
    assert [s[0] for s in v51.stones] == ["keepsake", "message", "Alice's", "Bob's"]
    keep, msg, alice, bob = v51.stones
    assert keep == ("keepsake", "cloth", PC.XKEEP, None, "tight")
    assert all((s[1], s[3], s[4]) == ("stone", "restless", "slack") for s in (msg, alice, bob))
    assert v51.cloth and v51.letter is None and v51.tokens_out is None
    assert v51.frame == PC.GLASS_FRAME and v51.ends == PC.SLAB_ENDS
    assert b51.labels == ("keepsake", "message", "Alice's", "Bob's", "reflection", "black", "drawing-cloth")
    assert "as a half of a pair's always is" in a1 and "nothing about it shows anything at all" in a1
    assert f1 == "**Tortoise:** The letter never went, and then the glass chimed."
    # L2 (a rule, not coordinates): the near group in the slab's left half, in order,
    # Alice's left of the middle; the Alice-Bob gap the widest, at least twice each near
    # gap; no two stones and no two reflections touch; Bob's reflection on the black
    xs = [s[2] for s in v51.stones]
    assert PC.SLAB_ENDS[0] < xs[0] < xs[1] < xs[2] < 0 < xs[3]
    near = [xs[1] - xs[0], xs[2] - xs[1]]
    assert round(xs[3] - xs[2], 2) >= round(2 * max(near), 2) and xs[3] - xs[2] > max(near)
    refl = [(s[2] + PC.reach(s[1], s[4])[0], s[2] + PC.reach(s[1], s[4])[1]) for s in v51.stones]
    body = [(s[2] + PC.reach(s[1], "tight")[0], s[2] + PC.reach(s[1], "tight")[1]) for s in v51.stones]
    for spans in (refl, body):
        assert all(spans[i][1] < spans[i + 1][0] for i in range(3)), spans
    assert PC.SLAB_ENDS[0] < refl[0][0] and refl[-1][1] < PC.SLAB_ENDS[1]
    assert "one stone he leaves at the near end, by the others" in DIALOGUE.read(DIALOGUE.dialog)

    # P5.2: the glass chimed -- the second stone in, the stones in its cups (spent), the
    # trays tipped, both tokens face-up in the dish; the very tokens the letter then
    # carries (B5.4)
    assert (g52.wrapped, g52.cups, g52.tokens, g52.pattern, g52.tipped, g52.labels) == \
        (False, "stones", "dish", PC.Pattern(True, True), True, ())
    assert g52.pattern == b54.state.letter
    assert "sets it in the left cup" in a1 and "the brass face-up, the copper face-up" in a1
    assert "The glass waits. He lifts Alice's stone into the right cup" in a1
    # the glass reads: its trays are tipped exactly when both cups are full
    for cups in ("cups empty", "stones in the cups"):
        for tokens in ("tokens in the trays", "both up", "brass up", "tokens away"):
            spec = cups + ", " + tokens
            if cups == "stones in the cups" and tokens == "tokens in the trays":
                assert PC.faults(PC.pairingglass, "", spec) == ["glass-full"]
                continue
            assert PC.pairingglass("", spec).tipped == (cups == "stones in the cups")
    assert PC.pairingglass("", "message stone in the left cup").tipped is False   # one stone: it waits
    # the chime happens as the second stone settles: from the glass waiting only
    assert PC.faults(PC.pairingglass, "step=chime both up", "cups empty") == ["chime-early"]

    # P5.3: read -- the message stone and Alice's gone with their reflections; the
    # keepsake and Bob's stone left exactly where and as they were (fixed parts)
    v53 = b53.shown
    assert v53.stones == (keep, bob) and v53.cloth and v53.letter is None
    assert (v53.frame, v53.ends) == (v51.frame, v51.ends)

    # B5.4: the letter obeyed, a strip of three -- flip, then spin (the state; nothing
    # shows), then the tokens tipped out before Bob's stone.  Frame k is the single call
    # [<after and the steps before>; step=<k>]; no before-frame: frame 1 without its arc is
    # that before-state to the byte (a stone carries no mark)
    fc = PC.frame_calls(st54)
    assert [c.opt for c in fc] == ["into the glass, letter both up; step=flip",
                                   "into the glass, letter both up, flip; step=spin",
                                   "into the glass, letter both up, flip, spin; step=tip out"]
    for c, (fr, _) in zip(fc, strip.pictures):
        assert PC.frame_of(PC.picture(c)) == PC.frame_of(fr)
    opened = PC.mirrortable("into the glass, letter both up", WC)
    for c in fc[:2]:
        assert PC.twin(c).shown == opened.shown and PC.twin(c).marks == ()    # flip and spin: nothing seen
    assert PC.frame_of(PC.twin(fc[2])) == PC.frame_of(
        PC.mirrortable("into the glass, letter both up, flip, spin, tip out", WC))   # round 3's P5.4
    assert f21.shown == f22.shown == opened.shown and f21.shown.letter == PC.Pattern(True, True)
    assert (f21.state.flip, f21.state.spin, f22.state.flip, f22.state.spin) == (True, False, True, True)
    assert [p.moved for p in (f21, f22, f23)] == ["flip", "spin", "tip"]
    # F21's mark: one swing arc round Bob's stone, over its crown and down its right side
    # -- the far end (T2) -- radius 0.25 about (0.88, 0.086), 100 -> -8 degrees
    arc, = f21.marks
    (cx, cy), (rx, ry, a0, a1_, b1) = arc.pts
    assert (arc.kind, arc.name) == ("swing", "flip")
    assert r3((cx, cy, rx, ry, a0, a1_)) == (0.88, 0.086, 0.25, 0.25, 100, -8)
    assert np.isclose(cy, PC.SRY * PC.K) and cx == PC.XBOB
    assert a0 > 90 > 0 > a1_ and b1 > a1_                         # over the crown, down the right (far) side
    paras = DIALOGUE.paragraphs("Fifth")
    flip_para = paras[paras.index(a2) - 1]                        # the flip's direction, before the spin's
    assert "its crown tipping over towards the far end" in flip_para and flip_para.startswith("*[Bob's stone sits")
    # F22's: half round and more above Bob's stone, about (0.88, 0.273), 160 -> 380
    # degrees, passing below its centre -- the near, company's side (T3)
    arc, = f22.marks
    (cx, cy), (rx, ry, a0, a1_, b1) = arc.pts
    assert (arc.kind, arc.name) == ("swing", "spin")
    assert r3((cx, cy, rx, ry, a0, a1_)) == (0.88, 0.273, 0.21, 0.055, 160, 380)
    assert a0 < 270 < b1 < a1_                                    # through 270: the company's side
    assert "coming round on the company's side" in a2
    # F23's: the envelope's ghost in outline (the tokens cover its place), no arc; its
    # destination, Achilles's hand, is the direction's
    ghost, = f23.marks
    assert (ghost.kind, ghost.name, ghost.how, r3(ghost.at)) == ("ghost", "envelope", "outline", (0.88, -0.865))
    assert np.isclose(ghost.at[1], (PC.BANDY0 + PC.BANDY1) / 2) and f23.mark("swing") == []
    assert "hands the empty envelope up to Achilles" in a2

    # P5.4's end state: flip, then spin (the state), the tokens tipped out before Bob's
    # stone, both up, no envelope; the stones EXACTLY P5.3's: the turn shows nothing
    T54 = b54.state
    assert (T54.glass, T54.letter, T54.flip, T54.spin, T54.tipped) == \
        (True, PC.Pattern(True, True), True, True, True)
    v54 = b54.shown
    assert v54.letter is None and v54.tokens_out == PC.Pattern(True, True)
    assert replace(v54, tokens_out=None) == v53                   # nothing else changed
    assert "with no mark on it, it looks just as it did" in a2
    assert f2 == "**Tortoise:** The glass chimed, and then the letter went."
    assert DIALOGUE.following("Fifth")[-1] == f2                  # nothing pictured after B5.4
    assert PC.lines_drawn(f23) == 4                               # the tokens on the cloth aside

    # the tie, slack <-> a half: every stone the table draws is 1/2 I, purity 1/2, at every
    # stage the plan draws.  Wires d (keepsake), a (message), b (Alice's), c (Bob's).
    two_pairs = kron(BELL, BELL).ravel()
    names = ["keepsake", "message", "Alice's", "Bob's"]
    shown51 = {s[0]: s for s in v51.stones}
    for name, rho in zip(names, halves(two_pairs, 4)):           # the opening (P5.1)
        assert np.allclose(rho, I2 / 2) and np.isclose(np.trace(rho @ rho).real, 0.5)
        assert name == "keepsake" or shown51[name][4] == "slack"  # the keepsake's cloth: an object
    m1, m2 = PC.bits(g52.pattern)                                 # both up: the answer 11
    assert (m1, m2) == (1, 1)
    for k in KS:                                                  # after the reading (P5.3), every answer
        read = bell(*k).conj().reshape(1, 4)
        left = (kron(I2, read, I2) @ two_pairs).ravel()           # the keepsake and Bob's stone
        assert np.allclose(left, 0.5 * bell(*k))                  # the named Bell pair
        for rho in halves(left / np.linalg.norm(left), 2):
            assert np.allclose(rho, I2 / 2)
    assert [s[4] for s in v53.stones if s[0] == "Bob's"] == ["slack"]
    # the letter obeyed (P5.4): flip, then spin, as the table's operations run --
    # X then Z, the correction Z^m1 X^m2 -- takes the pair from B_11 to Phi+, and Bob's
    # stone alone stays 1/2 I throughout: a turn changes the pair, not the stone
    turned = (Z if T54.spin else I2) @ (X if T54.flip else I2)
    assert np.allclose(turned, correction(m1, m2))
    pair = bell(m1, m2)
    after_flip = (kron(I2, X) @ pair).ravel()
    after_both = (kron(I2, turned) @ pair).ravel()
    assert np.allclose(after_both, BELL) and not np.allclose(pair, BELL)
    for state in (pair, after_flip, after_both):
        for rho in halves(state, 2):
            assert np.allclose(rho, I2 / 2) and np.isclose(np.trace(rho @ rho).real, 0.5)
    for R in (X, Z):
        assert np.allclose(R @ (I2 / 2) @ R.conj().T, I2 / 2)
    assert [s[4] for s in v54.stones] == ["tight", "slack"]      # the cloth, and Bob's: slack


# --- pictures -----------------------------------------------------------------

def swap_branch(m1, m2, corrected=True):
    """One branch, pure (Section 10.7): rows d=0, a=1, b=2, c=3.  Cup (d, a); along a the
    reading's rings, green m1*pi then red m2*pi, into the cap (a, b); cup (b, c); along c
    Bob's correction, red m2*pi then green m1*pi.  Outputs d, c.
    Scalar: two pairs and one reading, three factors 1/sqrt2 -> add_power(-3)."""
    p = Picture()
    cup_da, cap_ab, cup_bc = p.bend(0.5, 1), p.bend(1.5, 4), p.bend(2.5, 1)
    p.wire(p.out(0), cup_da, p.z(1, 2, m1), p.x(1, 3, m2), cap_ab)
    bob = [p.x(3, 2, m2), p.z(3, 3, m1)] if corrected else []
    p.wire(cap_ab, cup_bc, *bob, p.out(3))
    g = p.done()
    g.scalar.add_power(-3)
    return g


def half_pair(rings=()):
    """1/2 |Phi+> = cup / (2 sqrt2) on (d, c), optionally with rings on c listed outward
    from the bend.  add_power(-3) draws the 1/(2 sqrt2)."""
    p = Picture()
    bend = p.bend(1.5, 1)
    dots = [p.z(3, 2 + i, a) if c == "Z" else p.x(3, 2 + i, a) for i, (c, a) in enumerate(rings)]
    p.wire(p.out(0), bend, *dots, p.out(3))
    g = p.done()
    g.scalar.add_power(-3)
    return g


def master(corrected=True, forget=False):
    """The Mended Junction, hand-built as a doubled (CPM) diagram with its classical
    record (pyzx 0.10.6 refuses native ground vertices, so nothing here is a ground).
    Every quantum wire is drawn with its mirror copy; the Bell states, H, CNOT and the
    Paulis are real, so each mirror copy uses the same picture as its original.
      * the pairs: cups (d, a), (b, c) and their mirror cups (dbar, abar), (bbar, cbar);
      * the reading box M: CNOT a -> b on each copy; H on a and abar folded in as
        Hadamard EDGES into the read-spider r1 (the turn bit m1, classical wire z); the
        read-spider r2 on b and bbar (the flip bit m2, classical wire x);
      * the correction box C, classically controlled: r2 feeds red dots on c and cbar
        (a flip when m2 = 1), then r1 feeds green dots on c and cbar through Hadamard
        edges (a spin when m1 = 1) -- flip first, then spin, along c;
      * the record kept: r2 and r1 run out as the classical wires x and z.  With
        forget=True each ends instead in the phase-free one-legged green effect.
    Outputs, top to bottom as the chapter draws them: dbar, d, x, z, c, cbar
    (forget=True: dbar, d, c, cbar).
    Scalar compensations (the bare PyZX picture is 1/2 x the book's master):
      +2  each CNOT picture is CNOT/sqrt2, two of them;
      -4  each normalized pair carries 1/sqrt2, four cups (two pairs, doubled);
      +2  each classically controlled flip (a red dot fed by a green one) carries 1/sqrt2;
      +2  each classically controlled spin (a Hadamard edge between greens) carries 1/sqrt2.
    """
    g = zx.Graph()
    V = g.add_vertex
    E = lambda a, b, t=ET.SIMPLE: g.add_edge((a, b), edgetype=t)
    o_db, o_d = V(VT.BOUNDARY, 0, 20), V(VT.BOUNDARY, 1, 20)
    o_c, o_cb = V(VT.BOUNDARY, 4, 20), V(VT.BOUNDARY, 5, 20)
    cup_da, cup_da_b = V(VT.Z, 1.5, 1), V(VT.Z, 1, 1)        # pair (d,a) and its mirror
    cup_bc, cup_bc_b = V(VT.Z, 3.5, 1), V(VT.Z, 4, 1)        # pair (b,c) and its mirror
    ctl, ctl_b = V(VT.Z, 2, 3), V(VT.Z, 2.2, 3)              # CNOT control on a, abar
    tgt, tgt_b = V(VT.X, 3, 3), V(VT.X, 2.8, 3)              # CNOT target on b, bbar
    r1, r2 = V(VT.Z, 2.3, 6), V(VT.Z, 2.7, 6)                # read-spiders: m1 (z), m2 (x)
    E(cup_da, o_d); E(cup_da, ctl); E(cup_da_b, o_db); E(cup_da_b, ctl_b)
    E(cup_bc, tgt); E(cup_bc_b, tgt_b)
    E(ctl, tgt); E(ctl_b, tgt_b)
    E(ctl, r1, ET.HADAMARD); E(ctl_b, r1, ET.HADAMARD)       # H on a, abar; then read m1
    E(tgt, r2); E(tgt_b, r2)                                 # read m2
    if corrected:
        fx, fx_b = V(VT.X, 4, 8), V(VT.X, 5, 8)              # flip on c, cbar, fed by r2
        sz, sz_b = V(VT.Z, 4, 9), V(VT.Z, 5, 9)              # spin on c, cbar, fed by r1
        E(r2, fx); E(r2, fx_b)
        E(r1, sz, ET.HADAMARD); E(r1, sz_b, ET.HADAMARD)
        E(cup_bc, fx); E(fx, sz); E(sz, o_c)                 # along c: flip, then spin
        E(cup_bc_b, fx_b); E(fx_b, sz_b); E(sz_b, o_cb)
    else:
        E(cup_bc, o_c); E(cup_bc_b, o_cb)
    if forget:
        f1, f2 = V(VT.Z, 2.3, 7), V(VT.Z, 2.7, 7)
        E(r1, f1); E(r2, f2)
        outs = (o_db, o_d, o_c, o_cb)
    else:
        o_x, o_z = V(VT.BOUNDARY, 2, 20), V(VT.BOUNDARY, 3, 20)
        E(r2, o_x); E(r1, o_z)                               # the two classical wires
        outs = (o_db, o_d, o_x, o_z, o_c, o_cb)
    g.set_inputs(()); g.set_outputs(outs)
    g.scalar.add_power(2)                                    # two CNOT pictures
    g.scalar.add_power(-4)                                   # four cups of two normalized pairs
    if corrected:
        g.scalar.add_power(2)                                # two controlled flips
        g.scalar.add_power(2)                                # two controlled spins
    return g


def expected_master():
    """The right-hand side of Result 10.8, drawn: a cup on (d, c) and its mirror cup on
    (dbar, cbar) -- 2 |Phi+><Phi+| -- beside a phase-free one-legged green state on each
    classical wire (|0> + |1>, every record value equally).  Scalar 1/4 x 1/2 = 1/8
    -> add_power(-6).  Outputs dbar, d, x, z, c, cbar."""
    g = zx.Graph()
    V = g.add_vertex
    o_db, o_d, o_x, o_z, o_c, o_cb = (V(VT.BOUNDARY, i, 5) for i in range(6))
    dc, dc_b, sx, sz = V(VT.Z, 1, 1), V(VT.Z, 2, 1), V(VT.Z, 3, 1), V(VT.Z, 4, 1)
    for a, b in ((dc, o_d), (dc, o_c), (dc_b, o_db), (dc_b, o_cb), (sx, o_x), (sz, o_z)):
        g.add_edge((a, b))
    g.set_inputs(()); g.set_outputs((o_db, o_d, o_x, o_z, o_c, o_cb))
    g.scalar.add_power(-6)
    return g


def doubled_pair(across, power):
    """Two bends on the four outputs (dbar, d, c, cbar) = rows 0..3: across=True joins
    d-c and dbar-cbar (a doubled pair); across=False joins each wire to its own mirror
    (dbar-d, c-cbar: two discards bent round -- the maximally mixed state)."""
    p = Picture()
    if across:
        p.wire(p.out(1), p.bend(1.5, 1), p.out(2)); p.wire(p.out(0), p.bend(1.5, 0), p.out(3))
    else:
        p.wire(p.out(0), p.bend(0.5, 1), p.out(1)); p.wire(p.out(2), p.bend(2.5, 1), p.out(3))
    g = p.done()
    g.scalar.add_power(power)
    return g


def skeleton(with_box=False):
    """The master's wiring, one edge per line: M drawn as ONE vertex (four doubled lines
    in, two classical out; its insides are the value build above) and C as the identity
    (the record passing straight through to the outputs).  With with_box=True, C is drawn
    as four explicit identity dots on c, cbar, x, z, and their ids are returned."""
    g = zx.Graph()
    V = g.add_vertex
    o_db, o_d, o_x, o_z, o_c, o_cb = (V(VT.BOUNDARY, i, 9) for i in range(6))
    cup_da, cup_da_b = V(VT.Z, 1, 1), V(VT.Z, 0, 1)
    cup_bc, cup_bc_b = V(VT.Z, 4, 1), V(VT.Z, 5, 1)
    M = V(VT.Z, 2.5, 4)                                      # the reading box, as one vertex
    E = lambda a, b: g.add_edge((a, b))
    E(cup_da, o_d)          # 1  d     : keepsake half, ket copy
    E(cup_da_b, o_db)       # 2  dbar  : keepsake half, mirror copy
    E(cup_da, M)            # 3  a     : message half -> M
    E(cup_da_b, M)          # 4  abar
    E(cup_bc, M)            # 5  b     : Alice's half -> M
    E(cup_bc_b, M)          # 6  bbar
    ids = []
    if with_box:
        cc, ccb = V(VT.Z, 4, 6), V(VT.Z, 5, 6)
        cx, cz = V(VT.Z, 2, 6), V(VT.Z, 3, 6)
        ids = [cc, ccb, cx, cz]
        E(cup_bc, cc); E(cc, o_c); E(cup_bc_b, ccb); E(ccb, o_cb)
        E(M, cx); E(cx, o_x); E(M, cz); E(cz, o_z)
    else:
        E(cup_bc, o_c)      # 7  c     : Bob's half, through C
        E(cup_bc_b, o_cb)   # 8  cbar
        E(M, o_x)           # 9  x     : CLASSICAL, the flip bit m2   <-- classical wire
        E(M, o_z)           # 10 z     : CLASSICAL, the turn bit m1   <-- classical wire
    g.set_inputs(()); g.set_outputs((o_db, o_d, o_x, o_z, o_c, o_cb))
    return (g, ids) if with_box else g


def remove_identity(g, v):
    """The identity rule (Chapter 8) applied to exactly one dot: a phase-free green dot
    with two legs is a bare wire.  (pyzx 0.10.6's id_simp takes no vertex filter and
    would swallow the cups too, so the rule is applied here by splicing the dot out.)"""
    assert g.type(v) == VT.Z and g.phase(v) == 0 and len(list(g.neighbors(v))) == 2
    a, b = g.neighbors(v)
    g.remove_vertex(v)
    g.add_edge((a, b))


def sketchbook():
    # --- (a) the pure branch, and its DERIVATION: one yank, red meets red, green meets
    # green, a bare cup on (d, c) and 1/(2 sqrt2) -- that is, 1/2 Phi+ -- on every answer
    for m1, m2 in KS:
        g = swap_branch(m1, m2)
        assert np.allclose(matrix_of(g).ravel(), 0.5 * BELL)
        assert same_picture(g, half_pair())
        h = g.copy()
        simp.full_reduce(h)
        assert same_picture(h, half_pair())                   # derived, scalar kept
        # uncorrected: the cup still carries the reading's rings, in their bottom-leg
        # order (green nearer the bend, then red): 1/2 B_k, the very Bell state named
        u = swap_branch(m1, m2, corrected=False)
        assert np.allclose(matrix_of(u).ravel(), 0.5 * bell(m1, m2))
        assert same_picture(u, half_pair([("Z", m1), ("X", m2)]))
        if (m1, m2) != (0, 0):
            assert not same_picture(u, half_pair())

    # --- (b) the VALUE of the master, doubled, the record kept (Result 10.8, exact)
    Phi = np.outer(BELL, BELL.conj())
    want = np.zeros((2,) * 6, dtype=complex)                  # [dbar, d, x, z, c, cbar]
    for m1, m2 in KS:
        want[:, :, m2, m1, :, :] = 0.25 * np.einsum('dc,DC->DdcC', BELL.reshape(2, 2),
                                                     BELL.conj().reshape(2, 2))
    m = master()
    assert np.allclose(matrix_of(m).reshape((2,) * 6), want)
    assert same_picture(m, expected_master())
    # the record is a fair toss, the pair exactly Phi+ on every value, independent
    T = matrix_of(m).reshape((2,) * 6)
    for m1, m2 in KS:
        blk = np.einsum('DdcC->dcDC', T[:, :, m2, m1, :, :]).reshape(4, 4)   # pair on (d,c)
        assert np.allclose(blk, 0.25 * Phi)
    # DERIVED: the rules reduce the whole doubled master to the right-hand side
    h = master()
    simp.full_reduce(h)
    assert h.num_vertices() == 8 and h.num_edges() == 4      # six ends, two bare cups, two toss dots
    assert same_picture(h, expected_master())

    # uncorrected, record kept: each record value carries ITS OWN Bell pair -- this pins
    # the wiring of the two classical wires (a swapped x/z would fail here, though the
    # corrected value, uniform in the record, could not tell)
    U = matrix_of(master(corrected=False)).reshape((2,) * 6)
    for m1, m2 in KS:
        blk = np.einsum('DdcC->dcDC', U[:, :, m2, m1, :, :]).reshape(4, 4)
        B = bell(m1, m2)
        assert np.allclose(blk, 0.25 * np.outer(B, B.conj()))
        if (m1, m2) != (0, 0):
            assert not np.allclose(blk, 0.25 * Phi)

    # the record forgotten: the pair is exactly Phi+ (1/2 cup (x) mirror cup); without
    # the correction, 1/4 I -- each far end cupped to its own mirror, two strangers' coins
    assert same_picture(master(forget=True), doubled_pair(across=True, power=-2))
    assert same_picture(master(corrected=False, forget=True), doubled_pair(across=False, power=-4))
    assert not same_picture(master(corrected=False, forget=True), doubled_pair(across=True, power=-2))
    h = master(forget=True)
    simp.full_reduce(h)
    assert same_picture(h, doubled_pair(across=True, power=-2))

    # --- (c) the SKELETON: ten lines = ten edges (8 doubled quantum + 2 classical)
    s = skeleton()
    assert s.num_edges() == 10
    assert len(s.outputs()) == 6                             # dbar, d, x, z, c, cbar
    # the two classical wires are the edges M -> x and M -> z (lines 9 and 10)
    # C drawn as four identity dots (fourteen edges), then the identity rule applied to
    # exactly those dots PRODUCES the ten
    boxed, ids = skeleton(with_box=True)
    assert boxed.num_edges() == 14
    for v in ids:
        remove_identity(boxed, v)
    assert boxed.num_edges() == 10 and same_picture(boxed, s)


if __name__ == "__main__":
    verdict("MendedJunction", blackboard, sketchbook)
