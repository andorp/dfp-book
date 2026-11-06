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
# dish, and the table read (the message stone and Alice's gone into the glass);
# then the table after the letter is obeyed -- flip, spin, the tokens tipped out
# before Bob's stone -- which looks exactly as before,
#   \mirrortable[operations]{scene}{caption}, \pairingglass{state}{caption}
#                                                         (pdf/props/ch10.tex)
# tools/props_ch10.py transcribes the Lua (with L1 and L2); the calls are read
# out of dialog.md, never copied.  This file also checks the whole dialogue: the
# seventeen pictures, the keyed rule, the frames, the places, the Lua's
# forty-seven faults, that nothing is pictured after the letter is obeyed, and
# that the six "Let Bob draw" hooks still spell C-O-E-C-K-E.
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
SIZES = {"First": [1, 1, 1, 1], "Second": [1, 1], "Third": [1, 1, 2], "Fourth": [1, 1, 1],
         "Fifth": [3, 1]}
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
    """Every one of the Lua's forty-seven fault sites, each raised by a call of its own
    and by nothing else."""
    R, T, G, K = PC.ringstring, PC.mirrortable, PC.pairingglass, PC.twotokens
    bad = [
        ("pattern-word", K, "", "brass up, copper sideways"),
        ("pattern-twice", K, "", "brass up, brass down"),
        ("pattern-none", K, "", ""),
        ("glass-cups-twice", G, "", "cups empty, stones in the cups, both up"),
        ("glass-tokens-twice", G, "", "tokens in the trays, tokens away"),
        ("glass-word", G, "", "cups full"),
        ("glass-pattern-twice", G, "", "tokens away, brass up"),
        ("glass-wrapped", G, "", "wrapped, cups empty"),
        ("glass-full", G, "", "stones in the cups"),
        ("tokens-place", K, "envelope", "both up"),
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
        ("scene-words", R, "", "the four bends, both up"),
        ("loop-twice", R, "", "the loop, writing both up, writing both down, answer both up"),
        ("loop-stray", R, "", "the loop, brass up, writing both up, answer both up"),
        ("loop-missing", R, "", "the loop, writing both up"),
        ("string-scene", R, "", "the knot"),
        ("bends-op", R, "pull", "the four bends"),
        ("bends-round", R, "slide the flip round, slide the flip round", "the four bends"),
        ("pull-twice", R, "pull, pull", "the tale, both up"),
        ("corrections-early", R, "corrections", "the tale, both up"),
        ("corrections-twice", R, "pull, corrections, corrections", "the tale, both up"),
        ("meet-early", R, "kin to kin", "the tale, both up"),
        ("tale-op", R, "slide round", "the tale, both up"),
        ("slid-twice", R, "slide round, slide round", "the loop, writing copper up, answer brass up"),
        ("slide-count", R, "slide round", "the loop, writing copper up, answer both up"),
        ("slide-room", R, "slide round", "the loop, writing brass up, answer copper up"),
        ("loop-op", R, "pull", "the loop, writing copper up, answer copper up"),
        ("end-op", R, "pull", "end to end"),
        ("kin-none", R, "pull, kin to kin", "the tale, both down"),
        ("meet-few", R, "pull, corrections, red to red", "the tale, copper up"),
        ("meet-blocked", R, "pull, corrections spin first, red to red", "the tale, both up"),
    ]
    for site, fn, *args in bad:
        assert PC.faults(fn, *args) == [site], (site, PC.faults(fn, *args))
    assert [b[0] for b in bad] == list(PC.ERROR_SITES) and len(set(PC.ERROR_SITES)) == 47
    # the cascades: a bad token word and no good one leaves nothing said; both cups words
    # leave the later one, full, with the tokens still in the trays
    assert PC.faults(G, "", "brass sideways") == ["pattern-word", "pattern-none"]
    assert PC.faults(G, "", "cups empty, stones in the cups") == ["glass-cups-twice", "glass-full"]
    assert PC.faults(T, "into the glass, letter", WC) == ["pattern-none"]
    # the Lua's quirks, transcribed: 'neither' lays both down; 'bobs stone' is Bob's
    assert PC.pattern("neither up", "x") == PC.pattern("neither down", "x") == PC.Pattern(False, False)
    assert PC.faults(T, "into the glass, letter both up, flip bobs stone, spin bob's stone, tip out", WC) == []


def check_lua():
    """The transcription's constants are the Lua's (a drift guard: the places after L2,
    the frames, the reflection geometry), and the Lua has as many fault sites as
    ERROR_SITES -- its direct tex.error calls and its serr calls, the definition aside."""
    lua = LUA.read_text(encoding="utf-8")
    nums = lambda pat: tuple(float(x) for x in re.search(pat, lua).groups())
    assert nums(r"local XKEEP, XMSG, XALICE, XBOB = (\S+), (\S+), (\S+), (\S+)\n") == \
        (PC.XKEEP, PC.XMSG, PC.XALICE, PC.XBOB)
    assert nums(r"local FTOP, FBOT = (\S+), (\S+)\n") == (PC.FTOP, PC.FBOT)
    assert nums(r"local TTOP, TBOT = (\S+), (\S+)\n") == (PC.TTOP, PC.TBOT)
    assert nums(r"local SX0, SX1, STOP, SBOT = (\S+), (\S+), \S+, \S+\n") == PC.SLAB_ENDS
    assert nums(r"local SRX, SRY = (\S+), \S+\n") == (PC.SRX,)
    assert nums(r"local K = (\S+) ") == (PC.K,)
    assert nums(r"local SLACKDX = \{ (\S+), (\S+) \}") == PC.SLACKDX
    code = re.sub(r"--[^\n]*", "", lua)                           # comments aside
    direct = len(re.findall(r"tex\.error\(", code)) - 1           # the one inside serr
    via = len(re.findall(r"\bserr\(", code)) - 1                  # serr's own definition
    assert (direct, via) == (27, 20) and direct + via == len(PC.ERROR_SITES) == 47


def check_whole_dialogue():
    """All seventeen pictures: their number and places, the keyed rule, the captions,
    the frames, no fault anywhere, nothing after the letter is obeyed; and the hooks."""
    assert DIALOGUE.check_whole(PC.ORDER) == 17                  # 4 + 2 + 4 + 3 + 4
    text = DIALOGUE.read(DIALOGUE.dialog)
    assert len(DIALOGUE.call_re.findall(DIALOGUE.read(DIALOGUE.raw))) == 17   # dialog.raw.md too
    assert not DIALOGUE.call_re.findall(text[:text.index("## First Challenge")])   # none before
    DIALOGUE.keyed_first(PC.ORDER, kind=PC.keyed_kind)
    placed = [(o, i, c) for o in PC.ORDER
              for i, c in enumerate(c for b in DIALOGUE.blocks(o) for c in b)]
    assert [(o, i) for o, i, c in placed if c.keyed] == [
        ("First", 0), ("First", 1), ("First", 2), ("Fifth", 0)]  # P1.1 P1.2 P1.3 P5.1
    for o in PC.ORDER:
        assert [len(b) for b in DIALOGUE.blocks(o)] == SIZES[o], o
        for b in DIALOGUE.blocks(o):
            assert len({PC.height(c) for c in b}) == 1, (o, [c.text for c in b])
    for o, i, c in placed:
        PC.picture(c)                                            # the macro never faults
        assert words_only(c.caption, forbidden=PC.CAPTION_FORBIDDEN), c.caption
        if c.macro == "ringstring":
            assert CLOTH.caption_ok(c.caption), c.caption        # no word for the string's lengths
        assert PC.place_ok(o, c), (o, c.text)
        assert len(c.args) == 1 and c.args[0]                    # one state/scene argument
    # the forbidden places, as the card lists them, nowhere in the dialogue
    assert {c.macro for o, i, c in placed if o != "Fifth"} == {"pairingglass", "twotokens", "ringstring"}
    assert [o for o, i, c in placed if c.macro == "mirrortable"] == ["Fifth", "Fifth", "Fifth"]
    assert {o for o, i, c in placed if c.macro == "pairingglass"} == {"First", "Fifth"}
    for bad in ("end to end", "slide both round", "corrections spin first",
                "\\twotokens[cloth]", "\\twotokens*[cloth]"):
        assert all(bad not in c.text for o, i, c in placed), bad
    assert all(not {"wrapped", "under its chamois"} & set(PC.split(PC.lower(c.args[0]), ","))
               for o, i, c in placed if c.macro == "pairingglass")        # never wrapped
    # nothing pictured after the letter is obeyed: P5.4 is the last call of the dialogue
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


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    check_faults()
    check_lua()
    check_whole_dialogue()
    calls = DIALOGUE.check_layout("Fifth", sizes=[3, 1])
    p51, p52, p53, p54 = calls
    for c in calls:
        assert words_only(c.caption, forbidden=PC.CAPTION_FORBIDDEN), c.caption
        assert PC.place_ok("Fifth", c), c.text
    first, second = DIALOGUE.blocks("Fifth")
    assert {PC.height(c) for c in first} == {PC.GLASS_FRAME} == {1.58}   # glass and table level
    assert [c.macro for c in calls] == ["mirrortable", "pairingglass", "mirrortable", "mirrortable"]
    assert [c.keyed for c in calls] == [True, False, False, False]
    assert [(c.opt, c.args[0]) for c in calls] == [
        ("", WC), ("", "stones in the cups, both up"), ("into the glass", WC),
        ("into the glass, letter both up, flip, spin, tip out", WC)]
    assert [c.caption for c in calls] == [
        "as they lay all morning", "chimed, and tipped", "read, and Bob's untouched",
        "turned by feel, and unread"]
    b51, g52, b53, b54 = (PC.picture(c) for c in calls)
    a1, a2 = DIALOGUE.anchors("Fifth")
    f1, f2 = DIALOGUE.following("Fifth")
    assert QUOTE in DIALOGUE.section("Fifth")                     # the docstring's line

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

    # P5.2: the glass chimed -- the stones in its cups (spent), the trays tipped, both
    # tokens face-up in the dish; the very tokens the letter then carries (P5.4)
    assert (g52.wrapped, g52.cups, g52.tokens, g52.pattern, g52.tipped, g52.labels) == \
        (False, "stones", "dish", PC.Pattern(True, True), True, ())
    assert g52.pattern == b54.state.letter
    assert "sets it in the left cup" in a1 and "the brass face-up, the copper face-up" in a1
    # the glass reads: its trays are tipped exactly when both cups are full
    for cups in ("cups empty", "stones in the cups"):
        for tokens in ("tokens in the trays", "both up", "brass up", "tokens away"):
            spec = cups + ", " + tokens
            if cups == "stones in the cups" and tokens == "tokens in the trays":
                assert PC.faults(PC.pairingglass, "", spec) == ["glass-full"]
                continue
            assert PC.pairingglass("", spec).tipped == (cups == "stones in the cups")

    # P5.3: read -- the message stone and Alice's gone with their reflections; the
    # keepsake and Bob's stone left exactly where and as they were (fixed parts)
    v53 = b53.shown
    assert v53.stones == (keep, bob) and v53.cloth and v53.letter is None
    assert (v53.frame, v53.ends) == (v51.frame, v51.ends)

    # P5.4: the letter obeyed -- flip, then spin (the state), the tokens tipped out before
    # Bob's stone, both up, no envelope; the stones EXACTLY P5.3's: the turn shows nothing
    T54 = b54.state
    assert (T54.glass, T54.letter, T54.flip, T54.spin, T54.tipped) == \
        (True, PC.Pattern(True, True), True, True, True)
    v54 = b54.shown
    assert v54.letter is None and v54.tokens_out == PC.Pattern(True, True)
    assert replace(v54, tokens_out=None) == v53                   # nothing else changed
    assert "hands the empty envelope up to Achilles" in a2
    assert "with no mark on it, it looks just as it did" in a2
    assert f2 == "**Tortoise:** The glass chimed, and then the letter went."
    assert not DIALOGUE.following("Fifth")[-1].startswith("\\begin")    # nothing after P5.4

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
