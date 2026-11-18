"""Challenge V: Euclid's Strips  (from the reading to the period; from the period to the factors)
============================================================================================
"Fifteen is three fives."  -- Achilles

Claim: Euclid's algorithm turns a good reading c/q into k/r (the continued fraction: with
q >= N^2 the fraction within 1/(2q) with denominator below N is unique, Result 14.16), and
turns the period into factors: if N is odd and not a prime power, r is even and
a^{r/2} != -1 mod N, then gcd(a^{r/2} - 1, N) is a proper factor (Result 14.17).  For 15
and 7: 3 x 5; for 21 and 2: 3 x 7.  In pictures only the squaring reaches: C_7^2 is the
pure crossing rot_2, which multiplies by 4 = 7^2 mod 15 on the residues.

Translation exception (recorded on the identity card, section 4 and section 10(n)): the
gcd and the continued fraction have no picture (Rosetta row 109).  sketchbook() checks the
squaring step a^{r/2} only -- in the spec's words, "the diagram shows the architecture;
the number theory stays on the blackboard".
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
import re
from fractions import Fraction
from math import gcd

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import ket, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch14 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)


# --- the blackboard ---------------------------------------------------------------------

def strips(long, short):
    """Lay the short strip along the long one as many times as it fits, cut off what is
    left, and repeat with what was cut off.  Returns the counts of fits and the last
    short strip (which measures both)."""
    counts = []
    while short:
        counts.append(long // short)
        long, short = short, long % short
    return counts, long


def continued_fraction(c, q):
    counts, _ = strips(q, c)                  # c/q < 1: the leading quotient is 0
    return [0] + counts


def convergents(cf):
    out, (h0, h1), (k0, k1) = [], (0, 1), (1, 0)
    for a in cf:
        h0, h1 = h1, a * h1 + h0
        k0, k1 = k1, a * k1 + k0
        out.append(Fraction(h1, k1))
    return out


def order(a, N):
    return next(r for r in range(1, N + 1) if pow(a, r, N) == 1)


def shor_rule(c, q, N):
    """Shor's rule: the convergent of c/q with denominator below N closest to c/q."""
    conv = [f for f in convergents(continued_fraction(c, q)) if f.denominator < N]
    return min(conv, key=lambda f: abs(f - Fraction(c, q)))


def splits(a, N):
    """The reduction: returns a proper factor of N, or None when it fails."""
    r = order(a, N)
    if r % 2 or pow(a, r // 2, N) == N - 1:
        return None
    f = gcd(pow(a, r // 2, N) - 1, N)
    assert 1 < f < N                                                   # Result 14.17
    return f


def prime_factors(N):
    out, p = set(), 2
    while p * p <= N:
        while N % p == 0:
            out.add(p); N //= p
        p += 1
    return out | ({N} if N > 1 else set())


def blackboard():
    # --- the strips of the dialogue: fifteen and three, fifteen and five
    assert strips(15, 3) == ([5], 3)                  # five times, nothing left
    assert strips(15, 5) == ([3], 5)
    assert strips(15, 4) == ([3, 1, 3], 1)            # the hour itself: one thumb, nothing
    assert 3 * 5 == 15
    # "Multiply them and you get the hour times itself, less one": 3 x 5 = 4^2 - 1, and the
    # hour times itself is the hand ticked on as many ticks again, home at one
    assert 3 * 5 == 4 * 4 - 1 and (4 * 4) % 15 == 1 and pow(7, 4, 15) == 1

    # --- the harder dial: twenty-one hours, the tick doubles; five parts in thirty-two
    assert strips(32, 5) == ([6, 2, 2], 1)            # six times, two left; twice; twice
    assert continued_fraction(5, 32) == [0, 6, 2, 2]
    assert convergents([0, 6, 2, 2])[1] == Fraction(1, 6)          # the first convergent
    assert continued_fraction(27, 32) == [0, 1, 5, 2, 2]
    assert convergents([0, 1, 5, 2, 2])[2] == Fraction(5, 6)
    hand, ticks = 1, []
    while True:                                       # two, four, eight, sixteen, eleven, one
        hand = (2 * hand) % 21
        ticks.append(hand)
        if hand == 1:
            break
    assert ticks == [2, 4, 8, 16, 11, 1] and order(2, 21) == 6
    assert ticks[2] == 8 and (7, 9) == (ticks[2] - 1, ticks[2] + 1)  # half of six ticks: eight
    assert strips(21, 7)[1] == 7 and strips(21, 9)[1] == 3          # the strips give 7 and 3
    assert 3 * 7 == 21
    # the coarse reading is safe only by luck: with q = 32 < 21^2 the uniqueness of Result
    # 14.16(i) fails -- 2/13 is ALSO within 1/(2q) of 5/32, and closer, so Shor's rule would
    # return 13, which the check 2^13 = 1 mod 21 rejects.  (The dialogue: "A real case for a
    # dial this size would be made finer still, so that the strips can't mislead.")
    near = [f for f in (Fraction(1, 6), Fraction(2, 13)) if abs(f - Fraction(5, 32)) <= Fraction(1, 64)]
    assert len(near) == 2 and shor_rule(5, 32, 21) == Fraction(2, 13)
    assert pow(2, 13, 21) != 1
    assert shor_rule(27, 32, 21) == Fraction(11, 13)

    # --- section 14.5 in Shor's letters: N = 21, a = 2, q = 512, reading 85
    q = 512
    assert 21 ** 2 <= q < 2 * 21 ** 2
    cf = continued_fraction(85, q)
    assert cf == [0, 6, 42, 2]
    assert convergents(cf)[1:] == [Fraction(1, 6), Fraction(42, 253), Fraction(85, 512)]
    assert [f for f in convergents(cf)[1:] if f.denominator < 21] == [Fraction(1, 6)]
    assert abs(Fraction(85, q) - Fraction(1, 6)) == Fraction(1, 1536) <= Fraction(1, 2 * q)
    assert shor_rule(85, q, 21) == Fraction(1, 6) and 2 ** 6 == 3 * 21 + 1

    # --- Result 14.16 (i): with q >= N^2, d/r (r < N) within 1/(2q) of c/q is the ONLY
    #     fraction with denominator below N that close.  (ii), stated and cited, checked
    #     here as documentation: it is a convergent of c/q.  Every r < N and every d < r, for
    #     three dials at Shor's q.
    for N, q in [(15, 256), (21, 512), (33, 2048)]:
        assert N * N <= q
        for r in range(1, N):
            for d in range(r):
                c = round(q * d / r)
                if abs(Fraction(c, q) - Fraction(d, r)) > Fraction(1, 2 * q):
                    continue
                close = {Fraction(n, m) for m in range(1, N) for n in range(m + 1)
                         if abs(Fraction(n, m) - Fraction(c, q)) <= Fraction(1, 2 * q)}
                assert close == {Fraction(d, r)}
                assert Fraction(d, r) in convergents(continued_fraction(c, q))

    # --- the reduction on the chapter's two dials
    assert pow(7, 2, 15) == 4 and gcd(4 - 1, 15) == 3 and gcd(4 + 1, 15) == 5
    assert pow(2, 3, 21) == 8 and gcd(8 - 1, 21) == 7 and gcd(8 + 1, 21) == 3

    # --- Result 14.17 for every base of several odd N that are not prime powers, and the two
    #     ways it can fail.  Shor's rate: the fraction of bases (sharing no factor with N)
    #     that split N is at least 1 - 1/2^(l-1), l the number of distinct prime factors.
    rates = {}
    for N in (15, 21, 33, 35, 39, 45, 51, 55, 63, 105):
        assert N % 2 and len(prime_factors(N)) >= 2
        bases = [a for a in range(1, N) if gcd(a, N) == 1]
        ok = [a for a in bases if splits(a, N)]
        for a in bases:
            r = order(a, N)
            if a not in ok:
                # the only failures: r odd, or a^{r/2} = -1; and then the gcd really is not
                # a proper factor
                assert r % 2 or pow(a, r // 2, N) == N - 1
                if r % 2 == 0:
                    assert gcd(pow(a, r // 2, N) - 1, N) in (1, N)
        l = len(prime_factors(N))
        rates[N] = (len(ok), len(bases))
        assert Fraction(len(ok), len(bases)) >= 1 - Fraction(1, 2 ** (l - 1))
    assert rates[15] == (6, 8) and rates[21] == (6, 12) and rates[33] == (10, 20)
    fails15 = [a for a in range(1, 15) if gcd(a, 15) == 1 and not splits(a, 15)]
    N = 15
    assert fails15 == [1, N - 1]                      # r = 1 (odd), and a = -1 (r = 2)
    assert order(1, N) == 1 and order(N - 1, N) == 2 and pow(N - 1, 1, N) == N - 1

    pictures()


# --- the pictures in the dialogue ----------------------------------------------------------

ARIA = (("Achilles", "How do you tell how long a thing takes to come round, if you may look at it only once?"),
        ("Tortoise", "You don't count the ticks. You hear how far one tick turns it."))
GROUND = ("Tortoise", "The ticking changes where it stands, never what it is.")


def frame_top(own):
    """The frame's bounding path (its top read off), and the other lines a frame draws."""
    m = re.fullmatch(r"\\path \(-1\.250,-0\.260\) rectangle \(1\.250,(\d\.\d{3})\);", own[0])
    return float(m.group(1)), own[1:]


def pictures():
    """The Fifth Challenge's prop pictures (identity card, sections 4 and 3e), and the whole
    dialogue's, read through tools/dfp_props.py and the transcription tools/props_ch14.py:
    two strips and a single call, seven frames (F29-F35).  The ties below hold the
    transcription against this file's blackboard: a structural check of the pictures, not a
    second proof."""
    calls = DIALOGUE.check_layout("Fifth", sizes=[1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("tickstripssteps", True, "", ("fifteen and four", "lay; lay; lay"), "as cut; laid along; laid on; and on again"),
        ("tickstripssteps", False, "first laid three times; from=off", ("fifteen and four", "set back; snip"),
         "the short strip set back; after the snip"),
        ("tickdial", False, "", ("twenty-one hours; hand ticked six times",), "the new hand, home")]
    PC.check_captions(DIALOGUE, "Fifth", calls)
    PC.check_section(DIALOGUE, "Fifth", calls)
    strips_h, dial_h = round(PC.SFTOP4 - PC.SFBOT, 2), round(PC.DIAL_FTOP - PC.DIAL_FBOT, 2)
    assert (strips_h, dial_h) == (0.88, 2.42)                         # read from the Lua
    assert round(PC.SFTOP - PC.SFBOT, 2) == 0.78                      # a plain strips call's (round 3's)
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Fifth")] == [[strips_h], [strips_h], [dial_h]]
    frames = [(fc, p) for c in calls for fc, (p, _) in zip(PC.frame_calls(c), PC.pictures(c))]
    names = ["F%02d" % i for i in range(29, 36)]
    assert len(frames) == len(names) == 7
    F = dict(zip(names, [p for _, p in frames]))
    FC = dict(zip(names, [fc for fc, _ in frames]))
    for fc, p in frames:                                              # a strip frame IS its single call
        assert PC.drawn(PC.picture(fc)) == PC.drawn(p) and PC.picture(fc).log == p.log, fc.text
    assert [(fc.keyed, fc.opt) for fc in FC.values()] == [
        (True, "start"), ("new", "start; step=lay"), ("new", "first laid once; step=lay"),
        ("new", "first laid twice; step=lay"), (False, "first laid three times; step=set back"),
        (False, "first; step=snip"), (False, "")]

    # the anchors (card sections 4 and 3e.4), in dialog.md; the raw file's following paragraphs
    # agree.  T5: B17's paragraph ends at the third lay, and the set-back is a direction of its
    # own, drawn as B18's first frame
    a1, a2, a3 = DIALOGUE.anchors("Fifth")
    f1, f2, f3 = DIALOGUE.following("Fifth")
    assert "along the table's edge on his side" in a1 and a1.rstrip().endswith("that the four cannot cover.]*")  # D29, T5
    assert "He sets the short strip back beside the long one." not in a1
    assert f1 == "*[He sets the short strip back beside the long one.]*"                            # T5
    paras = DIALOGUE.paragraphs("Fifth")
    b17 = paras.index(a1) + 1
    assert paras[b17 + 1] == f1 and paras[b17 + 2] == a2 and paras[b17 + 3].startswith("\\begin{center}")
    assert a2.startswith("*[He pencils the count on the long strip and snips it at the last tick;")
    assert "the measured part, with its count on it, he lays aside" in a2                          # D29
    assert f2.startswith("*[The three goes along the four once")
    assert "now stands again on the first mark past the top" in a3
    assert f3.startswith("**Achilles:** Six ticks, and it's on the mark it left.")
    assert DIALOGUE.following("Fifth", DIALOGUE.raw) == [f1, f2, f3]
    assert DIALOGUE.anchors("Fifth", DIALOGUE.raw)[0] == a1
    section = DIALOGUE.section("Fifth")
    assert "She draws a second circle on the fresh sheet" in section                               # D30
    assert "the first at the top, on the side furthest from Achilles, as on the first" in section  # D30

    # F29-F34: the strips ARE Euclid's run on (15, 4) -- the counts 3, 1, 3, the last strip
    # that measures both 1 = gcd(15, 4); the ticks at 4q thumbs (q = 1, 2, 3), laid at 0, 4, 8;
    # set back to 0; the piece left 15 mod 4 = 3
    counts, last = strips(15, 4)
    for n in names[:6]:
        T = F[n].drawn
        assert [s["fits"] for s in T.steps] == counts == [3, 1, 3] and T.measure == last == gcd(15, 4)
    heads = [F[n].head.split(" view=1 ")[1] for n in names[:6]]
    assert heads == ["phase=start edge=15 beside=4 ticks=none", "phase=lay edge=15 beside=4 ticks=4 at=0",
                     "phase=lay edge=15 beside=4 ticks=4/8 at=4", "phase=lay edge=15 beside=4 ticks=4/8/12 at=8",
                     "phase=laid edge=15 beside=4 ticks=4/8/12", "phase=cut edge=3 beside=4 ticks=none"]
    for q, n in enumerate(names[1:4], 1):
        assert PC.strip_ticks(F[n].drawn) == [4 * i for i in range(1, q + 1)]
        v = F[n].view
        assert np.isclose(v.beside[0], PC.SX0 + 4 * (q - 1) * PC.THUMB)            # laid at 0, 4, 8
        assert np.allclose(v.ticks, [PC.SX0 + 4 * i * PC.THUMB for i in range(1, q + 1)])
        assert np.isclose(v.beside[2], v.ticks[-1])                                 # a tick where it stops
    assert PC.strip_ticks(F["F33"].drawn) == [4, 8, 12] and np.isclose(F["F33"].view.beside[0], PC.SX0)
    assert F["F34"].drawn.steps[0]["left"] == 15 % 4 == 3
    # what is drawn: the long strip along the near edge, the short one beside it, further from
    # Achilles, their left ends level when set back, the two never touching; after the snip
    # the piece left lies where the long strip began, the short strip unmoved, the measured
    # part not drawn
    v1, v2 = F["F33"].view, F["F34"].view
    stroke = 0.3 * PC.PT / 2
    for n in names[:6]:
        (ex0, ey0, ex1, ey1), (bx0, by0, bx1, by1) = F[n].view.edge, F[n].view.beside
        assert ex0 == PC.SX0 and by0 > ey1 and by0 - ey1 - 2 * stroke > 0.1
        assert F[n].view.top == PC.SFTOP4
    assert v1.beside[0] == v2.beside[0] == PC.SX0
    assert np.isclose(v1.edge[2] - v1.edge[0], 15 * PC.THUMB) and np.isclose(v2.edge[2] - v2.edge[0], 3 * PC.THUMB)
    assert v1.beside == v2.beside and v1.edge[:2] == v2.edge[:2] and v2.ticks == ()
    assert np.isclose(15 * PC.THUMB, 2.25) and v1.edge[2] < 1.25        # fifteen thumbs span the frame
    # the keyed words (measured node boxes): "short strip" to the short strip's right and
    # "long strip" below the strip's right end (F29); "tick" below the first tick, the part
    # the first lay brought (F30, keyed new: L11)
    p51, p30 = F["F29"], F["F30"]
    assert p51.words == ("short strip", "long strip") and p30.words == ("tick",)
    assert all(F[n].words == () for n in names[2:])
    assert "keyed new: tick, below the first tick at -0.550 (the part this step brought)" in p30.notes
    assert p30.notes == ["step lay -> " + p30.head,
                         "no marks: the first lay is at the long strip's left end, where the short strip lies; the "
                         "tick shows it", "keyed new: tick, below the first tick at -0.550 (the part this step brought)"]
    w = p51.view
    labels = {l[0]: l[1:] for l in w.labels}
    (labels["tick"],) = [l[1:] for l in p30.labels if l[0] == "tick"]
    assert labels["long strip"] == (w.edge[2], w.edge[1] - 0.03, "north east")
    assert labels["tick"] == (p30.view.ticks[0], w.edge[1] - 0.03, "north")
    boxes = {k: PC.label_box(k, *labels[k]) for k in labels}
    for k, (x0, y0, x1, y1) in boxes.items():
        assert min(x0 + 1.25, 1.25 - x1, y0 - PC.SFBOT, PC.SFTOP4 - y1) > 0, k
        i0, j0, i1, j1 = PC.label_box(k, *labels[k], ink=True)
        assert min(i0 + 1.25, 1.25 - i1, j0 - PC.SFBOT, PC.SFTOP4 - j1) > 0.03, k
    for k in ("tick", "long strip"):
        assert w.edge[1] - stroke - boxes[k][3] > 0                    # below the edge strip
    assert boxes["short strip"][0] - (w.beside[2] + stroke) > 0.03    # right of the short strip
    A, B = boxes["tick"], PC.label_box("short strip", *labels["short strip"])
    assert max(A[0] - B[2], B[0] - A[2], A[1] - B[3], B[1] - A[3]) > 0.1    # "tick" clear of F29's other words
    # the other rule: when the strip's word goes below a long strip and the first tick
    # stands near its right end, "tick" steps to its tick's left (fifteen and nine: the
    # tick at nine)
    w9 = {k: (x, y, a) for k, x, y, a in PC.strips_places(PC.strips_state("", "fifteen and nine"), True).labels}
    assert w9["tick"] == (PC.SX0 + 9 * PC.THUMB - 0.02, PC.STRIP_Y0[0] - 0.03, "north east")
    A, B = PC.label_box("tick", *w9["tick"]), PC.label_box("long strip", *w9["long strip"])
    assert B[0] - A[2] > 0

    # the marks (card 3e.6): every lay after the first, and the set-back, lift the short strip
    # over the top (its ghost where it lay); the snip slides the piece beneath the edge strip
    # (the long strip's ghost).  Each arc clear of the strips' outlines and the ticks, and
    # inside the taller frame (L8): no overlay above its top
    assert F["F29"].notes == [] and F["F29"].mark_tikz == [] and F["F30"].mark_tikz == []
    moves = {"F31": ("swing", -1), "F32": ("swing", -1), "F33": ("swing", -1), "F34": ("slide", 1)}
    for n, (kind, below) in moves.items():
        p = F[n]
        (arc,), (c,) = p.arcs, p.clearances
        assert arc.kind == kind and not arc.headless and len(p.mark_tikz) == 2 and len(p.ghosts) == 1
        assert c["outline"] > 0.02 and c["frame"] > 0.02 and (c["tick"] is None or c["tick"] > 0.3), (n, c)
        before = F[names[names.index(n) - 1]].view if n != "F34" else None
        if kind == "swing":                                            # over the top, by the short strip
            assert arc.p0[1] == arc.p1[1] > p.view.beside[3] and c["what"].startswith("lift arc over the top")
            (gw, (gx, gy), *_), = p.ghosts
            assert (gx, gy) == (before.beside[0], before.beside[1])    # the ghost where it lay
            assert np.isclose(arc.p1[0] - arc.p0[0], p.view.beside[0] - before.beside[0])   # it went there
        else:                                                          # beneath the edge strip, leftwards
            assert arc.p0[1] == arc.p1[1] < p.view.edge[1] and arc.p1[0] < arc.p0[0]
            assert c["what"].startswith("slide arc beneath the edge strip")
    assert np.allclose([F[n].view.beside[0] - F[names[names.index(n) - 1]].view.beside[0] for n in ("F31", "F32", "F33")],
                       [4 * PC.THUMB, 4 * PC.THUMB, -8 * PC.THUMB])               # on 4, on 4, back 8: to 0
    # the identities (card 3e.6): marks=off draws the same; F33 is round 3's P5.1 and F34 its
    # P5.2, up to keying, words and the taller frame (the frame's top 0.62 for 0.52: L8)
    for n in names[1:6]:
        assert PC.drawn(PC.twin(FC[n])) == PC.drawn(F[n]) and PC.twin(FC[n]).mark_tikz == []
    for n, br in (("F33", ""), ("F34", "first cut")):
        r3 = PC.run_frame("strips", br, "fifteen and four", False)
        t4 = PC.drawn(PC.twin(FC[n]))
        assert t4[0] == r3.head
        (top4, rest4), (top3, rest3) = frame_top(t4[1]), frame_top(PC.drawn(r3)[1])
        assert (top4, top3) == (PC.SFTOP4, PC.SFTOP) and rest4 == rest3

    # the runs read out but not pictured (l.701, l.713, l.729): fifteen and three, fifteen
    # and five -- the macro's strips, the same counts, against gcd.  The longer ones (thirty-
    # two and five, twenty-one and seven, twenty-one and nine) are longer than the evening's
    # scale draws: the macro refuses them (fifteen thumbs span the picture), and the
    # blackboard's strips carry them -- five parts in thirty-two, 5/32 = [0; 6, 2, 2]
    PC.TEX_ERRORS.clear()
    for words, (x, y) in (("fifteen and three", (15, 3)), ("fifteen and five", (15, 5))):
        T = PC.strips_state("", words)
        assert [s["fits"] for s in T.steps] == strips(x, y)[0] and T.measure == strips(x, y)[1] == gcd(x, y)
    assert not PC.TEX_ERRORS
    for words, (x, y) in (("thirty-two and five", (32, 5)), ("twenty-one and seven", (21, 7)),
                          ("twenty-one and nine", (21, 9))):
        PC.strips_state("", words)
        assert [site for site, _ in PC.TEX_ERRORS] == ["strips-long"] and x > PC.MAXLEN == 15
        PC.TEX_ERRORS.clear()
        assert strips(x, y)[1] == gcd(x, y)
    assert strips(32, 5)[0] == [6, 2, 2] and continued_fraction(5, 32) == [0, 6, 2, 2]
    assert strips(21, 9)[1] == 3 and strips(21, 7) == ([3], 7)
    # a starred strips strip keys only from the strips as cut: from a lay view it is refused
    assert PC.faults(PC.strip, "strips", "first laid once", "fifteen and four", "lay; lay", "a; b; c", True)
    PC.TEX_ERRORS.clear()

    # F35 (P5.3): the harder dial, the tick doubles: the hand visits 1, 2, 4, 8, 16, 11 and is
    # home after six ticks -- the order of 2 mod 21
    p53 = F["F35"]
    assert p53.dump == "hours=21 hand=1 rings=none" and p53.notes == [] and p53.words == ()
    assert PC.dial_orbit(21) == [pow(2, s, 21) for s in range(6)] == [1, 2, 4, 8, 16, 11]
    assert len(PC.dial_orbit(21)) == order(2, 21) == 6
    for m, w_ in enumerate(("once", "twice", "three times", "four times", "five times", "six times"), 1):
        assert PC.dial_state("twenty-one hours; hand ticked " + w_).hand == pow(2, m, 21)
    assert p53.view.rings == () and p53.view.pairs == ()

    whole()


def whole():
    """The dialogue's pictures as a whole (card 3e.6): nineteen calls and thirty-five frames, all
    inside the five challenges, none after F35; none in the forbidden ranges; each frame in its
    place; the keyed rule over true frames; the aria and the Genie."""
    D = DIALOGUE
    assert D.check_whole(PC.ORDER) == 19                                 # calls (a strip is one call)
    text = D.read(D.dialog)
    calls = [(o, c) for o in PC.ORDER for b in D.blocks(o) for c in b]
    assert not D.call_re.search(text[:text.index("## First Challenge")])
    last = calls[-1][1]
    assert last.text.startswith("\\tickdial{twenty-one hours; hand ticked six times}")
    assert not D.call_re.search(text[text.index(last.text) + len(last.text):])
    assert [len(D.blocks(o)) for o in PC.ORDER] == [3, 4, 5, 4, 3]       # 19 blocks
    frames = [(o, c, fc, p) for o, c in calls for fc, (p, _) in zip(PC.frame_calls(c), PC.pictures(c))]
    names = ["F%02d" % i for i in range(1, 36)]
    assert len(frames) == len(names) == 35 and sum(PC.is_strip(c) for _, c in calls) == 9
    assert [n for n, (o, c, fc, p) in zip(names, frames) if PC.is_prediction(fc)] == ["F27"]
    # the keyed rule over true frames only (a prediction is never keyed), the strips' frames
    # judged in turn: a starred strip keys its first frame; a starred strips strip keys its
    # later frames "new", naming only the part their step brought (F30: the tick)
    D.keyed_first(PC.ORDER, kind=PC.keyed_kind, select=PC.judged)
    keyed = [n for n, (o, c, fc, p) in zip(names, frames) if fc.keyed is True]
    assert keyed == ["F01", "F06", "F08", "F14", "F22", "F24", "F29"]
    new = [n for n, (o, c, fc, p) in zip(names, frames) if fc.keyed == "new"]
    assert new == ["F30", "F31", "F32"]
    seen = set()
    for n, (o, c, fc, p) in zip(names, frames):
        if fc.keyed == "new":
            assert PC.is_strip(c) and c.keyed is True and c.macro == "tickstripssteps", n
        elif PC.judged(fc):
            assert bool(PC.words(p)) == (PC.keyed_kind(fc) not in seen), n     # a family's first frame names it
            seen.add(PC.keyed_kind(fc))
        else:
            assert PC.words(p) == () and fc.keyed is False, n
        assert PC.place_ok(o, fc), (o, fc.text)                          # the places, globally
    assert [n for n, (o, c, fc, p) in zip(names, frames) if p.words and fc.keyed == "new"] == ["F30"]
    # no keyed frame carries a mark (card 3e.5), and the counted frame (F24) none either
    assert not any(p.mark_tikz for n, (o, c, fc, p) in zip(names, frames) if fc.keyed is True)
    marked = [n for n, (o, c, fc, p) in zip(names, frames) if p.mark_tikz]
    assert marked == ["F02", "F03", "F04", "F05", "F07", "F10", "F11", "F12", "F13", "F15", "F16", "F19",
                      "F26", "F28", "F31", "F32", "F33", "F34"]
    # a frame with no step draws no mark; a step's frame logs it first
    for n, (o, c, fc, p) in zip(names, frames):
        step = PC.options(fc).get("step")
        if step is None:
            assert p.mark_tikz == [] and p.notes == [], n
        else:
            assert p.notes[0].startswith("step " + step + " -> "), n
    # the aria with variations and the Genie, after the revision, on both files (the card's
    # section 1c one-liner, transcribed); no "Let Bob draw"; three "Referred.", the Genie's
    for path in (D.dialog, D.raw):
        t = D.read(path)
        assert PC.form_check(t) == (True, ARIA, GROUND, "Third Challenge"), path
        assert t.count("Let Bob draw") == 0 and "fourteen" not in t.lower()
        spoken = re.findall(r"^\*\*([^*]+):\*\*(.*)$", t, re.M)
        referred = [who for who, line in spoken if re.sub(r"\*\[.*?\]\*", "", line).strip().startswith("Referred.")]
        assert referred == ["Genie"] * 3, referred
    check_faults()


FAULT_CASES = [
    # the bracket (oerr)
    ('opt-unknown', 'case', False, 'knob; colour=red', 'knob up'),
    ('opt-twice', 'case', False, 'step=granted; step=front open', ''),
    ('opt-bare-two', 'case', False, 'knob; windows; step=knob down', 'knob up'),
    ('opt-onoff', 'case', False, 'step=knob up; marks=maybe', ''),
    ('marks-no-step', 'case', False, 'marks=on', ''),
    ('expects-keyed', 'dial', True, 'step=tick; expects=two ticks home', 'hand ticked once'),
    ('expects-marked', 'dial', False, 'step=tick; marks=on; expects=two ticks home', 'hand ticked once'),
    # the case (cerr; oerr)
    ('case-locked', 'case', False, '', 'granted; knob up'),
    ('case-knob', 'case', False, '', 'knob sideways'),
    ('case-twice', 'case', False, '', 'granted; granted'),
    ('case-open-early', 'case', False, '', 'front open'),
    ('case-open-twice', 'case', False, 'inside; step=front open', 'knob up; granted; front open'),
    ('case-clause', 'case', False, '', 'lid open'),
    ('case-rule', 'case', False, 'step=knob up; expects=two ticks home', ''),
    ('case-knob-view', 'case', False, 'windows; step=knob up', ''),
    ('case-knob-same', 'case', False, 'step=knob down', ''),
    ('case-front-view', 'case', False, 'step=front open', 'knob up; granted'),
    ('case-inside-shut', 'case', False, 'inside', ''),
    ('case-view', 'case', False, 'sideways', ''),
    # the dial (derr; oerr)
    ('dial-hours', 'dial', False, '', 'eleven hours'),
    ('dial-ticks', 'dial', False, '', 'hand ticked sometimes'),
    ('dial-first-twice', 'dial', False, '', 'first round; first round'),
    ('dial-round-ticks', 'dial', False, '', 'first round ticked sometimes'),
    ('dial-pairs-twice', 'dial', False, '', 'every round; pairs face down; pairs face down'),
    ('dial-pairsat-twice', 'dial', False, '', 'every round; pairs face down at seven; pairs face down at four'),
    ('dial-pairsat-word', 'dial', False, '', 'every round; pairs face down at bogus'),
    ('dial-pairsat-dup', 'dial', False, '', 'every round; pairs face down at seven and seven'),
    ('dial-clause', 'dial', False, '', 'hand at one; spin it'),
    ('dial-round-21', 'dial', False, '', 'twenty-one hours; first round'),
    ('dial-first-every', 'dial', False, '', 'first round; every round'),
    ('dial-pairs-alone', 'dial', False, '', 'pairs face down'),
    ('dial-pairsat-both', 'dial', False, '', 'every round; pairs face down; pairs face down at four'),
    ('dial-pairsat-alone', 'dial', False, '', 'pairs face down at four'),
    ('dial-pairsat-one', 'dial', False, '', 'every round; pairs face down at one'),
    ('dial-pairsat-hour', 'dial', False, '', 'every round; pairs face down at two'),
    ('dial-rule-step', 'dial', False, 'step=tick the round; expects=two ticks home', 'first round'),
    ('dial-rule-same', 'dial', False, 'step=tick; expects=two ticks home', 'hand at one'),
    ('dial-step-count', 'dial', False, 'step=tick zero times', 'hand at one'),
    ('dial-step', 'dial', False, 'step=wind', 'hand at one'),
    ('dial-step-round', 'dial', False, 'step=tick the round', 'hand at one'),
    ('dial-step-pairs', 'dial', False, 'step=pairs face down', 'hand at one'),
    ('dial-step-pairs-done', 'dial', False, 'step=pairs face down', 'every round; pairs face down'),
    ('dial-step-pairsat-done', 'dial', False, 'step=pairs face down at seven', 'every round; pairs face down at seven'),
    ('dial-step-pairs-left', 'dial', False, 'step=pairs face down', 'every round; pairs face down at seven'),
    ('dial-bare', 'dial', False, 'knob', 'hand at one'),
    ('dial-rule', 'dial', False, 'step=tick; expects=one tick home', 'hand at one'),
    ('dial-rule-nostep', 'dial', False, 'expects=two ticks home', 'hand at one'),
    # the cloth (terr; oerr)
    ('cloth-push-none', 'cloth', False, 'push the upper', 'two listeners'),
    ('cloth-rings-op', 'cloth', False, 'cross', 'loose rings a quarter and a half'),
    ('cloth-push-parts', 'cloth', False, 'slide off the eighth, push the wheel', 'one listener'),
    ('cloth-tube-who', 'cloth', False, 'step=tube on the upper', 'one listener'),
    ('cloth-tube-early', 'cloth', False, 'step=tube on the upper', 'two listeners'),
    ('cloth-tube-twice', 'cloth', False, 'push the wheel, tube on the listener; step=tube on the listener', 'one listener'),
    ('cloth-corr-one', 'cloth', False, 'lay the correction', 'one listener'),
    ('cloth-corr-twice', 'cloth', False, 'push the wheel, on to the far end, tube on the upper, lay the correction; step=lay the correction', 'two listeners'),
    ('cloth-corr-green', 'cloth', False, 'lay the correction', 'two listeners'),
    ('cloth-far-early', 'cloth', False, 'step=on to the far end', 'one listener'),
    ('cloth-far-twice', 'cloth', False, 'push the wheel, on to the far end; step=on to the far end', 'one listener'),
    ('cloth-push-upper-one', 'cloth', False, 'push the upper', 'one listener'),
    ('cloth-part-one', 'cloth', False, 'push into the ring', 'two listeners'),
    ('cloth-part-none', 'cloth', False, 'slide off the eighth, push into the ring, hub through the bob, bob into the ring, slide together', 'one listener; wheel dark'),
    ('cloth-part-order', 'cloth', False, 'push into the ring', 'one listener'),
    ('cloth-cross-one', 'cloth', False, 'cross', 'one listener'),
    ('cloth-cross-twice', 'cloth', False, 'push the wheel, cross, cross', 'two listeners'),
    ('cloth-cross-turns', 'cloth', False, 'cross', 'two listeners'),
    ('cloth-op', 'cloth', False, 'bogus op', 'one listener'),
    ('cloth-scene-clause', 'cloth', False, '', 'three listeners', ('cloth-scene-clause', 'cloth-no-scene')),
    ('cloth-no-scene', 'cloth', False, '', 'wheel lit'),
    ('cloth-rule-step', 'cloth', False, 'push the wheel; step=cross; expects=gave its quarter', 'two listeners'),
    ('cloth-rule-scene', 'cloth', False, 'step=push the wheel; expects=gave its quarter', 'two listeners'),
    ('cloth-rule-dark', 'cloth', False, 'step=push the wheel; expects=gave its quarter', 'one listener; wheel dark'),
    ('cloth-step-commas', 'cloth', False, 'step=push the wheel, cross', 'one listener'),
    ('cloth-rule', 'cloth', False, 'step=push the wheel; expects=gave nothing', 'one listener'),
    ('cloth-rule-nostep', 'cloth', False, 'expects=gave its quarter', 'one listener'),
    # the strips (serr; oerr)
    ('strips-form', 'strips', False, '', 'fifteen'),
    ('strips-word', 'strips', False, '', 'fifteen and many'),
    ('strips-order', 'strips', False, '', 'four and nine'),
    ('strips-long', 'strips', False, '', 'sixteen and three'),
    ('strips-view', 'strips', False, 'last cut', 'ten and three'),
    ('strips-lays', 'strips', False, 'third', 'ten and three'),
    ('strips-cut-exact', 'strips', False, 'step=cut', 'nine and three'),
    ('strips-lay-count', 'strips', False, 'first laid nothing', 'ten and three'),
    ('strips-lay-fits', 'strips', False, 'first laid twice', 'fifteen and nine'),
    ('strips-step-nofit', 'strips', False, 'last laid twice; step=lay', 'eleven and two'),
    ('strips-step-words', 'strips', False, 'first laid thirteen times; step=lay', 'fifteen and one'),
    ('strips-step-lay', 'strips', False, 'step=lay', 'ten and three'),
    ('strips-step-setback', 'strips', False, 'step=set back', 'ten and three'),
    ('strips-step-fitsagain', 'strips', False, 'first laid once; step=set back', 'ten and three'),
    ('strips-step-snip', 'strips', False, 'start; step=cut', 'ten and three'),
    ('strips-step', 'strips', False, 'step=fold', 'ten and three'),
    ('strips-keylay', 'strips', True, 'start; step=lay', 'ten and three'),
    ('strips-rule', 'strips', False, 'step=snip; expects=x', 'ten and three'),
    # a strip of steps (oerr: the strip's own faults, its frames not run)
    ('strip-bare', 'steps', True, 'dial', 'hand at one', 'knob', 'tick', 'a; b'),
    ('strip-none', 'steps', True, 'case', 'knob up', '', '', 'a'),
    ('strip-commas', 'steps', True, 'cloth', 'two listeners', '', 'push the wheel, cross', 'a; b'),
    ('strip-long', 'steps', True, 'strips', 'fifteen and four', '', 'lay; lay; lay; lay', 'a; b; c; d; e'),
    ('strip-captions', 'steps', True, 'case', 'knob up', 'windows', 'granted', 'a'),
]


def check_faults():
    """Every fault site raised by its own call, alone (or with the one it entails), with a
    message of its own; the 97 sites are the Lua's, all of them; the ';' bracket is order-free
    for all four macros and their strips, each order run, and an item given twice is refused."""
    seen = {}
    for case in FAULT_CASES:
        site = case[0]
        if case[1] == "steps":
            _, _, keyed, kind, scene, opt, stepspec, caps = case[:8]
            want = case[8] if len(case) > 8 else (site,)
            fn, args = PC.steps, (kind, keyed, scene, opt, stepspec, caps)
        else:
            _, kind, keyed, br, state = case[:5]
            want = case[5] if len(case) > 5 else (site,)
            fn, args = PC.run_frame, (kind, br, state, keyed)
        assert tuple(dict.fromkeys(PC.faults(fn, *args))) == tuple(want), site
        msgs = {m for s, m in zip(PC.faults(fn, *args), PC.messages(fn, *args)) if s == site}
        assert len(msgs) == 1 and not (msgs & set(seen)), site           # one message, its own
        seen[msgs.pop()] = site
    # a cloth step the Lua refuses draws no marks (the parts they would follow may not be there);
    # the same step taken draws them
    refused = [c for c in FAULT_CASES if c[1] == "cloth" and "step=" in c[3]]
    assert len(refused) >= 6
    for _, kind, keyed, br, state in (c[:5] for c in refused):
        assert PC.run_frame(kind, br, state, keyed).mark_tikz == [], br
    assert PC.run_frame("cloth", "step=push the wheel", "one listener; wheel lit", False).mark_tikz
    assert [c[0] for c in FAULT_CASES] == [k for k, _ in PC.TEX_SITES] and len(PC.ERROR_SITES) == 97
    PC.check_sites()                                                     # 8 + 16 + 21 + 9 + 43, five tex.error calls
    PC.check_lua()                                                       # every chunk as transcribed, pinned
    # the bracket is order-free: every order of a frame's items draws the same frame
    for kind, state, items in (
            ("case", "knob up", ["windows", "step=granted", "marks=off"]),
            ("dial", "hand at one", ["step=tick twice", "marks=on"]),
            ("dial", "every round; pairs face down", ["step=tick twice", "expects=two ticks home"]),
            ("cloth", "two listeners", ["push the wheel", "step=on to the far end", "marks=on"]),
            ("cloth", "one listener; wheel lit", ["step=push the wheel", "expects=gave its quarter"]),
            ("strips", "fifteen and four", ["first laid once", "step=lay", "marks=on"])):
        outs = set()
        for order in itertools.permutations(items):
            br = "; ".join(order)
            assert PC.faults(PC.run_frame, kind, br, state, False) == [], br
            p = PC.run_frame(kind, br, state, False)
            outs.add((PC.drawn(p), tuple(p.mark_tikz), tuple(p.notes)))
        assert len(outs) == 1, (kind, items)
        # an item given twice is refused
        twice = "; ".join(items + [items[-1]])
        assert "opt-twice" in PC.faults(PC.run_frame, kind, twice, state, False), twice
    for macro, scene, items, stepspec, caps in (
            ("case", "knob up", ["windows", "from=off", "marks=on"], "granted", "a"),
            ("dial", "hand at one", ["from=off", "marks=off"], "tick; tick", "a; b"),
            ("cloth", "two listeners", ["after=push the wheel", "from=off", "marks=on"], "on to the far end", "a"),
            ("strips", "fifteen and four", ["first laid once", "from=off"], "lay; lay", "a; b")):
        outs = set()
        for order in itertools.permutations(items):
            PC.TEX_ERRORS.clear()
            S = PC.steps(macro, False, scene, "; ".join(order), stepspec, caps)
            assert not PC.TEX_ERRORS, order
            outs.add(tuple(S.frames))
        assert len(outs) == 1, (macro, items)
        twice = "; ".join(items + [items[-1]])
        assert "opt-twice" in PC.faults(PC.steps, macro, False, scene, twice, stepspec, caps), twice
    PC.TEX_ERRORS.clear()


# --- the sketchbook ---------------------------------------------------------------------

def dial_picture():
    """C_7 (as in HandsRound.py): input row i runs to output row (i + 1) mod 4 -- the bottom
    wire climbs to the top -- then a red pi dot (NOT) on every wire."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, i, 0) for i in range(4)]
    outs = [g.add_vertex(VT.BOUNDARY, i, 10) for i in range(4)]
    for i in range(4):
        row = (i + 1) % 4
        nd = g.add_vertex(VT.X, row, 6, phase=Fraction(1))
        g.add_edge((ins[i], nd)); g.add_edge((nd, outs[row]))
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    return g


def crossing(shift):
    """A pure crossing: input row i runs straight to output row (i + shift) mod 4."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, i, 0) for i in range(4)]
    outs = [g.add_vertex(VT.BOUNDARY, i, 10) for i in range(4)]
    for i in range(4):
        g.add_edge((ins[i], outs[(i + shift) % 4]))
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    return g


def sketchbook():
    N = 15
    C7 = dial_picture()
    C2 = C7.copy()
    C2.compose(C7.copy())                                    # C_7 twice (compose mutates)
    rot2 = crossing(2)                                       # wires 1 <-> 3 and 2 <-> 4

    # (a) C_7^2 = rot_2, DERIVED: full_reduce on a copy fuses each wire's two red pi dots to
    #     a phase-free dot and removes it as a plain wire; the crossings compose to two
    #     rotations.  Nothing is left but four bare wires, each input two rows round.
    assert same_picture(C2, rot2)
    h = C2.copy()
    simp.full_reduce(h)
    assert h.num_vertices() == 8 and h.num_edges() == 4                   # boundaries only
    ins, outs = h.inputs(), h.outputs()
    for i in range(4):
        assert h.connected(ins[i], outs[(i + 2) % 4])
    assert np.isclose(h.scalar.to_number(), 1)                            # no scalar
    assert same_picture(h, rot2)
    assert not same_picture(C2, crossing(0)) and not same_picture(C2, C7)

    # a^{r/2} as a picture: rot_2 is x -> 4x mod 15 = 7^2 x mod 15 on the residues
    M = matrix_of(rot2)
    for x in range(1, N):
        assert np.allclose(M[:, x], ket(*[int(b) for b in f"{(4 * x) % N:04b}"]))
    assert (7 * 7) % N == 4
    # top wire first; the rotation's direction: 0001 -> 0100 (1 -> 4)
    assert np.allclose(M @ ket(0, 0, 0, 1), ket(0, 1, 0, 0))
    # here the squaring is a pure crossing, so it fixes 0 and 15 as well
    assert np.allclose(M @ ket(0, 0, 0, 0), ket(0, 0, 0, 0))
    assert np.allclose(M @ ket(1, 1, 1, 1), ket(1, 1, 1, 1))
    # NOTE (translation exception): the subtraction of one and the gcd -- the rest of
    # Result 14.17 -- and the continued fraction of Result 14.16 have no picture.


if __name__ == "__main__":
    verdict("SharedFactor", blackboard, sketchbook)
