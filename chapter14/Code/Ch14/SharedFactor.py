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
NAMES = ("P1.1", "P1.2", "P1.3", "P2.1", "P2.2", "P2.3", "P2.4", "P3.1", "P3.2", "P3.3", "P3.4",
         "P4.1", "P4.2", "P4.3", "P5.1", "P5.2", "P5.3")


def pictures():
    """The Fifth Challenge's prop pictures (identity card, section 4), and the whole
    dialogue's, read through tools/dfp_props.py and the transcription
    tools/props_ch14.py.  The ties below hold the transcription against this file's
    blackboard: a structural check of the pictures, not a second proof."""
    calls = DIALOGUE.check_layout("Fifth", sizes=[1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("tickstrips", True, "", ("fifteen and four",), "the short strip laid along"),
        ("tickstrips", False, "first cut", ("fifteen and four",), "after the snip"),
        ("tickdial", False, "", ("twenty-one hours; hand ticked six times",), "the new hand, home")]
    PC.check_captions(DIALOGUE, "Fifth", calls)
    PC.check_section(DIALOGUE, "Fifth", calls)
    strips_h, dial_h = round(PC.SFTOP - PC.SFBOT, 2), round(PC.DIAL_FTOP - PC.DIAL_FBOT, 2)
    assert (strips_h, dial_h) == (0.78, 2.42)                         # read from the Lua
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Fifth")] == [[strips_h], [strips_h], [dial_h]]

    # the anchors (card section 4), in dialog.md; the raw file's following paragraphs agree
    a1, a2, a3 = DIALOGUE.anchors("Fifth")
    f1, f2, f3 = DIALOGUE.following("Fifth")
    assert "along the table's edge on his side" in a1 and a1.rstrip().endswith("He sets the short strip back beside it.]*")  # D29
    assert f1.startswith("*[He pencils the count on the long strip and snips it at the last tick;")
    assert "the measured part, with its count on it, he lays aside" in a2                          # D29
    assert f2.startswith("*[The three goes along the four once")
    assert "now stands again on the first mark past the top" in a3
    assert f3.startswith("**Achilles:** Six ticks, and it's on the mark it left.")
    assert DIALOGUE.following("Fifth", DIALOGUE.raw) == [f1, f2, f3]
    section = DIALOGUE.section("Fifth")
    assert "She draws a second circle on the fresh sheet" in section                               # D30
    assert "the first at the top, on the side furthest from Achilles, as on the first" in section  # D30

    p51, p52, p53 = (PC.picture(c) for c in calls)

    # P5.1 and P5.2: the strips ARE Euclid's run on (15, 4) -- the counts 3, 1, 3, the last
    # strip that measures both 1 = gcd(15, 4); the ticks at 4, 8, 12 thumbs; the piece left 3
    assert p51.dump == "steps=15/4=3 3, 4/3=1 1, 3/1=3 0 view=1 phase=laid edge=15 beside=4 ticks=4/8/12"
    assert p52.dump == "steps=15/4=3 3, 4/3=1 1, 3/1=3 0 view=1 phase=cut edge=3 beside=4 ticks=none"
    counts, last = strips(15, 4)
    assert [s[2] for s in p51.state.steps] == counts == [3, 1, 3] and p51.state.measure == last == gcd(15, 4)
    assert PC._strip_ticks(p51.state) == [4 * q for q in range(1, counts[0] + 1)]
    assert p52.state.steps[0][3] == 15 % 4 == 3
    assert p51.labels == ("tick", "short strip", "long strip") and p52.labels == ()
    # the runs read out but not pictured (l.701, l.713, l.729): fifteen and three, fifteen
    # and five -- the macro's strips, the same counts, against gcd.  The longer ones (thirty-
    # two and five, twenty-one and seven, twenty-one and nine) are longer than the evening's
    # scale draws: the macro refuses them (fifteen thumbs span the picture), and the
    # blackboard's strips carry them -- five parts in thirty-two, 5/32 = [0; 6, 2, 2]
    for words, (x, y) in (("fifteen and three", (15, 3)), ("fifteen and five", (15, 5))):
        T = PC.strips_state("", words)
        assert [s[2] for s in T.steps] == strips(x, y)[0] and T.measure == strips(x, y)[1] == gcd(x, y)
    assert not PC.TEX_ERRORS
    for words, (x, y) in (("thirty-two and five", (32, 5)), ("twenty-one and seven", (21, 7)),
                          ("twenty-one and nine", (21, 9))):
        PC.strips_state("", words)
        assert [site for site, _ in PC.TEX_ERRORS] == ["strips-long"] and x > PC.MAXLEN == 15
        PC.TEX_ERRORS.clear()
        assert strips(x, y)[1] == gcd(x, y)
    assert strips(32, 5)[0] == [6, 2, 2] and continued_fraction(5, 32) == [0, 6, 2, 2]
    assert strips(21, 9)[1] == 3 and strips(21, 7) == ([3], 7)
    # drawn: the long strip along the near edge, the short one beside it, further from
    # Achilles, their left ends level, the two never touching; after the snip the piece
    # left lies where the long strip began, the short strip unmoved, the measured part
    # not drawn
    v1, v2 = p51.shown, p52.shown
    stroke = 0.3 * PC.PT / 2
    for v in (v1, v2):
        (ex0, ey0, ex1, ey1), (bx0, by0, bx1, by1) = v.edge, v.beside
        assert ex0 == bx0 == PC.SX0 and by0 > ey1 and by0 - ey1 - 2 * stroke > 0.1
    assert np.isclose(v1.edge[2] - v1.edge[0], 15 * PC.THUMB) and np.isclose(v2.edge[2] - v2.edge[0], 3 * PC.THUMB)
    assert v1.beside == v2.beside and v1.edge[:2] == v2.edge[:2] and v2.ticks == ()
    assert np.allclose(v1.ticks, [PC.SX0 + 4 * q * PC.THUMB for q in (1, 2, 3)])
    assert np.isclose(15 * PC.THUMB, 2.25) and v1.edge[2] < 1.25        # fifteen thumbs span the frame
    # the keyed words (measured node boxes): "long strip" below the strip's right end, beside
    # "tick", which stands below the first tick; "short strip" to the short strip's right.
    # All three inside the frame, below or beside their strips, none touching another
    labels = {w: (x, y, a) for w, x, y, a in v1.labels}
    assert labels["long strip"] == (v1.edge[2], v1.edge[1] - 0.03, "north east")
    assert labels["tick"] == (v1.ticks[0], v1.edge[1] - 0.03, "north")
    boxes = {w: PC.label_box(w, *labels[w]) for w in labels}
    # (the tightest is "long strip" over the frame's foot: its node box by 0.0024 cm, its
    # letters by 0.037; the caption hangs below the foot)
    for w, (x0, y0, x1, y1) in boxes.items():
        assert min(x0 + 1.25, 1.25 - x1, y0 - PC.SFBOT, PC.SFTOP - y1) > 0, w
        i0, j0, i1, j1 = PC.label_box(w, *labels[w], ink=True)
        assert min(i0 + 1.25, 1.25 - i1, j0 - PC.SFBOT, PC.SFTOP - j1) > 0.03, w
    for w in ("tick", "long strip"):
        assert v1.edge[1] - stroke - boxes[w][3] > 0                  # below the edge strip
    assert boxes["short strip"][0] - (v1.beside[2] + stroke) > 0.03   # right of the short strip
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = boxes[a], boxes[b]
            assert max(A[0] - B[2], B[0] - A[2], A[1] - B[3], B[1] - A[3]) > 0.1, (a, b)
    # the other rule: when the strip's word goes below a long strip and the first tick
    # stands near its right end, "tick" steps to its tick's left (fifteen and nine: the
    # tick at nine)
    w9 = {w: (x, y, a) for w, x, y, a in PC.strips_view(PC.strips_state("", "fifteen and nine"), True).labels}
    assert w9["tick"] == (PC.SX0 + 9 * PC.THUMB - 0.02, PC.STRIP_Y0[0] - 0.03, "north east")
    A, B = PC.label_box("tick", *w9["tick"]), PC.label_box("long strip", *w9["long strip"])
    assert B[0] - A[2] > 0

    # P5.3: the harder dial, the tick doubles: the hand visits 1, 2, 4, 8, 16, 11 and is home
    # after six ticks -- the order of 2 mod 21
    assert p53.dump == "hours=21 hand=1 rings=none"
    assert PC.dial_orbit(21) == [pow(2, s, 21) for s in range(6)] == [1, 2, 4, 8, 16, 11]
    assert len(PC.dial_orbit(21)) == order(2, 21) == 6
    for m, w in enumerate(("once", "twice", "three times", "four times", "five times", "six times"), 1):
        assert PC.dial_state("twenty-one hours; hand ticked " + w).hand == pow(2, m, 21)
    assert p53.shown.rings == () and p53.shown.pairs == ()

    # the whole dialogue: seventeen pictures, all inside the five challenges, none after
    # P5.3; each in its place; the keyed form exactly on the first picture of each kind
    assert DIALOGUE.check_whole(PC.ORDER) == 17
    text = DIALOGUE.read(DIALOGUE.dialog)
    assert not DIALOGUE.call_re.search(text[text.index(calls[-1].text) + len(calls[-1].text):])
    everything = []
    for o in PC.ORDER:
        for b in DIALOGUE.blocks(o):
            for c in b:
                assert PC.place_ok(o, c), (o, c.text)
                PC.picture(c)                                          # no fault
                everything.append(c)
    DIALOGUE.keyed_first(PC.ORDER, kind=PC.keyed_kind)
    assert [NAMES[i] for i, c in enumerate(everything) if c.keyed] == [
        "P1.1", "P2.1", "P2.3", "P3.1", "P3.3", "P3.4", "P5.1"]
    # the aria with variations and the Genie, after the revision, on both files (the card's
    # section 1c one-liner, transcribed); no "Let Bob draw"; three "Referred.", the Genie's
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        t = DIALOGUE.read(path)
        assert PC.form_check(t) == (True, ARIA, GROUND, "Third Challenge"), path
        assert t.count("Let Bob draw") == 0 and "fourteen" not in t.lower()
        spoken = re.findall(r"^\*\*([^*]+):\*\*(.*)$", t, re.M)
        referred = [who for who, line in spoken if re.sub(r"\*\[.*?\]\*", "", line).strip().startswith("Referred.")]
        assert referred == ["Genie"] * 3, referred


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
