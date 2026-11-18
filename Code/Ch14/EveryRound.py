"""Challenge IV: Every Round at Once  (the hand at one is the equal mix of the rounds; a random reading)
===================================================================================================
"Two of the four readings give it, then. Half the time."  -- Achilles

Claim: |1> = r^{-1/2} sum_k |psi_k> (Result 14.12; Cleve et al.'s (6.4), whose unnormalized
rounds sum to r|1>), so phase estimation with U_a started on |1> ends in
r^{-1/2} sum_k |kq/r>|psi_k> when r divides q and reads k/r for a uniformly random k
(Result 14.13): for N = 15, a = 7 one run returns r = 4 with probability exactly 1/2; Shor's
section 5 bounds (Result 14.15, stated and cited) hold on three small instances; and in
pictures the Quarter Clock with a GREEN wheel point, the equal mix of the dark and the lit
wheel, is exactly (|000> + |011>)/sqrt2 (Result 14.18).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction
from functools import reduce
from itertools import product
from math import gcd

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import ket, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch14 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def order(a, N):
    return next(r for r in range(1, N + 1) if pow(a, r, N) == 1)


def tick_box(a, N, wires):
    d = 2 ** wires
    M = np.zeros((d, d), dtype=complex)
    for x in range(d):
        M[(a * x) % N if x < N else x, x] = 1
    return M


def rounds(a, N, wires, r, normalized=True):
    """|psi_k>, k = 0..r-1, on the register; unnormalized = Cleve et al.'s (6.1)."""
    out = []
    for k in range(r):
        v = np.zeros(2 ** wires, dtype=complex)
        for s in range(r):
            v[pow(a, s, N)] = np.exp(-2j * np.pi * k * s / r)
        out.append(v / np.sqrt(r) if normalized else v)
    return out


def fourier(t):
    q = 2 ** t
    x, y = np.meshgrid(np.arange(q), np.arange(q))
    return np.exp(2j * np.pi * x * y / q) / np.sqrt(q)


def controlled_on(m, V, t):
    d = V.shape[0]
    out = np.zeros((2 ** t * d, 2 ** t * d), dtype=complex)
    for idx, bits in enumerate(product((0, 1), repeat=t)):
        out[idx * d:(idx + 1) * d, idx * d:(idx + 1) * d] = V if bits[m] else np.eye(d)
    return out


def phase_estimation(U, start, t):
    """(F_q^{-1} (x) I) prod_m Lambda_m(U^{2^{t-m}}) (H^t|0^t> (x) start)."""
    d = U.shape[0]
    state = np.kron(reduce(np.kron, [H @ ket(0)] * t, np.ones(1)), start)
    for m in range(t):
        state = controlled_on(m, np.linalg.matrix_power(U, 2 ** (t - 1 - m)), t) @ state
    return np.kron(fourier(t).conj().T, np.eye(d)) @ state


def readings(out, t):
    return (np.abs(out) ** 2).reshape(2 ** t, -1).sum(axis=1)


def convergents(c, q):
    """Euclid's algorithm on c/q: its quotients and the convergents of the continued fraction."""
    quotients, n, d = [], c, q
    while d:
        quotients.append(n // d)
        n, d = d, n % d
    conv, (h0, h1), (k0, k1) = [], (0, 1), (1, 0)
    for a in quotients:
        h0, h1 = h1, a * h1 + h0
        k0, k1 = k1, a * k1 + k0
        conv.append(Fraction(h1, k1))
    return quotients, conv


def shor_candidate(c, q, N):
    """Shor's rule: the convergent of c/q with denominator below N that is closest to c/q."""
    _, conv = convergents(c, q)
    best = min((f for f in conv if f.denominator < N), key=lambda f: abs(f - Fraction(c, q)))
    return best.denominator


# --- the blackboard ---------------------------------------------------------------------

def blackboard():
    N, a, r = 15, 7, 4
    U7 = tick_box(a, N, 4)
    psis = rounds(a, N, 4, r)
    one = ket(0, 0, 0, 1)

    # --- Result 14.12: |1> = r^{-1/2} sum_k |psi_k>.  The sum of the normalized rounds is
    #     sqrt(r)|1> = 2|1>; Cleve et al.'s rounds carry no r^{-1/2} (and run k = 1..r, the
    #     same set of rounds, since k = r is k = 0), so their (6.4) sum is r|1> = 4|1>.
    assert np.allclose(sum(psis) / np.sqrt(r), one)
    assert np.allclose(sum(psis), 2 * one)
    assert not np.allclose(sum(psis), one)                      # the normalization matters
    cleve = rounds(a, N, 4, r, normalized=False)
    assert np.allclose(sum(cleve[k % r] for k in range(1, r + 1)), 4 * one)
    # ... and for other dials
    for N2, a2 in [(21, 2), (33, 5), (35, 3)]:
        r2 = order(a2, N2)
        w = int(np.ceil(np.log2(N2)))
        e1 = np.zeros(2 ** w); e1[1] = 1
        assert np.allclose(sum(rounds(a2, N2, w, r2)) / np.sqrt(r2), e1)

    # --- the dialogue's adding of beads: the four rounds laid one over another.  At hour one
    #     all four beads are at rest; at seven and thirteen they point four ways; at four two
    #     rest and two at a half.  They pair off, each against one pointing the opposite way.
    orbit = [pow(a, s, N) for s in range(r)]
    for s, hour in enumerate(orbit):
        beads = [np.exp(-2j * np.pi * k * s / r) for k in range(r)]
        if hour == 1:
            assert np.allclose(beads, 1) and np.isclose(sum(beads), 4)
        else:
            assert np.isclose(sum(beads), 0)
            # at seven and thirteen: rest faces half, quarter faces three quarters;
            # at four: rest faces half, twice over
            pairs = [(0, 1), (2, 3)] if hour == 4 else [(0, 2), (1, 3)]
            assert all(np.isclose(beads[i] + beads[j], 0) for i, j in pairs)
    assert np.allclose([np.exp(-2j * np.pi * k * 2 / 4) for k in range(4)], [1, -1, 1, -1])
    # "At seven, the one at rest faces the half, and the quarter back faces the three
    #  quarters back": the beads at seven are 1, -i, -1, i
    assert np.allclose([np.exp(-2j * np.pi * k / 4) for k in range(4)], [1, -1j, -1, 1j])

    # --- Result 14.13: phase estimation with U_7 on the four-wire register, started on |1>.
    for t in (2, 3):
        q = 2 ** t
        out = phase_estimation(U7, one, t)
        want = sum(np.kron(ket(*[int(b) for b in f"{k * q // r:0{t}b}"]), psis[k]) for k in range(r)) / np.sqrt(r)
        assert np.allclose(out, want)
        p = readings(out, t)
        for c in range(q):
            assert np.isclose(p[c], 1 / r if c % (q // r) == 0 else 0)
        # the bookkeeping: the candidate period is the denominator of c/q in lowest terms,
        # checked by one multiplication; success exactly half the time
        success = sum(p[c] for c in range(q) if pow(a, Fraction(c, q).denominator, N) == 1)
        assert np.isclose(success, 0.5)
        assert Fraction(q // 2, q) == Fraction(1, 2) and pow(a, 2, N) == 4 != 1   # the half fails
        assert pow(a, Fraction(0, q).denominator, N) != 1                      # nought: nothing
    # the dialogue's check of the half by ticking: one, seven, four -- not home
    assert [1, (a * 1) % N, (a * a) % N] == [1, 7, 4]

    # --- Result 14.15, DOCUMENTATION ONLY (Shor section 5, stated and cited, not proved; Shor
    #     proves it for sufficiently large N, and it happens to hold on these three).  Shor's
    #     circuit reads the pair (c, a^k mod N); its amplitude is
    #     (1/q) sum_{x < q, x = k mod r} e^{2 pi i x c / q}.
    expected = {15: (0.500, 0.500), 21: (0.228, 0.321), 33: (0.290, 0.391)}
    for N2, a2, q in [(15, 7, 256), (21, 2, 512), (33, 5, 2048)]:
        assert N2 ** 2 <= q < 2 * N2 ** 2                       # Shor's choice of q
        r2 = order(a2, N2)
        P = np.zeros((q, r2))
        for k in range(r2):
            indicator = np.zeros(q)
            indicator[k::r2] = 1
            P[:, k] = np.abs(np.fft.ifft(indicator)) ** 2       # |(1/q) sum e^{+2pi i x c/q}|^2
        assert np.isclose(P.sum(), 1)
        rc = lambda c: (r2 * c) % q - q * ((r2 * c) % q > q / 2)   # Shor's {rc}_q
        good = [c for c in range(q) if abs(rc(c)) <= r2 / 2]
        # each good outcome has probability at least 1/(3 r^2)
        assert all(P[c, k] >= 1 / (3 * r2 ** 2) for c in good for k in range(r2))
        # ... and then |c/q - d/r| <= 1/(2q) for some d (Shor's Eq. (5.13))
        for c in good:
            assert abs(Fraction(c, q) - Fraction(round(c * r2 / q), r2)) <= Fraction(1, 2 * q)
        coprime = sum(1 for d in range(r2) if gcd(d, r2) == 1)
        # Shor's count: the good outcomes whose d shares no factor with r
        p_good = sum(P[c].sum() for c in good if gcd(round(c * r2 / q) % r2, r2) == 1)
        # the whole rule: continued fractions return exactly r
        p_rule = sum(P[c].sum() for c in range(q) if shor_candidate(c, q, N2) == r2)
        assert np.isclose(p_good, expected[N2][0], atol=5e-4)
        assert np.isclose(p_rule, expected[N2][1], atol=5e-4)
        assert p_rule >= p_good >= coprime / (3 * r2)

    pictures()


# --- the pictures in the dialogue ----------------------------------------------------------

def bead_phase(t):
    """A bead turned t twenty-fourths clockwise, as a phase."""
    return np.exp(2j * np.pi * t / 24)


def dial_clearances(p):
    """On the dial as drawn (edge strokes included, read from the Lua; the soft shadows
    aside), the least gap, in cm, between: two rings (or face-down pairs); a ring and the
    hand; two beads; a bead and any ring but its own (outside it, or inside its hole); a
    bead and the hand.  All positive: nothing at an hour overlaps or touches."""
    v = p.view
    S = PC.DIAL_STROKE
    ro, ri, rb = PC.RRO + S["ring"], PC.RRI - S["ring"], PC.RBR + S["bead"]
    places = [(x, y) for _, _, x, y in v.pairs] + [(r[3], r[4]) for r in v.rings]
    beads = [(r[5], r[6], (r[3], r[4])) for r in v.rings]
    hand = list(v.hand_outline)
    edges = list(zip(hand, hand[1:] + hand[:1]))

    def to_hand(c):
        if PC.inside(c, hand):
            return -1.0
        return min(PC.seg_dist(c, a, b) for a, b in edges) - S["hand"]
    gaps = {"ring-ring": min(np.hypot(a[0] - b[0], a[1] - b[1]) - 2 * ro
                             for i, a in enumerate(places) for b in places[i + 1:]),
            "ring-hand": min(to_hand(c) - ro for c in places)}
    if beads:
        gaps["bead-bead"] = min(np.hypot(a[0] - b[0], a[1] - b[1]) - 2 * rb
                                for i, a in enumerate(beads) for b in beads[i + 1:]) if len(beads) > 1 else 1.0
        gaps["bead-ring"] = min(max(np.hypot(bx - c[0], by - c[1]) - rb - ro, ri - np.hypot(bx - c[0], by - c[1]) - rb)
                                for bx, by, own in beads for c in places if c != own)
        gaps["bead-hand"] = min(to_hand((bx, by)) - rb for bx, by, _ in beads)
    return gaps


COUNT = ("once", "twice", "three times", "four times", "five times", "six times", "seven times", "eight times")


def pictures():
    """The Fourth Challenge's prop pictures (identity card, sections 4 and 3e), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch14.py: four single
    calls, four frames (F25-F28), the third a prediction (pale: as Achilles expects).  The ties
    below hold the transcription against this file's blackboard: a structural check of the
    pictures, not a second proof."""
    calls = DIALOGUE.check_layout("Fourth", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("tickdial", False, "", ("every round",), "the rounds together"),
        ("tickdial", False, "step=pairs face down", ("every round",), "turned face down"),
        ("tickdial", False, "step=tick twice; expects=two ticks home", ("every round; pairs face down",),
         "as Achilles expects"),
        ("tickdial", False, "step=tick twice", ("every round; pairs face down",), "the hand gone on")]
    PC.check_captions(DIALOGUE, "Fourth", calls)
    PC.check_section(DIALOGUE, "Fourth", calls)
    dial_h = round(PC.DIAL_FTOP - PC.DIAL_FBOT, 2)
    assert dial_h == 2.42                                             # read from the Lua
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Fourth")] == [[dial_h]] * 4
    assert [PC.is_prediction(c) for c in calls] == [False, False, True, False]
    p41, p42, pw, p43 = (PC.picture(c) for c in calls)               # F25, F26, F27 (pale), F28

    # the anchors (card sections 4 and 3e.4), in dialog.md; the raw file's following paragraphs agree
    a1, a2, a3, a4 = DIALOGUE.anchors("Fourth")
    f1, f2, f3, f4 = DIALOGUE.following("Fourth")
    assert "in a little square, side by side so that every bead shows" in a1                      # D27
    assert f1.startswith("**Achilles:** *[going round with a finger]* At one they all agree.")
    assert "the pair lies as one plain ring" in a2 and "in their little square" in a2             # D28
    assert f2.startswith("**Achilles:** So all the rounds laid together are the hand at one")
    assert a3 == "**Tortoise:** Tick it and see."                                                 # B15
    assert f3.startswith("*[The paper hand stands at one. Achilles carries it once round to seven")
    assert "It points at four, not at one." in a4 and a4 == f3                                    # B16
    assert f4.startswith("**Achilles:** Four. Not home.")
    assert DIALOGUE.following("Fourth", DIALOGUE.raw) == [f1, f2, f3, f4]
    assert "four rings sit together, one over another." not in DIALOGUE.section("Fourth")          # D27's old

    # the wrong rule (card 3e.1): every spoken reaction, in order, around the pale frame -- the
    # premise, the prediction, "Tick it and see." (the pale frame), the look (the true frame),
    # the refutation naming the true hour, the wrong rule named, and the true count agreed
    paras = DIALOGUE.paragraphs("Fourth")
    at = lambda s: [k for k, q in enumerate(paras) if s in q]
    (premise,), (said,), (see,), (b15,), (look,), (b16,), (four,), (named,), (agreed,) = (
        at("had shown you a half."), at("**Achilles:** A half at every tick: two ticks home."),
        at("**Tortoise:** Tick it and see."), at(calls[2].text), at("Two ticks have moved it twice, and it has not come home."),
        at(calls[3].text), at("**Achilles:** Four. Not home."),
        at("The half came from the round that goes a half at each step, and that round comes home in two. The hand doesn't."),
        at("Two of the four readings give it, then. Half the time."))
    assert [said, see, b15, look, b16, four, named] == list(range(premise + 1, premise + 8)) and agreed == named + 2
    assert paras[premise].startswith("**Tortoise:**") and "A quarter says four ticks home" in paras[agreed]
    # the Lua's wrong rule says the same, in the dialogue's words (its comment pinned by check_lua)
    rule = PC.rule_comment("THE DIAL'S STEP, ITS MARKS AND ITS WRONG RULE")
    quotes = PC.comment_quotes("THE DIAL'S STEP, ITS MARKS AND ITS WRONG RULE")
    for q in ("A half at every tick: two ticks home.",
              "It points at four, not at one. Two ticks have moved it twice, and it has not come home.",
              "Four. Not home."):
        assert q in quotes and q in DIALOGUE.section("Fourth"), q
    assert "-- THE DIAL'S WRONG RULE, given with expects=<rule>" in rule
    assert "only where it differs" in rule

    orb = PC.dial_orbit(15)
    psis = rounds(7, 15, 4, 4)

    # F25 (P4.1): every round laid as first laid -- at the i-th hour of the hand's way the k-th
    # round's bead stands k i quarters back; adding the beads at each hour gives 4 at one
    # and 0 elsewhere: the four rounds (each with its 1/2) add up to 2|1>, the hand at one
    assert p41.dump == ("hours=15 hand=1 rings=1:0:0,1:1:0,1:2:0,1:3:0,7:0:0,7:1:18,7:2:12,7:3:6,"
                        "4:0:0,4:1:12,4:2:0,4:3:12,13:0:0,13:1:6,13:2:12,13:3:18")
    assert p41.words == () and p41.notes == [] and p41.mark_tikz == []   # keyed in the Second (F06)
    for k in range(4):
        v = np.zeros(16, dtype=complex)
        for ring in p41.after.rings:
            if ring["k"] == k:
                v[ring["hour"]] = bead_phase(ring["t"]) / 2
        assert np.allclose(v, psis[k])
    total = np.zeros(16, dtype=complex)
    for ring in p41.after.rings:
        total[ring["hour"]] += bead_phase(ring["t"]) / 2
    assert np.allclose(total, sum(psis)) and np.allclose(total, 2 * ket(0, 0, 0, 1))
    assert np.isclose(2 * total[1], 4) and all(np.isclose(total[h], 0) for h in orb[1:])
    # the little square, as Achilles sees it: the ticked round (k = 1) far left, the round
    # at rest (k = 0) far right, the half's (k = 2) near left, three quarters back's near right
    for h in orb:
        hx, hy = PC.hourxy(15, h)
        at_ = {k: (x - hx, y - hy) for hour, k, t, x, y, bx, by in p41.view.rings if hour == h}
        assert at_[1][0] < 0 < at_[1][1] and at_[0][0] > 0 < at_[0][1]
        assert at_[2][0] < 0 > at_[2][1] and at_[3][0] > 0 > at_[3][1]
    # "so that every bead shows": at every hour the rings, the beads and the hand clear one
    # another on the drawn extents, strokes included -- no overlap, no touching stroke
    gaps = dial_clearances(p41)
    assert len(p41.view.rings) == 16 and min(gaps.values()) > 0.01, gaps

    # F26 (P4.2): the beads that point opposite ways (a half apart) turned face down in pairs --
    # exactly the pairs whose phases cancel; the four at one stay face up, all at rest
    assert p42.dump == ("hours=15 hand=1 rings=1:0:0,1:1:0,1:2:0,1:3:0,7:0:0:down2,7:1:18:down1,7:2:12:down2,"
                        "7:3:6:down1,4:0:0:down1,4:1:12:down1,4:2:0:down2,4:3:12:down2,13:0:0:down2,"
                        "13:1:6:down1,13:2:12:down2,13:3:18:down1")
    up = [ring for ring in p42.after.rings if not ring["down"]]
    assert {(r_["hour"], r_["t"]) for r_ in up} == {(1, 0)} and len(up) == 4
    for h in orb[1:]:
        here = [r_ for r_ in p42.after.rings if r_["hour"] == h]
        for n in (1, 2):
            pair = [r_ for r_ in here if r_["pair"] == n]
            assert len(pair) == 2 and (pair[0]["t"] - pair[1]["t"]) % 24 == 12
            assert np.isclose(bead_phase(pair[0]["t"]) + bead_phase(pair[1]["t"]), 0)
    # drawn: each pair one plain ring, the two at an hour side by side; nothing else moved
    assert len(p42.view.pairs) == 6 and len(p42.view.rings) == 4
    for h in orb[1:]:
        hx, hy = PC.hourxy(15, h)
        xs = sorted(x - hx for hour, n, x, y in p42.view.pairs if hour == h)
        assert np.allclose(xs, [-PC.SLOT, PC.SLOT])
    assert p42.view.rings == tuple(r_ for r_ in p41.view.rings if r_[0] == 1)
    assert p42.view.hand_outline == p41.view.hand_outline
    # the twine follows its rings: at one, where nothing turned over, every loop is as it lay
    assert all(p42.view.loops[k][0] == p41.view.loops[k][0] for k in range(4))
    gaps = dial_clearances(p42)
    assert min(gaps.values()) > 0.01, gaps
    # F26's marks (card 3e.6): every face-up ring at seven, four and thirteen ghosted where it
    # lay in F25 (its dotted outline cut away under the pairs); no arc (each moves under the
    # 0.2 cm a head and its gap need)
    assert p42.notes[0] == "step pairs face down -> " + p42.dump and p42.arcs == []
    ghosted = set()
    for note in p42.notes[1:]:
        m = re.fullmatch(r"ghost of the ring \(round (\d)\) at (\d+) in dotted outline at \((-?\d\.\d{3}),(-?\d\.\d{3})\), "
                         r"cut away under the pairs: turned face down", note)
        assert m, note
        k, h, x, y = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
        (was,) = [r_ for r_ in p41.view.rings if (r_[0], r_[1]) == (h, k)]
        assert np.allclose((x, y), (was[3], was[4]), atol=6e-4)
        ghosted.add((h, k))
    assert ghosted == {(h, k) for h in orb[1:] for k in range(4)} and len(p42.notes) == 13
    # the cut: one circle per pair at each hour turned (both rings of a pair lie at one place,
    # and the same circle given twice would cancel itself under the even odd rule, leaving the
    # ghosts drawn over the pairs)
    clips = [l for l in p42.mark_tikz if l.startswith("\\clip ")]
    assert len(clips) == 3 and p42.mark_tikz.count("\\begin{scope}[even odd rule]") == 3
    for line in clips:
        circles = re.findall(r"\((-?\d\.\d{3}),(-?\d\.\d{3})\) circle \(([\d.]+)\)", line)
        assert len(circles) == len(set(circles)) == 2, line
        centres = sorted((float(x), float(y)) for x, y, _ in circles)
        (h,) = [hour for hour in orb[1:]
                if np.allclose(centres, sorted((x, y) for hr, n, x, y in p42.view.pairs if hr == hour), atol=6e-4)]
        assert all(float(rad) == round(PC.RRO + 0.012, 3) for _, _, rad in circles)
    assert PC.drawn(PC.twin(calls[1])) == PC.drawn(PC.run_frame("dial", "", "every round; pairs face down", False))

    # F27: the prediction, two ticks home -- the dial paler, the hand where Achilles expects
    # it (at one) at full strength; the true hand stands at four (its '+ step' line).  The
    # wrong rule: the hand at one iff the ticks in all are even (else at seven); it is drawn
    # only where it differs from the true hand, so it is refused after 1, 4, 5 and 8 ticks
    assert pw.drawn.hand == pw.wrong.hand == 1 and pw.after.hand == p43.drawn.hand == 4
    assert pw.head == p42.dump and pw.marked is False and pw.mark_tikz == [] and pw.arcs == []
    assert pw.notes == ["step tick twice -> " + p43.dump + " (no marks)",
                        "expects two ticks home: the dial paler, the hand at 1 at full strength, a shade darker, "
                        "its outline heavier (it stands at 4)"]
    assert pw.view.hand_outline == p42.view.hand_outline != p43.view.hand_outline
    refused = []
    for m, w in enumerate(COUNT, 1):
        f = PC.faults(PC.run_frame, "dial", "step=tick " + w + "; expects=two ticks home", "every round; pairs face down", False)
        wrong = 1 if m % 2 == 0 else 7
        assert bool(f) == (wrong == pow(7, m, 15)), (m, f)
        if f:
            refused.append(m)
    assert refused == [1, 4, 5, 8]
    PC.TEX_ERRORS.clear()

    # F28 (P4.3): the hand ticked twice -- at four, 7^2 mod 15, not home: the half's "two ticks"
    # fails; the rounds lie as they lay, only the hand has moved
    assert p43.drawn.hand == pow(7, 2, 15) == 4 != 1
    assert p43.dump == p42.dump.replace("hand=1", "hand=4")
    assert (p43.view.rings, p43.view.pairs, p43.view.loops) == (p42.view.rings, p42.view.pairs, p42.view.loops)
    assert p43.view.hand_outline != p42.view.hand_outline
    gaps = dial_clearances(p43)
    assert min(gaps.values()) > 0.01, gaps
    # its marks (the Editor's item 3): two ticks, one after another, as B03 draws single ticks --
    # a ghost at each hour the hand stood at before a tick (one, seven), and one clockwise arc
    # per tick with its own head: one to seven over 6 marks, seven to four over 12 (6 + 12 =
    # 18, more than a turn); the second arc, whose way would lie on the first's, a whole step in
    assert [7 * 1 % 15, 7 * 7 % 15] == [7, 4] and [(7 - 1) % 15, (4 - 7) % 15] == [6, 12]
    assert 6 + 12 == 18 and 18 * 24 > 360
    plan = p43.plan
    assert plan["ghosts"] == [1, 7] and plan["count"] == 2
    assert [(a["from"], a["to"], a["marks"]) for a in plan["arcs"]] == [(1, 7, 6), (7, 4, 12)]
    assert [a["r"] for a in plan["arcs"]] == [PC.ARCIN, PC.ARCIN - PC.ARCSTEP] and PC.ARCIN == 0.45
    assert np.isclose(PC.ARCIN - PC.ARCSTEP, 0.37) and [a["lev"] for a in plan["arcs"]] == [0, 1]
    assert not any(a["short"] or a["shared"] for a in plan["arcs"])
    veil = ", filled with the sheet's shade (opacity 0.850) over the twine"
    assert p43.notes[0] == "step tick twice -> " + p43.dump
    assert p43.notes[1:3] == ["ghost of the hand in dotted outline at hour %d, under the hand and the pin: where it "
                              "stood before a tick" % h + veil for h in (1, 7)]
    cased = ", drawn OVER the twine, cased where it crosses a loop (the casing 1.800 pt wide, in the sheet's shade; "
    assert p43.notes[3].startswith("swing arc round the pin, clockwise, tick 1 of 2, from hour 1 to hour 7: 6 marks "
                                   "(144.000 degrees), radius 0.450, ")
    assert p43.notes[4].startswith("swing arc round the pin, clockwise, tick 2 of 2, from hour 7 to hour 4: 12 marks "
                                   "(288.000 degrees), radius 0.370, ")
    assert "short of the ghost at 7" + cased in p43.notes[3] and "short of the hand" + cased in p43.notes[4]
    # drawn: the two ghosts veiled, then every casing, then the two arcs, each with its head
    tk = p43.mark_tikz
    ghost_lines = [l for l in tk if l.startswith("\\draw[propchfourteenghost")]
    veils = [l for l in tk if l.startswith("\\fill[propchfourteenpaper!") and "opacity=0.850" in l]
    casings = [l for l in tk if "line width=1.800pt, line cap=butt, overlay]" in l]
    swings = [l for l in tk if l.startswith("\\draw[propchfourteenswing, overlay, -{Stealth")]
    assert len(ghost_lines) == 2 and len(veils) == 2 and len(swings) == 2
    assert len(casings) == sum(a["cases"] for a in plan["arcs"]) > 0
    assert tk == [veils[0], ghost_lines[0], veils[1], ghost_lines[1]] + casings + swings     # casings before dots
    assert [p_[1] for p_ in p43.ghosts] == [1, 7] and all(p_[3] for p_ in p43.ghosts)
    one, two = p43.arcs
    assert not one.headless and not two.headless                                        # two heads
    clock = lambda q: (90 - np.degrees(np.arctan2(q[1] - PC.DCY, q[0] - PC.DCX))) % 360
    radius = lambda q: np.hypot(q[0] - PC.DCX, q[1] - PC.DCY)
    assert 24 * 1 < clock(one.p0) < clock(one.p1) < 24 * 7                              # one -> seven
    assert 24 * 7 < clock(two.p0) and clock(two.p1) < 24 * 4                            # seven -> past the top -> four
    assert np.allclose([radius(one.p0), radius(one.p1)], 0.45, atol=1e-3)
    assert np.allclose([radius(two.p0), radius(two.p1)], 0.37, atol=1e-3)
    # the clearances as logged: every arc and ghost clear of the pin, the rings, the circle and
    # the sheet; the arcs lie OVER the twine on their casings (the twine figure as logged), and
    # tick 2 passes over the ghost at one, where the hand stood before (crossed, as it should)
    c1, c2, g1, g7 = p43.notes[5:9]
    assert len(p43.notes) == 9
    assert c1.startswith("clearances of tick 1's arc (edge to edge): the hand passed under (the arc lies beneath it), "
                         "the ghost at 1 0.031, the ghost at 7 0.031, tick 2's arc 0.016, ")
    assert c2.startswith("clearances of tick 2's arc (edge to edge): the hand 0.034, the ghost at 1 crossed, "
                         "the ghost at 7 0.031, tick 1's arc 0.016, ")
    for c in (c1, c2):
        assert "the twine -0.028 (it lies over the twine, on its casing)" in c
        g = {k: float(v) for k, v in re.findall(r"(the pin|the rings|the circle|its marks|inside the sheet by) "
                                                r"(-?\d\.\d{3})", c)}
        assert len(g) == 5 and min(g.values()) > 0.2, g
    assert g1 == ("clearances of the ghost at 1 (edge to edge, its blade beyond 0.200 of the pin): the hand 0.131, "
                  "the ghost at 7 0.059, the rings 0.030, the twine crossed (the ghost lies over it, by 0.025)")
    assert g7 == ("clearances of the ghost at 7 (edge to edge, its blade beyond 0.200 of the pin): the hand 0.131, "
                  "the ghost at 1 0.059, the rings 0.172, the twine crossed (the ghost lies over it, by 0.024)")
    # a single tick keeps its own way (one ghost, one arc under the twine, its head over it)
    single = PC.run_frame("dial", "step=tick", "every round; pairs face down", False)
    assert len(single.arcs) == 2 and single.arcs[0].headless and single.plan is None
    assert single.notes[2].startswith("swing arc round the pin, clockwise, 6 marks") and "UNDER the twine" in single.notes[2]
    assert PC.drawn(PC.twin(calls[3])) == PC.drawn(PC.run_frame("dial", "", "hand ticked twice; every round; "
                                                                "pairs face down", False))
    assert not PC.TEX_ERRORS


# --- the sketchbook ---------------------------------------------------------------------

def master(vartheta, wheel="lit"):
    """The Quarter Clock (as in QuarterClock.py), the wheel's point lit, dark or green.
    Phases in units of pi; vartheta = 2 phi.  The collars are Hadamard edges; the crossing
    is the output order.  No scalar added."""
    v = Fraction(vartheta)
    g = zx.Graph()
    T1 = g.add_vertex(VT.Z, 0, 1, phase=v)
    T2 = g.add_vertex(VT.Z, 0, 5, phase=Fraction(-1, 4))
    B = g.add_vertex(VT.Z, 1, 3, phase=v / 2 - Fraction(1, 4))
    if wheel == "green":
        P = g.add_vertex(VT.Z, 2, 0)                          # green point: |0> + |1>
    else:
        P = g.add_vertex(VT.X, 2, 0, phase=Fraction(1 if wheel == "lit" else 0))
    G = g.add_vertex(VT.Z, 2, 2, phase=3 * v / 2)
    R1, R2, R3 = (g.add_vertex(VT.X, 1.5, c) for c in (1, 2, 4))
    L1 = g.add_vertex(VT.Z, 1.5, 0, phase=-v)
    L2 = g.add_vertex(VT.Z, 1.5, 3, phase=-v / 2)
    L3 = g.add_vertex(VT.Z, 0.5, 5, phase=Fraction(1, 4))
    o1, o2, o3 = (g.add_vertex(VT.BOUNDARY, r, 8) for r in (0, 1, 2))
    g.add_edge((T1, T2), ET.HADAMARD)
    g.add_edge((T2, o2))
    g.add_edge((B, o1), ET.HADAMARD)
    g.add_edges([(P, G), (G, o3), (T1, R1), (G, R1), (G, R2), (B, R2), (T2, R3), (B, R3),
                 (R1, L1), (R2, L2), (R3, L3)])
    g.set_inputs(()); g.set_outputs((o1, o2, o3))
    return g


def one_point(kind, phase=0):
    g = zx.Graph()
    p = g.add_vertex(kind, 0, 0, phase=Fraction(phase))
    o = g.add_vertex(VT.BOUNDARY, 0, 2)
    g.add_edge((p, o)); g.set_inputs(()); g.set_outputs((o,))
    return g


def sketchbook():
    quarter = Fraction(1, 2)                                   # vartheta = 2 x 1/4

    # (b) the green wheel point is the wheel dark and the wheel lit at once: a phase-free
    #     green point is |0> + |1>, the red points are sqrt2|0> and sqrt2|1>
    green, dark, lit = (matrix_of(one_point(k, ph)) for k, ph in ((VT.Z, 0), (VT.X, 0), (VT.X, 1)))
    assert np.allclose(green, (dark + lit) / np.sqrt(2))
    assert not np.allclose(green, dark + lit)                  # the 1/sqrt2 is real
    assert np.allclose(green.ravel(), ket(0) + ket(1))

    # (a) the master with the wheel point GREEN: exactly (|000> + |011>)/sqrt2, no
    #     compensation -- the reading correlated with the wheel's round
    g = master(quarter, "green")
    vec = matrix_of(g).ravel()
    assert np.allclose(vec, (ket(0, 0, 0) + ket(0, 1, 1)) / np.sqrt(2))
    assert np.isclose(abs(vec[0b000]) ** 2, 0.5) and np.isclose(abs(vec[0b011]) ** 2, 0.5)
    assert np.isclose(vec[0b101], 0) and np.isclose(vec[0b010], 0)    # never a mismatched round
    # ... and it is 1/sqrt2 times (the master with the wheel dark + the master with it lit)
    both = matrix_of(master(quarter, "dark")) + matrix_of(master(quarter, "lit"))
    assert np.allclose(matrix_of(g), both / np.sqrt(2))
    # DERIVED: full_reduce on a copy keeps the value; and the reference picture -- a red
    # point (sqrt2|0>) on the first output and a phase-free green dot with two legs
    # (|00> + |11>) on the second and third, times 1/2 (add_power(-2)), which is
    # sqrt2 (|000> + |011>) / 2 -- reduces to the same state.  The eighth turns are non-Clifford, so both are reduced and
    # compared as tensors.
    ref = zx.Graph()
    o1, o2, o3 = (ref.add_vertex(VT.BOUNDARY, r, 8) for r in (0, 1, 2))
    ref.add_edge((ref.add_vertex(VT.X, 0, 0), o1))
    z = ref.add_vertex(VT.Z, 1.5, 4)
    ref.add_edges([(z, o2), (z, o3)])
    ref.set_inputs(()); ref.set_outputs((o1, o2, o3))
    ref.scalar.add_power(-2)
    assert same_picture(g, ref)
    h, hr = g.copy(), ref.copy()
    simp.full_reduce(h); simp.full_reduce(hr)
    assert same_picture(h, g) and same_picture(h, hr)
    # the green point with the wheel's secret changed: every whole number of quarter-turns,
    # the dark branch always reads 00, the lit branch the secret
    for j in range(4):
        v = matrix_of(master(Fraction(2 * j, 4), "green")).ravel()
        b = [int(x) for x in f"{j:02b}"]
        assert np.allclose(v, (ket(0, 0, 0) + ket(*b, 1)) / np.sqrt(2))


if __name__ == "__main__":
    verdict("EveryRound", blackboard, sketchbook)
