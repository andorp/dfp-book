"""Challenge IV: The Turning Plane  (two folds make a turn; BBHT's closed form; when to stop)
============================================================================================
"All but a hair of eight. Just past upright."  -- Achilles
"It goes round. Too many, and it turns away."  -- Tortoise

Claim: in the real plane, the reflection across the line at beta after the one at alpha is
the rotation by 2(beta - alpha) (Result 15.9); O_w and D map span{|w>, |w_perp>} to itself as
the reflections across the |w_perp> line and the |s> line, so G^k|s> = sin((2k+1)theta)|w>
+ cos((2k+1)theta)|w_perp> with sin theta = 1/sqrt N (Result 15.10, BBHT's Eq. (2): their
j, k_j, l_j are our k, a_k, b_k); m = floor(pi/(4 theta)) steps fail with probability at most
1/N and m <= (pi/4) sqrt N, and the success is not monotone (Result 15.11); with t marked
drawers the same holds with sin^2 theta = t/N (Result 15.12, BBHT's §3, stated and cited
in the chapter, checked here on samples).

The Turning Plane is not a string diagram and not a PyZX picture: the plane itself (the fold
lemma, the invariant plane, the closed form) is checked on the blackboard.  The sketchbook's
second strand is an eight-drawer search built from dots, whose state the plane predicts,
and the four-drawer master with a second step appended: the overshoot, in pictures.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch15 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)


def fold(alpha):
    """F_alpha, the reflection across the line at angle alpha to the first axis."""
    return np.array([[np.cos(2 * alpha), np.sin(2 * alpha)],
                     [np.sin(2 * alpha), -np.cos(2 * alpha)]])


def turn(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def apply_marker(v, marked):
    """O_w (or I - 2 sum_{w marked} |w><w|): turn the marked drawers over."""
    out = v.copy()
    out[list(marked)] *= -1
    return out


def apply_turnabout(v):
    """D v = 2 vbar - v (inversion about average; D itself for N <= 64 below)."""
    return 2 * v.mean() - v


def grover_D(N):
    D = np.full((N, N), 2 / N, dtype=complex)
    np.fill_diagonal(D, -1 + 2 / N)
    return D


def tilt(N, t=1):
    return np.arcsin(np.sqrt(t / N))


def stop(N, t=1):
    return int(np.floor(np.pi / (4 * tilt(N, t))))


def blackboard():
    rng = np.random.default_rng(1504)

    # --- Result 15.9, the fold lemma, for random angles
    for _ in range(20):
        a, b = rng.uniform(-np.pi, np.pi, size=2)
        Fa = fold(a)
        assert np.allclose(Fa @ [np.cos(a), np.sin(a)], [np.cos(a), np.sin(a)])       # keeps its line
        assert np.allclose(Fa @ [-np.sin(a), np.cos(a)], [np.sin(a), -np.cos(a)])     # reverses the rest
        assert np.allclose(fold(b) @ Fa, turn(2 * (b - a)))
        assert not np.allclose(fold(b) @ Fa, turn(2 * (a - b)))                       # the direction
        assert not np.allclose(fold(b) @ Fa, turn(b - a))                             # TWICE the angle
        assert np.isclose(np.linalg.det(fold(b) @ Fa), 1)                             # not turned over

    # --- Results 15.10 and 15.11, direct simulation against BBHT's Eq. (2)
    for N in (4, 8, 16, 64, 4096):
        th = tilt(N)
        m = stop(N)
        w = int(rng.integers(N))
        e_w = np.zeros(N); e_w[w] = 1
        perp = (np.ones(N) - e_w) / np.sqrt(N - 1)
        s = np.ones(N) / np.sqrt(N)
        assert np.allclose(s, np.sin(th) * e_w + np.cos(th) * perp)
        if N <= 64:
            D = grover_D(N)
            for u in (s, perp, e_w, rng.normal(size=N) + 1j * rng.normal(size=N)):
                assert np.allclose(D @ u, apply_turnabout(u))
        # the plane is kept; restricted, the marker is F_0 and the turn-about F_theta
        basis = np.array([perp, e_w]).T                                   # first axis w_perp
        for op, F in ((lambda v: apply_marker(v, [w]), fold(0)), (apply_turnabout, fold(th))):
            images = np.array([op(basis[:, 0]), op(basis[:, 1])]).T
            coords = basis.T @ images
            assert np.allclose(basis @ coords, images)                    # stays in the plane
            assert np.allclose(coords, F)
        assert np.allclose(fold(th) @ fold(0), turn(2 * th))              # a step turns by 2 theta
        # BBHT's recurrence (their Eq. (1)), cos 2theta = (N-2)/N, sin 2theta = 2 sqrt(N-1)/N
        assert np.isclose(np.cos(2 * th), (N - 2) / N)
        assert np.isclose(np.sin(2 * th), 2 * np.sqrt(N - 1) / N)
        v = s.copy()
        for k in range(m + 2):
            a_k, b_k = np.sin((2 * k + 1) * th), np.cos((2 * k + 1) * th) / np.sqrt(N - 1)
            assert np.isclose(v[w], a_k)
            assert np.allclose(np.delete(v, w), b_k)
            nxt = apply_turnabout(apply_marker(v, [w]))
            assert np.isclose(nxt[w], (N - 2) / N * v[w] + 2 * (N - 1) / N * v[(w + 1) % N])
            assert np.isclose(nxt[(w + 1) % N], (N - 2) / N * v[(w + 1) % N] - 2 / N * v[w])
            v = nxt
        # the stopping rule: failure after m steps is at most sin^2 theta = 1/N
        fail = np.cos((2 * m + 1) * th) ** 2
        assert fail <= 1 / N + 1e-15 and m <= np.pi / 4 * np.sqrt(N)
        assert abs(m - (np.pi / (4 * th) - 0.5)) <= 0.5                   # |m - m~| <= 1/2
    assert [stop(N) for N in (4, 8, 16, 64, 4096, 2 ** 20)] == [1, 2, 3, 6, 50, 804]
    for n in range(2, 31):                                                # 2^20 and beyond: formula
        N = 2 ** n
        assert np.cos((2 * stop(N) + 1) * tilt(N)) ** 2 <= 1 / N + 1e-15
    # the table's successes, to the digits printed
    table = {4: (1, "1"), 64: (6, "0.9966"), 4096: (50, "0.99995"), 2 ** 20: (804, "0.9999998")}
    for N, (m, shown) in table.items():
        succ = np.sin((2 * m + 1) * tilt(N)) ** 2
        digits = len(shown.split(".")[1]) if "." in shown else 0
        assert round(succ, digits) == float(shown)
    # for large N, m is close to (pi/4) sqrt N; four times the drawers, about twice the steps
    for n in range(10, 29, 2):
        N = 2 ** n
        assert abs(stop(N) / (np.pi / 4 * np.sqrt(N)) - 1) < 0.01
        assert abs(stop(4 * N) / stop(N) - 2) < 0.01
    # the square root against asking one by one (worst case N - 1): 6 against 63, and so on
    assert [(stop(N), N - 1) for N in (4, 64, 4096)] == [(1, 3), (6, 63), (50, 4095)]

    # --- the four-drawer sheet: tilt a third of a right angle; the tip 4, -4, 8 thumbs
    th4 = tilt(4)
    assert np.isclose(th4, np.pi / 6)
    hand = 8 * np.array([np.cos(th4), np.sin(th4)])
    assert np.isclose(hand[1], 4)
    after_marker = fold(0) @ hand
    assert np.isclose(after_marker[1], -4)                                # four thumbs under
    after_turn = fold(th4) @ after_marker
    assert np.allclose(after_turn, [0, 8])                                # straight up
    odds4 = [np.sin((2 * k + 1) * th4) ** 2 for k in range(6)]
    assert np.allclose(odds4, [0.25, 1, 0.25, 0.25, 1, 0.25])             # a second step undoes
    # four drawers are the lucky fit: no larger N = 2^n lands exactly on the marked line
    for n in range(3, 21):
        assert np.sin((2 * stop(2 ** n) + 1) * tilt(2 ** n)) ** 2 < 1 - 1e-12

    # --- the sixty-four-drawer sheet (the seed-cabinet): the dialogue's numbers
    th = tilt(64)
    assert 64 == 16 * 4                                                   # "sixteen times as many"
    assert np.isclose(8 * np.sin(th), 1)                                  # the tip one thumb up
    assert np.isclose(np.degrees(th), 7.18, atol=5e-3)
    odds = [np.sin((2 * k + 1) * th) ** 2 for k in range(9)]
    shown = [0.0156, 0.135, 0.344, 0.591, 0.816, 0.964, 0.9966, 0.907, 0.718]
    for p, q in zip(odds, shown):
        digits = len(str(q).split(".")[1])
        assert round(p, digits) == q
    assert np.isclose(1 - odds[6], 3 / 1000, atol=5e-4)                   # "all but about three in a thousand"
    assert np.isclose(odds[7], 0.9, atol=0.01)                            # "about nine in ten"
    tips = 8 * np.sin((2 * np.arange(8) + 1) * th)
    assert np.allclose(np.round(tips, 2), [1, 2.94, 4.69, 6.15, 7.23, 7.85, 7.99, 7.62])
    spoken = [1, 3 - Fraction(1, 16), 4 + Fraction(3, 4), 6 + Fraction(1, 8), 7 + Fraction(1, 4),
              7 + Fraction(7, 8), 8, 7 + Fraction(5, 8)]                  # "a hair of 8": 8
    for tip, said in zip(tips, spoken):
        assert abs(tip - float(said)) <= 1 / 16                           # to the sixteenth, inclusive
    assert np.isclose(tips[1], 3 - 1 / 16) and tips[1] < 3               # "just under three"
    assert 8 - tips[6] < 1 / 16 and tips[6] < 8                           # "all but a hair of eight"
    assert np.degrees(13 * th) > 90                                       # "just past upright"
    assert np.isclose(np.degrees(13 * th), 93.35, atol=5e-3) and np.isclose(np.degrees((2 * 7 + 1) * th), 107.7, atol=0.05)
    assert tips[7] < tips[6]                                              # "It's come down!"
    assert abs(np.degrees((2 * 7 + 1) * th) - 90) > abs(np.degrees(13 * th) - 90)  # farther past the line
    climbs = np.diff(tips[:7])
    assert np.all(np.diff(climbs) < 0)                                    # "the tip climbs less"
    # a million drawers: about eight hundred touches; asking plainly, up to a million less one
    assert stop(2 ** 20) == 804 and abs(stop(10 ** 6) - 800) < 20

    # --- the figure's sixteen drawers: the hands at 14.5, 43.4 and 72.4 degrees
    th16 = tilt(16)
    assert np.isclose(np.sin(th16), 1 / 4)
    assert np.allclose(np.round(np.degrees([th16, 3 * th16, 5 * th16]), 1), [14.5, 43.4, 72.4])

    # --- not monotone: BBHT's remarks, for 2^20 drawers
    N = 2 ** 20
    th = tilt(N)
    succ = lambda k: np.sin((2 * k + 1) * th) ** 2
    assert succ(804) > succ(805) > succ(1000)
    assert succ(int(np.sqrt(2 * N))) < 0.095                              # "less than 9.5%"
    assert succ(int(round(np.pi / 2 * np.sqrt(N)))) < 1e-9                # "negligible"
    half = min(k for k in range(2000) if (2 * k + 1) * th >= np.pi / 4)   # even odds
    assert half == 402 and succ(half) >= 0.5 > succ(half - 1)
    assert abs(half / (np.pi / 8 * np.sqrt(N)) - 1) < 0.01                # about (pi/8) sqrt N

    # --- Result 15.12 (BBHT §3; stated and cited in the chapter), checked on samples
    for N, t in [(8, 2), (16, 3), (64, 5), (64, 16), (256, 7), (4096, 4)]:
        th = tilt(N, t)
        marked = rng.choice(N, size=t, replace=False)
        v = np.ones(N) / np.sqrt(N)
        m = stop(N, t)
        for k in range(m + 2):
            assert np.allclose(v[marked], np.sin((2 * k + 1) * th) / np.sqrt(t))
            assert np.allclose(np.delete(v, marked), np.cos((2 * k + 1) * th) / np.sqrt(N - t))
            v = apply_turnabout(apply_marker(v, marked))
        assert np.cos((2 * m + 1) * th) ** 2 <= t / N + 1e-15
        assert m <= np.pi / 4 * np.sqrt(N / t)
    for N in (8, 16, 64, 4096):                                           # t = N/4: one step, certain
        th = tilt(N, N // 4)
        assert np.isclose(th, np.pi / 6) and np.isclose(np.sin(3 * th), 1)
    # BBHT's warning: 804 steps with four solutions among 2^20 succeed < one in a million
    assert np.sin((2 * 804 + 1) * tilt(2 ** 20, 4)) ** 2 < 1e-6

    pictures()


# --- the prop pictures ------------------------------------------------------------------

STEPS = ("no", "a", "two", "three", "four", "five", "six", "seven")


def figure_pairs_clear(paras):
    """Every touch line of the figure is answered by its turn line with no picture block
    between them (the toccata's figure, identity card section 3c)."""
    def spoken(p):
        m = re.match(r"^\*\*(\w+):\*\*(.*)$", p, re.S)
        return (m.group(1), re.sub(r"\*\[.*?\]\*", "", m.group(2)).strip()) if m else None
    touches = [i for i, p in enumerate(paras) if spoken(p) == ("Achilles", "The marker turns it over.")]
    for i in touches:
        j = next(k for k in range(i + 1, len(paras)) if spoken(paras[k]))
        assert spoken(paras[j]) == ("Tortoise", "And we turn it about."), paras[j][:60]
        assert not any(DIALOGUE._is_block(p) for p in paras[i:j]), i
    return len(touches)


def pictures():
    """The Fourth Challenge's prop pictures (identity card, section 4), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch15.py.  The
    ties hold the transcription against this file's blackboard (a structural check of the
    pictures, not a second proof): the hand on its pin is the state in the turning plane,
    its angle up from the unmarked line (|w_perp>) towards the marked line (|w>); the
    marker's fold is the reflection across the unmarked line, the turn-about's across the
    even line at theta, two folds a turn by 2 theta; the hand is eight thumbs long, so its
    tip stands 8 sin((2k+1) theta) thumbs up after k steps."""
    calls = DIALOGUE.check_layout("Fourth", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("handsheet", True, "", ("four drawers",), "the sheet ruled, the hand laid"),
        ("handsheet", False, "a step", ("four drawers",), "after the folds, upright"),
        ("handsheet", False, "six steps", ("sixty-four drawers",), "climbed to the top"),
        ("handsheet", False, "seven steps", ("sixty-four drawers",), "turned too far")]
    PC.check_captions(DIALOGUE, "Fourth", calls)
    PC.check_section(DIALOGUE, "Fourth", calls)
    sheet_h = round(PC.HFTOP - PC.HFBOT, 2)
    assert sheet_h == 2.72                                               # read from the Lua
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Fourth")] == [[sheet_h]] * 4

    # the anchors (card section 4)
    a1, a2, a3, a4 = DIALOGUE.anchors("Fourth")
    f1, f2, f3, f4 = DIALOGUE.following("Fourth")
    assert "and this tilted one is the even line." in a1
    assert f1.startswith("**Achilles:** Half the hand, and half squared is a quarter.")
    assert "now stands eight thumbs up, the hand's whole length.]*" in a2
    assert f2.startswith("**Achilles:** *[laying her rule from the unmarked line to the tip]* Eight thumbs of eight.")
    assert a3 == "**Achilles:** *[measuring]* All but a hair of eight. Just past upright."
    assert f3.startswith("**Tortoise:** Stop there. Six touches.")
    assert "on the far side of the marked line, to its left, and farther from it than before;" in a4          # D19
    assert f4.startswith("**Achilles:** *[measuring]* Seven and five eighths.")
    assert DIALOGUE.following("Fourth", DIALOGUE.raw) == [f1, f2, f3, f4]
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        sec = DIALOGUE.section("Fourth", path)
        for phrase in ("She slides the peg-card along to Achilles's left and lays the sheet on the table before "
                       "him, where the card lay, between him and the case, square to his edge of the table",       # D16
                       "nearer its left edge than its right, and a line across it, half way up",
                       "She lays it out to the right of the pin as Achilles sees it",
                       "by a third of a right angle, rising to the right.]*",
                       "She lifts the hand off the first sheet, pin and all, and sets that sheet aside on the table.",  # D17
                       "at the same places as on the first",
                       "She lays the hand out to the right of the pin, so that its tip stands one thumb above",
                       "just on the far side of the marked line from the even line, to its left as Achilles sees it.",  # D18
                       "on the far side of the marked line, to its left, and farther from it than before;"):       # D19
            assert sec.count(phrase) == 1, phrase
        # no picture between P4.2's block and P4.3's (the four-drawer surprise, the fresh
        # sheet, the Escher moment and the first six steps), nor inside a figure
        paras = DIALOGUE.paragraphs("Fourth", path)
        blocks = [i for i, p in enumerate(paras) if DIALOGUE._is_block(p)]
        escher = [i for i, p in enumerate(paras) if "Cubic Space Division" in p]
        assert len(escher) == 1 and blocks[1] < escher[0] < blocks[2]
        assert figure_pairs_clear(paras) == 6

    p41, p42, p43, p44 = (PC.picture(c) for c in calls)

    def plane(N, k):
        """(D O_w)^k |s> simulated on N drawers: (its |w_perp> part, its |w> part)."""
        w = 1
        v = np.ones(N) / np.sqrt(N)
        for _ in range(k):
            v = apply_turnabout(apply_marker(v, [w]))
        perp = np.ones(N) / np.sqrt(N - 1)
        perp[w] = 0
        return v @ perp, v[w]

    for N, sheet in ((4, "four drawers"), (64, "sixty-four drawers")):
        th = tilt(N)
        assert np.isclose(PC.TILT[N], th)
        for k in range(8):
            H = PC.sheet_state("%s step%s" % (STEPS[k], "" if k == 1 else "s") if k else "", sheet)
            b, a = plane(N, k)
            ang = (2 * k + 1) * th
            assert np.isclose(np.cos(H.a), np.cos(ang)) and np.isclose(np.sin(H.a), np.sin(ang))
            assert np.allclose((np.cos(H.a), np.sin(H.a)), (b, a))          # the hand IS the state
            assert PC.sheet_dump(H).split()[3] == "tip=%.2f" % (8 * a)
            # the holes: every place the hand was folded from or to, in order (the sheet
            # pricks a place once; on the four-drawer sheet the hand comes round past half
            # a turn, so a place may be met again under another angle)
            seq, u = [np.array([np.cos(th), np.sin(th)])], np.array([np.cos(th), np.sin(th)])
            for _ in range(k):
                u = fold(0) @ u
                seq.append(u)
                u = fold(th) @ u
                seq.append(u)

            def places(vs):
                out = []
                for q in vs:
                    if not any(np.allclose(q, r) for r in out):
                        out.append(q)
                return out
            if k:
                assert np.allclose(places([(np.cos(h), np.sin(h)) for h in H.holes]), places(seq))
            else:
                assert H.holes == []
        assert not PC.TEX_ERRORS
    # the marker's fold is the reflection across the unmarked line, a -> -a; the
    # turn-about's across the even line, a -> 2 theta - a; the two, a turn by 2 theta
    for N in (4, 64):
        th = tilt(N)
        Hm = PC.sheet_state("the marker's fold", "four drawers" if N == 4 else "sixty-four drawers")
        assert np.isclose(Hm.a, -th) and np.allclose(fold(0) @ [np.cos(th), np.sin(th)], [np.cos(Hm.a), np.sin(Hm.a)])
        Ht = PC.sheet_state("the marker's fold, the turn-about's fold", "four drawers" if N == 4 else "sixty-four drawers")
        assert np.allclose(turn(2 * th) @ [np.cos(th), np.sin(th)], [np.cos(Ht.a), np.sin(Ht.a)])
        assert not np.isclose(Ht.a, 2 * th - th)                          # not the turn-about alone
    assert PC.sheet_dump(PC.sheet_state("the marker's fold", "four drawers")).split()[2] == "angle=-30.00"   # "Four thumbs under."

    # P4.1 and P4.2: theta and 3 theta for sin theta = 1/2 -- tips four and eight
    assert p41.dump == "drawers=4 marked=true angle=30.00 tip=4.00 holes=none"
    assert p42.dump == "drawers=4 marked=true angle=90.00 tip=8.00 holes=30/-30/90"
    assert np.isclose(np.degrees(tilt(4)), 30) and np.isclose(8 * np.sin(3 * tilt(4)), 8)
    # P4.3 and P4.4: 13 theta and 15 theta for sin theta = 1/8 -- all but eight, then down
    assert p43.dump == "drawers=64 marked=true angle=93.35 tip=7.99 holes=7/-7/22/-22/36/-36/50/-50/65/-65/79/-79/93"
    assert p44.dump == ("drawers=64 marked=true angle=107.71 tip=7.62 "
                        "holes=7/-7/22/-22/36/-36/50/-50/65/-65/79/-79/93/-93/108")
    th = tilt(64)
    assert np.isclose(p43.state.a, 13 * th) and np.isclose(p44.state.a, 15 * th)
    assert 8 * np.sin(15 * th) < 8 * np.sin(13 * th) and stop(64) == 6      # the seventh turns away
    assert p44.state.holes[:len(p43.state.holes)] == p43.state.holes      # holes only added

    # what is drawn: the pin, the marked and unmarked lines the same on every sheet (the
    # fresh sheet's at the same places as the first's), the even line the same on one
    # sheet; a fold changes only the hand and adds holes
    v1, v2, v3, v4 = (p.shown for p in (p41, p42, p43, p44))
    for v in (v2, v3, v4):
        assert v.pin == v1.pin and v.lines[:2] == v1.lines[:2]
    assert v2.lines == v1.lines and v4.lines == v3.lines and v3.lines[2] != v1.lines[2]
    assert v3.marks == v4.marks and v1.marks == v2.marks
    # the even line rises to the right; the seventh step's hand leans over the marked line,
    # to its left (the far side from the even line)
    assert v1.lines[2][3] > v1.lines[2][1] and v1.lines[2][4] > v1.lines[2][2]
    assert v4.tip[0] < PC.PINX < v3.lines[2][3] and v3.tip[0] < PC.PINX
    # every hole inside the sheet, and the hand's tip clear of the sheet's edges, at every
    # pictured angle -- and at the marker's fold after six steps ("The marker turns it over.", not pictured)
    marked6 = PC.handsheet("six steps, the marker's fold", "sixty-four drawers")
    assert marked6.dump.split()[2] == "angle=-93.35"
    for p in (p41, p42, p43, p44, marked6):
        parts, boxes, _ = PC.drawn_parts(p)
        for name, part in parts.items():
            if name.startswith(("hole", "dot", "hand")) and name != "hand shadow":
                assert PC.edge_gap(part, boxes["sheet"]) > 0.03 and PC.inside(part[0][0], boxes["sheet"][0]), name
    assert set(marked6.state.holes) >= set(p43.state.holes)

    # the keyed words, as set (node boxes measured in the book's preamble): every word
    # clears the drawing, the others and the frame, `marked line` the frame's top by more
    # than 0.02.  `unmarked line` and `even line` stand beyond the sheet's right edge, where
    # its drop shadow (a 0.12 shade, 0.03 wide) shows: their node boxes' inner sep, which
    # nothing draws, reaches that shade, so there they are held on their letters
    assert p41.labels == ("marked line", "unmarked line", "even line", "pin", "hand")
    box, ink = PC.word_gaps(p41), PC.word_gaps(p41, ink=True)
    for w in p41.labels:
        held = ("sheet shadow right",) if w in ("unmarked line", "even line") else ()
        assert min(g for k, g in box[w].items() if k not in held) > 0.01, (w, min(box[w], key=box[w].get))
        assert min(ink[w].values()) > 0.02, w
    assert box["unmarked line"]["sheet shadow right"] < 0 and box["even line"]["sheet shadow right"] < 0
    assert box["marked line"]["frame"] > 0.02
    # the Lua finds room for `pin` and `hand` with the words' boxes as set (WORDSIZE, to the
    # nearest thousandth of the measured box), and places them clear of the hand
    for w in ("pin", "hand"):
        assert all(abs(PC.WORDSIZE[w][d] - PC.WORD_PT[w][d] * PC.PT) < 5e-4 for d in (0, 1)), w
    words = {w: (x, y) for w, x, y, _ in v1.labels}
    assert np.allclose(words["pin"], (-0.950, 1.415), atol=1e-3)
    assert np.allclose(words["hand"], (-0.442, 1.699), atol=1e-3)


# --- the sketchbook ---------------------------------------------------------------------

class Cords:
    """n cords read left to right, row 0 the top cord, each starting at a point (kind,
    phase).  A collar (the yellow box) is a Hadamard EDGE on the next stretch of its cord;
    two collars with nothing between them cancel (H H = I)."""

    def __init__(self, n, points):
        self.g, self.n, self.col = zx.Graph(), n, 1
        self.end, self.pend = [], [ET.SIMPLE] * n
        for r in range(n):
            kind, phase = points[r]
            self.end.append(self.g.add_vertex(kind, r, 0, phase=Fraction(phase)))

    def dot(self, r, kind, phase=0):
        v = self.g.add_vertex(kind, r, self.col, phase=Fraction(phase))
        self.g.add_edge((self.end[r], v), self.pend[r])
        self.pend[r], self.end[r] = ET.SIMPLE, v
        self.col += 1
        return v

    def collar(self, *rows):
        for r in rows:
            self.pend[r] = ET.HADAMARD if self.pend[r] == ET.SIMPLE else ET.SIMPLE

    def gadget(self, rows, ring, pendant):
        """Rings (green, `ring`) on `rows`, tied to a phase-free red hub, a pendant green
        point `pendant`.  Bare, a hub of k ties with its pendant gives 2^{-(k-1)/2}."""
        rings = [self.dot(r, VT.Z, ring) for r in rows]
        hub = self.g.add_vertex(VT.X, sum(rows) / len(rows) + 0.5, self.col)
        bob = self.g.add_vertex(VT.Z, sum(rows) / len(rows) + 0.5, self.col + 1, phase=Fraction(pendant))
        for v in rings:
            self.g.add_edge((v, hub))
        self.g.add_edge((hub, bob))
        self.col += 2

    def ccz(self):
        """CCZ as phase gadgets, bare 2^{-5/2} CCZ (Result 15.17)."""
        for r in range(3):
            self.dot(r, VT.Z, Fraction(1, 4))
        for pair in ([0, 1], [1, 2], [0, 2]):
            self.gadget(pair, 0, Fraction(-1, 4))
        self.gadget([0, 1, 2], 0, Fraction(1, 4))

    def done(self):
        outs = []
        for r in range(self.n):
            o = self.g.add_vertex(VT.BOUNDARY, r, self.col + 1)
            self.g.add_edge((self.end[r], o), self.pend[r])
            outs.append(o)
        self.g.set_inputs(())
        self.g.set_outputs(tuple(outs))
        return self.g


def scaled(g, power=0, sign=0):
    """g times sqrt2^power times (-1)^sign (add_power counts powers of sqrt2)."""
    h = g.copy()
    h.scalar.add_power(power)
    if sign:
        h.scalar.add_phase(Fraction(1))
    return h


def eight_drawer_search(steps, w=0b011):
    """Three dark points, collars, then `steps` times: the marker for w in X-form (red pi's
    on the cords where w has a 0, either side of the CCZ gadgets) and the X-form
    turn-about (collars, red pi's, CCZ gadgets, red pi's, collars)."""
    zeros = [r for r in range(3) if (w >> (2 - r)) & 1 == 0]
    c = Cords(3, [(VT.X, 0)] * 3)
    c.collar(0, 1, 2)
    for _ in range(steps):
        for r in zeros:
            c.dot(r, VT.X, 1)
        c.ccz()
        for r in zeros:
            c.dot(r, VT.X, 1)
        c.collar(0, 1, 2)
        for r in range(3):
            c.dot(r, VT.X, 1)
        c.ccz()
        for r in range(3):
            c.dot(r, VT.X, 1)
        c.collar(0, 1, 2)
    return c.done()


def four_drawer_search(steps, w=0b01):
    """The Second Drawer's build, with `steps` steps: the marker (controlled half-turn,
    half-rings by w) and the turn-about (collars, half-rings, controlled half-turn,
    collars).  One step is the master, exact with no compensation."""
    w1, w2 = (w >> 1) & 1, w & 1
    c = Cords(2, [(VT.X, 0)] * 2)
    c.collar(0, 1)
    for _ in range(steps):
        c.gadget([0, 1], Fraction(1, 2), Fraction(-1, 2))
        if w2 == 0:
            c.dot(0, VT.Z, 1)
        if w1 == 0:
            c.dot(1, VT.Z, 1)
        c.collar(0, 1)
        c.dot(0, VT.Z, 1)
        c.dot(1, VT.Z, 1)
        c.gadget([0, 1], Fraction(1, 2), Fraction(-1, 2))
        c.collar(0, 1)
    return c.done()


def sketchbook():
    # (a) The eight-drawer search in dots, marker 011.  Compensation: each step has two
    #     CCZ gadget blocks, 2^{-5/2} each, so add_power(10) per step; the three dark points
    #     carry 2^{3/2}, so add_power(-3); the X-form turn-about carries -1 per step.  The
    #     string diagram COMPUTES the state; the Turning Plane PREDICTS it.  Eighth-turns are
    #     not Clifford: full_reduce a copy and compare by matrix_of.
    th = tilt(8)
    for k, marked_amp in ((1, 0.883883), (2, 0.972272)):
        g = scaled(eight_drawer_search(k), 10 * k - 3, sign=k % 2)
        v = matrix_of(g).ravel()
        assert np.isclose(v[0b011], np.sin((2 * k + 1) * th))
        assert np.isclose(v[0b011], marked_amp, atol=1e-6)
        others = np.delete(v, 0b011)
        assert np.allclose(others, np.cos((2 * k + 1) * th) / np.sqrt(7))
        assert np.isclose(np.linalg.norm(v), 1)
        assert not np.isclose(v[0b110], np.sin((2 * k + 1) * th))        # upside down: unmarked
        h = g.copy()
        simp.full_reduce(h)
        assert np.allclose(matrix_of(h).ravel(), v)
        # without the X-form's sign after one step, the state is minus the prediction
        if k == 1:
            assert not same_picture(g, scaled(eight_drawer_search(k), 10 * k - 3))
    # the plane's prediction for eight drawers: m = 2 steps, success sin^2(5 theta)
    assert stop(8) == 2 and np.isclose(0.972272 ** 2, np.sin(5 * th) ** 2, atol=1e-5)

    # (b) The four-drawer check in dots: the master, and the master with a second step
    #     appended.  One step is exactly |01> (no compensation); the second step's two bare
    #     controlled half-turns need add_power(2), and the state is |w> - |s>: amplitude 1/2
    #     at 01 (sin 5 theta with theta = pi/6), -1/2 elsewhere, odds 1/4 -- the overshoot,
    #     in pictures.  Clifford: DERIVED by full_reduce on a copy, the value kept.
    one = four_drawer_search(1)
    assert np.allclose(matrix_of(one).ravel(), [0, 1, 0, 0])
    assert one.num_edges() == 15
    two = scaled(four_drawer_search(2), 2)
    v = matrix_of(two).ravel()
    assert np.isclose(v[0b01], 0.5) and np.isclose(v[0b01], np.sin(5 * np.pi / 6))
    assert np.allclose(np.delete(v, 0b01), -0.5)
    assert np.allclose(np.abs(v) ** 2, 0.25)                              # every drawer a quarter
    for g in (one, two):
        h = g.copy()
        simp.full_reduce(h)
        assert same_picture(h, g)
        assert h.num_vertices() < g.num_vertices()
    three = scaled(four_drawer_search(3), 4)                             # a third step: back to 1/4
    assert np.allclose(np.abs(matrix_of(three).ravel()) ** 2, 0.25)

    # (c) The picture measures the tilt (Coecke's Sketchbook, "two mirrors of one shape"):
    #     a red w-effect closed against a green point on every cord is sqrt2 per cord, so
    #     2^{n/2} in all; divided by the 2^{n/2} 2^{n/2} the effects and points carry, it is
    #     <w|s> = 2^{-n/2} = sin theta.  DERIVED: full_reduce empties the closed picture
    #     (every phase 0 or pi) and leaves that number as its scalar.
    for n in (2, 3, 4):
        for w in (0, 2 ** n - 1, 1):
            g = zx.Graph()
            for r in range(n):
                eff = g.add_vertex(VT.X, r, 1, phase=Fraction((w >> (n - 1 - r)) & 1))
                g.add_edge((g.add_vertex(VT.Z, r, 0), eff))
            value = matrix_of(g).item()
            assert np.isclose(value, 2 ** (n / 2))
            assert np.isclose(value / 2 ** n, np.sin(tilt(2 ** n)))
            h = g.copy()
            simp.full_reduce(h)
            assert h.num_vertices() == 0 and np.isclose(h.scalar.to_number(), 2 ** (n / 2))


if __name__ == "__main__":
    verdict("TurningPlane", blackboard, sketchbook)
