"""Challenge I: A Name, a Colour, a Turn (fusion)
================================================
"It has a family now. Has it a name?"  -- Achilles

Claim: two same-colour spiders that share at least one leg are a single spider,
and their phases add -- exactly, with no scalar:  Z(1,1,b) Z(m,1,a) = Z(m,1,a+b).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
import string
from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import ket, same_picture, same_picture_up_to_scalar, verdict
from dfp_props import Dialogue, words_only
import props_ch08 as P


# --- spiders as matrices: the chapter's own maths, pinned in every file ---
KET_PLUS = (ket(0) + ket(1)) / np.sqrt(2)
KET_MINUS = (ket(0) - ket(1)) / np.sqrt(2)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def zspider(m, k, alpha):
    """Z spider (green): |0..0><0..0| + e^{i alpha}|1..1><1..1|, m legs in / k out."""
    M = np.zeros((2 ** k, 2 ** m), dtype=complex)
    M[0, 0] = 1.0
    M[-1, -1] = np.exp(1j * alpha)
    return M


def _pow(v, n):
    out = np.array([1.0 + 0j])
    for _ in range(n):
        out = np.kron(out, v)
    return out


def xspider(m, k, beta):
    """X spider (red), built INTRINSICALLY from normalized |+>,|-> --
    NOT as H^(x)k Z H^(x)m, so colour change stays a theorem, not a definition."""
    pk, pm = _pow(KET_PLUS, k), _pow(KET_PLUS, m)
    mk, mm = _pow(KET_MINUS, k), _pow(KET_MINUS, m)
    return np.outer(pk, pm.conj()) + np.exp(1j * beta) * np.outer(mk, mm.conj())


def pin_spiders():
    """Spiders-as-matrices, asserted once per file."""
    assert np.allclose(zspider(1, 1, np.pi), np.diag([1, -1]))          # Z(pi) = phase-flip
    assert np.allclose(xspider(1, 1, np.pi), np.array([[0, 1], [1, 0]]))  # X(pi) = NOT
    assert np.allclose(H @ H, np.eye(2))                               # H^2 = I
    assert np.allclose(KET_PLUS + KET_MINUS, np.sqrt(2) * ket(0))       # red 1-leg state = sqrt2|0>


def blackboard():
    pin_spiders()
    rng = np.random.default_rng(8)
    a, b = rng.normal(), rng.normal()
    assert abs(a) > 0.1 and abs(b) > 0.1                                # generic, not the trivial 0

    # fusion along one shared leg: phases add, exactly
    assert np.allclose(zspider(1, 1, b) @ zspider(1, 1, a), zspider(1, 1, a + b))
    assert not np.allclose(zspider(1, 1, b) @ zspider(1, 1, a), zspider(1, 1, a))  # they really ADD

    # a multi-leg chain still fuses to one spider, phases adding
    assert np.allclose(zspider(1, 2, a) @ zspider(1, 1, b), zspider(1, 2, a + b))
    # general (m,k): a 2->1 green into a 1->3 green fuses to a 2->3 green
    assert np.allclose(zspider(1, 3, b) @ zspider(2, 1, a), zspider(2, 3, a + b))
    # more than one shared leg: two wires between the two greens fuse just the same
    assert np.allclose(zspider(2, 1, a) @ zspider(1, 2, b), zspider(1, 1, a + b))

    # conventions: a one-in/one-out phase-0 green spider is the bare wire (identity);
    # and ket order is top wire first -- the green copier duplicates a basis bit
    assert np.allclose(zspider(1, 1, 0), np.eye(2))
    assert np.allclose(zspider(1, 2, 0) @ ket(0), ket(0, 0))
    assert np.allclose(zspider(1, 2, 0) @ ket(1), ket(1, 1))

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the Crab's cords and curtain-rings
#   \ringcords[opts]{tangle}{caption}           (pdf/props/ch08.tex)
# and (round 4, "Show the process") a step strip of them,
#   \ringsteps[opts]{tangle}{steps}{captions}.
# The Lua COMPUTES the slid view (the left ring slid onto the right, one ring
# left at the right ring's place, its turn the two turns ADDED) and its marks
# (the near ring's ghost, the slide arc, the far bead's ghost and swing, the pin
# on the cord's far end), and each strip frame from the one before (a cord added
# to the first free hook).  tools/props_ch08.py transcribes the Lua (proved
# against texlua; P.check_lua holds the Lua and the .tex to the text it was
# proved on); the calls are read out of this challenge's section of dialog.md
# and dialog.raw.md, never copied.
#
# The state model (identity card, sections 3 and 3e): a green ring is a Z
# spider, a red ring an X spider; a ring's turn t (twenty-fourths, clockwise
# from the top as the company sees it) is the phase 2 pi t / 24; cords in on
# the left are inputs, out on the right outputs, the upper line first.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
QUOTE = "It has a family now. Has it a name?"


def phase(t):
    """A turn in twenty-fourths as a phase in radians."""
    return 2 * np.pi * t / P.WHOLE


def tangle_map(T):
    """The map a tangle of rings and cords denotes, contracted in NumPy: each
    ring a spider of its colour with one leg per cord segment at it; ends on
    the left are inputs, on the right outputs, the upper line first (top wire
    first).  Leg order and in/out do not matter: these Z and X spiders are
    symmetric in all their legs (|0>,|1>,|+>,|-> are real)."""
    letters = iter(string.ascii_letters)
    legs = {n: [] for n in range(1, len(T.rings) + 1)}
    ins, outs, ops, subs = [], [], [], []
    for a, b in P.segments(T):
        if isinstance(a, P.End) and isinstance(b, P.End):     # a bare cord, hook to hook
            la, lb = next(letters), next(letters)
            ops.append(np.eye(2)); subs.append(la + lb)
            pairs = ((a, la), (b, lb))
        else:
            l = next(letters)
            pairs = ((a, l), (b, l))
        for e, l in pairs:
            if isinstance(e, P.End):
                (ins if e.side == "L" else outs).append((-e.y, l))
            else:
                legs[e].append(l)
    for n, r in enumerate(T.rings, 1):
        k = len(legs[n])
        spider = zspider if r.c == "green" else xspider
        ops.append(spider(0, k, phase(r.t)).reshape([2] * k)); subs.append("".join(legs[n]))
    out = "".join(l for _, l in sorted(outs)); inn = "".join(l for _, l in sorted(ins))
    M = np.einsum(",".join(subs) + "->" + out + inn, *ops)
    return M.reshape(2 ** len(outs), 2 ** len(ins))


def check_transcription():
    """The Lua's words and its slid view against the chapter's rules (a
    structural check, not a second proof)."""
    P.check_lua()
    P.check_sites()
    # the turn words, in twenty-fourths, as the dialogue says them
    assert P.parse_turn("a third") == 8 and P.parse_turn("a quarter") == 6
    assert P.parse_turn("a third of the way round") == 8
    assert P.parse_turn("a third and a quarter") == 14 == P.parse_turn("seven twelfths")
    assert P.parse_turn("at the top") == P.parse_turn("no turn") == 0
    assert P.parse_turn("the halfway mark") == 12 and P.parse_turn("four quarters") == 0   # whole = none
    # every twenty-fourth has words (one term, or two joined by "and"), and "and" adds
    words = [f"{n} {u}" for n in P.NUMS for u in P.UNITS]
    assert {P.parse_turn(w) for w in words} | \
        {P.parse_turn(f"{a} and {b}") for a in words[::7] for b in words} == set(range(P.WHOLE))
    for a, b in itertools.product(("a third", "three quarters", "five twelfths"), repeat=2):
        assert P.parse_turn(f"{a} and {b}") == (P.parse_turn(a) + P.parse_turn(b)) % P.WHOLE
    # the macro's refusals on this challenge's tangles
    assert P.faults(P.parse_turn, "a fifth") == ["turn"]
    assert P.faults(P.parse_tangle, "green ring, seven cords") == ["count"]
    assert P.faults(P.parse_tangle, "green ring, sideways") == ["ring-item"]
    assert P.faults(P.parse_tangle, "green ring, none in, none out") == ["no-cords"]
    assert P.faults(P.parse_tangle, "two green rings") == ["two-turns"]
    assert P.faults(P.parse_tangle, "two green rings, turns top and half, loose") == ["two-extra"]
    assert P.faults(P.ringcords, "slid", "green ring, turn a third") == ["slid"]
    assert P.faults(P.ringcords, "slid", "two green rings, turns top and half") == []
    # <k> cords is floor(k/2) in and the rest out
    for k in range(1, 7):
        T = P.parse_tangle(f"green ring, {['one','two','three','four','five','six'][k - 1]} cords")
        ends = [e for s in P.segments(T) for e in s if isinstance(e, P.End)]
        assert sum(e.side == "L" for e in ends) == k // 2 and sum(e.side == "R" for e in ends) == k - k // 2

    # slid <-> fusion, every pair of turns, both colours: the slid ring is the fused spider,
    # EXACTLY (fusion holds on the nose), and the cord it sits on is unmoved
    for c, spider in (("green", zspider), ("red", xspider)):
        for t1, t2 in itertools.product(range(P.WHOLE), repeat=2):
            T = P.two(c, t1, t2)
            S = P.view_slid(T)
            (r,) = S.rings
            assert r.c == c and r.t == (t1 + t2) % P.WHOLE
            assert (r.x, r.y) == (T.rings[1].x, T.rings[1].y)          # at the far ring's place
            assert P.segments(S) == ((P.End("L", 0.0), 1), (1, P.End("R", 0.0)))
            assert np.allclose(spider(1, 1, phase(t2)) @ spider(1, 1, phase(t1)),
                               spider(1, 1, phase(r.t)))
            assert np.allclose(tangle_map(T), tangle_map(S))
    # a slid ring that only kept the far turn would be wrong (the turns really ADD)
    assert not np.allclose(tangle_map(P.two("green", 8, 6)), tangle_map(P.single("green", 6, 1, 1)))
    # "A green ring, one in, one out, no turn, is nothing but wire."
    assert np.allclose(tangle_map(P.parse_tangle("green ring")), np.eye(2))
    # the map of a tied ring is its spider: two in, three out (the grown ring, P1.1)
    assert np.allclose(tangle_map(P.parse_tangle("green ring, five cords, turn a third")),
                       zspider(2, 3, phase(8)))


def check_add_step():
    """The add step against the chapter's ring (a structural check): one more cord, to the
    first free hook of top, bottom, middle on its side, the cords already there kept on their
    hooks; a green ring with one more leg is the spider with one more leg, the same turn."""
    assert P.ADDORDER == ("top", "bottom", "middle") and P.PLACE_Y == {"top": 0.42, "middle": 0, "bottom": -0.42}
    names = ("top", "middle", "bottom")
    subsets = [frozenset(s) for n in range(4) for s in itertools.combinations(names, n)]
    for pin, pout in itertools.product(subsets, repeat=2):
        if not pin and not pout:
            continue
        spec = ("green ring, turn a third, " + ("in at " + P.places_words(pin) if pin else "none in") + ", " +
                ("out at " + P.places_words(pout) if pout else "none out"))
        T = P.parse_tangle(spec)
        assert (T.pin, T.pout) == (pin, pout)
        for side, have in (("in", pin), ("out", pout)):
            free = [w for w in P.ADDORDER if w not in have]
            pic = P.ringcords("beam; add=" + side, spec)
            if not free:
                assert P.faults(P.ringcords, "beam; add=" + side, spec) == ["add-full"]
                continue
            assert P.faults(P.ringcords, "beam; add=" + side, spec) == []
            assert pic.moved.hook == free[0] and pic.marks == () and pic.mark_tikz == ()   # from the hand
            new = P.End("L" if side == "in" else "R", P.PLACE_Y[free[0]])
            ends = lambda t: sorted((e for s in P.segments(t) for e in s if isinstance(e, P.End)), key=str)
            assert ends(pic.drawn) == sorted(ends(T) + [new], key=str)
            assert pic.drawn.rings[0].t == T.rings[0].t == 8 and pic.drawn.rings[0].c == "green"
            nin, nout = len(pin) + (side == "in"), len(pout) + (side == "out")
            assert np.allclose(tangle_map(pic.drawn), zspider(nin, nout, phase(8)))
    # the order is the Lua's: a cord in with the top taken goes to the bottom, never the middle
    assert P.ringcords("beam; add=in", "green ring, in at top, out at top and bottom").moved.hook == "bottom"
    assert P.faults(P.ringcords, "beam; add=out", "green ring, five cords") == ["add-full"]    # three out: full
    assert P.ringcords("beam; add=in", "green ring, five cords").moved.hook == "middle"     # two in: the gap
    assert P.faults(P.ringcords, "beam; add=in", "green ring, four in") == ["add-hooks"]      # no names
    assert P.faults(P.ringcords, "slid; add=in", "green ring") == ["step-view"]


def pictures():
    assert f'"{QUOTE}"' in __doc__                              # quoted byte for byte
    check_transcription()
    check_add_step()
    found = P.check_section(DIALOGUE, "First", sizes=[1, 2], frames=[3, 2])
    (p11, [(f01, cap01), (f02, cap02), (f03, cap03)]), = found[0]
    (p12, [(b12, cap12)]), (p13, [(b13, cap13)]) = found[1]
    calls = (p11, p12, p13)
    assert [c.macro for c in calls] == ["ringsteps", "ringcords", "ringcords"]
    assert (cap01, cap02, cap03, cap12, cap13) == ("as he found it", "another in", "another out",
                                                   "two rings, two turns", "slid into one")
    for c in calls:
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS) and P.caption_ok(c), c.caption
    # only what the First draws: a ring on the beam (a strip), two rings loose, the slide --
    # the views as the Lua parses the brackets
    assert [c.opt for c in calls] == ["view=beam", "", "slid; marks=on"]          # as written
    assert [P.view_of(c) for c in calls] == ["beam", "", "slid"]
    assert P.options(p11) == {"view": "beam", "bare": ""} and P.options(p13) == {"marks": "on", "bare": "slid"}
    assert not any(c.args[0].lower() in P.FRAME for c in calls)
    assert [c.keyed for c in calls] == [True, False, False]          # a demonstration ring, keyed once
    sec = DIALOGUE.section("First")
    a1, a2 = DIALOGUE.anchors("First")
    f1, f2 = DIALOGUE.following("First")
    assert QUOTE in sec and f1 == "**Achilles:** " + QUOTE             # the docstring's line, right after
    # the directions edited this round (N1, N2 with the Dramatist's wording, N3), in both files
    P.check_edits(DIALOGUE, "First", [
        ("two running out to hooks at its right]*",
         "two running out to hooks at its right, the left hook high and the right ones high and low]*"),
        ("then a fifth out of it to the right.]*",                    # HEAD's close: N2 adds
         "Each hangs from a hook of its own along the beam: the new one in, from a hook low on the "
         "left; the new one out, from a hook on the right between the other two."),
        ("lifts the first ring down off the beam, cords and all,",
         "lifts the first ring down off the beam, peg, cords and all,")])

    # Strip 1.1: the grown ring on its peg -- one in and two out as found, then one more of
    # each, in the order the Crab runs them ("a fourth cord into the ring from the left, then
    # a fifth out of it to the right"); each frame starts from the state the one before left
    assert ("one running in from a hook at the beam's left end, two running out to hooks at its right"
            in sec)
    assert "He runs a fourth cord into the ring from the left, then a fifth out of it to the right." in a1
    assert a1.endswith("The dot has grown up.")
    assert p11.args == ("green ring, in at top, out at top and bottom", "add=in; add=out")
    assert a1.index("into the ring from the left") < a1.index("out of it to the right")
    frames = P.frame_calls(p11)
    assert [(c.opt, c.args[0]) for c in frames] == [
        ("beam", "green ring, in at top, out at top and bottom"),
        ("beam; add=in; keyroom=on", "green ring, in at top, out at top and bottom"),
        ("beam; add=out; keyroom=on", "green ring, in at top and bottom, out at top and bottom")]
    assert [c.keyed for c in frames] == [True, False, False]
    assert f02.made.cords == f01.drawn.cords and f03.made.cords == f02.drawn.cords   # in order
    for fr in (f01, f02, f03):
        assert fr.setting == "beam" and fr.drawn.kind == "single" and fr.labels == ("green", "bead")
        (r,) = fr.drawn.rings
        assert (r.c, r.t, r.x, r.y) == ("green", 0, 0, 0)
        assert fr.marks == () and fr.mark_tikz == () and not fr.pale           # no mark: from the hand
    assert f01.keyed and not f01.phantom and f02.phantom and f03.phantom   # the frames stand level
    assert f01.step is None and f02.step == ("add", "in") and f03.step == ("add", "out")
    ends = lambda t: [e for s in P.segments(t) for e in s if isinstance(e, P.End)]
    L, R = (lambda y: P.End("L", y)), (lambda y: P.End("R", y))
    # F01: "the left hook high and the right ones high and low"
    assert ends(f01.drawn) == [L(0.42), R(0.42), R(-0.42)]
    # F02: "the new one in, from a hook low on the left"; the cords there stay on their hooks
    assert f02.moved.hook == "bottom" and ends(f02.drawn) == [L(0.42), L(-0.42), R(0.42), R(-0.42)]
    # F03: "the new one out, from a hook on the right between the other two"
    assert f03.moved.hook == "middle" and ends(f03.drawn) == [L(0.42), L(-0.42), R(0.42), R(0), R(-0.42)]
    assert -0.42 < 0 < 0.42
    # the strip's last frame IS round 3's P1.1, \ringcords*[beam]{green ring, five cords}, cord
    # for cord (HEAD's call, written out here)
    p11_round3 = P.ringcords("beam", "green ring, five cords", True)
    assert f03.drawn.cords == p11_round3.drawn.cords and f03.drawn.rings == p11_round3.drawn.rings
    assert f03.labels == p11_round3.labels
    assert f03.log[0].split(" | ")[1] == p11_round3.log[0].split(" | ")[1]   # the same dump reading
    # the wrong order (out first) would pass a different state between: the strip shows the order
    assert P.ringcords("beam; add=out", p11.args[0]).drawn.cords != f02.drawn.cords
    # three cords, then four, then five: one in two out, two in two out, two in three out
    for fr, (nin, nout) in zip((f01, f02, f03), ((1, 2), (2, 2), (2, 3))):
        assert (sum(e.side == "L" for e in ends(fr.drawn)), sum(e.side == "R" for e in ends(fr.drawn))) == (nin, nout)
        assert np.allclose(tangle_map(fr.drawn), zspider(nin, nout, 0))
    assert "Bead at the top, no turn at all." in sec and f03.drawn.rings[0].t == P.parse_turn("at the top")
    assert P.describe(p11)[0] == "ringsteps: strip of 3 frames" and len(P.describe(p11)) == 6

    # Block 1.2: the slide, before and after, the same tangle
    assert p12.args == p13.args
    assert "one bead — the left-hand ring's — a third of the way round, the other a quarter" in a2
    assert "the near ring, the left-hand one as they face it," in a2
    assert "the two turns added -- a third and a quarter together" in a2
    assert a2.endswith("the far end of the cord is exactly where it was.]*")
    assert b12.setting == "loose" and b12.drawn.kind == "two" and b12.marks == ()
    left, right = b12.drawn.rings
    assert left.c == right.c == "green" and left.x < right.x
    assert (left.t, right.t) == (P.parse_turn("a third of the way round"), P.parse_turn("a quarter"))
    assert b13.setting == "loose" and b13.drawn.kind == "single"
    (s,) = b13.drawn.rings
    assert s.c == "green" and s.t == P.parse_turn("a third and a quarter") == 14   # seven o'clock
    assert (s.x, s.y) == (right.x, right.y)                                   # where the far one was
    ends12 = {e for seg in P.segments(b12.drawn) for e in seg if isinstance(e, P.End)}
    ends13 = {e for seg in P.segments(b13.drawn) for e in seg if isinstance(e, P.End)}
    assert ends12 == ends13                                                   # the cord's ends unmoved
    assert np.allclose(tangle_map(b12.drawn), tangle_map(b13.drawn))          # fusion, the pictured turns
    assert np.allclose(tangle_map(b13.drawn), zspider(1, 1, phase(8) + phase(6)))
    # its marks (marks=on): "She sets a finger to the near ring ... slides it along the cord until
    # it meets the far one; the cord does not move and its far end does not stir"
    assert "slides it along the cord until it meets the far one; the cord does not move and its far end does not stir" in a2
    assert [m.kind for m in b13.marks] == ["ghost", "slide", "bead", "bead-arc", "pin"]
    ghost, slide, bead, arc, pin = b13.marks
    assert ghost.ring == left and ghost.at == (-0.4, 0)                      # the near ring, where it stood
    assert all(np.hypot(ghost.at[0] - q.x, ghost.at[1] - q.y) > 2 * P.RO for q in b13.drawn.rings)  # uncovered: in full
    (x0, y0), c1, c2, (x1, y1) = slide.pts                                   # the slide: from it to the far ring
    assert (x0, x1) == (left.x, right.x) and y0 == y1 == P.SLIDE_Y > P.RO   # above the cord, clear of the rings
    assert all(y > P.RO for _, y in (c1, c2))                                # bowed upward
    assert np.isclose(np.hypot(pin.at[0] - P.BX, pin.at[1]), 0.01) and pin.r < 0.1   # the cord's far end held
    # "at a turn that is the two turns added": the far ring's bead from its own turn on by the
    # near ring's, clockwise (the way the Crab measures a turn: "clockwise as you face it")
    assert "clockwise as you face it" in sec
    assert bead.ring == right and np.allclose(bead.at, P.bead_xy(right)) and arc.dir == 1
    (cx, cy), (rad, b0, b1) = arc.pts
    assert (cx, cy) == (right.x, right.y) and rad == P.RO + 0.06
    assert np.isclose((b0 - b1) % 360 + 2 * 0.07 / rad * 180 / np.pi, 15 * left.t)   # by the near turn
    assert P.describe(p13)[1:] == ["ringcords: + ghost green ring 8 at (-0.400,0.000)",
                                   "ringcords: + slide arc -0.400 -> 0.400 above the cord",
                                   "ringcords: + ghost bead 6 on the far ring",
                                   "ringcords: + bead arc 6 -> 14 clockwise",
                                   "ringcords: + pin the cord's far end at (1.000,0.000)"]
    # without marks=on the old call draws what it drew (round 3): no mark at all
    assert P.ringcords("slid", p13.args[0]).marks == () and P.ringcords("slid", p13.args[0]).drawn == b13.drawn


# --- pictures: phases below are Fractions in units of pi ---
def two_green(a, b):
    """Two green spiders in a row, joined by one wire (phases a, b)."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    z1 = g.add_vertex(VT.Z, 0, 1, phase=Fraction(a))
    z2 = g.add_vertex(VT.Z, 0, 2, phase=Fraction(b))
    o = g.add_vertex(VT.BOUNDARY, 0, 3)
    g.add_edge((i, z1)); g.add_edge((z1, z2)); g.add_edge((z2, o))
    g.set_inputs((i,)); g.set_outputs((o,))
    return g


def one_green(a):
    """A single green spider (phase a)."""
    g = zx.Graph()
    i = g.add_vertex(VT.BOUNDARY, 0, 0)
    z = g.add_vertex(VT.Z, 0, 1, phase=Fraction(a))
    o = g.add_vertex(VT.BOUNDARY, 0, 2)
    g.add_edge((i, z)); g.add_edge((z, o))
    g.set_inputs((i,)); g.set_outputs((o,))
    return g


def sketchbook():
    a, b = Fraction(1, 4), Fraction(1, 3)
    # fusion is EXACT -- the scalar is included (card: fusion holds on the nose)
    assert same_picture(two_green(a, b), one_green(a + b))
    assert not same_picture(two_green(a, b), one_green(a))          # phases really add

    # derive it with the rewrite rule: spider_simp fuses the two greens into one
    g = two_green(a, b)
    simp.spider_simp(g)
    internal = [v for v in g.vertices() if g.type(v) == VT.Z]
    assert len(internal) == 1                                       # the two became one spider
    assert g.phase(internal[0]) == (a + b) % 2                      # phase is the sum
    assert same_picture_up_to_scalar(g, one_green(a + b))


if __name__ == "__main__":
    verdict("Fusion", blackboard, sketchbook)
