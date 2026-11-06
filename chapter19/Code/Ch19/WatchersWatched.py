"""Challenge V: Watchers Watched  (the sealed rooms; the synthesis)
==================================================================
"Aunt Hillary has played it round after round. One round in twelve, both hear "ok"."  -- Anteater

Claim: in Frauchiger and Renner's experiment the three certainties are exact zeros,
(<t| (x) <ok|) psi = (<h| (x) <up|) psi = (<ok| (x) <down|) psi = 0, and both watchers
announce "ok" with probability 1/12 (Result 19.14) -- an instance of the paper's numbers.
Their theorem (Result 19.15) is quoted in the chapter, not checked; the code settles nothing
about which assumption to give up.  MIP* = RE (Result 19.16) has no code: it is a statement
about all games at once, and no computation on one picture speaks to it.
"""
import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import ket, kron, matrix_of, verdict
from dfp_props import Dialogue
import props_ch19 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

# heads = 0, tails = 1 (lab L-bar);  z = -1/2 (down) = 0, z = +1/2 (up) = 1 (lab L)
DOWN, UP = ket(0), ket(1)
RIGHT = np.sqrt(0.5) * (DOWN + UP)                         # |->> = sqrt(1/2)(|down> + |up>)
OK = np.sqrt(0.5) * (ket(0) - ket(1))                      # both labs' |ok>


def amp(bra_top, bra_bottom, psi):
    """(<a| (x) <b|) psi, lab L-bar on top."""
    return np.vdot(np.kron(bra_top, bra_bottom), psi)


def hardy_state(p_heads):
    """F-bar's quantum coin sqrt(p)|h> + sqrt(1-p)|t>; on heads |down>, on tails |->>, sent to L."""
    return (np.sqrt(p_heads) * np.kron(ket(0), DOWN)
            + np.sqrt(1 - p_heads) * np.kron(ket(1), RIGHT))


def blackboard():
    psi = hardy_state(1 / 3)
    assert np.allclose(psi, (ket(0, 0) + ket(1, 0) + ket(1, 1)) / np.sqrt(3))
    assert np.isclose(np.linalg.norm(psi), 1)
    assert np.allclose(kron(ket(1), ket(0)).ravel(), ket(1, 0))          # lab L-bar is the top wire
    # the three certainties, exact zeros
    assert np.isclose(amp(ket(1), OK, psi), 0)        # F-bar saw tails: W will say "fail"
    assert np.isclose(amp(ket(0), ket(1), psi), 0)    # F saw z = +1/2: r was not heads
    assert np.isclose(amp(OK, ket(0), psi), 0)        # W-bar said "ok": F saw z = +1/2
    # ...and yet both say "ok", one round in twelve
    a = amp(OK, OK, psi)
    assert np.isclose(a, 1 / (2 * np.sqrt(3)))
    assert np.isclose(abs(a) ** 2, 1 / 12)
    assert np.allclose(np.kron(OK, OK), 0.5 * (ket(0, 0) - ket(0, 1) - ket(1, 0) + ket(1, 1)))
    # NEGATIVE: the wrong |ok> (a plus sign) breaks the first certainty; an even coin
    # changes the one in twelve
    wrong_ok = np.sqrt(0.5) * (ket(0) + ket(1))
    assert not np.isclose(amp(ket(1), wrong_ok, psi), 0)
    even = hardy_state(1 / 2)
    assert not np.isclose(abs(amp(OK, OK, even)) ** 2, 1 / 12)
    assert np.isclose(amp(ket(1), OK, even), 0)                         # (the tails zero survives)

    pictures()


# --------------------------------------------------------------- the dialogue as a whole (no picture)

def pictures():
    """The Fifth Challenge has no picture (identity card, section 4); this file runs the
    whole-dialogue checks: the sections in order with every picture inside one, the starred form
    on the first picture of each macro, the forty negative anchors, the counting scan, and the
    card's section 1c form check, forbidden-word grep, label and Cat counts, on both files."""
    calls = DIALOGUE.check_layout("Fifth", sizes=[])
    assert calls == [] and DIALOGUE.blocks("Fifth") == [] and DIALOGUE.blocks("Fifth", DIALOGUE.raw) == []
    assert DIALOGUE.anchors("Fifth") == [] and PC.SECTION_PICS["Fifth"] == ()
    ORD = PC.ORDER
    assert DIALOGUE.check_whole(ORD) == 15
    DIALOGUE.keyed_first(ORD, kind=PC.keyed_kind)
    starred = [(o, c.macro) for o in ORD for b in DIALOGUE.blocks(o) for c in b if c.keyed]
    assert starred == [("First", "rulecard"), ("Second", "hingecloth"), ("Third", "darecloth"),
                       ("Fourth", "boxcloth"), ("Fourth", "crablute")]
    PC.check_negative_anchors(DIALOGUE)
    every_call = [c for o in ORD for b in DIALOGUE.blocks(o) for c in b]
    assert [c.macro for c in every_call].count("rulecard") == 4 and len(every_call) == 15
    # the Cat and the Genie's lamp are never pictured: no macro draws them, and no caption, keyed
    # word or dump names them (the table lamp a card is held to is not the Genie's)
    assert PC.MACROS == ("rulecard", "hingecloth", "darecloth", "boxcloth", "crablute")
    for c in every_call:
        pic = PC.picture(c)
        said = " ".join([c.caption, pic.dump] + list(pic.labels))
        assert not re.search(r"\b(cat|cat's|genie|genie's|purr)\b", said, re.I), c.text
        assert not re.search(r"\blamp\b", " ".join([pic.dump] + list(pic.labels)), re.I), c.text
    # the counting scan (section 6.17 (d)): no caption and no keyed word counts, each alone and
    # all joined; the whole files' hits are exactly the approved ones, ruled kept
    words = [c.caption for c in every_call] + [w for c in every_call for w in PC.picture(c).labels]
    assert len(set(w for c in every_call for w in PC.picture(c).labels)) == 11
    for w in words + [" ".join(words)]:
        assert not PC.COUNT_SCAN.search(w), w
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        text = pathlib.Path(path).read_text(encoding="utf-8")
        assert tuple(PC.count_hits(text)) == PC.COUNT_KEPT[pathlib.Path(path).name], path
        # section 1c: the form check, the forbidden-word grep, the hooks, the Genie, the Cat
        E, seen = PC.form_check(text)
        assert E == [], E
        assert seen["subject"] == [("Second Challenge", "Tortoise"), ("Third Challenge", "Crab"),
                                   ("Fourth Challenge", "Achilles")]
        assert seen["countersubject"] == [("First Challenge", "Achilles")]
        assert seen["speakers"] == ["Achilles", "Anteater", "Cat", "Crab", "Genie", "Tortoise"]
        assert PC.forbidden_lines(text) == []
        assert text.count("Let Bob draw") == 0
        assert PC.label_count(text, "Genie") == 6 and PC.label_count(text, "Cat") == 2
        assert PC.cat_count(text) == 25
    # the form check bites: a stray marker line, a Cat answering early, the index, a course total
    text = DIALOGUE.read(DIALOGUE.dialog)
    first = text.index("**Achilles:**")
    for bad in ("**Achilles:** Whatever the cards say.\n\n", "**Cat:** Asleep.\n\n",
                "**Crab:** Nineteen, then.\n\n", "**Crab:** Nine courses.\n\n"):
        assert PC.form_check(text[:first] + bad + text[first:])[0] != [], bad
    assert PC.forbidden_lines(text + "\nThe ZX of it.\n") != []


# ----------------------------------------------------------------------------- the sketchbook
# sqrt3 psi = (two green points at rest) - 1/2 (a red point at rest beside a red point at a half).
# Effects: <ok| = 1/sqrt2 x a green effect at a half;  <0|, <1| = 1/sqrt2 x red effects at rest,
# at a half.  Each closed term below is one picture, closed by a pair of effects.

EFFECT = {"ok": (VT.Z, Fraction(1)), "0": (VT.X, Fraction(0)), "1": (VT.X, Fraction(1))}
TERMS = {"green": ((VT.Z, 0), (VT.Z, 0)), "red": ((VT.X, 0), (VT.X, 1))}


def closed_term(term, top, bottom):
    """One of the two pictures of sqrt3 psi, closed by the effects `top` and `bottom` (each a
    spider with one leg: colour and turn), with the effects' number (1/sqrt2)^2 on the scalar."""
    g = zx.Graph()
    n = 0
    for row, (pt, eff) in enumerate(zip(TERMS[term], (top, bottom))):
        u = g.add_vertex(pt[0], row, 0, Fraction(pt[1]))
        v = g.add_vertex(eff[0], row, 1, eff[1])
        g.add_edge((u, v))
        n += 1
    assert g.num_edges() == n == 2
    g.scalar.add_power(-2)
    return g


def reduced_number(g):
    simp.full_reduce(g)
    assert g.num_vertices() == 0          # a closed picture reduced to nothing: its number
    return g.scalar.to_number()


def state_term(term):
    g = zx.Graph()
    outs = []
    for row, pt in enumerate(TERMS[term]):
        u = g.add_vertex(pt[0], row, 0, Fraction(pt[1]))
        o = g.add_vertex(VT.BOUNDARY, row, 2)
        g.add_edge((u, o))
        outs.append(o)
    g.set_outputs(tuple(outs))
    assert g.num_edges() == 2
    return g


def sketchbook():
    psi = (ket(0, 0) + ket(1, 0) + ket(1, 1)) / np.sqrt(3)
    # the state as a sum of two pictures (a sum of pictures is not one picture: it is taken here)
    s = matrix_of(state_term("green")).ravel() - 0.5 * matrix_of(state_term("red")).ravel()
    assert np.allclose(s, np.sqrt(3) * psi)
    assert np.allclose(matrix_of(state_term("red")).ravel(), 2 * ket(0, 1))    # the one corner

    # DERIVATION: each closed term reduced by full_reduce to its number (no vertex left);
    # (T3) TRANSLATION CHECK: the two terms are added with their numbers on the blackboard.
    targets = {("1", "ok"): 0, ("0", "1"): 0, ("ok", "0"): 0, ("ok", "ok"): 1 / (2 * np.sqrt(3))}
    for (top, bottom), want in targets.items():
        total = 0
        for term, weight in (("green", 1), ("red", -0.5)):
            g = closed_term(term, EFFECT[top], EFFECT[bottom])
            unreduced = matrix_of(g)[0, 0]
            number = reduced_number(g)
            assert np.isclose(number, unreduced)
            total += weight * number
        assert np.isclose(total / np.sqrt(3), want), (top, bottom)
    assert np.isclose(abs(targets[("ok", "ok")]) ** 2, 1 / 12)
    # the first zero, as Coecke's Sketchbook reads it: <ok| on a green point at rest fuses to a
    # green spider with no legs at a half, 1 + e^{i pi} = 0; <1| on a red point at rest, the same
    for term in ("green", "red"):
        assert np.isclose(matrix_of(closed_term(term, EFFECT["1"], EFFECT["ok"]))[0, 0], 0)
    ring = zx.Graph()
    ring.add_vertex(VT.Z, 0, 0, Fraction(1))
    assert np.isclose(matrix_of(ring)[0, 0], 0)
    # NEGATIVE: a green effect at rest (a plus sign) in place of <ok| does not give the zero
    plus = (VT.Z, Fraction(0))
    wrong = sum(w * reduced_number(closed_term(t, EFFECT["1"], plus))
                for t, w in (("green", 1), ("red", -0.5)))
    assert not np.isclose(wrong, 0)


if __name__ == "__main__":
    verdict("WatchersWatched", blackboard, sketchbook)
