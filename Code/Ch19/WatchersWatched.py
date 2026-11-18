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
    assert DIALOGUE.check_whole(ORD) == 21
    DIALOGUE.keyed_first(ORD, kind=PC.keyed_kind)
    starred = [(o, c.macro) for o in ORD for b in DIALOGUE.blocks(o) for c in b if c.keyed]
    assert starred == [("First", "rulecard"), ("Second", "hingecloth"), ("Third", "darecloth"),
                       ("Fourth", "boxcloth"), ("Fourth", "crablutesteps")]
    PC.check_negative_anchors(DIALOGUE)
    every_call = [c for o in ORD for b in DIALOGUE.blocks(o) for c in b]
    assert len(every_call) == 21 and [c.macro for c in every_call].count("rulecard") == 2
    assert sum(c.macro in PC.STRIPS_WHO for c in every_call) == 5                # five strips, sixteen singles
    every = [fr for c in every_call for fr in PC.frame_runs(c)]                  # every frame: (Run, caption, keyed)
    assert len(every) == 26 and sum(k for _, _, k in every) == 5
    # the frames per challenge (card section 3e.2, as applied)
    assert [sum(len(PC.frame_runs(c)) for b in DIALOGUE.blocks(o) for c in b) for o in ORD] == [5, 7, 7, 7, 0]
    # a marked frame is never keyed; no card and no lute frame is marked
    assert all(not (R.marked and k) for R, _, k in every)
    assert not any(R.marked or R.mark_tikz for R, _, _ in every if R.kind in ("card", "lute"))
    assert sum(R.marked for R, _, _ in every) == 13 == 4 + 5 + 4            # the hub, the dare, the box
    # the Cat and the Genie's lamp are never pictured: no macro draws them, and no caption, keyed
    # word or dump names them (the table lamp a card is held to is not the Genie's)
    assert PC.MACROS == ("rulecardsteps", "hingeclothsteps", "dareclothsteps", "boxclothsteps", "crablutesteps",
                         "rulecard", "hingecloth", "darecloth", "boxcloth", "crablute")
    for R, cap, k in every:
        said = " ".join([cap, R.dump] + list(R.labels))                    # what is drawn (the log may name the Cat's point)
        assert not re.search(r"\b(cat|cat's|genie|genie's|purr)\b", said, re.I), cap
        assert not re.search(r"\blamp\b", " ".join([R.dump] + list(R.labels)), re.I), cap
    # the Cat is never drawn: no picture after the Fifth Challenge begins (the dialogue ends when
    # someone looks at the Cat), in either file
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        text = DIALOGUE.read(path)
        assert text.rindex("\\begin{center}") < text.index("## Fifth Challenge")
        assert [l for l in text.split("\n") if l.startswith("**")][-1] == "**Cat:** Awake."   # the last word spoken
    # the counting scan (section 6.17 (d)): no caption and no keyed word counts, each alone and
    # all joined; the whole files' hits are exactly the approved ones, ruled kept
    words = [cap for _, cap, _ in every] + [w for R, _, _ in every for w in R.labels]
    assert len(set(w for R, _, _ in every for w in R.labels)) == 11
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
    check_faults()


# the Lua's fault sites (card section 3e.6): one call raising each, in TEX_SITES order
FAULT_CASES = [
    ("step-unknown", "hinge", False, "red frob", ""),
    ("step-done", "hinge", False, "red point, red point", ""),
    ("step-after", "hinge", False, "slid into the hub", ""),
    ("card", "card", False, "face", "a joker"),
    ("view", "card", False, "sideways", "new"),
    ("lute", "lute", False, "humming", ""),
    ("opt-unknown", "hinge", False, "step=red point; colour=red", ""),
    ("opt-twice", "hinge", False, "step=red point; step=red point", ""),
    ("opt-bare-two", "hinge", False, "red point; red point; marks=off", ""),
    ("opt-onoff", "hinge", False, "step=red point; marks=maybe", ""),
    ("marks-no-step", "hinge", False, "red point; marks=on", ""),
    ("marks-keyed", "hinge", True, "red point; step=slid into the hub", ""),
    ("step-commas", "hinge", False, "step=red point, slid into the hub", ""),
    ("step-empty", "hinge", False, "red point; step=", ""),
    ("card-expects", "card", False, "expects=the same card", "new"),
    ("card-step", "card", False, "step=turned over", "new"),
    ("card-through", "card", False, "through; step=held to the lamp", "new"),
    ("hinge-expects", "hinge", False, "expects=red into red", ""),
    ("dare-expects", "dare", False, "expects=an eighth", ""),
    ("box-part-turn", "box", False, "red points slid in, her look's ring slid along", ""),
    ("box-part-begin", "box", False, "his look's ring slid along", ""),
    ("box-part-next", "box", False, "red points slid in, his look's ring slid along, loops off", ""),
    ("box-expects", "box", False, "expects=awake", ""),
    ("box-part-keyed", "box", True, "red points slid in, his look's ring slid along", ""),
    ("lute-expects", "lute", False, "expects=a chord", ""),
    ("lute-same", "lute", False, "step=still", ""),
    ("lute-still-first", "lute", False, "pairs sounding; step=top string sounding", ""),
    ("strip-unknown", "steps", False, "nope", "", "", "x", "a"),
    ("strip-bare", "steps", False, "hinge", "", "loud", "red point", "a; b"),
    ("strip-star-from", "steps", True, "hinge", "", "from=off", "red point", "a"),
    ("strip-none", "steps", False, "hinge", "", "", "", "a"),
    ("strip-commas", "steps", False, "hinge", "", "", "red point, slid into the hub", "a; b"),
    ("strip-long", "steps", False, "hinge", "", "", "red point; slid into the hub; feet slid apart; tube", "a; b; c; d; e"),
    ("strip-captions", "steps", False, "hinge", "", "", "red point", "a"),
]
# a legal step for each macro (its own words), and its state argument
STEP = {"card": ("held to the lamp", "new"), "hinge": ("red point", ""), "dare": ("drawn through", ""),
        "box": ("red points slid in", ""), "lute": ("top string sounding", "")}
MARKED = {"hinge": ("red point", "slid into the hub"), "dare": ("", "drawn through"), "box": ("", "red points slid in")}
STRIPSTEP = {"card": "held to the lamp", "hinge": "red point", "dare": "drawn through", "box": "red points slid in",
             "lute": "top string sounding"}
def pair_cases():
    """A call for every macro-site pair the fault cases leave (the shared sites, from each macro)."""
    out = []
    for k, (st, arg) in STEP.items():
        out += [("opt-unknown", k, False, "step=%s; colour=red" % st, arg),
                ("opt-twice", k, False, "step=%s; step=%s" % (st, st), arg),
                ("opt-bare-two", k, False, "a; b; marks=off", arg),
                ("opt-onoff", k, False, "step=%s; marks=maybe" % st, arg),
                ("marks-no-step", k, False, "; marks=on", arg),
                ("step-commas", k, False, "step=%s, %s" % (st, st), arg),
                ("step-empty", k, False, "; step=", arg)]
    for k, (bare, st) in MARKED.items():
        out.append(("marks-keyed", k, True, "%s; step=%s" % (bare, st), ""))
    for k, st in STRIPSTEP.items():
        arg = "new" if k == "card" else ""
        out += [("opt-unknown", "steps", False, k, arg, "colour=red", st, "a; b"),
                ("opt-twice", "steps", False, k, arg, "from=on; from=on", st, "a; b"),
                ("opt-bare-two", "steps", False, k, arg, "loud; soft", st, "a; b"),
                ("opt-onoff", "steps", False, k, arg, "marks=maybe", st, "a; b"),
                ("strip-bare", "steps", False, k, arg, "loud", st, "a; b"),
                ("strip-star-from", "steps", True, k, arg, "from=off", st, "a"),
                ("strip-none", "steps", False, k, arg, "", "", "a"),
                ("strip-commas", "steps", False, k, arg, "", st + ", " + st, "a; b"),
                ("strip-long", "steps", False, k, arg, "", "; ".join([st] * 4), "a; b; c; d; e"),
                ("strip-captions", "steps", False, k, arg, "", st, "a")]
    for k in ("hinge", "dare", "box"):
        out += [("step-unknown", k, False, "frob", ""), ("step-done", k, False, ", ".join([STEP[k][0]] * 2), "")]
    out += [("step-after", "dare", False, "card backwards", ""), ("step-after", "box", False, "slid together", "")]
    return out


def check_faults():
    """Every fault site of ch19.lua raised by its own call, alone, each with messages of its own;
    the 34 sites are the Lua's, all of them, and the 113 macro-site pairs each reached; the ';'
    bracket is order-free (every order of a refused bracket raises the same faults); a call with a
    fault draws no marks."""
    def run(case):
        if case[1] == "steps":
            _, _, keyed, kind, state, opt, stepspec, caps = case[:8]
            return PC.ch19_steps, (kind, keyed, state, opt, stepspec, caps)
        _, kind, keyed, br, state = case[:5]
        return PC.run_frame, (kind, br, state, keyed)
    seen = {}
    for case in FAULT_CASES:
        site = case[0]
        fn, args = run(case)
        assert PC.faults(fn, *args) == [site], (site, PC.TEX_ERRORS)
        (m,) = PC.messages(fn, *args)
        assert m not in seen, (site, seen.get(m))                          # no two sites share a message
        seen[m] = site
    assert [c[0] for c in FAULT_CASES] == [k for k, _ in PC.TEX_SITES] and len(PC.ERROR_SITES) == 34
    assert {k.rstrip("("): v for k, v in PC.SITE_COUNTS.items()} == {"errf": 3, "cerr": 2, "lerr": 1, "xerr": 28}
    assert sum(PC.SITE_COUNTS.values()) == 34 == 6 + 28
    # every pair: the site raised with the macro's own name, from every macro that reaches it
    reached = set()
    for case in FAULT_CASES + pair_cases():
        fn, args = run(case)
        for m in PC.messages(fn, *args):
            reached.add((case[0], m.split(": ", 1)[0]))
    want = {(k, w) for k, ws in PC.SITE_MACROS.items() for w in ws}
    assert len(want) == 113 and reached == want | {("strip-unknown", "steps")}, sorted(want ^ reached)
    # the bracket's items in any order: the same faults
    import itertools
    orders = 0
    for case in FAULT_CASES:
        k = 5 if case[1] == "steps" else 3
        items = case[k].split("; ")
        if len(items) < 2:
            continue
        fn, args = run(case)
        want_f = sorted(PC.faults(fn, *args))
        for order in itertools.permutations(items):
            c2 = case[:k] + ("; ".join(order),) + case[k + 1:]
            fn, args = run(c2)
            assert sorted(PC.faults(fn, *args)) == want_f, (case[0], order)
            orders += 1
    assert orders >= 10
    # a call with a fault draws no marks; a refused step none either
    for case in FAULT_CASES:
        if case[1] != "steps":
            PC.TEX_ERRORS.clear()
            R = PC.run_frame(case[1], case[3], case[4], case[2])
            assert PC.TEX_ERRORS and R.mark_tikz == [] and R.arcs == [] and R.ghosts == [], case[0]
    # the arcs' floor: an arc bowed under the Lua's MINSAG is refused, with the reason, and the same
    # arc bowed just over it fits (no arc of the dialogue is refused for its bow, so the rule is tried
    # on a drawn one: the hub's hop into the hub, its bow changed)
    import copy
    # the floor as the Lua writes it (read here from the Lua's text, not through the transcription),
    # and its comment's figure
    floor = float(re.search(r"^local MINSAG = ([\d.]+) ", PC.LUA_TEXT, re.M).group(1))
    assert PC.MINSAG == floor == 0.055 and "bows at least MINSAG (0.055)" in PC.LUA_TEXT
    R = PC.run_frame("hinge", "red point; step=slid into the hub", "", False)
    S1 = PC.hinge_scene(R.after)
    g = PC.G(S1, PC.HOPTS, PC.HWHO)
    O_ = PC.c_obstacles(S1, g)
    assert PC.c_fits(R.arcs[0], O_, list(R.ghosts), g)[0]
    flat = copy.copy(R.arcs[0])
    flat.sag = floor - 0.001
    assert PC.c_fits(flat, O_, list(R.ghosts), g) == (False, "it bows only %s from its chord, under %s" % (
        PC.f(flat.sag), PC.f(floor)))
    flat.sag = floor + 0.0005
    assert PC.c_fits(flat, O_, list(R.ghosts), g)[0]
    PC.TEX_ERRORS.clear()
    PC.check_sites()                                                     # the wrappers' tex.error calls
    PC.check_lua()                                                       # every rule and place, in the Lua's words
    PC.check_pins()


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
