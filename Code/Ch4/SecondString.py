"""Challenge 1: A String Below
==============================
"Then read me off the share of heads above and tails below, together."  -- Tortoise

Claim: a second wire drawn below the first gives one entry per PAIR of answers,
and for two separate coins that entry is a product: the joint column is the
Kronecker column (a,b) (x) (c,d) = (ac, ad, bc, bd), top wire the left factor.
Brackets multiply: (<p'| (x) <f'|)(|p> (x) |f>) = <p'|p> <f'|f>.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import Picture, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch04 as P


def blackboard():
    rng = np.random.default_rng(41)

    # --- ket order: the top wire is the first label -------------------------
    assert np.allclose(ket(0, 1), np.array([0, 1, 0, 0]))       # 00,01,10,11
    assert np.allclose(ket(1, 0), np.array([0, 0, 1, 0]))
    assert not np.allclose(ket(0, 1), ket(1, 0))
    for a in (0, 1):
        for b in (0, 1):
            assert np.allclose(ket(a, b), np.kron(ket(a), ket(b)))

    # --- the dialogue's four panels, on shares ------------------------------
    upper = np.array([1 / 2, 1 / 2])                   # the tossed silk coin
    lower = np.array([3 / 4, 1 / 4])                   # the hemp coin: heads 3 in 4
    panels = np.kron(upper, lower)
    assert np.allclose(panels, [3 / 8, 1 / 8, 3 / 8, 1 / 8])
    assert np.isclose(panels.sum(), 1)                 # "Eight eighths."
    # Achilles's two little lists: four numbers that sum to two, and no line
    # for "heads above and tails below together"
    two_lists = np.array([1 / 2, 1 / 2, 3 / 4, 1 / 4])
    assert np.isclose(two_lists.sum(), 2)
    assert not np.isclose(1 / 2 + 1 / 4, panels[1])    # adding: not the panel
    assert not np.isclose(1 / 2 + 3 / 4, panels[0])    # "More than the whole morning."
    assert np.isclose(panels[1], 1 / 2 * 1 / 4)        # multiplying: the panel

    # --- the Kronecker column, on generic complex columns -------------------
    psi, phi = random_state(rng), random_state(rng)
    assert not np.allclose(psi.imag, 0) and not np.allclose(phi.imag, 0)
    a, b = psi
    c, d = phi
    assert np.allclose(np.kron(psi, phi), [a * c, a * d, b * c, b * d])
    assert not np.allclose(np.kron(psi, phi), np.kron(phi, psi))   # order matters
    assert np.isclose(np.linalg.norm(np.kron(psi, phi)), 1)        # a product IS a state

    # bilinear in each factor separately, with a complex multiple
    psi2 = random_state(rng)
    lam = 0.3 - 1.7j
    assert np.allclose(np.kron(psi + psi2, phi), np.kron(psi, phi) + np.kron(psi2, phi))
    assert np.allclose(np.kron(psi, phi + psi2), np.kron(psi, phi) + np.kron(psi, psi2))
    assert np.allclose(np.kron(lam * psi, phi), lam * np.kron(psi, phi))
    assert np.allclose(np.kron(lam * psi, phi), np.kron(psi, lam * phi))

    # --- the bra of a product is the product of the bras ---------------------
    bra = np.kron(psi, phi).conj()
    assert np.allclose(bra, np.kron(psi.conj(), phi.conj()))
    assert not np.allclose(bra, np.kron(psi, phi))     # conjugation is not free

    # --- Proposition 4.1: brackets multiply ---------------------------------
    for _ in range(50):
        p, q, pp, qq = (random_state(rng) for _ in range(4))
        left = np.kron(pp, qq).conj() @ np.kron(p, q)
        assert np.isclose(left, np.vdot(pp, p) * np.vdot(qq, q))
        # the plausible mix-ups: no conjugation, or the factors crossed
        assert not np.isclose(left, np.kron(pp, qq) @ np.kron(p, q))
        assert not np.isclose(left, np.vdot(pp, q) * np.vdot(qq, p))
    assert not np.isclose(np.vdot(pp, p).imag, 0)      # the brackets are not real

    # --- the chapter's worked numbers (Sec 4.2) ------------------------------
    arrow = np.array([3 / 5, 4 / 5], dtype=complex)
    omega = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    joint = np.kron(arrow, omega)
    assert np.allclose(joint, np.array([3, 3j, 4, 4j]) / (5 * np.sqrt(2)))
    assert np.allclose(np.abs(joint) ** 2, [9 / 50, 9 / 50, 16 / 50, 16 / 50])
    assert np.isclose((np.abs(joint) ** 2).sum(), 1)
    assert np.allclose(np.kron(omega, arrow),
                       np.array([3, 4, 3j, 4j]) / (5 * np.sqrt(2)))
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    both_plus = np.kron(plus, plus)
    assert np.isclose(both_plus.conj() @ joint, (7 + 7j) / (10 * np.sqrt(2)))
    assert np.isclose(np.vdot(plus, arrow), 7 / (5 * np.sqrt(2)))
    assert np.isclose(np.vdot(plus, omega), (1 + 1j) / 2)

    pictures(panels, two_lists)


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws its props with five macros
# (pdf/props/ch04.tex; the rules in pdf/props/ch04.lua):
#   \stringframe[opts]{state}{caption}    the Tortoise's frame, face on; opts
#        the place of the coins, and a step (slide=, lift=, exchange=) with its
#        marks, or what a wrong rule predicts (expects=), on cols= columns
#   \framewalk[opts]{state}{captions}     a step strip of one walk of the coins
#   \framesteps[opts]{state}{steps}{captions}   a step strip of moves (unused)
#   \foldsheet[opts]{folds}{marks}{caption}     the sheet, folded and opened flat
#   \foldsteps[opts]{first}{last}{captions}     a step strip of the folding
# The Lua COMPUTES what a picture shows -- rows, coins, crossings, panels,
# arrows, the moves, the marks, a wrong rule's prediction -- and
# tools/props_ch04.py transcribes it.  The calls are read out of this
# challenge's section of dialog.md (tools/dfp_props.py), never copied, so an
# edit to the dialogue that breaks a picture breaks here.  This first file also
# checks the dialogue as a whole: twenty-nine calls, thirty-six frames, the
# keyed rule, the marks and the predictions.
# --------------------------------------------------------------------------
MACROS = ("stringframe", "foldsheet", "framewalk", "framesteps", "foldsteps")
DIALOGUE = Dialogue(__file__, macros=MACROS)
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")
X = np.array([[0, 1], [1, 0]], dtype=complex)            # "turn it over"
S0 = np.array([[1, 1], [0, 0]], dtype=complex)           # "heads up, whatever it was"
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
GATES = {"over": X, "heads": S0, "yoke": CNOT, "cross": SWAP}


def marked(pic):
    """Whether a picture carries motion marks (a step's, a walk's, a fold's)."""
    if isinstance(pic, P.Sheet):
        return pic.marks
    return pic.moved is not None or pic.walkmarks


def check_whole_dialogue():
    total, nframes, starred = P.check_dialogue(DIALOGUE, ORDINALS)
    assert total == 29 and nframes == 36                    # 3+3+3+4+4+4 moments
    assert [len(DIALOGUE.blocks(o)) for o in ORDINALS] == [3, 3, 3, 4, 4, 4]
    assert starred == [("First", 0), ("First", 2), ("Second", 2)]
    calls = [(o, c) for o in ORDINALS for b in DIALOGUE.blocks(o) for c in b]
    # the keyed rule is by labelling: by macro alone, P2.2 would be no first
    second = [c for o, c in calls if o == "Second"]
    assert second[2].keyed and second[2].macro == "stringframe"
    assert any(c.macro == "stringframe" for o, c in calls if o == "First")
    # four step strips, of 2, 3, 3 and 3 frames; \framesteps is not used
    strips = [(o, c.macro, len(P.pictures(c))) for o, c in calls
              if c.macro in ("framewalk", "framesteps", "foldsteps")]
    assert strips == [("First", "foldsteps", 2), ("Third", "framewalk", 3),
                      ("Fifth", "framewalk", 3), ("Sixth", "framewalk", 3)]
    # eight marked single frames; in a strip, the frames of a step carry its marks
    # (a fold, the walk through a column), the first and the arrival none
    singles = [(o, c) for o, c in calls if c.macro in ("stringframe", "foldsheet")]
    assert len([1 for o, c in singles if marked(P.picture(c))]) == 8
    for o, m, n in strips:
        (c,) = [c for oo, c in calls if oo == o and c.macro == m]
        want = [False, True] if m == "foldsteps" else [False, True, False]
        assert [marked(p) for p, _ in P.pictures(c)] == want
    # five predictions, each a refuted rule, never keyed, never marked
    rules = [(o, P.picture(c).expects) for o, c in calls if P.is_prediction(c)]
    assert rules == [("Second", "edge"), ("Third", "untouched"), ("Fifth", "row"),
                     ("Fifth", "flat"), ("Sixth", "patchwork")]
    for o, c in calls:
        if P.is_prediction(c):
            pic = P.picture(c)
            assert not c.keyed and not marked(pic) and not pic.labels
    # the hold rings: at the two exchanges only, on the pair's left-hand hooks
    pinned = [(o, c.caption, [(n, r, x) for n, r, x, y in pic.pins])
              for o, c in calls for pic, _ in P.pictures(c)
              if isinstance(pic, P.Frame) and pic.pins]
    assert pinned == [("Fifth", "crossed once", [("gut", 3, 0), ("horsehair", 4, 0)]),
                      ("Fifth", "wound", [("horsehair", 4, 0), ("gut", 3, 0)])]
    for o, c in calls:
        for pic, _ in P.pictures(c):
            if isinstance(pic, P.Frame):
                assert bool(pic.pins) == (pic.step == "exchange")
    # the old captions are gone (C1, C2), in both files
    P.check_edits(DIALOGUE, "Third", [("{blue: over the first fold}",
                                       "{blue: across the first fold}")])
    P.check_edits(DIALOGUE, "Sixth", [("{every one home}", "{all home}")])


def check_colon_rule():
    """One word, two meanings, fixed by the first colon."""
    coin = P.stringframe("start", "silk heads")
    clip = P.stringframe("start", "silk: heads")
    both = P.stringframe("start", "silk heads: heads")
    assert coin.face == {"silk": "heads"} and coin.ncol == 0 and coin.clips() == []
    assert clip.face == {"silk": None} and clip.clips() == [(1, "silk", 1, ("heads",))]
    assert clip.coins() == {}
    assert both.face == {"silk": "heads"} and both.clips() == [(1, "silk", 1, ("heads",))]
    assert P.faults(P.stringframe, "start", "silk heads heads") != []
    assert P.faults(P.stringframe, "start", "flax") != []  # an unknown string


def check_round4_rules():
    """The macros' own refusals, which a dialogue edit could otherwise slip past."""
    # an expects picture is never keyed and carries no marks
    assert P.faults(P.stringframe, "end;expects=row", "silk heads: cross, over; hemp: cross",
                    True) == ["stringframe: an expects picture is not keyed"]
    assert P.faults(P.foldsheet, "three", "blank", False, "expects=edge;marks=on") == [
        "foldsheet: an expects picture carries no marks"]
    # a wrong rule only where it differs from the paper
    assert P.faults(P.foldsheet, "twice", "blank", False, "expects=edge") == [
        "foldsheet: expects=edge differs from the paper from the third fold on"]
    # the fold strip: forwards only, at most four frames, a caption for each
    assert P.faults(P.foldsteps, "once", "none", "", "a; b") == [
        "foldsteps: the last fold must come after the first"]
    assert P.faults(P.foldsteps, "none", "four", "", "a; b; c; d; e") == [
        "foldsteps: 5 frames; at most four fit a line (from=off)"]
    assert P.faults(P.foldsteps, "none", "once", "", "one whole face") == [
        "foldsteps: 2 frames but 1 captions"]
    # marks on an unfolded sheet, and the key on a sheet not folded twice
    assert P.faults(P.foldsheet, "none", "blank", False, "marks=on") == [
        "foldsheet: marks=on needs a fold (once .. four)"]
    assert P.faults(P.foldsheet, "once", "blank", True) == [
        "foldsheet: the key names the four panels of the sheet folded twice"]


def pictures(panels, two_lists):
    P.check_lua()                     # ch04.lua and ch04.tex as the transcription was proved
    check_whole_dialogue()
    check_colon_rule()
    check_round4_rules()
    # Three moments.  P1.1 the frame; block 1.2 the fold strip (none, once)
    # beside the keyed sheet folded twice, with its marks; P1.3 the patchwork.
    found = P.check_section(DIALOGUE, "First", sizes=[1, 2, 1], frames=[1, 3, 1])
    (p11, [(fr, _)]), = found[0]
    (p12a, strip), (p12b, [(sh, _)]) = found[1]
    (p13, [(pw, _)]), = found[2]

    # P1.1: the frame, a string below; the coins are behind the card, not drawn
    assert p11.macro == "stringframe" and p11.keyed and fr.place == ("start", None)
    assert fr.hooked == ("silk", "hemp") and fr.ncol == 0
    assert fr.rows == {"silk": [1], "hemp": [2]}           # hemp BELOW silk
    assert fr.clips() == [] and fr.yokes() == [] and fr.crossings() == []
    assert fr.face == {"silk": None, "hemp": None} and fr.coins() == {}
    assert fr.labels == {"silk": 1, "hemp": 2}
    assert not marked(fr)
    # a bare frame: every coin arrives as it was laid, whatever the pair
    assert np.allclose(P.check_walks(fr, GATES), np.kron(np.eye(2), np.eye(2)))

    # Block 1.2: the folding, frame by frame.  Each fold goes through every
    # panel there is (D1: once across, the lower half up onto the upper, and
    # opened again; D2: end to end, the right half over onto the left).  The
    # swing arcs of a fold stand one in every panel there was before it.
    assert p12a.macro == "foldsteps" and not p12a.keyed and p12a.opt == ""
    assert p12a.args == ("none", "once")
    assert p12b.macro == "foldsheet" and p12b.keyed and P.options(p12b) == {"marks": "on"}
    sheets = [s for s, _ in strip] + [sh]
    assert [s.folds for s in sheets] == [0, 1, 2]
    assert [s.shape for s in sheets] == [(1, 1), (2, 1), (2, 2)]
    for before, after in zip(sheets, sheets[1:]):
        assert after.panels == 2 * before.panels          # every panel split in two
        assert len(after.swings) == before.panels         # one arc in each old panel
    assert not sheets[0].marks and sheets[0].swings == []
    once, twice = sheets[1], sheets[2]
    assert once.swing_kind == "across" and twice.swing_kind == "endtoend"
    # D1: the arc of the first fold runs up, from the lower half over the crease
    # onto the upper
    (a,) = once.swings
    assert P.up(a) and a[0][1] < P.SH / 2 < a[3][1]
    assert (once.panel_at(*a[0]), once.panel_at(*a[3])) == ((2, 1), (1, 1))
    # D2: the arcs of the second fold, one in each row, run to the left, from
    # the right half over the new crease onto the left
    assert all(P.leftward(a) and a[3][0] < P.SW / 2 < a[0][0] for a in twice.swings)
    assert sorted((twice.panel_at(*a[0]), twice.panel_at(*a[3])) for a in twice.swings) == [
        ((1, 2), (1, 1)), ((2, 2), (2, 1))]
    # the keyed sheet: blank, the names of its four panels
    assert sh.shares is None and sh.arrows == [] and sh.tallies is None and not sh.borders
    assert sh.panels == 4 == len(panels)
    assert sh.labels == {"columns": ("hemp heads", "hemp tails"),
                         "rows": ("silk heads", "silk tails")}
    assert P.panel("heads", "tails") == 1                   # the panel asked for: upper right
    # D1 and D2, where the folding is done: in the paragraph the block follows
    D1 = ("and folds it once across the middle; the sheet is no smaller folded, but the crease "
          "has scored it into an upper half and a lower half.]*",
          "and folds it once across the middle, the lower half up onto the upper, and opens it "
          "again.]*")
    D2 = ("*[She folds it again, this time end to end, and opens it flat: the outline is "
          "unchanged, yet the two creases have quartered the single face into four panels.]*",
          "*[She folds it again, this time end to end, the right half over onto the left, and "
          "opens it flat.]*")
    P.check_edits(DIALOGUE, "First", [D1, D2])
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a11, a12, a13 = DIALOGUE.anchors("First", path)
        assert a12.endswith("For the hemp: left of the crease heads, right of it tails. "
                            "Four panels. Fill them.")
        assert D1[1] in a12 and D2[1] in a12
        assert a12.index(D1[1]) < a12.index("For the silk: above the crease") < a12.index(D2[1])

    # P1.3: the patchwork -- the panels the blackboard computed, the slips its lists
    assert not p13.keyed and pw.shape == (2, 2) and not marked(pw)
    assert np.allclose(pw.shares, panels)                   # 3/8, 1/8, 3/8, 1/8
    assert np.allclose(pw.shares, [3 / 8, 1 / 8, 3 / 8, 1 / 8])
    assert np.isclose(sum(pw.shares), 1)                    # "They make one whole"
    assert pw.borders
    assert np.allclose(pw.border["silk"] + pw.border["hemp"], two_lists)
    assert not np.isclose(sum(pw.border["silk"] + pw.border["hemp"]), 1)   # the lists: two
    # the macro's patchwork is the Kronecker product of the two slips, for any words
    for sw, s in (("a half", 1 / 2), ("one in three", 1 / 3), ("nothing", 0.0), ("whole", 1.0)):
        for hw, h in (("three-quarters", 3 / 4), ("two-fifths", 2 / 5), ("one in four", 1 / 4)):
            got = P.foldsheet("twice", f"patchwork: {sw}, {hw}").shares
            assert np.allclose(got, np.kron([s, 1 - s], [h, 1 - h]))
            assert np.linalg.matrix_rank(np.reshape(got, (2, 2))) <= 1   # always a patchwork
            if 0 < s < 1:
                assert not np.allclose(got, np.kron([h, 1 - h], [s, 1 - s]))   # silk the rows
    assert P.faults(P.foldsheet, "twice", "patchwork: a half") != []

    assert P.section_dumps(found) == [
        "stringframe: start",
        "stringframe: + cells silk bare",
        "stringframe: + cells hemp bare",
        "foldsheet: none rows 1 cols 1",
        "foldsheet: once rows 2 cols 1",
        "foldsheet: + swing 1 across",
        "foldsheet: twice rows 2 cols 2",
        "foldsheet: + swing 2 endtoend",
        "foldsheet: twice rows 2 cols 2",
        "foldsheet: + shares 0.3750,0.1250,0.3750,0.1250",
        "foldsheet: + slips 0.5000,0.7500"]


# --------------------------------------------------------------------------
# The sketchbook.  Rows are wires, row 0 is the top one; `g1 @ g2` stacks g1
# above g2, `g1 + g2` puts g1 first along the wire.
# --------------------------------------------------------------------------
def basis_state(b):
    p = Picture()
    n = p.x(0, 1, b)        # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def some_state(phases):
    """A state with no name: the state 0 followed by a chain of three nodes."""
    p = Picture()
    n = p.x(0, 1, 0)
    a, b, c = phases
    vs = [p.z(0, 2, a), p.x(0, 3, b), p.z(0, 4, c)]
    p.wire(n, *vs, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def sketchbook():
    # --- stacked basis states are the column for the pair -------------------
    for a in (0, 1):
        for b in (0, 1):
            stack = basis_state(a) @ basis_state(b)
            assert np.allclose(matrix_of(stack).ravel(), ket(a, b))
    assert not same_picture(basis_state(0) @ basis_state(1),
                            basis_state(1) @ basis_state(0))    # top wire first

    # --- stacked states with no name are the Kronecker column ---------------
    ph1 = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))
    ph2 = (Fraction(1, 2), Fraction(1, 4), Fraction(1, 3))
    s1, s2 = some_state(ph1), some_state(ph2)
    m1 = matrix_of(s1).ravel()
    m2 = matrix_of(s2).ravel()
    assert not np.allclose(m1.imag, 0) and not np.allclose(m1, m2)
    assert np.allclose(matrix_of(s1 @ s2).ravel(), np.kron(m1, m2))
    assert not np.allclose(matrix_of(s1 @ s2).ravel(), np.kron(m2, m1))

    # --- two closed pictures stacked, and the same two in a row -------------
    e1, e2 = some_state(ph2).adjoint(), some_state(ph1).adjoint()
    closed1, closed2 = s1 + e1, s2 + e2
    n1 = matrix_of(closed1)[0, 0]
    n2 = matrix_of(closed2)[0, 0]
    assert not np.isclose(n1.imag, 0)                   # not real: nothing hides
    assert same_picture(closed1 @ closed2, closed1 + closed2)   # stacked = in a row
    assert np.isclose(matrix_of(closed1 @ closed2)[0, 0], n1 * n2)
    # the mix-up a bra invites: the product of the numbers, not of one conjugated
    assert not np.isclose(n1 * n2, np.conj(n1) * n2)

    # --- Proposition 4.1 as a picture equation ------------------------------
    # states stacked, then effects stacked, is the two closed pictures stacked
    assert same_picture((s1 @ s2) + (e1 @ e2), closed1 @ closed2)
    assert not same_picture((s1 @ s2) + (e2 @ e1), closed1 @ closed2)


if __name__ == "__main__":
    verdict("SecondString", blackboard, sketchbook)
