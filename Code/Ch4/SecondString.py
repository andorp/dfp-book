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
# The pictures.  The dialogue draws its props with two macros
# (pdf/props/ch04.tex; the rules in pdf/props/ch04.lua):
#   \stringframe[place]{state}{caption}   the Tortoise's frame, face on
#   \foldsheet{folds}{marks}{caption}     the sheet, folded and opened flat
# The Lua COMPUTES what a picture shows -- rows, coins, crossings, panels,
# arrows -- and tools/props_ch04.py transcribes it.  The calls are read out of
# this challenge's section of dialog.md (tools/dfp_props.py), never copied, so
# an edit to the dialogue that breaks a picture breaks here.  This first file
# also checks the dialogue as a whole: twenty pictures, and the keyed rule.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("stringframe", "foldsheet"))
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")
NOT_IN_CAPTIONS = ("strings", "wires", "wire", "diagram", "diagrams", "drawing", "drawings",
                   "picture", "pictures", "box", "boxes")
X = np.array([[0, 1], [1, 0]], dtype=complex)            # "turn it over"
S0 = np.array([[1, 1], [0, 0]], dtype=complex)           # "heads up, whatever it was"
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
GATES = {"over": X, "heads": S0, "yoke": CNOT, "cross": SWAP}


def kind(call):
    """What a keyed picture labels: the frame with its hooked strings, the sheet."""
    if call.macro == "stringframe":
        return (call.macro, P.hooked_of(call))
    return call.macro


def check_whole_dialogue():
    assert DIALOGUE.check_whole(ORDINALS) == 20                  # 3 + 3 + 3 + 4 + 3 + 4
    DIALOGUE.keyed_first(ORDINALS, kind=kind)
    placed = [(o, i, c) for o in ORDINALS
              for i, c in enumerate(c for b in DIALOGUE.blocks(o) for c in b)]
    assert [(o, i) for o, i, c in placed if c.keyed] == [("First", 0), ("First", 1),
                                                         ("Second", 1)]
    # the keyed rule is by labelling: by macro alone, P2.2 would be no first
    seen = [c.macro for o, i, c in placed[:4]]
    assert placed[4][2].keyed and placed[4][2].macro in seen
    for o, i, c in placed:
        P.picture(c)                                       # no macro ever faults


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


def pictures(panels, two_lists):
    check_whole_dialogue()
    check_colon_rule()
    p11, p12, p13 = DIALOGUE.check_layout("First", sizes=[1, 1, 1])
    for c in (p11, p12, p13):
        assert words_only(c.caption, forbidden=NOT_IN_CAPTIONS), c.caption

    # P1.1: the frame, a string below; the coins are behind the card, not drawn
    fr = P.picture(p11)
    assert p11.macro == "stringframe" and p11.keyed and fr.place == ("start", None)
    assert fr.hooked == ("silk", "hemp") and fr.ncol == 0
    assert fr.rows == {"silk": [1], "hemp": [2]}           # hemp BELOW silk
    assert fr.clips() == [] and fr.yokes() == [] and fr.crossings() == []
    assert fr.face == {"silk": None, "hemp": None} and fr.coins() == {}
    assert fr.labels == {"silk": 1, "hemp": 2}
    # a bare frame: every coin arrives as it was laid, whatever the pair
    assert np.allclose(P.check_walks(fr, GATES), np.kron(np.eye(2), np.eye(2)))

    # P1.2: the sheet folded twice, blank, keyed
    sh = P.picture(p12)
    assert p12.macro == "foldsheet" and p12.keyed
    assert sh.shape == (2, 2) and sh.panels == 4 == len(panels)
    assert sh.shares is None and sh.arrows == [] and sh.tallies is None and not sh.borders
    assert sh.labels == {"columns": ("hemp heads", "hemp tails"),
                         "rows": ("silk heads", "silk tails")}
    assert P.panel("heads", "tails") == 1                   # the panel asked for: upper right

    # P1.3: the patchwork -- the panels the blackboard computed, the slips its lists
    sh = P.picture(p13)
    assert not p13.keyed and sh.shape == (2, 2)
    assert np.allclose(sh.shares, panels)                   # 3/8, 1/8, 3/8, 1/8
    assert np.allclose(sh.shares, [3 / 8, 1 / 8, 3 / 8, 1 / 8])
    assert np.isclose(sum(sh.shares), 1)                    # "They make one whole"
    assert sh.borders
    assert np.allclose(sh.border["silk"] + sh.border["hemp"], two_lists)
    assert not np.isclose(sum(sh.border["silk"] + sh.border["hemp"]), 1)   # the lists: two
    # the macro's patchwork is the Kronecker product of the two slips, for any words
    for sw, s in (("a half", 1 / 2), ("one in three", 1 / 3), ("nothing", 0.0), ("whole", 1.0)):
        for hw, h in (("three-quarters", 3 / 4), ("two-fifths", 2 / 5), ("one in four", 1 / 4)):
            got = P.foldsheet("twice", f"patchwork: {sw}, {hw}").shares
            assert np.allclose(got, np.kron([s, 1 - s], [h, 1 - h]))
            assert np.linalg.matrix_rank(np.reshape(got, (2, 2))) <= 1   # always a patchwork
            if 0 < s < 1:
                assert not np.allclose(got, np.kron([h, 1 - h], [s, 1 - s]))   # silk the rows
    assert P.faults(P.foldsheet, "twice", "patchwork: a half") != []


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
