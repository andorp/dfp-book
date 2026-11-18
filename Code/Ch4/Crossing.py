"""Challenge 5: The Crossing
=============================
"The clip kept its string; the string changed its place."  -- Achilles

Claim: the crossing is SWAP, the matrix that exchanges the two labels of every
pair.  Its two rules: a box slides through a crossing and changes row,
SWAP (A (x) B) = (B (x) A) SWAP; and two crossings are two bare wires,
SWAP.SWAP = I.  SWAP is itself no pair of boxes: it is not A (x) B for any
A and B.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
from dfp import I2, Picture, ket, matrix_of, random_matrix, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch04 as P

X = np.array([[0, 1], [1, 0]], dtype=complex)


def reshuffle(M):
    """The 4x4 matrix rewritten so that A (x) B becomes a rank-one matrix."""
    return M.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)


def blackboard():
    rng = np.random.default_rng(45)

    # --- SWAP, read off the rule |ab> -> |ba> --------------------------------
    SWAP = np.column_stack([ket(b, a) for a in (0, 1) for b in (0, 1)])
    assert np.allclose(SWAP, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    assert np.allclose(SWAP @ ket(0, 1), ket(1, 0))
    assert np.allclose(SWAP @ ket(1, 0), ket(0, 1))
    assert np.allclose(SWAP @ ket(0, 0), ket(0, 0))
    assert np.allclose(SWAP @ ket(1, 1), ket(1, 1))
    assert np.allclose(SWAP, SWAP.real) and np.allclose(SWAP, SWAP.T)
    assert np.allclose(SWAP.conj().T @ SWAP, np.eye(4))       # a unitary box
    assert np.allclose(SWAP @ SWAP, np.eye(4))                # twice is nothing

    # --- it exchanges the factors of a product ------------------------------
    for _ in range(30):
        psi, phi = random_state(rng), random_state(rng)
        assert not np.allclose(psi.imag, 0)
        assert np.allclose(SWAP @ np.kron(psi, phi), np.kron(phi, psi))
        assert not np.allclose(SWAP @ np.kron(psi, phi), np.kron(psi, phi))
        assert np.allclose(SWAP @ SWAP @ np.kron(psi, phi), np.kron(psi, phi))
    arrow = np.array([3 / 5, 4 / 5], dtype=complex)
    omega = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    assert np.allclose(SWAP @ np.kron(arrow, omega), np.kron(omega, arrow))
    assert np.allclose(SWAP @ np.kron(arrow, omega),
                       np.array([3, 4, 3j, 4j]) / (5 * np.sqrt(2)))

    # --- a box slides through and changes row --------------------------------
    for _ in range(30):
        A, B = random_matrix(rng), random_matrix(rng)
        assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())
        assert np.allclose(SWAP @ np.kron(A, B), np.kron(B, A) @ SWAP)
        assert not np.allclose(SWAP @ np.kron(A, B), np.kron(A, B) @ SWAP)   # must change row
        assert not np.allclose(SWAP @ np.kron(A, B), np.kron(B.T, A.T) @ SWAP)
        assert not np.allclose(SWAP @ np.kron(A, B), np.kron(B, A).conj().T @ SWAP)
        assert np.allclose(SWAP @ np.kron(A, B) @ SWAP, np.kron(B, A))
        assert not np.allclose(np.kron(A, B), np.kron(B, A))
    # the only way two stacked boxes commute is one being a multiple of the other
    A = random_matrix(rng)
    lam = 1.3 - 0.7j
    assert np.allclose(np.kron(A, lam * A), np.kron(lam * A, A))

    # --- SWAP is not a pair of boxes -----------------------------------------
    for _ in range(20):
        A, B = random_matrix(rng), random_matrix(rng)
        assert np.linalg.matrix_rank(reshuffle(np.kron(A, B))) == 1
    assert np.linalg.matrix_rank(reshuffle(SWAP)) == 4
    assert np.linalg.matrix_rank(reshuffle(np.kron(X, I2))) == 1
    # the elementary argument of Sec 4.6, in numbers: the upper left block of
    # a product is a multiple of the lower right block, and SWAP's are not
    upper_left, lower_right = SWAP[:2, :2], SWAP[2:, 2:]
    assert np.allclose(upper_left, [[1, 0], [0, 0]])
    assert np.allclose(lower_right, [[0, 0], [0, 1]])
    assert np.linalg.matrix_rank(np.stack([upper_left.ravel(), lower_right.ravel()])) == 2

    # --- the dialogue's coin walks -------------------------------------------
    # crossed once: the coins change hooks, and neither changes its face
    assert np.allclose(SWAP @ ket(0, 1), ket(1, 0))
    # "turn it over" on the gut string, left of the crossing, then slid right
    before = SWAP @ np.kron(X, I2)
    after = np.kron(I2, X) @ SWAP
    assert np.allclose(before, after)
    assert np.allclose(before @ ket(0, 0), ket(0, 1))     # heads on the gut's hook
    for a in (0, 1):
        for b in (0, 1):
            assert np.allclose(before @ ket(a, b), after @ ket(a, b))
    # the same clip on the horsehair string instead
    lower_before = SWAP @ np.kron(I2, X)
    lower_after = np.kron(X, I2) @ SWAP
    assert np.allclose(lower_before, lower_after)
    assert not np.allclose(before, lower_before)
    # crossed again the same way round: every coin on its own hook, face unchanged
    assert np.allclose(SWAP @ SWAP, np.kron(I2, I2))
    for a in (0, 1):
        for b in (0, 1):
            assert np.allclose(SWAP @ SWAP @ ket(a, b), ket(a, b))

    pictures(SWAP)


# --------------------------------------------------------------------------
# The pictures (pdf/props/ch04.tex; the Lua's rules transcribed in
# tools/props_ch04.py), read out of this challenge's section of dialog.md.
# The macro computes each string's row column by column, the row a clip sits
# on after a crossing, and which string passes in front at each crossing (the
# one that was upper there).  The coins' arrival never reads that last fact:
# the string remembers which went in front, the coin cannot tell.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("stringframe", "foldsheet", "framewalk", "framesteps",
                                      "foldsteps"))
S0 = np.array([[1, 1], [0, 0]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
I4 = np.eye(4, dtype=complex)
LOWER = [3, 4]                                             # the gut's and the horsehair's rows


def faces(v):
    """The faces a basis column shows, top row first (heads = 0)."""
    (k,) = np.flatnonzero(np.abs(v) > 0.5)
    n = int(np.log2(len(v)))
    return [P.FACE[int(b)] for b in format(int(k), "0%db" % n)]


def check_exchange(fr, upper, lower, col):
    """An exchange frame: the loops of the pair's right ends trade hooks, the upper
    coming down in front, the lower going up behind; the left-hand ends held."""
    m = fr.moved
    assert fr.step == "exchange" and (m["upper"], m["lower"], m["col"]) == (upper, lower, col)
    assert (col, upper, lower) in fr.crossings()               # the upper goes in front
    assert fr.drawn_coins == [] and fr.ghost_coins == []       # the coins laid aside
    (front, a), (behind, b) = fr.exchange_arcs
    assert front and not behind
    ya, yb = P.rowy(m["rowup"]), P.rowy(m["rowdn"])
    assert {m["rowup"], m["rowdn"]} == set(LOWER)
    assert not P.up(a) and a[0][1] > a[3][1] and P.up(b)        # in front down, behind up
    assert all(x > P.L for x, y in a) and all(x < P.L for x, y in b)   # at the right-hand hooks
    assert ya > a[0][1] > a[3][1] > yb and yb < b[0][1] < b[3][1] < ya
    # the hold rings: on the pair's left-hand hooks, which never move
    assert sorted((n, r) for n, r, x, y in fr.pins) == [("gut", 3), ("horsehair", 4)]
    assert all(x == 0 and y == P.rowy(r) for n, r, x, y in fr.pins)
    assert [(n, r) for n, r, x, y in fr.pins] == [(upper, fr.rows[upper][0]),
                                                  (lower, fr.rows[lower][0])]


def pictures(SWAP):
    P.check_lua()                     # ch04.lua and ch04.tex as the transcription was proved
    GATES = {"over": X, "heads": S0, "yoke": CNOT, "cross": SWAP}
    # Four moments: the exchange beside the walk through it (5.1), the slide
    # through as Achilles expects it beside the truth (5.2), the second
    # exchange as he expects it beside the truth (5.3), the same report (5.4).
    found = P.check_section(DIALOGUE, "Fifth", sizes=[2, 2, 2, 1], frames=[4, 2, 2, 1])
    (p51x, [(fx, _)]), (p51, walk) = found[0]
    (p52e, [(fr_row, _)]), (p52, [(f2, _)]) = found[1]
    (p53e, [(fr_flat, _)]), (p53, [(fw, _)]) = found[2]
    (p54, [(f3, _)]), = found[3]
    for c in (p51x, p51, p52e, p52, p53e, p53, p54):
        assert not c.keyed
    (g0, _), (g1, _), (g2, _) = walk
    frames = (fx, g0, g1, g2, fr_row, f2, fr_flat, fw, f3)
    for fr in frames:
        assert fr.hooked == ("silk", "hemp", "gut", "horsehair") and fr.yokes() == []
        assert fr.cellwords("silk") == fr.cellwords("hemp") == ["-"] * fr.ncol
    assert [fr.ncol for fr in frames] == [3, 3, 3, 3, 3, 3, 4, 4, 4]

    # Block 5.1, the exchange (l.563): a bare pair in an empty frame crosses
    # midway, the gut's loop onto the lower hook in front, the horsehair's onto
    # the upper behind; the strip after it walks a coin on each through it
    assert P.options(p51x) == {"place": "off", "exchange": "gut and horsehair"}
    assert p51x.args[0] == "silk; hemp; gut; horsehair"       # a bare frame
    check_exchange(fx, "gut", "horsehair", 2)
    assert np.isclose(P.colx(fx.state, 2), P.L / 2)            # midway
    assert fx.cells == g0.cells                                # the crossing the walk goes through
    assert np.allclose(P.check_walks(fx, GATES), np.kron(I4, SWAP))
    assert p51.macro == "framewalk" and P.options(p51) == {"to": "end"}
    assert [g.place for g in (g0, g1, g2)] == [("start", None), ("past", 2), ("end", None)]
    # P5.1, crossed once: hooks exchanged, faces kept
    f1 = g2
    assert f1.ncol == 3 and f1.crossings() == [(2, "gut", "horsehair")]
    assert f1.rows["gut"] == [3, 3, 4, 4] and f1.rows["horsehair"] == [4, 4, 3, 3]
    assert f1.clips() == []
    M1 = P.check_walks(f1, GATES)
    assert np.allclose(M1, np.kron(I4, SWAP))
    assert not np.allclose(M1, np.eye(16))
    assert f1.coins() == {"gut": ("heads", 4), "horsehair": ("tails", 3)}
    assert np.allclose(P.column({1: "tails", 2: "heads"}, 2), SWAP @ ket(0, 1))
    # frame by frame, the model's coins on the lower pair: laid on, past the
    # crossing, at the right-hand hooks
    start = ket(0, 1)                                          # gut heads, horsehair tails
    for g, v in zip((g0, g1, g2), (start, SWAP @ start, SWAP @ start)):
        assert np.allclose(P.drawn_column(g, rows=LOWER), v)
    # the step through the crossing: each coin changes its row, not its face --
    # a ghost on its old row, as it was, and no hop
    assert sorted(faces(start)) == sorted(faces(SWAP @ start))
    assert np.allclose(P.drawn_column(g1, rows=LOWER, ghosts=True), start)
    assert [(n, f, r) for n, f, r, x, y in g1.ghost_coins] == [("gut", "heads", 3),
                                                               ("horsehair", "tails", 4)]
    assert g1.hops == [] and g0.ghost_coins == [] and g2.ghost_coins == []
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a = DIALOGUE.anchors("Fifth", path)[0]
        assert a.endswith("They have changed hooks, and neither has changed its face.")
        assert ("the heads coin goes down with the gut to the lower row, the tails coin up with "
                "the horsehair to the upper") in a

    # Block 5.2: the clip slid through -- still on the gut, now on the lower row
    assert P.options(p52) == {"place": "end", "slide": "gut first to third"}
    before = P.clean(P.stringframe, "end", p52.args[0])         # the call's state: before it
    assert before.clips() == [(1, "gut", 3, ("over",))]
    assert f2.crossings() == [(2, "gut", "horsehair")]
    assert f2.clips() == [(3, "gut", 4, ("over",))]
    M2 = P.check_walks(f2, GATES)
    assert np.allclose(M2, np.kron(I4, np.kron(I2, X) @ SWAP))
    assert np.allclose(M2, np.kron(I4, SWAP @ np.kron(X, I2)))     # slid: row changed
    assert f2.coins() == {"gut": ("tails", 4), "horsehair": ("heads", 3)}
    truth = np.kron(I2, X) @ SWAP @ ket(0, 0)                 # both heads up
    assert np.allclose(P.drawn_column(f2, rows=LOWER), truth)
    # the clip before the crossing (l.421): the same arrival, face for face
    assert np.allclose(P.check_walks(before, GATES), M2)
    assert before.coins() == f2.coins()
    for start in P.starts(f2):
        assert before.walk(3, start) == f2.walk(3, start)
    # the marks: the ghost where the clip stood, on the upper row; the arc down
    # through the crossing to the lower row, to the right
    (_, _, _, bx, by), = before.clips_drawn
    (_, _, _, nx, ny), = f2.clips_drawn
    assert f2.ghost_clip == ("gut", "over", 1, 3, bx, by)
    assert by == P.rowy(3) and ny == P.rowy(4) and bx < P.colx(f2.state, 2) < nx
    # the slide's mark follows the clip's own string through the crossing, a little
    # above it all the way (the horsehair, its partner there, starts below): the
    # clip went along the gut, not over the horsehair
    assert f2.move_arc is None and f2.slide_ceiling is None and not f2.lowered
    side, off, arc = f2.follow_side, f2.follow_off, f2.follow_arc
    assert (side, off) == (1, P.SLIDEFOLLOWUP)
    assert np.isclose(arc[0][0], bx + P.SLIDESIDE) and np.isclose(arc[-1][0], nx - P.SLIDESIDE)
    assert all(p[0] < q[0] for p, q in zip(arc, arc[1:]))       # left to right
    gut = P.follow_centre(f2.state, f2.rows, "gut")
    hh = P.follow_centre(f2.state, f2.rows, "horsehair")
    pts = P.densify(arc)
    for q in pts:
        d = P.polyline_dist(q, gut)
        assert q[1] > P.polyline_y(gut, q[0])                    # on its side, all along
        gap = d - P.STROKE["gut"] / 2 - P.SWING / 2
        assert 0.02 <= gap < 0.05                                # clear of the gut's stroke,
        assert off - 0.003 < d < off + 0.003                     # and close beside it
    # it meets the horsehair once, inside the crossing's column
    above = [q[1] > P.polyline_y(hh, q[0]) for q in pts]
    meets = [i for i in range(1, len(pts)) if above[i] != above[i - 1]]
    w = min(0.2, 0.46 * P.colstep(f2.state))
    assert len(meets) == 1 and abs(pts[meets[0]][0] - P.colx(f2.state, 2)) < w
    # no lift's hump: never higher than its offset above the gut where it starts
    top = max(y for _, y in arc)
    assert top <= P.rowy(3) + off + 0.001
    assert top < by + P.CLIPTOP + 0.03 + 0.75 * 0.07             # below the old slide arc's apex
    # the macro raises no new fault on this frame; its guard needs a partner, which
    # every crossing has
    assert P.faults(P.stringframe, "end;slide=gut first to third",
                    p52.args[0]) == []
    # the prediction beside it (l.579): the clip turns the coin of the string
    # hooked on its row at the left upright -- the horsehair's, which after the crossing
    # lies on the upper row
    assert P.options(p52e) == {"place": "end", "expects": "row"} and fr_row.expects == "row"
    assert p52e.args[0] == P.serialize(f2.state)               # the frame after the slide
    assert fr_row.cells == f2.cells and fr_row.moved is None   # drawn as it is, no marks
    assert fr_row.coinframe.cellwords("horsehair") == ["-", "cross", "over"]
    assert fr_row.coinframe.cellwords("gut") == ["-", "cross", "-"]
    expected = np.kron(X, I2) @ SWAP @ ket(0, 0)
    assert np.allclose(P.drawn_column(fr_row, rows=LOWER), expected)
    assert fr_row.coins() == {"gut": ("heads", 4), "horsehair": ("tails", 3)}
    assert not np.allclose(expected, truth)                     # refuted
    assert fr_row.coins() != f2.coins()
    # Achilles's guess as a frame of its own: the clip moved to the horsehair
    guess = P.stringframe("end", "silk; hemp; gut heads: -, cross, -; "
                                 "horsehair heads: -, cross, over")
    assert guess.clips() == [(3, "horsehair", 3, ("over",))]
    assert guess.coins() == fr_row.coins() != f2.coins()
    assert not np.allclose(P.check_walks(guess, GATES), M2)
    # the rule predicts nothing where a clip is on its own row's string
    assert P.faults(P.stringframe, "end;expects=row", "silk; hemp; gut heads: over, cross, -; "
                    "horsehair heads: -, cross, -") == [
        "stringframe: expects=row: every clip is on its own row's string; the rule predicts what "
        "happens"]
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        assert DIALOGUE.anchors("Fifth", path)[1].endswith(
            "The clip kept its string; the string changed its place.")

    # Block 5.3: the same hand-movement again (l.593) -- this time the horsehair
    # in front, the left-hand ends held, the right ends home, and the pair wound
    assert P.options(p53) == {"place": "off", "exchange": "gut and horsehair"}
    assert p53.args[0] == "silk; hemp; gut: -, cross, -; horsehair: -, cross, -"   # crossed once
    check_exchange(fw, "horsehair", "gut", 3)
    assert fw.crossings() == [(2, "gut", "horsehair"), (3, "horsehair", "gut")]
    assert fw.rows["gut"][-1] == 3 and fw.rows["horsehair"][-1] == 4   # own hooks
    Mw = P.check_walks(fw, GATES)
    assert np.allclose(Mw, np.kron(I4, SWAP @ SWAP)) and np.allclose(Mw, np.eye(16))
    # D9: the two crossings lie close together about the middle
    x2, x3 = P.colx(fw.state, 2), P.colx(fw.state, 3)
    assert x2 < P.L / 2 < x3 and np.isclose(x3 - x2, P.colstep(fw.state))
    # the prediction beside it (l.597, "I said what you said and did what you
    # did!"): the same movement made again undoes the first, so the pair lies
    # straight -- the wound frame with its two crossings taken out.  The coins
    # cannot tell the two apart; the strings can.
    assert P.options(p53e) == {"place": "off", "expects": "flat"} and fr_flat.expects == "flat"
    assert p53e.args[0] == P.serialize(fw.state)               # the true frame, as Achilles reads it
    assert fr_flat.crossings() == [] and fr_flat.clips() == []
    assert all(c == P.NONE for n in fr_flat.hooked for c in fr_flat.cells[n])
    straightened = {n: tuple(P.NONE if c.kind == "cross" else c for c in fw.cells[n])
                    for n in fw.hooked}
    assert fr_flat.cells == straightened
    assert fr_flat.cells != fw.cells                           # refuted: in the strings only
    assert fr_flat.drawn_coins == [] == fw.drawn_coins
    assert np.allclose(P.check_walks(fr_flat, GATES), Mw)      # the coins report the same
    assert P.faults(P.stringframe, "off;expects=flat", p53.args[0]) == [
        "stringframe: expects=flat needs two crossings of one pair one after the other"]
    D8 = ("*[The lower pair still lies crossed once:",
          "*[Achilles lifts the two coins off the lower pair's right-hand hooks and lays them "
          "aside with the others. The lower pair still lies crossed once:")
    D9 = ("The right ends come home to their own hooks, and still the pair will not lie flat.]*",
          "The right ends come home to their own hooks, and still the pair will not lie flat. "
          "The two crossings lie close together about the middle.]*")
    P.check_edits(DIALOGUE, "Fifth", [D8, D9])
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        at8, blocks = P.where(DIALOGUE, "Fifth", D8[1], path)
        at9, _ = P.where(DIALOGUE, "Fifth", D9[1], path)
        assert blocks[1] < at8 <= at9 < blocks[2]               # after the slide, before 5.3
        para = DIALOGUE.paragraphs("Fifth", path)[at9]
        assert "this time the horsehair passes in front and the gut behind" in para
        assert DIALOGUE.anchors("Fifth", path)[2].endswith(
            "It won't pull out. And I said what you said and did what you did!")

    # P5.4 (the old P5.3): crossed again the same way round -- wound, each string
    # in front once, the same report as a straight pair
    assert f3.ncol == 4 and f3.place == ("end", None) and f3.moved is None
    assert f3.crossings() == [(2, "gut", "horsehair"), (3, "horsehair", "gut")]
    assert f3.cells == fw.cells                                # the wound frame, walked
    assert sorted(front for col, front, behind in f3.crossings()) == ["gut", "horsehair"]
    assert f3.rows["gut"][-1] == 3 and f3.rows["horsehair"][-1] == 4   # own hooks
    M3 = P.check_walks(f3, GATES)
    assert np.allclose(M3, np.kron(I4, SWAP @ SWAP)) and np.allclose(M3, np.eye(16))
    assert f3.coins() == {"silk": ("heads", 1), "hemp": ("tails", 2),
                          "gut": ("heads", 3), "horsehair": ("tails", 4)}
    # the same report as the straight pair: every coin home with its face
    for n, (face, row) in f3.coins().items():
        assert face == f3.face[n] and row == P.HOOKROW[n]
    # and the coin cannot tell wound from straight: the walk never reads the front
    straight = P.stringframe("end", "silk heads; hemp tails; gut heads; horsehair tails")
    assert straight.crossings() == []
    for start in P.starts(f3):
        assert f3.coins(start) == straight.coins(start)
    assert np.allclose(P.check_walks(straight, GATES), M3)
    # the macro refuses a walk strip of the slid-through frame: no coin fits
    # between the crossing and the clip
    assert any("too crowded to walk" in e for e in
               P.faults(P.run_call, "framewalk", False, "to=end", (p52.args[0],),
                        "laid on; past the clip; past the crossing; at the hooks"))

    # the macro's own faults: a crossing needs its partner on the next row, and
    # an exchange a pair whose right ends are neighbours
    assert P.faults(P.stringframe, "end", "gut: cross; horsehair") != []
    assert P.faults(P.stringframe, "end", "silk: cross; gut: cross") != []
    assert P.faults(P.stringframe, "off;exchange=silk and gut", "silk; hemp; gut") == [
        "stringframe: exchange: the right-hand ends of the silk and the gut are not on "
        "neighbouring rows"]

    assert P.section_dumps(found) == [
        "stringframe: off exchange",
        "stringframe: + cells silk -,-,-",
        "stringframe: + cells hemp -,-,-",
        "stringframe: + cells gut -,cross,-",
        "stringframe: + cells horsehair -,cross,-",
        "stringframe: + front gut col 2",
        "stringframe: + exchange gut horsehair col 2",
        "stringframe: + pin hook gut row 3",
        "stringframe: + pin hook horsehair row 4",
        "stringframe: start",
        "stringframe: + coin gut heads row 3",
        "stringframe: + coin horsehair tails row 4",
        "stringframe: + cells silk -,-,-",
        "stringframe: + cells hemp -,-,-",
        "stringframe: + cells gut -,cross,-",
        "stringframe: + cells horsehair -,cross,-",
        "stringframe: + front gut col 2",
        "stringframe: past-second",
        "stringframe: + coin gut heads row 4",
        "stringframe: + coin horsehair tails row 3",
        "stringframe: + cells silk -,-,-",
        "stringframe: + cells hemp -,-,-",
        "stringframe: + cells gut -,cross,-",
        "stringframe: + cells horsehair -,cross,-",
        "stringframe: + front gut col 2",
        "stringframe: + ghost coin gut heads row 3",
        "stringframe: + ghost coin horsehair tails row 4",
        "stringframe: end",
        "stringframe: + coin gut heads row 4",
        "stringframe: + coin horsehair tails row 3",
        "stringframe: + cells silk -,-,-",
        "stringframe: + cells hemp -,-,-",
        "stringframe: + cells gut -,cross,-",
        "stringframe: + cells horsehair -,cross,-",
        "stringframe: + front gut col 2",
        "stringframe: end expects row",
        "stringframe: + coin gut heads row 4",
        "stringframe: + coin horsehair tails row 3",
        "stringframe: + cells silk -,-,-",
        "stringframe: + cells hemp -,-,-",
        "stringframe: + cells gut -,cross,over",
        "stringframe: + cells horsehair -,cross,-",
        "stringframe: + front gut col 2",
        "stringframe: end slide",
        "stringframe: + coin gut tails row 4",
        "stringframe: + coin horsehair heads row 3",
        "stringframe: + cells silk -,-,-",
        "stringframe: + cells hemp -,-,-",
        "stringframe: + cells gut -,cross,over",
        "stringframe: + cells horsehair -,cross,-",
        "stringframe: + front gut col 2",
        "stringframe: + slid gut 1 to 3",
        "stringframe: + ghost clip gut over col 1 row 3",
        "stringframe: off expects flat",
        "stringframe: + cells silk -,-,-,-",
        "stringframe: + cells hemp -,-,-,-",
        "stringframe: + cells gut -,-,-,-",
        "stringframe: + cells horsehair -,-,-,-",
        "stringframe: off exchange",
        "stringframe: + cells silk -,-,-,-",
        "stringframe: + cells hemp -,-,-,-",
        "stringframe: + cells gut -,cross,cross,-",
        "stringframe: + cells horsehair -,cross,cross,-",
        "stringframe: + front gut col 2",
        "stringframe: + front horsehair col 3",
        "stringframe: + exchange horsehair gut col 3",
        "stringframe: + pin hook horsehair row 4",
        "stringframe: + pin hook gut row 3",
        "stringframe: end",
        "stringframe: + coin silk heads row 1",
        "stringframe: + coin hemp tails row 2",
        "stringframe: + coin gut heads row 3",
        "stringframe: + coin horsehair tails row 4",
        "stringframe: + cells silk -,-,-,-",
        "stringframe: + cells hemp -,-,-,-",
        "stringframe: + cells gut -,cross,cross,-",
        "stringframe: + cells horsehair -,cross,cross,-",
        "stringframe: + front gut col 2",
        "stringframe: + front horsehair col 3"]


# --------------------------------------------------------------------------
# The sketchbook.  Rows are wires, row 0 is the top one; `g1 @ g2` stacks g1
# above g2, `g1 + g2` puts g1 first along the wire.  A crossing is drawn by
# wiring the upper input to the lower output and the lower input to the upper:
# flat, with no node at all, so the drawing records which output each input
# reaches and nothing else.  "SWAP is no pair of boxes" is a property of one
# matrix and stays on the blackboard.
# --------------------------------------------------------------------------
def crossing():
    p = Picture()
    i0, i1 = p.inp(0), p.inp(1)
    o0, o1 = p.out(0), p.out(1)
    p.wire(i0, o1)
    p.wire(i1, o0)
    return p.done()


def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def box(phases):
    p = Picture()
    a, b, c = phases
    # PyZX draws this with the spiders you meet in Chapter 8.
    vs = [p.z(0, 1, a), p.x(0, 2, b), p.z(0, 3, c)]
    p.wire(p.inp(0), *vs, p.out(0))
    return p.done()


def not_box():
    p = Picture()
    n = p.x(0, 1, 1)
    p.wire(p.inp(0), n, p.out(0))
    return p.done()


def basis(b):
    p = Picture()
    n = p.x(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def state(phases):
    p = Picture()
    n = p.x(0, 1, 0)
    a, b, c = phases
    vs = [p.z(0, 2, a), p.x(0, 3, b), p.z(0, 4, c)]
    p.wire(n, *vs, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def sketchbook():
    S, W = crossing(), bare_wire()
    assert np.allclose(matrix_of(S),
                       [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    # --- two states stacked, crossed, are the two states the other way round -
    s1 = state((Fraction(1, 4), Fraction(1, 3), Fraction(3, 4)))
    s2 = state((Fraction(1, 2), Fraction(1, 4), Fraction(1, 3)))
    assert not np.allclose(matrix_of(s1).ravel(), matrix_of(s2).ravel())
    assert same_picture((s1 @ s2) + S, s2 @ s1)
    assert not same_picture((s1 @ s2) + S, s1 @ s2)
    for a in (0, 1):
        for b in (0, 1):
            assert same_picture((basis(a) @ basis(b)) + S, basis(b) @ basis(a))

    # --- a box slides through a crossing and changes row ---------------------
    A = box((Fraction(1, 4), Fraction(1, 3), Fraction(3, 4)))
    Bx = box((Fraction(1, 2), Fraction(1, 4), Fraction(1, 3)))
    assert same_picture((A @ Bx) + S, S + (Bx @ A))
    assert not same_picture((A @ Bx) + S, S + (A @ Bx))     # it must change row
    assert not same_picture((A @ Bx) + S, S + (Bx @ Bx))
    # the clip on either string, left of the crossing and right of it
    assert same_picture((A @ W) + S, S + (W @ A))
    assert same_picture((W @ A) + S, S + (A @ W))
    assert not same_picture((A @ W) + S, S + (A @ W))
    N = not_box()
    assert same_picture((N @ W) + S, S + (W @ N))           # the gut clip, slid through
    assert same_picture((basis(0) @ basis(0)) + (N @ W) + S, basis(0) @ basis(1))

    # --- two crossings are two bare wires ------------------------------------
    assert same_picture(S + S, W @ W)
    assert not same_picture(S, W @ W)                       # one crossing is not
    assert same_picture(S + S + S, S)
    # and a box may be slid through both, arriving on the row it set out from
    assert same_picture((A @ Bx) + S + S, A @ Bx)


if __name__ == "__main__":
    verdict("Crossing", blackboard, sketchbook)
