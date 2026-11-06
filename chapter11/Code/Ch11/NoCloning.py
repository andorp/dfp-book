"""Challenge II: One Loop, Two Loops  (no-cloning, unitarity form; The Theorem, Twice)
===================================================================================
"A half against a quarter."  -- Achilles

Claim: for any unitary U on (stone, blank, apparatus),
<U(phi 0 A), U(psi 0 A)> = <phi|psi> = s; if U cloned both states, the same number
would be s^2 a with a = <A_phi|A_psi>, |a| <= 1, so s = s^2 a, impossible for
0 < |s| < 1.  In pictures: a box meeting its mirror leaves bare wires, so the likeness
is ONE loop; faithful copies close into TWO separate loops, which multiply.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import I2, Picture, kron, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch11 as PC
import props_cloth as CLOTH


DELTA = np.outer(ket(0, 0), ket(0)) + np.outer(ket(1, 1), ket(1))
PLUS = (ket(0) + ket(1)) / np.sqrt(2)


def random_unitary(rng, d):
    """A random unitary (QR of a complex Gaussian), generically non-real."""
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def cnot(n, c, t):
    """CNOT on n wires, control c, target t (wire 0 is the TOP wire)."""
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for k in range(2 ** n):
        bits = [(k >> (n - 1 - i)) & 1 for i in range(n)]
        if bits[c]:
            bits[t] ^= 1
        M[int("".join(map(str, bits)), 2), k] = 1
    return M


def blackboard():
    rng = np.random.default_rng(111)
    blank = ket(0)

    # the headline: a random 8x8 unitary on (stone, blank, apparatus), non-real states
    U = random_unitary(rng, 8)
    assert np.allclose(U.conj().T @ U, np.eye(8)) and not np.allclose(U, U.conj())
    psi, phi, A = random_state(rng), random_state(rng), random_state(rng)
    for v in (psi, phi, A):
        assert not np.allclose(v, v.conj())                  # non-real: the mirror conjugates
    s = np.vdot(phi, psi)                                    # the likeness <phi|psi>
    assert not np.isclose(s.imag, 0)
    x, y = kron(phi, blank, A).ravel(), kron(psi, blank, A).ravel()
    # left: the inputs' likeness is ONE loop, times the blank's loop and the apparatus's, each 1
    assert np.isclose(np.vdot(blank, blank), 1) and np.isclose(np.vdot(A, A), 1)
    assert np.isclose(np.vdot(x, y), s * 1 * 1)
    # unitarity keeps it: <U x, U y> = <x|y> = s  (the press "loses nothing")
    assert np.isclose(np.vdot(U @ x, U @ y), s)
    # the mirror conjugates: the unconjugated pairing is a different number
    assert not np.isclose(np.dot(phi, psi), s)

    # right: were both cloned, the same number would be s * s * a, |a| <= 1
    assert 0 < abs(s) < 1 and abs(s) > abs(s) ** 2
    for _ in range(200):
        Aphi, Apsi = random_state(rng, 4), random_state(rng, 4)       # any apparatus finals
        a = np.vdot(Aphi, Apsi)
        assert abs(a) <= 1 + 1e-12                                     # Cauchy-Schwarz
        assert np.isclose(np.vdot(kron(phi, phi, Aphi).ravel(), kron(psi, psi, Apsi).ravel()),
                          s * s * a)                                   # two loops and a
        assert abs(s * s * a) <= abs(s) ** 2 < abs(s)                  # so never equal to s
        assert not np.isclose(s * s * a, s)
    # "one loop equals two only when the loop is worth all or nothing"
    for t in (0, 1, 0.5, 1 / np.sqrt(2), 0.3 + 0.4j, np.exp(0.9j)):
        assert np.isclose(t, t * t) == (np.isclose(t, 0) or np.isclose(t, 1))
    # a loop is "a whole if the very same, nothing at all if wholly unalike"
    assert np.isclose(np.vdot(psi, psi), 1) and np.isclose(np.vdot(ket(0), ket(1)), 0)

    # Achilles's numbers: a plain-low stone and a crosswise one.  Asked the plain
    # question the crosswise stone says low half the time: "half alike" (a probability,
    # the likeness squared; the amplitude is 1/sqrt2).  After faithful copying: a quarter.
    s0 = np.vdot(ket(0), PLUS)
    assert np.isclose(s0, 1 / np.sqrt(2))
    assert np.isclose(abs(s0) ** 2, 0.5)                                      # a half
    assert np.isclose(abs(np.vdot(kron(ket(0), ket(0)).ravel(), kron(PLUS, PLUS).ravel())) ** 2, 0.25)
    for _ in range(50):                                                       # the works' loop
        a = np.vdot(random_state(rng, 4), random_state(rng, 4))              # only shrinks it
        assert abs(s0 * s0 * a) ** 2 <= 0.25 + 1e-12
    # ... while the press, which loses nothing, keeps the half it was given
    assert np.isclose(abs(np.vdot(U @ kron(ket(0), blank, A).ravel(),
                                  U @ kron(PLUS, blank, A).ravel())) ** 2, 0.5)

    # linearity alone allows L|0> = |00>, L|+> = |++>; but L stretches |1> out of the
    # unit sphere: ||L|1>||^2 = 3 - sqrt2 (Section 11.2), so L is no physical process
    L = np.column_stack([ket(0, 0), np.sqrt(2) * kron(PLUS, PLUS).ravel() - ket(0, 0)])
    assert np.allclose(L @ ket(0), ket(0, 0)) and np.allclose(L @ PLUS, kron(PLUS, PLUS).ravel())
    assert np.isclose(np.linalg.norm(L @ ket(1)) ** 2, 3 - np.sqrt(2))
    assert not np.allclose(L.conj().T @ L, I2)

    # the basis cloner V = CNOT_{0->2} CNOT_{0->1}: on |0>, |1> it clones, and s = s^2 a
    # holds trivially (s = 0); it does not clone |+>
    V = cnot(3, 0, 2) @ cnot(3, 0, 1)
    out0, out1 = V @ ket(0, 0, 0), V @ ket(1, 0, 0)
    s01 = np.vdot(ket(0), ket(1))
    a01 = np.vdot(ket(0), ket(1))                            # apparatus finals |0>, |1>
    assert np.isclose(np.vdot(out0, out1), s01) and np.isclose(s01, s01 ** 2 * a01)
    outp = V @ kron(PLUS, blank, ket(0)).ravel()
    on_clones = kron(kron(PLUS, PLUS).conj().reshape(1, 4), I2) @ outp   # its part on |++>
    assert np.isclose(np.linalg.norm(on_clones), 0.5)        # a clone would have norm 1 here

    # the worked instance: the copier is an isometry, delta^dagger delta = I; not unitary
    assert np.allclose(DELTA.conj().T @ DELTA, I2)
    assert not np.allclose(DELTA @ DELTA.conj().T, np.eye(4))
    assert np.isclose(np.vdot(DELTA @ phi, DELTA @ psi), s)  # it keeps likeness

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the Tortoise's likeness loop, keyed: a
# red point (a stone settled plain-low), the card, the mirror card, the mirror of a
# green point (a stone settled crosswise), closed on itself; then the cards slid
# face to face and lifted off, the loop and its points as before; then, a
# close-up beside it, the same two stones after a faithful pressing: two separate
# loops and a small plain loop,
#   \copycloth[operations]{the likeness loop}{caption}   (pdf/props/ch11.tex)
# tools/props_ch11.py transcribes the Lua (with L1-L5); the calls are read out of
# dialog.md, never copied.
#
# The ties (identity card, section 4), a structural check against this file's
# blackboard, not a second proof.  The card is the press with its blank and works
# plugged in, the isometry V = U( . (x) |0> (x) |A>); the mirror card is V^dagger,
# threaded facing it output to output, so slid together they are V^dagger V = I --
# the bare loop -- while the other way round, V V^dagger, is a projector and would
# not come off.  A loop closing a point on a mirrored point is the likeness
# s = <phi|psi>; the pressing's two separate loops multiply, s^2, and the small plain
# loop is the works' likeness, a number of modulus at most 1 (comment only).
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = "A half against a quarter."
LOOP = "the likeness loop"


def cloth_loops(S):
    """The closed loops of a cloth scene, left to right, each the list of (colour, x) of
    the rings threaded on it, left to right."""
    out = []
    for d in S["cords"]:
        if d["kind"] == "cord" and d.get("closed"):
            on = sorted((r["x"], r["c"]) for r in S["things"]
                        if CLOTH._onpath(d["pts"], r["x"], r["y"], True))
            out.append((min(p[0] for p in d["pts"]), [(c, x) for x, c in on]))
    return [rings for _, rings in sorted(out)]


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    calls = DIALOGUE.check_layout("Second", sizes=[1, 1, 1])
    p21, p22, p23 = calls
    text, raw = DIALOGUE.read(DIALOGUE.dialog), DIALOGUE.read(DIALOGUE.raw)
    spoken = PC.spoken_sentences(text) | PC.spoken_sentences(raw)
    for c in calls:
        assert words_only(c.caption, forbidden=PC.CAPTION_FORBIDDEN), c.caption
        assert CLOTH.caption_ok(c.caption), c.caption
        assert PC.normal_sentence(c.caption) not in spoken, c.caption   # the gigue: never said twice
        assert PC.place_ok("Second", c), c.text
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Second")] == [[1.24], [1.24], [1.24]]
    assert [c.macro for c in calls] == ["copycloth"] * 3
    assert [c.keyed for c in calls] == [True, False, False]
    assert [(c.opt, c.args[0]) for c in calls] == [
        ("", LOOP), ("slide the cards together", LOOP), ("after a faithful pressing", LOOP)]
    assert [c.caption for c in calls] == [
        "the press and its mirror in the way", "the cards gone, the points still on the loop",
        "side by side, not touching"]
    c21, c22, c23 = (PC.picture(c) for c in calls)
    a1, a2, a3 = DIALOGUE.anchors("Second")
    f1, f2, f3 = DIALOGUE.following("Second")
    section = DIALOGUE.section("Second")
    assert QUOTE in section                                       # the docstring's line
    assert "gathers the bend off the cloth" in section            # D5: the last layout put away

    # P2.1: one closed loop, keyed; along the top the red point, the card, the mirror
    # card, the mirrored green point; the loop's bends beyond the points, never between
    # the cards; the card faces the mirror card (the mirror card to its right, reversed)
    assert c21.labels == ("point", "card", "mirror card")
    assert c21.dump == ("scene=loop cards=true pressed=false | cols=8 rows=2 y=0..1 near=false "
                        "folded=false; 1:ring red 0 @1,0; 2:ring green 0 @6,0; "
                        "cord nil-nil closed [0,0 7,0 7,1 0,1] | card @3,0 | mirror card @4,0")
    (one,) = cloth_loops(c21.cloth)
    assert one == [("red", 1), ("green", 6)]
    (card, mirror) = c21.cards
    assert card[0] == "card" and mirror[0] == "mirror card"
    assert one[0][1] < card[1] < mirror[1] < one[1][1]            # point, card, mirror card, point
    assert card[2] == mirror[2] == 0 and card[3] < mirror[3]      # on the top run, the card left
    xs = [p[0] for p in c21.cloth["cords"][0]["pts"]]
    assert min(xs) < one[0][1] and max(xs) > one[1][1]            # the turns lie beyond the points
    before = text[:text.index(p21.text)].lower()
    for w in c21.labels:
        assert w in before, w                                     # named before it is keyed
    assert "the two cards between them" in a1
    assert f1.startswith("**Tortoise:** That is how alike")
    # P2.2: the cards slid face to face and lifted off: the same loop, the same points
    assert c22.dump == c21.dump.split(" | card")[0].replace("cards=true", "cards=false")
    assert cloth_loops(c22.cloth) == [one] and c22.cards == ()
    assert "still face to face" in a2                             # D6
    assert f2.startswith("**Tortoise:** One loop.")
    # P2.3: two separate loops, not touching, each a red point then a mirrored green one,
    # and a little way off a small plain loop
    ls = cloth_loops(c23.cloth)
    assert [[c for c, _ in L] for L in ls] == [["red", "green"], ["red", "green"], []]
    boxes = [(min(p[0] for p in d["pts"]), max(p[0] for p in d["pts"])) for d in c23.cloth["cords"]]
    boxes.sort()
    assert all(boxes[i][1] < boxes[i + 1][0] for i in range(2))  # side by side, not touching
    assert c23.cards == () and c23.frame == c21.frame == 1.24
    assert "a small plain loop" in a3
    assert f3.startswith("**Tortoise:** And that little one")
    assert PC.faults(PC.copycloth, "after a faithful pressing, slide the cards together", LOOP) == \
        ["loop-slide-pressed"]                                    # no cards on the pressed loops

    # the ties.  The press card with its blank and works plugged in is an isometry;
    # face to face (V^dagger V) it is the bare string, the other way round not.
    rng = np.random.default_rng(1102)
    U = random_unitary(rng, 8)
    A = random_state(rng)
    assert not np.allclose(U, U.conj()) and not np.allclose(A, A.conj())
    V = U @ np.kron(np.eye(2), kron(ket(0), A).reshape(4, 1))    # 8 x 2: stone in, all out
    assert np.allclose(V.conj().T @ V, I2)                        # cards off: bare string
    P8 = V @ V.conj().T
    assert not np.allclose(P8, np.eye(8)) and np.allclose(P8 @ P8, P8)
    assert np.linalg.matrix_rank(P8) == 2                         # the other way: a projector
    # the loop's value: s to the power of the loops that close a point on a mirrored
    # point.  One loop with the cards (and without them): s; the pressing: s^2.
    psi, phi = random_state(rng), random_state(rng)
    s = np.vdot(phi, psi)
    assert not np.isclose(s.imag, 0) and 0 < abs(s) < 1
    n = lambda pic: sum(1 for L in cloth_loops(pic.cloth) if L)
    assert (n(c21), n(c22), n(c23)) == (1, 1, 2)
    assert np.isclose(np.vdot(V @ phi, V @ psi), s ** n(c21))    # the cards in the way
    assert np.isclose(np.vdot(phi, psi), s ** n(c22))             # the cards off
    assert np.isclose(np.vdot(np.kron(phi, phi), np.kron(psi, psi)), s ** n(c23))   # pressed
    assert not np.isclose(s ** n(c23), s)                         # two loops are not one
    # the drawn stones: a red point (plain-low) and a green point (crosswise), 1/sqrt2
    s0 = np.vdot(ket(0), PLUS)
    assert np.isclose(abs(s0 ** n(c22)) ** 2, 0.5) and np.isclose(abs(s0 ** n(c23)) ** 2, 0.25)


# --- pictures -----------------------------------------------------------------
# A one-legged green dot of phase t*pi is |0> + e^{i t pi}|1> = sqrt2 times a unit state;
# as an effect it is the mirror, <0| + e^{i t pi}<1|.  The mirror of the point t is the
# effect -t.  Each closed loop point--effect carries 2, hence add_power(-2).

def copier_meets_mirror():
    """delta then delta^dagger: the copier Z(1->2) and its mirror Z(2->1), joined by two
    edges.  PyZX keeps one edge per pair of dots, so the second edge runs through a
    phase-0 green identity dot."""
    g = zx.Graph()
    i, o = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 0, 4)
    c, m = g.add_vertex(VT.Z, 0, 1), g.add_vertex(VT.Z, 0, 3)
    mid = g.add_vertex(VT.Z, 1, 2)                           # identity dot on the second edge
    g.add_edge((i, c)); g.add_edge((c, m)); g.add_edge((c, mid)); g.add_edge((mid, m))
    g.add_edge((m, o))
    g.set_inputs((i,)); g.set_outputs((o,))
    return g


def bare_wires(n):
    p = Picture()
    for r in range(n):
        p.wire(p.inp(r), p.out(r))
    return p.done()


def one_loop(point, effect, colour=VT.Z):
    """point -> copier -> (two edges) -> mirror copier -> effect: <phi| delta^dag delta |psi>.
    Closed.  add_power(-2) for the point's and the effect's 1/sqrt2."""
    g = zx.Graph()
    p = g.add_vertex(colour, 0, 0, phase=Fraction(point))
    c, m = g.add_vertex(VT.Z, 0, 1), g.add_vertex(VT.Z, 0, 3)
    mid = g.add_vertex(VT.Z, 1, 2)
    e = g.add_vertex(colour, 0, 4, phase=Fraction(effect))
    g.add_edge((p, c)); g.add_edge((c, m)); g.add_edge((c, mid)); g.add_edge((mid, m))
    g.add_edge((m, e))
    g.set_inputs(()); g.set_outputs(())
    g.scalar.add_power(-2)
    return g


def loops(point, effect, k, colour=VT.Z):
    """k separate bare likeness loops point -- effect, side by side.  add_power(-2) each."""
    g = zx.Graph()
    for r in range(k):
        p = g.add_vertex(colour, r, 0, phase=Fraction(point))
        e = g.add_vertex(colour, r, 1, phase=Fraction(effect))
        g.add_edge((p, e))
    g.set_inputs(()); g.set_outputs(())
    g.scalar.add_power(-2 * k)
    return g


def value(g):
    return matrix_of(g).ravel()[0]                           # closed: through matrix_of only


def chain_state(phases):
    """A generic one-wire state: green point a, red b, green c (units of pi), normalized
    by its own norm (add_float) so that it is a unit state.  Returns (graph, vector)."""
    a, b, c = (Fraction(t) for t in phases)
    g = zx.Graph()
    v = [g.add_vertex(VT.Z, 0, 0, phase=a), g.add_vertex(VT.X, 0, 1, phase=b),
         g.add_vertex(VT.Z, 0, 2, phase=c)]
    o = g.add_vertex(VT.BOUNDARY, 0, 3)
    g.add_edge((v[0], v[1])); g.add_edge((v[1], v[2])); g.add_edge((v[2], o))
    g.set_inputs(()); g.set_outputs((o,))
    n = np.linalg.norm(matrix_of(g))
    g.scalar.add_float(1 / n)
    return g, matrix_of(g).ravel()


def sketchbook():
    # (a) the copier meets its mirror and they leave a bare wire -- exact, and DERIVED
    # (fusion along the two edges, then the self-loop removed)
    g = copier_meets_mirror()
    assert np.allclose(matrix_of(g), DELTA.conj().T @ DELTA)
    assert same_picture(g, bare_wires(1))
    h = g.copy(); simp.full_reduce(h)
    assert h.num_vertices() == 2 and h.num_edges() == 1      # a bare wire, in to out
    assert same_picture(h, bare_wires(1))

    # (a') EVERY cloner box, not only delta: a random Clifford+T box U on three wires
    # (stone, blank, apparatus) followed by its mirror U^dagger reduces to three bare
    # wires, scalar exactly 1 -- the unitarity picture the headline's left side uses
    U = zx.generate.cliffordT(3, 40, p_t=0.2, seed=11)
    UU = U.copy(); UU.compose(U.adjoint())
    assert np.allclose(matrix_of(UU), np.eye(8))
    assert not np.allclose(matrix_of(U), matrix_of(U).conj())    # a non-real box
    h = UU.copy(); simp.full_reduce(h)
    assert h.num_vertices() == 6 and h.num_edges() == 3      # three boundary-to-boundary edges
    assert np.isclose(h.scalar.to_number(), 1)
    assert same_picture(h, bare_wires(3))

    # the headline picture itself: states psi, 0, A; then U, U^dagger; then the effects
    # phi, 0, A (the mirrors of the states).  Its value is s: one loop that matters.
    gpsi, psi = chain_state((Fraction(1, 4), Fraction(1, 3), Fraction(3, 5)))
    gphi, phi = chain_state((Fraction(2, 7), Fraction(3, 4), Fraction(1, 6)))
    gA, A = chain_state((Fraction(1, 5), Fraction(2, 3), Fraction(1, 8)))
    for v in (psi, phi, A):
        assert not np.allclose(v, v.conj())
    s = np.vdot(phi, psi)
    assert not np.isclose(s.imag, 0) and 0 < abs(s) < 1

    def blank():
        g = zx.Graph()
        p, o = g.add_vertex(VT.X, 0, 0), g.add_vertex(VT.BOUNDARY, 0, 1)
        g.add_edge((p, o)); g.set_inputs(()); g.set_outputs((o,))
        g.scalar.add_power(-1)                               # red point = sqrt2 |0>
        return g

    def inputs(gs):
        return gs.tensor(blank()).tensor(gA)                 # stone, blank, apparatus

    closed = inputs(gpsi)
    closed.compose(U); closed.compose(U.adjoint())
    closed.compose(inputs(gphi).adjoint())                  # the mirror: effects phi, 0, A
    assert len(closed.inputs()) == 0 and len(closed.outputs()) == 0   # a closed picture
    assert np.isclose(value(closed), s)
    # DERIVED: full_reduce takes U U^dagger out of the closed picture; what is left is
    # exactly what the same rules leave of the picture drawn with NO box at all -- the
    # three likenesses (psi against phi, blank against blank, A against A).  (The
    # generic phases of psi, phi, A are not Clifford, so a small diagram remains; its
    # value is read through matrix_of.)
    no_box = inputs(gpsi); no_box.compose(inputs(gphi).adjoint())
    assert np.isclose(value(no_box), s)
    h, hn = closed.copy(), no_box.copy()
    simp.full_reduce(h); simp.full_reduce(hn)
    assert h.num_vertices() == hn.num_vertices() and h.num_edges() == hn.num_edges()
    assert np.isclose(value(h), s) and np.isclose(value(hn), s)
    # the unconjugated effect (a transpose, not the mirror) gives a different number
    wrong = inputs(gpsi); wrong.compose(U); wrong.compose(U.adjoint())
    tr = inputs(gphi).adjoint()                              # (its scalar is real)
    for v in tr.vertices():
        tr.set_phase(v, -tr.phase(v))                        # undo the conjugation
    wrong.compose(tr)
    assert not np.isclose(value(wrong), s)

    # (b) ONE loop through the copier and its mirror: green point 1/3, effect -1/4 (the
    # mirror of the green point 1/4); value exactly s = <phi|psi>, non-real
    psi1 = (ket(0) + np.exp(1j * np.pi / 3) * ket(1)) / np.sqrt(2)
    phi1 = (ket(0) + np.exp(1j * np.pi / 4) * ket(1)) / np.sqrt(2)
    s1 = np.vdot(phi1, psi1)
    assert not np.isclose(s1.imag, 0) and 0 < abs(s1) < 1
    g1 = one_loop(Fraction(1, 3), Fraction(-1, 4))
    assert np.isclose(value(g1), s1)
    assert not np.isclose(value(one_loop(Fraction(1, 3), Fraction(1, 4))), s1)   # unconjugated: no
    # the copier and its mirror are invisible in the loop: it equals the bare loop
    assert same_picture(g1, loops(Fraction(1, 3), Fraction(-1, 4), 1))

    # (c) TWO separate loops: value exactly s^2
    g2 = loops(Fraction(1, 3), Fraction(-1, 4), 2)
    assert np.isclose(value(g2), s1 ** 2)

    # (d) one loop is not two loops ...
    assert not np.isclose(value(g1), value(g2))
    assert not same_picture(g1, g2)
    # ... except when the loop is all or nothing: orthogonal (red points 0 and pi, value
    # 0) or the very same (green 1/3 against its own mirror, value 1)
    o1, o2 = one_loop(0, 1, colour=VT.X), loops(0, 1, 2, colour=VT.X)
    assert np.isclose(value(o1), 0) and np.isclose(value(o2), 0) and same_picture(o1, o2)
    e1, e2 = one_loop(Fraction(1, 3), Fraction(-1, 3)), loops(Fraction(1, 3), Fraction(-1, 3), 2)
    assert np.isclose(value(e1), 1) and np.isclose(value(e2), 1) and same_picture(e1, e2)

    # (e) DERIVE (b): full_reduce takes the one loop to the empty picture, scalar s
    h = g1.copy(); simp.full_reduce(h)
    assert h.num_vertices() == 0 and np.isclose(h.scalar.to_number(), s1)
    h = g2.copy(); simp.full_reduce(h)
    assert h.num_vertices() == 0 and np.isclose(h.scalar.to_number(), s1 ** 2)


if __name__ == "__main__":
    verdict("NoCloning", blackboard, sketchbook)
