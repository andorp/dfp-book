"""Challenge IV: Eve's Blanks  (no-deleting: a copy cannot be unmade, only moved)
===============================================================================
"The tell always came out holding the stone she had blanked."  -- Crab

Claim: a unitary deleter U(psi psi A) = psi Sigma A_psi, U(phi phi A) = phi Sigma A_phi
with s = <phi|psi> != 0 has <A_phi|A_psi> = s (two loops before, one loop and the
tell's loop after): the tell holds the stone.  The three-CNOT knot does exactly this --
it is a swap, psi psi 0 -> psi 0 psi.

Pati and Braunstein (arXiv:quant-ph/9911090; Nature 404, 164 (2000)), paraphrased:
for a merely LINEAR deleter acting on |H>|H> and |V>|V> with final tells |A_H>, |A_V>,
linearity forces the tell to be |A_psi> = alpha|A_H> + beta|A_V>, and normalization for
every alpha, beta forces <A_H|A_V> = 0: the second copy is swapped into the tell.  They
explicitly exclude swapping as a form of deleting -- and linearity leaves nothing else.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import I2, kron, ket, matrix_of, random_state, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch11 as PC
import props_cloth as CLOTH


PLUS = (ket(0) + ket(1)) / np.sqrt(2)
MINUS = (ket(0) - ket(1)) / np.sqrt(2)
SIGMA = ket(0)                                               # the blank: settled low


def cnot(n, c, t):
    """CNOT on n wires, control c, target t (wire 0 is the TOP wire)."""
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for k in range(2 ** n):
        bits = [(k >> (n - 1 - i)) & 1 for i in range(n)]
        if bits[c]:
            bits[t] ^= 1
        M[int("".join(map(str, bits)), 2), k] = 1
    return M


# Eve's frame on (first stone, second stone, tell): three cords between the second
# stone's line (1) and the tell's (2), "the middle one tied the other way about"
FRAME = cnot(3, 1, 2) @ cnot(3, 2, 1) @ cnot(3, 1, 2)


def blackboard():
    rng = np.random.default_rng(1104)
    psi, phi = random_state(rng), random_state(rng)
    for v in (psi, phi):
        assert not np.allclose(v, v.conj())                  # non-real
    s = np.vdot(phi, psi)
    assert not np.isclose(s, 0) and not np.isclose(s.imag, 0)

    # the knot is a swap of the second stone and the tell
    SWAP12 = kron(I2, cnot(2, 0, 1) @ cnot(2, 1, 0) @ cnot(2, 0, 1))
    assert np.allclose(FRAME, SWAP12)
    # all three cords tied the SAME way is one CNOT again, not a swap
    same_way = cnot(3, 1, 2) @ cnot(3, 1, 2) @ cnot(3, 1, 2)
    assert np.allclose(same_way, cnot(3, 1, 2)) and not np.allclose(same_way, FRAME)

    # two stones settled alike, the tell starting settled low, "as a blank does":
    # psi psi 0 -> psi 0 psi, exactly
    for x in (psi, phi):
        out = FRAME @ kron(x, x, SIGMA).ravel()
        assert np.allclose(out, kron(x, SIGMA, x).ravel())
        assert not np.allclose(out, kron(x, SIGMA, x.conj()).ravel())   # psi, never psi-bar
        assert not np.allclose(out, kron(x, x, SIGMA).ravel())          # something did move
    # so A_psi = psi: the loop form <A_phi|A_psi> = s, and s^2 = s <A_phi|A_psi>
    A_psi = FRAME @ kron(psi, psi, SIGMA).ravel()
    A_psi = A_psi.reshape(2, 2, 2)[0, 0, :] / psi[0]          # the tell, given stone psi
    A_phi = (FRAME @ kron(phi, phi, SIGMA).ravel()).reshape(2, 2, 2)[0, 0, :] / phi[0]
    assert np.allclose(A_psi, psi) and np.allclose(A_phi, phi)
    a = np.vdot(A_phi, A_psi)
    assert np.isclose(a, s)
    before = np.vdot(kron(phi, phi, SIGMA).ravel(), kron(psi, psi, SIGMA).ravel())
    after = np.vdot(FRAME @ kron(phi, phi, SIGMA).ravel(), FRAME @ kron(psi, psi, SIGMA).ravel())
    assert np.isclose(before, s * s * 1) and np.isclose(after, s * 1 * a)
    assert np.isclose(s * s, s * a)                          # two loops = one loop and the tell's
    assert np.isclose(np.vdot(SIGMA, SIGMA), 1)              # the blank's loop is 1

    # the letter, every settling: the frame "always gave her a blank", left the first
    # stone as it was, and the tell, read at the setting the stones had been given,
    # read exactly as the blanked stone would have read
    for setting in ((ket(0), ket(1)), (PLUS, MINUS)):
        for k, x in enumerate(setting):
            out = FRAME @ kron(x, x, SIGMA).ravel()
            assert np.allclose(out, kron(x, SIGMA, x).ravel())
            tell = out.reshape(2, 2, 2)[:, 0, :]                          # blank is exactly |0>
            p = [np.linalg.norm(tell @ setting[j].conj()) ** 2 for j in (0, 1)]
            assert np.isclose(p[k], 1) and np.isclose(p[1 - k], 0)       # the stone's answer

    # Pati-Braunstein's linear form, with the "mixed" input forced by linearity:
    # D|HH>|A> = |H>|Sigma>|A_H>, D|VV>|A> = |V>|Sigma>|A_V>,
    # D (|HV>+|VH>)/sqrt2 |A> = (|H>|Sigma>|A_V> + |V>|Sigma>|A_H>)/sqrt2.
    # On psi psi it gives psi Sigma (alpha A_H + beta A_V).
    def deleter_output(x, AH, AV):
        al, be = x
        return (al ** 2 * kron(ket(0), SIGMA, AH) + be ** 2 * kron(ket(1), SIGMA, AV)
                + al * be * (kron(ket(0), SIGMA, AV) + kron(ket(1), SIGMA, AH))).ravel()
    AH, AV = ket(0), ket(1)                                  # orthogonal tells: the swap's
    assert np.allclose(deleter_output(psi, AH, AV), kron(psi, SIGMA, psi[0] * AH + psi[1] * AV).ravel())
    assert np.isclose(np.linalg.norm(deleter_output(psi, AH, AV)), 1)
    # NEGATIVE: a tell that ignores the stone (A_H = A_V) -- "just a bit of ballast" --
    # sends a superposed stone to a vector of norm |alpha + beta| != 1: no fair process
    A0 = random_state(rng)
    bad = deleter_output(psi, A0, A0)
    assert np.allclose(bad, (psi[0] + psi[1]) * kron(psi, SIGMA, A0).ravel())
    assert not np.isclose(np.linalg.norm(bad), 1)
    # ... and so does any pair of non-orthogonal tells
    B1, B2 = random_state(rng), random_state(rng)
    assert not np.isclose(np.vdot(B1, B2), 0)
    assert not np.isclose(np.linalg.norm(deleter_output(psi, B1, B2)), 1)

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws, in one motion, the Tortoise's knot of
# Chapter 8's Sixth: the upper line loose from the left (her second stone), the
# lower line from a red point (the tell's settled start), and three short cords
# across, green-over-red, red-over-green, green-over-red (keyed: cord, line, knot);
# then the knot pulled: a plain crossing, the stone's line down to the lower right,
# into the tell's place, the tell's line up to the upper right,
#   \copycloth[operations]{the knot}{caption}   (pdf/props/ch11.tex)
# tools/props_ch11.py transcribes the Lua (with L1-L5); the calls are read out of
# dialog.md, never copied.
#
# The ties (identity card, section 4), a structural check against this file's
# blackboard, not a second proof.  A cord green on one line and red on the other is a
# CNOT, green the control; three of them alternating are CNOT CNOT CNOT = SWAP,
# Eve's frame, exactly -- the only knot the Lua's pull undoes; three the same way are
# one CNOT again, not a swap, and the Lua has no pull for them.  The pulled crossing
# is that swap: what came in on the stone's line leaves in the tell's place.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = "The tell always came out holding the stone she had blanked."
KNOT = "the knot"


def cord_cnot(tie):
    """A cord across (upper, lower colours) on (first stone, second stone = upper line,
    tell = lower line): green the control, red the target."""
    return cnot(3, 1, 2) if (tie["upper"], tie["lower"]) == ("green", "red") else cnot(3, 2, 1)


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    calls = DIALOGUE.check_layout("Fourth", sizes=[2])
    p41, p42 = calls
    text, raw = DIALOGUE.read(DIALOGUE.dialog), DIALOGUE.read(DIALOGUE.raw)
    spoken = PC.spoken_sentences(text) | PC.spoken_sentences(raw)
    for c in calls:
        assert words_only(c.caption, forbidden=PC.CAPTION_FORBIDDEN), c.caption
        assert CLOTH.caption_ok(c.caption), c.caption
        assert PC.normal_sentence(c.caption) not in spoken, c.caption   # the gigue: never said twice
        assert PC.place_ok("Fourth", c), c.text
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Fourth")] == [[1.24, 1.24]]
    assert [c.macro for c in calls] == ["copycloth"] * 2
    assert [c.keyed for c in calls] == [True, False]
    assert [(c.opt, c.args[0]) for c in calls] == [("", KNOT), ("pull", KNOT)]
    assert [c.caption for c in calls] == ["the middle tied the other way about", "pulled, and undone"]
    c41, c42 = (PC.picture(c) for c in calls)
    (a1,) = DIALOGUE.anchors("Fourth")
    (f1,) = DIALOGUE.following("Fourth")
    section = DIALOGUE.section("Fourth")
    assert QUOTE in section                                       # the docstring's line
    assert "the ring and its cords" in section                    # D8: the last layout put away
    assert "only which one ends where has changed" in a1
    assert f1.startswith("**Achilles:** A crossing.")

    # P4.1: the knot, keyed; the upper line loose from the left, the lower from a red
    # point at rest; three cords across, alternating
    assert c41.labels == ("cord", "line", "knot")
    th = c41.cloth["things"]
    ties = [(th[d["a"] - 1], th[d["b"] - 1]) for d in c41.cloth["cords"] if d["kind"] == "tie"]
    ties = sorted(({"upper": u["c"], "lower": l["c"], "x": u["x"]} for u, l in ties), key=lambda t: t["x"])
    assert [(t["upper"], t["lower"]) for t in ties] == [("green", "red"), ("red", "green"), ("green", "red")]
    assert [(t["upper"], t["lower"]) for t in ties] == [(t["upper"], t["lower"]) for t in PC.KNOT]
    points = [(r["x"], r["y"]) for r in th if r["kind"] == "end"]
    assert points == [(0, 1)] and [r["c"] for r in th if r["kind"] == "end"] == ["red"]
    before = text[:text.index(p41.text)].lower()
    for w in c41.labels:
        assert w in before, w                                     # named before it is keyed
    # P4.2: pulled, a plain crossing: the red point still at the tell's start, the line
    # from the stone out at the lower right, the tell's line out at the upper right
    assert c42.dump.endswith("1:end red 0 @0,1; cord loose-1 [6,0 4,0 2,1 0,1]; "
                             "cord loose-loose [0,0 2,0 4,1 6,1]")
    assert not [d for d in c42.cloth["cords"] if d["kind"] == "tie"] and c42.frame == c41.frame
    start = lambda d: d["pts"][0] if d["pts"][0][0] == 0 else d["pts"][-1]
    finish = lambda d: d["pts"][-1] if d["pts"][0][0] == 0 else d["pts"][0]
    moved = {start(d)[1]: finish(d)[1] for d in c42.cloth["cords"]}
    assert moved == {0: 1, 1: 0}                                  # the lines exchange rows
    assert PC.faults(PC.copycloth, "pull, pull", KNOT) == ["knot-pull-twice"]

    # the ties.  The drawn knot is Eve's frame, and the frame is the swap
    knot = np.eye(8)
    for t in ties:                                                # left to right: first applied first
        knot = cord_cnot(t) @ knot
    assert np.allclose(knot, FRAME)
    rng = np.random.default_rng(11041)
    x, u, w = random_state(rng), random_state(rng), random_state(rng)
    assert not np.allclose(u, u.conj()) and not np.allclose(w, w.conj())
    assert np.allclose(knot @ kron(x, u, w).ravel(), kron(x, w, u).ravel())   # rows exchanged
    assert np.allclose(knot @ kron(x, x, SIGMA).ravel(), kron(x, SIGMA, x).ravel())   # the blank
    # the only knot the pull undoes: three alternating (either way first) -- three the
    # same way are one CNOT, not a swap, and two are not a knot at all
    assert PC.knot_pull(list(PC.KNOT)) and PC.faults(PC.knot_pull, list(PC.KNOT)) == []
    other = [{"upper": "red", "lower": "green"}, {"upper": "green", "lower": "red"},
             {"upper": "red", "lower": "green"}]
    assert PC.faults(PC.knot_pull, other) == []
    same = [{"upper": "green", "lower": "red"}] * 3
    assert PC.faults(PC.knot_pull, same) == ["knot-not-alternating"]
    same_way = cord_cnot(same[2]) @ cord_cnot(same[1]) @ cord_cnot(same[0])
    assert np.allclose(same_way, cnot(3, 1, 2)) and not np.allclose(same_way, FRAME)
    assert PC.faults(PC.knot_pull, same[:2]) == ["knot-count"]
    alt = cord_cnot(other[2]) @ cord_cnot(other[1]) @ cord_cnot(other[0])
    assert np.allclose(alt, FRAME)                                # either way first: the swap


# --- pictures -----------------------------------------------------------------

def chain_state(phases, row):
    """A generic one-wire state on the given row: green a, red b, green c (units of pi),
    normalized by its own norm (add_float).  Non-real and not on the equator."""
    a, b, c = (Fraction(t) for t in phases)
    g = zx.Graph()
    v = [g.add_vertex(VT.Z, row, 0, phase=a), g.add_vertex(VT.X, row, 1, phase=b),
         g.add_vertex(VT.Z, row, 2, phase=c)]
    o = g.add_vertex(VT.BOUNDARY, row, 3)
    g.add_edge((v[0], v[1])); g.add_edge((v[1], v[2])); g.add_edge((v[2], o))
    g.set_inputs(()); g.set_outputs((o,))
    g.scalar.add_float(1 / np.linalg.norm(matrix_of(g)))
    return g


def blank_state(row):
    """The blank |0>: a red point, sqrt2|0>, with its 1/sqrt2 (add_power(-1))."""
    g = zx.Graph()
    p, o = g.add_vertex(VT.X, row, 0), g.add_vertex(VT.BOUNDARY, row, 3)
    g.add_edge((p, o)); g.set_inputs(()); g.set_outputs((o,))
    g.scalar.add_power(-1)
    return g


def knot(middle_reversed=True):
    """Three spider-CNOTs between the second stone (row 1) and the tell (row 2); the
    first stone (row 0) runs straight through.  Each bare spider-CNOT is CNOT/sqrt2, so
    the knot drawn bare is SWAP/(2 sqrt2): the chapter's 2 sqrt2 in front is add_power(3)."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in range(3)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 5) for r in range(3)]
    g.add_edge((ins[0], outs[0]))
    last = {1: ins[1], 2: ins[2]}
    for k, col in enumerate((1, 2, 3)):
        top_is_ctl = not (middle_reversed and k == 1)
        up = g.add_vertex(VT.Z if top_is_ctl else VT.X, 1, col)
        dn = g.add_vertex(VT.X if top_is_ctl else VT.Z, 2, col)
        g.add_edge((last[1], up)); g.add_edge((last[2], dn)); g.add_edge((up, dn))
        last = {1: up, 2: dn}
    g.add_edge((last[1], outs[1])); g.add_edge((last[2], outs[2]))
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    g.scalar.add_power(3)                                    # the 2 sqrt2 in front
    return g


def crossing():
    """The first stone straight; the second stone and the tell cross."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in range(3)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 2) for r in range(3)]
    g.add_edge((ins[0], outs[0])); g.add_edge((ins[1], outs[2])); g.add_edge((ins[2], outs[1]))
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    return g


def connected(g, u, v):
    seen, todo = {u}, [u]
    while todo:
        w = todo.pop()
        for n in g.neighbors(w):
            if n not in seen:
                seen.add(n); todo.append(n)
    return v in seen


def sketchbook():
    PH = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 5))
    psi = matrix_of(chain_state(PH, 0)).ravel()
    assert np.isclose(np.linalg.norm(psi), 1)
    assert not np.allclose(psi, psi.conj()) and not np.isclose(abs(psi[0]), abs(psi[1]))

    # the knot, with its 2 sqrt2, is exactly the crossing (Chapter 8's three CNOTs = swap)
    k = knot()
    SWAP12 = kron(I2, cnot(2, 0, 1) @ cnot(2, 1, 0) @ cnot(2, 0, 1))
    assert np.allclose(matrix_of(k), SWAP12)
    assert same_picture(k, crossing())
    c = zx.Circuit(3)                                        # cross-check with PyZX's own CNOTs
    for ct, tg in ((1, 2), (2, 1), (1, 2)):
        c.add_gate("CNOT", ct, tg)
    assert same_picture(k, c.to_graph())
    assert not same_picture(knot(middle_reversed=False), crossing())   # all tied one way: no
    # DERIVED: the rules pull the knot into the crossing
    h = k.copy(); simp.full_reduce(h)
    assert same_picture(h, crossing())
    assert h.num_vertices() == 6 and h.num_edges() == 3      # three bare wires, two crossed

    # the stones plugged in: psi, psi, and the tell settled low
    def plugged(states, m):
        g = states[0].tensor(states[1]).tensor(states[2])
        if m is not None:
            g.compose(m)
        return g
    lhs = plugged([chain_state(PH, 0), chain_state(PH, 1), blank_state(2)], knot())
    rhs = plugged([chain_state(PH, 0), blank_state(1), chain_state(PH, 2)], None)
    assert np.allclose(matrix_of(lhs).ravel(), kron(psi, ket(0), psi).ravel())
    assert same_picture(lhs, rhs)                            # psi psi 0 -> psi 0 psi, exact
    assert not np.allclose(matrix_of(lhs).ravel(), kron(psi, ket(0), psi.conj()).ravel())
    # DERIVED: after full_reduce the three outputs are three separate pieces -- the blank
    # on the middle wire, the stone on the tell's wire, nothing tying them together
    h = lhs.copy(); simp.full_reduce(h)
    assert same_picture(h, rhs)
    o = h.outputs()
    assert not connected(h, o[0], o[1]) and not connected(h, o[1], o[2]) and not connected(h, o[0], o[2])


if __name__ == "__main__":
    verdict("NoDeleting", blackboard, sketchbook)
