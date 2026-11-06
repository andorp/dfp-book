"""Challenge III: Eve's Tap  (copying is looking; the easy half of no-broadcasting)
================================================================================
"Nothing in them to spoil: they were settled already."  -- Achilles

Claim: copy, then discard one copy, is dephasing -- both marginals of
delta rho delta^dagger equal sum_i P_i rho P_i, the stone after a plain look nobody
read, and equal rho exactly when rho is diagonal; the copier in a common eigenbasis
broadcasts every pair of commuting states, correlated but not cloned.  In pictures:
the doubled copier with one output capped onto its mirror fuses into the four-legged
green dot, the same dot as a read followed by its mirror with the record forgotten.

Barnum, Caves, Fuchs, Jozsa and Schumacher (arXiv:quant-ph/9511010): a pair
{rho_0, rho_1} can be broadcast IF AND ONLY IF rho_0 rho_1 = rho_1 rho_0, and cloned
iff identical or orthogonal.  Only the easy direction (commuting => broadcast, by the
copier of the common eigenbasis) is checked here; the converse is stated, not tested.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import CAP, CUP, I2, kron, ket, matrix_of, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch11 as PC
import props_cloth as CLOTH


DELTA = np.outer(ket(0, 0), ket(0)) + np.outer(ket(1, 1), ket(1))
PLUS = (ket(0) + ket(1)) / np.sqrt(2)
MINUS = (ket(0) - ket(1)) / np.sqrt(2)
P = [np.outer(ket(i), ket(i)) for i in (0, 1)]


def random_unitary(rng, d):
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_rho(rng):
    M = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho = M @ M.conj().T
    return rho / np.trace(rho)


def marginals(R):
    """(tr_2 R, tr_1 R) of a two-wire density matrix R (top wire first)."""
    T = R.reshape(2, 2, 2, 2)
    return np.einsum('ikjk->ij', T), np.einsum('kikj->ij', T)


def blackboard():
    rng = np.random.default_rng(1103)
    rho = random_rho(rng)
    assert not np.allclose(rho, rho.real)                    # non-real
    assert not np.isclose(rho[0, 1], 0)                      # non-diagonal

    # copy, then discard either copy: dephasing, the plain look with the outcome unread
    R = DELTA @ rho @ DELTA.conj().T
    look = sum(Pi @ rho @ Pi for Pi in P)
    for m in marginals(R):
        assert np.allclose(m, np.diag(np.diag(rho)))
        assert np.allclose(m, look)                          # = Chapter 9's look-and-forget
        assert not np.allclose(m, rho)                       # the phase is gone
    # the two copies, read plain, always agree: the kept copy is the look, in the drawer
    assert np.isclose(R[1, 1], 0) and np.isclose(R[2, 2], 0)
    # thrown away altogether, the tapped stone is a stone thrown away (causality)
    assert np.isclose(np.trace(R), np.trace(rho))

    # a diagonal (commuting-with-the-plain-basis) mixed rho: broadcast, but not cloned
    lam = np.array([0.3, 0.7])
    D = np.diag(lam).astype(complex)
    RD = DELTA @ D @ DELTA.conj().T
    assert all(np.allclose(m, D) for m in marginals(RD))     # each marginal is rho: broadcast
    assert np.allclose(RD, sum(l * np.outer(ket(b, b), ket(b, b)) for b, l in enumerate(lam)))
    assert not np.allclose(RD, np.kron(D, D))                # ... not rho (x) rho: not cloned
    assert np.isclose(np.trace(D @ D).real, 0.58)            # (it is mixed, not pure)

    # the easy half of Barnum et al., in a generic basis: two commuting states, diagonal
    # in a random NON-REAL orthonormal basis |b> = W|b>, are broadcast by that basis's
    # copier delta_B = sum_b |bb><b| = (W (x) W) delta W^dagger, and not cloned
    W = random_unitary(rng, 2)
    assert not np.allclose(W, W.conj())
    dB = np.kron(W, W) @ DELTA @ W.conj().T
    rhos = [W @ np.diag(l).astype(complex) @ W.conj().T for l in ([0.2, 0.8], [0.9, 0.1])]
    assert np.allclose(rhos[0] @ rhos[1], rhos[1] @ rhos[0])  # they commute
    for r in rhos:
        Rb = dB @ r @ dB.conj().T
        assert all(np.allclose(m, r) for m in marginals(Rb))  # broadcast
        assert not np.allclose(Rb, np.kron(r, r))             # not cloned
    # the plain copier would not do for them (they are not diagonal in the plain basis)
    assert not all(np.allclose(m, rhos[0]) for m in marginals(DELTA @ rhos[0] @ DELTA.conj().T))
    # Eve wanted |0><0| and |+><+|: they do NOT commute (the theorem's hypothesis fails)
    r0, rp = np.outer(ket(0), ket(0)), np.outer(PLUS, PLUS)
    assert not np.allclose(r0 @ rp, rp @ r0)

    # --- the letter (every settling the scene allows) ---------------------------
    customer = lambda r: marginals(DELTA @ r @ DELTA.conj().T)[1]    # Eve keeps copy 1
    # settled plain, high or low: arrives as good as ever
    for i in (0, 1):
        assert np.allclose(customer(P[i]), P[i])
    # settled crosswise, EITHER way: arrives looked-at -- at cross, a coin toss
    for s in (PLUS, MINUS):
        c = customer(np.outer(s, s.conj()))
        assert np.allclose(c, I2 / 2)
        assert not np.allclose(c, np.outer(s, s.conj()))
        at_cross = [np.vdot(t, c @ t).real for t in (PLUS, MINUS)]
        assert np.allclose(at_cross, [0.5, 0.5])
    # a press that copies high and low "leaves high and low alone, and looks at
    # everything in between": the customer's stone is always the plain look of the sent one
    assert np.allclose(customer(rho), look)

    # doubling is a wire plus its CONJUGATE copy: rho as a state on (e, ebar) is
    # vec(rho) (top wire first), and a box K acts as K (x) conj(K).  The copier is real,
    # so its conjugate copy is the green fork again -- but a non-real box shows the rule.
    vec = lambda r: r.reshape(-1)
    K = random_unitary(rng, 2)
    assert np.allclose(np.kron(K, K.conj()) @ vec(rho), vec(K @ rho @ K.conj().T))
    assert not np.allclose(np.kron(K, K) @ vec(rho), vec(K @ rho @ K.conj().T))
    assert np.allclose(DELTA, DELTA.conj())
    # and discard is the cap between the two copies: <cap| vec(rho) = tr rho
    assert np.isclose((CAP @ vec(rho))[0], np.trace(rho))
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))       # cup/cap unnormalized

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws, in one motion, the Tortoise's doubled
# fork: the first string along the top with the fork on it, the mirror string
# along the bottom, upside down and running the same way, with the fork's
# reflection (keyed: fork, string, mirror string); the second copy's end folded
# over onto its reflection's, one cord from fork to mirror fork turned round at
# the right; and the forks slid together along it into one ring with four cords,
# two in from the left, two out to the right,
#   \copycloth[operations]{the fork and its mirror}{caption}   (pdf/props/ch11.tex)
# tools/props_ch11.py transcribes the Lua (with L1-L5); the calls are read out of
# dialog.md, never copied.
#
# The ties (identity card, section 4), a structural check against this file's
# blackboard, not a second proof.  The mirror string is the first string reflected
# top to bottom and running the same way -- the doubling's conjugate copy, not the
# mirror card's reversal (the dagger) -- so the fold at the right lays the second
# copy's end onto its reflection's end: a trace.  Copy, then that trace, is the
# dephasing channel, and as a map on the doubled wire (e, ebar) it is exactly the
# four-legged green ring |00><00| + |11><11|, two cords in, two out.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = "Nothing in them to spoil: they were settled already."
MIRROR = "the fork and its mirror"


def reflected(d):
    """A cord of the first string reflected top to bottom (y -> 2 - y), running the same
    way; its ring ends taken to the mirror fork (ring 1 -> ring 2)."""
    end = lambda e: 2 if e == 1 else e
    return (end(d.get("a")), end(d.get("b")), tuple((p[0], 2 - p[1]) for p in d["pts"]))


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    calls = DIALOGUE.check_layout("Third", sizes=[3])
    p31, p32, p33 = calls
    text, raw = DIALOGUE.read(DIALOGUE.dialog), DIALOGUE.read(DIALOGUE.raw)
    spoken = PC.spoken_sentences(text) | PC.spoken_sentences(raw)
    for c in calls:
        assert words_only(c.caption, forbidden=PC.CAPTION_FORBIDDEN), c.caption
        assert CLOTH.caption_ok(c.caption), c.caption
        assert PC.normal_sentence(c.caption) not in spoken, c.caption   # the gigue: never said twice
        assert PC.place_ok("Third", c), c.text
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Third")] == [[1.78, 1.78, 1.78]]
    assert [c.macro for c in calls] == ["copycloth"] * 3
    assert [c.keyed for c in calls] == [True, False, False]
    assert [(c.opt, c.args[0]) for c in calls] == [
        ("", MIRROR), ("fold the second copy", MIRROR),
        ("fold the second copy, slide the forks together", MIRROR)]
    assert [c.caption for c in calls] == [
        "as the slab would show it", "fork to mirror fork", "the forks slid into each other"]
    c31, c32, c33 = (PC.picture(c) for c in calls)
    (a1,) = DIALOGUE.anchors("Third")
    (f1,) = DIALOGUE.following("Third")
    section = DIALOGUE.section("Third")
    assert QUOTE in section                                       # the docstring's line
    paras = DIALOGUE.paragraphs("Third")
    k = paras.index(next(q for q in paras if p31.text in q))
    assert "upside down, and running the same way" in paras[k - 2]           # D7, two before
    assert "gathers the loops off the cloth into her shell" in paras[k - 2]  # the last layout put away

    # P3.1: the fork and its reflection, keyed; both strings run left to right, the
    # mirror string the first reflected top to bottom (D7)
    assert c31.labels == ("fork", "string", "mirror string")
    rings = [(r["kind"], r["c"], r["x"], r["y"]) for r in c31.cloth["things"]]
    assert rings == [("ring", "green", 2, 0), ("ring", "green", 2, 2)]
    top = [d for d in c31.cloth["cords"] if all(p[1] < 1 for p in d["pts"])]
    bottom = [d for d in c31.cloth["cords"] if all(p[1] > 1 for p in d["pts"])]
    assert len(top) == len(bottom) == 3 and len(top) + len(bottom) == len(c31.cloth["cords"])
    assert {reflected(d) for d in top} == {(d.get("a"), d.get("b"), tuple(map(tuple, d["pts"]))) for d in bottom}
    # running the SAME way: reflected left for right instead, the strings would not match
    assert {(d.get("a"), d.get("b"), tuple((6 - p[0], 2 - p[1]) for p in d["pts"])) for d in top} != \
        {(d.get("a"), d.get("b"), tuple(map(tuple, d["pts"]))) for d in bottom}
    before = text[:text.index(p31.text)].lower()
    for w in c31.labels:
        assert w in before, w                                     # named before it is keyed
    # P3.2: the second copy's end folded onto its reflection's: one cord from fork to
    # mirror fork, turned round at the RIGHT, beyond where the loose copies ended
    fold = [d for d in c32.cloth["cords"] if d.get("a") == 1 and d.get("b") == 2]
    assert len(fold) == 1 and len(c32.cloth["cords"]) == 5
    loose_end = max(p[0] for d in c31.cloth["cords"] for p in d["pts"] if 0 < p[1] < 2)
    turn = max(p[0] for p in fold[0]["pts"])
    assert turn > loose_end > 2                                   # at the right: an end onto an end
    assert c32.cloth["things"] == c31.cloth["things"]             # the forks keep their place
    # P3.3: the forks slid together: one green ring with four cords, two in from the
    # left (the stone's and its reflection's), two out to the right (the kept copy's)
    (ring,) = c33.cloth["things"]
    assert (ring["kind"], ring["c"], ring["x"], ring["y"]) == ("ring", "green", 3, 1)
    far = [d["pts"][-1][0] if d["pts"][0] == [3, 1] else d["pts"][0][0] for d in c33.cloth["cords"]]
    assert sorted(far) == [0, 0, 6, 6] and all(d.get("b") == 1 or d.get("a") == 1 for d in c33.cloth["cords"])
    assert c33.dump.endswith("1:ring green 0 @3,1; cord loose-1 [0,0 2,0 3,1]; cord loose-1 [6,0 4,0 3,1]; "
                             "cord loose-1 [0,2 2,2 3,1]; cord loose-1 [6,2 4,2 3,1]")
    assert "still go out" in a1
    assert f1.startswith("**Tortoise:** Stone and reflection in")
    assert PC.faults(PC.copycloth, "slide the forks together", MIRROR) == ["mirror-slide-early"]

    # the ties.  Copy, then fold the second copy onto its reflection (a trace): as a
    # map on vec(rho) (e, ebar; top wire first) it is the four-legged green ring
    rng = np.random.default_rng(11031)
    rho = random_rho(rng)
    assert not np.allclose(rho, rho.real) and not np.isclose(rho[0, 1], 0)
    vec = lambda r: r.reshape(-1)
    sup = np.column_stack([vec(marginals(DELTA @ np.outer(ket(i), ket(j)) @ DELTA.conj().T)[0])
                           for i in (0, 1) for j in (0, 1)])
    ring4 = np.outer(ket(0, 0), ket(0, 0)) + np.outer(ket(1, 1), ket(1, 1))
    assert np.allclose(sup, ring4)                                # two in, two out, exactly
    assert np.allclose(sup @ vec(rho), vec(np.diag(np.diag(rho))))
    assert sum(1 for x in far if x == 0) == sum(1 for x in far if x == 6) == 2   # 2 in, 2 out
    # the mirror string is the CONJUGATE copy: a non-real box K is doubled as K (x) conj K;
    # a reversed string (conj K transposed, the mirror card's rule) is not the doubling
    K = random_unitary(rng, 2)
    assert not np.allclose(K, K.conj()) and not np.allclose(K, K.T)
    assert np.allclose(np.kron(K, K.conj()) @ vec(rho), vec(K @ rho @ K.conj().T))
    assert not np.allclose(np.kron(K, K.conj().T) @ vec(rho), vec(K @ rho @ K.conj().T))


# --- pictures (hand-built doubled diagrams; pyzx 0.10.6 refuses ground vertices) -----
# Wire order on every doubled system: the wire, then its mirror copy (e, ebar).

def tap():
    """The doubled copier with one output discarded: the copier on e (outputs e1, e2),
    its conjugate -- the green fork again, delta being real -- on ebar (outputs ebar1,
    ebar2), and the discard: a cap (a phase-free two-legged green dot) joining e2 to
    ebar2.  Inputs (e, ebar), outputs (e1, ebar1)."""
    g = zx.Graph()
    ie, ieb = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 3, 0)
    c, cb = g.add_vertex(VT.Z, 0, 1), g.add_vertex(VT.Z, 3, 1)
    cap = g.add_vertex(VT.Z, 1.5, 2)                         # discard: e2 bent onto ebar2
    oe, oeb = g.add_vertex(VT.BOUNDARY, 0, 3), g.add_vertex(VT.BOUNDARY, 3, 3)
    for a, b in ((ie, c), (c, oe), (c, cap), (ieb, cb), (cb, oeb), (cb, cap)):
        g.add_edge((a, b))
    g.set_inputs((ie, ieb)); g.set_outputs((oe, oeb))
    return g


def four_legged():
    """Z(2->2) = sum_i |ii><ii|: a phase-free green dot, legs e, ebar in and e1, ebar1 out."""
    g = zx.Graph()
    ie, ieb = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0)
    z = g.add_vertex(VT.Z, 0.5, 1)
    oe, oeb = g.add_vertex(VT.BOUNDARY, 0, 2), g.add_vertex(VT.BOUNDARY, 1, 2)
    for v in (ie, ieb, oe, oeb):
        g.add_edge((v, z))
    g.set_inputs((ie, ieb)); g.set_outputs((oe, oeb))
    return g


def read_then_set():
    """Chapter 9's read-spider Z(2->1) on (e, ebar), its classical record, then the
    mirror of the read-spider Z(1->2) setting a stone as read.  The record runs between
    the two and is read by nobody."""
    g = zx.Graph()
    ie, ieb = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0)
    r, s = g.add_vertex(VT.Z, 0.5, 1), g.add_vertex(VT.Z, 0.5, 3)
    oe, oeb = g.add_vertex(VT.BOUNDARY, 0, 4), g.add_vertex(VT.BOUNDARY, 1, 4)
    for a, b in ((ie, r), (ieb, r), (r, s), (s, oe), (s, oeb)):  # r -- s: the classical wire
        g.add_edge((a, b))
    g.set_inputs((ie, ieb)); g.set_outputs((oe, oeb))
    return g


def look_record_forgotten():
    """The look that leaves the stone in place, Z(2 -> 2 + 1 classical), with its record
    forgotten by the phase-free one-legged green effect."""
    g = zx.Graph()
    ie, ieb = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0)
    z, forget = g.add_vertex(VT.Z, 0.5, 1), g.add_vertex(VT.Z, 2, 2)
    oe, oeb = g.add_vertex(VT.BOUNDARY, 0, 2), g.add_vertex(VT.BOUNDARY, 1, 2)
    for a, b in ((ie, z), (ieb, z), (z, oe), (z, oeb), (z, forget)):
        g.add_edge((a, b))
    g.set_inputs((ie, ieb)); g.set_outputs((oe, oeb))
    return g


def fed(g_map, colour, phase):
    """A doubled settled stone -- a normalized point on e and its mirror on ebar -- fed
    into a map on (e, ebar).  Each point carries 1/sqrt2: add_power(-2).  (The points
    used are real, phase 0 or pi, so each mirror point is the same point.)"""
    g = zx.Graph()
    pe, peb = g.add_vertex(colour, 0, 0, phase=phase), g.add_vertex(colour, 1, 0, phase=phase)
    oe, oeb = g.add_vertex(VT.BOUNDARY, 0, 1), g.add_vertex(VT.BOUNDARY, 1, 1)
    g.add_edge((pe, oe)); g.add_edge((peb, oeb))
    g.set_inputs(()); g.set_outputs((oe, oeb))
    g.scalar.add_power(-2)
    g.compose(g_map)
    return g


def doubled_state(colour, phase):
    return fed(bare(), colour, phase)


def bare():
    g = zx.Graph()
    ie, ieb = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0)
    oe, oeb = g.add_vertex(VT.BOUNDARY, 0, 1), g.add_vertex(VT.BOUNDARY, 1, 1)
    g.add_edge((ie, oe)); g.add_edge((ieb, oeb))
    g.set_inputs((ie, ieb)); g.set_outputs((oe, oeb))
    return g


def grey():
    """I/2 on (e, ebar): half a cup joining the stone to its own mirror.  cup = |00>+|11>
    unnormalized, so vec(I/2) = cup/2: add_power(-2)."""
    g = zx.Graph()
    b = g.add_vertex(VT.Z, 0.5, 0)
    oe, oeb = g.add_vertex(VT.BOUNDARY, 0, 1), g.add_vertex(VT.BOUNDARY, 1, 1)
    g.add_edge((b, oe)); g.add_edge((b, oeb))
    g.set_inputs(()); g.set_outputs((oe, oeb))
    g.scalar.add_power(-2)
    return g


def sketchbook():
    rng = np.random.default_rng(1104)
    rho = random_rho(rng)
    assert not np.allclose(rho, rho.real) and not np.isclose(rho[0, 1], 0)
    dephased = sum(Pi @ rho @ Pi for Pi in P)

    # the tap, exactly the four-legged green dot; on doubled wires it dephases rho
    t = tap()
    assert same_picture(t, four_legged())
    assert np.allclose(matrix_of(t) @ rho.reshape(-1), dephased.reshape(-1))
    assert not np.allclose(matrix_of(t) @ rho.reshape(-1), rho.reshape(-1))
    # DERIVED: fusion takes the two forks and the cap into one dot
    h = t.copy(); simp.full_reduce(h)
    assert h.num_vertices() == 5                             # one dot, four ends
    assert same_picture(h, four_legged())

    # a look nobody read is the same dot, both ways of drawing it, and the rules derive it
    for look in (read_then_set(), look_record_forgotten()):
        assert same_picture(look, four_legged())
        assert same_picture(look, t)                         # copy-and-forget = look-and-forget
        h = look.copy(); simp.full_reduce(h)
        assert h.num_vertices() == 5 and same_picture(h, four_legged())

    # the letter, in pictures: a plain stone (red point, phase 0 or pi) arrives unchanged;
    # a crosswise stone (green point, phase 0 or pi) arrives as I/2, grey -- derived
    for a in (0, 1):
        g = fed(tap(), VT.X, a)
        assert same_picture(g, doubled_state(VT.X, a))
        h = g.copy(); simp.full_reduce(h)
        assert same_picture(h, doubled_state(VT.X, a))
        g = fed(tap(), VT.Z, a)
        assert np.allclose(matrix_of(g).ravel(), (I2 / 2).reshape(-1))
        assert same_picture(g, grey())
        assert not same_picture(g, doubled_state(VT.Z, a))   # not the stone that was sent
        h = g.copy(); simp.full_reduce(h)
        assert same_picture(h, grey())

    # cap the kept copy too: the whole tap, discarded, is a discard (causality).  The
    # reduced picture is a bare edge joining two INPUTS, which matrix_of cannot evaluate
    # in 0.10.6, so it is checked as a state, its inputs turned to outputs.
    def capped(g):
        c = zx.Graph()
        a, b = c.add_vertex(VT.BOUNDARY, 0, 0), c.add_vertex(VT.BOUNDARY, 1, 0)
        k = c.add_vertex(VT.Z, 0.5, 1)
        c.add_edge((a, k)); c.add_edge((b, k))
        c.set_inputs((a, b)); c.set_outputs(())
        out = g.copy(); out.compose(c)
        return out
    whole = capped(tap())
    assert np.allclose(matrix_of(whole), CAP)
    h = whole.copy(); simp.full_reduce(h)
    as_state = h.copy(); as_state.set_outputs(h.inputs()); as_state.set_inputs(())
    assert np.allclose(matrix_of(as_state).ravel(), CUP)     # the cap, bent round: the cup


if __name__ == "__main__":
    verdict("NoBroadcasting", blackboard, sketchbook)
