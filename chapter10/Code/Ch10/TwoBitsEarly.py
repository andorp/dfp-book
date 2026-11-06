"""Challenge II: Two Bits Early  (the full Bell-basis calculation)
================================================================
"...Oh. It's there before the letter — and it's no use to him."  -- Achilles

Claim: for every one-qubit psi, psi (x) Phi+ = 1/2 sum_k |B_k> (x) X^m2 Z^m1 psi,
so each of Alice's four answers comes a quarter of the time and leaves Bob's
wire already holding the message, turned (X^m2 Z^m1 psi); but before the
letter Bob's wire, the answer forgotten, is exactly 1/2 I for every input,
pure or mixed -- "A coin toss, every time, whatever Alice had sent."
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import BELL, CUP, I2, Picture, kron, ket, matrix_of, random_state, \
    same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch10 as PC


X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
KS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def P(M, m):
    return np.linalg.matrix_power(M, m)


def bell(m1, m2):
    """|B_{m1m2}> = CNOT (H (x) I)|m1 m2>."""
    return (CNOT @ kron(H, I2) @ ket(m1, m2).reshape(4, 1)).ravel()


def random_density(rng):
    """A generic mixed state: full rank, non-real, trace 1."""
    M = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho = M @ M.conj().T
    return rho / np.trace(rho)


def blackboard():
    rng = np.random.default_rng(1002)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))       # cup unnormalized; Phi+ = cup/sqrt2
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())                  # a NON-REAL message

    # Result 10.3: the regrouping in the Bell basis, both sides evaluated
    lhs = kron(psi, BELL).ravel()                            # wires 1, a, c (top first)
    rhs = 0.5 * sum(kron(bell(m1, m2), P(X, m2) @ P(Z, m1) @ psi).ravel() for m1, m2 in KS)
    assert np.allclose(lhs, rhs)

    total = 0.0
    for m1, m2 in KS:
        branch = (kron(bell(m1, m2).conj().reshape(1, 4), I2) @ lhs).ravel()
        garbled = P(X, m2) @ P(Z, m1) @ psi
        assert np.allclose(branch, 0.5 * garbled)            # there already, turned, times 1/2
        p = np.linalg.norm(branch) ** 2
        assert np.isclose(p, 0.25)                           # each answer a quarter
        total += p
        if (m1, m2) != (0, 0):
            assert not np.allclose(branch, 0.5 * psi)        # ... and turned, not yet the message
    assert np.isclose(total, 1.0)
    # the turn on answer 11 is X Z (spin, then flip), not Z X: they differ by a sign
    b11 = (kron(bell(1, 1).conj().reshape(1, 4), I2) @ lhs).ravel()
    assert not np.allclose(b11, 0.5 * Z @ X @ psi)
    assert not np.allclose(b11, 0.5 * psi.conj())            # the message, not its conjugate

    # Result 10.4: before the letter Bob holds 1/2 I, for EVERY input, pure or mixed
    Phi = np.outer(BELL, BELL.conj())
    rho = random_density(rng)
    assert not np.allclose(rho, rho.conj())                                  # non-real
    assert np.real(np.trace(rho @ rho)) < 1 - 1e-6                           # genuinely mixed
    pure = np.outer(psi, psi.conj())
    for inp in (rho, pure):
        R = np.kron(inp, Phi)                                                # on 1, a, c
        bob = sum(kron(bell(*k).conj().reshape(1, 4), I2) @ R @ kron(bell(*k).reshape(4, 1), I2)
                  for k in KS)
        assert np.allclose(bob, I2 / 2)
        # the proof's own line: 1/4 (rho + Z rho Z + X rho X + XZ rho ZX) = 1/2 I
        assert np.allclose(0.25 * (inp + Z @ inp @ Z + X @ inp @ X + X @ Z @ inp @ Z @ X), I2 / 2)
        # Bob's early peek: a fair coin, whatever Alice sent
        assert np.allclose(np.real(np.diag(bob)), [0.5, 0.5])
        # but on a KNOWN answer his wire is not grey: it is 1/4 P rho P^dagger
        k = (1, 1); Pk = X @ Z
        one = kron(bell(*k).conj().reshape(1, 4), I2) @ R @ kron(bell(*k).reshape(4, 1), I2)
        assert np.allclose(one, 0.25 * Pk @ inp @ Pk.conj().T)
        assert not np.allclose(one, I2 / 8)

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the whole tale laid flat on the
# Tortoise's cloth -- the S: in with the message, round Alice's reading (the
# fourth bend's red and green on its incoming leg), back, round the pair, and
# out to Bob -- and then pulled straight,
#   \ringstring[operations]{scene}{caption}              (pdf/props/ch10.tex)
# tools/props_ch10.py transcribes the Lua (the string's layout read by the
# shared cloth kit, tools/props_cloth.py); the calls are read out of this
# challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3, state model 6): the ring sequence
# along the pulled string, from the message end, read left to right, is the order
# in which the turns act (red X, green Z); so the pulled tale is the branch
# operator of the Bell-basis expansion, X^m2 Z^m1, for each of the four answers.
# A structural check against this file's blackboard, not a second proof.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = "...Oh. It's there before the letter — and it's no use to him."
RING = {"red": X, "green": Z}


def applied(colours):
    """The operator a ring sequence makes, read in the order the turns act."""
    R = I2
    for c in colours:
        R = RING[c] @ R
    return R


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    calls = DIALOGUE.check_layout("Second", sizes=[1, 1])
    p21, p22 = calls
    for c in calls:
        assert words_only(c.caption, forbidden=PC.CAPTION_FORBIDDEN), c.caption
        assert PC.place_ok("Second", c), c.text
    for b in DIALOGUE.blocks("Second"):
        assert len({PC.height(c) for c in b}) == 1                # no block mixes frames
    assert [c.macro for c in calls] == ["ringstring"] * 2
    assert [c.keyed for c in calls] == [False, False]             # the string was keyed at P1.3
    assert [(c.opt, c.args[0]) for c in calls] == [("", "the tale, both up"), ("pull", "the tale, both up")]
    assert [c.caption for c in calls] == ["in, round, back, round and out", "pulled straight"]
    s21, s22 = (PC.picture(c) for c in calls)
    a1, a2 = DIALOGUE.anchors("Second")
    f1, f2 = DIALOGUE.following("Second")
    assert QUOTE in DIALOGUE.section("Second")                    # the docstring's line

    # P2.1: the tale laid flat, an S -- Alice's rings on the incoming leg, red close in
    # by the bend, green further out: exactly the fourth little bend's two rings, kept back
    assert s21.dump == "tale S; incoming leg from the bend [red green]"
    assert PC.bends_of(s21.cloth) == 2 and len(s21.cloth["cords"]) == 1 and s21.frame == 1.78
    p13 = DIALOGUE.blocks("First")[2][0]
    assert p13.args == ("the four bends",)
    assert PC.picture(p13).state.bends[3][1] == s21.state.alice   # the fourth bend's red, green
    assert "She gathers her first drawings" in a1 and "all but the fourth bend's red and green" in a1
    assert "the red close in by the bend and the green further out" in a1
    assert "round the pair, and out to Bob" in a1
    assert f1.startswith("**Achilles:** Let me work it through.")

    # P2.2: pulled straight -- the bends gone, the rings riding with the string, green
    # nearer the message, red nearer Bob, both left of the middle; the same frame and grid
    assert s22.dump == "tale straight; from the message end [alice green, alice red]"
    assert PC.bends_of(s22.cloth) == 0 and len(s22.cloth["cords"]) == 1
    assert (s22.frame, s22.cloth["cols"], s22.cloth["rows"]) == (s21.frame, 7, 3)   # no longer, no shorter
    assert [t["x"] for t in s22.cloth["things"]] == [1, 2]                     # left of the middle (3)
    assert [t["c"] for t in s22.cloth["things"]] == ["green", "red"]
    assert "only the bends are gone" in a2
    assert "a green ring nearer the message and a red ring nearer Bob" in a2
    assert f2.startswith("**Achilles:** It's at Bob's end.")

    # the pulled tale IS the branch: for each of the four answers the rings, read from
    # the message end as the order of application, are X^m2 Z^m1 (a green, then a red:
    # first Z, then X), and the branch of the Bell-basis expansion is 1/2 X^m2 Z^m1 psi
    # on a NON-REAL message
    rng = np.random.default_rng(1012)
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())
    whole = kron(psi, BELL).ravel()
    bob = np.zeros((2, 2), dtype=complex)
    for p in PC.PATTERNS:
        m1, m2 = PC.bits(p)
        pulled = PC.apply_string(PC.scene_state("the tale, " + PC.patname(p)), "pull")
        seq = PC.sequence(pulled)
        assert seq == list(reversed(PC.legrings(p)))              # green first, if any
        O = applied(seq)
        assert np.allclose(O, P(X, m2) @ P(Z, m1))
        branch = (kron(bell(m1, m2).conj().reshape(1, 4), I2) @ whole).ravel()
        assert np.allclose(branch, 0.5 * O @ psi)                 # the string's rings: the branch
        if seq:
            assert not np.allclose(branch, 0.5 * psi)             # turned: not yet the message
        # the ledger: each pattern a quarter of the time
        assert np.isclose(np.linalg.norm(branch) ** 2, 0.25)
        bob += np.outer(branch, branch.conj())
    # Bob's early peek, before the letter: a coin, high or low a half each, whatever was sent
    assert np.allclose(bob, I2 / 2) and np.allclose(np.real(np.diag(bob)), [0.5, 0.5])
    # the pull is drawn once: the Lua refuses a second pull, and rings meeting on the S
    assert PC.faults(PC.ringstring, "pull, pull", "the tale, both up") == ["pull-twice"]
    assert PC.faults(PC.ringstring, "kin to kin", "the tale, both up") == ["meet-early"]


# --- pictures -----------------------------------------------------------------

def snake(m1, m2):
    """One branch, drawn: the message wire (row 0) through the reading's rings (green
    m1*pi, then red m2*pi) into the cap on rows 0-1, back through the cup on rows 1-2,
    and out along Bob's wire (row 2).  No correction yet.
    Scalar: PyZX's bare bends are exactly the unnormalized cap and cup; the reading's
    1/sqrt2 and the pair's 1/sqrt2 make 1/2 -> add_power(-2)."""
    p = Picture()
    cap, cup = p.bend(0.5, 3), p.bend(1.5, 1)
    p.wire(p.inp(0), p.z(0, 1, m1), p.x(0, 2, m2), cap, cup, p.out(2))
    g = p.done()
    g.scalar.add_power(-2)
    return g


def half_turned_wire(m1, m2):
    """1/2 times a straight wire still wearing Alice's rings: green m1*pi then red m2*pi,
    i.e. 1/2 X^m2 Z^m1.  add_power(-2) draws the 1/2."""
    p = Picture()
    p.wire(p.inp(0), p.z(0, 1, m1), p.x(0, 2, m2), p.out(0))
    g = p.done()
    g.scalar.add_power(-2)
    return g


def doubled_reading_forgotten(as_state=False):
    """Section 10.2's doubled picture, hand-built (pyzx 0.10.6 refuses native ground
    vertices): every quantum wire is drawn with its mirror copy.  Inputs: the message
    and its mirror (1, 1bar).  Outputs: Bob's wire and its mirror (c, cbar).
      * the pair: a cup on (a, c) and its mirror cup on (abar, cbar);
      * Alice's reading on each copy: CNOT 1 -> a, then H on 1 (a Hadamard EDGE),
        then the read-spider of Chapter 9 joining each wire to its mirror
        (r1 on 1/1bar gives the turn bit m1, r2 on a/abar the flip bit m2);
      * the record FORGOTTEN: each read-spider's classical wire ends in the phase-free
        one-legged green effect <0|+<1|.
    The Paulis, H and CNOT are real, so the mirror copy uses the same pictures.
    Scalar: each CNOT picture is CNOT/sqrt2 (two of them: add_power(+2)); the doubled
    pair is 1/2 cup (x) cup (add_power(-2)); read-spiders, forgets and Hadamard edges
    are exact.  Net add_power(0)."""
    g = zx.Graph()
    V = g.add_vertex
    i1, i1b = V(VT.BOUNDARY, 0, 0), V(VT.BOUNDARY, 1, 0)
    oc, ocb = V(VT.BOUNDARY, 4, 8), V(VT.BOUNDARY, 5, 8)
    cup, cupb = V(VT.Z, 2.5, 1), V(VT.Z, 3.5, 1)             # the pair and its mirror
    ctl, ctlb = V(VT.Z, 0, 3), V(VT.Z, 1, 3)                 # CNOT control on 1, 1bar
    tgt, tgtb = V(VT.X, 2, 3), V(VT.X, 3, 3)                 # CNOT target on a, abar
    r1, r2 = V(VT.Z, 0.5, 5), V(VT.Z, 2.5, 5)                # the two read-spiders
    f1, f2 = V(VT.Z, 0.5, 6), V(VT.Z, 2.5, 6)                # forget each outcome
    E = lambda a, b, t=ET.SIMPLE: g.add_edge((a, b), edgetype=t)
    E(i1, ctl); E(i1b, ctlb); E(ctl, tgt); E(ctlb, tgtb)
    E(cup, tgt); E(cup, oc); E(cupb, tgtb); E(cupb, ocb)
    E(ctl, r1, ET.HADAMARD); E(ctlb, r1, ET.HADAMARD)        # H on 1 and 1bar, then read m1
    E(tgt, r2); E(tgtb, r2)                                  # read m2
    E(r1, f1); E(r2, f2)                                     # the record forgotten
    if as_state:                                             # the same picture, inputs bent to outputs
        g.set_inputs(()); g.set_outputs((i1, i1b, oc, ocb))
    else:
        g.set_inputs((i1, i1b)); g.set_outputs((oc, ocb))
    g.scalar.add_power(2)        # the two CNOT pictures: x sqrt2 each
    g.scalar.add_power(-2)       # the doubled pair: 1/2 cup (x) cup
    return g


def discard_then_grey(as_state=False):
    """The right-hand side: the message discarded (a cap on 1/1bar) and Bob's half
    1/2 cup(c, cbar) -- as a doubled state, 1/2 I.  add_power(-2) draws the 1/2."""
    p = Picture()
    cap, cup = p.bend(0.5, 1), p.bend(2.5, 1)
    end = p.out if as_state else p.inp
    p.wire(end(0), cap, end(1))
    p.wire(p.out(2), cup, p.out(3))
    g = p.done()
    g.scalar.add_power(-2)
    return g


def sketchbook():
    rng = np.random.default_rng(1020)
    for m1, m2 in KS:
        g = snake(m1, m2)
        # tensor check: the S pulled straight still wears Alice's rings, times 1/2
        assert np.allclose(matrix_of(g), 0.5 * P(X, m2) @ P(Z, m1))
        assert same_picture(g, half_turned_wire(m1, m2))
        if (m1, m2) != (0, 0):
            bare = Picture(); bare.wire(bare.inp(0), bare.out(0)); bare = bare.done()
            bare.scalar.add_power(-2)
            assert not same_picture(g, bare)                  # turned: not yet the message
        # DERIVED: full_reduce yanks the S and keeps the scalar -- the rings stay on
        h = g.copy()
        simp.full_reduce(h)
        assert same_picture(h, half_turned_wire(m1, m2))

    # Result 10.4 in the doubled picture: reading with the record forgotten leaves Bob
    # exactly 1/2 I and has discarded the message (a map rho -> tr(rho) 1/2 I)
    d = doubled_reading_forgotten()
    assert same_picture(d, discard_then_grey())
    # DERIVED as well: the rules reduce the doubled reading to the discard and the grey
    # half.  The reduced picture is two bare bends, i.e. boundary-to-boundary edges, which
    # matrix_of cannot evaluate with boundaries on the input side (pyzx 0.10.6), so the
    # derivation runs on the same picture with its inputs bent round to outputs.
    h = doubled_reading_forgotten(as_state=True)
    assert same_picture(h, discard_then_grey(as_state=True))
    simp.full_reduce(h)
    assert h.num_vertices() == 4 and h.num_edges() == 2       # nothing left but two bends
    assert same_picture(h, discard_then_grey(as_state=True))
    # fed a generic non-real mixed state (doubled: vec(rho)), Bob's half is 1/2 I
    rho = random_density(rng)
    assert not np.allclose(rho, rho.conj())
    bob = (matrix_of(d) @ rho.reshape(4, 1)).reshape(2, 2)   # rows c, columns cbar
    assert np.allclose(bob, I2 / 2)


if __name__ == "__main__":
    verdict("TwoBitsEarly", blackboard, sketchbook)
