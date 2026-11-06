"""Challenge III: The Letter  (Pauli corrections; the headline, The Theorem, Twice)
=================================================================================
"A whole turn is no turn."  -- Tortoise

Claim: for every one-qubit psi and each of the four answers (m1, m2),
(Z^m1 X^m2)_c (<B_{m1m2}|_{1a} (x) I_c)(psi_1 (x) Phi+_{ac}) = 1/2 psi_c exactly,
scalar included -- each answer a quarter, and after the correction (flip first,
then spin) Bob holds exactly psi; as a channel, read-send-correct-forget is the
identity, each branch exactly rho/4.  In the picture: one yank, then red meets
red and green meets green, and each pair, a whole turn, is no turn.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
import pyzx.simplify as simp

from dfp import BELL, CAP, CUP, I2, Picture, kron, ket, matrix_of, random_state, \
    same_picture, same_picture_up_to_scalar, verdict
from dfp_props import Dialogue, words_only
import props_ch10 as PC
import props_cloth as CLOTH


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


def correction(m1, m2):
    """Bob's correction for answer m1 m2, as the letter says: flip first, then spin."""
    return P(Z, m1) @ P(X, m2)


def random_unitary(rng, d=2):
    """A random unitary (QR of a complex Gaussian), generically non-real, non-symmetric."""
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_density(rng):
    M = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho = M @ M.conj().T
    return rho / np.trace(rho)


def branch(effect_row, psi):
    """Alice's reading `effect_row` (1x4, on wires 1, a) applied to psi_1 (x) Phi+_{ac}:
    what Bob's wire c holds on that branch (unnormalized: its length^2 is the probability)."""
    return (kron(effect_row, I2) @ kron(psi, BELL)).ravel()


def blackboard():
    rng = np.random.default_rng(1003)
    # conventions: the cup is UNNORMALIZED, so the bare snake is exactly the identity;
    # with the normalized pair it is 1/2 -- the 1/2 of every branch
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())                  # a NON-REAL message
    assert np.allclose((kron(CAP, I2) @ kron(psi, CUP)).ravel(), psi)          # yank: exactly psi
    assert np.allclose((kron(CAP, I2) @ kron(psi, BELL)).ravel(), psi / np.sqrt(2))

    # Result 10.5: all four answers, exact, scalar included
    total = 0.0
    for m1, m2 in KS:
        held = branch(bell(m1, m2).conj().reshape(1, 4), psi)
        out = correction(m1, m2) @ held
        assert np.allclose(out, 0.5 * psi)                    # 1/2 psi, exactly
        assert not np.allclose(out, 0.5 * psi.conj())         # the message, never its conjugate
        p = np.linalg.norm(held) ** 2
        assert np.isclose(p, 0.25)
        total += p
        # the wrong order (spin first, then flip): harmless where there is at most one
        # half-turn, a minus sign where there are two
        wrong = P(X, m2) @ P(Z, m1) @ held
        if (m1, m2) == (1, 1):
            assert np.allclose(wrong, -0.5 * psi)             # -1/2 psi on answer 11
            assert not np.allclose(wrong, 0.5 * psi)          # so not the exact equation
        else:
            assert np.allclose(wrong, 0.5 * psi)
    assert np.isclose(total, 1.0)
    # why: of the four corrections, three are their own inverses and ZX is minus its own
    assert np.allclose((Z @ X) @ (Z @ X), -I2)

    # the channel form, on a generic NON-REAL mixed input: each branch exactly rho/4
    rho = random_density(rng)
    assert not np.allclose(rho, rho.conj()) and np.real(np.trace(rho @ rho)) < 1 - 1e-6
    Phi = np.outer(BELL, BELL.conj())
    channel = np.zeros((2, 2), dtype=complex)
    for m1, m2 in KS:
        K = correction(m1, m2) @ kron(bell(m1, m2).conj().reshape(1, 4), I2)  # read, then correct
        piece = K @ np.kron(rho, Phi) @ K.conj().T
        assert np.allclose(piece, rho / 4)
        channel += piece
    assert np.allclose(channel, rho)                          # the identity channel

    # "A mirror is not a half-turn" (Section 10.4's sketchbook, and card 1c.6): replace the
    # real Paulis by a generic twisted basis U_k = Z^m1 X^m2 W, W NON-REAL and NON-SYMMETRIC
    W = random_unitary(rng)
    assert np.allclose(W.conj().T @ W, I2)
    assert not np.allclose(W, W.conj()) and not np.allclose(W, W.T)
    for m1, m2 in KS:
        U = correction(m1, m2) @ W
        # genericity that keeps the three wrong corrections wrong
        assert not np.allclose(U, U.T) and not np.allclose(U, U.conj())
        assert not np.allclose(U @ U, I2)
        E = (kron(U, I2) @ BELL).ravel()                      # |E_k> = (U_k (x) I)|Phi+>
        held = branch(E.conj().reshape(1, 4), psi)            # reading = the MIRROR <E_k|
        assert np.allclose(held, 0.5 * U.conj().T @ psi)      # 1/2 U_k^dagger psi
        assert np.allclose(U @ held, 0.5 * psi)               # correction U_k: yes
        for bad in (U.T, U.conj(), U.conj().T):               # U^T, Ubar, U^dagger: no
            assert not np.allclose(bad @ held, 0.5 * psi)
        # the ring on the OTHER leg: sliding brings it round transposed, and only now
        # is the transpose the right correction
        E2 = (kron(I2, U) @ BELL).ravel()                     # |E'_k> = (I (x) U_k)|Phi+>
        held2 = branch(E2.conj().reshape(1, 4), psi)
        assert np.allclose(U.T @ held2, 0.5 * psi)            # correction U_k^T: yes
        assert not np.allclose(U @ held2, 0.5 * psi)          # U_k: no
        assert not np.allclose(U.conj().T @ held2, 0.5 * psi)

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the letter opened at Bob's end (both
# tokens up), Bob's corrections threaded on the pulled tale as the letter says,
# and, in one motion, red meeting red and green meeting green and each lifted
# off, till the string lies bare,
#   \twotokens[place]{pattern}{caption}, \ringstring[operations]{scene}{caption}
#                                                         (pdf/props/ch10.tex)
# tools/props_ch10.py transcribes the Lua (the string's layout read by the
# shared cloth kit, tools/props_cloth.py); the calls are read out of this
# challenge's section of dialog.md, never copied.
#
# The state model (identity card, section 3, state model 6): the ring sequence
# from the message end, read left to right, is the order the turns act in (red
# X, green Z).  Alice's rings are the branch's X^m2 Z^m1, Bob's the correction
# Z^m1 X^m2 (flip first, then spin); together they make the identity, and like
# rings meet and come off till the string is the bare pulled tale -- for every
# one of the four letters.  Spin first, nothing meets, and the product is -I.
# A structural check against this file's blackboard, not a second proof.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = "A whole turn is no turn."
RING = {"red": X, "green": Z}
CLEARING = {"both down": [], "brass up": ["red to red"], "copper up": ["green to green"],
            "both up": ["red to red", "green to green"]}       # l.379's moves, letter by letter


def applied(colours):
    """The operator a ring sequence makes, read in the order the turns act."""
    R = I2
    for c in colours:
        R = RING[c] @ R
    return R


def tale(p, *ops):
    M = PC.scene_state("the tale, " + PC.patname(p))
    for op in ops:
        M = PC.apply_string(M, op)
    return M


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    calls = DIALOGUE.check_layout("Third", sizes=[1, 1, 2])
    p31, p32, p33, p34 = calls
    for c in calls:
        assert words_only(c.caption, forbidden=PC.CAPTION_FORBIDDEN), c.caption
        assert PC.place_ok("Third", c), c.text
    for b in DIALOGUE.blocks("Third"):
        assert len({PC.height(c) for c in b}) == 1                # no block mixes frames
    assert PC.height(p33) == PC.height(p34) == 1.78
    assert [c.macro for c in calls] == ["twotokens"] + ["ringstring"] * 3
    assert not any(c.keyed for c in calls)                        # tokens and string keyed in the First
    assert [(c.opt, c.args[0]) for c in calls] == [
        ("letter", "both up"), ("pull, corrections", "the tale, both up"),
        ("pull, corrections, red to red", "the tale, both up"),
        ("pull, corrections, kin to kin", "the tale, both up")]
    assert [c.caption for c in calls] == [
        "opened at Bob's end", "as the letter says", "red to red, lifted off", "green to green, and bare"]
    t31, s32, s33, s34 = (PC.picture(c) for c in calls)
    a1, a2, a3 = DIALOGUE.anchors("Third")
    f1, f2, f3 = DIALOGUE.following("Third")
    assert QUOTE in DIALOGUE.section("Third")                     # the docstring's line

    # P3.1: the letter opened at Bob's end -- brass up, copper up, as laid in; the very
    # pattern of the tale the Second pulled
    assert (t31.place, t31.pattern, t31.labels) == ("letter", PC.Pattern(True, True), ())
    assert t31.frame == PC.TOKENS_FRAME
    assert PC.picture(DIALOGUE.blocks("Second")[1][0]).state.pattern == t31.pattern
    assert a1.startswith("**Achilles:** *[opening it]* Brass up: flip.") and "brass first" in a1
    assert f1.startswith("**Tortoise:** Let Bob draw corrections")

    # P3.2: Bob's rings threaded from his end as the letter says -- red first, a little way
    # below Alice's red, then green: green, red | red, green
    assert s32.dump == "tale straight; from the message end [alice green, alice red, bob red, bob green]"
    assert [(t["x"], t["c"]) for t in s32.cloth["things"]] == \
        [(1, "green"), (2, "red"), (4, "red"), (5, "green")]         # facing across the middle (3)
    assert "facing each other across the middle" in a2 and "green, red, red, green" in a2
    assert f2.startswith("**Tortoise:** Let Bob draw kin to kin.")

    # P3.3 | P3.4: red to red, lifted off; then green to green, and the string bare -- no
    # longer or shorter: exactly the pulled tale with no reading at all (both down)
    assert s33.dump == "tale straight; from the message end [alice green, bob green]"
    assert s34.dump == "tale straight; from the message end []"
    bare = tale(PC.Pattern(False, False), "pull")
    assert CLOTH.same(s34.layout, PC.layout(bare)[0])             # one arrangement
    assert CLOTH.dump(s34.cloth) == CLOTH.dump(PC.ringstring("pull", "the tale, both down").cloth)
    assert s34.frame == s33.frame == s32.frame
    assert "in her pouch" in a3 and "no longer or shorter than before" in a3
    assert "no green between the two reds" in a3
    assert f3 == "**Tortoise:** A whole turn is no turn."

    # the corrections, every letter: Bob's rings are the pattern's own (a red iff brass
    # up, a green iff copper up, red nearer Alice's), they make the file's correction
    # Z^m1 X^m2, and the whole string makes Z^m1 X^m2 X^m2 Z^m1 = I; the l.379 moves
    # (or kin to kin) leave the bare pulled tale, and the blackboard's branch, corrected,
    # is 1/2 psi on a NON-REAL message
    rng = np.random.default_rng(1013)
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())
    for p in PC.PATTERNS:
        m1, m2 = PC.bits(p)
        on = tale(p, "pull", "corrections")
        bob = [r.c for r in on.seq if r.who == "bob"]
        assert bob == PC.legrings(p)                              # as the letter says
        assert np.allclose(applied(bob), correction(m1, m2))      # flip first, then spin
        assert np.allclose(applied(PC.sequence(on)), I2)          # the whole string: no turn
        held = branch(bell(m1, m2).conj().reshape(1, 4), psi)
        assert np.allclose(applied(PC.sequence(on)) @ (0.5 * psi), correction(m1, m2) @ held)
        assert np.allclose(correction(m1, m2) @ held, 0.5 * psi)
        cleared = tale(p, "pull", "corrections", *CLEARING[PC.patname(p)])
        assert PC.dump(cleared) == PC.dump(bare) == "tale straight; from the message end []"
        assert CLOTH.same(PC.layout(cleared)[0], PC.layout(bare)[0])
        if bob:
            kin = PC.ringstring("pull, corrections, kin to kin", "the tale, " + PC.patname(p))
            assert kin.dump == PC.dump(cleared) and CLOTH.same(kin.layout, PC.layout(bare)[0])
        else:   # both down: nothing to meet -- kin to kin on a bare string is a fault
            assert PC.faults(PC.ringstring, "pull, corrections, kin to kin", "the tale, both down") == \
                ["kin-none"]
    assert "Every pattern, and every time the string lies clean" in DIALOGUE.section("Third")

    # spin first (modelled; the dialogue never draws it): Bob's green goes on first,
    # nothing meets its own kind, and the string's product is (XZ)(XZ) = -I -- the sign
    # the stone cannot see and the exact equation can
    wrong = tale(PC.Pattern(True, True), "pull", "corrections spin first", "kin to kin")
    assert PC.dump(wrong) == \
        "tale straight; from the message end [alice green, alice red, bob green, bob red]"
    assert np.allclose(applied(PC.sequence(wrong)), -I2)
    held = branch(bell(1, 1).conj().reshape(1, 4), psi)
    assert np.allclose(P(X, 1) @ P(Z, 1) @ held, -0.5 * psi)      # the wrong order: -1/2 psi
    assert PC.faults(PC.ringstring, "pull, corrections spin first, red to red", "the tale, both up") == \
        ["meet-blocked"]                                          # a red cannot pass a green
    assert "a green would have to get past a red to reach its partner" in DIALOGUE.section("Third")
    assert PC.faults(PC.ringstring, "pull, corrections, corrections", "the tale, both up") == \
        ["corrections-twice"]


# --- pictures -----------------------------------------------------------------

def teleport_snake(m1, m2, bob_order=("X", "Z")):
    """One branch, drawn (Section 10.6): the message (row 0) through the reading's rings,
    green m1*pi then red m2*pi, into the cap on rows 0-1; the cup on rows 1-2; Bob's
    wire (row 2) through his correction, red m2*pi then green m1*pi (flip first), out.
    bob_order=("Z", "X") draws the correction in the wrong order.
    Scalar: PyZX's bare bends are the unnormalized cap and cup, and its one-in/one-out
    pi rings are exactly X and Z; the reading's 1/sqrt2 and the pair's 1/sqrt2 make the
    drawn 1/2 -> add_power(-2)."""
    p = Picture()
    cap, cup = p.bend(0.5, 3), p.bend(1.5, 1)
    bob = [p.x(2, 2 + i, m2) if c == "X" else p.z(2, 2 + i, m1) for i, c in enumerate(bob_order)]
    p.wire(p.inp(0), p.z(0, 1, m1), p.x(0, 2, m2), cap, cup, *bob, p.out(2))
    g = p.done()
    g.scalar.add_power(-2)
    return g


def uncorrected_snake(m1, m2):
    p = Picture()
    cap, cup = p.bend(0.5, 3), p.bend(1.5, 1)
    p.wire(p.inp(0), p.z(0, 1, m1), p.x(0, 2, m2), cap, cup, p.out(2))
    g = p.done()
    g.scalar.add_power(-2)
    return g


def half_wire():
    """1/2 times a bare wire.  add_power(-2) draws the 1/2."""
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    g = p.done()
    g.scalar.add_power(-2)
    return g


def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def two_like_rings(colour):
    """Two half-turn rings of one colour side by side on a wire."""
    p = Picture()
    ring = p.z if colour == "Z" else p.x
    p.wire(p.inp(0), ring(0, 1, 1), ring(0, 2, 1), p.out(0))
    return p.done()


def sketchbook():
    # "A whole turn is no turn": two like half-turns fuse (phases add to 2pi = 0) and the
    # phase-free dot is removed by the identity rule -- a bare wire, EXACTLY, no scalar
    for colour in ("Z", "X"):
        g = two_like_rings(colour)
        assert same_picture(g, bare_wire())                   # tensor check
        assert simp.spider_simp(g)                            # fusion
        assert simp.id_simp(g)                                # identity rule
        assert g.num_vertices() == 2                          # only the two ends are left
        assert same_picture(g, bare_wire())                   # derived, scalar 1

    for m1, m2 in KS:
        g = teleport_snake(m1, m2)
        # tensor check: the headline, exact, 1/2 included
        assert np.allclose(matrix_of(g), 0.5 * I2)
        assert same_picture(g, half_wire())
        # DERIVED: full_reduce (yank, fusion, identity rule) leaves a bare wire and 1/2
        h = g.copy()
        simp.full_reduce(h)
        assert h.num_vertices() == 2 and h.num_edges() == 1    # a bare wire
        assert np.isclose(h.scalar.to_number(), 0.5)            # the drawn 1/2, kept
        assert same_picture(h, half_wire())
        # without the letter the wire is still turned (except on the plain answer)
        if (m1, m2) != (0, 0):
            assert not same_picture(uncorrected_snake(m1, m2), half_wire())
        # the wrong order: the same on three answers; on answer 11 the rings would have
        # to pass a ring of the other colour, which costs a sign
        wrong = teleport_snake(m1, m2, bob_order=("Z", "X"))
        if (m1, m2) == (1, 1):
            assert not same_picture(wrong, half_wire())
            assert same_picture_up_to_scalar(wrong, half_wire())
            assert np.allclose(matrix_of(wrong), -0.5 * I2)
        else:
            assert same_picture(wrong, half_wire())


if __name__ == "__main__":
    verdict("Teleportation", blackboard, sketchbook)
