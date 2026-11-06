"""Challenge III: Twice the Pace  (doubling; the inverse transform; phase estimation; The Theorem, Twice; the master)
===============================================================================================================
"The first plain, the second lit, and the wheel lit. No half, and an odd quarter: one quarter, no more."  -- Achilles

Claim: for every unitary U with U|u> = e^{2 pi i phi}|u>, |u> normalized and phi = j/2^t,
(F_q^{-1} (x) I) prod_m Lambda_m(U^{2^{t-m}}) (H^t |0^t> (x) |u>) = |j>|u> exactly, j_1 on
the top wire (Result 14.11, in Cleve et al.'s one-network form); F_q is Shor's Eq. (4.1),
has Cleve et al.'s product form (their Eq. (4.2), Result 14.9), and is built by the
network of Hadamards and controlled turns CP(pi/2^d) with the wires reversed at the end
(Result 14.10; Shor's own order (4.4) leaves the bits reversed); and the fused master,
The Quarter Clock (t = 2, U = S, the wheel's point lit), is exactly |0>|1>|1>, with no
scalar (Result 14.18).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re
from fractions import Fraction
from functools import reduce
from itertools import product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import ket, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch14 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def Zph(theta):
    return np.diag([1, np.exp(1j * theta)])


def fourier(t):
    """Shor's Eq. (4.1): F_q |x> = q^{-1/2} sum_y e^{2 pi i x y / q} |y>, q = 2^t."""
    q = 2 ** t
    x, y = np.meshgrid(np.arange(q), np.arange(q))
    return np.exp(2j * np.pi * x * y / q) / np.sqrt(q)


def on_wire(G, m, t, extra=1):
    """G on wire m (0-based, top first) of t wires, then `extra` dims below."""
    return reduce(np.kron, [G if i == m else np.eye(2) for i in range(t)] + [np.eye(extra)])


def cp_between(m, n, alpha, t):
    """CP(alpha) between wires m and n of t wires: e^{i alpha} when both are 1."""
    d = np.ones(2 ** t, dtype=complex)
    for idx, bits in enumerate(product((0, 1), repeat=t)):
        if bits[m] and bits[n]:
            d[idx] = np.exp(1j * alpha)
    return np.diag(d)


def reverse_wires(t):
    P = np.zeros((2 ** t, 2 ** t))
    for idx, bits in enumerate(product((0, 1), repeat=t)):
        P[int("".join(map(str, bits[::-1])), 2), idx] = 1
    return P


def network(t):
    """Result 14.10: for m = 1..t, a Hadamard on wire m, then CP(pi/2^{m'-m}) between wire m
    and each lower wire m'; finally the wires reversed.  Gates in time order."""
    gates = []
    for m in range(t):
        gates.append(on_wire(H, m, t))
        for n in range(m + 1, t):
            gates.append(cp_between(m, n, np.pi / 2 ** (n - m), t))
    gates.append(reverse_wires(t))
    return gates


def run(gates, d):
    M = np.eye(d, dtype=complex)
    for G in gates:
        M = G @ M
    return M


def shor_order(t):
    """Shor's Eq. (4.4), in his letters: bit j (weight 2^j) sits on wire t-1-j from the
    top; R_j is the Hadamard on bit j; S_{j,k} (j < k) the controlled phase theta_{k-j} =
    pi/2^{k-j}.  Applied from left to right: R_{l-1}, then for j = l-2, ..., 0 the gates
    S_{j,l-1}, ..., S_{j,j+1} and then R_j."""
    w = lambda j: t - 1 - j
    gates = [on_wire(H, w(t - 1), t)]
    for j in range(t - 2, -1, -1):
        for k in range(t - 1, j, -1):
            gates.append(cp_between(w(j), w(k), np.pi / 2 ** (k - j), t))
        gates.append(on_wire(H, w(j), t))
    return gates


def controlled_on(m, V, t):
    """Lambda_m(V): V on the register (below the t counting wires) when wire m is 1."""
    d = V.shape[0]
    out = np.zeros((2 ** t * d, 2 ** t * d), dtype=complex)
    for idx, bits in enumerate(product((0, 1), repeat=t)):
        blk = V if bits[m] else np.eye(d)
        out[idx * d:(idx + 1) * d, idx * d:(idx + 1) * d] = blk
    return out


def phase_estimation(U, u, t, order=None, powers=None):
    """(F_q^{-1} (x) I) prod_m Lambda_m(U^{2^{t-m}}) (H^t|0^t> (x) |u>)."""
    d = U.shape[0]
    state = np.kron(reduce(np.kron, [H @ ket(0)] * t, np.ones(1)), u)
    order = range(t) if order is None else order
    powers = (lambda m: 2 ** (t - 1 - m)) if powers is None else powers
    for m in order:
        state = controlled_on(m, np.linalg.matrix_power(U, powers(m)), t) @ state
    return np.kron(fourier(t).conj().T, np.eye(d)) @ state


def unitary_with_eigenphase(rng, phi, d=3):
    V, _ = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    U = V @ np.diag(np.exp(2j * np.pi * np.r_[phi, rng.random(d - 1)])) @ V.conj().T
    return U, V[:, 0]


def bits_of(j, t):
    return [int(b) for b in f"{j:0{t}b}"]


# --- the blackboard ---------------------------------------------------------------------

def blackboard():
    rng = np.random.default_rng(1403)

    # --- the transform (Shor's (4.1)): unitary; F_2 = H; the t = 2 matrix of section 14.3
    for t in (1, 2, 3, 4):
        F = fourier(t)
        assert np.allclose(F.conj().T @ F, np.eye(2 ** t))
    assert np.allclose(fourier(1), H)
    F4 = fourier(2)
    assert np.allclose(2 * F4, [[1, 1, 1, 1], [1, 1j, -1, -1j], [1, -1, 1, -1], [1, -1j, -1, 1j]])
    # F_4|1> = (|0> - |1>)(|0> + i|1>)/2: the top wire a half, the lower a quarter
    assert np.allclose(F4 @ ket(0, 1), np.kron(ket(0) - ket(1), ket(0) + 1j * ket(1)) / 2)

    # --- Result 14.9, the product form (Cleve et al. (4.2)), t = 3, every x: wire m carries
    #     0.x_{t-m+1}...x_t, i.e. the fractional part of x / 2^m; the top wire only x_t
    t = 3
    for x in range(8):
        factors = [ket(0) + np.exp(2j * np.pi * ((x / 2 ** m) % 1)) * ket(1) for m in range(1, t + 1)]
        prod_state = reduce(np.kron, factors) / np.sqrt(8)
        assert np.allclose(fourier(t) @ ket(*bits_of(x, t)), prod_state)
        if x not in (0, 7):                                   # the order of the factors matters
            assert not np.allclose(fourier(t) @ ket(*bits_of(x, t)), reduce(np.kron, factors[::-1]) / np.sqrt(8))

    # --- Result 14.10, the network, for t = 2, 3, 4; its inverse is the same run backwards
    #     with every turn negated.  Without the final reversal it is NOT F_q.
    for t in (2, 3, 4):
        d = 2 ** t
        gates = network(t)
        assert np.allclose(run(gates, d), fourier(t))
        assert not np.allclose(run(gates[:-1], d), fourier(t))
        assert np.allclose(reverse_wires(t) @ run(gates[:-1], d), fourier(t))
        # run backwards, every turn negated, written out gate by gate: the crossing first,
        # then for m = t..1 the controlled turns CP(-pi/2^{m'-m}) and the Hadamard on wire m
        inverse = [reverse_wires(t)]
        for m in range(t - 1, -1, -1):
            for n in range(t - 1, m, -1):
                inverse.append(cp_between(m, n, -np.pi / 2 ** (n - m), t))
            inverse.append(on_wire(H, m, t))
        assert np.allclose(run(inverse, d), fourier(t).conj().T)
        assert np.allclose(run(inverse, d), np.linalg.inv(run(gates, d)))
        # Shor's literal order (4.4) gives F_q with its output bits reversed, which Shor lets
        # us "read ... in reverse order"
        M = run(shor_order(t), d)
        assert np.allclose(M, reverse_wires(t) @ fourier(t))
        assert not np.allclose(M, fourier(t))
        assert not np.allclose(run(shor_order(t)[::-1], d), reverse_wires(t) @ fourier(t))  # R_0 first: wrong
    # the t = 2 inverse of section 14.3: a Hadamard on top, CP(-pi/2), a Hadamard below, the crossing
    inv2 = run([on_wire(H, 0, 2), cp_between(0, 1, -np.pi / 2, 2), on_wire(H, 1, 2), reverse_wires(2)], 4)
    assert np.allclose(inv2, fourier(2).conj().T)
    assert not np.allclose(run([on_wire(H, 0, 2), cp_between(0, 1, np.pi / 2, 2), on_wire(H, 1, 2),
                                reverse_wires(2)], 4), fourier(2).conj().T)   # the turn negated

    # --- Result 14.8, doubling shifts the point: Lambda(U^{2^{t-m}}) kicks 0.j_{t-m+1}...j_t
    for t in (2, 3, 4):
        for j in range(2 ** t):
            phi, b = j / 2 ** t, bits_of(j, t)
            for m in range(1, t + 1):
                tail = sum(bit / 2 ** (i + 1) for i, bit in enumerate(b[t - m:]))
                assert np.isclose(np.exp(2j * np.pi * 2 ** (t - m) * phi), np.exp(2j * np.pi * tail))

    # --- Result 14.11, the headline: t = 1, 2, 3, every j, random complex U (3x3) and a
    #     random complex eigenvector.  Exact, scalar included; the order of the Lambda_m does
    #     not matter.
    for t in (1, 2, 3):
        for j in range(2 ** t):
            U, u = unitary_with_eigenphase(rng, j / 2 ** t)
            assert not np.allclose(U, np.diag(np.diag(U))) and not np.allclose(u, u.conj())
            out = phase_estimation(U, u, t)
            want = np.kron(ket(*bits_of(j, t)), u)
            assert np.allclose(out, want)
            assert np.allclose(phase_estimation(U, u, t, order=range(t - 1, -1, -1)), want)
            # reading the counting wires returns j with certainty
            p = (np.abs(out) ** 2).reshape(2 ** t, -1).sum(axis=1)
            assert np.isclose(p[j], 1)
            rev = int(f"{j:0{t}b}"[::-1], 2)
            if rev != j:
                # the controlled powers assigned bottom-up (the fastest listener at the
                # bottom) no longer read j with certainty
                wrong = phase_estimation(U, u, t, powers=lambda m: 2 ** m)
                assert not np.allclose(wrong, want)
                pw = (np.abs(wrong) ** 2).reshape(2 ** t, -1).sum(axis=1)
                assert pw[j] < 1 - 1e-6
            if 2 * j % 2 ** t:
                # F_q instead of F_q^{-1}: the reading is -j, not j
                d = U.shape[0]
                fwd = np.kron(fourier(t) @ fourier(t), np.eye(d)) @ out
                assert not np.allclose(fwd, want)
            # the conjugated phase is caught
            if 2 * j % 2 ** t:
                assert not np.allclose(phase_estimation(U.conj(), u.conj(), t), want)

    # --- G1: one passage is one application of the whole network, whatever the case does
    #     inside: the doubled pace is the SAME controlled box applied twice
    U, u = unitary_with_eigenphase(rng, 0.25)
    L1 = controlled_on(0, U, 1)
    assert np.allclose(controlled_on(0, U @ U, 1), L1 @ L1)
    assert np.allclose(controlled_on(0, np.linalg.matrix_power(U, 4), 1), np.linalg.matrix_power(L1, 4))
    # the whole passage is a single unitary on counting wires and register
    t, d = 2, 3
    W = np.kron(fourier(t).conj().T, np.eye(d)) @ controlled_on(0, U @ U, t) @ controlled_on(1, U, t)
    assert np.allclose(W.conj().T @ W, np.eye(2 ** t * d))

    # --- the master on the blackboard: t = 2, U = S = Z(pi/2), |u> = |1>: |01>|1>; every
    #     secret the Terms allow (a whole number of quarter-turns) is read exactly; the dark
    #     wheel |0> reads |00>|0> (the Terms, clause (iii))
    S = Zph(np.pi / 2)
    out = phase_estimation(S, ket(1), 2)
    assert np.allclose(out, ket(0, 1, 1))
    assert np.isclose(out[0b011], 1) and np.isclose(out[0b101], 0)   # top wire first, not "a half"
    for j in range(4):
        assert np.allclose(phase_estimation(Zph(2 * np.pi * j / 4), ket(1), 2), ket(*bits_of(j, 2), 1))
        assert np.allclose(phase_estimation(Zph(2 * np.pi * j / 4), ket(0), 2), ket(0, 0, 0))

    # --- Result 14.14, DOCUMENTATION ONLY (stated and cited, not proved: Cleve et al. Eq.
    #     (5.4)): for a phase that is not dyadic, the nearest t-bit reading has probability at
    #     least 4/pi^2.  Checked on random phases for t = 2..6; this is not a proof.
    for t in range(2, 7):
        q = 2 ** t
        for _ in range(40):
            phi = rng.random()
            counting = np.exp(2j * np.pi * phi * np.arange(q)) / np.sqrt(q)   # Cleve's (5.1)
            probs = np.abs(fourier(t).conj().T @ counting) ** 2
            nearest = int(np.round(phi * q)) % q
            assert probs[nearest] >= 4 / np.pi ** 2

    pictures()


# --- the pictures in the dialogue ----------------------------------------------------------

# the Genie's thirteen lines, frozen from the approved text (card section 3c), inline
# directions included, in order
GENIE = (
    "The Office of Wishes. *[He glances along the shelf to the case.]* Ah. You have the newer model. "
    "The Office is rather proud of it.",
    "One wish is one passage of the whole case, however often its wheel turns inside. What the case does "
    "inside is the case's business.",
    "Setting the knob is not a wish. Nor is any arithmetic you do afterwards with what the windows show. "
    "There is one wish this evening.",
    "Referred. *[He tilts his head, as though listening up a stairwell, and straightens at once.]* The "
    "Meta-Genie regrets. The quarter-turns are the case's to tell, and the case tells only by passage. "
    "Nothing has been spent.",
    "Referred. *[A blink.]* The reply has come down from well above the Meta-Genie: a wish to watch is a "
    "wish to see inside, and the front opens on a spent case and on no other. Your wish is untouched.",
    "No floor of the Office keeps a reply waiting. It goes all the way up, and the answer comes all the way "
    "down, at once.",
    "Referred. *[A blink.]* The reply is that there is one wish this evening. A reading that fails to give "
    "what the wisher wanted is still a spent wish; another reading would be another wish, and the Office "
    "grants one.",
    "Nobody has yet.",
    "Carrying is not a wish either. Keep it upright. The Office labels its boxes for a reason.",
    "One passage of the whole case, as it is now set. One wish.",
    "Granted.",
    "The case is spent. A secret read is no longer a secret, and the front may be opened.",
    "The wish has been spent in accordance with the Terms. The Office does not collect spent boxes.")

P32_OPS = ("push the wheel, on to the far end, tube on the upper, lay the correction, push the upper, "
           "tube on the lower, cross")


def cloth_reading(j):
    """The Third Challenge's cloth laid for a secret of j quarter-turns a tick (the lower
    listener's turn j quarters, the upper's twice that), every piece of P3.2 then worked
    by the transcription's own rules: the two listeners' readings, first window first
    (after the crossing, the lower's), as bits."""
    M = PC.cloth_setup("two listeners")
    lower, upper = M["cts"]
    assert (lower["a"], upper["a"]) == ("lower", "upper") and lower["turn"] == 6 and upper["turn"] == 12
    lower["turn"], upper["turn"] = 6 * j, 12 * j
    M = PC.cloth_apply(M, P32_OPS)
    assert not PC.TEX_ERRORS and M["crossed"]
    read = {}
    for name in ("upper", "lower"):
        e = M["cords"][name]
        assert e["c"] == "red" and e["t"] in (0, 12)                   # read: dark or lit, nothing between
        read[name] = int(e["t"] == 12)
    return read["lower"], read["upper"]


def master_names(g):
    """The master's points and hubs by their places in `master` (row, column)."""
    at = {(0, 1): "T1", (0, 5): "T2", (1, 3): "B", (2, 2): "G", (2, 0): "knob",
          (1.5, 1): "H1", (1.5, 2): "H2", (1.5, 4): "H3", (1.5, 0): "P1", (1.5, 3): "P2", (0.5, 5): "P3"}
    names = {}
    for v in g.vertices():
        if g.type(v) == VT.BOUNDARY:
            names[v] = "window %d" % (g.qubit(v) + 1)
        else:
            names[v] = at[(g.qubit(v), g.row(v))]
    return names


def pictures():
    """The Third Challenge's prop pictures (identity card, section 4), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch14.py.  The
    ties below hold the transcription against this file's blackboard and its master: a
    structural check of the pictures, not a second proof."""
    calls = DIALOGUE.check_layout("Third", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("tickcloth", True, "", ("two listeners",), "another listener above"),
        ("tickcloth", False, P32_OPS, ("two listeners",), "read, and the far ends crossed"),
        ("clockcase", True, "windows", ("knob up; granted",), "after the passage"),
        ("clockcase", True, "inside", ("knob up; granted; front open",), "the front open")]
    PC.check_captions(DIALOGUE, "Third", calls)
    PC.check_section(DIALOGUE, "Third", calls)
    case_h = round(PC.CFTOP - PC.CFBOT, 2)
    assert case_h == 2.72                                             # read from the Lua
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Third")] == [[2.2], [2.2], [case_h], [case_h]]

    # the anchors (card section 4), in dialog.md; the raw file's following paragraphs agree
    a1, a2, a3, a4 = DIALOGUE.anchors("Third")
    f1, f2, f3, f4 = DIALOGUE.following("Third")
    assert "it is the upper listener now, and the first the lower" in a1                           # D16
    assert "passing beneath the lower listener's cord, joined to it nowhere" in a1                 # D16
    assert f1.startswith("*[She pushes the lit wheel end along its cord")
    assert "the lower's end passing over the other without touching it" in a2
    assert f2.startswith("**Tortoise:** The half first, then the quarter.")
    assert "The seal down the front's right-hand edge has parted of itself, in the middle." in a3  # D24
    assert f3.startswith("**Achilles:** The first plain, the second lit, and the wheel lit.")
    assert "the one going up passing over the one going down" in a4 and "Among the rows hang red hubs" in a4  # D26
    assert f4.startswith("**Achilles:** A quarter round on the upper row's first ring.")
    assert DIALOGUE.following("Third", DIALOGUE.raw) == [f1, f2, f3, f4]
    section = DIALOGUE.section("Third")
    for phrase in (
            "She gathers the cloth clear", "close to the end-rings",                                     # D15
            "On each listener's cord the green ring slides back along it into the end-ring",              # D17
            "Then she runs the wheel's end-ring on to the wheel cord's far end.",                         # D17
            "taking a tube from the packet and sliding it onto the upper listener's end-ring, where it "
            "goes in and is gone",                                                                      # D18
            "just past its end-ring", "threaded on from the cord's loose far end",                        # D19
            "and that ring slides back along its cord into the lower listener's end-ring",                # D20
            "comes to rest just past where its green ring stood",                                         # D20
            "taking another tube from the packet and sliding it onto the lower listener's end-ring",      # D21
            "where a small oval of enamel is set",                                                        # D22
            "its front towards him, its knob on the left side, its windows down the right",               # D23
            "pasted down the join: the seal.",                                                            # D23
            "swings it open on its hinge at the left."):                                                  # D25
        assert phrase in section, phrase
    for old in ("just past its tube", "Off to one side hang red hubs", "sliding a tube onto the upper listener's ring;"):
        assert old not in section, old
    # the Genie's lines are unchanged, in both files; and no picture sits among the lamp,
    # the Genie and the meta-wishes up to `Granted.`, nor at the Escher moment
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        sec = DIALOGUE.section("Third", path)
        assert tuple(re.findall(r"^\*\*Genie:\*\* (.*)$", sec, re.M)) == GENIE
        lamp = sec.index("**Achilles:** *[reaching up, and taking the lamp down from the shelf]*")
        granted = sec.index("**Genie:** Granted.")
        escher = sec.index("*[Achilles looks up from the cloth, past the candle, to the print on the wall")
        after_escher = escher
        for _ in range(3):                                             # the moment's three paragraphs
            after_escher = sec.index("\n\n", after_escher + 2)
        places = [sec.index(c.text) for c in calls]
        assert places == sorted(places)
        assert not any(lamp < x < granted for x in places)
        assert not any(escher < x < after_escher for x in places)
        assert places[0] < escher < places[1] < lamp < granted < places[2]   # P3.3, P3.4 after `Granted.`

    p31, p32, p33, p34 = (PC.picture(c) for c in calls)

    # P3.1: the doubled listener laid -- the lower's turn a quarter, the upper's a half
    # (the wheel ticked twice for it), the wheel lit
    assert p31.dump == (
        "scene=two wheel=lit upper=green:0 lower=green:0 ct=lower-wheel:6 ct=upper-wheel:12 crossed=false | "
        "cols=8 rows=3 y=0..2 near=false folded=false; gaps=0.5 1.5; 1:end green 0 @0,0; 2:ring green 6 @4,0; "
        "3:end green 0 @0,1; 4:ring green 3 @1,1; 5:hub red 0 @2,1.5; 6:bob green 21 @3,1.5; 7:hub red 0 @5,1.5; "
        "8:bob green 18 @6,1.5; 9:end red 12 @0,2; 10:ring green 3 @1,2; 11:ring green 6 @4,2; tie 2-7 []; "
        "tie 4-5 []; tie 5-10 []; tie 7-11 []; pendant 5-6 []; pendant 7-8 []; cord loose-1 [7,0 0,0]; "
        "cord loose-3 [7,1 0,1]; cord loose-9 [7,2 0,2]")
    assert p31.labels == ("upper listener", "wheel cord", "hub", "bob", "tie")
    rings, cords = PC.cloth_drawn(p31)
    kind = {i: k for i, k, *_ in rings}
    least = 1.0
    for ck, a, b, pts, reach in cords:
        for i, rk, x, y, r, rreach in rings:
            if ck != "cord" and i not in (a, b):                       # a tie meets no ring but its two
                least = min(least, PC.point_to_polyline((x, y), pts) - rreach - reach)
            if rk in ("hub", "bob") and i not in (a, b):               # hubs and bobs clear the cords
                least = min(least, PC.point_to_polyline((x, y), pts) - rreach - reach)
    assert least > 0.03, least
    # the upper's tie runs from its ring down past the lower listener's cord to its hub,
    # between the lower's cord and the wheel cord: it passes UNDER that cord (cloth.lua draws
    # every tie before the cords), joined to it nowhere; it crosses no other cord
    upper_tie = [pts for ck, a, b, pts, _ in cords if ck == "tie" and {a, b} == {2, 7}][0]
    lower_cord = [pts for ck, a, b, pts, _ in cords if ck == "cord" and 3 in (a, b)][0]
    assert PC.crosses(upper_tie, lower_cord)
    assert [b for ck, a, b, pts, _ in cords if ck == "cord" and PC.crosses(upper_tie, pts)] == [3]
    assert kind[7] == "hub" and 1 < p31.scene["things"][6]["y"] < 2

    # P3.2: read, and the far ends crossed: the upper red at a half (lit), the lower red at
    # rest (dark) -- through the crossing the first window shows the lower's: |j1 j2>|1>
    # with j = 1, the blackboard's phase estimation of the quarter-turn
    assert p32.dump == (
        "scene=two wheel=lit upper=red:12 lower=red:0 crossed=true | cols=8 rows=3 y=0..2 near=false "
        "folded=false; gaps=0.5 1.5; 1:end red 12 @1.5,0; 2:end red 0 @0,1; 3:end red 12 @7,2; "
        "cord loose-1 [7,1 6,1 5,0 1.5,0]; cord loose-2 [7,0 6,0 5,1 0,1]; cord loose-3 [7,2 7,2]")
    for j in range(4):
        j1, j2 = bits_of(j, 2)
        assert cloth_reading(j) == (j1, j2)
        assert np.allclose(phase_estimation(Zph(2 * np.pi * j / 4), ket(1), 2), ket(j1, j2, 1))
    assert cloth_reading(1) == (0, 1) != (1, 0)                          # not read reversed
    # the fixed parts, P3.1 -> P3.2: the lower's end-ring keeps its place; the upper's rests
    # just past the correction's ring (x + 0.5); the wheel's runs on to the far end, its cord
    # kept as the zero-length record, which draws nothing; no ring is added but the
    # correction's, and it is gone again
    ends1 = {x["y"]: x["x"] for x in p31.scene["things"] if x["kind"] == "end"}
    ends2 = {x["y"]: x["x"] for x in p32.scene["things"] if x["kind"] == "end"}
    assert ends1 == {0: 0, 1: 0, 2: 0} and ends2 == {0: 1.5, 1: 0, 2: 7}
    assert [x["kind"] for x in p32.scene["things"]] == ["end"] * 3
    rings2, cords2 = PC.cloth_drawn(p32)
    assert len(p32.scene["cords"]) == 3 and len(cords2) == 2           # the record is not drawn
    (k1, _, _, c1, w1), (k2, _, _, c2, w2) = cords2
    crossings = [(s, t) for s in zip(c1, c1[1:]) for t in zip(c2, c2[1:]) if PC.crosses(s, t)]
    assert len(crossings) == 1                                         # once, at the crossing
    for (_, a, b, pts, reach) in cords2:
        own = b if a == "loose" else a
        for i, rk, x, y, r, rreach in rings2:
            if i != own:
                assert PC.point_to_polyline((x, y), pts) - rreach - reach > 0.03
    lower_cord = [d for d in p32.scene["cords"] if d["a"] == 2][0]["pts"]
    assert lower_cord == [[0, 1], [5, 1], [6, 0], [7, 0]]              # the lower's end goes up, over
    assert not PC.TEX_ERRORS

    # P3.3: the passage, the knob up: windows dll -- the master's |0>|1>|1>; with the knob
    # down (not pictured) every pane plain, |000>
    assert p33.dump == "knob=up granted=true windows=dll seal=parted front=shut"
    assert p33.labels == ("window", "wheel window", "plate", "seal")
    lit = [int(w) for w in p33.state.windows]
    assert np.allclose(phase_estimation(Zph(2 * np.pi * PC.CASE_SECRET / 4), ket(1), 2), ket(*lit))
    down = PC.case_state("windows", "knob down; granted")
    assert PC.case_dump(down) == "knob=down granted=true windows=ddd seal=parted front=shut"
    assert np.allclose(phase_estimation(Zph(2 * np.pi * PC.CASE_SECRET / 4), ket(0), 2), ket(*map(int, down.windows)))
    v = p33.shown
    assert [g for _, _, _, g in v.windows] == [False, True, True] and len(v.seal) == 4   # parted, wrapped
    # the fixed parts: in every view and state the windows, the knob and the plate keep
    # their places; `granted` changes only the panes, the seal and the lock
    for view in ("knob", "windows"):
        shown = [PC.case_view(PC.case_state(view, s)) for s in ("knob down", "knob up", "knob up; granted")]
        assert len({(w.plate, tuple((k, x, y) for k, x, y, _ in w.windows), w.knob and w.knob[:2]) for w in shown}) == 1
        assert [len(w.seal) for w in shown] == ([1, 1, 2] if view == "knob" else [2, 2, 4])
    assert [y for _, _, y, _ in v.windows] == list(PC.ROWY)
    assert PC.case_view(PC.case_state("knob", "knob up")).knob[1] == PC.ROWY[2]          # level with the wheel window
    assert not PC.TEX_ERRORS

    # P3.4: the inside -- its fourteen lengths ARE the master's fourteen edges, its collars
    # the master's two Hadamard edges, its beads the master's phases (vartheta = 1/2, the
    # quarter: a phase of p pi is a bead 12 p twenty-fourths round)
    assert p34.dump == "knob=up granted=true windows=dll seal=parted front=open"
    assert p34.labels == ("collar", "bob", "hub", "window")
    w = p34.shown
    g = master(Fraction(1, 2))
    names = master_names(g)
    edges = {frozenset((names[s], names[t])) for s, t in g.edges()}
    lengths = [(k, a, b) for k, a, b, _, _ in w.lengths]
    assert lengths == [
        ("tie", "H1", "T1"), ("tie", "H1", "G"), ("pendant", "H1", "P1"), ("tie", "H2", "G"), ("tie", "H2", "B"),
        ("pendant", "H2", "P2"), ("tie", "H3", "B"), ("tie", "H3", "T2"), ("pendant", "H3", "P3"),
        ("cord", "T1", "T2"), ("cordline", "T2", "window 2"), ("cordline", "B", "window 1"),
        ("cord", "knob", "G"), ("cord", "G", "window 3")]
    assert {frozenset((a, b)) for _, a, b in lengths} == edges and len(edges) == g.num_edges()
    hadamard = {frozenset((names[s], names[t])) for s, t in g.edges() if g.edge_type((s, t)) == ET.HADAMARD}
    assert hadamard == {frozenset(("T1", "T2")), frozenset(("B", "window 1"))}
    on = {}
    for name, x, y in w.sleeves:
        for k, a, b, pts, _ in w.lengths:
            if k.startswith("cord") and PC.point_to_polyline((x, y), list(pts)) < 1e-9:
                on[name] = frozenset((a, b))
    assert set(on.values()) == hadamard
    phase = {names[v_]: g.phase(v_) for v_ in g.vertices() if names[v_] in PC.INBEAD}
    assert {k: int(12 * phase[k]) % 24 for k in phase} == PC.INBEAD
    assert [t for _, t, *_ in w.beads] == [PC.INBEAD[k] for k in ("T1", "T2", "B", "G", "P1", "P2", "P3")]
    # the crossing: the cord going up (from the lower row's ring to the first window) and the
    # one going down cross once, at (XA + XB)/2, and the one going up is drawn after: over
    up_ = [pts for k, a, b, pts, _ in w.lengths if a == "B"][0]
    dn_ = [pts for k, a, b, pts, _ in w.lengths if a == "T2" and k == "cordline"][0]
    assert up_[2][1] > up_[1][1] and dn_[2][1] < dn_[1][1]
    assert PC.crosses(up_[1:3], dn_[1:3]) and not PC.crosses(up_[:2], dn_) and not PC.crosses(up_[2:], dn_)
    assert lengths.index(("cordline", "B", "window 1")) > lengths.index(("cordline", "T2", "window 2"))
    # no tie crosses another; each hub between the rows it ties, the hubs left to right
    ties = [list(pts) for k, a, b, pts, _ in w.lengths if k in ("tie", "pendant")]
    assert not any(PC.crosses(s, t) for i, s in enumerate(ties) for t in ties[i + 1:])
    for h in ("H1", "H2", "H3"):
        ys = [PC.IN[b][1] for k, a, b, _, _ in w.lengths if k == "tie" and a == h]
        assert min(ys) < PC.IN[h][1] < max(ys)
    assert PC.IN["H1"][0] < PC.IN["H2"][0] < PC.IN["H3"][0]
    # the backs gleam where the panes gleam; nothing on the stud's row but the knob's cord
    assert [gl for _, gl in w.backs] == [False, True, True]
    assert not PC.TEX_ERRORS


# --- the sketchbook ---------------------------------------------------------------------

def master(vartheta, wheel="lit"):
    """The Quarter Clock, fused, exactly as the card lists it.  Phases in units of pi;
    vartheta = 2 phi (1/2 for the quarter).  Two collars (yellow boxes) are Hadamard EDGES:
    each is a yellow box on one line, never a break in it.  The crossing is the output
    order: T2 runs to the SECOND output, B to the FIRST.  No scalar is added: three points
    give 2^{3/2}, three bare controlled turns 2^{-3/2}."""
    v = Fraction(vartheta)
    g = zx.Graph()
    T1 = g.add_vertex(VT.Z, 0, 1, phase=v)
    T2 = g.add_vertex(VT.Z, 0, 5, phase=Fraction(-1, 4))
    B = g.add_vertex(VT.Z, 1, 3, phase=v / 2 - Fraction(1, 4))
    if wheel == "green":
        P = g.add_vertex(VT.Z, 2, 0)                          # a green wheel point
    else:
        P = g.add_vertex(VT.X, 2, 0, phase=Fraction(1 if wheel == "lit" else 0))
    G = g.add_vertex(VT.Z, 2, 2, phase=3 * v / 2)
    R1, R2, R3 = (g.add_vertex(VT.X, 1.5, c) for c in (1, 2, 4))        # the hubs
    L1 = g.add_vertex(VT.Z, 1.5, 0, phase=-v)                          # the pendant points
    L2 = g.add_vertex(VT.Z, 1.5, 3, phase=-v / 2)
    L3 = g.add_vertex(VT.Z, 0.5, 5, phase=Fraction(1, 4))
    o1, o2, o3 = (g.add_vertex(VT.BOUNDARY, r, 8) for r in (0, 1, 2))
    g.add_edge((T1, T2), ET.HADAMARD)                                   # collar, upper row
    g.add_edge((T2, o2))                                                # crosses down
    g.add_edge((B, o1), ET.HADAMARD)                                    # collar, crosses up
    g.add_edges([(P, G), (G, o3),
                 (T1, R1), (G, R1), (G, R2), (B, R2), (T2, R3), (B, R3),    # the ties
                 (R1, L1), (R2, L2), (R3, L3)])                               # the pendants
    g.set_inputs(()); g.set_outputs((o1, o2, o3))
    return g


def unfused(vartheta):
    """The same instance gate by gate (section 14.7, Step 0): green counting points, three
    separate controlled turns (CP(2 theta) top-wheel, CP(theta) lower-wheel, CP(-pi/2)
    top-lower), the two yellow boxes as Hadamard edges, the crossing by output order."""
    v = Fraction(vartheta)
    g = zx.Graph()
    c1, c2 = g.add_vertex(VT.Z, 0, 0), g.add_vertex(VT.Z, 1, 0)          # H|0> sqrt2, each
    P = g.add_vertex(VT.X, 2, 0, phase=Fraction(1))                      # the lit wheel
    a1, w1 = g.add_vertex(VT.Z, 0, 1, phase=v), g.add_vertex(VT.Z, 2, 1, phase=v)
    a2, w2 = g.add_vertex(VT.Z, 1, 2, phase=v / 2), g.add_vertex(VT.Z, 2, 2, phase=v / 2)
    a3, b3 = g.add_vertex(VT.Z, 0, 4, phase=Fraction(-1, 4)), g.add_vertex(VT.Z, 1, 4, phase=Fraction(-1, 4))
    h1, h2, h3 = (g.add_vertex(VT.X, 1.5, c) for c in (1, 2, 4))
    p1 = g.add_vertex(VT.Z, 1.5, 0, phase=-v)
    p2 = g.add_vertex(VT.Z, 1.5, 3, phase=-v / 2)
    p3 = g.add_vertex(VT.Z, 0.5, 5, phase=Fraction(1, 4))
    o1, o2, o3 = (g.add_vertex(VT.BOUNDARY, r, 8) for r in (0, 1, 2))
    g.add_edges([(c1, a1), (a3, o2), (c2, a2), (a2, b3), (P, w1), (w1, w2), (w2, o3),
                 (a1, h1), (w1, h1), (a2, h2), (w2, h2), (a3, h3), (b3, h3),
                 (h1, p1), (h2, p2), (h3, p3)])
    g.add_edge((a1, a3), ET.HADAMARD)
    g.add_edge((b3, o1), ET.HADAMARD)
    g.set_inputs(()); g.set_outputs((o1, o2, o3))
    return g


def points_after_reduce(g):
    """full_reduce a copy; each output must hang on its own one-legged dot; return the
    phases, top output first."""
    h = g.copy()
    simp.full_reduce(h)
    assert same_picture(h, g)                                           # the value is kept
    got = []
    for o in h.outputs():
        (v,) = h.neighbors(o)
        assert h.type(v) != VT.BOUNDARY and h.vertex_degree(v) == 1
        got.append(h.phase(v))
    assert h.num_vertices() == 2 * len(h.outputs())
    return got


def transform_picture(t, inverse=False, crossed=True):
    """The transform's triangle for t = 2 (inverse=True: F_4^{-1}) or t = 3 (F_8): a yellow
    box per wire (a Hadamard edge), a controlled turn between each wire and every wire
    below it, then the crossing (the output order).  Bare gadgets: F_8 = 2^{3/2} x this,
    F_4^{-1} = sqrt2 x this."""
    g = zx.Graph()
    ins = [g.add_vertex(VT.BOUNDARY, r, 0) for r in range(t)]
    outs = [g.add_vertex(VT.BOUNDARY, r, 20) for r in range(t)]
    sign = -1 if inverse else 1
    rings = {}                                         # (wire, gadget) -> ring vertex
    pairs = [(m, n) for m in range(t) for n in range(m + 1, t)]
    col = 1
    last = {r: ins[r] for r in range(t)}
    boxed = {r: False for r in range(t)}
    for m in range(t):
        # the yellow box on wire m goes onto the next edge of wire m
        boxed[m] = True
        for n in range(m + 1, t):
            half = Fraction(sign, 2 ** (n - m + 1))    # alpha/2 for CP(sign pi/2^{n-m})
            hub = g.add_vertex(VT.X, t + 0.5, col)
            pend = g.add_vertex(VT.Z, t + 0.5, col + 0.5, phase=-half)
            g.add_edge((hub, pend))
            for r in (m, n):
                ring = g.add_vertex(VT.Z, r, col, phase=half)
                g.add_edge((last[r], ring), ET.HADAMARD if boxed[r] and r == m else ET.SIMPLE)
                if r == m:
                    boxed[m] = False
                last[r] = ring
                g.add_edge((ring, hub))
            col += 1
    # the last wire's box (and any box not yet placed) sits on the edge to the output
    for r in range(t):
        target = outs[t - 1 - r] if crossed else outs[r]
        g.add_edge((last[r], target), ET.HADAMARD if boxed[r] else ET.SIMPLE)
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs))
    return g


def scaled(g, power):
    h = g.copy()
    h.scalar.add_power(power)
    return h


def sketchbook():
    # (a) The Quarter Clock, built exactly as specified; value exactly |011>, no scalar
    g = master(Fraction(1, 2))
    vec = matrix_of(g).ravel()
    assert np.allclose(vec, ket(0, 1, 1))
    # (f) the reversal test: the reading is 01 (a quarter), not 10 (a half), top wire first
    assert np.isclose(vec[0b011], 1) and np.isclose(vec[0b101], 0)

    # (b) the line count: 2 (upper counting row) + 1 (lower counting row) + 2 (wheel row)
    #     + 6 (ties) + 3 (pendants); the vertices are eleven dots and three outputs.
    #     This is the FUSED form, as fusion leaves the gate-by-gate circuit.
    assert g.num_edges() == 14 == 2 + 1 + 2 + 6 + 3
    assert g.num_vertices() == 11 + 3
    assert sum(1 for e in g.edges() if g.edge_type(e) == ET.HADAMARD) == 2      # two collars

    # (c) every secret the Terms allow, phi = j/4, vartheta = 2 phi: exactly |j1 j2>|1>;
    #     the wheel knob dark: exactly |000> (Terms (iii)).  DERIVED for each by full_reduce:
    #     one point per output, with phase pi on exactly the 1-bits.
    for j in range(4):
        gj = master(Fraction(2 * j, 4))
        assert np.allclose(matrix_of(gj).ravel(), ket(*bits_of(j, 2), 1))
        assert points_after_reduce(gj) == bits_of(j, 2) + [1]
        gd = master(Fraction(2 * j, 4), wheel="dark")
        assert np.allclose(matrix_of(gd).ravel(), ket(0, 0, 0))
        assert points_after_reduce(gd) == [0, 0, 0]

    # (d) fusion DERIVES the master: the gate-by-gate circuit (eighteen edges) is the same
    #     picture, and spider_simp on a copy fuses it into exactly the master, edge for edge
    u = unfused(Fraction(1, 2))
    assert u.num_edges() == 18
    assert same_picture(u, g)
    h = u.copy()
    assert simp.spider_simp(h)                                          # fusion fired
    assert h.num_edges() == 14 and h.num_vertices() == 11 + 3
    assert same_picture(h, g)
    phases = lambda k: sorted((k.type(x), k.phase(x)) for x in k.vertices() if k.type(x) != VT.BOUNDARY)
    assert phases(h) == phases(g)                                       # the same dots
    for j in range(4):                                                  # every secret
        assert same_picture(unfused(Fraction(2 * j, 4)), master(Fraction(2 * j, 4)))
    # the unfused circuit IS the headline's left side: each bare gadget is CP/sqrt2, each
    # green point sqrt2 H|0>, the lit point sqrt2|1>, so the drawn picture equals the matrix
    # network with no scalar added
    S = Zph(np.pi / 2)
    net = phase_estimation(S, ket(1), 2)
    assert np.allclose(matrix_of(u).ravel(), net)

    # (e) the value DERIVED: full_reduce keeps it and leaves three one-legged dots, with
    #     phase pi on exactly the second and third outputs
    assert points_after_reduce(g) == [0, 1, 1]
    assert not same_picture(g, scaled(g, 1))                            # the scalar is exact

    # the transform's triangle (Result 14.10 in pictures): F_8 = 2^{3/2} x the bare
    # picture (add_power(3)), F_4^{-1} = sqrt2 x its picture (add_power(1)).  Non-Clifford
    # turns (pi/8), so both sides are reduced by full_reduce and compared as tensors.
    F8 = transform_picture(3)
    assert np.allclose(matrix_of(scaled(F8, 3)), fourier(3))
    assert not np.allclose(matrix_of(scaled(transform_picture(3, crossed=False), 3)), fourier(3))
    inv = transform_picture(2, inverse=True)
    assert np.allclose(matrix_of(scaled(inv, 1)), fourier(2).conj().T)
    assert not np.allclose(matrix_of(scaled(transform_picture(2), 1)), fourier(2).conj().T)
    for pic in (F8, inv):
        h = pic.copy()
        simp.full_reduce(h)
        assert same_picture(h, pic)


if __name__ == "__main__":
    verdict("QuarterClock", blackboard, sketchbook)
