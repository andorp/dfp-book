"""Challenge IV: The Crab's Ledger  (Gottesman-Knill, twice: the ledger and the shrink)
===================================================================================
"Not with five. With a hundred bells, twenty thousand one hundred marks, against a count thirty-one digits long."  -- Crab
"...Oh."  -- Achilles

Claim: a stabilizer state recorded as n signed Pauli rows of 2n+1 bits is followed exactly
through H, S, CNOT and CZ by the bit rules of Result 16.13 -- every row fixes the state after
every gate -- and a Pauli measurement is certain when it commutes with every row and even
odds when it anticommutes with one; the record is n(2n+1) bits (55 for the bow tie, 20100
for a hundred bells, against 2^100, a number of 31 digits).  In pictures a closed
quarter-turn picture shrinks to a number: the bow tie capped with green quarter-turn
effects is exactly 1 + i, by two pull-outs, each setting aside e^{i pi/4}/2 (Result 16.14);
PyZX's rules empty it, and every random closed Clifford circuit, keeping the value.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from itertools import product

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import I2, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue
import props_ch16 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j])
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)

BOWTIE = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]   # 0 left-back ... 2 knot ... 4 right-front
KNOT = 2


def on(n, ops):
    return kron(*[ops.get(w, I2) for w in range(n)])


def graph_state(n, E):
    amps = [(-1) ** sum(((x >> (n - 1 - b)) & 1) * ((x >> (n - 1 - c)) & 1) for b, c in E)
            for x in range(2 ** n)]
    return np.array(amps, dtype=complex) / 2 ** (n / 2)


def cnot(n, a, b):
    M = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for x in range(2 ** n):
        y = x ^ (1 << (n - 1 - b)) if (x >> (n - 1 - a)) & 1 else x
        M[y, x] = 1
    return M


def cz(n, a, b):
    return np.diag([(-1) ** (((x >> (n - 1 - a)) & 1) * ((x >> (n - 1 - b)) & 1))
                    for x in range(2 ** n)]).astype(complex)


class Ledger:
    """n rows, each 2n bits and a sign bit.  On one wire (x, z) is the letter i^{xz} X^x Z^z:
    I = 00, X = 10, Z = 01, Y = 11.  The rules are Aaronson and Gottesman's."""

    def __init__(self, x, z, r):
        self.x, self.z, self.r = x.copy(), z.copy(), r.copy()
        self.n = len(r)

    @classmethod
    def of_graph(cls, n, E):
        x, z = np.eye(n, dtype=int), np.zeros((n, n), dtype=int)
        for b, c in E:
            z[b, c] = z[c, b] = 1
        return cls(x, z, np.zeros(n, dtype=int))

    def h(self, a, sign=True):
        if sign:
            self.r ^= self.x[:, a] & self.z[:, a]
        self.x[:, a], self.z[:, a] = self.z[:, a].copy(), self.x[:, a].copy()

    def s(self, a, sign=True):
        if sign:
            self.r ^= self.x[:, a] & self.z[:, a]
        self.z[:, a] ^= self.x[:, a]

    def cnot(self, a, b):
        self.r ^= self.x[:, a] & self.z[:, b] & (self.x[:, b] ^ self.z[:, a] ^ 1)
        self.x[:, b] ^= self.x[:, a]
        self.z[:, a] ^= self.z[:, b]

    def cz(self, a, b):
        self.h(b)
        self.cnot(a, b)
        self.h(b)

    def row(self, i):
        letters = [np.linalg.matrix_power(X, self.x[i, w]) @ np.linalg.matrix_power(Z, self.z[i, w])
                   * 1j ** (self.x[i, w] * self.z[i, w]) for w in range(self.n)]
        return (-1) ** self.r[i] * kron(*letters)

    def text(self, i):
        name = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
        return "+-"[self.r[i]] + "".join(name[(self.x[i, w], self.z[i, w])] for w in range(self.n))

    def colours(self, i):
        name = {(0, 0): "-", (1, 0): "red", (0, 1): "green", (1, 1): "gold"}
        return [name[(self.x[i, w], self.z[i, w])] for w in range(self.n)]

    def bits(self):
        return self.x.size + self.z.size + self.r.size

    def fixes(self, psi):
        return all(np.allclose(self.row(i) @ psi, psi) for i in range(self.n))


def random_gate(rng, n):
    kind = ["h", "s", "cnot", "cz"][int(rng.integers(4))]
    if kind in ("h", "s"):
        return (kind, int(rng.integers(n)))
    a, b = rng.choice(n, size=2, replace=False)
    return (kind, int(a), int(b))


def apply(led, psi, gate):
    n = led.n
    kind = gate[0]
    if kind == "h":
        led.h(gate[1])
        return on(n, {gate[1]: H}) @ psi
    if kind == "s":
        led.s(gate[1])
        return on(n, {gate[1]: S}) @ psi
    if kind == "cnot":
        led.cnot(*gate[1:])
        return cnot(n, *gate[1:]) @ psi
    led.cz(*gate[1:])
    return cz(n, *gate[1:]) @ psi


def blackboard():
    rng = np.random.default_rng(1604)
    n = 5
    G = graph_state(n, BOWTIE)

    # --- the tray's rows: red in the bell's own cell, green for every bell tied to it
    led = Ledger.of_graph(n, BOWTIE)
    assert [led.colours(i) for i in range(n)] == [
        ["red", "green", "green", "-", "-"],       # left-back
        ["green", "red", "green", "-", "-"],       # left-front
        ["green", "green", "red", "green", "green"],   # the knot
        ["-", "-", "green", "red", "green"],       # right-back
        ["-", "-", "green", "green", "red"]]       # right-front
    assert [led.text(i) for i in range(n)] == ["+XZZII", "+ZXZII", "+ZZXZZ", "+IIZXZ", "+IIZZX"]
    assert led.fixes(G)

    # --- the one update: an over (H) on the knot swaps red and green in its column only
    after = Ledger.of_graph(n, BOWTIE)
    after.h(KNOT)
    assert [after.text(i) for i in range(n)] == ["+XZXII", "+ZXXII", "+ZZZZZ", "+IIXXZ", "+IIXZX"]
    for i in range(n):
        for w in range(n):
            if w != KNOT:
                assert after.colours(i)[w] == led.colours(i)[w]
    assert after.fixes(on(n, {KNOT: H}) @ G)
    assert not after.fixes(G)                                          # it is a new state

    # --- the ledger against the state vector through random quarter-turn circuits: every
    #     row after every gate.  From the bow tie, and from |00000> (rows Z_i).
    zero = np.zeros(2 ** n, dtype=complex)
    zero[0] = 1
    starts = [(Ledger.of_graph(n, BOWTIE), G)]
    for _ in range(4):
        starts.append((Ledger(np.zeros((n, n), int), np.eye(n, dtype=int), np.zeros(n, int)), zero))
    finals = []
    for led, psi in starts:
        assert led.fixes(psi)
        for _ in range(40):
            psi = apply(led, psi, random_gate(rng, n))
            assert led.fixes(psi)
        finals.append((led, psi))
    # (iii) a CNOT or a CZ changes each row only in the two columns of the gate
    for led, _ in finals:
        for kind in ("cnot", "cz"):
            a, b = (int(v) for v in rng.choice(n, size=2, replace=False))
            new = Ledger(led.x, led.z, led.r)
            getattr(new, kind)(a, b)
            others = [w for w in range(n) if w not in (a, b)]
            assert np.array_equal(new.x[:, others], led.x[:, others])
            assert np.array_equal(new.z[:, others], led.z[:, others])
    # the signs matter: dropping the sign flip from H's rule, or from S's, goes wrong
    for bad in ("h", "s"):
        led, psi = Ledger.of_graph(n, BOWTIE), G.copy()
        failed = False
        for _ in range(60):
            gate = random_gate(rng, n)
            if gate[0] == bad:
                getattr(led, bad)(gate[1], sign=False)
                psi = on(n, {gate[1]: H if bad == "h" else S}) @ psi
            else:
                psi = apply(led, psi, gate)
            failed = failed or not led.fixes(psi)
        assert failed

    # --- Result 16.13 (iv): a Pauli measurement.  On random ledger states, random Pauli
    #     strings M: if M anticommutes with some row the odds are exactly 1/2; if it commutes
    #     with every row, the reading is certain, +1 if M is in the group and -1 if -M is.
    letters = [I2, X, 1j * X @ Z, Z]
    seen = {"odds": 0, "certain": 0}
    for led, psi in finals:
        rows = [led.row(i) for i in range(n)]
        group = []
        for W in product((0, 1), repeat=n):
            P = np.eye(2 ** n, dtype=complex)
            for i in range(n):
                if W[i]:
                    P = P @ rows[i]
            group.append(P)
        for _ in range(30):
            M = kron(*[letters[int(c)] for c in rng.integers(4, size=n)])
            p_plus = np.linalg.norm((np.eye(2 ** n) + M) @ psi / 2) ** 2
            if any(np.allclose(M @ R, -R @ M) for R in rows):
                assert np.isclose(p_plus, 0.5)
                seen["odds"] += 1
            else:
                seen["certain"] += 1
                inside = any(np.allclose(M, P) for P in group)
                minus = any(np.allclose(-M, P) for P in group)
                assert inside != minus
                assert np.isclose(p_plus, 1.0 if inside else 0.0)
    assert seen["odds"] > 0 and seen["certain"] > 0                    # both cases were met
    # on the bow tie: K_3 reads +1 with certainty; Z on the left-back bell has odds 1/2
    K3 = Ledger.of_graph(n, BOWTIE).row(KNOT)
    assert np.isclose(np.linalg.norm((np.eye(32) + K3) @ G / 2) ** 2, 1)
    Z1 = on(n, {0: Z})
    assert np.allclose(Z1 @ Ledger.of_graph(n, BOWTIE).row(0), -Ledger.of_graph(n, BOWTIE).row(0) @ Z1)
    assert np.isclose(np.linalg.norm((np.eye(32) + Z1) @ G / 2) ** 2, 0.5)

    # --- the bit rules, written out: the letter (x, z) = i^{xz} X^x Z^z under H and S
    Yl = 1j * X @ Z
    for (x, z), (x2, z2, flip) in {(1, 0): (0, 1, 0), (0, 1): (1, 0, 0), (1, 1): (1, 1, 1)}.items():
        L = [I2, X][x] @ [I2, Z][z] * 1j ** (x * z)
        L2 = [I2, X][x2] @ [I2, Z][z2] * 1j ** (x2 * z2)
        assert np.allclose(H @ L @ H.conj().T, (-1) ** flip * L2)
    for (x, z), (x2, z2, flip) in {(1, 0): (1, 1, 0), (1, 1): (1, 0, 1), (0, 1): (0, 1, 0)}.items():
        L = [I2, X][x] @ [I2, Z][z] * 1j ** (x * z)
        L2 = [I2, X][x2] @ [I2, Z][z2] * 1j ** (x2 * z2)
        assert np.allclose(S @ L @ S.conj().T, (-1) ** flip * L2)
    assert np.allclose(S @ X @ S.conj().T, Yl) and np.allclose(S @ Yl @ S.conj().T, -X)

    # --- the count: n(2n+1) bits; eleven marks a row
    assert Ledger.of_graph(n, BOWTIE).bits() == n * (2 * n + 1) == 55
    assert 2 * n + 1 == 11 and 2 ** n == 32 == len(G)
    assert 100 * (2 * 100 + 1) == 20100
    assert len(str(2 ** 100)) == 31
    assert 2 * n * (2 * n + 1) == 110                                   # AG's tableau with destabilizers


    pictures()


# --- the prop pictures --------------------------------------------------------------------

def clip_op(colour, turn):
    """A clip (or a piece threaded on a loose cord) as an operator on its bell's wire: a
    two-legged spider of its colour with phase turn/12 (pi at a half, 12; pi/2 at a quarter
    round, 6; 3pi/2 at a quarter back, 18).  Red: |+><+| + e^{i a}|-><-|; green: diag(1, e^{i a})."""
    a = np.exp(1j * np.pi * turn / 12)
    if colour == "green":
        return np.diag([1, a])
    return (np.outer(PLUS, PLUS) + a * np.outer(MINUS, MINUS)).astype(complex)


def tray_op(T):
    """The clips on the tray as one operator, ket order = the bells' order (top wire first)."""
    ops = []
    for b in PC.BELLS:
        M = I2
        for colour, turn in T.rings[b]:
            M = clip_op(colour, turn) @ M
        ops.append(M)
    return kron(*ops)


def bow_net(B):
    """The net a \\bowcloth state stands for: a green spider per ring on the cloth, phase
    (its bead's turn)/12; a Hadamard edge per tie; an output per loose cord (a capped cord
    has none: the cap's quarter is in the ring's turn), through its piece, a two-legged
    spider, if one is threaded on.  Outputs in the bells' order.  The counters by the
    teapot are numbers, not part of the net."""
    g = zx.Graph()
    v, outs = {}, []
    for w, b in enumerate(PC.BELLS):
        R = B.rings[b]
        if R["on"]:
            v[b] = g.add_vertex(VT.Z, w, 1, phase=Fraction(R["t"], 12))
    for key in sorted(B.ties):
        a, c = key.split("-", 1)
        if a in v and c in v:
            g.add_edge((v[a], v[c]), ET.HADAMARD)
    for w, b in enumerate(PC.BELLS):
        R = B.rings[b]
        if R["on"] and R["cord"]:
            last = v[b]
            if R["piece"]:
                colour, turn = R["piece"]
                p = g.add_vertex(VT.X if colour == "red" else VT.Z, w, 2, phase=Fraction(turn, 12))
                g.add_edge((last, p))
                last = p
            o = g.add_vertex(VT.BOUNDARY, w, 3)
            g.add_edge((last, o))
            outs.append(o)
    g.set_inputs(())
    g.set_outputs(tuple(outs))
    assert g.num_edges() == len([k for k in B.ties if all(x in v for x in k.split("-", 1))]) + \
        sum(1 + bool(B.rings[b]["piece"]) for b in v if B.rings[b]["cord"])
    return g


def check_bells():
    """Ket order is the bells' order: the transcription's bells and ties are this file's
    BOWTIE, wire for wire (0 left-back, 1 left-front, 2 knot, 3 right-back, 4 right-front)."""
    assert PC.BELLS == ("lb", "lf", "knot", "rb", "rf") and PC.BELLS.index("knot") == KNOT
    assert sorted(tuple(sorted((PC.BELLS.index(a), PC.BELLS.index(b)))) for a, b in PC.TIES) == BOWTIE


def rings_in_place(pic):
    """Every ring still on the cloth lies at its grid place (BOWAT), in the canonical scene."""
    for b in PC.BELLS:
        if pic.state.rings[b]["on"]:
            assert any((t["x"], t["y"]) == tuple(PC.BOWAT[b]) and t["c"] == "green" for t in pic.cloth["things"]), b


def page_rows(block):
    """A block of the ledger's page as colours per cell ('-' for a blank cell) and signs."""
    return [([c or "-" for c in cells], turned) for cells, turned in block]


def pictures():
    """The Fourth Challenge's prop pictures (identity card, section 4), read from the
    dialogue through tools/dfp_props.py and the transcription tools/props_ch16.py.  The ties
    below hold the transcription against this file's blackboard: a structural check of the
    pictures, not a second proof."""
    check_bells()
    calls = DIALOGUE.check_layout("Fourth", sizes=[1, 1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("trayledger", True, "", ("the tray's rows",), "The tray's rows"),
        ("trayledger", False, "an over on the knot", ("the tray's rows",), "The knot's column swapped"),
        ("bowcloth", False, "cap every loose cord", ("bare",), "No loose end left"),
        ("bowcloth", False, "cap every loose cord, pull out the left-back, pull out the right-back", ("bare",),
         "Pulled out to the last")]
    PC.check_captions(DIALOGUE, "Fourth", calls)
    PC.check_section(DIALOGUE, "Fourth", calls)
    PC.check_anchors(DIALOGUE, "Fourth", ["P4.1", "P4.2", "P4.3", "P4.4"])
    ledger_h, cloth_h = round(PC.LFTOP - PC.LFBOT, 2), 2.32
    assert ledger_h == 2.61                                             # read from the Lua
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Fourth")] == [[ledger_h], [ledger_h], [cloth_h], [cloth_h]]
    p41, p42, p43, p44 = (PC.picture(c) for c in calls)
    n, G = 5, graph_state(5, BOWTIE)

    # P4.1: the tray's rows -- red X, green Z, gold Y, blank I, every sign plain: K_1 ... K_5,
    # each fixing the tray, the ledger's own rows
    assert p41.dump == "rgg--+;grg--+;ggrgg+;--grg+;--ggr+"
    led = Ledger.of_graph(n, BOWTIE)
    assert page_rows(p41.state.blocks[0]) == [(led.colours(i), bool(led.r[i])) for i in range(n)]
    letter = {"red": X, "green": Z, "gold": 1j * X @ Z, "-": I2}
    for i, (cells, turned) in enumerate(page_rows(p41.state.blocks[0])):
        Ka = (-1) ** turned * kron(*[letter[c] for c in cells])
        assert np.allclose(Ka, led.row(i)) and np.allclose(Ka @ G, G)
        assert np.allclose(Ka, on(n, {**{b: Z for b, c in enumerate(cells) if c == "green"}, i: X}))
    # read bottom-up, the first row is not the left-back's touch; with the colours read the
    # wrong way (red Z, green X) it does not fix the tray
    first = page_rows(p41.state.blocks[0])[0][0]
    assert not np.allclose(kron(*[letter[c] for c in first[::-1]]), Ledger.of_graph(n, BOWTIE).row(0))
    swap = {"red": Z, "green": X, "gold": 1j * X @ Z, "-": I2}
    assert not np.allclose(kron(*[swap[c] for c in first]) @ G, G)
    # P4.2: an over on the knot -- the rows copied with H on the knot: H K_a H^dagger, the knot's
    # column red for green and green for red, the signs plain (no gold in that column)
    assert p42.dump == "rgg--+;grg--+;ggrgg+;--grg+;--ggr+/rgr--+;grr--+;ggggg+;--rrg+;--rgr+"
    assert p42.state.blocks[0] == p41.state.blocks[0]
    led.h(KNOT)
    Hk = on(n, {KNOT: H})
    for i, (cells, turned) in enumerate(page_rows(p42.state.blocks[1])):
        assert (cells, turned) == (led.colours(i), bool(led.r[i]))
        assert np.allclose((-1) ** turned * kron(*[letter[c] for c in cells]), Hk @ Ledger.of_graph(n, BOWTIE).row(i) @ Hk.conj().T)
        assert np.allclose((-1) ** turned * kron(*[letter[c] for c in cells]) @ Hk @ G, Hk @ G)
    old, new = page_rows(p42.state.blocks[0]), page_rows(p42.state.blocks[1])
    for (c0, s0), (c1, s1) in zip(old, new):                            # "The knot's column swapped"
        assert [a for w, a in enumerate(c0) if w != KNOT] == [a for w, a in enumerate(c1) if w != KNOT]
        assert c1[KNOT] == {"red": "green", "green": "red", "-": "-"}[c0[KNOT]] and s0 == s1
    assert "gold" not in [c[KNOT] for c, _ in old]
    # the keyed words: letters clear of the page and of each other, the frame by 0.025
    assert p41.labels == ("left-back", "left-front", "knot", "right-back", "right-front", "sign")
    assert p42.labels == ()
    PC.check_keyed_words(p41)

    # P4.3: every loose cord capped -- every ring a quarter round, no loose cord: the closed
    # net with a green quarter-round cap on every output, exactly 1 + i
    assert p43.dump.split(" | ")[0] == ("lb=6 lf=6 knot=6 rb=6 rf=6 ties=knot-rb,knot-rf,lb-knot,lb-lf,lf-knot,rb-rf "
                                        "edge= aside=0")
    assert "+cord" not in p43.dump                                      # "No loose end left"
    closed = bow_net(p43.state)
    assert not closed.outputs()
    assert np.isclose(matrix_of(closed).ravel()[0], 1 + 1j)
    assert same_picture(closed, closed_net(n, BOWTIE, [Fraction(1, 2)] * n))
    assert not np.isclose(matrix_of(closed_net(n, BOWTIE, [Fraction(3, 2)] * n)).ravel()[0], 1 + 1j)   # quarter back: not
    # P4.4: the left-back and the right-back pulled out -- three lone rings, the numbers 2,
    # 1 - i and 2, two counters aside; the value is the same, with the two pull-outs' numbers
    # (e^{i pi/4}/2 each)
    assert p44.dump.split(" | ")[0] == "lb=out lf=0 knot=18 rb=out rf=0 ties= edge= aside=2"
    lone = matrix_of(bow_net(p44.state)).ravel()[0]
    assert [1 + np.exp(1j * np.pi * p44.state.rings[b]["t"] / 12) for b in ("lf", "knot", "rf")] == \
        [2, 1 + np.exp(1j * np.pi * 18 / 12), 2]
    assert np.isclose(1 + np.exp(1j * np.pi * 18 / 12), 1 - 1j)
    assert np.isclose(lone, 2 * (1 - 1j) * 2)
    pull = np.exp(1j * np.pi / 4) / 2
    assert p44.state.aside == 2 and np.isclose(lone * pull ** 2, 1 + 1j)
    assert not np.isclose(lone * pull, 1 + 1j) and not np.isclose(lone, 1 + 1j)
    assert [b for b in PC.BELLS if not p44.state.rings[b]["on"]] == ["lb", "rb"]   # "Pulled out to the last"
    assert "ties= " in p44.dump
    # the counters: two by the teapot, off the linen and inside the frame
    v = p44.shown
    assert len(v.counters) == 2 and p43.shown.counters == ()
    for cx, cy in v.counters:
        box = PC.counter_box(cx, cy)
        assert box[3] < v.linen[1] - PC.LINEN_SHADOW and box[1] > v.fbot - PC.ASIDE
    # fixed parts: the rings on the cloth stay at their places
    for p in (p43, p44):
        rings_in_place(p)
    assert not PC.TEX_ERRORS


# --- the sketchbook ----------------------------------------------------------------------

def closed_net(n, E, caps):
    """The net of E with each output capped by a green effect of phase caps[w] (unfused)."""
    g = zx.Graph()
    bells = [g.add_vertex(VT.Z, w, 1) for w in range(n)]
    for b, c in E:
        g.add_edge((bells[b], bells[c]), ET.HADAMARD)
    for w in range(n):
        c = g.add_vertex(VT.Z, w, 2, phase=Fraction(caps[w]))
        g.add_edge((bells[w], c))
    g.set_inputs(())
    g.set_outputs(())
    return g


def lone_net(phases, E, power=0, phase=0):
    """Interior green spiders with the given phases, tied by E; times sqrt2^power e^{i pi phase}."""
    g = zx.Graph()
    vs = {w: g.add_vertex(VT.Z, w, 1, phase=Fraction(p)) for w, p in phases.items()}
    for b, c in E:
        g.add_edge((vs[b], vs[c]), ET.HADAMARD)
    g.set_inputs(())
    g.set_outputs(())
    g.scalar.add_power(power)
    g.scalar.add_phase(Fraction(phase))
    return g


def sketchbook():
    q = Fraction(1, 2)
    n = 5

    # --- a spider with no legs is a number: 1 + e^{i alpha}
    for alpha, number in [(0, 2), (q, 1 + 1j), (-q, 1 - 1j), (1, 0)]:
        assert np.isclose(matrix_of(lone_net({0: alpha}, [])).item(), number)

    # --- Result 16.14: the bow tie capped with green quarter-round effects is exactly 1 + i
    P0 = closed_net(n, BOWTIE, [q] * n)
    value = matrix_of(P0).item()
    assert np.isclose(value, 1 + 1j)
    x_sum = sum(1j ** bin(x).count("1") * (-1) ** sum(((x >> (4 - b)) & 1) * ((x >> (4 - c)) & 1)
                                                      for b, c in BOWTIE) for x in range(32))
    assert np.isclose(x_sum / 8, 1 + 1j)                               # the blackboard's 32 terms
    assert not np.isclose(value, 1 - 1j)
    # the hand shrink, each picture against the one before, with its number.
    # (a) each cap fuses into its bell: five interior quarter-round spiders
    F0 = lone_net({w: q for w in range(n)}, BOWTIE)
    assert same_picture(F0, P0)
    # (b) pull out the left-back bell: the tie left-front--knot goes, both turned a quarter
    #     back (to rest); e^{i pi/4}/2 set aside (sqrt2^-2, and phase pi/4)
    F1 = lone_net({1: 0, 2: 0, 3: q, 4: q}, [(2, 3), (2, 4), (3, 4)], power=-2, phase=Fraction(1, 4))
    assert same_picture(F1, F0)
    # (c) pull out the right-back bell: the tie knot--right-front goes; the knot to a quarter
    #     back, the right-front to rest; another e^{i pi/4}/2
    F2 = lone_net({1: 0, 2: -q, 4: 0}, [], power=-4, phase=Fraction(1, 2))
    assert same_picture(F2, F1)
    # (d) nothing but lone spiders: (e^{i pi/4}/2)^2 * 2 * (1 - i) * 2 = 1 + i
    assert np.isclose((np.exp(1j * np.pi / 4) / 2) ** 2 * 2 * (1 - 1j) * 2, 1 + 1j)
    assert np.isclose(matrix_of(F2).item(), 1 + 1j)
    assert not same_picture(lone_net({1: 0, 2: -q, 4: 0}, []), F1)      # the number is needed
    # where the number comes from: sum over the pulled-out bit, four cases
    for x2, x3 in product((0, 1), repeat=2):
        lhs = 0.5 * (1 + 1j * (-1) ** (x2 + x3))
        rhs = 0.5 * np.sqrt(2) * np.exp(1j * np.pi / 4) * (-1j) ** x2 * (-1j) ** x3 * (-1) ** (x2 * x3)
        assert np.isclose(lhs, rhs)
    assert np.isclose(0.5 * np.sqrt(2) * np.exp(1j * np.pi / 4) * 2 ** -0.5, np.exp(1j * np.pi / 4) / 2)

    # --- DERIVED by PyZX's rules: spider_simp, then lcomp_simp alone empties it, value kept
    h = P0.copy()
    assert simp.spider_simp(h)
    assert same_picture(h, P0)
    assert simp.lcomp_simp(h)
    assert h.num_vertices() == 0 and np.isclose(h.scalar.to_number(), 1 + 1j)
    # the order does not matter: pull out the knot first (one lcomp at the knot) -- a
    # different picture, the ring of four at rest -- and the same 1 + i
    h = P0.copy()
    simp.spider_simp(h)
    knot = [v for v in h.vertices() if h.vertex_degree(v) == 4]
    assert len(knot) == 1 and simp.check_lcomp(h, knot[0]) and simp.lcomp(h, knot[0])
    assert same_picture(h, P0)
    assert h.num_vertices() == 4 and h.num_edges() == 4
    assert all(h.phase(v) == 0 for v in h.vertices())
    # a different picture from F1 (three ties, two quarters), though -- being closed pictures
    # of the same value -- the same number
    assert (h.num_edges(), sorted(h.phase(v) for v in h.vertices())) != \
        (F1.num_edges(), sorted(F1.phase(v) for v in F1.vertices()))
    assert simp.pivot_simp(h)
    assert h.num_vertices() == 0 and np.isclose(h.scalar.to_number(), 1 + 1j)

    # --- with the caps at rest the closed net is exactly 1, and pivots empty it
    P = closed_net(n, BOWTIE, [0] * n)
    assert np.isclose(matrix_of(P).item(), 1)
    h = P.copy()
    simp.spider_simp(h)
    assert simp.pivot_simp(h)
    assert h.num_vertices() == 0 and np.isclose(h.scalar.to_number(), 1)

    # --- random closed quarter-turn circuits: <0000| C |0000>, shrunk to a number
    for seed in range(1, 13):
        c = zx.generate.cliffordT(4, 40, p_t=0, seed=seed)
        M = matrix_of(c)
        assert np.allclose(M.conj().T @ M, np.eye(16))                  # a circuit: unitary
        assert all(Fraction(c.phase(v)).denominator <= 2 for v in c.vertices())
        g = c.copy()
        g.apply_state("0000")
        g.apply_effect("0000")
        amp = matrix_of(g).item()
        assert np.isclose(amp, M[0, 0])                                 # the amplitude
        h = g.copy()
        simp.clifford_simp(h)
        assert h.num_vertices() == 0                                    # nothing left but a number
        assert np.isclose(h.scalar.to_number(), amp)                    # and it is the amplitude


if __name__ == "__main__":
    verdict("CrabsLedger", blackboard, sketchbook)
