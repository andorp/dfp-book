"""Challenge I: Every Card Balances  (soundness; the mirror is the dagger)
=========================================================================
"Whatever the pictures say, the numbers say."  -- Achilles
"Hold any of them to the lamp, and what you see from behind is a card of the deck."  -- Tortoise

Claim: every card of the deck holds exactly as matrices, number included, for every choice of
its turns (Result 19.2 (a)); a card used inside any surrounding picture keeps the whole equal
(Result 19.2 (b)); and the mirror of a picture -- reflected left to right, every turn reversed,
every box replaced by its mirror -- has the adjoint as its matrix (Result 19.1), so every card
seen through is again a true card, with its number conjugated.  Soundness is exemplified here
card by card; the deck is the book's, listed in Section 19.1 with its Rosetta rows.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction
from functools import reduce

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import CAP, CUP, I2, ket, kron, matrix_of, random_matrix, same_picture, verdict
from dfp_props import Dialogue
import props_ch19 as PC

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)

PI = np.pi
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
SWAP = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
PLUS, MINUS = (ket(0) + ket(1)) / np.sqrt(2), (ket(0) - ket(1)) / np.sqrt(2)


def power(v, n):
    return reduce(np.kron, [v] * n, np.array([1.0 + 0j]))


def zsp(m, k, a):
    """Green spider, m legs in, k out, turned a (in units of pi): |0..0><0..0| + e^{i pi a}|1..1><1..1|."""
    return (np.outer(power(ket(0), k), power(ket(0), m))
            + np.exp(1j * PI * float(a)) * np.outer(power(ket(1), k), power(ket(1), m)))


def xsp(m, k, a):
    """Red spider, built from |+>, |-> on its own (row 54), NOT as H Z H: colour change is a card."""
    return (np.outer(power(PLUS, k), power(PLUS, m).conj())
            + np.exp(1j * PI * float(a)) * np.outer(power(MINUS, k), power(MINUS, m).conj()))


# ----------------------------------------------------------------- pictures as layers of pieces
# A side of a card is a list of layers, read left to right; a layer is a list of pieces stacked
# top wire first.  Pieces: ("Z", m, k, a), ("X", m, k, a), ("H",), ("I",), ("SW",), ("cup",),
# ("cap",), ("box", a, b, c) -- a generic box, green a, red b, green c on one wire.  Turns are
# Fractions, in units of pi.

LEGS = {"H": (1, 1), "I": (1, 1), "SW": (2, 2), "cup": (0, 2), "cap": (2, 0), "box": (1, 1)}


def legs(p):
    return (p[1], p[2]) if p[0] in "ZX" else LEGS[p[0]]


def matrix_piece(p):
    kind = p[0]
    if kind == "Z":
        return zsp(*p[1:])
    if kind == "X":
        return xsp(*p[1:])
    if kind == "box":
        a, b, c = p[1:]
        return zsp(1, 1, c) @ xsp(1, 1, b) @ zsp(1, 1, a)
    return {"H": H, "I": I2, "SW": SWAP, "cup": CUP.reshape(4, 1), "cap": CAP}[kind]


def evaluate(layers):
    """The matrix of a side: one after another is the product, side by side the Kronecker product."""
    out = None
    for layer in layers:
        L = kron(*[matrix_piece(p) for p in layer])
        out = L if out is None else L @ out
    return out


def mirror_piece(p):
    kind = p[0]
    if kind in "ZX":
        return (kind, p[2], p[1], -p[3])            # legs swap sides, the turn reverses
    if kind == "box":
        return ("box", -p[3], -p[2], -p[1])        # the box drawn the other way about
    return {"cup": ("cap",), "cap": ("cup",)}.get(kind, p)


def mirror(layers):
    """Seen through at the lamp: left and right exchanged, every turn the other way round."""
    return [[mirror_piece(p) for p in layer] for layer in reversed(layers)]


def reflect_only(layers):
    """A WRONG see-through that swaps left and right but forgets to reverse the turns."""
    def piece(p):
        if p[0] in "ZX":
            return (p[0], p[2], p[1], p[3])
        if p[0] == "box":
            return ("box", p[3], p[2], p[1])
        return mirror_piece(p)
    return [[piece(q) for q in layer] for layer in reversed(layers)]


def legs_of(side):
    return (sum(legs(p)[0] for p in side[0]), sum(legs(p)[1] for p in side[-1]))


def norm_turn(p):
    if p[0] in "ZX":
        return p[:3] + (Fraction(p[3]) % 2,)
    if p[0] == "box":
        return ("box",) + tuple(Fraction(t) % 2 for t in p[1:])
    return p


def same_layers(s1, s2):
    return [[norm_turn(p) for p in l] for l in s1] == [[norm_turn(p) for p in l] for l in s2]


# ----------------------------------------------------------------------------------- the deck

def deck(a, b, boxes):
    """(name, left side, number, right side), for turns a, b and generic boxes.
    Numbers are complex; turns in units of pi."""
    A, B, C = boxes
    Q = Fraction(1, 2)
    cards = [
        ("boxes on different wires slide past each other", [[A, ("I",)], [("I",), B]], 1,
         [[("I",), B], [A, ("I",)]]),
        ("the crossing twice", [[("SW",)], [("SW",)]], 1, [[("I",), ("I",)]]),
        ("a box slides through the crossing", [[A, B], [("SW",)]], 1, [[("SW",)], [B, A]]),
        ("the bend pulls straight", [[("I",), ("cup",)], [("cap",), ("I",)]], 1, [[("I",)]]),
        ("the other bend pulls straight", [[("cup",), ("I",)], [("I",), ("cap",)]], 1, [[("I",)]]),
        ("sliding together", [[("Z", 1, 1, a)], [("Z", 1, 1, b)]], 1, [[("Z", 1, 1, a + b)]]),
        ("sliding together, more legs", [[("Z", 1, 2, a)], [("Z", 1, 2, b), ("I",)]], 1,
         [[("Z", 1, 3, a + b)]]),
        ("sliding together, red", [[("X", 2, 1, a)], [("X", 1, 2, b)]], 1, [[("X", 2, 2, a + b)]]),
        ("a green ring at rest is cord", [[("Z", 1, 1, 0)]], 1, [[("I",)]]),
        ("a red ring at rest is cord", [[("X", 1, 1, 0)]], 1, [[("I",)]]),
        ("the tube changes colour", [[("H",)], [("Z", 1, 1, a)], [("H",)]], 1, [[("X", 1, 1, a)]]),
        ("the tubes change colour, more legs", [[("H",)], [("Z", 1, 2, a)], [("H",), ("H",)]], 1,
         [[("X", 1, 2, a)]]),
        ("two tubes are none", [[("H",)], [("H",)]], 1, [[("I",)]]),
        ("a half passes a red turn", [[("X", 1, 1, a)], [("Z", 1, 1, 1)]], np.exp(1j * PI * float(a)),
         [[("Z", 1, 1, 1)], [("X", 1, 1, -a)]]),
        ("a half passes a red turn, other order", [[("Z", 1, 1, 1)], [("X", 1, 1, a)]],
         np.exp(1j * PI * float(a)), [[("X", 1, 1, -a)], [("Z", 1, 1, 1)]]),
        ("a red half passes a green turn", [[("Z", 1, 1, a)], [("X", 1, 1, 1)]], np.exp(1j * PI * float(a)),
         [[("X", 1, 1, 1)], [("Z", 1, 1, -a)]]),
        ("the cross undone", [[("X", 1, 2, 0), ("X", 1, 2, 0)], [("I",), ("SW",), ("I",)],
                              [("Z", 2, 1, 0), ("Z", 2, 1, 0)]], 1 / np.sqrt(2),
         [[("Z", 2, 1, 0)], [("X", 1, 2, 0)]]),
        ("two cords are none", [[("Z", 1, 2, 0)], [("X", 2, 1, 0)]], 0.5,
         [[("Z", 1, 0, 0)], [("X", 0, 1, 0)]]),
        ("a loop comes off", [[("Z", 1, 3, a)], [("I",), ("cap",)]], 1, [[("Z", 1, 1, a)]]),
        ("the Euler card (joins the deck in II)", [[("Z", 1, 1, Q)], [("X", 1, 1, Q)], [("Z", 1, 1, Q)]],
         np.exp(1j * PI / 4), [[("H",)]]),
        ("read, then let the answer go, is letting go", [[("Z", 2, 1, 0)], [("Z", 1, 0, 0)]], 1,
         [[("cap",)]]),
    ]
    for k in (2, 3, 4):                                          # copy, with its number, both colours
        for c in (0, 1):
            cards.append((f"a red point copies, c={c}, k={k}", [[("X", 0, 1, c)], [("Z", 1, k, 0)]],
                          2 ** ((1 - k) / 2), [[("X", 0, 1, c)] * k]))
            cards.append((f"a green point copies, c={c}, k={k}", [[("Z", 0, 1, c)], [("X", 1, k, 0)]],
                          2 ** ((1 - k) / 2), [[("Z", 0, 1, c)] * k]))
    return cards


def random_turn(rng):
    """A generic turn: a Fraction of pi with a non-dyadic denominator."""
    return Fraction(int(rng.integers(1, 96)), 97)


def random_box(rng):
    return ("box", random_turn(rng), random_turn(rng), random_turn(rng))


def in_company(M_, D, L, K):
    """K first, then D beside L, then M (Result 19.2 (b))."""
    return M_ @ np.kron(D, L) @ K


def blackboard():
    rng = np.random.default_rng(148)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))              # the bend is unweighted
    assert np.allclose(zsp(0, 1, 0).ravel(), ket(0) + ket(1))        # green point at rest
    assert np.allclose(xsp(0, 1, 0).ravel(), np.sqrt(2) * ket(0))    # red point at rest
    assert np.allclose(xsp(0, 1, 1).ravel(), np.sqrt(2) * ket(1))    # red point at a half
    assert np.allclose(kron(ket(0), ket(1)).ravel(), ket(0, 1))       # top wire first

    for trial in range(3):                                           # every turn: several draws
        a, b = random_turn(rng), random_turn(rng)
        boxes = [random_box(rng) for _ in range(3)]
        A = matrix_piece(boxes[0])
        assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())   # a generic box
        for name, lhs, num, rhs in deck(a, b, boxes):
            L, R = evaluate(lhs), evaluate(rhs)
            n = complex(num)
            # (a) the card holds, number included
            assert np.allclose(L, n * R), name
            if not np.isclose(n, 1):
                assert not np.allclose(L, R), name                     # ...and not without it
            # (b) in company: random NON-REAL boxes before, alongside and after, on all wires
            Kc = random_matrix(rng, 2 * L.shape[1])
            Lc = random_matrix(rng, 2)
            Mc = random_matrix(rng, 2 * L.shape[0])
            assert np.allclose(in_company(Mc, L, Lc, Kc), n * in_company(Mc, R, Lc, Kc)), name
            # Result 19.1, card by card: the see-through of each side is its adjoint
            mL, mR = evaluate(mirror(lhs)), evaluate(mirror(rhs))
            assert np.allclose(mL, L.conj().T) and np.allclose(mR, R.conj().T), name
            # ...so the card seen through is a true card, its number conjugated
            assert np.allclose(mL, np.conj(n) * mR), name

    # The three cards the Tortoise holds to the lamp.
    a, b = random_turn(rng), random_turn(rng)
    boxes = [random_box(rng) for _ in range(3)]
    D = dict((name, (lhs, num, rhs)) for name, lhs, num, rhs in deck(a, b, boxes))
    # the sliding card, seen through, is the sliding card with other turns (-b, -a)
    lhs, _, rhs = D["sliding together"]
    lhs2, _, rhs2 = dict((nm, (l, n, r)) for nm, l, n, r in deck(-b, -a, boxes))["sliding together"]
    assert same_layers(mirror(lhs), lhs2) and same_layers(mirror(rhs), rhs2)
    # the half-passing card, seen through, is the same card read in the other order, turn -a
    lhs, num, rhs = D["a half passes a red turn"]
    other = dict((nm, (l, n, r)) for nm, l, n, r in deck(-a, b, boxes))
    lhs2, num2, rhs2 = other["a half passes a red turn, other order"]
    assert same_layers(mirror(lhs), lhs2) and same_layers(mirror(rhs), rhs2)
    assert np.isclose(np.conj(num), num2)                            # the dial turned back
    # the mirror card: a picture D (Result 19.1's example: a box, then a green spider turned a,
    # one leg in and two out) is the card's one side, its see-through the other.  Seen through,
    # the card's sides are (mirror D, D): "the same card, with its two sides changed over"
    pic = [[boxes[0]], [("Z", 1, 2, a)]]
    assert same_layers(mirror(mirror(pic)), pic)
    K = matrix_piece(boxes[0])
    Z12 = zsp(1, 2, a)
    assert np.allclose(evaluate(pic), Z12 @ K)
    assert np.allclose(evaluate(mirror(pic)), K.conj().T @ zsp(2, 1, -a))   # K^dag after the spider
    assert np.allclose(evaluate(mirror(pic)), evaluate(pic).conj().T)
    # NEGATIVE: forgetting to reverse the turns gives the transpose, not the mirror
    assert np.allclose(evaluate(reflect_only(pic)), evaluate(pic).T)
    assert not np.allclose(evaluate(reflect_only(pic)), evaluate(pic).conj().T)

    # Dirac's Blackboard, three reflections kept apart, on the spider and on a general box
    Zt = np.outer(ket(0), ket(0, 0)) + np.exp(1j * PI * float(a)) * np.outer(ket(1), ket(1, 1))
    Zc = np.outer(ket(0, 0), ket(0)) + np.exp(-1j * PI * float(a)) * np.outer(ket(1, 1), ket(1))
    Zd = np.outer(ket(0), ket(0, 0)) + np.exp(-1j * PI * float(a)) * np.outer(ket(1), ket(1, 1))
    assert np.allclose(Z12.T, Zt) and np.allclose(Z12.conj(), Zc) and np.allclose(Z12.conj().T, Zd)
    assert np.allclose(Zd, zsp(2, 1, -a))                        # the mirror of a spider is a spider
    G = random_matrix(rng)
    assert not np.allclose(G.conj(), G.conj().T)                 # conjugate is not the mirror
    assert not np.allclose(G.T, G.conj().T)                      # nor is the transpose

    # the laws that tie the mirror to the two ways of combining, on generic matrices
    F, G2, F2 = random_matrix(rng), random_matrix(rng), random_matrix(rng, 4)
    assert np.allclose((F @ G2).conj().T, G2.conj().T @ F.conj().T)
    assert not np.allclose((F @ G2).conj().T, F.conj().T @ G2.conj().T)    # the order reverses
    assert np.allclose(np.kron(F, F2).conj().T, np.kron(F.conj().T, F2.conj().T))
    assert np.allclose(SWAP.conj().T, SWAP) and np.allclose(H.conj().T, H)
    assert np.allclose(CUP.reshape(4, 1).conj().T, CAP)          # the mirror of the cup is the cap

    # The Hopf card without its 1/2: feed a green point at rest into both sides
    _, lhs, num, rhs = [c for c in deck(a, b, boxes) if c[0] == "two cords are none"][0]
    g_pt = ket(0) + ket(1)
    assert np.allclose(evaluate(lhs) @ g_pt, np.sqrt(2) * ket(0))
    assert np.allclose(evaluate(rhs) @ g_pt, 2 * np.sqrt(2) * ket(0))      # without the 1/2: twice
    assert np.allclose(evaluate(lhs) @ g_pt, num * evaluate(rhs) @ g_pt)
    # a card without its number is wrong in company, too
    Kc, Lc, Mc = random_matrix(rng, 4), random_matrix(rng), random_matrix(rng, 4)
    assert not np.allclose(in_company(Mc, evaluate(lhs), Lc, Kc), in_company(Mc, evaluate(rhs), Lc, Kc))

    # causality (row 72): a unitary on a wire, its conjugate on the mirror copy, then the cap
    Q_, _ = np.linalg.qr(random_matrix(rng))
    assert np.allclose(Q_.conj().T @ Q_, I2) and not np.allclose(Q_, Q_.conj())
    assert np.allclose(CAP @ np.kron(Q_, Q_.conj()), CAP)
    assert not np.allclose(CAP @ np.kron(Q_, Q_), CAP)                     # the copy conjugates
    # and seen through: the cup, then U^dag and its conjugate, is the cup
    assert np.allclose(np.kron(Q_.conj().T, Q_.T) @ CUP, CUP)
    # the read-spider takes the diagonal of a doubled state (row 68)
    rho = random_matrix(rng)
    rho = rho @ rho.conj().T
    doubled = rho.reshape(4)                                       # sum rho_ij |i>|j>, copy second
    assert np.allclose(zsp(2, 1, 0) @ doubled, np.diag(rho))
    assert np.isclose((CAP @ doubled)[0], np.trace(rho))           # letting go is the trace

    pictures()


# ------------------------------------------------------------------- the prop pictures (cards)

FIRST_DUMPS = ("half passing face | left: g12 r6 | right: r18 g12 | dial=right:6 | cross=none",
               "sliding through | left: g15 | right: g18 g21 | dial=none | cross=none",
               "half passing through | left: g12 r6 | right: r18 g12 | dial=left:18 | cross=none")
CARD_NAMES = ("sliding", "half passing", "new", "three quarters back", "green point")


def read_dump(dump):
    """A card's dump: (left tokens, right tokens, (dial side, dial turn) or None, cross side)."""
    head, left, right, dial, cross = dump.split(" | ")
    assert left.startswith("left: ") and right.startswith("right: ")
    d = dial[len("dial="):]
    return (left[6:].split(), right[6:].split(), None if d == "none" else (d.split(":")[0], int(d.split(":")[1])),
            cross[len("cross="):])


def token(tok):
    """A drawn piece: ('ring' | 'point', colour, turn in twenty-fourths, mirrored) or ('tube',)."""
    if tok == "tube":
        return ("tube",)
    point = tok.startswith("p")
    t = tok[1:] if point else tok
    mirrored = t.endswith("<")
    t = t.rstrip("<")
    assert t[0] in "gr" and (point or not mirrored)
    return ("point" if point else "ring", "Z" if t[0] == "g" else "X", int(t[1:]), mirrored)


def side_matrix(tokens):
    """A card side, read left to right, built from this file's own spiders: a ring a two-legged
    spider turned t/12 (in units of pi; a bead's turn is t twenty-fourths of a whole turn), a tube
    the yellow box, a point a one-legged spider (an effect when it is drawn facing the other way)."""
    M = None
    for p in map(token, tokens):
        if p[0] == "tube":
            P = H
        else:
            kind, c, t, mirrored = p
            m, k = ((1, 0) if mirrored else (0, 1)) if kind == "point" else (1, 1)
            P = (zsp if c == "Z" else xsp)(m, k, Fraction(t, 12))
        M = P if M is None else P @ M
    return M


def side_graph(tokens, phase=0):
    """The same side as a chain on one wire (card section 4): a ring a two-legged spider, a tube
    a yellow edge, a point a one-legged spider; the number written on the scalar."""
    g = zx.Graph()
    pieces = [token(t) for t in tokens]
    prev, et, col, n = None, ET.SIMPLE, 0, 0
    if not (pieces[0][0] == "point" and not pieces[0][3]):
        prev = g.add_vertex(VT.BOUNDARY, 0, col)
        g.set_inputs((prev,))
    for p in pieces:
        if p[0] == "tube":
            et = ET.HADAMARD if et == ET.SIMPLE else ET.SIMPLE
            continue
        col += 1
        v = g.add_vertex(VT.Z if p[1] == "Z" else VT.X, 0, col, Fraction(p[2], 12))
        if prev is not None:
            g.add_edge((prev, v), et)
            n += 1
        prev, et = v, ET.SIMPLE
    if not (pieces[-1][0] == "point" and pieces[-1][3]):
        o = g.add_vertex(VT.BOUNDARY, 0, col + 1)
        g.add_edge((prev, o), et)
        n += 1
        g.set_outputs((o,))
    assert g.num_edges() == n
    g.scalar.add_phase(Fraction(phase))
    return g


def dial_number(d):
    return np.exp(1j * PI * d / 12)


def check_card_views(name):
    """The see-through's rule on one card (card section 4, D2, D16): the card held to the lamp is
    the card's mirror -- every drawn part reflected left to right about the card's middle -- and
    its two sides are the adjoints of the face's, crossed over; the dial crosses with its picture
    and reverses; on both views the dial multiplies the picture it stands under."""
    face, thru = PC.rulecard(name, "face"), PC.rulecard(name, "through")
    fl, fr, fd, fc = read_dump(face.dump)
    tl, tr, td, tc = read_dump(thru.dump)
    Fl, Fr, Tl, Tr = map(side_matrix, (fl, fr, tl, tr))
    # the sides: the adjoints of the face's, crossed over (in numbers and in pictures)
    assert np.allclose(Tl, Fr.conj().T) and np.allclose(Tr, Fl.conj().T), name
    assert same_picture(side_graph(tl), side_graph(fr).adjoint()), name
    assert same_picture(side_graph(tr), side_graph(fl).adjoint()), name
    for T_, F_ in ((Tl, Fr), (Tr, Fl)):
        if not np.allclose(F_.T, F_.conj().T):
            assert not np.allclose(T_, F_.T), name                    # not the bare reflection
    assert not np.allclose(Tl, Fl.conj().T) or np.allclose(Fl, Fr), name   # crossed over, not in place
    # the dial: under the right-hand picture on the face, crossed to the left and reversed through
    assert (fd is None) == (td is None), name
    for (L_, R_, d, lt, rt) in ((Fl, Fr, fd, fl, fr), (Tl, Tr, td, tl, tr)):
        if d is None:
            assert np.allclose(L_, R_), name
            assert same_picture(side_graph(lt), side_graph(rt)), name
            continue
        under, other = (R_, L_) if d[0] == "right" else (L_, R_)
        assert np.allclose(dial_number(d[1]) * under, other), name     # the dial multiplies its picture
        assert not np.allclose(dial_number(d[1]) * other, under), name # not the other one
        assert not np.allclose(under, other), name                      # and it is load-bearing
        ut, ot = (rt, lt) if d[0] == "right" else (lt, rt)
        assert same_picture(side_graph(ut, Fraction(d[1], 12)), side_graph(ot)), name
        assert not same_picture(side_graph(ut), side_graph(ot)), name
    if fd is not None:
        assert fd[0] == "right" and td == ("left", (24 - fd[1]) % 24), name     # crossed, reversed
        assert not np.allclose(dial_number(fd[1]) * Tl, Tr), name               # crossed, not reversed: no
    # the cross: top right on the face, top left seen through
    assert (fc, tc) in (("none", "none"), ("right", "left")), name
    # D16, the card held to the lamp is the mirror: left and right exchanged, every drawn part
    # (ring, bead, tube, cord, dial and its needle, cross, the written squiggle) reflected
    vf, vt = face.shown, thru.shown
    flip = lambda p: (round(-p[0], 9), round(p[1], 9))
    places = lambda v: sorted((round(x, 9), p.kind, p.c) for side, x, p in v.pieces)
    assert places(vt) == sorted((round(-x, 9), k, c) for x, k, c in places(vf)), name
    beads = lambda v: [PC.bead_at(x, PC.CLY, p.t, PC.CRO) for side, x, p in v.pieces if p.kind != "tube"]
    assert sorted(flip(b) for b in beads(vf)) == sorted((round(x, 9), round(y, 9)) for x, y in beads(vt)), name
    assert sorted((round(-b, 9), round(-a, 9)) for a, b in vf.cords) == sorted((round(a, 9), round(b, 9)) for a, b in vt.cords)
    sq_f = PC.squiggle_points(0, PC.CLY, vf.squiggle)
    sq_t = PC.squiggle_points(0, PC.CLY, vt.squiggle)
    assert np.allclose([(-x, y) for x, y in sq_f], sq_t), name
    if vf.dial:
        assert np.allclose(vt.dial[:2], (-vf.dial[0], vf.dial[1])), name
        assert sorted(flip(q) for q in PC.needle(*vf.dial)) == sorted(flip(flip(q)) for q in PC.needle(*vt.dial)), name
        assert np.allclose(PC.needle(*vt.dial)[2], flip(PC.needle(*vf.dial)[2])), name     # its tip
    if vf.cross:
        assert np.allclose(vt.cross, (-vf.cross[0], vf.cross[1])), name
    # ...and the through view is NOT the face with its turns kept (a reflection that forgot to
    # turn the beads round), wherever a bead is off its rest and its half
    kept = [(round(-x, 9), PC.bead_at(-x, PC.CLY, p.t, PC.CRO)) for side, x, p in vf.pieces
            if p.kind != "tube" and p.t % 12]
    for x, b in kept:
        assert not any(np.allclose(b, PC.bead_at(xt, PC.CLY, q.t, PC.CRO)) for s_, xt, q in vt.pieces
                       if q.kind != "tube" and abs(xt - x) < 1e-9), name
    return face, thru


def pictures():
    """The First Challenge's prop pictures (identity card, section 4), read from the dialogue
    through tools/dfp_props.py and the transcription tools/props_ch19.py, and held against this
    file's deck: each card side built as a chain on one wire from its dump."""
    calls = DIALOGUE.check_layout("First", sizes=[1, 1, 1])
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("rulecard", True, "", ("half passing",), "Face up on the table"),
        ("rulecard", False, "through", ("sliding",), "Held to the lamp"),
        ("rulecard", False, "through", ("half passing",), "Read from behind")]
    PC.check_captions(DIALOGUE, "First", calls)
    PC.check_section(DIALOGUE, "First", calls)
    PC.check_anchors(DIALOGUE, "First")
    pics = [PC.picture(c) for c in calls]
    p11, p12, p13 = pics
    assert tuple(p.dump for p in pics) == FIRST_DUMPS
    # the captions are true of their dumps: face up; held to the lamp and read from behind, through
    assert (p11.state.view, p12.state.view, p13.state.view) == ("face", "through", "through")
    # one frame, every card, both views
    assert {PC.rulecard(n, v).frame for n in CARD_NAMES for v in ("face", "through")} == {p11.frame} == {1.12}

    # the see-through's rule, on every card of the deck the Tortoise draws (not only the pictured)
    for name in CARD_NAMES:
        check_card_views(name)
    crossed = [n for n in CARD_NAMES if PC.rulecard(n).state.crossed]
    assert crossed == ["three quarters back", "green point"]         # "taken from the others"

    Q = Fraction(1, 2)
    # P1.1, the half passing card face up: a green half, then a red quarter round, is the dial's
    # quarter round times a red quarter back, then a green half -- this file's deck card
    fl, fr, fd, fc = read_dump(p11.dump)
    lhs, num, rhs = [(l, n, r) for nm, l, n, r in deck(Q, Q, [random_box(np.random.default_rng(1))] * 3)
                     if nm == "a half passes a red turn, other order"][0]
    assert np.isclose(num, dial_number(fd[1])) and fd == ("right", 6)
    assert np.allclose(side_matrix(fl), evaluate(lhs)) and np.allclose(side_matrix(fr), evaluate(rhs))
    assert same_picture(side_graph(fl), picture(lhs)) and same_picture(side_graph(fr), picture(rhs))
    assert np.allclose(side_matrix(fl), 1j * side_matrix(fr))
    assert not np.allclose(side_matrix(fl), side_matrix(fr))           # the dial is load-bearing

    # P1.2, the sliding card held to the lamp: the face's sides' adjoints, crossed over, and again
    # a sliding card -- an eighth back and a quarter back make three eighths back; no dial
    face = PC.rulecard("sliding")
    sl, sr, _, _ = read_dump(face.dump)
    tl, tr, td, _ = read_dump(p12.dump)
    assert (sl, sr) == (["g3", "g6"], ["g9"]) and td is None
    assert np.allclose(side_matrix(tl), side_matrix(sr).conj().T)
    assert np.allclose(side_matrix(tr), side_matrix(sl).conj().T)
    lhs, _, rhs = [(l, n, r) for nm, l, n, r in deck(-Q, -Fraction(1, 4), [random_box(np.random.default_rng(1))] * 3)
                   if nm == "sliding together"][0]
    assert same_picture(side_graph(tr), picture(lhs)) and same_picture(side_graph(tl), picture(rhs))
    assert np.allclose(side_matrix(tl), zsp(1, 1, -Fraction(3, 4)))    # three eighths back
    assert not np.allclose(side_matrix(tl), zsp(1, 1, Fraction(3, 4)))  # not round

    # P1.3, the half passing card read from behind (D2 run): through-left is face-right's adjoint,
    # through-right face-left's, and the dial, now under the left-hand picture and pointing the
    # other way round, multiplies it: e^{-i pi/2} (face-right)^dag = (face-left)^dag
    tl, tr, td, _ = read_dump(p13.dump)
    Fl, Fr = side_matrix(fl), side_matrix(fr)
    assert td == ("left", 18)
    assert np.allclose(side_matrix(tl), Fr.conj().T) and np.allclose(side_matrix(tr), Fl.conj().T)
    assert np.allclose(np.exp(-1j * PI / 2) * Fr.conj().T, Fl.conj().T)
    assert not np.allclose(np.exp(1j * PI / 2) * Fr.conj().T, Fl.conj().T)    # the face's dial: no
    assert same_picture(side_graph(tl, Fraction(18, 12)), side_graph(tr))

    # unpictured, run: the new card is e^{i pi/4} H; three quarters back is e^{-i pi/4} H; a green
    # point a quarter round is the red point a quarter back, the whole an eighth round
    nl, nr, nd, _ = read_dump(PC.rulecard("new").dump)
    assert np.allclose(side_matrix(nl), np.exp(1j * PI / 4) * H) and nr == ["tube"] and nd == ("right", 3)
    bl, br, bd, bc = read_dump(PC.rulecard("three quarters back").dump)
    assert np.allclose(side_matrix(bl), np.exp(-1j * PI / 4) * H) and br == ["tube"] and bd == ("right", 21)
    assert bc == "right"
    gl, gr, gd, gc = read_dump(PC.rulecard("green point").dump)
    assert (gl, gr, gd, gc) == (["pg6"], ["pr18"], ("right", 3), "right")
    assert np.allclose(side_matrix(gl), np.exp(1j * PI / 4) * side_matrix(gr))
    assert np.allclose(side_matrix(gl).ravel(), ket(0) + 1j * ket(1))
    # the new card's sides are this file's Euler card
    lhs, num, rhs = [(l, n, r) for nm, l, n, r in deck(Q, Q, [random_box(np.random.default_rng(1))] * 3)
                     if nm.startswith("the Euler card")][0]
    assert same_picture(side_graph(nl), picture(lhs)) and same_picture(side_graph(nr, Fraction(1, 4)), picture(rhs, num))

    # the keyed words (P1.1): is and dial, set by the Lua, at the compiled boxes and measured
    # clear; no cross on this card, so none printed; the other pictures unkeyed
    assert p11.labels == ("is", "dial") and p12.labels == p13.labels == ()
    PC.check_compiled_boxes(p11)
    assert min(PC.check_keyed_words(p11).values()) >= PC.LETTER_FLOOR
    for name in CARD_NAMES:                                           # every card the Lua can key
        for view in ("face", "through"):
            pic = PC.rulecard(name, view, True)
            assert [w for w, *_ in PC.own_words(pic)] == ["is"] + ["dial"] * (pic.state.dial is not None) + \
                ["cross"] * pic.state.crossed
            PC.check_compiled_boxes(pic)
            assert min(PC.check_keyed_words(pic).values()) >= PC.LETTER_FLOOR

    # the Lua's refusals: an unknown card; an unknown view
    PC.TEX_ERRORS.clear()
    PC.card_state("blank card", "face")
    PC.card_state("sliding", "upside down")
    assert [k for k, _ in PC.TEX_ERRORS] == ["card", "view"]
    PC.TEX_ERRORS.clear()


# ----------------------------------------------------------------------------- the sketchbook

def picture(layers, number=1):
    """Build a side as a PyZX graph, piece by piece.  Every strand between two pieces carries a
    green ring at rest (bare cord, row 57), so no two strands between the same pieces collapse
    into one edge; the number is written on the picture's scalar."""
    g = zx.Graph()
    n_in = sum(legs(p)[0] for p in layers[0])
    ins = [g.add_vertex(VT.BOUNDARY, q, 0) for q in range(n_in)]
    open_legs = [(v, ET.SIMPLE) for v in ins]
    edges = 0

    def join(u, et, w, row, col):
        nonlocal edges
        r = g.add_vertex(VT.Z, row, col)
        g.add_edge((u, r), et)
        g.add_edge((r, w))
        edges += 2

    col = 0
    for layer in layers:
        col += 4
        new_open, row = [], 0
        for p in layer:
            m, k = legs(p)
            kind = p[0]
            if kind in "ZX":
                v = g.add_vertex(VT.Z if kind == "Z" else VT.X, row, col, phase=Fraction(p[3]))
                vin, vout = [v] * m, [(v, ET.SIMPLE)] * k
            elif kind == "box":
                z1 = g.add_vertex(VT.Z, row, col - 1, phase=Fraction(p[1]))
                x = g.add_vertex(VT.X, row, col, phase=Fraction(p[2]))
                z2 = g.add_vertex(VT.Z, row, col + 1, phase=Fraction(p[3]))
                g.add_edge((z1, x))
                g.add_edge((x, z2))
                edges += 2
                vin, vout = [z1], [(z2, ET.SIMPLE)]
            elif kind == "SW":
                v0, v1 = g.add_vertex(VT.Z, row, col), g.add_vertex(VT.Z, row + 1, col)
                vin, vout = [v0, v1], [(v1, ET.SIMPLE), (v0, ET.SIMPLE)]
            else:                                                # H, I, cup, cap: one ring at rest
                v = g.add_vertex(VT.Z, row + (m + k) / 4, col)
                vin = [v] * m
                vout = [(v, ET.HADAMARD if kind == "H" else ET.SIMPLE)] * k
            for w in vin:
                u, et = open_legs.pop(0)
                join(u, et, w, row, col - 2)
                row += 1
            row = max(row, len(new_open) + k)
            new_open += vout
        assert not open_legs
        open_legs = new_open
    outs = []
    for q, (u, et) in enumerate(open_legs):
        o = g.add_vertex(VT.BOUNDARY, q, col + 4)
        join(u, et, o, q, col + 2)
        outs.append(o)
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    assert g.num_edges() == edges                          # no strand silently dropped
    g.scalar.add_float(complex(number))
    return g


def joiner():
    """Achilles's kind of look, bare: a green spider on the top wire tied to a red one below."""
    g = zx.Graph()
    i0, i1 = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0)
    o0, o1 = g.add_vertex(VT.BOUNDARY, 0, 2), g.add_vertex(VT.BOUNDARY, 1, 2)
    z, x = g.add_vertex(VT.Z, 0, 1), g.add_vertex(VT.X, 1, 1)
    for e in ((i0, z), (z, o0), (i1, x), (x, o1), (z, x)):
        g.add_edge(e)
    g.set_inputs((i0, i1))
    g.set_outputs((o0, o1))
    assert g.num_edges() == 5
    return g


def bare_wires(g, n):
    """Every output joined straight to its own input by one plain edge, nothing else."""
    if g.num_vertices() != 2 * n or g.num_edges() != n:
        return False
    return all(g.connected(i, o) and g.edge_type(g.edge(i, o)) == ET.SIMPLE
               for i, o in zip(g.inputs(), g.outputs()))


def sketchbook():
    rng = np.random.default_rng(1480)
    a, b = random_turn(rng), random_turn(rng)
    boxes = [random_box(rng) for _ in range(3)]

    # (T1) TRANSLATION CHECK, card by card: each card's two pictures evaluated and compared.
    # Soundness IS agreement in numbers; a card cannot be derived from itself, so this block
    # evaluates rather than rewrites.  The derivations of this file are further down.
    for name, lhs, num, rhs in deck(a, b, boxes):
        L, R = picture(lhs), picture(rhs, num)
        assert same_picture(L, R), name
        assert np.allclose(matrix_of(L), evaluate(lhs)), name       # the picture is the matrix
        if not np.isclose(complex(num), 1):
            assert not same_picture(L, picture(rhs)), name          # the number is load-bearing
        # the see-through, in pictures: PyZX's mirror of each side is the side built from the
        # deck's own pieces, reflected with every turn reversed
        assert same_picture(L.adjoint(), picture(mirror(lhs))), name
        assert same_picture(R.adjoint(), picture(mirror(rhs), np.conj(complex(num)))), name

    # The context test in pictures: each card inside a non-product surrounding.  Before it,
    # a generic box on every wire and a tie from the card's last wire to a wire alongside;
    # beside it, a generic box on that wire; after it, the same again with other turns.
    def context(n):
        """On the card's n wires and one wire alongside: a generic box on every wire, then a
        look (a green spider on the card's last wire tied to a red one alongside)."""
        if n == 0:
            return [[random_box(rng)]]
        tie = [("I",)] * (n - 1) + [("Z", 1, 2, random_turn(rng)), ("I",)]
        close = [("I",)] * n + [("X", 2, 1, random_turn(rng))]
        return [[random_box(rng) for _ in range(n + 1)], tie, close]

    for name, lhs, num, rhs in deck(a, b, boxes):
        n_in, n_out = legs_of(lhs)
        before, after, beside = context(n_in), context(n_out), random_box(rng)
        wrap = lambda side: (before + [layer + [("I",)] for layer in side]
                             + [[("I",)] * n_out + [beside]] + after)
        Lw, Rw = picture(wrap(lhs)), picture(wrap(rhs), num)
        assert same_picture(Lw, Rw), name
        if not np.isclose(complex(num), 1):
            assert not same_picture(Lw, picture(wrap(rhs))), name   # its number matters in company

    # Result 19.1 on a larger picture with every kind of piece: generic boxes, spiders of both
    # colours with random turns, yellow boxes, a cup, a cap and a crossing.
    big = [[random_box(rng), ("cup",), ("Z", 1, 2, a)],
           [("SW",), ("H",), ("X", 1, 1, b), ("I",)],
           [("I",), ("cap",), ("Z", 1, 1, random_turn(rng)), ("I",)],
           [("X", 2, 1, random_turn(rng)), ("I",)],
           [("H",), ("I",)],
           [random_box(rng), ("Z", 1, 1, random_turn(rng))]]
    assert legs_of(big) == (2, 2)
    P = picture(big, np.exp(2j * PI * float(a)))
    Pm = matrix_of(P)
    assert not np.allclose(Pm, Pm.conj())                          # non-real
    assert np.allclose(matrix_of(P.adjoint()), Pm.conj().T)         # the mirror is the dagger
    assert not np.allclose(matrix_of(P.adjoint()), Pm.T)            # not the transpose
    assert same_picture(P.adjoint(), picture(mirror(big), np.exp(-2j * PI * float(a))))
    # and a random circuit-shaped picture with non-Clifford turns on every spider
    c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=3, depth=24, clifford=False, seed=1481)
    g = c.to_graph()
    for v in g.vertices():
        if g.type(v) in (VT.Z, VT.X):
            g.set_phase(v, random_turn(rng))
    M = matrix_of(g)
    assert not np.allclose(M, M.conj()) and not np.allclose(M, M.T)          # generic
    assert np.allclose(matrix_of(g.adjoint()), M.conj().T)

    # DERIVATION 1 (named rules): the sliding card's left picture, rewritten by PyZX's fusion
    # and identity rules, is one green spider turned a + b.
    for name in ("sliding together", "sliding together, more legs"):
        lhs, rhs = [(l, r) for nm, l, n, r in deck(a, b, boxes) if nm == name][0]
        h = picture(lhs)
        while simp.spider_simp(h) or simp.id_simp(h):
            pass
        inner = [v for v in h.vertices() if h.type(v) != VT.BOUNDARY]
        assert len(inner) == 1 and h.phase(inner[0]) == (a + b) % 2
        assert same_picture(h, picture(rhs))

    # DERIVATION 2: a box meeting its mirror reduces to bare wires, with its number.
    # A random Clifford circuit: the number is 1.
    c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=3, depth=40, clifford=True, seed=1482)
    g = c.to_graph()
    h = g.copy()
    h.compose(g.adjoint())
    simp.full_reduce(h)
    assert bare_wires(h, 3) and np.isclose(h.scalar.to_number(), 1)
    # The bare joiner (a look, Chapter 8's row 64) meeting its mirror: bare wires, number 1/2.
    j = joiner()
    assert np.allclose(matrix_of(j), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                                               [0, 0, 1, 0]]) / np.sqrt(2))   # CNOT / sqrt2
    h = j.copy()
    h.compose(j.adjoint())
    simp.full_reduce(h)
    assert bare_wires(h, 2) and np.isclose(h.scalar.to_number(), 0.5)
    assert not np.isclose(h.scalar.to_number(), 1)                  # the joiner is not unitary


if __name__ == "__main__":
    verdict("EveryCardBalances", blackboard, sketchbook)
