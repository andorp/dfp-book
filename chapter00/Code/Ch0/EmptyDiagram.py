"""The exchange: Prelude on a Blank Sheet
=========================================
"Twice as much, and then nothing done: still twice. Twice, and then three times, would have been six times. So your picture is the one that leaves it as it was: once. One, not zero. ...Oh."  -- Achilles

Claim: the empty diagram is the number 1, and composing with it changes nothing:
the product of no numbers is 1, 1*a = a for every a, and 0 does not serve.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import math
from fractions import Fraction

import numpy as np
import pyzx as zx
from pyzx.utils import VertexType as VT
from dfp import matrix_of, same_picture, verdict


def blackboard():
    rng = np.random.default_rng(0)
    a = complex(rng.normal() + 1j * rng.normal())
    assert a.imag != 0                                   # a generic, non-real number

    # the product of no numbers is 1
    assert np.prod([]) == 1
    assert math.prod([]) == 1

    # 1 changes nothing: 1*a is a, exactly
    assert 1 * a == a
    # appending a factor multiplies the product by it: a = (product of none) * a
    assert np.prod([a]) == np.prod([]) * a

    # the only such number: whatever leaves every number as it was leaves 1 as it
    # was, and a number times 1 is itself -- so the product of none is 1
    u = np.prod([])
    assert u * 1 == 1
    assert u == 1

    # Achilles's first guess, zero, fails: it changes nothing only under adding
    assert 0 * a != a
    assert 0 * a == 0
    assert 0 + a == a


def check_no_edges_no_boundaries(g):
    """Every picture in this file has no edge and no boundary at all."""
    assert g.num_edges() == 0
    assert tuple(g.inputs()) == () and tuple(g.outputs()) == ()
    assert all(g.type(v) != VT.BOUNDARY for v in g.vertices())


def lone(phase):
    """A picture with one phase node in it and nothing else."""
    g = zx.Graph()
    # PyZX draws this with the spiders you meet in Chapter 8.
    g.add_vertex(VT.Z, 0, 0, phase=phase)
    check_no_edges_no_boundaries(g)
    return g


def sketchbook():
    rng = np.random.default_rng(0)

    # --- The Empty Diagram --------------------------------------------------
    # PyZX draws pictures with the spiders you meet in Chapter 8; this one has none.
    E = zx.Graph()
    check_no_edges_no_boundaries(E)
    assert E.num_vertices() == 0
    assert matrix_of(E).shape == (1, 1)
    assert np.allclose(matrix_of(E), [[1]])              # the empty diagram is 1

    # --- Blank beside blank is blank ----------------------------------------
    side_by_side = E.tensor(zx.Graph())                  # tensor returns a new graph
    check_no_edges_no_boundaries(side_by_side)
    assert side_by_side.num_vertices() == 0
    assert same_picture(side_by_side, E)

    one_after_other = zx.Graph()
    one_after_other.compose(zx.Graph())                  # compose mutates in place
    check_no_edges_no_boundaries(one_after_other)
    assert one_after_other.num_vertices() == 0
    assert same_picture(one_after_other, E)

    # --- Beside a picture, nothing changes ----------------------------------
    # A translation check, not a proof: the chapter proves "beside any picture"
    # by looking (the two sides are the same page).  Here we check one closed
    # picture, a special case, with a random phase.
    # PyZX draws this with the spiders you meet in Chapter 8.
    k = int(rng.integers(1, 97))
    P = lone(Fraction(k, 97))
    value = 1 + np.exp(1j * np.pi * k / 97)
    assert not np.isclose(value.imag, 0)                 # a non-real value
    assert np.allclose(matrix_of(P), [[value]])

    EP = E.tensor(P)
    check_no_edges_no_boundaries(EP)
    assert same_picture(EP, P)                           # blank, then the picture

    PE = P.tensor(zx.Graph())
    check_no_edges_no_boundaries(PE)
    assert same_picture(PE, P)                           # the picture, then blank

    C = P.copy()
    C.compose(zx.Graph())
    check_no_edges_no_boundaries(C)
    assert same_picture(C, P)                            # composed with blank

    # Achilles's zero, in pictures: a picture worth 0 does change what it is put beside
    Z0 = lone(Fraction(1))
    assert np.allclose(matrix_of(Z0), [[0]])
    PZ = P.tensor(Z0)
    check_no_edges_no_boundaries(PZ)
    assert not same_picture(PZ, P)


if __name__ == "__main__":
    verdict("EmptyDiagram", blackboard, sketchbook)
