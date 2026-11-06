"""Challenge 6: The Relay
=======================
"They never shared a bench."  -- Anteater

Claim: on six wires, three cups (wires 1-2, 3-4, 5-6) followed by two caps
(wires 2-3, 4-5) are exactly one cup on wires 1 and 6:
(I (x) cap (x) cap (x) I)(cup (x) cup (x) cup) = cup.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
from dfp import BELL, CAP, CUP, I2, Picture, kron, matrix_of, same_picture, verdict
from dfp_props import Dialogue, words_only
import props_ch06 as P


def blackboard():
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))      # unnormalized cup
    assert np.allclose(CAP, CUP.conj().reshape(1, 4))       # ... and cap
    assert np.allclose(BELL, CUP / np.sqrt(2))

    # one junction: two pairs, one cap across the middle
    one_junction = kron(I2, CAP, I2) @ kron(CUP, CUP)
    assert np.allclose(one_junction.ravel(), CUP)

    # the chain: six wires in, wires 1 and 6 out
    three_cups = kron(CUP, CUP, CUP)
    assert three_cups.shape == (2 ** 6, 1)                  # six wires
    chain = kron(I2, CAP, CAP, I2) @ three_cups
    assert np.allclose(chain.ravel(), CUP)                  # exact: no stray number

    # what the pull costs, with normalized pairs and the normalized outcome
    bell_bra = BELL.conj().reshape(1, 4)
    relay = (kron(I2, bell_bra, bell_bra, I2) @ kron(BELL, BELL, BELL)).ravel()
    assert np.allclose(relay, BELL / 4)                     # 1/2 per junction
    assert not np.allclose(relay, BELL)
    assert np.isclose(np.linalg.norm(relay) ** 2, 1 / 16)   # = 1/4 * 1/4

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the six boxes in a row with the string
# laid before them, and tonight's page of the ledger, with
#   \musicboxes[string]{boxes}{caption}   \crabledger[blotter]{lines}{caption}
# (pdf/props/ch06.tex).  The Lua COMPUTES the pulled string -- bends of
# opposite kinds cancel, far bends minus near turns, one wide bend from the
# first leg to the last -- and the ledger's right-hand column.
# tools/props_ch06.py transcribes it; the calls are read out of dialog.md,
# never copied.  This file also checks the dialogue's pictures as a whole.
#
# The state model (identity card, section 3): in the row the boxes stand in
# wire order, left pearwood = 1 ... right ebony = 6; the string's leg k runs
# from the near side up to box k; a far bend at each pair (at the boxes), a
# near turn between pairs (at the stations), the free ends at the near edge.
# --------------------------------------------------------------------------
MACROS = ("musicboxes", "walnutstring", "beadstring", "crabledger")
DIALOGUE = Dialogue(__file__, macros=MACROS)
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")
DRAWS = {"First": {"musicboxes", "crabledger"}, "Second": {"crabledger"},
         "Third": {"walnutstring"}, "Fourth": {"beadstring"}, "Fifth": {"walnutstring"},
         "Sixth": {"musicboxes", "crabledger"}}


def kind(c):
    """The keyed form comes once per macro -- and once more for the row, whose
    labels (above the pairs) differ from the tray's (beside the rows)."""
    return (c.macro, c.args[0].split(";")[0].strip() if c.macro == "musicboxes" else "")


def string_map(s, early):
    """A laid string as a map, read with time running away from the `early`
    side of the table: a bend on the early side is a cup, on the late side a
    cap; a free end on the early side is an input, on the late side an output;
    the legs, left to right, are the wires, top first.  In the row the boxes
    are at the far side and the free ends at the near edge: early = far, so a
    far bend at a pair is a cup and a near turn at a station is a cap."""
    late = {"far": "near", "near": "far"}[early]
    order = s.names

    def layer(side, joined):
        out, i = [], 0
        while i < len(order):
            bend = [b for b in s.bends if b.side == side and order[i] in b.legs]
            if bend:
                assert bend[0].legs == (order[i], order[i + 1])
                out.append(joined)
                i += 2
            else:
                out.append(I2)
                i += 1
        return kron(*out)

    return layer(late, CAP) @ layer(early, CUP)


def check_transcription():
    """The Lua's pull rule against the blackboard (a structural check, not a
    second proof): three far bends and two near turns pull to 3 - 2 = 1 far
    bend from leg 1 to leg 6, and the map is the Chain's, before and after."""
    laid = P.row_string("three bends")
    assert P.continuous(laid) and laid.names == (1, 2, 3, 4, 5, 6)
    assert [(b.side, b.legs) for b in laid.bends] == \
        [("far", (1, 2)), ("near", (2, 3)), ("far", (3, 4)), ("near", (4, 5)), ("far", (5, 6))]
    assert laid.ends == ("near", "near")
    assert np.allclose(string_map(laid, "far"), kron(I2, CAP, CAP, I2) @ kron(CUP, CUP, CUP))
    assert P.faults(P.row_string, "three bends pulled") == []
    one = P.row_string("three bends pulled")
    assert P.continuous(one) and [(b.side, b.legs) for b in one.bends] == [("far", (1, 6))]
    assert one.ends == laid.ends
    assert np.allclose(string_map(one, "far").ravel(), CUP)
    assert np.allclose(string_map(laid, "far"), string_map(one, "far"))
    # the partial strings are laid pair by pair (never drawn: the card's gap 5)
    assert P.row_string("two bends").ends == ("near", "slack")
    assert P.faults(P.row_string, "two bends pulled") != []            # pulled only when laid
    assert P.faults(P.row_string, "four bends") != []


FORBIDDEN = {"walnutstring": ("nudged", "hairpin out", "hairpin close", "hairpin aside"),
             "musicboxes": ("bare", "one bend", "two bends")}


def check_whole_dialogue():
    assert DIALOGUE.check_whole(ORDINALS) == 20                  # 4 + 3 + 4 + 3 + 2 + 4
    DIALOGUE.keyed_first(ORDINALS, kind=kind)
    placed = [(o, i, c) for o in ORDINALS
              for i, c in enumerate(c for b in DIALOGUE.blocks(o) for c in b)]
    assert [(o, i) for o, i, c in placed if c.keyed] == \
        [("First", 0), ("First", 2), ("Third", 0), ("Fourth", 0), ("Sixth", 0)]
    for o in ORDINALS:
        assert {c.macro for oo, _, c in placed if oo == o} == DRAWS[o], o
        for b in DIALOGUE.blocks(o):                              # the walnut pictures stand alone
            assert len({c.macro for c in b}) == 1 or "walnutstring" not in {c.macro for c in b}
    for o, i, c in placed:
        pic = P.picture(c)                                        # no macro ever faults
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS), c.caption
        assert "chain" not in c.caption.lower()                  # never the master's name
        if c.macro == "walnutstring":
            assert not pic.nudged and pic.shape not in ("hairpin out", "hairpin close",
                                                         "hairpin aside")
        if c.macro == "beadstring":
            assert pic.place not in ("in the bend", "back", "napkin")
        if c.macro == "musicboxes":
            assert pic.string_word.replace(" pulled", "") not in FORBIDDEN["musicboxes"]
            assert (pic.layout == "row") == (o == "Sixth")         # the tray before, the row after
            assert all(b.state != "kept" for b in pic.boxes.values()) or o == "First"
    # no picture in the relay's opening (the carriers), and none after the last note
    six = DIALOGUE.section("Sixth")
    first_block = six.index("\\begin{center}")
    assert six.index("Even so — the boxes in between?") < first_block
    assert six.index("Let Bob draw on to the second station; the last pair.") < first_block
    last = DIALOGUE.blocks("Sixth")[-1][-1]
    assert not any("\\" + m in six.split(last.text)[1] for m in MACROS)


def pictures():
    check_transcription()
    check_whole_dialogue()
    p61, p62, p63, p64 = DIALOGUE.check_layout("Sixth", sizes=[1, 1, 2])
    assert [c.macro for c in (p61, p62, p63, p64)] == ["musicboxes"] * 3 + ["crabledger"]
    r61, r62, r63, t64 = (P.picture(c) for c in (p61, p62, p63, p64))
    six = DIALOGUE.section("Sixth")

    a1, a2, a3 = P.anchors(DIALOGUE, "Sixth")
    assert "one below the far-left box and one below the far-right" in a1
    assert "joined now by one unbroken bend across the whole front of the row" in a2
    assert "The Crab turns the ledger to tonight's page" in a3
    assert "a single high tone, the same as the first" in a3

    # P6.1: the row, keyed; boxes in wire order; box 1 spent; the whole string laid
    assert p61.keyed and r61.layout == "row" and r61.order == (1, 2, 3, 4, 5, 6)
    assert r61.labels == ("pearwood", "walnut", "ebony") and r61.labels_at == "above the pairs"
    assert r61.state(1) == "spent" and all(r61.state(w) == "shut" for w in range(2, 7))
    assert r61.string == P.row_string("three bends")
    assert r61.string.count("far") == 3 and r61.string.count("near") == 2
    assert {b.legs for b in r61.string.bends if b.side == "far"} == \
        {(P.WIRE[("left", p)], P.WIRE[("right", p)]) for p in P.PAIRS}   # a bend at each pair
    assert np.allclose(string_map(r61.string, "far"), kron(I2, CAP, CAP, I2) @ kron(CUP, CUP, CUP))

    # P6.2: both ends drawn in -- nothing else moved; one far bend, box 1 to box 6: the Chain
    assert not p62.keyed and r62.boxes == r61.boxes and p62.args == p61.args
    assert [(b.side, b.legs) for b in r62.string.bends] == [("far", (1, 6))]
    assert np.allclose(string_map(r62.string, "far").ravel(), CUP)
    assert "One bend. From Alice's box to Bob's." in six
    # "the hairpin you laid for the walnut boxes, only wider": the Third's pair, the same map
    hairpin = P.picture(DIALOGUE.blocks("Third")[0][0])
    assert hairpin.shape == "hairpin"
    assert np.allclose(string_map(hairpin.string, "near").ravel(), string_map(r62.string, "far").ravel())

    # P6.3: the last note -- box 2's lid up, the string's one bend unchanged
    assert r63.string == r62.string and r63.state(2) == "open" and r63.boxes[2].lid_up
    assert r63.state(1) == "spent" and all(r63.state(w) == "shut" for w in range(3, 7))
    # the first note and the last are the two halves of one pair
    p11 = DIALOGUE.blocks("First")[0][0]
    b11 = P.picture(p11)
    assert [w for w in range(1, 7) if b11.boxes[w].lid_up] == [1]
    assert [w for w in range(1, 7) if r63.boxes[w].lid_up] == [2]
    assert (b11.boxes[1].wood, b11.boxes[1].side, r63.boxes[2].wood, r63.boxes[2].side) == \
        ("pearwood", "left", "pearwood", "right")

    # P6.4: tonight's page (P1.3's), completed: the same word twice
    t13 = P.picture(DIALOGUE.blocks("First")[1][1])
    assert t13.written[:-1] == t64.written[:-1]
    assert (t13.written[-1], t64.written[-1]) == ("tonight high", "tonight high completed")
    assert t13.lines[:-1] == t64.lines[:-1]
    assert (t64.lines[-1].left, t64.lines[-1].right, t64.lines[-1].ink) == ("high", "high", "fresh")
    assert t13.lines[-1].right is None
    assert "**Achilles:** High." in six


def the_chain():
    """Six rows (row k-1 is wire k).  Bends sit between the rows they join:
    cups on the left at 0.5, 2.5, 4.5; caps on the right at 1.5, 3.5."""
    p = Picture()
    cup12 = p.bend(0.5, 1)   # PyZX draws this with the spiders you meet in Chapter 8.
    cup34, cup56 = p.bend(2.5, 1), p.bend(4.5, 1)
    cap23, cap45 = p.bend(1.5, 3), p.bend(3.5, 3)
    # the string: out round 1-2, back between 2 and 3, out round 3-4, back, out round 5-6
    p.wire(p.out(0), cup12, cap23, cup34, cap45, cup56, p.out(5))
    return p.done()


def one_bend():
    p = Picture()
    bend = p.bend(0.5, 1)
    p.wire(p.out(0), bend, p.out(1))
    return p.done()


def sketchbook():
    chain = the_chain()
    assert chain.num_edges() == 6                           # one stretch of string per wire
    assert len(chain.inputs()) == 0 and len(chain.outputs()) == 2
    top, bottom = chain.outputs()                           # outputs sorted top wire first;
    assert (chain.qubit(top), chain.qubit(bottom)) == (0, 5)    # PyZX calls our rows "qubits"
    assert np.allclose(matrix_of(one_bend()).ravel(), CUP)

    assert same_picture(chain, one_bend())                  # pull: one bend, wire 1 to wire 6

    # the normalized version: 1/sqrt(2) beside each of the five bends
    relay = the_chain()
    relay.scalar.add_power(-5)
    assert np.allclose(matrix_of(relay).ravel(), BELL / 4)
    assert not same_picture(relay, one_bend())


if __name__ == "__main__":
    verdict("Chain", blackboard, sketchbook)
