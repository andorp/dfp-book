"""Challenge VI: Octet (the count, the revelation)
==================================================
"Eight cords. Eight, and not one of them spare — every one carrying its part, and the whole thing meaning nothing if you pull a single one out."  -- Achilles

Claim: the Octet (the bialgebra cradle) is a genuine 8-wire diagram that
evaluates, up to sqrt2, to a single green-red pair -- and only its topology
matters (deform the drawing, the tensor does not move).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
import re
import string
from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import ket, kron, matrix_of, same_picture, same_picture_up_to_scalar, verdict
from dfp_props import Dialogue, words_only
import props_ch08 as P


# --- spiders as matrices: the chapter's own maths, pinned in every file ---
KET_PLUS = (ket(0) + ket(1)) / np.sqrt(2)
KET_MINUS = (ket(0) - ket(1)) / np.sqrt(2)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I2 = np.eye(2, dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def zspider(m, k, alpha):
    """Z spider (green): |0..0><0..0| + e^{i alpha}|1..1><1..1|, m legs in / k out."""
    M = np.zeros((2 ** k, 2 ** m), dtype=complex)
    M[0, 0] = 1.0
    M[-1, -1] = np.exp(1j * alpha)
    return M


def _pow(v, n):
    out = np.array([1.0 + 0j])
    for _ in range(n):
        out = np.kron(out, v)
    return out


def xspider(m, k, beta):
    """X spider (red), built INTRINSICALLY from normalized |+>,|-> --
    NOT as H^(x)k Z H^(x)m, so colour change stays a theorem, not a definition."""
    pk, pm = _pow(KET_PLUS, k), _pow(KET_PLUS, m)
    mk, mm = _pow(KET_MINUS, k), _pow(KET_MINUS, m)
    return np.outer(pk, pm.conj()) + np.exp(1j * beta) * np.outer(mk, mm.conj())


def pin_spiders():
    assert np.allclose(zspider(1, 1, np.pi), np.diag([1, -1]))
    assert np.allclose(xspider(1, 1, np.pi), np.array([[0, 1], [1, 0]]))
    assert np.allclose(H @ H, np.eye(2))                               # H^2 = I
    assert np.allclose(KET_PLUS + KET_MINUS, np.sqrt(2) * ket(0))


def blackboard():
    pin_spiders()

    # the Octet as a 2->2 map: two green copiers into a complete cross into two red
    # adders (all phase-free) -- the same cradle Complementarity.py built for bialgebra
    Z12, X21 = zspider(1, 2, 0), xspider(2, 1, 0)
    cross = kron(I2, SWAP, I2)                         # I (x) SWAP (x) I on 4 wires = the K_2,2
    octet_m = kron(X21, X21) @ cross @ kron(Z12, Z12)
    pair_m = Z12 @ X21                                 # one red gathers, one green issues

    # it reads the parity of its two inputs and writes that parity on BOTH outputs
    parity = 0.5 * (np.outer(ket(0, 0), (ket(0, 0) + ket(1, 1)).conj())
                    + np.outer(ket(1, 1), (ket(0, 1) + ket(1, 0)).conj()))
    assert np.allclose(octet_m, parity)
    # the master equality: Octet = a single green-red pair, UP TO the scalar sqrt2
    assert np.allclose(octet_m, pair_m / np.sqrt(2))  # exact scalar: cradle = pair / sqrt2
    # NEGATIVE: without the full cross (only the two straight cords) it is NOT the pair
    straight = kron(X21, X21) @ kron(Z12, Z12)
    assert not np.allclose(straight, pair_m / np.sqrt(2))

    pictures(octet_m, pair_m)


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the Crab's cords and curtain-rings
#   \ringcords[view]{tangle}{caption}           (pdf/props/ch08.tex)
# and the Crab's press, \koekpan[view]{caption}.  The Lua COMPUTES the views:
# slid, flipped, pushed, pressed, and the ring come clear of the flip-slot.
# Round 4 ("Show the process") adds the marks (marks=on: a moved part's ghost,
# its arc, a pin on what is held), the prediction frames (expects=: Achilles's
# refuted rule, drawn paler) and \ringsteps[opts]{tangle}{steps}{captions}, a
# strip whose every frame starts from the state the one before left.
# tools/props_ch08.py transcribes the Lua (proved against texlua; P.check_lua
# holds the Lua and the .tex to the text it was proved on); the calls are read
# out of this challenge's section of dialog.md and dialog.raw.md, never copied.
#
# The state model (identity card, section 3): a green ring is a Z spider, a red
# ring an X spider; a ring's turn t (twenty-fourths, clockwise from the top as
# the company sees it) is the phase 2 pi t / 24; cords in on the left are
# inputs, out on the right outputs, the upper line first.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
QUOTE = ("Eight cords. Eight, and not one of them spare — every one carrying its part, "
         "and the whole thing meaning nothing if you pull a single one out.")
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")
SIZES = {"First": [1, 2], "Second": [1, 1, 2], "Third": [1, 2, 2], "Fourth": [1, 1, 3],
         "Fifth": [1, 1], "Sixth": [1, 1, 1]}                     # calls a block (a strip is one)


def phase(t):
    """A turn in twenty-fourths as a phase in radians."""
    return 2 * np.pi * t / P.WHOLE


def tangle_map(T):
    """The map a tangle of rings and cords denotes, contracted in NumPy: each
    ring a spider of its colour with one leg per cord segment at it; ends on
    the left are inputs, on the right outputs, the upper line first (top wire
    first).  Leg order and in/out do not matter: these Z and X spiders are
    symmetric in all their legs (|0>,|1>,|+>,|-> are real)."""
    letters = iter(string.ascii_letters)
    legs = {n: [] for n in range(1, len(T.rings) + 1)}
    ins, outs, ops, subs = [], [], [], []
    for a, b in P.segments(T):
        if isinstance(a, P.End) and isinstance(b, P.End):     # a bare cord, hook to hook
            la, lb = next(letters), next(letters)
            ops.append(np.eye(2)); subs.append(la + lb)
            pairs = ((a, la), (b, lb))
        else:
            l = next(letters)
            pairs = ((a, l), (b, l))
        for e, l in pairs:
            if isinstance(e, P.End):
                (ins if e.side == "L" else outs).append((-e.y, l))
            else:
                legs[e].append(l)
    for n, r in enumerate(T.rings, 1):
        k = len(legs[n])
        spider = zspider if r.c == "green" else xspider
        ops.append(spider(0, k, phase(r.t)).reshape([2] * k)); subs.append("".join(legs[n]))
    out = "".join(l for _, l in sorted(outs)); inn = "".join(l for _, l in sorted(ins))
    M = np.einsum(",".join(subs) + "->" + out + inn, *ops)
    return M.reshape(2 ** len(outs), 2 ** len(ins))


def unordered(conns):
    """A multiset of connections "a-b", each read either way round."""
    return sorted("-".join(sorted(c.split("-"))) for c in conns)


def check_faults():
    """Every one of the Lua's sixty-two tex.error sites, each raised by a call of its own and
    by nothing else.  Two no macro call reaches (the fuzz against texlua raises the other 60):
    'reshaped' guards the reshaped cradle's connections, which the Lua's own RESHAPE tables
    keep, and 'step-key' guards ring_step against a key the bracket parser has already
    refused; each is raised here through the model's function itself."""
    cradle = P.parse_tangle("cradle")
    R, K, S = P.ringcords, P.koekpan, P.ringsteps
    two = "two green rings, turns top and half"
    bad = [
        ("opt-unknown", R, "colour=red", "green ring"),
        ("opt-twice", R, "keyroom=on; keyroom=on", "green ring"),
        ("opt-bare", R, "slid; pushed", two),
        ("onoff", R, "slid; marks=yes", two),
        ("turn", R, "", "green ring, turn a fifth"),
        ("count", R, "", "green ring, seven cords"),
        ("place", R, "", "green ring, in at side"),
        ("place-twice", R, "", "green ring, in at top and top"),
        ("ring-item", R, "", "green ring, sideways"),
        ("turn-twice", R, "", "green ring, turn a third, turn half"),
        ("in-once", R, "", "green ring, in at top, in at bottom"),
        ("out-once", R, "", "green ring, out at top, out at bottom"),
        ("cords-twice", R, "", "green ring, four cords, two cords"),
        ("in-twice", R, "", "green ring, two in, one in"),
        ("out-twice", R, "", "green ring, two out, one out"),
        ("cords-places", R, "", "green ring, four cords, in at top"),
        ("cords-counts", R, "", "green ring, two in, four cords"),
        ("in-places", R, "", "green ring, two in, in at top"),
        ("out-places", R, "", "green ring, out at top, two out"),
        ("no-cords", R, "", "green ring, none in, none out"),
        ("two-turns", R, "", "two green rings"),
        ("two-extra", R, "", "two green rings, turns top and half, loose"),
        ("point-fork", R, "", "red point, red fork"),
        ("half-turn", R, "", "red half-turn, red ring"),
        ("half-turn-item", R, "", "red half-turn, green ring, loose"),
        ("green-and-red", R, "", "green and red, three cords"),
        ("reshaped", P.check_connections, cradle.cords[:-1]),          # a cord lost
        ("tangle", R, "", "blue ring"),
        ("slid", R, "slid", "green ring"),
        ("flipped", R, "flipped", "three gates"),
        ("pushed", R, "pushed", "gate"),
        ("pressed", R, "pressed", "gate"),
        ("wrong-passes", R, "pressed; expects=passes", "cradle"),
        ("wrong-tighter", R, "pressed; expects=tighter", "red point, green fork"),
        ("add-single", R, "add=in", "cradle"),
        ("add-value", R, "add=up", "green ring"),
        ("add-hooks", R, "add=in", "green ring, four in"),
        ("add-full", R, "add=out", "green ring, out at top and middle and bottom"),
        ("slide-value", R, "slide=far ring", two),
        ("slide-two", R, "slide=near ring", "green ring"),
        ("reshape-value", R, "reshape=all", "cradle"),
        ("reshape-takes", R, "reshape=cords", "cradle"),
        ("step-key", P.ring_step, P.parse_tangle("green ring"), "green ring", "bogus", "x", "ringcords"),
        ("keyroom", R, "keyroom=off", "green ring"),
        ("one-step", R, "add=in; reshape=rings", "green ring, in at top"),
        ("step-view", R, "pressed; reshape=rings", "cradle"),
        ("view", R, "sideways", "gate"),
        ("frame-loose", R, "loose", "cradle"),
        ("beam", R, "beam", "gate"),
        ("marks-view", R, "flipped; marks=on", "green ring"),
        ("expects-pressed", R, "expects=passes", "red point, green fork"),
        ("expects-rule", R, "pressed; expects=sooner", "red point, green fork"),
        ("expects-keyed", R, "pressed; expects=passes", "red point, green fork", True),
        ("koekpan-marks", K, "shut; marks=on"),
        ("koekpan-view", K, "open"),
        ("strip-bare", S, "beam", "green ring, in at top", "add=in", "a; b"),
        ("strip-view", S, "view=pressed", "green ring, in at top", "add=in", "a; b"),
        ("strip-done", S, "", two, "slid; slid", "a; b; c"),
        ("strip-step", S, "", "green ring, in at top", "add=in; wiggle", "a; b"),
        ("strip-none", S, "", "green ring, in at top", "", "a"),
        ("strip-four", S, "", "green ring, in at top", "add=in; add=in; add=out; add=out", "a; b; c; d; e"),
        ("strip-captions", S, "", "green ring, in at top", "add=in", "a"),
    ]
    for site, fn, *args in bad:
        assert P.faults(fn, *args) == [site], (site, P.faults(fn, *args))
    assert [b[0] for b in bad] == list(P.ERROR_SITES) and len(set(P.ERROR_SITES)) == 62
    # the ring's items are checked after the whole list is read: a mix or a repeat is the same
    # fault whichever item comes first, and a ring written in any order is the same ring
    pairs = [("cords-places", "four cords", "in at top"), ("cords-places", "four cords", "out at bottom"),
             ("cords-counts", "four cords", "two in"), ("cords-counts", "four cords", "one out"),
             ("in-places", "two in", "in at top"), ("out-places", "two out", "out at top"),
             ("turn-twice", "turn a third", "turn half"), ("in-once", "in at top", "in at bottom"),
             ("out-once", "out at top", "out at middle"), ("cords-twice", "four cords", "two cords"),
             ("in-twice", "two in", "one in"), ("out-twice", "two out", "one out")]
    for site, a, b in pairs:
        for x, y in ((a, b), (b, a)):
            assert P.faults(R, "", f"green ring, {x}, {y}") == [site], (site, x, y)
    items = ["turn a third", "in at top and bottom", "out at top and middle and bottom"]
    for perm in itertools.permutations(items):
        T = P.parse_tangle("green ring, " + ", ".join(perm))
        assert P.faults(P.parse_tangle, "green ring, " + ", ".join(perm)) == []
        assert (T.rings, T.cords, T.pin, T.pout) == (P.parse_tangle("green ring, " + ", ".join(items)).rings,
                                                      P.parse_tangle("green ring, " + ", ".join(items)).cords,
                                                      frozenset({"top", "bottom"}), frozenset({"top", "middle", "bottom"}))
    assert P.faults(R, "pressed", " ") == ["tangle", "pressed"]                  # an empty name: '?'
    assert P.TEX_ERRORS[-1][1].endswith("(got '?')")
    assert P.lua_sites() == P.TEX_SITES                             # the Lua's own, in its order
    # a re-joined cord is caught as well as a lost one; the reshape as drawn loses none
    moved = cradle.cords[:4] + (P.Cord(1, 2),) + cradle.cords[5:]
    assert P.faults(P.check_connections, moved) == ["reshaped"]
    assert P.faults(P.parse_tangle, "reshaped cradle") == []
    assert P.faults(P.parse_tangle, "cradle, rings slid") == []


MARK_STYLES = ("propcheightghost", "propcheightswing", "propcheightslide", "propcheightpinmark")


def check_marks(frames):
    """The card's section 3e.5: marks are neither wires nor notation.  No mark on a frame on
    the beam (F01-F03, F15, F22-F24); every mark line drawn in a mark style, each style dotted,
    never dashed (the Hadamard edge is dashed); the arcs carry a head; the ghosts are the part
    itself, paler (a transparency group); no cord ghost anywhere; marks only where a part
    moves in the scene (the slide, the slot, the push)."""
    tex = P.TEX.read_text()
    for st in MARK_STYLES:
        (style,) = re.findall(r"^\s*" + st + r"/\.style=\{([^}]*)\}", tex, re.M)
        assert "densely dotted" in style and "dash" not in style, (st, style)
    marked = []
    for o, fr in frames:
        if isinstance(fr, P.Pan):
            kinds = [m.kind for m in fr.marks]
            assert kinds in ([], ["ghost", "swing"]) and (not kinds or fr.view.startswith("slot")), fr.log
        else:
            kinds = [m.kind for m in fr.marks]
            if fr.setting == "beam":
                assert kinds == [] and fr.mark_tikz == () and not fr.pale, fr.log      # the counted frames
            assert "outline" not in kinds, fr.log                                  # no outline on a cord
        for line in fr.mark_tikz:
            if line.startswith("\\begin{scope}"):
                assert line in ("\\begin{scope}[transparency group, opacity=" + P.f(P.GHOST) + "]",
                                "\\begin{scope}[transparency group, opacity=" + P.f(P.PALE) + "]"), line
                continue
            if line.startswith("\\draw[propcheight") and not any(s in line for s in MARK_STYLES):
                continue                                # a ghost's own body (ring, bead), in its scope
            assert "dashed" not in line and "dash pattern" not in line, line
            if "propcheightswing" in line or "propcheightslide" in line:
                assert P.TIP in line, line                                         # an arc, headed
        if kinds:
            marked.append((o, tuple(kinds)))
    assert marked == [("First", ("ghost", "slide", "bead", "bead-arc", "pin")), ("Second", ("ghost", "swing")),
                      ("Third", ("ghost", "slide", "bead", "bead-arc"))]
    assert P.GHOST == 0.35 and P.PALE == 0.5


def check_expansion():
    """What the transcription takes from ch08.tex (held by P.check_lua's TEX_PINS; read here so
    that a re-pinned edit to the expansion still breaks the check): the brackets TeX hands the
    Lua when a call has none (P.bracket: [] for the cords and the strip, [shut] for the pan),
    a strip frame run as a \ringcords call is (P.frame_calls), and the strip's arguments in the
    order P.ringsteps takes them."""
    tex = P.TEX.read_text()
    for line in (r"\newcommand{\prop@ringcords}[1]{\@ifnextchar[{\prop@ringcords@{#1}}{\prop@ringcords@{#1}[]}}",
                 r"\newcommand{\prop@koekpan}[1]{\@ifnextchar[{\prop@koekpan@{#1}}{\prop@koekpan@{#1}[shut]}}",
                 r"\newcommand{\propcheightringframe}[4]{\prop@ringcords@{#1}[#2]{#3}{#4}}",
                 r"\newcommand{\prop@ringstepso}[1]{\@ifnextchar[{\prop@ringsteps{#1}}{\prop@ringsteps{#1}[]}}",
                 "\\def\\prop@ringsteps#1[#2]#3#4#5{%\n  \\directlua{props.ch08_ringsteps(#1, "
                 "\"\\luaescapestring{#3}\", \"\\luaescapestring{#2}\",\n    \"\\luaescapestring{#4}\", "
                 "\"\\luaescapestring{#5}\")}}"):
        assert tex.count(line) == 1, line
    # the card's section 3e.6 pins, as the transcription holds them (the Lua's own, proved)
    assert (P.ROW, P.BX, P.RO, P.RI) == (0.34, 1.0, 0.17, 0.10) and np.isclose(P.MET_X, 0.048)
    assert np.isclose(P.SLIDE_Y, P.RO + 0.07) and P.SLOT_GHOST == (0.56, 0.31)
    assert P.PLACE_Y == {"top": 0.42, "middle": 0, "bottom": -0.42}
    assert P.ADDORDER == ("top", "bottom", "middle")
    assert P.RESHAPE_AT == ((-0.12, 0.44), (-0.66, -0.12), (0.70, 0.02), (0.20, -0.40))
    assert (P.GHOST, P.PALE) == (0.35, 0.5)
    assert [P.mirror_dir(t) for t in range(24)] == [0] + [-1] * 5 + [1] * 6 + [0] + [-1] * 6 + [1] * 5


def check_whole_dialogue():
    """All twenty-two calls and their twenty-five frames: their number and places, the keyed
    rule (true frames only), the captions, the views each challenge may use, the marks and the
    prediction frames, the dump, and no fault anywhere."""
    assert DIALOGUE.check_whole(ORDINALS) == 22                       # 3 + 4 + 5 + 5 + 2 + 3
    raw = DIALOGUE.read(DIALOGUE.raw)
    assert len(DIALOGUE.call_re.findall(raw)) == 22                   # dialog.raw.md, the same
    first = DIALOGUE.read(DIALOGUE.dialog).index("## First Challenge")
    assert not DIALOGUE.call_re.findall(DIALOGUE.read(DIALOGUE.dialog)[:first])   # none before it
    DIALOGUE.keyed_first(ORDINALS, kind=P.keyed_kind, select=P.judged)
    placed = [(o, i, c) for o in ORDINALS
              for i, c in enumerate(c for b in DIALOGUE.blocks(o) for c in b)]
    assert [(o, i) for o, i, c in placed if c.keyed] == [
        ("First", 0), ("Second", 0), ("Third", 2), ("Fourth", 0)]      # B1.1, P2.1, B3.2, P4.1
    assert [(o, i) for o, i, c in placed if P.is_prediction(c)] == [("Third", 1), ("Fourth", 3)]
    assert [(o, i) for o, i, c in placed if P.is_strip(c)] == [("First", 0), ("Sixth", 1)]
    frames, dump = [], []
    for o in ORDINALS:
        blocks = DIALOGUE.blocks(o)
        assert [len(b) for b in blocks] == SIZES[o], o
        assert len(blocks) <= 4, o                                     # at most four moments
        for b in blocks:
            fr = [p for c in b for p, _ in P.pictures(c)]
            assert len(fr) <= 4, o                                     # at most four frames a block
            frames += [(o, p) for p in fr]
    assert len(frames) == 25
    for o, i, c in placed:
        P.picture(c)                                                   # the macro never faults
        dump += P.describe(c)
        assert words_only(c.caption, forbidden=P.NOT_IN_CAPTIONS) and P.caption_ok(c), c.caption
        if c.macro == "koekpan":
            assert o == "Second", c.text
            continue
        for fc in P.frame_calls(c):
            view = P.view_of(fc)
            tangle = fc.args[0].lower()
            assert (view != "slid" or o == "First") and (view != "flipped" or o == "Second")
            assert (view != "pushed" or o == "Third")
            assert view != "pressed" or o in ("Third", "Fourth", "Fifth", "Sixth"), c.text
            assert tangle not in ("greens", "greens and reds"), c.text
            assert tangle.split(",")[0] not in P.FRAME or o in ("Fourth", "Sixth"), c.text
    assert len(dump) == 44 and dump.count("ringsteps: strip of 3 frames") == 1
    assert dump.count("ringsteps: strip of 2 frames") == 1
    # the prediction frames: paler, never keyed, each beside the true press of the same tangle
    for o, i in (("Third", 1), ("Fourth", 3)):
        calls = [c for b in DIALOGUE.blocks(o) for c in b]
        pred, true = calls[i], calls[i + 1]
        p, t = P.picture(pred), P.picture(true)
        assert p.pale and not p.keyed and p.expects in P.WRONG and t.expects is None and not t.pale
        assert pred.args == true.args and P.view_of(true) == "pressed"
        assert p.drawn != t.drawn                                      # the rule, refuted
    check_marks(frames)
    check_expansion()
    # the Lua's comments quote the dialogue (its steps, its wrong rules): every quote stands in
    # a challenge section of both files, so an edit to a line the Lua cites breaks the check
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        spoken = "\n".join(DIALOGUE.section(o, path) for o in ORDINALS)
        for q in P.step_quotes() + P.rule_quotes():
            assert q in spoken, (path.name, q)
    assert len(P.step_quotes()) == 8 and len(P.rule_quotes()) == 4


def pictures(octet_m, pair_m):
    assert f'"{QUOTE}"' in __doc__                              # quoted byte for byte
    P.check_lua()
    P.check_sites()
    check_faults()
    check_whole_dialogue()
    found = P.check_section(DIALOGUE, "Sixth", sizes=[1, 1, 1], frames=[1, 2, 1])
    calls = tuple(c for b in found for c, _ in b)
    p61, p62, p63 = calls
    (b61, cap61), = found[0][0][1]
    (f23, cap23), (f24, cap24) = found[1][0][1]
    (b63, cap63), = found[2][0][1]
    assert [c.macro for c in calls] == ["ringcords", "ringsteps", "ringcords"]
    assert not any(c.keyed for c in calls)
    assert [(c.opt, c.args) for c in calls] == [
        ("", ("cradle",)), ("from=off; marks=off", ("cradle", "reshape=rings; reshape=cords")),
        ("pressed", ("reshaped cradle",))]
    assert P.options(p62) == {"from": "off", "marks": "off", "bare": ""}
    assert (cap61, cap23, cap24, cap63) == ("untouched", "the rings moved", "as she left it",
                                            "from the crooked one")
    for cap in (cap61, cap23, cap24):
        assert not P.quantity_words(cap), cap                          # drawn copies of the frame
    sec = DIALOGUE.section("Sixth")
    paras = DIALOGUE.paragraphs("Sixth")
    a1, a2, a3 = DIALOGUE.anchors("Sixth")
    f1, f2, f3 = DIALOGUE.following("Sixth")
    # the directions and the caption edited this round (N6, N7 -- the read-out dropped, the
    # text closing over it --, N8; C4), in both files
    P.check_edits(DIALOGUE, "Sixth", [
        ("sliding the rings along the beam, bending", "sliding the rings along the beam, each ring with its peg, bending"),
        ("The cradle as it stood: two greens left, two reds right, a neat cross between.",
         "*[Her hands move the rings along the beam, and then coax"),
        ("Her hands move the rings along the beam and coax the four crossed cords",
         "Her hands move the rings along the beam, and then coax the cords, the crossed ones most of all,"),
        ("{waiting on the beam}", "{untouched}")])

    # P6.1: the cradle alone on the beam, just before the count; the count is its cords
    assert a1.endswith("and nothing else strung anywhere.]*")
    assert ("the two feed-cords in from the left, the four cords of the cross, "
            "the two draw-cords out to the right") in a1
    assert f1.startswith("**Achilles:** *[reaching up, touching each cord as he counts]*")
    assert QUOTE in f1                                                 # the docstring's line
    assert "One... two — the two coming in on the left." in f1
    assert "Three, four, five, six — the whole cross of them, every green to every red." in f1
    assert "Seven... eight, going out." in f1
    segs = P.segments(b61.drawn)
    feeds = [s for s in segs if isinstance(s[0], P.End)]
    cross = [s for s in segs if all(isinstance(e, int) for e in s)]
    draws = [s for s in segs if isinstance(s[1], P.End)]
    assert (len(feeds), len(cross), len(draws), len(segs)) == (2, 4, 2, 8)   # counted as spoken
    assert b61.setting == "beam" and b61.labels == ()
    assert np.allclose(tangle_map(b61.drawn), octet_m)

    # B6.2: the reshape's strip -- from=off, so its before-state is P6.1 four paragraphs above;
    # first the rings moved, each with its peg, cords straight to them; then the cords bent.
    # Every connection and every hook end kept, every frame the Octet's eight wires on four
    # rings, and (marks=off, Q6) no ghost and no outline on a counted frame
    assert "sliding the rings along the beam, each ring with its peg, bending the cross into new curves" in sec
    assert a2.endswith("has changed.]*")
    assert "every cord still runs from the ring it left to the ring it reached" in a2
    assert a2.index("move the rings along the beam") < a2.index("and then coax the cords")   # in order
    assert f2.startswith("*[The snow-hush deepens")                    # the lit print, after
    glance = [i for i, q in enumerate(paras) if "Metamorphosis II" in q]
    assert len(glance) == 1 and paras[glance[0] - 1] == f2               # no block before the glance
    frames = P.frame_calls(p62)
    assert [(c.opt, c.args[0]) for c in frames] == [("reshape=rings; marks=off", "cradle"),
                                                   ("reshape=cords; marks=off", "cradle, rings slid")]
    assert f23.made == b61.drawn and f24.made == f23.drawn              # each from the one before
    assert (f23.step, f24.step) == (("reshape", "rings"), ("reshape", "cords"))
    assert (f23.after, f24.after) == ("cradle, rings slid", "reshaped cradle")
    ends = lambda T: sorted({e for s in P.segments(T) for e in s if isinstance(e, P.End)}, key=str)
    for fr in (f23, f24):
        assert fr.setting == "beam" and fr.drawn.frame and fr.labels == ()
        assert fr.marks == () and fr.mark_tikz == () and not fr.pale    # no mark on the counted frame
        assert len(fr.drawn.cords) == len(P.segments(fr.drawn)) == 8 and len(fr.drawn.rings) == 4
        assert P.connections(fr.drawn.cords) == P.connections(b61.drawn.cords)
        assert [r.c for r in fr.drawn.rings] == [r.c for r in b61.drawn.rings]
        assert all((r.x, r.y) != (q.x, q.y) for r, q in zip(fr.drawn.rings, b61.drawn.rings))
        assert ends(fr.drawn) == ends(b61.drawn)                       # the hooks where they were
        assert np.allclose(tangle_map(fr.drawn), octet_m)
        g = graph_of(fr.drawn)
        assert g.num_edges() == 8 and same_picture(g, octet())          # the same eight wires
    assert [(r.x, r.y) for r in f23.drawn.rings] == list(P.RESHAPE_AT)  # the rings moved ...
    assert all(d.bend is None for d in f23.drawn.cords)                 # ... cords straight to them
    assert f24.drawn.rings == f23.drawn.rings                           # then the cords alone
    assert all(d.bend is not None for d in f24.drawn.cords)             # every one bent (N8)
    assert tuple(d.bend for d in f24.drawn.cords) == P.RESHAPE_BENDS
    # the strip's last frame IS round 3's P6.2, \ringcords{reshaped cradle}{as she left it}
    old = P.ringcords("", "reshaped cradle")
    assert f24.drawn.cords == old.drawn.cords and f24.drawn.rings == old.drawn.rings
    # marks=off is load-bearing: the reshape's own marks would outline the four rings where they
    # stood, and cords of the counted frame run through them
    assert [m.kind for m in P.ringcords("reshape=rings", "cradle").marks] == ["outline"] * 4
    assert P.describe(p62) == ["ringsteps: strip of 2 frames",
                               f23.log[0], "ringcords: + step reshape=rings -> {cradle, rings slid}",
                               f24.log[0], "ringcords: + step reshape=cords -> {reshaped cradle}"]

    # P6.3: pressed from the crooked one -- the very pair the neat cradle gave (P4.2)
    (p42,) = [c for c in DIALOGUE.blocks("Fourth")[1]]
    b42 = P.picture(p42)
    assert "Off the second hooks comes the very same knot-pair the neat cradle gave." in a3
    assert b63.setting == "hooks" and b63.drawn.rings == b42.drawn.rings
    assert b63.drawn.cords == b42.drawn.cords
    assert np.allclose(tangle_map(b63.drawn), pair_m)
    assert np.allclose(tangle_map(f24.drawn), tangle_map(b63.drawn) / np.sqrt(2))
    assert P.view_pressed(f24.drawn).cords == b63.drawn.cords           # pressed from the strip's end
    assert f3.startswith("*[Beyond the pane")
    last = max(i for i, q in enumerate(paras) if DIALOGUE.is_block(q))
    assert paras[last].count("\\ringcords") == 1 and p63.text in paras[last]   # the section's last call
    assert "The same little pair, out of that mare's nest. It didn't matter how you drew it." in sec


# --- pictures: all spiders phase-free (the bialgebra rule needs it) ---
def octet():
    """The Octet: two greens, two reds, a complete cross (K_2,2) between,
    two input cords and two output cords.  Eight edges = eight wires."""
    g = zx.Graph()
    i1 = g.add_vertex(VT.BOUNDARY, 0, 0); i2 = g.add_vertex(VT.BOUNDARY, 1, 0)
    gt = g.add_vertex(VT.Z, 0, 1); gb = g.add_vertex(VT.Z, 1, 1)
    rt = g.add_vertex(VT.X, 0, 2); rb = g.add_vertex(VT.X, 1, 2)
    o1 = g.add_vertex(VT.BOUNDARY, 0, 3); o2 = g.add_vertex(VT.BOUNDARY, 1, 3)
    g.add_edge((i1, gt)); g.add_edge((i2, gb))         # (1) top feed, (2) bottom feed
    g.add_edge((gt, rt)); g.add_edge((gt, rb))         # (3),(4) top green to both reds
    g.add_edge((gb, rt)); g.add_edge((gb, rb))         # (5),(6) bottom green to both reds
    g.add_edge((rt, o1)); g.add_edge((rb, o2))         # (7) top draw, (8) bottom draw
    g.set_inputs((i1, i2)); g.set_outputs((o1, o2))
    return g


def pair():
    """A single green-red pair: both inputs into one RED spider, one cord to one
    GREEN spider, both outputs from the green.  (= Complementarity.py's bialgebra RHS.)"""
    g = zx.Graph()
    i1 = g.add_vertex(VT.BOUNDARY, 0, 0); i2 = g.add_vertex(VT.BOUNDARY, 1, 0)
    r = g.add_vertex(VT.X, 0, 1); z = g.add_vertex(VT.Z, 0, 2)
    o1 = g.add_vertex(VT.BOUNDARY, 0, 3); o2 = g.add_vertex(VT.BOUNDARY, 1, 3)
    g.add_edge((i1, r)); g.add_edge((i2, r)); g.add_edge((r, z))
    g.add_edge((z, o1)); g.add_edge((z, o2))
    g.set_inputs((i1, i2)); g.set_outputs((o1, o2))
    return g


def octet_deformed():
    """The SAME Octet, drawn differently: spiders placed at other coordinates and
    the cords added in a different order.  Same connectivity, so it must have the
    same tensor -- only topology matters."""
    g = zx.Graph()
    # the four boundary endpoints keep their top/bottom identity (that IS which wire is
    # which); the interior rings are placed at scrambled coordinates and the cords are
    # strung in a different order -- a genuine redrawing of the same connectivity
    rb = g.add_vertex(VT.X, 5, 9); gt = g.add_vertex(VT.Z, 2, 1)
    o2 = g.add_vertex(VT.BOUNDARY, 1, 4); i1 = g.add_vertex(VT.BOUNDARY, 0, 0)
    gb = g.add_vertex(VT.Z, 3, 8); rt = g.add_vertex(VT.X, 6, 0)
    o1 = g.add_vertex(VT.BOUNDARY, 0, 4); i2 = g.add_vertex(VT.BOUNDARY, 1, 0)
    g.add_edge((gb, rb)); g.add_edge((gt, rt)); g.add_edge((i2, gb))
    g.add_edge((rt, o1)); g.add_edge((gt, rb)); g.add_edge((i1, gt))
    g.add_edge((gb, rt)); g.add_edge((rb, o2))
    g.set_inputs((i1, i2)); g.set_outputs((o1, o2))
    return g


def sketchbook():
    # the wire count, checked in code: eight edges = eight wires
    o = octet()
    assert o.num_edges() == 8
    # each interior ring has degree 3 (one boundary cord + two cross cords): the K_2,2
    assert all(len(list(o.neighbors(v))) == 3
               for v in o.vertices() if o.type(v) in (VT.Z, VT.X))

    # the master equality: Octet = single green-red pair, UP TO sqrt2
    assert same_picture_up_to_scalar(o, pair())
    assert np.allclose(matrix_of(o), matrix_of(pair()) / np.sqrt(2))   # cradle = pair / sqrt2

    # DERIVE it with the NAMED bialgebra rule, in the clean unfold direction pair -> cradle.
    # Locate the two interior spiders by colour (the rule renumbers vertices, so we must
    # not trust saved handles).
    g = pair()
    v_red = [v for v in g.vertices() if g.type(v) == VT.X][0]
    v_green = [v for v in g.vertices() if g.type(v) == VT.Z][0]
    assert simp.check_bialgebra(g, v_red, v_green)
    assert simp.bialgebra(g, v_red, v_green)                           # unfold
    assert g.num_edges() == 8                                          # the count is PRODUCED
    assert same_picture_up_to_scalar(g, octet())                       # and it is the Octet

    # ONLY TOPOLOGY MATTERS: the same connectivity drawn differently is the same tensor,
    # scalar included (a genuine deformation, not a rewrite -- no rule was applied)
    assert same_picture(octet_deformed(), octet())

    frame_is_the_octet()


# --------------------------------------------------------------------------
# The frame on the beam is the Octet.  The transcription's cradle, strung as a
# PyZX graph ring by ring and cord by cord, is octet() -- the same eight edges
# under the wire map of the identity card (left hook top <-> i_top, top green
# <-> g_top, ...), and the same picture, exactly; the reshaped cradle too; and
# the pressed pair is pair().  (A structural tie, not a second proof.)
# --------------------------------------------------------------------------
def graph_of(T):
    """A tangle as a PyZX graph: a green ring a Z spider, a red ring an X spider,
    its turn t the phase t/12 (in units of pi); every cord segment one edge; the
    hooks on the left the inputs, on the right the outputs, the upper line first."""
    g = zx.Graph()
    v = {}
    segs = P.segments(T)
    hooks = sorted({e for s in segs for e in s if isinstance(e, P.End)}, key=lambda e: -e.y)
    for e in hooks:
        v[e] = g.add_vertex(VT.BOUNDARY, 0 if e.y > 0 else 1, 0 if e.side == "L" else 3)
    for n, r in enumerate(T.rings, 1):
        v[n] = g.add_vertex(VT.Z if r.c == "green" else VT.X, 0 if r.y > 0 else 1,
                            1 if r.c == "green" else 2, phase=Fraction(r.t, 12))
    for a, b in segs:
        g.add_edge((v[a], v[b]))
    assert g.num_edges() == len(segs)                    # no strand collapsed
    g.set_inputs(tuple(v[e] for e in hooks if e.side == "L"))
    g.set_outputs(tuple(v[e] for e in hooks if e.side == "R"))
    return g


def roles(g):
    """octet()'s edges as the transcription names their ends: L1/L2 the input
    hooks, 1/2 the top/bottom green, 3/4 the top/bottom red, R1/R2 the outputs."""
    ring = {(VT.Z, 0): "1", (VT.Z, 1): "2", (VT.X, 0): "3", (VT.X, 1): "4"}

    def role(u):
        q = int(g.qubit(u))
        if u in g.inputs():
            return f"L{q + 1}"
        if u in g.outputs():
            return f"R{q + 1}"
        return ring[(g.type(u), q)]
    return unordered(f"{role(a)}-{role(b)}" for a, b in (g.edge_st(e) for e in g.edges()))


def frame_is_the_octet():
    cradle = P.parse_tangle("cradle")
    reshaped = P.parse_tangle("reshaped cradle")
    o = octet()
    assert roles(o) == unordered(P.connections(cradle.cords))          # the eight edges, by name
    assert unordered(P.connections(reshaped.cords)) == roles(o)
    for T in (cradle, reshaped):
        g = graph_of(T)
        assert g.num_edges() == 8 == len(P.CRADLE_CORDS)                # eight cords, eight wires
        assert same_picture(g, o) and same_picture(g, octet_deformed())
    assert same_picture(graph_of(P.view_pressed(reshaped)), pair())
    assert same_picture(graph_of(P.view_pressed(cradle)), pair())
    # a turned ring: the graph's phase (t/12, in units of pi) is the NumPy turn 2 pi t/24
    for spec in ("green ring, turn a third", "red ring, five cords, turn seven twelfths"):
        T = P.parse_tangle(spec)
        assert np.allclose(matrix_of(graph_of(T)), tangle_map(T))
    assert not np.allclose(matrix_of(graph_of(P.parse_tangle("green ring, turn a third"))),
                           tangle_map(P.parse_tangle("green ring, turn a quarter")))
    # the frame before the cross is not the Octet
    assert graph_of(P.parse_tangle("greens and reds")).num_edges() == 4


if __name__ == "__main__":
    verdict("Octet", blackboard, sketchbook)
