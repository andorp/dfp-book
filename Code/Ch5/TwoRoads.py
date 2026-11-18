"""Challenge 1: Two Roads
=======================
"Do the amplitudes add — and can they cancel?"  -- Achilles

Claim: two mirrors undo each other, HH = I (the two roads to |0> pile up, the
walker comes home); a half-turn of phase between them is a flip, HZH = X (the
roads to |0> cancel, the walker is turned away).  Both exact, scalar included.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import numpy as np
from dfp import I2, Picture, ket, matrix_of, same_picture, verdict
from dfp_props import Dialogue, words_only
from pyzx.utils import EdgeType
import props_ch05 as P

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def blackboard():
    # two mirrors undo each other: no phase => the walker comes home
    assert np.allclose(H @ H, I2)                       # HH = I, exact, no stray scalar
    assert not np.allclose(H @ H, X)                    # not a flip
    assert np.allclose(H.conj().T, H) and np.allclose(H @ H, I2)   # H is self-inverse

    # a half-turn of phase between the mirrors is a flip
    assert np.allclose(H @ Z @ H, X)                    # HZH = X, exact
    assert not np.allclose(H @ Z @ H, I2)               # not the identity
    assert not np.allclose(H @ Z @ H, Z)               # the phase did not survive as itself

    # read the two arrival amplitudes on |0>.  With no phase the two roads to
    # |0> have amplitude +1/2 each and PILE UP to 1; the roads to |1> are
    # +1/2 and -1/2 and CANCEL to 0 -- the walker comes home.
    home = H @ H @ ket(0)
    assert np.allclose(home, ket(0))                    # amplitude of |0> = 1, of |1> = 0
    # a half-turn between the mirrors swaps the fortunes: |0> cancels, |1> fills
    away = H @ Z @ H @ ket(0)
    assert np.allclose(away, ket(1))                    # turned away to |1>

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the Tortoise's conduit-board with
#   \conduitboard[view; opts]{setting}{caption}
# and a step strip of one walk on one lane with
#   \conduitwalk[opts]{setting}{caption; caption; ...}   (pdf/props/ch05.tex).
# The Lua COMPUTES the roads, the tokens' faces, the deal, the settling, the
# motion marks of each step and what a named wrong rule predicts;
# tools/props_ch05.py transcribes it.  The calls are read out of this
# challenge's section of dialog.md (tools/dfp_props.py), never copied, so an
# edit to the dialogue that breaks a picture breaks here.
#
# The state model (identity card, section 3): a road token's face is its
# road's sign, silver = +1, slate = -1; the pocket it lies in is its value,
# home = <0|, away = <1|; each spreader is H.  A walk down one lane with both
# spreaders and M between them (I for the bare lane, Z for the peg) has four
# moments, and the strip draws them in order:
#   set out   sqrt2 H|0>                 one token per road, low road first
#   past      sqrt2 M H|0>               just past the peg
#   dealt     2 H[y,v] (M H|0>)_v        road v's token in cup y: one path each
#   end       2 <y|H M H|0>              the cup's net: the paths summed
# A ghost stands at the moment before; a flip marks a token whose sign the
# step changed; a cup whose two paths cancel loses the pair (lifted).
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=("conduitboard", "conduitwalk", "roadtiles"))
NOT_IN_CAPTIONS = ("wire", "wires", "diagram", "diagrams", "drawing", "drawings", "picture",
                   "pictures", "box", "boxes", "dot", "dots", "node", "nodes", "lanes",
                   "strings", "five")


def heap_net(heap):
    """A drawn heap (face, how many) as a signed count; a bare cup is 0."""
    return 0 if heap is None else P.SIGN[heap[0]] * heap[1]


def one_lane_model(M):
    """What a one-lane walk with both spreaders and M between them must show:
    the tokens just past M (sqrt2 x M H|0>, one per road) and the two settled
    pockets (2 x <y| H M H |0>)."""
    past = np.sqrt(2) * (M @ H @ ket(0)).ravel()
    arrive = 2 * (H @ M @ H @ ket(0)).ravel()
    return past, arrive


def walk_model(M):
    """The four moments of the walk, as signs: set out, past, the paths
    (paths[y, v]: road v's token in cup y), and each cup's net."""
    h0 = (H @ ket(0)).ravel()
    before = np.sqrt(2) * h0
    past = np.sqrt(2) * (M @ h0)
    paths = 2 * H * (M @ h0)[None, :]
    assert np.allclose(paths.sum(axis=1), 2 * (H @ M @ H @ ket(0)).ravel())   # the sum of paths
    return before, past, paths, paths.sum(axis=1)


def faces(signs):
    """Signs +-1 as the faces the board shows."""
    assert np.allclose(np.abs(signs), 1) and np.allclose(np.imag(signs), 0)
    return [P.face(float(np.real(s))) for s in signs]


def check_transcription():
    """The Lua's walk, deal and settling against the blackboard: the peg is Z
    on the lane, the foot spreader's deal is the second H."""
    for peg, M in ((False, I2), (True, Z)):
        setting = "willow peg" if peg else "willow"
        past, arrive = one_lane_model(M)
        b = P.conduitboard("past", setting)
        assert [v for v, _, _ in b.walking] == [0, 1]                  # low road, then high
        assert np.allclose([P.SIGN[f] for _, f, _ in b.walking], past)
        assert [ahead for _, _, ahead in b.walking] == [False, True]    # the high road ahead
        assert all(f == "silver" for _, f, _ in P.conduitboard("before", setting).walking)
        e = P.conduitboard("end", setting).heaps["willow"]
        assert np.allclose([heap_net(e["home"]), heap_net(e["away"])], arrive)
        for view in ("pockets", "pockets settled"):
            c = P.conduitboard(view, setting)
            assert np.allclose([P.net(c.cups["home"]), P.net(c.cups["away"])], arrive)
        c = P.conduitboard("pockets settled", setting)
        assert [sum(P.SIGN[f] * k for f, k in c.heaps[p]) for p in ("home", "away")] \
            == [heap_net(e["home"]), heap_net(e["away"])]
        # the dealt cups are the paths, one token each; a cup whose two paths agree
        # piles, a cup whose paths are opposite lies side by side
        _, _, paths, _ = walk_model(M)
        d = P.clean(P.conduitboard, "dealt", setting)
        for y, cup in enumerate(("home", "away")):
            got, how, _ = d.cups[cup]
            assert got == faces(paths[y])
            assert how == ("piled" if np.isclose(paths[y, 0], paths[y, 1]) else "side")
    # the macro's own rules: a peg never shares the board with the bar; no
    # arrival with a foot spreader lifted; past on one lane only
    assert P.faults(P.conduitboard, "set", "willow peg; reed two-faced; bar never") != []
    assert P.faults(P.conduitboard, "end", "willow no foot") != []
    assert P.faults(P.conduitboard, "past", "willow; alder") != []
    assert P.faults(P.conduitboard, "set", "willow; willow") != []
    assert P.faults(P.conduitboard, "sideways", "willow") != []
    # and the strip's: one lane, and a dealt frame only where the foot spreader stands
    assert P.faults(P.walkstrip, False, "willow; alder", "", "a") != []
    assert P.faults(P.walkstrip, False, "willow no foot", "", "a; b") != []
    # the wrong rule `add` refuses a lane where it agrees with the board (no peg)
    assert P.faults(P.conduitboard, "end; expects=add", "willow") == [
        "conduitboard: expects=add: no peg in any lane; the rule agrees with the board"]


def check_set_out(fr, before):
    """A strip's first frame: the tokens set out at the head, one per road, unmarked;
    the high road's token a little ahead of the low road's (D1)."""
    assert fr.view == "before" and not fr.marked and fr.moved is None
    assert [t["road"] for t in fr.tokens] == [0, 1]
    assert [t["face"] for t in fr.tokens] == faces(before)
    low, high = fr.tokens
    assert high["x"] - low["x"] > 0 and low["x"] < P.XBEFORE < high["x"]   # high a little ahead


def check_past(fr, ghosts, before, past):
    """Past the peg: each token's ghost where it stood before (the frame before
    it, faces and places); the tokens past the peg with the model's faces; a flip
    on each token the peg turned, and on no other; nothing else marked."""
    assert fr.view == "past" and fr.pegs == ("willow",)
    assert [(g["piece"], g["place"], g["face"], g["x"], g["y"]) for g in fr.ghosts] == \
        [("token", "before", t["face"], t["x"], t["y"]) for t in ghosts]
    assert [g["face"] for g in fr.ghosts] == faces(before)
    assert [t["face"] for t in fr.tokens] == faces(past)
    turned = [v for v in (0, 1) if not np.isclose(past[v], before[v])]
    assert [f["on"] for f in fr.flips] == [("willow", P.roadword(v)) for v in turned]
    for f, v in zip(fr.flips, turned):
        t = fr.tokens[v]
        (x0, _), _, _, (x1, _) = f["pts"]
        assert np.isclose((x0 + x1) / 2, t["x"]) and f["s"] == (1 if v == 1 else -1)
    assert fr.halos == [] and fr.pins == [] and fr.arcs == [] and fr.lifted == []
    for t in fr.tokens:                        # past the pin: clear of it, on the foot side
        assert t["x"] - P.TOKR > P.XPEG + P.PEG_R


def check_dealt(fr, came, past, paths, says):
    """Dealt: the ghosts where the tokens came past (the model's `past`), the two
    deal arcs from them to the cups, each cup's tokens the model's paths -- piled
    where the two paths agree, side by side where they are opposite, as the text
    says -- and a flip under each token the deal turned, at its own x."""
    assert fr.view == "dealt" and fr.heaps == {}
    assert [(g["place"], g["face"]) for g in fr.ghosts] == [("past", f) for f in faces(past)]
    if came is not None:                       # the frame before drew them there
        assert [(g["x"], g["y"], g["face"]) for g in fr.ghosts] == \
            [(t["x"], t["y"], t["face"]) for t in came]
    assert [(k, to) for k, to, _ in fr.arcs] == [("deal", "home"), ("deal", "away")]
    xmax = max(g["x"] for g in fr.ghosts)
    for _, to, pts in fr.arcs:
        (x0, _), _, _, (x1, y1) = pts
        cup = P.XHOME if to == "home" else P.XAWAY
        assert x0 > xmax and np.hypot(x1 - cup, y1) < P.CUPR     # from the tokens, into its cup
    home_arc = fr.arcs[0][2]
    assert P.curve_gap_circle(home_arc, P.XAWAY, 0, P.CUP_EXTENT) > 0  # clear of the away cup
    for y, cup in enumerate(("home", "away")):
        got, how, places = fr.cups[cup]
        assert got == faces(paths[y])
        assert how == ("piled" if np.isclose(paths[y, 0], paths[y, 1]) else "side")
        assert how == says[cup]                # as the text says they lie
    turned = [v for v in (0, 1) if not np.isclose(paths[1, v], paths[0, v])]
    assert turned == [1]                       # the deal turns the high road's token, away
    assert [(f["on"], f["cup"]) for f in fr.flips] == [(("willow", "high"), "away")]
    (f,) = fr.flips
    (x0, y0), _, _, (x1, _) = f["pts"]
    x_own = fr.cups["away"][2][1][0]                         # the turned token's own x
    assert np.isclose((x0 + x1) / 2, x_own) and y0 < -P.CUPR and f["s"] == -1
    for c in (P.XAWAY, P.XHOME):                             # clear of the alder's cups
        assert P.curve_gap_circle(f["pts"], c, P.ROWY["alder"], P.CUP_EXTENT) > 0
    assert fr.pins == [] and fr.halos == [] and fr.lifted == []


def check_end(fr, dealt, paths, net):
    """End: the pair of a cup whose paths cancel lifts out, its ghosts where it lay
    dealt; what is left is each cup's net."""
    assert fr.view == "end" and fr.ghosts == [] and fr.flips == [] and fr.arcs == []
    gone = [cup for y, cup in enumerate(("home", "away")) if np.isclose(paths[y].sum(), 0)]
    assert [(n, cup, fs, places) for n, cup, fs, places in fr.lifted] == \
        [("willow", cup, dealt.cups[cup][0], dealt.cups[cup][2]) for cup in gone]
    e = fr.heaps["willow"]
    assert np.allclose([heap_net(e["home"]), heap_net(e["away"])], net)


def pictures():
    P.check_lua()                                  # the Lua and .tex the transcription was proved on
    check_transcription()
    found = P.check_section(DIALOGUE, "First", sizes=[1, 1, 2], frames=[1, 3, 4],
                            forbidden=NOT_IN_CAPTIONS)
    (p11, [(b11, _)]), = found[0]
    (p12, walk1), = found[1]
    (p13e, [(b13e, _)]), (p13, walk2) = found[2]
    for c in (p11, p12, p13e, p13):
        assert c.macro in ("conduitboard", "conduitwalk")
    frames = [b11] + [f for f, _ in walk1] + [b13e] + [f for f, _ in walk2]
    for b in frames:
        assert isinstance(b, P.Board) and b.grooves == P.GROOVES        # the whole board
        assert b.live == ("willow",) and b.spreaders == {"willow": (True, True)}
        assert b.bar is None and b.reed is None                         # no bar, no reed token
        assert b.moved is None                                          # no piece lifted

    # anchors: each block sits after the speech or walk it shows (both files)
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        a1, a2, a3 = DIALOGUE.anchors("First", path)
        assert a1.endswith("without the channel ever splitting.")
        assert "Both come up silver, the bright face." in a2 and "peg" not in a2
        assert "as the high road passes the peg, its token turns over" in a3
    a1, a2, a3 = DIALOGUE.anchors("First")

    # P1.1: the board as set, keyed; the willow alone in play
    assert p11.keyed and b11.view == "set" and not b11.marked
    assert b11.labels == ("willow", "alder", "hazel", "rowan", "reed", "home", "away")
    assert b11.pegs == () and b11.walking == () and b11.heaps == {}

    # Block 1.2, walk 1 (the bare lane): a strip of three frames, set out -> dealt
    # -> end; no past frame, for nothing stands in the lane's way
    assert p12.macro == "conduitwalk" and not p12.keyed and P.options(p12) == {"view": ""}
    strip = P.clean(P.walkstrip, p12.keyed, p12.args[0], p12.opt, p12.caption)
    assert [(f.opts, f.state, f.caption) for f in strip] == [
        ("before", "willow", "set out"), ("dealt;marks=on", "willow", "dealt"),
        ("end;marks=on", "willow", "home heaped, away bare")]
    f02, f03, f04 = (f for f, _ in walk1)
    before, past, paths, net = walk_model(I2)
    assert np.allclose(past, before)                                    # nothing turned them
    check_set_out(f02, before)
    # card l.231: "silver upon silver, and they pile" (home); "silver beside slate" (away)
    says = {"home": "piled" if "silver upon silver, and they pile" in a2 else None,
            "away": "side" if "silver beside slate, unlike faces" in a2 else None}
    check_dealt(f03, None, past, paths, says)
    assert f03.cups["home"][:2] == (["silver", "silver"], "piled")
    assert f03.cups["away"][:2] == (["silver", "slate"], "side")
    check_end(f04, f03, paths, net)
    assert f04.lifted[0][:3] == ("willow", "away", ["silver", "slate"])
    assert f04.heaps["willow"] == {"home": ("silver", 2), "away": None}
    assert np.allclose(net, 2 * ket(0).ravel())                         # HH = I: home, net +2

    # Block 1.3, the peg walk: Achilles's prediction (rule `add`, card l.239 "I *said*
    # the chances only add"), then the walk from past the peg
    assert P.options(p13e) == {"view": "end", "expects": "add"} and not p13e.keyed
    assert p13e.args == p13.args == ("willow peg",)
    assert b13e.expects == "add" and not b13e.marked and b13e.pegs == ("willow",)  # peg drawn
    assert b13e.setting.nopeg                                           # ... and passed untouched
    _, _, _, net_add = walk_model(I2)                                   # the peg turns nothing
    assert np.allclose([heap_net(b13e.heaps["willow"][p]) for p in ("home", "away")], net_add)
    assert b13e.heaps == f04.heaps                                      # the bare lane's walk
    assert P.options(p13) == {"view": "", "from": "off"} and not p13.keyed
    strip = P.clean(P.walkstrip, p13.keyed, p13.args[0], p13.opt, p13.caption)
    assert [(f.opts, f.caption) for f in strip] == [
        ("past;marks=on", "past the peg"), ("dealt;marks=on", "dealt"),
        ("end;marks=on", "home bare, away heaped")]
    f06, f07, f08 = (f for f, _ in walk2)
    before, past, paths, net = walk_model(Z)
    # past the peg: the ghosts at the head, as walk 1 set them out (the pockets
    # tipped and the peg pressed in between); the high road's token turned
    check_past(f06, f02.tokens, before, past)
    assert [f["on"] for f in f06.flips] == [("willow", "high")]         # the low road: no flip
    assert np.allclose(past, np.diag(Z))                                # +1 low, -1 high
    # "deals as before" (card l.249) and "slate against your silver" (card l.263)
    assert "At the foot, the spreader deals as before." in a3
    at263, blocks = P.where(DIALOGUE, "First", "lay in the home pocket slate against your silver")
    assert blocks[2] < at263
    says = {"home": "side" if "slate against your silver" in DIALOGUE.paragraphs("First")[at263]
            else None,
            "away": "piled" if "deals as before" in a3 else None}     # like faces, as in walk 1
    check_dealt(f07, f06.tokens, past, paths, says)
    assert f07.cups["home"][:2] == (["silver", "slate"], "side")
    assert f07.cups["away"][:2] == (["silver", "silver"], "piled")
    check_end(f08, f07, paths, net)
    assert f08.lifted[0][:3] == ("willow", "home", ["silver", "slate"])
    assert f08.heaps["willow"] == {"home": None, "away": ("silver", 2)}
    assert np.allclose(net, 2 * (X @ ket(0)).ravel())                   # HZH = X: away
    # the prediction differs from the true frame beside it
    assert b13e.heaps != f08.heaps
    assert heap_net(b13e.heaps["willow"]["home"]) != heap_net(f08.heaps["willow"]["home"])

    # where the pictures stand: the tokens tipped out between the two walks (the
    # re-lay's clearing); the prediction after the walk, before card l.257, and nothing
    # inside the returning passage that follows
    for path in (DIALOGUE.dialog, DIALOGUE.raw):
        at, blocks = P.where(DIALOGUE, "First", "She tips the tokens out of the pockets", path)
        assert blocks[1] < at < blocks[2]
        at257, _ = P.where(DIALOGUE, "First", "nothing touched the roads except a pin", path)
        assert blocks[2] < at257 and len(blocks) == 3
    # the peg midway between the spreaders (card l.245), halfway down (card l.247)
    text = DIALOGUE.section("First")
    assert "midway between the two spreaders" in text and "halfway down" in text
    assert abs(P.XPEG - (P.XHEAD + P.XFOOT) / 2) < P.PEG_R             # the midpoint under the peg
    # the flip over the high road's token: clear of the board's edge and the foot spreader
    (fl,) = f06.flips
    assert max(y for _, y in P.bezier(fl["pts"])) + P.SWING_W / 2 < P.TOP
    assert P.curve_gap_circle(fl["pts"], P.XFOOT, 0, P.SPREADER_EXTENT) > 0

    # the directions edited this round (D1, D2a, D2b), in both files
    P.check_edits(DIALOGUE, "First", [
        ("down the lane the two roads run together, and each lets fall its token at the foot.",
         "down the lane the two roads run together, the high road's token a little ahead of the "
         "low road's, and each lets fall its token at the foot."),
        ("Two roads again from the one head spreader, the high road's token running a little "
         "ahead of the low road's; but this time,",
         "Two roads again from the one head spreader; but this time,"),
        ("while the low road slips past the pin and stays silver.",
         "while the low road slips past the pin.")])
    assert "the high road's token a little ahead of the low road's" in a2   # D1: at its first use

    assert P.section_dumps(found) == [
        "conduitboard: set | willow",
        "conduitboard: + reed none",
        "conduitboard: before | willow",
        "conduitboard: + road willow=low silver",
        "conduitboard: + road willow=high silver",
        "conduitboard: + reed none",
        "conduitboard: + token willow low silver at before",
        "conduitboard: + token willow high silver at before",
        "conduitboard: dealt marks | willow",
        "conduitboard: + road willow=low silver",
        "conduitboard: + road willow=high silver",
        "conduitboard: + reed none",
        "conduitboard: + ghost token willow low silver at past",
        "conduitboard: + ghost token willow high silver at past",
        "conduitboard: + deal willow home away",
        "conduitboard: + cup willow home silver,silver piled",
        "conduitboard: + cup willow away silver,slate side",
        "conduitboard: + flip willow high in away",
        "conduitboard: end marks | willow",
        "conduitboard: + road willow=low silver",
        "conduitboard: + road willow=high silver",
        "conduitboard: + reed none",
        "conduitboard: + lifted willow away silver,slate",
        "conduitboard: + cup willow home heap silver 2",
        "conduitboard: + cup willow away bare",
        "conduitboard: end expects add | willow peg",
        "conduitboard: + road willow=low silver",
        "conduitboard: + road willow=high silver",
        "conduitboard: + reed none",
        "conduitboard: + cup willow home heap silver 2",
        "conduitboard: + cup willow away bare",
        "conduitboard: past marks | willow peg",
        "conduitboard: + road willow=low silver",
        "conduitboard: + road willow=high slate",
        "conduitboard: + reed none",
        "conduitboard: + ghost token willow low silver at before",
        "conduitboard: + ghost token willow high silver at before",
        "conduitboard: + token willow low silver at past",
        "conduitboard: + token willow high slate at past",
        "conduitboard: + flip willow high",
        "conduitboard: dealt marks | willow peg",
        "conduitboard: + road willow=low silver",
        "conduitboard: + road willow=high slate",
        "conduitboard: + reed none",
        "conduitboard: + ghost token willow low silver at past",
        "conduitboard: + ghost token willow high slate at past",
        "conduitboard: + deal willow home away",
        "conduitboard: + cup willow home silver,slate side",
        "conduitboard: + cup willow away silver,silver piled",
        "conduitboard: + flip willow high in away",
        "conduitboard: end marks | willow peg",
        "conduitboard: + road willow=low silver",
        "conduitboard: + road willow=high slate",
        "conduitboard: + reed none",
        "conduitboard: + lifted willow home silver,slate",
        "conduitboard: + cup willow home bare",
        "conduitboard: + cup willow away heap silver 2"]


def hedge(p, a, b):
    """A Hadamard edge -- the box H, drawn the exact Stage-A way (see the card):
    NOT an Euler phase-node chain, which would carry a stray e^{i pi/4}."""
    p.g.add_edge((a, b), EdgeType.HADAMARD)


def two_mirrors(middle_phase):
    """One wire: H, then a Z-phase of `middle_phase` half-turns, then H."""
    p = Picture()
    i = p.inp(0)
    mid = p.z(0, 2, middle_phase)   # PyZX draws this with the spiders you meet in Chapter 8.
    o = p.out(0)
    hedge(p, i, mid)                # first mirror
    hedge(p, mid, o)               # second mirror
    return p.done()


def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def gate(kind, phase):
    p = Picture()
    node = p.z(0, 2, phase) if kind == "Z" else p.x(0, 2, phase)
    p.wire(p.inp(0), node, p.out(0))
    return p.done()


def sketchbook():
    # no phase between the mirrors: H-H is a bare wire, exactly
    assert np.allclose(matrix_of(two_mirrors(0)), I2)
    assert same_picture(two_mirrors(0), bare_wire())            # HH = I
    assert not same_picture(two_mirrors(0), gate("X", 1))       # not a flip

    # a half-turn (Z, one half-turn = phase pi) between the mirrors is a flip
    assert same_picture(two_mirrors(1), gate("X", 1))           # HZH = X
    assert not same_picture(two_mirrors(1), bare_wire())        # not the identity
    assert not same_picture(two_mirrors(1), gate("Z", 1))       # not a bare Z


if __name__ == "__main__":
    verdict("TwoRoads", blackboard, sketchbook)
