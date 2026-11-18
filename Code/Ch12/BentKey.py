"""Challenge IV: Backwards to the Lamp  (E91 is BB84 bent; transpose versus conjugate)
====================================================================================
"Round the bend, your question reaches my end as its reflection."  -- Tortoise

Claim: (i) for all 2x2 U1, U2, (U1 (x) U2)|Phi+> = (I (x) U2 U1^T)|Phi+> -- a
TRANSPOSE, not a dagger; (ii) for every state phi, (<phi| (x) I)|Phi+> =
(1/sqrt2)|phi*> -- a reading slid round the bend arrives CONJUGATED; (iii) with
Alice at the plain or cross setting and Bob at the plain, cross or circular one,
the pair gives p(kA, kB | r, t) = 1/2 P_BB84(y = kB | x = kA, r, t): the BB84 round
fed a fair coin, and in the picture the cup and Alice's reader fuse into her sender;
(iv) with both at the circular setting the pair ALWAYS disagrees,
p = 1/2 [kA != kB], where the BB84 round at the circular setting always agrees.
"""
import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from fractions import Fraction

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT, EdgeType as ET

from dfp import BELL, CUP, I2, Picture, kron, ket, matrix_of, random_matrix, random_state, \
    same_picture, verdict
from dfp_props import Dialogue
import props_ch12 as PC


H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j])                                         # the quarter-turn Z(pi/2)
DELTA = np.outer(ket(0, 0), ket(0)) + np.outer(ket(1, 1), ket(1))
C1 = np.array([1, 1j]) / np.sqrt(2)                          # BB84's circular basis, as printed
C2 = np.array([1j, 1]) / np.sqrt(2)
PLAIN, CROSS, CIRC = 0, 1, 2
BASIS = {PLAIN: (ket(0), ket(1)), CROSS: (H @ ket(0), H @ ket(1)), CIRC: (C1, C2)}


def bb84(y, x, r, t):
    """P(y | x, r, t): Alice sends BASIS[r][x]; Bob's effects are <BASIS[t][y]|."""
    return abs(np.vdot(BASIS[t][y], BASIS[r][x])) ** 2


def pair(kA, kB, r, t):
    """p(kA, kB | r, t): Phi+ read with the effects <BASIS[r][kA]| (x) <BASIS[t][kB]|."""
    return abs(np.vdot(np.kron(BASIS[r][kA], BASIS[t][kB]), BELL)) ** 2


def blackboard():
    rng = np.random.default_rng(1204)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))      # cup unnormalized
    assert np.allclose(BELL, CUP / np.sqrt(2))

    # (ii) a reading slid round the bend is a CONJUGATE ket -- on a NON-REAL phi
    for _ in range(5):
        phi = random_state(rng)
        assert not np.isclose(abs(np.vdot(phi.conj(), phi)), 1)   # phi* is not phi (times a phase)
        slid = (kron(phi.conj().reshape(1, 2), I2) @ BELL).ravel()
        assert np.allclose(slid, phi.conj() / np.sqrt(2))
        assert not np.allclose(slid, phi / np.sqrt(2))

    # (i) Gisin et al.'s Eq. (10) on random complex, non-symmetric, non-unitary U1, U2
    for _ in range(5):
        U1, U2 = random_matrix(rng), random_matrix(rng)
        for U in (U1, U2):
            assert not np.allclose(U, U.T) and not np.allclose(U, U.conj())
        assert not np.allclose(U1 @ U2, U2 @ U1)
        lhs = (kron(U1, U2) @ BELL).ravel()
        assert np.allclose(lhs, (kron(I2, U2 @ U1.T) @ BELL).ravel())          # transpose: yes
        assert not np.allclose(lhs, (kron(I2, U2 @ U1.conj().T) @ BELL).ravel())  # dagger: no
        assert not np.allclose(lhs, (kron(I2, U2 @ U1) @ BELL).ravel())        # bare U1: no
        assert not np.allclose(lhs, (kron(I2, U2 @ U1.conj()) @ BELL).ravel()) # conjugate: no
        assert not np.allclose(lhs, (kron(I2, U1.T @ U2) @ BELL).ravel())      # wrong order: no
    # Dirac's Blackboard's one-entry check: U1 = diag(1, i)
    U1 = np.diag([1, 1j])
    assert np.allclose((kron(U1, I2) @ CUP).ravel(), ket(0, 0) + 1j * ket(1, 1))
    assert np.allclose((kron(I2, U1.T) @ CUP).ravel(), ket(0, 0) + 1j * ket(1, 1))
    assert np.allclose((kron(I2, U1.conj().T) @ CUP).ravel(), ket(0, 0) - 1j * ket(1, 1))

    # (iii) the pair is the BB84 round fed a fair coin: Alice plain/cross, Bob all three
    for r in (PLAIN, CROSS):
        for t in (PLAIN, CROSS, CIRC):
            for kA in (0, 1):
                for kB in (0, 1):
                    assert np.isclose(pair(kA, kB, r, t), 0.5 * bb84(kB, kA, r, t))
    # the plain and cross readings are real, so nothing shows; the circular ones are not
    assert np.allclose(BASIS[CROSS][1], BASIS[CROSS][1].conj())
    assert not np.allclose(C1, C1.conj())
    # (iv) circular-circular: c1* = -i c2 and c2* = -i c1, so the pair ALWAYS disagrees ...
    assert np.allclose(C1.conj(), -1j * C2) and np.allclose(C2.conj(), -1j * C1)
    p_pair = np.array([[pair(a, b, CIRC, CIRC) for b in (0, 1)] for a in (0, 1)])
    p_bb84 = np.array([[0.5 * bb84(b, a, CIRC, CIRC) for b in (0, 1)] for a in (0, 1)])
    assert np.allclose(p_pair, 0.5 * (1 - np.eye(2)))          # 1/2 [kA != kB]
    assert np.allclose(p_bb84, 0.5 * np.eye(2))                # BB84: [y = x], every time
    assert not np.allclose(p_pair, p_bb84)                     # REQUIRED: here they differ
    # ... and had the Tortoise read her windows the other way about, they would have agreed
    assert np.allclose(p_pair[:, ::-1], p_bb84)
    # the ledger of IV (yes = 0, no = 1): upright Yes/Yes, crosswise No/No, then at the
    # curl Yes/No, No/Yes, No/Yes -- every round possible, every curl round a discord
    for r, a, b in ((PLAIN, 0, 0), (CROSS, 1, 1), (CIRC, 0, 1), (CIRC, 1, 0), (CIRC, 1, 0)):
        assert pair(a, b, r, r) > 0
        assert (a != b) == (r == CIRC)

    # the BB84 paper's EPR state is the singlet 0.7071(r1 r2 - r2 r1): anticorrelated at
    # EVERY setting -- tested on random orthonormal bases, the circular one, and cross
    singlet = (ket(0, 1) - ket(1, 0)) / np.sqrt(2)
    for _ in range(5):
        Q, _r = np.linalg.qr(random_matrix(rng))
        u = (Q[:, 0], Q[:, 1])
        assert np.isclose(sum(abs(np.vdot(np.kron(u[k], u[k]), singlet)) ** 2 for k in (0, 1)), 0)
        # while Phi+ read at a random non-real basis at both ends neither always agrees nor
        # always disagrees: it agrees always only for real settings like plain and cross
        agree = sum(abs(np.vdot(np.kron(u[k], u[k]), BELL)) ** 2 for k in (0, 1))
        assert 1e-6 < agree < 1 - 1e-6
    for s in (PLAIN, CROSS, CIRC):
        b = BASIS[s]
        assert np.isclose(sum(abs(np.vdot(np.kron(b[k], b[k]), singlet)) ** 2 for k in (0, 1)), 0)
    assert np.allclose((np.kron(C1, C2) - np.kron(C2, C1)) / np.sqrt(2), singlet)
    d1, d2 = BASIS[CROSS]                                       # BB84's diagonal d1, d2
    # "the same state vector is demonstrably equal to 0.7071(d1 d2 - d2 d1)": equal up to
    # the global sign -1 (the same physical state; as a vector it is -singlet)
    assert np.allclose(np.kron(d1, d2) - np.kron(d2, d1), -np.sqrt(2) * singlet)
    pictures()


# --- the prop pictures (dialog.md: \keycloth, \sparkey; tools/props_ch12.py) -------------

DIALOGUE = Dialogue(__file__, macros=PC.MACROS)
QUOTE = "Round the bend, your question reaches my end as its reflection."
STOP = {"upright": PLAIN, "crosswise": CROSS, "twist": CIRC}     # the collar's stops, the settings


def engraved(p):
    """The key's engraved marks as drawn on the page (round 5: keymarks, after both bodies),
    read off the picture's layers by composing each scope's shift, rotate, scale and yscale
    in TikZ's order: [(ink, mark, line width in pt, the mark's origin, its points), ...], the
    reflection's three first.  The mark is told by its points: 2 a stroke, 4 a cross, more
    the curl."""
    out, stack = [], [np.eye(3)]
    for line in p.layers:
        if line.startswith("\\begin{scope}"):
            M = np.eye(3)
            opts = re.finditer(r"(shift)=\{\(([-\d.]+),([-\d.]+)\)\}|(rotate)=([-\d.]+)|\b(y?scale)=([-\d.]+)", line)
            for m in opts:
                key, a, b = m.group(1) or m.group(4) or m.group(6), m.group(2) or m.group(5) or m.group(7), m.group(3)
                if key == "shift":
                    T = np.array([[1, 0, float(a)], [0, 1, float(b)], [0, 0, 1]])
                elif key == "rotate":
                    c, s = np.cos(np.radians(float(a))), np.sin(np.radians(float(a)))
                    T = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                elif key == "scale":
                    T = np.diag([float(a), float(a), 1])
                else:
                    T = np.diag([1, float(a), 1])
                M = M @ T
            stack.append(stack[-1] @ M)
        elif line == "\\end{scope}":
            stack.pop()
        else:
            m = re.match(r"\\draw\[(\w+), line width=([\d.]+)pt[^\]]*\] (.*);$", line)
            pts = [tuple(map(float, q)) for q in re.findall(r"\(([-\d.]+),([-\d.]+)\)", m.group(3))]
            page = [tuple((stack[-1] @ np.array([x, y, 1]))[:2]) for x, y in pts]
            mark = {2: "stroke", 4: "cross"}.get(len(pts), "curl")
            out.append((m.group(1), mark, float(m.group(2)), tuple((stack[-1] @ np.array([0, 0, 1]))[:2]), page))
    assert len(stack) == 1
    return out


def winding(origin, pts):
    """The turning of a drawn spiral about its origin, in degrees: negative is clockwise on the
    page, the way the sun goes round."""
    ang = np.unwrap([np.arctan2(y - origin[1], x - origin[0]) for x, y in pts])
    return np.degrees(ang[-1] - ang[0])


def tex_rgb(name):
    """A colour of ch12.tex, as its RGB triple."""
    (rgb,) = re.findall(r"\\definecolor\{" + name + r"\}\{RGB\}\{(\d+),(\d+),(\d+)\}", PC.TEX.read_text())
    return np.array([float(v) for v in rgb])


def grey(rgb):
    return float(np.dot([0.299, 0.587, 0.114], rgb))


def box_clear(box, pts, reach):
    """The least distance from a polyline's ink (half-width reach) to a box
    (x0, y0, x1, y1), sampled finely along the polyline."""
    x0, y0, x1, y1 = box
    best = None
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        for k in range(101):
            x, y = ax + (bx - ax) * k / 100, ay + (by - ay) * k / 100
            d = np.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1)) - reach
            best = d if best is None else min(best, d)
    return best


def pictures():
    assert f'"{QUOTE}' in __doc__                                # quoted byte for byte
    calls = DIALOGUE.check_layout("Fourth", sizes=[1, 1, 1])                # round 4: two strips, a prediction
    b41, b42a, b42 = calls
    assert [(c.macro, c.keyed, c.opt, c.args, c.caption) for c in calls] == [
        ("keyclothsteps", True, "", ("the lamp's bend", "pull straight"), "bent round where the lamp stands; pulled, the bend run out"),
        ("sparkey", False, "expects=same setting", ("the curl",), "as Achilles expects"),
        ("sparksteps", True, "", ("the curl", "the cross; the stroke; the curl"),
         "as he found it; back to the cross; on to the stroke; round again, winding the other way beneath")]
    PC.check_captions(DIALOGUE, "Fourth", calls)
    PC.check_section(DIALOGUE, "Fourth", calls)
    assert [[PC.height(c) for c in b] for b in DIALOGUE.blocks("Fourth")] == [[1.78], [PC.KEY_FRAME], [PC.KEY_FRAME]]

    a1, a2a, a2 = DIALOGUE.anchors("Fourth")
    f1, f2a, f2 = DIALOGUE.following("Fourth")
    for phrase in ("where each short cord ends", "the table's own line, from his key through the lamp to hers",
                   "The rings are drawn apart towards the edges, riding on their cords"):       # D12
        assert phrase in a1, phrase
    assert f1.startswith("**Achilles:** A sending ring and a reading ring on the same cords")      # S1
    # B4.2a (round 4): the prediction stands where Achilles still holds the rule -- after he has
    # heard every curl round disagree and blames the line, after "Look at your collar in the
    # tray.", before he bends down and looks (card section 3e.1, Q-E); B4.2 then after the look
    assert "Every one wrong!" in DIALOGUE.paragraphs("Fourth")[DIALOGUE.paragraphs("Fourth").index(a2a) - 1]
    assert a2a == "**Tortoise:** Nothing is on the line. Look at your collar in the tray."
    assert f2a.startswith("*[Achilles bends down to the lacquer.") and f2a == a2
    assert "only the curl changes hands" in a2
    assert "He turns it onward, the same way round, to the curl again" in a2                  # T1
    assert "He turns it on past the cross" not in a2
    assert f2.startswith("**Tortoise:** Round the bend, your question reaches my end as its reflection.")
    assert DIALOGUE.following("Fourth", DIALOGUE.raw) == [f1, f2a, f2]
    assert DIALOGUE.anchors("Fourth", DIALOGUE.raw) == [a1, a2a, a2]
    section = DIALOGUE.section("Fourth")
    for phrase in ("on one cord", "both run out to the left", "one pair of cords"):          # S1, D12
        assert phrase not in section, phrase
    assert section.index(b42a.text) < section.index(b42.text)                    # prediction, then the look

    # the frames, as TeX runs them
    assert [(f.keyed, PC.bracket(f), f.args[0]) for f in PC.frame_calls(b41)] == [
        (True, "", "the lamp's bend"), (False, "step=pull straight", "the lamp's bend")]
    assert [(f.keyed, PC.bracket(f), f.args[0]) for f in PC.frame_calls(b42)] == [
        (True, "", "the curl"), (False, "step=the cross", "the curl"), (False, "step=the stroke", "the cross"),
        (False, "step=the curl", "the stroke")]
    (c41, _), (c42, _) = PC.pictures(b41)
    k20 = PC.picture(b42a)
    (k21, _), (k22, _), (k23, _), (k24, _) = PC.pictures(b42)
    k43, k44 = k22, k24                                          # round 3's P4.3 (the cross) and P4.4 (the curl)

    # F19: pulled straight -- each ring's ghost where it stood, no arc (the Lua: beneath the
    # row the mirror cord leaves each ring inwards, no room for a head)
    assert c42.before == c41.state and c42.step == "pull straight"
    assert PC.ghost_places(c42) == [((-0.42, -0.18), 0.09), ((0.42, -0.18), 0.09)] and PC.arc_ends(c42) == []
    assert c42.log[-1] == ("keycloth: + no arc: beneath the row the mirror cord leaves each ring inwards, "
                           "leaving room for a head and no arc")
    ghosts = [g for g, _ in PC.ghost_places(c42)]                # where the rings stood, on the pulled frame's grid
    assert np.allclose(ghosts, [PC.cloth_cm(c42, r["x"], r["y"]) for r in c41.cloth["things"]], atol=5e-4)
    assert not np.allclose(ghosts, [PC.cloth_cm(c42, r["x"], r["y"]) for r in c42.cloth["things"]], atol=5e-4)
    assert round(PC.mark_clearance(c42), 3) == 0.082 and PC.mark_clearance(c41) is None
    assert PC.frame_of(PC.twin(PC.frame_calls(b41)[1])) == PC.frame_of(PC.keycloth("pull straight", "the lamp's bend"))

    # F20: the prediction, Achilles's rule "same setting" drawn: at the curl, the reflection's
    # curl turned back over (winding as the curl above), the key's body and the lacquer paler
    # (card 3e.8 E1: the marks are drawn after the pale group closes, unpaled), unmarked,
    # unkeyed.  Its dump head still gives the true winding; the '+' line gives what is drawn.
    assert PC.is_prediction(b42a) and not PC.judged(b42a) and k20.rule == "same setting"
    assert k20.dump == "stop twist; top curl; above curl sunwise; beneath curl against the sun"
    assert k20.log[1] == ("sparkey: + expects same setting: the reflection's curl winds as the curl above "
                          "(the body and the lacquer paler, the marks not)")
    assert k20.shown["above"] == k20.shown["beneath"] == "sunwise"                  # the wrong rule ...
    assert k21.shown["beneath"] == "against the sun" != k20.shown["beneath"]        # ... the true frame differs
    assert k20.mark_tikz == ("\\begin{scope}[transparency group, opacity=0.500]", "\\end{scope}")
    assert [s for s in k20.scopes if "yscale=-1" in s] == ["\\begin{scope}[shift={(0.000,0.237)}, rotate=0.000, yscale=-1]"]
    assert [s.replace(", yscale=-1", "") for s in k20.scopes] == list(k21.scopes)   # only the reflection's curl differs
    assert k20.labels == () and PC.arc_lines(k20) == []
    # E1 (card 3e.8), the curl's winding legible at print size.  The marks lie outside the pale
    # group: it opens and closes first, and every engraved line comes after it, unpaled
    assert k20.layers[:2] == k20.mark_tikz and not any("opacity" in l for l in k20.layers[2:])
    assert all("transparency" not in l for l in k21.layers) and len(engraved(k21)) == len(engraved(k20)) == 6
    for k in (k20, k21):
        marks = engraved(k)
        assert [(ink, m) for ink, m, *_ in marks] == [("propchtwelvemarkglint", m) for m in ("stroke", "cross", "curl")] + \
            [("propchtwelveengrave", m) for m in ("stroke", "cross", "curl")]      # the reflection's in the glint
        assert [w for _, _, w, *_ in marks] == [1.1, 1.1, 0.75] * 2
        (_, _, _, o_up, up), = [x for x in marks if x[:2] == ("propchtwelveengrave", "curl")]
        (_, _, _, o_dn, dn), = [x for x in marks if x[:2] == ("propchtwelvemarkglint", "curl")]
        rel_up, rel_dn = np.subtract(up, o_up), np.subtract(dn, o_dn)
        # the curl 0.18 cm across at print size (the picture is drawn in cm, unscaled), its stroke
        # and its cross the same size class; its winding sunwise above, a turn and a half
        assert 2 * max(np.hypot(*q) for q in rel_up) >= 0.18 and 2 * PC.MARKR * PC.KEYSC >= 0.18
        assert np.isclose(winding(o_up, up), -540, atol=1)
        if k is k20:      # Achilles's rule: the reflection's curl is the very glyph above
            assert np.allclose(rel_dn, rel_up, atol=2e-3) and np.isclose(winding(o_dn, dn), -540, atol=1)
        else:             # the mirror: the same glyph flipped top to bottom, winding against the sun
            assert np.allclose(rel_dn, rel_up * [1, -1], atol=2e-3) and np.isclose(winding(o_dn, dn), 540, atol=1)
            assert not np.allclose(rel_dn, rel_up, atol=2e-3)
    # the glint against what lies beneath it: the reflection's collar, the brass dimmed into the
    # lacquer (C(): 40 per cent brass); in F20 paled at PALE over the white page (the frame's
    # bounding path draws nothing).  Grey-level contrast at least 60/255 everywhere on the ring
    glint, lacquer = tex_rgb("propchtwelvemarkglint"), tex_rgb("propchtwelvelacquer")
    for b in ("propchtwelvebrasslight", "propchtwelvebrass", "propchtwelvebrassdark"):
        dim = 0.4 * tex_rgb(b) + 0.6 * lacquer
        pale = PC.PALE * dim + (1 - PC.PALE) * 255
        assert grey(glint) - grey(dim) >= 60 and grey(glint) - grey(pale) >= 60, b
    assert grey(glint) - grey(PC.PALE * (0.4 * tex_rgb("propchtwelvebrasslight") + 0.6 * lacquer) + (1 - PC.PALE) * 255) < 70
    # every spoken reaction, checked against the rule (card section 3e.1's table): stated; the
    # upright and crosswise rounds agree (as the rule says, and as the true frames show: the
    # mirror agrees there); the three curl rounds all disagree; still held after them (he
    # blames the line); the pale frame; refuted in the lacquer; recanted
    paras = DIALOGUE.paragraphs("Fourth")
    at = lambda s: next(i for i, q in enumerate(paras) if s in q)
    said = lambda i: [re.match(r"\*\*(\w+):\*\* (Yes|No)\.$", q).group(2) for q in paras[i + 1:i + 3]]
    i_rule = at("**Achilles:** Then that's settled: same setting, same answer, whichever way it's done.")
    i_up, i_cross = at("Two collars turn back from the cross to the upright stroke"), at("Two collars go to the cross.")
    i_curl = at("Each turns a collar on past the cross until it stops at the little curl.")
    curl_rounds = [said(i_curl)] + [said(i) for i in range(i_curl + 3, at("Every one wrong!")) if paras[i] == "*[The lever.]*"]
    assert said(i_up) == ["Yes", "Yes"] and said(i_cross) == ["No", "No"]
    assert len(curl_rounds) == 3 and all(a != b for a, b in curl_rounds)        # same setting, never the same answer
    i_held, i_pale = at("Is it the lamp? Or has somebody put something on the line?"), paras.index(b42a.text.join(
        ["\\begin{center}\n", "\n\\end{center}"]))
    i_look = at("the curl beneath turns against the curl above")
    i_oh = at("**Achilles:** ...Oh. Then if you'd read your windows the other way about, we'd have agreed every time.")
    assert i_rule < i_up < i_cross < i_curl < i_held < i_pale < i_look < i_oh
    assert paras[i_pale + 2] == b42.text.join(["\\begin{center}\n", "\n\\end{center}"])
    # the rule agrees with the true key at the stroke and the cross: the Lua refuses it there
    for stop in ("the stroke", "the cross"):
        assert PC.faults(PC.sparkey, stop, False, "expects=same setting") == ["key-rule-stop"]
        assert PC.key_visible(stop)["beneath"] == "same"

    # F21-F24: as he found it (keyed, unmarked), then three turns, each a third of a turn against
    # the sun, its swing arc round the collar from where the stop now set stood to just short of
    # the index (the angle read off the drawn arc, the sense off its travel)
    assert k21.labels == ("yes", "no", "collar", "reflection") and PC.arc_lines(k21) == []
    cy, r = 0.36 * PC.KEYSC, PC.KEYRO * PC.KEYSC + 0.12
    for k, frm, to in ((k22, "the curl", "the cross"), (k23, "the cross", "the stroke"), (k24, "the stroke", "the curl")):
        assert k.before == PC.key_stop(frm) and k.step == to and k.labels == ()
        (line,) = PC.arc_lines(k)
        _, _, pts = PC._mark_line(line)
        ang = [np.degrees(np.arctan2(y - cy, x)) for x, y in pts]
        assert all(np.isclose(np.hypot(x, y - cy), r, atol=1e-3) for x, y in pts)          # round the collar
        mark = PC.MARKOF[PC.key_stop(to)]
        assert np.isclose((ang[0] - 90) % 360, PC.key_visible(frm)["angles"][mark], atol=0.05)   # from where it stood (3 decimals)
        assert np.isclose(ang[-1], 90 - 24, atol=0.05)                                               # to short of the index
        assert all(b > a for a, b in zip(ang, ang[1:]))                                     # counter-clockwise ...
        assert "a third of a turn against the sun" in k.log[2]                              # ... is against the sun
        assert sorted(PC.key_visible(to)["angles"].values()) == [0, 120, 240]
        # the drawn arc: inside the frame, above the lacquer, outside the brass rim
        # (round 5: the collar's rim widened to KEYRO, so the arc stands 0.040 from the frame's edge)
        assert round(PC.mark_clearance(k), 3) == 0.030 and k.log[3].endswith(
            "the least gap to its edge 0.040; clear of the collar's rim by 0.090 (centre lines)")
        assert round(min(p[1] for p in pts) - PC.LACQUER_TOP, 3) == 0.060               # centre line
        assert round(PC.lacquer_clearance(k), 3) == 0.049                                # drawn extent
        assert min(np.hypot(x, y - cy) for x, y in pts) - PC.KEYRO * PC.KEYSC > 0.09
        # with marks=off, the frame is the single key at its new stop (round 3's P4.3, P4.4)
        assert PC.frame_of(PC.twin(PC.frame_calls(b42)[[k22, k23, k24].index(k) + 1])) == \
            PC.frame_of(PC.sparkey(to, False))
    # a full turn backwards, so the sun's way is the other: a turn with the sun brings the
    # stroke, the cross, the curl to the index (the order of their angles, ascending)
    assert [PC.MARKOF[PC.key_stop(s)] for s in ("the curl", "the cross", "the stroke", "the curl")] == \
        ["curl", "cross", "stroke", "curl"]
    assert PC.frame_of(k21) == PC.frame_of(PC.sparkey("the curl", True))

    # P4.1: the lamp's bend (after L3), the rings a little way in from the edges
    assert c41.dump == ("scene=bend rings=green-green pulled=false | cols=7 rows=3 y=0..2 near=false folded=false; "
                        "1:ring green 0 @1.6,0; 2:ring green 0 @4.4,0; cord 1-2 [1.6,0 1.9,1 1.9,2 4.1,2 4.1,1 4.4,0]; "
                        "cord 1-2 [1.6,0 2.5,0 2.5,1 3.5,1 3.5,0 4.4,0]; cord loose-1 [-0.45,0 1.6,0]; "
                        "cord loose-2 [6.45,0 4.4,0]")
    assert c41.labels == ("his", "hers", "bend")
    PC.check_band_words(c41)
    # the glint's narrow turn nested in the mirror's wider one, a row deeper, crossing nothing
    turns = [d["pts"] for d in c41.cloth["cords"] if d["a"] == 1 and d["b"] == 2]
    (mirror,), (glint,) = [t for t in turns if max(y for _, y in t) == 2], [t for t in turns if max(y for _, y in t) == 1]
    cm = lambda pts: [PC.cloth_cm(c41, x, y) for x, y in pts]
    assert PC.crossings(cm(mirror), cm(glint)) == 0
    assert min(x for x, y in mirror if y > 0) < min(x for x, y in glint if y > 0) and \
        max(x for x, y in mirror if y > 0) > max(x for x, y in glint if y > 0)
    # L3: the keyed 'bend' clear of both turns (its box by the Lua's width estimate and a
    # \tiny line's height, 0.2 cm)
    (word, side, gx, gy, x, y, anchor, w), = [o for o in c41.own if o[0] == "bend"]
    assert side is None and anchor == "west"
    _, cords = PC.cloth_drawn(c41)
    box = (x, y - 0.1, x + w, y + 0.1)
    assert min(box_clear(box, pts, reach - 0.0) for kind, a, b, pts, reach in cords) > 0
    # P4.2: pulled straight -- the kit's table is 'green to green' exactly (the model's words
    # before ' | ' name the scene and differ; the cloth after it is the same)
    straight = PC.keycloth("", "green to green")
    assert c42.dump.split(" | ")[1] == straight.dump.split(" | ")[1]
    assert c42.dump.split(" | ")[0] == "scene=bend rings=green-green pulled=true"
    assert c42.frame == c41.frame == 1.78                       # one frame for both states (L3)
    # the fixed parts: only the rings' x changes, outwards, and nothing is added or lost
    xs = lambda pic: [(r["c"], r["x"], r["y"]) for r in pic.cloth["things"]]
    assert xs(c41) == [("green", PC.BAX, 0), ("green", PC.BBX, 0)] and xs(c42) == [("green", PC.AX, 0), ("green", PC.BX, 0)]
    assert PC.AX < PC.BAX and PC.BX > PC.BBX
    assert len(c41.cloth["cords"]) == len(c42.cloth["cords"]) == 4
    assert [(d["a"], d["b"]) for d in c41.cloth["cords"]] == [(d["a"], d["b"]) for d in c42.cloth["cords"]]
    for pic in (c41, c42):                                       # every ring threaded as before
        assert any({1, 2, ("left", 0), ("right", 0)} <= comp for comp in PC.cloth_links(pic.cloth))

    # the tie (structural): the pair read at (r, t) is half the BB84 round fed a fair coin
    # -- the bend pulled straight is a sending ring and a reading ring on the same cords
    for r in (PLAIN, CROSS):
        for t in (PLAIN, CROSS, CIRC):
            for kA in (0, 1):
                for kB in (0, 1):
                    assert np.isclose(pair(kA, kB, r, t), 0.5 * bb84(kB, kA, r, t))

    # P4.3 | P4.4: the key close to -- the cross the same beneath, the curl the other way
    assert k43.dump == "stop crosswise; top cross; above cross same; beneath cross same"
    assert k44.dump == "stop twist; top curl; above curl sunwise; beneath curl against the sun"
    assert k43.labels == k44.labels == ()                       # unkeyed now: the strip keys its first frame
    for k in (k43, k44):                                         # computed from the drawn marks
        v = k.shown
        assert k.dump == f"stop {v['stop']}; top {v['top']}; above {v['top']} {v['above']}; " \
                         f"beneath {v['top']} {v['beneath']}"
    assert PC.key_visible("the stroke")["beneath"] == "same"
    # a turn with the sun (clockwise) brings the stroke, the cross, the curl to the index
    angles = PC.key_visible("the stroke")["angles"]
    assert sorted(angles, key=lambda m: angles[m]) == ["stroke", "cross", "curl"]
    # the ties (structural): the curl's setting is the circular one, whose mirror (the
    # conjugate) is the other circular vector -- c1* is c2 up to a phase -- while the plain
    # and cross vectors are real, so their reflection is the same setting
    c1, c2 = BASIS[STOP[k44.shown["stop"]]]
    assert not np.allclose(c1, c1.conj())
    assert np.isclose(abs(np.vdot(c1.conj(), c2)), 1) and np.isclose(abs(np.vdot(c1.conj(), c1)), 0)
    assert STOP[k43.shown["stop"]] == CROSS
    for s in (PLAIN, CROSS):
        assert all(np.allclose(v, v.conj()) for v in BASIS[s])
    # and so at the curl the pair always disagrees, at the cross always agrees
    assert np.isclose(sum(pair(k, 1 - k, CIRC, CIRC) for k in (0, 1)), 1)
    assert np.isclose(sum(pair(k, k, CROSS, CROSS) for k in (0, 1)), 1)


# --- pictures (hand-built doubled diagrams; pyzx 0.10.6 refuses ground vertices) -----

def attach_reader(g, q, qb, setting, row, col):
    """Chapter 9's read-spider on the doubled line (q, qbar), returns the record's dot.
    Plain: straight in.  Cross: a Hadamard edge on both legs.  Circular: a green -1/2
    (units of pi) dot on q and +1/2 on qbar -- the mirror conjugates, so its phase is
    NEGATED -- then a Hadamard edge each into the read-spider (it undoes the circular
    sender's H-then-quarter-turn)."""
    rd = g.add_vertex(VT.Z, row, col + 1)
    if setting == CIRC:
        pq = g.add_vertex(VT.Z, row - 0.5, col, phase=Fraction(-1, 2))
        pqb = g.add_vertex(VT.Z, row + 0.5, col, phase=Fraction(1, 2))
        g.add_edge((q, pq)); g.add_edge((qb, pqb))
        g.add_edge((pq, rd), ET.HADAMARD); g.add_edge((pqb, rd), ET.HADAMARD)
    else:
        h = ET.HADAMARD if setting == CROSS else ET.SIMPLE
        g.add_edge((q, rd), h); g.add_edge((qb, rd), h)
    return rd


def twin(r, t):
    """The E91 twin, doubled: the cup on (a, b), the mirror cup on (abar, bbar); Alice's
    reader at r on (a, abar), Bob's at t on (b, bbar); outputs kA, kB.
    add_power(-2): the doubled pair is 1/2 cup (x) cup, the cups unnormalized."""
    g = zx.Graph()
    cup, cupb = g.add_vertex(VT.Z, 1, 0), g.add_vertex(VT.Z, 2, 0)
    rdA = attach_reader(g, cup, cupb, r, 0, 1)
    rdB = attach_reader(g, cup, cupb, t, 3, 1)
    kA, kB = g.add_vertex(VT.BOUNDARY, 0, 4), g.add_vertex(VT.BOUNDARY, 3, 4)
    g.add_edge((rdA, kA)); g.add_edge((rdB, kB))
    g.set_inputs(()); g.set_outputs((kA, kB))
    g.scalar.add_power(-2)
    return g


def fair_coin_bb84(r, t):
    """The BB84 round fed a fair coin, with Alice's copy of her bit: ONE green dot holding
    Alice's record leg kA and the doubled pair (q, qbar) -- Hadamard edges at cross; at the
    circular setting a box then a green +1/2 dot on q and -1/2 on qbar (the circular
    SENDER) -- then Bob's reader at t.  Two phase-0 identity dots carry the doubled line
    (PyZX keeps one edge per pair of dots).  add_power(-2): the fair coin's 1/2."""
    g = zx.Graph()
    G = g.add_vertex(VT.Z, 1.5, 0)
    if r == CIRC:
        q = g.add_vertex(VT.Z, 1, 1, phase=Fraction(1, 2))
        qb = g.add_vertex(VT.Z, 2, 1, phase=Fraction(-1, 2))
        g.add_edge((G, q), ET.HADAMARD); g.add_edge((G, qb), ET.HADAMARD)
    else:
        q, qb = g.add_vertex(VT.Z, 1, 1), g.add_vertex(VT.Z, 2, 1)
        h = ET.HADAMARD if r == CROSS else ET.SIMPLE
        g.add_edge((G, q), h); g.add_edge((G, qb), h)
    rdB = attach_reader(g, q, qb, t, 1.5, 2)
    kA, kB = g.add_vertex(VT.BOUNDARY, 0, 4), g.add_vertex(VT.BOUNDARY, 3, 4)
    g.add_edge((G, kA)); g.add_edge((rdB, kB))
    g.set_inputs(()); g.set_outputs((kA, kB))
    g.scalar.add_power(-2)
    return g


def bend(flip=False):
    """1/2 times a green bend on (kA, kB), with a red half-turn on kB if flip:
    1/2 [kA = kB] or 1/2 [kA != kB].  add_power(-2)."""
    g = zx.Graph()
    z = g.add_vertex(VT.Z, 1, 0)
    kA, kB = g.add_vertex(VT.BOUNDARY, 0, 2), g.add_vertex(VT.BOUNDARY, 2, 2)
    g.add_edge((z, kA))
    if flip:
        x = g.add_vertex(VT.X, 2, 1, phase=Fraction(1))
        g.add_edge((z, x)); g.add_edge((x, kB))
    else:
        g.add_edge((z, kB))
    g.set_inputs(()); g.set_outputs((kA, kB))
    g.scalar.add_power(-2)
    return g


def lone_reader(setting, negate=True):
    """A reader alone, (q, qbar) -> y, to compare with the blackboard's
    delta^dagger (M (x) M*).  negate=False puts the SAME phase on both copies (wrong)."""
    g = zx.Graph()
    q, qb = g.add_vertex(VT.BOUNDARY, 0, 0), g.add_vertex(VT.BOUNDARY, 1, 0)
    if setting == CIRC and not negate:
        pq = g.add_vertex(VT.Z, 0, 1, phase=Fraction(-1, 2))
        pqb = g.add_vertex(VT.Z, 1, 1, phase=Fraction(-1, 2))
        rd = g.add_vertex(VT.Z, 0.5, 2)
        g.add_edge((q, pq)); g.add_edge((qb, pqb))
        g.add_edge((pq, rd), ET.HADAMARD); g.add_edge((pqb, rd), ET.HADAMARD)
    else:
        rd = attach_reader(g, q, qb, setting, 0.5, 1)
    y = g.add_vertex(VT.BOUNDARY, 0.5, 4)
    g.add_edge((rd, y))
    g.set_inputs((q, qb)); g.set_outputs((y,))
    return g


def chain_on_cup(top, bottom):
    """A cup with the box chain `top` on its top leg and `bottom` on its bottom leg, each
    listed outward from the bend.  A chain is ((colour, phase), ...): a generic one-wire
    box is z(a) - x(b) - z(c) with three different phases; its transpose is the chain
    reversed."""
    p = Picture()
    b = p.bend(0.5, 1)
    for row, chain in ((0, top), (1, bottom)):
        vs = [(p.z if c == "z" else p.x)(row, 2 + i, ph) for i, (c, ph) in enumerate(chain)]
        p.wire(b, *vs, p.out(row))
    return p.done()


def connected(g, u, v):
    seen, todo = {u}, [u]
    while todo:
        w = todo.pop()
        for n in g.neighbors(w):
            if n not in seen:
                seen.add(n); todo.append(n)
    return v in seen


def sketchbook():
    # the readers, against the blackboard: R = delta^dagger (M (x) M*), M the setting's
    # change of basis (M = H S^dagger at circular: <y| H S^dagger = <c_y| up to a phase)
    M = {PLAIN: I2, CROSS: H, CIRC: H @ S.conj().T}
    for s in (PLAIN, CROSS, CIRC):
        R = DELTA.conj().T @ np.kron(M[s], M[s].conj())
        assert np.allclose(matrix_of(lone_reader(s)), R)
        for y in (0, 1):
            assert np.isclose(abs(np.vdot(BASIS[s][y], M[s].conj().T @ ket(y))), 1)
    # the mirror copy's phase is the NEGATED phase: the same phase on both copies is wrong
    assert not same_picture(lone_reader(CIRC, negate=False), lone_reader(CIRC))
    psi = np.array([0.6, 0.8j])                                # a non-real test state
    probs = np.array([abs(np.vdot(c, psi)) ** 2 for c in (C1, C2)])
    assert np.allclose((matrix_of(lone_reader(CIRC)) @ np.kron(psi, psi.conj())).ravel(), probs)
    assert not np.allclose((matrix_of(lone_reader(CIRC, negate=False)) @ np.kron(psi, psi.conj())).ravel(), probs)

    # (iii) the cup and Alice's reader fuse into her sender: the twin IS the fair-coin
    # BB84 round, for Alice plain or cross and Bob plain, cross or circular
    for r in (PLAIN, CROSS):
        for t in (PLAIN, CROSS, CIRC):
            g = twin(r, t)
            want = np.array([[0.5 * bb84(kB, kA, r, t) for kB in (0, 1)] for kA in (0, 1)])
            assert np.allclose(matrix_of(g).ravel(), want.ravel())
            assert same_picture(g, fair_coin_bb84(r, t))
            h = g.copy(); simp.full_reduce(h)                  # DERIVATION, scalar kept
            assert same_picture(h, g)
            kA, kB = h.outputs()
            if r == t:
                # a bare edge between the two records, and the 1/2: a shared coin
                assert h.num_vertices() == 2 and h.num_edges() == 1 and connected(h, kA, kB)
                assert same_picture(h, bend())
            else:
                assert not connected(h, kA, kB)
    # (iv) circular-circular: the twin is 1/2 x a bend with a red half-turn -- always
    # disagree -- and NOT the BB84 round, which is the bare bend -- always agree
    g = twin(CIRC, CIRC)
    assert same_picture(g, bend(flip=True))
    assert same_picture(fair_coin_bb84(CIRC, CIRC), bend())
    assert not same_picture(g, fair_coin_bb84(CIRC, CIRC))           # REQUIRED negative
    h = g.copy(); simp.full_reduce(h)
    assert same_picture(h, bend(flip=True)) and connected(h, *h.outputs())

    # (i) Eq. (10) as a picture: a generic box slides round the bend as its TRANSPOSE,
    # the chain reversed; U2 U1^T means U1^T nearer the bend
    U1 = (("z", Fraction(1, 4)), ("x", Fraction(1, 3)), ("z", Fraction(3, 5)))
    U2 = (("z", Fraction(1, 5)), ("x", Fraction(2, 3)), ("z", Fraction(-1, 7)))
    one = Picture(); vs = [(one.z if c == "z" else one.x)(0, 1 + i, ph) for i, (c, ph) in enumerate(U1)]
    one.wire(one.inp(0), *vs, one.out(0))
    MU1 = matrix_of(one.done())
    assert not np.allclose(MU1, MU1.T) and not np.allclose(MU1, MU1.conj())   # generic
    lhs = chain_on_cup(U1, U2)
    assert same_picture(lhs, chain_on_cup((), U1[::-1] + U2))                 # slid: reversed
    assert not same_picture(lhs, chain_on_cup((), U1 + U2))                   # not the same box
    assert not same_picture(lhs, chain_on_cup((), U2 + U1[::-1]))             # wrong order


if __name__ == "__main__":
    verdict("BentKey", blackboard, sketchbook)
