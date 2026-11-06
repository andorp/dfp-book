"""Challenge V: A Look with the Lights On  (channels, Stinespring, causality; the master)
========================================================================================
"Because discarding after the works is discarding straight off."  -- Tortoise

Claim: every channel is an isometry V followed by discarding the environment
(Stinespring), Lambda(rho) = tr_E(V rho V^dagger) with V^dagger V = I; and
discarding after the works is discarding straight off (causality).  In the master
diagram "The Unmoved Half" -- the near half sent through a dilated, looked-at
channel, the environment vented and the outcome AVERAGED -- the far half is 1/2 I
for EVERY dilation V; outcome-CONDITIONED, it is correlated, as a Bell pair must be.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re

import numpy as np
import pyzx as zx
import pyzx.simplify as simp
from pyzx.utils import VertexType as VT

from dfp import BELL, CAP, CUP, I2, ket, matrix_of, same_picture, Picture, verdict
from dfp_props import Dialogue, words_only
import props_ch09 as P


def random_unitary(rng, d):
    """A Haar-ish random unitary (QR of a complex Gaussian), generically non-real."""
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def blackboard():
    rng = np.random.default_rng(95)
    assert np.isclose(np.linalg.norm(CUP), np.sqrt(2))       # cups are UNNORMALIZED
    assert np.isclose(np.linalg.norm(BELL) ** 2, 1)          # |Phi+> = cup/sqrt2, so 1/2 = (1/sqrt2)^2

    # a Stinespring dilation of a one-qubit channel: isometry V: A -> B (x) E, built
    # from a 2-qubit unitary U on (A, E) with the environment started in |0>.
    U = random_unitary(rng, 4)
    assert not np.allclose(U, U.conj())                      # non-real
    V = U @ np.kron(I2, ket(0).reshape(2, 1))                # 4x2, A -> B (x) E
    assert np.allclose(V.conj().T @ V, I2)                   # V^dagger V = I (an isometry)

    # Kraus operators read straight off the dilation: K_i = (I_B (x) <e_i|) V
    Ks = [np.kron(I2, ket(i).conj().reshape(1, 2)) @ V for i in (0, 1)]
    assert np.allclose(sum(K.conj().T @ K for K in Ks), I2)  # sum K^dagger K = V^dagger V = I
    assert not np.allclose(Ks[0].conj().T @ Ks[0], I2)       # a genuine (non-unitary) channel

    # the channel Lambda(rho) = sum K_i rho K_i^dagger = tr_E(V rho V^dagger)
    M = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho = M @ M.conj().T; rho /= np.trace(rho)
    lam_kraus = sum(K @ rho @ K.conj().T for K in Ks)
    lam_dilate = np.einsum('aebe->ab', (V @ rho @ V.conj().T).reshape(2, 2, 2, 2))  # tr_E
    assert np.allclose(lam_kraus, lam_dilate)
    # causality: the channel preserves the trace (discard cannot feel it)
    assert np.isclose(np.trace(lam_kraus), np.trace(rho))

    pictures()


# --------------------------------------------------------------------------
# The prop pictures.  The dialogue draws the black slab with the look laid out
# on it -- the far stone under its cloth, the near stone, the needle and the
# flask, each with its reflection, a clean slate off the black -- the needle
# pinned and the flask folded, then the works on the Tortoise's string, and the
# pull,
#   \blackslab[operations]{scene}{caption}                 (pdf/props/ch09.tex)
# and, in the First and Second, the viewer (\crabviewer) and the slate
# (\notchslate).  tools/props_ch09.py transcribes the Lua; the calls are read out
# of dialog.md, never copied.  This file also checks the whole dialogue: the
# nineteen pictures, the keyed rule, the places, and that nothing is pictured
# after the pull -- least of all the Cat.
#
# The state model (identity card, section 3): the works are an isometry V from
# the near stone to near stone (x) needle (x) flask; the far stone keeps the Bell
# pair's top wire; the pin is the read map sum_i |i><ii| (the needle and its
# reflection in, one notch out); a fold is a discard; the pull after every leg is
# folded is causality -- discard after the works = discard straight off.  A
# structural check against this file's blackboard, not a second proof.
# --------------------------------------------------------------------------
DIALOGUE = Dialogue(__file__, macros=P.MACROS)
QUOTE = "Because discarding after the works is discarding straight off."
SIZES = {"First": [1, 2, 1], "Second": [2, 1, 1], "Third": [3], "Fourth": [1, 2, 1],
         "Fifth": [1, 1, 2]}


def check_faults():
    """Every one of the Lua's twenty-four tex.error sites, each raised by a call of
    its own and by nothing else."""
    bad = [
        ("viewer-word", P.crabviewer, "", "restless stone, needle sideways"),
        ("viewer-twice", P.crabviewer, "", "spent stone, spent stone, needle high"),
        ("viewer-restless", P.crabviewer, "", "restless stone, needle low"),
        ("viewer-spent", P.crabviewer, "", "spent stone, needle middle"),
        ("viewer-view", P.crabviewer, "open", "restless stone"),
        ("notch-word", P.notchslate, "", "high, middle"),
        ("notch-none", P.notchslate, "", ""),
        ("notch-many", P.notchslate, "", ", ".join(["high"] * 13)),
        ("slate-view", P.notchslate, "open", "high"),
        ("slate-unread", P.notchslate, "", "high, unread"),
        ("pull-laid", P.blackslab, "pull", "pair of one piece, strung"),
        ("pull-pulled", P.blackslab, "pull, pull", "the look on a string"),
        ("pull-none", P.blackslab, "pull", "the look laid out"),
        ("pin-word", P.blackslab, "pin needle sideways", "the look laid out"),
        ("pin-none", P.blackslab, "pin needle low", "plain pair"),
        ("pin-twice", P.blackslab, "pin needle low, pin needle high", "the look laid out"),
        ("far-stone", P.blackslab, "fold far stone", "the look laid out"),
        ("needle", P.blackslab, "fold needle", "the look laid out"),
        ("tip-fold-word", P.blackslab, "fold works", "the look on a string"),
        ("single", P.blackslab, "fold near stone", "looked-at stone"),
        ("gone", P.blackslab, "fold flask, fold flask", "the look laid out"),
        ("tipped", P.blackslab, "tip flask, tip flask", "the look laid out"),
        ("operation", P.blackslab, "look near stone", "the look laid out"),
        ("scene", P.blackslab, "", "a pair of boots"),
    ]
    for site, fn, *args in bad:
        assert P.faults(fn, *args) == [site], (site, P.faults(fn, *args))
    assert [b[0] for b in bad] == list(P.ERROR_SITES) and len(set(P.ERROR_SITES)) == 24
    # the stand-ins: a bad notch word leaves no notch, so the empty slate is refused too
    assert P.faults(P.notchslate, "", "middle") == ["notch-word", "notch-none"]
    # a leg of the look on a string is already folded: there is nothing there to fold
    assert P.faults(P.blackslab, "fold near stone", "the look on a string") == ["gone"]
    assert P.faults(P.blackslab, "pull", "the look on a string") == []   # every leg folded


def check_whole_dialogue():
    """All nineteen pictures: their number and places, the keyed rule, the
    captions, the frames, no fault anywhere, and nothing after the pull."""
    assert DIALOGUE.check_whole(P.ORDER) == 19                   # 4 + 4 + 3 + 4 + 4
    text = DIALOGUE.read(DIALOGUE.dialog)
    assert len(DIALOGUE.call_re.findall(DIALOGUE.read(DIALOGUE.raw))) == 19   # dialog.raw.md too
    assert not DIALOGUE.call_re.findall(text[:text.index("## First Challenge")])   # none before
    DIALOGUE.keyed_first(P.ORDER, kind=P.keyed_kind)
    placed = [(o, i, c) for o in P.ORDER
              for i, c in enumerate(c for b in DIALOGUE.blocks(o) for c in b)]
    assert [(o, i) for o, i, c in placed if c.keyed] == [
        ("First", 0), ("First", 2), ("Second", 2), ("Third", 0), ("Fourth", 0),
        ("Fifth", 0), ("Fifth", 2)]                               # P1.1 P1.3 P2.3 P3.1 P4.1 P5.1 P5.3
    for o in P.ORDER:
        assert [len(b) for b in DIALOGUE.blocks(o)] == SIZES[o], o
        for b in DIALOGUE.blocks(o):
            assert len({P.height(c) for c in b}) == 1, (o, [c.text for c in b])
    for o, i, c in placed:
        P.picture(c)                                             # the macro never faults
        assert words_only(c.caption, forbidden=P.CAPTION_FORBIDDEN), c.caption
        assert P.place_ok(o, c), (o, c.text)
        assert len(c.args) == 1 and c.args[0]                    # one state/scene argument
    # the pull equality, as the dialogue draws it: P4.4 and P5.4 the same picture,
    # with the same caption, each beside its own before
    (p44,) = DIALOGUE.blocks("Fourth")[-1]
    p54 = DIALOGUE.blocks("Fifth")[-1][-1]
    assert P.picture(p44).shown == P.picture(p54).shown and p44.caption == p54.caption
    # no picture after the pull: the last call of the dialogue is P5.4, and every
    # paragraph from the Escher glance to the Cat's answer comes after its block
    paras = [q.strip() for q in re.split(r"\n[ \t]*\n", text) if q.strip()]
    pics = [i for i, q in enumerate(paras) if DIALOGUE.call_re.search(q)]
    assert p54.text in paras[pics[-1]]
    question = [i for i, q in enumerate(paras) if "Cat — asleep, or awake?" in q]
    answer = [i for i, q in enumerate(paras) if q == "**Cat:** Awake."]
    escher = [i for i, q in enumerate(paras) if "Escher's \\textup{Eye}, a single vast eye with" in q]
    assert len(question) == len(answer) == 1 and answer[0] == question[0] + 1
    assert escher and pics[-1] < min(escher) and pics[-1] < question[0]
    assert all("crabviewer" not in q and "blackslab" not in q and "notchslate" not in q
               for q in paras[pics[-1] + 1:])


def pictures():
    assert f'"{QUOTE}"' in __doc__                               # quoted byte for byte
    check_faults()
    check_whole_dialogue()
    calls = DIALOGUE.check_layout("Fifth", sizes=[1, 1, 2])
    p51, p52, p53, p54 = calls
    for c in calls:
        assert words_only(c.caption, forbidden=P.CAPTION_FORBIDDEN), c.caption
        assert P.place_ok("Fifth", c), c.text
    assert [c.macro for c in calls] == ["blackslab"] * 4
    assert [c.keyed for c in calls] == [True, False, True, False]
    assert [(c.opt, c.args[0]) for c in calls] == [
        ("", "the look laid out"), ("pin needle low, fold flask", "the look laid out"),
        ("", "the look on a string"), ("pull", "the look on a string")]
    b51, b52, b53, b54 = (P.picture(c) for c in calls)
    paras = DIALOGUE.paragraphs("Fifth")
    a1, a2, a3 = DIALOGUE.anchors("Fifth")
    f1, f2, f3 = DIALOGUE.following("Fifth")
    assert QUOTE in DIALOGUE.section("Fifth")                     # the docstring's line

    # the model: a random non-real isometry V, near stone -> near (x) needle (x) flask
    rng = np.random.default_rng(951)
    z = rng.normal(size=(8, 2)) + 1j * rng.normal(size=(8, 2))
    V = np.linalg.qr(z)[0]
    assert np.allclose(V.conj().T @ V, I2) and not np.allclose(V, V.conj())
    after = np.einsum("sm,xm->sx", BELL.reshape(2, 2), V).reshape(2, 2, 2, 2)   # far, near, needle, flask
    R3 = matrix_of(read_spider()).reshape(2, 2, 2)                # the pin: [notch, needle, twin]

    # P5.1: the look laid out, keyed -- cloth, near stone, needle, flask, each with its
    # reflection, the black full length, a clean slate off the black; the laid frame (L2)
    assert "clean slate" in a1 and "Off the black, in front of the slab at Achilles's end" in a1
    assert "the near stone and its twin, the needle and its twin, the flask and its twin" in a1
    cloth, near, needle, flask = b51.shown.things
    assert [t.name for t in b51.shown.things] == ["far stone", "near stone", "needle", "flask"]
    assert cloth.x < near.x < needle.x < flask.x                  # cloth, near, needle, flask
    assert (near.gloss, near.refl) == ("restless", "slack")       # a Bell half (L1)
    assert (needle.mode, needle.refl, flask.mode, flask.refl) == ("standing", "tight", "standing", "tight")
    assert b51.shown.slate == () and b51.shown.frame == P.LAID_FRAME
    assert b51.labels == ("cloth", "near", "needle", "flask", "slate", "black slab")
    # fixed parts held still (L2): the far cloth and the slab's ends as in the Fourth
    b41 = P.picture(DIALOGUE.blocks("Fourth")[0][0])
    for b in (b51, b52, b53, b54):
        assert b.shown.things[0] == b41.shown.things[0] == cloth   # the far cloth at -0.8
        assert b.shown.ends == b41.shown.ends == P.SLAB_ENDS       # the black, full length

    # P5.2: the needle pinned (low: the pointer lies left, toward the stop it read), a
    # notch cut at the slate's foot -- exactly one more than P5.1's; the flask folded
    pin = [q for q in paras if "pins the needle down onto its own reflection" in q]
    assert len(pin) == 1 and paras.index(a1) < paras.index(pin[0]) < paras.index(a2)
    assert "low, as it happens; she scratches it at the slate's foot" in pin[0]
    assert "Two pointers in; one settled mark out" in pin[0]
    assert "breathed into" in a2 and "gone from the black" in a2
    assert [t.name for t in b52.shown.things] == ["far stone", "near stone", "needle"]
    pinned = b52.shown.things[2]
    assert (pinned.mode, pinned.refl, pinned.lie) == ("pinned", None, "left")   # one pointer, no twin
    assert pinned.lie == P.LIE["low"] == P.STOP["low"] and P.EDGE["low"] == "foot"
    assert b52.shown.slate == b51.shown.slate + ("low",)          # exactly one notch more
    assert b52.shown.things[1] == near and b52.shown.frame == b51.shown.frame
    assert f2.startswith("**Achilles:** So pinning is reading")
    # the pin <-> the read map: the needle and its reflection in, one notch out, with
    # the Born probabilities of the needle's answer
    assert matrix_of(read_spider()).shape == (2, 4)               # two legs in, one out
    needle_rho = np.einsum("sape,saqe->pq", after, after.conj())
    born = np.einsum("cpq,pq->c", R3, needle_rho)
    probs = [np.einsum("sae,sae->", after[:, :, i, :], after[:, :, i, :].conj()) for i in (0, 1)]
    assert np.allclose(born, probs) and np.isclose(born.sum(), 1)
    assert np.allclose(born.imag, 0) and np.all(born.real > 0)    # either notch possible

    # P5.3: the look on a string, keyed -- the works on the black, every leg folded
    assert "exactly as for the clean fold" in a3
    assert ("every leg it comes out on — the near stone's, the flask's and the needle's, "
            "its answer already cut on the slate — is folded into its reflection") in a3
    assert [t.kind for t in b53.shown.things] == ["cloth", "works"] and b53.shown.string == "works"
    assert b53.labels == ("cloth", "works", "black slab") and b53.shown.frame == P.TALL_FRAME
    assert b53.shown.slate is None                                # the notch is not in its frame
    # the needle's folded leg <-> the outcome averaged: the far half is 1/2 I; the
    # outcome kept (conditioned), it is not
    far_avg = np.einsum("sape,Saqe,cpq->sS", after, after.conj(), R3)
    assert np.allclose(far_avg, I2 / 2)
    cond = np.einsum("sae,Sae->sS", after[:, :, 0, :], after[:, :, 0, :].conj())
    assert not np.allclose(cond / np.trace(cond), I2 / 2)
    assert "chimney" in f3                                        # the smoke, after the block

    # P5.4: pulled -- the same picture as the Fourth's pull (causality: discard after
    # the works = discard straight off), beside the blackboard's trace preservation
    strung = P.apply(P.apply(P.SCENES["pair of one piece, strung"], "fold near stone"), "pull")
    works = P.apply(P.SCENES["the look on a string"], "pull")
    assert P.visible(strung) == P.visible(works) == b54.shown
    assert b54.shown.things == (cloth,) and b54.shown.string == "pulled"
    M = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho = M @ M.conj().T
    rho /= np.trace(rho)
    assert np.isclose(np.trace(V @ rho @ V.conj().T), np.trace(rho))   # tr(V rho V^dag) = tr rho
    assert np.allclose(far_avg, np.einsum("abcb->ac", np.outer(BELL, BELL.conj()).reshape(2, 2, 2, 2)))
    # the SCENES are values: reading them again sees nothing folded or pulled
    assert P.visible(P.SCENES["pair of one piece, strung"]).string == "laid"
    last = max(i for i, q in enumerate(paras) if DIALOGUE.call_re.search(q))
    assert p54.text in paras[last] and paras[last + 1] == f3      # the section's last call
    assert paras.index(f3) < min(i for i, q in enumerate(paras) if "asleep, or awake" in q)


def read_spider():
    """The read-spider Z(2->1) = sum_i |i><ii|, one classical wire out."""
    g = zx.Graph()
    i1 = g.add_vertex(VT.BOUNDARY, 0, 0); i2 = g.add_vertex(VT.BOUNDARY, 1, 0)
    z = g.add_vertex(VT.Z, 0.5, 1); o = g.add_vertex(VT.BOUNDARY, 0.5, 2)
    g.add_edge((i1, z)); g.add_edge((i2, z)); g.add_edge((z, o))
    g.set_inputs((i1, i2)); g.set_outputs((o,))
    return g


def value_build():
    """The master 'The Unmoved Half' as a hand-built CPM (doubled) diagram, evaluated
    for a concrete NON-REAL dilation V = U (x) Ubar.  The near half M comes from the
    Bell cup; the pointer P and environment E start in |0>; U couples (M,P,E); then M
    and E are discarded (caps), P is READ (the read-spider), and the outcome c is
    AVERAGED (forgotten).  Returns the far half's marginal and the outcome-conditioned
    pieces."""
    rng = np.random.default_rng(950)
    U = random_unitary(rng, 8)                               # the works: U on (M, P, E)
    assert not np.allclose(U, U.conj())
    Ut = U.reshape(2, 2, 2, 2, 2, 2)                         # [m', p', e', m, p, e]

    # Bell fixes m = s (the far half's index); pointer/environment start in |0> (p=e=0)
    Kket = np.moveaxis(Ut[:, :, :, :, 0, 0], 3, 0)          # [s, m', p', e']  (ket copy: U)
    Kcon = np.moveaxis(U.conj().reshape(2, 2, 2, 2, 2, 2)[:, :, :, :, 0, 0], 3, 0)  # Ubar
    R3 = matrix_of(read_spider()).reshape(2, 2, 2)          # [c, p', pbar']  (a picture primitive)

    # Result 9.5: reading, then forgetting c, is a plain cap on (P, Pbar)
    assert np.allclose(np.einsum('cpq->pq', R3), CAP.reshape(2, 2))

    # discard M' (a shared m'), discard E' (a shared e'), read P' & forget c (sum over c);
    # scalar 1/2 = the doubled Bell normalization, = (1/sqrt2)^2 (cup = |00>+|11>)
    far = 0.5 * np.einsum('sape,Saqe,cpq->sS', Kket, Kcon, R3)
    cond0 = 0.5 * np.einsum('sape,Saqe,pq->sS', Kket, Kcon, R3[0])   # outcome c = 0
    cond1 = 0.5 * np.einsum('sape,Saqe,pq->sS', Kket, Kcon, R3[1])   # outcome c = 1
    # NEGATIVE: without the CONJUGATE mirror copy the causality yank fails
    far_noconj = 0.5 * np.einsum('sape,Saqe,cpq->sS', Kket, Kket, R3)
    return far, cond0, cond1, far_noconj


def master_skeleton(with_box=False):
    """The master in unfolded doubled form, drawn as an ordinary graph so its nine LINES
    are nine edges.  Outputs: S, c (classical), Sbar.  With with_box=True, an explicit
    phase-0 Z spider (an identity box) sits on each of the six lines through V -- the
    dilation V = doubled identity, drawn out -- and the six ids are returned so the
    identity rule can be applied to exactly them."""
    g = zx.Graph()
    outS = g.add_vertex(VT.BOUNDARY, 0, 6)
    outSb = g.add_vertex(VT.BOUNDARY, 7, 6)
    outC = g.add_vertex(VT.BOUNDARY, 3.5, 6)              # the single CLASSICAL wire
    cupSM = g.add_vertex(VT.Z, 0.5, 1); cupSbMb = g.add_vertex(VT.Z, 6.5, 1)
    capM = g.add_vertex(VT.Z, 1.5, 4); capE = g.add_vertex(VT.Z, 5.5, 4)
    read = g.add_vertex(VT.Z, 3.5, 4)
    p0 = g.add_vertex(VT.X, 3, 1); p0b = g.add_vertex(VT.X, 4, 1)
    e0 = g.add_vertex(VT.X, 5, 1); e0b = g.add_vertex(VT.X, 6, 1)
    E = g.add_edge
    box_ids = []
    if with_box:
        # V = doubled identity: a phase-0 Z (identity box) on M, Mbar, P, Pbar, E, Ebar
        m, mb = g.add_vertex(VT.Z, 1, 2.5), g.add_vertex(VT.Z, 2, 2.5)
        p, pb = g.add_vertex(VT.Z, 3, 2.5), g.add_vertex(VT.Z, 4, 2.5)
        e, eb = g.add_vertex(VT.Z, 5, 2.5), g.add_vertex(VT.Z, 6, 2.5)
        box_ids = [m, mb, p, pb, e, eb]
        E((cupSM, outS)); E((cupSM, m)); E((m, capM)); E((cupSbMb, mb)); E((mb, capM)); E((cupSbMb, outSb))
        E((p0, p)); E((p, read)); E((p0b, pb)); E((pb, read)); E((read, outC))
        E((e0, e)); E((e, capE)); E((e0b, eb)); E((eb, capE))
    else:
        E((cupSM, outS))        # 1  S      : far half, ket copy
        E((cupSM, capM))        # 2  M      : near half, ket copy -> discard
        E((cupSbMb, capM))      # 3  Mbar   : near half, mirror copy -> discard
        E((cupSbMb, outSb))     # 4  Sbar   : far half, mirror copy
        E((p0, read))           # 5  P      : pointer, ket copy -> read
        E((p0b, read))          # 6  Pbar   : pointer, mirror copy -> read
        E((read, outC))         # 7  c      : the classical outcome  <-- the one classical wire
        E((e0, capE))           # 8  E      : environment, ket copy -> discard
        E((e0b, capE))          # 9  Ebar   : environment, mirror copy -> discard
    g.set_inputs(()); g.set_outputs((outS, outC, outSb))
    return (g, box_ids) if with_box else g


def remove_identity(g, v):
    """The identity rule of Chapter 8: a degree-2, phase-0 Z spider is a bare wire.
    (pyzx 0.10.6's id_simp takes no vertex filter and would also swallow the cups and
    caps, so we apply the id rule to exactly the box spiders by splicing them out.)"""
    assert g.type(v) == VT.Z and g.phase(v) == 0 and len(list(g.neighbors(v))) == 2
    a, b = g.neighbors(v)
    g.remove_vertex(v)
    g.add_edge((a, b))


def master_forgotten():
    """The skeleton with the outcome c FORGOTTEN in-picture: the read-spider's classical
    wire ends in the 'forget' effect <0|+<1| = a phase-free Z(1->0), instead of the open
    output c.  A state on (S, Sbar).  The scalar (sqrt2)^-6 = (1/2)^3 collects the four
    |0> apparatus/environment factors (each carries sqrt2) and the doubled Bell 1/2."""
    g = zx.Graph()
    outS = g.add_vertex(VT.BOUNDARY, 0, 6); outSb = g.add_vertex(VT.BOUNDARY, 7, 6)
    cupSM = g.add_vertex(VT.Z, 0.5, 1); cupSbMb = g.add_vertex(VT.Z, 6.5, 1)
    capM = g.add_vertex(VT.Z, 1.5, 4); capE = g.add_vertex(VT.Z, 5.5, 4)
    read = g.add_vertex(VT.Z, 3.5, 4); forget = g.add_vertex(VT.Z, 3.5, 5)   # read, then forget c
    p0 = g.add_vertex(VT.X, 3, 1); p0b = g.add_vertex(VT.X, 4, 1)
    e0 = g.add_vertex(VT.X, 5, 1); e0b = g.add_vertex(VT.X, 6, 1)
    E = g.add_edge
    E((cupSM, outS)); E((cupSM, capM)); E((cupSbMb, capM)); E((cupSbMb, outSb))
    E((p0, read)); E((p0b, read)); E((read, forget))
    E((e0, capE)); E((e0b, capE))
    g.set_inputs(()); g.set_outputs((outS, outSb))
    g.scalar.add_power(-6)
    return g


def half_cup():
    """1/2 cup(S, Sbar) = the far half cupped to its own mirror = 1/2 I."""
    p = Picture(); p.wire(p.out(0), p.bend(1.5, 1), p.out(3))
    g = p.done(); g.scalar.add_power(-2)
    return g


def sketchbook():
    # --- (a) VALUE BUILD ---  averaged over the outcome, the far half is 1/2 I for a
    # concrete non-real dilation; conditioned on an outcome, it is correlated.
    far, cond0, cond1, far_noconj = value_build()
    assert np.allclose(far, I2 / 2)                          # unmoved half, EVERY V, exact
    assert np.allclose(cond0 + cond1, I2 / 2)                # the pieces average back to 1/2 I
    assert not np.allclose(cond0, I2 / 2)                    # ... but each piece is correlated
    assert np.isclose((np.trace(cond0) + np.trace(cond1)).real, 1)   # P(c=0)+P(c=1) = 1
    assert not np.allclose(far_noconj, I2 / 2)               # conjugate copy is load-bearing

    # Result 9.5 in the picture: read-spider then forget its classical wire FUSES to a cap
    def read_then_forget():
        g = zx.Graph()
        i1 = g.add_vertex(VT.BOUNDARY, 0, 0); i2 = g.add_vertex(VT.BOUNDARY, 1, 0)
        read = g.add_vertex(VT.Z, 0.5, 1); forget = g.add_vertex(VT.Z, 0.5, 2)   # Z(2->1), Z(1->0)
        g.add_edge((i1, read)); g.add_edge((i2, read)); g.add_edge((read, forget))
        g.set_inputs((i1, i2)); g.set_outputs(())
        return g
    cap = Picture(); cb = cap.bend(0.5, 1); cap.wire(cap.inp(0), cb, cap.inp(1))
    cap = cap.done()
    rf = read_then_forget()
    assert same_picture(rf, cap)                             # read then forget = cap
    assert simp.spider_simp(rf) and same_picture(rf, cap)   # the fusion rule derives it

    # --- (b) SKELETON BUILD ---  the wire count, checked in code: nine lines = nine edges
    s = master_skeleton()
    assert s.num_edges() == 9                                # eight doubled + one classical
    assert len(s.outputs()) == 3                             # S, c, Sbar
    # the single classical wire is the read-spider's output edge (line 7, read -> outC)

    # V = doubled identity, drawn out (six identity boxes, fifteen edges) and then id'd:
    # the identity rule PRODUCES the nine-wire count (as Octet produced eight via bialgebra)
    boxed, box_ids = master_skeleton(with_box=True)
    assert boxed.num_edges() == 15
    for v in box_ids:
        remove_identity(boxed, v)                           # apply the id rule to the six boxes
    assert boxed.num_edges() == 9 and same_picture(boxed, s)

    # with the outcome forgotten, the whole master yanks to 1/2 I -- exactly the picture
    # of Result 9.6; and the rewrite rules derive the yank (causality: discard after the
    # works = discard straight off)
    mf = master_forgotten()
    assert mf.num_edges() == 9                               # same wiring: forget replaces line 7
    assert np.allclose(matrix_of(mf).reshape(2, 2), I2 / 2)
    assert same_picture(mf, half_cup())
    g = master_forgotten(); simp.full_reduce(g)
    assert same_picture(g, half_cup())


if __name__ == "__main__":
    verdict("Stinespring", blackboard, sketchbook)
