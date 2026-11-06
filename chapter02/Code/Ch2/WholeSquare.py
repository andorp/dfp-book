"""Challenge 6: The Whole Square
==================================
"None over, none short. ...Oh. A string tagged at both ends is a number, and
the number may be backwards. The number with its twin from the dark string is a
square of paper, and that never is."  -- Achilles

Claim: a state meets its own mirror image in the number 1 -- and therefore the
chances it gives make up one whole for every question there is.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import math
import re
from fractions import Fraction

import numpy as np
import pyzx as zx
from dfp import (Picture, ket, matrix_of, random_matrix, random_state,
                 same_picture, verdict)

PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)
T = np.array([3 / 5, 4 / 5], dtype=complex)
CHI = np.array([3 / 5, 4j / 5], dtype=complex)      # blackboard only
PHASES = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))


def born(state, b):
    return (np.vdot(state, b) * np.vdot(b, state)).real


def blackboard():
    rng = np.random.default_rng(26)
    psi = random_state(rng)
    assert not np.allclose(psi.imag, 0) and not np.allclose(psi, psi.conj())

    # the theorem: the conjugated row meets its column in one
    assert np.isclose(np.vdot(psi, psi), 1)
    # the plain row does not: the unconjugated mirror fails
    assert not np.isclose(psi @ psi, 1)
    assert np.isclose(np.vdot(CHI, CHI), 1) and np.isclose(CHI @ CHI, -7 / 25)
    assert np.isclose(np.vdot(T, T), 1)

    # the arrow asked about itself: five in five, and the square on it is
    # made of the nine and the sixteen
    assert 9 + 16 == 25
    assert np.isclose(born(T, ket(0)) + born(T, ket(1)), 1)
    # the dark string repeats the pale string's number, sign included, because
    # T is real -- here for the answer that reaches backwards
    assert np.isclose(np.vdot(MINUS, T), np.vdot(T, MINUS))
    assert np.isclose(np.vdot(MINUS, T), -1 / (5 * np.sqrt(2)))
    # for a state with a non-real entry the two halves are conjugates, not equal
    assert np.isclose(np.vdot(PLUS, CHI), np.conj(np.vdot(CHI, PLUS)))
    assert not np.isclose(np.vdot(PLUS, CHI), np.vdot(CHI, PLUS))

    # the corollary: the totals make one whole for every question.
    # the standard basis, the turned basis, and a basis drawn at random
    random_q, _ = np.linalg.qr(random_matrix(rng))
    assert np.allclose(random_q.conj().T @ random_q, np.eye(2))
    assert not np.allclose(random_q.imag, 0)                   # a genuinely complex basis
    for basis in ([ket(0), ket(1)], [PLUS, MINUS],
                  [random_q[:, 0], random_q[:, 1]]):
        gram = np.array([[np.vdot(bi, bj) for bj in basis] for bi in basis])
        assert np.allclose(gram, np.eye(2))
        for state in (psi, T, CHI):
            total = sum(born(state, b) for b in basis)
            assert np.isclose(total, np.vdot(state, state).real)
            assert np.isclose(total, 1)
    # and the expansion the corollary rests on
    for basis in ([PLUS, MINUS], [random_q[:, 0], random_q[:, 1]]):
        assert np.allclose(sum(np.vdot(b, psi) * b for b in basis), psi)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws its props with five macros
# (pdf/props/ch02.tex; the rules are in pdf/props/ch02.lua):
#   \echostrings[strip]{pale}{dark}{caption}   the two strings and their tags
#   \crosssheet{arrow}{position}{read}{caption} the pale string's reading, close up
#   \tailontail{start}{closing}{caption}       the dark string's reading
#   \sharestrip{share}{caption}                a strip, k parts in n shaded
#   \fivesquare{laid}{caption}                 the five-by-five outline
# The Lua COMPUTES what a picture shows -- the dark string's tags under "echo",
# where a tag hangs and which corner is knotted, every reach with its sign,
# where the card's foot falls, how many matches and snapped pieces, where the
# little squares lie, how many parts of a strip are shaded, how the outline
# fills -- so the dialogue never writes a reading by hand.  The functions
# below, down to fivesquare(), are a transcription of ch02.lua (all but the
# drawing), so what is asserted here is what the printed page shows.  The
# calls are read out of this challenge's section of dialog.md, never copied,
# so an edit to the dialogue that breaks a picture breaks here.
# --------------------------------------------------------------------------
DIALOG = pathlib.Path(__file__).resolve().parents[2] / "dialog.md"
RAW = DIALOG.with_name("dialog.raw.md")
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")
MACROS = ("echostrings", "crosssheet", "tailontail", "sharestrip", "fivesquare")
# the regex of identity.md, section 3 "Props and their pictures"
CALL = (r"\\(echostrings|crosssheet|tailontail|sharestrip|fivesquare)(\*?)"
        r"(?:\[([^\]]*)\])?((?:\{[^{}]*\})+)")
ARGS = {"echostrings": ("pale", "dark"), "crosssheet": ("arrow", "position", "read"),
        "tailontail": ("start", "closing"), "sharestrip": ("share",), "fivesquare": ("laid",)}
MARKUP = set(MACROS) | {"quad", "begin", "end", "textup"}
# the keyed pictures of identity.md section 3: (challenge, place in it)
KEYED = [("First", 0), ("First", 1), ("Second", 1), ("Second", 3), ("Fourth", 0), ("Sixth", 2)]


def read(path):
    """The file, its planning comments left out (they are notes, not pictures)."""
    return re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)


def section(path, ordinal):
    """The '## <ordinal> Challenge' section, up to the next heading."""
    text = read(path)
    heads = list(re.finditer(r"^## " + ordinal + r" Challenge\b.*$", text, re.M))
    assert len(heads) == 1, (path.name, ordinal)
    rest = text[heads[0].end():]
    after = re.search(r"^## ", rest, re.M)
    return rest[:after.start()] if after else rest


def picture(call):
    """One call, parsed: the macro, keyed or not, its arguments and caption."""
    m = re.fullmatch(CALL, call)
    assert m, call                                        # nothing malformed
    macro, star, strip, group = m.groups()
    parts = re.findall(r"\{([^{}]*)\}", group)
    assert len(parts) == len(ARGS[macro]) + 1, call
    assert strip is None or macro == "echostrings", call  # only the strings take a strip
    p = dict(zip(ARGS[macro], parts[:-1]))
    p.update(macro=macro, keyed=star == "*", strip=strip, caption=parts[-1])
    return p


def blocks(path, ordinal):
    """The picture blocks of a section, in order, each the list of its calls."""
    text = section(path, ordinal)
    found = []
    for para in re.split(r"\n[ \t]*\n", text):
        if not any("\\" + m in para for m in MACROS):
            continue
        # a raw centre block that is a paragraph of its own, one line of calls
        # joined by \quad -- so never inside a *[...]* direction
        m = re.fullmatch(r"\\begin\{center\}\n([^\n]*)\n\\end\{center\}", para.strip())
        assert m, para
        found.append([c.strip() for c in re.split(r"\\quad", m.group(1))])
    calls = [c for b in found for c in b]
    assert sum(text.count("\\" + m) for m in MACROS) == len(calls)
    return found


def words_only(caption):
    """Words (hyphened or with an apostrophe), commas between them: no digit,
    brace, symbol or lone letter, and never the strings counted or named in
    the plural."""
    return (re.fullmatch(r"[A-Za-z'-]+(?:,? [A-Za-z'-]+)*", caption) is not None
            and all(len(w) > 1 or w == "a" for w in re.findall(r"[A-Za-z'-]+", caption))
            and not re.search(r"\bstrings\b|\bboth\b|\bpair\b", caption))


def check_layout(ordinal, sizes):
    """The section's blocks: their sizes, markup and captions, and the very same
    calls in dialog.raw.md.  Returns the parsed pictures, in order."""
    found = blocks(DIALOG, ordinal)
    assert [len(b) for b in found] == sizes, (ordinal, [len(b) for b in found])
    assert blocks(RAW, ordinal) == found, ordinal         # identical in the raw dialogue
    assert set(re.findall(r"\\([A-Za-z]+)", section(DIALOG, ordinal))) <= MARKUP
    for b in found:
        assert len(b) <= 4
        assert sum(picture(c)["macro"] == "echostrings" for c in b) <= 1, b   # two strings, never more
    pics = [picture(c) for b in found for c in b]
    for p in pics:
        assert words_only(p["caption"]), p["caption"]
        for a in ARGS[p["macro"]]:
            assert not re.search(r"[0-9]", p[a]), p       # plain words, never digits
        assert p["strip"] is None, p                      # the strip is told, not drawn
    return pics


def kind(p):
    """What a keyed picture labels: each macro, and the square and the turned
    cross-sheet each with their own line names.  None: nothing to label."""
    if p["macro"] != "crosssheet":
        return p["macro"]
    pos = norm(p["position"])
    return "crosssheet " + pos if pos in ("square", "turned") else None


def check_whole_dialogue():
    """The six challenge sections, in order; every picture of the dialogue in
    one of them; and the keyed rule: a picture is starred exactly when it is
    the first of its prop, or of its labelling -- P1.1, P1.2, P2.2, P2.4, P4.1
    and P6.3 -- and nowhere else."""
    text = read(DIALOG)
    assert tuple(re.findall(r"^## (\w+) Challenge\b", text, re.M)) == ORDINALS
    everything = re.findall(CALL, text)
    assert sum(text.count("\\" + m) for m in MACROS) == len(everything) == 20
    placed = [(o, i, picture(c)) for o in ORDINALS
              for i, c in enumerate(c for b in blocks(DIALOG, o) for c in b)]
    assert len(placed) == 20 and sum(len(blocks(DIALOG, o)) for o in ORDINALS) == 19
    seen, starred = set(), []
    for o, i, p in placed:
        k = kind(p)
        first = k is not None and k not in seen
        seen.add(k)
        assert p["keyed"] == first, (o, i, p)
        if p["keyed"]:
            starred.append((o, i))
    assert starred == KEYED


# --- ch02.lua, transcribed ---------------------------------------------------
# The macro's tex.error (an unknown word, a line read under the wrong position,
# squares on a reach that is not whole, ...) records the fault and carries on
# with a stand-in, as the Lua does; the build then stops.  Here the faults are
# recorded in TEX_ERRORS, and every picture of the dialogue must leave it empty.
TEX_ERRORS = []


def tex_error(msg):
    TEX_ERRORS.append(msg)


def evaluate(macro, *args):
    """Run one transcribed macro; return what it computes and the faults it raised."""
    TEX_ERRORS.clear()
    out = macro(*args)
    return out, list(TEX_ERRORS)


def clean(macro, *args):
    """A picture as the dialogue draws it: it must raise no fault at all."""
    out, errors = evaluate(macro, *args)
    assert errors == [], errors
    return out

WORDS = {"zero": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
         "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22,
         "twenty-three": 23, "twenty-four": 24, "twenty-five": 25}
BASE = {"three-and-four": (3, 4, "nl"), "heads-ward": (5, 0, "nl"), "tails-ward": (0, 5, "nl"),
        "slant": (1, 1, "nl"), "back-slant": (1, -1, "fl")}
OPP = {"nl": "fr", "fr": "nl", "fl": "nr", "nr": "fl"}


def norm(s):
    return " ".join(s.split())


def number(w):
    w = norm(w)
    if w in WORDS:
        return WORDS[w]
    return int(w) if re.fullmatch(r"-?[0-9]+", w) else None


def parse_arrow(spec):
    """An arrow word: its direction and length in matches, and the corner of
    its square that holds the tail.  End for end (and 'reversed', which is the
    three-and-four arrow end for end) negates the direction and moves the tail
    to the opposite corner."""
    s, turned = norm(spec), False
    if s == "reversed":
        s, turned = "three-and-four", True
    while s.endswith(" end-for-end"):
        s, turned = s[:-len(" end-for-end")], not turned
    if s not in BASE:
        tex_error("unknown arrow " + spec)
        s = "three-and-four"
    x, y, corner = BASE[s]
    if turned:
        x, y, corner = -x, -y, OPP[corner]
    L = math.hypot(x, y)
    return dict(name=s, v=np.array([x, y], dtype=float), corner=corner, L=L,
                u=np.array([x, y], dtype=float) / L)


def pair_length(a, b):
    """Both arrows of a pair are drawn as long as each other."""
    names = {x["name"] for x in (a, b) if x is not None}
    if "three-and-four" in names:
        return 5.0
    if names & {"slant", "back-slant"}:
        return math.sqrt(2)
    return 5.0


def tail_on_square(corner, S, m):
    return {"nl": (m, m), "fl": (m, S - m), "fr": (S - m, S - m)}.get(corner, (S - m, m))


def parse_pair(spec):
    s = norm(spec)
    if s in ("", "bare"):
        return None, None
    m = re.fullmatch(r"([^/]*)/([^/]*)", s)
    a, b = (m.group(1), m.group(2)) if m else (s, "")
    a, b = norm(a), norm(b)
    return (parse_arrow(a) if a else None), (parse_arrow(b) if b else None)


def parse_share(spec):
    s = norm(spec)
    if s == "all dark":
        return 2, 2
    if s == "all white":
        return 0, 2
    m = re.fullmatch(r"(.*?) in (.*)", s)
    k, n = (number(m.group(1)), number(m.group(2))) if m else (None, None)
    if m and m.group(1) == "all" and n is not None:
        k = n
    if k is None or n is None or not (1 <= n <= 25 and 0 <= k <= n):
        tex_error("unknown share " + spec)
        return 1, 2
    return k, n


def string_tag(kx, role, S, a, Lp):
    """A tag on a string: its square (four corners, near left first), the
    knot, and its inked arrow (tail and head).  A start tag is knotted by its
    near left-hand corner and stands off beyond the string (y > 0); a closing
    tag is the same square as it lies, knotted by its far right-hand corner,
    hanging on the near side (y < 0)."""
    x0, y0 = (kx, 0.012) if role == "start" else (kx - S, -S - 0.012)
    m = 0.09 * S
    tx, ty = tail_on_square(a["corner"], S, m)
    Ld = S * (0.40 + 0.09 * Lp)
    tail = np.array([x0 + tx, y0 + ty])
    return dict(role=role, arrow=a, knot=np.array([kx, 0.0]), drawn=Ld, tail=tail,
                head=tail + Ld * a["u"],
                square=np.array([[x0, y0], [x0 + S, y0], [x0 + S, y0 + S], [x0, y0 + S]]))


def echostrings(stripspec, palespec, darkspec, keyed):
    ps, pc = parse_pair(palespec)
    if norm(darkspec) == "echo":
        ds, dc = pc, ps
        if ps is None or pc is None:
            tex_error("the dark string can echo only a pale string tagged at both ends")
    else:
        ds, dc = parse_pair(darkspec)
    Ls, G, S, kin = 1.03, 0.26, 0.46, 0.07
    p0, p1, d0, d1 = 0.0, Ls, Ls + G, 2 * Ls + G
    Lpale, Ldark = pair_length(ps, pc), pair_length(ds, dc)
    tags = {}
    for name, a, kx, role, Lp in (("pale start", ps, p0 + kin, "start", Lpale),
                                  ("pale closing", pc, p1 - kin, "closing", Lpale),
                                  ("dark start", ds, d0 + kin, "start", Ldark),
                                  ("dark closing", dc, d1 - kin, "closing", Ldark)):
        if a is not None:
            tags[name] = string_tag(kx, role, S, a, Lp)
    strip = None
    if norm(stripspec or ""):
        if ps is None:
            tex_error("a strip is tied on beside the start tag; the pale string has none")
        strip = parse_share(stripspec)
    labels = set()
    if keyed:
        labels |= {"start tag"} if ps else set()
        labels |= {"closing tag"} if pc else set()
        labels |= {"pale", "dark"}
    return dict(tags=tags, lengths=(Lpale, Ldark), strings=((p0, p1), (d0, d1)),
                strip=strip, labels=labels)


def close_len(L):
    return 5.0 if L > 2 else 3.8


def match_row(r):
    """Matches laid end to end from the crossing, r matches (sign: which way):
    each (from, to, snapped) in matches along the line; a remainder is a
    snapped piece with no head."""
    sgn = -1 if r < 0 else 1
    a = abs(r)
    whole = math.floor(a + 1e-6)
    row = [(sgn * i, sgn * (i + 1), False) for i in range(whole)]
    if a - whole > 0.04:
        row.append((sgn * whole, sgn * a, True))
    return row


POSITIONS = {"square": 0.0, "turned": -45.0, "own": 20.0}
LINE = {"square": {"heads-ward": "A", "tails-ward": "B"},
        "turned": {"back-slant": "A", "slant": "B"},
        "along": {"along": "A", "across": "B"}, "own": {}}
LINE_NAMES = {"square": {"A": "heads-ward", "B": "tails-ward"},
              "turned": {"A": "back-slant", "B": "slant"}}


def cross(e, v):
    return e[0] * v[1] - e[1] * v[0]


def crosssheet(arrowspec, posspec, readspec, keyed):
    """The pale string's reading: everything in matches, the crossing on the
    arrow's tail at the origin."""
    a = parse_arrow(arrowspec)
    pos = norm(posspec)
    if pos == "along":
        th = math.degrees(math.atan2(a["u"][1], a["u"][0]))
    else:
        if pos not in POSITIONS:
            tex_error("unknown position " + posspec)
            pos = "square"
        th = POSITIONS[pos]
    read, order = set(), []
    for w in readspec.split(","):
        w = norm(w)
        if w in ("", "none"):
            continue
        if w == "both":
            read |= {"A", "B"}
            order.append("B")
        elif w in LINE[pos]:
            read.add(LINE[pos][w])
            order.append(LINE[pos][w])
        elif w in ("arrow", "squares", "halves", "both-ways", "no-matches"):
            read.add(w)
        else:
            tex_error("'%s' is not read under the %s sheet" % (w, pos))
    if "squares" in read and "halves" in read:
        tex_error("the halves are the two little squares cut up; squares or halves, not both")
    AL = close_len(a["L"])
    mu = AL / a["L"]                                   # drawn units per match
    t = math.radians(th)
    e = {"A": np.array([math.cos(t), math.sin(t)]), "B": np.array([-math.sin(t), math.cos(t)])}
    tip = AL * a["u"]
    reach = {l: float(a["v"] @ e[l]) for l in "AB"}   # in matches, signed

    def side(l):                                       # which side of the line the tip is
        c = cross(e[l], tip)
        return 1 if abs(c) < 1e-6 or c > 0 else -1

    out = dict(arrow=a, position=pos, theta=th, e=e, reach=reach, read=read, order=order,
               squares={}, halves=None, rows={}, back_rows={}, arrow_row=None, card=None,
               labels=set())
    for l in "AB":
        if "squares" in read and l in read:
            r = reach[l]
            if abs(r - math.floor(r + 0.5)) > 1e-6:
                tex_error("squares only on a whole reach of matches")
                continue
            n = math.floor(abs(r) + 0.5)
            sg = -1 if r < 0 else 1
            away = -side(l) * np.array([-e[l][1], e[l][0]])     # the side away from the tip
            out["squares"][l] = [sg * (i + 0.5) * e[l] + (j + 0.5) * away
                                 for i in range(n) for j in range(n)]
    if "halves" in read and (abs(abs(reach["A"]) - 1) > 1e-6 or abs(abs(reach["B"]) - 1) > 1e-6):
        tex_error("the halves close round an arrow whose two reaches are one match each")
    elif "halves" in read:
        nrm = np.array([-a["u"][1], a["u"][0]]) * AL
        P = [np.zeros(2), tip, tip + nrm, nrm]
        c = (tip + nrm) / 2
        out["halves"] = dict(corners=[p / mu for p in P], centre=c / mu,
                             pieces=[(c / mu, P[i] / mu, P[(i + 1) % 4] / mu) for i in range(4)])
    for l in "AB":
        if l in read and "no-matches" not in read:
            out["rows"][l] = match_row(reach[l])
            if "both-ways" in read:
                out["back_rows"][l] = match_row(-reach[l])
    if "arrow" in read:
        out["arrow_row"] = match_row(a["L"])
    last = order[-1] if order else None
    if last and not ("squares" in read or "halves" in read):
        r = reach[last]
        out["card"] = dict(line=last, foot=r * e[last], out=-1 if r < -1e-6 else 1,
                           side=side(last))
    if keyed and pos in LINE_NAMES:
        out["labels"] = set(LINE_NAMES[pos].values())
    return out


def tailontail(startspec, closingspec, keyed):
    """The dark string's reading: its start laid over its closing, tail on
    tail; the card from the start's tip onto the closing arrow's line."""
    s, c = parse_arrow(startspec), parse_arrow(closingspec)
    Lp = pair_length(s, c)
    r = float(s["u"] @ c["u"]) * Lp                   # in matches
    tip = Lp * s["u"]
    cr = cross(c["u"], tip)
    return dict(start=s, closing=c, length=Lp, number=r, row=match_row(r),
                card=dict(foot=r * c["u"], out=-1 if r < -1e-6 else 1,
                          side=1 if abs(cr) < 1e-6 or cr > 0 else -1),
                tip=tip, labels={"start over closing"} if keyed else set())


def sharestrip(sharespec, keyed):
    """The strip folded in n parts, k shaded from the knot end (left)."""
    k, n = parse_share(sharespec)
    labels = set()
    if keyed:
        labels |= {"heads' share"} if k > 0 else set()
        labels |= {"tails' share"} if k < n else set()
    return dict(k=k, n=n, shaded=[i < k for i in range(n)], labels=labels)


def fivesquare(laidspec, keyed):
    """The outline: the loose squares set in, row filling into row from the
    near left; the three-by-three broken up first, then the four-by-four."""
    s = norm(laidspec)
    if s in ("", "empty", "none"):
        k = 0
    elif s in ("filled", "all"):
        k = 25
    else:
        k = number(s)
    if k is None or not 0 <= k <= 25:
        tex_error("unknown laid " + laidspec)
        k = 0
    return dict(laid=[(i % 5, i // 5) for i in range(k)], left9=9 - min(k, 9),
                left16=16 - max(0, k - 9), labels={"the outline"} if keyed else set())


# --- the state model: the blackboard's own vectors, in matches ----------------
# An arrow is a real state times its drawn length: the three-and-four arrow is
# five times T, heads-ward and tails-ward five times ket(0) and ket(1), the
# slant and the back-slant root two times PLUS and MINUS.  A line of the sheet
# is a question's answer: the square sheet asks ket(0), ket(1); the turned
# sheet MINUS (line A, where heads-ward swung to) and PLUS (line B).  A reach is
# the amplitude times the drawn length, sign included; its square over the
# length squared is the Born value.
MODEL = {"three-and-four": 5 * T, "heads-ward": 5 * ket(0), "tails-ward": 5 * ket(1),
         "slant": np.sqrt(2) * PLUS, "back-slant": np.sqrt(2) * MINUS}


def rotated(v, degrees):
    t = math.radians(degrees)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]) @ v


def model_lines(position, arrow=None):
    """The two answers a position of the sheet asks, as unit columns."""
    if position == "square":
        return ket(0), ket(1)
    if position == "turned":
        return MINUS, PLUS
    if position == "own":
        return rotated(ket(0).real, 20).astype(complex), rotated(ket(1).real, 20).astype(complex)
    u = arrow / np.linalg.norm(arrow)                  # along: the arrow's own line
    return u, rotated(u.real, 90).astype(complex)


def amplitude_reach(arrow, line):
    """The reach, from the blackboard: <line|arrow>, the arrow at its length."""
    r = np.vdot(line, arrow)
    assert np.isclose(r.imag, 0)                       # real arrows: no conjugate to see
    return r.real


def same_arrow(a, vec):
    """A parsed arrow is the model's column, length and direction."""
    return np.allclose(a["v"], vec.real) and np.allclose(vec.imag, 0)


def model_vec(word):
    """The model's column for an arrow word, end for end as often as it says."""
    s, flips = norm(word), 0
    if s == "reversed":
        s, flips = "three-and-four", 1
    while s.endswith(" end-for-end"):
        s, flips = s[:-len(" end-for-end")], flips + 1
    return (-1) ** flips * MODEL[s]


def check_rules():
    """The transcription against the model, on every arrow, turned end for end
    or not, under every position of the sheet and every legal reading."""
    words = list(BASE) + ["reversed"] + [w + " end-for-end" for w in BASE]
    for w in words:
        a = clean(parse_arrow, w)
        vec = model_vec(w)
        flips = (w == "reversed") + w.count("end-for-end")
        assert same_arrow(a, vec), w
        assert np.isclose(a["L"], np.linalg.norm(vec))
        # end for end: the square turned half round about its centre, so the
        # tail goes to the opposite corner and the arrow points the other way
        S, m = 1.0, 0.1
        t0 = np.array(tail_on_square(BASE[a["name"]][2], S, m))
        t1 = np.array(tail_on_square(a["corner"], S, m))
        assert np.allclose(t1, t0 if flips % 2 == 0 else np.array([S, S]) - t0), w
        for pos in ("square", "turned", "own", "along"):
            lines = model_lines(pos, vec)
            words_here = list(LINE[pos])
            reads = ["both", "none"] + words_here + [",".join(words_here[::-1])]
            for rd in reads:
                for extra in ("", ",squares", ",halves", ",both-ways", ",no-matches", ",arrow"):
                    spec = rd + extra
                    cs, errors = evaluate(crosssheet, w, pos, spec, True)
                    reaches = [amplitude_reach(vec, ln) for ln in lines]
                    off_whole = [l for l, r in zip("AB", reaches)
                                 if l in cs["read"] and abs(r - round(r)) > 1e-6]
                    want = []
                    if "squares" in extra:
                        want += ["squares only on a whole reach of matches"] * len(off_whole)
                    if "halves" in extra and not all(np.isclose(abs(r), 1) for r in reaches):
                        want.append("the halves close round an arrow whose two reaches are one match each")
                    assert errors == want, (w, pos, spec, errors)
                    if want:                                   # the fault is not drawn, and stops the build
                        assert not any(l in cs["squares"] for l in off_whole) and cs["halves"] is None
                        continue
                    for l, ln in zip("AB", lines):
                        assert np.allclose(cs["e"][l], ln.real), (w, pos)
                        assert np.isclose(cs["reach"][l], amplitude_reach(vec, ln)), (w, pos, l)
                        # the reach squared, over the length squared, is the Born value
                        assert np.isclose(cs["reach"][l] ** 2 / cs["arrow"]["L"] ** 2,
                                          abs(np.vdot(ln, vec / np.linalg.norm(vec))) ** 2)
                    # the squares on the two reaches make the square on the arrow
                    assert np.isclose(cs["reach"]["A"] ** 2 + cs["reach"]["B"] ** 2, a["L"] ** 2)
                    card = cs["card"]
                    if card is not None:
                        e = cs["e"][card["line"]]
                        assert np.isclose((a["v"] - card["foot"]) @ e, 0)   # dropped square
                        assert (card["out"] < 0) == (cs["reach"][card["line"]] < -1e-6)
                    for l, row in cs["rows"].items():
                        r = cs["reach"][l]
                        assert np.isclose(sum(b - f for f, b, _ in row), r)  # matches make the reach
                        assert sum(not s for _, _, s in row) == math.floor(abs(r) + 1e-6)
                        assert all((b - f) * r > 0 for f, b, _ in row)      # laid the reach's way
                        if l in cs["back_rows"]:                            # both-ways: the other side
                            back = cs["back_rows"][l]
                            assert len(back) == len(row)
                            assert all(np.isclose(f2, -f) and np.isclose(b2, -b) and s2 == s
                                       for (f, b, s), (f2, b2, s2) in zip(row, back))
                    for l, sq in cs["squares"].items():
                        r = cs["reach"][l]
                        assert len(sq) == round(r) ** 2                     # side |reach|, count reach^2
                        c_tip = cross(cs["e"][l], a["v"])
                        tip_side = 0 if abs(c_tip) < 1e-9 else np.sign(c_tip)
                        for c in sq:
                            assert np.sign(c @ cs["e"][l]) == np.sign(r)    # behind, if backwards
                            if tip_side != 0:
                                assert np.sign(cross(cs["e"][l], c)) == -tip_side   # away from the arrow
                    if cs["halves"] is not None:
                        pieces = cs["halves"]["pieces"]
                        area = sum(abs(cross(p - c0, q - c0)) / 2 for c0, p, q in pieces)
                        assert np.isclose(area, a["L"] ** 2)                # nothing over
                        assert np.isclose(area, cs["reach"]["A"] ** 2 + cs["reach"]["B"] ** 2)
                        assert all(np.isclose(abs(cross(p - c0, q - c0)) / 2, 0.5) for c0, p, q in pieces)
        # the strings: an echo is the pale pair with its ends exchanged, and the
        # dark string's number is the pale string's, sign included
        checked = []
        for w2 in words:
            b = clean(parse_arrow, w2)
            Lp = pair_length(a, b)
            assert any(np.isclose(Lp, x["L"]) for x in (a, b))      # one of the two's own lengths
            if np.isclose(a["L"], b["L"]):
                assert np.isclose(Lp, a["L"])
            if not np.isclose(Lp, a["L"]):
                continue                                   # a question is as long as its start
            checked.append(w2)
            es = clean(echostrings, None, w + "/" + w2, "echo", False)
            ds, dc = es["tags"]["dark start"]["arrow"], es["tags"]["dark closing"]["arrow"]
            assert ds["name"] == b["name"] and np.allclose(ds["v"], b["v"])
            assert dc["name"] == a["name"] and np.allclose(dc["v"], a["v"])
            pale = float(a["v"] @ b["u"])                  # the start's reach along the closing line
            dark = clean(tailontail, w2, w, False)["number"]      # the dark start laid over its closing
            assert np.isclose(pale, dark), (w, w2)
            assert np.isclose(pale, amplitude_reach(vec, model_vec(w2) / np.linalg.norm(model_vec(w2))))
        assert w in checked                                # every arrow asked about itself
    # every other fault the macros know, each raised once and only there
    for macro, args in ((parse_arrow, ("three-and-five",)),
                        (crosssheet, ("slant", "sideways", "both", False)),
                        (crosssheet, ("slant", "square", "slant", False)),
                        (crosssheet, ("slant", "turned", "heads-ward", False)),
                        (crosssheet, ("slant", "square", "both,squares,halves", False)),
                        (echostrings, (None, "three-and-four/", "echo", False)),
                        (echostrings, ("one in two", "bare", "bare", False)),
                        (sharestrip, ("nine in thirty", False)),
                        (sharestrip, ("ten in nine", False)),
                        (fivesquare, ("twenty-six", False))):
        assert len(evaluate(macro, *args)[1]) == 1, (macro.__name__, args)
    for k in range(26):
        f = clean(fivesquare, str(k), False)
        assert len(f["laid"]) + f["left9"] + f["left16"] == 25     # none over, none short
        assert len(set(f["laid"])) == k and all(0 <= x < 5 and 0 <= y < 5 for x, y in f["laid"])
        assert f["left9"] == 9 - min(k, 9)                          # the three-by-three first
    for n in range(1, 26):
        for k in range(n + 1):
            s = clean(sharestrip, "%d in %d" % (k, n), False)
            assert sum(s["shaded"]) == k and s["shaded"] == sorted(s["shaded"], reverse=True)


def laid(row):
    """A row of matches, read as the dialogue reads it: how many whole matches,
    how long the snapped piece (0 if none), and which way (+1 forward, -1
    backwards, 0 for no row at all)."""
    whole = sum(not s for _, _, s in row)
    piece = sum(abs(b - f) for f, b, s in row if s)
    way = int(np.sign(row[0][1] - row[0][0])) if row else 0
    return whole, round(piece, 2), way


def check_mirror_gesture(es, string):
    """The mirror gesture, as the macro computes it: a start tag knotted by its
    near left-hand corner, standing off beyond the string; the closing tag the
    same kind of square carried AS IT LIES -- its arrow neither turned round nor
    turned over -- knotted by the far right-hand corner, so it hangs on the
    near side.  Turn the string's picture half round about its midpoint and
    the two squares change places, while the arrows do not turn."""
    (x0, x1) = es["strings"][0 if string == "pale" else 1]
    start, closing = es["tags"][string + " start"], es["tags"][string + " closing"]
    Lp = es["lengths"][0 if string == "pale" else 1]
    assert np.isclose(Lp, start["arrow"]["L"])             # a question is as long as its start
    assert np.isclose(start["drawn"], closing["drawn"])    # both drawn as long as each other
    assert (start["square"][:, 1] > 0).all()               # beyond the string
    assert (closing["square"][:, 1] < 0).all()             # the near side, toward the two of them
    assert np.allclose(start["knot"], [x0 + 0.07, 0]) and np.allclose(closing["knot"], [x1 - 0.07, 0])
    assert np.allclose(start["knot"], start["square"][0] - [0, 0.012])      # near left-hand corner
    assert np.allclose(closing["knot"], closing["square"][2] + [0, 0.012])  # far right-hand corner
    # the two squares are each other's half-turn about the string's midpoint
    half = {tuple(np.round([x0 + x1 - x, -y], 9)) for x, y in start["square"]}
    assert half == {tuple(np.round(p, 9)) for p in closing["square"]}
    # ... but each arrow keeps its place in its own square and its heading
    for tag in (start, closing):
        a = tag["arrow"]
        S = tag["square"][2][0] - tag["square"][0][0]
        offset = tag["tail"] - tag["square"][0]
        assert np.allclose(offset, tail_on_square(a["corner"], S, 0.09 * S))
        assert np.allclose(tag["head"] - tag["tail"], tag["drawn"] * a["u"])
    # the half-turn would have turned the closing arrow round: it did not
    turned = -(start["head"] - start["tail"])
    if start["arrow"]["name"] == closing["arrow"]["name"]:
        assert not np.allclose(closing["head"] - closing["tail"], turned)
        assert np.allclose(closing["head"] - closing["tail"], -turned)


def pictures():
    check_whole_dialogue()
    check_rules()
    along, strings, empty, filled = check_layout("Sixth", [1, 1, 1, 1])

    # P6.1: the sheet turned until one line lies along the arrow itself: the
    # card's foot falls on the arrow's own tip -- five in five -- and along the
    # other line, no reach at all.
    assert along["macro"] == "crosssheet" and not along["keyed"], along
    cs = clean(crosssheet, along["arrow"], along["position"], along["read"], False)
    assert cs["position"] == "along" and same_arrow(cs["arrow"], 5 * T)
    assert np.allclose(cs["e"]["A"], T.real)                           # the arrow's own line
    assert cs["order"][-1] == "A" and cs["card"]["line"] == "A"        # the card at the line read last
    assert np.isclose(cs["reach"]["A"], amplitude_reach(5 * T, T)) and np.isclose(cs["reach"]["A"], 5)
    assert np.isclose(cs["reach"]["B"], 0)
    assert np.allclose(cs["card"]["foot"], cs["arrow"]["v"])           # on the arrow's own tip
    assert laid(cs["rows"]["A"]) == (5, 0, 1) and laid(cs["rows"]["B"]) == (0, 0, 0)
    assert np.isclose(cs["reach"]["A"] / cs["arrow"]["L"], np.vdot(T, T).real)   # five parts in five

    # P6.2: the arrow at both ends of the pale string, and -- once Achilles has
    # tied the dark string's copies -- the echo of that pair on the dark.
    assert strings["macro"] == "echostrings" and not strings["keyed"], strings
    assert norm(strings["dark"]) == "echo"
    es = clean(echostrings, strings["strip"], strings["pale"], strings["dark"], False)
    assert set(es["tags"]) == {"pale start", "pale closing", "dark start", "dark closing"}
    assert all(same_arrow(t["arrow"], 5 * T) for t in es["tags"].values())
    for s in ("pale", "dark"):
        check_mirror_gesture(es, s)
    pale = float(es["tags"]["pale start"]["arrow"]["v"] @ es["tags"]["pale closing"]["arrow"]["u"])
    dark = clean(tailontail, es["tags"]["dark start"]["arrow"]["name"],
                      es["tags"]["dark closing"]["arrow"]["name"], False)["number"]
    assert np.isclose(pale, 5) and np.isclose(dark, pale) and np.isclose(pale, cs["reach"]["A"])

    # P6.3, keyed: the empty five-by-five, and beside it the three-by-three and
    # the four-by-four -- the very blocks laid in the Second Challenge.
    assert empty["macro"] == "fivesquare" and empty["keyed"], empty
    f0 = clean(fivesquare, empty["laid"], True)
    assert f0["laid"] == [] and (f0["left9"], f0["left16"]) == (9, 16)
    assert f0["labels"] == {"the outline"}
    second = [picture(c) for b in blocks(DIALOG, "Second") for c in b]
    laid_then = [p for p in second if p["macro"] == "crosssheet" and "squares" in p["read"]]
    assert len(laid_then) == 1
    then = clean(crosssheet, laid_then[0]["arrow"], laid_then[0]["position"], laid_then[0]["read"], False)
    assert (len(then["squares"]["A"]), len(then["squares"]["B"])) == (f0["left9"], f0["left16"])

    # P6.4: the loose squares re-laid, row filling into row: the outline is a
    # full field, nothing left of either block.  None over, none short.
    assert filled["macro"] == "fivesquare" and not filled["keyed"], filled
    f1 = clean(fivesquare, filled["laid"], False)
    assert sorted(f1["laid"]) == [(x, y) for x in range(5) for y in range(5)]
    assert (f1["left9"], f1["left16"]) == (0, 0)
    assert len(f1["laid"]) == f0["left9"] + f0["left16"] == 25
    assert np.isclose(len(f1["laid"]), cs["reach"]["A"] ** 2)          # the square on the arrow
    assert np.isclose(len(f1["laid"]), 25 * np.vdot(T, T).real)        # asked about itself: one whole



def basis_state(b):
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    n = p.x(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def turned_state(b):
    p = Picture()
    n = p.z(0, 1, b)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def generic_state():
    p = Picture()
    n = p.x(0, 1, 0)
    a, b, c = PHASES
    chain = [p.z(0, 2, a), p.x(0, 3, b), p.z(0, 4, c)]
    p.wire(n, *chain, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def unconjugated_mirror():
    """The generic state's chain reversed but its phases left alone: the plain
    transpose, which is not the mirror image."""
    a, b, c = PHASES
    p = Picture()
    chain = [p.z(0, 1, c), p.x(0, 2, b), p.z(0, 3, a)]
    n = p.x(0, 4, 0)
    p.wire(p.inp(0), *chain, n)
    g = p.done()
    g.scalar.add_power(-1)
    return g


def echo(state, b):
    return state + b.adjoint() + b + state.adjoint()


def sketchbook():
    psi = generic_state()
    col = matrix_of(psi).ravel()
    assert np.isclose(np.linalg.norm(col), 1)          # a state, exactly

    # the theorem as a picture: state psi, wire, effect psi
    closed = psi + psi.adjoint()
    assert closed.inputs() == () and closed.outputs() == ()
    assert np.allclose(matrix_of(closed), [[1]])
    # a picture with no vertices at all, used here only as the number 1
    number_one = zx.Graph()
    assert same_picture(closed, number_one)

    # the unconjugated mirror closes the same state on a different number
    wrong = psi + unconjugated_mirror()
    assert not np.isclose(matrix_of(wrong)[0, 0], 1)
    assert not same_picture(wrong, number_one)                # the partner for the test above

    # The Echo with b = psi is the theorem twice over
    assert np.isclose(matrix_of(echo(psi, psi))[0, 0], 1)

    # the corollary, as arithmetic on evaluated pictures: the two Echo numbers
    # of a question add to the number of the closed diagram.  Adding numbers,
    # not pictures.
    whole = matrix_of(closed)[0, 0]
    for question in ((basis_state(0), basis_state(1)),
                     (turned_state(0), turned_state(1))):
        total = sum(matrix_of(echo(psi, b))[0, 0] for b in question)
        assert np.isclose(total, whole) and np.isclose(total, 1)

    # the dark string, once more: for a real state both halves agree, sign and
    # all; for the generic state they are conjugates and differ
    minus, one_state = turned_state(1), basis_state(1)
    left = matrix_of(minus + one_state.adjoint())[0, 0]
    right = matrix_of(one_state + minus.adjoint())[0, 0]
    assert np.isclose(left, -1 / np.sqrt(2)) and np.isclose(right, left)
    gl = matrix_of(psi + one_state.adjoint())[0, 0]
    gr = matrix_of(one_state + psi.adjoint())[0, 0]
    assert np.isclose(gr, np.conj(gl)) and not np.isclose(gr, gl)


if __name__ == "__main__":
    verdict("WholeSquare", blackboard, sketchbook)
