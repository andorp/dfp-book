"""Challenge 4: The Covered Coin
=================================
"My strip couldn't show it: turned end for end, half-and-half is half-and-half."  -- Achilles

Claim: a start may be a division of one whole into a heads share and a tails
share; NOT exchanges the two shares and keeps them summing to one; and the fair
half-and-half start is left where it is by NOT, although NOT is not the bare
wire -- one start is not enough to tell a tag from no tag.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import re

import numpy as np
from dfp import I2, Picture, matrix_of, same_picture, verdict

X = np.array([[0, 1], [1, 0]], dtype=complex)      # NOT
S0 = np.array([[1, 1], [0, 0]], dtype=complex)     # SET-0
S1 = np.array([[0, 0], [1, 1]], dtype=complex)     # SET-1

# convention: the first entry is the heads (0) share, the second the tails share
FIVE_ONE = np.array([5 / 6, 1 / 6], dtype=complex)   # Tortoise's strip: the die
FAIR = np.array([1 / 2, 1 / 2], dtype=complex)       # Achilles's strip: a tossed coin


def blackboard():
    assert np.isclose(FIVE_ONE.sum(), 1) and np.all(FIVE_ONE.real >= 0)

    # the strip turned end for end: the division moves from five parts out to one
    assert np.allclose(X @ FIVE_ONE, [1 / 6, 5 / 6])
    assert np.isclose((X @ FIVE_ONE).sum(), 1)          # still one strip
    assert np.allclose(X @ (X @ FIVE_ONE), FIVE_ONE)    # and back: home

    # a random division, seeded; kept away from half-and-half so the move shows
    rng = np.random.default_rng(4)
    q = rng.uniform(0.05, 0.45)
    p = np.array([q, 1 - q], dtype=complex)
    assert np.isclose(p.sum(), 1) and not np.isclose(q, 0.5)
    assert np.allclose(X @ p, p[::-1]) and not np.allclose(X @ p, p)
    assert np.isclose((X @ p).sum(), 1)
    assert np.allclose(X @ (X @ p), p)

    # every one-bit box is column-stochastic, which is exactly why the two shares
    # still make up the whole after the walk
    for M in (I2, X, S0, S1):
        assert np.allclose(M.imag, 0) and np.all(M.real >= 0)
        assert np.allclose(M.sum(axis=0), [1, 1])
        for v in (FIVE_ONE, FAIR, p):
            assert np.isclose((M @ v).sum(), 1)

    # Achilles's trap: half-and-half reads the same from either end ...
    assert np.allclose(X @ FAIR, FAIR)
    # ... and yet the tag is at work -- it is not the bare wire
    assert not np.allclose(X, I2)

    pictures()


# --------------------------------------------------------------------------
# The pictures.  The dialogue draws the Tortoise's string with
# \tagstring[place]{start}{tags}{caption} (pdf/props/ch01.tex; the rules are in
# pdf/props/ch01.lua).  start is the end tag as tied; tags are hung left to
# right, "over" for "turn it over" and "heads" for "heads up, whatever it was",
# a "+" joining two tags slid together; place is how far the traveller has got.
# The macro COMPUTES what the traveller looks like there; the dialogue never
# writes a turned result.  parse_start, parse_tags, travel and tagstring below
# are a line-for-line transcription of ch01.lua, so what is asserted here is
# what the printed page shows.  The calls are read out of this challenge's
# section of dialog.md, never copied, so an edit to the dialogue that breaks a
# picture breaks here.
# --------------------------------------------------------------------------
DIALOG = pathlib.Path(__file__).resolve().parents[2] / "dialog.md"
RAW = DIALOG.with_name("dialog.raw.md")
ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth")
# the regexes of identity.md, section 3 "Props and their pictures"
TAGSTRING = r"\\tagstring(\*?)(?:\[([^\]]*)\])?\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}"
ARROWSQUARE = r"\\arrowsquare(\*?)(?:\[([^\]]*)\])?\{([^}]*)\}\{([^}]*)\}"
MARKUP = {"tagstring", "arrowsquare", "quad", "qquad", "begin", "end", "textup"}


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
    """One call, parsed: a dict naming the macro and its arguments."""
    m = re.fullmatch(TAGSTRING, call)
    if m:
        star, place, start, tags, caption = m.groups()
        return dict(macro="tagstring", keyed=star == "*", place=place or "", start=start,
                    tags=tags, caption=caption)
    m = re.fullmatch(ARROWSQUARE, call)
    assert m, call                                   # nothing malformed
    star, view, matches, caption = m.groups()
    return dict(macro="arrowsquare", keyed=star == "*", view=view or "", matches=matches,
                caption=caption)


def blocks(path, ordinal):
    """The picture blocks of a section, in order, each the list of its calls."""
    text = section(path, ordinal)
    found = []
    for para in re.split(r"\n[ \t]*\n", text):
        if "\\tagstring" not in para and "\\arrowsquare" not in para:
            continue
        # a raw centre block that is a paragraph of its own, one line of calls
        # joined by \quad or \qquad -- so never inside a *[...]* direction
        m = re.fullmatch(r"\\begin\{center\}\n([^\n]*)\n\\end\{center\}", para.strip())
        assert m, para
        found.append([c.strip() for c in re.split(r"\\q?quad", m.group(1))])
    calls = [c for b in found for c in b]
    assert text.count("\\tagstring") + text.count("\\arrowsquare") == len(calls)
    return found


def words_only(caption):
    """Words, commas and apostrophes: no digit, brace, symbol or lone letter,
    and never a count of strings."""
    return (re.fullmatch(r"[A-Za-z']+(?:,? [A-Za-z']+)*", caption) is not None
            and all(len(w) > 1 or w == "a" for w in re.findall(r"[A-Za-z']+", caption))
            and not re.search(r"\bstrings\b", caption))


def check_layout(ordinal, sizes):
    """The section's blocks: their sizes, markup and captions, and the very same
    calls in dialog.raw.md.  Returns the parsed pictures, in order."""
    found = blocks(DIALOG, ordinal)
    assert [len(b) for b in found] == sizes, (ordinal, [len(b) for b in found])
    assert blocks(RAW, ordinal) == found, ordinal       # identical in the raw dialogue
    assert set(re.findall(r"\\([A-Za-z]+)", section(DIALOG, ordinal))) <= MARKUP
    pics = [picture(c) for b in found for c in b]
    for b in found:
        assert len(b) <= 4
        squares = [picture(c)["macro"] == "tagstring" and picture(c)["start"].strip() == "square"
                   for c in b]
        assert all(squares) or not any(squares), b      # a square-tied string keeps to its own line
    for p in pics:
        assert words_only(p["caption"]), p["caption"]
        assert p.get("place", "").strip() != "off"      # [off] is not used
    return pics


def check_whole_dialogue():
    """The five challenge sections, in order, and every picture of the dialogue
    in one of them."""
    text = read(DIALOG)
    assert tuple(re.findall(r"^## (\w+) Challenge\b", text, re.M)) == ORDINALS
    total = len(re.findall(TAGSTRING, text)) + len(re.findall(ARROWSQUARE, text))
    assert text.count("\\tagstring") + text.count("\\arrowsquare") == total
    assert total == sum(len(c) for o in ORDINALS for c in blocks(DIALOG, o))


# --- ch01.lua, transcribed ---------------------------------------------------
WORDNUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
POS = {"start": 0, "first": 1, "second": 2, "third": 3, "fourth": 4}


def parse_start(spec):
    spec = re.sub(r"\s", "", spec)
    if spec in ("heads", "tails"):
        return dict(kind="coin", face=spec)
    if spec in ("none", ""):
        return dict(kind="none")
    if spec == "square":
        return dict(kind="square", turned=False)
    m = re.fullmatch(r"strip-(\w+)", spec)
    if m:
        k = m.group(1)
        n = WORDNUM.get(k, int(k) if k.isdigit() else None)
        if n is not None and 0 <= n <= 6:
            return dict(kind="strip", dark=n, turned=False)
    raise AssertionError("tagstring: unknown start " + spec)


def parse_tags(spec):
    """Groups of touching tags, and the tags one by one, left to right."""
    groups, flat = [], []
    for g in re.sub(r"\s", "", spec).split(","):
        grp = []
        for t in g.split("+"):
            if t == "":
                continue
            assert t in ("over", "heads"), "tagstring: unknown tag " + t
            grp.append(t)
            flat.append(t)
        if grp:
            groups.append(grp)
    return groups, flat


def travel(st, tags, n):
    """The traveller after the first n tags."""
    s = dict(st)
    for t in tags[:n]:
        if s["kind"] == "coin":
            if t == "over":
                s["face"] = "tails" if s["face"] == "heads" else "heads"
            else:
                s["face"] = "heads"
        elif s["kind"] in ("strip", "square"):
            assert t == "over", "a 'heads up, whatever it was' tag cannot be done to a " + s["kind"]
            s["turned"] = not s["turned"]
    return s


def tagstring(place, startspec, tagspec):
    """props.ch01_tagstring, all but the drawing: its error checks, how many
    tags the traveller has passed, and the traveller there."""
    where = re.sub(r"\s", "", place) or "start"
    assert where in POS or where in ("end", "off"), "tagstring: unknown place " + where
    st = parse_start(startspec)
    groups, tags = parse_tags(tagspec)
    n = len(tags)
    passed = n if where == "end" else POS.get(where, 0)
    assert passed <= n, "tagstring: the traveller cannot be past tag %d of %d" % (passed, n)
    travel(st, tags, n)                    # every tag is checked, wherever the traveller is
    s = travel(st, tags, passed)
    walked = where == "end" or passed > 0
    if st["kind"] == "square" and where != "off":
        assert not walked, "tagstring: the square is only drawn tied on at the start"
    return dict(start=st, groups=groups, tags=tags, where=where, passed=passed, now=s)


# --- the strip, in the chapter's state model ---------------------------------
# strip-k, tied on by its dark end, is the division (k/6, (6-k)/6), heads'
# share first: from the knot out to the change of shade.  "turn it over" is
# NOT, the matrix blackboard() uses; on the strip it is the turn end for end,
# heads' share h becoming six minus h.
PLACES = ("start", "first", "second", "third", "fourth")


def strip_start(dark):
    return np.array([dark / 6, (6 - dark) / 6], dtype=complex)


def carry(v, tags):
    """The tags met, in the order met: the string read left to right."""
    for t in tags:
        assert t == "over"
        v = X @ v
    return v


def strip_panels(dark, turned):
    """ch01.lua: the six panels from the knot end, True = shaded."""
    return [(7 - i if turned else i) <= dark for i in range(1, 7)]


def share(now):
    """ch01.lua: heads' share, as the keyed picture labels it."""
    return 6 - now["dark"] if now["turned"] else now["dark"]


def share_by_eye(panels):
    """Heads' share read off the drawing: panels from the knot to the change of
    shade (for a strip that has a change)."""
    k = 0
    while k < 6 and panels[k] == panels[0]:
        k += 1
    return k


def check_strip_rule():
    """The macro's strip against the model: every strip from none to six
    shaded panels, up to four turn-over tags, every place it can have got to."""
    for w, k in WORDNUM.items():                        # strip-five is strip-5
        assert parse_start("strip-" + w) == parse_start("strip-%d" % k)
    for dark in range(7):
        for n in range(5):
            tags = ["over"] * n
            for place in PLACES[:n + 1] + ("end",):
                t = tagstring(place, "strip-%d" % dark, ",".join(tags))
                v = carry(strip_start(dark), tags[:t["passed"]])
                assert np.isclose(6 * v[0], share(t["now"]))       # macro = model
                assert np.isclose(v.sum(), 1)                        # one strip, one length
                panels = strip_panels(dark, t["now"]["turned"])
                assert sum(panels) == dark                           # turning shades nothing
                if 0 < dark < 6:
                    assert share_by_eye(panels) == share(t["now"])  # as drawn


def pictures():
    check_whole_dialogue()
    check_strip_rule()
    assert np.allclose(strip_start(3), FAIR) and np.allclose(strip_start(5), FIVE_ONE)

    # P4.1, the keyed strip picture: Achilles's tossed coin tied on, the string
    # still bare.  P4.2: carried to the tag and turned there.  P4.3: the
    # Tortoise's strip, one tag on the string.  P4.4: carried and turned.
    # P4.5: tied on again, the pair lifted off.  Each row: shaded panels as tied,
    # the tags' groups, where the strip is, keyed?, heads' share in sixths as
    # identity.md section 3b reads it, and whether the white end is at the string.
    one = [["over"]]
    scene = [(3, [], "start", True, 3, False),
             (3, one, "first", False, 3, True),
             (5, one, "start", False, 5, False),
             (5, one, "first", False, 1, True),
             (5, [], "start", False, 5, False)]
    pics = check_layout("Fourth", [1, 1, 1, 1, 1])
    assert len(pics) == len(scene)
    strips = []
    for p, (dark, groups, place, keyed, reading, white) in zip(pics, scene):
        assert p["macro"] == "tagstring" and p["keyed"] == keyed, p
        t = tagstring(p["place"], p["start"], p["tags"])
        assert t["start"] == dict(kind="strip", dark=dark, turned=False), p   # as tied
        assert t["groups"] == groups and t["where"] == place, p
        if keyed:
            assert len(t["tags"]) <= 1, p               # a keyed strip carries one tag at most
        v = carry(strip_start(dark), t["tags"][:t["passed"]])                 # the model
        assert np.isclose(6 * v[0], share(t["now"])) and share(t["now"]) == reading, p
        panels = strip_panels(dark, t["now"]["turned"])
        assert (not panels[0]) == white, p              # which end is at the string
        strips.append((v, panels))
    (tossed, _), (tossed_turned, _), (die, _), (die_turned, turned_panels), (home, _) = strips

    # the trap: half-and-half turned end for end reads as it did ...
    assert np.allclose(tossed_turned, tossed)
    # ... and five-and-one does not: "It moved!"
    assert not np.allclose(die_turned, die)
    assert np.allclose(die_turned, X @ FIVE_ONE)
    # the shading only marks the place: dark does not mean heads
    assert sum(turned_panels) == 5 and not np.isclose(6 * die_turned[0], 5)
    # the pair lifted off, and the strip is as it began: the round trip
    assert np.allclose(home, die) and np.allclose(carry(die, ["over", "over"]), die)

    # the keyed rule: of all the dialogue's strip pictures, the first -- this
    # challenge's first -- is keyed, and no other
    strips_all = [m for m in re.finditer(TAGSTRING, read(DIALOG))
                  if parse_start(m.group(3))["kind"] == "strip"]
    assert strips_all[0].group(0) == blocks(DIALOG, "Fourth")[0][0]
    assert [m.group(1) == "*" for m in strips_all] == [True] + [False] * (len(strips_all) - 1)


def fair_coin():
    """The fair start IS a picture: one one-legged node, its two stray sqrt(2)s removed."""
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    n = p.z(0, 1, 0)
    p.wire(n, p.out(0))
    g = p.done()
    g.scalar.add_power(-2)
    return g


def fair_then_nots(k):
    """The fair start, then k NOT boxes, in a row on one wire."""
    p = Picture()
    n = p.z(0, 1, 0)
    nodes = [p.x(0, 3 + 2 * i, 1) for i in range(k)]
    p.wire(n, *nodes, p.out(0))
    g = p.done()
    g.scalar.add_power(-2)
    return g


def not_box():
    p = Picture()
    n = p.x(0, 2, 1)
    p.wire(p.inp(0), n, p.out(0))
    return p.done()


def set0_box():
    p = Picture()
    p.wire(p.inp(0), p.z(0, 2, 0))
    p.wire(p.x(0, 3, 0), p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def two_not_boxes():
    p = Picture()
    a, b = p.x(0, 2, 1), p.x(0, 4, 1)
    p.wire(p.inp(0), a, b, p.out(0))
    return p.done()


def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def sketchbook():
    assert np.allclose(matrix_of(fair_coin()).ravel(), FAIR)

    # the trap, as pictures: the fair start is where it was, and yet the tag is
    # not the bare wire.  Exact, scalar included.
    assert same_picture(fair_then_nots(1), fair_coin())
    assert not same_picture(not_box(), bare_wire())
    assert same_picture(fair_then_nots(2), fair_coin())

    # A general division such as five-and-one has no exact picture built from these
    # nodes, so for it we check the boxes instead: what the pictures denote is
    # column-stochastic, and acts on the strip as claimed.
    xm, s0m = matrix_of(not_box()), matrix_of(set0_box())
    for M in (xm, s0m):
        assert np.allclose(M.imag, 0) and np.all(M.real >= -1e-12)
        assert np.allclose(M.sum(axis=0), [1, 1])
    rng = np.random.default_rng(44)
    q = rng.uniform(0.05, 0.45)
    p = np.array([q, 1 - q], dtype=complex)
    for v in (FIVE_ONE, p):
        assert np.allclose(xm @ v, v[::-1]) and not np.allclose(xm @ v, v)
        assert np.isclose((xm @ v).sum(), 1)
        assert np.allclose(matrix_of(two_not_boxes()) @ v, v)


if __name__ == "__main__":
    verdict("HiddenCoin", blackboard, sketchbook)
