"""Challenge 1: A Coin on a String
==================================
"Read it from the left. The end tag is where it starts."  -- Tortoise

Claim: NOT turns the state 0 into the state 1 and the state 1 into the state 0
-- X e0 = e1 and X e1 = e0 -- and NOT is not the identity.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
import re

import numpy as np
from dfp import I2, Picture, ket, matrix_of, same_picture, verdict

X = np.array([[0, 1], [1, 0]], dtype=complex)      # NOT
S0 = np.array([[1, 1], [0, 0]], dtype=complex)     # SET-0: "heads up, whatever it was"


def blackboard():
    # convention: the first (top) entry belongs to 0, the second to 1
    e0, e1 = ket(0), ket(1)
    assert np.allclose(e0, [1, 0]) and np.allclose(e1, [0, 1])

    # one tag on the string: "turn it over"
    assert np.allclose(X @ e0, e1)                 # heads in, tails out
    assert np.allclose(X @ e1, e0)                 # and the other way about
    assert not np.allclose(X @ e0, e0)             # it really moved
    assert not np.allclose(X, I2)                  # the tag is not a bare string

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


# --- the coin, in the chapter's state model ----------------------------------
# heads is the first entry, as on the blackboard; "turn it over" is NOT and
# "heads up, whatever it was" is SET-0, the very matrices blackboard() uses.
HEADS = np.array([1, 0], dtype=complex)
TAILS = np.array([0, 1], dtype=complex)
FACE = {"heads": HEADS, "tails": TAILS}
TAG = {"over": X, "heads": S0}
PLACES = ("start", "first", "second", "third", "fourth")


def carry(v, tags):
    """The tags met, in the order met: the string read left to right."""
    for t in tags:
        v = TAG[t] @ v
    return v


def face_of(v):
    names = [f for f, u in FACE.items() if np.allclose(v, u)]
    assert len(names) == 1, v
    return names[0]


def check_coin_rule():
    """The macro's coin against the model: every start, every string of up to
    four tags, every place the coin can have got to."""
    for start in FACE:
        for n in range(5):
            for tags in itertools.product(("over", "heads"), repeat=n):
                for place in PLACES[:n + 1] + ("end",):
                    t = tagstring(place, start, ",".join(tags))
                    assert t["now"]["face"] == face_of(carry(FACE[start], tags[:t["passed"]]))


def check_coin_pictures(ordinal, sizes, scene):
    """The section's pictures against the scene, one row per picture:
    (end tag as tied, the tags' groups on the string, where the coin is,
    keyed?, the face identity.md section 3b reads).  Returns the model's
    coin at each picture."""
    check_whole_dialogue()
    check_coin_rule()
    pics = check_layout(ordinal, sizes)
    assert len(pics) == len(scene), (ordinal, len(pics))
    coins = []
    for p, (start, groups, place, keyed, reading) in zip(pics, scene):
        assert p["macro"] == "tagstring" and p["keyed"] == keyed, p
        t = tagstring(p["place"], p["start"], p["tags"])
        assert t["start"] == dict(kind="coin", face=start), p       # the end tag as tied
        assert t["groups"] == groups and t["where"] == place, p
        v = carry(FACE[start], t["tags"][:t["passed"]])              # the model
        assert t["now"]["face"] == face_of(v) == reading, p         # macro = model = card
        coins.append(v)
    return coins


def pictures():
    # P1.1, the keyed coin picture: the first walk, from "heads up", past one
    # "turn it over" tag, to the right-hand end.  P1.2: the other way about.
    one = [["over"]]
    first, other = check_coin_pictures("First", [1, 1], [
        ("heads", one, "end", True, "tails"),
        ("tails", one, "end", False, "heads"),
    ])
    assert np.allclose(first, X @ HEADS) and np.allclose(other, X @ TAILS)
    assert not np.allclose(first, HEADS)           # the tag did something

    # the keyed rule: of all the dialogue's coin pictures, the first -- this
    # challenge's first -- is keyed, and no other
    coins = [m for m in re.finditer(TAGSTRING, read(DIALOG))
             if parse_start(m.group(3))["kind"] == "coin"]
    assert coins[0].group(0) == blocks(DIALOG, "First")[0][0]
    assert [m.group(1) == "*" for m in coins] == [True] + [False] * (len(coins) - 1)


def not_box():
    p = Picture()
    n = p.x(0, 2, 1)                 # PyZX draws this with the spiders you meet in Chapter 8.
    p.wire(p.inp(0), n, p.out(0))    # input boundary on the left: read left to right
    return p.done()


def bare_wire():
    p = Picture()
    p.wire(p.inp(0), p.out(0))
    return p.done()


def state(bit):
    """The state box labelled 0 or 1: a one-legged node, and the stray sqrt(2) removed."""
    p = Picture()
    s = p.x(0, 1, bit)
    p.wire(s, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def state_then_not(bit):
    """A state box, then a NOT box, in a row on one wire, read left to right."""
    p = Picture()
    s = p.x(0, 1, bit)
    n = p.x(0, 3, 1)
    p.wire(s, n, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def sketchbook():
    # the boxes denote exactly what the blackboard writes, scalar included
    assert np.allclose(matrix_of(not_box()), X)
    assert np.allclose(matrix_of(bare_wire()), I2)
    assert np.allclose(matrix_of(state(0)).ravel(), ket(0))
    assert np.allclose(matrix_of(state(1)).ravel(), ket(1))
    assert np.isclose(np.linalg.norm(matrix_of(state(0))), 1)
    assert np.isclose(np.linalg.norm(matrix_of(state(1))), 1)

    # picture against picture: "state 0, then NOT" IS the picture "state 1"
    assert same_picture(state_then_not(0), state(1))
    assert same_picture(state_then_not(1), state(0))
    assert not same_picture(state_then_not(0), state(0))
    assert not same_picture(state_then_not(1), state(1))

    # and the tag is not a bare string
    assert not same_picture(not_box(), bare_wire())


if __name__ == "__main__":
    verdict("Flip", blackboard, sketchbook)
