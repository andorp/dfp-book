"""Challenge 3: Which Tag First?
=================================
"Same tags. Walk it again."  -- Tortoise

Claim: a row of boxes read left to right is the matrix product read right to
left, and the order matters: NOT then SET-0 is SET-0, while SET-0 then NOT is
SET-1.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
import re

import numpy as np
from dfp import I2, Picture, ket, matrix_of, random_matrix, random_state, same_picture, verdict

X = np.array([[0, 1], [1, 0]], dtype=complex)      # NOT
S0 = np.array([[1, 1], [0, 0]], dtype=complex)     # SET-0: "heads up, whatever it was"
S1 = np.array([[0, 0], [1, 1]], dtype=complex)     # SET-1
FOUR = {"I": I2, "X": X, "S0": S0, "S1": S1}


def blackboard():
    # a row "A then B" is the product B A -- the picture left to right, the product
    # right to left.  With two different boxes the convention becomes visible:
    assert np.allclose(S0 @ X, S0)                 # NOT then SET-0
    assert np.allclose(X @ S0, S1)                 # SET-0 then NOT
    assert not np.allclose(S0 @ X, X @ S0)         # same tags, other face

    # the same coin, the same start, the two tags in the two orders
    assert np.allclose(S0 @ X @ ket(0), ket(0))
    assert np.allclose(X @ S0 @ ket(0), ket(1))

    # one of the twins comes off, and never both: SET-0 twice does what it does once
    assert np.allclose(S0 @ S0, S0)
    assert not np.allclose(S0 @ S0, I2)
    # after SET-0 nothing on the string can tell how the coin started ...
    assert np.allclose(S0 @ ket(0), S0 @ ket(1))
    assert np.allclose(X @ S0 @ ket(0), X @ S0 @ ket(1))   # "both starts arrive alike"
    # ... and nothing hung further along would bring it back
    assert np.isclose(np.linalg.det(S0), 0)
    rng = np.random.default_rng(3)
    B = random_matrix(rng)
    assert np.allclose((B @ S0)[:, 0], (B @ S0)[:, 1])     # the two columns collapse
    assert not np.allclose(B @ S0, I2)
    # while the turn-over tag never loses anything
    assert not np.allclose(X @ ket(0), X @ ket(1))
    assert not np.isclose(np.linalg.det(X), 0)

    # the four one-bit boxes are closed under rows: all sixteen products
    for A in FOUR.values():
        for C in FOUR.values():
            assert any(np.allclose(C @ A, D) for D in FOUR.values())

    # left to right versus right to left, on a generic non-commuting pair
    A, C = random_matrix(rng), random_matrix(rng)
    assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())
    assert not np.allclose(A @ C, C @ A)
    v = random_state(rng)
    assert np.allclose(C @ (A @ v), (C @ A) @ v)           # first A, then C
    assert not np.allclose(C @ (A @ v), (A @ C) @ v)

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
    # P3.1: the twins, walked from tails.  P3.2: "turn it over" then "heads up,
    # whatever it was", from heads.  P3.3: the same two tags, places swapped.
    # P3.4: swapped, walked from tails.
    twins, as_hung, swapped = [["heads"], ["heads"]], [["over"], ["heads"]], [["heads"], ["over"]]
    rows = [("tails", twins, "end", False, "heads"),
            ("heads", as_hung, "end", False, "heads"),
            ("heads", swapped, "end", False, "tails"),
            ("tails", swapped, "end", False, "tails")]
    p1, p2, p3, p4 = check_coin_pictures("Third", [1, 1, 1, 1], rows)
    # "along a bare string it would have arrived tails!": the twins can't come off
    assert not np.allclose(p1, TAILS)
    # and one of them lifted, the other does what the pair did
    assert np.allclose(carry(TAILS, ["heads"]), p1)
    # same tags, same start, the other face
    assert np.allclose(p2, HEADS) and np.allclose(p3, TAILS)
    # "Both starts arrive alike!"
    assert np.allclose(p3, p4)

    # The mix-up the convention rules out: the string read from the right.  On
    # the swapped string that reading gives heads, not the tails the page shows.
    pics = check_layout("Third", [1, 1, 1, 1])
    hung = [tagstring(p["place"], p["start"], p["tags"]) for p in pics]
    assert sorted(hung[1]["tags"]) == sorted(hung[2]["tags"])      # the very same two tags
    assert hung[1]["tags"] == hung[2]["tags"][::-1]                 # in the other order
    assert face_of(carry(HEADS, hung[2]["tags"][::-1])) != hung[2]["now"]["face"]
    # and so it does on many strings: of the 31 strings of up to four tags,
    # each walked from both starts (62 walks), the two readings part on 24
    differ = sum(face_of(carry(FACE[start], tags[::-1]))
                 != tagstring("end", start, ",".join(tags))["now"]["face"]
                 for start in FACE for n in range(5)
                 for tags in itertools.product(("over", "heads"), repeat=n))
    assert differ == 24, differ


def row_of(*tags):
    """A row of boxes on one wire, read left to right.  Tags: "X", "S0", "S1".

    NOT is one two-legged node of phase pi.  SET-0 and SET-1 are drawn in the book
    as ordinary labelled boxes; here each is a one-legged node closing the incoming
    wire and a one-legged node opening the outgoing one, which together denote
    exactly [[1,1],[0,0]] and [[0,0],[1,1]].
    """
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    prev, col, power = p.inp(0), 2, 0
    for tag in tags:
        if tag == "X":
            node = p.x(0, col, 1)
            p.wire(prev, node)
            prev = node
        else:
            p.wire(prev, p.z(0, col, 0))
            prev = p.x(0, col + 1, 0 if tag == "S0" else 1)
            power -= 1
        col += 2
    p.wire(prev, p.out(0))
    g = p.done()
    g.scalar.add_power(power)
    return g


def sketchbook():
    # the boxes denote exactly what the blackboard writes
    assert np.allclose(matrix_of(row_of("X")), X)
    assert np.allclose(matrix_of(row_of("S0")), S0)
    assert np.allclose(matrix_of(row_of("S1")), S1)
    assert np.allclose(matrix_of(row_of()), I2)

    # picture against picture: the order along the wire is part of what it says
    assert same_picture(row_of("X", "S0"), row_of("S0"))
    assert same_picture(row_of("S0", "X"), row_of("S1"))
    assert not same_picture(row_of("X", "S0"), row_of("S0", "X"))

    # one of the twins comes off, and never both
    assert same_picture(row_of("S0", "S0"), row_of("S0"))
    assert not same_picture(row_of("S0", "S0"), row_of())

    # a row read left to right is the product read right to left
    for tags in (("X", "S0"), ("S0", "X"), ("S0", "S0"), ("X", "X")):
        first, then = (matrix_of(row_of(t)) for t in tags)
        assert np.allclose(matrix_of(row_of(*tags)), then @ first)


if __name__ == "__main__":
    verdict("OrderMatters", blackboard, sketchbook)
