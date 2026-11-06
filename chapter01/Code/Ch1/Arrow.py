"""Challenge 5: The Arrow
==========================
"Five. Not seven."  -- Achilles

Claim: a qubit state is normalized by its squared moduli and not by its entries;
NOT exchanges the two reaches and keeps the length; and two NOT boxes in a row
return the arrow exactly -- the third passenger of The Round Trip.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import itertools
import math
import re
from fractions import Fraction

import numpy as np
from dfp import I2, Picture, ket, matrix_of, random_state, same_picture, verdict

X = np.array([[0, 1], [1, 0]], dtype=complex)      # NOT
S0 = np.array([[1, 1], [0, 0]], dtype=complex)     # SET-0

ARROW = np.array([3 / 5, 4 / 5], dtype=complex)    # three matches along, four up


def blackboard():
    # nine and sixteen make twenty-five: the squares on the reaches make the whole
    assert np.isclose(abs(ARROW[0]) ** 2 + abs(ARROW[1]) ** 2, 1)
    assert np.isclose(np.linalg.norm(ARROW), 1)
    # three and four overshoot: as a wager it is nonsense
    assert np.isclose(ARROW.sum().real, 7 / 5) and not np.isclose(ARROW.sum().real, 1)

    # turned over: the reaches change places, and it is still one arrow
    assert np.allclose(X @ ARROW, [4 / 5, 3 / 5])
    assert np.isclose(np.linalg.norm(X @ ARROW), 1)
    assert not np.allclose(X @ ARROW, ARROW)
    assert np.allclose(X @ (X @ ARROW), ARROW)              # home

    # a generic passenger: a complex unit vector, asserted non-real with a
    # non-real ratio, so nothing here can hide behind a symmetry
    rng = np.random.default_rng(5)
    psi = random_state(rng)
    assert not np.allclose(psi, psi.conj())
    assert not np.isclose(psi[0], 0) and not np.isclose(psi[1], 0)
    assert not np.isclose((psi[1] / psi[0]).imag, 0)
    assert np.isclose(np.linalg.norm(psi), 1)
    assert np.allclose(X @ psi, psi[::-1])
    assert np.isclose(np.linalg.norm(X @ psi), 1)
    assert not np.allclose(X @ psi, psi)
    assert np.allclose(X @ (X @ psi), psi)                  # The Round Trip

    # the two definite starts are legal under all three rules
    for e in (ket(0), ket(1)):
        assert np.isclose(e.sum().real, 1)                  # a bit, and a wager
        assert np.isclose(np.linalg.norm(e), 1)             # and an arrow

    # not every classical box may be hung on a wire carrying an arrow (chapter.md,
    # end of 1.5): SET-0 keeps sums but not sums of squares
    assert np.isclose((S0 @ ARROW).sum(), ARROW.sum())      # the sum survives
    assert np.allclose(S0 @ ARROW, [7 / 5, 0])
    assert not np.isclose(np.linalg.norm(S0 @ ARROW), 1)

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


# --- the square, in the chapter's state model --------------------------------
# The arrow pencilled from the near left-hand corner is ARROW = (3/5, 4/5),
# heads-ward first; with the matches for a ruler its reaches are five times its
# entries and its length five times its norm.  "turn it over" is NOT, the
# matrix blackboard() uses: the square turned over about the line from the
# arrow's corner to the far corner, the two reaches changing places.
def in_matches(v):
    """The reaches heads-ward and tails-ward, and the length, in matches."""
    return (5 * v[0].real, 5 * v[1].real, 5 * np.linalg.norm(v))


def carry_arrow(v, tags):
    """The tags met, in the order met: the string read left to right."""
    for t in tags:
        assert t == "over"
        v = X @ v
    return v


def arrowsquare(view, rowspec):
    """props.ch01_arrowsquare, all but the drawing: seen from behind or not,
    the reaches drawn, and how many matches lie in each row."""
    view = re.sub(r"\s", "", view)
    if view in ("drawn", "", "twice", "home"):
        back = False
    else:
        assert view == "over", "arrowsquare: unknown view " + view
        back = True
    rows = set()
    for t in re.sub(r"\s", "", rowspec).split(","):
        if t in ("along", "up", "arrow"):
            rows.add(t)
        elif t == "edges":
            rows |= {"along", "up"}
        elif t == "all":
            rows |= {"along", "up", "arrow"}
        else:
            assert t in ("none", ""), "arrowsquare: unknown row " + t
    a, b = (4, 3) if back else (3, 4)
    count = dict(along=a if "along" in rows else 0, up=b if "up" in rows else 0,
                 arrow=math.floor(math.sqrt(a * a + b * b) + 0.5) if "arrow" in rows else 0)
    return dict(back=back, reaches=(a, b), rows=rows, count=count)


def square_on_string(now):
    """props.ch01_tagstring, the square tied on: its reaches as drawn."""
    return (4, 3) if now["turned"] else (3, 4)


def check_square_rule():
    """The macro's square against the model, on every view and every choice of
    rows, and tied on under up to four turn-over tags."""
    for view, turns in (("drawn", 0), ("", 0), ("home", 0), ("twice", 2), ("over", 1)):
        v = np.linalg.matrix_power(X, turns) @ ARROW
        along, up, length = in_matches(v)
        for k in range(4):
            for rows in itertools.combinations(("along", "up", "arrow"), k):
                for spec in (",".join(rows),) + (("none",) if k == 0 else ()):
                    sq = arrowsquare(view, spec)
                    assert sq["back"] == (turns % 2 == 1)         # seen through the paper
                    assert np.allclose(sq["reaches"], (along, up))
                    assert sq["count"]["along"] == (round(along) if "along" in rows else 0)
                    assert sq["count"]["up"] == (round(up) if "up" in rows else 0)
                    assert sq["count"]["arrow"] == (round(length) if "arrow" in rows else 0)
        assert np.isclose(along ** 2 + up ** 2, 25) and np.isclose(length, 5)
        assert not np.isclose(along + up, length)          # three and four are not seven
    assert arrowsquare("", "edges") == arrowsquare("drawn", "along,up")
    assert arrowsquare("", "all") == arrowsquare("drawn", "along,up,arrow")
    for n in range(5):
        t = tagstring("start", "square", ",".join(["over"] * n))
        assert square_on_string(t["now"]) == (3, 4)          # drawn tied on, as tied
        whole = travel(t["start"], t["tags"], n)             # what the tags would do
        assert np.allclose(square_on_string(whole),
                           in_matches(np.linalg.matrix_power(X, n) @ ARROW)[:2])


def pictures():
    check_whole_dialogue()
    check_square_rule()
    pics = check_layout("Fifth", [1, 1, 1, 2])

    # P5.1, the keyed square picture: the matches along the two edges.  P5.2:
    # along the arrow too, "five, not seven".  P5.3: turned over, matches laid
    # again.  Each row: turns, the rows of matches, keyed?, and the reaches and
    # length identity.md section 3b reads (3/4/5 as drawn, 4/3/5 turned over).
    close_ups = [(0, {"along", "up"}, True, (3, 4, 0)),
                 (0, {"along", "up", "arrow"}, False, (3, 4, 5)),
                 (1, {"along", "up", "arrow"}, False, (4, 3, 5))]
    for p, (turns, rows, keyed, reading) in zip(pics[:3], close_ups):
        assert p["macro"] == "arrowsquare" and p["keyed"] == keyed, p
        sq = arrowsquare(p["view"], p["matches"])
        assert sq["rows"] == rows and sq["back"] == (turns == 1), p
        v = np.linalg.matrix_power(X, turns) @ ARROW                     # the model
        assert np.allclose(sq["reaches"], in_matches(v)[:2]), p
        c = sq["count"]
        assert (c["along"], c["up"], c["arrow"]) == reading, p
        if "arrow" in rows:
            assert c["arrow"] == round(in_matches(v)[2]) == 5, p          # five
            assert c["along"] + c["up"] == 7 != c["arrow"], p            # not seven
            assert c["along"] ** 2 + c["up"] ** 2 == c["arrow"] ** 2, p  # 9 + 16 = 25

    # P5.4, one block of two: the square tied on by the arrow's corner with two
    # turn-over tags on the string, and the same square with them lifted off.
    # Nothing is walked: the square is drawn at the start, as drawn.
    two = [["over"], ["over"]]
    for p, groups in zip(pics[3:], (two, [])):
        assert p["macro"] == "tagstring" and not p["keyed"], p
        t = tagstring(p["place"], p["start"], p["tags"])
        assert t["start"] == dict(kind="square", turned=False), p
        assert t["groups"] == groups and t["where"] == "start" and t["passed"] == 0, p
        assert square_on_string(t["now"]) == (3, 4), p
        assert np.allclose(square_on_string(t["now"]), in_matches(ARROW)[:2]), p
        # and had it been walked past both tags, it would arrive as it left
        whole = travel(t["start"], t["tags"], len(t["tags"]))
        assert np.allclose(square_on_string(whole),
                           in_matches(carry_arrow(ARROW, t["tags"]))[:2]), p
        assert square_on_string(whole) == (3, 4), p

    # the keyed rule: the first close-up of the square in the dialogue -- this
    # challenge's first picture -- is keyed, no other close-up is, and no
    # picture of the square tied on is
    text = read(DIALOG)
    ups = list(re.finditer(ARROWSQUARE, text))
    assert ups[0].group(0) == blocks(DIALOG, "Fifth")[0][0]
    assert [m.group(1) == "*" for m in ups] == [True] + [False] * (len(ups) - 1)
    tied = [m for m in re.finditer(TAGSTRING, text) if parse_start(m.group(3))["kind"] == "square"]
    assert tied and not any(m.group(1) == "*" for m in tied)


PHASES = (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4))


def arrow_then_nots(k):
    """A generic arrow -- the state box 0 followed by three phase nodes -- then k
    NOT boxes, in a row on one wire, read left to right."""
    p = Picture()
    # PyZX draws this with the spiders you meet in Chapter 8.
    s = p.x(0, 1, 0)
    a, b, c = PHASES
    chain = [p.z(0, 2, a), p.x(0, 3, b), p.z(0, 4, c)]
    nots = [p.x(0, 6 + 2 * i, 1) for i in range(k)]
    p.wire(s, *chain, *nots, p.out(0))
    g = p.done()
    g.scalar.add_power(-1)
    return g


def not_box():
    p = Picture()
    n = p.x(0, 2, 1)
    p.wire(p.inp(0), n, p.out(0))
    return p.done()


def two_not_boxes():
    p = Picture()
    a, b = p.x(0, 2, 1), p.x(0, 4, 1)
    p.wire(p.inp(0), a, b, p.out(0))
    return p.done()


def sketchbook():
    psi = matrix_of(arrow_then_nots(0)).ravel()
    assert np.isclose(np.linalg.norm(psi), 1)               # exact after the scalar fix
    assert not np.isclose(psi[0], 0) and not np.isclose(psi[1], 0)
    assert not np.isclose((psi[1] / psi[0]).imag, 0)        # a genuine arrow, not a wager

    # one turn-over: the two reaches change places
    turned = matrix_of(arrow_then_nots(1)).ravel()
    assert np.allclose(turned, psi[::-1])
    assert np.allclose(turned, X @ psi)
    assert not same_picture(arrow_then_nots(1), arrow_then_nots(0))

    # two turn-overs: the same picture again, exactly -- The Round Trip, read with
    # its third passenger
    assert same_picture(arrow_then_nots(2), arrow_then_nots(0))

    # The three-four-five arrow itself has no exact picture built from these nodes,
    # so it is checked on the blackboard above and against the box here.
    assert np.allclose(matrix_of(not_box()) @ ARROW, [4 / 5, 3 / 5])
    assert np.allclose(matrix_of(two_not_boxes()) @ ARROW, ARROW)
    assert np.allclose(matrix_of(two_not_boxes()), I2)


if __name__ == "__main__":
    verdict("Arrow", blackboard, sketchbook)
