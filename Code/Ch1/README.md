# Chapter 1 — Not Just a Bit: the companion code

Five challenges in the dialogue, five files here. Each file checks its claim twice, as the
chapter proves its theorem twice: `blackboard()` with NumPy columns and $2\times2$ matrices,
`sketchbook()` with pictures built in PyZX and compared as exact tensors — every number
included, because no equation in this chapter holds only "up to a scalar".

Run one file, or the whole chapter, from the repository root:

```bash
.venv/bin/python chapter01/Code/Ch1/TwiceOver.py
bash check-code.sh 1          # must end with: 5 files, 0 failed
```

Each file prints one line per strand and exits non-zero if any `assert` fails. Running the
files is this book's analogue of type-checking.

**Conventions, asserted where they are used.** The first (top) entry of a column belongs to
the state $0$, the second to the state $1$: `ket(0)` is $(1,0)$. A picture is read **left to
right** — the input boundary is on the left — and the matrix product is read **right to
left**: the row "$A$ then $B$" is the matrix $BA$. Everything is exact; nothing in Chapter 1
is up to a scalar. There is no cup, cap, transpose, dagger or conjugate here, so those
conventions have nothing to say.

**What the pictures are made of.** The chapter draws a wire, a box, a state box, a row of
boxes and the bare wire. PyZX has its own nodes, which is plumbing: NOT is one two-legged node
of phase $\pi$; a state box is a one-legged node with the stray $\sqrt2$ divided out; SET-0 and
SET-1, which the book draws as ordinary labelled boxes, are a node closing the incoming wire
and a node opening the outgoing one. Each file says so in a comment where its first node is
made. The shared plumbing lives in `tools/dfp.py`; the mathematics of each challenge is
written out in its own file.

The first three files fix one wire and the boxes that may sit on it — one box, the same box
twice, two different boxes in either order — with a definite bit as the passenger. The last
two keep every box and every picture unchanged and change only the passenger: first to a
division of one whole into two shares, then to an arrow of length one in $\mathbb{C}^2$. Two
turn-overs still come off.

**The pictures in the dialogue.** The dialogue draws its props at the moments that matter:
the Tortoise's string with its end tag, its tags and the traveller
(`\tagstring[place]{start}{tags}{caption}`) and the square of tracing paper with its matches
(`\arrowsquare[view]{matches}{caption}`), twenty pictures in eighteen blocks. The macros
live in `pdf/props/ch01.tex` and `ch01.lua`, and the Lua *computes* what a picture shows —
the coin's face, which end of the strip is at the string, how many matches lie in each row —
from the start as tied and the tags passed, so the dialogue never writes a turned result by
hand. Each file reads the calls out of its own `## <Ordinal> Challenge` section of
`chapter01/dialog.md` (never a copy: an edit that breaks a picture breaks the file), and in
`blackboard()` checks them against the chapter's state model, with the very $X$ and $S_0$ the
blackboard uses: heads is the first entry, `over` is NOT, the `heads` tag is SET-0; the strip
`strip-k` is $(k/6,(6-k)/6)$; the square is $(3/5,4/5)$ with reaches five times its entries.
The part of `ch01.lua` that decides what is drawn — `parse_start`, `parse_tags`, `travel`,
the place rule, the strip's panels and heads' share, the square's reaches and match counts —
is transcribed line for line, and first checked against the model on *every* start, every
string of up to four tags and every place, not only on the dialogue's own pictures. Then each
picture is checked against the scene: the end tag as tied, the tags on the string (and which
touch), where the traveller is, and the reading recorded for it on the identity card (§3b). The
files also check the card's layout rules: the only starred pictures are the first coin
picture, the first strip picture and the first close-up of the square; every caption is words
only; every block is a paragraph of its own (never inside a stage direction), holds at most
four pictures, and keeps a square-tied string to itself; `[off]` is never used; no picture
stands outside a challenge section; and every block appears identically in
`chapter01/dialog.raw.md`.

---

## 1. A Coin on a String — `Ch1/Flip.py`

Tortoise pins the string, ties on the end tag and hangs one "turn it over" tag; Achilles walks
the coin from the left end and it arrives the other way up, from either start. The file checks
that NOT exchanges the two definite states and is not the bare wire. In the sketchbook the
comparison is picture against picture: the row "state $0$, then NOT" *is* the picture "state
$1$", exactly. The negative asserts rule out the reading in which the tag does nothing — "state
$0$ then NOT" is not the picture "state $0$", and the NOT box is not the bare wire — which is
the whole content of Achilles's "well, of course it's tails up".

```python
assert np.allclose(X @ e0, e1) and np.allclose(X @ e1, e0)
assert not np.allclose(X, I2)
assert same_picture(state_then_not(0), state(1))
assert not same_picture(state_then_not(0), state(0))
assert not same_picture(not_box(), bare_wire())
```

Its two pictures are the two walks: from "heads up" past one tag, the coin is drawn tails (the
keyed picture), and from "tails up", heads. `Flip.py` also holds the keyed rule for the coin:
of all the dialogue's coin pictures, the first is starred and no other.

```python
first, other = check_coin_pictures("First", [1, 1], [
    ("heads", one, "end", True, "tails"),
    ("tails", one, "end", False, "heads"),
])
assert np.allclose(first, X @ HEADS) and np.allclose(other, X @ TAILS)
```

## 2. There and Back — `Ch1/TwiceOver.py`   *(The Theorem, Twice; the master diagram)*

"Two tags must do twice as much." They do not: the coin arrives as it left, from either start,
and Tortoise slides the pair together and lifts it off. On the blackboard this is one matrix
product, $XX=I$, and the negative assert is Achilles's guess itself — two turn-overs are not one.
The theorem is then run on passengers the picture never named: a random division of one whole,
a random complex unit vector asserted non-real, and both definite starts; each comes back
unchanged, while one tag alone moves both generic passengers.

In the sketchbook, two NOT boxes in a row are exactly the bare wire, one is not, and three are
one. The file also runs the chapter's own picture proof — both state boxes through the row —
and then checks the principle that closes it, that a box is fixed by what it does to the states
$0$ and $1$: those two pictures are precisely the two columns of the row's matrix.

```python
assert np.allclose(X @ X, I2)
assert not np.allclose(X @ X, X)                 # not "twice as much"
assert same_picture(row_of_nots(2), bare)
assert not same_picture(row_of_nots(1), bare)
assert np.allclose(two[:, bit], matrix_of(state_then_nots(bit, 2)).ravel())
```

Its four pictures: the coin past two tags from each start arrives as it left (and one tag
alone would have turned it — not "twice as much"); then one block of two, the pair slid
together until the tags touch (`over+over`, one group) and lifted off, the coin lying tails
where the last walk left it. A bare string with the coin at its end reads as its start.

```python
assert np.allclose(from_heads, HEADS) and np.allclose(from_tails, TAILS)
assert not np.allclose(carry(HEADS, ["over"]), from_heads)
assert np.allclose(slid, from_tails) and np.allclose(lifted, TAILS)
```

## 3. Which Tag First? — `Ch1/OrderMatters.py`

"Heads up, whatever it was" hung twice does not come off; hung beside a turn-over tag and then
exchanged with it, it sends the same coin to the other face. The file checks the two orders,
$S_0X=S_0$ and $XS_0=S_1$, and that they differ; that one twin comes off and never both
($S_0S_0=S_0\neq I$); that after SET-0 both starts arrive alike and nothing hung further along
brings the start back — for a random complex $B$, the two columns of $BS_0$ are equal — while the
turn-over tag loses nothing. All sixteen products of the four boxes are again among the four.
The left-to-right/right-to-left convention is checked on a random complex pair asserted not to
commute, so the negative carries weight.

In the sketchbook the rows are built left to right and compared as pictures, and each row's
matrix is checked against the product of its boxes' matrices in the reverse order.

```python
assert np.allclose(S0 @ X, S0) and np.allclose(X @ S0, S1)
assert not np.allclose(S0 @ X, X @ S0)
assert np.allclose((B @ S0)[:, 0], (B @ S0)[:, 1])
assert same_picture(row_of("X", "S0"), row_of("S0"))
assert not same_picture(row_of("X", "S0"), row_of("S0", "X"))
assert np.allclose(matrix_of(row_of(*tags)), then @ first)
```

Its four pictures: the twins from tails arrive heads, where a bare string gives tails; "as
hung" and "swapped" are the very same two tags in the other order and give the other face;
and swapped from tails arrives as swapped from heads did — both starts alike. The negative
assert is the mix-up the convention rules out: read from the right, the swapped string would
give heads, not the tails drawn (and over all strings of up to four tags, the two readings part
on 24 of the 62 walks of the 31 strings of up to four tags, from either start).

```python
assert hung[1]["tags"] == hung[2]["tags"][::-1]
assert face_of(carry(HEADS, hung[2]["tags"][::-1])) != hung[2]["now"]["face"]
assert np.allclose(p3, p4)                       # "Both starts arrive alike!"
```

## 4. The Covered Coin — `Ch1/HiddenCoin.py`

The coin is under a foot and the end tag becomes a strip: from the knot to the change of shade
is heads' share, the rest is tails'. Turning it end for end exchanges the shares, and the
five-and-one strip shows the tag at work where Achilles's half-and-half strip could not. The
file checks that NOT exchanges the shares of $(5/6,1/6)$ and of a random division, that the two
shares still make up the whole afterwards, and that each of the four boxes is column-stochastic
— which is exactly why they do. Then the trap: NOT leaves the fair half-and-half start where it
was, and NOT is still not the bare wire. One start is not enough to tell a tag from no tag.

The half-and-half start really is a picture here — one node with its two stray factors divided
out, denoting exactly $(1/2,1/2)$ — so the trap is checked picture against picture. A general
division such as five-and-one has no exact picture built from these nodes; the file says so in a
comment and checks the boxes instead, against the shares as NumPy columns.

```python
assert np.allclose(X @ FIVE_ONE, [1 / 6, 5 / 6])
assert np.isclose((X @ FIVE_ONE).sum(), 1)
assert np.allclose(M.sum(axis=0), [1, 1])        # every box is column-stochastic
assert same_picture(fair_then_nots(1), fair_coin())
assert not same_picture(not_box(), bare_wire())
```

Its five pictures are the two strips. Achilles's tossed coin (the keyed strip picture, on a
string still bare) reads three in six, and turned end for end at the tag it still reads three,
with its white end now at the string — the trap. The Tortoise's strip reads five, turned at the
tag it reads one ("It moved!"), and tied on again on a bare string it reads five, as it began.
`strip-three` and `strip-five` are asserted to be the blackboard's `FAIR` and `FIVE_ONE`.
The turned five-and-one strip still has five dark panels while heads' share is one: the shading
only marks the place.

```python
assert np.allclose(tossed_turned, tossed)
assert not np.allclose(die_turned, die) and np.allclose(die_turned, X @ FIVE_ONE)
assert sum(turned_panels) == 5 and not np.isclose(6 * die_turned[0], 5)
assert np.allclose(home, die)
```

## 5. The Arrow — `Ch1/Arrow.py`

Three matches heads-ward and four tails-ward, and the arrow is five long, not seven. The file
checks the two rules apart on that one example: the squares of $(3/5,4/5)$ make the whole, while
the entries sum to $7/5$, so it is an arrow and no wager. Turning the square over exchanges the
reaches and keeps the length; turning it twice is home. The same is then run on a random complex
unit vector, asserted non-real and with a non-real ratio of entries, so no symmetry can hide a
mistake. The two definite starts are legal under all three rules, and — as the chapter remarks
at the end of §1.5 — SET-0 keeps sums but not sums of squares, so not every classical box may be
hung on a wire that carries an arrow.

In the sketchbook the generic arrow is built as a picture: the state box $0$ followed by three
phase nodes, which gives length one, both reaches non-zero and a ratio that is not real. One
turn-over exchanges its two entries and gives a different picture; two turn-overs give the same
picture exactly. That is *The Round Trip* read with its third passenger. The three-four-five
arrow itself has no exact picture from these nodes, and is checked against the boxes' matrices.

```python
assert np.isclose(np.linalg.norm(ARROW), 1) and np.isclose(ARROW.sum().real, 7 / 5)
assert np.allclose(X @ ARROW, [4 / 5, 3 / 5])
assert not np.isclose(np.linalg.norm(S0 @ ARROW), 1)
assert not same_picture(arrow_then_nots(1), arrow_then_nots(0))
assert same_picture(arrow_then_nots(2), arrow_then_nots(0))
```

Its five pictures: three close-ups of the square — the keyed one with matches along the two
edges (three heads-ward, four tails-ward), then along the arrow too (five, not seven, and
nine and sixteen make twenty-five), then turned over (four and three, and still five) — and one
block of two with the square tied on, under two turn-over tags and with them lifted off, drawn
as drawn since nothing is walked. The views `home` and `twice` are checked to read as two
turns, `over` as one, against `X @ ARROW`.

```python
assert (c["along"], c["up"], c["arrow"]) == reading          # (3,4,5) or (4,3,5)
assert c["along"] + c["up"] == 7 != c["arrow"]
assert c["along"] ** 2 + c["up"] ** 2 == c["arrow"] ** 2
assert square_on_string(whole) == (3, 4)          # two tags come off for the arrow
```
