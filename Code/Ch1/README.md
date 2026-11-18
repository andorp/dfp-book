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

**The pictures in the dialogue.** The dialogue draws its props at the moments that matter,
with three macros from `pdf/props/ch01.tex`: the Tortoise's string with its end tag, its tags
and the traveller (`\tagstring[place;opts]{start}{tags}{caption}`), a **step strip** of one
walk along it (`\tagwalk[opts]{start}{tags}{caption; caption; ...}`: the start, then one frame
just past each tag, then with `to=end` the arrival), and the square of tracing paper with its
matches (`\arrowsquare[view;opts]{matches}{caption}`) — 25 calls, 30 frames, in seventeen
blocks, at most four moments to a challenge (2, 3, 4, 4, 4). The Lua (`ch01.lua`) *computes*
what a frame shows — the coin's face, which end of the strip is at the string, the square's
reaches and how many matches lie in each row — from the start as tied and the tags passed;
with `marks=on` it draws the step just made (a **ghost**, the traveller one tag earlier, and a
dotted hop under the tag, or for the square a ghost arrow, a swing and a pin ring), and none
where the tag changed nothing; with `expects=<rule>` it draws, pale, what a **named wrong rule**
predicts (`twice`, `twins`, `anyorder`, `idle`, `sum`), from the call's own start and tags. The
dialogue never writes a turned result by hand.

`tools/props_ch01.py` transcribes `ch01.lua`, all but the drawing: the option grammar, the
start and the tags, the walk, the marks, the five rules (`twins` with its cascade of pairs,
its lifted set, the place read on the string as hung, and its fault for a place inside a
pair), the step strip's frames, the close-up, every fault the macro raises (recorded in
`TEX_ERRORS`; `picture()` insists there are none) and the log dump. Each file reads its own
`## <Ordinal> Challenge` section of `chapter01/dialog.md` through `tools/dfp_props.Dialogue`
(never a copy: an edit that breaks a picture breaks the file), and checks **every frame**
against the chapter's state model, written out in the file with the very $X$ and $S_0$ the
blackboard uses: heads is the first entry, `over` is NOT, the `heads` tag is SET-0; the strip
`strip-k` is $(k/6,(6-k)/6)$ and which end is held; the square is $(3/5,4/5)$ with reaches five
times its entries. Each ghost is the model one tag earlier, there exactly when the tag moved
the traveller. Each prediction is three things at once: the named rule of `props_ch01`, the
file's own model of what Achilles says, and *not* the true frame beside it at the same place.
The transcription is first checked against the model on every start, every string of up to
four tags, every place, with and without marks, and on every step strip of such a walk — not
only on the dialogue's own pictures; each file also checks the log dumps the card records for
its section (identity.md §3e.2). Then the card's layout rules: the only starred pictures are
the first of the coin (the first frame of the First's step strip), of the strip and of the
square close up; every caption, and each caption of a step strip, is words only; every block
is a paragraph of its own, holds at most three string frames or two squares to a line, and a
line of three frames made of two calls is joined by a bare `%` line end; the place `off` is
never used; each block stands after the line it pictures (its anchor phrases are asserted);
and every block appears identically in `chapter01/dialog.raw.md`.

**Pins.** `props_ch01.check_lua()` (run when the module is imported, unless `DFP_REPIN` is set
to re-pin, and again at the head of every file's picture checks) holds `pdf/props/ch01.lua`
and `ch01.tex` to the text the transcription was proved against with texlua: `LUA_PINS`
hashes the Lua's 40 top-level chunks (comments left out), `TEX_PINS` the `.tex`'s 31 colours,
styles and macros, and `COMMENT_PINS` the Lua's two comments that quote the dialogue, word for
word, each quote still in `dialog.md`. A drawing-only or style-only edit then fails all five
files, naming the chunk, until the transcription is proved again and re-pinned
(`print(P.lua_pins())` with `DFP_REPIN=1`; the module's docstring has the line); a plain
comment edit passes.

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

Its pictures are two moments. The evening's first walk is a keyed step strip of three frames
— heads beside the end tag; just past the tag, tails, with the ghost of the heads coin just
before it; arrived, tails — and the same walk from "tails up" is one picture, heads. There is
no prediction: Achilles is right. The file also runs every step strip of every coin walk of up
to four tags, with and without a start frame and an arrival, and checks that a caption short
or one too many is a fault.

```python
assert np.allclose(check_coin(start, False), HEADS)            # from heads up
assert np.allclose(check_coin(turned, True), X @ HEADS)       # turned at the tag ...
assert turned.ghost["face"] == "heads"                       # ... from heads, the ghost
assert np.allclose(check_coin(arrived, False), TAILS)          # carried on: tails
assert start.keyed and start.labels == ("the end tag", "a tag")
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

Its pictures are three moments. The first two-tag walk is a step strip of two marked frames —
tails just past the first tag (ghost heads), heads just past the second (ghost tails) — with
the prediction beside it, pale, at the same place: `twice`, "Very... thoroughly tails", the
coin the other way up from its start once it has been turned at all. The rule is checked
against the file's model of what he says on every row of up to four turn-overs, and it parts
from the truth from the second tag on. Then the other start, and one block of two: the pair
slid together (`over+over`, one group) and lifted off, the coin lying tails where the last walk
left it.

```python
assert np.allclose(check_coin(second, True), X @ X @ HEADS) and second.ghost["face"] == "tails"
assert guess.where == second.where and guess.passed == second.passed == 2
assert guess.shown["face"] == face_of(twice_as_much(HEADS, guess.passed)) == "tails"
assert guess.shown != second.shown
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

Its pictures are four moments. The first walk past the "heads up, whatever it was" twins is a
step strip of two frames: heads just past the first (ghost tails), and heads just past the
second with **no marks**, for there the tag changed nothing ("at the second it is heads up
already, and he leaves it"). Beside it, pale, `twins`: "Twins. Off they come." — the lifted set
is exactly the two twins, the string is bare where they hung, and the coin, read at the same
place, is tails. The file's own model lifts pairs as the Tortoise would (the leftmost two of a
kind side by side, and look again) and agrees with the rule, its lifted set and its fault for a
place inside a pair on every string of up to four tags. Then "as hung", and the swapped string
with the prediction `anyorder` beside the true frame: the tags walked in the other order give
heads, the page gives tails. That wrong rule is exactly the mix-up the convention rules out —
the string read from the right — and over the 28 strings of two to four tags, from either
start, it parts from the truth on 24 of the 56 walks. Last, swapped from tails arrives as
swapped from heads did: both starts alike.

```python
assert second.marks and second.ghost is None and np.allclose(p1, S0 @ S0 @ TAILS)
assert twins.lifted == {1, 2} and twins.hanging == ()
assert twins.shown["face"] == face_of(guess) == "tails" != second.shown["face"]
assert P.faults(P.tagstring, "first;expects=twins", "tails", "heads,heads") == [
    "tagstring: expects=twins: the place falls inside a pair that comes off "
    "(past tag 1, not past tag 2)"]
assert anyorder.shown["face"] == face_of(guess) == face_of(p2) == "heads"
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

Its pictures are four moments of the two strips. Achilles's tossed coin, the keyed strip
picture on a string still bare, reads three in six. Turned end for end at the tag, it still
reads three, with its white end now at the string, and the ghost of the strip as tied is
drawn: the strip did move, though the division reads the same — the trap, and the tag at work
in it. The file's strip model is therefore two facts, the shaded panels and the end held,
and a ghost is there exactly when the end held changes. The Tortoise's strip is a step strip —
tied on, five in six; just past the tag, one in six, the white end at the string, the ghost
five-dark ("It moved!") — with the prediction `idle` beside it, pale, at the same place: "this
tag does nothing", the strip as tied, five in six. Last, tied on again on a bare string, it
reads five, as it began. `strip-three` and `strip-five` are asserted to be the blackboard's
`FAIR` and `FIVE_ONE`; the turned five-and-one strip still has five dark panels while heads'
share is one: the shading only marks the place.

```python
assert np.allclose(tossed_turned, tossed) and np.allclose(X @ FAIR, FAIR)
assert turned_pic.ghost == {"kind": "strip", "dark": 3, "turned": False}
assert not np.allclose(die_turned, die) and np.allclose(die_turned, X @ FIVE_ONE)
assert idle.where == die_turned_pic.where and idle.passed == die_turned_pic.passed == 1
assert idle.shown != die_turned_pic.shown
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

Its pictures are four moments. The keyed close-up has matches along the two edges (three
heads-ward, four tails-ward). Then the measure, five, with the prediction `sum` beside it,
pale: the reaches added, seven matches along the arrow, a row that runs past the tip and off
the paper by a clear gap. Then the turn, two frames: the square turned over and cleared of
matches, with the ghost arrow where it lay (three and four), its swing and its pin; then
measured once more, four and three, and still five. Last, the square tied on under two
turn-over tags and with them lifted off, drawn as tied, since nothing is walked. The views
`home` and `twice` are checked to read as two turns and `over` as one, against `X @ ARROW`,
on every choice of rows, with and without the turn's marks.

```python
assert five.count == (3, 4, 5) and not five.sum
assert five.count[0] ** 2 + five.count[1] ** 2 == five.count[2] ** 2
assert seven.count == (3, 4, 7) and seven.count[2] == as_achilles_counts(ARROW)
assert seven.far_end - P.SIDE > 0.1                     # off the paper
assert turn.ghost == (3, 4) and turn.reaches == (4, 3)
assert again.count == (4, 3, 5)
```
