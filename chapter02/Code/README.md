# Chapter 2 — Amplitudes: the companion code

Six challenges in the dialogue, six files here. Each file checks its claim twice, as the
chapter proves its theorem twice: `blackboard()` with NumPy columns and rows, `sketchbook()`
with pictures built in PyZX and compared as exact tensors — every number included, because no
equation in this chapter holds only up to a scalar.

Run one file, or the whole chapter, from the repository root:

```bash
.venv/bin/python chapter02/Code/Ch2/Closing.py
bash check-code.sh 2          # must end with: 6 files, 0 failed
```

Each file prints one line per strand and exits non-zero if any `assert` fails. Running the
files is this book's analogue of type-checking.

**Conventions, asserted where they are used.** The first (top) entry of a column belongs to
the state $0$. A picture is read **left to right**, so the effect is on the right; a bracket is
a matrix product read **right to left**, so the bra is on the left — `Closing.py` asserts that
the picture "state then effect" and the product "row times column" are the same number. The
mirror image of a state **conjugates**: every file that tests it uses a state with non-real
entries, so that the plain transpose gives a different number and can be ruled out by a
negative assert. Cups, caps and their normalization have nothing to say in this chapter.

**What the pictures are made of.** The chapter draws a state box, a wire, an effect box and
rows of these. PyZX has its own nodes, which is plumbing; each file says so in a comment where
its first node is made. The states $0$ and $1$ are one-legged nodes of one kind, the states $+$
and $-$ one-legged nodes of the other, with PyZX's stray $\sqrt2$ divided out so that the
columns are exact. The generic state — the one that carries the weight of every test — is the
state $0$ followed by a chain of three different phases: its entries are both non-zero and
neither is real. Its mirror image is that chain reversed **with every phase negated**; reversed
without negating, it is the plain transpose, and that counterfeit appears in `Closing.py` and
`WholeSquare.py` as a negative assert. An effect is obtained either that way or by PyZX's
`adjoint()`, its name for the mirror image of a picture; the two are checked against each other.
Two pictures in a row are `g1 + g2`. A closed diagram has no inputs and no outputs, and its
matrix is $1\times1$.

Two things the files are careful about. A zero is never checked by comparing two pictures —
two pictures that both denote zero agree vacuously — so every zero is checked with NumPy and
paired with a non-zero check that the same comparison does distinguish. And the arrows of the
dialogue, $(3/5,4/5)$ and its non-real cousin $(3/5,4i/5)$, have no exact picture built from
these nodes; they are checked on the blackboard only, and each file says so in a comment.

**The pictures in the dialogue.** The dialogue draws its props at the moments that matter —
twenty pictures in nineteen blocks — with five macros: the pale and the dark string with their
start and closing tags (`\echostrings{pale}{dark}{caption}`), the pale string's reading close up,
cross-sheet, card and matches (`\crosssheet{arrow}{position}{read}{caption}`), the dark string's
reading, start laid over closing tail on tail (`\tailontail{start}{closing}{caption}`), a strip
(`\sharestrip{share}{caption}`) and the five-by-five outline (`\fivesquare{laid}{caption}`). The
macros live in `pdf/props/ch02.tex` and `ch02.lua`, and the Lua *computes* what a picture shows:
the dark string's tags under `echo`, where each tag hangs and by which corner it is knotted,
every reach with its sign, where the card's foot falls, the whole matches and the snapped piece,
where the little squares lie, how many parts of a strip are shaded, how the outline fills. Each
file reads the calls out of its own `## <Ordinal> Challenge` section of `chapter02/dialog.md`
(never a copy: an edit that breaks a picture breaks the file), and in `blackboard()` checks
them against the chapter's state model, built from the very columns the blackboard uses: an
arrow is a real state times its length in matches (the three-and-four arrow is $5T$, heads-ward
and tails-ward $5\ket0$ and $5\ket1$, the slant and the back-slant $\sqrt2\ket+$ and
$\sqrt2\ket-$, the arrow reversed $-5T$); a line of the sheet is an answer (square: $\ket0$,
$\ket1$; turned: $\ket-$ on line A, where heads-ward swung to, and $\ket+$ on line B; Achilles's
own position the square sheet swung $20^\circ$; along: the arrow's own line). A reach is the
amplitude times the length, sign included, and its square over the length squared is the Born
value.

The part of `ch02.lua` that decides what is drawn is transcribed (everything but the drawing),
and its faults — an unknown word, a line read under the wrong position, squares on a reach that
is not whole, halves round an arrow whose reaches are not one match each, `echo` without both
pale tags — are recorded as the Lua's `tex.error` records them. The transcription is first
checked against the model on *every* arrow word, turned end for end or not, under every position
of the sheet and every reading, not only on the dialogue's own pictures: every reach is
$\langle\text{line}|\text{arrow}\rangle$; the card's foot is the perpendicular from the tip; the
matches add up to the reach and lie its way; the squares number reach$^2$ and lie on the side of
their reach away from the arrow, behind the crossing when it is backwards; the halves close into
area $2$; a fault arises exactly where the model says the paper cannot be laid; an `echo` is the
pale pair with its ends exchanged, and the dark string's number (start over closing) is the pale
string's, sign included; the outline never gains or loses a square. Then each picture is checked
against the scene at its moment and against the reading recorded for it on the identity card
(§3b). The files also check the card's layout rules: the only starred pictures are the first
of each prop or of each labelling (the strings, the square sheet, the start over closing, the
strip, the turned sheet, the outline), and no other; every caption is words only and never
counts or names the strings in the plural; no argument holds a digit; every block is a paragraph
of its own (never inside a stage direction), holds at most four pictures and at most one pair of
strings; the optional strip beside the start tag is never drawn; no picture stands outside a
challenge section; and every block appears identically in `chapter02/dialog.raw.md`.

The first two files close a wire and find first a number and then, from the number and its
mirror twin, a chance. The next three put a minus sign where one question cannot see it, where
another can, and where none can. The last closes a state with its own mirror image and finds
the whole.

---

## 1. The Right-Hand End — `Ch2/Closing.py`

Tortoise carries a start tag to the far end of the pale string and ties it on with the knot
facing back; the cross-sheet, the card and the matches give one number, three parts in five.
The file builds the bra as the conjugated row, checks that the entries of a column are its
amplitudes for $0$ and for $1$, and that exchanging the two labels conjugates the bracket. The
chapter's own example does the ruling out: $\ket\chi=(3/5,4i/5)$ meets itself in $1$ through the
conjugated row and in $-7/25$ through the plain one.

```python
assert np.isclose(np.vdot(CHI, CHI), 1)          # conjugated row: one
assert np.isclose(CHI @ CHI, -7 / 25)            # plain row: minus seven in twenty-five
```

In the sketchbook the effect is built twice — by `adjoint()` and by hand — and the
un-negated chain is rejected, which is the picture's form of the same ruling out. The closed
diagram is confirmed to have no open end at all, and the left/right convention is asserted
outright.

```python
assert same_picture(effect_by_hand(True), mirror)
assert not same_picture(effect_by_hand(False), mirror)
assert closed.inputs() == () and closed.outputs() == ()
assert np.allclose(matrix_of(psi + e), matrix_of(e) @ matrix_of(psi))
```

Its two pictures, both keyed. The strings: last evening's arrow at the pale string's left end,
the heads-ward square carried to the right-hand end as it lies and knotted by its far
right-hand corner, so that it hangs on the near side; the dark string bare. The mirror gesture
is asserted as the macro computes it — the closing square is the start square turned half round
about the string's midpoint, while its arrow keeps its heading and its place in its own square.
Then the cross-sheet, square, its heads-ward line along the closing tag's arrow: three whole
matches, the card's foot at three, the arrow five.

```python
check_mirror_gesture(es, "pale")
assert np.isclose(number, amplitude_reach(5 * T, ket(0))) and np.isclose(number, 3)
assert np.allclose(cs["e"]["A"], closing["u"])                    # the line the question asks
assert laid(cs["rows"]["A"]) == (3, 0, 1) and set(cs["rows"]) == {"A"}   # three whole, forward
```

## 2. Why Squares — `Ch2/Squares.py`   *(the master diagram, first drawn)*

Achilles lays three by three on one reach and four by four on the other: nine and sixteen,
twenty-five in all. The file first rules out his earlier guess — the reaches themselves
overshoot the whole, $3/5+4/5=7/5$ — and then checks that the amplitude times its mirror twin
is real, never negative, and sums to one: a covered coin in the exact sense of Chapter 1. The
plain square, without the conjugate, is not even real for a generic state.

```python
assert np.isclose(reaches.sum(), 7 / 5) and not np.isclose(reaches.sum(), 1)
assert np.isclose(p, abs(amp) ** 2)
assert not np.isclose((amp1 ** 2).imag, 0)
```

The sketchbook builds *The Echo* — state $\psi$, effect $b$, then a gap, then state $b$, effect
$\psi$ — and checks that its number is the product of the two halves, that the right half
really is the left half mirrored, and that the two numbers of a question add to one. That last
line adds numbers, not pictures, and says so. The dialogue's dark string is asserted here as
the card asks: for a real state the two halves are equal, sign and all, while for the generic
state they are conjugates and differ.

```python
assert np.isclose(value, left * right)
assert same_picture((psi + b.adjoint()).adjoint(), b + psi.adjoint())
assert np.isclose(left, -1 / np.sqrt(2)) and np.isclose(right, left)
assert np.isclose(gr, np.conj(gl)) and not np.isclose(gr, gl)
```

Its four pictures. The dark string tagged by `echo` (never written out): the pale pair with its
ends exchanged, each string's tags obeying the mirror gesture, the dark number equal to the pale
one, three. The dark string's reading (keyed): the heads-ward start over the three-and-four
closing — the dark string's own two tags — three matches forward. The little squares: nine and
sixteen, each block on the side of its reach away from the arrow, the two blocks disjoint, no
card once paper is laid. The strip (keyed): nine of twenty-five parts shaded from the knot, the
Born value of $T$ for $0$ and not three in five.

```python
assert same_arrow(tags["dark start"], 5 * ket(0)) and same_arrow(tags["dark closing"], 5 * T)
assert np.isclose(tt["number"], 3) and np.isclose(tt["number"], pale)   # identity 3b: 3
assert (len(nine), len(sixteen)) == (9, 16) and len(nine) + len(sixteen) == 25
assert np.isclose(float(heads), abs(np.vdot(ket(0), T)) ** 2)     # the Born value, 9/25
```

## 3. A Reach Backwards — `Ch2/HiddenSign.py`

Achilles ties one strip on for both arrows: same strip, same start. The file confirms the half
of his conclusion that is true — $+$ and $-$ give the identical wager under the standard
question, and so do $\alpha\ket0\pm\beta\ket1$ for a generic complex state — and then the part
that undoes it: $\braket{+|-}=0$, and the two are different columns. The sign is in the
amplitude the whole time; only the square eats it.

```python
assert np.isclose(born(s, ket(0)), 1 / 2)
assert np.isclose(np.vdot(PLUS, MINUS), 0)
assert np.isclose(np.vdot(ket(1), flipped), -np.vdot(ket(1), psi))
```

In the sketchbook the two *Echo* pictures are the same picture, exactly, while the two state
pictures are not. The zero appears here for the first time, and is checked with NumPy and given
a non-zero partner, so that nothing passes vacuously.

```python
assert same_picture(echo(plus, b), echo(minus, b))
assert not same_picture(plus, minus)
assert np.allclose(matrix_of(zero_diagram), 0)
assert np.isclose(matrix_of(one_diagram)[0, 0], 1)
```

Its three pictures. The slant under the square sheet: one match each way, and the four
half-squares closed into the square on the slant, area two, on its tails-ward side — nothing
over. The back-slant, its tail at the far left-hand corner: heads-ward one, tails-ward one
*backwards*, the card's foot behind the crossing. The back-slant again with a little square on
each reach, the tails-ward one behind the crossing: the shares are one in two, exactly the
slant's, although one reach has changed its sign — the trap's evidence, asserted both ways.

```python
assert np.isclose(area, cs["reach"]["A"] ** 2 + cs["reach"]["B"] ** 2)   # made of those two
assert b["card"]["line"] == "B" and b["card"]["out"] == -1         # the card at the tails-ward line
assert np.isclose(cs["reach"]["B"], -s["reach"]["B"])
assert np.allclose(shares, [abs(np.vdot(ket(k), PLUS)) ** 2 for k in (0, 1)])
```

## 4. The Turned Question — `Ch2/TurnedQuestion.py`

The cross-sheet turns through half a right angle and the two starts that one question could not
tell apart are told apart every time. The file checks that $\{+,-\}$ is an orthonormal basis by
its Gram matrix, that the expansion of a generic state in it is exact, and that the arrow gives
$49/50$ and $1/50$ — the sliver coming from an amplitude that reaches backwards. The
chapter's non-real state is here too, and shows what only a turned question can: it matches the
arrow under the standard question and parts from it under the turned one.

```python
assert np.allclose(gram, np.eye(2))
assert np.allclose(rebuilt, psi)
assert np.isclose(np.vdot(MINUS, T), -1 / (5 * np.sqrt(2)))
assert np.isclose(born(CHI, PLUS), 1 / 2)
assert not np.isclose(born(CHI, PLUS), born(T, PLUS))
```

In the sketchbook the certainties come out as $1$ and $0$ — the zeros with NumPy again, and
with a picture comparison as their non-zero partner. The file also asserts that no wire in any
of its pictures carries a mark of its own: $+$ and $-$ are states here, and the box that turns
one question into the other belongs to Chapter 3 and is built nowhere.

```python
assert plain_wires(g)
assert np.isclose(value, abs(np.vdot(vec, col)) ** 2)
assert not same_picture(echo(plus, minus), echo(plus, plus))
```

Its four pictures, in three blocks. The turned sheet (keyed, with its own line names): the
back-slant asked along the slant's line has no reach — the card's foot on the crossing, no
matches — and lies wholly along the other. The file asserts the sense of the turn: line A is
$\ket-$, the heads-ward line come round toward the two of them; swung the other way the
back-slant's whole arrow would lie backwards along its line, contrary to the dialogue. The two
strips: under the turned sheet the share from the knot is line A's, so the back-slant's strip
is all dark and the slant's all white — $|\langle -|-\rangle|^2$ and $|\langle -|+\rangle|^2$.
Last, the three-and-four arrow under the turned sheet: just short of five along the slant's line
(four matches and a nearly whole piece), short of one match backwards along the back-slant's.

```python
assert np.isclose(cs["theta"], -45) and cs["e"]["A"][1] < 0
assert np.isclose(abs(BACK.real @ other), cs["arrow"]["L"]) and BACK.real @ other < 0
assert shares == [1, 0]                                           # all dark, all white
assert laid(t["rows"]["B"]) == (4, 0.95, 1)                        # four and a nearly whole piece
assert laid(t["rows"]["A"]) == (0, 0.71, -1)                       # a piece behind the crossing
```

## 5. The Arrow Reversed — `Ch2/Reversed.py`

Tortoise turns the square end for end and asks for the question that knows it. The file starts
with the dialogue's own case — every amplitude of $-T$ changes sign, every chance stays — and
then does the general one: a phase on the whole column leaves the Born value alone for the four
effects of the chapter and for a hundred drawn at random. The contrast is the point of the
chapter, so it is asserted: a sign on one entry alone *is* caught.

```python
assert np.isclose(np.vdot(MINUS, rev), +1 / (5 * np.sqrt(2)))
assert np.isclose(born(primed, b), born(psi, b))
assert not np.isclose(born(flipped, PLUS), born(psi, PLUS))
```

In the sketchbook the two state pictures are compared with every number kept, and they are
different; their *Echo* pictures are the same. The halves show where the factor goes — the call
gains $e^{i\theta}$ and the echo gives back $e^{-i\theta}$. Half a turn is the reversed arrow
exactly.

```python
assert not same_picture(primed, psi)
assert same_picture(echo(primed, b), echo(psi, b))
assert np.allclose(matrix_of(with_phase(psi, Fraction(1))).ravel(), -col)
```

Its three pictures, all of the arrow reversed — the square turned end for end, its tail at what
had been the far corner. Square sheet: three and four backwards, and the nine and the sixteen
built again behind the crossing, the old blocks turned round. Turned sheet: just short of five
backwards, the sliver forwards now, every square as before. Achilles's own position, the sheet
swung twenty degrees with the heads-ward line tipped away from them: $-4.19$ and $-2.73$, and
the both-ways row for the square turned back, $+4.19$ and $+2.73$ — the same lengths, each on
the other side of the crossing.

```python
assert same_arrow(a, REV) and a["corner"] == "fr"
assert {tuple(np.round(-c, 6)) for c in t["squares"][l]} == \
       {tuple(np.round(c, 6)) for c in sq["squares"][l]}     # the same block, turned round
assert laid(ow["rows"]["A"]) == (4, 0.19, -1) and laid(ow["rows"]["B"]) == (2, 0.73, -1)
assert laid(ow["back_rows"]["A"]) == (4, 0.19, 1) and laid(ow["back_rows"]["B"]) == (2, 0.73, 1)
```

## 6. The Whole Square — `Ch2/WholeSquare.py`   *(The Theorem, Twice)*

Achilles takes the nine and the sixteen apart and lays them into the five-by-five: it fills.
The file proves the theorem for a generic complex state and rules out the unconjugated version
in the same breath; then the corollary, for the standard question, the turned one, and a
question drawn at random by taking the QR factor of a random complex matrix — checked to be
orthonormal and genuinely complex, so that it is no easier than the others.

```python
assert np.isclose(np.vdot(psi, psi), 1)
assert not np.isclose(psi @ psi, 1)
assert np.allclose(random_q.conj().T @ random_q, np.eye(2))
assert np.isclose(total, 1)
```

In the sketchbook the closed diagram has no open end, its matrix is $[[1]]$, and it agrees with
a picture that has no vertices at all — used only as the number $1$, with the counterfeit
closing as its partner so that the comparison is shown to discriminate. The corollary is then
arithmetic on evaluated pictures, borrowed from the blackboard and labelled as such.

```python
assert np.allclose(matrix_of(closed), [[1]])
assert same_picture(closed, number_one)
assert not same_picture(wrong, number_one)
total = sum(matrix_of(echo(psi, b))[0, 0] for b in question)
assert np.isclose(total, whole) and np.isclose(total, 1)
```

Its four pictures. The sheet turned until a line lies along the arrow: the card's foot on the
arrow's own tip, five in five, and no reach across. The arrow at both ends of the pale string
and, by `echo`, of the dark: both numbers five. The outline (keyed), empty, with the
three-by-three and the four-by-four beside it — the very blocks laid in the Second Challenge,
read from that section's picture. Then the outline filled: twenty-five squares on twenty-five
cells, nothing left of either block, the square on the arrow asked about itself.

```python
assert np.allclose(cs["card"]["foot"], cs["arrow"]["v"])           # on the arrow's own tip
assert (len(then["squares"]["A"]), len(then["squares"]["B"])) == (f0["left9"], f0["left16"])
assert sorted(f1["laid"]) == [(x, y) for x in range(5) for y in range(5)]
assert np.isclose(len(f1["laid"]), cs["reach"]["A"] ** 2)          # the square on the arrow
```
