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

**The pictures in the dialogue.** The dialogue draws its props at the moments that matter, and
shows a process as a strip of frames: 27 calls in 21 blocks, 31 frames, four of the blocks step strips. There are six macros.
`\echostrings{pale}{dark}{caption}` draws the pale and the dark string with their start and
closing tags. `\crosssheet[opts]{arrow}{position}{read}{caption}` draws the pale string's
reading close up: cross-sheet, card and matches. `\crosssteps[opts]{arrow}{position}{read}{steps}{caption; ...}`
is a step strip of that reading, one frame per step. `\tailontail[show]{start}{closing}{caption}`
draws the dark string's reading, the start laid over the closing tail on tail. Then there are
the strip, `\sharestrip[opts]{share}{caption}`, and the five-by-five outline,
`\fivesquare[opts]{laid}{caption}`. The macros live in `pdf/props/ch02.tex` and `ch02.lua`. The
Lua *computes* what a picture shows:

- the dark string's tags under `echo`, and where each tag hangs;
- every reach with its sign, where the card's foot falls, the matches and the snapped piece;
- the little squares, the cut, and the halves closed round the slant;
- the shaded parts of a strip, and how the outline fills;
- each frame of a step strip, computed from the frame before;
- the motion marks of a step:
  - the swung sheet's old lines and their arcs (`swung=`);
  - the square turned end for end, its old arrow dotted (`endforend=on`);
  - the halves that moved, ghosted where they lay (`marks=on`, which the `close` step sets);
  - the squares taken from their blocks (`from=`);
- and, drawn pale, what a named wrong rule predicts (`expects=coin|sum|bookkeeping` on the
  sheet, `expects=reaches` on the strip). A prediction stands level beside the true picture
  with `deep=on`, which is allowed on the prediction strip and on no keyed strip.

`tools/props_ch02.py` is a transcription of `ch02.lua`: everything but the drawing. It covers
every `tex.error` site (47 of them, listed in `TEX_SITES`) and every `\typeout` dump; a call's
faults are recorded in `TEX_ERRORS`, in the Lua's order and with the Lua's duplicates.
It was proved against the Lua itself, and proved again when the pins below were added. A
driver ran `ch02.lua` under `texlua` with `tex` stubbed. It made 61,911 calls: the 54 dialogue
calls of `dialog.md` and `dialog.raw.md`, 57 aimed calls that between them raise all 47 sites,
and 61,800 random calls over five seeds. Those calls expanded 27,622 step-strip frames, and the
driver also ran 43,775 probes of the Lua's exported model (rules, positions, moves, frames,
blocks). The Python agreed byte for byte in every error message, dump line, keyed label and
frame expansion. It also agreed on the counts of what each picture draws, and on the geometry
that no dump carries: the card's foot, the end-for-end ghost and its half-turn, the swing arcs,
the halves' travel and the outline's ghosts. Every site was raised, and the Lua raised no Lua
error. The 35 dump lines of the real chapter build log are byte-identical to the
transcription's.

**The pins.** Each file calls `props_ch02.check_lua()` at the start of `pictures()`. It holds `ch02.lua` and `ch02.tex` to the text the
transcription was proved against, by hash:

- `LUA_PINS` covers the Lua's 66 top-level chunks, with comments left out and white space
  closed up outside strings;
- `COMMENT_PINS` covers the Lua's two comments that quote the dialogue (the wrong rules coin,
  sum, bookkeeping and reaches);
- `TEX_PINS` covers the 45 chunks of the `.tex`.

A drawing-only change, a style-only change or a reworded quote fails all six files, and the
failure names the chunk. Any other comment may be reworded freely. After a prop edit, prove the
transcription again, then renew the pins with `lua_pins()`, `comment_pins()` and `tex_pins()`.
The check does not run at import, so the module still loads for that.

Each file reads the calls out of its own `## <Ordinal> Challenge`
section through `tools/dfp_props.py`: `Dialogue(__file__, macros=(six))`. It never reads a
copy, so an edit that breaks a picture breaks the file. `check_dialogue` checks the card's
layout rules once for the whole dialogue:

- the key goes on the first picture of each kind (`keyed_first`). Predictions, and pictures with
  the sheet off, along the arrow or in Achilles's own position, are left out of that count;
- exactly six pictures are starred, the six of `identity.md` §3e.2;
- every block sits in its own paragraph and appears identically in `dialog.raw.md`;
- no picture stands outside a challenge section.

`check_section` then checks each section:

- at most four moments, and each block has the expected number of calls and frames;
- at most four frames in a row, and at most one pair of strings per block;
- no digit in any argument, and the optional strip beside the start tag is never drawn;
- every caption, and every caption of a strip separated by `;`, is words only, without
  "strings", "wires", "wire", "both" or "pair";
- in both dialogue files, no caption repeats a two-word phrase of the paragraph that follows its
  block.

Each file keeps its own state model and checks every picture of its section against it. The
model is built from the very columns the blackboard uses. An arrow is a real state times its
length in matches: the three-and-four arrow is $5T$, heads-ward and tails-ward are $5\ket0$ and
$5\ket1$, the slant and the back-slant are $\sqrt2\ket+$ and $\sqrt2\ket-$, and the arrow
reversed is $-5T$. A line of the sheet is an answer:

- square: $\ket0$, $\ket1$;
- turned: $\ket-$ on line A, where heads-ward swung to, and $\ket+$ on line B;
- Achilles's own: the square sheet swung $20^\circ$;
- along: the arrow's own line.

A reach is the amplitude times the length, sign included, and its square over the length
squared is the Born value. The model also gives:

- a step's marks: the swing's signed angle the short way round, the end-for-end turn $a\mapsto-a$
  about the square's centre, and the least-travel assignment of the halves by brute force;
- each named wrong rule, written out from Achilles's own words, so that each prediction equals
  the rule *and* differs from the true frame.

The files first check the transcription against the model on every arrow word, under every
position and reading, and not only on the dialogue's own pictures. Then each frame is checked
against the scene at its moment and against the reading on the identity card (§3b). The stage
directions a frame stands for are checked too, in both dialogue files.

Mutation tests ran on a scratch copy of the chapter, each mutation applied to both dialogue
files, and all 34 were caught. They included:

- a step dropped from `\crosssteps`;
- `swung=` or `endforend=on` removed;
- an `expects=` rule swapped;
- `deep=on` removed from block 2.4, or moved onto its starred strip;
- `[laid]` changed to `[card]`;
- a caption removed from a strip;
- a block caption made to echo its next paragraph;
- a star moved;
- a wrong arrow or position;
- `from=` broken;
- a share changed;
- the prose of a step changed in one file only.

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

Its two moments, both keyed. The strings: last evening's arrow at the pale string's left end,
and the heads-ward square carried to the right-hand end as it lies and knotted by its far
right-hand corner, so that it hangs on the near side. The dark string is bare. The mirror
gesture is asserted as the macro computes it: the closing square is the start square turned
half round about the string's midpoint, while its arrow keeps its heading and its place in its
own square. Then comes the evening's first reading, as a three-frame strip:

1. the cross-sheet laid square, its heads-ward line along the closing tag's arrow, with the key
   on this frame only;
2. the card dropped from the tip, its foot at three;
3. three whole matches.

Each frame is the model after that many steps. The last frame is the plain reading of the
sheet, dump for dump. A sweep runs the same three-step strip on every arrow, position and line.
It also checks the coin rule, which the dialogue does not draw: the whole arrow handed to the
line, and not three.

```python
states = model_reading(5 * T, ket(0), ["card", "matches"])
for (pic, _), state in zip(steps, states):
    check_frame(pic, 5 * T, lines, "A", state)
assert [p.keyed for p, _ in steps] == [True, False, False]        # the key on the first frame
assert laid(read_.rows["A"]) == (3, 0, 1) and set(read_.rows) == {"A"}   # three whole, forward
assert (old.dump, old.rows, old.card.line) == (read_.dump, read_.rows, read_.card.line)
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

Its four moments.

1. The dark string, tagged by `echo` (never written out): the pale pair with its ends exchanged,
   each string's tags obeying the mirror gesture.
2. The dark string's reading as three frames, the key on the first: the start laid tail on tail,
   the card dropped, three matches. These are the dark string's own two tags, at each stage of
   the model, in the order she made them.
3. The little squares, one picture: nine and sixteen laid on the bare reaches, each block on the
   side of its reach away from the arrow. No card stands on this sheet, and the matches are those
   of the bare reading: only paper has been added.
4. Achilles's prediction beside the true strip. Drawn pale and level (`deep=on`), it applies his
   rule that each reach is its own share: the strip folded in five, three shaded, the tails' four
   running two past the paper's end. That is exactly his rule, and refuted by the keyed strip,
   which has nine of twenty-five parts shaded: the Born value of $T$ for $0$, not three in five.

```python
assert shows == ["laid", "card", "matches"]                         # in the order she did it
assert cs.read == {"A", "B", "squares"} and cs.card is None        # no card stands here
assert bare.rows == cs.rows                                        # the matches do not move
assert (p.folded, p.heads, p.tails, p.parts, p.past_end) == (5, 3, 4, 7, 2)
assert expected.deep is True and ss.deep is None                  # level beside the key
assert Fraction(p.heads, p.folded) != heads and p.parts > p.folded  # the prediction is refuted
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

Its three moments.

1. A three-frame strip of the slant under the square sheet, one match each way:
   1. a little square on each reach;
   2. each square cut from the crossing to its far corner, where it lies;
   3. the four halves closed into the square on the slant, on its tails-ward side, area two,
      nothing over.

   The direction just before the strip, in which Achilles takes the matches back off the slant's
   own arrow and sets out the little squares, is anchored too: nothing is drawn on the arrow.
   The close carries its marks: the little squares ghosted where they lay, and an arc for each
   half that moved. The file finds each half's place itself, by brute force over the 24
   assignments, and checks that the least travel is unique. Three halves move. One stays: the
   tails-ward square's half, which already lay against the tails-ward reach. Its edge from the
   crossing is that reach, and no edge of it lies along the slant.
2. The back-slant, its tail at the far left-hand corner: heads-ward one, tails-ward one
   *backwards*, the card's foot behind the crossing.
3. The back-slant again, with a little square on each reach. The shares are one in two, exactly
   the slant's, although one reach has changed its sign. That is the trap's evidence, asserted
   both ways.

```python
cs = check_cut_strip(cut_frames, SLANT)                           # every frame, every mark
assert cs.moved == 3 and [m.moved for m in cs.moves] == [True, True, False, True]
assert stay.square == "B" and same_place(stay.frm, stay.to)       # it lies in its place
assert has(stay.frm, np.zeros(2)) and has(stay.frm, tails_end)
assert not (has(stay.frm, np.zeros(2)) and has(stay.frm, cs.arrow.v))   # no edge along the slant
assert b.card.line == "B" and b.card.out == -1                    # the card at the tails-ward line
assert np.isclose(cs.reach["B"], -s.reach["B"])
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

Its six pictures, in four blocks.

1. The turned sheet, keyed, with its own line names, just swung (`swung=square`) about the
   crossing from where it lay under the last picture of the Third. The old lines are drawn where
   they were. The swing is $-45^\circ$, clockwise, the short way round, so the heads-ward line
   comes round toward the two of them, onto the back-slant. Swung the other way, the
   back-slant's whole arrow would lie backwards along its line, contrary to the dialogue.
2. Achilles's prediction (`expects=bookkeeping`, every backwards reach counted forwards), beside
   the true reading of the back-slant along the slant's line. The prediction lays the whole
   arrow, one match and a nearly whole piece. The truth is no reach: the card's foot on the
   crossing, and no matches.
3. The two strips, captioned "the back-slant's strip" and "the slant's strip". Under the turned
   sheet the share from the knot belongs to line A, so the back-slant's strip is all dark and
   the slant's all white: $|\langle -|-\rangle|^2$ and $|\langle -|+\rangle|^2$.
4. The three-and-four arrow under the turned sheet: just short of five along the slant's line
   (four matches and a nearly whole piece), and short of one match, backwards, along the
   back-slant's.

```python
d, ghosts = model_swing("square", "turned", BACK)
assert np.isclose(swung.swing, d) and np.isclose(d, -45)          # an eighth, clockwise
assert swung.swung == P.norm(before.args[1]) == "square"           # swung from where it lay
want = amplitude_reach(as_bookkeeping(BACK), PLUS)
assert np.isclose(expected.laid["B"], want) and np.isclose(want, np.linalg.norm(BACK))
assert not np.isclose(expected.laid["B"], cs.reach["B"])           # the prediction is refuted
assert shares == [1, 0]                                           # all dark, all white
assert laid(t.rows["A"]) == (0, 0.71, -1)                          # a piece behind the crossing
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

Its four moments.

1. The square turned end for end (`endforend=on`), the sheet lifted aside. The reversed arrow is
   inked, and the arrow it was is dotted where it lay: that is P4.4's arrow, and `flip_arrow`
   of the square it was. The half-turn about the square's centre carries the old tail onto the
   new tail and the old tip onto the new tip, clockwise.
2. Then three readings of the arrow reversed. The square sheet: three and four backwards, and
   the nine and the sixteen built again behind the crossing, the old blocks turned round.
3. The turned sheet: just short of five backwards, the sliver forwards now, every square as
   before.
4. Achilles's own position, the sheet swung twenty degrees with the heads-ward line tipped away
   from them: $-4.19$ and $-2.73$. Beside them is the both-ways row for the square turned back,
   $+4.19$ and $+2.73$: the same lengths, each on the other side of the crossing.

```python
was = check_turned_square(turn, REV)                               # a -> -a, tail to tail, tip to tip
assert np.allclose(was, before.arrow.v)                           # P4.4's arrow, dotted
assert P.norm(c_turn.args[0]) == P.flip_arrow(before.arrow.name)
assert {tuple(np.round(-c, 6)) for c in t.squares[l]} == \
       {tuple(np.round(c, 6)) for c in sq.squares[l]}             # the same block, turned round
assert laid(ow.back_rows["A"]) == (4, 0.19, 1) and laid(ow.back_rows["B"]) == (2, 0.73, 1)
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

Its four moments.

1. The sheet turned until a line lies along the arrow: the card's foot on the arrow's own tip,
   five in five, and no reach across.
2. The arrow at both ends of the pale string and, by `echo`, of the dark: both numbers five.
3. The outline, keyed and empty, with the three-by-three and the four-by-four beside it. These
   are the very blocks laid in the Second Challenge, read from that section's picture.
4. The re-laying as a step pair. First the nine, from the near left-hand corner, row filling
   into row, with the three-by-three ghosted where it stood (`from=empty`). Then the sixteen,
   with the four-by-four ghosted (`from=nine`). In the end there are twenty-five squares on
   twenty-five cells and nothing left of either block: the square on the arrow, asked about
   itself.

The file's own re-laying model fixes every cell, every square taken and every ghost's place,
for any count from any smaller one.

```python
assert np.allclose(cs.card.foot, cs.arrow.v)                       # on the arrow's own tip
assert (len(then.squares["A"]), len(then.squares["B"])) == (f0.left9, f0.left16)
check_outline(f1, 9, 0)
check_outline(f2, 25, 9)
assert {b for b, _, _ in f2.ghosts} == {16} and len(f2.ghosts) == 16 and (f2.left9, f2.left16) == (0, 0)
assert np.isclose(len(f2.laid), cs.reach["A"] ** 2)                # the square on the arrow
```
