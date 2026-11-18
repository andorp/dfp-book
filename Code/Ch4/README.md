# Chapter 4 — Uniting Wires: the companion code

Six files, one per challenge of the dialogue, in the order they are met. Each one
checks its claim twice: once on Dirac's blackboard, in NumPy, and once in Penrose's
sketchbook, as pictures compared through `tools/dfp.py`. Nothing is asserted here
that the chapter does not claim.

```bash
bash check-code.sh 4                              # all six
.venv/bin/python chapter04/Code/Ch4/Yoke.py       # just one
```

Each file prints one line per strand and exits non-zero the moment an assert fails.

Two conventions run through everything, and both are asserted rather than assumed.
The **top wire is the first label and the left factor**: `ket(0,1)` is the second
entry of four, and a box drawn on the upper wire of two is `np.kron(A, I2)`.
PyZX's own agreement with that is not taken on trust — `OneClip.py` asks the
drawing, comparing `matrix_of(A @ W)` against both `kron(A, I)` and `kron(I, A)`
and keeping the failing one as a negative assert. In the pictures, `g1 @ g2`
stacks `g1` above `g2` and `g1 + g2` puts `g1` first along the wire.

**The pictures.** The dialogue shows the state of its props in twenty-nine calls
drawing thirty-six frames, made by five macros of `pdf/props/ch04.tex`:

- `\stringframe[opts]{state}{caption}`, the Tortoise's frame with its strings, clips,
  bars and coins. The options give the coins' place, a step (`slide=`, `lift=`,
  `exchange=`) with its motion marks, a wrong rule's prediction (`expects=row|flat|stick`),
  or a fixed number of columns (`cols=`);
- `\framewalk[opts]{state}{captions}`, a step strip of one walk of the coins;
- `\framesteps[...]`, a strip of moves, which the dialogue does not use;
- `\foldsheet[opts]{folds}{marks}{caption}`, the folded sheet with its panels, slips,
  tallies and pencil arrows, and with `marks=on` or `expects=edge|untouched|patchwork`;
- `\foldsteps{first}{last}{captions}`, a step strip of the folding.

The macros *compute* what they show: each string's row column by column, the row a
clip rides on, which string passes in front at a crossing, where each coin arrives and
with which face. They also compute a clip slid along its string (stopped and *knocked*
against a clip on the same string), lifted past anything, the loop exchange at the
right-hand hooks, a patchwork's panels, the arrows a clip draws, and what a named
wrong rule predicts. `tools/props_ch04.py` transcribes that Lua once, for all six files,
all but the drawing. Where a mark's place is what a file asserts, its coordinates are
transcribed from the Lua's own constants: the side a swing arc passes, the gap between
a coin and a clip.

Each file reads its own challenge's pictures out of `dialog.md` with
`tools/dfp_props.py`, never a copy, so an edit to the dialogue that breaks a picture
breaks a file. `P.check_section` checks the layout of a section:

- the calls and the frames per block, and at most four moments;
- word-only captions that never say wire, drawing, picture or box, with a tall letter in
  every caption of a block of several frames;
- the same blocks in `dialog.raw.md`;
- no caption echoing the paragraph after its block, in either file;
- no macro that would stop the build.

Then every frame is asserted against the file's own model. The model is the
transcription checked against the blackboard. For every way of laying the coins, and
at every stop along the frame, the macro's walk is the product, column by column, of
`X` ("turn it over"), `S0` ("heads up, whatever it was"), `CNOT` (the bar, upper row
looking) and `SWAP` (a crossing), placed on their rows:

```python
M = P.check_walks(fr, GATES)          # every start, every stop: walk == matrices
assert np.allclose(M, np.kron(np.kron(S0 @ X, X @ S0), I4))   # the chapter's matrix
```

The strips, the marks and the predictions are held to the same model:

- Each frame of a walk strip draws the model's column at that stop.
- Its ghost coins draw the column just before the step, and only for the coins that
  step changed. A hop is drawn exactly where a face changed.
- A step's ghost clip stands where the clip stood in the call's own state, the state
  before the step.
- The hold rings appear in the two exchange frames and nowhere else.
- Every `expects=` frame equals the named wrong rule as the file computes it, and
  differs from the true frame beside it.

```python
expected = np.kron(X, I2) @ SWAP @ ket(0, 0)              # the row rule's coins
assert np.allclose(P.drawn_column(fr_row, rows=LOWER), expected)
assert not np.allclose(expected, truth)                     # refuted
```

The round's direction edits (D1–D10) and caption renames (C1, C2) are pinned too:
each new text stands once in its section of both files, its old text nowhere, in the
paragraph the frame needs. A placed phrase is asserted on the drawn extent with a
positive gap:

- D1–D3 give the side each fold's swing arcs pass;
- "passes clean under his" for the slide. The macro lowers a slide's arc under any clip
  or bar standing above it, and its stroke clears his clip, drop shadow included, by
  about 0.05 cm;
- the slide through the crossing, whose mark follows the gut's own string. It keeps
  0.075 above the gut all the way and at least 0.02 clear of its stroke, meets the
  horsehair only inside the crossing's column, and has no lift's hump;
- "beyond its neighbour" for the lift;
- "The two crossings lie close together about the middle" for the wound pair;
- the bar "near the left upright".

Each section's log lines are asserted verbatim. They are the build log's dump, all
225 lines of it, which `describe` reproduces exactly in both dialogue files.

`GATES` and each picture's matrix are written in the challenge file; the helper only
places the matrices it is handed.

That these checks bite was shown in three ways:

- **The transcription against `texlua`.** A driver ran `ch04.lua` under `texlua` with `tex`
  stubbed, and the transcription beside it. It was proved first in round 4 and proved again,
  from scratch, when the pins below were added. The second proof made 66,132 calls: the 29
  calls of `dialog.md` and the 29 of `dialog.raw.md`, 74 aimed calls, and 66,000 random calls
  over six seeds. Half of the random calls are frames built to act (slides through crossings
  both ways, knocks, lifts, exchanges, walks); the other half sweep the grammar's words,
  faults and white space. They drew 65,884 strip frames, among them 6,858 slides, 1,538
  knocks, 4,169 lifts and 5,500 exchanges, and ran 8,269 probes of the Lua's exported model.
  Of the slides, 2,233 went through a crossing, their arc following the string: rightward
  647 with the arc above and 918 below, leftward 267 above and 401 below. The faults, the
  log lines, the strips' expansions and the marks' coordinates agree byte for byte on every call, and
  the Lua raised no Lua error. That holds with Lua's `pairs` walking the strings in hook
  order. The same calls were run again under reversed order and three times under Lua's own
  `pairs`. No dialogue call differs. Every call or probe that differs falls under the first
  liberty in `props_ch04.py`, by a mechanical test: two step keys in one bracket, three
  neighbouring strings saying yoke or cross in one column, or faults reordered within one
  column. A copy of the transcription with one arc constant moved by 0.01 failed the
  comparison.
  The inputs on which the Lua itself crashes are not imitated: a bare `+` cell a coin or a
  slide must pass, a walk over eight columns, `cols=2.5`, shares or tallies not four, and
  marks asked past the end of a frame of no columns. Two more liberties are refusals. The
  Lua's `tonumber` takes `cols=4.0` and `cols=4e0` as four, where the transcription refuses
  them. The Lua lower-cases only ASCII, so a word with a non-ASCII capital is refused by both
  but echoed differently in the fault.
  Of the Lua's 76 `tex.error` sites, the aimed calls raise 74. In round 4, removing any one
  of them was caught, and so was reversing an arc, flipping a hop, undoing the slide's
  lowered arc or its arc through a crossing, or putting that arc on the wrong side.
  Two sites are guards that no input reaches, so the proof raised them on crafted input
  (`debug.getupvalue`):
  - "the slide's arc cannot pass under the clip above". Rows lie 0.44 apart, so a clip
    on a higher row always leaves the lowered arc about 0.12 of rise, and the site
    asks for less than 0.04.
  - "no partner string at the crossing". A row changes only at a crossing, which
    exchanges two strings, so a partner is always there.
- **Thirty dialogue mutations**, on scratch copies of the dialogue. These were a wrong
  frame, a dropped mark, a prediction drawn as the truth, a reverted direction, a
  removed re-lay or clearing, a restored old caption, a caption with no tall letter,
  the raw dialogue out of step, the removed P4.1 put back, a starred prediction, the
  Fourth's columns unfixed, and a fifth moment. Every one made a file fail.
- **The round-3 breaks**, five more mutations, each of which still makes a file fail:
  a face turned, a clip moved a column, a crossing dropped, a star removed, a stray
  word in a caption.

**The pins.** Each file calls `props_ch04.check_lua()` first in `pictures()`. It holds
`ch04.lua` and `ch04.tex` to the text the transcription was proved against, by hash:

- `LUA_PINS` covers the Lua's 93 top-level chunks, with comments left out and white space
  closed up outside strings;
- `COMMENT_PINS` covers the Lua's four comments that quote the dialogue (the header, the two
  Round 4 blocks, and the wrong rules row, flat and stick), together with which of their
  quotes the dialogue still says (case, quotation marks and apostrophes aside);
- `TEX_PINS` covers the 50 chunks of the `.tex`.

A drawing-only change, a style-only change, a reworded quote or a quoted line the dialogue no
longer says fails all six files, and the failure names the chunk. Any other comment may be
reworded freely, and re-indenting, blank lines or a longer run of blanks between tokens pass.
A blank added or removed inside a string, or the only blank between two tokens removed, is
caught. After a prop edit, prove the transcription again, then renew the pins with
`lua_pins()`, `comment_pins()` and `tex_pins()`. The check does not run at import, so the
module still loads for that.

---

## 1. `SecondString.py` — a second string below

The Tortoise hangs a second coin below the first and asks for "the share of heads
above and tails below, together". Achilles offers two little lists; they sum to
two, and neither of them has a line for the question asked. The file makes that
failure exact and then builds the right object.

```python
assert np.isclose(two_lists.sum(), 2)
assert not np.isclose(1 / 2 + 1 / 4, panels[1])    # adding: not the panel
assert np.isclose(panels[1], 1 / 2 * 1 / 4)        # multiplying: the panel
assert np.allclose(np.kron(psi, phi), [a * c, a * d, b * c, b * d])
```

Then the general claims of §4.2: the joint column of two states is itself a state,
the product is bilinear in each factor with complex multiples, the mirror of a
product is the product of the mirrors, and Proposition 4.1 — brackets multiply —
over fifty random quadruples, with the two mix-ups it invites ruled out:

```python
assert np.isclose(left, np.vdot(pp, p) * np.vdot(qq, q))
assert not np.isclose(left, np.kron(pp, qq) @ np.kron(p, q))   # no conjugation
assert not np.isclose(left, np.vdot(pp, q) * np.vdot(qq, p))   # factors crossed
```

The states are random complex, so the brackets are not real and a dropped
conjugation cannot hide. In the sketchbook the two states are drawn stacked and
their tensor compared with the Kronecker column, top factor first; Proposition 4.1
becomes a picture equation — states stacked and then effects stacked is the two
closed drawings side by side — with the crossed pairing as its negative.

*Pictures.* This file also checks the dialogue whole:

- twenty-nine calls and thirty-six frames, all inside the six challenges, with three,
  three, three, four, four and four moments;
- four step strips (the fold strip of two frames, three walk strips of three);
- eight marked single frames;
- five predictions, in order `edge`, `untouched`, `row`, `flat`, `patchwork`, none
  keyed and none marked;
- the hold rings in the two exchange frames only;
- the starred (keyed) form on exactly the first frame of each set of hooked strings and
  the first sheet: P1.1, the marked sheet of block 1.2, and P2.2. P2.2 is the third call of
  the Second, which by macro alone would be no first.

It pins the colon rule (`silk heads` is a coin, `silk: heads` a clip, `silk heads: heads`
both) and the macros' refusals: a starred or marked prediction, `edge` before the third
fold, a fold strip run backwards, five frames, captions short of frames.

The first frame hooks silk above hemp and draws no coin, for both lie behind the card.
Block 1.2 is the folding, frame by frame:

- blank, folded once, folded twice;
- each fold doubles the panels, and draws one swing arc in every panel there was before
  it;
- D1: the first fold's arc runs up, from the lower half over the crease;
- D2: the second fold's arcs run left, from the right half, one in each row.

The patchwork's panels are the very `panels` the blackboard computed, and its slips are
the two little lists. For any shares, the macro's patchwork is the Kronecker product of
the slips, silk down the side:

```python
assert P.up(a) and a[0][1] < P.SH / 2 < a[3][1]          # D1: lower half up onto the upper
assert np.allclose(pw.shares, panels)                   # 3/8, 1/8, 3/8, 1/8
assert np.allclose(got, np.kron([s, 1 - s], [h, 1 - h]))
```

## 2. `Doubling.py` — sixteen panels

Achilles folds a third time and guesses six. The file walks one to four strings and
keeps both counts side by side: `2**n` entries in the column against `2*n` numbers
for the same states kept apart.

```python
assert column.shape == (2 ** n,)
assert separate.shape == (2 * n,)
assert 2 ** 3 != 6 and 2 ** 4 != 8
assert 2 ** 10 == 1024 and 2 * 10 == 20
```

The two counts agree for one string and for two — which is exactly why the trap
works — and part for good from three on; that is asserted in both directions.
Bracketing is checked to be free (`(p (x) q) (x) r = p (x) (q (x) r)`, entry by
entry as $p_i q_k r_m$) while order is not. The sketchbook half is a size claim,
not a second proof, and says so in the docstring: `n` lines of ink drawn,
`2**n` entries down each side of the tensor, and all sixteen stacked basis
drawings of four strings landing on their own single entry.

*Pictures.* The macro's folds give `2**n` panels for `n` folds.

- The third fold goes across again, four rows of two, and the sheet has eight panels,
  not six.
- Its marks are four arcs, one in each panel there was, each running up from a panel's
  lower half over the new crease: D3's "bottom edge up to top".
- Beside it stands the `edge` prediction, the sheet of "So a third string makes six.",
  with three rows of two panels added at the edge. The file computes it from its own
  rule and shows it is not the paper's. The rule agrees with the paper up to two folds,
  which is why the macro draws it only from the third.
- The fourth fold gives sixteen where the separate coins need eight lines.
- The frame with all four strings hooked is bare, and its matrix is the
  sixteen-by-sixteen identity.

```python
assert edge.shape == edge_rule(3) and edge.panels == 6 == 2 + 2 + 2
assert sh.panels == 2 ** 3 == 8 and sh.panels != 6      # "Two, four, six... eight."
assert 2 * len(fr.hooked) == 8 != sh4.panels            # "Eight lines in all."
```

## 3. `OneClip.py` — one clip, nothing on the other

A clip on the silk string alone. Achilles expects the panels of the other coin to
sit still. The file builds both matrices from the rule on the four pairs and
matches them to the stacked forms, then shows the block shapes of §4.4 on a matrix
that is neither real nor symmetric — with a stray transpose and a stray mirror as
negatives.

```python
assert np.allclose(top, np.kron(X, I2))              # NOT on the TOP wire
assert np.allclose(bottom, np.kron(I2, X))           # NOT on the bottom wire
assert not np.allclose(np.kron(A, I2), np.kron(A.T, I2))
```

That the two are the same matrix only when the box is a multiple of nothing-at-all
is checked both ways: a positive control with $\lambda I$, and twenty random boxes
for which they differ. The dialogue's own trap is the last block — on the evenly
tossed sheet the whole list of shares comes back unchanged, and yet not one panel
stays where it was, and a lopsided sheet gives the difference away in the numbers.

The empirical reading of the drawing lives here, and is the file's hinge:

```python
assert np.allclose(matrix_of(A @ W), np.kron(Am, I2))     # box on the TOP wire
assert not np.allclose(matrix_of(A @ W), np.kron(I2, Am))
assert np.allclose(matrix_of(W @ A), np.kron(I2, Am))     # box on the bottom wire
```

*Pictures.* The walk strip draws the clip on the silk walking heads above and tails
below. The coins are laid on at the left upright (D4 sets them back there), and the
three frames are the model's columns:

- the start, `ket(0, 1)`;
- just past the clip, `kron(X, I2) @ ket(0, 1)`;
- the arrival, the same column, tails and tails.

Only the silk coin has a ghost and a hop, for only its face changed. The ghost stands
clear of the clip before it, and the coin clear of it after. The arrival frame is the
old P3.1, unchanged.

Each of the macro's five arrow rules is checked against its matrix on the four panels.

- The blue arrows are the permutation of `X (x) I`: all four vertical, all across the
  first fold (caption C1).
- Beside them, the `untouched` prediction ("The whole sheet is untouched!") is the same
  patchwork with no arrow, the identity's, which is no clip. The numbers agree only
  because the two halves are twins.
- The red ones are the permutation of `I (x) X`, all four sideways inside a half. No
  arrow is in common, and every panel moved both times.

```python
assert np.allclose(P.drawn_column(f1, rows=changed, ghosts=True), ket(0))   # as laid
assert all(a % 2 == b % 2 and a // 2 != b // 2 for a, b in blue)   # across the first fold
assert all(a // 2 == b // 2 and a % 2 != b % 2 for a, b in red)    # inside each half
```

## 4. `Interchange.py` — which first?

Achilles walks the frame two ways, touches no clip, and finds the same arrival.
The file checks the Interchange Law (Theorem 4.2) on the chapter's four named
matrices — including the printed products and the traced entry $-2+2i$, computed
three ways — and then on thirty random quadruples, each first certified non-real,
non-symmetric, non-Hermitian and pairwise distinct.

```python
assert np.allclose(np.kron(a, b) @ np.kron(c, d), np.kron(a @ c, b @ d))
assert not np.allclose(np.kron(a, b) @ np.kron(c, d), np.kron(c @ a, d @ b))
assert not np.allclose(np.kron(a, b) @ np.kron(c, d), np.kron(a @ d, b @ c))
assert not np.allclose(np.kron(a, b) @ np.kron(c, d), np.kron(a @ c, b @ d).T)
```

Seven wrong readings fail: the two factors composed the other way round, the boxes
paired across wires, and the whole answer reflected, mirrored or conjugated.
Sliding (Corollary 4.3) is checked in both directions, the mirror of a stack is the
stack of the mirrors (with the swapped version failing), and unitary above unitary
is unitary. Against all this, one wire keeps its grammar: `S0 @ X` is not
`X @ S0`, while those same two boxes on different wires have nothing to quarrel
about. The dialogue's coin walks close the blackboard: one clip each — left, right
and level are three drawings and one matrix — and two clips each, where "First both firsts and then both seconds" and "all of the silk and then all of the
hemp" agree entry
for entry, and reversing the silk pair changes the arrival.

The sketchbook builds one frame and reads it two ways,

```python
by_columns = (c @ d) + (a @ b)
by_wires = (c + a) @ (d + b)
assert same_picture(by_columns, by_wires)
```

with the crossed pairings as negatives. A comment says what the pictures do not
cover: the chains available at this stage are unitary, so the law for boxes that
keep no promise is the blackboard's business.

*Pictures.* Every frame of the Fourth has four columns, so no clip shifts from
picture to picture. The first four calls set `cols=4`; the last has four cells of its
own.

- **The slide.** The frame starts from her clip to the left of his, the call's own
  state and the old P4.1, and slides her clip past his. The two states have one matrix
  and the same arrival from every start (Sliding). The ghost stands where her clip
  stood. The arc is lowered for his clip: it starts beside her ghost and ends beside
  the clip where it lands, just above the hemp. It runs rightwards, and its stroke
  clears his clip, drop shadow included, by more than 0.04 ("passes clean under his").
  The clip lands in column three, beyond his.
- **Nothing moved.** The first clips stand level and the second clips level. The
  frame's matrix is both readings at once, and silk arrives heads and hemp tails
  whatever the coins were. The macro's stopping places (`at-`, `past-`, `off`) are
  checked on this frame too, and so is its refusal to walk a strip through two columns
  of clips.
- **The knock.** D6 takes the coins off first. "Turn it over", pushed to the right, is
  stopped in column three against "heads", which stays at its column's centre. The two
  stand struck, a hair apart, and walk exactly as before: along one string, sliding
  cannot change the order.
- **The lift.** "Turn it over" is lifted from where the knock left it, higher than a
  slide and over its neighbour, and lands beyond it. The frame beside it walks the silk
  alone from tails: the silk arrives tails, where a moment before it arrived heads. The
  hemp clips never moved.

```python
assert P.clip_extent(sx, sy)[2] - (hi + P.SWING / 2) > 0.04       # under his clip
assert fk.cellwords("silk") == ["-", "-", "over>heads", "-"]      # struck
assert np.allclose(M3, np.kron(by_columns, I4))                    # the Interchange Law
```

## 5. `Crossing.py` — the crossing

"The clip kept its string; the string changed its place." SWAP is built from the
rule on the four pairs, matched to the printed matrix, and shown real, symmetric,
unitary and its own undoing. It exchanges the factors of thirty random product
states — complex, so a crossing cannot be mistaken for a conjugation — and its two
rules are checked with the wrong versions failing:

```python
assert np.allclose(SWAP @ np.kron(A, B), np.kron(B, A) @ SWAP)
assert not np.allclose(SWAP @ np.kron(A, B), np.kron(A, B) @ SWAP)   # must change row
assert np.allclose(SWAP @ SWAP, np.eye(4))
```

That the crossing is no pair of boxes is the rank test of §4.6 — a stacked pair
always has rank one when rewritten, the crossing has rank four — together with the
elementary block argument in numbers. The dialogue's clip is then put on the gut
string, slid across, and put on the horsehair instead, each on all four starts, and
the two cases shown to differ. In the sketchbook a crossing is drawn flat, with no
node at all: the upper input wired to the lower output and back. States crossed,
a box slid through and changing row, a clip left of the crossing equal to the
other clip right of it, and two crossings equal to two bare wires.

*Pictures.*

- **The exchange.** It takes a bare frame to three columns crossed midway, `I4 (x)
  SWAP`. The gut, upper, comes down in front over the upright and the horsehair goes up
  behind it. Hold rings sit on the pair's left-hand hooks, which never move.
- **The walk strip.** It takes the two coins through the crossing frame by frame. Each
  changes its row and keeps its face, so it has a ghost on its old row and no hop.
- **The slide through.** The clip is slid through the crossing and stays on the gut,
  now on row four. Its frame has the same matrix and the same arrival as the clip
  before the crossing. Its mark follows the gut through the crossing, on the side away
  from the horsehair, about 0.034 clear of the gut's stroke. It meets the horsehair
  once, inside the crossing's column, and never rises above the gut's first row by
  more than its offset. The clip went along its string, not over the other one.
- **The `row` prediction.** It walks the same frame with the clip handed to the
  horsehair, whose hook is on the clip's row at the left upright, so
  `kron(X, I2) @ SWAP` where the truth is `kron(I2, X) @ SWAP`. Refuted.
- **The second exchange.** It is the same hand-movement, with the horsehair in front
  this time. It winds the pair: two crossings close together about the middle (D9), the
  right ends home, the matrix the identity.
- **The `flat` prediction.** It is that frame with both crossings taken out. The coins
  cannot tell the two apart, and the strings can.
- **The same report.** From every start, the wound pair walks exactly as a straight
  one: the walk never reads which string went in front.

```python
assert np.allclose(P.check_walks(fr_flat, GATES), Mw)      # the coins report the same
assert fr_flat.cells != fw.cells                           # refuted: in the strings only
for start in P.starts(f3):
    assert f3.coins(start) == straight.coins(start)
```

## 6. `Yoke.py` — the yoke

The stick between the two strings is not a clip on each: it is how the lower coin
looks up. The file reads the box off the rule $|a,b\rangle \mapsto |a, a \oplus b\rangle$,
finds the block form, and shows it real, symmetric, unitary, stochastic and its own
undoing. Then the claim the chapter rests on (Proposition 4.5):

```python
assert np.linalg.matrix_rank(reshuffle(CNOT)) == 2
assert np.linalg.matrix_rank(reshuffle(np.kron(P, Q))) == 1     # every pair of boxes
```

Achilles's two attempts are repeated exactly as he made them — the clip on the
lower string agrees when the upper coin shows tails and fails when it shows heads;
no clip at all does the opposite — and neither of them is the yoke. On shares, the
tossed coin above and the plain coin below arrive matched:

```python
assert np.allclose(matched, [1 / 2, 0, 0, 1 / 2])
assert np.linalg.matrix_rank(matched.reshape(2, 2)) == 2      # no product of two lists
```

while any pair of clips carries a product list to a product list, and a second yoke
gives the coins back. Finally the master picture, *Twice Yoked, Twice Crossed*: the
yoked pair and the crossed pair, twice over, is bare line on every row, in both
column orders, and with any two boxes on the upper pair the crossed pair below
contributes nothing.

```python
paired = np.kron(CNOT, SWAP)
assert np.allclose(paired @ paired, np.eye(16))
assert np.allclose(whole, np.kron(Q @ P, I4))
assert not np.allclose(whole, np.kron(P @ Q, I4))
```

The sketchbook draws the same thing: the yoke as two joined nodes, the four rules
as picture equations, and

```python
alternating = (cn @ W2) + (W2 @ S) + (cn @ W2) + (W2 @ S)
paired = (cn @ S) + (cn @ S)
assert same_picture(alternating, paired)                 # sliding
assert same_picture(alternating, bare_wires(4))          # bare wire on every row
```

"The yoke is no pair of boxes" is a property of one matrix with no picture equation
at this stage, so it stays on the blackboard, and a comment in the file says so.

*Pictures.*

- **The walk strip past the bar.** The bar stands in column one, near the left
  upright, and the coins are set back there by D10. The strip draws `CNOT` on tails
  above and heads below frame by frame: the hemp coin turned, its ghost as laid, its
  hop drawn under it and under the bar with a gap, and the silk coin never touched.
  The arrival is the old P6.1, `CNOT (x) I4`; the wound pair below contributes
  nothing.
- **No `stick` prediction.** The stick rule agrees with the bar when the silk shows
  tails, which is the strip's walk, and differs only for heads (ruling Q1).
- **No bar.** With no bar the same pair arrives tails and heads. No pair of clips on
  the silk and the hemp walks as the bar does on every start: none, either rule, or two
  knocked together in either order.
- **The sheet of tallies.** It is seven, none, none, five, a dozen. Its shaded shares
  are `CNOT` applied to the tossed coin and the plain one, of rank two, so no
  patchwork, and its slips lie empty. Beside it, the `patchwork` prediction multiplies
  out the sheet's own borders, a half and a half: a quarter in every panel, which is
  not the sheet.
- **The closing frame.** It is captioned "all home" (C2): bar, crossing, bar,
  crossing, the gut in front at the first crossing and the horsehair at the second.
  Its matrix is the master picture's, and every coin comes home with its face from all
  sixteen starts.

```python
assert np.allclose(pw.shares, np.kron(silk, hemp)) and np.allclose(pw.shares, [1 / 4] * 4)
assert not np.allclose(pw.shares, sh.shares)             # refuted
assert np.allclose(M4, alternating)                     # the master picture
for start in P.starts(f4):
    assert f4.coins(start) == {n: (start[n], P.HOOKROW[n]) for n in f4.hooked}
```

---

All six files draw their pictures with PyZX, and evaluate them only through
`tools/dfp.py`'s `matrix_of` and `same_picture`, which contract naively and keep
the scalar. Comparisons are exact: at this point in the book an equation of
drawings means the numbers agree, scalar included, and never merely up to a factor.
