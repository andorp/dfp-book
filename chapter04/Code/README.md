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

**The pictures.** The dialogue shows the state of its props with twenty pictures,
drawn by two macros of `pdf/props/ch04.tex`: `\stringframe[place]{state}{caption}`,
the Tortoise's frame with its strings, clips, bars and coins, and
`\foldsheet{folds}{marks}{caption}`, the folded sheet with its panels, slips,
tallies and pencil arrows. The macros *compute* what they show (each string's row
column by column, the row a clip rides on, which string passes in front at a
crossing, where each coin arrives and with which face, a patchwork's panels, the
arrows a clip draws), and `tools/props_ch04.py` transcribes that Lua once, for all six
files. Each file reads its own challenge's pictures out of `dialog.md` with
`tools/dfp_props.py` — never a copy — so an edit to the dialogue that breaks a
picture breaks a file. Every file checks the layout of its section (one picture per
block, word-only captions that never say wire, drawing, picture or box, the same
blocks in `dialog.raw.md`), that no macro would stop the build, and then the
transcription against the blackboard: for every way of laying the coins, and at
every stop along the frame, the macro's walk is the product, column by column, of
`X` ("turn it over"), `S0` ("heads up, whatever it was"), `CNOT` (the bar, upper row
looking) and `SWAP` (a crossing) placed on their rows:

```python
M = P.check_walks(fr, GATES)          # every start, every stop: walk == matrices
assert np.allclose(M, np.kron(np.kron(S0 @ X, X @ S0), I4))   # the chapter's matrix
```

`GATES` and each picture's matrix are written in the challenge file; the helper only
places the matrices it is handed. That these checks bite was shown by breaking
pictures in scratch copies of the dialogue (a face turned, a clip moved a column, a
crossing dropped, a star removed, a stray word in a caption, the raw dialogue out of
step) and rules in a scratch copy of the transcription: every break made a file fail.

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

*Pictures.* This file also checks the dialogue whole: twenty pictures, all inside
the six challenges, and the starred (keyed) form on exactly the first frame of each
set of hooked strings and the first sheet — P1.1, P1.2, P2.2. It pins the colon rule
(`silk heads` is a coin, `silk: heads` a clip, `silk heads: heads` both). The first
frame hooks silk above hemp and draws no coin, for both lie behind the card; the
patchwork's panels are the very `panels` the blackboard computed, its slips the two
little lists, and for any shares the macro's patchwork is the Kronecker product of
the slips, silk down the side:

```python
assert np.allclose(sh.shares, panels)                   # 3/8, 1/8, 3/8, 1/8
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

*Pictures.* The macro's folds give `2**n` panels for `n` folds; the third fold goes
across again, four rows of two, and the sheet has eight panels, not six; the fourth
gives sixteen where the separate coins need eight lines. The frame with all four
strings hooked is bare, and its matrix is the sixteen-by-sixteen identity.

```python
assert sh.panels == 2 ** 3 == 8 and sh.panels != 6      # "Two, four, six... eight."
assert 2 * len(fr.hooked) == 8 != sh.panels             # "Eight lines in all."
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

*Pictures.* The clip on the silk walks heads above and tails below to tails and
tails, as `X (x) I` says. Each of the macro's five arrow rules is checked against its
matrix on the four panels; the blue arrows are the permutation of `X (x) I`, all four
vertical and all across the first fold, and the red ones that of `I (x) X`, all four
sideways inside a half — no arrow in common, and every panel moved both times.

```python
assert all(a % 2 == b % 2 and a // 2 != b // 2 for a, b in blue)   # over the first fold
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

*Pictures.* Her clip to the left of his and slid past it are two frames with one
matrix and the same arrival from every start (Sliding). In *nothing moved* the first
clips stand level and the second clips level, and the frame's matrix is both
readings at once; silk arrives heads and hemp tails whatever the coins were. Lifted
off and put back beyond its neighbour, the silk's pair reads `X @ S0`: the silk coin
arrives tails, where a moment before it arrived heads. The macro's stopping places
(`at-`, `past-`, `off`) are checked on this frame too.

```python
by_columns = np.kron(S0, X) @ np.kron(X, S0)             # both firsts, then both seconds
by_strings = np.kron(I2, X @ S0) @ np.kron(S0 @ X, I2)   # all of the silk, then the hemp
assert np.allclose(M3, np.kron(by_columns, I4))
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

*Pictures.* Crossed once, the gut, upper, passes in front, and the coins change
hooks with their faces kept (`I4 (x) SWAP`). The clip slid through stays on the gut
and now sits on row four; its frame has the same matrix and the same arrival as the
clip before the crossing, and Achilles's guess — a clip turning the horsehair coin —
is a different frame. Crossed again, each string is in front once (wound), yet every
coin is home with its face, and from every start the wound pair walks exactly as a
straight one: the walk never reads which string went in front.

```python
assert f3.crossings() == [(2, "gut", "horsehair"), (3, "horsehair", "gut")]
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
assert np.allclose(np.kron(CNOT, SWAP) @ np.kron(CNOT, SWAP), np.eye(16))
assert np.allclose(whole, np.kron(Q @ P, I4))
assert not np.allclose(whole, np.kron(P @ Q, I4))
```

The sketchbook draws the same thing: the yoke as two joined nodes, the four rules
as picture equations, and

```python
alternating = (cn @ W2) + (W2 @ S) + (cn @ W2) + (W2 @ S)
assert same_picture(alternating, (cn @ S) + (cn @ S))    # sliding
assert same_picture(alternating, bare_wires(4))          # bare wire on every row
```

"The yoke is no pair of boxes" is a property of one matrix with no picture equation
at this stage, so it stays on the blackboard, and a comment in the file says so.

*Pictures.* The bar near the left upright is `CNOT (x) I4` (the wound pair below
contributes nothing) and walks tails and heads to tails and tails; with no bar the
same pair arrives tails and heads. No pair of clips on the silk and the hemp — none,
either rule, or two knocked together in either order — walks as the bar does on
every start. The sheet of tallies is seven, none, none, five, a dozen; its shaded
shares are `CNOT` applied to the tossed coin and the plain one, of rank two, so no
patchwork, and its slips lie empty. The closing frame is bar, crossing, bar,
crossing, the gut in front at the first crossing and the horsehair at the second; its
matrix is the master picture's, and every coin comes home with its face from all
sixteen starts.

```python
assert np.allclose(M4, alternating)                     # the master picture
for start in P.starts(f4):
    assert f4.coins(start) == {n: (start[n], P.HOOKROW[n]) for n in f4.hooked}
```

---

All six files draw their pictures with PyZX, and evaluate them only through
`tools/dfp.py`'s `matrix_of` and `same_picture`, which contract naively and keep
the scalar. Comparisons are exact: at this point in the book an equation of
drawings means the numbers agree, scalar included, and never merely up to a factor.
