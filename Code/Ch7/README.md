# Chapter 7 — Companion Code: *Inequalities Nature Breaks*

Five challenges, five files, each checked twice: once on **Dirac's Blackboard**
(NumPy `assert`s) and once in **Coecke's Sketchbook** (PyZX pictures compared as
tensors, only through `tools/dfp.py`'s `matrix_of` / `same_picture`, which force
the correct `strategy="naive"`). Every picture equality here is **exact, scalar
included**. Cups, caps and the three-legged node are **unnormalized**
(`||cup|| = ||node|| = sqrt(2)`); a one-legged state or effect costs a `sqrt(2)`,
paid back with `g.scalar.add_power(-1)`.

```bash
bash check-code.sh 7                 # run all five
.venv/bin/python chapter07/Code/Ch7/SealedNotes.py   # run one
```

The first two files set a ceiling and then go through it: sealed notes stop at
three deals in four, a bent wire with one tilted box reaches
`cos^2(pi/8) ~ 85%`. The last three replace *more often* with *always*: the
three-legged node, the six notes that cannot be written, and the round that is
never lost.

**The prop pictures.** The dialogue draws its props with four macros from
`pdf/props/ch07.tex` — Achilles's card with its counters, masks, buttons and
egg-cup (`\notecard`), the hall table (`\halltable`), one trio box close up
(`\triobox`) and a page of Aunt Hillary's score-book (`\scorebook`) — and, since
the "Show the process" round, three step strips (`\cardsteps`, `\hallsteps`,
`\triosteps`). Their Lua computes part of what is drawn: a button on every mask
whose word the counters break, the egg-cup that stands only on counters keeping
the word, the slip's word and the spent boxes' open ways computed from the deal,
Bob's strokes computed by the pair game's rule — and each *step*: a frame names
the step (`turn=`, `lay=`, `lift=`, `cup=` on the card; `open=`, `lift=` on the
hall table; `open=` on a box) on the state before it, and the Lua draws the state
after, a pale *ghost* of the moving part where it was (for the box's door and
drawer, and the egg-cup, its dotted outline only, so it never paints over the
part moved), a curved *arc* for its
motion (a swing, or a shallower slide), and a dotted *ring* where a hand holds a
box still. Two frames are *predictions* (`expects=mended`): the buttons
Achilles's wrong rule leaves — the buttons before, less those on the masks that
show the turned counter — drawn pale beside the true frame. In a starred strip
every frame keeps the key's room (`keyroom=on`): the first draws the labels, the
others draw them as vertical phantoms, so the frames stand level.

`tools/props_ch07.py` transcribes all of it except the drawing (no matrices
there): every option and step, every `tex.error` of the Lua (67 sites, in Lua
order, matched by `check_sites()` against the Lua source), the marks' TikZ text,
the label nodes, and the log line the build writes for every picture — the 88
lines of a real build, byte for byte. It was proved against `texlua` on the
dialogue's 22 calls and on tens of thousands of random and aimed ones.
`check_lua()` pins the Lua chunk by chunk (comments stripped), and
`pdf/props/ch07.tex` with it — its colours, its mark, label and caption styles,
its macros' expansion — so that a change to the drawing or the styles alone,
which no log line shows, breaks every file until the transcription is looked at
again (then renew `LUA_PINS` and `TEX_PINS` from `lua_pins()` and `tex_pins()`).

Each file reads the pictures of its own section of `dialog.md` and
`dialog.raw.md` through `tools/dfp_props.py`, never a copy of them, and checks
their layout (22 calls in 19 blocks, 32 frames; blocks of calls `[1,1,2,1]`,
`[1,1,1,1]`, `[2,1,1,1]`, `[1,1,2]`, `[1,1,1,1]`; at most four frames to a row
and four moments to a challenge), their captions (words only, no digit, no echo
of the paragraph after; no count at all under a hall table or a trio box, which
draw the wire-objects), the speech or direction each follows, the directions and
captions edited this round, and, frame by frame and in order: the state each
frame draws against the scene at that moment; each step frame's before-state
against the picture of that surface before it; each ghost at the state before
the step; each arc from the moving part's old place to its new one; each ring on
the box the words hold still; each prediction equal to `mended` and different
from the true frame; and the section's log, line for line. Each file also ties
the transcription to its own blackboard — a structural check, not a second
proof.

---

## 1. `SealedNotes.py` — *The Sealed Notes*

Achilles brings his theory: a note for the lid and a note for the door, written
inside each box before the players part. The Tortoise reads the game from the
Crab's letter — the answers must be *alike*, except when Alice is asked door and
Bob lid, when they must be *unlike*. That is the CHSH game (win iff
`a xor b = x.y`), and sealed notes are a **product** strategy.

The blackboard enumerates the 16 deterministic strategies and shows the best
wins exactly 3 of 4; every strategy has `|S| = 2`; and the count of broken
pair-masks is **always odd** — 8 cards break one mask, 8 break three, never
none and never an even number:

```python
assert max(win_counts) == 3
assert all(L % 2 == 1 for L in losses)               # always odd
assert losses.count(1) == 8 and losses.count(3) == 8
```

The dialogue's moving button is the fact that **one flip changes exactly two
verdicts** (each note lies under two masks), asserted for all 16 cards; and a
shared coin cannot do better, checked on 200 random mixtures. The sketchbook
draws the point: a sealed-note number **factors** into two one-wire numbers
(`joint == left * right`), and — Chapter 6's test — two disconnected states make
a rank-one (determinant 0) table, while the cup does not (`det = 1`).

*Pictures.* Each pair-mask reads one question pair, and for all 16 cards it is
kept exactly when that pair is won — so the buttons are the losses. The pictures
are read off the card: all white; then the laying strip, four frames, the four
masks laid one at a time in the order the Tortoise reads them, each a ghost on
the card and an arc down to its place below, two by two — no button until the
fourth, then one on the unlike mask, from her pocket, with no mark (frame 1
keyed with the rows and the first mask's word, the others keeping its room as
phantoms). Then the turn, Achilles's forecast beside it: by `mended` the right
lid turned leaves no button; the true frame keeps one, crossed — a ghost on the
unlike mask, an arc to the two-lids mask. The left lid turned again, unmarked (a
repeat): the button on the left-lid-right-door mask. Each flip one counter, two
verdicts changed, one button still:

```python
assert (not P.broken(m, cs)) == win(a[x], b[y], x, y)          # all 16 cards, all four masks
assert [f.buttons for f in lays] == [(), (), (), ("left door right lid",)]
assert f06.buttons == tuple(P.rule_mended("front", f05.counters, ["right lid"])) == ()
assert f07.buttons == ("two lids",) != f06.buttons
```

## 2. `Wedge.py` — *The Wedge*

The Anteater's tale: a hundred evenings, one shared pair each, Bob's box stood
on a wedge so it is asked its question *tilted halfway between lid and door*.
The wedge is the real rotation `R_{pi/8}` on Bob's leg. The blackboard builds
all 16 amplitudes on `|Phi+> = cup/sqrt(2)` and finds every one of the four
question pairs won with probability `cos^2(pi/8)`, with `E = (+,+,+,-)/sqrt(2)`
and `S = 2*sqrt(2) > 2`:

```python
assert np.isclose(pwin, cos2)                        # every pair wins cos^2(pi/8)
assert np.isclose(S, 2 * np.sqrt(2)) and S > 2
```

`R_{pi/8}` is **real but not symmetric** (`R^T = R_{-pi/8}`), so sliding it round
the cup is a genuine transpose — shown first with a lopsided complex box where
transpose, dagger and conjugate all differ, then specialised to `R`. And no kind
of evening is the losing one: each player's own answer is `1/2, 1/2` for every
pair. The sketchbook reproduces every one of the 16 amplitudes as a closed
picture (cup, `R` chain, two effects) and shows the wedge slides: `R` on Bob's
leg equals `R` turned the other way on Alice's — but not `R` the same way up.

*Pictures.* Bob's stroke rule is the CHSH rule under the card's labelling, and
the very rule of the First's masks. The open page has ten strokes in twelve,
computed, with every kind of evening on it; Bob's sorted page has four heaps of
six in the order the hook names them, five strokes in each, the loss on a
different row each time. The heaps are the open page and the page facing it
(the direction says so since this round): the open page's twelve evenings are
among the heaps' twenty-four, and what is left — the facing page, never drawn —
is twelve evenings with ten strokes, so 10 + 10 = 20, *not a stroke added or
lost*. With Alice's column covered, Bob's shows six dark in twelve, three in six
whatever Alice was asked. The page is a fiction, and no probability is asserted
from it. Then the hall table, a strip: the case shut, keyed (the first hall
table), and the lid lifted, the boxes in their nests — the lid's swing arc from
its front edge at the case's right end, bowed out and short of the slip.

```python
assert P.won(ea, eh, wa, wh) == ((a ^ b) == (x & y))
assert sum(r.stroke for r in heap) == 5                 # "About the same in each."
facing = heaps - page
assert sum(facing.values()) == 12 and sum(n for e, n in facing.items() if e[-1]) == 10
```

## 3. `Trio.py` — *Three from One Bench*

Three boxes from one bench: the three-legged node `N = |000> + |111>`. The
blackboard fixes its norm (`sqrt(2)`, `<N|N> = 2`) and shows that on any
guaranteed deal the **wrong-parity amplitudes are exactly 0** — two answers fix
the third, which is EPR's criterion enacted with the egg-cup. It also proves the
rule the next file needs: diagonal boxes slide through the node and merge, a
generic box does not:

```python
assert np.allclose(kron(D, I2, I2) @ N, kron(I2, I2, D) @ N)       # slide
assert np.allclose(kron(D, Dp, I2) @ N, kron(I2, I2, D @ Dp) @ N)  # merge
assert not np.allclose(kron(B, I2, I2) @ N, kron(I2, B, I2) @ N)   # a generic box: no
```

Closing one leg with a bright door-effect leaves `(1/sqrt(2)) cup`; a dark one
leaves the same with a `Z` box on one leg. The sketchbook confirms the node is
exactly `N` (one dot, no scalar), slides and merges phase boxes leg to leg, and
matches the two leg-closings against the cup — with the negative check that the
`1/sqrt(2)` matters.

*Pictures.* The egg-cup is tied to the node's certainty: for every mask, every
window under the cup and every two faces shown, the one face the macro accepts
under the cup is the one answer with a non-zero amplitude. The pictures: the
boxes lifted from their nests one at a time, from the left (a ghost in each nest,
an arc to its place on the cloth, between the case and the envelopes), beside one
of them close up, in one block; the EVEN mask on the back of the card, two whites
shown, the cup over the right door, the hidden face forced white; the cup lifted
(its ghost in outline, no arc: the text names no place it goes), the white
counter shown; then the left and right counters set back black in their own
windows (a flip over each, the middle one unmarked), and the cup stood over the
left door from the hand (no mark), the hidden face forced black.

```python
assert live == keep                                     # the node's one answer
assert [(g.what, g.name, g.outline) for g in c19.ghosts] == [("cup", "right door", True)]
assert P.keeping_faces(c34.mask, c34.shown, c34.cup) == ("black",) == (c34.counters[c34.cup],)
```

## 4. `SixNotes.py` — *Six Notes, Four Masks* (*The Theorem, Twice*)

The headline. The blackboard verifies the four eigen-relations exactly —
`XXX N = +N` and `XYY N = YXY N = YYX N = -N` — the operator identity
`(XYY)(YXY)(YYX) = -XXX`, and that no six written notes meet all four demands:

```python
assert np.allclose(kron(X, X, X) @ N, +N)
assert np.allclose(kron(X, Y, Y) @ N, -N)
assert max(kept_counts) == 3                         # at most three demands hold
assert broken_counts.count(1) == 32 and broken_counts.count(3) == 32   # always odd
```

The drawer question is the door question behind `S*`; because the state `|+i>`
is genuinely complex, the file also checks that the *unconjugated* version
(`<+|S`, or `|+i>` transposed) is wrong — the conjugate matters. The sketchbook
draws the rewrite step by step, each an exact picture equality: two drawer boxes
(`S*`) on two legs slide to one leg and merge into a `Z`, and that `Z` may sit
on any leg. Then the closed pictures: all-door effects on `N` give `1/sqrt(2)` on
even parity and `0` on odd; with the `Z` in place, the parities exchange — the
sign lives exactly there.

*Pictures.* Each trio-mask's windows are one deal, and the product of its
questions sends `N` to `+N` for EVEN and to `-N` for the three ODD masks; a mask
is kept exactly when its windows' faces multiply to that sign, which is the
blackboard's `demands` — part (ii), as buttons. The pictures: six white, buttons
on the three ODD masks; the middle drawer turned, two buttons lifted off (their
ghosts, no arc), one left on the mask over the middle door; the left drawer
turned, Achilles's forecast beside it (`mended`: no button), the true frame with
the button crossing — a ghost on the middle ODD mask, an arc to the mask drawn
next to it. The flip the dialogue does not draw (the right door) sends it to
EVEN.

```python
assert np.allclose(op @ N, EIGEN[m.rule] * N)
assert [not P.broken(m, cs) for m in back] == demands(x, y)
assert c24.buttons == tuple(P.rule_mended("back", c42.counters, ["left drawer"])) == ()
assert c41.buttons == ("odd left", "odd middle", "odd right") and c43.buttons == ("odd right",)
```

## 5. `Round.py` — *The Round* (master diagram)

The seven-wire picture, played apart. The blackboard builds it honestly — three
player boxes `P = diag(H, HS*)`, the node on the legs, three CNOTs and two
crossings on the tally wire — and shows that for each dealt triple, with
`r = q_A or q_B or q_C` and the losing slip `<1|`, it is **the zero vector**;
with the winning slip `<0|` it equals the tally-less picture:

```python
assert np.allclose(round_state(qA, qB, qC, r, s=1), 0)                 # never a loss
assert np.allclose(round_state(qA, qB, qC, r, s=0), tally_less(qA, qB, qC))
assert not np.allclose(round_state(qA, qB, qC, 1 - r, s=1), 0)         # wrong slip: not zero
```

An undealt triple (`100`) is non-zero either way; the box `P` pinned between
`|q>` and `<q|` is exactly its block; and nothing travels — each player's answer
is `1/2, 1/2` for all eight triples. PyZX has no controlled box, so the
sketchbook draws the **four dealt instances** with `P` replaced by its block
(H = a Hadamard edge; HS* = a phase box then a Hadamard edge), the CNOTs copied
from Chapter 4's `Yoke.py`, the crossings free (connectivity is all that
counts). The losing slip evaluates to the zero picture (asserted with
`np.allclose(matrix_of(g), 0)`, since an equality-of-pictures test is not
available against zero); the winning slip matches the node-with-blocks picture
exactly.

*Pictures.* The slip is tied to the tally wire: its word is `r = q_A or q_B or
q_C`, and for every deal the counters the macro accepts are exactly the answers
with a non-zero amplitude — those it refuses have amplitude `0`, which is why
the losing slip gives the zero vector. The close-ups follow the three solos: the
attic's and the cellar's are strips, the box shut and full beside its opened
envelope and then opened the named way — the door's ghost in the side face and a
swing arc round its hinge, the drawer's ghost at the foot and a slide arc out to
the front, parallel to the pull and clear of the drawer drawn out (each ghost an
outline only: no fill, no knob), and a ring on the box's top for the hand that
holds it still; the
garden room's, a repeat, unmarked. Each is captioned by its room's name alone. On
the return the table is a strip: the spent boxes, left to right Achilles's, the
Tortoise's, the Anteater's, with the counters black, white, white by them and
the slip folded under its glass; then the glass slid off towards the case (its
ghost on the slip, a slide arc to where it is set aside, the same glass at the
same size, between the case and the slip) and the slip's computed word *odd*:
the dialogue's deal `011` and outcome `100`, amplitude `1/sqrt(2)`. This file
also checks the whole dialogue — 22 calls, 32 frames, 7 strips, 2 predictions,
all inside the challenge sections, the starred (keyed) form on exactly the first
picture of each kind, none of the forbidden views, the log's 88 lines — and that
every one of the Lua's 67 refusals is reached by a call of its own.

```python
assert (props.faults(props.halltable, "", state) == []) == live
total, frames, starred, log = props.check_dialogue(DIALOGUE, ORDINALS)
assert total == 22 and len(frames) == 32 and len(log) == 88
assert r0 == r1 and x1 < x0 and props.CX1 < x1 - r1 and x1 + r1 < props.SLIPX[0]   # the glass
assert drawn[0].startswith("\\draw[propchsevenghost") and "fill" not in drawn[0]  # outline only
assert min(seg_dist(q, a, b) for q in pts for a, b in edges) > 0.1          # clear of the drawer
```
