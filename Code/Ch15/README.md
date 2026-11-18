# Chapter 15 — *Searching in Square-Root Time*: companion code

There are five challenge files, one for each dialogue challenge. Each file checks its claim
**twice**. The blackboard does it with NumPy `assert`s. The sketchbook compares PyZX diagrams
as tensors through `tools/dfp.py`'s `matrix_of` / `same_picture`, which force
`strategy="naive"`, and it also derives each equality with the rewrite rules of
`pyzx.simplify`: `full_reduce` on a copy with the scalar kept, `spider_simp` for the fused
master, or the named rules `fuse` and `color_change` for the steps of the headline's picture
chain. Running a file is the book's analogue of type-checking.

```bash
# one challenge at a time
.venv/bin/python chapter15/Code/Ch15/SecondDrawer.py
# the whole chapter (must end "5 files, 0 failed")
bash check-code.sh 15
```

Each file also runs `pictures()`, at the end of its blackboard: the dialogue's prop pictures
are read out of `dialog.md` and `dialog.raw.md` through `tools/dfp_props.py`, and their state is
computed by `tools/props_ch15.py`, a transcription of `pdf/props/ch15.lua` (with the shared cloth
and Office kits). Since round 4 (*show the process*) there are eight macros: the four pictures
`\drawercase`, `\pegcard`, `\aboutcloth`, `\handsheet`, each now able to take one `step=` (drawn
with its marks: a dotted ghost where a part stood, a curved headed arc the way it went),
`marks=off` and, on the pegs and the sheet, `expects=` (a named wrong rule, drawn pale), and the
four strips `\drawercasesteps`, `\pegcardsteps`, `\aboutclothsteps`, `\handsheetsteps`, which
draw a move frame by frame. The dialogue holds 21 blocks, one call each, and 35 frames. The
transcription is pinned to the Lua chunk by chunk (190 chunks), to its four round-4 comments
word for word and to `ch15.tex` (56 chunks), so a changed Lua stops every file at import. It was
diffed against the Lua under texlua, line for line (every emitted picture line, every log line,
every fault): the dialogue's 21 calls (148 log lines, which are also the real TeX build's), one
call for each of the 68 fault sites (the Lua raises them through 5 `tex.error` calls), a sweep of 42,264 calls and 10,713 spelling variants
of the brackets, with no difference and no Lua runtime error. `pictures()` ties each frame to the
blackboard, in order: the state before and after each step is this file's model, each ghost
stands where the part stood before the step, each marks=off twin draws round 3's picture of the
same state line for line, and each prediction draws its wrong rule's state, which is not the
true one. The pegs' turn-about is $D$, inversion about the average. The case's windows read
$DO_w\ket s$. The cloth's set-aside halves count $D$'s sign, push by push, and the hand's angle
and tip are $(2k+1)\theta$ and $8\sin((2k+1)\theta)$, fold by fold. It also checks the dialogue's
rules: blocks, sizes and frames, captions in words only (each frame's, no echoes, no step word
repeated within a strip), the anchor phrases before and after each block, the forbidden places,
the starred pictures P1.1, P1.2, P2.2, P3.3 and P4.1, and clearances on drawn extents. This is a
structural check of the pictures, not a third proof.

**Pins (the cloth kit).** The shared cloth kit, `pdf/props/cloth.{lua,tex}`, is pinned in
`tools/props_cloth.py`, which every file imports through `props_ch15`; a change to the kit fails
every file of this chapter (see the note in `chapter10/Code/README.md`).

Five things hold in every file.

- **Scalars are exact.** As in Chapters 9–14, the "up to a non-zero scalar" licence is never
  used. A red point is $\sqrt2\ket0$ (dark) or $\sqrt2\ket1$ (lit), a green point
  $\ket0+\ket1$; a Hadamard **edge** (the yellow box, the collar) is exactly $H$, and two
  collars with nothing between them cancel to a plain edge. `add_power(n)` multiplies by
  $\sqrt2^{\,n}$ and `add_phase(Fraction(1))` by $-1$. The compensations, each commented
  where it is made, are:
  - `add_power(1)` for a hub of two ties with its pendant ($2^{-1/2}$): the bare controlled
    half-turn is $\mathrm{CZ}/\sqrt2$, the bare marker $O_w/\sqrt2$, the bare cloth build
    $D/\sqrt2$;
  - `add_power(2)` for a hub of three ties ($2^{-1}$), and `add_power(5)` for the CCZ
    gadgets, bare $2^{-3/2}\cdot2^{-1}=2^{-5/2}\,\mathrm{CCZ}$ (the files assert that 4 and
    6 are wrong);
  - `add_phase` for the $-1$ of the X-form turn-about, and for the marker of the first
    drawer in its half-ring form;
  - `add_power(10k-3)` and the sign $(-1)^k$ for the eight-drawer search after $k$ steps;
  - **none** for *The Second Drawer*: two points give $2$, two bare hubs give $\tfrac12$.
- **Top wire first.** The marked drawer $01$ is not a palindrome. The master's amplitude at
  `0b01` is $1$ and at `0b10` (the reading upside down) $0$; the half-ring on the wrong cord
  marks the third drawer and is caught.
- **Generic inputs where the claim is general.** Inversion about average is tested on a
  random **complex** vector (so that $2\bar v-v$ and $2\overline{\bar v}-v$ differ); the
  marker, the query box and "no odds change" on random complex states; every marker
  $w$ on two cords in both forms; the fold lemma on random angles; Lemma 6(1) on 200
  Haar-random methods with a workspace.
- **Non-Clifford turns.** The CCZ gadgets need eighths, so `full_reduce` cannot close them.
  Three-cord pictures are reduced on a copy and compared by `matrix_of`; the chapter derives
  that drawing by no rule, and the files say so.
- **No picture proves a bound.** The Turning Plane is not a string diagram: it is checked on
  the blackboard. The optimality sketch has no picture: `NoFasterHand.py`'s sketchbook
  computes two processes and compares their distance with the bound (the card's recorded
  translation exception, deviation (p)).

The files begin with a marker that turns one drawer over and changes nobody's odds, and a
turn-about that inverts every amplitude about the average. They prove that the turn-about is
the reflection $2\ket s\bra s-I$ in every size, in algebra and as a sum of pictures, and build
the fifteen-line Second Drawer, where one touch and one turn find one drawer in four exactly.
They then fold the same two reflections in a plane to get BBHT's $\sin((2k+1)\theta)$, check
it against an eight-drawer search drawn in dots, and end with the bound, computed but not
proved by any picture, that no way of touching does much better than the square root.

---

## I. `FourDrawers.py` — *Four Drawers, One Wish* (the classical cost; the marker; the even state)

> "A quarter of the time the first is the one, and I have it at once. The other three quarters, I guess among three and I'm right one time in three, which is another quarter. Half the time, all told." — Achilles

Achilles reads the Terms, reckons the plain way (half the time with one ask and a guess,
three asks to be sure), then plays the marker on the peg-card: the second peg slides under,
and the squares do not change. He concludes that the touch is wasted.

The blackboard checks Result 15.1 (worst case $N-1$; mean $\tfrac{N-1}2+\tfrac{N-1}N$, $2.25$
and $32.48$, for every asking order of four drawers) and Result 15.2 **over every
deterministic decision tree** for four, five and six drawers: the most a tree with $k$ asks
can find is $k+1$ drawers, so one ask and a guess is exactly $2/N$, and certainty needs
$N-1$. It checks Result 15.3 for $n=1,2,3$ on every drawer and a random complex query state,
with the negative that a flag at $\ket+$ kicks nothing back, and Result 15.4 for $n\le6$.

```python
assert best == min(k + 1, N)                                       # every tree, N = 4, 5, 6
assert np.allclose(U @ np.kron(psi, MINUS), np.kron(O @ psi, MINUS))
assert not np.allclose(U @ np.kron(psi, PLUS), np.kron(O @ psi, PLUS))
assert np.allclose(touched ** 2, pegs ** 2)                        # the squares haven't changed
```

The sketchbook draws the marker for all four drawers in its Z-form (the controlled half-turn
with half-rings) and its X-form (red $\pi$s either side), derives that they are one process
(the first composed with the mirror of the second `full_reduce`s to two bare cords, scalar
$1$), shows a red point through a collar is a green point, runs the basis-point check of
the marker for $01$ (price $2^{-1/2}$, sign $-1$ on $01$ only), and draws Result 15.3 itself:
the query box on two rods, built from CCZ gadgets, with a green $\pi$ point on its flag,
equals the marker with the point passing by.

```python
assert reduces_to_bare_cords(both, 2)                              # Z-form . X-form^dagger = I
assert same_picture(fed, scaled(pair, -1, sign=sign))              # the basis-point check
assert same_picture(scaled(query_picture(w, 1), 5), scaled(marker_beside_flag(w, 1), 1))
```

**The pictures.** There are three, all in the First Challenge. P1.1 shows the case sealed. P1.2 is the even row of pegs on the card, $2\ket s$. P1.3 (F03, round 4: `\pegcard[step=touch the second]{even row}`) is the touch, the first instance of the marker, as a marked step frame: before it the even row, after it $(1,-1,1,1)=O_{01}(1,1,1,1)$, never $O_{10}$'s; the second peg's ghost stands where it stood, and one dotted slide arc runs beside its slit, on the slit's right, from its old height to its new one, through the middle line it passes. Every $w$ gives the dump `touch the <ord>` equal to $O_w(1,1,1,1)$, and the squares of the pegs still sum to four. Its twin with `marks=off` draws round 3's P1.3 line for line. The card's slits, notches and middle line are the same in all three pictures, and every peg stays inside its slit. The keyed words of P1.1 and P1.2 clear the drawing, each other and the frame. Their node boxes clear by at least $0.001$ (a peg's word against peg 4's shadow) and their letters by more than $0.03$.

```python
peg_step(p, ones, marker(0b01, 2) @ ones)           # before, after; ghost where it stood; the arc
assert [xy for *_, xy in p.ghosts] == [vb.pegs[k][1:] for k in moved]
assert PC.drawn(tw) == PC.drawn(r3) == PC.drawn(p)  # the marks=off twin is round 3's picture
```

## II. `TurnAbout.py` — *The Turn-About* (the headline; *The Theorem, Twice*)

> "...Oh. All the odds are on the second drawer." — Achilles

The figure sounds once: touch, then turn-about, and the pegs go to nothing, two, nothing,
nothing. Achilles says the touch is a formality; the Tortoise has him turn the even row
about, and nothing moves. On the cloth her build lets the even setting through and turns a
tilted one over, and the tilted row on the pegs agrees.

The blackboard proves Result 15.5 for $n=1,\dots,6$ against Grover's own entries, with the
negative that $I-2\ket s\bra s$ (the sign no odds reveal) and the X-form without its $-1$
both fail; inversion about average on a random complex vector, $D\ket s=\ket s$, $Du=-u$,
$D^2=I$; §15.7 Step 2's $-(I-2\ket{0\dots0}\bra{0\dots0})=R$; Result 15.6 for every $c$;
Result 15.7 (i); the pegs of the challenge; and the eight-case parity identity behind the CCZ
gadgets.

```python
assert np.allclose(W @ grover_R(N) @ W, D) and np.allclose(2 * proj - np.eye(N), D)
assert not np.allclose(np.eye(N) - 2 * proj, D)                    # the sign
assert np.allclose(D @ v, 2 * vbar - v)                            # v random complex
assert not np.allclose(D @ v, 2 * np.conj(vbar) - v)
assert np.allclose(D4 @ pegs, [0, 2, 0, 0])
```

The sketchbook walks the chapter's picture chain for $n=2$ and $n=3$ one picture at a time:
the tall box as bare cords minus $2^{1-n}$ lit effects beside lit points; the bare-cords term
`full_reduce`d to bare cords; each red $\pi$ **fused** (named rule) into its lit effect or
point, leaving it dark; each dark end **colour-changed** (named rule) through its collar into
a green one; and $2^{1-n}$ (green effects beside green points) minus bare cords is $D$. The
middle alone is checked the same way: two red $\pi$ dots reduce to bare cords, the red
$\pi$s fuse into the lit ends to leave dark ones, and $2^{1-n}$ (dark effects beside dark
points) minus bare cords is $R$ (not $-R$), with $WRW=D$. For
two cords the cloth build is exactly $D/\sqrt2$, is derived equal to the X-form, and
composed with itself reduces to bare cords ($D^2=I$); $R$ is derived by feeding basis
points; the cloth claims are derived on all four settings. For three cords the CCZ gadgets
and the X-form turn-about are checked by matrix, and one step of the eight-drawer search
built from them gives $\sin3\theta$.

```python
assert simp.check_fuse(Bf, tgt, dot) and simp.fuse(Bf, tgt, dot)   # red pi into lit: dark
assert simp.check_color_change(Bf, v) and simp.color_change(Bf, v)  # dark + collar: green
assert np.allclose(2 ** (1 - n) * matrix_of(greens) - matrix_of(bare_cords(n)), D)
assert same_picture(fed, scaled(same, 0, sign=tilted))             # the cloth claims
assert np.allclose(matrix_of(scaled(G, 5)), CCZ)                   # 2^{-5/2}, checked not derived
```

**The pictures.** The Second has eight blocks and eighteen frames, each checked against the file's model in order. B04 (P2.1's place) plays the turn-about as the dialogue plays it, from the touched row $v=(1,-1,1,1)$: the tick at $\bar v=\tfrac12$ (no peg moved, no marks); the first, third and fourth pegs reflected about it, $(0,-1,0,0)$ (three ghosts, three arcs); then the second, $(0,2,0,0)=D v=2\bar v-v$ (one ghost, one arc). B05 is `does the finding`, Achilles's claim that the turn-about finds the drawer by itself, drawn as a live prediction before the Tortoise plays it: the card pale, the four pegs where he expects them, $DO_{01}\,2\ket s=(0,2,0,0)$, each a notch (0.17 cm, asserted over the 0.15 cm floor) from its true place; the true state is $D\,2\ket s=2\ket s$, and B06 shows it, not a peg moved. The rule's comment is the Lua's word for word, and its quotations are the dialogue's. B11 (P2.4's place) turns one, one, under, under, $2\ket{-+}$ with the top cord first (never $2\ket{+-}$), into $-u$: four ghosts, four arcs, the same slide arc as the touch's. On the cloth, P2.2 is the build as laid and B08's before-frame. B08 pushes the even setting through piece by piece (captions "dark through the tubes; the ends past the rings; the turn fallen away; the ends green again, at rest"); B09 and B10 push the tilted setting, the direction split around B09 with no word changed (T1). The model reads an end as a basis point through its collars: the tubes make it red, a half-ring sets a half aside when a lit end passes it, the controlled turn sets one aside when both ends are lit, and pushed one end at a time it leaves a half-ring on the other cord (`landed=lower`) which the lower end then passes; the far tubes make it green again. Each frame's ghosts are exactly the parts the push takes (the landed ring has none), each arc is level, on its cord's gap side, from its ghost towards the new place, clear of every cord, tie, ring and ghost. The even setting comes back unchanged with no half aside, the tilted with one: the sign $D$ gives $\ket{-+}$. Every marks=off twin is round 3's picture. On drawn extents, no tie of P2.2 meets a ring that is not its own, the hub and the bob clear every cord, and along each cord every piece clears the next. In the last frame (P2.3's state) the ends rest at $6.0$; the set-aside bob lies wholly below the linen and its shadow, inside the frame.

```python
assert np.allclose(p5.wrong.h, D4 @ O(0b01, 4) @ s2) and np.allclose(p5.wrong.h, p21.after.h)
assert DOES_THE_FINDING in PC.LUA_TEXT
cloth_step(p, states[k - 1], states[k])             # ghosts = what the push takes; arcs on the gap side
assert np.allclose(D4 @ ket, (-1) ** states[-1]["aside"] * ket)
```

## III. `SecondDrawer.py` — *One Touch* (the master)

> "Granted." — Genie
>
> "Upper dark, lower lit. The second drawer." — Achilles

Achilles rubs the lamp, his meta-wishes are referred, and he declines the plain question.
One passage: the upper pane stays plain, the lower gleams. The second drawer holds the key,
and the back opens on the cords.

The blackboard checks Result 15.8 for all four drawers, with the passage written as its
three factors and the marker counted once (the Terms' clause (iii), the Genie's G1); the
negative that the turn-about before the marker finds nothing; the blackboard proof of
Result 15.16; and one step on $N$ drawers, $(1-\tfrac4N)\ket s+\tfrac2{\sqrt N}\ket w$, with
$0.367$ and odds $0.135$ for sixty-four.

The sketchbook builds *The Second Drawer* vertex by vertex from the card's list, unfused
(upper row 4 edges, lower row 5, each hub's ties and pendant 3 + 3, six of them collars),
with no compensation: exactly $\ket0\ket1$. `spider_simp` fuses it to 12 edges with the same value;
`full_reduce` leaves two one-legged green dots behind collars, phase $0$ above and $\pi$
below, scalar $\tfrac12$. The same builder gives $\ket{10}$, $\ket{11}$ and $-\ket{00}$ for
the other drawers; the master is the composition of its three separately built pieces
(powers $-2+1+1=0$); and the ledger's three closings are derived to their numbers.

The chapter's picture proof of Result 15.8 (§15.3) runs for every drawer, both strands. On the
blackboard, $O_w\,2\ket s=2\ket s-2\ket w$ (not the upside-down pair), $D\,2\ket s=2\ket s$
($\tfrac12\cdot4-1$) and $D\,2\ket w=2\ket s-2\ket w$ ($\tfrac12\cdot2$), so
$DO_w\,2\ket s=2\ket w$. In the sketchbook the marker fed two green points equals two green
points minus the dark/lit pair that spells $w$ (a sum of pictures, compared by `matrix_of`);
the picture of green points closed on red $w$-effects beside red $w$-points is
`full_reduce`d to twice the pair (two closings of $\sqrt2$); the turn-about takes the green
points to themselves and the pair to the green points minus the pair, each derived; and the
difference is the pair, $2\ket w$.

```python
assert np.allclose(marker(w, 4) @ two_s, two_s - two_w)            # the marker: green minus the pair
assert np.allclose(lhs, matrix_of(greens).ravel() - matrix_of(pair).ravel())
assert np.allclose(matrix_of(on_greens) - matrix_of(on_pair), matrix_of(pair))   # 2|w>
assert np.allclose(matrix_of(G).ravel(), ket(0, 1).ravel())        # no compensation
assert np.isclose(v[0b10], 0)                                      # upside down: nothing
assert G.num_edges() == 15
assert h.phase(a) == 0 and h.phase(b) == 1                          # after full_reduce
assert np.isclose(h.scalar.to_number(), 0.5)
assert same_picture(whole, G)                                      # even state . marker . turn-about
```

**The pictures.** The Third has three blocks and six frames. B12 (P3.1's place, captions "the case as set on the table; upper plain, lower gleaming") shows the case as set, then after the one passage: the windows read upper dark, lower lit, $DO_{01}\ket s=\ket{01}$, top cord first, and with the marked drawer set to each $w$ in turn they read $\ket w$; the passage draws no marks (the gleam shows it). B13 (P3.2's place) begins after the passage: the first drawer pulled out (no marks: it comes out 0.18 cm, under the 0.20 cm an arc needs), pushed shut (its front's dotted ghost exactly where the frame before drew it), and the second drawn out with the key on its floor. Each frame's state is the file's case model, clause by clause, and every marks=off twin draws round 3's picture. P3.3, the counted frame, is round 3's call, unchanged and unmarked; `step=back open` is accepted only in the back view and never draws a mark, and keyed it is refused. P3.3 is the inside from behind: the master's fifteen lengths, by name. The upper row has four cords, the lower five, and the two hubs have three ties and pendants each. The six collars sit on lines 1, 2, 4, 5, 7 and 9, and every colour and bead turn is checked. Each hub stands straight between its rings, the marker's bob points towards the end-rings, the turn-about's towards the windows, and the lower window's back is lit. No tie meets a cord or another tie. The eighteen Genie lines are frozen, word for word, all of them in the Third section, and no picture comes before `Granted.`

```python
assert (what, k) == ("drawer front", 1) and front[0] == 1 and np.allclose(rect, front[1:])
assert "step" not in PC.options(calls[2]) and p33.opts.get("round4") is None
assert p33.mark_tikz == [] and p33.ghosts == [] and p33.arcs == [] and not p33.marked
assert PC.faults(PC.run_frame, "case", "step=back open", before, False) == ["case-back-view"]
```

## IV. `TurningPlane.py` — *The Turning Plane* (two folds make a turn; when to stop)

> "All but a hair of eight. Just past upright." — Achilles
>
> "It goes round. Too many, and it turns away." — Tortoise

On the four-drawer sheet two folds turn the hand straight up. On the sixty-four sheet the
figure sounds six times, the tip climbing to just past upright; a seventh step, without the
figure's words, brings it down to seven and five eighths.

The Turning Plane is not a PyZX picture, so the plane is checked on the blackboard: the fold
lemma on random angles (with the negatives: the other direction, and once the angle instead
of twice); the plane kept by $O_w$ and $D$, restricted to $F_0$ and $F_\theta$; BBHT's Eq.
(2) and their recurrence against direct simulation for $N=4,8,16,64,4096$; $m=1,2,3,6,50,804$
and the failure bound up to $2^{30}$; the table's successes to the digits printed; the
sixty-four odds sequence; the dialogue's tip heights within a sixteenth of a thumb; the
non-monotone remarks for $2^{20}$; and Result 15.12 on samples of $(N,t)$.

```python
assert np.allclose(fold(b) @ Fa, turn(2 * (b - a)))
assert np.isclose(v[w], np.sin((2 * k + 1) * th))                   # BBHT's Eq. (2)
assert [stop(N) for N in (4, 8, 16, 64, 4096, 2 ** 20)] == [1, 2, 3, 6, 50, 804]
assert abs(tip - float(said)) <= 1 / 16                            # the heights Achilles reads
assert tips[7] < tips[6]                                           # "It's come down!"
```

The sketchbook is the eight-drawer search drawn in dots (marker $011$, CCZ gadgets): after
one and two steps its marked amplitude is $\sin3\theta=0.883883$ and $\sin5\theta=0.972272$,
and every other drawer $\cos((2k+1)\theta)/\sqrt7$ — the diagram computes the state, the plane
predicts it. The four-drawer master with a second step appended gives $\tfrac12$ at $01$ and
odds $\tfrac14$ everywhere, the overshoot in pictures, derived by `full_reduce`. A red
$w$-effect closed against green points measures the tilt, $\sin\theta=2^{-n/2}$.

```python
g = scaled(eight_drawer_search(k), 10 * k - 3, sign=k % 2)
assert np.isclose(v[0b011], np.sin((2 * k + 1) * th))
assert np.isclose(v[0b01], 0.5) and np.allclose(np.abs(v) ** 2, 0.25)   # two steps, four drawers
```

**The pictures.** The hand on its pin is the state in the turning plane. For four and for sixty-four drawers and $k=0,\dots,7$, the angle is $(2k+1)\theta$, and it equals $(DO_w)^k\ket s$ simulated. The tip is `"%.2f" % (8 a_k)`, and the holes are the places folded from and to, in order. The Fourth has six blocks and seven frames, one fold to a frame: F29 the marker's fold on the four-drawer sheet, $\theta\to-\theta$ ($-30^\circ$), and F30 the turn-about's fold, $-\theta\to3\theta$ ($90^\circ$), each at the moment told; F33 and F34 the seventh step (captions "the hand down below the pin; turned too far"), $13\theta\to-13\theta\to15\theta$ ($107.71^\circ$, tip $7.62$, turned away). Each fold has the hand's ghost where it lay and one swing arc round the pin, the short way, the arc's ends on its circle inside the sweep, crossing exactly the pencil lines the hand passes (computed from the angles: the unmarked line, and for F30 the even line too), on the sheet and clear of the hand, its ghost, the holes and the pin, never over the ghost. F32 is `surer`, a live prediction: the hand upright at $90^\circ$, tip eight, over the sheet with the holes it had before the step, 0.32 cm from the true tip (asserted over 0.3 cm); the rule is refused at every earlier step, where the next step raises the tip ($k<6$), and its comment is the Lua's word for word. Every marks=off twin is round 3's picture. The pin and the marked and unmarked lines stay fixed, and the fresh sheet's lines sit at the first sheet's places. Every hole and tip stays inside the sheet. No picture falls between a figure's touch line and its turn line; B17 and B18 stay consecutive, B18 straight after "All but a hair of eight", with the Escher paragraph between them. P4.1's keyed words clear everything. `unmarked line` and `even line` are held on their letters against the sheet's right shadow strip, because their node boxes' inner sep reaches that shade.

```python
c30 = sheet_fold(f30, -th4, 3 * th4, "about", th4)  # ghost, one swing arc, the lines it crosses
assert c["crossed"] == lines_passed(a0, a1, th)
assert min(c["sheet"], c["hand"], c["ghost"], c["holes"], c["pin"]) > 0.03 and c["passed"] == ()
assert got == (["sheet-rule-rises"] if rises else []) and rises == (k < 6)
```

## V. `NoFasterHand.py` — *No Faster Hand* (the fair share)

> "Four touches, one hand: not enough. Five, one and a quarter hands: still short. Six, one and a half: enough. That's the six we found." — Achilles

With nothing marked the hand stays on the even line. A touch pulls it at most twice the
touched drawer's height, each peg is an eighth, the tip must travel a little under one and a
third hands: five touches are short. Any way of touching leaves some drawer with no more
than its fair share. The Tortoise: the people who counted touches proved it properly.

The blackboard runs Result 15.13 (BBHT's Lemma 6(1)) on 200 Haar-random methods with a
workspace, touching with the marker or with the query box and its flag in whatever state the
method leaves it, for every $w$; the fair share and Cauchy–Schwarz; the counting steps of
Result 15.14's proof on each method; the bound $181$ for $2^{20}$ and Grover meeting it up to
$2^{30}$; Grover on sixty-four drawers ($0.496$, $1.366$, $1.323$; five short, six enough; the
ruler's ten and a half thumbs). BBHT's Theorem 7 is checked by its ratio: Grover's steps for even
odds over $\lfloor\sin(\pi/8)\sqrt N\rfloor$ fall ($1.083$, $1.028$, $1.026$), and at $2^{40}$
the ratio is $(\pi/8)/\sin(\pi/8)\approx1.026$ to within $10^{-4}$.

```python
assert dist <= 2 * np.sum(alpha[:, w]) + 1e-9                      # Lemma 6(1), random methods
assert S[least] <= T / N + 1e-12                                   # the fair share
assert int(np.ceil(bound(2 ** 20))) == 181
assert 4 * 0.25 < gap and 5 * 0.25 < gap <= 6 * 0.25              # five short, six enough
```

The sketchbook is the recorded translation exception: no picture proves a bound. It builds
*The Second Drawer* and the same picture with no drawer marked, where the collars either side
of the missing marker meet and are drawn as one plain edge; bare, the unmarked picture is
exactly $\sqrt2\ket s$, derived by `full_reduce` to two green points. It is the master with
the marker taken out (five dots and six edges fewer). The chapter's route to its value is
also checked: the turn-about's middle is $R/\sqrt2$; $R$ is $\tfrac12$ (dark effects beside
dark points) minus bare cords; a dark effect closed on a dark point `full_reduce`s to $2$;
the middle fed two dark points gives $2^{-1/2}$ times them, derived; and the last collars
colour-change them green. The two processes end with difference exactly
$\tfrac12(-1,1,-1,-1)$, distance $1$, which is BBHT's bound $2\cdot\tfrac12$ for one touch on
four drawers: attained.

```python
assert same_picture(scaled(unmarked, -1), scaled(greens, -2))     # both exactly |s>
assert np.isclose(dist, 2 * alpha)                                 # the bound, attained
```

**The pictures.** P5.1 is the hand back at the first dot, with no drawer marked, after the turn-about's fold: round 3's plain call, no step, no marks. It has not moved, because $D\ket s=\ket s$. Its tip reads $1.00=8\sin\theta$, and its holes are those of the seventh step's last frame, with none added. The Lua refuses a marker's fold when nothing is marked (`sheet-nomark`). This file also checks the whole dialogue:
- 21 blocks and 35 frames, asserted as the sums of the sections' tables, all inside challenge sections, none of them after P5.1;
- every stepped frame marked unless it is a prediction, no marked or predicted frame keyed, no strip starred;
- the starred set is exactly P1.1, P1.2, P2.2, P3.3 and P4.1 (predictions are never judged);
- the eleven Old fragments are gone from both files;
- the toccata's figure appears once in II and six times in IV, with one `Granted.`, the Genie's, in III, and three `Referred.`, all the Genie's;
- no picture falls inside a figure;
- "Let Bob draw" never occurs, and the signal grep finds nothing, comments included;
- `FAULT_CASES`: each of the Lua's 68 fault sites raised by its own call, with a message of its own (the strip faults by the strip alone; `strip-unknown` with the caption count it entails); every order of a refused bracket's items raises the same faults; a refused step draws no marks; the `;` bracket is order-free for all four macros and their strips, and an item given twice is refused.

```python
assert np.allclose(grover_D(N) @ s, s)
assert [c[0] for c in FAULT_CASES] == [k for k, _ in PC.TEX_SITES] and len(PC.ERROR_SITES) == 68
assert sorted(fn("; ".join(order))) == want, (case[0], order)
```
