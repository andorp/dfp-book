# Chapter 11 — *Copying Is Forbidden*: companion code

Five challenge files, one per dialogue challenge. Each checks its claim **twice**: once
on the blackboard (NumPy `assert`s) and once in the sketchbook. The sketchbook compares
PyZX diagrams as tensors through `tools/dfp.py`'s `matrix_of` / `same_picture`, which force
`strategy="naive"`, *and* derives each equality with the rewrite rules of `pyzx.simplify`.
Running a file is the book's analogue of type-checking.

```bash
# one challenge at a time
.venv/bin/python chapter11/Code/Ch11/OwnBasis.py
# the whole chapter (must end "5 files, 0 failed")
bash check-code.sh 11
```

Four things hold throughout.

- **Scalars are exact.** A likeness is an amplitude and the master is a probability
  distribution, and the headline turns on the gap between a number and its square. So the
  "up to a non-zero scalar" licence of Chapter 8 is not used anywhere. PyZX's one-legged
  dots are $\sqrt2$ times a unit state (a red point of phase $a\pi$ is $\sqrt2\ket a$, a green
  point of phase $0$ is $\ket0+\ket1$). Every point therefore carries `add_power(-1)`, the
  $\tfrac1{\sqrt2}$ the chapter writes in front of it. Each closed likeness loop carries
  `add_power(-2)`. The three bare spider-CNOTs carry `add_power(3)`, the chapter's $2\sqrt2$.
  The doubled master carries `add_power(-2)`, because the doubled pair is $\tfrac12\,$cup
  $\otimes$ cup. The copier, the phase-free green dot with one leg in and two out, is exactly
  $\delta$ and needs no compensation.
- **Doubled pictures are hand-built.** PyZX 0.10.6 refuses ground vertices. So, as in
  Chapters 9 and 10, every doubled picture draws a wire beside its mirror copy, discard is a
  cap joining the two copies, and reading is the read-spider. The copier is real, so its
  conjugate copy is the green fork again. A non-real box is used on the blackboard to show
  that the mirror copy really is conjugated.
- **Double edges go through an identity dot.** PyZX keeps one edge per pair of dots, so
  where the copier meets its mirror along two wires, the second wire runs through a
  phase-free green dot with two legs.
- **Generic inputs.** Every likeness is taken between **non-real** states, so the mirror's
  conjugation is exercised. The dephasing is run on a non-real, non-diagonal $\rho$. The
  deleter is fed a state that is neither real nor on the equator.

The files start from what a copier *can* do, which is to copy its own basis, and show
what it does to anything else: a green point fuses into a cup, a pair instead of two
copies. They then prove that no unitary machine can do more, because one loop would have
to equal two. The next two files follow a copier through the doubled world. A copy thrown
away is a look nobody read, and a copy un-made is only a copy moved into the machine. The
last builds the eleven-line Forbidden Telegraph, counts its edges, and shows by rewriting
that at the cross setting the near record falls away from the far ones. So the only
telegraph a copier could make is one no linear picture can draw.

**The prop pictures.** The dialogue draws its props with four macros from
`pdf/props/ch11.tex`: the black slab with the fresh pair, the copying press, the tray, the
pewter counters and the viewer (`\presstable`), the Tortoise's string, rings, points, cards,
loops and knot on her drawing-cloth (`\copycloth`, drawn on the shared cloth of
`pdf/props/cloth.lua`), and, from round 4 (*Show the process*), their step strips
(`\presssteps`, `\copysteps`), a row of frames each one step on from the last. Their Lua
rules are transcribed once in `tools/props_ch11.py` (no matrices there). For the table it
holds the places read from the Lua and the computed operations: a reading spends its stone,
draws its reflection tight and lays the tray's top counter opposite it; the far stone goes
into the hopper unread; the one strike drops two struck stones, their reflections slack. For
the cloth it holds the six scenes and their moves (thread, lift off, slide the card along,
lift the cards off, a faithful pressing, fold, slide the forks together, pull, slide
together, draw apart), laid out and framed through the book's one transcription of the cloth
kit, `tools/props_cloth.py`. Round 4 adds the bracket's `;` grammar (one bare item and
`step=`, `marks=`, `expects=`, `after=`, `from=`, in any order), the **step** (the frame drawn
from the state before it, with the **marks** of the move: a ghost where a thing was, a short
curved arc with a head for its path), the two **wrong rules** (`expects=copies anything`,
`expects=deleted`: Achilles's predictions drawn pale, never keyed, never marked), and the
strips. The Lua's fifty-five fault sites are mirrored one for one (twelve direct ones outside
its two wrappers, fourteen of the table's, twenty-nine of the cloth's), each message byte for
byte. The transcription was proved against `texlua`: 9335 calls (the dialogue's thirty-four,
both files, and sweeps of every scene, step, prediction and strip), every emitted line equal,
every fault site hit; and every marked frame with `marks=off` added draws round 3's picture
byte for byte. `check_lua` then holds it there: the Lua's code (107 chunks), the `.tex`'s
definitions (61) and the three rule comments that quote the dialogue are pinned by hash.

**Pins (the cloth kit).** The shared cloth kit, `pdf/props/cloth.{lua,tex}`, is pinned in
`tools/props_cloth.py`, which every file imports through `props_ch11`; a change to the kit fails
every file of this chapter (see the note in `chapter10/Code/README.md`).

Each file reads the pictures of its own section of `dialog.md` through `tools/dfp_props.py`
and `PC.check_section`: the calls per block (`[1,2,2]`, `[1,1,1]`, `[1]`, `[3]`, `[1,1,1,2]`;
seventeen calls) and the frames (6, 4, 3, 3, 6; twenty-two), the same in `dialog.raw.md`; at
most four frames to a block, one frame height and one macro family to a block; the captions
(words only, from a forbidden list with no quantity word and no Stage-D word, never a cloth
word, never a sentence someone speaks); the places each scene may appear; the paragraph each
block follows and the one after it; the log each call writes, line by line; and every
frame's state, each ghost's place and each arc's ends. Every frame of a strip is checked to be
the single call `[<the steps before>; step=<its step>]`, and every marked frame's marks-off
twin to be round 3's picture. Each file also **ties the transcription to its own
mathematics**, NumPy against its blackboard. This is a structural check, not a second proof.
A red point is a stone settled plain, a green point a stone settled crosswise; a separate
piece of cord is a factor of a product. A slack reflection is a stone that is $\tfrac12 I$
averaged over every record laid elsewhere, a tight one a stone that is pure given the counter
laid opposite it. `ForbiddenTelegraph.py` checks the whole dialogue besides: the keyed rule
(true frames only), the predictions, the ten marked frames, the counted frames, every mark
inside its picture, the log (77 lines), the fifty-five fault sites one by one, and the
transcription's constants against the Lua itself.

---

## I. `OwnBasis.py` — *The Press Copies Its Own* (no-cloning from linearity; complementarity)

> "...Oh. It doesn't copy the crosswise stone. It makes a *pair* of it." — Achilles

The Crab reads the proving-book. Plain stones come out as two alike. Crosswise stones come
out as a matched pair: each struck stone is a coin toss at cross, yet the two always agree
for one settling and always disagree for the other. Read plain, they come out both high or
both low, never one of each. The Tortoise threads a red point through the forked ring (two
red points) and a green point into it (one bend).

The **blackboard** checks the following.
- $\delta\ket0=\ket{00}$ and $\delta\ket1=\ket{11}$ exactly.
- $\delta\ket\psi=\alpha\ket{00}+\beta\ket{11}$ on a non-real $\psi$. It is not
  $\psi\otimes\psi$: its matrix of coefficients has determinant $\alpha\beta$ and rank 2.
- $\delta\ket\pm=\ket{\Phi^\pm}$.
- $\mathrm{CNOT}\,(I\otimes\ket0)=\delta$: the blank is inside the copier.
- Result 11.1, twice. Once for the basis cloner $\mathrm{CNOT}_{0\to2}\mathrm{CNOT}_{0\to1}$,
  and once for a **random non-real linear machine** with a two-wire apparatus, fixed only on
  the two plain inputs. In both cases the $\ket{01}$ part is zero, where a clone's is
  $\alpha\beta\ket{A_\psi}$.
- The Sketchbook sidebar's remark: a red point with a turn $\theta$ has
  $\alpha\beta=\tfrac14(1-e^{2i\theta})$, and is copied only at $\theta\in\{0,\pi\}$.
- Every proving-book statistic, for every settling the scene allows.
- A negative: Achilles's guess, "two crosswise stones", would read $(+,+)$ every time.

```python
assert np.allclose(part01 @ got, 0)                          # no |01> part at all
assert np.allclose(part01 @ clone, a * b * A_psi)            # a clone's is ab A_psi
assert np.allclose(readings(DELTA @ PLUS, cross), [[0.5, 0], [0, 0.5]])    # always agree
assert np.allclose(readings(DELTA @ MINUS, cross), [[0, 0.5], [0.5, 0]])   # always disagree
assert np.allclose(readings(DELTA @ s, plain).ravel(), [0.5, 0, 0, 0.5])   # never one of each
```

The **sketchbook** first checks that PyZX's green dot with one leg in and two out is
$\delta$. Then come three pictures, each derived with `full_reduce` and scalars kept.
- The CNOT with its blank loaded reduces to the copier.
- A red point of phase $0$ or $\pi$ through the copier is two red points (four vertices, two
  edges left). A red point with phase $\tfrac13$ is not copied.
- A green point fuses into $\tfrac1{\sqrt2}\,$cup $=\ket{\Phi^+}$. The reduced picture is a
  bare edge between two outputs, which evaluates as a state. With phase $\pi$ it gives
  $\ket{\Phi^-}$.

A green point of phase $\tfrac13$ comes out with Schmidt rank 2.

```python
assert same_picture(fed_copier("red", a), two_points(a))     # red copies
assert same_picture(fed_copier("green", 0), bell_picture(0)) # green fuses: 1/sqrt2 cup
h = g.copy(); simp.full_reduce(h); assert same_picture(h, bell_picture(0))
```

*Pictures.* The slab as the press is set down, keyed: the near stone and the far stone,
each restless over a slack reflection, the press at the far end, the counters stacked in the
tray, the viewer beside it. It is one of the two **counted frames**: unmarked, seven lines
(each stone and its reflection two, each stacked counter one: the discs stand up out of the
tray, each a little up and every other one a little aside, so they count at print size, and
the ruled fallback of four, the discs counted by the dialogue instead, is unused). The file
asserts the stack countable: consecutive disc faces at least 0.05 apart, every face above the
tray's front wall. The file checks the places the card's L1 fixes and the
drawn extents, reflections included, each a positive gap, and that each stone of the fresh
pair is $\tfrac12 I$ alone. Then the fork with a red point, keyed, and the point threaded, a
step: two red points either side of where the fork stood, each on a cord of its own that
keeps its course (within a twentieth of its length), a product of Schmidt rank 1; its marks
are the point's ghost where it lay, the fork's ghost where it stood (into her shell; no cord
is ghosted; clipped away round the two red points that came to lie over its place, which
the file checks), and a slide arc above the in-cord. Then, at the naming beat, Achilles's
prediction recalled, pale (`copies anything`: two green points where the red ones lay, two
pieces, the product $\ket+\ket+$, which the copier does not make), and the green point's
process as a strip of two: threaded, one green ring (the point's ghost, a slide arc stopping
short of the ring), then the bare ring lifted off, one bend with both ends loose (the fork's
ghost). One piece, which is $\delta\ket+=\tfrac1{\sqrt2}\,$cup exactly (the cup unnormalized,
Schmidt rank 2).

```python
assert PC.lines_drawn(t11) == 7 and PC.lines_drawn(t11, tray=False) == 4    # a counted frame
assert PC.frame_of(t)[:3] == PC.frame_of(PC.copycloth("thread", RED))[:3]  # marks off: round 3's P1.3
wrong = pieces(w14p.cloth)                                      # the pale frame: two pieces
assert not np.allclose(DELTA @ PLUS, kron(PLUS, PLUS).ravel())   # not two green points
(piece,) = pieces(c16.cloth)                                     # green: one bend, one piece
assert np.allclose(DELTA @ PLUS, CUP / np.sqrt(2))
```


## II. `NoCloning.py` — *One Loop, Two Loops* (the headline; *The Theorem, Twice*)

> "A half against a quarter." — Achilles

The Tortoise slides the press card and its mirror card off the string, which leaves one
loop. Beside it she lays the two separate loops a faithful pressing would leave. The Crab
tries to repeat her sentence word for word, and it comes out changed.

The **blackboard** runs the headline with a random $8\times8$ unitary and non-real
$\varphi,\psi,A$.
- $\langle U(\varphi0A),U(\psi0A)\rangle=s$, with the blank's loop and the apparatus's loop
  each equal to 1.
- For 200 random apparatus finals, $\lvert a\rvert\le1$ and the two-loop value is
  $s^2a\neq s$.
- $s=s^2$ only for $s\in\{0,1\}$.
- Achilles's numbers, taken as probabilities: $\lvert\braket{0|+}\rvert^2=\tfrac12$ against
  $\tfrac14$ after pressing. The works' loop can only shrink the $\tfrac14$, and the unitary
  keeps the $\tfrac12$. (The amplitude is $\tfrac1{\sqrt2}$; §11.2 reconciles the two.)
- §11.2's linear map with $L\ket0=\ket{00}$ and $L\ket+=\ket{++}$ has
  $\lVert L\ket1\rVert^2=3-\sqrt2$.
- The basis cloner satisfies $s=s^2a$ trivially on $\ket0,\ket1$ and does not clone $\ket+$.
- $\delta^\dagger\delta=I$, but $\delta\delta^\dagger\neq I$.

```python
assert np.isclose(np.vdot(U @ x, U @ y), s)                  # the press loses nothing
assert abs(s * s * a) <= abs(s) ** 2 < abs(s)                # two loops never equal one
```

The **sketchbook** has five parts.
- (a) The copier meets its mirror along two edges (one through an identity dot) and leaves
  a bare wire. `full_reduce` derives it: two vertices, one edge.
- (a′) The same holds for every box. A seeded random Clifford+T $U$ on three wires,
  composed with `U.adjoint()`, reduces to three bare wires with scalar exactly $1$.
- The headline picture itself: states $\psi,0,A$, then $U$, then $U^\dagger$, then the
  mirrored effects $\varphi,0,A$. It evaluates to $s$. After `full_reduce` nothing of $U$
  is left: the reduced picture matches, vertex for vertex, the reduction of the picture
  drawn with no box at all. The unconjugated (transposed) effects give a different number.
- (b)–(d) One loop through the copier and its mirror (green point $\tfrac13$, effect
  $-\tfrac14$) is exactly $s$. The unconjugated effect is not. Two separate loops are exactly
  $s^2$, and one loop is not two. They agree only in the orthogonal case (red $0$ against
  red $\pi$, both $0$) and the same-state case (both $1$).
- (e) `full_reduce` takes each loop picture to the empty picture, with scalar $s$ and $s^2$.

```python
h = UU.copy(); simp.full_reduce(h); assert same_picture(h, bare_wires(3))
assert np.isclose(value(one_loop(Fraction(1, 3), Fraction(-1, 4))), s1)      # one loop: s
assert np.isclose(value(loops(Fraction(1, 3), Fraction(-1, 4), 2)), s1 ** 2) # two loops: s^2
```

*Pictures.* The likeness loop, keyed: a point, the press card, its mirror card, the other
point, on one closed loop (the caption now says what the picture shows: "the cards between
the points"). Then the slide as a strip of two: the card slid along, up onto the mirror card,
face to face (its ghost where it stood, a slide arc inside the loop below the cards), then
both lifted off together, the points still on the loop (both cards' ghosts, into her shell). Each card's ghost is broken where the string passes;
the file checks the break is a band across the whole ghost, wider than the cord and the
dotted line together.
The Lua refuses to lift the cards before they touch. On the blackboard the press with its
blank and works plugged in is an isometry $V$: face to face, $V^\dagger V=I$; the other way
round, $VV^\dagger$ is a projector, not the bare string. Last, after a faithful pressing, two
point-carrying loops and a small plain one, side by side and not touching: the likeness read
off the picture is $s$ to the number of point-carrying loops, $s$ before, during and after
the slide and $s^2$ after the pressing, and $s\neq s^2$ for a non-real $s$.

```python
assert (under[:3], over[:3]) == (("mirror card", 4, 0), ("card", 4 - PC.CARDTOUCH, 0))
assert PC.faults(PC.copycloth, "lift the cards off", LOOP) == ["loop-lift-early"]
assert np.isclose(np.vdot(phi, V.conj().T @ (V @ psi)), s ** n(c22a))   # face to face: V^dagger V
assert np.isclose(np.vdot(np.kron(phi, phi), np.kron(psi, psi)), s ** n(c23))   # pressed
```


## III. `NoBroadcasting.py` — *Eve's Tap* (copying is looking; no-broadcasting, easy half)

> "Nothing in them to spoil: they were settled already." — Achilles

The maker's letter says Eve's plain stones arrived as good as ever, and her crosswise
stones, settled either way, arrived "looked-at". The Tortoise lays the forked ring beside
its mirror, folds one copy onto its mirror twin, and the two forks fuse into one
four-legged ring. That ring is Chapter 9's look that nobody read.

The **blackboard** checks the following.
- Both marginals of $\delta\rho\delta^\dagger$ equal $\sum_iP_i\rho P_i$ on a non-real,
  non-diagonal $\rho$, and differ from $\rho$.
- The two copies, read plain, always agree: Eve's drawer holds the look.
- A diagonal mixed $\rho$ is broadcast but not cloned, as Barnum et al.'s correlated state
  $\sum_b\lambda_b\ket{bb}\bra{bb}$.
- The easy half of Barnum et al. in a **random non-real basis**: two commuting states are
  broadcast by that basis's copier and not cloned.
- $\ket0\bra0$ and $\ket+\bra+$ do not commute.
- The letter's statistics, for every settling. The customer's stone is left alone when it
  was settled plain. When it was settled crosswise either way it becomes $\tfrac12I$, a coin
  toss at cross.
- The doubling convention: a box acts on doubled wires as $K\otimes\bar K$. Plain
  $K\otimes K$ fails for a non-real $K$.

Barnum et al.'s "iff" is stated in the docstring and not tested.

```python
assert np.allclose(m, look) and not np.allclose(m, rho)      # copy-and-forget = dephasing
assert np.allclose(customer(np.outer(s, s.conj())), I2 / 2)  # crosswise: looked-at
```

The **sketchbook** builds the tap as a doubled picture: the copier on $e$, its conjugate on
$\bar e$, and a cap on the second copies.
- It equals the four-legged green dot exactly, and on doubled wires it dephases $\rho$.
  `full_reduce` derives it (five vertices).
- The read-spider, its record, and its mirror setting the stone again give the same dot.
  So does the non-destructive look with its record forgotten. Both are derived.
- Fed a doubled red point, the tap returns it unchanged. Fed a doubled green point, it
  returns $\tfrac12$ cup (grey). Both are derived.
- Capped on its output as well, the tap is the cap. The reduced picture joins two inputs,
  so it is checked as a state.

```python
assert same_picture(tap(), four_legged())
assert same_picture(read_then_set(), tap())                  # copy-and-forget = look-and-forget
assert same_picture(fed(tap(), VT.Z, a), grey())             # the crosswise stone turns grey
```

*Pictures.* One strip of three, its first frame keyed. The fork and its mirror: the mirror
string is the first one reflected top to bottom and runs the same way (a left-to-right
reflection would not match). Then the second copy folded onto its reflection at the right,
fork to mirror fork, with no mark (a curve beside the turn would read as a second fold). Then
the forks slid into one ring with two cords in from the left and two out to the right, each
fork's ghost where it stood and a slide arc from each, inside the wedge between the cords,
the two arcs mirror images, clear of every cord. On the blackboard copy-then-forget is the
dephasing, exactly; the doubling of a non-real map is $K\otimes\bar K$, and the transposed
pairing fails.

```python
assert sorted(far) == [0, 0, 6, 6]                               # two in, two out: a trace
assert (c32.marks, c32.mark_tikz) == ((), ())                    # the fold: no mark
assert np.allclose(sup, ring4)                                    # copy-then-forget: the ring
assert not np.allclose(np.kron(K, K.conj().T) @ vec(rho), vec(K @ rho @ K.conj().T))
```


## IV. `NoDeleting.py` — *Eve's Blanks* (no-deleting)

> "The tell always came out holding the stone she had blanked." — Crab

Eve's frame always gave a blank, and its tell always read as the blanked stone would have
read. The Tortoise pulls Chapter 8's three-cord knot, middle cord reversed, and it comes
undone to a crossing.

The **blackboard** checks the following.
- The frame is the swap of the second stone and the tell. Three cords tied the same way
  make one CNOT again, not a swap.
- $\psi\psi0\mapsto\psi0\psi$ exactly on non-real states. The output is never
  $\psi0\bar\psi$ and never unchanged.
- The loop form: $A_\psi=\psi$, $\braket{A_\varphi|A_\psi}=s$ and $s^2=s\cdot1\cdot a$.
- The letter's statistics, for all four settlings. The middle output is the blank, the first
  stone is untouched, and the tell, read at the settling's own setting, gives the stone's
  answer with probability 1.
- Pati–Braunstein's linear form: with the mixed input forced by linearity, the output is
  $\psi\,\Sigma\,(\alpha A_H+\beta A_V)$.
- The negative. A tell that ignores the stone ($A_H=A_V$, Achilles's "ballast") gives
  $(\alpha+\beta)\,\psi\Sigma A_0$, of norm $\ne1$. Non-orthogonal tells fail as well.

```python
assert np.allclose(FRAME @ kron(x, x, SIGMA).ravel(), kron(x, SIGMA, x).ravel())
assert np.isclose(np.vdot(A_phi, A_psi), s)                  # the tell holds the stone
assert not np.isclose(np.linalg.norm(bad), 1)                # a tell that ignores it: unfair
```

The **sketchbook** draws the knot as three bare spider-CNOTs with the chapter's $2\sqrt2$.
- It is exactly the crossing, it agrees with PyZX's own circuit CNOTs, and the all-one-way
  knot is not a crossing.
- `full_reduce` pulls it into three bare wires, two of them crossed.
- With $\psi,\psi$ and the blank plugged in, it gives exactly $\psi,0,\psi$.
- After `full_reduce` the three outputs are three separate pieces, with no path between
  any two.

```python
assert same_picture(knot(), crossing())
h = lhs.copy(); simp.full_reduce(h); assert same_picture(h, rhs)   # psi psi 0 -> psi 0 psi
```

*Pictures.* One block of three. The knot, keyed: the upper line loose from the left, the
lower from a red point, three cords across, green-over-red, red-over-green, green-over-red.
Then Achilles's prediction, pale (`deleted`: the stone's line stops where the knot was, a
fresh blank starts the upper line out of nothing, the tell runs on as it came in). Then the
knot pulled, a step: a plain crossing, the lines exchanging rows, the six knot rings' ghosts
where they stood and no arc. Each cord is a CNOT with green the control, and the three
alternating ones make the swap, Eve's frame, exactly; three the same way make one CNOT
again, and the Lua's pull refuses them (`knot-not-alternating`). The pale frame is the
deletion the theorem forbids: it would take two copies' likeness $s^2$ to $s$, while the
swap keeps $s^2$.

```python
assert rows == [(0, 0, PC.DELSTOP), (0, PC.DELBLANK, 6), (1, 0, 6)]   # stops; from nothing; runs on
assert np.allclose(knot, FRAME)                                # the drawn knot is the swap
assert np.isclose(np.vdot(knot @ ins[1], knot @ ins[0]), s ** 2)   # the swap keeps the likeness
```


## V. `ForbiddenTelegraph.py` — *The Forbidden Telegraph* (the master)

> "So Delft sees the same whatever I do here." — Achilles

Achilles reckons what a perfect copier would show: a telegraph. Then the real press runs
once, blind. The near stone is read at cross, and the two struck stones are read plain and
agree. The proving-book shows the same at either setting. On her cloth the Tortoise lays
the reading rings out: at plain they slide together into one green ring holding every
counter's cord. For the crosswise reading she lays them out afresh, a red ring at the near
end joined to the far green ring by two cords, and draws the red ring away; the two cords
fall away together.

The **blackboard** checks the following.
- $p(m,z_1,z_2)$ is $\tfrac12\delta_{mz_1}\delta_{z_1z_2}$ at plain and
  $\tfrac14\delta_{z_1z_2}$ at cross. The far marginal is $\tfrac12\delta$ at both, and the
  struck pair never shows one of each.
- At plain the pair matches the near reading. At cross the near record is independent of
  the far ones.
- The single run seen on the table has probability $\tfrac14$.
- The counterfactual ideal cloner is commented "not linear; it has no picture". It gives
  agreement $1$ at plain against $\tfrac12$ at cross. It tells apart the two mixtures of
  $\tfrac12I$, which $\delta$ cannot.
- The rank across the cut $m\,|\,z_1z_2$ is 2 at plain and 1 at cross.
- Result 11.8, parts (i)–(iv):
  - (i) rank $\le2$ for 50 random $D$;
  - (ii) the clones span 3 dimensions;
  - (iii) on a wire of dimension $r$ they span $r(r+1)/2$ dimensions, for $r=1,\dots,4$;
  - (iv) $(I\otimes\delta)\,$cup $=\ket{000}+\ket{111}$, of rank 2 across every cut.

```python
assert np.allclose(real["plain"].sum(axis=0), real["cross"].sum(axis=0))   # no signal
assert np.isclose(agree["plain"], 1.0) and np.isclose(agree["cross"], 0.5) # ideal cloner: a telegraph
assert np.linalg.matrix_rank(clones) == 3
```

The **sketchbook** builds the master as a doubled diagram at both settings. At cross,
$a$ and $\bar a$ reach the near read-spider through Hadamard edges.
- (a) It is exactly the eight-entry distribution at each setting (`add_power(-2)`), and it
  matches the drawn right-hand side of Result 11.7.
- (b) `num_edges() == 11`. The comments name the three classical edges.
- Forgetting $m$ gives the same picture at both settings: no signal. The reduction is
  derived.
- (c) After `full_reduce`, $m$ has **no path** to $z_1,z_2$ at cross (a point on $m$ and a
  bend on $z_1z_2$). At plain one green dot holds all three.
- (c′) The Tortoise's cloth. With the cross reader drawn as a red dot (colour change),
  fusion alone (`spider_simp`, whose handling of the two parallel red–green edges is Hopf)
  leaves the red ring holding only $m$ and one green ring holding $z_1,z_2$. At plain it
  leaves a single green ring with all three cords.
- (d) The copier on one half of a cup fuses into the three-legged green dot.

```python
assert same_picture(master(s), expected(s)) and master(s).num_edges() == 11
assert not connected(h, m, z1) and not connected(h, m, z2)   # at cross: disconnected
assert set(h.neighbors(red)) == {m} and set(h.neighbors(green)) == {z1, z2}   # the cloth
```

*Pictures.* First a strip of two from the slab as set down: the near stone read high at
cross, spent, its reflection tight, its counter laid face-up opposite it, the viewer at the
window end (a pale ghost of the viewer by the tray; no ghost for the counter taken off the
stack, which would read as one more counter; no arc),
and **the far stone beside it, still restless, its reflection slack**; then the far stone
lifted into the hopper (a dotted outline of the pebble where it stood, its reflection not
ghosted, a lift arc from above the outline up and over into the cup). Then the slab just after the strike, computed with `marks=off`
(the same picture as round 3's, and as the strike with its marks on, less the lever's swing
and the two falls): the far stone's place bare, two struck stones below the chutes, slack.
It is the second counted frame: nine lines (with P1.1's seven, all eleven of the chapter's
wires). The full run of l.436 is checked on the transcription and not pictured. The reflection
rule is tested on every stone shown: slack is $\tfrac12 I$ averaged over every record laid
elsewhere, tight is pure given the stone's own counter. At cross each struck stone is
$\tfrac12 I$ on either near outcome; at plain it is $\tfrac12 I$ only averaged and pure given
the near record. The far stone after the near reading is $\tfrac12 I$ averaged at both
settings, and pure but setting-dependent given the record: a reflection that tightened would
be the telegraph. So it stays slack in every table frame after the near reading, the rule
this file asserts on every frame and every reachable step. The run has probability
$\tfrac14$; the far counters never disagree. Then the reading rings: at plain, keyed, one
green ring with every counter's cord, and the three records always agree; at cross, the red
ring joined to the far green one by the link, and then drawn away, a step (its ghost, a swing
arc above the row), holding only the near counter's cord. The distribution is the product
over the picture's pieces, and each picture's pieces fail on the other setting's
distribution.

This file also checks the whole dialogue: seventeen calls, twenty-two frames, four strips,
two predictions (each followed by the true step it foresees), ten marked frames, six keyed
frames (true frames only), the log of 77 lines in both files, the nineteen dialogue phrases
the Lua's rule comments quote (each in one challenge), every mark inside its linen or frame
by more than 0.01 cm (line width and arrow head included; the closest a table reading's
viewer ghost, 0.0375 cm from the frame's foot) and no cloth arc within 0.05 cm of a cord,
over the dialogue's frames and every step the Lua draws marks for; the fifty-five fault
sites, one call each; the bracket order-free; and the transcription's constants against the
Lua, the round-4 marks among them (the ghost's and the pale frame's opacities, the card ghost's
size, the card's overlap, the deleted line's stop and blank, the lever's arc, the three mark
styles). In every state the table can reach, each stone, standing or reflected, keeps a
positive gap from the press's legs, body and chutes, from every other stone and from the
slab's ends.

```python
assert things["far"] == PC.Thing("far", "stone", PC.XFAR, "restless", "slack")   # F17: pictured slack
assert PC.frame_of(t51) == PC.frame_of(PC.presstable(STRIKE, F))   # round 3's P5.1, exactly
assert [PC.lines_drawn(p) for p in counted] == [7, 9]              # the counted frames
assert (direct, table, cloth) == PC.SITE_COUNTS == (12, 14, 29)    # 55 fault sites, by location
assert not np.allclose(cond["plain"], cond["crosswise"])           # why it must stay slack
assert pieces_hold(real["cross"], pieces(rings4)) and not pieces_hold(real["cross"], pieces(rings))
```
