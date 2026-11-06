# Chapter 12 — *Locks Made of Light*: companion code

There are five challenge files, one for each dialogue challenge. Each file checks its claim
**twice**. The blackboard does it with NumPy `assert`s. The sketchbook compares PyZX diagrams
as tensors through `tools/dfp.py`'s `matrix_of` / `same_picture`, which force
`strategy="naive"`, and it also derives each equality with the rewrite rules of
`pyzx.simplify` (`full_reduce` on a copy, with the scalar kept). Running a file is the book's
analogue of type-checking.

```bash
# one challenge at a time
.venv/bin/python chapter12/Code/Ch12/TwoSettings.py
# the whole chapter (must end "5 files, 0 failed")
bash check-code.sh 12
```

Five things hold in every file.

- **Scalars are exact.** Every picture here is a probability distribution, a stochastic
  map or a projector, so Chapter 8's "up to a non-zero scalar" licence is never used. The
  compensations, each commented where it is made, are:
  - **none** for the round (sender, reader), Eve's box, the copier with Eve's leg discarded,
    and the disagreement projectors. The copier, the read-spider and the sender are exact,
    and a Hadamard **edge** is exactly $H$.
  - `add_power(-2)` for every doubled pair. The doubled pair is $\tfrac12\,$cup $\otimes$
    cup, and PyZX's bend is the unnormalized cup $\ket{00}+\ket{11}$. The same power is
    used for a fair coin (the green point with $\tfrac12$), for $\tfrac12 J$, for the shared
    coin $\tfrac12\,$bend, and for the pass projector $\tfrac12\,$cap-then-cup.
  - `add_power(-1)` for the master, The Alarm: $\tfrac12$ for the pair times $\sqrt2$,
    because the red $X(2\to1)$ dot is $\tfrac1{\sqrt2}$ times parity. The same holds for
    the bell on a ringed pair, and $\tfrac1{\sqrt2}$ for a ringed cup (a Bell state).
  - `add_power(-3)` for the master's plain answer (a green point and a red point), and
    `add_power(-4)` for two independent coins.
- **Doubled pictures are hand-built.** PyZX 0.10.6 refuses ground vertices. So, as in
  Chapters 9–11, a quantum wire is drawn beside its mirror copy, a classical wire is one
  wire, a reading is the read-spider, a sending is the same green dot run backwards, and
  forgetting a record is a one-legged green effect. The copier is real, so its mirror copy
  is the green fork again. The circular reader's mirror copy carries the **negated** phase.
- **Double edges go through an identity dot.** PyZX keeps only one edge between any pair of
  dots. Where a sender meets a reader along both lines, each line runs through a
  phase-free two-legged green dot.
- **Generic inputs.** The transpose of Eq. (10) is tested on random complex, non-symmetric
  matrices, with the dagger, the bare matrix, the conjugate and the wrong order as negative
  controls. The slid reading and the mirror's conjugation are tested on random non-real
  states. The circular setting is the non-real test on the table.
- **The dialogue's ledger is checked.** For every round the dialogue plays, the model must
  allow the calls and pieces laid, and it must force what the dialogue says is forced.
  Alice's letter must fit the model too: its counts are plausible draws, and its "every
  time" statements are exact.

The files begin with one glint and two settings. At the same setting, a sender and a reader
fuse into a bare wire; at different settings they fall apart into a fair coin. From that
the key is made by saying only the settings aloud. The third file puts Eve on the wire as
Chapter 11's copier and measures what her look costs. The fourth bends the sender into a
shared pair, with a circular test that catches the difference between a transpose and a
conjugate. The last builds the twelve-line Alarm, and its rewriting shows that at the cross
setting all a spy learns is whether she was caught.

**The prop pictures.** The dialogue draws its props with three macros from
`pdf/props/ch12.tex`: the black tray with the two keys, the lamp, the sleeve or the
tell-glass, the tell's key and the side rod, the rows of pieces and the bell
(`\glinttray`); one key close to, its collar's mark above and its reflection beneath
(`\sparkey`); and the Tortoise's cords and rings on her drawing-cloth (`\keycloth`, drawn on
the shared cloth of `pdf/props/cloth.lua`). Their Lua rules are transcribed once, with the
card's L1–L3 and the Prop Artist's later fixes in force, in `tools/props_ch12.py` (no
matrices there). It reads every place, pitch, scale and frame the Lua names as a constant
from `pdf/props/ch12.lua` itself (a drift guard), and holds the three state models, the
dumps, the keyed words and where the Lua places them, the drawn outlines the clearances
need, the one pitch of the rows, and the Lua's fifty-one fault sites, each keyed and each message byte for byte. Every
dump, frame, keyed word and word place was compared with `texlua` on the dialogue's nineteen
calls and thirty-eight more, every fault message on its own call, and four thousand random
calls on both sides (no difference). Each file reads the pictures of its own section of
`dialog.md` through `tools/dfp_props.py` and checks their layout (4, 3, 4, 4, 4 pictures;
blocks `[1,2,1]`, `[1,2]`, `[1,2,1]`, `[2,2]`, `[1,1,1,1]`; the same in `dialog.raw.md`),
one frame and one macro per block, the captions (words only, from the card's forbidden
list, never a sentence someone speaks, never the first sentence after the block), the places
each prop may appear, the paragraph each block follows and the one after it (with the card's
direction phrases), every keyed word named before its picture, and every picture's state.
Each file also **ties the transcription to its own mathematics**, NumPy against its
blackboard. This is a structural check, not a second proof: a lit window is a reading, two
same-colour rings slid into one cord are $R_rE_r=I$, rings drawn apart on cords of their own
are $\tfrac12J$, a piece laid is a record the model must allow. Clearances are asserted on
drawn extents (strokes, shadows and reflections included) with a positive gap, never on a
centre line.

---

## I. `TwoSettings.py` — *Same Setting, Same Answer* (conjugate bases; BB84 correctness)

> "...Oh. There's no crosswise answer in an upright glint, waiting to be read out." — Achilles

Both collars are upright, then both crosswise: every round the two calls sound in unison.
Achilles guesses that at mixed settings the calls will still agree three times in four; in
the three mixed rounds played they agree once. Alice's lock-book gives the long run, and the
Tortoise's rings show the reason. Same-colour rings slide into one cord; a green ring and a
red ring let their two cords fall away.

The blackboard checks all sixteen amplitudes $\bra yH^tH^r\ket x$, keeping the sign, and
then the doubled round:

```python
if r == t:
    assert np.isclose(amp, iv(y == x))
else:
    assert np.isclose(amp, (-1) ** (x * y) / np.sqrt(2))
...
RE = reader(t) @ sender(r)
assert np.allclose(RE, I2 if r == t else J / 2)
```

Result 12.1 is checked for the plain and cross bases and for BB84's circular basis. The
circular vectors are asserted to be non-real, and their four cross overlaps are compared
with the values the proof prints. A sweep over $\theta$ confirms Dirac's Blackboard: no
real vector is conjugate to both bases. The reader turns $\psi\otimes\psi^*$ into
probabilities; on a non-real $\psi$, the version without the conjugate fails. Achilles's
guess is a negative assert: at mixed settings the agreement is $\tfrac12$, not
$\tfrac34$. Each of the eight rounds of I is checked as possible, and the rows left after
the strike are white, black, white, black, white at both ends. Alice's counts add up:
$96+104=200$ and $53+51=104$. Both 96 of 200 and 53 of 104 lie within three standard
deviations of a fair coin.

The sketchbook builds the round with no compensation and derives both halves of the
headline:

```python
if r == t:   # matching settings fuse into a wire
    assert h.num_vertices() == 2 and h.num_edges() == 1 and connected(h, x, y)
else:        # mismatched settings disconnect into noise
    assert h.num_vertices() == 4 and h.num_edges() == 2 and not connected(h, x, y)
    assert same_picture(h, noise_picture())
```

It also checks that a cross reader drawn as a red ring, as on the cloth, is the same
picture. The closed picture of a red point meeting a green effect reduces to
$(-1)^{ab}/\sqrt2$.

*Pictures.* The tray lit at both ends, keyed: the two windows flash yes at once, and at
matching settings the blackboard's pair disagrees with probability exactly $0$. The
convention, keyed (green to green: Alice's short cord running straight on into Bob's, the
mirror cord beneath it and running the same way); slid together and lifted off, one cord
from edge to edge, and $R_0E_0=I$ (red to red likewise, $R_1E_1=I$). Green to red drawn
apart: each ring rides out on its own short cord (0.6 to 0.1, 5.4 to 5.9), nothing joins
them, the slack cords touch neither ring on drawn extents, and $R_1E_0=\tfrac12J$ of rank 1.

```python
assert {1, ("left", 0)} in comps and {2, ("right", 0)} in comps
assert np.allclose(RE, J / 2) and np.linalg.matrix_rank(RE) == 1
```

## II. `SiftedKey.py` — *Say Only the Setting* (the protocol; sifting)

> "Anyone with an ear to the keyhole has heard "upright, upright" and learned that we kept the round — and not a whisper of which piece we laid." — Achilles

The settings are secret and the pieces are laid in silence. Only the settings are said
aloud. Rounds 2, 4 and 6 are struck, and the kept rows match piece for piece.

The blackboard enumerates the eight triples $(x,r,t)$ exactly:

```python
assert np.isclose(kept, 0.5)          # (i)
assert np.isclose(kept_err, 0)        # (ii) every kept round: y = x
assert np.isclose(raw_err, 0.25)      # (iii) the raw key: 25% errors
assert np.isclose(joint.sum(axis=1)[0] / p_rt, 0.5)   # (iv) P(x = 0 | r, t) = 1/2
```

In the twin form, a kept round is $\tfrac12[\,k_A=k_B\,]$ and a struck round is a product
of two fair coins. Twenty seeded runs of 400 table rounds give identical rows on every kept
round. The file also checks the worked run in §12.2 (kept rounds 1, 3, 5 and 6; sifted key
$0,1,0,1$) and the six-round ledger of II.

The sketchbook shows the twin at a matching setting as $\tfrac12$ times a bend on the two
records. `full_reduce` leaves one bare edge between the outputs. At mismatched settings
the records fall apart. It also builds the left-hand picture of §12.2: a fair coin, Alice's
copy of her bit and the kept round. That picture fuses into the same shared coin, and it
equals the twin.

*Pictures.* The first tray picture comes only after the screens are down (G11): the two
rows laid open, piece for piece the same, and a kept round's error is exactly $0$. Then two
rounds on the cloth, green to green above red to red below, and the pair slid together and
lifted off: two straight cords whose dumps differ only in their row, while a listener told
the colour learns nothing of the piece, $p(x\mid r,t,\text{kept})=\tfrac12$.

## III. `EvesBox.py` — *The Tell-Glass* (Eve is a box on the wire; information–disturbance)

> "And the upright rounds came through untouched because the tell-glass copies its own kind, as the press did: an upright glint it copies and lets be, and a crosswise one it can only make a pair of." — Achilles

The tell-glass goes into the line and the third key is set upright. In six secret rounds
the tell's piece is laid in silence. The first wrong note sounds, unrung, on a crosswise
round.

On the blackboard, Result 12.4 is checked at both settings and for both of Eve's readings:

```python
assert np.allclose(press_round(x, PLAIN, PLAIN), want)            # [kB = x][kE = x]
assert np.allclose(press_round(x, CROSS, PLAIN), 0.25)
assert np.allclose(press_round(x, CROSS, CROSS), want)            # 1/2 [kB xor kE = x]
assert np.isclose(b, 0.5) and np.isclose(d, 0.25)                 # 1/2 bit, 1/4 disagreement
```

Bennett and Brassard's general bound is cited, not tested. The file checks only that the
press sits on it ($d=b/2$). Achilles's "cleverer Eve" is a separate assert: a copier in
the cross basis spoils the upright rounds instead. With Eve's copy discarded, the press is
dephasing, tested on a non-real $\rho$. A cross round through it is $\tfrac12J$ even
though the settings matched.

The table form puts the copier on the Tortoise's half and leaves the tell's key upright.
Upright–upright gives $k_A=k_B=k_E$. Crosswise–crosswise gives $\tfrac18$ on every triple.
Achilles upright against the Tortoise crosswise gives tell = Achilles; the reverse gives
tell = Tortoise. The six ledger rounds are each checked as possible.

The sketchbook builds Eve's box with nine edges and no compensation. It asserts the "six
lines join five dots" count and the value at every setting. It then derives three results:

```python
g = eves_box(s, PLAIN, forget_bob=True)          # Bob's record forgotten
assert connected(h, x, kE) == (s == PLAIN)        # at cross: a cut through empty paper
assert same_picture(press_discarded(), four_legged())            # row 85
assert same_picture(dephasing_round(CROSS, CROSS), noise())      # a kept round struck
```

The III table configuration is built as a doubled picture with `add_power(-2)`. At
crosswise–crosswise every record is disconnected from every other.

*Pictures.* The tray with the tell-glass in the sleeve's place, keyed, then the run as
laid and struck. The file reads the settings the two of them say aloud and the pieces as
laid, and they are the blackboard's ledger round for round, each possible under
$(I\otimes\delta)\ket{\Phi^+}$ read at (his, hers, upright): $\tfrac12,\tfrac14,\tfrac18,
\tfrac14,\tfrac12,\tfrac18$. The strike takes rounds 2 and 4 (the mixed ones), the gaps
close, and on every kept upright round the tell's piece is his. The tell's row lies level
with her last pieces. Every part of the tell's key standing is clear of every part of her
key standing (least: its foot to her crystal, 0.018 cm with the crystal's stroke), and
their reflections likewise; the tell's reflection itself runs down behind her standing key,
which the Lua draws over it (the draw order is asserted; the overlap is reported, not asserted). The fork threaded, keyed:
Eve's ring green, the upper side cord crossing only the mirror cord, once, and the copier
real, $\bar\delta=\delta$.

## IV. `BentKey.py` — *Backwards to the Lamp* (E91 is BB84 bent)

> "Round the bend, your question reaches my end as its reflection." — Tortoise

The Tortoise pulls the string straight, and Achilles's reading ring becomes a sending ring.
Then she springs the trap: at the curl, every call differs. The curl is the one mark that
looks reversed in the tray.

The blackboard checks Eq. (10) as a **transpose** on random complex, non-symmetric
matrices:

```python
assert np.allclose(lhs, (kron(I2, U2 @ U1.T) @ BELL).ravel())             # transpose: yes
assert not np.allclose(lhs, (kron(I2, U2 @ U1.conj().T) @ BELL).ravel())  # dagger: no
```

A reading slid round the bend arrives **conjugated**. This is tested on random non-real
states, with $\tfrac1{\sqrt2}\varphi$ as the negative control. The pair is the BB84 round
fed a fair coin for Alice at plain or cross and Bob at plain, cross or circular. The
required negative assert is at circular–circular:

```python
assert np.allclose(p_pair, 0.5 * (1 - np.eye(2)))   # the pair: always disagree
assert np.allclose(p_bb84, 0.5 * np.eye(2))         # BB84: always agree
assert not np.allclose(p_pair, p_bb84)
```

The file checks the BB84 paper's EPR remark as well. On random bases the singlet is
anticorrelated at every setting, while $\ket{\Phi^+}$ at a random non-real setting neither
always agrees nor always disagrees.

The sketchbook's circular reader carries $-\tfrac12$ on $q$ and $+\tfrac12$ on $\bar q$. The
same phase on both copies is asserted wrong. The twin (`add_power(-2)`) is `same_picture` the
fair-coin sender form for all six setting pairs. At circular–circular it equals a bend with
a red half-turn, and it is not the BB84 round. `full_reduce` leaves a bare edge between the
records at matching settings. Eq. (10) is also drawn: a generic box chain slides round the
bend reversed. The unreversed chain and the wrong order both fail.

*Pictures.* The lamp's bend, keyed: the glint's narrow turn nested inside the mirror's
wider one, crossing nothing, and the word "bend" clear of both turns. Pulled straight, the
kit's table is `green to green`'s exactly (the half of the dump after " | "; the words
before it name the scene), the rings ride out from 1.6 and 4.4 to 0.6 and 5.4, and no ring
or cord is added or lost: on the blackboard the pair read at $(r,t)$ is $\tfrac12$ times
BB84 fed a fair coin. The key close to: the cross is the same beneath, keyed; the curl winds
against the sun beneath, computed from the drawn marks. The curl's setting is non-real, its
conjugate is the other circular vector, and the plain and cross vectors are real.

```python
assert c42.dump.split(" | ")[1] == PC.keycloth("", "green to green").dump.split(" | ")[1]
assert np.isclose(pair(kA, kB, r, t), 0.5 * bb84(kB, kA, r, t))
```

## V. `Alarm.py` — *The Alarm* (the master; checks; monogamy)

> "Then that's the whole of it. Two calls at the same setting sing the same; a listener has to look; a look at the wrong setting is a wrong note, and the bell hears it; and a pair that never sings a wrong note, upright or crosswise, has nothing to spare for anyone else." — Achilles

In the checked crosswise round, the calls differ, the bell rings, and the tell's piece is
black. In the next crosswise round the calls agree, the bell is silent, and the tell's
piece is white. In the upright round the calls agree, the bell is silent, and the tell's
piece is black, like Achilles's. The half-turn rings come off the bend and leave it bare.

The blackboard computes the master distribution, the three ledger rounds (each possible,
with the tell's piece forced), the parity dot, Result 12.6 and Result 12.7:

```python
assert np.allclose(alarm_dist(PLAIN), 0.5 * np.array([[1, 0], [1, 0]]))   # 1/2 [w = 0]
assert np.allclose(alarm_dist(CROSS), 0.5 * np.eye(2))                     # 1/2 [kE = w]
assert np.allclose(proj(PSI_P) + proj(PSI_M), dis_plain)
assert np.allclose(proj(PHI_M) + proj(PSI_M), dis_cross)
assert np.linalg.matrix_rank(M, tol=1e-9) == 1        # projected random states: AB | Eve
```

For monogamy, random pure and mixed three-part states are projected. Each passes both
checks with certainty and factors as $\ket{\Phi^+}\bra{\Phi^+}\otimes\rho_{\mathrm{Eve}}$.
Each check alone can be fooled: the file shows two entangled states, one passing only the
plain check and one passing only the cross check. Result 12.9 is cited, not tested. Only
the arithmetic of the numbers the chapter quotes is checked: the root of $1-2h$ near 11%,
$\tfrac12(1-\tfrac1{\sqrt2})\approx15\%$, and $0.6^2+0.4^2=0.52$.

The sketchbook builds the master and counts its lines:

```python
assert g.num_edges() == 12        # 8 doubled quantum lines + 4 classical (kA, kB, kE, w)
...
if S == PLAIN:
    assert not connected(h, kE, w)                    # Eve's record and the bell fall apart
else:
    assert connected(h, kE, w)                        # one bend joins them
```

The master yanks into the prepare-and-measure form (`bent=True`). The pass projector
$\tfrac12\,$cap-then-cup is rank 1, with no path from its inputs to its outputs. After
`full_reduce` its cap is a bare edge between two inputs, so it is compared as a state.

The checks are drawn as half-turn rings. The four ringed cups are the Bell states. The
disagreement projectors are drawn as pictures, and each check picks out its ring:
$P_{\mathrm{dis}}B_{m_1m_2}=m_2B$ at plain and $m_1B$ at cross. The pass projector leaves
only the bare bend: $PB_{m_1m_2}=[\,m_1=m_2=0\,]B_{00}$. Summing the four ringed projectors
as matrices is a translation check, not a second proof. The bell on a doubled ringed pair
rings exactly on the red ring at plain and on the green ring at cross.

*Pictures.* The bell rung (the rows as III left them, only the bell lifted and
swinging), then his white, hers black and the tell's black. The file reads the three rounds
of V off the dialogue (the calls spoken, the pieces laid) and they are the ledger, each
possible under the master at the agreed setting, $\tfrac14,\tfrac14,\tfrac12$; at cross the
tell's piece is black exactly when the bell rings. The pair's bend, keyed, red close in and
green further out, is $\ket{B_{m_1m_2}}\propto(Z^{m_1}X^{m_2}\otimes I)\,$cup; the other
order differs by a sign. L1's route is keyed on the ring's x: red goes round the turn, green
keeps its place, and $(Z\otimes X)\,$cup $=(ZX\otimes I)\,$cup since $X^T=X$; the plain
`slide off red` is refused with the Lua's own message. The bare bend is the pass projector,
and there is room for the claw's line beyond its turn.

The file then checks the chapter's whole run: nineteen pictures, none outside the five
sections, the keyed ones exactly P1.1, P1.2, P3.1, P3.4, P4.1, P4.3, P5.3, and every call
legal and in its place. Over all seven tray pictures the things on the black keep their
places and the bell moves only when rung. There is one pitch for the evening, read from the
Lua, so every piece keeps its x from call to call; only the strike of P3.3 closes gaps. On
drawn extents the bell (still or rung) clears every piece by at least 0.034 cm, shadows
included; the Prop Artist's 0.0376 cm is the foot against her first piece's body on centre
lines. The rows at fifteen, from the full run of V, clear each other by 0.030 cm, shadows
included; the Prop Artist's 0.04 cm is between the bodies, before strokes. It runs the duet's one-liners on both dialogue files (two speakers, 44
calls, Eve never speaks, `COECKE`, six "Let Bob draw"), trips each of the fifty-one fault
sites with its own call and a message of its own, and checks the Lua still says what the
transcription assumes where a value is written inline (the tell's key's scale, the rung
bell's shift and turn, the key's frame, the side rod's end, the pieces' and the bell's
shapes, the pitch rule).

```python
assert D.check_whole(PC.ORDER) == 19
assert {t.shown.d for t in trays} == {PC.PITCH}
assert set(PC.faults(fn, *args)) == {site}
```
