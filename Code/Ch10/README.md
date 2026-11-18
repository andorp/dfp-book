# Chapter 10 — *Teleportation*: companion code

Five challenge files, one per dialogue challenge, each checking its claim **twice**: once
on the blackboard (NumPy `assert`s) and once in the sketchbook (PyZX diagrams compared as
tensors through `tools/dfp.py`'s `matrix_of` / `same_picture`, which force
`strategy="naive"`, *and* derived by the rewrite rules with `pyzx.simplify`). Running a file
is the book's analogue of type-checking.

```bash
# one challenge at a time
.venv/bin/python chapter10/Code/Ch10/FourAnswers.py
# the whole chapter (must end "5 files, 0 failed")
bash check-code.sh 10
```

Three things hold throughout.

- **Scalars are exact.** Every stated result carries a probability, so the "up to a
  non-zero scalar" licence of Chapter 8 is not used for any of them. PyZX's own spider
  normalizations are compensated with `g.scalar.add_power(n)` (a factor $\sqrt2^{\,n}$), and
  every compensation is written out in a docstring: $-2$ for the snake's $\tfrac12$ and for
  the dense-coding loops, $-1$ for a Bell state or effect, $-3$ for the swap's
  $\tfrac1{2\sqrt2}$, and $+2$ net for the doubled master.
- **Doubled pictures are hand-built.** PyZX 0.10.6 refuses ground vertices, so, as in
  Chapter 9, every doubled picture draws a wire beside its mirror copy, reading is the
  read-spider, and forgetting a record is the phase-free one-legged green effect.
- **Generic inputs.** The message is a random non-real $\psi$, the mixed input a random
  non-real, full-rank $\rho$. The Paulis are real and symmetric, so they cannot tell a
  mirror (dagger) from a half-turn (transpose). That job goes to a twisted basis
  $U_k=Z^{k_1}X^{k_2}W$ with $W$ a random non-real, non-symmetric unitary.

The files start from the four answers a joint reading can give: Chapter 6's one bend, and
the same bend carrying a flip, a spin, or both. Then they follow a message through them. It
arrives at once, but turned (the Bell-basis calculation), and two classical bits set it
right, their half-turns meeting their own kind and vanishing (teleportation, all four
outcomes, exact). The last two files run the same pieces the other way round, in superdense
coding, whose codebook is the correction table and whose preparations are the mirror images
of the readings. Then they send half a pair, building the ten-line Mended Junction in doubled
form, counting its edges, and checking that every outcome leaves the same pair.

**The prop pictures.** The dialogue draws its props with four macros from
`pdf/props/ch10.tex`: the Crab's pairing-glass (`\pairingglass`), the two tokens close to
(`\twotokens`), the mirror-table with the last two pairs and their reflections
(`\mirrortable`), and the Tortoise's string and rings on her drawing-cloth
(`\ringstring`). Their Lua rules are transcribed once, with the card's L1 and L2 corrections
in force, in `tools/props_ch10.py` (no matrices there). It holds the token pattern (brass up
a flip, copper up a spin, a token not named lying down) and the glass's state words, with
the trays tipped exactly when the cups are full. It holds the table's places after L2 and its
computed operations: into the glass, the letter, flip, spin, tip out. A flip or a spin
changes the state and never the picture, since a stone carries no mark. It holds the string's
four scenes and their moves: one ring slid round a bend unchanged, two rings the other way
about (L1), the pull, Bob's corrections, like rings meeting (cyclically on the loop), and the
answer's ring carried round. It also holds the three frames (1.58, 1.10 and the cloth's
1.95, 1.78, 1.54, 0.70 cm) and the Lua's fault sites, each keyed. The string's
picture is a `\tortoisecloth` layout, which the transcription writes as the Lua does and
reads with the book's one transcription of the cloth kit, `tools/props_cloth.py`.

Round 4 (*Show the process*) adds the bracket's `;` items, read whole so that their order
never matters: a **step** (`step=turn brass`, `step=slide the flip round`, `step=pull`,
`step=corrections`, `step=red meets red`, `step=lift off`, `step=slide round`,
`step=chime both up`, `step=into the glass`, `step=flip`, `step=spin`, `step=tip out`) takes
the state written as the state *before* it and draws the state after it, computed, with the
step's **motion marks** — a ring's ghost in dotted outline where it lay (a halo where another
ring arrives), the slide arc the string takes it along, a swing arc over a turned token or
round Bob's stone, the pin round the token that held still, the envelope's ghost — unless
`marks=off`; a **prediction** (`expects=apart`, `expects=more turned`, `expects=either way`)
draws, paler, what one of Achilles's three named wrong rules foresaw; and `\stringsteps` and
`\tablesteps` expand into their frames, frame $k$ being the single call
`[<after and the steps before it>; step=<k>]`. The new string moves are `pull halfway` (the
S drawn in), `red meets red` and `green meets green` (two like rings run together into one
ring at rest, a whole turn, where the one that held still lay) and `lift off`. A ring on the
straight string now keeps its column while it lies still (the P3.3 fix: after red to red the
greens stay at columns 1 and 5). The transcription carries the marks down to their TikZ
text and the log lines byte for byte, and it was **proved against texlua**: every dialogue
call of both files and a sweep of some 67 000 aimed and random calls (every glass state with
every step and bracket, every token pattern, place and step, the table's scenes with
operation prefixes and every step, every string scene with up to five operations, every
step, every prediction, and the strips with every option) — 0 mismatches in about 570 000
records, all **seventy-three** fault sites raised (forty-four direct `tex.error` calls,
twenty-nine through the string's `serr`), each message byte for byte. `check_lua` then holds
the Lua and the `.tex` to the text proved (`LUA_PINS`, `TEX_PINS`, chunk by chunk, comments
left out) and the five round-4 comments that quote the dialogue word for word
(`COMMENT_PINS`); `check_sites` holds the seventy-three sites to the Lua's own list.

**Pins (the cloth kit).** The cloth kit this chapter introduced, shared by Chapters 10–19, is pinned
on its own, in `tools/props_cloth.py`: `check_cloth()`, run when the module is imported (unless
`DFP_REPIN` is set, to re-pin), holds `pdf/props/cloth.lua` to its 79 top-level chunks (`LUA_PINS`:
comments left out, white space closed up outside string literals) and `cloth.tex` to its 39 colours,
styles and macro (`TEX_PINS`). `COMMENT_PINS` is empty: only a kit comment that quotes *spoken*
dialogue would be pinned, and none does. The header's turn-grammar examples ("an eighth round and a
half") are grammar, not quotes, even where a stage direction writes the same words. Every challenge
file of Chapters 10–19 imports its `props_chNN`, which imports `props_cloth`, so a drawing-only or
style-only edit to the kit fails all of them, naming the chunk. A plain comment edit passes, and so
does white space outside the Lua's string literals (or trailing blanks and blank lines in the
`.tex`). The kit may hold no Lua long bracket (`[[`), which the chunker could not see into:
`check_cloth` refuses one. Before pinning, the transcription was re-proved against the committed kit
under texlua (every call the twenty dialogue files make into the kit, all 74 transcribed fault sites
(the drawing-time "no room" fault stays untranscribed) and 63,000 random calls: 0 mismatches). To
re-pin after a kit edit, prove it again, then print the pins with `DFP_REPIN=1` (the module's
docstring has the line).

Each file reads the pictures of its own section of `dialog.md` through `tools/dfp_props.py`
and checks the following: their layout (calls `[1,1,1,1]`, `[1,2]`, `[1,1,3]`, `[1,1,1]`,
`[3,1]`; frames 4, 4, 6, 3, 6; the same in `dialog.raw.md`); one frame height per block; the
captions (words only, no Stage-D word, no quantity word, and on the string no word for its
lengths); the round's direction edits (T1–T3) in both files; the places each macro and
operation may appear; the speech or direction each block follows and the line after it;
the Lua comments' quotes of its section; and every reading against the scene at that
moment: each step from the state before it, each mark where the card's dump list puts it,
each marked frame's `marks=off` twin against its round-3 picture, each strip frame against
its single call, and each prediction against the truth that refutes it. Each file also
**ties the transcription to its own mathematics**, NumPy against its blackboard. This is a
structural check, not a second proof. A ring on a leg is a Pauli on a wire. The ring
sequence along the string, read from the message end, is the order in which the turns act.
A slack reflection is a stone whose state alone is $\tfrac12 I$.

---

## I. `FourAnswers.py` — *Four Ways to Answer* (the Bell basis and the Bell reading)

> "It has two cups side by side under one lens, and it reads two stones together, never one." — Crab

The Crab unwraps the pairing-glass and lays out a run of ledger lines. The tokens lie in
four ways, each about a quarter of the time. The Tortoise ties four bends, plain and ringed,
and slides a ring round a bend. The **blackboard** builds the four Bell states from the
circuit $\mathrm{CNOT}(H\otimes I)\ket{m_1m_2}$, checks them against the table of §10.1, and
checks that each is the plain pair with a Pauli on either leg. On the bottom leg the product
comes out **reversed**, and the same-order product gives $-B_{11}$. The four are
orthonormal and complete. The reading's effects are the mirrors of the states and equal the
circuit run backwards. Chapter 6's cap is $\sqrt2\,\bra{B_{00}}$. On a non-real message
each answer comes a quarter of the time. Sliding is checked on a generic non-symmetric,
non-real box: a single box comes round **transposed** (not daggered), and a product comes
round in reversed order.

```python
assert np.allclose(B, top) and np.allclose(B, bottom)                    # Result 10.1
assert not np.allclose((kron(I2, Z @ X) @ BELL).ravel(), bell(1, 1))    # same order: no
assert np.allclose(eff, ringed_cap) and np.allclose(eff, circuit)       # Result 10.2
assert np.allclose(kron(B @ A, I2) @ CUP, kron(I2, A.T @ B.T) @ CUP)    # a chain reverses
```

The **sketchbook** pins the rings (one-in/one-out $\pi$ dots are exactly $X$ and $Z$) and
checks that the reading as built, a CNOT with a Hadamard **edge** and two one-legged red
effects, is exactly the ringed cap for all four answers. Then `full_reduce` **derives** it:
colour change and fusion boil the circuit down to the ringed bend, scalar kept, with no red
dot left. The reduced $B_{00}$ effect is a bare boundary-to-boundary edge that `matrix_of`
cannot evaluate, so the derivation runs on the same picture with its two ends turned to
outputs. The rings' order is fixed: the other order is right only up to a sign. One ring
slides round a bend unchanged, and a chain of two generic rings arrives reversed, not in
the same order.

```python
assert same_picture(circuit_reading(m1, m2), ringed_cap(m1, m2))        # tensor check
simp.full_reduce(g); assert same_picture(g, ringed_cup(m1, m2))         # derived
assert same_picture(top, chain_on_cup(1, [("X", b), ("Z", a)]))         # reversed: yes
assert not same_picture(top, chain_on_cup(1, [("Z", a), ("X", b)]))     # same order: no
```

*Pictures.* The glass unwrapped, keyed: cups empty, both tokens blank in their trays (brass
left, copper right), the trays level. Then the tokens close to in the dish, keyed, brass up
and copper down. Then the four bends, two above two, keyed: plain, a red close in, a green,
and a red close in with a green further out. Last, the flip's red slid round to the lower
leg, the other three bends unchanged. The file ties each bend to its Bell state: with the
leg's rings read from the bend as the order of application, $(R\otimes I)\,$cup is $\sqrt2$
times the state the answer names. One ring slid round gives $(I\otimes R)\,$cup, exactly the
same. It also checks the move the dialogue only speaks of, "Two together come round the
other way about". After `slide both round` the lower leg reads green, red (L1), and
$(I\otimes XZ)\,$cup $=(ZX\otimes I)\,$cup, while the order the Lua used before L1 is off by
a sign. P1.2's brass-up pattern is the flip bend's one red ring, and each of the four
patterns comes a quarter of the time on a non-real message.

Round 4 makes P1.2 and P1.4 steps. P1.2 is `[step=turn brass]{both down}`: the swing arc runs
over the brass from its left rim ($-0.71$) to its right ($-0.09$) at $y=-0.52$, its head
towards the copper (T1, "turns the brass token over towards the copper"), with no ghost (the
token covers its own place) and a pin round the copper at $(0.4,-0.53)$. P1.4 is
`[step=slide the flip round]`: the red's ghost in outline at $(0.44,-0.132)$, grid (6, 0), and
one slide arc round the outside of the bend (its turn at $x=0.128$) to the same distance out
on the lower leg. Each `marks=off` twin is round 3's picture to the byte, and turning a token
flips exactly its bit of the answer.

```python
assert r3((x0, y0, x1, y1)) == (-0.71, -0.52, -0.09, -0.52)                       # over the brass
assert (pin.kind, pin.name, r3(pin.at), pin.r) == ("pin", "copper", (0.4, -0.53), (0.33, 0.2))
assert PC.frame_of(t) == PC.frame_of(PC.twotokens(None, "brass up", True))        # round 3's P1.2
assert np.isclose(ghost.at[0], ring_x)                       # the same distance from the bend
```

```python
assert np.allclose(kron(leg_operator(up), I2) @ CUP, np.sqrt(2) * bell(*m))      # each bend
assert both == ("both", [], ["green", "red"])                                     # L1
assert np.allclose(kron(I2, leg_operator(both[2])) @ CUP, kron(ZX, I2) @ CUP)     # transposed
assert np.allclose(kron(I2, ZX) @ CUP, -kron(ZX, I2) @ CUP)                       # pre-L1: a sign
```

## II. `TwoBitsEarly.py` — *Two Bits Early* (the full Bell-basis calculation)

> "...Oh. It's there before the letter — and it's no use to him." — Achilles

The Tortoise lays the whole tale in string and pulls it straight. The message is at Bob's
end already, still wearing Alice's rings. The Crab's ledger shows Bob's early peeks: a coin
toss, every time, whatever Alice had sent. The **blackboard** evaluates both sides of
Result 10.3 on a non-real $\psi$. Each branch is exactly $\tfrac12X^{m_2}Z^{m_1}\psi$, a
quarter each, turned and not yet the message. Result 10.4 is checked on a non-real mixed
$\rho$ (purity below 1) and on the pure input: with the answer forgotten, Bob holds exactly
$\tfrac12 I$, the proof's own line
$\tfrac14(\rho+Z\rho Z+X\rho X+XZ\rho ZX)=\tfrac12 I$ holds, and his peek is a fair coin. On
a *known* answer his wire is not grey.

```python
assert np.allclose(lhs, rhs)                                   # Result 10.3, both sides
assert np.allclose(branch, 0.5 * garbled)                      # there already, turned
assert np.allclose(bob, I2 / 2)                                # Result 10.4, pure or mixed
assert not np.allclose(one, I2 / 8)                            # a known answer is not grey
```

The **sketchbook** draws the S with the reading's rings. Its tensor is
$\tfrac12X^{m_2}Z^{m_1}$, and `full_reduce` **derives** the yank, leaving the rings on
the wire and the $\tfrac12$ kept. Then Section 10.2's doubled picture is hand-built: the
pair and its mirror, Alice's reading on both copies (CNOT, Hadamard edges into the
read-spider) and both records forgotten. It equals the message discarded beside
$\tfrac12$cup$(c,\bar c)$, which is $\tfrac12 I$, exactly. The rules derive this too, down
to two bare bends. Fed a non-real $\rho$, it returns $\tfrac12 I$.

```python
assert same_picture(h, half_turned_wire(m1, m2))               # derived: rings stay on
assert same_picture(d, discard_then_grey())                    # doubled: 1/2 I, every input
assert h.num_vertices() == 4 and h.num_edges() == 2            # reduced to two bends
```

*Pictures.* The tale laid flat as an S, with Alice's red and green on its incoming leg,
which are exactly the fourth little bend's two rings, kept back when the first drawings were
gathered. Then the tale pulled straight: the bends gone, the same frame and grid (no longer,
no shorter), a green nearer the message and a red nearer Bob, both left of the middle. The
file ties the pulled string to the branch. For all four patterns, the rings read from the
message end as the order of application make $X^{m_2}Z^{m_1}$, and the Bell-basis branch is
$\tfrac12X^{m_2}Z^{m_1}\psi$ on a non-real message, a quarter each. Summed, Bob's early peek
is a coin.

Round 4 makes the pull one block, in the spoken order: Achilles's prediction first, paler,
never keyed or marked — rule `apart`, the string parted at the pair's bend, the message's
length straight with Alice's rings and Bob's length bare — then the pull as a strip of two.
*Drawn in* (`pull halfway`): the two bends slide towards each other, the green from column 2
to 1 (a halo: the red now lies on its place) and the red from 3 to 2, both arcs below the
string (the cloth's top edge leaves no room above it), bowed down, end to end and apart.
*Pulled straight*: both drop onto the straight row, each with
its ghost and a curved arc straight down; its `marks=off` twin is round 3's P2.2. The tale's
ends keep columns 0 and 6 and the frame stays 1.78 throughout (look, not length). The
prediction is refuted: the true frame is one line to Bob's end with the same rings on it.

```python
assert w.dump == ("tale apart; the message's length from the message end [alice green, alice red]; "
                  "Bob's length bare")
assert PC.ringpos(f07.state) == {"alice red": ("red", 2, 0, False), "alice green": ("green", 1, 0, False)}
assert PC.frame_of(t) == PC.frame_of(PC.ringstring("pull", "the tale, both up"))   # round 3's P2.2
```

```python
assert PC.picture(p13).state.bends[3][1] == s21.state.alice    # the fourth bend's rings
assert np.allclose(O, P(X, m2) @ P(Z, m1))                     # the pulled string's rings
assert np.allclose(branch, 0.5 * O @ psi)                      # ... are the branch
```

## III. `Teleportation.py` — *The Letter* (Pauli corrections; *The Theorem, Twice*)

> "A whole turn is no turn." — Tortoise

Achilles carries the letter round the table. Bob's rings go on as the tokens say, red meets
red and green meets green, and all come off, for every pattern. The **blackboard** proves
the headline on a non-real $\psi$: for all four answers
$(Z^{m_1}X^{m_2})(\bra{B_k}\otimes I)(\psi\otimes\Phi^+)=\tfrac12\psi$, exactly. It is a
quarter each, and never $\tfrac12\bar\psi$. The unnormalized snake is exactly the identity.
The wrong order (spin first) agrees on three answers and gives $-\tfrac12\psi$ on answer
$11$, because $ZX$ is minus its own inverse. As a channel on a non-real mixed $\rho$, each
branch is exactly $\rho/4$ and the whole is $\rho$. The **twisted-basis test** keeps mirror
and half-turn apart. With the reading $\bra{E_k}=\ket{E_k}^\dagger$, Bob holds
$\tfrac12U_k^\dagger\psi$ and the correction is $U_k$. $U_k^T$, $\bar U_k$ and $U_k^\dagger$
all fail, and genericity is asserted so that each failure is real. With the box on the other
leg, sliding brings it round transposed, and only then is $U_k^T$ right.

```python
assert np.allclose(out, 0.5 * psi)                             # Result 10.5, all four
assert np.allclose(wrong, -0.5 * psi)                          # wrong order, answer 11
assert np.allclose(channel, rho)                               # the identity channel
for bad in (U.T, U.conj(), U.conj().T):
    assert not np.allclose(bad @ held, 0.5 * psi)              # only U_k corrects
assert np.allclose(U.T @ held2, 0.5 * psi)                     # other leg: the transpose
```

The **sketchbook** is *The Theorem, Twice*'s picture proof. First the Tortoise's line is
checked: two like half-turns **fuse** (`spider_simp`) to a whole turn, which the **identity
rule** (`id_simp`) removes, leaving a bare wire with scalar 1. Then the full snake, the
reading's rings, the cap, the cup and Bob's rings, is exactly $\tfrac12$ times a bare wire
for all four answers. `full_reduce` **derives** it, down to two ends, one edge and the
scalar $\tfrac12$. The wrong-order picture is right on three answers. On answer $11$ it is
`same_picture` **false** and `same_picture_up_to_scalar` **true**: the sign.

```python
assert simp.spider_simp(g)                                     # fusion: a whole turn ...
assert simp.id_simp(g)                                         # ... is no turn
assert same_picture(g, bare_wire())
simp.full_reduce(h); assert np.isclose(h.scalar.to_number(), 0.5)    # derived, 1/2 kept
assert not same_picture(wrong, half_wire())                    # answer 11, wrong order
assert same_picture_up_to_scalar(wrong, half_wire())           # ... right up to a sign
```

*Pictures.* The letter opened at Bob's end with both tokens up, the pattern of the tale
already pulled. Then Bob's rings threaded from his end as the letter says, reading green,
red | red, green, facing each other across the middle. Then, in one block, red to red
lifted off and green to green lifted off, leaving the string bare. The bare string is the
same arrangement as the pulled tale with no reading at all (compared through the cloth
kit). For every letter, Bob's rings are the pattern's own and make the file's correction
$Z^{m_1}X^{m_2}$, and the whole string makes the identity. The moves the dialogue runs
letter by letter (red to red, green to green, or nothing at all for both down, where
`kin to kin` would be a fault) leave the bare pulled tale. Spin first, which is modelled
and never drawn, leaves the rings where nothing meets, the product is $-I$, and the Lua
refuses red to red past a green.

Round 4 makes P3.2 a step (`[pull; step=corrections]`): no ghost, two slide arcs from Bob's
end ($x=0.873$), the red's above the string to column 4 and the green's below to column 5.
The meeting block is in the spoken order: the prediction, paler — rule `more turned`, all
four rings as threaded — then the strip *red to red* (Alice's red slid to Bob's, the two run
together into one ring at rest at column 4, its bead all the way round; the ghost at column 2,
the arc above) and *lifted off* (the met ring's ghost at column 4, into her pouch, no arc),
then round 3's own *green to green, and bare*, unmarked. After red to red the greens lie at
columns 1 and 5, where they were put (the P3.3 fix), and the reading is unchanged. The third
rule, `either way`, is modelled only (its precondition, spin first, never stands in the
dialogue): paler, the string bare; the true step leaves all four rings, nothing met.

```python
assert ring_cols(f13) == [(1, "green"), (5, "green")]                 # the greens lie still
assert f12.dump == "tale straight; from the message end [alice green, met red at rest, bob green]"
assert w.dump == s32.dump                                             # more turned: all four
assert ew.dump == "tale straight; from the message end []" and true.dump == PC.dump(wrong)
```

```python
assert np.allclose(applied(bob), correction(m1, m2))           # flip first, then spin
assert np.allclose(applied(PC.sequence(on)), I2)               # the whole string: no turn
assert CLOTH.same(s34.layout, PC.layout(bare)[0])              # bare = the unread tale
assert np.allclose(applied(PC.sequence(wrong)), -I2)           # spin first: -I
```

## IV. `DenseCoding.py` — *The Other Way Round* (superdense coding; the mirror)

> "...Oh. It's the letter's own table." — Achilles

In the Crab's tale Alice turns her stone first and sends the stone, and Bob's glass drops
her own tokens. The Tortoise ties a closed loop, and like rings come off. The
**blackboard** checks the loop as a trace on a generic box (Result 10.6): the bare loop is
$2$, and the normalized pair and effect give *half* the trace. It checks all sixteen
amplitudes, $\delta_{jk}=\tfrac12\operatorname{tr}(X^{j_2}Z^{j_1}Z^{k_1}X^{k_2})$, and
$\operatorname{tr}(\sigma_j^\dagger\sigma_k)=2\delta_{jk}$. The codebook entry $k$ is exactly
the correction that undoes teleportation's answer $k$. With the twisted basis the reading
$\bra{E_j}$ is the mirror of the preparation, not its transpose, and decodes exactly, while
the transpose does not decode. Dense coding is **not** the dagger of the protocol:
teleportation's corrected branch is $\tfrac12I$ on a qubit and so is its dagger, while dense
coding is a table from two bits to two bits.

```python
assert np.isclose((CAP @ kron(A, I2) @ CUP).item(), np.trace(A))         # Result 10.6
assert np.isclose(amp, 1.0 if j == k else 0.0)                           # Result 10.7
assert np.allclose(codebook(k1, k2) @ held, 0.5 * psi)                   # the letter's table
assert not np.allclose(np.abs(trans) ** 2, np.eye(4))                    # transpose: no
```

The **sketchbook** draws the sixteen closed loops: the cup, Alice's rings, the reading's
rings, the cap, and a phase-free spacer dot on the return leg (PyZX cancels a doubled edge).
They are closed diagrams, so they are evaluated only through `matrix_of`, and each is
exactly $\delta_{jk}$. `full_reduce` **derives** each to nothing but a scalar, $1$ for a
clean loop and $0$ with a ring left over. The bare loop is $2$, and a loop through a generic
chain is its trace. Last comes the mirror. The adjoint of the Bell-state picture is the
Bell-effect picture, because half-turn rings are their own mirror images. On a generic
chain the mirror conjugates the phases and the half-turn does not, and the two differ.

```python
assert np.isclose(matrix_of(g).item(), want)                            # sixteen loops
simp.full_reduce(h); assert h.num_vertices() == 0 and np.isclose(h.scalar.to_number(), want)
assert same_picture(state.adjoint(), effect)                            # reading = mirror
assert not same_picture(prep.adjoint(), half_turn)                      # not the half-turn
```

*Pictures.* The closed loop, with Alice's writing (no red, a green) and the answer asked
about (a green) side by side. Kin to kin leaves a clean loop. Then the answer brass up: its
red is carried the long way round to beside the writing's green, and nothing meets. The
writing's rings follow the same rule as Bob's corrections in the Third and make the
codebook entry. For all sixteen (writing, answer) pairs, the loop's rings read from the near
bend make $X^{j_2}Z^{j_1}Z^{k_1}X^{k_2}$. Half its trace is the dense-coding amplitude.
Kin to kin, and slide round first where the Lua allows it, keeps that half-trace, and the
loop is clean exactly when the writing is the answer.

Round 4 makes P4.3 a step (`[step=slide round]`): the state before, the red close in by the
far bend, is not pictured; its ghost places it, at place 4, and the slide arc runs inside the
loop from $55.6°$ down through the far bend, Bob's leg and the near bend to $-235.6°$, beside
the green. Its `marks=off` twin is round 3's P4.3; the two greens may also meet in two steps
(green meets green, then lift off), and nothing goes round while a ring is at rest.

```python
assert r3((cx, cy, a0, a1)) == (0.0, -0.65, 55.6, -235.6)            # inside the loop
assert np.isclose(ghost.at[0], px[1])                                 # from place 4
```

```python
assert np.isclose(0.5 * np.trace(R), amp)                      # the loop's rings: the amplitude
assert (PC.sequence(N) == []) == (w == a)                      # clean iff writing = answer
```

## V. `MendedJunction.py` — *The Mended Junction* (entanglement swapping, corrected; the master)

> "Nothing crossed the room but a letter." — Achilles

The real run on the table: the glass chimes, the letter goes the long way round, and Bob's
stone is flipped, then spun. The keepsake stone and Bob's stone are one piece, never having
met. The **blackboard** checks each branch: corrected, the far ends are exactly
$\tfrac12\Phi^+$, a quarter each. Uncorrected they are $\tfrac12B_k$, the very Bell state
the reading named. In doubled form with the record kept, the result is exactly
$\tfrac14\sum_k\ket{m_2m_1}\!\bra{m_2m_1}\otimes\ket{\Phi^+}\!\bra{\Phi^+}$: the record a
fair toss, the pair $\Phi^+$, and the two a product. With the record forgotten the pair is
$\Phi^+$. Uncorrected and forgotten it is $\tfrac14I$, two strangers' coins. Pushing half a
pair through Result 10.5's channel gives the same $\Phi^+$: the headline, bent.

```python
assert np.allclose(corrected, 0.5 * BELL)                      # every answer
assert np.allclose(uncorrected, 0.5 * bell(m1, m2))            # the named Bell state
assert np.allclose(full, np.kron(I4 / 4, Phi))                 # Result 10.8
assert np.allclose(pair_unc, I4 / 4)                           # the letters astray
```

The **sketchbook** makes three builds.

- **(a)** The pure branch, the S of two cups and a ringed cap with Bob's rings, is exactly
  $\tfrac1{2\sqrt2}$cup $=\tfrac12\Phi^+$, and `full_reduce` derives it. Uncorrected, it is
  the cup still carrying the reading's rings in bottom-leg order.
- **(b)** The master itself is hand-built in doubled form: two pairs and their mirrors, the
  reading box $M$ (CNOTs, Hadamard edges, two read-spiders), and the classically controlled
  correction box $C$. Along $c$ the flip comes first, fed by $x$, and the spin second, fed
  by $z$. The scalar compensations are itemized ($+2-4+2+2$). With the record kept, the
  master is exactly the drawn right-hand side: a cup on $(d,c)$, its mirror cup, and a fair
  toss dot on each classical wire, with $\tfrac18$ in front. `full_reduce` derives the
  whole master down to that. The corrected value cannot see a swapped record label, since
  it is uniform in the record, so the uncorrected master with the record kept is also
  checked: each record value carries its own Bell pair. Forgetting the record gives
  $\Phi^+$, and forgetting it without correcting gives $\tfrac14I$.
- **(c)** The skeleton, with $M$ drawn as one vertex and $C$ as the identity, has **ten
  lines as ten edges**: eight doubled quantum lines and the two classical wires
  $M\to x$ and $M\to z$. Drawing $C$ as four identity dots (fourteen edges) and applying
  the identity rule to exactly those dots produces the ten.

```python
assert same_picture(m, expected_master())                      # Result 10.8, exact
simp.full_reduce(h); assert same_picture(h, expected_master()) # derived
assert np.allclose(blk, 0.25 * np.outer(B, B.conj()))          # uncorrected: B_k per record
assert s.num_edges() == 10                                     # the ten lines
assert boxed.num_edges() == 10 and same_picture(boxed, s)      # produced by the id rule
```

*Pictures.* One block, one motion. First the table as it lay all morning, keyed: the
keepsake under its small cloth, the message stone, Alice's stone, and far off Bob's, every
stone's reflection slack, the drawing-cloth bare along the front. Then the glass chimed, with
its trays tipped and both tokens up in the dish. Then the table read, with the message stone
and Alice's gone into the glass and the keepsake and Bob's stone exactly where and as they
were. Last, the table after the letter is obeyed. The file checks that flip and spin are in
the state and the tokens are tipped out before Bob's stone, and that the picture is otherwise
**exactly** the one before, because the turned stone looks just as it did. The places obey
the L2 rule: the near group in the slab's left half, the Alice–Bob gap the widest and at
least twice each near gap, and no two stones or reflections touching. They are also checked
against the Lua's own constants.

Round 4 computes the glass and the table as steps, `marks=off`: the glass from the message
stone in the left cup (`[step=chime both up]`), the table `[step=into the glass]`, each the
same picture as round 3's. The block holds the ten, and a counted frame carries no mark: the
three frames draw **8, 2 and 4** lines (four things on the black with their reflections; the
two tokens in the dish, classical; the keepsake and Bob's stone), with no ghost, arc or pin,
and their log is the head and the step only. The letter is obeyed as a strip of three:
*flipped* (a swing arc over the crown of Bob's stone and down its far side, radius 0.25 about
$(0.88,0.086)$, $100°\to-8°$, T2), *spun* (half round and more about $(0.88,0.273)$,
$160°\to380°$, passing the company's side, T3) and *tokens out* (the envelope's ghost at
$(0.88,-0.865)$, handed up to Achilles). The flip and spin frames without their arcs are the
letter-open table exactly; the tip-out frame without its ghost is round 3's P5.4.

The tie is that a slack reflection means a half. Every stone the table draws is
$\tfrac12 I$ with purity $\tfrac12$ at the opening, after the reading on each of the four
answers (the keepsake and Bob's stone then the named Bell pair), and after the corrections.
Flip then spin is the correction $Z^{m_1}X^{m_2}$ for the answer the glass dropped. It takes
the pair from $B_{11}$ to $\Phi^+$ and leaves Bob's stone alone at $\tfrac12 I$. The
keepsake's tight reflection is its cloth's, an object outside the tie.

This file also checks the whole dialogue:
- 19 calls and 23 frames in 14 blocks (three strips), with none before the First Challenge;
- the keyed rule on true frames only (a prediction is never keyed), whose keyed pictures
  are exactly P1.1, P1.2, P1.3 and P5.1;
- the two predictions (`apart`, `more turned`), pale, unkeyed, unmarked; the eleven marked
  frames (2, 2, 3, 1, 3), each with its `marks=off` twin and every ghost's destination named;
- the log: 65 lines (23 heads, 39 entries, 3 strip lines), the same for both files; the
  25 quotes of the Lua's round-4 comments, each in its one challenge (the flip's and the
  spin's sides among them, T2 and T3); and the header's quote of the opening direction for
  the keepsake's hidden reflection, once, before the First Challenge;
- every mark inside its picture: a mark is drawn `overlay`, so nothing keeps it on the
  page. Its extent is read from its TikZ text (sampled cubics and arcs, circles, ellipses,
  rectangles, half its style's line width, the arrow's head) and must clear the linen's edge
  on the cloth, or the frame's box elsewhere, by more than 0.01 cm. This holds on the eleven
  marked frames and on 56 marked pictures of every step the Lua marks: the lower pair round
  the inside of its bend, the loop's short meeting inside it with the long arc kept lower,
  and the tale drawn in on every pattern;
- every caption and frame, with no fault on any call;
- the places, with the table only in the Fifth, the glass only in the First and Fifth, and
  never `wrapped`, `[cloth]`, `end to end`, `slide both round` or `corrections spin first`;
- P5.4 as the last picture, so the straight string of "Let Bob draw end to end" is not
  drawn;
- the six "Let Bob draw" hooks spelling C-O-E-C-K-E in order;
- the seventy-three fault sites, each raised by a call of its own; the bracket's items in
  every order, one picture.

```python
assert [PC.lines_drawn(p) for p in (b51, g52, b53)] == [8, 2, 4]   # the counted block, unmarked
assert r3((cx, cy, rx, ry, a0, a1_)) == (0.88, 0.086, 0.25, 0.25, 100, -8)   # the flip's arc
assert replace(v54, tokens_out=None) == v53                    # the turn shows nothing
assert np.allclose(after_both, BELL) and not np.allclose(pair, BELL)   # B_11 -> Phi+
assert np.allclose(rho, I2 / 2)                                # every stone: a half
assert "".join(h[0].upper() for h in hooks) == "COECKE"        # the six hooks
```
