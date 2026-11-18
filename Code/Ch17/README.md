# Chapter 17 — *Compiling by Rewriting*: companion code

There are four challenge files, one for each dialogue challenge. Each file checks its claim
**twice**. The blackboard does it with NumPy `assert`s. The sketchbook compares PyZX diagrams
as tensors through `tools/dfp.py`'s `matrix_of` / `same_picture`, which force
`strategy="naive"`. Where the chapter derives an equality by rules, the file derives it too,
with the rewrites of `pyzx.simplify`: `color_change`, `spider_simp`, `id_simp`, `pi_commute`,
`to_graph_like`, `full_reduce` and `teleport_reduce`, plus `zx.extract_circuit`. Running a
file is the book's analogue of type-checking.

```bash
# one challenge at a time
.venv/bin/python chapter17/Code/Ch17/FourEighths.py
# the whole chapter (must end "4 files, 0 failed")
bash check-code.sh 17
```

The files follow one recipe through a compiler. First they count its eighths and turn it,
move by move and at no cost, into a net. Then they cook that net until its four eighths meet,
add and vanish, by hand and by PyZX, checked against the matrices. The last two files run a
net without a recipe, reading it ring by ring with the halves sent forward, and take a recipe
back out of a cooked net, which works when such a forward order exists.

Each file also runs `pictures()`, at the end of its blackboard. The dialogue's prop pictures
(`\cardcloth`, `\secondhooks`, `\onewaytin`, `\tincloth`, `\outcloth`, and since round 4,
"Show the process" (card §3e), the strips `\cardclothsteps`, `\onewaytinsteps`,
`\tinclothsteps` and `\outclothsteps`: 18 blocks, 19 calls, 39 frames, eight strips of 28
frames among them, and one prediction frame, `expects=the joiners cancel`) are read out of
`dialog.md` and `dialog.raw.md` through `tools/dfp_props.py`. Their state is computed by
`tools/props_ch17.py`, a transcription of `pdf/props/ch17.lua` on the shared cloth and Office
kits. The transcription reads the Lua's places from the Lua itself, and is pinned to the Lua it
was proved against: every one of the Lua's 234 top-level chunks and the `.tex`'s 78 by its hash,
and the wrong rule's comment word for word (`check_pins`, run at import). It was diffed against
the Lua under texlua, every strip frame run as TeX runs it: the 19 dialogue calls (47 records,
keyed and unkeyed), 119 fault calls and about 45 700 swept calls (every card step and step pair,
every tin path, the hooks, the row's travels, the out-cloth's moves, the strips) give 0
differences. The one gap is the cloth kit's own "no room for a keyed word" fault, which
`tools/props_cloth.py` leaves untranscribed: 646 swept records with it are set aside on both
sides, none a call of the dialogue. The Lua's `tex.error` has 83 keyed sites (`terr` 4, `herr`
2, `cerr` 23, `rerr` 12, `oerr` 9, `xerr` 33; four bare re-raisings of a caught message are not
sites), each reached by its own case in `CookingDown.py`'s `FAULT_CASES`, with its own message,
and in any order with the others. Two sites fire only past another fault, as in the Lua:
`expects-marked` (`marks=on` on an expects frame comes with `marks-no-step`, or with a step
`card-expects-step`) and `strip-unknown` (an unknown strip has no frames, so any caption brings
`strip-captions`). `pictures()` ties each picture to the blackboard. A
cloth dump is read as a PyZX graph (`PC.graph_of`: a ring a spider of its colour turned as its
bead, a tube an odd number of times a Hadamard edge, a loose end a boundary), a tin's dial is an
end's turn, and the hooks' pair is the cooked picture.

**Pins (the cloth kit).** The shared cloth kit, `pdf/props/cloth.{lua,tex}`, is pinned in
`tools/props_cloth.py`, which every file imports through `props_ch17`; a change to the kit fails
every file of this chapter (see the note in `chapter10/Code/README.md`).

Round 4's strips are checked frame by frame. Each frame is the state its steps leave, in order
(`PC.frame_chain`), and the last frame of a block is round 3's picture of it. **Every frame of a
strip is tied to the blackboard by itself**, not only the last: every part-way frame of the card
is exactly $\tfrac14U$, every cooking frame times its counters is $\tfrac14U$, every tin frame's
next reading has the chance the row gives it, every frame of the half sent on has P3.3's value,
and every frame of the turns slid off is $U/\sqrt2$. Every ghost is a part of the state before
the step, drawn where that part was; every arc runs from where a part was to where it arrives
and clears the parts it passes on the drawn extent, with a positive gap; every hop and slide arc
bows at least its floor from the chord (a turn arc instead follows its circle round the knob). A frame with marks is never keyed, and a step that is refused draws none.
`pictures()` also checks the dialogue's rules: blocks and sizes, one frame height and one macro
in a block, captions in words only (no *it*, no echo of the paragraph after), every picture in
its place, the anchor phrases of both files (24 sentences start after a picture and 40 phrases
come before one, both files counted), the starred pictures (P1.1, P2.4, B3.1, P3.3, B4.1), and
the keyed words' clearances on drawn extents. A word's node box clears the frame by at least
$0.025$ cm and its letters clear every part drawn and every other word by at least $0.02$ cm.
Five words the Lua sets stand **above** their frame's top by design (the lids' names over the
tin, by $0.046$ cm; *second hooks* on the rack's top edge, by $0.175$; *bead* in the band over
the linen, by $0.075$), so the files hold them to that overhang and to the other three sides.
This is a structural check of the pictures, not a third proof.

Five things hold in every file.

- **Scalars are exact.** A bare joiner is $\mathrm{CNOT}/\sqrt2$, a Hadamard edge between two
  spiders at rest is $\mathrm{CZ}/\sqrt2$, and every picture compared with another carries its
  number as `scalar.add_power(k)` (a factor $\sqrt2^{\,k}$) or `scalar.add_phase(phi)`. PyZX's
  `c.to_graph()` carries a compensating number (4 for the recipe); the files reset it with
  `g.scalar = zx.Scalar()` to get the bare picture. **One block uses a licence:** in
  `OutAgain.py`, extraction of a general random circuit is asserted only up to a number $r$
  with $\lvert r\rvert=1$, as Duncan et al. give it (their Remark 2.2). The recipe's own
  extraction is asserted exact.
- **Top wire first.** The upper wire is qubit 0 (the joiners' controls), the lower is qubit 1;
  $U=(\mathrm{CNOT}_{12}(I\otimes T))^4$ is built from `add_gate("T", 1)` then
  `add_gate("CNOT", 0, 1)`, four times. The files assert that the joiner turned the other way
  up, or the quarter and half on the wrong wires, is a different picture.
- **Generic inputs where the claim is general.** $S$, $T$, CZ and the gadget $\Gamma(\alpha)$
  are diagonal, so symmetric, and are never used as generic matrices. The one-step formula is
  checked on a seeded random **non-real** state; the gadget rules, the red half against an end
  and the travelling half on seeded random angles (as `Fraction(k, 97)` in pictures).
- **Negative controls.** The trap $I\otimes Z$, a picture without its number, the turn on the
  control wire, the feet a quarter round instead of back, the joiner the other way up, the
  survey's label read as a turn, the tins read without the rule or with half of it, the
  master with a Hadamard lead unsleeved: each is asserted to fail.
- **No rewrite proves a bound.** `full_reduce` is asserted to keep the value and to lower the
  count of eighths where it does; nothing claims the count it reaches is the fewest.

## Four Eighths — `FourEighths.py`

Achilles counts the four eighths on the Crab's card and falls for the trap: the joiners come
in pairs, so they cancel, and four eighths make a half on the lower line. Following the card
with the upper line lit, two of the eighths turn: "Two of them turned. A quarter — and on the
upper line's account. ...Oh." The blackboard multiplies the recipe out, walks Achilles's
input step by step, and runs the parity count on all four inputs:

```python
assert np.allclose(U, kron(S, Z) @ CZ)                  # the headline, exact, scalar 1
assert not np.allclose(U, trap)                         # ...but the recipe is not I (x) Z
assert turned == 2
assert np.allclose(state, 1j * basis(1, 0))
assert np.allclose(U @ basis(x1, x2), 1j ** (x2 + (x1 ^ x2)) * basis(x1, x2))
```

It also evaluates *The Uncooked Net* directly, as a sum over the bits of its eleven
spiders (a sign and $1/\sqrt2$ per Hadamard edge), and gets exactly $\tfrac14U$. The
sketchbook turns the recipe's picture into the net by the chapter's three moves, and checks
the picture after each: `color_change` on each red target, then `spider_simp` for the four
controls on the upper line (the Tortoise's "Slide them together."), then the two dummies at
rest with their yellow boxes. Before the dummies the picture is still exact but not graph-like
(the big ring holds two ends); after them it is. The hand-built master is then compared with
all of these:

```python
assert net.num_edges() == 17                    # the upper wire's four lines, the lower wire's nine, the four ties
assert np.allclose(matrix_of(net), U / 4)       # Result 17.18, exact
assert isomorphic(net, h)                       # = the three moves' result
assert isomorphic(net, pg)                      # = PyZX's own to_graph_like of the recipe
```

`isomorphic` compares vertex types, phases, boundary sides and qubits, and edge types, with
`networkx`. The file also shows the trap in pictures: two joiners with nothing between them
fuse and fall apart (Hopf) to $\tfrac12$ times two bare wires, and an eighth between them
leaves the two reds unable to meet.

**The pictures.** P1.1–P1.4 are the card laid out on the Tortoise's cloth and the three moves
made on it. Every one of them, read as a graph, is the recipe's net, exactly $\tfrac14U$:
nothing is set aside in the First Challenge. P1.1 is the bare circuit picture. P1.4 has
seventeen lines and is the master spider for spider:

```python
assert np.allclose(matrix_of(g), U / 4)                       # each of P1.1-P1.4
assert graphs[3].num_edges() == 17                            # P1.4: seventeen lines
assert isomorphic(graphs[3], net) and is_graph_like(graphs[3])
assert isomorphic(graphs[3], pg)                              # PyZX's own net
assert not is_graph_like(graphs[2])                           # P1.3: the big ring holds two ends
```

One small ring's tube left off is a yellow box on the upper lead; the lower right lead without
its tube is a yellow box on the lower; both are asserted to be different pictures. The card's
columns hold from P1.1 to P1.2 and the lower line from P1.2 to P1.4.

**Round 4.** P1.3 becomes a strip (B1.3, `\cardclothsteps`): the upper rings slid together one
at a time, the first and second into one, the third into those, the fourth into those. **Every
part-way frame is exactly $\tfrac14U$**, and none of them is yet the master; with a ring's tie
dropped it is not $\tfrac14U$:

```python
assert np.allclose(matrix_of(g), U / 4) and R.after.aside == 0, R.step      # each frame of B1.3
assert not isomorphic(g, net)                                                # not yet the master
assert [round(a.sag, 3) for R in b13 for a in R.arcs] == [0.058] * 4        # L4: every hop 0.058
```

Each hop runs on its line's outer side from a ring's old place to its new one, bows at least the
hop floor `HOPSAG` (0.055 cm; the Lua's own) and clears the net; the ghosts stand at the old
places. P1.4 (the small rings put on) is unmarked: the rings come from her box, and a marked
tangle is refused (`card-marks-tangle`). The parts a step does not touch stay where they were,
frame to frame.

## Cooking Down — `CookingDown.py`

The Tortoise's cloth turns the sandwiches into hubs and the hubs into one; Achilles checks the
four settings on the card: "A quarter whenever the upper is lit, a half when the lower is —
and when both are, a half taken back." The koekpan presses a copy of the tangle to the same
three pieces, and the Crab is honest about it: "But it never promises the fewest." The
blackboard checks the gadget $\Gamma(\alpha)=\mathrm{diag}(1,e^{i\alpha},e^{i\alpha},1)$ on
twenty seeded angles, gadget fusion, $U=(I\otimes S)\Gamma(\pi/2)$, Achilles's four settings
as the angles of $U$'s diagonal, and Result 17.7's choice of quarters. The sketchbook builds
every picture of §17.6's chain and compares each with the one before, with its number:

```python
assert same_picture(sw, hub_picture(a).done(-1))           # sandwich = 2^-1/2 hub (Result 17.2)
assert same_picture(two, hub_picture(al + be, top, bot).done(-1))   # two hubs = 2^-1/2 one hub
assert same_picture(circuit, step1_n)                      # = 1/2 (eighth, hub, eighth, hub)
assert np.allclose(matrix_of(qhub), CZ / R2) and np.allclose(matrix_of(hedge), CZ / R2)
assert same_picture(circuit, cooked_n)                     # circuit = 2^-3/2 cooked
```

Steps (ii) and (v) are derived by `spider_simp`; step (iv), the quarter hub for the Hadamard
edge, is a translation checked by evaluation, as the chapter says. Then PyZX cooks the master
by its own moves:

```python
simp.full_reduce(h)
assert same_picture(h, net)                                # value kept, number tracked
assert zx.tcount(net) == 4 and zx.tcount(h) == 0
assert h.phase(up) == Fraction(1, 2) and h.phase(down) == 1
assert np.isclose(h.scalar.to_number(), 2 ** -1.5)
assert np.allclose(matrix_of(ce.to_graph()), U)            # extraction: exact
assert names == [("S", 1), ("CNOT", 1), ("CNOT", 1), ("CNOT", 1), ("S", 1), ("CNOT", 1)]
```

The last line is `teleport_reduce` (which returns a new graph and leaves its input alone,
also asserted): every joiner kept, quarters in the first and fourth eighths' places. The
recipe followed by the adjoint of the extracted circuit cooks to two bare wires. On twenty
seeded random circuits (`zx.generate.cliffordT(3, 40, p_t=0.2)`, seeds 1–20), `full_reduce`
keeps the value exactly in all twenty and lowers the count of eighths in eighteen (seeds 2
and 13 unchanged): `assert lowered == 18 and flat == [2, 13]`.

**The pictures.** P2.1–P2.3 are the card cooked by hand, with a bone counter set by the teapot
for each hub hung and for the two hubs made one. What the counters stand for is computed from
the blackboard, not read: a sandwich is $2^{-1/2}$ times its hub, and two hubs on the same feet
$2^{-1/2}$ times one hub, so three counters are $2^{-3/2}$. Each cloth times its counters is the
recipe, and P2.3 alone is the cooked picture. P2.4, the pair on the second hooks, is that same
picture:

```python
assert np.allclose(number[p.state.aside] * matrix_of(g), U / 4)     # P2.1, P2.2, P2.3
assert np.allclose(matrix_of(cooked), U / R2)                        # P2.3: the cooked picture
assert same_picture(hooks_graph(p24.state), cooked)                  # P2.4: the pan's answer
assert not same_picture(hooks_graph(PC.hooks_state("at a half, a quarter round")), cooked)
```

The first hub and its end stay where they hung from P2.1 to P2.2, and the counters lie in the
band below the linen, clear of it and its shadow.

**Round 4.** The cooking is drawn move by move: B2.1 (the greens met above the second eighth,
its bead sent to an end, the cradle undone), B2.2 (the upper feet met, the lower feet slid
together), B2.2a (the hubs merged) and B2.3's strip (the feet slid apart, the hub lifted with the
sleeve laid, nothing hanging). **Every one of these eleven frames, times its counters, is exactly
$\tfrac14U$**, and a move that sets no counter aside leaves the picture exactly as it was:

```python
assert [R.drawn.aside for R in cooking] == [0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3]
assert np.allclose(number[R.drawn.aside] * matrix_of(g), U / 4), R.step
assert same_picture(PC.graph_of(R.dump), PC.graph_of(P.dump)), R.step       # no counter, no change
```

The bead's turn lost on the way is asserted to fail. The eighth's end sent up from its ring has
its arc (L5: 0.152 cm, bowed 0.030, clear of the cords and ties by 0.020 and of the rings by
0.022), and the direction before B2.1 says where the end lies (E3: up from the ring, between the
short strings), checked in both files against the drawn end. B2.3 opens with Achilles's guess,
drawn pale beside it (`expects=the joiners cancel`). Its ring is the Lua's wrong rule applied by
the model, every joiner gone and the eighths' turns summed on the lower line. Read as a picture,
the guess is $I\otimes Z$, which is $(I\otimes T)^4$ and not $U$ at any normalization. It agrees
with $U$ when the upper line is dark and fails when it is lit, as the dialogue finds. The pale
ring reads under the veil (L8: an extra edge, 0.169 cm across). The rule's comment stands word for
word in the card and its name in both dialogue files.

`check_faults()` runs the 83 `FAULT_CASES`. Each site is reached by its own case, with its own
message (alone, but for `expects-marked` and `strip-unknown`, which fire only past another fault),
and in any order with the others (at least twenty permutations of every multi-fault case). A
refused step draws no marks, and a step taken draws them. The hooks pressed draw nothing ("It
jams if watched"), while lifted off they draw the two rings' ghosts and no arc.

## The One-Way Row — `OneWayRow.py`

The Crab brings out his tins: "These are the one thing I own that works only because you do."
Tin 1, read as preset, comes up lit at the last lid; the tins read by his rules come up dark.
The Tortoise shows why on her cloth: "A lit reading sends a half forward: the next end turns
the other way, and the one after gets a half." The blackboard checks Result 17.8 on a random
non-real state, with Danos and Kashefi's effect written as the end turned $-\alpha+s\pi$, and
computes every branch of the three-dish row:

```python
assert np.allclose(branch, want)                    # (1/sqrt2) X^s H Z(-alpha) psi
assert not np.allclose(wrong, want)                 # the end turned +alpha: no
assert np.isclose(lit, 0)                           # by the rule: the last is dark, surely
assert np.allclose([none[k][1] for k in sorted(none)], [0, 0.5, 1, 0.5])   # without it
assert np.isclose(overall(other_only), 0.5) and np.isclose(overall(half_only), 0.25)
assert np.allclose(kept, pauli @ MU)                # kept unread: X^{s2} Z^{s1} |mu>
```

It also runs Danos and Kashefi's corrections ($X_2Z_3$ after reading 1, $X_3$ after
reading 2) on random angles: all four branches leave exactly the same state. The sketchbook
builds the closed row and checks it is exactly four times the amplitude for all eight
readings; fusing each end into its ring (a reading, `spider_simp`) changes nothing. It checks
the one-step picture is exactly $X^sHZ(\varphi)$ (derived by `spider_simp` and `id_simp`),
Result 17.10 with its number $e^{i\varphi}$ (derived by `pi_commute`), and both equations of
Result 17.11 by building both sides:

```python
assert np.isclose(P([p1 + 1, p2, p3]), ex(p2) * P([p1, -p2, p3 + 1]))
assert np.isclose(P([p1, p2 + 1, p3]), ex(p3) * P([p1, p2, -p3]))
```

The first equation is also derived: `pi_commute` at the second ring sends the half out on
every leg, and `pi_commute` at the second end takes it off, turning the end the other way.
PyZX leaves the number $e^{i\varphi_2}$ unwritten, so the file writes it. (In pyzx 0.10.6,
`pi_commute` puts a red *at rest*, not a red half, on a leg whose neighbour cannot take the
half; the file puts a red at rest on the end's cord first, so that branch is never used, and
checks the value after each step.)

**The pictures.** P3.1 is Tin 1 read as preset: its dials at $\pi/4, 0, -\pi/4$, read lit,
dark, and then lit at the last lid, as the branch $(1,0)$ must be without the rules. P3.2 is a
tin read by the rules in the branch $(1,1)$: its last dial stands at $5\pi/4$ and its last
hollow is dark, with probability 1 there. The tin row's two pictures are closed rows, each four
times its branch's amplitude, and exactly equal, since the half sent on meets an end at rest:

```python
assert same_angle(setting(p32)[2], rule[(1, 1)][2]) and same_angle(setting(p32)[2], 5 * np.pi / 4)
assert np.isclose(rule[(1, 1)][1], 0)
assert PC.tin_state(p32.drawn_bracket) == PC.tin_state(r3)            # read by the rules, round 3's bracket
assert np.isclose(value(g33), 4 * amp(t33)) and np.isclose(value(g34), 4 * amp(t34))
assert t33[1] == 0 and same_picture(g33, g34)              # the travelling half's number is 1
assert p34.state.ends == list(PC.PRESET)                   # like a tin read dark
```

A dial turned after its lid is lifted, or a lid lifted out of order, is refused as the Lua
refuses it; the row's rings stay at 0, 2 and 4.

**Round 4.** B3.1 lifts Tin 1's lids one at a time: the tin as set, then the first lid lit,
the second dark, the last lit. Each reading has the chance the row gives it as the dials stand,
a coin each for the first two and, with no rule turned, the last lit surely. B3.1a turns Tin 2's
last dial the other way, back through rest, from seven eighths to one, which is the rule's
setting for $(0,1)$. B3.2 reads Tin 3 by the rules frame by frame, and its last lid is dark
surely:

```python
assert [np.round(reading_chance(R.drawn, k), 12) for k, R in enumerate(b31[1:])] == [0.5, 0.5, 1.0]
assert same_angle(np.pi / 4 * b31a.after.lids[2]["s"], rule[(0, 1)][2])
assert np.isclose(reading_chance(b32[1].drawn, 1), 0.5) and np.isclose(reading_chance(b32[3].drawn, 2), 1)
```

A lid lifted has a lift arc bowed to its right (0.064 cm, clear of the lid where it lay and where
it is laid). A dial turned leaves the pointer's ghost where it stood and a turn arc on its circle
round the knob, the way it went. The other way sweeps across rest, and its arc's body clears the
dots, the knob and both pointers (L7: B3.1a's arc 0.170 cm; clear of the dots by 0.032, the knob
by 0.097, the pointer by 0.028; its head's corner comes 0.010 from a dot). B3.4 sends the half on
step by step: off the first end onto the tie (green, by its ring), past the tube (red), out on
every other cord, on to the ends. **Every frame, its halves read in, has exactly P3.3's value,
number included.** Read without its halves, or with a half the other colour, a frame does not
have that value:

```python
g = row_graph(R)
assert same_picture(g, g33), R.step                              # each frame of B3.4
assert not same_picture(other, g33), R.step                      # a half the other colour
assert sorted((round(a.len, 3), round(a.sag, 3)) for a in b34[2].arcs) == [(0.152, 0.047), (0.268, 0.068)]
```

Each half's ghost lies where it lay the frame before, and every arc runs from a ghost to a half or
an end. Both arcs of "out on every other cord" are drawn (L6), clear of the net by 0.026 and
0.023.

## Out Again — `OutAgain.py`

Achilles takes the recipe out of the cooked pieces: "Slide them apart." Then the Tortoise
lays out the V, and he looks for an order in which to read its arms: "...Oh. Each has to go
before the other." The blackboard searches every map and every order, by brute force, for a
flow (Danos–Kashefi's Definition 2) and a gFlow (the chapter's three conditions):

```python
assert fl is not None and fl[0] == {1: 2, 2: 3}               # the row
for V_in in ([], ["a"]):
    assert find_flow(V, E, V_in, ["c"]) is None               # the V: no flow
    assert find_gflow(V, E, V_in, ["c"]) is None              # and no gFlow
```

PyZX's own `gflow` gives the same verdicts. The blackboard also reduces
$B=\bigl(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\bigr)$ to $I$ with one row operation
and checks Result 17.15 as a sum over bits. The sketchbook checks the V's travelling half
(a lit half on $a$'s end lands on $b$'s end and on $c$'s output, with no number), and then:

```python
assert np.allclose(matrix_of(frontier([[1, 0], [0, 1]])), kron(H, H))
assert same_picture(lhs, good)                    # joiner green up, red down (Result 17.15)
assert not same_picture(lhs, flip)                # the other way up: no
assert same_picture(fused, apart)                 # Result 17.16: the turns slide off
assert is_flow(E, V_in, V_out, f, order)          # the master's causal flow (Section 17.4)
assert gflow(net) is not None                     # Lemma 3.7
```

Result 17.15 is also checked by the rules: the left side followed by the adjoint of the right
side cooks under `full_reduce` to two bare wires with the number $\tfrac12$. On the twenty
random circuits, extraction succeeds on all twenty, each equal to the original times a unit
number and none exactly (`assert np.isclose(abs(r), 1)`, `assert exact == 0`). A gFlow is
asserted on the sixteen cooked nets without a phase gadget, where Duncan et al.'s theorem
covers it (`assert gflow(r_g) is not None`); on the four with a gadget (seeds 6, 8, 10, 18)
PyZX's finder finds none (`assert gflow(r_g) is None`). That second assert is a statement about
PyZX's finder and *ordinary* gFlow only; it claims nothing about the extended notion of
Backens et al., which the chapter has not read.

**The pictures.** P4.1 is the pan's answer slid apart, exactly $U/\sqrt2$ and the new card: a
sleeved tie, then a quarter on the upper line and a half on the lower. P4.2 is the frontier
picture with the crossing tie, and P4.3, the joiner laid at its ends, is the same picture; with
the joiner's colours swapped it is not. P4.4 is the V, read from its dump as an open graph:

```python
assert same_picture(g41, two_wire([(0, Fraction(1, 2)), (1, 1)]))
assert same_picture(g42, frontier(B))
assert same_picture(g42, g43)
assert swapped != p43.dump and not same_picture(g42, PC.graph_of(swapped))         # the joiner the other way up
assert find_flow(V, E, vin, corner) is None and find_gflow(V, E, vin, corner) is None
```

**Round 4.** P4.1 becomes a strip (B4.1, `\outclothsteps`): the pan's answer laid back, then
the upper ring's turn carried right, then the lower's. **Every frame is the same picture,
$U/\sqrt2$**, the first frame included. A turn copied instead of slid, or carried to the wrong
value, is not it. Each move is a hop on its line's outer side from the ring at rest to the ring
that took the turn, bowed 0.058 cm (at least `HOPSAG`), with the bead ghosted on the ring at
rest. The tie and the rings at rest do not move.

This file also runs the checks on the whole dialogue: the sections and every picture in one,
18 blocks, 19 calls and 39 frames (asserted as sums of the per-file tables), eight strips of 28
frames, the one prediction frame, only a strip's before-frame ever keyed, the starred form on the
first picture of each macro (a strip counted by its frames' macro), every call in its place in
both files, no negative anchor in either file, the
form check of the identity card (the Tortoise's "Slide them together." once in each of the
first three challenges, Achilles's "Slide them apart." in the Fourth), and the card's greps.

There is no proof-assistant tier; `bash check-code.sh 17` is the chapter's check.
