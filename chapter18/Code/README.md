# Chapter 18 — *Against Noise*: companion code

There are five challenge files, one for each dialogue challenge. Each file checks its claim
**twice**. The blackboard does it with NumPy `assert`s. The sketchbook compares PyZX diagrams
as tensors through `tools/dfp.py`'s `matrix_of` / `same_picture`, which force
`strategy="naive"`, and every sketchbook also lets PyZX's own rewrites derive what the chapter
derives: `full_reduce`, `spider_simp`, `to_gh`. Running a file is the book's analogue of
type-checking. There is no proof-assistant tier.

```bash
# one challenge at a time
.venv/bin/python chapter18/Code/Ch18/BlindVote.py
# the whole chapter (must end "5 files, 0 failed")
bash check-code.sh 18
```

The files follow one message through a storm. First they count what a vote buys and draw the
one copier that does not clone. Then they send the message down the storm line, ask only
pairs, mend blind and get it back exactly, in NumPy and by PyZX's own rewriting. The last
three find what the pairs miss and build the nine strands, decide which questions a bell may
ask at all, and join two protected messages by a split and a merge, the same ring asked in two
colours.

Each file also runs `pictures()`, at the end of its blackboard. The dialogue's prop pictures
(`\plaintin`, `\stormline`, `\partcloth`, `\linecloth`, `\tubecloth`, `\splitcloth`:
seventeen calls, four, four, three, two and four by challenge) are read out of `dialog.md` and
`dialog.raw.md` through `tools/dfp_props.py`. Their state is computed by `tools/props_ch18.py`,
a transcription of `pdf/props/ch18.lua` on the shared cloth kit. The transcription reads the
Lua's places and tables from the Lua itself, and a drift guard finds every rule it transcribes,
word for word, in the Lua and counts the Lua's `tex.error` call sites (25: `\plaintin` 3,
`\stormline` 7, `\partcloth` 3, `\linecloth` 4, `\tubecloth` 6, `\splitcloth` 2). It was
diffed against the Lua under texlua: 96 reference lines (the seventeen calls of the dialogue,
keyed and unkeyed, and the unpictured states between them), 28 error calls and 3000 fuzzed
calls give 0 differences, and all 25 sites are reached. `pictures()` ties each picture to the
blackboard. A cloth dump is read as a PyZX graph (`PC.graph_of`: a ring a spider of its colour
turned as its bead, a ring lying on a cord spliced into it, a cord with an odd number of tubes a
Hadamard edge, a loose end a boundary), and a storm-line state is read as the labels of the
file's `storm_line`. It also checks the dialogue's rules: blocks and sizes, captions in words
only and no echoes, every picture in its place, the anchor phrases, the starred pictures (P1.1,
P1.2, P1.4, P2.2, P3.2, P5.1), and the keyed words' clearances on drawn extents: a word's node
box clears the frame by at least $0.025$ cm and its letters clear every part drawn and every
other word by at least $0.02$ cm. This is a structural check of the pictures, not a third proof.

Six conventions hold across the files.

- **Scalars are exact.** The encoder, a green spider with one leg in and three out, is
  $\mathrm{Enc}=\ket{000}\bra0+\ket{111}\bra1$ with no number; a two-legged red $\pi$ is $X$
  and a green $\pi$ is $Z$. The only numbers that are not 1 are written in the code: the
  copy's $\tfrac12$ (`scalar.add_power(-2)`), a weight-$w$ check's $2^{1-w/2}$, the double-tied
  bell's $(1+e^{i\theta})/2$ (`scalar.add_node(theta)` then `add_power(-2)`), and the bare
  joiner's $1/\sqrt2$. `zx.Circuit(...).to_graph()` already carries its compensating
  $\sqrt2$ and is exact; `SplitAndMerge.py` asserts that and then resets it with
  `g.scalar = zx.Scalar()` for the bare joiner. **The one licensed "up to"** is Shor's sign:
  `NineStrands.py` asserts that a corrected single-wire Pauli returns the code state times a
  number $r$ with $\lvert r\rvert=1$, a global phase, the same for every message.
- **Top wire first.** Strand 1 is qubit 0. Operators multiply right to left, pictures read
  left to right.
- **Generic inputs.** $\mathrm{Enc}$, $X$, $Z$, $H$ and the check projectors are real, so no
  transpose/dagger mix-up can hide in them; the messages, density matrices, Kraus weights and
  one-wire errors are seeded random and **non-real**, and every "for every" of the chapter is
  checked on them.
- **Separate strands.** PyZX's simple graphs keep at most one edge between two vertices, and a
  second `add_edge` between the same pair is silently dropped. So wherever the chapter draws
  two spiders joined by more than one strand (the encoder and its mirror, Shor's round trip,
  the Knill–Laflamme pictures, the double-tied bell) the file puts a ring on every such
  strand (the error's own ring, or a ring at rest: bare cord, row 57) and asserts
  `num_edges()` after building.
- **No ground vertex.** The one doubled picture (row 136) is built by hand as a wire and its
  mirror copy: pyzx 0.10.6 refuses grounded graphs.
- **No rewrite proves a bound.** The threshold theorem has no file; the vote of votes is a
  classical toy on the blackboard, and the distances are found by listing Pauli words.

## Three for One — `ThreeForOne.py`

The Crab reads his ledger ("About one hollow in ten read false.") and lifts a triple, dark,
dark, lit; on the Tortoise's cloth a dark point copies through the green ring, and a point
turned a quarter round slides in instead: "...Oh. You can't copy a turn; you can only spread
it." The blackboard enumerates the eight flip patterns for a bit set dark and a bit set lit,
at $p=\tfrac1{10}$, $0.37$ and a random $p$, and compares the quantum code's logical channel
with the bit-flip channel of probability $F(p)$ as $4\times4$ superoperators:

```python
assert np.isclose(wrong, F(p))                                   # both settings
assert np.isclose(F(F(0.1)), 0.002308096, atol=1e-12)            # the ledger's vote of votes
assert not np.allclose(ENC @ psi, kron(psi, psi, psi).ravel())   # spread, not cloned
assert np.allclose(full, superop([np.sqrt(1 - pL) * I2, np.sqrt(pL) * X]))
```

The sketchbook checks that the green 1→3 spider is $\mathrm{Enc}$ (and not the red one), and
derives the rest. The encoder followed by its mirror reduces under `full_reduce` to one bare
wire with the number 1. A red point, dark or lit, pushed into the encoder reduces to three
separate points with nothing joining them and the number $\tfrac12$. A green point fuses in
under `spider_simp`, and the result has rank 2 across a cut, which no three points have:

```python
assert h.num_vertices() == 2 and h.num_edges() == 1              # Enc then its mirror
assert len(spiders) == 3                                         # the copy: three points...
assert np.isclose(h.scalar.to_number(), 0.5)                     # ...and the number 1/2
assert np.linalg.matrix_rank(m) == 2                             # the turned point: entangled
```

One block is a **translation check, not a derivation** (T1): a weighted sum of doubled
pictures is not one diagram, so nothing rewrites it. The doubled wire with a red half on the
wire and one on its mirror copy evaluates to $X\otimes\bar X$; the weighted sums reproduce the
bit-flip, phase-flip and depolarizing channels on a random non-real $\rho$ (for $Y$ the
mirror's $-i$ cancels the wire's $i$, so the doubled $Y$ needs no number), and a half on the
wire alone does not.

**The pictures.** P1.1 is the spent tin, read dark, dark, lit: with the ledger's "Set dark"
one hollow is false, the last, and the vote of the three is the setting. P1.2 is the dark point
pushed into the parting ring: its graph is **exactly twice** the point knotted in
(`point_into_encoder`), so the bone counter by the teapot stands for $\tfrac12$, the number the
blackboard asserts (`ENC @ red_point == three / 2`); once is asserted wrong. P1.3 is the turned
point pushed in, equal to one green ring a quarter round with its three cords, and not equal to
three turned points at any power of $\sqrt2$: nothing is set aside. P1.4 is the storm line as
strung, whose cords are counted by the Lua itself (`line_cords()` is 18, the master's edges).

```python
assert same_picture(g12, twice) and not same_picture(g12, knotted)
assert same_picture(g13, point_into_encoder(VT.Z, Fraction(1, 2)))
assert PC.line_cords() == 18 == len(p14.shown.spans)
```

## The Blind Vote — `BlindVote.py`

A quiet run comes through clear. Then a gust at the door, both bells ring, and the Crab mends
without looking: "Both bells rang. So the middle strand disagrees with both its neighbours, and
it's the middle I mend." The subject, in the Tortoise's voice first: "Ask the pair, not the
one." The blackboard checks Result 18.6 for all eight gust patterns and all four outcomes on a
random non-real message, the headline for the four single gusts, Result 18.9 for all eight,
and the free-mending formula for all $2^8$ labellings; the negative controls leave the gust
unmended or mend the wrong strand. Five random non-real Kraus operators in
$\operatorname{span}\mathcal E_X$ are undone exactly, and the kitchen glass is clear for all six
messages under one gust, while two gusts cloud dark, lit and the quarter turns:

```python
assert np.allclose(lhs, (s == sigma(e)) * I2)                    # the headline, scalar 1
assert np.allclose(recovered, coded)                             # every mixture of single gusts
assert np.isclose(glass(MARKS[name], two), 0)                    # two gusts: the vote goes wrong
```

The sketchbook builds *The Storm Line* by hand, as the chapter draws it: five green and eight
red spiders, plain edges only, every ring drawn whatever its bit.

```python
# the lead-in and the lead-out, twelve strand lengths (four on each strand), four ties
assert g0.num_edges() == 18
assert np.allclose(matrix_of(g), storm_value(e, s, c))           # all 2^8 labellings
assert same_picture(g, moved)                                    # step (i) of Section 18.7
```

Then `full_reduce` derives, labelling by labelling, what the chapter's five moves derive: a
bare wire with the number 1 when $s=\sigma(e)$ and $c=e$; one green $\pi$ between two
Hadamard edges (PyZX's graph-like form of a red half, $X$) with the number 1 when
$c=e\oplus111$; and `scalar.is_zero` otherwise. The double-tied bell (Result 18.8) is checked
at rest, at a half and at a random turn, on a green spider with one, two or three other legs
at a random turn; `full_reduce` takes it off with the number 1 at rest and to zero at a half.
Row 138, the bell that asks a pair, is exactly $\tfrac12(I+(-1)^sZ\otimes Z)$, and the two
bells together are the four $P_s$.

**The pictures.** P2.1 is the line after the blind mend: the middle mending ring at a half is
$c(s)$ for the bells' $s=(1,1)$, and `storm_line((0,1,0),(1,1),(0,1,0))` is the identity, so
the glass is clear (unmended it is not). Against P1.4 only that bead and the glass differ. P2.2
is the line laid on the cloth ring for ring: its graph has the master's eighteen edges, fifteen
vertices and vertex signature, and is `same_picture` with the master. P2.3, the gust's half
sent through the tell ring, equals `halves set`, which equals the line with the gust, the bells
and the mend all at a half; without the halves on the ties it does not. P2.4, the line drawn
straight, equals P2.2.

```python
assert signature(g22) == signature(master) and same_picture(g22, master)
assert same_picture(halves, gust) and same_picture(g23, halves)
assert same_picture(g24, g22) and np.allclose(matrix_of(g24), I2)
```

## The Turn the Bells Miss — `NineStrands.py`

The Tortoise turns the middle tell ring a half by hand; both bells stay silent, and the glass
comes up cloudy: "Cloudy! ...Oh. The bells never heard a thing, and it's spoilt all the same."
The Crab: "Then my storm line is half deaf." The blackboard checks Result 18.12 for each wire
and outcome, the glass for all six messages (dark and lit clear, the four turns cloudy), the
phase-flip code for all single phase flips (and its blindness to bit flips), and Shor's code
against Gottesman's eqs. (9)–(10) divided by $2\sqrt2$, with Table 2's eight checks. Of the
twenty-eight single-wire operators it counts the syndromes class by class and mends each by
the chapter's rule, an $X$ inside the block and a $Z$ on the block's first wire:

```python
assert len(set(syn.values())) == 22
assert len(cls["X"]) == 9 and len(cls["Y"]) == 9 and len(cls["Z"]) == 3
assert np.isclose(abs(r), 1)                                     # up to a global phase |r| = 1
assert np.allclose(out, r * ENC9)                                # the same r on both codewords
assert rs <= {1, -1, 1j, -1j} and rs != {1}
assert not np.allclose(out, r * ENC9)                            # two wires hit: not mended
assert np.isclose(weight, np.linalg.norm(A) ** 2 / 2)           # any error on one wire (Theorem 2)
```

The last line runs for a random non-real error on each of the nine wires, over all $2^8$
outcomes; the single-wire depolarizing channel is also undone exactly on a density matrix. The
sketchbook derives Result 18.12's picture: *The Storm Line* with a green half on a tell ring
reduces under `full_reduce` to one green $\pi$ on plain edges, number 1. `to_gh` turns the red
encoder green with a Hadamard edge on every leg, and the red encoder is
$H^{\otimes3}\mathrm{Enc}\,H$; the whole storm line in the other colour is the old one between
two yellow boxes, for all labellings. Shor's encoder, green, three Hadamard edges, green, is
$\mathrm{Enc}_9$ exactly (and not without its yellow boxes), and its round trip, nine separate
strands, reduces to a bare wire with the number 1.

**The pictures.** P3.1 is the Tortoise's turn: the line with the middle tell ring at a half is
**exactly** $Z$, not the identity, though no bell rings; dark and lit would read clear, every
turned mark cloudy. Against P2.1 the middle mending ring is back at rest and the tell ring is at
a half. P3.2 is the fresh parting ring, with its pairs of tubes, and drawn through: all three
graphs equal the green encoder, and the red ring with its tubes taken off is $H^{\otimes3}
\mathrm{Enc}\,H$. P3.3, the nine strands, is `shor_encoder()` and $\mathrm{Enc}_9$ exactly;
without its tubes it is not.

```python
assert np.allclose(line, Z) and not np.allclose(line, I2)
assert same_picture(g, green)                      # fresh, pairs, drawn through
assert same_picture(g33, shor_encoder()) and np.allclose(matrix_of(g33), ENC9)
```

## What a Bell May Ask — `WhatABellMayAsk.py`

The Crab clips his new bell to one ring, proudly: "Ask the one, not the pair." Dark and lit
come through clear; a quarter round comes up cloudy, and the Tortoise says why: "A bell that
can tell you the message spoils every turned one." The blackboard checks Knill and Laflamme's
conditions in Gottesman's two forms: $C=I_4$ for the single gusts, failure for a question
about the value, and for Shor's twenty-eight a Hermitian $C$ of rank 22, with the rows of
$Z_1,Z_2,Z_3$ equal. The clip-bell's joint table is the product of its margins, a fair coin in
each for a turned message:

```python
assert np.allclose(C, np.eye(4)) and np.linalg.matrix_rank(C) == 4
assert np.allclose(DEC @ clip @ ENC, np.outer(ket(0), ket(0)))  # the clip-bell: not a number
assert np.linalg.matrix_rank(C) == 22                            # Shor: degenerate
assert np.allclose(joint, np.outer(p_bell, p_clear))             # independent of the bell
```

It also lists Pauli words over GF(2): the bit-flip code has $Z_1$ in
$N(\mathcal S)\smallsetminus\mathcal S$, so $d=1$; Shor's code, and the seven- and five-qubit
codes of Gottesman's Tables 3 and 4, have nothing of weight at most 2 there and something of
weight 3. The toric code is built at $\ell=2,4,5$ only: the stars and faces commute, satisfy
exactly two relations, and protect a space of dimension 4 (at $\ell=2$ also as the trace of
the product of the check projectors). A loop of $Z$ round the torus is exhibited as an
undetected error of weight $\ell$; that is an example, not Kitaev's bound. The quantum Hamming
and Knill–Laflamme bounds are checked as arithmetic, and the vote of votes as a toy: fixed
points $0,\tfrac12,1$, $F(0.6)=0.648$.

The sketchbook derives the sixteen Knill–Laflamme pictures of $\mathcal E_X$ with
`full_reduce` (a bare wire with the number 1 on the diagonal, zero off it), leaves one green
$\pi$ for $Z_1$, and shows that the clip-bell's picture inside the code reduces to a picture
with no path from input to output: the wire is cut. One block is a **translation check, not a
derivation** (T2): a weight-$w$ check, $w=1,\dots,5$, both $s$, evaluates to $2^{1-w/2}$ times
its projector, and its colour-swapped twin to the same number times the $X$ projector. The
projector property is then derived: `full_reduce` of the check followed by itself is
$2^{1-w/2}$ times the check, for $w=2,3,4$.

**The pictures.** P4.1 and P4.2 carry the clip-bell on the first tell ring, every bead at rest.
The clip-bell asks $Z_1$ alone, with both branches asserted: silent, $(I+Z_1)/2$, which leaves
the encoded dark message as it was (P4.1, read at dark: clear); ringing, $(I-Z_1)/2$, which
leaves the encoded lit message whatever was sent, so the Tortoise's quarter round reads cloudy
with probability $\tfrac12$ (P4.2). Between the pictures only the glass changes.

## Split and Merge — `SplitAndMerge.py`

The Tortoise unties her joiner into a split and a merge; the penny comes down heads, then
tails, and the green half left on the merge's first incoming cord slides back to the upper
line, where the Crab's half cancels it: "A joiner again." Then the synthesis: "Everything
tonight was one ring, asked in two colours." The blackboard writes de Beaudrap and Horsman's
four operations (as $U_{\mathrm{sm}},U_{\mathrm{ro}},K_{j,\mathrm{sm}},K_{j,\mathrm{ro}}$),
checks both negative branches in the first of their two conventions (and that the other
differs by a half on the child), and the joiner in both branches:

```python
assert np.allclose(B1, kron(Z, I2) @ CNOT / np.sqrt(2))
assert np.allclose(kron(Z, I2) @ B1, B0)                         # mended
assert np.allclose(B.conj().T @ B, np.eye(4) / 2)               # probability 1/2 for every input
assert np.allclose(K1_RO @ pp, 0)                                # a bare merge on |++>: certain
```

The coin is also checked on five random product inputs and one random entangled one. The
sketchbook checks the four spiders and both negative-branch pictures exactly, then derives:
`full_reduce` takes the surgery picture and the bare circuit picture to the same picture;
`spider_simp` slides the negative branch's half into the split, from the merge's cord and from
the upper wire alike, so the two pictures become one; and after the Crab's half, `spider_simp`
leaves the bare joiner, spider for spider. Each branch followed by its mirror reduces to two
bare wires with the number $\tfrac12$, which is the fair coin drawn. Last, the synthesis: the
split is the copier $\delta$, and two splits in a row fuse into the parting ring.

**The pictures.** P5.1, the joiner clipped across, is the bare joiner $\mathrm{CNOT}/\sqrt2$,
and P5.2, drawn out long, is the same picture: a split and a merge. The half on the string and
the half slid into the split are $(Z\otimes I)\,\mathrm{CNOT}/\sqrt2$; P5.3, slid apart, is
that branch with the half on the upper line; P5.4, the Crab's half slid into hers, is the
joiner again, and its table is the drawn-out one's. The split's green ring stays at $(1,0)$ and
the merge's red at $(3,2)$ throughout. The file then checks the whole dialogue: the sections in
order with seventeen pictures, the starred form on the first of each macro, none of the 35
negative anchors, the form check of card §1c on both files, no signal, no "Let Bob draw", no
Genie, and the Cat four times in each file and never in a picture.

```python
assert same_picture(g[0], bare) and same_picture(g[1], surgery(0))
assert same_picture(g[2], surgery(1, half_on_upper=True)) and np.allclose(matrix_of(g[2]), zc)
assert same_picture(g[3], bare)
```
