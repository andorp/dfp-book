# Chapter 19 — *Logic of Pictures*: companion code

There are five challenge files, one for each dialogue challenge. Each file checks its claim
**twice**. The blackboard does it with NumPy `assert`s. The sketchbook builds the pictures in
PyZX and evaluates them only through `tools/dfp.py`'s `matrix_of` / `same_picture`, which force
`strategy="naive"`. Every sketchbook also carries at least one derivation by PyZX's own rewrites
(`full_reduce`, or the named rules `spider_simp` and `id_simp`), so no file is tensor-only.
Three blocks are marked as **translation checks** (evaluation, not rewriting), each beside its
file's own derivation. Running a file is the book's analogue of type-checking. There is no
proof-assistant tier.

```bash
# one challenge at a time
.venv/bin/python chapter19/Code/Ch19/EveryCardBalances.py
# the whole chapter (must end "5 files, 0 failed")
bash check-code.sh 19
```

The files follow the evening's argument. First they check the deck card by card. Then they add
the one card the book was missing and use it to close the gap Chapter 17 left. Next come three
*instances* of what a complete deck promises. **No file checks completeness itself.** The
theorems of Result 19.8 (Backens; Jeandel, Perdrix and Vilmart; Ng and Wang; Carette, Jeandel,
Perdrix and Vilmart) are stated in the chapter and cited to their papers, and none of them is
proved here. After that the code builds the closed box, which answers one question whoever
watched, by a look that copies the Cat's asleep-or-awake or watches only the watchers. It ends
with the Anteater's sealed rooms, whose numbers the code confirms but whose paradox it cannot
settle.

Each file also runs `pictures()`, at the end of its blackboard. The dialogue's prop pictures
(`\rulecard`, `\hingecloth`, `\darecloth`, `\boxcloth`, `\crablute`: fifteen calls, three, four,
four and four by challenge, none in the Fifth) are read out of `dialog.md` and `dialog.raw.md`
through `tools/dfp_props.py`. Their state is computed by `tools/props_ch19.py`, a transcription of
`pdf/props/ch19.lua` on the shared cloth kit. The transcription reads the Lua's places and tables
from the Lua itself; its drift guard holds every top-level statement of the Lua against a pinned
fingerprint and counts the Lua's `tex.error` call sites (6 sites, 12 macro–site pairs: the shared
step list's three through each cloth macro, `\rulecard` 2, `\crablute` 1). It was diffed against
the Lua under texlua: 272 reference lines (the fifteen calls, keyed and unkeyed, every state of
every trace, every card and view, and the parser's spellings), 30 error calls and 4000 fuzzed
calls give 0 differences and no raw Lua error, and all twelve pairs are reached. `pictures()`
ties each picture to the blackboard: a card side is built as a chain on one wire from its dump,
and a cloth dump is read as a PyZX graph (`PC.graph_of`: a ring a spider of its colour turned as
its bead, a ring lying on a cord spliced into it, a cord, tie or pendant with an odd number of
tubes a Hadamard edge, a second cord between the same two rings kept apart by a ring at rest, a
loose end a boundary; the corner's slips are the picture's number). This is a structural check
of the pictures, not a third proof. It also checks the dialogue's rules: blocks and sizes,
captions in words only (no echo, no count, true of the dump), every picture in its place, the
anchor phrases and D-item phrases, the starred pictures, and the keyed words at their compiled
boxes and clear of every part drawn (letters at least $0.02$ cm from every stroke, shadow and
other word, node boxes at least $0.025$ cm inside the frame).

Six conventions hold across the files.

- **Scalars are exact.** Spiders are bare: a green point at rest is $\ket0+\ket1$, a red point
  at rest $\sqrt2\ket0$, a red point at a half $\sqrt2\ket1$; the bare look (a green spider tied
  to a red one) is $\mathrm{CNOT}/\sqrt2$. Every number a picture carries is written on its
  scalar: `add_phase` for a turn of the whole (in units of $\pi$), `add_power` for powers of
  $\sqrt2$, `add_float` for a general number such as $(1-e^{2i\alpha})/2$. No file uses
  `same_picture_up_to_scalar`.
- **Top wire first.** Operators multiply right to left; pictures read left to right.
- **Generic inputs.** Turns are `Fraction`s of $\pi$ with the non-dyadic denominator 97 (or
  seeded random floats on the blackboard); states are `random_state` (non-real); the boxes around
  a card are `random_matrix` (non-real, non-symmetric, non-unitary); unitaries come from QR of a
  seeded complex Gaussian matrix. Every generator is seeded.
- **One strand, one edge.** PyZX's simple graphs drop a second edge between the same two vertices.
  So wherever strands could double up, a green ring at rest (bare cord) sits on each strand, and
  every hand-built picture asserts `num_edges()`.
- **No ground vertex.** The doubled pictures (the causality instance and *The Closed Box*) are
  built by hand as a wire and its mirror copy, with each discard drawn as a cap: a two-legged
  green spider at rest joining the two. pyzx 0.10.6 refuses grounded graphs.
- **No rewrite proves a bound.** The majority-of-three numbers are the blackboard's; MIP\* = RE
  has no file; the Frauchiger–Renner file checks numbers, not the theorem.

## Every Card Balances — `EveryCardBalances.py`

The Tortoise deals her deck and Achilles ticks every card on his pad. He hunts for the wrong
card and finds only a number he had missed ("Halved." ...Oh.). He puts one card in company.
She holds three cards to the lamp, and he hands back the squared deck with the countersubject:
"Whatever the pictures say, the numbers say." Her line on the see-through is the file's second
quotation: "Hold any of them to the lamp, and what you see from behind is a card of the deck."

The deck is written once, as data. Each card is a list of layers of pieces (spiders with any
number of legs and a turn, yellow boxes, crossings, cups, caps, generic boxes), with its number.
The same description is evaluated as a matrix on the blackboard and built as a PyZX graph in the
sketchbook. The red spider is built from $\ket\pm$ on its own, so that colour change is a real
check and not a definition. For three draws of random turns and boxes, every card must balance,
number included. Where the number is not 1, the card must fail without it. Every card must also
balance in company, with random non-real matrices before it, alongside it and after it (Result
19.2). Each side's see-through must be its adjoint, so each card seen through is a true card with
its number conjugated (Result 19.1):

```python
assert np.allclose(L, n * R), name
assert np.allclose(in_company(Mc, L, Lc, Kc), n * in_company(Mc, R, Lc, Kc)), name
assert np.allclose(mL, L.conj().T) and np.allclose(mR, R.conj().T), name
assert np.allclose(mL, np.conj(n) * mR), name
```

The three cards held to the lamp are checked as the dialogue has them. The sliding card seen
through is the sliding card with the turns $(-b,-a)$. The half-passing card seen through is "the
same card read in the other order", at turn $-a$, with its dial turned back. The mirror card's
two sides are exchanged. The negatives cover the plausible mix-ups. A see-through that forgets to
reverse the turns gives the transpose, not the adjoint. On a general box the conjugate is not the
adjoint. Dealt without its $\tfrac12$, the Hopf card sends a green point to $2\sqrt2\ket0$
instead of $\sqrt2\ket0$, and it stays wrong in company. On the mirror copy, $U$ is not swallowed
by the cap; $\bar U$ is.

The sketchbook's **(T1)** block is the translation check: each card's two pictures are built and
compared, and PyZX's `adjoint()` of each side is compared with the side rebuilt from the deck's
own pieces, reflected with every turn reversed. The context test is repeated in pictures, with a
generic box on every wire and a look tying the card's last wire to the wire alongside, so the
surrounding is not a product. Result 19.1 is then checked on a larger picture that contains
every kind of piece, and on a random circuit-shaped picture with non-Clifford turns on every
spider. The derivations: the named rules `spider_simp` and `id_simp` rewrite the sliding cards'
left sides to one green spider turned $a+b$; a random Clifford circuit composed with its mirror
reduces by `full_reduce` to bare wires, number 1; the bare look composed with its mirror reduces
to bare wires, number $\tfrac12$.

`pictures()` reads the First Challenge's three cards: the half passing card face up (keyed:
`is`, `dial`), the sliding card held to the lamp, and the half passing card read from behind. For
every card of the deck, face and through, it checks the see-through's rule: the through sides are
the adjoints of the face's sides, crossed over (in matrices and as PyZX `adjoint()`), and not
their transposes; the dial multiplies the picture it stands under, crosses with its picture and
reverses (D2), and the face's dial left unreversed fails; and the card held to the lamp is the
mirror of the face, every drawn part (rings, beads, tubes, cords, dial, needle, cross, the
written "is") reflected left to right, while the face reflected with its beads kept is not the
through view. P1.1's sides are the file's deck card "a half passes a red turn" at $a=\tfrac12$,
number $e^{i\pi/2}$; P1.2 is again a sliding card, three eighths back; P1.3 is D2 run,
$e^{-i\pi/2}(\text{face-right})^\dagger=(\text{face-left})^\dagger$. Unpictured: the new card is
$e^{i\pi/4}H$ (the file's Euler card), three quarters back is $e^{-i\pi/4}H$, and the green point
a quarter round is $e^{i\pi/4}$ times the red point a quarter back.

```python
assert np.allclose(Tl, Fr.conj().T) and np.allclose(Tr, Fl.conj().T), name
assert np.allclose(dial_number(d[1]) * under, other), name     # the dial multiplies its picture
assert fd[0] == "right" and td == ("left", (24 - fd[1]) % 24), name     # crossed, reversed
```

## The Hinge — `TheHinge.py`

The Tortoise recalls the gap from the day before: "A hub with a quarter hanging from it, standing
on a quarter back on each line, is a tie in a sleeve. Yesterday I hadn't a move for it." She draws
the Euler card, which Achilles knows from the three quarter clicks, and three crossed cards from
it. Then she runs the hub on the cloth to the sleeved tie, laying two slips that cancel.

The blackboard checks the Euler card $Z(\tfrac\pi2)X(\tfrac\pi2)Z(\tfrac\pi2)=e^{i\pi/4}H$, its
mirror with $e^{-i\pi/4}$, and its colour-swapped form. It checks the mirror Euler's own
derivation (a pair of yellow boxes added, one read backwards, the opposite turns fused away from
the inside out), Lemma P for random $\alpha$ and $c=0,1$ along the sketchbook's route (slide
apart, copy with $\tfrac1{\sqrt2}$, close against the green point), and Lemma A with its five
moves. The hinge is checked in two ways. The first is the bit identity
$x_1+x_2-(x_1\oplus x_2)=2x_1x_2$ on all four words. The second contracts $\mathrm{Hub}$ and
$\mathrm{Tie}$ leg by leg, as drawn, and finds both equal to $\mathrm{CZ}/\sqrt2$:

```python
Hub, Tie = hub_matrix(), tie_matrix()
assert np.allclose(Hub, CZ / np.sqrt(2)) and np.allclose(Tie, CZ / np.sqrt(2))
assert not np.allclose(hub_matrix(foot=+q), Tie)       # feet a quarter ROUND
assert not np.allclose(hub_matrix(point=0), Tie)       # pendant at rest
assert not np.allclose(tie_matrix(I2), Tie)            # a tie with no sleeve
```

The sketchbook builds $\mathrm{Hub}$ (seven edges: feet at $-\tfrac12$, a red hub at rest, a
green $\tfrac12$ point) and $\mathrm{Tie}$ (a Hadamard edge between two green spiders at rest).
It builds every picture of the chain of §19.7 and checks each against the one before, with its
slip: (i) Lemma A with an eighth round, (ii) fusion, (iii) fusion backwards, (iv) the mirror
Euler with an eighth back, and (v) the slips cancel, so `same_picture(hub, tie)` holds exactly.
Without their slips, the middle pictures are an eighth off. The derivations: `full_reduce` takes
each Euler chain to a lone yellow box with its number and Lemma A's red point to the green one;
it takes $\mathrm{Hub}$ to a picture equal to $\mathrm{Tie}$ (PyZX's simplifier is not the
book's deck, as the chapter says); and $\mathrm{Hub}$ composed with the mirror of $\mathrm{Tie}$
reduces to two bare wires with the number $\tfrac12$.


`pictures()` reads the new card (P2.1: green, red, green a quarter round each is $e^{i\pi/4}$
times the tube) and the hinge cloth three times. `graph_of` of the hub as laid (keyed: the kit's
`hub`, `point`, `tie`, `line`, and `corner`, set by the Lua) is `hinge_picture("hub")` and
`hub_matrix()` exactly; every state of the trace, with its corner's slips, is the hub as laid,
and the three middle states are an eighth off without them. The states are the file's own chain:
(i), (ii) and (iii) read ring for ring. P2.4, the corner empty, is `hinge_picture("tie")`,
$\mathrm{CZ}/\sqrt2$, and without its tube it is not. The negatives: the point a quarter back, or
the feet a quarter round, is not the hub. Fixed parts: the lines' loose ends and the feet's
column in every picture.

## Same Number, Same Picture — `SameNumberSamePicture.py`

The Crab's dare, and Achilles sets it: "Here's one. A tube, then a quarter round, three times
over, on one cord; and a bare cord with the whole turned an eighth round. My pad says they're the
same. Get from one to the other with your cards, and don't look at my numbers." The Tortoise wins
it with the deck alone, and the Crab says the subject.

The file checks three **instances**; it checks no theorem. (1) The dare:
$(SH)^3=e^{i\pi/4}I$, with the matrix of every picture of the five moves of Result 19.7 checked
against the first, and the card's hand chain. (2) The eighth-turn instance of Result 19.11, van
de Wetering's eq. (173) with its number worked out: at $\alpha=\tfrac\pi4$, where the number is
$\tfrac{1-i}2$, and at random $\alpha$, since the chapter claims it for every $\alpha$. (3) Letting
go after any turn (Result 19.9), for random turns: the copy's reversed turns make the conjugate,
and the cap swallows it.

```python
assert np.allclose(dare, E8 * I2)
assert np.allclose(eighth_turn_lhs(alpha), number * red_pt)
assert np.allclose(CAP @ np.kron(U, Ubar), CAP)
assert not np.allclose(CAP @ np.kron(U, U.conj().T), CAP)   # reversed AND reordered: no
```

In the sketchbook, `full_reduce` takes the dare's picture to 2 boundaries, 1 plain edge and the
scalar $e^{i\pi/4}$, and each of the Tortoise's five pictures, with its slips, equals the dare.
The **(T2)** block is a translation check. Each side of the eighth-turn instance is reduced by
`full_reduce`, and the reductions are compared through `matrix_of` with the number
$\tfrac{1-i}2$ (and at random non-dyadic $\alpha$ with its own number). The comment says plainly
that PyZX's simplifier is not the book's deck: the book has derived no such chain. The doubled
causality picture is built by hand as a state (its open ends are outputs), with the cap as a
two-legged green spider at rest. `full_reduce` leaves the bare cap, 2 vertices and 1 edge with
number 1, for multiples of $\tfrac\pi4$ and for generic turns. Two negatives: a copy with the
same turns, and a copy with reversed turns in reversed order (the dagger, not the conjugate), are
not let go.


`pictures()` reads Achilles's dare on the cloth (keyed: the kit's `tube`) and three of its
steps. P3.1 is the file's dare picture, exactly $e^{i\pi/4}I$. Every state of the trace, read
ring for ring, is the picture of the Tortoise's moves (laid, then (i) to (v)), and with its
corner's slips it equals P3.1; without them every state after the first two is off. The two slips
of P3.4's step before ($21+6$ twenty-fourths) and P3.4's one ($3$) are the same eighth round.
Fixed parts: the cord's ends at 0 and 10 in every picture, and the red ring at 2 from P3.2 to
P3.3.

## One Answer Out — `OneAnswerOut.py`

The Tortoise lays *The Closed Box* twice on the cloth, as it is and seen through, and slides it
together: "There's the Cat's own point, on the answer's cord, just as it lay before anybody
looked." Achilles over-reaches, and she scopes him: "Not any look. Yours copied asleep-or-awake,
and mine only watched you. A look that asked the Cat some other question could change them."
Then Achilles counts two times in three, and twenty times in twenty-seven.

The blackboard first shows that the bare look, built from its two spiders, is $\mathrm{CNOT}/\sqrt2$,
a look of the first kind. The three points and two looks make $\ket{000}+\ket{111}$. Doubled,
with both pages capped and the Cat read, the value is $(1,1)$. Closing the answer wire would give
2, not 1. The normalized odds are the Cat's own $\tfrac12,\tfrac12$. For Result 19.13 the file
takes a random entangled non-real state of all three and a random mixed one, and runs four rounds
of looks: one copying asleep-or-awake with random unitaries, the Tortoise's look at Achilles, and
a random unitary on the pages alone. The Cat's odds never move, though the rest of the Cat's
state does. The negatives are the Tortoise's "other question". In the chapter's own case, a
surely-asleep Cat and a look that copies $\ket\pm$ make the odds even. In the generic case, a
$\ket\pm$-controlled look moves random odds. The file also checks a run of §19.4 (the value
with $\tfrac14$ is $(p_0,p_1)$) and Result 19.10 with `Fraction`s: $f$ counted over the eight
outcomes, $f(\tfrac23)=\tfrac{20}{27}$, $f\ge\tfrac{20}{27}$ on a grid from $\tfrac23$ to 1 with
$f'\ge0$, and the sidebar's identities ($f(p)=1-F(1-p)$, $f(p)-p=p(1-p)(2p-1)$,
$f(\tfrac{20}{27})=\tfrac{16400}{19683}$, resting points $0,\tfrac12,1$).

The master is built by hand, as the chapter draws it:

```python
box = closed_box()
assert box.num_edges() == 19          # the master's lines, doubled and classical (Section 19.8)
assert np.allclose(matrix_of(box).ravel(), [1, 1])
simp.full_reduce(h)                   # one green spider at rest, one edge, number 1
```

The builder's docstring names the nine undoubled edges, their mirror copies and the classical
wire. `full_reduce` leaves one green spider at rest on the answer wire with the scalar 1, and
`spider_simp` / `id_simp` alone (fusion, the identity card, the loops) reach the same point. In
pictures, a surely-asleep Cat (a red point in its place) still reads $(1,0)$ after both looks.
Achilles's look turned into one that asks the other question (a red spider on the Cat's wire,
tied by a yellow box) makes the odds even.


`pictures()` reads the box cloth three times (keyed: `read ring`) and the Crab's lute. `graph_of`
of every box state (laid, red points slid in, slid together, loops off) is `closed_box()` and
`green_point()`, value $(1,1)$, exactly. The cloth's graph has 17 edges against the builder's 19,
because each let-go cord is one edge, knot and all, where `closed_box()` draws a cap with two
legs. D16 is asserted on the dump: the copy is the upper layer with its rows in the other order,
left and right kept, every bead turned the other way round (the conjugate), and the lamp's mirror
(left and right exchanged) does not reproduce it. The turns are load-bearing: a page's ring turned
a generic step on the picture and back on its copy leaves the Cat's point, while the same turn on
both does not. The knots sit on the cords' rounded turns, her page's inside Achilles's. The lute
(R7-Ch19): its strings, the top string alone and nine pairs, are $1+2\times9=19$, equal to
`closed_box().num_edges()`, the master's lines. A sounding string's swing and the motion arcs are
marks of the one string, not wires, so they are not counted: each swing edge runs from its own
string's nut end to its bridge end, and no arc reaches the nut or the bridge. The count is
asserted in code and never said.

```python
assert len(strings) == 1 + 2 * 9 == box.num_edges() == 19
assert same_picture(PC.graph_of(turned(t, 24 - t)), point), (up, t)
assert not same_picture(PC.graph_of(turned(t, t)), point), (up, t)
```

## Watchers Watched — `WatchersWatched.py`

The Anteater tells the story of Aunt Hillary's sealed chambers: "Aunt Hillary has played it
round after round. One round in twelve, both hear "ok"." Achilles rubs the lamp, the Genie's
answer comes back at once, and the Cat has the last word.

The blackboard builds Frauchiger and Renner's state from the protocol: a quantum coin with heads
$\tfrac13$, $\ket\downarrow$ on heads and $\ket\to$ on tails, giving
$\psi=\tfrac1{\sqrt3}(\ket{00}+\ket{10}+\ket{11})$. It checks the three certainties as exact zeros
and the "ok, ok" amplitude $\tfrac1{2\sqrt3}$, probability $\tfrac1{12}$. The negatives: $\ket{\mathrm{ok}}$
with a plus sign breaks the first zero, and an even coin changes the one in twelve.

```python
assert np.isclose(amp(ket(1), OK, psi), 0)        # F-bar saw tails: W will say "fail"
assert np.isclose(amp(ket(0), ket(1), psi), 0)    # F saw z = +1/2: r was not heads
assert np.isclose(amp(OK, ket(0), psi), 0)        # W-bar said "ok": F saw z = +1/2
assert np.isclose(abs(a) ** 2, 1 / 12)
```

The sketchbook writes $\sqrt3\,\psi$ as two green points at rest minus $\tfrac12$ times a red
point at rest beside a red point at a half, and closes each term with the effects ($\bra{\mathrm{ok}}$
a green effect at a half, $\bra0,\bra1$ red effects, each with $\tfrac1{\sqrt2}$). The
derivation: `full_reduce` removes each closed term entirely (`num_vertices() == 0`), leaving its
number. The **(T3)** block is the translation check: the two terms are added with their weights
on the blackboard, because a sum of pictures is not one picture, and the four values come out as
the blackboard's. The theorem itself (Result 19.15) is quoted in the chapter, not checked.
MIP\* = RE (Result 19.16) is a statement about all games at once, and has no code.

The Fifth Challenge draws no picture. `pictures()` here runs the whole-dialogue checks: the five
sections in order with all fifteen pictures inside them, the starred form on the first picture of
each macro, none of the forty negative anchors in either file, the Cat and the Genie never named
by a caption, keyed word or dump, no count in a caption or keyed word, and the counting scan's
whole-file hits exactly the approved ones (32 in `dialog.md`, 29 in `dialog.raw.md`). On both
files it runs the card's form check (the subject and countersubject by speaker and section, the
cast, the Cat's last word), the forbidden-word grep (no line), no "Let Bob draw", six Genie lines,
two Cat lines and the name Cat 25 times. Four broken samples show that the form check bites.
