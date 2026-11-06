# Chapter 14 — *Shor's Hidden Periods*: companion code

There are five challenge files, one for each dialogue challenge. Each file checks its claim
**twice**. The blackboard does it with NumPy `assert`s. The sketchbook compares PyZX diagrams
as tensors through `tools/dfp.py`'s `matrix_of` / `same_picture`, which force
`strategy="naive"`, and it also derives each equality with the rewrite rules of
`pyzx.simplify`: `full_reduce` on a copy with the scalar kept, or `spider_simp` (fusion) where
the chapter names fusion. Running a file is the book's analogue of type-checking.

```bash
# one challenge at a time
.venv/bin/python chapter14/Code/Ch14/QuarterClock.py
# the whole chapter (must end "5 files, 0 failed")
bash check-code.sh 14
```

Five things hold in every file.

- **Scalars are exact.** As in Chapters 9–13, the "up to a non-zero scalar" licence is never
  used. A red point is $\sqrt2\ket0$, a red $\pi$ point $\sqrt2\ket1$, a phase-free green
  point $\ket0+\ket1$; a bare controlled turn is $\mathrm{CP}(\alpha)/\sqrt2$; a Hadamard
  **edge** (the yellow box) is exactly $H$. `add_power(n)` multiplies by $\sqrt2^{\,n}$. The
  compensations, each commented where it is made, are:
  - `add_power(1)` for the bare controlled turn against $\mathrm{CP}(\alpha)$, and for the
    left side of the basis-point lemma;
  - `add_power(-2)` for the copy step of the lemma's ledger ($\tfrac12$);
  - `add_power(3)` for the $t=3$ transform's triangle against $F_8$, `add_power(1)` for the
    $t=2$ inverse against $F_4^{-1}$;
  - **none** for *The Quarter Clock* (three points $2^{3/2}$, three bare controlled turns
    $2^{-3/2}$), none for it with the wheel point green, and none for the dial pictures
    ($C_7$, $C_7^2$, $C_7^4$), which are red $\pi$ dots and crossings.
- **Top wire first.** The master's reading is $01$, which is not a palindrome: the file
  checks that the amplitude at `0b011` is $1$ and at `0b101` (the reading upside down, "a
  half") is $0$. The dial picture sends `0b0001` to `0b0111` ($1\mapsto7$), which catches a
  rotation the wrong way.
- **Generic inputs where the claim is general.** Kickback and the headline use a random
  complex, non-diagonal $3\times3$ unitary with a random complex eigenvector, so that a
  slip between $\varphi$ and $-\varphi$ shows; the controlled turn and the lemma run at
  $\alpha=3\pi/7$ as well as $\pi/2$; the order, the rounds, $\ket1$ as their mix, the
  reduction and the continued-fraction step are checked on several dials, not only on
  fifteen.
- **Non-Clifford turns.** The eighths ($\pi/4$, $\pi/8$) and $3\pi/7$ are not Clifford, so
  `full_reduce` does not empty a closed diagram. Where it does not reduce to a readable
  shape, both sides are reduced and compared as tensors.
- **Stated-and-cited bounds are documentation.** Cleve et al.'s $4/\pi^2$ and Shor's §5
  bounds are checked on samples and three small instances, with a comment saying so; no file
  claims to prove a running time or a success bound.

The files begin with a dial ticked by hand until it comes home, and a picture of that dial
made of flips and crossings. They then find the rounds that the ticking only turns, tie a
listener to one by a controlled turn so that the turn kicks back, and double the pace until
one wish reads the turn exactly (the fourteen-line Quarter Clock). Then they start from the
hand at one, which is every round at once, and end with Euclid's algorithm, which turns a
reading into a period and a period into factors, with no picture at all.


**The prop pictures.** The dialogue draws its devices seventeen times, with four macros from
`pdf/props/ch14.{tex,lua}`: `\tickdial` (the paper dial, its hand and the rounds' rings),
`\tickcloth` (the Tortoise's cloth), `\clockcase` (the Office's case) and `\tickstrips`
(Euclid's strips). `tools/props_ch14.py` transcribes the Lua's rules: the argument grammar,
the 41 error sites, the state models and the drawn geometry. It reads every constant from the
Lua and guards every rule with a literal snippet, so an edit to `ch14.lua` that it does not
follow fails the import. The cloth and the case come from the shared kits
(`props_cloth.py`, `props_office.py`). Each challenge file's `pictures()`, called at the end
of its `blackboard()`, parses its own section's calls out of `dialog.md` through
`tools/dfp_props.py`. It then checks:

- the layout, the captions (words only, none spoken, none forbidden), the block heights and
  the anchors (the paragraph before each block, and the one after it, in both files);
- that each picture's state is the one this file's mathematics computes for that moment;
- that what is drawn clears what it must, on the drawn extents with the strokes included.

These are structural checks of the pictures, not a second proof.

---

## I. `HandsRound.py` — *The Fifteen-Hour Dial* (the order; the tick box; the dial as a picture)

> "Four ticks. Seven thirteens are ninety-one; less six fifteens, one. Home." — Achilles

Achilles ticks the paper hand from one: seven, four, thirteen, one. He thinks a bigger dial
only takes more ticks; the Tortoise answers that the Office's dials may take nearly as many
ticks as they have hours, and nobody knows a quicker way.

The blackboard asserts the orbit `[1, 7, 4, 13]` and $r=4$, the orders $6$ (of $2$ mod $21$)
and $10$ (of $5$ mod $33$), and Result 14.1 for 25 random pairs $(N,a)$: the tick is a
permutation, $U_a$ is unitary, $U_a^r=I$ and $U_a^{r-1}\neq I$. A base sharing a factor with
$N$ is shown not to permute, so the hypothesis is needed. For odd primes up to $31$ some base
has order `N - 1`, the chapter's "nearly as large as $N$".

The sketchbook builds $C_7$ as the chapter draws it: the bottom wire climbs to the top over
the other three, then a red $\pi$ on every wire.

```python
assert np.allclose(M, P)                                  # x -> 15 - rot3(x), all sixteen
assert np.allclose(M @ ket(0, 0, 0, 1), ket(0, 1, 1, 1))  # 1 -> 7, top wire first
assert np.allclose(M @ ket(0, 0, 0, 0), ket(1, 1, 1, 1))  # 0 -> 15: not a residue
assert not same_picture(C7, wrong)                        # the other rotation: 1 -> 13
```

It equals $7x\bmod15$ only on the residues $1\le x<N$. The gate list of three SWAPs and four
NOTs is the same picture. $C_7^4=I$ is **derived**: `full_reduce` on four copies composed
leaves only the eight boundaries, each input wired straight across, with scalar $1$. $C_7$,
$C_7^2$ and $C_7^3$ are not the identity.

**The pictures (P1.1–P1.3).** Each is the dial with the hand at one, at seven (ticked once)
and at one again (ticked four times). The hand's tip points at its hour, counted clockwise
from the top mark: the first tick turns it $144^\circ$, not $216^\circ$. The hour sequence
is the file's `orbit(7, 15)`. The keyed word "top" stands just right of the top mark's outer
end, above the circle. Its measured box clears the sheet's edge, the circle, the marks and the
hand, and the letters of "hand" clear the hand.

## II. `KickedTurn.py` — *The Round Turned Whole* (eigenrounds; kickback; the controlled turn; the blur)

> "...Oh. The round hasn't gone anywhere. Every ring has moved, and the round lies where it lay, only turned, all of it at once, a quarter round." — Achilles

The Tortoise lays a round on the dial and Achilles ticks it: every ring moves, and every bead
stands a quarter on. On her cloth she builds a controlled turn between a wheel cord and a
listener and pushes the lit end through. Then the trap: a quarter and three quarters stand at
the same slant, even odds both.

The blackboard checks Result 14.3 on the dial (the four rounds are orthonormal, and $U_7$
turns $\ket{\psi_k}$ by $e^{2\pi ik/4}$, not $e^{-2\pi ik/4}$) and on three more dials. The
dialogue's beads are asserted hour by hour (`ticked[h] == 1j * bead[h]`). Kickback is checked
on a random complex non-diagonal $U$, with the conjugate phase as a negative control:

```python
assert np.allclose(L @ np.kron(chi, u), np.kron(Zph(theta) @ chi, u))
assert not np.allclose(L @ np.kron(chi, u), np.kron(Zph(-theta) @ chi, u))
```

Kitaev's Eq. (17) gives $P(0)=\tfrac12(1+\cos2\pi\varphi)$: $1$, $\tfrac12$, $0$, $\tfrac12$ for
the four quarter-turns. $\varphi$ and $1-\varphi$ give the same odds, and $iU$ turns the cosine
into minus the sine.

The sketchbook checks Result 14.5 (the bare gadget is $\mathrm{CP}(\alpha)/\sqrt2$, and
`same_picture` holds with its wires exchanged) and the hub's parity. The basis-point lemma
is **derived**: `full_reduce` on the left side removes the whole gadget and leaves exactly
the right side's shape.

```python
assert h.vertex_degree(pt) == 1 and h.phase(pt) == c   # the point passes straight through
assert h.phase(z) == alpha                             # lit: its turn on the other wire
assert turned == [] and h.connected(h.inputs()[0], o1) # dark: a bare wire
```

The ledger is checked step by step at $\alpha=\pi/2$ and $3\pi/7$: the copy is $\tfrac12$, the
closing is $\sqrt2\,e^{ic\alpha/2}$, the $\pi$-commutation is $e^{-i\alpha/2}$, and the product
is $2^{-1/2}$. The trap is drawn too: the listener behind a yellow box comes out at even odds
for $\pi/2$ and $3\pi/2$, with different amplitudes of the same size. The closed picture of a
green $\theta$ point, a yellow box and a dark point is the number $1+e^{i\theta}$. For a
quarter and three quarters, `full_reduce` empties it and leaves exactly that number.

**The pictures (P2.1–P2.4).** P2.1, the first round laid, *is* $\ket{\psi_1}$: each hour's
bead stands a quarter back for every step of the orbit, and it is not $\ket{\psi_3}$. P2.2, one
tick later, is $U_7\ket{\psi_1}=i\ket{\psi_1}$: every ring is carried on to the next hour,
and every bead is turned a quarter. P2.3's cloth is the controlled quarter: the halves at
$e^{i\pi/4}$ and the bob at $e^{-i\pi/4}$. P2.4 reads as $\mathrm{CP}(\pi/2)$ on
$\ket+\ket1$ for the lit wheel and the dark one. The keyed words "ring" and "bead" clear
everything drawn. On the cloth, ties clear the other rings, and hubs and bobs clear the cords.

## III. `QuarterClock.py` — *Twice the Pace* (the transform; phase estimation; the master)

> "The first plain, the second lit, and the wheel lit. No half, and an odd quarter: one quarter, no more." — Achilles

The Tortoise adds a listener ticked twice, which hears only whether the quarters are odd,
then corrects the first listener by a quarter back and crosses the cords: the half first,
then the quarter. The Genie grants one passage with the wheel knob lit. The windows show
dark, lit, lit, and the opened case holds the picture.

The blackboard checks the transform against Shor's (4.1), the product form (Cleve et al.
(4.2)) for $t=3$ on every basis state, and the network for $t=2,3,4$ (without the final
crossing it is not $F_q$). The inverse, written gate by gate with every turn negated, is
checked too. **Shor's own gate order**, built from his §4, Eq. (4.4), with bit $j$ on wire
$t-1-j$, gives $F_q$ with its bits reversed, as Shor says:

```python
assert np.allclose(M, reverse_wires(t) @ fourier(t))
assert not np.allclose(M, fourier(t))
```

The headline, Result 14.11, is exact for $t=1,2,3$ and every $j$, with a random complex $U$.
Both orders of the $\Lambda_m$ give the same result. Assigning the controlled powers
bottom-up, using $F_q$ for $F_q^{-1}$, or conjugating the phase each fails. G1 is asserted as
$\Lambda(U^2)=\Lambda(U)^2$. Cleve et al.'s $4/\pi^2$ bound is checked on random phases for
$t=2,\dots,6$, commented as documentation.

The sketchbook builds *The Quarter Clock* exactly as the card lists it, fused, with its two
collars as Hadamard edges and the crossing as the output order:

```python
assert np.allclose(vec, ket(0, 1, 1))                            # no scalar added
assert np.isclose(vec[0b011], 1) and np.isclose(vec[0b101], 0)   # a quarter, not a half
assert g.num_edges() == 14 == 2 + 1 + 2 + 6 + 3
```

All four secrets give $\ket{j_1j_2}\ket1$, and the dark wheel gives $\ket{000}$ (the Terms,
clause (iii)). Each value is **derived** by `full_reduce`: one point per output, with phase
$\pi$ on exactly the 1-bits. **Fusion derives the master**: the gate-by-gate circuit equals the
matrix network, and `spider_simp` on it gives exactly the master, with the same edges and the
same dots.

```python
assert simp.spider_simp(h)
assert h.num_edges() == 14 and same_picture(h, g) and phases(h) == phases(g)
```

The transform's triangles of §14.3 are checked as drawn: $F_8=2^{3/2}\times$ its picture and
$F_4^{-1}=\sqrt2\times$ its picture. Without the crossing it fails.

**The pictures (P3.1–P3.4).**
- P3.2's cloth is worked by the transcription's own rules for all four quarter-secrets. It
  reads $\ket{j_1j_2}$, the file's `phase_estimation` and `bits_of`, and not the reading
  reversed. The cords cross once and meet no other end-ring.
- The case after the passage (P3.3) shows `dll`, which is $\ket{011}$. With the knob down it
  shows $\ket{000}$.
- The inside view (P3.4) *is* the master at $\vartheta=\tfrac12$:
  - its fourteen lengths are the master's fourteen edges, by name;
  - its two collars sit on the two Hadamard edges;
  - its seven beads are the master's phases in twenty-fourths;
  - the cord going up crosses the one going down once, and is drawn over it;
  - no tie crosses another.
- The Genie's thirteen lines are frozen as a literal list. No picture sits between the lamp
  and `Granted.`, or at the Escher moment.

## IV. `EveryRound.py` — *Every Round at Once* (the hand at one; a random reading)

> "Two of the four readings give it, then. Half the time." — Achilles

The Tortoise lays all four rounds on the dial. At one the beads agree, and at every other
hour they pair off and cancel. She supposes a reading of a half; Achilles ticks from one to
seven to four, not home. A quarter or three quarters gives four ticks home, and nought says
nothing.

The blackboard checks Result 14.12, $\ket1=r^{-1/2}\sum_k\ket{\psi_k}$, on four dials. Cleve
et al.'s unnormalized (6.4) sums to $4\ket1$. The dialogue's pairing of the beads is asserted
at every hour. Phase estimation with $U_7$ on the four-wire register, started on $\ket1$, is
exactly $r^{-1/2}\sum_k\ket{kq/r}\ket{\psi_k}$. At $t=2$ the four readings come up a quarter
each; at $t=3$ only the even readings occur, a quarter each. The bookkeeping gives success
exactly $\tfrac12$:

```python
success = sum(p[c] for c in range(q) if pow(a, Fraction(c, q).denominator, N) == 1)
assert np.isclose(success, 0.5)
```

Shor's §5 bounds are checked, as documentation, for $(15,7,256)$, $(21,2,512)$ and
$(33,5,2048)$. Every good outcome has probability at least $1/(3r^2)$, and it lies within
$1/(2q)$ of some $d/r$. Two success figures are computed. Shor's count (the good outcomes
whose $d$ shares no factor with $r$) gives $0.500$, $0.228$, $0.290$. The full
continued-fraction rule gives $0.500$, $0.321$, $0.391$. Both are at least the number of $d<r$
coprime to $r$, divided by $3r$.

The sketchbook checks that a green point is the dark and the lit red points added and
divided by $\sqrt2$. The master with its wheel point green is exactly
$\tfrac1{\sqrt2}(\ket{000}+\ket{011})$ with no compensation. That is $\tfrac1{\sqrt2}$ times
(the master dark plus the master lit), and it holds for every secret. It is **derived** by
reducing it and a reference picture (a red point beside a two-legged green dot, times
$\tfrac12$) with `full_reduce` and comparing the two.

**The pictures (P4.1–P4.3).** On P4.1 the four rounds laid together (each with its
$\tfrac12$) add up to $2\ket1$: four at one, nothing elsewhere. P4.2 turns face down exactly
the pairs a half apart, whose phases cancel, and leaves the four rings at one face up. P4.3's
hand stands at $4=7^2\bmod15$. The square at each hour is ordered as Achilles sees it:
- the ticked round far left;
- the round at rest far right;
- the half's round near left;
- three quarters back near right.

Every bead shows. On the drawn extents, with the strokes read from the Lua, nothing at an hour
touches anything else:
- bead to bead: 0.015 cm;
- bead to another ring: 0.021 cm;
- ring to ring: 0.027 cm;
- ring to hand: 0.028 cm.

## V. `SharedFactor.py` — *Euclid's Strips* (continued fractions; the reduction)

> "Fifteen is three fives." — Achilles

Achilles lays strips: three measures fifteen five times, five measures it three times. On the
supposed twenty-one-hour dial, five parts in thirty-two give the counts six, two, two. Six
ticks bring the hand home; three ticks leave it at eight, and the strips give seven and
three.

The blackboard runs the strips as Euclid's algorithm: `strips(32, 5) == ([6, 2, 2], 1)`. It
checks $5/32=[0;6,2,2]$ (first convergent $1/6$), $27/32=[0;1,5,2,2]$ ($5/6$) and Shor's
example $85/512=[0;6,42,2]$, whose only convergent with denominator below $21$ is $1/6$, within
$1/(2q)$. It also checks why the dialogue's coarse reading needs a finer case: at $q=32<21^2$,
$2/13$ is also within $1/(2q)$ of $5/32$, and closer, so Shor's rule would return $13$, which
the check $2^{13}\not\equiv1$ rejects.

```python
assert len(near) == 2 and shor_rule(5, 32, 21) == Fraction(2, 13)
assert pow(2, 13, 21) != 1
```

Result 14.16 (i) is checked for every $r<N$ and $d<r$ at Shor's $q$ for three dials. Result
14.17 is checked for every base of ten odd $N$ that are not prime powers. The only failures are
$r$ odd and $a^{r/2}\equiv-1$. The success counts are $6/8$, $6/12$ and $10/20$ for
$15$, $21$ and $33$, each at least $1-1/2^{\ell-1}$. The failures mod $15$ are
`[1, N - 1]`.

The sketchbook takes the one step the pictures reach, the squaring. $C_7^2$ is **derived**
by `full_reduce` to the pure crossing $\mathrm{rot}_2$: only boundaries remain, each input two
rows round, with scalar $1$. $\mathrm{rot}_2$ multiplies by $4=7^2\bmod15$ on the residues and
sends `0b0001` to `0b0100`. The gcd and the continued fraction have no picture (Rosetta row
109). This is the translation exception recorded on the card, and the docstring quotes the
spec: *the diagram shows the architecture; the number theory stays on the blackboard*.

**The pictures (P5.1–P5.3).**
- The strips *are* Euclid's run on $(15,4)$: the counts $3,1,3$, ticks at $4,8,12$, and $3$
  left after the snip.
- One scale serves the whole evening: 0.15 cm a thumb, read from the Lua, so the
  fifteen-thumb strip spans 2.25 cm of the 2.5 cm frame.
- The runs read out but not pictured are checked as follows:
  - fifteen and three, and fifteen and five, run through the macro's strips and are checked
    against `strips` and `math.gcd`;
  - thirty-two and five, twenty-one and seven, and twenty-one and nine are longer than
    fifteen thumbs. The macro refuses them, and the file's `strips` checks them instead.
- The two strips never touch, and the short one stays put.
- The keyed words' measured boxes sit inside the 0.78 cm frame. They touch neither each other
  nor the strips:
  - "long strip" stands below the strip's right end, beside "tick";
  - the other placement rule, where "tick" steps to the left of its tick, is checked on
    fifteen and nine.
- P5.3's hand is home after six ticks, `dial_orbit(21) == [1, 2, 4, 8, 16, 11]`, the order of
  $2\bmod21$.
- This file also checks the whole dialogue:
  - 17 pictures, none after P5.3, each in its place;
  - the keyed form exactly on P1.1, P2.1, P2.3, P3.1, P3.3, P3.4 and P5.1;
  - the aria-and-Genie one-liner of the card's §1c on both files;
  - no "Let Bob draw";
  - three "Referred." lines, all the Genie's.
