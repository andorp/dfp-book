# Chapter 13 — *Asking the Oracle*: companion code

There are four challenge files, one for each dialogue challenge. Each file checks its claim
**twice**. The blackboard does it with NumPy `assert`s. The sketchbook compares PyZX diagrams
as tensors through `tools/dfp.py`'s `matrix_of` / `same_picture`, which force
`strategy="naive"`, and it also derives each equality with the rewrite rules of
`pyzx.simplify`: `full_reduce` on a copy with the scalar kept, and, where the chapter names
the move, the named rule itself (`copy`, `spider_simp`, `id_simp`, `color_change`). Running a
file is the book's analogue of type-checking.

```bash
# one challenge at a time
.venv/bin/python chapter13/Code/Ch13/TurnedWeb.py
# the whole chapter (must end "4 files, 0 failed")
bash check-code.sh 13
```

Five things hold in every file.

- **Scalars are exact.** As in Chapters 9–12, the "up to a non-zero scalar" licence is never
  used. A red point is $\sqrt2\ket0$, a red $\pi$ point $\sqrt2\ket1$, a green point
  $\sqrt2\ket+$, a green $\pi$ point $\sqrt2\ket-$; a bare green–red tie is
  $\mathrm{CNOT}/\sqrt2$; a Hadamard **edge** is exactly $H$. The compensations, each
  commented where it is made, are:
  - `add_power(w)` for the web against the matrix $U_f$ ($w$ ties), and `add_power(2w)` for
    two webs in a row against bare wires;
  - `add_power(-1)` for the right-hand side of kickback-as-a-copy, and for the trap;
  - `add_power(-5)` for the master, *The Turned Web* (seven points $2^{7/2}$, two ties
    $2^{-1}$), and `add_power(w-n-1)` for Cleve's form;
  - **none** for the turned web against the fan-out (both sides carry the same $2^{-w/2}$);
  - `add_power(-4)` for the doubled Simon round (both instances), and `add_power(|s|-2)` for
    the parity gadget against $P_s$ — none only when $|s|=2$.
- **Top wire first.** Inputs and outputs are set explicitly, rods from the top and the flag
  last. The master's secret $010100$ is not a palindrome, so the file checks the output index
  `0b0101001` and that its reversal `0b0010101` is zero.
- **The web is hand-built with its targets unfused**, one red dot per tie, as the gate-by-gate
  translation gives. PyZX keeps one edge between two dots, so two webs in a row are built as
  one graph with four separate targets, never by adding a tie twice.
- **Generic inputs where the claim is general.** Result 13.1 is checked for every $f$ on up
  to three rods and for random non-linear $f$; kickback and the trap on random non-linear $f$
  and random complex rod states; Chapter 5's test on a random balanced $f$ whose tally is
  checked first; Simon's round on random non-linear pairs boxes (redrawn until not affine).
  This chapter has no transpose trick and no mirror at the table, so its hazards are reversal,
  a missing scalar, a missing sign, and a box that passes only because it is a web.
- **The dialogue's props are checked.** The disc, the slips, the cloth web with its end-rings
  and tubes, the casket's windows, the fold of the cloth and the supposed clues are each an
  assert.

The files begin with a box asked plainly: one wish, one bit, and a box of marks that turns
out to be a web of ties, needing a wish per knob. They then set the flag crosswise, so that
each mark is copied back along its tie; turn the whole web with Hadamards, so that one wish
reads the secret straight off the wiring (the thirteen-line Turned Web); and end at a pairs
box, where one wish returns only a line through the secret, and no picture says how many
wishes a wisher must spend.

**The prop pictures.** The dialogue draws its props with three macros from
`pdf/props/ch13.tex`: the Office's casket, from the knob face, the window face or inside
with the lid lifted (`\wishcasket`, drawn with the shared Office parts of
`pdf/props/office.lua`); the wisher's slips of seven squares, with the disc (`\wishslip`);
and the Tortoise's web, end-rings, tubes and pairs box on her drawing-cloth (`\wishcloth`,
drawn on the shared cloth of `pdf/props/cloth.lua`). Their Lua rules are transcribed once in
`tools/props_ch13.py` (no matrices there), on top of `tools/props_office.py`, which holds
the Office parts' sizes and drawn outlines for this chapter and the next two, and
`tools/props_cloth.py`, the shared cloth (where an open cord of length zero draws nothing,
as in `cloth.lua`). Every constant and every inline place, scale and frame is read from the
Lua itself, and every rule the Python mirrors is guarded by a literal snippet of it (a drift
guard). The transcription holds the three state models, among them the casket's passage
simulated on its 128 entries as the Lua does it, and also the dumps, frames, keyed words
and where they go, the drawn outlines, and the Lua's forty-six fault sites, each keyed and
each message byte for byte. Each was compared with `texlua` on the fifteen dialogue calls,
on fifty fault calls (all 46 sites hit), and on four thousand random calls. There was no
difference. Each file reads the pictures of its own section through `tools/dfp_props.py` and
checks:

- the layout (4 pictures in each section, one per block, the same in `dialog.raw.md`), with
  one frame and one macro per block;
- the captions (words only, none on the card's forbidden list, never a spoken sentence,
  never the first sentence after the block);
- the places each prop may appear, and the paragraph each block follows and the one after
  it, with the card's direction phrases;
- every picture's state.

Each file also **ties the transcription to its own mathematics**, NumPy against its
blackboard. This is a structural check of the pictures, not a second proof. Fifteen
mutations of the dialogue (a caption, a setting, a mark, a keyed star, a phrase, a Genie
line, an extra picture) were each caught by the file that owns the section.

---

## I. `PlainWishes.py` — *One Wish, One Answer* (the query model; the web; the baseline)

> "Whatever I light, all it can ever say is whether the count came out odd or even." — Achilles

Achilles inks the first and fifth squares of his slip. The disc comes back lit for the first
and third; black for every place lit ("As if I'd made no marks at all"); turned the other way
when handed gilt; back as given when the same wish is asked twice. A wish per knob reads the
slip. Then the Tortoise's slip (second, sixth) and her open third strip (first, third,
fourth, sixth) answer all four of Achilles's fewer wishes alike, odd, even, odd, even.

The blackboard checks Result 13.1 for **every** $f$ on one to three rods and for random
non-linear $f$ on four and six: $U_f$ is a permutation, unitary, and its own inverse. The web
lemma is checked for every $(a,b)$ at $n=4$ and for the casket's, the slip's and random
secrets at $n=6$. The slip game is asserted move by move, and the "second slip" is found by
the kernel argument for 200 random sets of up to five wishes. Result 13.3 is exhaustive at
$n=4$, adaptive methods included — the fewest queries of any decision tree are computed by
recursion over every tree:

```python
assert not any(separates(qs, unknown, secret)
               for qs in combinations_with_replacement(strings(n), n))
assert separates(unit + [(0,) * n], unknown, secret)
assert min_queries(known, secret) == n
assert min_queries(unknown, secret) == n + 1
```

The sketchbook builds the web and checks $U_f=2^{w/2}\times$web, with the uncompensated web,
the wrong $b$ and the upside-down secret as negative controls. Asking twice is **derived**:
`full_reduce` leaves only boundaries, each input wired straight to its output, and the Hopf
price $2^{-w}$:

```python
assert h.num_vertices() == 2 * (n + 1)
assert np.isclose(h.scalar.to_number(), 2.0 ** (-w))
assert same_picture(scaled(h, 2 * w), bare_wires(n + 1))
```

The web composed with its PyZX mirror (`adjoint`) reduces to bare wires as well, the picture
form of unitarity.

**Pictures (P1.1–P1.4).** The casket from the knob face (P1.1) has its nibs at the top, its
rows on the Lua's pitch and the flag set apart. Every nib is clear of its own ball and
pointer and of the neighbouring collars, with either stop, and the seal is clear of the
plate. The checks are on drawn outlines, not body widths, and every gap is positive. The
unit wishes give the slip's inked squares (P1.2). The two slips of P1.3 answer Achilles's
four wishes alike (parities 1, 0, 1, 0). The disc's rule agrees with the parity for all 63
nonzero strings, both slips and both faces; the string of zeros is not expressible in the
disc's words. The slips are more than 0.04 apart, and both slips and their shadows lie inside the frame
the Lua draws, both read from `ch13.lua`. On P1.4 each tie's row is an inked square, and an X on the flag's wire in the PyZX
web of the slip. Each tie crosses every cord between its ends and clears every other ring.

## II. `KickedMark.py` — *The Mark on the Cords* (kickback as a copy; Deutsch–Jozsa read off the web)

> "...Oh. It's your token, wedged up on its edge on the porch in the spring. The turn had nothing on the flag to spend itself on, so it jumped back up the tie onto the cord that asked." — Achilles

The Tortoise sets every cord crosswise and dark, flag included, and pushes: every cord ends
as it began, and Achilles concludes there are no marks. With the flag's bead halfway, the
half-turn climbs exactly the first and fifth ties. A web with no ties returns nothing, and
the flag window, crosswise and lit, gleams every time.

The blackboard checks kickback on a random non-linear $f$ for every basis string and on a
random complex rod state, and Result 13.4 on a random complex $\ket\psi$:

```python
assert np.allclose(CNOT @ np.kron(psi, MINUS), np.kron(Z @ psi, MINUS))
assert not np.allclose(CNOT @ np.kron(psi, MINUS), np.kron(psi, MINUS))    # not nothing
assert np.allclose(CNOT @ np.kron(psi, PLUS), np.kron(psi, PLUS))          # the trap
```

The trap network returns $\ket{0^n}\ket+$ for **every** $f$ on up to three rods, every linear
$f$ and a random non-linear $f$ on four. The cloth of II is replayed on the slip $100010$.
The sign sum (Result 13.5) is checked for every $c$ up to $n=6$; Result 13.6 for every
$(a,b)$ at $n=4$; Chapter 5's test on constant and random balanced $f$ at $n=4,5,6$; and the
BPP remark exactly, $2^{1-k}$ over every $k$-tuple of strings.

The sketchbook **derives** kickback with the chapter's two named rules, in its order:

```python
assert simp.check_copy(h, p) and simp.copy(h, p)   # the red dot copies the green pi
assert simp.spider_simp(h)                          # the copy fuses into the control
assert same_picture(h, rod_and_point(1, 1))         # Z on the rod, the flag kept, 1/sqrt2
```

The trap's phase-free point copies as nothing (`add_power(-1)`, determined here). The cloth
is built with green end-rings; `full_reduce` leaves one point per cord, $\pi$ exactly on the
first, fifth and flag cords, and read crosswise the windows are $1000101$.

**Pictures (P2.1–P2.4).** The cloth's end-rings read crosswise give $H^{\otimes7}U_fH^{\otimes7}$
on the blackboard: $\ket{0}\mapsto\ket{0}$ (the trap), $\ket{0000001}\mapsto\ket{1000101}$,
and with no marks the flag alone. On P2.4 the flag cord has length zero and draws nothing
(`props_cloth.drawn`: six paths, six tips, six lashings; P2.2 has seven of each). The
end-rings' fixed parts are unmoved by the push.

## III. `TurnedWeb.py` — *The Turned Web* (Bernstein–Vazirani; *The Theorem, Twice*; the master)

> "The second and the fourth, and the flag. Not my marks at all. The casket's own." — Achilles

The Genie states G1 and G2 and refers every meta-wish. The Tortoise's tubes turn the slip's
web: every ring changes colour, and the ties run from the flag. Achilles sets every collar
crosswise, the wish-knobs dark and the flag lit; `Granted.`; the second and fourth windows
and the flag window gleam. Inside the casket the web is unturned: the turning was all in the
collars.

The blackboard checks the headline exactly, the sign $(-1)^b$ included, for every $a$ and
both $b$ at $n=4$ and for the casket's, the slip's and random secrets at $n=6$. It checks the
turned tie, the turned web for every $(a,b)$ at $n=4$ (with the untouched web and
Hadamards-on-the-rods-only as negative controls), and the master's value:

```python
out = Hn(7) @ Uc @ Hn(7) @ ket(0, 0, 0, 0, 0, 0, 1)
assert np.isclose(out[0b0101001], 1) and np.isclose(out[0b0010101], 0)
```

G1 is asserted as linearity (one passage is one map on any state, collars upright or
crosswise) and G2 as a definite reading for all 128 settings. The rods-crosswise,
flag-upright hazard is asserted **as documentation only**: four outcomes at $\tfrac14$, and
with the flag window crosswise the rods still read $a$ only half the time — "That would
gamble the only wish."

The sketchbook builds *The Turned Web* with each unmarked rod as **one** plain edge (its two
collars cancel) and the two red targets unfused:

```python
assert g.num_vertices() == 18 and g.num_edges() == 13 == 6 + 3 * 2 + 1
v = matrix_of(scaled(g, -5)).ravel()
assert np.allclose(v, ket(0, 1, 0, 1, 0, 0, 1))
pts = points_after_reduce(scaled(g, -5))
assert [ph for ph, _ in pts] == [0, 1, 0, 1, 0, 0, 1]
```

The tubes slid outward (named `color_change` on each point) give the same picture. The turned
web is **derived** with named rules — `id_simp` on the two-box stretches, then `color_change`
on every dot of the web — and lands on the fan-out with no scalar, every box gone and the
same ties ("Nothing has moved"), for every $(a,b)$ at $n=4$. Cleve's form is built for random
secrets at $n=6$ with `add_power(w-n-1)` and reduced to a red $\pi$ point on each marked rod
and a green $\pi$ point on the flag.

**Pictures (P3.1–P3.4).** The tubes swap every ring's colour and keep every place (P3.1).
Pushed and read plainly, the web gives the inked squares and the flag: $\ket{1000101}$,
amplitude $+1$ (P3.2). The swallowed flag cord draws nothing. The casket's passage, run as
the Lua runs it, **is** the blackboard's
$(H\text{ on window-crosswise rows})\,U_f\,(H\text{ on knob-crosswise rows})$. The file
checks this on several sets of cases:

- all 128 knob settings, every collar upright and every collar crosswise, through the
  macro's own words;
- every one of the $2^{14}$ collar patterns at two knob settings, where the Lua has exactly
  one reading in exactly the 160 cases where the blackboard's state is a basis state;
- 200 random settings.

At the setting the dialogue declines (l.570: the flag knob's collar upright), the macro
refuses with the gamble fault, and the likeliest reading has probability $\tfrac12$. P3.3's
windows `dldlddl` are $\ket{0101001}$. Inside (P3.4), section 1a's thirteen lengths are
there by name, with the ties behind. The lower red ring is below the upper and further
across. The backs gleam as the windows do. No length crosses a ring. The words "ring" and
"tie" sit in the gap above the flag row (y = 0.515). Their node boxes are measured once with
lualatex, as `\tiny\itshape` in the book's TeX Gyre Pagella with the 1pt inner sep, and they
clear every length, knot and ring. The margin is thin by design: the least gap is 0.005 cm,
"tie" against the first tie, inner sep included. The fourteen Genie lines
are pinned word for word in both files. Granting changes only the windows and the seal, and
lifting changes only the lid.

## IV. `PairsBox.py` — *The Pairs Box* (the *da capo*; one round of Simon's problem)

> "So that's the evening. A plain wish buys one answer. A crosswise wish on a box of marks buys the whole secret, read straight off its ties. On a pairs box it buys one clue: a line through the secret, not the secret. And no turning of the collars buys what the box doesn't hold." — Achilles

The Tortoise ties the first two places to the upper answer-cord and the third to the lower.
Achilles finds that settings answer alike exactly when they differ at the first two places.
She folds the cloth over the answer-cords' ends, slides like rings together, and supposes
two clues; Achilles eliminates his way to the secret, and the refrain returns.

The blackboard checks the box's promise as Achilles touches the ties, then Result 13.9:

```python
assert np.allclose(rho, (np.eye(8) + pauli_on(Z, s)) / 8)
assert not np.allclose(rho, (np.eye(8) + pauli_on(X, s)) / 8)   # the second layer matters
assert clues == [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)]
```

Linearity comes free at the drawn size: all $4^8$ functions from three rods to two answer
wires are enumerated, and every one with the promise (for every nonzero $s$; $24$ each) is a
web with some answer bits flipped, $f(x)=Mx\oplus c$. With three answer wires an explicit box
with the promise for $s=110$ is not a web. The same $\rho$ comes out for random **non-linear** pairs boxes — $s=110$, the weight-3
secret $1110$, the chapter's $100010$ and a random secret on five rods — so the round sees a
box only through $s$. A random complex unitary on the answer wires before the discard
changes nothing (causality). The supposed clues $001$ and $110$ leave only $110$; the
all-dark clue passes every check and removes nothing; every independent pair of clues fixes
$s$, and random independent sets of five do at $n=6$. The expected-rounds bound is checked
from the counted probabilities $2^k/2^{n-1}$.

The sketchbook builds the round **doubled** by hand (PyZX 0.10.6 refuses ground vertices):
the circuit and its mirror copy, which is the same circuit because everything is real, with
each answer wire joined to its own mirror by a plain edge — the unnormalized cap, the discard.
Fusion alone performs the Tortoise's fold:

```python
assert simp.spider_simp(h)
assert sorted(h.vertex_degree(v) for v in reds) == [2, 4]   # one red ring per answer-cord
assert same_picture(bare, reduced_4rho())                   # 4 rho: green dot on 1,2,1bar,2bar; cup on 3,3bar
```

The parity gadget obeys $P_s=2^{|s|/2-1}\times$gadget for $|s|=2,3,4$ (the bare gadget is
**not** $P_s$ unless $|s|=2$), and $\rho=2^{|s|/2-n}\times$gadget for the pairs box
($\tfrac14$) and the weight-3 secret, where $2^{1-n}\times$gadget fails. Simon's classical
separation and BBBV's Theorem 3.5 are stated in the chapter and cited; no file tests a query
bound.

**Pictures (P4.1–P4.3), and the whole run.** The pairs box's ties **are**
$f(x)=(x_1\oplus x_2,\,x_3)$, and Simon's promise holds for $s=110$ and not for $011$. The
fold moves nothing else. Its band covers both answer-cords' ends and no ring. The
answer-cords turn down at columns 5 and 6, cross nothing, and clear the other's rings. Run
together, they land on rings that were already there, and one round gives the clues
$000,001,110,111$, a quarter each, every one with $z\cdot s=0$. Over the whole dialogue the
file checks the following:

- there are fifteen pictures, all inside the four challenges, none after P4.3;
- the keyed pictures are exactly P1.1, P1.2, P1.4, P2.1, P3.3, P3.4, P4.1 and P4.2;
- every call is legal and in its place;
- no picture shows the gamble;
- section 1c's form check passes on both files, "Let Bob draw" never appears, and there are
  three spoken "Referred." lines, all the Genie's.
