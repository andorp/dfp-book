# Chapter 9 — *Now You Look*: companion code

Five challenge files, one per dialogue challenge, each checking its claim **twice** —
once on the blackboard (NumPy `assert`s) and once in the sketchbook (PyZX diagrams
compared as tensors through `tools/dfp.py`'s `matrix_of` / `same_picture`, which force
`strategy="naive"`). Running a file is the book's analogue of type-checking.

```bash
# one challenge at a time
.venv/bin/python chapter09/Code/Ch9/Measurement.py
# the whole chapter (must end "5 files, 0 failed")
bash check-code.sh 9
```

This chapter opens **Stage D** — classical wires, doubling, discard. PyZX's `to_matrix`
**rejects ground vertices** (`ValueError: Hybrid graphs are not supported`, pyzx 0.10.6),
so no diagram here uses a native ground. Every discard is hand-built as a **CPM (doubled)
diagram**: a quantum system is drawn as its wire **and its conjugate (mirror) copy**, and
a **discard is a cap** contracting the two copies — which is exactly the partial trace. The
Stage-C "up to a scalar" licence is **withdrawn**: every stated equality is exact, scalar
included, because doubling is what makes a picture's scalar a real, meaningful number.

The files walk from the first look to the whole apparatus of looking — a measurement that
forces one outcome and writes it on a new, classical wire; the mixed state a look leaves,
drawn by doubling a wire with its conjugate; discarding a half as the cap that is partial
trace; and the headline, that the far half of a Bell pair is $\tfrac12 I$ no matter what is
done across the room. The last file dilates a real, leaky measurement à la Stinespring,
hand-builds the nine-wire master, counts its edges in code, and shows causality keeps the
far half unmoved.

**The prop pictures.** The dialogue draws its props with four macros from
`pdf/props/ch09.tex` — the Crab's viewer (`\crabviewer`), the slate its looks are cut on
(`\notchslate`), the polished black slab with the things standing on it and their
reflections (`\blackslab`), and a step strip of slab pictures (`\slabsteps`). Their Lua
rules are transcribed once, with the card's L1 and L2 corrections in force, in
`tools/props_ch09.py` (no matrices there): the viewer's state words and needle rules, the
slate's notches (high in the top edge, low at the foot, unread only under the cloth), the
seven slab scenes with each stone's gloss and reflection (the near stone of a pair of one
piece restless **and slack**), the computed operations — `tip`, `fold`, `pin` (one still
pointer lying toward the stop it read, one notch on the slate), `pull` (only on a string
with every leg folded) — and the three frames (1.44, 1.26 and the laid 1.84 cm).

Round 4 (*Show the process*) adds the bracket's `;` items, read whole so that their order
never matters: a **step** (`step=look high`, `step=cut high`, `step=fold near stone`, …)
takes the state written as the state *before* it and draws the state after it, computed,
with the step's **motion marks** — a ghost where the moved part stood (in dotted outline
where the part covers its old place), a swing or slide arc, the pin ring of the finger that
holds the far stone — unless `marks=off`; a **prediction** (`expects=spyglass`,
`expects=tighter`, `expects=along`) draws, paler, what one of Achilles's three named wrong
rules foresaw; and `\slabsteps` expands into its frames, frame $k$ being the single call
`\blackslab[<the steps before it>; step=<k>]`. The transcription carries the marks down to
their TikZ text, the keyed labels down to their nodes, and the log lines byte for byte, and
it was **proved against texlua**: every dialogue call of both files, about 9 600 aimed calls
(every viewer state and step, every slate length and cut, every scene with every two-step
prefix and every step, every pair of strip steps, every order of a bracket's items) and
30 000 random ones per seed, three seeds — 0 mismatches in about 900 000 records, all
**fifty** `tex.error` sites raised, each message byte for byte. `check_lua` then holds the
Lua and the `.tex` to the text proved (`LUA_PINS`, `TEX_PINS`, chunk by chunk, comments left
out), and the five round-4 comments that quote the dialogue word for word (`COMMENT_PINS`):
a change to the drawing alone fails the chapter, a comment may be reworded freely unless it
quotes the dialogue. `check_sites` holds the fifty sites to the Lua's own list.

Each file reads the pictures of its own section of `dialog.md` through
`tools/dfp_props.py`, checks their layout (calls `[1,3,1]`, `[2,1,2]`, `[4]`, `[1,2,1]`,
`[1,1,2]`; frames 5, 5, 4, 4, 5; the same in `dialog.raw.md`), one frame height per block,
the captions (words only; no Stage-D word, no quantity word, never *slack*), the round's
caption and direction edits (C1–C7, T1) in both files, the places each macro and operation
may appear, the speech or direction each block follows, the Lua comments' quotes of its
section, and every reading against the scene at that moment: each step from the state
before it, each mark where the card's dump list puts it, each prediction against the truth
it fails, each marked frame's marks-off twin against round 3's picture. Each file also
**ties the transcription to its own mathematics**, NumPy against its blackboard — a
structural check, not a second proof: a tight reflection is a pure piece and a slack one a
mixed piece, on every scene; the fold is the partial trace; the pin is the read map; the
pull is causality.

---

## I. `Measurement.py` — *The Look That Forces*

> "The look didn't read the stone, it *made* it." — Achilles

Achilles finally looks: the needle swings to one stop at random, a notch is cut, the stone
is spent. The **blackboard** builds the two projectors $P_0,P_1$ and checks the Born rule on
a **non-real** state — the outcome probability is $p_i=\langle\psi|P_i|\psi\rangle=|\psi_i|^2$,
the weights sum to $1$, the post-measurement state lies wholly along the $|i\rangle$ axis (up
to a phase), and measuring twice returns the same $i$ (the answer is settled).

```python
assert np.isclose(p0, abs(psi[0]) ** 2) and np.isclose(p1, abs(psi[1]) ** 2)
assert np.allclose(P0 @ post0, post0)          # measure twice = once
```

The **sketchbook** builds the outcome's new home, the **classical wire**, out of the
**read-spider** $Z(2\!\to\!1)=\sum_i|i\rangle\langle ii|$ — the Chapter-8 green spider, put to
a new use. Fed a state on one leg and its **conjugate** on the other, its classical wire
carries the Born-probability vector exactly (scalar $1$). The negative asserts that the
conjugate leg is load-bearing: feeding $|\psi\rangle$ on both legs gives $\sum\psi_i^2$, which
is *not* the probabilities for a non-real state.

```python
target = np.outer(ket(0), ket(0, 0).conj()) + np.outer(ket(1), ket(1, 1).conj())
assert np.allclose(R, target)                                   # read-spider, scalar 1
assert np.allclose((R @ np.kron(psi, psi.conj())).ravel(), np.abs(psi) ** 2)
assert not np.allclose((R @ np.kron(psi, psi)).ravel(), np.abs(psi) ** 2)   # no conjugate: wrong
```

*Pictures.* The viewer, keyed, with a restless stone in the dish and the needle in the
middle (flask left, dish right, as the direction sets them); then the look as a **step**
from that state — the stone spent, the needle hard against the right stop, its ghost at the
middle and a swing arc round the pivot (83° to 47.8°, short of the needle at 36°) — with,
before it and paler, **Achilles's spyglass**: the needle at the same stop but the stone still
restless, as if the look had only shown what was there; beside them the keyed slate, its one
notch computed as `step=cut high` from the clean slate; then the row *high, high, low, low,
high, low*. The file runs the First through its own model of the viewer (a look is the
projective measurement, high the outcome $0$; the stylus cuts only when the needle leaves the
middle): the look's frame is the model's state after it, the spyglass's differs from it in
the stone, the re-look has probability $1$ and cuts nothing, and the last row is the first
notch followed by the words Achilles reads aloud, parsed out of his line:

```python
assert (v12.dish, v12.needle) == (V.dish(), V.needle) == ("spent", "high")
assert w12.dish == v11.dish == "restless" and w12.dish != V.dish()   # the spyglass, refuted
assert (ghost.name, ghost.at) == ("needle", (90,)) and (a0, round(b1, 1)) == (83, 47.8)
assert list(s14.shown) == V.slate == list(s13.shown) + heard        # a notch an answer
```

## II. `DensityMatrix.py` — *A Stone That's Been Looked At*

> "A thing and its reflection." — Tortoise

A looked-at-but-unread stone is neither restless nor a plain answer: it is a **mixture**. The
**blackboard** builds $\rho=\sum_i p_i|\psi_i\rangle\langle\psi_i|$, checks it is Hermitian,
positive and unit-trace, and that Achilles's unread $|+\rangle$ is the maximally mixed
$\tfrac12 I$ — differing from the pure $|+\rangle\langle+|$ exactly in the off-diagonal
coherence the look destroyed. Purity $\operatorname{tr}(\rho^2)$ is $1$ pure, $\tfrac12$ mixed.

```python
assert np.allclose(np.diag(pure_plus), np.diag(unread))   # same diagonal, lost phase
assert np.isclose(np.trace(unread @ unread), 0.5)         # tr(rho^2) = 1/d
```

The **sketchbook** is **doubling**: a state drawn as itself beside its **conjugate** copy. A
doubled pure state reshapes to $|\psi\rangle\langle\psi|$; the two copies of a non-real state
genuinely differ, and the mirror is the *conjugate*, not the transpose or dagger. The
negatives rule out the mix-ups: $|\psi\rangle\otimes|\psi\rangle$ (no conjugate) reshapes to
$\psi\psi^{T}\neq|\psi\rangle\langle\psi|$, and evolving the doubled state with $U\otimes U^{T}$
or $U\otimes U^{\dagger}$ (instead of $U\otimes\bar U$) gives the wrong $\rho'$.

```python
assert np.allclose(doubled.reshape(2, 2), np.outer(psi, psi.conj()))    # = |psi><psi|
assert not np.allclose(np.kron(psi, psi).reshape(2, 2), np.outer(psi, psi.conj()))
assert np.allclose((np.kron(U, U.conj()) @ doubled).reshape(2, 2),
                   U @ np.outer(psi, psi.conj()) @ U.conj().T)          # U (x) Ubar
```

*Pictures.* The look taken unseen, as steps from the state the First left (the dial and
the slate under their cloths, no mark: the cloth hides the swing; the slate's list is P1.4's
with one `unread` after, and the transcription refuses it uncovered); a stone cut alone on
the slab, keyed, its reflection tight; then, paler, **Achilles's "tighter still"** — the
looked-at stone drawn with its reflection tight — beside the truth, the same stone with its
reflection slack. The tie runs over **every scene of the slab**: each stone, and the needle
and flask standing ready, is drawn tight exactly when its own reduced state has
$\operatorname{tr}\rho^2=1$ — a Bell half $\tfrac12 I$ included. The negatives rebuild the Lua
as it was before L1 (the Bell half drawn tight), and take the prediction's own picture: the
tie fails on both.

```python
assert tie_holds(P.SCENES, states)                       # tight <=> pure, every piece (L1)
assert not tie_holds(pre_l1, states)                     # the pre-L1 Lua: a Bell half drawn tight
assert not tie_holds({"looked-at stone": w24.state}, states)   # "tighter": tight, yet mixed
assert s22.before == first_row.notches and s22.notches == first_row.notches + ("unread",)
```

## III. `PartialTrace.py` — *Cover the Other Half*

> That fold is "cover the other half." — Tortoise

To cover a half is to **discard** it. The **blackboard** takes a genuinely **mixed** random
$\rho_{AB}$ and checks that contracting the $B$-copies with the cap $\langle00|+\langle11|$
equals the partial trace $\operatorname{tr}_B$, and preserves the trace. It also pins the two
load-bearing scalars: the doubled bare cup discards to $I$ (a closed value of $2$), the
doubled Bell state to $\tfrac12 I$.

```python
discard = np.einsum('abAB,bB->aA', r, CAP.reshape(2, 2))
assert np.allclose(discard, trB)                          # discard IS the partial trace
```

The **sketchbook** draws the discard as a **cap** and shows it is the *unnormalized*
$\langle00|+\langle11|$ (not the normalized one). The doubled state with its near half
discarded is the unfolded chain $S$–cup–$M$–cap–$\bar M$–cup–$\bar S$, which yanks to the far
half cupped to its own mirror: the bare cup to $I$, the Bell state (scalar $\tfrac12$) to
$\tfrac12 I$ — exactly, scalar included. A generic non-real doubled state confirms cap =
partial trace.

```python
assert same_picture(discarded(), cup_ref())                        # bare cup -> vec(I)
assert same_picture(discarded(scalar_power=-2), cup_ref(scalar_power=-2))   # Bell -> vec(1/2 I)
assert not same_picture(discarded(scalar_power=-2), cup_ref())     # 1/2 I is not I
```

*Pictures.* One block, one motion: the plain pair (two stones each cut alone, keyed `far`,
`near`); the near stone tipped, a step — its ghost standing in dotted outline where it stood,
a swing arc round its right side, toward Achilles's end (the direction T1 gives that side);
then, paler, **Achilles's "along"** — the near stone folded and the far stone tipped down
too, "something of the far one" going with it — right before the truth: the fold, a step
from the tipped stone, its ghost above the surface going into its reflection, no arc, and
the far stone's drawn entry P3.1's to the letter, as
$\operatorname{tr}_{\text{near}}(\rho_{\text{far}}\otimes\rho_{\text{near}})=\rho_{\text{far}}$
for random non-real stones. Each marked frame's marks-off twin is round 3's picture. The
transcription refuses a tip or fold of the far stone and a second fold of a stone already
gone:

```python
assert far3 == far1 and np.allclose(kept, rho_far)          # unstirred, drawn and computed
assert (wfar.mode, far3.mode) == ("tipped", "standing")     # "along", refuted
assert (P.f(x0), P.f(y0), P.f(x1), P.f(y1)) == ("1.000", "0.236", "1.055", "0.035")
assert P.faults(P.blackslab, "fold far stone", "plain pair") == ["far-stone"]
```

## IV. `NoSignalling.py` — *What the Twin Cannot Tell* (*The Theorem, Twice*)

> "There's nothing I can do here that reaches there." — Achilles

The headline. The **blackboard** evaluates both sides: tracing the **near** half of
$|\Phi^+\rangle\langle\Phi^+|$ gives $\tfrac12 I$, exact — and it stays $\tfrac12 I$ after a
random **non-real, non-unitary, trace-preserving** Kraus channel on the near half. Causality
(trace-preservation) is the step that saves it; the negative confirms a **non**-trace-preserving
map *does* move the far half.

```python
assert np.allclose(sum(K.conj().T @ K for K in Ks), I2)   # trace-preserving
assert np.allclose(np.einsum('abcb->ac', apply_on_B(rho, Ks)), I2 / 2)   # far half unmoved
assert not np.allclose(far_bad, I2 / 2)                   # a non-TP map DOES move it
```

The **sketchbook** is the chapter's rarer verdict — the picture is the shorter proof. The
doubled Bell chain pulls straight in **one yank** to $\tfrac12$ cup$(S,\bar S)=\tfrac12 I$
(and `full_reduce` derives the same yank, carrying the scalar). Causality is drawn twice: the
discard-cap swallows any doubled channel, $\text{cap}\circ(U\otimes\bar U)=\text{cap}$, with
the plain, transpose and dagger mix-ups all failing; and the same box, ridden on the near half
of the chain with its conjugate reflection on the mirror leg, leaves the far half $\tfrac12 I$
unchanged.

```python
assert same_picture(half, far_cup)                                 # one yank, scalar 1/2
assert np.allclose((CAP @ np.kron(U, U.conj())).ravel(), CAP.ravel())     # Ubar: swallowed
assert not np.allclose((CAP @ np.kron(U, U.conj().T)).ravel(), CAP.ravel())   # dagger: no
assert same_picture(zigzag(mbox=PH, mbarbox=conj_reflection), zigzag())   # channel, far unchanged
```

*Pictures.* The pair of one piece, keyed — the far stone under its cloth, the near stone
restless and **slack** (a Bell half, $\operatorname{tr}\rho^2=\tfrac12$), placed after the
direction that shows it; the string laid far to near; the near stone folded (the string
joined round the fold — the same cover as the Third's, a repeat, unmarked); and the pull, a
step from the folded string, marked with the pin ring on the far stone's cloth (the finger
that holds it) and a slide arc above the string's old way, in toward the far stone — no
ghost, as the string is never ghosted. Its marks-off twin is nothing on the black but the
cloth, the string straight down to its own reflection, "half" chalked beside it, which is
$\tfrac12\,$cup$(S,\bar S)=\tfrac12 I$. A tip before the fold pulls to the same picture, as a
random non-real channel on the near half leaves the far half $\tfrac12 I$; the pull before
the fold, and any touch of the far stone, are refused:

```python
assert b44.shown.things == (cloth,) and b44.shown.string == "pulled"
assert (slide.kind, pin.kind) == ("slide", "pin") and b44.mark("ghost") == []
assert np.allclose(0.5 * CUP.reshape(2, 2), far_alone)      # the pulled far stone = 1/2 I
assert tipped.shown == b44.shown                           # any box on the middle leg: swallowed
```

## V. `Stinespring.py` — *A Look with the Lights On* (channels, dilation, causality; the master)

> "Because discarding after the works is discarding straight off." — Tortoise

A real look is a **channel**. The **blackboard** builds a Stinespring **isometry**
$V\colon A\to B\otimes E$ from a non-real 2-qubit unitary (checking $V^{\dagger}V=I$), reads
its **Kraus** operators $K_i=(I_B\otimes\langle e_i|)V$ straight off (checking
$\sum_i K_i^{\dagger}K_i=I$), and confirms $\Lambda(\rho)=\sum_i K_i\rho K_i^{\dagger}
=\operatorname{tr}_E(V\rho V^{\dagger})$ and $\operatorname{tr}(\Lambda\rho)=\operatorname{tr}\rho$.

```python
assert np.allclose(sum(K.conj().T @ K for K in Ks), I2)   # sum K^dag K = V^dag V = I
assert np.allclose(lam_kraus, lam_dilate)                 # Kraus = tr_E(V rho V^dag)
```

The **sketchbook** does the master two ways. The **value build** is the master "The Unmoved
Half" as a hand-built doubled diagram with a concrete **non-real** $V=U\otimes\bar U$: the near
half discarded, the environment discarded, the pointer read, and the outcome **averaged** — the
far half is $\tfrac12 I$ for every $V$. Outcome-**conditioned**, the far half is correlated (the
honest converse), and the conditioned pieces average back to $\tfrac12 I$ with
$P(0)+P(1)=1$; dropping the conjugate copy breaks it. Result 9.5 (reading, then forgetting, is
the plain cap) is shown as a spider fusion.

```python
assert np.allclose(far, I2 / 2)                           # unmoved half, every V
assert not np.allclose(cond0, I2 / 2)                     # conditioned: correlated
assert not np.allclose(far_noconj, I2 / 2)                # the conjugate copy is load-bearing
```

The **skeleton build** takes $V=$ identity and draws the master in unfolded doubled form as an
ordinary graph, so its **nine lines are nine edges** — eight doubled-quantum (four systems and
their mirror copies) plus **one** classical wire (the read-spider's output, line 7). The odd
count forces a classical wire by parity. Drawing $V$ as six explicit identity boxes (fifteen
edges) and then applying the Chapter-8 identity rule to exactly those boxes *produces* the
nine, as `Octet.py` produced its eight via the bialgebra rule. With the outcome forgotten
in-picture, the whole master yanks to $\tfrac12 I$, and the rewrite rules derive the yank.

```python
assert s.num_edges() == 9                                 # the wire count, checked in code
for v in box_ids: remove_identity(boxed, v)               # id rule on V = identity
assert boxed.num_edges() == 9 and same_picture(boxed, s)  # nine, produced
assert same_picture(master_forgotten(), half_cup())       # forget c -> 1/2 I (Result 9.6)
```

*Pictures.* The look laid out, keyed (cloth, near stone, needle, flask, each with its
reflection; a clean slate off the black; the laid frame); then a **step strip** from that
state, `\slabsteps[from=off; marks=off]`: the needle pinned — one still pointer lying left,
toward the low stop, the slate now carrying exactly one notch, at its foot, as the direction
says — and the flask folded, each frame the single call it stands for and the last one round
3's picture; then the works on the string, every leg folded, and the pull, unmarked. The far
cloth and the slab's ends stay where the Fourth had them (L2). The look laid out and the
strip are the **counted frames** of the master's wire map, so they carry no mark: counted as
CLAUDE.md counts lines (each thing with its reflection two, a notch off the black one) they
draw eight, then nine — the master's eight doubled lines and its one classical wire — then
seven; with the marks on, the strip would add an unreflected ghost, a line no count reads.
The ties use a random non-real isometry $V$ (near stone $\to$ near $\otimes$ needle $\otimes$
flask): the pin is the read map, giving the needle's Born probabilities; with the needle's
leg folded (the outcome averaged) the far half is $\tfrac12 I$, conditioned it is not; and
the **pull equality** — the Fourth's pulled strung pair, its marks off, and the Fifth's pulled
look on a string are the same picture — stands beside
$\operatorname{tr}(V\rho V^\dagger)=\operatorname{tr}\rho$; the marked Fourth pull adds the
slide arc and the pin and nothing else.

This file also checks the whole dialogue: twenty-two calls drawing twenty-three frames in
thirteen blocks (three predictions, one strip, four marked frames), none before the First
Challenge; the starred form on exactly the first true frame of each kind (P1.1, B1.2's
slate, P2.3, P3.1, P4.1, P5.1, P5.3; a prediction is never keyed); every call free of
faults; the log's forty-four lines, the same from both dialogue files, with the card's
readings of the changed frames; every marked frame equal to its marks-off twin but for the
marks; every one of the Lua's **fifty** refusals raised alone, in the Lua's order; every
phrase the Lua's round-4 comments quote standing in both files, each in one section; and no
picture after the pull — the Escher glance and the Cat's question come after it, and the Cat
is never drawn.

```python
assert P.visible(strung) == P.visible(works) == b54.shown   # discard after the works = straight off
assert t44.shown == b54.shown and [m.kind for m in b44.marks] == ["slide", "pin"]
assert np.isclose(np.trace(V @ rho @ V.conj().T), np.trace(rho))
assert f20.shown.slate == b51.shown.slate + ("low",)         # the pin: one notch, at the foot
assert [P.lines_drawn(b.shown) for b in (b51, f20, f21)] == [8, 9, 7]   # the counted frames
assert np.allclose(far_avg, I2 / 2) and not np.allclose(cond / np.trace(cond), I2 / 2)
assert DIALOGUE.check_whole(P.ORDER) == 22 and len(frames) == 23
```
