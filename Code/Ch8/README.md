# Chapter 8 — Spiders, Red and Green: the companion code

Six challenges in the dialogue, six files here. Each file checks its claim twice, as the
chapter proves its theorem twice: `blackboard()` with NumPy vectors and matrices,
`sketchbook()` with pictures built in PyZX. This is the revelation chapter — spiders are now
first-class — so from here on each file does two things in the sketchbook: it confirms the
two pictures agree as tensors, *and* it **derives** the equality with a named ZX rewrite rule
(`pyzx.simplify.spider_simp`, `color_change`, `bialgebra`, `full_reduce`).

Run one file, or the whole chapter, from the repository root:

```bash
.venv/bin/python chapter08/Code/Ch8/Fusion.py
bash check-code.sh 8          # must end with: 6 files, 0 failed
```

Each file prints one line per strand and exits non-zero if any `assert` fails. Running the
files is this book's analogue of type-checking. Every picture is evaluated through
`tools/dfp.py` (`matrix_of`, `same_picture`, `same_picture_up_to_scalar`), which force
`strategy="naive"`: PyZX 0.10.6's default contraction is wrong on some closed diagrams.

**Two conventions carried in every file's header.** The Z spider (green) is
$\ket{0\cdots0}\bra{0\cdots0}+e^{i\alpha}\ket{1\cdots1}\bra{1\cdots1}$. The X spider (red) is
built **intrinsically** from the *normalized* $\ket\pm=\tfrac1{\sqrt2}(\ket0\pm\ket1)$ —
$\ket{+\cdots+}\bra{+\cdots+}+e^{i\beta}\ket{-\cdots-}\bra{-\cdots-}$ — and **not** as
$H^{\otimes}ZH^{\otimes}$. That choice is the whole point of Challenge II: colour change
becomes a theorem the code *proves*, not a definition it assumes. Each header pins the two
spiders as matrices ($Z(\pi)=\mathrm{diag}(1,-1)$, $X(\pi)=\text{NOT}$) and records that the
red one-leg state is $\sqrt2\ket0$, the unnormalized cup convention.

**Scalars, stated honestly.** Fusion and colour change hold *on the nose* (exact, scalar
included). Copy carries $\sqrt2$, π-commutation a scalar exact only at phase 0, bialgebra
$\sqrt2$, the Hopf law $2$, three CNOTs $\tfrac1{2\sqrt2}$, the Octet $\sqrt2$. Where a rule
drops the global scalar, the reduced diagram is compared with `same_picture_up_to_scalar`, and
the exact factor is checked separately with `matrix_of`.

The files walk the ZX vocabulary the dialogue builds cord by cord — first a spider gets a
name, a colour and a turn, then the two colours meet and copy each other, then complementarity
lets the cradle come undone, which is exactly what cooks three CNOTs into a swap. The last
file assembles the eight-wire Octet, counts its edges in code, and checks that only its
topology matters — the picture the chapter is named for.

**The prop pictures.** The dialogue draws its props with three macros from
`pdf/props/ch08.tex` — the Crab's cords and curtain-rings, on the beam, loose in the hand
or on the second hooks (`\ringcords`), the Crab's press with its flip-slot (`\koekpan`), and
(round 4, *Show the process*) a step strip of rings, `\ringsteps[opts]{tangle}{steps}{captions}`,
whose every frame starts from the state the one before left. Their Lua rules are transcribed
once, in `tools/props_ch08.py` (no matrices there): the `;` bracket (a bare view word, then
`key=value`), the turn words in twenty-fourths, the hooks a cord hangs from (`in at top and
bottom`, …), every tangle the dialogue names, the computed views — `slid` (the turns added),
`flipped` (the other colour, the same cords and turn), `pushed` (the ring's turn mirrored),
`pressed` (what comes off onto the second hooks) — the steps (`add=` a cord to the first free
hook of top, bottom, middle; `slide=`; `reshape=` the rings, then the cords), the motion marks
(`marks=on`: a moved part's ghost, its arc, a pin on what is held), Achilles's two refuted rules
(`expects=passes`, `expects=tighter`, drawn paler), the strip's expansion (`keyroom=on` on its
later frames), the log lines the Lua writes, and all **62** `tex.error` sites in the Lua's
order, each keyed and each held to its message template.

The transcription was **proved against the Lua** before it was pinned: every call of both
dialogue files, 2 289 aimed calls (every turn on the slide and the push, every hook set on both
adds, a single ring's items in every order, each kind twice over too) and 30 000 random ones on each of seven seeds, run through
`texlua` with `tex` stubbed and through the Python, agree record for record — every fault
message, every log line, a strip's printed frames, every mark's TikZ line and every keyed
label node. The real build's log (44 dump lines, both files) is the files' `describe()` line
for line. `P.check_lua()`, called first in every file, holds `ch08.lua` chunk by chunk
(`LUA_PINS`), the wrong rules' comment word for word (`WRONG_PIN`) and `ch08.tex` (`TEX_PINS`:
the mark styles, the colours, the macros' expansion) to the text that proof was run on; a
comment-only edit passes, except in the wrong rules' comment (pinned whole, as in Chapter 7)
and to the Lua's quotes of the dialogue (each must stand in it).

Each file reads the pictures of its own section of `dialog.md` through `tools/dfp_props.py`
(`P.check_section`): its blocks (calls `[1,2]`, `[1,1,2]`, `[1,2,2]`, `[1,1,3]`, `[1,1]`,
`[1,1,1]`; frames `[3,2]`, `[1,1,2]`, `[1,2,2]`, `[1,1,3]`, `[1,1]`, `[1,2,1]`; the same in
`dialog.raw.md`), their captions (words only, frame by frame; no ring called a spider or a turn
a phase, never the pan's name; no quantity word under a drawn copy of the frame or beside a
cord), the speech or direction each follows, the round's direction and caption edits
(`P.check_edits`: N1–N8, C1–C4, each new text once in its section and its old text gone, in
both files), and every reading against the scene at that moment — where the dialogue says a
turn or a hook in words, the file parses those words and compares. Each file also **ties the
transcription to its own rules**, contracting the drawn tangle in NumPy (a green ring a Z
spider, a red ring an X spider, a turn $t$ the phase $2\pi t/24$): a structural check, not a
second proof. A prediction frame is checked against the challenge's own model of what
Achilles said, and shown to differ from the true frame beside it.

---

## I. A Name, a Colour, a Turn — `Ch8/Fusion.py`

"It has a family now. Has it a name?" The unnamed three-legged node of Chapter 7 grows a
family: any number of legs, a colour, a phase. Two same-colour spiders that share a leg are
one spider, and their phases add. The blackboard uses generic random phases (seeded, never
the trivial $0$) and asserts fusion along one leg, along a multi-leg chain, for a general
$(m,k)$ shape, and across two shared legs:

```python
assert np.allclose(zspider(1, 1, b) @ zspider(1, 1, a), zspider(1, 1, a + b))
assert not np.allclose(zspider(1, 1, b) @ zspider(1, 1, a), zspider(1, 1, a))  # they really ADD
assert np.allclose(zspider(1, 3, b) @ zspider(2, 1, a), zspider(2, 3, a + b))
```

The sketchbook compares two green spiders on a wire with one, **exactly** (`same_picture`),
rules out `one_green(a)` as the wrong sum, then derives the fusion with `spider_simp`: the two
dots become one, whose phase is `(a + b) % 2`.

*Pictures.* The lay strip on the beam (keyed on its first frame; the later frames keep the
labels' room as phantoms, so the three stand level): the ring as the Crab found it, one cord
in from the high left hook and two out to the high and low right hooks (N1); another in, to
the **low left** hook; another out, to the right hook **between** the other two (N2) — each
frame computed from the one before, the earlier cords kept on their hooks, no mark (a cord
from the hand). Its last frame is round 3's keyed `\ringcords*[beam]{green ring, five cords}`
**cord for cord**. Then the two rings on one cord at a third and a quarter and the same cord
slid, marked: the near ring's ghost where it stood, the slide arc above the cord from it to
the far ring, the far bead's ghost at a quarter and its arc clockwise to **14**, the dialogue's
"a third and a quarter together" parsed to the same number, and a pin on the cord's far end,
which "does not stir". The ties: every add, from every set of hooks, is the spider with one
more leg; for all $24\times24$ pairs of turns, both colours, the slid ring is the fused spider
**exactly**:

```python
assert f03.drawn.cords == p11_round3.drawn.cords and f03.drawn.rings == p11_round3.drawn.rings
assert f03.moved.hook == "middle" and ends(f03.drawn) == [L(0.42), L(-0.42), R(0.42), R(0), R(-0.42)]
assert np.isclose((b0 - b1) % 360 + 2 * 0.07 / rad * 180 / np.pi, 15 * left.t)   # by the near turn
assert np.allclose(tangle_map(b12.drawn), tangle_map(b13.drawn))          # fusion, the pictured turns
```

## II. The Other Colour — `Ch8/ColourChange.py`

"A red ring and a green ring see different things." A Hadamard on *every* leg turns a Z
spider into an X spider of the same phase. Because the red spider was defined from $\ket\pm$
and not from $H$, this is a genuine theorem:

```python
for alpha in [0.0, 0.7, 1.3, 2.9]:
    assert np.allclose(H @ zspider(1, 1, alpha) @ H, xspider(1, 1, alpha))
assert not np.allclose(H @ zspider(1, 1, 1.3), xspider(1, 1, 1.3))  # one leg is not enough
```

The genericity assert needs a *non-zero* phase, since $Z(0)=X(0)=I$ would let a false claim
pass. The sketchbook dresses a green spider in Hadamard-**edges** on both legs and finds it
equal to a red spider **exactly**, then derives the change with the *named* rule
`color_change` — not `full_reduce`, whose first move (`to_gh`) *is* the colour-change and would
read circular. After the rule the dot's type is `VT.X` and every edge is simple again.

*Pictures.* The pan shut (keyed: rim, flip-slot, handle, second hooks; never marked — it
works unseen), the slot close-up with the ring come clear **red** from green and, marked, the
green ring's ghost where it sat **on the far side of the rim** (N4) and its way down through
the slot — the ghost's whole disk above the rim's upper edge, the ring's below its lower edge,
the arc crossing the rim inside the slot's opening, all measured on the drawn extent; then the
four-cord ring before and after every cord went through (a surprise, kept as before and after,
unmarked): red, the same cords, the same turn. The tie: for every turn and both colours, on
the two shapes drawn, a flip on every leg is the flipped ring exactly — and a flip on one leg
alone never is:

```python
check_slot_marks(b22, "green")                          # ghost green, far side; swing through the slot
assert np.allclose(Hout @ tangle_map(T) @ Hin, tangle_map(F))
assert not np.allclose(kron(H, I2) @ T22, tangle_map(P.view_flipped(P.single(c, t, 2, 2))))
```

## III. Each Copies the Other's Points — `Ch8/Copy.py`

"...Oh." Achilles sends in one red point and two come out. A phase-free spider copies the
*other* colour's basis points — up to $\sqrt2$ — and a half-turn pushed through the other
colour reverses that colour's turn:

```python
assert np.allclose(zspider(1, 2, 0) @ r0, (1 / np.sqrt(2)) * kron(r0, r0).ravel())  # copy, up to sqrt2
r_phase = xspider(0, 1, 0.9).ravel()
assert not _proportional(zspider(1, 2, 0) @ r_phase, kron(r_phase, r_phase).ravel())  # only BASIS points
for a in [0.7, 1.3]:                                                    # pi-commutation, a != 0
    lhs = xspider(1, 1, np.pi) @ zspider(1, 1, a)
    rhs = zspider(1, 1, -a) @ xspider(1, 1, np.pi)
    assert _proportional(lhs, rhs) and not np.allclose(lhs, rhs)        # up to scalar, not exact
```

The negative asserts are the teeth: a red point *with a phase* is not copied even up to a
scalar, and π-commutation is checked at $\alpha\neq0$ so it is a genuine up-to-scalar
equality, not a hidden exact one. The sketchbook compares point-into-fork with two points
(up to scalar, exact factor $1/\sqrt2$ checked separately) and derives it with `full_reduce`;
π-commutation is exact at phase 0 and only up-to-scalar at $\tfrac{7}{10}\pi$.

*Pictures.* The red point and the green fork (the state Achilles predicts on), then the
prediction frame beside the true press: "as Achilles expects", drawn paler and never keyed,
**one** red point on the upper right hook (his "comes out on one leg, unchanged", the file's
own model of the rule), and — keyed, the first true pressed picture — **two** red points and
no green ring; the red half-turn before the green ring at a third (the dialogue's own words),
then pushed and marked: the traveller's ghost where it waited, its arc clear **over** the
ring, the green bead's ghost at a third and its swing to **16** "round beneath the cord" (N5).
The ties: pressed point and fork is copy, $1/\sqrt2$, both colourings; pushed is
π-commutation for every turn, both colourings, **up to a non-zero scalar** and equal only at
turn 0 (at the pictured third the scalar is $e^{2\pi i/3}$):

```python
assert np.allclose(tangle_map(b32.drawn).ravel(), passes(b31.drawn).ravel())   # the rule drawn
assert len(b32.drawn.rings) == 1 and len(b33.drawn.rings) == 2                 # refuted
assert s0 > bottom > s1 and P.turn_angle(g.t) > s0 and s1 > P.turn_angle(g2.t)  # 8 -> 16 by the bottom
assert exact == [0]                                    # the push: equal only at no turn
```

## IV. The Cradle Comes Undone — `Ch8/Complementarity.py`

"Fewer cords for the same job." Two facts that are one: the bialgebra (two greens crossed to
two reds equal one green–red pair, up to $\sqrt2$) and its twin the Hopf law (a green and a
red joined by *two* wires equal the two disconnected, up to $2$). The blackboard builds the
$K_{2,2}$ cradle as $[X\!\otimes\!X]\circ(\text{cross})\circ[Z\!\otimes\!Z]$, shows it reads
parity and equals the single pair over $\sqrt2$, and rules out swapping the colours:

```python
lhs = kron(X21, X21) @ kron(I2, SWAP, I2) @ kron(Z12, Z12)
assert np.allclose(lhs, rhs / np.sqrt(2))                       # cradle = pair / sqrt2
assert np.allclose(X21 @ Z12, disc / 2)                         # Hopf: link = disconnected / 2
```

The sketchbook checks the cradle has 8 edges with every interior dot at degree 3, and derives
the Hopf disconnection with `full_reduce` — after which the input and output boundaries lie in
different components (a small graph walk confirms it). It also documents a construction trap:
in PyZX 0.10.6, calling `add_edge` on the same pair twice *cancels* the edge, so the genuine
double link is built with a phase-0 identity spider on one of the two paths.

*Pictures.* The cradle on the beam (keyed: in, green, red, out), its four cross cords read
out of the direction that strings them ("top green to top red; …") and matched to the
transcription's — a counted frame: eight cords, four rings, no mark; off the pan, the pair —
red on the left gathering, green on the right issuing; the twice-tied pair, the prediction
frame ("Two cords must bind them twice as tightly": both cords kept, short and taut, the rings
pulled close, paler, never keyed), then nothing between. The ties: the transcribed cradle *is*
the file's `lhs`, the pressed pair its `rhs`, and the pair with the colours the other way
round is not the cradle even up to a scalar; the twice-tied pair is half the untied one; the
prediction keeps its two cords where the press leaves none:

```python
assert np.allclose(tangle_map(b41.drawn), tangle_map(b42.drawn) / np.sqrt(2))   # bialgebra
assert not _proportional(tangle_map(swapped), lhs)     # the colours change sides: the rule
assert np.allclose(tangle_map(b43.drawn), tangle_map(b45.drawn) / 2)            # Hopf
assert len(between(b44.drawn)) == 2 and len(between(b45.drawn)) == 0            # refuted
```

## V. Three Crossings, One Swap — `Ch8/SwapFromCNOTs.py`

"Three of them, and they just... swap." The headline theorem. Three CNOTs, the middle one
reversed, make a SWAP — exact on the blackboard, three sparse matrices in a line:

```python
assert np.allclose(CNOT @ CNOTp @ CNOT, SWAP)                  # exact, no scalar
assert np.allclose(CNOT @ CNOT @ CNOT, CNOT)                   # NEGATIVE: same orientation gives CNOT
assert np.allclose(spider_cnot, CNOT / np.sqrt(2))             # the honest per-gate scalar
```

The picture carries a wart the algebra does not: each bare spider-CNOT is $\mathrm{CNOT}/\sqrt2$,
so the assembled three-CNOT tangle is a bare crossing only up to $\tfrac1{2\sqrt2}$. The
sketchbook asserts that up-to-scalar equality, checks the exact factor with `matrix_of`, and
lets the **rules derive** it: `full_reduce` cooks the tangle down to exactly 4 vertices, all
of them boundaries — a plain crossing of wires.

*Pictures.* The three gates, their colours top and bottom read in the order the Crab names
them ("green-red, then red-green, then green-red"), and what came off: no ring, the upper
left hook to the lower right and the lower left to the upper right. The ties: one gate is
$\mathrm{CNOT}/\sqrt2$ with the control on the green's line, left to right along the lines is
the order of application, and the crossing is SWAP exactly:

```python
assert np.allclose(tangle_map(T3), CNOT @ CNOTp @ CNOT / (2 * np.sqrt(2)))
assert np.allclose(tangle_map(D), SWAP)
```

The pan's results alone are drawn (it is never stripped), so neither picture carries a mark;
the first caption is "its middle reversed" (C3).

## VI. Octet — `Ch8/Octet.py`

"Eight cords. Eight, and not one of them spare — every one carrying its part, and the whole thing meaning nothing if you pull a single one out."

The master diagram, counted. The Octet is
the bialgebra cradle read as an 8-wire diagram; it evaluates, up to $\sqrt2$, to a single
green–red pair, and only its topology matters:

```python
assert o.num_edges() == 8                                      # eight edges = eight wires
assert same_picture_up_to_scalar(o, pair())                    # Octet = pair, up to sqrt2
assert simp.bialgebra(g, v_red, v_green) and g.num_edges() == 8  # the count is PRODUCED by the rule
assert same_picture(octet_deformed(), octet())                 # only topology matters
```

The derivation runs in the clean **unfold** direction: start from the single pair
(identical to Challenge IV's bialgebra right-hand side), apply the named `bialgebra` rule, and
watch it grow into the eight-edge cradle — the wire count is produced by the rule, not merely
asserted. The last line redraws the Octet with its interior rings at scrambled coordinates and
its cords strung in a different order, and finds the same tensor exactly: slide, bend and
stretch the picture freely — only the connections count.

*Pictures.* The cradle alone on the beam just before the count, "untouched" (C4; its cords
two in, four across, two out, as Achilles counts them); the reshape's strip, from P6.1 four
paragraphs above (`from=off`) and without marks (`marks=off`, so no outline is threaded on a
counted cord): the rings moved, each with its peg (N6), cords straight to them, and then the
cords bent (N8; the read-out N7 dropped) — both frames keep the eight connections, the four
rings and every hook end, and the last is round 3's reshaped cradle cord for cord; and, pressed
from the crooked one, the very pair of the Fourth. The tie: the transcription's cradle, strung
as a PyZX graph cord by cord, has exactly `octet()`'s eight edges under the card's wire map
and is the same picture (`same_picture`); so is each frame of the reshape; the pressed pair is
`pair()`. This file also checks the whole dialogue: 22 calls and 25 frames, all inside the
challenge sections, at most four moments a section and four frames a block; the starred form
on exactly the first true picture of each kind (the lay strip, the pan, the pressed copy, the
frame — a prediction is never keyed); the views each challenge may use; the 44 dump lines; the
marks (none on a frame on the beam, F01–F03, F15, F22–F24; every mark line in a dotted mark
style, never dashed; arcs headed; marks only at the slide, the slot and the push); the quotes
the Lua's comments take from the dialogue; what the code takes from `ch08.tex` (the brackets
a call without one is handed, the strip's frame call) and the card's pinned numbers; and the Lua's 62 error sites, each raised by a call
of its own (`reshaped` and `step-key`, which no macro call reaches, through the model's own
functions) — a single ring's items checked after the whole list is read, so a mix or a
repeat is the same fault whichever comes first, and a ring written in any order the same ring:

```python
assert DIALOGUE.check_whole(ORDINALS) == 22 and len(frames) == 25 and len(dump) == 44
assert [b[0] for b in bad] == list(P.ERROR_SITES) and len(set(P.ERROR_SITES)) == 62
assert fr.marks == () and fr.mark_tikz == () and not fr.pale    # no mark on the counted frame
assert g.num_edges() == 8 and same_picture(g, octet())          # the same eight wires
```
