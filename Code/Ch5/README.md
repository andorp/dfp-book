# Chapter 5 — Roads That Cancel: the companion code

Five challenges in the dialogue, five files here. Each file checks its claim twice, as the
chapter proves its theorem twice: `blackboard()` with NumPy vectors and matrices,
`sketchbook()` with pictures built in PyZX and compared as exact tensors. Every equality in
this chapter is exact, on the nose — scalar and sign included, never "up to a scalar"
(that begins in Chapter 8, and so does deriving an equality by rewriting; here PyZX only
*confirms* what the blackboard found).

Run one file, or the whole chapter, from the repository root:

```bash
.venv/bin/python chapter05/Code/Ch5/Deutsch.py
bash check-code.sh 5          # must end with: 5 files, 0 failed
```

Each file prints one line per strand and exits non-zero if any `assert` fails. Running the
files is this book's analogue of type-checking.

**Conventions, asserted where they are used.** Kets are written top wire first, so `kron(A, I2)`
puts `A` on the top wire and, on four query wires, the top wire is the most significant bit.
The gate $H=\tfrac1{\sqrt2}\begin{psmallmatrix}1&1\\1&-1\end{psmallmatrix}$ is a labelled box, built
directly in NumPy; there is no Hadamard helper. In a `Picture`, $H$ is drawn as a *Hadamard
edge* — `p.g.add_edge((a, b), EdgeType.HADAMARD)` — which is exactly $H$; an Euler chain of phase
nodes would carry a stray $e^{i\pi/4}$ and break the equalities, so it is never used. The
difference state $\ket-=\tfrac1{\sqrt2}(\ket0-\ket1)$ is kept **normalized everywhere**, at both
ends of every closed diagram, so the master value comes out clean. The shared plumbing is in
`tools/dfp.py`; the mathematics of each challenge is in its own file. (One helper is missing from
`tools/dfp.py` and defined locally in each file: `hedge(p, a, b)`, the one-line Hadamard edge just
described.)

**The pictures.** The dialogue shows the state of the Tortoise's conduit-board and her two
one-road tiles in eighteen calls drawing twenty-two frames (F01–F22), in eleven blocks, made by
three prop macros of `pdf/props/ch05.tex`:

- `\conduitboard[view; opts]{setting}{caption}`, the board: the whole board (`set`, `before`,
  `past`, `dealt`, `end`) or a close-up of a cup (`home`, `away`, `pockets`, settled or not). The
  options draw the walk's step into the view with its motion marks (`marks=on`), a step on the
  setting before it (`lift=<lane> head high|low|foot|peg`, `wedge=reed`), or a named wrong rule's
  prediction (`expects=add|reed`);
- `\conduitwalk[from=; to=; marks=]{setting}{captions}`, a step strip of one walk down one lane,
  a frame per moment: set out, past the peg or bar, dealt into the cups, the end;
- `\roadtiles[view; opts]{low face, high face}{caption}`, the two tiles: `apart` on the table,
  `laid` in the home pocket (`turn=low|high` turns one), and `read` (`marks=on` lifts an unlike
  pair out).

The Lua behind them (`pdf/props/ch05.lua`) *computes* what each frame shows: the roads (one per
choice of values over the lanes in play, willow first, low before high), each road's token (set
out silver, turned by the peg on a high road, turned by the bar's kick where the rule acts and the
reed token is two-faced — or, the reed token lying flat, the reed's own token turned), the deal
into the cups (two like faces piled, two unlike side by side), the settling, the tiles' turn and
read, the marks of each step, and what a wrong rule predicts. `tools/props_ch05.py` transcribes
that Lua once, for all five files, all but the drawing; where a mark's place is what a file
asserts, its coordinates come from the Lua's own constants. `tools/dfp_props.py` reads each
challenge's pictures straight out of `chapter05/dialog.md` (never copied), and `P.check_section`
checks a section's layout: the calls and frames per block, at most four moments, word-only
captions that never say wire, drawing, picture or box, the same blocks in `dialog.raw.md`, no
caption echoing the paragraph after its block, no call the macro would refuse.

Each file first checks the transcription against the blackboard — silver $=+1$, slate $=-1$,
home $=\bra0$, away $=\bra1$, the flat reed token's face its value, the two-faced reed token
$=\ket-$ closed by $\bra-$, each spreader $=H$, the peg $=Z$, the bar $=U_f$ with its rule read as
$f$ — and then asserts every frame of every strip, in order, against the file's own model at that
moment:

- each ghost stands where its piece stood the moment before (the frame before it, when there is
  one), and a flip mark sits only on a token whose sign the step changed;
- the two deal arcs run from the tokens into their cups, a dealt cup's tokens are the paths
  $2H_{yv}(MH\ket0)_v$, piled where the two paths agree and side by side where they are opposite,
  as the text says;
- the pin ring stands only where the bar's kick lands with the reed token held (F17), and the kick's
  travel has no arc;
- each `expects=` frame equals its named wrong rule as the file computes it, and differs from the
  true frame beside it; the macro itself refuses a prediction that would equal the truth.

```python
check_dealt(f03, None, past, paths, says)          # walk 1's deal: the model's paths, as the text says
assert b16.expects == "reed" and b16.setting.nokick and not b16.marked
assert not np.allclose(expected, truth)             # the rule `reed`: refuted
```

The round's direction edits (D1–D8) and caption renames (C1–C7) are pinned with `P.check_edits`:
each new text stands once in its section of both files, its old text nowhere. Each clearing (the
pockets tipped before a walk) is found in the paragraph the frame needs with `P.where`. A placed
phrase is asserted on the drawn extent with a positive gap: the high road's token "a little ahead"
(D1); the peg "midway between the two spreaders"; the reed token "carried over the bar" (D4), its
lift arc passing above the bar and under the rowan's brass; the wedge "just short of the bar"; the
flip marks clear of the board's edge, the foot spreader and the neighbouring cups. Each section's
log lines are asserted verbatim: they are the build log's dump, all 145 lines, which `describe`
reproduces byte for byte for both dialogue files.

That these checks bite was shown two ways.

- **The transcription against `texlua`.** 60,000 calls, random and aimed (20,000 of them biased
  to well-formed calls, 6,159 with marks), were run through the Lua itself (TeX stubbed) and
  through the transcription; the faults, the log lines, the strips' expansions and the marks'
  coordinates agree on every one. Of the Lua's 60 `tex.error` sites, 59 are raised. One is a guard
  that no input reaches: "at most four fit a line (from=off)", since a strip has at most four
  views. `TEX_SITES` lists the sites in Lua line order and `P.check_sites()` (run by
  `SixteenRoads.py`) asserts the list is the Lua's own, so removing a `tex.error` breaks the
  chapter's check.
- **Mutations** of scratch copies of the dialogue, the Lua and the transcription: a wrong frame (walk
  1 drawn with the peg; a strip started at the wrong moment; the wedge from a silver reed), a
  dropped mark, a pin drawn without a kick, a flip on an unturned token, a like pair dealt side by
  side, a prediction equal to the truth, a reverted direction, a removed clearing, a restored old
  caption and a removed `tex.error` all fail a file. A change to the Lua's drawing alone (a mark
  dropped, a pile spread) is caught by the `texlua` comparison and by the build's dump against
  `describe`.

**Pins.** `props_ch05.check_lua()` (run when the module is imported, unless `DFP_REPIN` is set
to re-pin, and again as the first line of every file's `pictures()`) holds `pdf/props/ch05.lua`
and `ch05.tex` to the text the transcription was proved against with texlua: `LUA_PINS` hashes
the Lua's 85 top-level chunks (comments left out, white space closed up outside string literals,
so a blank added inside the TikZ a literal writes is caught), `TEX_PINS` the `.tex`'s 40
colours, styles and macros, and `COMMENT_PINS` the Lua's two comments that quote the dialogue —
the wrong rules Achilles reasons by ("I *said* the chances only add", "The bar wrote its mark on
the reed — the lane I asked about — …") and how the dealt tokens lie in a cup ("silver upon
silver, and they pile", "silver beside slate") — word for word, each with the phrases it quotes,
every one still spoken in `dialog.md` (a spaced ` -- ` in a Lua comment stands for the
dialogue's em dash). A drawing-only or style-only edit (a spreader's lobes, a token's side
colour, a ghost's dotting) then fails all five files, naming the chunk, until the transcription
is proved again and re-pinned (`print(P.lua_pins())` with `DFP_REPIN=1`; the module's
docstring has the line); a plain comment or white-space edit passes, and a `--[[` block comment, which would hide code
from the line-by-line hashes, is refused outright. The re-proof before
pinning (2026-11-14): the dialogue's 36 calls, 74 aimed calls (59 of the 60 sites; the 60th
cannot fire), 305,394 exhaustive, 30,000 fuzzed and 6,692 edge calls with 24,169 model
probes, byte for byte with 0 raw Lua errors, and the chapter build's 145 dump lines equal to
`describe`'s. The transcription now lower-cases as the Lua does (A–Z only), so an accented
capital in a setting reads alike in both.

The first file watches two roads add or cancel; the second gives the Stage-A language for
"both roads at once"; the third finds the sign that jumps from one wire to another; the fourth
spends that sign on a single certain question; the last widens the board to sixteen roads and
watches eight cancel eight.

---

## 1. Two Roads — `Ch5/TwoRoads.py`

*"Do the amplitudes add — and can they cancel?"* The Cat walks the willow lane both ways; with
no phase the two roads to the home pocket pile up and the walker comes home, and a half-turn of
phase makes them cancel and turns the walker away. The file checks the two facts of §5.1:
$HH=I$ (the roads pile, exact — no stray scalar) and $HZH=X$ (the roads cancel and flip). On
$\ket0$ the first gives $\ket0$, the second $\ket1$. In the sketchbook two Hadamard edges through
a phase-free node are a bare wire, and two Hadamard edges around a $Z$ are a bit-flip; the
negatives rule out mistaking $HH$ for a flip or $HZH$ for the identity.

```python
assert np.allclose(H @ H, I2)                    # HH = I, exact
assert np.allclose(H @ Z @ H, X)                 # HZH = X
assert same_picture(two_mirrors(0), bare_wire())         # HH = I, as pictures
assert same_picture(two_mirrors(1), gate("X", 1))        # HZH = X, as pictures
assert not same_picture(two_mirrors(1), bare_wire())     # not the identity
```

*Pictures* (First Challenge, blocks `[1, 1, 2]`, frames F01–F08). P1.1 is the keyed board as
set — the willow alone in play with both spreaders, no token, peg, bar or reed token, all five lanes
and the home/away labels drawn. Block 1.2 is walk 1 on the bare lane, one `\conduitwalk` of three
frames: the two tokens set out silver at the head, the high road's a little ahead (D1); dealt —
the home cup silver upon silver, piled, the away cup silver beside slate, the high road's token
turned by the deal and flipped; and the end, the away pair lifted out, home heaped with two silver:
$2HH\ket0=2\ket0$. Block 1.3 is the peg walk. First Achilles's prediction, the rule `add`: the peg
passed untouched, the chances only adding, so the same home heap as walk 1. Then the
`\conduitwalk[from=off]`: past the peg — the ghosts at the head as walk 1 set them out, the high
road's token turned slate and flipped, the low road's not; dealt — home silver beside slate, away
silver upon silver; the end, home bare, away two silver: $2HZH\ket0=2\ket1$. The prediction differs
from that end. Between the walks the Tortoise tips the tokens out, and that clearing is pinned
between the two blocks.

```python
before, past, paths, net = walk_model(Z)
check_past(f06, f02.tokens, before, past)          # ghosts as walk 1 set them out; a flip on high only
check_dealt(f07, f06.tokens, past, paths, says)    # home side by side, away piled
check_end(f08, f07, paths, net)                    # 2 HZH|0> = 2|1>: away
assert b13e.heaps != f08.heaps                     # Achilles's `add`: refuted
```

## 2. A Sum of Pictures — `Ch5/SumOfPictures.py`

*"Where did the other road go — can you draw the both-at-once?"* The Tortoise refuses to fork the
wire and instead sums two one-road tiles in the pocket. The file checks the new picture element
of §5.2 with **generic random complex** boxes (non-symmetric, non-real, so a transpose slip is
caught): a box on a state is the weighted list of output values, $A\ket{x}=\sum_y A_{yx}\ket{y}$,
and two boxes in a row are a sum over the middle value, the roads. In the sketchbook the two-box
picture evaluates to the matrix product $BA$ (a translation check — no Stage-A rewrite derives it),
and the sum of pictures itself is a genuine two-graph equality: $H\ket0$ *is* the sum of a
$\ket0$-road and a $\ket1$-road, and $H\ket1$ carries a minus on its $\ket1$-road.

```python
assert np.allclose(A @ ket(x), sum(A[y, x] * ket(y) for y in (0, 1)))     # the column, road by road
assert not np.allclose(A @ ket(x), sum(A[x, y] * ket(y) for y in (0, 1))) # not the transpose
assert np.allclose(matrix_of(P), Bmat @ Amat)                # two boxes = the road-sum
assert np.allclose(matrix_of(h_on(0)), (1 / np.sqrt(2)) * (k0 + k1))      # a sum of whole pictures
assert not same_picture(h_on(1), superposed(0))              # the |1>-road is signed minus
```

*Pictures* (Second Challenge, blocks `[1, 3]`, frames F09–F12). P2.1 is the keyed pair of tiles
on the table, both silver, named low road and high road. F10 reads silver over silver: a doubled
stack at the pile, net $+2$. F11 is drawn on the faces *before* the turn (`laid; turn=high` on
silver, silver): the high tile lifted, turned slate and laid back where it lay, so both tiles stay
at the pile, the turned one with its halo and a flip mark — $Z$ on the high road. F12 reads that
pair with `marks=on`: it lifts out from the pile, its ghosts where F11 laid it, the pocket bare,
net $0$. Nothing was walked, and the file asserts the tiles give the two walks' home pockets
exactly. For all four pairs of faces the transcribed `read` is
$s_{\rm low}+s_{\rm high}=2\bra0H\,\mathrm{diag}(s)\,H\ket0$, a turn is $Z$ on its road, and the
faces of $\sqrt2\,H\ket0$ and $\sqrt2\,H\ket1$ are the tiles (silver, silver) and (silver,
slate) — row 33.

```python
assert t23.before == t22.faces                           # drawn on the faces before the turn
assert t23.turned == "high" and t23.faces == faces(turned)   # turned = Z @ s: Z on the high road
assert t23.places == t22.places == PILE_PLACES                  # laid back where it lay
assert t24.ghosts == tuple((f,) + pl for f, pl in zip(t23.faces, t23.places))   # lifted out
assert tile_net(t22) == home_net(walk1[-1]) == 2                # a pile, as walk 1's home pocket
```

## 3. The Mark on the Other Wire — `Ch5/Kickback.py`

*"The mark lands on the road I asked from, not the road I asked about."* The reed token is wedged
two-faced ($\ket-$) and the bar laid across; where a road's value is $1$ the turn jumps to the
willow token and flips its face. The file checks phase kickback for **all four** functions
$f:\{0,1\}\to\{0,1\}$: each oracle is built as the reversible permutation
$\ket{x}\ket{y}\mapsto\ket{x}\ket{y\oplus f(x)}$, and on $\ket{x}\ket-$ it comes back
$(-1)^{f(x)}\ket{x}\ket-$ — the sign on the query, the answer register returned to $\ket-$ exactly.
The dialogue's two false starts are the negatives: a *definite* answer $\ket0$ puts the mark on the
answer wire ($\ket{x}\ket{f(x)}$), and $\ket+$ takes no mark at all. The sketchbook compares the
oracle picture with the signed picture for all eight cases (four $f$, two $x$), exact.

```python
assert np.allclose(U @ kron(ket(x), KMINUS), sign * kron(ket(x), KMINUS))  # kickback
assert np.allclose(U @ kron(ket(x), k0), kron(ket(x), ket(f[x])))          # definite answer: mark on the answer
assert np.allclose(U @ kron(ket(x), KPLUS), kron(ket(x), KPLUS))           # |+>: no mark at all
assert same_picture(kicked(f, x, signed=False), kicked(f, x, signed=True)) # exact, sign included
```

*Pictures* (Third Challenge, blocks `[2, 3]`, frames F13–F17). The bar is told aloud as
`willow high`, read through the transcription as the function `id`, $(f(0),f(1))=(0,1)$. F13 is
drawn on the setting before its step (two roads, $H\ket0$): the head spreader lifted off, its ghost
at the head, one token walked by hand on the high road, $\ket1$. F14, past the bar with the reed's
token lying flat: the reed turned silver to slate, with a halo and a flip, the willow's token
unmarked — $U_f\ket1\ket0=\ket1\ket1$. F15 wedges the reed's token *as it lay* (slate, from F14):
its ghost in the cup, the lift arc over the bar to the wedge, the token stood two-faced just short
of the bar, $\ket-$. F16 is Achilles's prediction, the rule `reed`: the bar's turn spent on the
reed, the willow's token silver. F17 is the truth: the willow's token turned slate (its ghost at the
head, a flip), the reed token held still (the pin ring), no arc for the kick —
$U_f\ket1\ket-=-\ket1\ket-$. The transcription is checked for every one-lane rule and both roads:
two-faced, the token past the bar is $(-1)^{f(x)}$; flat, the road's token stays silver and the
reed's face is the value $U_f$ writes; a wedge from either face gives $\ket-$.

```python
assert np.allclose(U @ kron(q, ket(REED_VALUE[b13.reed])), kron(q, ket(REED_VALUE[b14.reed])))
assert P.parse_setting(p33.args[0]).reed == b14.reed == "slate"   # wedged as it lay
assert np.allclose(truth, sign * kron(q, KMINUS))                 # U_f|1>|-> = -|1>|->
assert not np.allclose(expected, truth)                           # Achilles's `reed`: refuted
assert b17.halos == [] and b17.arcs == []                         # the reed unmoved; no arc
```

## 4. The One Question — `Ch5/Deutsch.py`   *(The Theorem, Twice)*

*"I never learned either value, only whether they agree."* One walk, one consultation of the bar,
settles same-or-different. The file checks Deutsch's algorithm for all four $f$:
$(H\otimes I)U_f(H\otimes I)\ket0\ket-=(-1)^{f(0)}\ket{f(0)\oplus f(1)}\ket-$ — **exactly, the
global sign $(-1)^{f(0)}$ included** (dropping it is a falsehood whenever $f(0)=1$, asserted as a
negative). The query reads $\ket0$ for constant and $\ket1$ for balanced, with certainty. Two
honest side-checks show why one *classical* evaluation cannot do it — the four $f$ grouped by
$f(0)$ each hold one constant and one balanced — and why a quantum peek cannot either: with no
second $H$, reading the query is a fair coin for every $f$. This is the deterministic one-query
algorithm of Cleve, Ekert, Macchiavello and Mosca (1997), with Alain Tapp independent; Deutsch's
1985 original was probabilistic only. The sketchbook draws the whole two-wire circuit and compares
it, sign and all, with the target state.

```python
assert np.allclose(out, sign * kron(ket(a), KMINUS))         # exact, global sign included
if f[0] == 1:
    assert not np.allclose(out, kron(ket(a), KMINUS))        # the sign is really there
assert same_picture(circuit_out(f), target(a, sign))         # the picture, sign and all
```

*Pictures* (Fourth Challenge, blocks `[1, 1]`, frames F18–F19, no marks: ruling (d)). Only
close-ups of the willow's home cup: the away cup is never pictured, and no picture of the walk with
both spreaders is `past`, `end`, `pockets` or `home` as they fell — any of them would show the low
road's answer, which Achilles says he cannot tell. Each walk starts on emptied pockets (D5, D6, D7).
P4.1, keyed "home", is the plod: the head lifted, the low road walked by hand under the hidden rule
`never` (`zero`), one silver token $=\sqrt2\,(\bra0\bra-)(H\otimes I)U_f\ket0\ket-=+1$ — the low
road's answer. P4.2 is the one walk under `willow low` (`not`), settled: home bare, net $0$, the
$\ket0\ket-$ amplitude of Deutsch's output $-\ket1\ket-$. The tosses between the two pictures are
drawn by no picture. The two pictured rules differ from each other and from the Third Challenge's.
The transcription is checked for every one-lane rule: settled pockets
$2(\bra{y}\bra-)(H\otimes I)U_f(H\otimes I)\ket0\ket-$, a hand-walked road's token
$\sqrt2(\bra{y}\bra-)(H\otimes I)U_f\ket{x}\ket-$.

```python
assert f_of(b41) == FOURS["zero"] and f_of(b42) == FOURS["not"]
assert b41.cups == {"home": ["silver"]}                          # the low road's answer, +1
assert b42.heaps == {"home": []} and P.net(b42.cups["home"]) == 0   # home bare
```

## 5. The Sixteen Roads — `Ch5/SixteenRoads.py`   *(the master diagram)*

*"are the roads all of one kind, or split?"* The Cat walks all sixteen roads; under a balanced bar
the home pocket empties. The file checks the Deutsch–Jozsa value of §5.5: the closed five-wire
board equals $\tfrac1{16}\sum_x(-1)^{f(x)}$ — $\pm1$ for a constant $f$, $0$ for a balanced one.
The balanced example ($f(x)=$ the top query bit) is asserted **genuinely balanced**, eight and
eight, before the zero is trusted, and the closed value is checked against the formula directly.
The sketchbook builds the closed diagram — four query wires $\ket0\to H\to U_f\to H\to\bra0$ and one
answer wire $\ket-\to U_f\to\bra-$ — and reads it as a single number, evaluated only through
`tools/dfp.py` (which forces the correct contraction; pyzx's default is wrong on some closed
diagrams). The negative shows why $\ket-$ must stay normalized at both ends: an unnormalized answer
wire multiplies the constant value by $\braket{-|-}=2$.

```python
assert tally == [8, 8]                                       # genuinely balanced before we trust the 0
assert np.isclose(closed, sum((-1) ** f(x) for x in range(16)) / 16)   # the road-sum formula
val = matrix_of(closed_board(name)).ravel()[0]               # the closed picture is a number
assert np.isclose(val, want)                                 # exact: 1, -1, or 0
assert np.isclose(stray, 2 * VALUE[name])                    # unnormalized |->: a stray factor of 2
```

*Pictures* (Fifth Challenge, blocks `[1, 2]`, frames F20–F22, no marks: ruling (e)). P5.1 is the
whole board set, after the willow's pockets are tipped (D8): the four upper lanes in play with both
spreaders, the reed two-faced, the bar hidden with the rule `willow high` — read through the
transcription, exactly this file's balanced $f(x)=$ `(x >> 3) & 1`, eight and eight. F21 is the
whole board's home pocket as the tokens fell, road by road in binary counting with the willow the
top digit: eight silver, then eight slate, the list $(-1)^{f(x)}$; F22 is the same pocket settled,
all lifted out, net $0=16\times$ the closed master value. No picture is `end`, and none follows the
block — the Cat's measurement ends the dialogue. The transcription is checked on four lanes for
seven rules: the home list is $(-1)^{f(x)}$ in road order, its net is $16\times$ the closed board,
and the whole board's `end` is drawn exactly when the blackboard puts the walk in one definite
pocket per lane.

This file also checks the dialogue as a whole: eighteen calls drawing twenty-two frames, every one
inside a challenge section and none refused by the macro; three, two, two, two and two moments; the
keyed rule (the first board, the first tiles and the first close-up are keyed, and nothing else; a
prediction is never judged); five step strips (blocks 1.2, 1.3, 2.2, 3.1, 3.2), two of them
`\conduitwalk`; eleven marked frames, F03, F04, F06, F07, F08, F11, F12, F13, F14, F15, F17; two
predictions, `add` and `reed`, unkeyed, unmarked and stepless; one pin ring, F17's; the 145 dump
lines with a head line per frame; the bar's rule in every frame, as fixed by the identity card;
and `TEX_SITES` against the Lua's own `tex.error` sites.

```python
total, frames, starred = P.check_dialogue(DIALOGUE, ORDINALS)
assert total == 18 and len(frames) == 22                     # calls; frames F01-F22
assert [n for n, (_, _, pic, _) in enumerate(frames, 1) if pic.marked] == \
    [3, 4, 6, 7, 8, 11, 12, 13, 14, 15, 17]
assert preds == [("First", "add"), ("Third", "reed")]
assert b52.cups["home"] == [P.face((-1) ** FS["balanced"](x)) for x in range(16)]   # eight and eight
assert P.net(b53.cups["home"]) == 0 and np.isclose(16 * closed, 0)  # all lifted out
P.check_sites()
```
