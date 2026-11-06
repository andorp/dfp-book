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

**The pictures.** The dialogue draws the Tortoise's conduit-board and her two one-road tiles
with two prop macros from `pdf/props/ch05.tex` — `\conduitboard[view]{setting}{caption}` and
`\roadtiles[view]{low face, high face}{caption}` — seventeen pictures in eleven blocks. The Lua
behind them (`pdf/props/ch05.lua`) *computes* what each picture shows: the roads (one per choice of
values over the lanes in play, willow first, low before high), each road's token (set out silver,
turned by the peg on a high road, turned by the bar's kick where the rule acts and the reed token
is two-faced — or, the reed token lying flat, the reed's own token turned), the deal into the
pockets, the settling, and the tiles' `read`. `tools/props_ch05.py` transcribes those rules once,
for all five files; `tools/dfp_props.py` reads each challenge's pictures straight out of
`chapter05/dialog.md` (never copied), checks the block sizes, the word-only captions, and that the
blocks stand identically in `dialog.raw.md`. Each file first checks the transcription itself
against the blackboard — silver $=+1$, slate $=-1$, home $=\bra0$, the two-faced reed token
$=\ket-$ closed by $\bra-$, each spreader $=H$, the bar $=U_f$ with its rule read as $f$, and a
settled pocket's net equal to the amplitude times $\sqrt2$ per spreader walked through — and then
asserts every picture's reading, so that an edit to the dialogue that breaks a picture breaks the
code.

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

*Pictures* (First Challenge, blocks `[1, 1, 2]`). P1.1 is the keyed board as set — the willow alone
in play with both spreaders, no token, peg, bar or reed token, all five lanes and the home/away
labels drawn. P1.2, the first walk, settles to home heaped with two silver and away bare: net
$(2,0)=2\,HH\ket0$. P1.3 shows the two tokens just past the peg, the low road's silver and the high
road's slate, running a little ahead — the diagonal of $Z$; P1.4 settles to home bare, away two
silver: $2\,HZH\ket0=2\ket1$. Before that, the transcription is checked with the peg in and out:
the tokens past it are $\sqrt2\,MH\ket0$ and the settled pockets $2\bra{y}HMH\ket0$, $M\in\{I,Z\}$.

```python
assert b12.heaps["willow"] == {"home": ("silver", 2), "away": None}           # 2 HH|0>
assert b13.walking == ((0, "silver", False), (1, "slate", True))             # Z on the two roads
assert np.allclose([heap_net(b14.heaps["willow"][p]) for p in ("home", "away")], arrive)  # 2 HZH|0>
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

*Pictures* (Second Challenge, blocks `[1, 3]`). P2.1 is the keyed pair of tiles on the table, both
silver, named low road and high road. P2.2 reads silver over silver: a doubled stack, net $+2$;
P2.3 lays them in the home pocket, the slate high tile over the silver low one; P2.4 reads that
pair: lifted out, the pocket bare, net $0$. Nothing was walked, and the file asserts the tiles give
P1.2's and P1.4's home pockets exactly. For all four pairs of faces the transcribed `read` is
$s_{\rm low}+s_{\rm high}=2\bra0H\,\mathrm{diag}(s)\,H\ket0$, and the faces of $\sqrt2\,H\ket0$
and $\sqrt2\,H\ket1$ are the tiles (silver, silver) and (silver, slate) — row 33.

```python
assert tile_net(t22) == home_net(walk1) == 2        # a pile, as the first walk's home pocket
assert t23.drawn == ("silver", "slate")             # silver beneath slate
assert tile_net(t24) == home_net(walk2) == 0        # the pocket bare, as the pegged walk's
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

*Pictures* (Third Challenge, blocks `[2, 2]`). Every picture has the willow's head spreader
lifted and its high road walked by hand under the bar, whose rule — told aloud — is `willow high`,
read through the transcription as the function `id`, $(f(0),f(1))=(0,1)$. P3.1–P3.2: the reed's
token lying flat, silver before the bar and slate past it, the willow's token silver throughout —
$U_f\ket1\ket0=\ket1\ket1$. P3.3–P3.4: the reed's token wedged two-faced and unchanged, the willow's
token silver before and slate past — $U_f\ket1\ket-=-\ket1\ket-$. The transcription is checked for
every one-lane rule and both roads: two-faced, the token past the bar is $(-1)^{f(x)}$; flat, the
road's token stays silver and the reed's face is the value $U_f$ writes.

```python
assert b31.reed == "silver" and b32.reed == "slate"              # the mark on the reed
assert b34.walking == ((1, "slate", False),) and b34.reed == "two-faced"   # the mark on the willow
assert np.allclose(U @ kron(q, KMINUS), sign * kron(q, KMINUS))  # U_f|1>|-> = -|1>|->
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

*Pictures* (Fourth Challenge, blocks `[1, 1]`). Only close-ups of the willow's home cup: the away
cup is never pictured, and no picture of the walk with both spreaders is `past`, `end`, `pockets`
or `home` as they fell — any of them would show the low road's answer, which Achilles says he
cannot tell. P4.1, keyed "home", is the plod: the head lifted, the low road walked by hand under
the hidden rule `never` (`zero`), one silver token $=\sqrt2\,(\bra0\bra-)(H\otimes I)U_f\ket0\ket-=+1$.
P4.2 is the one walk under `willow low` (`not`), settled: the home cup bare, net $0$, the
$\ket0\ket-$ amplitude of Deutsch's output $-\ket1\ket-$. The two pictured rules differ from each
other and from the Third Challenge's. The transcription is checked for every one-lane rule:
settled pockets $2(\bra{y}\bra-)(H\otimes I)U_f(H\otimes I)\ket0\ket-$, a hand-walked road's token
$\sqrt2(\bra{y}\bra-)(H\otimes I)U_f\ket{x}\ket-$.

```python
assert f_of(b41) == FOURS["zero"] and f_of(b42) == FOURS["not"]
assert b41.cups == {"home": ["silver"]}                          # one answer, +1
assert b42.heaps == {"home": []} and P.net(b42.cups["home"]) == 0   # the home pocket bare
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

*Pictures* (Fifth Challenge, blocks `[1, 2]`). P5.1 is the whole board set: the four upper lanes
in play with both spreaders, the reed two-faced, the bar hidden with the rule `willow high` — read
through the transcription, exactly this file's balanced $f(x)=$ `(x >> 3) & 1`, eight and eight.
P5.2 is the whole board's home pocket as the tokens fell, road by road in binary counting with the
willow the top digit: eight silver, then eight slate, the list $(-1)^{f(x)}$; P5.3 is the same
pocket settled, bare, net $0=16\times$ the closed master value. No picture is `end`, and none
follows the block — the Cat's measurement ends the dialogue. This file also checks the dialogue as
a whole: seventeen pictures, every one inside a challenge section and none faulting the macro; the
keyed rule (the first board, the first tiles and the first close-up are keyed, and nothing else);
and the bar's rule in every picture, as fixed by the identity card. The transcription is checked on
four lanes for seven rules: the home list is $(-1)^{f(x)}$ in road order, its net is $16\times$ the
closed board, and the whole board's `end` is drawn exactly when the blackboard puts the walk in one
definite pocket per lane.

```python
assert DIALOGUE.check_whole(ORDINALS) == 17                    # 4 + 4 + 4 + 2 + 3
assert b52.cups["home"] == [P.face((-1) ** FS["balanced"](x)) for x in range(16)]   # eight and eight
assert P.net(b53.cups["home"]) == 0 and np.isclose(16 * closed, 0)  # not a token left
```
