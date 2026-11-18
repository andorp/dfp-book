# Chapter 6 — Entanglement: the companion code

Six challenges in the dialogue, six files here. Each file checks its claim twice, as the
chapter proves its theorem twice: `blackboard()` with NumPy vectors and matrices,
`sketchbook()` with pictures built in PyZX and compared as exact tensors — every number
included, because no equation in this chapter holds only "up to a scalar".

Run one file, or the whole chapter, from the repository root:

```bash
.venv/bin/python chapter06/Code/Ch6/Sliding.py
bash check-code.sh 6          # must end with: 6 files, 0 failed
```

Each file prints one line per strand and exits non-zero if any `assert` fails. Running the
files is this book's analogue of type-checking.

**Conventions, asserted where they are used.** The cup $=\ket{00}+\ket{11}$ and the cap
$=\bra{00}+\bra{11}$ are *unnormalized* (`np.linalg.norm(CUP) == sqrt(2)`); the Bell state is
`BELL = CUP / sqrt(2)`. Kets are written top wire first: `kron(A, I2)` puts `A` on the top
wire. In a `Picture`, row 0 is the top wire, and `bend` is a bare bent wire — a cup or a cap
according to how it is connected. The shared plumbing is in `tools/dfp.py`; the mathematics
of each challenge is in its own file.

The first two files establish what a Bell pair *is* — one description, not two; random
parts, a definite whole — with nothing but vectors. The next three replace the vector by a
bent wire and check the three things one can do with a bend: pull it straight, slide a box
round it, bend a box into a state. The last file threads six wires with all of it and pulls
once.

**The prop pictures.** The dialogue draws its props — the Crab's boxes on their tray and in
their row, his ledger, the Tortoise's string at the walnut pair, the bead with the napkin and
the glass, the tuning key — with four macros from `pdf/props/ch06.tex` (`\musicboxes`,
`\walnutstring`, `\beadstring`, `\crabledger`) and their four step strips (`\boxsteps`,
`\walnutsteps`, `\beadsteps`, `\ledgersteps`). Their Lua computes part of what is drawn: the
string laid pair by pair and pulled (far bends minus near bends), the bead's picture at each
place on the string, the looking-glass bead, the ledger's right-hand column — and, since the
"Show the process" round, each *step*: a frame names the step (`open=`, `close=`, `keep=`,
`slide=`, `lift=`, `bend=`, `pull=`, `lay=`, `write=`) on the state before it, and the Lua
draws the state after, a pale *ghost* of the moving part where it was, a curved *arc* for its
motion, and a *ring* on the part held fast. Two frames are *predictions* (`expects=`): the
state a named wrong rule gives, drawn pale beside the true one. In a starred strip the frames
after the keyed one keep its room (`keyroom=on`, which the strip adds): they draw the keyed
labels as vertical phantoms, so that every frame stands as high as the keyed one and the
captions stand level.

`tools/props_ch06.py` transcribes all of it except the drawing: every option and every step,
every `tex.error` of the Lua (89 sites, in Lua order, each matched by `check_sites()` against
the Lua source), the marks' geometry, and the log line the build writes for every picture —
the 158 lines of a real build, byte for byte. It was proved against `texlua` on the dialogue's
calls and on thousands of random and aimed ones. `check_lua()` pins the Lua itself, chunk by
chunk (comments stripped), and `pdf/props/ch06.tex` with it — its colours, its mark, label
and caption styles, its macros' expansion — so that a change to the drawing or the styles
alone, which no log line shows, breaks every file until the transcription is looked at
again (then renew `LUA_PINS` and `TEX_PINS` from `lua_pins()` and `tex_pins()`).

Each file reads the pictures of its own section of `dialog.md` (and of `dialog.raw.md`, for the
anchors and the directions edited this round) through `tools/dfp_props.py`, never a copy of
them, and asserts, frame by frame and in order: the state each frame draws against the file's
own model of that moment; each ghost at the state before the step; each arc a curve from the
moving part's old place to its new one, clear of every part it does not pass; each ring on the
part the words hold fast; each prediction equal to its named rule and different from the true
frame beside it; and the section's log, line for line. The pictures are then checked against
the mathematics: the pulled string is the yank, the row pulled is the chain, the slide and the
glass are the transpose and the dagger, the ledger's agreement is the pair's joint table. An
edit to the dialogue that breaks a picture breaks a file here.

---

## 1. Two Recipe Cards — `Ch6/NotAProduct.py`

Achilles writes one card per box; coins played from the two cards disagree half the time, the
ledger never does. "Two cards won't do it. It needs one card for the pair." The file checks
the determinant test of §6.1: a product state has $c_{00}c_{11}-c_{01}c_{10}=0$, the Bell
state has $\tfrac12$. In the sketchbook, two cards are a *disconnected* picture, and the
circuit that prepares the pair ($H$ on the top wire, then CNOT, on $\ket{00}$) equals
$\tfrac1{\sqrt2}$ times the bent wire exactly. The negative asserts rule out the disconnected
picture, and the bent wire without its $\tfrac1{\sqrt2}$.

```python
assert np.isclose(det_of(np.kron(alpha, beta)), 0)      # two cards
assert np.isclose(det_of(BELL), 1 / 2)                  # one card for the pair
assert not same_picture(cards, bent_wire())
assert same_picture(prepared_pair(), bell)              # bell = bent wire with 1/sqrt(2)
```

*Pictures* (six frames, in three moments): a strip of three frames — the tray shut and full, the
first lid up, the lid down again (box 1 spent) — then box 2 kept apart beside the tray, and
tonight's page with one word waiting in fresh ink; then the leaf Achilles reads across, every
line the same word twice. The lid's arc runs into the gap between the pair and back the same
way; the closing frame's ghost is the raised lid; the kept box's ghost stands in its tray place,
shut and full, and its arc lifts it over the tray's right rim, clear of every other box. The
file checks Achilles's spoken reading against the drawn pages, and that two separate coins
would write a disagreeing line half the time while the pair never does.

```python
assert f03.arcs[0].pts == tuple(reversed(f02.arcs[0].pts))                 # the same swing, back
assert (ghost.what, ghost.state) == ("box", f03.state(2))                  # shut and full, as it was
assert min(xs) < P.TRAY[2] < max(xs)                                       # over the right rim
assert (tonight.left, tonight.right, tonight.ink) == ("high", None, "fresh")
assert np.isclose(two_cards_disagree(), 1 / 2) and np.isclose(the_pair_disagrees(), 0)
```

## 2. The Ledger — `Ch6/Marginals.py`

The blotter over one column: a coin toss, by lid or by side door. The blotter lifted: the
columns are identical. The file computes outcome probabilities straight from amplitudes (no
density matrices): asking the top wire for $b$ leaves $\ket{b^*}/\sqrt2$ on the bottom wire,
probability $\tfrac12$, for a random complex $b$. The negative assert rules out
$\ket{b}/\sqrt2$ — the conjugate is really there, and it is why the halves agree in a *real*
basis and not in every basis. In the sketchbook an effect plugged into the top leg of a bend
equals the same node as a state on the bottom wire; PyZX does not conjugate a node it turns
round, and that is exactly the blackboard's $b\mapsto b^*$. Nothing here refutes a note
written inside both boxes beforehand; that is Chapter 7.

```python
assert np.allclose(left_behind, b.conj() / np.sqrt(2))
assert not np.allclose(left_behind, b / np.sqrt(2))
assert np.allclose(joint_table(side_door), [1 / 2, 0, 0, 1 / 2])
assert same_picture(effect_on_top_leg(a), state_on_bottom_wire(a))
```

*Pictures* (three frames): one page under the blotter on the right; the blotter slid to the
left, its ghost on the right and its arc running flat over the top of the page from column to
column; the blotter lifted off, its ghost on the left and no arc (a lift draws none). The slide
is a flat arc and the First's lift a steep one, so the two motions read differently. The right
column shown alone is the left column's run, computed; Achilles's tally ("Five highs and three
lows on this page") is the drawn column. The Lua's rule — right = left, by either door — is
checked against the pair's joint table: nothing off its diagonal in the lid's basis or the
door's, and each column alone a fair coin.

```python
assert (g22.what, g22.state, g22.place) == ("blotter", a21.blotter, P.blotter_rect(a21.blotter))
assert min(y for _, y in pts) - P.SWING_W / 2 > ghost_top > P.PAGE[3]
assert rise_over_span(slide) < 0.15 and rise_over_span(lift) > 0.45
assert a22.column("right") == a21.column("left")
```

## 3. The String — `Ch6/Yanking.py`

Tortoise lays the string out, back and out again, and says "Pull." The file checks the
yanking equation in both chiralities, exactly, and the reason for the book's normalization:
with the normalized Bell state and its bra, the same bent wire is *half* a wire.

```python
assert np.allclose(kron(CAP, I2) @ kron(I2, CUP), I2)
assert np.allclose(kron(I2, CAP) @ kron(CUP, I2), I2)
assert np.allclose(kron(bell_bra, I2) @ kron(I2, BELL), I2 / 2)     # and not I2
assert same_picture(z_bend, straight()) and same_picture(s_bend, straight())
```

*Pictures* (six frames, in two strips and two single pictures): the straight string in front of
the left walnut box, then bent into the hairpin (a pair) — the straight string's ghost, the right
end's arc across to the left box, the left end's arc down, round and up to the right box, each
clear of the boxes and of the strands it does not pass; out, back, out; the same pulled half-way
(both bends' arcs towards the middle) and then by its ends, drawn apart; the other way about.
The pulled frame is the only one in the dialogue whose ends are drawn apart: its ends lie at or
beyond the laid ends along the line, the line keeps clear of both box bodies, and the meeting
point the arcs run to is the line's own crossing. The file reads each laid string as a map — a
bend on the side time starts from is a cup, on the other side a cap — so the out-back-out
string is one chirality of the yank and the other-way-about string the other; the pull rule
(far bends minus near bends: none left) gives the straight string, and the same map.

```python
assert np.allclose(string_map(w31.string, "near").ravel(), CUP)      # "There is a pair."
assert np.allclose(string_map(w33.string, "near"), string_map(w32.string, "near"))
assert w33.apart and w33.line == P.SZAPART["S"]
assert min(P.gap_rect(line, *d.body) for _, d in w33.drawn) > 0.11     # its centre (ch06.lua)
```

## 4. The Bead — `Ch6/Sliding.py`   *(The Theorem, Twice)*

"It went in point first. Look how it comes out." The bead arrives turned end for end — and
the Crab's mirror shows that turned is not mirrored. On the blackboard `A` is a random
complex matrix, asserted non-symmetric, non-real and non-unitary, so that the transpose, the
dagger, the conjugate and `A` itself are four different matrices; only the transpose passes.
The file also checks the picture of $A^T$ (the box with both ends bent round) and the
four-row substitution that the sketchbook proof of §6.7 starts from. In the sketchbook the
generic box is a chain of three phase nodes; turned end for end is the chain read backwards,
the mirror image is that with the phases negated.

```python
lhs = kron(A, I2) @ CUP
assert np.allclose(lhs, kron(I2, A.T) @ CUP)                # transpose: yes
assert not np.allclose(lhs, kron(I2, A.conj().T) @ CUP)     # dagger: no
assert not np.allclose(lhs, kron(I2, A.conj()) @ CUP)       # conjugate: no
assert not np.allclose(lhs, kron(I2, A) @ CUP)              # unchanged: no
assert same_picture(on_top, box_on_bend(1, turned(BEAD)))
assert not same_picture(on_top, box_on_bend(1, mirrored(BEAD)))
```

*Pictures* (five frames): the bead going in (point to the bend, notch to the window, as the
napkin records it); Achilles's prediction, pale, on the near leg by the *unchanged* rule; the
slide itself, a strip of two frames — in the bend (a quarter turn) and out on the near leg (point
right, notch towards the company: another quarter turn), each with the previous bead as its ghost
and an arc that keeps to the string, outside the bend, clear of the napkin and the boxes; and the
looking-glass bead behind the Crab's glass (point right, notch still towards the window). Every
frame on the cloth draws the walnut pair where the Crab's direction before them (D12) leaves it:
the right box drawn back towards the window, the left eased to the left and towards the company,
the string bent round it, both boxes clear of the string. The
prediction differs from the truth in both the point and the notch. The file checks that sliding
and the glass generate four poses and that these match $A$, $A^T$, $A^\dagger$, $\bar A$, the
slide standing for the transpose and the glass for the dagger (the flip "on its face" is the
conjugate), and that a pose keeps its handedness exactly when its matrix is not conjugated.

```python
assert b42e.bead == P.rule_unchanged(1, -1) == b41.bead
assert b42.bead == (RIGHT, DOWN) == quarter_turn(b42b.bead)
assert b42e.bead[0] != b42.bead[0] and b42e.bead[1] != b42.bead[1]   # the prediction, refuted
assert b43.image == (RIGHT, UP) and b43.image != b42.bead
assert (q1 == q2) == bool(np.allclose(M1, M2))        # every pair of move sequences
```

## 5. A Key in the Post — `Ch6/Bending.py`

The Crab's key, too delicate to post, becomes a pair of boxes when the string is bent, and a
key again when it is bent back. "...Oh. Every key is a pair." With the box on the **top** leg
the state's four amplitudes are the matrix's four entries; the file asserts the pitfall too —
on the bottom leg they are the transpose's entries. Recovery is one cap and one yank. Product
states correspond to rank-one boxes: the determinant test of the first challenge again.

```python
psi = kron(A, I2) @ CUP
assert np.allclose(psi.reshape(2, 2), A)                            # top leg
assert np.allclose((kron(I2, A) @ CUP).reshape(2, 2), A.T)          # bottom leg: transposed
assert np.allclose(kron(I2, CAP) @ kron(psi, I2), A)                # bend back
assert same_picture(pair_bent_back(), key())
```

*Pictures* (three frames): a strip — the key at work on the straight string, then the string
bent into a hairpin with the key at the same spot, on the leg to the left walnut box — and the
pair bent back, all three with the walnut pair set back where the Third's pictures stood it,
as the Crab's direction before them (D13) says. The bending frame carries the straight string's ghost, the two ends' arcs (clear
of the key's bow and shank) and a ring on the key, held fast ("The key has not moved a hair");
the frame bent back runs the same two arcs backwards, the key ringed again. The words say the
note comes in at the string's left end, and that the *right* end is carried up to the left
walnut box while the left end swings round to the right one — so the key's output side faces
box 3 and the bend lies on its input side: the drawn pair is $(A\otimes I)\,$cup, the key on
the top leg, whose amplitudes are the key's own entries, not the transposed ones. Bending the
left end back as the input and pulling gives the key again.

```python
assert (pin.name, pin.x, pin.y) == ("key", *w52.key_spot) and pin.r > P.KEY_BOW_R
assert back_round.pts == tuple(reversed(round_.pts))
psi = string_map(w52.string, "near", keys={w52.key_leg: (A, "finish")}).ravel()
assert np.allclose(psi, kron(A, I2) @ CUP) and np.allclose(psi.reshape(2, 2), A)
assert not np.allclose(psi.reshape(2, 2), A.T)
assert np.allclose(kron(I2, CAP) @ kron(psi, I2), A)
```

## 6. The Relay — `Ch6/Chain.py`   *(the master diagram)*

The Anteater's story: Alice, Bob, and two relay stations between them.
"They never shared a bench." Three cups, two caps, and one pull leave a single cup from the first wire to the
last — exactly, with the unnormalized cup and cap. With normalized pairs and outcomes the
same picture is `BELL / 4`: a factor $\tfrac12$ per junction, probability
$\tfrac1{16}=\tfrac14\cdot\tfrac14$ that both stations hear the chord. That is an equality
of processes and the honest price of two postselected outcomes, nothing more; what to do on
the other evenings is Chapter 10.

```python
chain = kron(I2, CAP, CAP, I2) @ kron(CUP, CUP, CUP)
assert np.allclose(chain.ravel(), CUP)
assert np.allclose(relay, BELL / 4) and np.isclose(np.linalg.norm(relay) ** 2, 1 / 16)
assert same_picture(the_chain(), one_bend())
```

*Pictures* (nine frames, in three moments): a strip of four — the string loose and bare along
the row, then laid pair by pair, each lay leaving the string before it as a ghost, each laid
part one pair from box 1 to box $2k$; Achilles's prediction (the *gaps* rule: three separate
pairs, cut at each station), pale, and beside it the pull — both ends drawn in, the walnut bend
held under a ring, the two stations' turns running up into the bend between the ghost legs they
do not pass, one bend from box 1 to box 6; then the last note and, in a ledger strip, tonight's
page completed. The row read as a map is the chain before the pull and the single cup after
it; the prediction is three cups, which the turns would have to join; the pulled row and the
Third's walnut hairpin are the same map ("the hairpin you laid for the walnut boxes, only
wider"). The first note and the last are the two boxes of one pair. This file also checks the
dialogue's pictures as a whole — 22 calls in 18 moments (at most four a section), 32 frames,
seven strips, 17 step frames, two predictions, the keyed form first for each macro (and again
for the row, whose labels differ from the tray's), the 158-line log of a real build — and that
the transcription refuses every input the Lua refuses: an aimed faulty call reaches each of the
Lua's error sites but one, which the Lua itself can never reach.

```python
assert np.allclose(string_map(r61.string, "far"), kron(I2, CAP, CAP, I2) @ kron(CUP, CUP, CUP))
assert np.allclose(kron(*[string_map(p, "far") for p in r62e.pieces]), kron(CUP, CUP, CUP))
assert [(b.side, b.legs) for b in r62.string.bends] == [("far", (1, 6))]
assert pin.name == "walnut bend" and (pin.x, pin.y) == ((P.ROWX[2] + P.ROWX[3]) / 2, P.LEGTOP)
assert len(frames) == 32 and len(log) == 158
assert hit == set(range(len(P.TEX_SITES))) - {dead}
```
