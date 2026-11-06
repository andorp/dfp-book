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
`\walnutstring`, `\beadstring`, `\crabledger`). Their Lua computes part of what is drawn: the
string pulled (far bends minus near bends), the bead's picture at each place on the string,
the looking-glass bead, the ledger's right-hand column. `tools/props_ch06.py` transcribes those
rules, and each file reads the pictures of its own section of `dialog.md` through
`tools/dfp_props.py` (never a copy of them), checks their layout, and asserts every picture
against the state the dialogue is in at that moment and against the mathematics: the pulled
string is the yank, the row pulled is the chain, the slide and the glass are the transpose and
the dagger, the ledger's agreement is the pair's joint table. An edit to the dialogue that
breaks a picture breaks a file here.

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

*Pictures* (four): the tray with the first lid up; box 1 spent and box 2 held apart beside
tonight's page, one word waiting in fresh ink; the leaf Achilles reads across, every line the
same word twice. The file checks Achilles's spoken reading against the drawn pages, and that
two separate coins would write a disagreeing line half the time while the pair never does.

```python
assert (tonight.left, tonight.right, tonight.ink) == ("high", None, "fresh")
assert all(l.right == l.left for l in c14.lines)
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

*Pictures* (three): one page under the blotter on the right, on the left, and lifted. The
right column shown alone is the left column's run, computed; Achilles's tally ("Five highs and
three lows on this page") is the drawn column. The Lua's rule — right = left, by either door —
is checked against the pair's joint table: nothing off its diagonal in the lid's basis or the
door's, and each column alone a fair coin.

```python
assert possible == {(l.left, l.right) for l in page.lines}          # what the Lua writes
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

*Pictures* (four): the hairpin (a pair); out, back, out; the same pulled straight; the other
way about. The file reads each laid string as a map — a bend on the side time starts from is a
cup, on the other side a cap — so the out-back-out string is one chirality of the yank and the
other-way-about string the other; the pull rule (far bends minus near bends: none left) gives
the straight string, and the same map.

```python
assert np.allclose(string_map(w32.string, "near"), kron(CAP, I2) @ kron(I2, CUP))   # = I2
assert w33.string.bends == () and np.allclose(string_map(w33.string, "near"), I2)
assert np.allclose(string_map(w31.string, "near").ravel(), CUP)                    # a pair
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

*Pictures* (three): the bead going in (point to the bend, notch to the window, as the napkin
records it), coming out on the near leg (point right, notch towards the company), and the
looking-glass bead behind the Crab's glass (point right, notch still towards the window). The
file checks that sliding and the glass generate four poses and that these match $A$, $A^T$,
$A^\dagger$, $\bar A$, the slide standing for the transpose and the glass for the dagger (the
flip "on its face" is the conjugate), and that a pose keeps its handedness exactly when its
matrix is not conjugated.

```python
assert b42.bead == (RIGHT, DOWN) and P.handedness(b42.bead) == P.handedness(b41.bead)
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

*Pictures* (two, side by side): the key at work on the straight string, and the hairpin with
the key at the same spot, on the leg to the left walnut box. The words say the note comes in
at the string's left end, and that the *right* end is carried up to the left walnut box while
the left end swings round to the right one — so the key's output side faces box 3 and the bend
lies on its input side: the drawn pair is $(A\otimes I)\,$cup, the key on the top leg, whose
amplitudes are the key's own entries, not the transposed ones. Bending the left end back as
the input and pulling gives the key again.

```python
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

*Pictures* (four): the row laid (three bends at the pairs, two turns at the stations), both
ends drawn in (one bend, box 1 to box 6), and the last note beside tonight's page completed.
The row read as a map is the chain before the pull and the single cup after it; the pulled
row and the Third's walnut hairpin are the same map ("the hairpin you laid for the walnut
boxes, only wider"). Tonight's page is the First's page with its lone word completed, and the
first note and the last are the two boxes of one pair. This file also checks the dialogue's
pictures as a whole: twenty, in six sections; the keyed form first for each macro (and again
for the row, whose labels differ from the tray's); no macro fault, no forbidden view, no
caption naming what it counts; nothing drawn in the relay's opening lines or after the last
note.

```python
assert np.allclose(string_map(r61.string, "far"), kron(I2, CAP, CAP, I2) @ kron(CUP, CUP, CUP))
assert [(b.side, b.legs) for b in r62.string.bends] == [("far", (1, 6))]
assert np.allclose(string_map(r62.string, "far").ravel(), CUP)
assert DIALOGUE.check_whole(ORDINALS) == 20
```
