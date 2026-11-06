# Chapter 3 — *Turning the Sphere* : the companion code

Five challenges, five files. Each one is checked twice: once on Dirac's
blackboard, in NumPy, and once in Penrose's sketchbook, as pictures compared
tensor against tensor with PyZX. A file that prints two `ok` lines and exits 0
has kept both promises.

```bash
bash check-code.sh 3                                 # all five
.venv/bin/python chapter03/Code/Ch3/Mirror.py        # one of them
```

Every file also carries a third strand that belongs to the dialogue rather
than to the chapter: the Crab's ball. A position of the ball is a rotation of
ordinary space; a needle is twirled *as it lies*, so a click multiplies the
position on the right; "with the clock" is the sense seen from that needle's
own bead. Every bead reading Achilles speaks in the scene is asserted — in
both senses of the clock, so that nothing depends on which way the Crab's
collars happen to be cut.

The dialogue also *draws* the ball, in small strips of pictures made with the
`\crabball{glass}{steel}{caption}` macro of `pdf/props/ch03.tex` — twenty pictures
in all: four, four, three, four and five. Each file reads the pictures out of
its own challenge's section of `dialog.md` (from its `## … Challenge` heading
to the next heading), never copying them, and checks each one against the
position it computes for that moment: the glass bead, the steel bead, the
brass needle midway between them, glass and steel at right angles, and the
number of pictures, so that a picture edited, dropped or added is caught. The
pictures use the room's frame — $x$ towards the dresser, $y$ up, $z$ towards
the company — and the files their own; the change of frame is written out,
checked on all six named directions, and checked to be a turn rather than a
reflection, without which every "with the clock" would quietly become
"against".

```python
assert np.isclose(np.linalg.det(ROOM), 1)                 # a turn, not a reflection
assert len(pictures) == len(moments)
assert np.allclose(glass, ROOM @ R @ HOME["glass"])
assert np.allclose(steel, ROOM @ R @ HOME["steel"])
```

The dialogue draws the Tortoise's strip of tracing paper too, with
`\scorestrip[view]{score}{caption}`: the score is the strip *as drawn*
(`glass+1` is glass, one click, with the clock; `|` a gummed join), and the
view — drawn, page, round, flip — is how it now lies; `pdf/props/ch03.lua` draws
what is then seen. Thirteen strips: four, four, two, none and three. Each file
parses its section's strips with a line-for-line transcription of
`parse_score` / `view_score` from `pdf/props/ch03.lua`, so what is asserted is what the
page prints, and checks that the arrows each strip shows are exactly the
twirls the scene then plays: the Tortoise's page is glass, one click, against,
and brings the ball home; Achilles's strip round on the table is steel two,
glass two (the trap); in the Second, round and flip each lead to their own
wrong ball picture and only the page to home; in the Third the page curls
every arrow the other way yet plays to the same turn; in the Fifth the gummed
strip is, piece for piece, the strips drawn in the First, Second and Third
sections, and its page — pieces reversed, each paged — brings the evening's
score home. The ball pictures of those moments are now computed *from* the
parsed strips. The Fourth asserts that its section has no strip at all, and
`NoMirror.py` that the whole dialogue has thirteen, all inside challenge
sections.

Every strip file also checks what the views mean on generic strips, seeded
and random: the page after the strip is always the identity turn (in both
senses of the clock, with the seams carried along); round and flip leave the
ball astray more often than not; a half-turn is the same turn either way
round, which is why `pdf/props/ch03.lua` may curl a two-click arrow the other way.
Strips of half-turns only are the trap of the First Challenge: round always
brings them home, and flip does too while the needles are glass and steel,
but not once the brass needle joins them — the Sandwich's lesson.

```python
for s in (+1, -1):
    assert at_home(play(drawn_ + arrows(view_score(items, "page")), s))
assert fails["round"] > 100 and fails["flip"] > 100      # astray more often than not
assert np.allclose(play([(n, 2, +1)]), play([(n, 2, -1)]))  # half-turn: no sense
bg = parse_score("brass+2,glass+2")
assert not at_home(play(bg + view_score(bg, "flip")))       # brass: order matters
```

Two notes on the machinery. PyZX draws this with the spiders you meet in
Chapter 8; nothing here needs them, but the library speaks nothing else.
And PyZX 0.10.6's default way of working out the number a closed picture
denotes returns the wrong answer for some of the pictures below — this was
found here, checked against NumPy by hand, and worked around: the shared
`matrix_of` in `tools/dfp.py` asks for the plain contraction instead.

---

## 1. `Mirror.py` — One Turn and Its Mirror

The Tortoise plays a score, turns the strip over like a page, and plays what
she sees: the last arrow first, every arrow the other way round. The ball comes
home. The file asks what that operation is on a box, and takes care to ask it
of a box on which the three cheap answers — transpose, conjugate, dagger —
are three different boxes.

```python
A = random_matrix(rng)
assert not np.allclose(A, A.T) and not np.allclose(A, A.conj())
assert np.allclose(A.conj().T.conj().T, A)            # twice through the page
assert not np.allclose(A.conj().T, A.T)               # not the transpose
assert not np.allclose(A.conj().T, A.conj())          # not the conjugate
```

The test box is $U = (HSH)S$, which is neither real nor symmetric, and the
promise is the one the chapter names: $U^\dagger U = I = U U^\dagger$, while
$U^T U$ is not $I$ at all but $\begin{pmatrix}0&i\\i&0\end{pmatrix}$. The six
boxes of §3.4 are each put through it: $X, Y, Z, H$ are their own mirror
images and square to a bare wire; $S$ and $T$ are not, and $S^2 = Z$,
$T^2 = S$. The trap of §3.4 is here too — $S X S^\dagger = Y$, but
$S X S = iX$, which is a different box.

In the sketchbook a box with no name is a row of three nodes with three
different phases; its mirror is the same row backwards with every phase
negated. Both cheaper reflections are checked and both fail:

```python
assert same_picture(by_hand, generic.adjoint())      # reversed AND negated
assert same_picture(generic + by_hand, wire)
assert not same_picture(generic + reversed_only, wire)   # that is the transpose
assert not same_picture(generic + negated_only, wire)    # that is the conjugate
```

On the ball: a glass quarter-turn sends the steel bead to the stove and the
glass bead nowhere; the page brings it home. Achilles's two-click trap —
where the order does not seem to matter, and all three strip motions undo the
score — is asserted exactly as he finds it, and the file records why it is a
trap by showing that a *single* glass click sends the steel bead to the
dresser when it is played against the clock.

---

## 2. `Backwards.py` — The Score Read Backwards

"Both at once. Last arrow first *and* every arrow the other way round, and
neither half will do alone." The file is that sentence:

```python
assert np.allclose((B @ A).conj().T, A.conj().T @ B.conj().T)
assert not np.allclose((B @ A).conj().T, B.conj().T @ A.conj().T)
```

with $A$ and $B$ generic, and then on the test box $U = VS$: the score played
backwards without mirroring is $U^T$, the score mirrored without reversing is
$\overline U$, and neither of them undoes $U$ — one of them, composed with
$U$, gives $iX$, a perfectly good turn that is simply the wrong one. The
worked number of §3.2, $\braket{+|U|a} = (3+4i)/(5\sqrt2)$, is checked with
its mirror $(3-4i)/(5\sqrt2)$, and both cheap versions give something else.

The same claim is put to the sphere, where it can be seen: a box acts on the
three numbers of a state as a real rotation matrix, the mirror box acts as the
inverse rotation, and the two cheap reflections do not — they are turns, but
not the turn that undoes this one. The rotation is checked against fifty
random states, not derived.

In the sketchbook the row is two named boxes, $S$ then $V$, and the value of a
closed picture with $U$ in the middle is compared with its mirror image: the
conjugate of a number, which is what a mirrored closed picture is. The two
cheap reflections give two other numbers.

On the ball: the four readings of the Second Challenge, one right and two
wrong — the page brings it home, the ball turned round on the table does not,
the strip flipped top over bottom does not. The Crab backs the wrong clicks
out last-first instead of replaying from home; the file asserts that this
lands in the same place as playing the score from home, which is why the
scene works.

---

## 3. `Sandwich.py` — The Sandwich

Brass, glass, brass — two clicks each — leaves the beads exactly where one
steel half-turn leaves them. On the blackboard that is the chapter's headline:

```python
assert np.allclose(H @ Z @ H, X)          # exact, scalar and all
assert np.allclose(H @ X @ H, Z)
assert np.allclose(H @ Y @ H, -Y)
assert not np.allclose(H @ Z @ H, Z)
```

and on the sphere it is the reason: $H$ exchanges the $x$- and $z$-axes and
reverses $y$, so a half-turn about $z$, seen through that exchange, is a
half-turn about $x$. The file checks the three numbers of §3.5 on a hundred
random states — each real, each on the unit sphere, none of them changed by a
global phase — and checks the turns of Proposition 3.6 the same way, by
following the point rather than by trusting the formula. The named points are
there too, including $(3/5,4/5) \mapsto (24/25, 0, -7/25)$, which stays on the
blackboard: the arrow is not built as a picture, as in Chapter 2.

The sketchbook proves $HZH = X$ as a picture equation, exactly, and then reads
the master diagram *The Turn in Three Steps* all three ways: as drawn, in the
mirror, and with three boxes of one's own choosing. The second reading is the
first one's mirror image, built by hand and compared with the mirror PyZX
makes; the third uses a state with no name and boxes $S$, $(HSH)S$, $T$, with
the middle one asserted to be neither real nor symmetric so that the dagger in
the effects is doing real work. Replacing it with the cheaper reflection
breaks the line:

```python
assert np.isclose(value(st1 + B + B.adjoint() + st1.adjoint()), 1)
assert not np.isclose(value(st1 + B + wrong + st1.adjoint()), 1)
```

Finally, the reason a closed picture with $X$ in it is a real number is given
as a picture and not as a computation: the picture is its own mirror image.
With $S$ in the middle it is not, and the number is not real.

---

## 4. `ThreeTurns.py` — Three Turns Make Any Turn

"Glass, steel, glass. It is always enough." The instance first, scalar
included, since the scalar is part of the claim:

```python
assert np.allclose(S @ (H @ S @ H) @ S, np.exp(1j * np.pi / 4) * H)
assert not np.allclose(S @ (H @ S @ H) @ S, H)
```

with the all-mirrored version giving $e^{-i\pi/4}H$ — the same turn of the
sphere, the opposite phase. Then the general statement, on a hundred random
boxes that keep the promise: the four angles are read off the four entries of
the box by hand, with `np.angle` and `np.abs` and nothing else, and the box is
rebuilt from them to better than $10^{-10}$. The reading is delicate in one
place, and the file says so where it happens: the two angle *differences* it
can see are each known only up to a whole turn, so the two $Z$-angles are
fixed only up to a common half-turn — which is why the middle angle is then
taken from a ratio, with its sign, and the phase from a single entry at the
end. That the ratio comes out real is itself the promise being kept, and is
asserted.

One needle is never enough, and that is checked rather than asserted in
passing: a $Z$-turn does not move the pole, and two of them are still one of
them.

In the sketchbook, `x(1/2)` is checked to be $HSH$ exactly — not
$(I + iX)/\sqrt2$ — and then the row $S$, $HSH$, $S$ is compared with the
Hadamard box twice: without its scalar phase it is a different box, with it it
is the same one, on the nose.

```python
assert not same_picture(row, hbox)      # the scalar is part of the claim
row.scalar.add_phase(-Fraction(1, 4))
assert same_picture(row, hbox)
```

Then any box at all: a row `z(c) – x(b) – z(a)` with a scalar phase reproduces
a random box's four angles. The angles are drawn at random and are not exact
fractions of $\pi$, so the phases are the nearest fraction with a denominator
under a billion and the comparison is `np.allclose` — said so in the code.
Dropping the scalar, or dropping the middle turn, gives a different box.

On the ball: glass, steel, glass with one click each is the brass half-turn;
the score reads the same from either end, so its page is the same three
turns with every arrow reversed — and, played from home, it lands in the
same place the score did, which is what a half-turn does. Then the Crab shuts
his eyes and tumbles the ball, and the file does what the Tortoise does: for a
hundred positions with no name, it finds a glass angle, a steel angle and a
glass angle that bring the ball home, and checks that the glass bead does not
stir while the glass needle is held, that the first glass turn lays the steel
level, and that the steel turn then stands the glass on top — the Tortoise's
recipe, and Achilles's reason for it.

The tumble the dialogue pictures is done once more by itself. The Crab's eyes
were shut, so that one position is given, not computed: glass along
$(12,-15,16)$ and steel along $(-8,16,21)$ in the room's frame, which is what
Achilles reads aloud — the glass low, towards the company and the dresser; the
steel high, towards the company and a little to the stove; nowhere with a name.
From there every twirl is solved and made about its needle *as it lies*, and
each of the four positions is checked against its picture: the glass twirl to
the nearer of the two level places leaves the steel along $(-4,0,3)$; the
steel twirl then stands the glass on top, leaving the ball wrong only by a
spin about the glass needle; the last glass twirl brings it home. None of the
three is a whole number of clicks, and no glass twirl alone would do.

```python
assert np.allclose(after, R @ turn(HOME[needle], angle))  # as it lies = before R
assert not np.isclose(s @ UP, 0)          # tumbled, the steel is not level
assert reading(R2, "glass") == "up"
assert at_home(R3)
```

---

## 5. `NoMirror.py` — What Has No Mirror

The catch on the pedestal sets the ball to home from wherever it is. Achilles
tries to draw it, and cannot: the page would have to lead to two places at
once. On the blackboard the same box is SET-0, and it fails the promise in
both ways at once —

```python
assert np.allclose(SET0.conj().T @ SET0, np.ones((2, 2)))   # not a bare wire
assert np.isclose(np.linalg.det(SET0), 0)                   # and no undoing
assert np.allclose(SET0 @ ket(0), SET0 @ ket(1))            # two places, one image
```

— so that no box whatever, drawn at random or otherwise, undoes it. Against
that, a box that keeps the promise keeps every number between two states, with
its corner: $\braket{U\phi|U\psi} = \braket{\phi|\psi}$, checked on fifty
random pairs, and checked to be a number that is *not* real, so that the test
is not passed by accident. And being invertible is not enough: $\mathrm{diag}(1,2)$
has an inverse and still stretches $(3/5,4/5)$ to squared length $73/25$.

In the sketchbook SET-0 is drawn the only way it can be: the incoming wire is
closed off and a fresh $0$ is opened. Its mirror image does not undo it, the
closed picture that should have been a certainty is not one, and Chapter 2's
echo — what leaves by every answer comes back to $1$ — fails, by exactly the
squared length of what SET-0 left:

```python
assert np.isclose(value(psi + u + u.adjoint() + psi.adjoint()), 1)
assert not np.isclose(value(psi + s0 + s0.adjoint() + psi.adjoint()), 1)
```

A long row of turns, on the other hand, is undone by its page however long it
is — and stops being so the moment the catch is gummed on the end.

On the ball, that is the evening's strips laid end to end: the Tortoise's one
arrow, Achilles's strip, and brass-glass-brass. The file asserts every reading
Achilles gives — glass level at the dresser, the far side from the one the
beads took in the Second Challenge; steel level away; brass level, away and
towards the dresser — then turns the whole strip over and checks that what he
reads off the back is, click for click, the score that brings the ball home.
Turned round on the table, or flipped top over bottom, the strip does not.
