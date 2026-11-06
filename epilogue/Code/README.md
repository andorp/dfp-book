# Epilogue — The Empty Diagram: the companion code

The Epilogue's one file lets a state go, on the blackboard as a trace and in the sketchbook as
a cap on its two copies, and finds exactly one both ways; PyZX's own rewriting takes the
unfolded picture down to a graph with nothing in it. The traps are checked beside it: a loop
is two, and only a state lets go to one.

Run it from the repository root:

```bash
.venv/bin/python epilogue/Code/Epilogue/LastRewrite.py
bash check-code.sh epilogue   # must end with: 1 files, 0 failed
```

The file prints one line per strand and exits non-zero if any `assert` fails. Running the
files is this book's analogue of type-checking. Pictures are evaluated only through
`matrix_of` and `same_picture` from `tools/dfp.py`, exactly, the scalar included; nothing is
compared "up to a scalar", because here the scalar is the theorem.

---

## The exchange: Postlude on the Same Sheet — `Epilogue/LastRewrite.py`

Back from the Crab's at first light, Achilles lays the Tortoise's blank sheet from the autumn
on the table. She tosses his coin, claps it down under her palm unseen, and asks what it is
worth, the whole of it, if they never look. Achilles says "A half. Heads, a half." She slides
the coin off the sheet into the open drawer and shuts it, and he reckons again: "A half and a
half: once. ... Whatever way it was set, once." What is left on the table is the sheet, as it
was, worth once. ...Oh.

Both strands check one state $\psi$, built from two seeded phases $a, b$ (units of $\pi$):
$\psi=\tfrac12\bigl(1+e^{i\pi a},\,(1-e^{i\pi a})e^{i\pi b}\bigr)$, normalized and **non-real**
(its two amplitudes differ by a non-real phase), so that the mirror image, which conjugates,
cannot be mistaken for a plain transpose.

**Blackboard.** Letting go is the trace. $\operatorname{tr}\ket\psi\bra\psi$ is
$\alpha\bar\alpha+\beta\bar\beta=\braket{\psi|\psi}=1$, for $\psi$ and for eight more seeded
non-real states. The doubled vector, $\psi$ beside its conjugate copy, closed by the
**unnormalized** cap $\bra{00}+\bra{11}$, is the $1\times1$ matrix $(1)$: a process from no
wire to no wire. A normalized cap would give the wrong number.

```python
assert np.isclose(np.linalg.norm(CAP), np.sqrt(2))       # the cap is unnormalized
assert np.isclose(np.trace(rho), psi.conj() @ psi) and np.isclose(np.trace(rho), 1)
closed = CAP @ np.kron(psi, psi.conj()).reshape(4, 1)
assert closed.shape == (1, 1) and np.allclose(closed, [[1]])
assert not np.isclose((CAP / 2) @ doubled, 1)
```

The bend: the conjugate copy carried round the cap arrives on the upper wire as the row
$\psi^{\dagger}$, and **not** $\psi^{T}$; used as the effect, $\psi^{T}$ would not close $\psi$
to one.

```python
bent = CAP @ np.kron(I, psi.conj().reshape(2, 1))
assert np.allclose(bent.ravel(), psi.conj())             # psi-dagger
assert not np.allclose(bent.ravel(), psi)                # not psi-transpose
assert np.isclose(bent @ psi, 1) and not np.isclose(psi @ psi, 1)
```

Then the example and the traps. $\ket+$ let go is $\tfrac12+\tfrac12=1$. Looking is not letting
go: the effect $0$ on each copy of the doubled $\ket+$ weighs one answer, the square of the
amplitude $\braket{0|+}=1/\sqrt2$, which is $\tfrac12$, not $1$ — Achilles's first "a half".
His reckoning is checked in exact fractions: a half and a half is once, five sixths and one
sixth is once still, and heads alone is not once. A wire does not always go for nothing: a
bare loop is $2$, an unnormalized point let go is $2$, and the half-cup, $\tfrac12$ times the
loop, is $1$. A mixed state, a weighted list of pure ones whose weights sum to one, lets go to
one as well.

```python
assert np.isclose(look, 0.5) and np.isclose(amplitude, 1 / np.sqrt(2))
assert Fraction(1, 2) + Fraction(1, 2) == 1 and Fraction(1, 2) != 1
assert Fraction(5, 6) + Fraction(1, 6) == 1
assert np.isclose(CAP @ CUP, 2) and np.isclose(CAP @ (CUP / 2), 1)
assert np.isclose(CAP @ np.kron(point, point.conj()), 2)  # point = |0> + |1>
assert np.linalg.matrix_rank(mixed) > 1 and np.isclose(np.trace(mixed), 1)
```

**Sketchbook.** The empty graph is exactly one. PyZX cannot evaluate a graph with a ground
vertex, so the discard is built unfolded, as the cap on the two copies; the picture "$\psi$,
then let go" *is* that unfolded picture, so it is not built separately. The state box is a red
and a green node (phases $a$, $b$) and evaluates to the blackboard's $\psi$; its conjugate copy
carries the negated phases and evaluates to $\psi^{*}$, not $\psi$. The lone cap is the
unnormalized one.

Then the chain, each term a graph: the unfolded picture (the two copies joined by the cap, with
its scalar $\tfrac12$), the state followed by its own mirror image (`s.adjoint()`), and the
empty graph. All three are the same picture, exactly. The bend is checked in pictures too:
the copy carried round the cap is the state's mirror image, and not the state bent round
without conjugating.

```python
assert same_picture(bent_copy(a, b), s.adjoint())             # psi-dagger
assert not same_picture(bent_copy(a, b), transposed_effect(a, b))
assert same_picture(P2, E) and same_picture(P3, E) and same_picture(P2, P3)
```

The rewrites derive it, not only confirm it: `full_reduce` takes a copy of the unfolded
picture, and a copy of the state-then-mirror picture, to a graph with no vertex and no edge,
whose scalar is one. The last rewrite leaves nothing.

```python
g = picture.copy(); simp.full_reduce(g)
assert g.num_vertices() == 0 and np.isclose(g.scalar.to_number(), 1)
assert same_picture(g, E)
```

The traps, as negatives: the bare loop (a cup closed by a cap) is $2$ and not the empty graph,
and `full_reduce` leaves it as the scalar $2$; the unfolded picture with its $\tfrac12$ left
off is $2$; the unnormalized point let go is $2$. The loop with its $\tfrac12$, the half-cup
let go, is one and the same picture as the empty graph. Looking at the doubled $\ket+$ with
the effect $0$ is $\tfrac12$, not the empty graph; one copy alone is $1/\sqrt2$.

```python
assert np.allclose(matrix_of(loop()), [[2]]) and not same_picture(loop(), E)
assert np.allclose(matrix_of(unfolded(a, b, power=0)), [[2]])
assert np.allclose(matrix_of(unnormalized_point_let_go()), [[2]])
assert same_picture(loop(power=-2), E)
assert np.allclose(matrix_of(look_at_plus()), [[0.5]])
```

Every hand-built graph asserts its number of edges, so no strand is silently merged.
