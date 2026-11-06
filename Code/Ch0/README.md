# Chapter 0 — The Empty Diagram: the companion code

The Prelude's one file checks, on the blackboard, that a product of no numbers is one and
that one is the number that changes nothing; in the sketchbook, that a picture with nothing
in it evaluates to exactly one, and leaves a picture it is put beside as it was. Every
picture in the file has no wire at all. The chapters after it are where something is drawn.

Run it, or the whole Prelude, from the repository root:

```bash
.venv/bin/python chapter00/Code/Ch0/EmptyDiagram.py
bash check-code.sh 0          # must end with: 1 files, 0 failed
```

The file prints one line per strand and exits non-zero if any `assert` fails. Running the
files is this book's analogue of type-checking. Pictures are evaluated only through
`matrix_of` and `same_picture` from `tools/dfp.py`, exactly, the scalar included.

---

## The exchange: Prelude on a Blank Sheet — `Ch0/EmptyDiagram.py`

The Tortoise shows a finished picture of nothing, and Achilles calls it worth zero. She lays
it beside his own blank sheet, and nothing about his changes. He objects that adding zero
does nothing either; she answers that she put hers *after* his, and he reckons: "Twice as
much, and then nothing done: still twice. ... One, not zero. ...Oh."

**Blackboard.** The product of no numbers is one; one times a random non-real $a$ is $a$,
exactly; appending $a$ to no numbers gives $a$; the number that leaves one as it was is one.
Achilles's zero is the negative: it leaves $a$ alone only under adding.

```python
assert np.prod([]) == 1 and math.prod([]) == 1
assert 1 * a == a
assert np.prod([a]) == np.prod([]) * a
assert u * 1 == 1 and u == 1                  # u = the product of no numbers
assert 0 * a != a and 0 * a == 0
```

**Sketchbook.** The empty graph evaluates to exactly one. Blank beside blank, side by side
(`tensor`) or one after the other (`compose`), is blank.

```python
E = zx.Graph()
assert matrix_of(E).shape == (1, 1) and np.allclose(matrix_of(E), [[1]])
assert same_picture(E.tensor(zx.Graph()), E)
one_after_other = zx.Graph(); one_after_other.compose(zx.Graph())
assert same_picture(one_after_other, E)
```

Then one closed picture `P` with a seeded random phase, whose value is not real, is put
beside the empty graph both ways and composed with it; each time it is left as it was. This
is a translation check on one picture, a special case, not a proof: the chapter proves the
"beside any picture" sentence by looking, since the two sides are the same page. The
negative is Achilles's zero again, in pictures: a closed picture worth $0$, put beside `P`,
does change it.

```python
assert same_picture(E.tensor(P), P)
assert same_picture(P.tensor(zx.Graph()), P)
assert same_picture(C, P)                     # C = P.copy(); C.compose(zx.Graph())
assert not same_picture(P.tensor(Z0), P)      # Z0 is worth 0
```

Every graph the file builds is checked to have no edge and no boundary.
