# Dirac, Feynman, Penrose: An Entangled Golden Braid

A book introducing quantum computing in the style of *Gödel, Escher, Bach*, in two
braided notations: classic (bra-ket, matrices) and diagrammatic (string diagrams,
ZX-calculus).

This repository holds the book as a single self-contained LaTeX file, together with
the PDF built from it. It is a work in progress and is updated as chapters are written.

- [`dfp-book.pdf`](dfp-book.pdf) — the book (6×9 in print layout)
- [`dfp-book.tex`](dfp-book.tex) — its LaTeX source
- [`Code/`](Code) — the companion Python code, one folder per chapter (`Code/Ch0` … `Code/Ch19`, `Code/Epilogue`)

## Companion code

Every challenge in a chapter's dialogue has a Python file in `Code/ChN/`
that checks its claim in both strands: the algebra with NumPy, the picture with
[PyZX](https://github.com/zxcalc/pyzx) (0.10.6), the two compared as tensors. Each
folder's `README.md` walks through its files in order.

The files are published to be read alongside the book. They import shared helpers
(`tools/dfp.py` and the prop-picture checkers) and read the chapter's dialogue
sources, none of which are part of this repository, so they do not run on their own.

## Building the PDF

The file needs LuaLaTeX (TeX Live 2023 or later). Compile it twice, with `makeindex` in between:

```bash
lualatex dfp-book.tex
makeindex dfp-book.idx
lualatex dfp-book.tex
```

Required fonts: TeX Gyre Pagella, TeX Gyre Pagella Math, TeX Gyre Heros, DejaVu Sans
Mono, FreeSerif, and a Japanese font for `luatexja` (the haiku). Required packages
include `unicode-math`, `luatexja`, `tikz` (with the `zx-calculus` library), `tikz-cd`,
`braket` and `imakeidx`. All of these ship with a full TeX Live install.

## License

© Andor Penzes. Licensed under
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/)
(CC BY-NC-ND 4.0); see [`LICENSE`](LICENSE). The license covers the companion code
too. You may read and share the book and its code with attribution, but not use them
commercially or distribute modified versions.
