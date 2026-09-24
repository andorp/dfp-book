# Dirac, Feynman, Penrose: An Entangled Golden Braid

A book introducing quantum computing in the style of *Gödel, Escher, Bach*, in two
braided notations: classic (bra-ket, matrices) and diagrammatic (string diagrams,
ZX-calculus).

This repository holds the book as a single self-contained LaTeX file, together with
the PDF built from it. It is a work in progress and is updated as chapters are written.

- [`dfp-book.pdf`](dfp-book.pdf) — the book (6×9 in print layout)
- [`dfp-book.tex`](dfp-book.tex) — its LaTeX source

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
(CC BY-NC-ND 4.0); see [`LICENSE`](LICENSE). You may read and share the book with
attribution, but not use it commercially or distribute modified versions.
