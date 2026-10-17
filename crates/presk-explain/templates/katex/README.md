# KaTeX, vendored

KaTeX 0.19.0 (MIT, `LICENSE`), the npm package `katex@0.19.0`
(`https://registry.npmjs.org/katex/-/katex-0.19.0.tgz`, integrity
`sha512-v6Tznz3zJ7u3niRCoDTsumM2+HA2XXcCu+WAacCeHD2z3p9A9Ks987o5FzfTGBN0e8A0vjEgIDvLbICrXpdw/Q==`, checked
on 2026-10-17). Only what the pipeline view needs is kept, as the package
ships it: `dist/katex.min.js`, `dist/katex.min.css` and the twenty
`dist/fonts/*.woff2`; `SHA256SUMS` lists them.

`presk explain --format html` inlines the three into `explain.html`, the
fonts as `data:` URLs in place of the stylesheet's `url(fonts/…)` (the
`.woff` and `.ttf` fallbacks dropped), so the page stays one file that
works offline (plan 0.13.0 step 4c, D137). The page renders a formula
only when it is shown.
