# Research paper

[PDF](main.pdf) · [Original LaTeX, revised](main.tex) · [Bibliography](refs.bib)

**CoGround-VLM: Differentiable Patch-Token Co-Clustering for Visual Grounding
in Vision-Language Models**

Hervé Trinh · Independent Researcher · trinh.herve@gmail.com

This version modifies the original manuscript supplied in `papers/`. Its
architecture diagram, proofs, original numerical tables, and bibliography are
retained. It uses the standard LaTeX `article` class, with no conference template,
submission checklist, or company affiliation.

Historical results are explicitly qualified. The earlier decoding-combo row
is excluded because it did not enable a decoding intervention. Corrected code
and prospective experiments are distinguished from those original observations.
The corrected campaign has not been executed for this version.

From the repository root:

```bash
tectonic --outdir papers papers/main.tex
```

Alternatively, from `papers/`, run `pdflatex main.tex`, `bibtex main`, then
`pdflatex main.tex` twice.

The original summary figure can be redrawn from the sanitized historical JSON:

```bash
python -m pip install -e '.[paper]'
MPLBACKEND=Agg python papers/figures/generate_results_figures.py
```

This only draws figures; it does not run training or evaluation. The JSON files
in `metrics/` are original figure-level summaries and sanitized RefCOCO aggregates.
They are not raw runs and do not independently validate the reported statistics.
No dataset images, model weights, prediction logs, or private conversations are
included. The compiled manuscript and vector summary figure are included.
