# Technical audit and release scope

The release contains an exploratory research paper and its implementation by
Hervé Trinh, Independent Researcher. It does not contain new benchmark results.
The canonical manuscript is the original `papers/main.tex`, revised directly
with its original proofs, figures, numerical tables, and bibliography. An earlier
reconstructed note was discarded from the published tree once the original
source was provided. A local backup preserves the unmodified originals.

## Scientific findings

| Finding | Consequence | Release treatment |
|---|---|---|
| Occupancy-normalized block means are not the general soft-membership least-squares optimum | The original unconditional optimality claim was too broad | Paper distinguishes Gram-pseudoinverse and hard-membership cases; code defaults to the former |
| Historical real-data head was trained, cosine was not | Its pointing gain cannot isolate co-clustering from learned capacity | Historical figures are qualified; corrected configs include learned bilinear and exactly parameter-matched MLP controls |
| Earlier real-data AUROC was near chance | Peak pointing does not establish global spatial ranking | Explicit limitation in the abstract and observations |
| Earlier decoding condition did not enable a generation intervention | Identical numbers do not establish compatibility or hallucination reduction | The corresponding conclusion is withdrawn; the actual vocabulary-logit intervention is documented separately |
| Earlier split definitions were custom and manifests are absent | Official split isolation cannot be retrospectively verified | Old values are not presented as standard RefCOCO test results |
| Raw synthetic outputs, real-data manifests, and trained checkpoints are absent; sanitized figure-level summaries are present | Historical significance and exact reproduction cannot be audited | Original historical tables and sanitized figure inputs retained with caveats; no new benchmark measurements |
| The strongest historical tap is embedding-level hidden state zero | It is an early feature probe | No claim of recovered grounding throughout the decoder |
| Built-in adjacent-method and caption-scoring implementations are simplified | Their names alone cannot support official benchmark comparisons | README and paper explicitly identify proxy/diagnostic status |

## Code corrections in this release

- **Shared spatial geometry.** Training and grounding evaluation use the same
  row-major patch-center masks. The evaluator no longer guesses processor
  geometry from original image aspect ratio; unsupported layouts fail clearly.
  Explicit verified rectangular grids are configurable. Tiny boxes are not
  expanded to neighboring cells. No-positive-center examples count as
  pointing/recall misses and have a separate AUROC denominator.
- **Comparable aggregation.** Grounding summaries separate dataset, backbone,
  split, layer, spatial protocol, and explicit grid. Duplicate seeds are errors.
  Transfer comparisons have their own matched pairs. Generation tests pair the
  same seed and checkpoint. Summary grouping also retains generation mode and
  prior weight. Student-t inference replaces a sparse critical-value lookup;
  degenerate tests are flagged and avoid nonstandard JSON infinity.
- **Generation boundary.** Features and vocabulary embeddings are cast to the
  projection-head dtype before float32 heads consume bf16 features. Independent
  decoding assignments use their own candidate-token head. Vocabulary chunks
  must be positive, and the assignment cache tracks parameter updates. Tests
  exercise the actual logits-processor boundary and continuation decoding.
- **Evaluation mode and provenance.** Backbones are placed in evaluation mode;
  a learned CoGround evaluation requires a checkpoint. Grounding outputs include
  the resolved config, asset registry, runtime metadata, spatial-protocol fields,
  and per-example records. LoRA/full-backbone configurations are rejected by the
  generic frozen-backbone evaluator to avoid silently ignoring trained weights.
- **Adapter boundaries.** The multimodal auto-loader is considered when present.
  Unaligned answer-only causal-LM labels now fail explicitly. Cross-attention
  inputs must have the expected patch/token axes. Real model integration remains
  an explicit validation step.
- **Training utilities.** Frozen mock adapters no longer require a `.model`
  member. Preflight finds an existing filesystem ancestor for disk checks, so a
  new clone need not already contain ignored result directories.
- **Synthetic controls.** Training data random streams reset after method-specific
  head initialization. Cluster-mismatch evaluation has its own shared stream.
  NMF restores the original affinity scale and propagates failures instead of
  silently substituting the input as a successful reference. No experiment was
  rerun; historical numbers are not comparable to these changed scripts.
- **Maintenance.** Ruff formatting/import checks cover the source, entry points,
  and tests. Dependency declarations have one source of truth in `pyproject.toml`;
  Gemma 4 and quantization dependencies are explicit extras. Existing CC BY 4.0
  licensing is preserved with the author's name.

The spatial conventions, synthetic random streams, NMF reference behavior,
aggregation, and decoding-assignment fixes affect evaluation behavior and must
be recorded in any future experiment report.

## Validation

Local validation uses Python 3.12.9, PyTorch 2.9.1+cu128 (CPU unit-test execution),
Transformers 4.57.3, Accelerate 1.12.0, OmegaConf 2.3.0, and Ruff 0.17.3.
From the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 python -m pytest -q -p no:cacheprovider
python -m ruff check src scripts tests papers/figures
python -m ruff format --check src scripts tests papers/figures
```

The suite covers 32 tests, including shared box geometry, special tokens,
unknown layouts, tiny boxes, paired inference, protocol separation, duplicate
seeds, checkpoint matching, bf16 generation plumbing, independent vocabulary
heads, block solvers, parameter counts, configs, checkpoint loading, metrics,
and preflight with a new nested output directory. An installed Requests
compatibility warning is environmental and does not fail the suite.

The paper compiles with Tectonic 0.17.0. Its rendered pages, author metadata,
email, equations, tables, and bibliography were checked. It uses the standard
`article` class. The bibliography links primary papers and upstream assets.
Minor underfull-line TeX warnings concern spacing; no overfull boxes or missing
references remain.

A fresh clone also passed the same validation before publication.
A GitHub Actions workflow repeats CPU unit tests and Ruff checks. Its remote
outcome is separate from the local validation and should be inspected after
publication.

## Remaining limits

- No training, synthetic campaign, benchmark evaluation, or distribution
  packaging job was executed for this cleanup.
- CPU tests do not establish correctness of Gemma 4/PaliGemma feature extraction,
  processor geometry, BF16 GPU numerics, two-process DDP, or resumed experiments.
- The generic spatial contract assumes uniform resizing and verified row-major
  tokens with optional trailing specials. Cropped, tiled, and packed layouts
  require model-specific coordinate mapping.
- Grounding BCE, an affinity reconstruction, and exact fixed-membership algebra
  do not establish semantic grounding gains or prevent all trivial solutions.
- The vocabulary prior is static over decoding steps and uses a representation
  different from contextual training tokens; paired generation is essential.
- Resume assumes unchanged manifests, configuration, topology, and compatible
  software. Worker-side random transforms and arbitrary environment changes are
  not guaranteed to replay bitwise.
- Official POPE/CHAIR comparability requires validating parsing, lexical rules,
  and denominators against the original evaluation implementations.
- Dependency lower bounds are not a frozen environment; future scientific runs
  must retain exact versions and resolved configurations with their outputs.

## Published tree

Source, configurations, scripts, unit tests, the original paper, its sanitized figure inputs, and relevant documentation
are committed. Local ZIPs, datasets, outputs, earlier manuscript material,
private reviews, cleanup scripts, and temporary files are ignored and preserved
on disk. No earlier result JSON is presented as a corrected benchmark artifact.


## Workspace cleanup

The Python package now follows a conventional repository-root `src/` layout,
with `configs/`, `scripts/`, and `tests/` alongside `pyproject.toml`. The code
guide is in `docs/code_guide.md`, and `papers/` is the only manuscript directory.
The original figures and proofs are retained; the inactive decoding-combo row is
excluded from the displayed comparison. Figure regeneration reads sanitized
JSON and does not train or evaluate any model.

Old templates, submission checklists, duplicate supplementary trees, rendered
page checks, ZIP files, reviews, conversations, and local outputs were moved to
a separate local backup outside the repository. The backup is not published
and is not a reproducibility dependency.
