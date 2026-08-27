# Corrected experiment contract

This document defines which runs may support a revised manuscript. The earlier aggregate is kept only in the private local workspace. It is not published as benchmark evidence. The corrected campaign has not been run.

## Main comparison

All three methods use the same frozen backbone, layer tap, RefCOCO records,
grounding target construction, optimizer schedule, number of updates, and five
seeds.

| Method | Learned components | Training target |
|---|---|---|
| CoGround | Bilinear affinity and two MLP heads | Reconstruction plus true grounding BCE |
| Matched MLP | The same bilinear affinity and the same two MLP heads | True grounding BCE |
| Bilinear | Bilinear affinity only | True grounding BCE |

The matched MLP has exactly the same trainable parameter count as CoGround when
the two cluster counts equal its rank. A test enforces this contract.

The primary metrics are pointing accuracy, Recall@1, Recall@5, Recall@10,
in-box versus out-of-box AUROC, and probability mass inside the union of all
reference boxes. Token spans restrict scoring to the referred expression when
they are available.

## Structural controls

The following controls each change one part of the corrected main
configuration:

- frozen cosine affinity instead of a learned bilinear affinity;
- occupancy-normalized B instead of the soft-assignment least-squares solver;
- identity B instead of an estimated block matrix;
- independent sigmoid memberships instead of row-wise simplex assignments;
- permuted patch and token grounding targets with the same positive-label
  count;
- independent decoding assignments instead of the shared U and V state.

## Transfer checks

Dataset and backbone transfer are isolated:

- Gemma 4 E2B on RefCOCOg changes the dataset while retaining the backbone;
- PaliGemma 2 on RefCOCO changes the backbone while retaining the dataset.

These results must be reported separately. A single run changing both axes
cannot establish which kind of transfer succeeded or failed.

## Decoding and hallucination evaluation

`decoding.mode=vocab_logit_bias` is an effective intervention. It assigns the
input vocabulary embeddings to textual clusters, maps the prompt-conditioned
block state to a vocabulary-sized score, and supplies that score to Hugging
Face generation as a logits processor. The former `logit_proxy` mode is
rejected at runtime because it did not change generated tokens.

The same checkpoint must be evaluated with the prior enabled and disabled.
POPE reports accuracy, precision, recall, F1, yes ratio, and unparsed count.
The built-in CHAIR-style scorer is an explicit-lexicon diagnostic, not an exact reproduction of official CHAIR. Official tokenization, disambiguation, and object-mention counting must be validated against the original evaluator before benchmark reporting. Generated captions are never used to infer their own ground-truth object set.

## Synthetic misspecification

The synthetic study includes three structures:

- block-constant affinities, which match the CoGround assumption;
- overlapping memberships, which violate hard partitioning;
- continuous low-rank affinities, which do not have block-constant structure.

A credible claim must describe all three regimes. Strong performance only on
the matched block generator is evidence for the inductive bias, not evidence
of broad superiority.

## Manifest schema

Every JSONL record requires:

```json
{
  "id": "sample-id",
  "image": "relative/or/absolute/path.jpg",
  "prompt": "the referring expression or benchmark question"
}
```

Grounding training and evaluation additionally require either:

```json
{
  "boxes": [[x1, y1, x2, y2]],
  "token_spans": [[start_token, end_token]]
}
```

or a dense `grounding_matrix` matching the extracted patch and token axes.

`token_spans` are positions in the final processed model sequence. For
RefCOCO-family data, prefer the tokenizer-independent form below. The phrase
must occur verbatim in `prompt`; at runtime it is located after the processor
has inserted image and chat-template tokens:

```json
{
  "boxes": [[x1, y1, x2, y2]],
  "grounding_text": "the referred expression"
}
```

POPE records require a yes/no `answer`. CHAIR records require official
`ground_truth_objects` and either a list or canonical-to-synonyms mapping in
`candidate_objects`.

## Reporting rule

No corrected result table may use the earlier local aggregate. Revised values must
come from new `results/runs/` directories containing the resolved config,
checkpoint, raw per-example output, metrics, asset registry, and run metadata.

## Two-A100 execution contract

Corrected transformer runs use two A100 processes, bf16, per-device batch size
1, gradient accumulation 8, and global batch 16. `max_steps` counts synchronized
optimizer updates. Checkpoints are written only at accumulation boundaries and
include enough state to resume on the same two-process topology. Changing the
world size during a resume is rejected because it would change data order and
the effective batch contract.


## Spatial protocol: patch_centers_v1

Training and grounding evaluation share `data/spatial.py`. A patch is positive
when its center lies in a reference xyxy box. Multiple boxes use their union for
scoring; each box is paired with a token span, or all share one phrase span.
Patch Recall@k means a top-k center hit, not IoU proposal recall.

The default layout is a square row-major grid, optionally followed by four
special tokens. Explicit `dataset.patch_grid: [rows, columns]` supports verified
rectangular grids; any remaining tokens must be trailing specials. Original
image aspect ratio does not determine the processor grid. Uniform resizing is
assumed. Crops, tiles, and packed token layouts need a model-specific mapping.

This changes the earlier training behavior, which rounded box coordinates to
patch cells, and the evaluator, which guessed a grid from the image aspect
ratio. Historical metrics are not directly comparable. Small boxes can have no
positive center. They remain all-negative training targets and count as
pointing/recall misses in evaluation; AUROC excludes examples with one class
and reports its own denominator. The positive-center fraction is the random
pointing reference.

The generic evaluator supports frozen backbones only and rejects LoRA or
full-backbone runs rather than silently discarding their trained parameters.

## Aggregation and synthetic reproducibility

Grounding comparisons are separated by dataset, backbone, split, layer tap,
spatial protocol, and explicit grid. Duplicate seeds in the same condition are
errors. Generation pairs must share a checkpoint and seed. Confidence intervals
use the Student distribution and sample standard deviation; zero-variance
paired tests are flagged as degenerate and serialize as valid JSON.

Synthetic learned methods reset the training data RNG after head initialization,
so corresponding seeds receive the same generated batches. The cluster-mismatch
evaluation uses a separate seed stream shared across methods and cluster counts.
These changes, together with the corrected block solver, mean the scripts do
not reproduce the historical PDF's synthetic summaries verbatim. The synthetic
low-rank control matches assignment-head architecture; the exactly matched
real-data MLP parameter contract is a separate comparison.

The NMF diagnostic restores the original affinity scale after nonnegative shifting; factorization failures propagate rather than falling back to the input matrix. This also changes its behavior relative to the historical script.
