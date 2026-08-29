# CoGround-VLM: code guide

PyTorch implementation of differentiable patch–token co-clustering and learned
grounding controls. The [research paper](../papers/main.pdf) is by Hervé Trinh,
Independent Researcher ([trinh.herve@gmail.com](mailto:trinh.herve@gmail.com)).

Run the commands below **from the repository root**. Training and evaluation are
explicit operations; they are not prerequisites for running the unit tests.
The corrected campaign has not been executed for this release.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
# Equivalent local developer install:
# python -m pip install -r requirements.txt
```

Python 3.10+ is required. Choose CPU/CUDA PyTorch wheels for your environment.
The `gemma4` extra requests Transformers >=5.5.0; confirm the architecture and
loader against the [official model card](https://huggingface.co/google/gemma-4-E2B-it)
before downloading weights. The `quantization` extra installs optional
bitsandbytes; the corrected two-GPU configs do not use 4-bit loading.

The local CPU tests pass with Python 3.12.9, PyTorch 2.9.1, Transformers 4.57.3,
Accelerate 1.12.0, and OmegaConf 2.3.0. These tests use small tensors and in-memory
adapters. Real-backbone extraction and two-A100 execution remain unvalidated.

## Library

- `models/coclustering.py`: membership heads, soft least-squares block solver,
  occupancy ablation, objectives, and block-conditioned prior.
- `models/grounding_baselines.py`: bilinear and parameter-matched MLP controls.
- `models/backbones.py`: generic Hugging Face adapter and a test-only mock.
- `data/`: JSONL manifests, text alignment, shared spatial masks, synthetic generators.
- `baselines/decoding.py`: vocabulary-logit processor and adjacent baseline proxies.
- `train/checkpointing.py`: heads, optimizer, scheduler, progress, and random states.

The baseline proxies are research scaffolds, not official implementations of
VCD, OPERA, PAINT, or SeGroS. The built-in CHAIR-style lexical scorer is also a
diagnostic approximation; use the official evaluator before benchmark claims.

## Data preparation

Obtain COCO train2014 images and RefCOCO-family annotations separately from
[COCO](https://cocodataset.org/#download) and [REFER](https://github.com/lichengunc/refer).
The exporter keeps the referring phrase verbatim and exports official splits
separately. It does not download or redistribute data.

```bash
python scripts/prepare/export_refcoco.py \
  --refer-root /path/to/refer \
  --data-root /path/to/refer/data \
  --image-root /path/to/coco/train2014 \
  --dataset refcoco \
  --output-dir data/refcoco/manifests

python scripts/prepare/validate_manifest.py \
  --manifest data/refcoco/manifests/refcoco_train.jsonl \
  --image-root /path/to/coco/train2014 \
  --require-grounding
```

Use `--dataset refcocog` with its own output directory for dataset transfer.
See the [manifest schema and metric definitions](experiments.md).
Set `dataset.image_root=/path/to/coco/train2014` through a CLI override.

Spatial targets and metrics use patch centers, with the same row-major grid.
Square grids and square grids plus four **trailing** special tokens have a
default mapping. Verified rectangular grids require
`dataset.patch_grid=[rows,columns]`. These assumptions must be checked against
the processor; cropped, tiled, and packed images need a separate mapping.
Grounding token spans refer to the processed sequence; prefer `grounding_text`
so image/chat tokens are accounted for at runtime.

## Transformer training

The corrected configurations are under `configs/experiments/revision/`.
The older `configs/experiments/gemma4/` files are retained as diagnostic settings;
they do not support the corrected claims and some need an explicit legacy
self-reconstruction override.

The target runtime is two A100 processes, bf16, batch 1 per GPU, and eight
accumulation steps: nominal global batch 16. `training.max_steps` counts
optimizer updates. End-of-epoch partial accumulation can contain fewer examples.
Preflight validates infrastructure and manifests, not model geometry.

```bash
export COGROUND_CUDA_DEVICES=0,1
export GEMMA4_MODEL_PATH=google/gemma-4-E2B-it
export HF_HOME=/path/to/persistent/huggingface-cache

python scripts/prepare/cache_backbone.py \
  --config configs/experiments/revision/gemma4_refcoco_coground.yaml

CUDA_VISIBLE_DEVICES="$COGROUND_CUDA_DEVICES" python scripts/train/preflight_2xa100.py \
  --config configs/experiments/revision/gemma4_refcoco_coground.yaml \
  dataset.image_root=/path/to/coco/train2014

bash scripts/train/launch_2xa100.sh \
  configs/experiments/revision/gemma4_refcoco_coground.yaml \
  experiment_id=revision_gemma4_refcoco_coground_seed17 \
  seed=17 dataset.image_root=/path/to/coco/train2014
```

The launcher resumes automatically unless `COGROUND_RESUME=none`. Resume
requires unchanged manifests/configuration and the same process topology.
Checkpoints live in `results/runs/<experiment_id>/step_<optimizer_step>/`.
No frozen backbone weights are saved. LoRA or full-backbone evaluation requires
an adapter that explicitly restores those trainable parameters; the generic
head-only evaluator is intended for the frozen-backbone protocol.

Campaign stages are sequential because each training job uses both GPUs:

```bash
bash scripts/train/run_revision_campaign_2xa100.sh core
bash scripts/train/run_revision_campaign_2xa100.sh ablations
bash scripts/train/run_revision_campaign_2xa100.sh layer-sweep
bash scripts/train/run_revision_campaign_2xa100.sh transfer
bash scripts/train/run_revision_campaign_2xa100.sh decoding
```

Remaining arguments are applied as OmegaConf overrides to every run. The `all`
stage trains each listed stage; it does not automatically evaluate checkpoints.
Validate each selected tap against the backbone's actual depth.

## Evaluation

Evaluate a trained frozen-backbone grounding head using its resolved config.
The same entry point supports the matched MLP, bilinear, and cosine controls.

```bash
python scripts/eval/run_grounding_metrics.py \
  --config results/runs/revision_gemma4_refcoco_coground_seed17/resolved_config.yaml \
  --checkpoint results/runs/revision_gemma4_refcoco_coground_seed17/step_00003000 \
  --output results/runs/grounding_revision_gemma4_refcoco_coground_seed17/metrics.json
```

Repeat for each method/seed and choose a distinct output directory. Report
pointing, patch Recall@1/5/10, AUROC, in-box mass, and their denominators.
Examples with no positive patch center count as pointing/recall misses and are
excluded from AUROC; all-positive examples also have undefined AUROC.

```bash
python scripts/eval/run_synthetic_multi.py \
  --structures block overlapping continuous \
  --out results/runs/synthetic_structure_stress/metrics.json
python scripts/eval/run_synthetic_kmismatch.py \
  --out results/runs/synthetic_kmismatch/metrics.json
python scripts/eval/aggregate_all.py
```

Synthetic jobs train small heads; run them only when you intend to execute an
experiment. They do not require downloaded backbones. The real-data matched MLP
parameter contract does not apply to every synthetic control.

`run_combo_decoding_prior.sh` trains a prior and evaluates the same checkpoint
with the prior enabled and disabled when POPE/CHAIR-style manifests exist.
The current intervention is a **static prompt-conditioned vocabulary bias**.
The former `logit_proxy` mode is rejected. Model-specific attention hooks are
not installed by the generic adapter.

## Validation

```bash
python -m pytest -q
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
```

See the [experiment protocol](experiments.md) for data, metrics, and evidence limits.
Original repository material is [CC BY 4.0](../LICENSE); upstream assets retain
their own terms.
