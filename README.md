# CoGround-VLM

**Differentiable patch–token co-clustering for visual grounding in vision–language models.**

[Paper (PDF)](papers/main.pdf) · [Code guide](docs/code_guide.md) · [Experiment protocol](docs/experiments.md)

Hervé Trinh · Independent Researcher · [trinh.herve@gmail.com](mailto:trinh.herve@gmail.com)

CoGround-VLM compresses a patch–token affinity matrix into soft visual memberships
`U`, soft textual memberships `V`, and a small interaction matrix `B`. The
reconstruction `U B Vᵀ` provides spatial evidence maps. A separate experimental
head can turn the block state into a vocabulary bias during generation.

The research paper is provided as a PDF alongside the PyTorch implementation.
This is an exploratory research implementation. The paper distinguishes
historical observations from the corrected experiment protocol. **The corrected
benchmark campaign has not been executed**, and this release does not establish
improved hallucination rates or superiority over matched learned baselines.

## Repository

```text
papers/main.pdf          Independent research manuscript (PDF)
src/coground_vlm/        Models, data, metrics, decoding, checkpoint utilities
configs/                 Grounding, synthetic, ablation, and runtime settings
scripts/                 Data preparation, training, evaluation, aggregation
tests/                   CPU unit tests; no model or dataset downloads
docs/                    Code guide and experiment protocol
pyproject.toml           Package metadata and dependencies
```

Datasets, model weights, checkpoints, experiment outputs, ZIP archives, temporary
files, and private review correspondence are excluded by `.gitignore`. Editable
manuscript sources and local working material are kept outside this repository.

## Install and validate

Use a Python 3.10+ virtual environment. Install a PyTorch wheel appropriate to
your CPU or CUDA environment before installing the package if needed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
```

The local CPU validation used Python 3.12.9, PyTorch 2.9.1, Transformers 4.57.3,
Accelerate 1.12.0, and OmegaConf 2.3.0. This validates software contracts; it does
not validate a Gemma 4 integration or GPU experiments. Gemma 4 needs the newer
Transformers architecture support; install `pip install -e '.[gemma4]'` and check
the [upstream model instructions](https://huggingface.co/google/gemma-4-E2B-it).
The generic spatial adapter assumes a verified row-major grid and uniform image
resizing; crop, tile, and packed-token layouts need a model-specific mapping.

Training and benchmark commands are documented in the [code guide](docs/code_guide.md).
They require separately obtained public datasets and model weights. No training
or benchmark evaluation runs automatically during installation or testing.

## Paper

[Read the research paper (PDF)](papers/main.pdf). It is an independent,
exploratory manuscript; the corrected benchmark campaign has not been executed.
The PDF is the only manuscript artifact distributed in this repository.

## Cite

```bibtex
@misc{trinh2026coground,
  author = {Trinh, Herv\'e},
  title = {CoGround-VLM: Differentiable Patch-Token Co-Clustering
           for Visual Grounding in Vision-Language Models},
  year = {2026},
  howpublished = {Independent research paper},
  url = {https://github.com/hervetrinh/coground-vlm}
}
```

## License

Original code, configuration, documentation, and paper: [CC BY 4.0](LICENSE).
Third-party models, libraries, images, and annotations retain their upstream
licenses and terms. None of their weights or datasets are included.
