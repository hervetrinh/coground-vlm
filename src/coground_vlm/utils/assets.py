from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from coground_vlm.utils.io import write_json

KNOWN_TERMS = {
    "google/paligemma2-3b-pt-896": {
        "family": "PaliGemma 2",
        "license_or_terms": "Hugging Face model card terms; access-gated by Google",
        "redistribution": "check model card before redistributing adapters or outputs",
        "inherited_restrictions": "Gemma/PaliGemma release terms",
    },
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": {
        "family": "LLaVA-OneVision",
        "license_or_terms": "Hugging Face model card terms",
        "redistribution": "check model card and base Qwen2 terms",
        "inherited_restrictions": "base LLM and vision encoder terms",
    },
    "google/gemma-4-E2B-it": {
        "family": "Gemma 4 E2B",
        "license_or_terms": "Apache 2.0 per Google Gemma 4 Hugging Face model card",
        "redistribution": "model weights are upstream assets; do not redistribute unless Google/HF terms permit",
        "inherited_restrictions": "Gemma 4 model card and Apache 2.0 terms",
    },
    "google/gemma-4-E4B-it": {
        "family": "Gemma 4 E4B",
        "license_or_terms": "Apache 2.0 per Google Gemma 4 Hugging Face model card",
        "redistribution": "model weights are upstream assets; do not redistribute unless Google/HF terms permit",
        "inherited_restrictions": "Gemma 4 model card and Apache 2.0 terms",
    },
}


DATASET_TERMS = {
    "refcoco": "REFER repository and annotation distribution terms; inherits COCO image restrictions",
    "refcoco+": "REFER repository and annotation distribution terms; inherits COCO image restrictions",
    "refcocog": "REFER repository and annotation distribution terms; inherits COCO image restrictions",
    "flickr30k_entities": "Flickr30k Entities terms; inherits Flickr30k image restrictions",
    "pope": "POPE benchmark terms; inherits upstream image restrictions",
    "coco_chair": "COCO annotations and website CC BY 4.0; images subject to Flickr/source-image terms; plus CHAIR evaluation protocol",
    "hallusionbench": "HallusionBench release terms",
    "gqa": "GQA terms; inherits Visual Genome image restrictions",
    "vqav2": "VQAv2 terms; inherits COCO image restrictions",
    "winoground": "Winoground terms",
    "sugarcrepe": "SugarCrepe terms",
    "aro": "ARO terms; inherits upstream image restrictions",
}


def file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_asset_registry(
    cfg: Any, *, split: str | None = None, checkpoint: str | None = None
) -> dict[str, Any]:
    model_id = str(cfg.backbone.get("model_id", cfg.backbone.get("kind", "unknown")))
    dataset_name = (
        str(cfg.dataset.get("name", "unknown")).lower() if cfg.get("dataset") else "synthetic"
    )
    manifest = None
    if cfg.get("dataset"):
        if split and cfg.dataset.get("splits") and split in cfg.dataset.splits:
            manifest = cfg.dataset.splits[split]
        else:
            manifest = cfg.dataset.get("manifest")
    model_terms = KNOWN_TERMS.get(
        model_id,
        {
            "family": cfg.backbone.get("kind", "unknown") if cfg.get("backbone") else "none",
            "license_or_terms": "TODO: record official release terms",
            "redistribution": "TODO",
            "inherited_restrictions": "TODO",
        },
    )
    return {
        "model": {
            "release_name": model_terms["family"],
            "checkpoint_id": model_id,
            "role": "backbone",
            "license_or_terms": model_terms["license_or_terms"],
            "redistribution": model_terms["redistribution"],
            "inherited_restrictions": model_terms["inherited_restrictions"],
        },
        "dataset": {
            "name": dataset_name,
            "split": split,
            "manifest": str(manifest) if manifest else None,
            "manifest_sha256": file_sha256(manifest),
            "license_or_terms": DATASET_TERMS.get(dataset_name, "TODO: record benchmark terms"),
            "redistribution": "do not redistribute upstream images unless source terms permit it",
            "inherited_restrictions": "upstream image/source annotations",
        },
        "run": {
            "experiment_id": str(cfg.get("experiment_id", "unknown")),
            "checkpoint": checkpoint,
            "seed": int(cfg.get("seed", 0)),
        },
    }


def write_asset_registry(
    path: str | Path, cfg: Any, *, split: str | None = None, checkpoint: str | None = None
) -> None:
    write_json(path, build_asset_registry(cfg, split=split, checkpoint=checkpoint))
