#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from PIL import Image

from coground_vlm.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a CoGround 2xA100 launch without training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--sample-records", type=int, default=32)
    parser.add_argument("--skip-gpu", action="store_true", help="Only for CPU CI/static checks.")
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def iter_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if len(records) >= limit:
                break
    return records


def validate_gpu(args: argparse.Namespace, cfg: Any) -> list[dict[str, Any]]:
    if args.skip_gpu:
        return []
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")
    if torch.cuda.device_count() != args.world_size:
        raise RuntimeError(
            f"Expected exactly {args.world_size} visible GPUs, found {torch.cuda.device_count()}. "
            "Set COGROUND_CUDA_DEVICES or CUDA_VISIBLE_DEVICES to the two A100 indices."
        )
    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        raise RuntimeError("PyTorch was not built with distributed NCCL support.")
    devices: list[dict[str, Any]] = []
    require_a100 = bool(cfg.runtime.get("require_a100", True)) and not args.allow_non_a100
    for index in range(args.world_size):
        properties = torch.cuda.get_device_properties(index)
        if require_a100 and "A100" not in properties.name.upper():
            raise RuntimeError(f"GPU {index} is {properties.name}, not an A100.")
        if str(cfg.runtime.get("mixed_precision", "no")) == "bf16":
            major, _minor = torch.cuda.get_device_capability(index)
            if major < 8:
                raise RuntimeError(f"GPU {index} compute capability is below bf16-capable Ampere.")
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "memory_gib": round(properties.total_memory / (1024**3), 2),
                "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
            }
        )
    return devices


def validate_manifest(cfg: Any, sample_records: int) -> dict[str, Any]:
    split = str(cfg.dataset.get("train_split", "train"))
    manifest_value = cfg.dataset.get("manifest")
    if cfg.dataset.get("splits") and split in cfg.dataset.splits:
        manifest_value = cfg.dataset.splits[split]
    if manifest_value is None:
        raise ValueError(f"No manifest configured for training split {split!r}.")
    manifest = Path(str(manifest_value))
    if not manifest.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {manifest}. Export it with "
            "scripts/prepare/export_refcoco.py or override dataset.splits.train."
        )
    records = iter_jsonl(manifest, max(1, sample_records))
    if not records:
        raise ValueError(f"Training manifest is empty: {manifest}")
    image_root = Path(str(cfg.dataset.get("image_root", manifest.parent)))
    for index, record in enumerate(records):
        record_id = record.get("id", index)
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Record {record_id!r} has no non-empty prompt.")
        image_value = record.get("image")
        if not image_value:
            raise ValueError(f"Record {record_id!r} has no image path.")
        image_path = Path(str(image_value))
        if not image_path.is_absolute():
            image_path = image_root / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Record {record_id!r} image not found: {image_path}")
        with Image.open(image_path) as image:
            image.verify()
        has_dense_target = record.get("grounding_matrix") is not None
        boxes = record.get("boxes")
        has_token_target = bool(record.get("token_spans") or record.get("grounding_text"))
        if not has_dense_target and (not boxes or not has_token_target):
            raise ValueError(
                f"Record {record_id!r} needs grounding_matrix or boxes plus "
                "token_spans/grounding_text."
            )
        grounding_text = record.get("grounding_text")
        if grounding_text:
            phrases = [grounding_text] if isinstance(grounding_text, str) else grounding_text
            for phrase in phrases:
                if str(phrase).strip() not in prompt:
                    raise ValueError(
                        f"Record {record_id!r} grounding_text={phrase!r} is not verbatim in prompt."
                    )
    return {
        "path": str(manifest.resolve()),
        "sampled_records": len(records),
        "image_root": str(image_root.resolve()),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    expected_world_size = cfg.runtime.get("expected_world_size")
    if expected_world_size is not None and int(expected_world_size) != args.world_size:
        raise ValueError(
            f"runtime.expected_world_size={expected_world_size}, but preflight uses {args.world_size}."
        )
    if cfg.backbone.get("device_map") is not None:
        raise ValueError("backbone.device_map must be null for the DDP launcher.")
    if bool(cfg.backbone.get("load_in_4bit", False)):
        raise ValueError("The validated 2-A100 configuration requires backbone.load_in_4bit=false.")
    effective_batch = (
        int(cfg.training.batch_size)
        * int(cfg.training.get("gradient_accumulation_steps", 1))
        * args.world_size
    )
    configured_global_batch = cfg.training.get("global_batch_size")
    if configured_global_batch is not None and int(configured_global_batch) != effective_batch:
        raise ValueError(
            f"training.global_batch_size={configured_global_batch}, computed={effective_batch}."
        )
    output_root = Path(str(cfg.output_dir))
    disk_root = output_root.resolve()
    while not disk_root.exists():
        disk_root = disk_root.parent
    free_disk_gib = shutil.disk_usage(disk_root).free / (1024**3)
    minimum_free = float(cfg.runtime.get("min_free_disk_gb", 10.0))
    if free_disk_gib < minimum_free:
        raise RuntimeError(
            f"Only {free_disk_gib:.1f} GiB free near {output_root}; {minimum_free:.1f} GiB required."
        )
    report = {
        "status": "ok",
        "config": str(Path(args.config).resolve()),
        "experiment_id": str(cfg.experiment_id),
        "world_size": args.world_size,
        "effective_global_batch_size": effective_batch,
        "mixed_precision": str(cfg.runtime.get("mixed_precision", "no")),
        "model_id": str(cfg.backbone.get("model_id", cfg.backbone.get("kind"))),
        "free_disk_gib": round(free_disk_gib, 2),
        "manifest": validate_manifest(cfg, args.sample_records),
        "gpus": validate_gpu(args, cfg),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
