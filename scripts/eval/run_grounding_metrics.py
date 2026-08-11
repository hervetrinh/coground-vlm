#!/usr/bin/env python
"""Evaluate spatial grounding from cosine, learned, or co-clustered affinity.

The evaluator uses referred token columns when token spans are available,
supports non-square patch grids, and evaluates the union of all reference
boxes. This keeps the learned-baseline and CoGround comparisons on exactly the
same examples and metric implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from coground_vlm.config import load_config, save_config
from coground_vlm.data.datasets import (
    build_dataset,
    build_grounding_targets,
    collate_vl,
    resolve_grounding_token_spans,
)
from coground_vlm.data.spatial import box_patch_mask, resolve_patch_grid
from coground_vlm.models.backbones import build_backbone
from coground_vlm.models.coclustering import coclustering_from_config
from coground_vlm.models.grounding_baselines import grounding_baseline_from_config
from coground_vlm.train.checkpointing import (
    load_coclustering_head,
    load_grounding_baseline,
)
from coground_vlm.utils.assets import write_asset_registry
from coground_vlm.utils.params import count_parameters
from coground_vlm.utils.repro import RunTimer, reproducibility_payload, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", required=True, help="Path to write metrics.json")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def resolve_feature_dim(backbone: torch.nn.Module, cfg: Any) -> int:
    inferred = int(backbone.hidden_size())
    configured = cfg.get("dim", "auto")
    if configured is None or str(configured).lower() == "auto":
        return inferred
    configured_dim = int(configured)
    if configured_dim != inferred:
        raise ValueError(
            f"Configured feature dimension {configured_dim} does not match backbone hidden "
            f"size {inferred}. Set method.dim=auto."
        )
    return configured_dim


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    if bool(cfg.get("lora", {}).get("enabled", False)) or not bool(
        cfg.get("training", {}).get("freeze_backbone", True)
    ):
        raise ValueError(
            "The generic evaluator supports frozen backbones only. Restore trained "
            "backbone/adapters in a model-specific evaluator before scoring this run."
        )
    set_seed(int(cfg.seed))
    timer = RunTimer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(
        cfg.dataset,
        split=cfg.dataset.get("eval_split", cfg.dataset.get("split", "eval")),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_vl,
    )
    backbone = build_backbone(cfg.backbone).to(device).eval()
    feature_dim = resolve_feature_dim(backbone, cfg.method)
    co_head = (
        coclustering_from_config(cfg.method, dim=feature_dim).to(device)
        if bool(cfg.method.get("enabled", False))
        else None
    )
    grounding_baseline_cfg = cfg.get("grounding_baseline", {})
    grounding_baseline = (
        grounding_baseline_from_config(
            grounding_baseline_cfg,
            dim=feature_dim,
        ).to(device)
        if bool(grounding_baseline_cfg.get("enabled", False))
        else None
    )
    if co_head is not None and grounding_baseline is not None:
        raise ValueError("Evaluate CoGround and learned baselines in separate runs.")
    if co_head is not None:
        if not args.checkpoint:
            raise ValueError("A checkpoint is required for CoGround evaluation.")
        load_coclustering_head(args.checkpoint, co_head, map_location=str(device))
        co_head.eval()
    if grounding_baseline is not None:
        if not args.checkpoint:
            raise ValueError("A checkpoint is required for learned-baseline evaluation.")
        load_grounding_baseline(
            args.checkpoint,
            grounding_baseline,
            map_location=str(device),
        )
        grounding_baseline.eval()

    pointing_correct: list[bool] = []
    box_mass: list[float] = []
    box_area_frac: list[float] = []
    recall_at: dict[int, list[bool]] = {1: [], 5: [], 10: []}
    aurocs: list[float] = []
    per_example: list[dict[str, Any]] = []

    for batch in tqdm(loader, desc="grounding"):
        boxes = batch.get("boxes")
        image_sizes = batch.get("image_size")
        if not boxes or not boxes[0] or not image_sizes:
            continue

        inputs = backbone.prepare_batch(batch, device)
        layer_tap = int(
            grounding_baseline_cfg.get(
                "layer_tap",
                cfg.method.get("layer_tap", -1),
            )
        )
        features = backbone.extract_features(
            inputs,
            layer_tap=layer_tap,
            require_attentions=(cfg.method.get("affinity_mode") == "cross_attention"),
        )
        patches = features.patch_embeddings.float()
        tokens = features.token_embeddings.float()
        grounding_targets = build_grounding_targets(
            batch,
            patches.shape[1],
            tokens.shape[1],
            device,
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            tokenizer=backbone.tokenizer(),
            patch_grid=cfg.dataset.get("patch_grid"),
        )

        if co_head is not None:
            co_out = co_head(
                patches,
                tokens,
                cross_attention=(
                    features.cross_attention
                    if cfg.method.get("affinity_mode") == "cross_attention"
                    else None
                ),
                grounding_targets=grounding_targets,
                loss_weights=cfg.losses,
            )
            affinity = co_out.reconstructed_affinity
        elif grounding_baseline is not None:
            affinity = grounding_baseline(patches, tokens).affinity
        else:
            patch_normalized = torch.nn.functional.normalize(patches, dim=-1)
            token_normalized = torch.nn.functional.normalize(tokens, dim=-1)
            affinity = patch_normalized @ token_normalized.transpose(1, 2)

        width, height = image_sizes[0]
        grid_rows, grid_cols = resolve_patch_grid(affinity.shape[1], cfg.dataset.get("patch_grid"))
        usable = grid_rows * grid_cols
        spatial_affinity = affinity[:, :usable, :]
        token_spans = batch.get("token_spans")
        if not token_spans or not any(token_spans):
            if batch.get("grounding_text"):
                token_spans = resolve_grounding_token_spans(
                    batch,
                    input_ids=inputs["input_ids"],
                    tokenizer=backbone.tokenizer(),
                    attention_mask=inputs.get("attention_mask"),
                )
        if token_spans and token_spans[0]:
            referred_tokens = torch.zeros(tokens.shape[1], dtype=torch.bool, device=device)
            for start, end in token_spans[0]:
                referred_tokens[start:end] = True
            spatial_affinity = spatial_affinity[..., referred_tokens]
        patch_scores = spatial_affinity[0].max(dim=-1).values.cpu().numpy()
        patch_distribution = np.exp(patch_scores - patch_scores.max())
        patch_distribution /= patch_distribution.sum()

        in_box = box_patch_mask(
            boxes=boxes[0],
            width=width,
            height=height,
            rows=grid_rows,
            cols=grid_cols,
        ).numpy()

        prediction = int(np.argmax(patch_scores))
        pointing_correct.append(bool(in_box[prediction]))
        mass_in = float(patch_distribution[in_box].sum())
        box_mass.append(mass_in)
        box_area_frac.append(float(in_box.mean()))

        ranking = np.argsort(-patch_scores)
        for k in recall_at:
            recall_at[k].append(bool(in_box[ranking[:k]].any()))
        if in_box.any() and not in_box.all():
            aurocs.append(float(roc_auc_score(in_box.astype(int), patch_scores)))

        per_example.append(
            {
                "id": batch["id"][0],
                "pointing_correct": bool(in_box[prediction]),
                "gt_box_mass": mass_in,
                "box_area_frac": float(in_box.mean()),
                "grid_rows": grid_rows,
                "grid_cols": grid_cols,
                "positive_patches": int(in_box.sum()),
            }
        )

    scoring_head = "cosine"
    scoring_parameters = 0
    if co_head is not None:
        scoring_head = "coclustering"
        scoring_parameters = count_parameters(co_head, trainable_only=True)
    elif grounding_baseline is not None:
        scoring_head = str(grounding_baseline_cfg.get("mode", "learned"))
        scoring_parameters = count_parameters(
            grounding_baseline,
            trainable_only=True,
        )
    metrics = {
        "num_examples": len(pointing_correct),
        "num_auroc_examples": len(aurocs),
        "num_examples_without_positive_patch": sum(
            row["positive_patches"] == 0 for row in per_example
        ),
        "dataset": str(cfg.dataset.name),
        "backbone": str(cfg.backbone.get("model_id", cfg.backbone.get("kind"))),
        "split": str(cfg.dataset.get("eval_split", "eval")),
        "layer_tap": int(grounding_baseline_cfg.get("layer_tap", cfg.method.get("layer_tap", -1))),
        "spatial_protocol": "patch_centers_v1",
        "patch_grid": list(cfg.dataset.patch_grid) if cfg.dataset.get("patch_grid") else None,
        "patch_pointing_accuracy": (float(np.mean(pointing_correct)) if pointing_correct else None),
        "gt_box_support_mass_mean": float(np.mean(box_mass)) if box_mass else None,
        "gt_box_support_mass_std": float(np.std(box_mass)) if box_mass else None,
        "random_pointing_baseline": (float(np.mean(box_area_frac)) if box_area_frac else None),
        "patch_recall_at_1": (float(np.mean(recall_at[1])) if recall_at[1] else None),
        "patch_recall_at_5": (float(np.mean(recall_at[5])) if recall_at[5] else None),
        "patch_recall_at_10": (float(np.mean(recall_at[10])) if recall_at[10] else None),
        "in_box_vs_out_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "experiment_id": str(cfg.experiment_id),
        "variant": str(cfg.get("variant", cfg.experiment_id)),
        "seed": int(cfg.seed),
        "checkpoint": args.checkpoint,
        "k": int(cfg.method.get("k", 0)),
        "scoring_head": scoring_head,
        "trainable_scoring_parameters": scoring_parameters,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output.parent / "resolved_eval_config.yaml")
    write_asset_registry(
        output.parent / "asset_registry.json",
        cfg,
        split=cfg.dataset.get("eval_split", "eval"),
        checkpoint=args.checkpoint,
    )
    (output.parent / "run_metadata.json").write_text(
        json.dumps(
            reproducibility_payload(cfg, {"wall_clock_seconds": timer.elapsed_seconds()}), indent=2
        ),
        encoding="utf-8",
    )
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output.parent / "per_example_grounding.jsonl").write_text(
        "\n".join(json.dumps(record) for record in per_example),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
