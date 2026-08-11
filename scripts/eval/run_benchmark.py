#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from coground_vlm.baselines import build_baseline, generate_with_baseline
from coground_vlm.config import load_config, save_config
from coground_vlm.data.datasets import build_dataset, build_grounding_targets, collate_vl
from coground_vlm.eval.metrics import summarize_predictions, table_row
from coground_vlm.models.backbones import build_backbone
from coground_vlm.models.coclustering import coclustering_from_config, decoding_prior_from_config
from coground_vlm.train.checkpointing import load_coclustering_head, load_decoding_prior
from coground_vlm.utils.assets import write_asset_registry
from coground_vlm.utils.io import ensure_dir, write_csv_row, write_json, write_jsonl
from coground_vlm.utils.repro import RunTimer, gpu_memory_mb, reproducibility_payload, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CoGround-VLM benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def resolve_feature_dim(backbone: torch.nn.Module, cfg) -> int:
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
    output_dir = ensure_dir(Path(cfg.output_dir) / str(cfg.experiment_id))
    save_config(cfg, output_dir / "resolved_eval_config.yaml")

    dataset = build_dataset(
        cfg.dataset, split=cfg.dataset.get("eval_split", cfg.dataset.get("split", "eval"))
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        num_workers=int(cfg.eval.get("num_workers", 4)),
        collate_fn=collate_vl,
    )
    backbone = build_backbone(cfg.backbone).to(device).eval()
    feature_dim = resolve_feature_dim(backbone, cfg.method)
    co_head = (
        coclustering_from_config(cfg.method, dim=feature_dim).to(device)
        if bool(cfg.method.get("enabled", False))
        else None
    )
    decoding_prior = (
        decoding_prior_from_config(cfg, hidden_dim=feature_dim).to(device)
        if co_head is not None and bool(cfg.decoding.get("enabled", False))
        else None
    )
    if co_head is not None:
        if not args.checkpoint:
            raise ValueError("A checkpoint is required for CoGround evaluation.")
        load_coclustering_head(args.checkpoint, co_head, map_location=str(device))
        co_head.eval()
    if decoding_prior is not None:
        if not args.checkpoint:
            raise ValueError("A checkpoint is required when decoding.enabled=true.")
        load_decoding_prior(args.checkpoint, decoding_prior, map_location=str(device))
        decoding_prior.eval()
    baseline = build_baseline(cfg.get("baseline"))

    records = []
    for batch in tqdm(loader):
        predictions = generate_with_baseline(
            backbone,
            batch,
            device,
            cfg.generation,
            baseline,
            cfg.get("decoding"),
            co_head=co_head,
            decoding_prior=decoding_prior,
            method_cfg=cfg.method,
        )
        support = [None] * len(predictions)
        coherence = [None] * len(predictions)
        if co_head is not None:
            inputs = backbone.prepare_batch(batch, device)
            features = backbone.extract_features(
                inputs,
                layer_tap=int(cfg.method.layer_tap),
                require_attentions=(cfg.method.get("affinity_mode") == "cross_attention"),
            )
            features.patch_embeddings = features.patch_embeddings.float()
            features.token_embeddings = features.token_embeddings.float()
            grounding_targets = build_grounding_targets(
                batch,
                features.patch_embeddings.shape[1],
                features.token_embeddings.shape[1],
                device,
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                tokenizer=backbone.tokenizer(),
                patch_grid=cfg.dataset.get("patch_grid"),
            )
            first_pass = co_head(
                features.patch_embeddings,
                features.token_embeddings,
                cross_attention=features.cross_attention
                if cfg.method.get("affinity_mode") == "cross_attention"
                else None,
                grounding_targets=grounding_targets,
                loss_weights=cfg.losses,
            )
            prior_scores = None
            if decoding_prior is not None:
                prior_scores = decoding_prior.token_prior_scores(
                    features.token_embeddings,
                    first_pass.patch_assignments,
                    first_pass.token_assignments,
                    patch_embeddings=features.patch_embeddings,
                    token_embeddings=features.token_embeddings,
                    stopgrad_assignments=bool(cfg.decoding.get("stopgrad_assignments", True)),
                )
            co_out = co_head(
                features.patch_embeddings,
                features.token_embeddings,
                cross_attention=features.cross_attention
                if cfg.method.get("affinity_mode") == "cross_attention"
                else None,
                grounding_targets=grounding_targets,
                loss_weights=cfg.losses,
                prior_scores=prior_scores,
            )
            support = co_out.support_scores.detach().float().mean(dim=1).cpu().tolist()
            coherence = co_out.block_coherence.detach().float().cpu().tolist()
        for idx, prediction in enumerate(predictions):
            records.append(
                {
                    "id": batch["id"][idx],
                    "dataset": str(cfg.dataset.get("name", "manifest")),
                    "metadata": batch.get("metadata", [None] * len(predictions))[idx],
                    "image_path": batch.get("image_path", [None] * len(predictions))[idx],
                    "prompt": batch["prompt"][idx],
                    "prediction": prediction,
                    "answer": batch.get("answer", [None] * len(predictions))[idx],
                    "label_text": batch.get("label_text", [None] * len(predictions))[idx],
                    "boxes": batch.get("boxes", [None] * len(predictions))[idx],
                    "point": batch.get("point", [None] * len(predictions))[idx],
                    "predicted_box": batch.get("predicted_box", [None] * len(predictions))[idx],
                    "ranked_boxes": batch.get("ranked_boxes", [None] * len(predictions))[idx],
                    "pair_id": batch.get("pair_id", [None] * len(predictions))[idx],
                    "group_id": batch.get("group_id", [None] * len(predictions))[idx],
                    "hallucinated_objects": batch.get(
                        "hallucinated_objects", [None] * len(predictions)
                    )[idx],
                    "mentioned_objects": batch.get("mentioned_objects", [None] * len(predictions))[
                        idx
                    ],
                    "ground_truth_objects": batch.get(
                        "ground_truth_objects",
                        [None] * len(predictions),
                    )[idx],
                    "candidate_objects": batch.get(
                        "candidate_objects",
                        [None] * len(predictions),
                    )[idx],
                    "support_mean": support[idx],
                    "block_coherence": coherence[idx],
                    "unsupported_label": batch.get("unsupported_label", [None] * len(predictions))[
                        idx
                    ],
                    "baseline": baseline.metadata(),
                }
            )

    metrics = summarize_predictions(records)
    metrics.update(
        {
            "experiment_id": str(cfg.experiment_id),
            "variant": str(cfg.get("variant", cfg.experiment_id)),
            "dataset": str(cfg.dataset.get("name", "manifest")),
            "decoding_prior_enabled": bool(cfg.decoding.get("enabled", False)),
            "decoding_mode": str(cfg.decoding.get("mode", "none")),
            "lambda_dec": float(cfg.decoding.get("lambda_dec", 0.0)),
            "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
            "backbone": str(cfg.backbone.get("model_id", cfg.backbone.get("kind"))),
            "split": str(cfg.dataset.get("eval_split", "eval")),
            "layer_tap": int(cfg.method.get("layer_tap", -1)),
        }
    )
    metrics.update(
        reproducibility_payload(
            cfg,
            {
                "wall_clock_seconds": timer.elapsed_seconds(),
                "max_gpu_memory_mb": gpu_memory_mb(),
            },
        )
    )
    raw_path = output_dir / "raw_predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    row_path = output_dir / "summary_row.csv"
    write_jsonl(raw_path, records)
    write_json(metrics_path, metrics)
    write_csv_row(
        row_path,
        table_row(
            cfg=cfg,
            metrics=metrics,
            output_paths={"raw_predictions": str(raw_path), "metrics_json": str(metrics_path)},
            baseline=baseline.metadata()["baseline"],
        ),
    )
    write_asset_registry(
        output_dir / "asset_registry.json",
        cfg,
        split=cfg.dataset.get("eval_split", cfg.dataset.get("split", "eval")),
        checkpoint=args.checkpoint,
    )
    print(f"Wrote {raw_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {row_path}")


if __name__ == "__main__":
    main()
