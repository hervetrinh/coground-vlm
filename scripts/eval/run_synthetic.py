#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from tqdm.auto import trange

from coground_vlm.config import load_config, save_config
from coground_vlm.data.synthetic import generate_block_batch
from coground_vlm.models.coclustering import CoClusteringHead
from coground_vlm.utils.io import ensure_dir, write_csv_row, write_json
from coground_vlm.utils.repro import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled synthetic CoGround study.")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(Path(cfg.output_dir) / str(cfg.experiment_id))
    save_config(cfg, out_dir / "resolved_config.yaml")

    head = CoClusteringHead(
        dim=int(cfg.synthetic.dim),
        num_patch_clusters=int(cfg.synthetic.k_patch),
        num_token_clusters=int(cfg.synthetic.k_token),
        hidden_dim=int(cfg.synthetic.hidden_dim),
        affinity_mode="bilinear",
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg.training.lr))

    for _ in trange(int(cfg.training.max_steps)):
        batch = generate_block_batch(
            batch_size=int(cfg.synthetic.batch_size),
            num_patches=int(cfg.synthetic.num_patches),
            num_tokens=int(cfg.synthetic.num_tokens),
            dim=int(cfg.synthetic.dim),
            k_patch=int(cfg.synthetic.k_patch),
            k_token=int(cfg.synthetic.k_token),
            noise_std=float(cfg.synthetic.noise_std),
            distractor_strength=float(cfg.synthetic.get("distractor_strength", 1.0)),
            support_margin=float(cfg.synthetic.get("support_margin", 0.5)),
            temperature=float(cfg.synthetic.get("assignment_temperature", 0.1)),
            device=device,
        )
        out = head(
            batch.patch_embeddings,
            batch.token_embeddings,
            external_affinity=batch.affinity,
            loss_weights=cfg.losses,
        )
        loss = out.losses["total_coclustering"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    eval_batch = generate_block_batch(
        batch_size=int(cfg.synthetic.eval_batch_size),
        num_patches=int(cfg.synthetic.num_patches),
        num_tokens=int(cfg.synthetic.num_tokens),
        dim=int(cfg.synthetic.dim),
        k_patch=int(cfg.synthetic.k_patch),
        k_token=int(cfg.synthetic.k_token),
        noise_std=float(cfg.synthetic.noise_std),
        distractor_strength=float(cfg.synthetic.get("distractor_strength", 1.0)),
        support_margin=float(cfg.synthetic.get("support_margin", 0.5)),
        temperature=float(cfg.synthetic.get("assignment_temperature", 0.1)),
        device=device,
    )
    with torch.no_grad():
        out = head(
            eval_batch.patch_embeddings,
            eval_batch.token_embeddings,
            external_affinity=eval_batch.affinity,
            loss_weights=cfg.losses,
        )
    patch_pred = out.patch_assignments.argmax(dim=-1).cpu().numpy()
    token_pred = out.token_assignments.argmax(dim=-1).cpu().numpy()
    patch_true = eval_batch.patch_clusters.cpu().numpy()
    token_true = eval_batch.token_clusters.cpu().numpy()
    support_score = out.support_scores.detach().cpu().numpy().reshape(-1)
    support_label = eval_batch.support_labels.cpu().numpy().reshape(-1)
    supported = support_score[support_label == 1]
    unsupported = support_score[support_label == 0]
    before_mass = supported_mass(eval_batch.decoding_logits, eval_batch.support_mask)
    after_mass = supported_mass(
        eval_batch.decoding_logits + out.reconstructed_affinity, eval_batch.support_mask
    )
    metrics = {
        "experiment_id": str(cfg.experiment_id),
        "variant": "synthetic",
        "reconstruction_mse_observed": float(
            torch.mean((out.reconstructed_affinity - out.affinity) ** 2).item()
        ),
        "reconstruction_mse_clean": float(
            torch.mean((out.reconstructed_affinity - eval_batch.clean_affinity) ** 2).item()
        ),
        "patch_cluster_ari": float(
            np.mean(
                [adjusted_rand_score(t, p) for t, p in zip(patch_true, patch_pred, strict=False)]
            )
        ),
        "token_cluster_ari": float(
            np.mean(
                [adjusted_rand_score(t, p) for t, p in zip(token_true, token_pred, strict=False)]
            )
        ),
        "support_score_auroc": safe_auroc(support_label, support_score),
        "support_score_average_precision": float(average_precision(support_label, support_score)),
        "support_score_gap": float(supported.mean() - unsupported.mean())
        if len(supported) and len(unsupported)
        else float("nan"),
        "supported_mass_before": before_mass,
        "supported_mass_after": after_mass,
        "supported_mass_shift": after_mass - before_mass,
        "seed": int(cfg.seed),
        "k_patch": int(cfg.synthetic.k_patch),
        "k_token": int(cfg.synthetic.k_token),
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(
        out_dir / "asset_registry.json",
        {
            "model": {
                "release_name": "none",
                "checkpoint_id": "synthetic",
                "role": "controlled study",
            },
            "dataset": {
                "name": "synthetic_block_recovery",
                "split": "generated",
                "license_or_terms": "repo-generated synthetic data",
                "redistribution": "per repository license",
                "inherited_restrictions": "none",
            },
            "run": {"experiment_id": str(cfg.experiment_id), "seed": int(cfg.seed)},
        },
    )
    write_csv_row(out_dir / "summary_row.csv", metrics)
    torch.save(head.state_dict(), out_dir / "coclustering_head.pt")
    print(f"Wrote {out_dir / 'metrics.json'}")


def supported_mass(logits: torch.Tensor, support_mask: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=1)
    mass = (probs * support_mask.float()).sum(dim=1)
    token_has_support = support_mask.any(dim=1)
    if not bool(token_has_support.any()):
        return 0.0
    return float(mass[token_has_support].mean().item())


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, scores))


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


if __name__ == "__main__":
    main()
