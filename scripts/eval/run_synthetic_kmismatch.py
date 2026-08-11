#!/usr/bin/env python
"""K-mismatch robustness: train coground / low_rank with K_pred != K_true=8.
Tests model misspecification - reviewer may ask 'what if K is wrong?'.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from coground_vlm.data.synthetic import generate_block_batch
from coground_vlm.models.coclustering import CoClusteringHead
from coground_vlm.utils.io import ensure_dir, write_json

# import LowRankHead from sibling
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_synthetic_multi import LowRankHead, supported_mass


def run(method, seed, k_pred, k_true, noise, num_steps, batch, eval_batch, dim, hidden, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if method == "coground":
        head = CoClusteringHead(
            dim=dim,
            num_patch_clusters=k_pred,
            num_token_clusters=k_pred,
            hidden_dim=hidden,
            affinity_mode="bilinear",
        ).to(device)
    else:
        head = LowRankHead(dim=dim, rank=k_pred, hidden_dim=hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    lw = {"lambda_cc": 1.0, "lambda_recon": 1.0, "lambda_entropy": 0.01, "lambda_balance": 0.1}
    torch.manual_seed(seed + 100_000)
    for _ in range(num_steps):
        tb = generate_block_batch(
            batch_size=batch,
            num_patches=64,
            num_tokens=32,
            dim=dim,
            k_patch=k_true,
            k_token=k_true,
            noise_std=noise,
            distractor_strength=1.0,
            support_margin=0.5,
            temperature=0.1,
            device=device,
        )
        if method == "coground":
            out = head(
                tb.patch_embeddings,
                tb.token_embeddings,
                external_affinity=tb.affinity,
                loss_weights=lw,
            )
            loss = out.losses["total_coclustering"]
        else:
            aff, s_hat = head(
                tb.patch_embeddings,
                tb.token_embeddings,
                external_affinity=tb.affinity,
            )
            loss = F.mse_loss(s_hat, aff)
        opt.zero_grad()
        loss.backward()
        opt.step()

    torch.manual_seed(seed + 1_000_000)
    eval_batch_ = generate_block_batch(
        batch_size=eval_batch,
        num_patches=64,
        num_tokens=32,
        dim=dim,
        k_patch=k_true,
        k_token=k_true,
        noise_std=noise,
        distractor_strength=1.0,
        support_margin=0.5,
        temperature=0.1,
        device=device,
    )
    with torch.no_grad():
        if method == "coground":
            out = head(
                eval_batch_.patch_embeddings,
                eval_batch_.token_embeddings,
                external_affinity=eval_batch_.affinity,
                loss_weights=lw,
            )
            s_hat = out.reconstructed_affinity
            aff = out.affinity
            support = out.support_scores.detach()
        else:
            aff, s_hat = head(
                eval_batch_.patch_embeddings,
                eval_batch_.token_embeddings,
                external_affinity=eval_batch_.affinity,
            )
            support = torch.sigmoid(s_hat.max(dim=1).values)

    label = eval_batch_.support_labels.cpu().numpy().reshape(-1)
    sup_np = support.detach().cpu().numpy().reshape(-1)
    auroc = float(roc_auc_score(label, sup_np)) if len(np.unique(label)) == 2 else float("nan")
    ap = (
        float(average_precision_score(label, sup_np))
        if len(np.unique(label)) == 2
        else float("nan")
    )
    before = supported_mass(eval_batch_.decoding_logits, eval_batch_.support_mask)
    after = supported_mass(eval_batch_.decoding_logits + s_hat, eval_batch_.support_mask)
    return {
        "support_auroc": auroc,
        "support_ap": ap,
        "mass_shift": after - before,
        "method": method,
        "seed": seed,
        "k_pred": k_pred,
        "k_true": k_true,
        "noise": noise,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    p.add_argument("--k-preds", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--k-true", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.2)
    p.add_argument("--num-steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--out", default="results/runs/synthetic_kmismatch/metrics.json")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(Path(args.out).parent)
    rows = []
    total = len(args.seeds) * len(args.k_preds) * 2
    pbar = tqdm(total=total)
    for seed in args.seeds:
        for kp in args.k_preds:
            for method in ("coground", "low_rank"):
                m = run(
                    method,
                    seed,
                    kp,
                    args.k_true,
                    args.noise,
                    args.num_steps,
                    args.batch,
                    args.eval_batch,
                    args.dim,
                    args.hidden,
                    device,
                )
                rows.append(m)
                pbar.update(1)
    pbar.close()
    write_json(Path(args.out), {"rows": rows})
    # aggregate
    agg = {}
    for method in ("coground", "low_rank"):
        for kp in args.k_preds:
            sel = [r for r in rows if r["method"] == method and r["k_pred"] == kp]
            for k in ("support_auroc", "support_ap", "mass_shift"):
                vals = np.array([r[k] for r in sel if not np.isnan(r[k])])
                if len(vals):
                    agg[f"{method}/k_pred={kp}/{k}_mean"] = float(vals.mean())
                    agg[f"{method}/k_pred={kp}/{k}_std"] = float(vals.std())
    write_json(out_dir / "aggregate.json", agg)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
