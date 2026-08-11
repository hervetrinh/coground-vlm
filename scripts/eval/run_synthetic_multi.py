#!/usr/bin/env python
"""Multi-seed synthetic block_recovery + structural baselines.

For each requested structure, seed, and noise level, trains four methods on
the same synthetic data:
  1) coground (UBV^T tri-factorization with simplex assignments)
  2) low_rank (PQ^T factorization with matching assignment heads)
  3) random   (untrained head; pure baseline)
  4) nmf      (sklearn nonneg matrix factorization on observed affinity)

Reports mean +/- std for ARI patch/token, AUROC support, supported_mass shift.
Synthetic comparisons are mechanism checks, not real-data grounding evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from coground_vlm.data.synthetic import generate_block_batch
from coground_vlm.models.coclustering import AssignmentHead, CoClusteringHead
from coground_vlm.utils.io import ensure_dir, write_json


def supported_mass(logits, mask):
    p = torch.softmax(logits, dim=-1)
    m = (p * mask).sum(dim=-1).mean().item()
    return float(m)


class LowRankHead(nn.Module):
    """Direct rank-K factorization with heads matched to CoGround's U/V heads."""

    def __init__(self, dim: int, rank: int, hidden_dim: int):
        super().__init__()
        self.patch_head = AssignmentHead(dim, rank, hidden_dim=hidden_dim)
        self.token_head = AssignmentHead(dim, rank, hidden_dim=hidden_dim)
        self.rank = rank

    def forward(self, patch_e, token_e, external_affinity=None):
        affinity = (
            external_affinity
            if external_affinity is not None
            else torch.einsum("bpd,btd->bpt", patch_e, token_e)
        )
        p = self.patch_head.logits(patch_e)
        q = self.token_head.logits(token_e)
        s_hat = torch.einsum("bpr,btr->bpt", p, q) * self.rank**-0.5
        return affinity, s_hat


def fit_nmf_on_affinity(affinity_np, rank):
    """Fit a per-example NMF reference and restore the original affinity scale."""
    out = np.zeros_like(affinity_np)
    for i in range(affinity_np.shape[0]):
        minimum = float(affinity_np[i].min())
        shifted = affinity_np[i] - minimum
        scale = float(shifted.max())
        if scale == 0:
            out[i] = affinity_np[i]
            continue
        normalized = shifted / scale
        # Errors must remain visible; returning the input would mislabel a
        # failed factorization as a successful NMF baseline.
        model = NMF(n_components=rank, init="nndsvda", max_iter=200, random_state=0)
        left = model.fit_transform(normalized)
        out[i] = (left @ model.components_) * scale + minimum
    return out


def run_method(
    method,
    seed,
    noise,
    num_steps,
    batch,
    eval_batch,
    k,
    dim,
    hidden_dim,
    structure,
    device,
):
    """Returns dict of metrics for one method × seed × noise."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Reset the data RNG after method-specific parameter initialization.
    if method in ("coground", "low_rank"):
        head = (
            CoClusteringHead(
                dim=dim,
                num_patch_clusters=k,
                num_token_clusters=k,
                hidden_dim=hidden_dim,
                affinity_mode="bilinear",
            )
            if method == "coground"
            else LowRankHead(dim=dim, rank=k, hidden_dim=hidden_dim)
        ).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        loss_weights = {
            "lambda_cc": 1.0,
            "lambda_recon": 1.0,
            "lambda_entropy": 0.01,
            "lambda_balance": 0.1,
        }
        torch.manual_seed(seed + 100_000)
        for _ in range(num_steps):
            tb = generate_block_batch(
                batch_size=batch,
                num_patches=64,
                num_tokens=32,
                dim=dim,
                k_patch=8,
                k_token=8,
                noise_std=noise,
                distractor_strength=1.0,
                support_margin=0.5,
                temperature=0.1,
                structure=structure,
                device=device,
            )
            if method == "coground":
                out = head(
                    tb.patch_embeddings,
                    tb.token_embeddings,
                    external_affinity=tb.affinity,
                    loss_weights=loss_weights,
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
        # eval
        with torch.no_grad():
            if method == "coground":
                out = head(
                    eval_batch.patch_embeddings,
                    eval_batch.token_embeddings,
                    external_affinity=eval_batch.affinity,
                    loss_weights=loss_weights,
                )
                s_hat = out.reconstructed_affinity
                aff = out.affinity
                u = out.patch_assignments
                v = out.token_assignments
                support = out.support_scores.detach()
            else:
                aff, s_hat = head(
                    eval_batch.patch_embeddings,
                    eval_batch.token_embeddings,
                    external_affinity=eval_batch.affinity,
                )
                # derive cluster assignments by post-hoc kmeans on s_hat
                u = derive_clusters(s_hat, k, dim_axis=1)
                v = derive_clusters(s_hat, k, dim_axis=2)
                # support = sigmoid of max across patches for each token
                support = torch.sigmoid(s_hat.max(dim=1).values)
    elif method == "random":
        head = CoClusteringHead(
            dim=dim,
            num_patch_clusters=k,
            num_token_clusters=k,
            hidden_dim=hidden_dim,
            affinity_mode="bilinear",
        ).to(device)
        loss_weights = {
            "lambda_cc": 1.0,
            "lambda_recon": 1.0,
            "lambda_entropy": 0.01,
            "lambda_balance": 0.1,
        }
        with torch.no_grad():
            out = head(
                eval_batch.patch_embeddings,
                eval_batch.token_embeddings,
                external_affinity=eval_batch.affinity,
                loss_weights=loss_weights,
            )
            s_hat = out.reconstructed_affinity
            aff = out.affinity
            u = out.patch_assignments
            v = out.token_assignments
            support = out.support_scores.detach()
    elif method == "nmf":
        with torch.no_grad():
            aff = eval_batch.affinity
            s_hat_np = fit_nmf_on_affinity(aff.cpu().numpy(), rank=k)
            s_hat = torch.from_numpy(s_hat_np).float().to(device)
            u = derive_clusters(s_hat, k, dim_axis=1)
            v = derive_clusters(s_hat, k, dim_axis=2)
            support = torch.sigmoid(s_hat.max(dim=1).values)

    # metrics
    patch_pred = u.argmax(dim=-1).cpu().numpy()
    token_pred = v.argmax(dim=-1).cpu().numpy()
    patch_true = eval_batch.patch_clusters.cpu().numpy()
    token_true = eval_batch.token_clusters.cpu().numpy()
    support_np = support.detach().cpu().numpy().reshape(-1)
    label_np = eval_batch.support_labels.cpu().numpy().reshape(-1)

    ari_p = float(
        np.mean([adjusted_rand_score(t, p) for t, p in zip(patch_true, patch_pred, strict=False)])
    )
    ari_t = float(
        np.mean([adjusted_rand_score(t, p) for t, p in zip(token_true, token_pred, strict=False)])
    )

    sup, unsup = support_np[label_np == 1], support_np[label_np == 0]
    auroc = (
        float(roc_auc_score(label_np, support_np))
        if len(np.unique(label_np)) == 2
        else float("nan")
    )
    ap = (
        float(average_precision_score(label_np, support_np))
        if len(np.unique(label_np)) == 2
        else float("nan")
    )

    before = supported_mass(eval_batch.decoding_logits, eval_batch.support_mask)
    after = supported_mass(eval_batch.decoding_logits + s_hat, eval_batch.support_mask)
    recon_obs = float(torch.mean((s_hat - aff) ** 2).item())
    recon_clean = float(torch.mean((s_hat - eval_batch.clean_affinity) ** 2).item())

    return {
        "ari_patch": ari_p,
        "ari_token": ari_t,
        "support_auroc": auroc,
        "support_ap": ap,
        "support_gap": float(sup.mean() - unsup.mean())
        if len(sup) and len(unsup)
        else float("nan"),
        "mass_shift": after - before,
        "mass_before": before,
        "mass_after": after,
        "recon_observed": recon_obs,
        "recon_clean": recon_clean,
    }


def derive_clusters(s_hat, k, dim_axis):
    """Simple soft cluster assignments derived from rank-k SVD of s_hat."""
    arr = s_hat.detach()
    if dim_axis == 1:
        # cluster patches: SVD on patch axis
        u_svd, _, _ = torch.linalg.svd(arr, full_matrices=False)  # [B,P,T] -> u_svd [B,P,T]
        u_topk = u_svd[..., :k]
    else:
        _, _, v_svd = torch.linalg.svd(arr, full_matrices=False)
        u_topk = v_svd.transpose(-1, -2)[..., :k]
    # softmax over k axis to get a simplex
    return torch.softmax(u_topk * 5.0, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    parser.add_argument("--noises", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.8])
    parser.add_argument("--methods", nargs="+", default=["coground", "low_rank", "random", "nmf"])
    parser.add_argument(
        "--structures",
        nargs="+",
        choices=["block", "overlapping", "continuous"],
        default=["block"],
    )
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out", default="results/runs/synthetic_multi/metrics.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(Path(args.out).parent)
    rows = []

    total = len(args.structures) * len(args.seeds) * len(args.noises) * len(args.methods)
    pbar = tqdm(total=total)
    for structure in args.structures:
        for seed in args.seeds:
            for noise in args.noises:
                torch.manual_seed(seed)
                np.random.seed(seed)
                eval_batch = generate_block_batch(
                    batch_size=args.eval_batch,
                    num_patches=64,
                    num_tokens=32,
                    dim=args.dim,
                    k_patch=8,
                    k_token=8,
                    noise_std=noise,
                    distractor_strength=1.0,
                    support_margin=0.5,
                    temperature=0.1,
                    structure=structure,
                    device=device,
                )
                for method in args.methods:
                    m = run_method(
                        method,
                        seed,
                        noise,
                        args.num_steps,
                        args.batch,
                        eval_batch,
                        args.k,
                        args.dim,
                        args.hidden,
                        structure,
                        device,
                    )
                    m.update(
                        {
                            "seed": seed,
                            "noise": noise,
                            "method": method,
                            "k": args.k,
                            "structure": structure,
                        }
                    )
                    rows.append(m)
                    pbar.update(1)
    pbar.close()

    write_json(Path(args.out), {"rows": rows})
    # aggregate
    agg = {}
    for structure in args.structures:
        for method in args.methods:
            for noise in args.noises:
                sel = [
                    r
                    for r in rows
                    if r["structure"] == structure and r["method"] == method and r["noise"] == noise
                ]
                for k in (
                    "ari_patch",
                    "ari_token",
                    "support_auroc",
                    "support_ap",
                    "mass_shift",
                    "recon_observed",
                ):
                    vals = np.array([r[k] for r in sel if not np.isnan(r[k])])
                    if len(vals):
                        prefix = f"{structure}/{method}/noise={noise}/{k}"
                        agg[f"{prefix}_mean"] = float(vals.mean())
                        agg[f"{prefix}_std"] = float(vals.std())
    write_json(out_dir / "aggregate.json", agg)
    print(json.dumps(agg, indent=2))
    print(f"\nWrote {args.out} and {out_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
