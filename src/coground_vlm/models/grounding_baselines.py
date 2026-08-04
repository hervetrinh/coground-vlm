from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from coground_vlm.models.coclustering import (
    AffinityBuilder,
    AssignmentHead,
    grounding_bce_loss,
)


@dataclass
class GroundingBaselineOutput:
    affinity: torch.Tensor
    losses: dict[str, torch.Tensor]


class LearnedAffinityBaseline(nn.Module):
    """Trainable grounding controls evaluated under the same protocol.

    ``bilinear`` learns only the patch-side bilinear projection. ``matched_mlp``
    uses the same bilinear projection and the same two MLP projection heads as
    the co-clustering model, but predicts affinity directly without simplex
    assignments or a block matrix. When patch and token ranks are equal, its
    trainable parameter count exactly matches a CoClusteringHead configured
    with the same dimensions.
    """

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        rank: int,
        mode: str = "matched_mlp",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"bilinear", "matched_mlp"}:
            raise ValueError(f"Unknown learned grounding baseline mode: {mode}")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.mode = mode
        self.affinity = AffinityBuilder(dim=dim, mode="bilinear")
        if mode == "matched_mlp":
            self.patch_projection = AssignmentHead(
                dim,
                rank,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            self.token_projection = AssignmentHead(
                dim,
                rank,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.patch_projection = None
            self.token_projection = None

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        token_embeddings: torch.Tensor,
        *,
        grounding_targets: torch.Tensor | None = None,
        grounding_weight: float = 1.0,
        positive_weight: float | str | None = "auto",
        max_auto_weight: float = 100.0,
    ) -> GroundingBaselineOutput:
        affinity = self.affinity(patch_embeddings, token_embeddings)
        if self.mode == "matched_mlp":
            if self.patch_projection is None or self.token_projection is None:
                raise RuntimeError("Matched MLP projections were not initialized.")
            patch_factors = self.patch_projection.logits(patch_embeddings)
            token_factors = self.token_projection.logits(token_embeddings)
            residual = torch.einsum(
                "bpr,btr->bpt",
                patch_factors,
                token_factors,
            ) * (self.rank**-0.5)
            affinity = affinity + residual

        grounding_bce = affinity.new_tensor(0.0)
        if grounding_targets is not None:
            if grounding_targets.shape != affinity.shape:
                raise ValueError(
                    "grounding_targets must match baseline affinity shape "
                    f"{tuple(affinity.shape)}, got {tuple(grounding_targets.shape)}"
                )
            grounding_bce = grounding_bce_loss(
                affinity,
                grounding_targets,
                positive_weight=positive_weight,
                max_auto_weight=max_auto_weight,
            )
        total = float(grounding_weight) * grounding_bce
        return GroundingBaselineOutput(
            affinity=affinity,
            losses={
                "grounding_bce": grounding_bce,
                "total": total,
            },
        )


def grounding_baseline_from_config(
    cfg: Any,
    *,
    dim: int | None = None,
) -> LearnedAffinityBaseline:
    resolved_dim = int(dim if dim is not None else cfg.dim)
    return LearnedAffinityBaseline(
        dim=resolved_dim,
        hidden_dim=int(cfg.get("hidden_dim", resolved_dim)),
        rank=int(cfg.get("rank", cfg.get("k", 16))),
        mode=str(cfg.get("mode", "matched_mlp")),
        dropout=float(cfg.get("dropout", 0.0)),
    )
