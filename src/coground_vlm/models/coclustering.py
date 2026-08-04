from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoClusterOutput:
    patch_embeddings: torch.Tensor
    token_embeddings: torch.Tensor
    affinity: torch.Tensor
    patch_assignments: torch.Tensor
    token_assignments: torch.Tensor
    block_matrix: torch.Tensor
    reconstructed_affinity: torch.Tensor
    support_scores: torch.Tensor
    prior_scores: torch.Tensor | None
    patch_occupancy: torch.Tensor
    token_occupancy: torch.Tensor
    block_coherence: torch.Tensor
    losses: dict[str, torch.Tensor]


class AssignmentHead(nn.Module):
    """Small projection head producing soft cluster assignments."""

    def __init__(
        self,
        input_dim: int,
        num_clusters: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_clusters),
        )

    def logits(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Return unconstrained assignment scores.

        Keeping the logits accessible lets the parameter-matched learned
        baseline reuse exactly the same projection heads without imposing a
        simplex or a block reconstruction.
        """

        return self.net(x) / max(float(temperature), 1e-6)

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        mode: str = "soft",
    ) -> torch.Tensor:
        logits = self.logits(x, temperature=temperature)
        probs = F.softmax(logits, dim=-1)
        if mode == "soft":
            return probs
        if mode == "hard":
            return F.one_hot(probs.argmax(dim=-1), probs.shape[-1]).to(probs.dtype)
        if mode == "straight_through":
            hard = F.one_hot(probs.argmax(dim=-1), probs.shape[-1]).to(probs.dtype)
            return hard + probs - probs.detach()
        if mode in {"sigmoid", "independent"}:
            # Independent memberships are a controlled ablation of the
            # row-wise simplex constraint. They remain non-negative so the
            # block reconstruction is still well-defined.
            return torch.sigmoid(logits)
        raise ValueError(f"Unknown assignment mode: {mode}")


class AffinityBuilder(nn.Module):
    """Build patch-token affinities with bilinear or provided cross-attention inputs."""

    def __init__(self, dim: int, mode: str = "bilinear"):
        super().__init__()
        if mode not in {"bilinear", "dot", "cosine", "cross_attention"}:
            raise ValueError(f"Unknown affinity mode: {mode}")
        self.mode = mode
        self.proj = nn.Linear(dim, dim, bias=False) if mode == "bilinear" else nn.Identity()
        self.scale = dim**-0.5

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        token_embeddings: torch.Tensor,
        cross_attention: torch.Tensor | None = None,
        external_affinity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "cross_attention":
            if cross_attention is None:
                raise ValueError("cross_attention affinity mode requires cross_attention tensor")
            if cross_attention.ndim == 4:
                cross_attention = cross_attention.mean(dim=1)
            if cross_attention.shape[-2:] == (token_embeddings.shape[1], patch_embeddings.shape[1]):
                cross_attention = cross_attention.transpose(-1, -2)
            expected = (patch_embeddings.shape[1], token_embeddings.shape[1])
            if cross_attention.shape[-2:] != expected:
                raise ValueError(
                    f"Expected patch-token cross-attention axes {expected}, "
                    f"got {tuple(cross_attention.shape[-2:])}. Decoder self-attention "
                    "requires explicit image-token extraction in a backbone-specific adapter."
                )
            return cross_attention.float()
        if self.mode == "cosine":
            patch_embeddings = F.normalize(patch_embeddings.float(), dim=-1)
            token_embeddings = F.normalize(token_embeddings.float(), dim=-1)
            return torch.einsum("bpd,btd->bpt", patch_embeddings, token_embeddings)
        left = self.proj(patch_embeddings)
        return torch.einsum("bpd,btd->bpt", left, token_embeddings) * self.scale


class CoClusteringHead(nn.Module):
    """Differentiable patch-token co-clustering bottleneck."""

    def __init__(
        self,
        dim: int,
        num_patch_clusters: int = 16,
        num_token_clusters: int | None = None,
        hidden_dim: int | None = None,
        affinity_mode: str = "bilinear",
        assignment_mode: str = "soft",
        control_mode: str = "coclustering",
        block_solver: str = "least_squares",
        block_ablation: str = "none",
        block_rcond: float = 1e-5,
        detach_reconstruction_affinity: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if block_solver not in {"least_squares", "occupancy"}:
            raise ValueError(f"Unknown block solver: {block_solver}")
        if block_ablation not in {"none", "identity"}:
            raise ValueError(f"Unknown block ablation: {block_ablation}")
        self.dim = dim
        self.num_patch_clusters = num_patch_clusters
        self.num_token_clusters = num_token_clusters or num_patch_clusters
        self.eps = eps
        self.assignment_mode = assignment_mode
        self.control_mode = control_mode
        self.block_solver = block_solver
        self.block_ablation = block_ablation
        self.block_rcond = block_rcond
        self.detach_reconstruction_affinity = detach_reconstruction_affinity
        self.affinity = AffinityBuilder(dim=dim, mode=affinity_mode)
        self.patch_head = AssignmentHead(dim, self.num_patch_clusters, hidden_dim, dropout)
        self.token_head = AssignmentHead(dim, self.num_token_clusters, hidden_dim, dropout)

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        token_embeddings: torch.Tensor,
        *,
        cross_attention: torch.Tensor | None = None,
        external_affinity: torch.Tensor | None = None,
        grounding_targets: torch.Tensor | None = None,
        loss_weights: dict[str, Any] | None = None,
        temperature: float = 1.0,
        prior_scores: torch.Tensor | None = None,
    ) -> CoClusterOutput:
        weights = loss_weights or {}
        s = (
            external_affinity.float()
            if external_affinity is not None
            else self.affinity(patch_embeddings, token_embeddings, cross_attention=cross_attention)
        )
        u = self.patch_head(patch_embeddings, temperature=temperature, mode=self.assignment_mode)
        v = self.token_head(token_embeddings, temperature=temperature, mode=self.assignment_mode)
        b = self.block_matrix(s, u, v)
        if self.control_mode == "random_low_rank":
            s_hat = random_low_rank_control(
                s,
                rank=min(self.num_patch_clusters, self.num_token_clusters),
            )
        else:
            s_hat = torch.einsum("bpk,bkl,btl->bpt", u, b, v)
        support_source = prior_scores if prior_scores is not None else s_hat
        support = self.support_scores(support_source)
        reconstruction_target = s.detach() if self.detach_reconstruction_affinity else s
        losses = self.losses(
            reconstruction_target,
            s_hat,
            u,
            v,
            grounding_targets,
            weights,
        )
        patch_occ = u.mean(dim=1)
        token_occ = v.mean(dim=1)
        return CoClusterOutput(
            patch_embeddings=patch_embeddings,
            token_embeddings=token_embeddings,
            affinity=s,
            patch_assignments=u,
            token_assignments=v,
            block_matrix=b,
            reconstructed_affinity=s_hat,
            support_scores=support,
            prior_scores=prior_scores,
            patch_occupancy=patch_occ,
            token_occupancy=token_occ,
            block_coherence=block_coherence(s, s_hat, u, v),
            losses=losses,
        )

    def block_matrix(
        self, affinity: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        if self.block_ablation == "identity":
            if self.num_patch_clusters != self.num_token_clusters:
                raise ValueError(
                    "The identity-B ablation requires equal patch and token cluster counts."
                )
            identity = torch.eye(
                self.num_patch_clusters,
                device=affinity.device,
                dtype=affinity.dtype,
            )
            return identity.unsqueeze(0).expand(affinity.shape[0], -1, -1)

        if self.block_solver == "least_squares":
            # For general soft assignments, the least-squares optimum is
            # (U^T U)^+ U^T S V (V^T V)^+. The previous occupancy-normalized
            # expression is exact only for hard one-hot assignments.
            compute_dtype = (
                torch.float32
                if affinity.dtype in {torch.float16, torch.bfloat16}
                else affinity.dtype
            )
            u_compute = u.to(compute_dtype)
            v_compute = v.to(compute_dtype)
            affinity_compute = affinity.to(compute_dtype)
            block_sum_compute = torch.einsum(
                "bpk,bpt,btl->bkl",
                u_compute,
                affinity_compute,
                v_compute,
            )
            patch_gram = torch.einsum("bpk,bpl->bkl", u_compute, u_compute)
            token_gram = torch.einsum("btk,btl->bkl", v_compute, v_compute)
            patch_inverse = torch.linalg.pinv(
                patch_gram,
                rcond=float(self.block_rcond),
            )
            token_inverse = torch.linalg.pinv(
                token_gram,
                rcond=float(self.block_rcond),
            )
            block = patch_inverse @ block_sum_compute @ token_inverse
            return block.to(affinity.dtype)

        block_sum = torch.einsum("bpk,bpt,btl->bkl", u, affinity, v)
        patch_occ = u.sum(dim=1).clamp_min(self.eps)
        token_occ = v.sum(dim=1).clamp_min(self.eps)
        return block_sum / patch_occ.unsqueeze(-1) / token_occ.unsqueeze(1)

    def support_scores(self, reconstructed_affinity: torch.Tensor) -> torch.Tensor:
        token_evidence = reconstructed_affinity.max(dim=1).values
        return torch.sigmoid(token_evidence)

    def losses(
        self,
        affinity: torch.Tensor,
        reconstructed_affinity: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        grounding_targets: torch.Tensor | None,
        weights: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        recon = F.mse_loss(reconstructed_affinity, affinity)
        if self.assignment_mode in {"sigmoid", "independent"}:
            entropy = affinity.new_tensor(0.0)
            balance = affinity.new_tensor(0.0)
        else:
            entropy = assignment_entropy(u).mean() + assignment_entropy(v).mean()
            balance = kl_balance_loss(u) + kl_balance_loss(v)
        out = {
            "reconstruction": recon,
            "entropy": entropy,
            "balance": balance,
            "grounding_bce": affinity.new_tensor(0.0),
        }
        if grounding_targets is not None:
            if grounding_targets.shape != affinity.shape:
                raise ValueError(
                    f"grounding_targets must match affinity shape {tuple(affinity.shape)}, "
                    f"got {tuple(grounding_targets.shape)}"
                )
            out["grounding_bce"] = grounding_bce_loss(
                reconstructed_affinity,
                grounding_targets,
                positive_weight=weights.get("grounding_pos_weight", "auto"),
                max_auto_weight=float(weights.get("grounding_max_pos_weight", 100.0)),
            )
        lcc = affinity.new_tensor(0.0)
        lcc = lcc + float(weights.get("lambda_recon", 1.0)) * out["reconstruction"]
        lcc = lcc + float(weights.get("lambda_entropy", 0.0)) * out["entropy"]
        lcc = lcc + float(weights.get("lambda_balance", 0.0)) * out["balance"]
        out["lcc"] = lcc
        out["total_coclustering"] = (
            float(weights.get("lambda_cc", 1.0)) * lcc
            + float(weights.get("lambda_ground", 0.0)) * out["grounding_bce"]
        )
        return out


class BlockAwareDecodingPrior(nn.Module):
    """Token-conditioned block prior that returns a patch-token attention bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_patch_clusters: int,
        num_token_clusters: int,
        *,
        share_assignments: bool = True,
        assignment_hidden_dim: int | None = None,
        assignment_mode: str = "soft",
    ):
        super().__init__()
        self.num_patch_clusters = num_patch_clusters
        self.num_token_clusters = num_token_clusters
        self.share_assignments = share_assignments
        self.assignment_mode = assignment_mode
        self.patch_head = (
            None
            if share_assignments
            else AssignmentHead(
                hidden_dim,
                num_patch_clusters,
                hidden_dim=assignment_hidden_dim,
            )
        )
        self.token_head = (
            None
            if share_assignments
            else AssignmentHead(
                hidden_dim,
                num_token_clusters,
                hidden_dim=assignment_hidden_dim,
            )
        )
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_patch_clusters * num_token_clusters),
        )

    def forward(
        self,
        token_hidden: torch.Tensor,
        patch_assignments: torch.Tensor,
        token_assignments: torch.Tensor,
        patch_embeddings: torch.Tensor | None = None,
        token_embeddings: torch.Tensor | None = None,
        lambda_dec: float = 1.0,
        stopgrad_assignments: bool = True,
    ) -> torch.Tensor:
        patch_assignments, token_assignments = self.resolve_assignments(
            patch_assignments,
            token_assignments,
            patch_embeddings=patch_embeddings,
            token_embeddings=token_embeddings,
        )
        if stopgrad_assignments:
            patch_assignments = patch_assignments.detach()
            token_assignments = token_assignments.detach()
        matrix = self.net(token_hidden).view(
            *token_hidden.shape[:-1], self.num_patch_clusters, self.num_token_clusters
        )
        patch_block = torch.einsum("bpk,bjkl->bjpl", patch_assignments, matrix)
        bias = torch.einsum("bjpl,btl->bjpt", patch_block, token_assignments)
        return float(lambda_dec) * bias

    def token_prior_scores(
        self,
        token_hidden: torch.Tensor,
        patch_assignments: torch.Tensor,
        token_assignments: torch.Tensor,
        patch_embeddings: torch.Tensor | None = None,
        token_embeddings: torch.Tensor | None = None,
        stopgrad_assignments: bool = True,
    ) -> torch.Tensor:
        bias = self.forward(
            token_hidden,
            patch_assignments,
            token_assignments,
            patch_embeddings=patch_embeddings,
            token_embeddings=token_embeddings,
            lambda_dec=1.0,
            stopgrad_assignments=stopgrad_assignments,
        )
        # Use each decoding position's own textual column when possible. Shape: B x P x T.
        diag_len = min(bias.shape[1], bias.shape[-1])
        scores = bias.new_zeros(bias.shape[0], bias.shape[2], bias.shape[-1])
        scores[..., :diag_len] = bias[:, :diag_len, :, :diag_len].diagonal(dim1=1, dim2=3)
        if diag_len < bias.shape[-1]:
            scores[..., diag_len:] = bias[:, -1, :, diag_len:]
        return scores

    def vocabulary_prior_scores(
        self,
        token_hidden: torch.Tensor,
        patch_assignments: torch.Tensor,
        candidate_token_assignments: torch.Tensor,
        *,
        patch_embeddings: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Map the block state to an explicit next-token vocabulary bias.

        This is a real generation-time intervention: the returned score has
        one value per vocabulary item and can be added directly to the model
        logits. It is deliberately named a logit prior rather than an
        attention hook.
        """

        if token_hidden.ndim == 3:
            token_hidden = token_hidden[:, -1, :]
        if not self.share_assignments:
            if patch_embeddings is None or self.patch_head is None:
                raise ValueError("Independent decoding assignments require patch_embeddings.")
            patch_assignments = self.patch_head(
                patch_embeddings,
                mode=self.assignment_mode,
            )
        matrix = self.net(token_hidden).view(
            token_hidden.shape[0],
            self.num_patch_clusters,
            self.num_token_clusters,
        )
        patch_cluster_support = torch.einsum(
            "bpk,bkl->bpl",
            patch_assignments,
            matrix,
        )
        visual_support = patch_cluster_support.max(dim=1).values
        scores = torch.einsum(
            "bl,vl->bv",
            visual_support.float(),
            candidate_token_assignments.float(),
        )
        if normalize:
            scores = scores - scores.mean(dim=-1, keepdim=True)
            scores = scores / scores.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return scores

    def resolve_assignments(
        self,
        patch_assignments: torch.Tensor,
        token_assignments: torch.Tensor,
        *,
        patch_embeddings: torch.Tensor | None,
        token_embeddings: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.share_assignments:
            return patch_assignments, token_assignments
        if (
            patch_embeddings is None
            or token_embeddings is None
            or self.patch_head is None
            or self.token_head is None
        ):
            raise ValueError("Independent decoding assignments require patch and token embeddings.")
        return (
            self.patch_head(patch_embeddings, mode=self.assignment_mode),
            self.token_head(token_embeddings, mode=self.assignment_mode),
        )


def assignment_entropy(assignments: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(assignments * assignments.clamp_min(eps).log()).sum(dim=-1)


def grounding_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_weight: float | str | None = "auto",
    max_auto_weight: float = 100.0,
) -> torch.Tensor:
    target = targets.float()
    pos_weight = None
    if isinstance(positive_weight, str):
        if positive_weight.lower() != "auto":
            raise ValueError(f"Unknown grounding positive weight: {positive_weight}")
        positives = target.sum().detach()
        negatives = target.numel() - positives
        if float(positives.item()) > 0:
            ratio = negatives / positives
            pos_weight = ratio.clamp(min=1.0, max=float(max_auto_weight)).to(
                device=logits.device,
                dtype=logits.dtype,
            )
    elif positive_weight is not None:
        pos_weight = logits.new_tensor(float(positive_weight))
    return F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
    )


def balance_loss(assignments: torch.Tensor) -> torch.Tensor:
    mean_occ = assignments.mean(dim=1)
    target = torch.full_like(mean_occ, 1.0 / assignments.shape[-1])
    return F.mse_loss(mean_occ, target)


def kl_balance_loss(assignments: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean_occ = assignments.mean(dim=1).clamp_min(eps)
    target = torch.full_like(mean_occ, 1.0 / assignments.shape[-1])
    return (mean_occ * (mean_occ / target).log()).sum(dim=-1).mean()


def block_coherence(
    affinity: torch.Tensor,
    reconstructed_affinity: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    residual = (affinity - reconstructed_affinity).pow(2)
    weighted = torch.einsum("bpk,bpt,btl->bkl", u, residual, v)
    occ = torch.einsum("bpk,btl->bkl", u, v).clamp_min(eps)
    return 1.0 / (weighted / occ).mean(dim=(1, 2)).clamp_min(eps)


def random_low_rank_control(affinity: torch.Tensor, rank: int) -> torch.Tensor:
    """Legacy name for a deterministic truncated-SVD reconstruction reference."""
    u, s, vh = torch.linalg.svd(affinity.float(), full_matrices=False)
    rank = min(rank, s.shape[-1])
    return torch.einsum("bpr,br,brt->bpt", u[..., :rank], s[..., :rank], vh[..., :rank, :])


def coclustering_from_config(cfg: Any, *, dim: int | None = None) -> CoClusteringHead:
    return CoClusteringHead(
        dim=int(dim if dim is not None else cfg.dim),
        num_patch_clusters=int(cfg.get("num_patch_clusters", cfg.get("k", 16))),
        num_token_clusters=int(cfg.get("num_token_clusters", cfg.get("k", 16))),
        hidden_dim=int(cfg.hidden_dim) if cfg.get("hidden_dim") else None,
        affinity_mode=str(cfg.get("affinity_mode", "bilinear")),
        assignment_mode=str(cfg.get("assignment_mode", "soft")),
        control_mode=str(cfg.get("control", "coclustering")),
        block_solver=str(cfg.get("block_solver", "least_squares")),
        block_ablation=str(cfg.get("block_ablation", "none")),
        block_rcond=float(cfg.get("block_rcond", 1e-5)),
        detach_reconstruction_affinity=bool(cfg.get("detach_reconstruction_affinity", True)),
        dropout=float(cfg.get("dropout", 0.0)),
    )


def decoding_prior_from_config(cfg: Any, hidden_dim: int | None = None) -> BlockAwareDecodingPrior:
    method = cfg.method if hasattr(cfg, "method") else cfg
    decoding = cfg.decoding if hasattr(cfg, "decoding") else cfg
    return BlockAwareDecodingPrior(
        hidden_dim=int(hidden_dim or decoding.get("hidden_dim", method.dim)),
        num_patch_clusters=int(method.get("num_patch_clusters", method.get("k", 16))),
        num_token_clusters=int(method.get("num_token_clusters", method.get("k", 16))),
        share_assignments=bool(decoding.get("share_assignments", True)),
        assignment_hidden_dim=(
            int(decoding.get("assignment_hidden_dim"))
            if decoding.get("assignment_hidden_dim")
            else None
        ),
        assignment_mode=str(decoding.get("assignment_mode", "soft")),
    )
