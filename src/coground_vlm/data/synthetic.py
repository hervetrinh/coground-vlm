from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SyntheticBatch:
    patch_embeddings: torch.Tensor
    token_embeddings: torch.Tensor
    patch_clusters: torch.Tensor
    token_clusters: torch.Tensor
    patch_memberships: torch.Tensor
    token_memberships: torch.Tensor
    clean_affinity: torch.Tensor
    affinity: torch.Tensor
    decoding_logits: torch.Tensor
    support_labels: torch.Tensor
    support_mask: torch.Tensor


def generate_block_batch(
    *,
    batch_size: int,
    num_patches: int,
    num_tokens: int,
    dim: int,
    k_patch: int,
    k_token: int,
    noise_std: float,
    distractor_strength: float = 1.0,
    support_margin: float = 0.5,
    temperature: float = 0.1,
    structure: str = "block",
    overlap_strength: float = 0.35,
    device: torch.device | str,
) -> SyntheticBatch:
    if structure not in {"block", "overlapping", "continuous"}:
        raise ValueError(f"Unknown synthetic structure: {structure}")
    if not 0.0 <= overlap_strength <= 0.5:
        raise ValueError("overlap_strength must be between 0 and 0.5.")
    if structure == "overlapping" and (k_patch < 2 or k_token < 2):
        raise ValueError("Overlapping memberships require at least two clusters per axis.")

    patch_proto = torch.randn(k_patch, dim, device=device)
    token_proto = torch.randn(k_token, dim, device=device)
    if structure in {"block", "overlapping"}:
        block = -0.5 * torch.ones(k_patch, k_token, device=device)
        high_patch = max(1, k_patch // 3)
        high_token = max(1, k_token // 3)
        block[:high_patch, :high_token] = 2.0 + torch.rand(
            high_patch,
            high_token,
            device=device,
        )
        sparse_extra = torch.rand(k_patch, k_token, device=device) < 0.08
        block = torch.where(sparse_extra, block + 1.5, block)
        patch_clusters = torch.randint(
            0,
            k_patch,
            (batch_size, num_patches),
            device=device,
        )
        token_clusters = torch.randint(
            0,
            k_token,
            (batch_size, num_tokens),
            device=device,
        )
        patch_memberships = torch.nn.functional.one_hot(
            patch_clusters,
            num_classes=k_patch,
        ).float()
        token_memberships = torch.nn.functional.one_hot(
            token_clusters,
            num_classes=k_token,
        ).float()
        if structure == "overlapping":
            patch_secondary = (
                patch_clusters
                + torch.randint(
                    1,
                    k_patch,
                    patch_clusters.shape,
                    device=device,
                )
            ) % k_patch
            token_secondary = (
                token_clusters
                + torch.randint(
                    1,
                    k_token,
                    token_clusters.shape,
                    device=device,
                )
            ) % k_token
            patch_memberships *= 1.0 - overlap_strength
            token_memberships *= 1.0 - overlap_strength
            patch_memberships.scatter_add_(
                -1,
                patch_secondary.unsqueeze(-1),
                torch.full(
                    (*patch_secondary.shape, 1),
                    overlap_strength,
                    device=device,
                ),
            )
            token_memberships.scatter_add_(
                -1,
                token_secondary.unsqueeze(-1),
                torch.full(
                    (*token_secondary.shape, 1),
                    overlap_strength,
                    device=device,
                ),
            )
        patch_embeddings = patch_memberships @ patch_proto
        token_embeddings = token_memberships @ token_proto
        clean_affinity = torch.einsum(
            "bpk,kl,btl->bpt",
            patch_memberships,
            block,
            token_memberships,
        )
    else:
        # A continuous low-rank generator deliberately violates the
        # block-constant assumption. It is a stress test in which an
        # unconstrained low-rank baseline should be competitive.
        latent_dim = max(2, min(k_patch, k_token))
        patch_latent = torch.randn(
            batch_size,
            num_patches,
            latent_dim,
            device=device,
        )
        token_latent = torch.randn(
            batch_size,
            num_tokens,
            latent_dim,
            device=device,
        )
        patch_map = torch.randn(latent_dim, dim, device=device) / latent_dim**0.5
        token_map = torch.randn(latent_dim, dim, device=device) / latent_dim**0.5
        patch_embeddings = patch_latent @ patch_map
        token_embeddings = token_latent @ token_map
        clean_affinity = (
            torch.einsum(
                "bpr,btr->bpt",
                patch_latent,
                token_latent,
            )
            / latent_dim**0.5
        )
        patch_membership_map = torch.randn(
            latent_dim,
            k_patch,
            device=device,
        )
        token_membership_map = torch.randn(
            latent_dim,
            k_token,
            device=device,
        )
        patch_memberships = torch.softmax(
            patch_latent @ patch_membership_map,
            dim=-1,
        )
        token_memberships = torch.softmax(
            token_latent @ token_membership_map,
            dim=-1,
        )
        patch_clusters = patch_memberships.argmax(dim=-1)
        token_clusters = token_memberships.argmax(dim=-1)

    patch_embeddings = patch_embeddings + noise_std * torch.randn_like(patch_embeddings)
    token_embeddings = token_embeddings + noise_std * torch.randn_like(token_embeddings)
    affinity = clean_affinity + noise_std * torch.randn_like(clean_affinity)
    distractors = distractor_strength * torch.rand_like(clean_affinity)
    if structure == "continuous":
        mean = clean_affinity.mean(dim=(1, 2), keepdim=True)
        std = clean_affinity.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        support_mask = clean_affinity > mean + float(support_margin) * std
    else:
        support_mask = clean_affinity > support_margin
    distractors = distractors.masked_fill(support_mask, 0.0)
    decoding_logits = affinity + distractors
    support_labels = support_mask.any(dim=1).long()
    if temperature > 0:
        patch_embeddings = patch_embeddings + temperature * torch.randn_like(patch_embeddings)
        token_embeddings = token_embeddings + temperature * torch.randn_like(token_embeddings)
    return SyntheticBatch(
        patch_embeddings=patch_embeddings,
        token_embeddings=token_embeddings,
        patch_clusters=patch_clusters,
        token_clusters=token_clusters,
        patch_memberships=patch_memberships,
        token_memberships=token_memberships,
        clean_affinity=clean_affinity,
        affinity=affinity,
        decoding_logits=decoding_logits,
        support_labels=support_labels,
        support_mask=support_mask,
    )
