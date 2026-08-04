from __future__ import annotations

import torch

from coground_vlm.baselines.decoding import StaticVocabularyPriorProcessor
from coground_vlm.models.coclustering import (
    BlockAwareDecodingPrior,
    CoClusteringHead,
    random_low_rank_control,
)
from coground_vlm.models.grounding_baselines import LearnedAffinityBaseline
from coground_vlm.utils.params import count_parameters


def test_coclustering_shapes_and_losses_are_finite() -> None:
    head = CoClusteringHead(dim=16, num_patch_clusters=4, num_token_clusters=3)
    patches = torch.randn(2, 9, 16)
    tokens = torch.randn(2, 7, 16)
    targets = torch.randint(0, 2, (2, 9, 7)).float()
    out = head(
        patches,
        tokens,
        grounding_targets=targets,
        loss_weights={
            "lambda_recon": 1.0,
            "lambda_entropy": 0.01,
            "lambda_balance": 0.1,
            "lambda_ground": 0.5,
        },
    )
    assert out.affinity.shape == (2, 9, 7)
    assert out.patch_assignments.shape == (2, 9, 4)
    assert out.token_assignments.shape == (2, 7, 3)
    assert out.block_matrix.shape == (2, 4, 3)
    assert out.reconstructed_affinity.shape == (2, 9, 7)
    assert out.support_scores.shape == (2, 7)
    assert torch.isfinite(out.losses["total_coclustering"])


def test_low_rank_control_shape() -> None:
    affinity = torch.randn(2, 5, 4)
    approx = random_low_rank_control(affinity, rank=2)
    assert approx.shape == affinity.shape


def test_hard_assignment_and_external_affinity() -> None:
    head = CoClusteringHead(
        dim=8, num_patch_clusters=2, num_token_clusters=2, assignment_mode="hard"
    )
    patches = torch.randn(1, 4, 8)
    tokens = torch.randn(1, 3, 8)
    affinity = torch.randn(1, 4, 3)
    out = head(patches, tokens, external_affinity=affinity, loss_weights={"lambda_cc": 1.0})
    assert torch.allclose(out.patch_assignments.sum(dim=-1), torch.ones(1, 4))
    assert torch.allclose(out.token_assignments.sum(dim=-1), torch.ones(1, 3))
    assert out.affinity.shape == affinity.shape


def test_block_prior_scores_shape() -> None:
    prior = BlockAwareDecodingPrior(hidden_dim=8, num_patch_clusters=2, num_token_clusters=3)
    hidden = torch.randn(2, 5, 8)
    u = torch.softmax(torch.randn(2, 4, 2), dim=-1)
    v = torch.softmax(torch.randn(2, 5, 3), dim=-1)
    bias = prior(hidden, u, v, lambda_dec=0.5)
    scores = prior.token_prior_scores(hidden, u, v)
    assert bias.shape == (2, 5, 4, 5)
    assert scores.shape == (2, 4, 5)


def test_soft_least_squares_block_is_no_worse_than_occupancy_average() -> None:
    affinity = torch.randn(2, 9, 7)
    u = torch.softmax(torch.randn(2, 9, 4), dim=-1)
    v = torch.softmax(torch.randn(2, 7, 3), dim=-1)
    least_squares = CoClusteringHead(
        dim=8,
        num_patch_clusters=4,
        num_token_clusters=3,
        block_solver="least_squares",
    )
    occupancy = CoClusteringHead(
        dim=8,
        num_patch_clusters=4,
        num_token_clusters=3,
        block_solver="occupancy",
    )
    b_ls = least_squares.block_matrix(affinity, u, v)
    b_occ = occupancy.block_matrix(affinity, u, v)
    reconstructed_ls = torch.einsum("bpk,bkl,btl->bpt", u, b_ls, v)
    reconstructed_occ = torch.einsum("bpk,bkl,btl->bpt", u, b_occ, v)
    error_ls = torch.mean((affinity - reconstructed_ls) ** 2)
    error_occ = torch.mean((affinity - reconstructed_occ) ** 2)
    assert error_ls <= error_occ + 1e-5


def test_identity_block_and_no_simplex_controls() -> None:
    head = CoClusteringHead(
        dim=8,
        num_patch_clusters=3,
        num_token_clusters=3,
        assignment_mode="sigmoid",
        block_ablation="identity",
    )
    output = head(torch.randn(1, 5, 8), torch.randn(1, 4, 8))
    assert torch.allclose(output.block_matrix, torch.eye(3).unsqueeze(0))
    assert not torch.allclose(
        output.patch_assignments.sum(dim=-1),
        torch.ones(1, 5),
    )
    assert torch.isfinite(output.losses["total_coclustering"])


def test_matched_mlp_has_same_parameter_count_as_coclustering_head() -> None:
    co_head = CoClusteringHead(
        dim=16,
        num_patch_clusters=4,
        num_token_clusters=4,
        hidden_dim=12,
        affinity_mode="bilinear",
    )
    baseline = LearnedAffinityBaseline(
        dim=16,
        hidden_dim=12,
        rank=4,
        mode="matched_mlp",
    )
    assert count_parameters(co_head) == count_parameters(baseline)


def test_vocabulary_prior_and_independent_state_shapes() -> None:
    prior = BlockAwareDecodingPrior(
        hidden_dim=8,
        num_patch_clusters=2,
        num_token_clusters=3,
        share_assignments=False,
        assignment_hidden_dim=6,
    )
    hidden = torch.randn(2, 8)
    patches = torch.randn(2, 4, 8)
    tokens = torch.randn(2, 5, 8)
    placeholder_u = torch.zeros(2, 4, 2)
    candidate_v = torch.softmax(torch.randn(11, 3), dim=-1)
    scores = prior.vocabulary_prior_scores(
        hidden,
        placeholder_u,
        candidate_v,
        patch_embeddings=patches,
    )
    placeholder_v = torch.zeros(2, 5, 3)
    bias = prior(
        tokens,
        placeholder_u,
        placeholder_v,
        patch_embeddings=patches,
        token_embeddings=tokens,
    )
    assert scores.shape == (2, 11)
    assert bias.shape == (2, 5, 4, 5)


def test_vocabulary_prior_processor_changes_generation_logits() -> None:
    processor = StaticVocabularyPriorProcessor(
        torch.tensor([[0.0, 1.0, -1.0, 0.5]]),
        weight=0.3,
    )
    base_scores = torch.zeros(1, 4)
    adjusted = processor(torch.zeros(1, 2, dtype=torch.long), base_scores)
    assert torch.allclose(
        adjusted,
        torch.tensor([[0.0, 0.3, -0.3, 0.15]]),
    )
