from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from coground_vlm.baselines.decoding import (
    candidate_vocabulary_assignments,
    generate_with_coclustering_logit_prior,
)
from coground_vlm.models.backbones import FeatureBatch
from coground_vlm.models.coclustering import BlockAwareDecodingPrior, CoClusteringHead


class TinyBackbone:
    """In-memory generation boundary; no weights or dataset downloads."""

    def __init__(self):
        self.embedding = torch.nn.Embedding(11, 8).to(torch.bfloat16)
        self.model = SimpleNamespace(generate=self.generate)
        self.processor = SimpleNamespace(batch_decode=lambda ids, **kw: ids.tolist())
        self.last_scores = None

    def input_embeddings(self):
        return self.embedding

    def prepare_batch(self, batch, device):
        return {"input_ids": torch.tensor([[1, 2]], device=device)}

    def extract_features(self, inputs, **kwargs):
        return FeatureBatch(
            patch_embeddings=torch.ones(1, 4, 8, dtype=torch.bfloat16),
            token_embeddings=self.embedding(inputs["input_ids"]),
        )

    def generate(self, input_ids, logits_processor, **kwargs):
        scores = torch.zeros(1, 11)
        self.last_scores = logits_processor(input_ids, scores)
        return torch.cat([input_ids, self.last_scores.argmax(-1, keepdim=True)], dim=1)


def test_bf16_features_reach_float32_heads_and_real_generation_processor() -> None:
    backbone = TinyBackbone()
    head = CoClusteringHead(dim=8, num_patch_clusters=2)
    prior = BlockAwareDecodingPrior(8, 2, 2)
    generation = OmegaConf.create({"max_new_tokens": 1})
    for weight in (0.0, 0.7):
        output = generate_with_coclustering_logit_prior(
            backbone,
            {"prompt": ["object"]},
            "cpu",
            generation,
            co_head=head,
            decoding_prior=prior,
            method_cfg=OmegaConf.create({}),
            decoding_cfg=OmegaConf.create({"lambda_dec": weight}),
        )
        assert len(output[0]) == 1  # Prompt tokens excluded from scoring.
        assert torch.isfinite(backbone.last_scores).all()
        if weight == 0:
            assert torch.count_nonzero(backbone.last_scores) == 0
        else:
            assert torch.count_nonzero(backbone.last_scores) > 0


def test_independent_prior_uses_its_own_vocabulary_head() -> None:
    backbone = TinyBackbone()
    head = CoClusteringHead(dim=8, num_patch_clusters=2)
    prior = BlockAwareDecodingPrior(8, 2, 2, share_assignments=False)
    with torch.no_grad():
        prior.token_head.net[-1].weight.zero_()
        prior.token_head.net[-1].bias.copy_(torch.tensor([10.0, -10.0]))
    assignments = candidate_vocabulary_assignments(backbone, prior, device="cpu", chunk_size=3)
    assert (assignments[:, 0] > 0.99).all()
    with pytest.raises(ValueError, match="positive"):
        candidate_vocabulary_assignments(backbone, head, device="cpu", chunk_size=0)
