from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from coground_vlm.data.datasets import (
    ManifestVLDataset,
    apply_grounding_target_control,
    resolve_grounding_token_spans,
)
from coground_vlm.data.synthetic import generate_block_batch


def test_manifest_dataset_loads_image(tmp_path: Path) -> None:
    image = Image.new("RGB", (8, 8), color="white")
    image.save(tmp_path / "image.jpg")
    record = {"id": "x", "image": "image.jpg", "prompt": "What is shown?", "answer": "white"}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = ManifestVLDataset(manifest)
    sample = dataset[0]
    assert sample["id"] == "x"
    assert sample["image_size"] == (8, 8)


def test_grounding_target_permutation_preserves_label_count() -> None:
    torch.manual_seed(17)
    targets = torch.zeros(1, 8, 5)
    targets[:, :3, 1:3] = 1
    controlled = apply_grounding_target_control(targets, mode="permute_both")
    assert controlled is not None
    assert controlled.sum() == targets.sum()
    assert not torch.equal(controlled, targets)


def test_synthetic_stress_structures_have_consistent_shapes() -> None:
    for structure in ("block", "overlapping", "continuous"):
        batch = generate_block_batch(
            batch_size=2,
            num_patches=12,
            num_tokens=7,
            dim=10,
            k_patch=4,
            k_token=3,
            noise_std=0.1,
            structure=structure,
            device="cpu",
        )
        assert batch.clean_affinity.shape == (2, 12, 7)
        assert batch.patch_memberships.shape == (2, 12, 4)
        assert batch.token_memberships.shape == (2, 7, 3)
        assert torch.allclose(
            batch.patch_memberships.sum(dim=-1),
            torch.ones(2, 12),
            atol=1e-5,
        )
        assert torch.allclose(
            batch.token_memberships.sum(dim=-1),
            torch.ones(2, 7),
            atol=1e-5,
        )


class _WhitespaceTokenizer:
    def __init__(self) -> None:
        self.vocabulary = {"find": 10, "the": 11, "red": 12, "car": 13, "now": 14}

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [self.vocabulary[token] for token in text.strip().split()]}


def test_grounding_text_is_aligned_after_multimodal_special_tokens() -> None:
    batch = {"id": ["sample"], "grounding_text": ["the red car"]}
    input_ids = torch.tensor([[99, 98, 10, 11, 12, 13, 14, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])
    spans = resolve_grounding_token_spans(
        batch,
        input_ids=input_ids,
        tokenizer=_WhitespaceTokenizer(),
        attention_mask=attention_mask,
    )
    assert spans == [[[3, 6]]]
