from __future__ import annotations

import pytest
import torch

from coground_vlm.data.datasets import build_grounding_targets
from coground_vlm.data.spatial import box_patch_mask, resolve_patch_grid


def test_training_and_scoring_use_patch_centers_on_rectangular_grid() -> None:
    boxes = [[0.0, 0.0, 0.49, 0.49], [0.7, 0.51, 1.0, 1.0]]
    batch = {"boxes": [boxes], "token_spans": [[[1, 3]]], "image_size": [(120, 80)]}
    targets = build_grounding_targets(batch, 8, 4, "cpu", patch_grid=[2, 4])
    expected = box_patch_mask(boxes=boxes, width=120, height=80, rows=2, cols=4)
    assert targets is not None
    assert torch.equal(targets[0, :, 1].bool(), expected)
    assert torch.equal(targets[0, :, 2].bool(), expected)
    assert targets[0, :, 0].sum() == 0
    assert targets[0, :, 3].sum() == 0


def test_special_tokens_are_not_spatial_targets() -> None:
    batch = {"boxes": [[[0, 0, 1, 1]]], "token_spans": [[[0, 2]]], "image_size": [(80, 120)]}
    targets = build_grounding_targets(batch, 260, 2, "cpu")
    assert resolve_patch_grid(260) == (16, 16)
    assert targets is not None
    assert targets[:, :256].sum() == 512
    assert targets[:, 256:].sum() == 0


def test_unknown_grid_requires_explicit_layout() -> None:
    with pytest.raises(ValueError, match="Unknown spatial token layout"):
        resolve_patch_grid(10)
    assert resolve_patch_grid(10, [2, 5]) == (2, 5)


def test_small_box_does_not_expand_to_neighboring_patch() -> None:
    batch = {
        "boxes": [[[0.01, 0.01, 0.1, 0.1]]],
        "token_spans": [[[0, 1]]],
        "image_size": [(100, 100)],
    }
    target = build_grounding_targets(batch, 4, 2, "cpu")
    assert target is not None
    assert target.sum() == 0
