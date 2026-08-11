from __future__ import annotations

from pathlib import Path

from coground_vlm.config import load_config


def test_synthetic_config_loads() -> None:
    cfg = load_config("configs/synthetic/block_recovery.yaml")
    assert cfg.experiment_id == "synthetic_block_recovery"
    assert cfg.synthetic.k_patch == 8


def test_revision_config_inheritance() -> None:
    cfg = load_config("configs/experiments/revision/ablations/shuffled_grounding.yaml")
    assert cfg.method.block_solver == "least_squares"
    assert cfg.losses.lambda_ground == 1.0
    assert cfg.target_control.mode == "permute_both"
    assert cfg.experiment_id == "revision_gemma4_refcoco_shuffled_grounding"
    assert cfg.runtime.expected_world_size == 2
    assert cfg.training.global_batch_size == 16


def test_transfer_baselines_preserve_backbone_and_dataset() -> None:
    refcocog = load_config("configs/experiments/revision/gemma4_refcocog_matched_mlp.yaml")
    assert refcocog.dataset.name == "refcocog"
    assert refcocog.method.enabled is False
    assert refcocog.grounding_baseline.mode == "matched_mlp"

    paligemma = load_config("configs/experiments/revision/paligemma2_refcoco_bilinear.yaml")
    assert paligemma.backbone.kind == "paligemma2"
    assert paligemma.method.enabled is False
    assert paligemma.grounding_baseline.mode == "bilinear"


def test_relative_config_inheritance(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text("a: 1\nnested:\n  left: 2\n", encoding="utf-8")
    child.write_text(
        "extends: parent.yaml\nnested:\n  right: 3\n",
        encoding="utf-8",
    )
    cfg = load_config(child)
    assert cfg.a == 1
    assert cfg.nested.left == 2
    assert cfg.nested.right == 3
