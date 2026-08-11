from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from scipy.stats import ttest_rel


def load_aggregator():
    path = Path(__file__).resolve().parents[1] / "scripts/eval/aggregate_all.py"
    spec = importlib.util.spec_from_file_location("aggregate_all", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_inference_uses_student_distribution_and_valid_json() -> None:
    module = load_aggregator()
    left, right = [0.4, 0.6, 0.7, 0.8], [0.3, 0.4, 0.6, 0.5]
    result = module.paired_t(left, right)
    assert result is not None
    expected = ttest_rel(left, right)
    assert result["t_stat"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue)
    zero = module.paired_t([1.0, 1.0], [1.0, 1.0])
    assert zero["p_value"] == 1.0
    json.dumps(zero, allow_nan=False)


def write_metrics(root: Path, run: str, variant: str, seed: int, dataset: str) -> None:
    path = root / "results/runs" / run / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "variant": variant,
                "seed": seed,
                "dataset": dataset,
                "backbone": "mock",
                "split": "val",
                "layer_tap": 0,
                "spatial_protocol": "patch_centers_v1",
                "patch_pointing_accuracy": 0.4 + seed * 0.01,
            }
        )
    )


def test_aggregation_keeps_transfer_protocols_separate_and_rejects_duplicates(tmp_path) -> None:
    module = load_aggregator()
    module.ROOT = tmp_path
    for dataset in ("refcoco", "refcocog"):
        for variant in ("coground_grounded", "matched_mlp_grounding"):
            for seed in (17, 18):
                write_metrics(tmp_path, f"{dataset}_{variant}_{seed}", variant, seed, dataset)
    result = module.aggregate_grounding("*")
    assert len(result["by_variant"]) == 4
    assert len(result["paired_tests"]) == 2
    write_metrics(tmp_path, "duplicate", "coground_grounded", 17, "refcoco")
    with pytest.raises(ValueError, match="Duplicate seed"):
        module.aggregate_grounding("*")


def test_generation_pairs_require_same_checkpoint(tmp_path) -> None:
    module = load_aggregator()
    module.ROOT = tmp_path
    for enabled in (True, False):
        for seed in (17, 18):
            folder = tmp_path / "results/runs" / f"generation_{enabled}_{seed}"
            folder.mkdir(parents=True)
            (folder / "metrics.json").write_text(
                json.dumps(
                    {
                        "dataset": "pope",
                        "backbone": "mock",
                        "split": "val",
                        "layer_tap": 0,
                        "decoding_mode": "vocab_logit_bias",
                        "decoding_prior_enabled": enabled,
                        "seed": seed,
                        "checkpoint": f"{'A' if enabled else 'B'}/{seed}",
                        "pope_accuracy": 0.5,
                    }
                )
            )
    assert module.aggregate_generation("generation_*")["paired_tests"] == {}
    for seed in (17, 18):
        path = tmp_path / "results/runs" / f"generation_False_{seed}/metrics.json"
        payload = json.loads(path.read_text())
        payload["checkpoint"] = f"A/{seed}"
        path.write_text(json.dumps(payload))
    assert len(module.aggregate_generation("generation_*")["paired_tests"]) == 1
