#!/usr/bin/env python
"""Aggregate corrected synthetic and grounding campaigns without mixing seeds."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    return json.loads(source.read_text(encoding="utf-8"))


def stats(values: list[float | None]) -> dict[str, float | int]:
    clean = [
        value
        for value in values
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    if not clean:
        return {"n": 0}
    if len(clean) == 1:
        return {"n": 1, "mean": clean[0], "std": 0.0}
    return {
        "n": len(clean),
        "mean": mean(clean),
        "std": stdev(clean),
    }


def paired_t(
    left: list[float],
    right: list[float],
) -> dict[str, Any] | None:
    differences = [x - y for x, y in zip(left, right, strict=True)]
    n = len(differences)
    if n < 2:
        return None
    mean_difference = mean(differences)
    std_difference = stdev(differences)
    from scipy.stats import t as student_t

    standard_error = std_difference / math.sqrt(n)
    if standard_error > 0:
        t_statistic = mean_difference / standard_error
        p_value = float(2 * student_t.sf(abs(t_statistic), df=n - 1))
    else:
        # JSON has no infinity. Equal paired differences are a degenerate test.
        t_statistic = None
        p_value = 1.0 if mean_difference == 0 else 0.0
    return {
        "n": n,
        "mean_diff": mean_difference,
        "std_diff": std_difference,
        "t_stat": t_statistic,
        "p_value": p_value,
        "degenerate": standard_error == 0,
        "df": n - 1,
        "ci95_half_width": float(student_t.ppf(0.975, n - 1)) * standard_error,
    }


def aggregate_synthetic(metrics_path: str | Path) -> dict[str, Any] | None:
    payload = load(metrics_path)
    if payload is None:
        return None
    rows = payload["rows"]
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("structure", "block"),
                row["method"],
                float(row["noise"]),
            )
        ].append(row)
    summaries = {}
    for (structure, method, noise), group in grouped.items():
        for metric in (
            "support_auroc",
            "support_ap",
            "mass_shift",
            "ari_patch",
            "ari_token",
            "recon_clean",
        ):
            summaries[f"{structure}/{method}/noise={noise}/{metric}"] = stats(
                [row.get(metric) for row in group]
            )

    paired_tests = {}
    structures = sorted({row.get("structure", "block") for row in rows})
    noises = sorted({float(row["noise"]) for row in rows})
    for structure in structures:
        for noise in noises:
            for metric in ("support_auroc", "mass_shift", "recon_clean"):
                coground = {
                    int(row["seed"]): float(row[metric])
                    for row in rows
                    if row.get("structure", "block") == structure
                    and row["method"] == "coground"
                    and float(row["noise"]) == noise
                    and row.get(metric) is not None
                }
                low_rank = {
                    int(row["seed"]): float(row[metric])
                    for row in rows
                    if row.get("structure", "block") == structure
                    and row["method"] == "low_rank"
                    and float(row["noise"]) == noise
                    and row.get(metric) is not None
                }
                common = sorted(set(coground) & set(low_rank))
                result = paired_t(
                    [coground[seed] for seed in common],
                    [low_rank[seed] for seed in common],
                )
                if result is not None:
                    paired_tests[f"{structure}/noise={noise}/{metric}_coground-vs-low_rank"] = (
                        result
                    )
    return {
        "by_method_noise": summaries,
        "paired_tests": paired_tests,
        "n_rows": len(rows),
    }


def aggregate_kmismatch(metrics_path: str | Path) -> dict[str, Any] | None:
    payload = load(metrics_path)
    if payload is None:
        return None
    rows = payload["rows"]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["k_pred"]))].append(row)
    summaries = {}
    for (method, k_pred), group in grouped.items():
        for metric in ("support_auroc", "support_ap", "mass_shift"):
            summaries[f"{method}/k_pred={k_pred}/{metric}"] = stats(
                [row.get(metric) for row in group]
            )
    return {"by_method_k": summaries, "n_rows": len(rows)}


GROUNDING_METRICS = (
    "patch_pointing_accuracy",
    "gt_box_support_mass_mean",
    "patch_recall_at_1",
    "patch_recall_at_5",
    "patch_recall_at_10",
    "in_box_vs_out_auroc_mean",
)


def aggregate_grounding(pattern: str) -> dict[str, Any]:
    paths = sorted((ROOT / "results" / "runs").glob(f"{pattern}/metrics.json"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = load(path)
        if payload is None:
            continue
        row = {
            key: payload.get(key)
            for key in (
                *GROUNDING_METRICS,
                "experiment_id",
                "variant",
                "seed",
                "random_pointing_baseline",
                "k",
                "checkpoint",
                "scoring_head",
                "trainable_scoring_parameters",
                "dataset",
                "backbone",
                "split",
                "layer_tap",
                "spatial_protocol",
                "patch_grid",
            )
        }
        row["source"] = str(path)
        row["variant"] = row["variant"] or path.parent.name.replace("grounding_", "")
        rows.append(row)

    protocol_fields = (
        "dataset",
        "backbone",
        "split",
        "layer_tap",
        "spatial_protocol",
        "patch_grid",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    protocol_groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        protocol = json.dumps([row.get(field) for field in protocol_fields])
        variant = str(row["variant"])
        group = protocol_groups[protocol][variant]
        if any(previous.get("seed") == row.get("seed") for previous in group):
            raise ValueError(f"Duplicate seed for {variant} in protocol {protocol}.")
        group.append(row)
        grouped[f"{protocol}/{variant}"].append(row)
    summaries = {
        variant: {metric: stats([row.get(metric) for row in group]) for metric in GROUNDING_METRICS}
        for variant, group in grouped.items()
    }
    for variant, group in grouped.items():
        summaries[variant]["scoring_head"] = group[0].get("scoring_head")
        summaries[variant]["trainable_scoring_parameters"] = group[0].get(
            "trainable_scoring_parameters"
        )

    paired_tests = {}
    comparisons = (
        ("coground_grounded", "matched_mlp_grounding"),
        ("coground_grounded", "bilinear_grounding"),
        ("coground_refcocog_transfer", "matched_mlp_refcocog_transfer"),
        ("coground_refcocog_transfer", "bilinear_refcocog_transfer"),
        ("coground_paligemma2_transfer", "matched_mlp_paligemma2_transfer"),
        ("coground_paligemma2_transfer", "bilinear_paligemma2_transfer"),
    )
    for protocol, variants in protocol_groups.items():
        # Legacy metrics without protocol provenance cannot establish matched comparisons.
        if any(value is None for value in json.loads(protocol)[:5]):
            continue
        for left_variant, right_variant in comparisons:
            for metric in GROUNDING_METRICS:
                left = {
                    int(row["seed"]): float(row[metric])
                    for row in variants.get(left_variant, [])
                    if row.get("seed") is not None and row.get(metric) is not None
                }
                right = {
                    int(row["seed"]): float(row[metric])
                    for row in variants.get(right_variant, [])
                    if row.get("seed") is not None and row.get(metric) is not None
                }
                common = sorted(set(left) & set(right))
                result = paired_t(
                    [left[seed] for seed in common],
                    [right[seed] for seed in common],
                )
                if result is not None:
                    paired_tests[f"{protocol}/{left_variant}-vs-{right_variant}/{metric}"] = result
    return {
        "runs": rows,
        "by_variant": summaries,
        "paired_tests": paired_tests,
        "n_rows": len(rows),
    }


def aggregate_generation(pattern: str) -> dict[str, Any]:
    metric_names = (
        "pope_accuracy",
        "pope_precision",
        "pope_recall",
        "pope_f1",
        "pope_yes_ratio",
        "chairs",
        "chairi",
        "exact_match",
    )
    paths = sorted((ROOT / "results" / "runs").glob(f"{pattern}/metrics.json"))
    rows = []
    for path in paths:
        payload = load(path)
        if payload is None:
            continue
        row = {
            key: payload.get(key)
            for key in (
                *metric_names,
                "experiment_id",
                "variant",
                "dataset",
                "seed",
                "decoding_prior_enabled",
                "decoding_mode",
                "lambda_dec",
                "checkpoint",
                "backbone",
                "split",
                "layer_tap",
            )
        }
        row["source"] = str(path)
        rows.append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition = json.dumps(
            [
                row.get(field)
                for field in (
                    "dataset",
                    "backbone",
                    "split",
                    "layer_tap",
                    "decoding_mode",
                    "lambda_dec",
                    "variant",
                )
            ]
        )
        grouped[condition].append(row)
    summaries = {
        condition: {metric: stats([row.get(metric) for row in group]) for metric in metric_names}
        for condition, group in grouped.items()
    }
    paired_tests = {}
    protocols: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        fields = ("dataset", "backbone", "split", "layer_tap", "decoding_mode")
        protocol = json.dumps([row.get(field) for field in fields])
        protocols[protocol].append(row)
    for protocol, group in protocols.items():
        if any(value is None for value in json.loads(protocol)):
            continue
        enabled, disabled = {}, {}
        for row in group:
            if row.get("seed") is None or row.get("checkpoint") is None:
                continue
            key = (int(row["seed"]), str(row["checkpoint"]))
            index = enabled if bool(row.get("decoding_prior_enabled")) else disabled
            if key in index:
                raise ValueError(f"Duplicate generation condition for {protocol}, {key}.")
            index[key] = row
        common = sorted(set(enabled) & set(disabled))
        for metric in metric_names:
            usable = [
                key
                for key in common
                if enabled[key].get(metric) is not None and disabled[key].get(metric) is not None
            ]
            result = paired_t(
                [float(enabled[key][metric]) for key in usable],
                [float(disabled[key][metric]) for key in usable],
            )
            if result is not None:
                paired_tests[f"{protocol}/prior-vs-no_prior/{metric}"] = result
    return {
        "runs": rows,
        "by_dataset_variant": summaries,
        "paired_tests": paired_tests,
        "n_rows": len(rows),
    }


def main() -> None:
    output = {
        "synthetic_structure_stress": aggregate_synthetic(
            ROOT / "results/runs/synthetic_structure_stress/metrics.json"
        ),
        "synthetic_block_recovery_legacy": aggregate_synthetic(
            ROOT / "results/runs/synthetic_multi/metrics.json"
        ),
        "synthetic_kmismatch": aggregate_kmismatch(
            ROOT / "results/runs/synthetic_kmismatch/metrics.json"
        ),
        "grounding_revision": aggregate_grounding("grounding_revision*"),
        "grounding_all": aggregate_grounding("grounding_*"),
        "generation_revision": aggregate_generation("revision_gemma4_*"),
    }
    output_path = ROOT / "results/runs/AGGREGATE_ALL.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    revision = output["grounding_revision"]
    for variant, summary in revision["by_variant"].items():
        pointing = summary["patch_pointing_accuracy"]
        auroc = summary["in_box_vs_out_auroc_mean"]
        print(
            f"{variant}: pointing={pointing.get('mean')} "
            f"AUROC={auroc.get('mean')} n={pointing.get('n')}"
        )


if __name__ == "__main__":
    main()
