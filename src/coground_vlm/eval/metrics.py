from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, answer: str | list[str]) -> float:
    answers = answer if isinstance(answer, list) else [answer]
    pred = normalize_text(prediction)
    return float(any(pred == normalize_text(str(candidate)) for candidate in answers))


def contains_answer(prediction: str, answer: str | list[str]) -> float:
    answers = answer if isinstance(answer, list) else [answer]
    pred = normalize_text(prediction)
    return float(any(normalize_text(str(candidate)) in pred for candidate in answers))


def parse_binary_answer(text: str) -> int | None:
    """Parse the final explicit yes/no token from a generated response."""

    tokens = normalize_text(text).split()
    for token in reversed(tokens):
        if token == "yes":
            return 1
        if token == "no":
            return 0
    return None


def summarize_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"num_examples": len(records)}
    answered = [r for r in records if r.get("answer") is not None]
    if answered:
        metrics["exact_match"] = float(
            np.mean([exact_match(r["prediction"], r["answer"]) for r in answered])
        )
        metrics["contains_answer"] = float(
            np.mean([contains_answer(r["prediction"], r["answer"]) for r in answered])
        )
    support_values = [r.get("support_mean") for r in records if r.get("support_mean") is not None]
    if support_values:
        metrics["support_mean"] = float(np.mean(support_values))
        metrics["support_std"] = float(np.std(support_values))
    coherence_values = [
        r.get("block_coherence") for r in records if r.get("block_coherence") is not None
    ]
    if coherence_values:
        metrics["block_coherence"] = float(np.mean(coherence_values))
    unsupported = [
        r
        for r in records
        if r.get("unsupported_label") is not None and r.get("support_mean") is not None
    ]
    if len({r["unsupported_label"] for r in unsupported}) == 2:
        labels = [int(r["unsupported_label"]) for r in unsupported]
        scores = [-float(r["support_mean"]) for r in unsupported]
        metrics["unsupported_auroc"] = float(roc_auc_score(labels, scores))
        metrics["unsupported_average_precision"] = float(average_precision_score(labels, scores))
    add_grounding_metrics(metrics, records)
    add_pair_group_metrics(metrics, records)
    add_pope_metrics(metrics, records)
    add_chair_metrics(metrics, records)
    return metrics


def add_grounding_metrics(metrics: dict[str, Any], records: list[dict[str, Any]]) -> None:
    grounding = [r for r in records if r.get("predicted_box") is not None and r.get("boxes")]
    if grounding:
        ious = [max(box_iou(r["predicted_box"], box) for box in r["boxes"]) for r in grounding]
        metrics["box_accuracy_iou50"] = float(np.mean([iou >= 0.5 for iou in ious]))
        metrics["mean_iou"] = float(np.mean(ious))
    pointing = [r for r in records if r.get("point") is not None and r.get("boxes")]
    if pointing:
        metrics["pointing_accuracy"] = float(
            np.mean([any(point_in_box(r["point"], box) for box in r["boxes"]) for r in pointing])
        )
    ranked = [r for r in records if r.get("ranked_boxes") is not None and r.get("boxes")]
    if ranked:
        for k in (1, 5, 10):
            metrics[f"recall_at_{k}"] = float(
                np.mean(
                    [
                        any(
                            box_iou(pred, gt) >= 0.5
                            for pred in r["ranked_boxes"][:k]
                            for gt in r["boxes"]
                        )
                        for r in ranked
                    ]
                )
            )


def add_pair_group_metrics(metrics: dict[str, Any], records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        group_id = record.get("pair_id") or record.get("group_id")
        if group_id is not None and record.get("answer") is not None:
            grouped.setdefault(str(group_id), []).append(record)
    if grouped:
        group_scores = []
        for group in grouped.values():
            group_scores.append(all(exact_match(r["prediction"], r["answer"]) for r in group))
        metrics["group_accuracy"] = float(np.mean(group_scores))
        metrics["pair_accuracy"] = metrics["group_accuracy"]


def add_pope_metrics(metrics: dict[str, Any], records: list[dict[str, Any]]) -> None:
    pope_records = [
        record
        for record in records
        if str(record.get("dataset", "")).lower() == "pope" and record.get("answer") is not None
    ]
    if not pope_records:
        return
    labels: list[int] = []
    predictions: list[int] = []
    unparsed = 0
    for record in pope_records:
        answer = record["answer"]
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        label = parse_binary_answer(str(answer))
        prediction = parse_binary_answer(str(record.get("prediction", "")))
        if label is None:
            continue
        if prediction is None:
            unparsed += 1
            # POPE's official convention treats a non-affirmative response as
            # negative. We retain the count so this fallback stays auditable.
            prediction = 0
        labels.append(label)
        predictions.append(prediction)
    if not labels:
        return
    metrics.update(
        {
            "pope_accuracy": float(accuracy_score(labels, predictions)),
            "pope_precision": float(precision_score(labels, predictions, zero_division=0)),
            "pope_recall": float(recall_score(labels, predictions, zero_division=0)),
            "pope_f1": float(f1_score(labels, predictions, zero_division=0)),
            "pope_yes_ratio": float(np.mean(predictions)),
            "pope_unparsed_count": unparsed,
        }
    )


def add_chair_metrics(metrics: dict[str, Any], records: list[dict[str, Any]]) -> None:
    for record in records:
        if (
            record.get("hallucinated_objects") is None
            and record.get("mentioned_objects") is None
            and record.get("ground_truth_objects") is not None
            and record.get("candidate_objects") is not None
        ):
            mentioned, hallucinated = chair_object_mentions(
                str(record.get("prediction", "")),
                record["ground_truth_objects"],
                record["candidate_objects"],
            )
            record["mentioned_objects"] = mentioned
            record["hallucinated_objects"] = hallucinated
    chair_records = [
        r
        for r in records
        if r.get("hallucinated_objects") is not None or r.get("mentioned_objects") is not None
    ]
    if not chair_records:
        return
    sent_hall = []
    hallucinated = 0
    mentioned = 0
    for record in chair_records:
        h = len(record.get("hallucinated_objects") or [])
        m = len(record.get("mentioned_objects") or [])
        hallucinated += h
        mentioned += m
        sent_hall.append(h > 0)
    metrics["chairs"] = float(np.mean(sent_hall))
    metrics["chairi"] = float(hallucinated / max(1, mentioned))


def chair_object_mentions(
    prediction: str,
    ground_truth_objects: list[str],
    candidate_objects: list[str] | dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Extract object mentions from an explicit, auditable object lexicon.

    For official CHAIR reporting, manifests should be produced with the
    official synonym and double-word lists. This function does not invent
    object annotations from the generated caption.
    """

    normalized_prediction = f" {normalize_text(prediction)} "
    aliases: dict[str, list[str]]
    if isinstance(candidate_objects, dict):
        aliases = {
            str(canonical): [str(canonical), *[str(item) for item in synonyms]]
            for canonical, synonyms in candidate_objects.items()
        }
    else:
        aliases = {str(item): [str(item)] for item in candidate_objects}
    ground_truth = {normalize_text(str(item)) for item in ground_truth_objects}
    mentioned: list[str] = []
    for canonical, synonyms in aliases.items():
        normalized_canonical = normalize_text(canonical)
        if any(
            f" {normalize_text(alias)} " in normalized_prediction
            for alias in synonyms
            if normalize_text(alias)
        ):
            mentioned.append(normalized_canonical)
    hallucinated = [item for item in mentioned if item not in ground_truth]
    return mentioned, hallucinated


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-8)


def point_in_box(point: list[float], box: list[float]) -> bool:
    x, y = map(float, point[:2])
    x1, y1, x2, y2 = map(float, box)
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)


def table_row(
    *,
    cfg: Any,
    metrics: dict[str, Any],
    output_paths: dict[str, str],
    baseline: str,
) -> dict[str, Any]:
    row = {
        "experiment_id": str(cfg.experiment_id),
        "variant": str(cfg.get("variant", cfg.experiment_id)),
        "baseline": baseline,
        "backbone": str(cfg.backbone.get("model_id", cfg.backbone.get("kind"))),
        "dataset": str(cfg.dataset.get("name")),
        "split": str(cfg.dataset.get("eval_split", cfg.dataset.get("split", "eval"))),
        "seed": int(cfg.seed),
        "k_patch": cfg.method.get("num_patch_clusters", cfg.method.get("k", "")),
        "k_token": cfg.method.get("num_token_clusters", cfg.method.get("k", "")),
        "lambda_recon": cfg.losses.get("lambda_recon", 0.0),
        "lambda_ground": cfg.losses.get("lambda_ground", 0.0),
        "lambda_dec": cfg.decoding.get("lambda_dec", 0.0),
        "lora_enabled": bool(cfg.get("lora", {}).get("enabled", False)),
        "decoding_prior_enabled": bool(cfg.decoding.get("enabled", False)),
        "raw_predictions": output_paths.get("raw_predictions", ""),
        "metrics_json": output_paths.get("metrics_json", ""),
    }
    row.update(metrics)
    return row
