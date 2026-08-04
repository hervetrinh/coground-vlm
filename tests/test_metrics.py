from __future__ import annotations

from coground_vlm.eval.metrics import (
    chair_object_mentions,
    parse_binary_answer,
    summarize_predictions,
)


def test_parse_binary_answer_uses_final_explicit_answer() -> None:
    assert parse_binary_answer("Question text. No, actually yes.") == 1
    assert parse_binary_answer("No.") == 0
    assert parse_binary_answer("uncertain") is None


def test_pope_metrics() -> None:
    records = [
        {"dataset": "pope", "answer": "yes", "prediction": "Yes."},
        {"dataset": "pope", "answer": "no", "prediction": "No."},
        {"dataset": "pope", "answer": "yes", "prediction": "No."},
    ]
    metrics = summarize_predictions(records)
    assert metrics["pope_accuracy"] == 2 / 3
    assert metrics["pope_recall"] == 0.5
    assert metrics["pope_unparsed_count"] == 0


def test_chair_mentions_use_explicit_lexicon() -> None:
    mentioned, hallucinated = chair_object_mentions(
        "A dog sits beside a bicycle.",
        ground_truth_objects=["dog"],
        candidate_objects={
            "dog": ["dogs"],
            "bicycle": ["bike", "bikes"],
            "cat": ["cats"],
        },
    )
    assert mentioned == ["dog", "bicycle"]
    assert hallucinated == ["bicycle"]
