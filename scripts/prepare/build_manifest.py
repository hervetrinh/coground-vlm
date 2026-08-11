#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coground_vlm.utils.io import write_jsonl

SUPPORTED = {
    "refcoco",
    "refcoco+",
    "refcocog",
    "flickr30k_entities",
    "pope",
    "coco_chair",
    "hallusionbench",
    "gqa",
    "vqav2",
    "winoground",
    "sugarcrepe",
    "aro",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized CoGround-VLM JSONL manifests.")
    parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED))
    parser.add_argument("--source", required=True, help="Source JSON/JSONL file or directory.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--split", default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)
    records = load_source_records(source)
    converted = [
        convert_record(args.dataset, record, args.image_root, args.split, idx)
        for idx, record in enumerate(records)
    ]
    write_jsonl(args.out, converted)
    print(f"Wrote {args.out} ({len(converted)} records)")


def load_source_records(source: Path) -> list[dict[str, Any]]:
    if source.is_dir():
        raise NotImplementedError(
            f"Directory source {source} needs a dataset-specific converter. Provide a JSON/JSONL export "
            "with image/prompt/answer/boxes fields or add the raw-format parser here."
        )
    if source.suffix == ".jsonl":
        out = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    out.append(json.loads(line))
        return out
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        for key in ("annotations", "data", "questions", "examples"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported source format: {source}")


def convert_record(
    dataset: str, record: dict[str, Any], image_root: str, split: str, idx: int
) -> dict[str, Any]:
    image = first(record, ["image", "image_path", "file_name", "filename"])
    prompt = first(record, ["prompt", "question", "caption", "expression", "text"])
    if image is None or prompt is None:
        raise KeyError(
            f"{dataset} record {idx} needs image and prompt/question/caption/expression fields. "
            "Add a dataset-specific parser if the raw schema differs."
        )
    out = {
        "id": str(
            first(
                record,
                ["id", "question_id", "ann_id", "sample_id"],
                default=f"{dataset}-{split}-{idx}",
            )
        ),
        "image": image,
        "prompt": prompt,
        "metadata": {"dataset": dataset, "split": split, "image_root": image_root},
    }
    for src, dst in [
        ("answer", "answer"),
        ("answers", "answer"),
        ("label_text", "label_text"),
        ("boxes", "boxes"),
        ("bbox", "boxes"),
        ("token_spans", "token_spans"),
        ("unsupported_label", "unsupported_label"),
        ("pair_id", "pair_id"),
        ("group_id", "group_id"),
        ("hallucinated_objects", "hallucinated_objects"),
        ("mentioned_objects", "mentioned_objects"),
        ("ground_truth_objects", "ground_truth_objects"),
        ("candidate_objects", "candidate_objects"),
        ("object_synonyms", "candidate_objects"),
    ]:
        if src in record:
            value = record[src]
            if src == "bbox" and value and not isinstance(value[0], list):
                value = [value]
            out[dst] = value
    if dataset in {"refcoco", "refcoco+", "refcocog"} and "boxes" in out:
        out.setdefault(
            "grounding_text",
            first(record, ["grounding_text", "expression", "text"], default=prompt),
        )
    return out


def first(record: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


if __name__ == "__main__":
    main()
