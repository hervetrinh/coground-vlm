#!/usr/bin/env python
"""Export RefCOCO-family REFER annotations to CoGround JSONL manifests.

The script intentionally does not download or redistribute COCO images or
RefCOCO annotations. Point ``--refer-root`` at a local checkout of
lichengunc/refer and ``--data-root`` at its data directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coground_vlm.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RefCOCO REFER splits to JSONL.")
    parser.add_argument("--refer-root", required=True, help="Local lichengunc/refer checkout.")
    parser.add_argument("--data-root", required=True, help="Directory containing refcoco datasets.")
    parser.add_argument("--image-root", required=True, help="Local COCO train2014 image directory.")
    parser.add_argument("--dataset", choices=["refcoco", "refcoco+", "refcocog"], default="refcoco")
    parser.add_argument(
        "--split-by", default=None, help="REFER splitBy value; defaults to unc or umd."
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prompt-template",
        default="{expression}",
        help="Template containing {expression}; keep the expression verbatim for token alignment.",
    )
    parser.add_argument("--skip-image-check", action="store_true")
    return parser.parse_args()


def load_refer_class(refer_root: Path) -> Any:
    sys.path.insert(0, str(refer_root.resolve()))
    try:
        from refer import REFER  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            f"Could not import REFER from {refer_root}. Clone https://github.com/lichengunc/refer "
            "and pass the checkout root with --refer-root."
        ) from exc
    return REFER


def default_splits(dataset: str) -> list[str]:
    return ["train", "val", "test"] if dataset == "refcocog" else ["train", "val", "testA", "testB"]


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def main() -> None:
    args = parse_args()
    refer_root = Path(args.refer_root)
    data_root = Path(args.data_root)
    image_root = Path(args.image_root)
    if "{expression}" not in args.prompt_template:
        raise ValueError("--prompt-template must contain the literal {expression} placeholder.")
    for path, label in (
        (refer_root, "refer root"),
        (data_root, "data root"),
        (image_root, "image root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    split_by = args.split_by or ("umd" if args.dataset == "refcocog" else "unc")
    refer_class = load_refer_class(refer_root)
    refer = refer_class(str(data_root), dataset=args.dataset, splitBy=split_by)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits or default_splits(args.dataset):
        records: list[dict[str, Any]] = []
        for ref_id in sorted(refer.getRefIds(split=split)):
            ref = refer.Refs[ref_id]
            annotation = refer.Anns[ref["ann_id"]]
            image = refer.Imgs[ref["image_id"]]
            image_name = str(image["file_name"])
            image_path = image_root / image_name
            if not args.skip_image_check and not image_path.exists():
                raise FileNotFoundError(f"COCO image for ref {ref_id} not found: {image_path}")
            box = xywh_to_xyxy(annotation["bbox"])
            for sentence_index, sentence in enumerate(ref.get("sentences", [])):
                expression = str(sentence.get("raw", sentence.get("sent", ""))).strip()
                if not expression:
                    continue
                sentence_id = sentence.get("sent_id", sentence_index)
                records.append(
                    {
                        "id": f"{args.dataset}-{split}-ref{ref_id}-sent{sentence_id}",
                        "image": image_name,
                        "prompt": args.prompt_template.format(expression=expression),
                        "grounding_text": expression,
                        "boxes": [box],
                        "metadata": {
                            "dataset": args.dataset,
                            "split": split,
                            "split_by": split_by,
                            "ref_id": int(ref_id),
                            "ann_id": int(ref["ann_id"]),
                            "image_id": int(ref["image_id"]),
                            "sentence_id": int(sentence_id),
                        },
                    }
                )
        if not records:
            raise RuntimeError(f"REFER returned no records for {args.dataset}/{split_by}/{split}.")
        output_path = output_dir / f"{args.dataset}_{split}.jsonl"
        write_jsonl(output_path, records)
        print(f"Wrote {output_path} ({len(records)} expressions)")


if __name__ == "__main__":
    main()
