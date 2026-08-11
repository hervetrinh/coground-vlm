#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coground_vlm.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a CoGround-VLM JSONL manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--require-labels", action="store_true")
    parser.add_argument("--require-grounding", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    image_root = Path(args.image_root) if args.image_root else manifest.parent
    records = read_jsonl(manifest)
    if not records:
        raise ValueError(f"Manifest has no records: {manifest}")
    for idx, record in enumerate(records):
        rid = record.get("id", idx)
        if "prompt" not in record:
            raise KeyError(f"Record {rid} missing prompt")
        if "image" not in record:
            raise KeyError(f"Record {rid} missing image")
        image_path = Path(record["image"])
        if not image_path.is_absolute():
            image_path = image_root / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Record {rid} image not found: {image_path}")
        if args.require_labels and "label_text" not in record and "answer" not in record:
            raise KeyError(f"Record {rid} missing label_text or answer")
        if args.require_grounding:
            has_dense = record.get("grounding_matrix") is not None
            has_sparse = bool(record.get("boxes")) and bool(
                record.get("token_spans") or record.get("grounding_text")
            )
            if not has_dense and not has_sparse:
                raise KeyError(
                    f"Record {rid} needs grounding_matrix or boxes plus token_spans/grounding_text"
                )
            grounding_text = record.get("grounding_text")
            phrases = [grounding_text] if isinstance(grounding_text, str) else grounding_text or []
            if any(str(phrase).strip() not in str(record["prompt"]) for phrase in phrases):
                raise ValueError(f"Record {rid} grounding_text is not verbatim in prompt")
    print(f"OK: {manifest} ({len(records)} records)")


if __name__ == "__main__":
    main()
