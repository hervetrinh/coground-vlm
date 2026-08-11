#!/usr/bin/env python
"""Resolve one configured Hugging Face backbone before a multi-process launch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from huggingface_hub import snapshot_download

from coground_vlm.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache a configured HF backbone once.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None, help="Prefer HF_TOKEN instead of shell history.")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    model_id = str(cfg.backbone.model_id)
    local_model = Path(model_id).expanduser()
    if local_model.exists():
        print(json.dumps({"status": "already_local", "path": str(local_model.resolve())}, indent=2))
        return
    resolved = snapshot_download(
        repo_id=model_id,
        revision=args.revision,
        local_dir=args.local_dir,
        token=args.token,
    )
    print(
        json.dumps(
            {
                "status": "cached",
                "model_id": model_id,
                "snapshot_path": str(Path(resolved).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
