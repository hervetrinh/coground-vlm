#!/usr/bin/env python
"""E0 cosine baseline at multiple layer_tap values (no training, eval only).
Tells us whether the trained head adds value over raw cosine similarity at each layer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import subprocess

LAYERS = [0, 2, 4, 6, 8, 12, 16, 20, 24, 32]
ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval/gemma4_refcoco_grounding.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    args = p.parse_args()

    for L in args.layers:
        out = ROOT / f"results/runs/grounding_E0_lt{L}/metrics.json"
        if out.exists():
            with open(out) as f:
                d = json.load(f)
            print(
                f"[skip] lt={L}: pointing={d['patch_pointing_accuracy']:.3f} AUROC={d['in_box_vs_out_auroc_mean']:.3f}"
            )
            continue
        cmd = [
            "python",
            "scripts/eval/run_grounding_metrics.py",
            "--config",
            args.config,
            "--output",
            str(out),
            f"experiment_id=grounding_E0_lt{L}",
            f"variant=E0_lt{L}",
            "method.enabled=false",
            f"method.layer_tap={L}",
        ]
        env = os.environ.copy()
        env.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "PYTHONPATH": "src",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        print(f"=== E0 cosine at lt={L} ===")
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL lt={L}:", r.stderr[-500:])
            continue
        with open(out) as f:
            d = json.load(f)
        print(
            f"  lt={L}: pointing={d['patch_pointing_accuracy']:.3f} recall@5={d['patch_recall_at_5']:.3f} recall@10={d['patch_recall_at_10']:.3f} AUROC={d['in_box_vs_out_auroc_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
