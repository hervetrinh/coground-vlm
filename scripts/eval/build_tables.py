#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CoGround-VLM result rows.")
    parser.add_argument("--results-dir", default="results/runs")
    parser.add_argument("--out-dir", default="results/tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for csv_path in Path(args.results_dir).glob("**/summary_row.csv"):
        rows.append(pd.read_csv(csv_path))
    if not rows:
        raise FileNotFoundError(f"No summary_row.csv files found under {args.results_dir}")
    df = pd.concat(rows, ignore_index=True)
    group_cols = ["variant", "baseline", "backbone", "dataset", "k_patch", "lora_enabled"]
    for col in group_cols:
        if col not in df.columns:
            df[col] = ""
    metric_cols = [
        col
        for col in df.columns
        if col not in set(group_cols + ["experiment_id", "seed", "raw_predictions", "metrics_json"])
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    summary = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "main_tables_runs.csv", index=False)
    summary.to_csv(out_dir / "main_tables.csv", index=False)
    with (out_dir / "main_tables.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs": df.to_dict(orient="records"),
                "summary": json.loads(summary.to_json(orient="records")),
            },
            handle,
            indent=2,
        )
    print(f"Wrote {out_dir / 'main_tables.csv'}")


if __name__ == "__main__":
    main()
