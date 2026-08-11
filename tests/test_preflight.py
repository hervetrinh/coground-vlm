from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_preflight_validates_data_without_starting_training(tmp_path: Path) -> None:
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (8, 8), color="white").save(image)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample",
                "image": image.name,
                "prompt": "the white square",
                "grounding_text": "white square",
                "boxes": [[0, 0, 8, 8]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
experiment_id: preflight_test
output_dir: {tmp_path / "new" / "nested" / "runs"}
runtime:
  mixed_precision: bf16
  expected_world_size: 2
  require_a100: true
  min_free_disk_gb: 0
backbone:
  kind: mock
  model_id: mock
  device_map: null
  load_in_4bit: false
dataset:
  name: refcoco
  image_root: {tmp_path}
  train_split: train
  splits:
    train: {manifest}
training:
  batch_size: 1
  gradient_accumulation_steps: 8
  global_batch_size: 16
""".strip()
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train/preflight_2xa100.py",
            "--config",
            str(config),
            "--world-size",
            "2",
            "--skip-gpu",
            "--sample-records",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["status"] == "ok"
    assert report["effective_global_batch_size"] == 16
    assert report["manifest"]["sampled_records"] == 1
