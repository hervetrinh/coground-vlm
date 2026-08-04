from __future__ import annotations

from pathlib import Path

import pytest
import torch

from coground_vlm.train.checkpointing import (
    load_training_checkpoint,
    resolve_resume_checkpoint,
    save_checkpoint,
)


def test_auto_resume_resolves_atomic_latest_pointer(tmp_path: Path) -> None:
    module = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1.0e-3)
    checkpoint = save_checkpoint(
        tmp_path,
        step=7,
        cfg={"seed": 17},
        co_head=module,
        optimizer=optimizer,
        progress={"global_step": 7, "epoch": 2, "batches_in_epoch": 3},
        world_size=2,
    )
    assert resolve_resume_checkpoint("auto", tmp_path) == checkpoint.resolve()


def test_resume_rejects_a_different_world_size(tmp_path: Path) -> None:
    module = torch.nn.Linear(4, 2)
    checkpoint = save_checkpoint(
        tmp_path,
        step=1,
        cfg={"seed": 17},
        co_head=module,
        world_size=2,
    )
    with pytest.raises(ValueError, match="world size"):
        load_training_checkpoint(
            checkpoint,
            co_head=module,
            expected_world_size=1,
        )
