from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from coground_vlm.config import as_plain_dict, save_config


def save_checkpoint(
    output_dir: str | Path,
    *,
    step: int,
    cfg: Any,
    co_head: torch.nn.Module | None = None,
    grounding_baseline: torch.nn.Module | None = None,
    decoding_prior: torch.nn.Module | None = None,
    backbone: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    progress: dict[str, int] | None = None,
    world_size: int = 1,
    extra: dict[str, Any] | None = None,
    update_latest: bool = True,
) -> Path:
    out = Path(output_dir) / f"step_{step:08d}"
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "progress": progress or {"global_step": step, "epoch": 0, "batches_in_epoch": 0},
        "world_size": int(world_size),
        "extra": extra or {},
    }
    if co_head is not None:
        torch.save(co_head.state_dict(), out / "coclustering_head.pt")
    if grounding_baseline is not None:
        torch.save(grounding_baseline.state_dict(), out / "grounding_baseline.pt")
    if decoding_prior is not None:
        torch.save(decoding_prior.state_dict(), out / "decoding_prior.pt")
    if backbone is not None and hasattr(backbone, "save_pretrained"):
        backbone.save_pretrained(out / "backbone_adapter")
        trainable_backbone = {
            name: parameter.detach().cpu()
            for name, parameter in backbone.named_parameters()
            if parameter.requires_grad
        }
        if trainable_backbone:
            torch.save(trainable_backbone, out / "backbone_trainable.pt")
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, out / "trainer_state.pt")
    save_config(as_plain_dict(cfg), out / "config.yaml")
    if update_latest:
        write_latest_checkpoint(output_dir, out)
    return out


def write_latest_checkpoint(output_dir: str | Path, checkpoint_dir: str | Path) -> None:
    """Atomically publish a completed checkpoint as the resume target."""

    root = Path(output_dir)
    latest = root / "latest.txt"
    temporary = root / ".latest.txt.tmp"
    temporary.write_text(str(Path(checkpoint_dir).resolve()), encoding="utf-8")
    os.replace(temporary, latest)


def resolve_resume_checkpoint(
    resume: str | Path | None,
    output_dir: str | Path,
) -> Path | None:
    """Resolve ``auto``, a checkpoint directory, or a ``latest.txt`` file."""

    if resume is None or str(resume).strip().lower() in {"", "none", "false"}:
        return None
    candidate = Path(output_dir) / "latest.txt" if str(resume).lower() == "auto" else Path(resume)
    if candidate.is_file():
        raw = candidate.read_text(encoding="utf-8").strip()
        if not raw:
            raise ValueError(f"Checkpoint pointer is empty: {candidate}")
        pointed = Path(raw)
        if not pointed.is_absolute():
            cwd_candidate = pointed.resolve()
            pointer_candidate = (candidate.parent / pointed).resolve()
            pointed = cwd_candidate if cwd_candidate.exists() else pointer_candidate
        candidate = pointed
    if not candidate.exists():
        if str(resume).lower() == "auto":
            return None
        raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"Resume checkpoint must be a directory: {candidate}")
    trainer_state = candidate / "trainer_state.pt"
    if not trainer_state.exists():
        raise FileNotFoundError(f"Incomplete checkpoint, missing {trainer_state}")
    return candidate.resolve()


def load_training_checkpoint(
    path: str | Path,
    *,
    co_head: torch.nn.Module | None = None,
    grounding_baseline: torch.nn.Module | None = None,
    decoding_prior: torch.nn.Module | None = None,
    backbone: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    expected_world_size: int | None = None,
    rank: int = 0,
) -> dict[str, Any]:
    """Restore a synchronized training checkpoint without serializing frozen weights."""

    checkpoint = Path(path)
    state = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
    saved_world_size = int(state.get("world_size", 1))
    if expected_world_size is not None and saved_world_size != int(expected_world_size):
        raise ValueError(
            "Checkpoint world size does not match this launch: "
            f"saved={saved_world_size}, current={expected_world_size}."
        )
    if co_head is not None:
        load_coclustering_head(checkpoint, co_head)
    if grounding_baseline is not None:
        load_grounding_baseline(checkpoint, grounding_baseline)
    if decoding_prior is not None:
        load_decoding_prior(checkpoint, decoding_prior)
    trainable_backbone_path = checkpoint / "backbone_trainable.pt"
    if backbone is not None and trainable_backbone_path.exists():
        saved_parameters = torch.load(
            trainable_backbone_path,
            map_location="cpu",
            weights_only=True,
        )
        current_parameters = dict(backbone.named_parameters())
        missing = sorted(set(saved_parameters) - set(current_parameters))
        if missing:
            raise KeyError(f"Trainable backbone parameters missing while resuming: {missing[:5]}")
        with torch.no_grad():
            for name, value in saved_parameters.items():
                current_parameters[name].copy_(value.to(current_parameters[name].device))
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    rng_path = checkpoint / f"rng_state_rank{rank:04d}.pt"
    if rng_path.exists():
        restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    return state


def save_rng_state(checkpoint_dir: str | Path, rank: int) -> Path:
    out = Path(checkpoint_dir) / f"rng_state_rank{rank:04d}.pt"
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state()
    torch.save(payload, out)
    return out


def restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and "cuda" in payload:
        torch.cuda.set_rng_state(payload["cuda"])


def load_coclustering_head(
    path: str | Path, co_head: torch.nn.Module, map_location: str = "cpu"
) -> None:
    state_path = Path(path)
    if state_path.is_dir():
        state_path = state_path / "coclustering_head.pt"
    state = torch.load(state_path, map_location=map_location)
    co_head.load_state_dict(state)


def load_decoding_prior(
    path: str | Path, decoding_prior: torch.nn.Module, map_location: str = "cpu"
) -> None:
    state_path = Path(path)
    if state_path.is_dir():
        state_path = state_path / "decoding_prior.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Decoding prior checkpoint not found: {state_path}")
    state = torch.load(state_path, map_location=map_location)
    decoding_prior.load_state_dict(state)


def load_grounding_baseline(
    path: str | Path,
    grounding_baseline: torch.nn.Module,
    map_location: str = "cpu",
) -> None:
    state_path = Path(path)
    if state_path.is_dir():
        state_path = state_path / "grounding_baseline.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Grounding baseline checkpoint not found: {state_path}")
    state = torch.load(state_path, map_location=map_location)
    grounding_baseline.load_state_dict(state)
