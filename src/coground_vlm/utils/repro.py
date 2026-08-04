from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class RunTimer:
    start_time: float = field(default_factory=time.time)

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time


def gpu_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def reproducibility_payload(cfg: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "seed": int(cfg.get("seed", 0)) if hasattr(cfg, "get") else None,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        payload["cuda_device_name"] = torch.cuda.get_device_name(0)
    if extra:
        payload.update(extra)
    return payload
