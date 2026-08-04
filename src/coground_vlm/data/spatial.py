"""Shared spatial contract for box supervision and patch-level metrics.

Rows are ordered top to bottom, columns left to right. Only uniform resizing
is supported; cropped/tiled/packed images need a backbone-specific mapping.
"""

from __future__ import annotations

from math import isqrt
from typing import Sequence

import torch


def resolve_patch_grid(
    num_patches: int, patch_grid: Sequence[int] | None = None
) -> tuple[int, int]:
    if num_patches <= 0:
        raise ValueError("num_patches must be positive.")
    if patch_grid is not None:
        if len(patch_grid) != 2:
            raise ValueError("patch_grid must contain [rows, columns].")
        rows, cols = map(int, patch_grid)
        if rows <= 0 or cols <= 0 or rows * cols > num_patches:
            raise ValueError("patch_grid must fit the extracted patch sequence.")
        return rows, cols
    side = isqrt(num_patches)
    if num_patches - side * side not in {0, 4}:
        raise ValueError(
            "Unknown spatial token layout. Set dataset.patch_grid=[rows,columns] "
            "only after verifying row-major spatial tokens followed by trailing specials."
        )
    return side, side


def normalize_box(
    box: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, box)
    if max(x1, y1, x2, y2) > 1.5:
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def box_patch_mask(
    *, boxes: Sequence[Sequence[float]], width: int, height: int, rows: int, cols: int
) -> torch.Tensor:
    """Select patch centers inside the union of xyxy boxes (inclusive edges)."""
    ys = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
    xs = (torch.arange(cols, dtype=torch.float32) + 0.5) / cols
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    mask = torch.zeros((rows, cols), dtype=torch.bool)
    for box in boxes:
        x1, y1, x2, y2 = normalize_box(box, width, height)
        mask |= (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
    return mask.flatten()
