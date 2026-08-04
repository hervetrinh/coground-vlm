from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from coground_vlm.data.spatial import box_patch_mask, resolve_patch_grid
from coground_vlm.utils.io import read_jsonl


class ManifestVLDataset(Dataset):
    """JSONL manifest dataset for grounding, QA, captioning, and robustness tasks.

    Required per-record fields:
      - image: relative or absolute image path
      - prompt: model input prompt

    Optional fields:
      - id, label_text, answer, boxes, token_spans, grounding_matrix, metadata
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path | None = None,
        limit: int | None = None,
        require_images: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.image_root = Path(image_root) if image_root else self.manifest_path.parent
        self.records = read_jsonl(self.manifest_path)
        if limit:
            self.records = self.records[:limit]
        self.require_images = require_images

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = dict(self.records[idx])
        record.setdefault("id", f"{self.manifest_path.stem}-{idx}")
        image_path = record.get("image")
        if image_path:
            path = Path(image_path)
            if not path.is_absolute():
                path = self.image_root / path
            if not path.exists():
                raise FileNotFoundError(f"Image not found for record {record['id']}: {path}")
            image = Image.open(path).convert("RGB")
            record["image"] = image
            record["image_path"] = str(path)
            record["image_size"] = image.size
        elif self.require_images:
            raise KeyError(f"Record {record['id']} is missing required image field.")
        return record


class MissingBenchmarkDataset(Dataset):
    def __init__(self, benchmark: str, manifest_path: str | Path | None = None):
        manifest = f" at {manifest_path}" if manifest_path else ""
        raise NotImplementedError(
            f"{benchmark} is scaffolded but not runnable until a JSONL manifest{manifest} is created. "
            "Use docs/experiments.md for the expected manifest schema."
        )


def build_dataset(cfg: Any, split: str | None = None) -> Dataset:
    name = str(cfg.get("name", "manifest")).lower()
    manifest = cfg.get("manifest")
    if split and cfg.get("splits") and split in cfg.splits:
        manifest = cfg.splits[split]
    limit = cfg.get("limit")
    if name in {
        "manifest",
        "refcoco",
        "refcoco+",
        "refcocog",
        "flickr30k_entities",
        "pope",
        "coco_chair",
        "gqa",
        "vqav2",
        "winoground",
        "sugarcrepe",
        "aro",
        "hallusionbench",
    }:
        if manifest is None:
            return MissingBenchmarkDataset(name)
        return ManifestVLDataset(
            manifest_path=manifest,
            image_root=cfg.get("image_root"),
            limit=int(limit) if limit else None,
            require_images=bool(cfg.get("require_images", True)),
        )
    raise ValueError(f"Unknown dataset name: {name}")


def collate_vl(records: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    keys = set().union(*(record.keys() for record in records))
    for key in keys:
        values = [record.get(key) for record in records]
        if key == "grounding_matrix" and all(value is not None for value in values):
            batch[key] = torch.tensor(values, dtype=torch.float32)
        elif key == "input_ids" and all(value is not None for value in values):
            batch[key] = torch.tensor(values, dtype=torch.long)
        else:
            batch[key] = values
    return batch


def build_grounding_targets(
    batch: dict[str, Any],
    num_patches: int,
    num_tokens: int,
    device: torch.device | str,
    *,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    tokenizer: Any | None = None,
    patch_grid: list[int] | None = None,
) -> torch.Tensor | None:
    matrix = batch.get("grounding_matrix")
    if torch.is_tensor(matrix):
        if matrix.shape[-2:] != (num_patches, num_tokens):
            raise ValueError(
                f"grounding_matrix shape must end with {(num_patches, num_tokens)}, got {tuple(matrix.shape)}"
            )
        return matrix.to(device)
    boxes = batch.get("boxes")
    token_spans = batch.get("token_spans")
    if (not token_spans or not any(token_spans)) and batch.get("grounding_text"):
        if input_ids is None or tokenizer is None:
            raise ValueError(
                "grounding_text requires the processed input_ids and tokenizer to resolve "
                "actual model-token positions."
            )
        token_spans = resolve_grounding_token_spans(
            batch,
            input_ids=input_ids,
            tokenizer=tokenizer,
            attention_mask=attention_mask,
        )
    image_sizes = batch.get("image_size")
    if not boxes or not token_spans or not any(token_spans) or not image_sizes:
        return None
    rows, cols = resolve_patch_grid(num_patches, patch_grid)
    targets = torch.zeros((len(boxes), num_patches, num_tokens), device=device)
    for bidx, (sample_boxes, sample_spans, image_size) in enumerate(
        zip(boxes, token_spans, image_sizes, strict=True)
    ):
        width, height = image_size
        if sample_spans and len(sample_spans) == 1 and len(sample_boxes or []) > 1:
            sample_spans = list(sample_spans) * len(sample_boxes)
        if len(sample_boxes or []) != len(sample_spans or []):
            raise ValueError("Every box requires a token span, or one shared phrase span.")
        for box, span in zip(sample_boxes or [], sample_spans or [], strict=True):
            ts, te = int(span[0]), int(span[1])
            if not 0 <= ts < te <= num_tokens:
                raise ValueError(f"Invalid token span {span} for {num_tokens} tokens.")
            mask = box_patch_mask(boxes=[box], width=width, height=height, rows=rows, cols=cols).to(
                device
            )
            targets[bidx, : rows * cols, ts:te] = torch.maximum(
                targets[bidx, : rows * cols, ts:te], mask[:, None].float()
            )
    return targets


def resolve_grounding_token_spans(
    batch: dict[str, Any],
    *,
    input_ids: torch.Tensor,
    tokenizer: Any,
    attention_mask: torch.Tensor | None = None,
) -> list[list[list[int]]]:
    """Locate grounding phrases in the exact multimodal token sequence.

    RefCOCO exports store the referred phrase as text rather than guessing token
    indices before the model-specific processor and optional chat template run.
    This resolver tokenizes the phrase with the backbone tokenizer and searches
    it in the final input ids, so image and chat-template special tokens are
    accounted for.
    """

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
    grounding_text = batch.get("grounding_text")
    if not grounding_text:
        return [[] for _ in range(input_ids.shape[0])]
    if len(grounding_text) != input_ids.shape[0]:
        raise ValueError(
            "grounding_text batch size does not match input_ids: "
            f"{len(grounding_text)} vs {input_ids.shape[0]}"
        )

    resolved: list[list[list[int]]] = []
    for batch_index, value in enumerate(grounding_text):
        phrases = [value] if isinstance(value, str) else list(value or [])
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask must match input_ids, got {tuple(attention_mask.shape)}"
            )
        # Search the full padded row so returned positions remain correct for
        # both left- and right-padding tokenizers.
        sequence = input_ids[batch_index].detach().cpu().tolist()
        sample_spans: list[list[int]] = []
        for phrase in phrases:
            span = _find_tokenized_phrase(sequence, str(phrase), tokenizer)
            if span is None:
                record_ids = batch.get("id", [batch_index] * input_ids.shape[0])
                raise ValueError(
                    f"Could not align grounding_text={phrase!r} to processed tokens for "
                    f"record {record_ids[batch_index]!r}. Keep the phrase verbatim in the prompt."
                )
            sample_spans.append([span[0], span[1]])
        resolved.append(sample_spans)
    return resolved


def _find_tokenized_phrase(
    sequence: list[int],
    phrase: str,
    tokenizer: Any,
) -> tuple[int, int] | None:
    candidates: list[list[int]] = []
    for variant in (phrase, phrase.strip(), f" {phrase.strip()}"):
        if not variant:
            continue
        encoded = tokenizer(variant, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        candidate = [int(value) for value in ids]
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        width = len(candidate)
        for start in range(0, len(sequence) - width + 1):
            if sequence[start : start + width] == candidate:
                return start, start + width
    return None


def apply_grounding_target_control(
    targets: torch.Tensor | None,
    mode: str = "none",
) -> torch.Tensor | None:
    """Apply a negative-control permutation without changing target density.

    These controls distinguish learning from real spatial or token alignment
    from learning driven only by the marginal number of positive labels.
    Randomness follows PyTorch's seeded generator, so runs remain reproducible.
    """

    if targets is None or mode in {"none", "", "null"}:
        return targets
    if targets.ndim != 3:
        raise ValueError(
            f"Grounding targets must have shape [B, P, T], got {tuple(targets.shape)}."
        )

    controlled = targets.clone()
    if mode in {"permute_patches", "permute_both"}:
        for batch_index in range(controlled.shape[0]):
            order = torch.randperm(controlled.shape[1], device=controlled.device)
            controlled[batch_index] = controlled[batch_index, order, :]
    if mode in {"permute_tokens", "permute_both"}:
        for batch_index in range(controlled.shape[0]):
            order = torch.randperm(controlled.shape[2], device=controlled.device)
            controlled[batch_index] = controlled[batch_index, :, order]
    if mode == "shuffle_batch":
        if controlled.shape[0] < 2:
            raise ValueError(
                "shuffle_batch requires batch_size >= 2; use permute_patches for batch-size-one runs."
            )
        order = torch.randperm(controlled.shape[0], device=controlled.device)
        controlled = controlled[order]
    elif mode not in {
        "permute_patches",
        "permute_tokens",
        "permute_both",
        "shuffle_batch",
    }:
        raise ValueError(f"Unknown grounding target control: {mode}")
    return controlled
