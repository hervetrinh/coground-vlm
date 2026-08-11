#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed as accelerate_set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from coground_vlm.baselines import build_baseline
from coground_vlm.config import as_plain_dict, load_config, save_config
from coground_vlm.data.datasets import (
    apply_grounding_target_control,
    build_dataset,
    build_grounding_targets,
    collate_vl,
)
from coground_vlm.models.backbones import build_backbone
from coground_vlm.models.coclustering import (
    coclustering_from_config,
    decoding_prior_from_config,
    grounding_bce_loss,
)
from coground_vlm.models.grounding_baselines import grounding_baseline_from_config
from coground_vlm.train.checkpointing import (
    load_training_checkpoint,
    resolve_resume_checkpoint,
    save_checkpoint,
    save_rng_state,
    write_latest_checkpoint,
)
from coground_vlm.utils.assets import write_asset_registry
from coground_vlm.utils.io import ensure_dir, write_json
from coground_vlm.utils.params import count_parameters
from coground_vlm.utils.repro import RunTimer, gpu_memory_mb, reproducibility_payload


@dataclass
class TrainingProgress:
    global_step: int = 0
    epoch: int = 0
    batches_in_epoch: int = 0

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> TrainingProgress:
        saved = payload.get("progress") or {"global_step": int(payload.get("step", 0))}
        return cls(
            global_step=int(saved.get("global_step", payload.get("step", 0))),
            epoch=int(saved.get("epoch", 0)),
            batches_in_epoch=int(saved.get("batches_in_epoch", 0)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CoGround-VLM variants.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from a checkpoint directory, latest.txt, or the run's latest checkpoint.",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides")
    return parser.parse_args()


def maybe_apply_lora(backbone: torch.nn.Module, cfg: Any) -> torch.nn.Module:
    lora_cfg = cfg.get("lora", {})
    if not lora_cfg or not bool(lora_cfg.get("enabled", False)):
        return backbone
    from peft import LoraConfig, get_peft_model

    target_modules = list(lora_cfg.get("target_modules", ["q_proj", "v_proj"]))
    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("rank", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
    )
    backbone.model = get_peft_model(backbone.model, peft_cfg)
    if hasattr(backbone.model, "enable_input_require_grads"):
        backbone.model.enable_input_require_grads()
    return backbone


def resolve_feature_dim(backbone: torch.nn.Module, cfg: Any) -> int:
    inferred = int(backbone.hidden_size())
    configured = cfg.get("dim", "auto")
    if configured is None or str(configured).lower() == "auto":
        return inferred
    configured_dim = int(configured)
    if configured_dim != inferred:
        raise ValueError(
            f"Configured feature dimension {configured_dim} does not match backbone hidden "
            f"size {inferred}. Set dim: auto for cross-backbone runs."
        )
    return configured_dim


def validate_distributed_contract(cfg: Any, accelerator: Accelerator) -> int:
    runtime = cfg.get("runtime", {})
    expected_world_size = runtime.get("expected_world_size")
    if expected_world_size is not None and int(expected_world_size) != accelerator.num_processes:
        raise RuntimeError(
            "This experiment was configured for "
            f"{expected_world_size} processes but was launched with {accelerator.num_processes}. "
            "Use scripts/train/launch_2xa100.sh or override runtime.expected_world_size."
        )
    if bool(runtime.get("require_cuda", False)) and accelerator.device.type != "cuda":
        raise RuntimeError("This configuration requires CUDA, but the launch is not using a GPU.")
    if (
        str(runtime.get("mixed_precision", "no")) == "bf16"
        and accelerator.device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("The selected GPU does not support bf16 training.")
    if accelerator.num_processes > 1 and cfg.backbone.get("device_map") is not None:
        raise ValueError("backbone.device_map must be null for one-process-per-GPU DDP training.")
    if accelerator.num_processes > 1 and bool(cfg.backbone.get("load_in_4bit", False)):
        raise ValueError(
            "The validated 2-A100 path does not use bitsandbytes 4-bit placement. "
            "Set backbone.load_in_4bit=false."
        )
    effective_batch = (
        int(cfg.training.batch_size)
        * int(cfg.training.get("gradient_accumulation_steps", 1))
        * accelerator.num_processes
    )
    configured_global_batch = cfg.training.get("global_batch_size")
    if configured_global_batch is not None and int(configured_global_batch) != effective_batch:
        raise ValueError(
            "Global batch mismatch: config requests "
            f"{configured_global_batch}, but batch_size x accumulation x world_size = "
            f"{effective_batch}."
        )
    return effective_batch


def set_loader_epoch(loader: Any, epoch: int) -> None:
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(epoch)
        return
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def train() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    seed = int(cfg.seed)
    tracker = cfg.logging.get("tracker")
    tracker_name = (
        None if tracker is None or str(tracker).lower() in {"", "none", "null"} else str(tracker)
    )
    pin_memory = bool(cfg.training.get("pin_memory", True))
    accelerator = Accelerator(
        mixed_precision=str(cfg.runtime.get("mixed_precision", "bf16")),
        gradient_accumulation_steps=int(cfg.training.get("gradient_accumulation_steps", 1)),
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            data_seed=seed,
            non_blocking=pin_memory,
        ),
        log_with=tracker_name,
        project_dir=str(cfg.output_dir),
    )
    accelerate_set_seed(seed, device_specific=True)
    effective_global_batch = validate_distributed_contract(cfg, accelerator)
    timer = RunTimer()
    output_dir = ensure_dir(Path(cfg.output_dir) / str(cfg.experiment_id))
    resume_request = args.resume if args.resume is not None else cfg.training.get("resume_from")
    resume_path = resolve_resume_checkpoint(resume_request, output_dir)
    if resume_path is None and (output_dir / "latest.txt").exists():
        raise RuntimeError(
            f"Run directory already contains checkpoints: {output_dir}. "
            "Use --resume auto or choose another experiment_id."
        )
    if accelerator.is_main_process:
        save_config(cfg, output_dir / "resolved_config.yaml")
    accelerator.wait_for_everyone()

    if tracker_name is not None:
        accelerator.init_trackers(
            project_name=str(cfg.logging.get("project_name", "coground-vlm")),
            config=as_plain_dict(cfg),
        )

    dataset = build_dataset(cfg.dataset, split=cfg.dataset.get("train_split", "train"))
    num_workers = int(cfg.training.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(cfg.training.get("persistent_workers", True)) and num_workers > 0,
        drop_last=bool(cfg.training.get("drop_last", False)),
        collate_fn=collate_vl,
    )
    if len(loader) == 0:
        raise RuntimeError("The training dataloader has no batches.")

    backbone = maybe_apply_lora(build_backbone(cfg.backbone), cfg)
    lora_enabled = bool(cfg.get("lora", {}).get("enabled", False))
    freeze_backbone = bool(cfg.training.get("freeze_backbone", True))
    if freeze_backbone and not lora_enabled:
        for parameter in (
            backbone.model.parameters() if hasattr(backbone, "model") else backbone.parameters()
        ):
            parameter.requires_grad_(False)
        backbone_model = getattr(backbone, "model", backbone)
        if hasattr(backbone_model, "gradient_checkpointing_disable"):
            backbone_model.gradient_checkpointing_disable()
        backbone_model.eval()
    else:
        getattr(backbone, "model", backbone).train()

    feature_dim = resolve_feature_dim(backbone, cfg.method)
    co_head = (
        coclustering_from_config(cfg.method, dim=feature_dim)
        if bool(cfg.method.get("enabled", False))
        else None
    )
    grounding_baseline_cfg = cfg.get("grounding_baseline", {})
    grounding_baseline = (
        grounding_baseline_from_config(grounding_baseline_cfg, dim=feature_dim)
        if bool(grounding_baseline_cfg.get("enabled", False))
        else None
    )
    if co_head is not None and grounding_baseline is not None:
        raise ValueError(
            "A run must train either CoGround or a learned grounding baseline, not both."
        )
    if (
        co_head is not None
        and str(cfg.method.get("affinity_mode", "bilinear")) == "bilinear"
        and float(cfg.losses.get("lambda_ground", 0.0)) <= 0
        and float(cfg.losses.get("lambda_task", 0.0)) <= 0
        and not bool(cfg.method.get("allow_self_referential_objective", False))
    ):
        raise ValueError(
            "A learned bilinear affinity cannot be trained only by reconstructing itself. "
            "Enable grounding/task supervision, use a frozen affinity, or explicitly set "
            "method.allow_self_referential_objective=true for a diagnostic legacy run."
        )
    generation_baseline = build_baseline(cfg.get("baseline"))
    decoding_prior = (
        decoding_prior_from_config(cfg, hidden_dim=feature_dim)
        if co_head is not None and bool(cfg.decoding.get("enabled", False))
        else None
    )

    trainable: list[torch.nn.Parameter] = []
    for module in (co_head, grounding_baseline, decoding_prior, backbone):
        if module is not None:
            trainable.extend(
                parameter for parameter in module.parameters() if parameter.requires_grad
            )
    if not trainable:
        raise RuntimeError(
            "No trainable parameters. Enable a method head, learned baseline, or LoRA."
        )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.get("weight_decay", 0.0)),
        betas=tuple(float(value) for value in cfg.training.get("betas", [0.9, 0.999])),
        eps=float(cfg.training.get("adam_epsilon", 1.0e-8)),
    )
    max_steps = int(cfg.training.max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_steps))

    model_to_prepare = backbone.model if hasattr(backbone, "model") else backbone
    prepare_items: list[Any] = [model_to_prepare, loader, optimizer, scheduler]
    if co_head is not None:
        prepare_items.append(co_head)
    if grounding_baseline is not None:
        prepare_items.append(grounding_baseline)
    if decoding_prior is not None:
        prepare_items.append(decoding_prior)
    prepared = accelerator.prepare(*prepare_items)
    prepared_model, loader, optimizer, scheduler = prepared[:4]
    if hasattr(backbone, "model"):
        backbone.model = prepared_model
    else:
        backbone = prepared_model
    offset = 4
    if co_head is not None:
        co_head = prepared[offset]
        offset += 1
    if grounding_baseline is not None:
        grounding_baseline = prepared[offset]
        offset += 1
    if decoding_prior is not None:
        decoding_prior = prepared[offset]

    progress = TrainingProgress()
    elapsed_before_resume = 0.0
    if resume_path is not None:
        state = load_training_checkpoint(
            resume_path,
            co_head=accelerator.unwrap_model(co_head) if co_head is not None else None,
            grounding_baseline=(
                accelerator.unwrap_model(grounding_baseline)
                if grounding_baseline is not None
                else None
            ),
            decoding_prior=(
                accelerator.unwrap_model(decoding_prior) if decoding_prior is not None else None
            ),
            backbone=accelerator.unwrap_model(backbone.model)
            if hasattr(backbone, "model")
            else None,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_world_size=accelerator.num_processes,
            rank=accelerator.process_index,
        )
        progress = TrainingProgress.from_checkpoint(state)
        elapsed_before_resume = float(state.get("extra", {}).get("wall_clock_seconds", 0.0))
        if progress.global_step > max_steps:
            raise ValueError(
                f"Checkpoint step {progress.global_step} exceeds training.max_steps={max_steps}."
            )

    if freeze_backbone and not lora_enabled:
        getattr(backbone, "model", backbone).eval()
    else:
        getattr(backbone, "model", backbone).train()
    for module in (co_head, grounding_baseline, decoding_prior):
        if module is not None:
            module.train()

    metrics_path = output_dir / "train_metrics.jsonl"
    last_saved_step = progress.global_step if resume_path is not None else -1

    def elapsed_total() -> float:
        return elapsed_before_resume + timer.elapsed_seconds()

    def save_synchronized_checkpoint() -> Path:
        nonlocal last_saved_step
        step = progress.global_step
        checkpoint_dir = output_dir / f"step_{step:08d}"
        if accelerator.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()
        save_rng_state(checkpoint_dir, accelerator.process_index)
        accelerator.wait_for_everyone()
        local_memory = gpu_memory_mb() or 0.0
        memory_tensor = torch.tensor([local_memory], device=accelerator.device)
        max_memory = float(accelerator.gather(memory_tensor).max().item())
        if accelerator.is_main_process:
            save_checkpoint(
                output_dir,
                step=step,
                cfg=cfg,
                co_head=accelerator.unwrap_model(co_head) if co_head is not None else None,
                grounding_baseline=(
                    accelerator.unwrap_model(grounding_baseline)
                    if grounding_baseline is not None
                    else None
                ),
                decoding_prior=(
                    accelerator.unwrap_model(decoding_prior) if decoding_prior is not None else None
                ),
                backbone=(
                    accelerator.unwrap_model(backbone.model)
                    if hasattr(backbone, "model")
                    and any(parameter.requires_grad for parameter in backbone.model.parameters())
                    else None
                ),
                optimizer=optimizer,
                scheduler=scheduler,
                progress=asdict(progress),
                world_size=accelerator.num_processes,
                extra={
                    "wall_clock_seconds": elapsed_total(),
                    "max_gpu_memory_mb_all_ranks": max_memory,
                    "effective_global_batch_size": effective_global_batch,
                },
                update_latest=False,
            )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            write_latest_checkpoint(output_dir, checkpoint_dir)
        accelerator.wait_for_everyone()
        last_saved_step = step
        return checkpoint_dir

    progress_bar = tqdm(
        total=max_steps,
        initial=progress.global_step,
        disable=not accelerator.is_main_process,
        desc=str(cfg.experiment_id),
    )
    save_every = int(cfg.training.get("save_every", max_steps))
    log_every = int(cfg.logging.get("log_every", 10))
    if save_every <= 0 or log_every <= 0:
        raise ValueError("training.save_every and logging.log_every must be positive.")

    while progress.global_step < max_steps:
        set_loader_epoch(loader, progress.epoch)
        epoch_loader = (
            accelerator.skip_first_batches(loader, progress.batches_in_epoch)
            if progress.batches_in_epoch > 0
            else loader
        )
        reached_step_limit = False
        for batch in epoch_loader:
            if progress.global_step >= max_steps:
                reached_step_limit = True
                break
            model_for_accum = backbone.model if hasattr(backbone, "model") else backbone
            accumulation_models = [model_for_accum]
            accumulation_models.extend(
                module
                for module in (co_head, grounding_baseline, decoding_prior)
                if module is not None
            )
            losses: dict[str, torch.Tensor] = {}
            with accelerator.accumulate(*accumulation_models):
                inputs = backbone.prepare_batch(batch, accelerator.device)
                total_loss = torch.zeros((), device=accelerator.device)
                if float(cfg.losses.get("lambda_task", 1.0)) > 0:
                    task_loss = backbone.forward_loss(inputs)
                    total_loss = total_loss + float(cfg.losses.get("lambda_task", 1.0)) * task_loss
                    losses["task"] = task_loss.detach().float()

                features = None
                grounding_targets = None
                if co_head is not None or grounding_baseline is not None:
                    layer_tap = int(
                        grounding_baseline_cfg.get("layer_tap", cfg.method.get("layer_tap", -1))
                    )
                    features = backbone.extract_features(
                        inputs,
                        layer_tap=layer_tap,
                        require_attentions=(cfg.method.get("affinity_mode") == "cross_attention"),
                    )
                    grounding_targets = build_grounding_targets(
                        batch,
                        features.patch_embeddings.shape[1],
                        features.token_embeddings.shape[1],
                        accelerator.device,
                        input_ids=inputs.get("input_ids"),
                        attention_mask=inputs.get("attention_mask"),
                        tokenizer=backbone.tokenizer(),
                        patch_grid=cfg.dataset.get("patch_grid"),
                    )
                    grounding_targets = apply_grounding_target_control(
                        grounding_targets,
                        mode=str(cfg.get("target_control", {}).get("mode", "none")),
                    )
                    target_required = (
                        float(cfg.losses.get("lambda_ground", 0.0)) > 0
                        or grounding_baseline is not None
                        or decoding_prior is not None
                    )
                    if grounding_targets is None and target_required:
                        raise RuntimeError(
                            "This run requires grounding targets, but the batch provides neither "
                            "boxes plus token_spans/grounding_text nor a grounding_matrix."
                        )

                if co_head is not None:
                    if features is None:
                        raise RuntimeError("CoGround features were not constructed.")
                    co_out = co_head(
                        features.patch_embeddings,
                        features.token_embeddings,
                        cross_attention=(
                            features.cross_attention
                            if cfg.method.get("affinity_mode") == "cross_attention"
                            else None
                        ),
                        grounding_targets=grounding_targets,
                        loss_weights=cfg.losses,
                        temperature=float(cfg.method.get("temperature", 1.0)),
                    )
                    if decoding_prior is not None:
                        prior_scores = decoding_prior.token_prior_scores(
                            features.token_embeddings,
                            co_out.patch_assignments,
                            co_out.token_assignments,
                            patch_embeddings=features.patch_embeddings,
                            token_embeddings=features.token_embeddings,
                            stopgrad_assignments=bool(
                                cfg.decoding.get("stopgrad_assignments", True)
                            ),
                        )
                        if grounding_targets is not None:
                            prior_loss = grounding_bce_loss(
                                prior_scores,
                                grounding_targets,
                                positive_weight=cfg.losses.get("grounding_pos_weight", "auto"),
                                max_auto_weight=float(
                                    cfg.losses.get("grounding_max_pos_weight", 100.0)
                                ),
                            )
                            total_loss = (
                                total_loss
                                + float(
                                    cfg.losses.get(
                                        "lambda_dec", cfg.decoding.get("lambda_dec", 0.0)
                                    )
                                )
                                * prior_loss
                            )
                            losses["decoding_prior_bce"] = prior_loss.detach().float()
                    baseline_loss = generation_baseline.extra_loss(
                        {
                            "grounding_logits": co_out.reconstructed_affinity,
                            "grounding_targets": grounding_targets,
                        }
                    )
                    if baseline_loss is not None:
                        total_loss = total_loss + baseline_loss
                        losses["baseline_extra"] = baseline_loss.detach().float()
                    total_loss = total_loss + co_out.losses["total_coclustering"]
                    losses.update(
                        {key: value.detach().float() for key, value in co_out.losses.items()}
                    )

                if grounding_baseline is not None:
                    if features is None or grounding_targets is None:
                        raise RuntimeError(
                            "The learned baseline requires features and grounding targets."
                        )
                    baseline_out = grounding_baseline(
                        features.patch_embeddings,
                        features.token_embeddings,
                        grounding_targets=grounding_targets,
                        grounding_weight=float(grounding_baseline_cfg.get("grounding_weight", 1.0)),
                        positive_weight=cfg.losses.get("grounding_pos_weight", "auto"),
                        max_auto_weight=float(cfg.losses.get("grounding_max_pos_weight", 100.0)),
                    )
                    total_loss = total_loss + baseline_out.losses["total"]
                    losses.update(
                        {
                            f"learned_baseline_{key}": value.detach().float()
                            for key, value in baseline_out.losses.items()
                        }
                    )

                if not total_loss.requires_grad:
                    raise RuntimeError(
                        "The configured total loss is not connected to trainable parameters."
                    )
                losses["total"] = total_loss.detach().float()
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable,
                        float(cfg.training.get("max_grad_norm", 1.0)),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.batches_in_epoch += 1
            if not accelerator.sync_gradients:
                continue
            progress.global_step += 1
            progress_bar.update(1)

            if progress.global_step % log_every == 0 or progress.global_step == 1:
                reduced_losses = {
                    key: float(accelerator.reduce(value, reduction="mean").item())
                    for key, value in losses.items()
                }
                log_payload = {
                    "step": progress.global_step,
                    "epoch": progress.epoch,
                    "batches_in_epoch": progress.batches_in_epoch,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "effective_global_batch_size": effective_global_batch,
                    **{f"loss/{key}": value for key, value in reduced_losses.items()},
                }
                if accelerator.is_main_process:
                    append_jsonl(metrics_path, log_payload)
                    progress_bar.set_postfix(
                        {key: f"{value:.4f}" for key, value in reduced_losses.items()}
                    )
                if tracker_name is not None:
                    accelerator.log(log_payload, step=progress.global_step)

            if progress.global_step % save_every == 0:
                save_synchronized_checkpoint()
            if progress.global_step >= max_steps:
                reached_step_limit = True
                break

        if reached_step_limit:
            break
        progress.epoch += 1
        progress.batches_in_epoch = 0

    progress_bar.close()
    final_dir = (
        save_synchronized_checkpoint()
        if last_saved_step != progress.global_step
        else output_dir / f"step_{progress.global_step:08d}"
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        write_json(
            output_dir / "run_metadata.json",
            reproducibility_payload(
                cfg,
                {
                    "experiment_id": str(cfg.experiment_id),
                    "checkpoint": str(final_dir.resolve()),
                    "resumed_from": str(resume_path) if resume_path is not None else None,
                    "optimizer_steps": progress.global_step,
                    "epochs_started": progress.epoch + 1,
                    "world_size": accelerator.num_processes,
                    "per_device_batch_size": int(cfg.training.batch_size),
                    "gradient_accumulation_steps": int(
                        cfg.training.get("gradient_accumulation_steps", 1)
                    ),
                    "effective_global_batch_size": effective_global_batch,
                    "wall_clock_seconds": elapsed_total(),
                    "backbone_model_id": str(
                        cfg.backbone.get("model_id", cfg.backbone.get("kind"))
                    ),
                    "feature_dim": feature_dim,
                    "method_enabled": bool(cfg.method.get("enabled", False)),
                    "grounding_baseline_enabled": grounding_baseline is not None,
                    "grounding_target_control": str(
                        cfg.get("target_control", {}).get("mode", "none")
                    ),
                    "decoding_prior_enabled": decoding_prior is not None,
                    "parameter_count_total": count_parameters(backbone)
                    + (count_parameters(co_head) if co_head is not None else 0)
                    + (
                        count_parameters(grounding_baseline)
                        if grounding_baseline is not None
                        else 0
                    )
                    + (count_parameters(decoding_prior) if decoding_prior is not None else 0),
                    "parameter_count_trainable": count_parameters(backbone, trainable_only=True)
                    + (count_parameters(co_head, trainable_only=True) if co_head is not None else 0)
                    + (
                        count_parameters(grounding_baseline, trainable_only=True)
                        if grounding_baseline is not None
                        else 0
                    )
                    + (
                        count_parameters(decoding_prior, trainable_only=True)
                        if decoding_prior is not None
                        else 0
                    ),
                },
            ),
        )
        write_asset_registry(
            output_dir / "asset_registry.json",
            cfg,
            split=cfg.dataset.get("train_split", "train"),
            checkpoint=str(final_dir.resolve()),
        )
    accelerator.end_training()


if __name__ == "__main__":
    train()
