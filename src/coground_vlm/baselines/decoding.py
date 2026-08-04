from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class BaselineResult:
    logits: torch.Tensor | None = None
    attention_bias: torch.Tensor | None = None
    penalty: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None


class BaselineAdapter:
    """Common interface for adjacent baseline controls."""

    name = "none"

    def prepare_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch

    def adjust_logits(self, logits: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        return logits

    def attention_bias(self, context: dict[str, Any]) -> torch.Tensor | None:
        return None

    def extra_loss(self, context: dict[str, Any]) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict[str, Any]:
        return {"baseline": self.name}


class VcdBaseline(BaselineAdapter):
    """Visual contrastive decoding proxy using visual and no-image logits."""

    name = "VCD"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def adjust_logits(self, logits: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        no_image_logits = context.get("no_image_logits")
        if no_image_logits is None:
            return logits
        return logits + self.alpha * (logits - no_image_logits)

    def metadata(self) -> dict[str, Any]:
        return {"baseline": self.name, "alpha": self.alpha}


class OperaBaseline(BaselineAdapter):
    """OPERA-style over-trust penalty based on peaked visual attention."""

    name = "OPERA"

    def __init__(self, penalty_weight: float = 1.0, entropy_floor: float = 0.25):
        self.penalty_weight = penalty_weight
        self.entropy_floor = entropy_floor

    def extra_loss(self, context: dict[str, Any]) -> torch.Tensor | None:
        attention = context.get("visual_attention")
        if attention is None:
            return None
        probs = attention.float().clamp_min(1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        overtrust = F.relu(self.entropy_floor - entropy).mean()
        return self.penalty_weight * overtrust

    def metadata(self) -> dict[str, Any]:
        return {
            "baseline": self.name,
            "penalty_weight": self.penalty_weight,
            "entropy_floor": self.entropy_floor,
        }


class PaintBaseline(BaselineAdapter):
    """PAINT-style informed-token attention reweighting proxy."""

    name = "PAINT"

    def __init__(
        self,
        informed_token_weight: float = 1.0,
        local_weight: float = 1.5,
        summary_weight: float = 1.2,
        topk_ratio: float = 0.2,
    ):
        self.informed_token_weight = informed_token_weight
        self.local_weight = local_weight
        self.summary_weight = summary_weight
        self.topk_ratio = topk_ratio

    def attention_bias(self, context: dict[str, Any]) -> torch.Tensor | None:
        informed_mask = context.get("informed_token_mask")
        if informed_mask is None and context.get("vision_attentions") is not None:
            informed_mask = paint_informed_token_mask(
                context["vision_attentions"],
                topk_ratio=self.topk_ratio,
                device=context.get("device"),
            )
        if informed_mask is None:
            return None
        while informed_mask.ndim < 4:
            informed_mask = informed_mask.unsqueeze(1)
        return self.informed_token_weight * informed_mask.float()

    def metadata(self) -> dict[str, Any]:
        return {
            "baseline": self.name,
            "informed_token_weight": self.informed_token_weight,
            "local_weight": self.local_weight,
            "summary_weight": self.summary_weight,
            "topk_ratio": self.topk_ratio,
        }


class SegrosStyleBaseline(BaselineAdapter):
    """Semantically grounded supervision proxy with the shared grounding target format."""

    name = "SeGroS-style"

    def __init__(self, grounding_weight: float = 1.0, discriminative_topk_ratio: float = 0.25):
        self.grounding_weight = grounding_weight
        self.discriminative_topk_ratio = discriminative_topk_ratio

    def extra_loss(self, context: dict[str, Any]) -> torch.Tensor | None:
        logits = context.get("grounding_logits")
        targets = context.get("grounding_targets")
        if logits is None or targets is None:
            return None
        return self.grounding_weight * F.binary_cross_entropy_with_logits(logits, targets.float())

    def metadata(self) -> dict[str, Any]:
        return {
            "baseline": self.name,
            "grounding_weight": self.grounding_weight,
            "discriminative_topk_ratio": self.discriminative_topk_ratio,
        }


def build_baseline(cfg: Any | None) -> BaselineAdapter:
    if cfg is None:
        return BaselineAdapter()
    name = str(cfg.get("name", "none")).lower()
    if name in {"none", "null", ""}:
        return BaselineAdapter()
    if name == "vcd":
        return VcdBaseline(alpha=float(cfg.get("alpha", 1.0)))
    if name == "opera":
        return OperaBaseline(
            penalty_weight=float(cfg.get("penalty_weight", 1.0)),
            entropy_floor=float(cfg.get("entropy_floor", 0.25)),
        )
    if name == "paint":
        return PaintBaseline(
            informed_token_weight=float(cfg.get("informed_token_weight", 1.0)),
            local_weight=float(cfg.get("local_weight", 1.5)),
            summary_weight=float(cfg.get("summary_weight", 1.2)),
            topk_ratio=float(cfg.get("topk_ratio", 0.2)),
        )
    if name in {"segros", "segros-style", "segros_style"}:
        return SegrosStyleBaseline(
            grounding_weight=float(cfg.get("grounding_weight", 1.0)),
            discriminative_topk_ratio=float(cfg.get("discriminative_topk_ratio", 0.25)),
        )
    raise ValueError(f"Unknown baseline: {cfg.get('name')}")


@torch.no_grad()
def generate_with_baseline(
    backbone: Any,
    batch: dict[str, Any],
    device: torch.device | str,
    generation_cfg: Any,
    baseline: BaselineAdapter,
    decoding_cfg: Any | None = None,
    *,
    co_head: Any | None = None,
    decoding_prior: Any | None = None,
    method_cfg: Any | None = None,
) -> list[str]:
    """Generate text with local baseline controls when possible."""
    if decoding_cfg and bool(decoding_cfg.get("enabled", False)):
        mode = str(decoding_cfg.get("mode", "vocab_logit_bias"))
        if baseline.name != "none":
            raise ValueError(
                "Evaluate the CoGround decoding prior and adjacent decoding baselines in "
                "separate runs so their effects remain attributable."
            )
        if mode == "vocab_logit_bias":
            if co_head is None or decoding_prior is None or method_cfg is None:
                raise ValueError(
                    "decoding.mode=vocab_logit_bias requires a loaded CoGround head, "
                    "decoding prior, and method configuration."
                )
            return generate_with_coclustering_logit_prior(
                backbone,
                batch,
                device,
                generation_cfg,
                co_head=co_head,
                decoding_prior=decoding_prior,
                method_cfg=method_cfg,
                decoding_cfg=decoding_cfg,
            )
        if mode == "attention_hook":
            if not hasattr(backbone, "generate_with_attention_bias"):
                raise RuntimeError(
                    "decoding.mode=attention_hook requires a validated backbone-specific "
                    "generate_with_attention_bias implementation."
                )
            return backbone.generate_with_attention_bias(
                batch,
                device,
                generation_cfg,
                co_head=co_head,
                decoding_prior=decoding_prior,
                decoding_cfg=decoding_cfg,
            )
        if bool(decoding_cfg.get("require_true_attention_bias", False)):
            raise RuntimeError(
                "True block-aware attention bias was requested, but this generic HF adapter does not "
                "install model-specific attention hooks."
            )
        raise ValueError(
            f"Unsupported decoding mode {mode!r}. The former logit_proxy mode was removed "
            "because it did not alter generation; use vocab_logit_bias or a validated "
            "attention_hook."
        )
    if isinstance(baseline, VcdBaseline):
        return greedy_vcd_generate(backbone, batch, device, generation_cfg, baseline)
    if isinstance(baseline, OperaBaseline):
        return greedy_opera_generate(backbone, batch, device, generation_cfg, baseline)
    return backbone.generate_text(batch, device, generation_cfg)


class StaticVocabularyPriorProcessor:
    """Transformers-compatible processor that adds an explicit vocabulary bias."""

    def __init__(
        self,
        prior_logits: torch.Tensor,
        *,
        weight: float,
        top_k: int = 0,
        probability_floor: float = 0.0,
    ) -> None:
        self.prior_logits = prior_logits
        self.weight = float(weight)
        self.top_k = int(top_k)
        self.probability_floor = float(probability_floor)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        del input_ids
        prior = self.prior_logits.to(device=scores.device, dtype=scores.dtype)
        if prior.shape != scores.shape:
            if prior.shape[0] == 1 and scores.shape[0] > 1:
                prior = prior.expand(scores.shape[0], -1)
            if prior.shape != scores.shape:
                raise ValueError(
                    f"Vocabulary prior shape {tuple(prior.shape)} does not match generation "
                    f"logits {tuple(scores.shape)}."
                )
        allowed = torch.ones_like(scores, dtype=torch.bool)
        if self.top_k > 0:
            k = min(self.top_k, scores.shape[-1])
            indices = scores.topk(k, dim=-1).indices
            allowed = torch.zeros_like(scores, dtype=torch.bool)
            allowed.scatter_(1, indices, True)
        if self.probability_floor > 0:
            probabilities = torch.softmax(scores.float(), dim=-1)
            relative_floor = self.probability_floor * probabilities.max(dim=-1, keepdim=True).values
            allowed &= probabilities >= relative_floor
        return scores + self.weight * prior.masked_fill(~allowed, 0.0)


@torch.no_grad()
def generate_with_coclustering_logit_prior(
    backbone: Any,
    batch: dict[str, Any],
    device: torch.device | str,
    generation_cfg: Any,
    *,
    co_head: Any,
    decoding_prior: Any,
    method_cfg: Any,
    decoding_cfg: Any,
) -> list[str]:
    """Generate with a block-derived bias that actually changes token logits.

    The generic implementation conditions the block matrix on the final prompt
    state and applies the resulting vocabulary prior at every decoding step.
    It does not claim to be the model-specific attention intervention described
    in the original draft.
    """

    if not hasattr(backbone, "model") or not hasattr(backbone, "processor"):
        raise RuntimeError("The vocabulary logit prior requires a Hugging Face VLM adapter.")
    if getattr(backbone, "use_chat_template", False):
        prompts = backbone._format_prompts(
            batch["prompt"],
            has_images=batch.get("image") is not None,
            generation=True,
        )
        encoded = backbone.processor(
            text=prompts,
            images=batch.get("image"),
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in encoded.items()
        }
    else:
        inputs = backbone.prepare_batch(batch, device)
    inputs = {key: value for key, value in inputs.items() if key != "labels"}

    features = backbone.extract_features(
        inputs,
        layer_tap=int(method_cfg.get("layer_tap", 0)),
        require_attentions=(method_cfg.get("affinity_mode") == "cross_attention"),
    )
    head_dtype = next(co_head.parameters()).dtype
    features.patch_embeddings = features.patch_embeddings.to(head_dtype)
    features.token_embeddings = features.token_embeddings.to(head_dtype)
    co_out = co_head(
        features.patch_embeddings,
        features.token_embeddings,
        cross_attention=(
            features.cross_attention
            if method_cfg.get("affinity_mode") == "cross_attention"
            else None
        ),
        loss_weights={},
        temperature=float(method_cfg.get("temperature", 1.0)),
    )
    candidate_assignments = candidate_vocabulary_assignments(
        backbone,
        decoding_prior if not decoding_prior.share_assignments else co_head,
        device=device,
        chunk_size=int(decoding_cfg.get("vocab_chunk_size", 4096)),
    )
    prompt_hidden = features.token_embeddings[:, -1, :]
    prior_logits = decoding_prior.vocabulary_prior_scores(
        prompt_hidden,
        co_out.patch_assignments,
        candidate_assignments,
        patch_embeddings=features.patch_embeddings,
        normalize=bool(decoding_cfg.get("normalize_prior", True)),
    )

    try:
        from transformers import LogitsProcessorList
    except ImportError as exc:
        raise RuntimeError("transformers is required for prior-guided generation.") from exc

    processor = StaticVocabularyPriorProcessor(
        prior_logits,
        weight=float(decoding_cfg.get("lambda_dec", 0.0)),
        top_k=int(decoding_cfg.get("plausibility_top_k", 128)),
        probability_floor=float(decoding_cfg.get("plausibility_floor", 0.0)),
    )
    generated = backbone.model.generate(
        **inputs,
        max_new_tokens=int(generation_cfg.get("max_new_tokens", 64)),
        do_sample=bool(generation_cfg.get("do_sample", False)),
        temperature=float(generation_cfg.get("temperature", 1.0)),
        logits_processor=LogitsProcessorList([processor]),
    )
    return decode_generated_sequences(
        backbone,
        generated,
        prompt_input_ids=inputs.get("input_ids"),
    )


@torch.no_grad()
def candidate_vocabulary_assignments(
    backbone: Any,
    co_head: Any,
    *,
    device: torch.device | str,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Assign vocabulary embeddings to textual clusters without a large MLP spike."""

    if chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive.")
    embeddings = backbone.input_embeddings()
    if not hasattr(embeddings, "weight"):
        raise RuntimeError("Input embedding module does not expose a vocabulary weight.")
    weight = embeddings.weight
    cache = getattr(co_head, "_candidate_vocabulary_assignments_cache", None)
    cache_key = (
        int(weight.shape[0]),
        int(weight.shape[1]),
        str(torch.device(device)),
        str(weight.dtype),
        weight._version,
        tuple(parameter._version for parameter in co_head.token_head.parameters()),
    )
    if cache is not None and cache[0] == cache_key:
        return cache[1]
    if chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive.")
    chunks = []
    head_dtype = next(co_head.token_head.parameters()).dtype
    for start in range(0, weight.shape[0], chunk_size):
        chunk = weight[start : start + chunk_size].to(device=device, dtype=head_dtype)
        chunks.append(co_head.token_head(chunk, mode=co_head.assignment_mode).float())
    assignments = torch.cat(chunks, dim=0)
    co_head._candidate_vocabulary_assignments_cache = (cache_key, assignments)
    return assignments


def decode_generated_sequences(
    backbone: Any,
    generated: torch.Tensor,
    *,
    prompt_input_ids: torch.Tensor | None,
) -> list[str]:
    """Decode generated continuations without leaking prompt words into metrics."""

    continuation = generated
    if (
        prompt_input_ids is not None
        and generated.ndim == 2
        and generated.shape[1] >= prompt_input_ids.shape[1]
        and torch.equal(
            generated[:, : prompt_input_ids.shape[1]],
            prompt_input_ids.to(generated.device),
        )
    ):
        continuation = generated[:, prompt_input_ids.shape[1] :]
    return backbone.processor.batch_decode(continuation, skip_special_tokens=True)


@torch.no_grad()
def greedy_vcd_generate(
    backbone: Any,
    batch: dict[str, Any],
    device: torch.device | str,
    generation_cfg: Any,
    baseline: VcdBaseline,
) -> list[str]:
    if not hasattr(backbone, "model") or not hasattr(backbone, "processor"):
        return backbone.generate_text(batch, device, generation_cfg)
    inputs = backbone.prepare_batch(batch, device)
    if "input_ids" not in inputs or "pixel_values" not in inputs:
        return backbone.generate_text(batch, device, generation_cfg)
    generated = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(generated))
    pixel_values = inputs["pixel_values"]
    contrast_mode = str(generation_cfg.get("vcd_contrast_mode", "noise"))
    if contrast_mode == "zero":
        contrast_pixel_values = torch.zeros_like(pixel_values)
    else:
        noise_std = float(generation_cfg.get("vcd_noise_std", 0.5))
        contrast_pixel_values = pixel_values + noise_std * torch.randn_like(pixel_values)
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 64))
    eos_id = getattr(backbone.processor.tokenizer, "eos_token_id", None)
    for _ in range(max_new_tokens):
        model_inputs = {
            k: v
            for k, v in inputs.items()
            if k not in {"labels", "input_ids", "attention_mask", "pixel_values"}
        }
        model_inputs.update(
            {"input_ids": generated, "attention_mask": attention_mask, "pixel_values": pixel_values}
        )
        contrast_inputs = dict(model_inputs)
        contrast_inputs["pixel_values"] = contrast_pixel_values
        logits = backbone.model(**model_inputs, return_dict=True).logits[:, -1, :]
        no_image_logits = backbone.model(**contrast_inputs, return_dict=True).logits[:, -1, :]
        adjusted = baseline.adjust_logits(logits, {"no_image_logits": no_image_logits})
        plausibility = float(generation_cfg.get("vcd_plausibility_cutoff", 0.0))
        if plausibility > 0:
            probs = torch.softmax(logits, dim=-1)
            adjusted = adjusted.masked_fill(
                probs < plausibility * probs.max(dim=-1, keepdim=True).values, -torch.inf
            )
        next_token = adjusted.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        if eos_id is not None and bool((next_token == eos_id).all()):
            break
    return decode_generated_sequences(
        backbone,
        generated,
        prompt_input_ids=inputs["input_ids"],
    )


@torch.no_grad()
def greedy_opera_generate(
    backbone: Any,
    batch: dict[str, Any],
    device: torch.device | str,
    generation_cfg: Any,
    baseline: OperaBaseline,
) -> list[str]:
    if not hasattr(backbone, "model") or not hasattr(backbone, "processor"):
        return backbone.generate_text(batch, device, generation_cfg)
    inputs = backbone.prepare_batch(batch, device)
    if "input_ids" not in inputs:
        return backbone.generate_text(batch, device, generation_cfg)
    generated = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(generated))
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 64))
    eos_id = getattr(backbone.processor.tokenizer, "eos_token_id", None)
    rollback_threshold = float(generation_cfg.get("opera_rollback_threshold", 0.85))
    rollback_window = int(generation_cfg.get("opera_rollback_window", 8))
    for _ in range(max_new_tokens):
        model_inputs = {
            k: v for k, v in inputs.items() if k not in {"labels", "input_ids", "attention_mask"}
        }
        model_inputs.update({"input_ids": generated, "attention_mask": attention_mask})
        outputs = backbone.model(**model_inputs, output_attentions=True, return_dict=True)
        logits = outputs.logits[:, -1, :]
        penalty = overtrust_penalty_from_outputs(outputs, baseline.penalty_weight)
        adjusted = logits - penalty.unsqueeze(-1)
        next_token = adjusted.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        if (
            float(penalty.max().item()) > rollback_threshold
            and generated.shape[1] > rollback_window
        ):
            generated = generated[:, :-rollback_window]
            attention_mask = attention_mask[:, :-rollback_window]
        if eos_id is not None and bool((next_token == eos_id).all()):
            break
    return decode_generated_sequences(
        backbone,
        generated,
        prompt_input_ids=inputs["input_ids"],
    )


def overtrust_penalty_from_outputs(outputs: Any, penalty_weight: float) -> torch.Tensor:
    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        logits = outputs.logits
        return logits.new_zeros(logits.shape[0])
    attn = attentions[-1].float()
    if attn.ndim == 4:
        attn = attn[:, :, -1, :].mean(dim=1)
    top = attn.max(dim=-1).values
    entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1)
    return float(penalty_weight) * top / entropy.clamp_min(1e-4)


def paint_informed_token_mask(
    vision_attentions: list[torch.Tensor] | tuple[torch.Tensor, ...],
    topk_ratio: float = 0.2,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Identify local and summary visual tokens from early/deep ViT attentions."""
    if not vision_attentions:
        raise ValueError("PAINT requires vision attentions.")
    early = vision_attentions[min(1, len(vision_attentions) - 1)].float()
    deep = vision_attentions[-1].float()
    if early.ndim == 4:
        early_score = early.mean(dim=1)[:, 0, :]
        deep_score = deep.mean(dim=1)[:, 0, :]
    else:
        early_score = early[:, 0, :]
        deep_score = deep[:, 0, :]
    k = max(1, int(topk_ratio * early_score.shape[-1]))
    local_idx = early_score.topk(k, dim=-1).indices
    summary_idx = deep_score.topk(k, dim=-1).indices
    mask = torch.zeros_like(early_score)
    mask.scatter_(1, local_idx, 1.0)
    mask.scatter_(1, summary_idx, 1.0)
    return mask.to(device) if device is not None else mask


def segros_grounding_map(
    patch_embeddings: torch.Tensor,
    token_embeddings: torch.Tensor,
    text_self_attention: torch.Tensor | None = None,
    cross_attention: torch.Tensor | None = None,
    topk_ratio: float = 0.25,
) -> torch.Tensor:
    """SeGroS-style semantic grounding map from discriminative tokens and patch similarity."""
    sim = torch.einsum("bpd,btd->bpt", patch_embeddings, token_embeddings)
    inter = sim.max(dim=1).values
    if cross_attention is not None:
        ca = cross_attention.float()
        if ca.ndim == 4:
            ca = ca.mean(dim=1)
        if ca.shape[-2:] == (token_embeddings.shape[1], patch_embeddings.shape[1]):
            ca = ca.transpose(-1, -2)
        inter = inter + ca.max(dim=1).values
    if text_self_attention is not None:
        intra = text_self_attention.float()
        if intra.ndim == 4:
            intra = intra.mean(dim=1)
        intra = intra.mean(dim=1)
        token_score = inter + intra
    else:
        token_score = inter
    k = max(1, int(topk_ratio * token_score.shape[-1]))
    keep = token_score.topk(k, dim=-1).indices
    token_mask = torch.zeros_like(token_score)
    token_mask.scatter_(1, keep, 1.0)
    return torch.sigmoid(sim) * token_mask.unsqueeze(1)
