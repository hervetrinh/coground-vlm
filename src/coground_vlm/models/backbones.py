from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class FeatureBatch:
    patch_embeddings: torch.Tensor
    token_embeddings: torch.Tensor
    cross_attention: torch.Tensor | None = None
    model_outputs: Any | None = None


class BaseVLMAdapter(nn.Module):
    name = "base"

    def prepare_batch(self, batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
        raise NotImplementedError

    def forward_loss(self, inputs: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def extract_features(
        self,
        inputs: dict[str, Any],
        layer_tap: int = -1,
        require_attentions: bool = False,
    ) -> FeatureBatch:
        raise NotImplementedError

    def hidden_size(self) -> int:
        raise NotImplementedError

    def input_embeddings(self) -> nn.Module:
        raise NotImplementedError

    def tokenizer(self) -> Any | None:
        return None

    @torch.no_grad()
    def generate_text(
        self, batch: dict[str, Any], device: torch.device | str, generation_cfg: Any
    ) -> list[str]:
        raise NotImplementedError


class HuggingFaceVLMAdapter(BaseVLMAdapter):
    """Generic HF VLM adapter for PaliGemma/LLaVA-family models."""

    def __init__(
        self,
        model_id: str,
        torch_dtype: str = "bfloat16",
        device_map: str | None = None,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        gradient_checkpointing: bool = False,
        use_chat_template: bool = False,
    ):
        super().__init__()
        from transformers import AutoProcessor

        self.model_id = model_id
        self.use_chat_template = use_chat_template
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        dtype = dtype_from_name(torch_dtype)
        self.model = load_hf_vlm_model(
            model_id=model_id,
            dtype=dtype,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
        )
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def prepare_batch(self, batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
        prompts = batch["prompt"]
        images = batch.get("image")
        labels = batch.get("label_text")
        if isinstance(labels, list) and all(label is None for label in labels):
            labels = None
        prompts = self._format_prompts(prompts, has_images=images is not None, generation=False)
        if labels is not None:
            try:
                encoded = self.processor(
                    text=prompts,
                    images=images,
                    suffix=labels,
                    padding=True,
                    return_tensors="pt",
                )
                if "labels" in encoded:
                    return {
                        key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in encoded.items()
                    }
            except TypeError:
                pass
        encoded = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        if labels is not None:
            raise ValueError(
                "This processor does not provide aligned suffix labels. Task-loss training "
                "requires a model-specific adapter that appends answers and masks prompt tokens."
            )
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in encoded.items()
        }

    def _format_prompts(self, prompts: list[str], has_images: bool, generation: bool) -> list[str]:
        if not self.use_chat_template or not hasattr(self.processor, "apply_chat_template"):
            return prompts
        formatted = []
        for prompt in prompts:
            content = []
            if has_images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            formatted.append(
                self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=generation,
                    tokenize=False,
                )
            )
        return formatted

    def forward_loss(self, inputs: dict[str, Any]) -> torch.Tensor:
        outputs = self.model(**inputs, return_dict=True)
        if not hasattr(outputs, "loss") or outputs.loss is None:
            raise RuntimeError(
                "Backbone did not return a loss; provide label_text in the dataset manifest."
            )
        return outputs.loss

    def hidden_size(self) -> int:
        model = unwrap_model_container(self.model)
        candidates = [
            getattr(model, "config", None),
            getattr(getattr(model, "config", None), "text_config", None),
            getattr(getattr(model, "config", None), "language_config", None),
            getattr(getattr(model, "config", None), "llm_config", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            value = getattr(candidate, "hidden_size", None)
            if value is not None:
                return int(value)
        embeddings = self.input_embeddings()
        if hasattr(embeddings, "embedding_dim"):
            return int(embeddings.embedding_dim)
        if hasattr(embeddings, "weight"):
            return int(embeddings.weight.shape[-1])
        raise RuntimeError("Could not infer the language hidden size from the backbone.")

    def input_embeddings(self) -> nn.Module:
        model = unwrap_model_container(self.model)
        candidates = [model, getattr(model, "model", None)]
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "get_input_embeddings"):
                embeddings = candidate.get_input_embeddings()
                if embeddings is not None:
                    return embeddings
        raise RuntimeError("The Hugging Face VLM does not expose input embeddings.")

    def tokenizer(self) -> Any | None:
        return getattr(self.processor, "tokenizer", None)

    def extract_features(
        self,
        inputs: dict[str, Any],
        layer_tap: int = -1,
        require_attentions: bool = False,
    ) -> FeatureBatch:
        outputs = self.model(
            **{k: v for k, v in inputs.items() if k != "labels"},
            output_hidden_states=True,
            output_attentions=require_attentions,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            hidden_states = getattr(
                getattr(outputs, "language_model_outputs", None), "hidden_states", None
            )
        if hidden_states is None:
            raise RuntimeError(
                "Backbone did not expose hidden_states with output_hidden_states=True."
            )
        token_embeddings = hidden_states[layer_tap]
        patch_embeddings = extract_projected_vision_features(outputs)
        if patch_embeddings is None:
            patch_embeddings = self._extract_vision_features(inputs)
        cross_attention = extract_cross_attention(outputs) if require_attentions else None
        if patch_embeddings.shape[-1] != token_embeddings.shape[-1]:
            model = unwrap_model_container(self.model)
            projector = first_existing_attr(
                model,
                [
                    "multi_modal_projector",
                    "multimodal_projector",
                    "visual_projection",
                    "vision_projector",
                    "embed_vision",
                ],
            )
            if projector is None and hasattr(model, "model"):
                projector = first_existing_attr(
                    model.model,
                    [
                        "multi_modal_projector",
                        "multimodal_projector",
                        "visual_projection",
                        "vision_projector",
                        "embed_vision",
                    ],
                )
            if projector is not None:
                patch_embeddings = projector(patch_embeddings)
        if patch_embeddings.shape[-1] != token_embeddings.shape[-1]:
            raise RuntimeError(
                "Patch and token feature dimensions differ and no known projector was found: "
                f"{patch_embeddings.shape[-1]} vs {token_embeddings.shape[-1]}."
            )
        return FeatureBatch(
            patch_embeddings=patch_embeddings,
            token_embeddings=token_embeddings,
            cross_attention=cross_attention,
            model_outputs=outputs,
        )

    def _extract_vision_features(self, inputs: dict[str, Any]) -> torch.Tensor:
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("Processor inputs did not contain pixel_values.")
        model = unwrap_model_container(self.model)
        vision = first_existing_attr(
            model,
            ["vision_tower", "vision_model", "vision_encoder", "visual"],
        )
        if vision is None and hasattr(model, "get_vision_tower"):
            vision = model.get_vision_tower()
        if vision is None and hasattr(model, "model"):
            vision = first_existing_attr(
                model.model,
                ["vision_tower", "vision_model", "vision_encoder", "visual"],
            )
        if vision is None and hasattr(model, "model"):
            vision = first_existing_attr(
                getattr(model.model, "multi_modal_projector", None),
                ["vision_model", "vision_encoder"],
            )
        if vision is None:
            raise RuntimeError("Could not locate a vision tower on the HF model.")
        try:
            vision_dtype = next(vision.parameters()).dtype
        except StopIteration:
            vision_dtype = pixel_values.dtype
        if pixel_values.is_floating_point() and pixel_values.dtype != vision_dtype:
            pixel_values = pixel_values.to(dtype=vision_dtype)
        kw = {"pixel_values": pixel_values, "output_hidden_states": True, "return_dict": True}
        if inputs.get("pixel_position_ids") is not None:
            kw["pixel_position_ids"] = inputs["pixel_position_ids"]
        elif inputs.get("image_position_ids") is not None:
            kw["pixel_position_ids"] = inputs["image_position_ids"]
        if inputs.get("image_grid_thw") is not None:
            kw["grid_thw"] = inputs["image_grid_thw"]
        try:
            outputs = vision(**kw)
        except TypeError:
            outputs = vision(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        feat = None
        if hasattr(outputs, "last_hidden_state"):
            feat = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
            feat = outputs.hidden_states[-1]
        elif isinstance(outputs, tuple):
            feat = outputs[0]
        if feat is None:
            raise RuntimeError("Vision tower did not expose patch embeddings.")
        # Some vision towers (e.g. Gemma 4) return a 2D tensor [N, d] without a batch axis.
        if feat.dim() == 2:
            batch_size = pixel_values.shape[0] if pixel_values.dim() >= 2 else 1
            feat = feat.view(batch_size, -1, feat.shape[-1])
        return feat

    @torch.no_grad()
    def generate_text(
        self, batch: dict[str, Any], device: torch.device | str, generation_cfg: Any
    ) -> list[str]:
        if self.use_chat_template:
            prompts = self._format_prompts(
                batch["prompt"], has_images=batch.get("image") is not None, generation=True
            )
            encoded = self.processor(
                text=prompts, images=batch.get("image"), padding=True, return_tensors="pt"
            )
            inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in encoded.items()
            }
        else:
            inputs = self.prepare_batch(batch, device)
        generate_kwargs = {
            "max_new_tokens": int(generation_cfg.get("max_new_tokens", 64)),
            "do_sample": bool(generation_cfg.get("do_sample", False)),
            "temperature": float(generation_cfg.get("temperature", 1.0)),
        }
        generated = self.model.generate(
            **{k: v for k, v in inputs.items() if k != "labels"},
            **generate_kwargs,
        )
        prompt_ids = inputs.get("input_ids")
        continuation = generated
        if (
            prompt_ids is not None
            and generated.ndim == 2
            and generated.shape[1] >= prompt_ids.shape[1]
            and torch.equal(
                generated[:, : prompt_ids.shape[1]],
                prompt_ids.to(generated.device),
            )
        ):
            continuation = generated[:, prompt_ids.shape[1] :]
        return self.processor.batch_decode(continuation, skip_special_tokens=True)


class MockVLMAdapter(BaseVLMAdapter):
    """Small deterministic adapter for unit tests and synthetic plumbing."""

    name = "mock"

    def __init__(self, dim: int = 64, num_patches: int = 32, vocab_size: int = 128):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.emb = nn.Embedding(vocab_size, dim)
        self.image_proj = nn.Linear(3, dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def prepare_batch(self, batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
        input_ids = batch.get("input_ids")
        if input_ids is None:
            size = len(batch["prompt"])
            input_ids = torch.arange(8, device=device).unsqueeze(0).repeat(size, 1)
        return {"input_ids": input_ids.to(device)}

    def forward_loss(self, inputs: dict[str, Any]) -> torch.Tensor:
        hidden = self.emb(inputs["input_ids"])
        logits = self.lm_head(hidden)
        return logits.float().pow(2).mean()

    def hidden_size(self) -> int:
        return self.dim

    def input_embeddings(self) -> nn.Module:
        return self.emb

    def extract_features(
        self,
        inputs: dict[str, Any],
        layer_tap: int = -1,
        require_attentions: bool = False,
    ) -> FeatureBatch:
        del layer_tap, require_attentions
        token_embeddings = self.emb(inputs["input_ids"])
        pooled = token_embeddings.mean(dim=1, keepdim=True)
        patch_embeddings = pooled.repeat(1, self.num_patches, 1)
        return FeatureBatch(patch_embeddings=patch_embeddings, token_embeddings=token_embeddings)

    @torch.no_grad()
    def generate_text(
        self, batch: dict[str, Any], device: torch.device | str, generation_cfg: Any
    ) -> list[str]:
        return ["mock output" for _ in batch["prompt"]]


def load_hf_vlm_model(
    model_id: str,
    dtype: torch.dtype,
    device_map: str | None,
    load_in_4bit: bool,
    trust_remote_code: bool,
) -> nn.Module:
    import transformers

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if device_map:
        kwargs["device_map"] = device_map
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
    class_names = [
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ]
    errors = []
    for class_name in class_names:
        cls = getattr(transformers, class_name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(model_id, **kwargs)
        except Exception as exc:
            errors.append(f"{class_name}: {exc}")
    raise RuntimeError(f"Could not load HF VLM model {model_id}. Attempts: {' | '.join(errors)}")


def build_backbone(cfg: Any) -> BaseVLMAdapter:
    kind = str(cfg.get("kind", "hf")).lower()
    if kind == "mock":
        return MockVLMAdapter(
            dim=int(cfg.get("dim", 64)),
            num_patches=int(cfg.get("num_patches", 32)),
            vocab_size=int(cfg.get("vocab_size", 128)),
        )
    if kind in {"hf", "paligemma2", "llava", "llava-onevision"}:
        return HuggingFaceVLMAdapter(
            model_id=str(cfg.model_id),
            torch_dtype=str(cfg.get("torch_dtype", "bfloat16")),
            device_map=cfg.get("device_map"),
            load_in_4bit=bool(cfg.get("load_in_4bit", False)),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
            use_chat_template=bool(cfg.get("use_chat_template", False)),
        )
    raise ValueError(f"Unknown backbone kind: {kind}")


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unknown torch dtype: {name}")


def first_existing_attr(obj: Any, names: list[str]) -> Any | None:
    if obj is None:
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def unwrap_model_container(model: nn.Module) -> nn.Module:
    """Unwrap DDP/Accelerate containers while preserving the HF inner ``model`` field."""

    current = model
    seen: set[int] = set()
    while hasattr(current, "module") and id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "module")
        if not isinstance(wrapped, nn.Module):
            break
        current = wrapped
    return current


def extract_cross_attention(outputs: Any) -> torch.Tensor | None:
    cross = getattr(outputs, "cross_attentions", None)
    if cross:
        return cross[-1]
    attentions = getattr(outputs, "attentions", None)
    if attentions:
        return attentions[-1]
    return None


def extract_projected_vision_features(outputs: Any) -> torch.Tensor | None:
    """Reuse projected image tokens returned by modern HF VLMs when available."""

    candidates = [
        getattr(outputs, "image_hidden_states", None),
        getattr(outputs, "vision_hidden_states", None),
        getattr(getattr(outputs, "language_model_outputs", None), "image_hidden_states", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (tuple, list)):
            if not candidate:
                continue
            candidate = candidate[-1]
        if torch.is_tensor(candidate):
            if candidate.dim() == 2:
                candidate = candidate.unsqueeze(0)
            if candidate.dim() == 4:
                candidate = candidate.flatten(1, 2)
            if candidate.dim() == 3:
                return candidate
    return None
