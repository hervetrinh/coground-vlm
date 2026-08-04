from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config, optional parent configs, and dot-list overrides."""

    cfg = _load_config_tree(Path(path).resolve(), seen=set())
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def _load_config_tree(path: Path, seen: set[Path]) -> DictConfig:
    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise ValueError(f"Cyclic config inheritance: {chain}")
    seen = {*seen, path}
    cfg = OmegaConf.load(path)
    parents = cfg.get("extends")
    if not parents:
        return cfg
    parent_paths = [parents] if isinstance(parents, str) else list(parents)
    merged = OmegaConf.create({})
    for parent in parent_paths:
        parent_path = Path(str(parent))
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = OmegaConf.merge(
            merged,
            _load_config_tree(parent_path.resolve(), seen=seen),
        )
    local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    del local["extends"]
    return OmegaConf.merge(merged, local)


def save_config(cfg: DictConfig | dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=out)


def as_plain_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


def require_keys(cfg: DictConfig, keys: list[str]) -> None:
    missing = [key for key in keys if OmegaConf.select(cfg, key) is None]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")
