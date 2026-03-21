from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch

from .models import build_enhancer
from .postfilters import config_from_metadata


CHECKPOINT_FORMAT = "sebench_checkpoint"
CHECKPOINT_VERSION = 2
_BUILD_ENHANCER_KWARGS = {
    name
    for name in inspect.signature(build_enhancer).parameters
    if name not in {"model_family", "variant"}
}


def _unwrap_model(model: torch.nn.Module) -> tuple[torch.nn.Module, dict[str, Any]]:
    metadata = {
        "postfilter": {
            "mode": "none",
            "preset": "medium",
        },
        "train_postfilter": False,
        "spectral_native_gate": False,
        "model_config": {},
    }
    base_model = model
    if hasattr(model, "postfilter_config"):
        metadata["postfilter"] = model.postfilter_config.to_metadata()
        metadata["train_postfilter"] = bool(getattr(model, "apply_in_train", False))
        base_model = model.base_model
    metadata["spectral_native_gate"] = bool(getattr(base_model, "spectral_native_gate", False) or getattr(base_model, "gate_head", None) is not None)
    metadata["model_config"] = dict(getattr(base_model, "model_config", {}))
    return base_model, metadata


def checkpoint_payload(
    model: torch.nn.Module,
    model_family: str,
    variant: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_model, metadata = _unwrap_model(model)
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_family": model_family,
        "variant": variant,
        "postfilter": metadata["postfilter"],
        "train_postfilter": metadata["train_postfilter"],
        "spectral_native_gate": metadata["spectral_native_gate"],
        "model_config": metadata["model_config"],
        "state_dict": base_model.state_dict(),
        "extra": extra or {},
    }


def save_checkpoint_package(
    path: str | Path,
    model: torch.nn.Module,
    model_family: str,
    variant: str,
    extra: dict[str, Any] | None = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, model_family, variant, extra=extra), out_path)


def load_checkpoint_package(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    raw = torch.load(Path(path), map_location=map_location)
    if isinstance(raw, dict) and "state_dict" in raw and "model_family" in raw:
        raw.setdefault("postfilter", {"mode": "none", "preset": "medium"})
        raw.setdefault("train_postfilter", False)
        raw.setdefault("spectral_native_gate", False)
        raw.setdefault("model_config", {})
        return raw
    if isinstance(raw, dict):
        return {
            "format": "legacy_atennuate_checkpoint",
            "version": 0,
            "model_family": "atennuate",
            "variant": "base",
            "postfilter": {"mode": "none", "preset": "medium"},
            "train_postfilter": False,
            "spectral_native_gate": False,
            "model_config": {},
            "state_dict": raw,
            "extra": {},
        }
    raise ValueError(f"Unsupported checkpoint structure in {path}")


def load_model_from_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    model_family: str | None = None,
    variant: str | None = None,
    postfilter_mode: str | None = None,
    postfilter_preset: str | None = None,
    train_postfilter: bool | None = None,
    spectral_native_gate: bool | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    package = load_checkpoint_package(path, map_location="cpu")
    family = model_family or package["model_family"]
    model_variant = variant or package.get("variant", "base")
    postfilter_metadata = config_from_metadata(package.get("postfilter"))
    model_config = {
        key: value
        for key, value in dict(package.get("model_config", {})).items()
        if key in _BUILD_ENHANCER_KWARGS
    }
    model = build_enhancer(
        family,
        model_variant,
        spectral_native_gate=package.get("spectral_native_gate", False) if spectral_native_gate is None else spectral_native_gate,
        postfilter_mode=postfilter_metadata.mode if postfilter_mode is None else postfilter_mode,
        postfilter_preset=postfilter_metadata.preset if postfilter_preset is None else postfilter_preset,
        train_postfilter=package.get("train_postfilter", False) if train_postfilter is None else train_postfilter,
        **model_config,
    )
    target = model.base_model if hasattr(model, "base_model") else model
    target.load_state_dict(package["state_dict"])
    model.to(device)
    model.eval()
    return model, package
