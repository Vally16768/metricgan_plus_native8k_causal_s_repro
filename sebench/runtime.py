from __future__ import annotations

import torch


def require_cuda_device(device: str | None) -> str:
    requested = (device or "").strip().lower()
    if requested in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if not requested.startswith("cuda"):
        raise ValueError(f"Unsupported device {device!r}. Use cpu, cuda or cuda:N.")
    if not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but CUDA is not available on this machine.")
    if requested == "cuda":
        return requested
    try:
        index = int(requested.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid CUDA device {device!r}. Use cuda or cuda:N.") from exc
    device_count = torch.cuda.device_count()
    if index < 0 or index >= device_count:
        raise ValueError(f"Invalid CUDA device index {index}; available device count is {device_count}.")
    return requested
