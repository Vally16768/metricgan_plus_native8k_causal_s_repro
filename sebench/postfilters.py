from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


POSTFILTER_MODES = ("none", "sg_residual_soft", "sg_input_floor")
POSTFILTER_PRESETS = ("light", "medium", "aggressive")

_PRESET_VALUES = {
    "light": {"strength": 0.20, "threshold_scale": 1.25, "temperature": 0.15, "min_mask": 0.12},
    "medium": {"strength": 0.35, "threshold_scale": 1.50, "temperature": 0.15, "min_mask": 0.08},
    "aggressive": {"strength": 0.50, "threshold_scale": 1.75, "temperature": 0.15, "min_mask": 0.05},
}


@dataclass(frozen=True)
class SpectralGateConfig:
    mode: str = "none"
    preset: str = "medium"
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    percentile: float = 0.10
    percentile_window: int = 31
    freq_window: int = 5
    time_window: int = 9
    strength: float = 0.35
    threshold_scale: float = 1.50
    temperature: float = 0.15
    min_mask: float = 0.08

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


def resolve_postfilter_config(mode: str = "none", preset: str = "medium") -> SpectralGateConfig:
    mode = (mode or "none").lower()
    preset = (preset or "medium").lower()
    if mode not in POSTFILTER_MODES:
        raise ValueError(f"Unsupported postfilter mode: {mode}")
    if mode == "none":
        return SpectralGateConfig(mode="none", preset=preset)
    if preset not in POSTFILTER_PRESETS:
        raise ValueError(f"Unsupported postfilter preset: {preset}")
    values = _PRESET_VALUES[preset]
    return SpectralGateConfig(mode=mode, preset=preset, **values)


def config_from_metadata(metadata: dict[str, object] | None) -> SpectralGateConfig:
    if not metadata:
        return resolve_postfilter_config("none", "medium")
    mode = str(metadata.get("mode", "none"))
    preset = str(metadata.get("preset", "medium"))
    if mode == "none":
        return resolve_postfilter_config(mode, preset)
    preset_values = _PRESET_VALUES[preset]
    return SpectralGateConfig(
        mode=mode,
        preset=preset,
        n_fft=int(metadata.get("n_fft", 512)),
        hop_length=int(metadata.get("hop_length", 128)),
        win_length=int(metadata.get("win_length", 512)),
        percentile=float(metadata.get("percentile", 0.10)),
        percentile_window=int(metadata.get("percentile_window", 31)),
        freq_window=int(metadata.get("freq_window", 5)),
        time_window=int(metadata.get("time_window", 9)),
        strength=float(metadata.get("strength", preset_values["strength"])),
        threshold_scale=float(metadata.get("threshold_scale", preset_values["threshold_scale"])),
        temperature=float(metadata.get("temperature", preset_values["temperature"])),
        min_mask=float(metadata.get("min_mask", preset_values["min_mask"])),
    )


def _ensure_waveform_batch(wav: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if wav.ndim == 2:
        return wav.unsqueeze(1), True
    if wav.ndim == 3 and wav.shape[1] == 1:
        return wav, False
    raise ValueError("Expected waveform tensor shaped (batch, length) or (batch, 1, length).")


def _average_pool_1d(spec: torch.Tensor, kernel_size: int, dim: int) -> torch.Tensor:
    if kernel_size <= 1:
        return spec
    if dim not in (-1, -2):
        raise ValueError("dim must be -1 or -2 for 4D spectrogram tensors.")
    if dim == -2:
        spec = spec.transpose(-1, -2)
    pad = kernel_size // 2
    pooled = F.avg_pool1d(
        spec.reshape(-1, 1, spec.shape[-1]),
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    ).reshape_as(spec)
    if dim == -2:
        pooled = pooled.transpose(-1, -2)
    return pooled


def _rolling_quantile(magnitude: torch.Tensor, window: int, quantile: float) -> torch.Tensor:
    if window <= 1:
        return magnitude
    pad = window // 2
    padded = F.pad(magnitude, (pad, pad), mode="replicate")
    unfolded = padded.unfold(-1, window, 1)
    return torch.quantile(unfolded, quantile, dim=-1)


def estimate_noise_floor(enhanced_mag: torch.Tensor, noisy_mag: torch.Tensor, config: SpectralGateConfig) -> torch.Tensor:
    source_mag = noisy_mag if config.mode == "sg_input_floor" else (noisy_mag - enhanced_mag).abs()
    floor = _rolling_quantile(source_mag, config.percentile_window, config.percentile)
    floor = _average_pool_1d(floor, config.freq_window, dim=-2)
    floor = _average_pool_1d(floor, config.time_window, dim=-1)
    return floor.clamp_min(1e-6)


def spectral_gate_waveform(enhanced: torch.Tensor, noisy: torch.Tensor, config: SpectralGateConfig) -> torch.Tensor:
    enhanced_3d, squeeze_output = _ensure_waveform_batch(enhanced)
    noisy_3d, _ = _ensure_waveform_batch(noisy)
    if not config.enabled:
        return enhanced_3d.squeeze(1) if squeeze_output else enhanced_3d

    length = enhanced_3d.shape[-1]
    window = torch.hann_window(config.win_length, device=enhanced_3d.device, dtype=enhanced_3d.dtype)

    enhanced_spec = torch.stft(
        enhanced_3d.squeeze(1),
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    noisy_spec = torch.stft(
        noisy_3d.squeeze(1),
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )

    enhanced_mag = enhanced_spec.abs().clamp_min(1e-6)
    noise_floor = estimate_noise_floor(enhanced_mag, noisy_spec.abs().clamp_min(1e-6), config)
    mask = torch.sigmoid(
        (enhanced_mag - config.threshold_scale * noise_floor) /
        (config.temperature * noise_floor + 1e-6)
    )
    final_mask = config.min_mask + (1.0 - config.min_mask) * ((1.0 - config.strength) + config.strength * mask)
    gated_spec = torch.polar(enhanced_mag * final_mask, torch.angle(enhanced_spec))
    gated = torch.istft(
        gated_spec,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        length=length,
    ).unsqueeze(1)
    return gated.squeeze(1) if squeeze_output else gated


class PostFilterEnhancer(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        postfilter_config: SpectralGateConfig,
        apply_in_train: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.postfilter_config = postfilter_config
        self.apply_in_train = apply_in_train

    @property
    def postfilter_active(self) -> bool:
        return self.postfilter_config.enabled

    def _base_model_device(self) -> torch.device:
        try:
            return next(self.base_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _maybe_filter(self, noisy: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        if not self.postfilter_config.enabled:
            return enhanced
        return spectral_gate_waveform(enhanced, noisy, self.postfilter_config)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        enhanced = self.base_model(noisy)
        if self.training and not self.apply_in_train:
            return enhanced
        return self._maybe_filter(noisy, enhanced)

    def denoise_raw(self, noisy: torch.Tensor) -> torch.Tensor:
        noisy_3d, squeeze_output = _ensure_waveform_batch(noisy)
        noisy_3d = noisy_3d.to(self._base_model_device(), non_blocking=True)
        enhanced = self.base_model(noisy_3d)
        return enhanced.squeeze(1) if squeeze_output else enhanced

    def denoise_single(self, noisy: torch.Tensor) -> torch.Tensor:
        noisy_3d, squeeze_output = _ensure_waveform_batch(noisy)
        noisy_3d = noisy_3d.to(self._base_model_device(), non_blocking=True)
        enhanced = self.base_model(noisy_3d)
        gated = self._maybe_filter(noisy_3d, enhanced)
        return gated.squeeze(1) if squeeze_output else gated
