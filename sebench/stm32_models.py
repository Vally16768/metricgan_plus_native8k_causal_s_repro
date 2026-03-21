from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sebench.postfilters import SpectralGateConfig, resolve_postfilter_config, spectral_gate_waveform


STM32_SAMPLE_RATE = 16_000
STM32_N_FFT = 512
STM32_WIN_LENGTH = 320
STM32_HOP_LENGTH = 160
STM32_ERB_BANDS = 32
STM32_CONTEXT_FRAMES = 5
STM32_GUIDANCE_MODE = "spectral_gating"


def frontend_defaults_for_sample_rate(sample_rate: int) -> tuple[int, int, int]:
    if int(sample_rate) <= 8000:
        return 256, 80, 160
    return STM32_N_FFT, STM32_HOP_LENGTH, STM32_WIN_LENGTH


def _erb_scale(freq_hz: torch.Tensor) -> torch.Tensor:
    return 21.4 * torch.log10(1.0 + 0.00437 * freq_hz.clamp_min(0.0))


def _inv_erb_scale(erb_value: torch.Tensor) -> torch.Tensor:
    return (10.0 ** (erb_value / 21.4) - 1.0) / 0.00437


def _fake_quant_tensor(tensor: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    if not tensor.is_floating_point():
        return tensor
    levels = float(2**num_bits - 1)
    max_val = tensor.detach().abs().max()
    if float(max_val) < 1e-8:
        return tensor
    scale = max_val / (levels / 2.0)
    quantized = torch.clamp(torch.round(tensor / scale), min=-(levels / 2.0), max=levels / 2.0)
    dequantized = quantized * scale
    return tensor + (dequantized - tensor).detach()


class QuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, qat: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.qat = qat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = _fake_quant_tensor(self.linear.weight) if self.qat else self.linear.weight
        bias = self.linear.bias
        if self.qat:
            x = _fake_quant_tensor(x)
        return F.linear(x, weight, bias)


class QuantConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        qat: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.qat = qat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = _fake_quant_tensor(self.conv.weight) if self.qat else self.conv.weight
        bias = self.conv.bias
        if self.qat:
            x = _fake_quant_tensor(x)
        return F.conv1d(
            x,
            weight,
            bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class TinyWaveformEnhancer(nn.Module):
    def denoise_single(self, noisy: torch.Tensor) -> torch.Tensor:
        if noisy.ndim != 2:
            raise ValueError("Expected noisy tensor shaped (batch, length).")
        return self.forward(noisy.unsqueeze(1)).squeeze(1)


@dataclass(frozen=True)
class TinySTM32Config:
    erb_bands: int = STM32_ERB_BANDS
    context_frames: int = STM32_CONTEXT_FRAMES
    sample_rate: int = STM32_SAMPLE_RATE
    n_fft: int = STM32_N_FFT
    hop_length: int = STM32_HOP_LENGTH
    win_length: int = STM32_WIN_LENGTH
    qat: bool = False
    guidance_classic: str = "none"

    @property
    def feature_dim(self) -> int:
        return self.erb_bands + 1


def padded_frame_count(length: int, *, n_fft: int = STM32_N_FFT, hop_length: int = STM32_HOP_LENGTH) -> int:
    return max(1, 1 + max(0, length // hop_length))


def _pad_for_stft(wav: torch.Tensor, *, n_fft: int = STM32_N_FFT, hop_length: int = STM32_HOP_LENGTH) -> tuple[torch.Tensor, int]:
    length = wav.shape[-1]
    padded_length = max(length, n_fft)
    remainder = (padded_length - n_fft) % hop_length
    if remainder:
        padded_length += hop_length - remainder
    pad = padded_length - length
    if pad:
        wav = F.pad(wav, (0, pad))
    return wav, pad


def build_erb_filterbank(
    *,
    n_fft: int = STM32_N_FFT,
    sample_rate: int = STM32_SAMPLE_RATE,
    bands: int = STM32_ERB_BANDS,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    freq_bins = n_fft // 2 + 1
    freqs = torch.linspace(0.0, sample_rate / 2.0, freq_bins, device=device, dtype=dtype or torch.float32)
    erb_min = _erb_scale(freqs[:1])
    erb_max = _erb_scale(freqs[-1:])
    edges = torch.linspace(float(erb_min), float(erb_max), bands + 2, device=device, dtype=freqs.dtype)
    hz_edges = _inv_erb_scale(edges)

    bank = torch.zeros(bands, freq_bins, device=device, dtype=freqs.dtype)
    for band_idx in range(bands):
        left = hz_edges[band_idx]
        center = hz_edges[band_idx + 1]
        right = hz_edges[band_idx + 2]
        rising = torch.clamp((freqs - left) / (center - left + 1e-6), min=0.0, max=1.0)
        falling = torch.clamp((right - freqs) / (right - center + 1e-6), min=0.0, max=1.0)
        bank[band_idx] = torch.minimum(rising, falling)
    bank = bank / bank.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return bank


def waveform_to_stft(
    wav: torch.Tensor,
    *,
    n_fft: int = STM32_N_FFT,
    hop_length: int = STM32_HOP_LENGTH,
    win_length: int = STM32_WIN_LENGTH,
) -> tuple[torch.Tensor, int]:
    if wav.ndim == 3:
        wav = wav.squeeze(1)
    window = torch.hann_window(win_length, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    return spec, 0


def stft_to_waveform(
    spec: torch.Tensor,
    *,
    length: int,
    n_fft: int = STM32_N_FFT,
    hop_length: int = STM32_HOP_LENGTH,
    win_length: int = STM32_WIN_LENGTH,
) -> torch.Tensor:
    window = torch.hann_window(win_length, device=spec.device, dtype=spec.real.dtype)
    wav = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length,
    )
    return wav[..., :length]


def project_mag_to_erb(magnitude: torch.Tensor, erb_filterbank: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bft,ef->bet", magnitude, erb_filterbank)


def expand_erb_gains(gains: torch.Tensor, erb_filterbank: torch.Tensor) -> torch.Tensor:
    freq_gains = torch.einsum("bet,ef->bft", gains, erb_filterbank)
    norm = erb_filterbank.sum(dim=0).view(1, -1, 1).clamp_min(1e-6)
    return freq_gains / norm


def waveform_to_erb_mask(
    noisy: torch.Tensor,
    enhanced: torch.Tensor,
    *,
    erb_bands: int = STM32_ERB_BANDS,
    sample_rate: int = STM32_SAMPLE_RATE,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
) -> torch.Tensor:
    fft_value, hop_value, win_value = frontend_defaults_for_sample_rate(sample_rate)
    if n_fft is None:
        n_fft = fft_value
    if hop_length is None:
        hop_length = hop_value
    if win_length is None:
        win_length = win_value
    noisy_spec, _ = waveform_to_stft(noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    enh_spec, _ = waveform_to_stft(enhanced, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    bank = build_erb_filterbank(
        n_fft=n_fft,
        sample_rate=sample_rate,
        bands=erb_bands,
        device=noisy_spec.device,
        dtype=noisy_spec.real.dtype,
    )
    noisy_erb = project_mag_to_erb(noisy_spec.abs().clamp_min(1e-5), bank)
    enh_erb = project_mag_to_erb(enh_spec.abs().clamp_min(1e-5), bank)
    return (enh_erb / noisy_erb.clamp_min(1e-5)).clamp(0.0, 2.0)


def compute_spectral_gating_guidance(
    noisy: torch.Tensor,
    *,
    erb_bands: int = STM32_ERB_BANDS,
    sample_rate: int = STM32_SAMPLE_RATE,
    preset: str = "medium",
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
) -> torch.Tensor:
    config = resolve_postfilter_config("sg_input_floor", preset)
    gated = spectral_gate_waveform(noisy, noisy, config)
    return waveform_to_erb_mask(
        noisy,
        gated,
        erb_bands=erb_bands,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )


class TinySTM32Base(TinyWaveformEnhancer):
    def __init__(self, config: TinySTM32Config) -> None:
        super().__init__()
        self.config = config
        bank = build_erb_filterbank(
            n_fft=config.n_fft,
            sample_rate=config.sample_rate,
            bands=config.erb_bands,
        )
        self.register_buffer("erb_filterbank", bank, persistent=False)
        self.model_config: dict[str, Any] = {
            "erb_bands": config.erb_bands,
            "context_frames": config.context_frames,
            "sample_rate": config.sample_rate,
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "win_length": config.win_length,
            "qat": config.qat,
            "guidance_classic": config.guidance_classic,
        }

    def _build_context(self, feature_frames: torch.Tensor) -> torch.Tensor:
        batch, channels, frames = feature_frames.shape
        context = self.config.context_frames
        if context <= 1:
            return feature_frames.transpose(1, 2)
        left_pad = context - 1
        padded = F.pad(feature_frames, (left_pad, 0), mode="replicate")
        unfolded = padded.unfold(dimension=-1, size=context, step=1)
        return unfolded.permute(0, 2, 1, 3).reshape(batch, frames, channels * context)

    def _extract_features(self, noisy: torch.Tensor, guidance: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noisy_spec, _ = waveform_to_stft(
            noisy,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
        )
        noisy_mag = noisy_spec.abs().clamp_min(1e-5)
        erb_mag = project_mag_to_erb(noisy_mag, self.erb_filterbank)
        log_erb = torch.log1p(erb_mag)
        energy = torch.log1p(noisy.pow(2).mean(dim=-1, keepdim=True)).unsqueeze(-1).expand(-1, -1, log_erb.shape[-1])
        features = torch.cat([log_erb, energy], dim=1)
        if guidance is not None:
            features = torch.cat([features[:, : self.config.erb_bands], guidance, energy], dim=1)
        return noisy_spec, noisy_mag, features

    def _guidance(self, noisy: torch.Tensor, guidance: torch.Tensor | None) -> torch.Tensor | None:
        if guidance is not None:
            return guidance
        if self.config.guidance_classic != STM32_GUIDANCE_MODE:
            return None
        return compute_spectral_gating_guidance(
            noisy,
            erb_bands=self.config.erb_bands,
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
        )

    def _reconstruct(self, noisy_spec: torch.Tensor, gains: torch.Tensor, length: int) -> torch.Tensor:
        freq_gains = expand_erb_gains(gains, self.erb_filterbank)
        enhanced_spec = torch.polar(noisy_spec.abs() * freq_gains.clamp(0.0, 2.0), torch.angle(noisy_spec))
        return stft_to_waveform(
            enhanced_spec,
            length=length,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
        ).unsqueeze(1)

    def stm32_spec(self) -> dict[str, Any]:
        raise NotImplementedError


class TinySTM32FC(TinySTM32Base):
    def __init__(
        self,
        variant: str,
        *,
        erb_bands: int,
        context_frames: int,
        guidance_classic: str,
        qat: bool,
        sample_rate: int = STM32_SAMPLE_RATE,
        n_fft: int = STM32_N_FFT,
        hop_length: int = STM32_HOP_LENGTH,
        win_length: int = STM32_WIN_LENGTH,
    ) -> None:
        hidden1, hidden2 = (128, 64) if variant == "small" else (160, 80)
        config = TinySTM32Config(
            erb_bands=erb_bands,
            context_frames=context_frames,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            qat=qat,
            guidance_classic="none",
        )
        super().__init__(config)
        input_dim = context_frames * (erb_bands + 1)
        self.net = nn.Sequential(
            QuantLinear(input_dim, hidden1, qat=qat),
            nn.SiLU(),
            QuantLinear(hidden1, hidden2, qat=qat),
            nn.SiLU(),
            QuantLinear(hidden2, erb_bands, qat=qat),
        )
        self.variant = variant

    def forward(self, noisy: torch.Tensor, guidance: torch.Tensor | None = None) -> torch.Tensor:
        if noisy.ndim != 3:
            raise ValueError("Expected input tensor shaped (batch, 1, length).")
        wav = noisy.squeeze(1)
        noisy_spec, _, features = self._extract_features(wav)
        context = self._build_context(features)
        gains = torch.sigmoid(self.net(context)).transpose(1, 2)
        return self._reconstruct(noisy_spec, gains, wav.shape[-1])

    def stm32_spec(self) -> dict[str, Any]:
        input_dim = self.config.context_frames * (self.config.erb_bands + 1)
        hidden1 = self.net[0].linear.out_features
        hidden2 = self.net[2].linear.out_features
        return {
            "arch": "tiny_stm32_fc",
            "variant": self.variant,
            "erb_bands": self.config.erb_bands,
            "context_frames": self.config.context_frames,
            "sample_rate": self.config.sample_rate,
            "n_fft": self.config.n_fft,
            "hop_length": self.config.hop_length,
            "win_length": self.config.win_length,
            "input_dim": input_dim,
            "layer_dims": [input_dim, hidden1, hidden2, self.config.erb_bands],
            "qat": self.config.qat,
            "guidance_classic": "none",
        }


class TinySTM32HybridSG(TinySTM32Base):
    def __init__(
        self,
        variant: str,
        *,
        erb_bands: int,
        context_frames: int,
        guidance_classic: str,
        qat: bool,
        sample_rate: int = STM32_SAMPLE_RATE,
        n_fft: int = STM32_N_FFT,
        hop_length: int = STM32_HOP_LENGTH,
        win_length: int = STM32_WIN_LENGTH,
    ) -> None:
        hidden1, hidden2 = (160, 80) if variant == "small" else (192, 96)
        config = TinySTM32Config(
            erb_bands=erb_bands,
            context_frames=context_frames,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            qat=qat,
            guidance_classic=guidance_classic,
        )
        super().__init__(config)
        input_dim = context_frames * (erb_bands * 2 + 1)
        self.net = nn.Sequential(
            QuantLinear(input_dim, hidden1, qat=qat),
            nn.SiLU(),
            QuantLinear(hidden1, hidden2, qat=qat),
            nn.SiLU(),
            QuantLinear(hidden2, erb_bands, qat=qat),
        )
        self.variant = variant

    def forward(self, noisy: torch.Tensor, guidance: torch.Tensor | None = None) -> torch.Tensor:
        if noisy.ndim != 3:
            raise ValueError("Expected input tensor shaped (batch, 1, length).")
        wav = noisy.squeeze(1)
        guidance = self._guidance(wav, guidance)
        noisy_spec, _, features = self._extract_features(wav, guidance=guidance)
        context = self._build_context(features)
        gains = torch.sigmoid(self.net(context)).transpose(1, 2)
        return self._reconstruct(noisy_spec, gains, wav.shape[-1])

    def stm32_spec(self) -> dict[str, Any]:
        input_dim = self.config.context_frames * (self.config.erb_bands * 2 + 1)
        hidden1 = self.net[0].linear.out_features
        hidden2 = self.net[2].linear.out_features
        return {
            "arch": "tiny_stm32_hybrid_sg",
            "variant": self.variant,
            "erb_bands": self.config.erb_bands,
            "context_frames": self.config.context_frames,
            "sample_rate": self.config.sample_rate,
            "n_fft": self.config.n_fft,
            "hop_length": self.config.hop_length,
            "win_length": self.config.win_length,
            "input_dim": input_dim,
            "layer_dims": [input_dim, hidden1, hidden2, self.config.erb_bands],
            "qat": self.config.qat,
            "guidance_classic": self.config.guidance_classic,
        }


class TinySTM32TCNHybrid(TinySTM32Base):
    def __init__(
        self,
        variant: str,
        *,
        erb_bands: int,
        context_frames: int,
        guidance_classic: str,
        qat: bool,
        sample_rate: int = STM32_SAMPLE_RATE,
        n_fft: int = STM32_N_FFT,
        hop_length: int = STM32_HOP_LENGTH,
        win_length: int = STM32_WIN_LENGTH,
    ) -> None:
        channels = 48 if variant == "small" else 64
        config = TinySTM32Config(
            erb_bands=erb_bands,
            context_frames=context_frames,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            qat=qat,
            guidance_classic=guidance_classic,
        )
        super().__init__(config)
        in_channels = erb_bands * 2 + 1
        self.in_proj = QuantConv1d(in_channels, channels, kernel_size=1, qat=qat)
        self.depthwise = nn.ModuleList(
            [
                QuantConv1d(channels, channels, kernel_size=3, padding=2, groups=channels, qat=qat),
                QuantConv1d(channels, channels, kernel_size=3, padding=4, groups=channels, qat=qat),
                QuantConv1d(channels, channels, kernel_size=3, padding=8, groups=channels, qat=qat),
            ]
        )
        self.pointwise = nn.ModuleList(
            [QuantConv1d(channels, channels, kernel_size=1, qat=qat) for _ in range(3)]
        )
        self.head = QuantConv1d(channels, erb_bands, kernel_size=1, qat=qat)
        self.variant = variant

    def forward(self, noisy: torch.Tensor, guidance: torch.Tensor | None = None) -> torch.Tensor:
        if noisy.ndim != 3:
            raise ValueError("Expected input tensor shaped (batch, 1, length).")
        wav = noisy.squeeze(1)
        guidance = self._guidance(wav, guidance)
        noisy_spec, _, features = self._extract_features(wav, guidance=guidance)
        encoded = F.silu(self.in_proj(features))
        for depthwise, pointwise in zip(self.depthwise, self.pointwise):
            residual = encoded
            encoded = F.silu(depthwise(encoded)[..., : residual.shape[-1]])
            encoded = F.silu(pointwise(encoded))
            encoded = encoded + residual
        gains = torch.sigmoid(self.head(encoded))
        return self._reconstruct(noisy_spec, gains, wav.shape[-1])

    def stm32_spec(self) -> dict[str, Any]:
        return {
            "arch": "tiny_stm32_tcn_hybrid",
            "variant": self.variant,
            "erb_bands": self.config.erb_bands,
            "context_frames": self.config.context_frames,
            "sample_rate": self.config.sample_rate,
            "n_fft": self.config.n_fft,
            "hop_length": self.config.hop_length,
            "win_length": self.config.win_length,
            "channels": self.in_proj.conv.out_channels,
            "layers": 3,
            "qat": self.config.qat,
            "guidance_classic": self.config.guidance_classic,
        }
