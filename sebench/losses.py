from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sebench.stm32_models import waveform_to_erb_mask


class SISDRLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if enhanced.dim() == 3:
            enhanced = enhanced.squeeze(1)
            clean = clean.squeeze(1)

        clean = clean - clean.mean(dim=-1, keepdim=True)
        enhanced = enhanced - enhanced.mean(dim=-1, keepdim=True)

        proj = (enhanced * clean).sum(dim=-1, keepdim=True) * clean
        proj = proj / (clean.pow(2).sum(dim=-1, keepdim=True) + self.eps)
        noise = enhanced - proj

        ratio = (proj.pow(2).sum(dim=-1) + self.eps) / (noise.pow(2).sum(dim=-1) + self.eps)
        return -10.0 * torch.log10(ratio).mean()


class ComplexSTFTLoss(nn.Module):
    def __init__(self, n_ffts: tuple[int, ...] = (256, 512, 1024)):
        super().__init__()
        self.n_ffts = n_ffts

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if enhanced.dim() == 3:
            enhanced = enhanced.squeeze(1)
            clean = clean.squeeze(1)

        total = enhanced.new_tensor(0.0)
        for n_fft in self.n_ffts:
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=enhanced.device)
            enh_spec = torch.stft(
                enhanced,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            clean_spec = torch.stft(
                clean,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            real_l1 = torch.mean(torch.abs(enh_spec.real - clean_spec.real))
            imag_l1 = torch.mean(torch.abs(enh_spec.imag - clean_spec.imag))
            mag_l1 = torch.mean(torch.abs(enh_spec.abs() - clean_spec.abs()))
            total = total + (real_l1 + imag_l1 + mag_l1) / 3.0
        return total / float(len(self.n_ffts))


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    wave: torch.Tensor
    spectral: torch.Tensor
    sisdr: torch.Tensor
    noise_gate: torch.Tensor
    speech_preserve: torch.Tensor
    teacher_mask: torch.Tensor
    teacher_wave: torch.Tensor


class CompositeEnhancementLoss(nn.Module):
    def __init__(
        self,
        recipe: str,
        sample_rate: int = 16000,
        erb_bands: int = 32,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 320,
    ):
        super().__init__()
        self.recipe = recipe.upper()
        if self.recipe not in {"D1", "D2"}:
            raise ValueError(
                f"Unsupported loss recipe for the standalone project: {recipe}. "
                "This project is scoped only for the D1/D2 teacher-lite lineage."
            )
        self.erb_bands = erb_bands
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.wave_loss = nn.SmoothL1Loss(beta=0.5)
        self.teacher_mask_loss = nn.L1Loss()
        self.complex_loss = ComplexSTFTLoss()
        self.sisdr_loss = SISDRLoss()

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        epoch: int,
        total_epochs: int,
        teacher_wav: torch.Tensor | None = None,
        teacher_mask_erb: torch.Tensor | None = None,
    ) -> LossBreakdown:
        del epoch, total_epochs
        if teacher_wav is None or teacher_mask_erb is None:
            raise ValueError(f"Loss recipe {self.recipe} requires teacher_wav and teacher_mask_erb.")

        wave = self.wave_loss(enhanced, clean)
        student_mask = waveform_to_erb_mask(
            noisy,
            enhanced,
            erb_bands=self.erb_bands,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        teacher_mask = self.teacher_mask_loss(student_mask, teacher_mask_erb)
        teacher_wave = self.complex_loss(enhanced, teacher_wav)
        spectral = self.complex_loss(enhanced, clean)
        sisdr = enhanced.new_tensor(0.0)

        total = 0.60 * teacher_mask + 0.25 * teacher_wave + 0.15 * spectral
        if self.recipe == "D2":
            sisdr = self.sisdr_loss(enhanced, clean)
            total = total + 0.05 * sisdr

        zero = enhanced.new_tensor(0.0)
        return LossBreakdown(
            total=total,
            wave=wave,
            spectral=spectral,
            sisdr=sisdr,
            noise_gate=zero,
            speech_preserve=zero,
            teacher_mask=teacher_mask,
            teacher_wave=teacher_wave,
        )
