from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from sebench.audio import TARGET_SAMPLE_RATE, load_mono_audio, resample_mono_audio, stable_hash_text
from sebench.data import ManifestRow, read_pair_manifest
from sebench.stm32_models import (
    STM32_HOP_LENGTH,
    compute_spectral_gating_guidance,
    frontend_defaults_for_sample_rate,
    padded_frame_count,
    waveform_to_erb_mask,
)


@dataclass(frozen=True)
class TeacherCacheRow:
    noisy: Path
    clean: Path
    teacher_wav: Path
    teacher_mask_erb: Path
    guidance_sg: Path | None


def _row_key(row: ManifestRow) -> str:
    return stable_hash_text([row.noisy.as_posix(), row.clean.as_posix()])[:16]


def build_teacher_cache(
    manifest_path: str | Path,
    teacher_model: torch.nn.Module,
    *,
    out_dir: str | Path,
    device: str,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    teacher_sample_rate: int = TARGET_SAMPLE_RATE,
    erb_bands: int = 32,
    guidance_classic: str = "none",
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    rows = read_pair_manifest(manifest_path)
    n_fft, hop_length, win_length = frontend_defaults_for_sample_rate(target_sample_rate)
    out_root = Path(out_dir)
    wav_dir = out_root / "teacher_wav"
    mask_dir = out_root / "teacher_mask_erb"
    guidance_dir = out_root / "guidance_sg"
    wav_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    guidance_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_root / f"{Path(manifest_path).stem}_teacher_cache.csv"

    teacher_model.eval()
    payload_rows: list[dict[str, str]] = []
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            noisy_teacher, _ = load_mono_audio(row.noisy, teacher_sample_rate)
            clean_teacher, _ = load_mono_audio(row.clean, teacher_sample_rate)
            noisy_batch = noisy_teacher.unsqueeze(0).to(device)
            teacher_wav = teacher_model.denoise_single(noisy_batch).squeeze(0).cpu()
            if target_sample_rate != teacher_sample_rate:
                noisy = resample_mono_audio(noisy_teacher, teacher_sample_rate, target_sample_rate)
                clean = resample_mono_audio(clean_teacher, teacher_sample_rate, target_sample_rate)
                teacher_wav = resample_mono_audio(teacher_wav, teacher_sample_rate, target_sample_rate)
            else:
                noisy = noisy_teacher
                clean = clean_teacher
            teacher_mask = waveform_to_erb_mask(
                noisy.unsqueeze(0),
                teacher_wav.unsqueeze(0),
                erb_bands=erb_bands,
                sample_rate=target_sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
            ).squeeze(0).cpu()
            guidance_path: Path | None = None
            if guidance_classic == "spectral_gating":
                guidance = compute_spectral_gating_guidance(
                    noisy.unsqueeze(0),
                    erb_bands=erb_bands,
                    sample_rate=target_sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                ).squeeze(0).cpu()
                guidance_path = guidance_dir / f"{_row_key(row)}.pt"
                torch.save(guidance, guidance_path)

            teacher_wav_path = wav_dir / f"{_row_key(row)}.pt"
            teacher_mask_path = mask_dir / f"{_row_key(row)}.pt"
            torch.save(teacher_wav, teacher_wav_path)
            torch.save(teacher_mask, teacher_mask_path)
            payload_rows.append(
                {
                    "noisy": row.noisy.as_posix(),
                    "clean": row.clean.as_posix(),
                    "teacher_wav": teacher_wav_path.as_posix(),
                    "teacher_mask_erb": teacher_mask_path.as_posix(),
                    "guidance_sg": guidance_path.as_posix() if guidance_path is not None else "",
                }
            )
            if progress_callback is not None and (index == len(rows) or index == 1 or index % 100 == 0):
                progress_callback(f"teacher cache {index}/{len(rows)} from {Path(manifest_path).name}")

    with manifest_out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("noisy", "clean", "teacher_wav", "teacher_mask_erb", "guidance_sg"),
        )
        writer.writeheader()
        writer.writerows(payload_rows)
    return manifest_out.as_posix()


def read_teacher_cache_manifest(csv_path: str | Path) -> list[TeacherCacheRow]:
    path = Path(csv_path)
    rows: list[TeacherCacheRow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                TeacherCacheRow(
                    noisy=Path(row["noisy"]),
                    clean=Path(row["clean"]),
                    teacher_wav=Path(row["teacher_wav"]),
                    teacher_mask_erb=Path(row["teacher_mask_erb"]),
                    guidance_sg=Path(row["guidance_sg"]) if row.get("guidance_sg") else None,
                )
            )
    if not rows:
        raise ValueError(f"Teacher cache manifest is empty: {path}")
    return rows


class TeacherCacheDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        *,
        segment_len: int,
        sample_rate: int = TARGET_SAMPLE_RATE,
        n_fft: int = 512,
        hop_length: int = STM32_HOP_LENGTH,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.rows = read_teacher_cache_manifest(csv_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        noisy, _ = load_mono_audio(row.noisy, self.sample_rate)
        clean, _ = load_mono_audio(row.clean, self.sample_rate)
        teacher_wav = torch.load(row.teacher_wav, map_location="cpu").float()
        teacher_mask = torch.load(row.teacher_mask_erb, map_location="cpu").float()
        guidance = torch.load(row.guidance_sg, map_location="cpu").float() if row.guidance_sg is not None else None

        total = noisy.shape[-1]
        segment = self.segment_len
        if total >= segment:
            max_start = total - segment
            aligned_max = max_start // self.hop_length
            frame_start = torch.randint(0, aligned_max + 1, (1,)).item() if aligned_max > 0 else 0
            start = frame_start * self.hop_length
            noisy = noisy[start:start + segment]
            clean = clean[start:start + segment]
            teacher_wav = teacher_wav[start:start + segment]
        else:
            pad = segment - total
            start = 0
            frame_start = 0
            noisy = torch.nn.functional.pad(noisy, (0, pad))
            clean = torch.nn.functional.pad(clean, (0, pad))
            teacher_wav = torch.nn.functional.pad(teacher_wav, (0, pad))

        segment_frames = padded_frame_count(segment, n_fft=self.n_fft, hop_length=self.hop_length)
        teacher_mask = teacher_mask[:, frame_start:frame_start + segment_frames]
        if teacher_mask.shape[-1] < segment_frames:
            teacher_mask = torch.nn.functional.pad(teacher_mask, (0, segment_frames - teacher_mask.shape[-1]), mode="replicate")
        if guidance is not None:
            guidance = guidance[:, frame_start:frame_start + segment_frames]
            if guidance.shape[-1] < segment_frames:
                guidance = torch.nn.functional.pad(guidance, (0, segment_frames - guidance.shape[-1]), mode="replicate")

        sample = {
            "noisy": noisy,
            "clean": clean,
            "teacher_wav": teacher_wav,
            "teacher_mask_erb": teacher_mask,
        }
        if guidance is not None:
            sample["guidance_sg"] = guidance
        return sample
