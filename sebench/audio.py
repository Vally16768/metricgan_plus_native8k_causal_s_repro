from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio


TARGET_SAMPLE_RATE = 16000


def resample_mono_audio(wav: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    if source_sr == target_sr:
        return wav
    needs_unsqueeze = wav.ndim == 1
    wav_2d = wav.unsqueeze(0) if needs_unsqueeze else wav
    resampled = torchaudio.functional.resample(wav_2d, source_sr, target_sr)
    return resampled.squeeze(0) if needs_unsqueeze else resampled


def load_mono_audio(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0), target_sr


def save_mono_audio(path: str | Path, wav: torch.Tensor, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu().float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(out_path.as_posix(), wav, sample_rate)


def crop_or_pad(wav: torch.Tensor, length: int, start: int | None = None) -> torch.Tensor:
    if wav.shape[-1] >= length:
        if start is None:
            start = 0
        return wav[..., start:start + length]
    pad = length - wav.shape[-1]
    return torch.nn.functional.pad(wav, (0, pad))


def loop_to_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    if wav.shape[-1] == 0:
        raise ValueError("Cannot loop an empty waveform.")
    if wav.shape[-1] >= length:
        return wav[..., :length]
    repeats = (length + wav.shape[-1] - 1) // wav.shape[-1]
    tiled = wav.repeat(repeats)
    return tiled[:length]


def tensor_to_numpy_mono(wav: torch.Tensor) -> np.ndarray:
    return wav.detach().cpu().reshape(-1).numpy().astype(np.float32, copy=False)


def stable_hash_text(values: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def manifest_hash(csv_path: str | Path) -> str:
    path = Path(csv_path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        values = [f"{row['noisy']}|{row['clean']}" for row in reader]
    return stable_hash_text(values)
