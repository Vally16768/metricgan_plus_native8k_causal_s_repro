from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .audio import load_mono_audio


@dataclass(frozen=True)
class ManifestRow:
    noisy: Path
    clean: Path


def read_pair_manifest(csv_path: str | Path) -> list[ManifestRow]:
    path = Path(csv_path)
    rows: list[ManifestRow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "noisy" not in (reader.fieldnames or ()) or "clean" not in (reader.fieldnames or ()):
            raise ValueError(f"Manifest {path} must contain columns: noisy, clean")
        for row in reader:
            rows.append(ManifestRow(noisy=Path(row["noisy"]), clean=Path(row["clean"])))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


class VoiceBankDemandDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        segment_len: int = 16000 * 2,
        sample_rate: int = 16000,
        rows: Iterable[ManifestRow] | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.rows = list(rows) if rows is not None else read_pair_manifest(csv_path)
        if not self.rows:
            raise ValueError(f"No rows found in {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        noisy, _ = load_mono_audio(row.noisy, self.sample_rate)
        clean, _ = load_mono_audio(row.clean, self.sample_rate)

        total = noisy.shape[-1]
        segment = self.segment_len
        if total >= segment:
            start = torch.randint(0, total - segment + 1, (1,)).item()
            noisy = noisy[start:start + segment]
            clean = clean[start:start + segment]
        else:
            pad = segment - total
            noisy = torch.nn.functional.pad(noisy, (0, pad))
            clean = torch.nn.functional.pad(clean, (0, pad))

        return noisy, clean
