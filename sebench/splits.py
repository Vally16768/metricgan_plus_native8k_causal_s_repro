from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from .data import ManifestRow, read_pair_manifest


SPEAKER_RE = re.compile(r"(p\d+)_")


def _speaker_id(path: Path) -> str:
    match = SPEAKER_RE.match(path.stem)
    if not match:
        raise ValueError(f"Cannot infer speaker id from {path.name}")
    return match.group(1)


def _stable_order(rows: list[ManifestRow]) -> list[ManifestRow]:
    def key(row: ManifestRow) -> str:
        return hashlib.sha1(f"{row.noisy}|{row.clean}".encode("utf-8")).hexdigest()

    return sorted(rows, key=key)


def _write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["noisy", "clean"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"noisy": row.noisy.as_posix(), "clean": row.clean.as_posix()})


def build_voicebank_campaign_splits(
    train_csv: str | Path,
    output_dir: str | Path,
    val_speakers: tuple[str, ...] = ("p239", "p286", "p244", "p270"),
    rank_count: int = 128,
) -> dict[str, str]:
    rows = read_pair_manifest(train_csv)
    train_fit: list[ManifestRow] = []
    val_pool: list[ManifestRow] = []

    by_speaker: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        by_speaker[_speaker_id(row.clean)].append(row)

    missing = [speaker for speaker in val_speakers if speaker not in by_speaker]
    if missing:
        raise ValueError(f"Missing requested validation speakers: {missing}")

    for speaker, speaker_rows in by_speaker.items():
        if speaker in val_speakers:
            val_pool.extend(speaker_rows)
        else:
            train_fit.extend(speaker_rows)

    val_pool = _stable_order(val_pool)
    val_rank = val_pool[:rank_count]
    val_select = val_pool[rank_count:]

    out_root = Path(output_dir)
    manifest_paths = {
        "train_fit": out_root / "train_fit.csv",
        "val_pool": out_root / "val_pool.csv",
        "val_rank": out_root / "val_rank.csv",
        "val_select": out_root / "val_select.csv",
    }
    _write_manifest(manifest_paths["train_fit"], train_fit)
    _write_manifest(manifest_paths["val_pool"], val_pool)
    _write_manifest(manifest_paths["val_rank"], val_rank)
    _write_manifest(manifest_paths["val_select"], val_select)

    summary = {
        "train_csv": str(Path(train_csv).resolve()),
        "val_speakers": list(val_speakers),
        "counts": {
            "train_fit": len(train_fit),
            "val_pool": len(val_pool),
            "val_rank": len(val_rank),
            "val_select": len(val_select),
        },
        "manifests": {key: path.as_posix() for key, path in manifest_paths.items()},
    }
    with (out_root / "split_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return {key: path.as_posix() for key, path in manifest_paths.items()}
