#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from sebench import build_voicebank_campaign_splits
from sebench.checkpoints import load_model_from_checkpoint
from sebench.data import read_pair_manifest
from sebench.mlflow_utils import DEFAULT_EXPERIMENT_NAME
from sebench.models import dynamic_quantize_metricgan
from sebench.reporting import generate_report, read_json, write_csv, write_json
from sebench.runtime import require_cuda_device
from sebench.stm32sim import simulate_model_across_profiles
from sebench.teacher_cache import build_teacher_cache
from sebench.training import ExperimentConfig, benchmark_inference, evaluate_manifest, run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
SPEAKER_RE = re.compile(r"^(p\d+)_", re.IGNORECASE)


def _expand_tree(payload: Any, context: dict[str, str]) -> Any:
    if isinstance(payload, str):
        rendered = payload
        for key, value in context.items():
            rendered = rendered.replace("{" + key + "}", value)
        return rendered
    if isinstance(payload, list):
        return [_expand_tree(item, context) for item in payload]
    if isinstance(payload, dict):
        return {key: _expand_tree(value, context) for key, value in payload.items()}
    return payload


def load_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    context = {"project_root": PROJECT_ROOT.as_posix()}
    resolved = _expand_tree(raw, context)
    for _ in range(3):
        paths = resolved.get("paths", {})
        context.update({key: str(value) for key, value in paths.items() if isinstance(value, (str, Path))})
        resolved = _expand_tree(resolved, context)
    return resolved


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _copy_manifest(src: str | Path, dst: str | Path) -> str:
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)
    return dst_path.as_posix()


def materialize_test_manifest_8k(config: dict[str, Any], *, force: bool = False) -> str:
    dst = Path(config["dataset"]["test_csv_8k"])
    if dst.exists() and not force:
        return dst.as_posix()
    return _copy_manifest(config["dataset"]["test_csv_16k"], dst)


def _concat_csvs(output_csv: str | Path, input_csvs: list[str | Path], force: bool = False) -> str:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path.as_posix()

    headers = None
    rows: list[list[str]] = []
    seen: set[str] = set()
    for in_csv in input_csvs:
        with Path(in_csv).open(newline="") as f:
            reader = csv.reader(f)
            file_headers = next(reader, None)
            if file_headers is None:
                continue
            if headers is None:
                headers = file_headers
            elif headers != file_headers:
                raise ValueError(f"CSV header mismatch: {in_csv}")
            for row in reader:
                if not row:
                    continue
                if len(row) >= 2 and headers[:2] == ["noisy", "clean"]:
                    key = f"{_normalize_path(row[0])}|{_normalize_path(row[1])}"
                else:
                    key = "\x1f".join(row)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

    if headers is None:
        raise ValueError(f"No rows to concat for {output_csv}")

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    return output_path.as_posix()


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def _pair_key(noisy_path: str, clean_path: str) -> str:
    return f"{_normalize_path(noisy_path)}|{_normalize_path(clean_path)}"


def _clean_key(clean_path: str) -> str:
    value = _normalize_path(clean_path)
    for marker in ("/clean_train/", "/clean_val/", "/clean_test/", "/clean_sources/"):
        if marker in value:
            return value.split(marker, 1)[1].lstrip("/")
    return value


def _speaker_key(clean_path: str) -> str | None:
    match = SPEAKER_RE.match(Path(clean_path).name)
    if not match:
        return None
    return match.group(1).lower()


def _write_manifest_rows(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["noisy", "clean"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"noisy": row.noisy.as_posix(), "clean": row.clean.as_posix()})


def _manifest_keysets(manifest_path: str | Path) -> dict[str, Any]:
    rows = read_pair_manifest(manifest_path)
    pair_set: set[str] = set()
    clean_set: set[str] = set()
    for row in rows:
        pair_set.add(_pair_key(row.noisy.as_posix(), row.clean.as_posix()))
        clean_set.add(_clean_key(row.clean.as_posix()))
    return {
        "rows": len(rows),
        "pair_set": pair_set,
        "clean_set": clean_set,
        "duplicate_pairs": len(rows) - len(pair_set),
        "duplicate_clean_keys": len(rows) - len(clean_set),
    }


def _audit_manifest_bundle(
    manifests: dict[str, str | Path],
    *,
    strict: bool,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    for label, path in manifests.items():
        loaded[label] = _manifest_keysets(path)

    per_manifest = {
        label: {
            "rows": payload["rows"],
            "duplicate_pairs": payload["duplicate_pairs"],
            "duplicate_clean_keys": payload["duplicate_clean_keys"],
            "unique_pairs": len(payload["pair_set"]),
            "unique_clean_keys": len(payload["clean_set"]),
        }
        for label, payload in loaded.items()
    }

    categories: dict[str, dict[str, set[str]]] = {
        "train": {"pair": set(), "clean": set()},
        "val": {"pair": set(), "clean": set()},
        "test": {"pair": set(), "clean": set()},
    }

    for label, payload in loaded.items():
        lower = label.lower()
        if lower.startswith("train"):
            bucket = "train"
        elif lower.startswith("val"):
            bucket = "val"
        elif lower.startswith("test"):
            bucket = "test"
        else:
            continue
        categories[bucket]["pair"].update(payload["pair_set"])
        categories[bucket]["clean"].update(payload["clean_set"])

    boundaries = {
        "train_vs_val": {
            "pair_overlap": len(categories["train"]["pair"] & categories["val"]["pair"]),
            "clean_overlap": len(categories["train"]["clean"] & categories["val"]["clean"]),
        },
        "train_vs_test": {
            "pair_overlap": len(categories["train"]["pair"] & categories["test"]["pair"]),
            "clean_overlap": len(categories["train"]["clean"] & categories["test"]["clean"]),
        },
        "val_vs_test": {
            "pair_overlap": len(categories["val"]["pair"] & categories["test"]["pair"]),
            "clean_overlap": len(categories["val"]["clean"] & categories["test"]["clean"]),
        },
    }

    summary = {
        "manifests": {label: Path(path).as_posix() for label, path in manifests.items()},
        "per_manifest": per_manifest,
        "boundaries": boundaries,
    }

    if out_path:
        write_json(Path(out_path), summary)

    if strict:
        dup_issues = [
            label
            for label, payload in per_manifest.items()
            if payload["duplicate_pairs"] > 0 or payload["duplicate_clean_keys"] > 0
        ]
        if dup_issues:
            raise ValueError(f"Duplicate rows/clean keys detected in manifests: {dup_issues}")
        for boundary, payload in boundaries.items():
            if payload["pair_overlap"] > 0 or payload["clean_overlap"] > 0:
                raise ValueError(f"Data leakage detected on {boundary}: {payload}")

    return summary


def _prepare_voicebank_dataset(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    manifests = build_voicebank_campaign_splits(
        train_csv=config["dataset"]["train_csv_16k"],
        output_dir=config["dataset"]["campaign_dir_8k"],
        val_speakers=tuple(config["dataset"]["val_speakers"]),
        rank_count=int(config["dataset"]["rank_count"]),
    )
    manifests["test_8k"] = materialize_test_manifest_8k(config, force=force)
    return {
        "campaign_manifests": manifests,
        "val_speakers": list(config["dataset"]["val_speakers"]),
        "rank_count": int(config["dataset"]["rank_count"]),
    }


def _prepare_dns5_dataset(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    train_csv = config["dataset"].get("train_csv_16k")
    test_csv = config["dataset"].get("test_csv_16k")
    val_rank_csv = config["dataset"].get("val_rank_csv")
    val_select_csv = config["dataset"].get("val_select_csv")
    if not train_csv or not test_csv:
        raise ValueError("DNS5 dataset requires train_csv_16k and test_csv_16k")
    return {
        "campaign_manifests": {
            "train_fit": train_csv,
            "val_rank": val_rank_csv or "",
            "val_select": val_select_csv or "",
            "test": test_csv,
        },
        "val_speakers": config["dataset"].get("val_speakers", []),
        "rank_count": int(config["dataset"].get("rank_count", 0)),
    }


def _prepare_combined_dataset(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    combined_dir = Path(config["paths"]["output_root"]) / "combined_manifest"
    combined_dir.mkdir(parents=True, exist_ok=True)
    vbd_train = config["dataset"]["vbd_train_csv_16k"]
    dns_train = config["dataset"]["dns5_train_csv_16k"]
    combined_train = combined_dir / "train_combined.csv"
    _concat_csvs(combined_train, [vbd_train, dns_train], force=force)

    vbd_test = config["dataset"]["vbd_test_csv_16k"]
    dns_test = config["dataset"]["dns5_test_csv_16k"]
    combined_test = combined_dir / "test_combined.csv"
    _concat_csvs(combined_test, [vbd_test, dns_test], force=force)

    return {
        "campaign_manifests": {
            "train_fit": combined_train.as_posix(),
            "test": combined_test.as_posix(),
        },
        "val_speakers": config["dataset"].get("val_speakers", []),
        "rank_count": int(config["dataset"].get("rank_count", 0)),
    }


def _split_manifest_rank_select(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    rank_count: int,
    force: bool = False,
    prefix: str = "split",
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_csv = output_dir / f"{prefix}_rank.csv"
    select_csv = output_dir / f"{prefix}_select.csv"
    if rank_csv.exists() and select_csv.exists() and not force:
        return {"rank": rank_csv.as_posix(), "select": select_csv.as_posix()}

    rows = read_pair_manifest(manifest_path)
    dedup: dict[str, Any] = {}
    for row in rows:
        key = _pair_key(row.noisy.as_posix(), row.clean.as_posix())
        if key not in dedup:
            dedup[key] = row
    ordered = sorted(dedup.values(), key=lambda row: _pair_key(row.noisy.as_posix(), row.clean.as_posix()))

    rank_size = max(0, min(int(rank_count), len(ordered)))
    rank_rows = ordered[:rank_size]
    select_rows = ordered[rank_size:]

    _write_manifest_rows(rank_csv, rank_rows)
    _write_manifest_rows(select_csv, select_rows)
    return {"rank": rank_csv.as_posix(), "select": select_csv.as_posix()}


def _split_manifest_train_test(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    test_fraction: float,
    seed: int = 42,
    force: bool = False,
    prefix: str = "split",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = output_dir / f"{prefix}_train.csv"
    test_csv = output_dir / f"{prefix}_test.csv"
    if train_csv.exists() and test_csv.exists() and not force:
        return {"train": train_csv.as_posix(), "test": test_csv.as_posix(), "reused": True}

    rows = read_pair_manifest(manifest_path)
    dedup: dict[str, Any] = {}
    for row in rows:
        key = _pair_key(row.noisy.as_posix(), row.clean.as_posix())
        if key not in dedup:
            dedup[key] = row
    unique_rows = list(dedup.values())

    target_fraction = max(0.0, min(0.5, float(test_fraction)))
    if not unique_rows:
        _write_manifest_rows(train_csv, [])
        _write_manifest_rows(test_csv, [])
        return {"train": train_csv.as_posix(), "test": test_csv.as_posix(), "reused": False, "rows": 0}

    groups: dict[str, list[Any]] = defaultdict(list)
    for row in unique_rows:
        groups[_clean_key(row.clean.as_posix())].append(row)
    group_keys = sorted(groups.keys())
    rng = random.Random(int(seed))
    rng.shuffle(group_keys)

    total = len(unique_rows)
    target_test = int(round(total * target_fraction))
    target_test = max(1, min(target_test, total - 1)) if total > 1 and target_fraction > 0 else 0

    train_rows: list[Any] = []
    test_rows: list[Any] = []
    for key in group_keys:
        bucket = test_rows if len(test_rows) < target_test else train_rows
        bucket.extend(groups[key])

    # Safety: never leave train empty when we have enough rows.
    if not train_rows and len(test_rows) > 1:
        train_rows.append(test_rows.pop())

    _write_manifest_rows(train_csv, train_rows)
    _write_manifest_rows(test_csv, test_rows)
    return {
        "train": train_csv.as_posix(),
        "test": test_csv.as_posix(),
        "reused": False,
        "rows_total": len(rows),
        "rows_unique": len(unique_rows),
        "rows_train": len(train_rows),
        "rows_test": len(test_rows),
        "duplicates_removed": len(rows) - len(unique_rows),
        "target_test_fraction": target_fraction,
    }


def _prepare_academic_combined_dataset(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    voicebank_root = Path(dataset_cfg["voicebank_root"])
    dns5_root = Path(dataset_cfg["dns5_root"])

    # VoiceBank+DEMAND uses official train/test; validation is built from official train.
    vbd_train = (voicebank_root / "16k" / "train.csv").as_posix()
    vbd_test = (voicebank_root / "16k" / "test.csv").as_posix()
    campaign_dir = dataset_cfg.get("voicebank_campaign_dir") or (voicebank_root / "16k" / "campaign").as_posix()
    val_speakers = tuple(dataset_cfg.get("val_speakers") or ("p239", "p286", "p244", "p270"))
    rank_count = int(dataset_cfg.get("rank_count", 128))
    vbd_campaign = build_voicebank_campaign_splits(
        train_csv=vbd_train,
        output_dir=campaign_dir,
        val_speakers=val_speakers,
        rank_count=rank_count,
    )

    # DNS5 has train+val locally. Split val into disjoint rank/select for stable training signals.
    dns5_train = dataset_cfg.get("dns5_train_csv") or (dns5_root / "train.csv").as_posix()
    dns5_val = dataset_cfg.get("dns5_val_csv") or (dns5_root / "val.csv").as_posix()
    dns5_rank_count = int(dataset_cfg.get("dns5_rank_count", 4096))
    dns5_test_from_train_fraction = float(dataset_cfg.get("dns5_test_from_train_fraction", 0.10))
    split_seed = int(dataset_cfg.get("split_seed", 42))
    combined_dir = Path(config["paths"]["output_root"]) / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    dns5_split = _split_manifest_rank_select(
        dns5_val,
        combined_dir / "dns5_val_split",
        rank_count=dns5_rank_count,
        force=force,
        prefix="dns5_val",
    )

    dns5_test_candidate = str(dataset_cfg.get("dns5_test_csv") or "").strip()
    dns5_train_for_fit = dns5_train
    dns5_test_for_combined: str | None = None
    dns5_test_source = "none"
    dns5_train_test_split: dict[str, Any] | None = None

    if dns5_test_candidate and Path(dns5_test_candidate).exists():
        dns5_test_for_combined = dns5_test_candidate
        dns5_test_source = "external_dns5_test_csv"
    elif dns5_test_from_train_fraction > 0.0:
        dns5_train_test_split = _split_manifest_train_test(
            dns5_train,
            combined_dir / "dns5_train_test_split",
            test_fraction=dns5_test_from_train_fraction,
            seed=split_seed,
            force=force,
            prefix="dns5",
        )
        dns5_train_for_fit = dns5_train_test_split["train"]
        dns5_test_for_combined = dns5_train_test_split["test"]
        dns5_test_source = "derived_from_dns5_train"

    # Build combined manifests.
    combined_train = _concat_csvs(
        dataset_cfg["combined_train_csv"],
        [vbd_campaign["train_fit"], dns5_train_for_fit],
        force=force,
    )
    combined_val_rank = _concat_csvs(
        dataset_cfg["combined_val_rank_csv"],
        [vbd_campaign["val_rank"], dns5_split["rank"]],
        force=force,
    )
    combined_val_select = _concat_csvs(
        dataset_cfg["combined_val_select_csv"],
        [vbd_campaign["val_select"], dns5_split["select"]],
        force=force,
    )

    test_inputs = [vbd_test]
    dns5_test_included = False
    if dns5_test_for_combined and Path(dns5_test_for_combined).exists():
        test_inputs.append(dns5_test_for_combined)
        dns5_test_included = True
    combined_test = _concat_csvs(
        dataset_cfg["combined_test_csv"],
        test_inputs,
        force=force,
    )

    combined_manifests = {
        "train_fit": combined_train,
        "val_rank": combined_val_rank,
        "val_select": combined_val_select,
        "test": combined_test,
    }
    integrity = _audit_manifest_bundle(
        combined_manifests,
        strict=True,
        out_path=combined_dir / "combined_integrity_summary.json",
    )

    return {
        "voicebank": {
            "train_csv": vbd_train,
            "test_csv": vbd_test,
            "campaign_manifests": vbd_campaign,
            "val_speakers": list(val_speakers),
            "rank_count": rank_count,
        },
        "dns5": {
            "train_csv": dns5_train,
            "train_fit_csv": dns5_train_for_fit,
            "val_csv": dns5_val,
            "val_split": dns5_split,
            "val_rank_count": dns5_rank_count,
            "test_csv": dns5_test_for_combined or "",
            "test_source": dns5_test_source,
            "test_included_in_combined": dns5_test_included,
            "train_test_split": dns5_train_test_split or {},
        },
        "combined_manifests": combined_manifests,
        "integrity": integrity,
        "notes": [
            "VoiceBank test uses official test.csv.",
            "VoiceBank validation is speaker-holdout from official train.csv.",
            "DNS5 val is split into disjoint val_rank/val_select.",
            "DNS5 test comes from explicit dns5_test_csv or (fallback) deterministic split from DNS5 train.",
        ],
    }


def _create_reference_splits(manifest_path: str | Path, output_dir: str | Path, *, force: bool = False) -> dict[str, str]:
    """Create deterministic, leakage-safe 80/10/10 splits for any dataset."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = output_dir / "train_80.csv"
    val_csv = output_dir / "val_10.csv"
    test_csv = output_dir / "test_10.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists() and not force:
        return {
            "train": train_csv.as_posix(),
            "val": val_csv.as_posix(),
            "test": test_csv.as_posix(),
        }

    # Read + dedupe exact pairs.
    rows = read_pair_manifest(manifest_path)
    seen_pairs: set[str] = set()
    unique_rows = []
    for row in rows:
        key = _pair_key(row.noisy.as_posix(), row.clean.as_posix())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_rows.append(row)
    total_samples = len(unique_rows)

    # Group rows so the same clean content never crosses split boundaries.
    speaker_keys = [_speaker_key(row.clean.as_posix()) for row in unique_rows]
    speaker_hits = [value for value in speaker_keys if value]
    speaker_ratio = (len(speaker_hits) / total_samples) if total_samples else 0.0
    speaker_count = len(set(speaker_hits))
    use_speaker_groups = speaker_ratio >= 0.98 and speaker_count >= 6

    groups: dict[str, list[Any]] = defaultdict(list)
    for row in unique_rows:
        if use_speaker_groups:
            group_key = _speaker_key(row.clean.as_posix()) or _clean_key(row.clean.as_posix())
        else:
            group_key = _clean_key(row.clean.as_posix())
        groups[group_key].append(row)

    # Deterministic assignment by groups.
    group_keys = sorted(groups.keys())
    rng = random.Random(42)
    rng.shuffle(group_keys)

    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size
    targets = {"train": train_size, "val": val_size, "test": test_size}
    assigned = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for group_key in group_keys:
        group_rows = groups[group_key]
        deficits = {name: targets[name] - counts[name] for name in ("train", "val", "test")}
        if max(deficits.values()) <= 0:
            split_name = min(counts, key=counts.get)
        else:
            split_name = max(deficits, key=deficits.get)
        assigned[split_name].extend(group_rows)
        counts[split_name] += len(group_rows)

    train_rows = assigned["train"]
    val_rows = assigned["val"]
    test_rows = assigned["test"]

    _write_manifest_rows(train_csv, train_rows)
    _write_manifest_rows(val_csv, val_rows)
    _write_manifest_rows(test_csv, test_rows)

    integrity = _audit_manifest_bundle(
        {"train": train_csv, "val": val_csv, "test": test_csv},
        strict=True,
        out_path=output_dir / "split_integrity_summary.json",
    )

    summary = {
        "original_manifest": str(Path(manifest_path).resolve()),
        "total_samples": total_samples,
        "deduplicated_rows_removed": len(rows) - len(unique_rows),
        "grouping_strategy": "speaker" if use_speaker_groups else "clean_key",
        "splits": {
            "train_80": len(train_rows),
            "val_10": len(val_rows),
            "test_10": len(test_rows),
        },
        "manifests": {
            "train": train_csv.as_posix(),
            "val": val_csv.as_posix(),
            "test": test_csv.as_posix(),
        },
        "integrity": integrity,
    }

    write_json(output_dir / "reference_split_summary.json", summary)
    return {
        "train": train_csv.as_posix(),
        "val": val_csv.as_posix(),
        "test": test_csv.as_posix(),
    }


def _prepare_reference_dataset(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Prepare reference 80/10/10 splits for VoiceBank and DNS5, and combined datasets."""

    # Create reference splits for VoiceBank
    voicebank_splits = _create_reference_splits(
        config["dataset"]["voicebank_root"] + "/16k/train.csv",
        config["dataset"]["voicebank_root"] + "/reference_splits",
        force=force
    )

    # Create reference splits for DNS5
    dns5_splits = _create_reference_splits(
        config["dataset"]["dns5_root"] + "/train.csv",
        config["dataset"]["dns5_root"] + "/reference_splits",
        force=force
    )

    # Create combined datasets
    combined_dir = Path(config["paths"]["output_root"]) / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Combined training set
    _concat_csvs(
        config["dataset"]["combined_train_csv"],
        [voicebank_splits["train"], dns5_splits["train"]],
        force=force
    )

    # Combined validation set
    _concat_csvs(
        config["dataset"]["combined_val_csv"],
        [voicebank_splits["val"], dns5_splits["val"]],
        force=force
    )

    # Combined test set
    _concat_csvs(
        config["dataset"]["combined_test_csv"],
        [voicebank_splits["test"], dns5_splits["test"]],
        force=force
    )

    combined_manifests = {
        "train_fit": config["dataset"]["combined_train_csv"],
        "val_rank": config["dataset"]["combined_val_csv"],
        "val_select": config["dataset"]["combined_val_csv"],  # Use same as val_rank for simplicity
        "test": config["dataset"]["combined_test_csv"],
    }
    integrity = _audit_manifest_bundle(
        combined_manifests,
        strict=True,
        out_path=combined_dir / "combined_integrity_summary.json",
    )

    return {
        "voicebank_splits": voicebank_splits,
        "dns5_splits": dns5_splits,
        "combined_manifests": combined_manifests,
        "integrity": integrity,
        "val_speakers": [],
        "rank_count": 0,
    }


def command_prepare_data(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    dataset_type = config["dataset"].get("dataset_type", "voicebank")
    if dataset_type == "voicebank":
        summary = _prepare_voicebank_dataset(config, force=force)
    elif dataset_type == "dns5":
        summary = _prepare_dns5_dataset(config, force=force)
    elif dataset_type == "combined":
        summary = _prepare_combined_dataset(config, force=force)
    elif dataset_type == "academic_combined":
        summary = _prepare_academic_combined_dataset(config, force=force)
    elif dataset_type == "reference_combined":
        summary = _prepare_reference_dataset(config, force=force)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    write_json(Path(config["paths"]["output_root"]) / "prepare_data" / "summary.json", summary)
    return summary


class _QuantizedTeacherWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def denoise_single(self, noisy: torch.Tensor) -> torch.Tensor:
        return self.model.denoise_single(noisy)


def command_build_teacher_cache(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    manifest_path = Path(config["teacher_cache"]["manifest"])
    if manifest_path.exists() and not force:
        return {
            "manifest": manifest_path.as_posix(),
            "out_dir": config["teacher_cache"]["out_dir"],
            "reused": True,
        }

    # Check if we have a trained teacher
    teacher_checkpoint = None
    teacher_training_results_path = Path(config["paths"]["output_root"]) / "teacher_training_results.json"
    if teacher_training_results_path.exists():
        teacher_results = read_json(teacher_training_results_path)
        winner = _select_teacher_winner(teacher_results["runs"])
        teacher_checkpoint = winner["checkpoint_out"]
    else:
        # Fall back to pre-trained teacher
        teacher_checkpoint = config["paths"]["teacher_source_checkpoint"]

    teacher_model, _ = load_model_from_checkpoint(
        teacher_checkpoint,
        device="cpu",
        model_family="metricgan_plus_native8k",
        variant="small",
    )
    quantized_teacher = _QuantizedTeacherWrapper(dynamic_quantize_metricgan(teacher_model))
    manifest = build_teacher_cache(
        config["dataset"]["train_fit_csv"],
        quantized_teacher,
        out_dir=config["teacher_cache"]["out_dir"],
        device="cpu",
        target_sample_rate=int(config["training"]["sample_rate"]),
        teacher_sample_rate=int(config["training"]["sample_rate"]),
        erb_bands=int(config["training"]["erb_bands"]),
        guidance_classic=str(config["training"]["guidance_classic"]),
    )
    summary = {
        "manifest": manifest,
        "out_dir": config["teacher_cache"]["out_dir"],
        "teacher_checkpoint": teacher_checkpoint,
        "quantized_teacher": True,
    }
    write_json(Path(config["paths"]["output_root"]) / "teacher_cache" / "summary.json", summary)
    return summary


def _teacher_experiment_configs(config: dict[str, Any], *, device: str) -> list[ExperimentConfig]:
    shared = _base_experiment_config(config, device=device)
    # For teacher training, use combined dataset
    shared["train_csv"] = config["dataset"]["combined_train_csv"]
    shared["val_rank_csv"] = config["dataset"].get("combined_val_rank_csv") or config["dataset"].get("combined_val_csv")
    shared["val_select_csv"] = config["dataset"].get("combined_val_select_csv") or config["dataset"].get("combined_val_csv")
    shared["test_csv"] = config["dataset"]["combined_test_csv"]

    configs: list[ExperimentConfig] = []
    for family in config["teacher_training"]["families"]:
        for variant in config["teacher_training"]["variants"]:
            for seed in config["teacher_training"]["seeds"]:
                run_name = f"{family}-{variant}-teacher-training-seed{seed}"
                checkpoint_out = Path(config["paths"]["output_root"]) / "checkpoints" / "teacher" / f"{run_name}.pt"
                payload = {
                    **shared,
                    "checkpoint_out": checkpoint_out.as_posix(),
                    "model_family": str(family),
                    "variant": str(variant),
                    "loss_recipe": str(config["teacher_training"]["loss_recipe"]),
                    "run_name": run_name,
                    "phase": "teacher_training",
                    "epochs": int(config["teacher_training"]["epochs"]),
                    "lr": float(config["teacher_training"]["lr"]),
                    "early_stop_patience": int(config["teacher_training"]["early_stop_patience"]),
                    "min_epochs": int(config["teacher_training"]["min_epochs"]),
                    "seed": int(seed),
                    "qat": False,
                    "init_checkpoint": None,
                    "teacher_cache_manifest": None,  # No teacher cache for teacher training
                }
                configs.append(ExperimentConfig(**payload))
    return configs


def command_train_teacher(config: dict[str, Any], *, device: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for experiment in _teacher_experiment_configs(config, device=device):
        results.append(run_experiment(experiment))
    summary = {
        "runs": results,
        "winner": _select_teacher_winner(results),
    }
    write_json(Path(config["paths"]["output_root"]) / "teacher_training_results.json", summary)
    return summary


def _select_teacher_winner(teacher_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not teacher_results:
        raise ValueError("No teacher training results available.")
    return max(teacher_results, key=lambda item: float(item.get("best_val_select_pesq") or float("-inf")))


def _base_experiment_config(config: dict[str, Any], *, device: str) -> dict[str, Any]:
    return {
        "train_csv": config["dataset"].get("train_fit_csv") or config["dataset"].get("train_csv_16k"),
        "val_rank_csv": config["dataset"].get("val_rank_csv"),
        "val_select_csv": config["dataset"].get("val_select_csv"),
        "test_csv": config["dataset"].get("test_csv") or config["dataset"].get("test_csv_16k"),
        "variant": "small",
        "segment_len": int(config["training"]["segment_len"]),
        "device": device,
        "scheduler": str(config["training"]["scheduler"]),
        "lr_factor": float(config["training"]["lr_factor"]),
        "lr_patience": int(config["training"]["lr_patience"]),
        "min_lr": float(config["training"]["min_lr"]),
        "eval_every": int(config["training"]["eval_every"]),
        "grad_clip": float(config["training"]["grad_clip"]),
        "amp": bool(config["training"]["amp"]),
        "selection_metric": "val_select_pesq_mean",
        "mlflow_uri": config["paths"]["tracking_root"],
        "mlflow_artifact_root": str(Path(config["paths"]["tracking_root"]) / "artifacts"),
        "experiment_name": str(config["tracking"]["experiment_name"]),
        "log_system_metrics": False,
        "log_torch_model": False,
        "sample_count": int(config["training"]["sample_count"]),
        "benchmark_seconds": int(config["training"]["benchmark_seconds"]),
        "benchmark_repeats": int(config["training"]["benchmark_repeats"]),
        "max_eval_files": None,
        "eval_batch_size": None,
        "cache_eval_audio": True,
        "rank_compute_composite": True,
        "select_compute_composite": True,
        "postfilter_mode": "none",
        "postfilter_preset": "medium",
        "train_postfilter": False,
        "spectral_native_gate": False,
        "teacher_source_run_id": None,
        "teacher_variant": None,
        "audit_only": False,
        "teacher_cache_manifest": config["teacher_cache"]["manifest"],
        "guidance_classic": str(config["training"]["guidance_classic"]),
        "erb_bands": int(config["training"]["erb_bands"]),
        "context_frames": int(config["training"]["context_frames"]),
        "mcu_profile": str(config["mcu"]["profile"]),
        "sample_rate": int(config["training"]["sample_rate"]),
        "n_fft": int(config["training"]["n_fft"]),
        "hop_length": int(config["training"]["hop_length"]),
        "win_length": int(config["training"]["win_length"]),
        "eval_dnsmos": False,
    }


def _stage1_run_name(family: str, config: dict[str, Any], seed: int) -> str:
    return (
        f"{family}-small-lr{config['stage1']['lr']}"
        f"-seg{config['training']['segment_len']}"
        f"-loss{config['stage1']['loss_recipe']}"
        f"-seed{seed}"
    )


def _stage1_experiment_configs(config: dict[str, Any], *, device: str) -> list[ExperimentConfig]:
    shared = _base_experiment_config(config, device=device)
    configs: list[ExperimentConfig] = []
    for family in config["stage1"]["families"]:
        for seed in config["stage1"]["seeds"]:
            run_name = _stage1_run_name(str(family), config, int(seed))
            checkpoint_out = Path(config["paths"]["output_root"]) / "checkpoints" / "stage1" / f"{run_name}.pt"
            payload = {
                **shared,
                "checkpoint_out": checkpoint_out.as_posix(),
                "model_family": str(family),
                "loss_recipe": str(config["stage1"]["loss_recipe"]),
                "run_name": run_name,
                "phase": "teacher_lite_stage1",
                "epochs": int(config["stage1"]["epochs"]),
                "lr": float(config["stage1"]["lr"]),
                "early_stop_patience": int(config["stage1"]["early_stop_patience"]),
                "min_epochs": int(config["stage1"]["min_epochs"]),
                "seed": int(seed),
                "qat": False,
                "init_checkpoint": None,
            }
            configs.append(ExperimentConfig(**payload))
    return configs


def command_train_stage1(config: dict[str, Any], *, device: str) -> dict[str, Any]:
    if not Path(config["teacher_cache"]["manifest"]).exists():
        command_build_teacher_cache(config)
    results: list[dict[str, Any]] = []
    for experiment in _stage1_experiment_configs(config, device=device):
        results.append(run_experiment(experiment))
    summary = {
        "runs": results,
        "winner": _select_stage1_winner(results),
    }
    write_json(Path(config["paths"]["output_root"]) / "stage1_results.json", summary)
    return summary


def _select_stage1_winner(stage1_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not stage1_results:
        raise ValueError("No stage1 results available.")
    return max(stage1_results, key=lambda item: float(item.get("best_val_select_pesq") or float("-inf")))


def command_train_qat(config: dict[str, Any], *, device: str) -> dict[str, Any]:
    stage1_summary_path = Path(config["paths"]["output_root"]) / "stage1_results.json"
    if not stage1_summary_path.exists():
        command_train_stage1(config, device=device)
    stage1_summary = read_json(stage1_summary_path)
    winner = _select_stage1_winner(stage1_summary["runs"])
    checkpoint_out = Path(config["paths"]["output_root"]) / "checkpoints" / "final" / "metricgan_plus_native8k_causal_s_qat.pt"
    payload = {
        **_base_experiment_config(config, device=device),
        "checkpoint_out": checkpoint_out.as_posix(),
        "model_family": winner["model_family"],
        "loss_recipe": str(config["qat"]["loss_recipe"]),
        "run_name": str(config["reference"]["final_qat_run_name"]),
        "phase": "teacher_lite_qat",
        "epochs": int(config["qat"]["epochs"]),
        "lr": float(config["qat"]["lr"]),
        "early_stop_patience": int(config["qat"]["early_stop_patience"]),
        "min_epochs": int(config["qat"]["min_epochs"]),
        "seed": int(winner["seed"]),
        "qat": True,
        "init_checkpoint": winner["checkpoint_out"],
    }
    result = run_experiment(ExperimentConfig(**payload))
    write_json(Path(config["paths"]["output_root"]) / "qat_result.json", result)
    return result


def _resolve_checkpoint_for_evaluation(config: dict[str, Any], checkpoint: str | None) -> str:
    if checkpoint:
        return checkpoint
    local_final = Path(config["paths"]["output_root"]) / "checkpoints" / "final" / "metricgan_plus_native8k_causal_s_qat.pt"
    if local_final.exists():
        return local_final.as_posix()
    return config["paths"]["reference_final_checkpoint"]


def _evaluation_output_dir(config: dict[str, Any], label: str | None, checkpoint: str) -> Path:
    if label:
        name = label
    else:
        name = Path(checkpoint).stem
    out_dir = Path(config["paths"]["output_root"]) / "evaluations" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_canonical_eval_csv(out_dir: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for split_name in ["val_rank_metrics", "val_select_metrics", "test_metrics"]:
        for key, value in summary.get(split_name, {}).items():
            if key == "sample_paths":
                continue
            rows.append({"scope": split_name, "metric": key, "value": value})
    rows.append({"scope": "benchmark", "metric": "benchmark_latency_10s", "value": summary.get("benchmark_latency_10s")})
    rollup = summary.get("mcu_rollup", {})
    rows.extend(
        [
            {"scope": "mcu_rollup", "metric": "best_profile_name", "value": rollup.get("best_profile_name")},
            {"scope": "mcu_rollup", "metric": "best_power_profile_name", "value": rollup.get("best_power_profile_name")},
            {"scope": "mcu_rollup", "metric": "best_power_profile_avg_power_mw", "value": rollup.get("best_power_profile_avg_power_mw")},
        ]
    )
    write_csv(out_dir / "canonical_metrics.csv", rows, ["scope", "metric", "value"])


def command_evaluate(
    config: dict[str, Any],
    *,
    device: str,
    checkpoint: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = _resolve_checkpoint_for_evaluation(config, checkpoint)
    out_dir = _evaluation_output_dir(config, label, checkpoint_path)
    model, package = load_model_from_checkpoint(checkpoint_path, device=device)
    val_rank_manifest = config["dataset"]["val_rank_csv"]
    val_select_manifest = config["dataset"]["val_select_csv"]
    test_manifest = materialize_test_manifest_8k(config)

    val_rank_metrics = evaluate_manifest(
        model,
        val_rank_manifest,
        device,
        sample_rate=int(config["training"]["sample_rate"]),
        compute_dnsmos=False,
        compute_composite=bool(config["evaluation"]["compute_composite"]),
        sample_dir=out_dir / "samples" / "val_rank",
        sample_count=int(config["training"]["sample_count"]),
        batch_size=int(config["evaluation"]["eval_batch_size"]),
        cache_audio=True,
    )
    val_select_metrics = evaluate_manifest(
        model,
        val_select_manifest,
        device,
        sample_rate=int(config["training"]["sample_rate"]),
        compute_dnsmos=False,
        compute_composite=bool(config["evaluation"]["compute_composite"]),
        sample_dir=out_dir / "samples" / "val_select",
        sample_count=int(config["training"]["sample_count"]),
        batch_size=int(config["evaluation"]["eval_batch_size"]),
        cache_audio=True,
    )
    test_metrics = evaluate_manifest(
        model,
        test_manifest,
        device,
        sample_rate=int(config["training"]["sample_rate"]),
        compute_dnsmos=False,
        compute_composite=bool(config["evaluation"]["compute_composite"]),
        sample_dir=out_dir / "samples" / "test",
        sample_count=int(config["training"]["sample_count"]),
        batch_size=int(config["evaluation"]["eval_batch_size"]),
        cache_audio=True,
    )
    sample_path = read_pair_manifest(val_rank_manifest)[0].noisy
    benchmark_seconds = benchmark_inference(
        model,
        sample_path=sample_path,
        device=device,
        sample_rate=int(config["training"]["sample_rate"]),
        duration_seconds=int(config["training"]["benchmark_seconds"]),
        repeats=int(config["training"]["benchmark_repeats"]),
    )
    mcu_rollup = simulate_model_across_profiles(model)

    write_json(out_dir / "val_rank_metrics.json", val_rank_metrics)
    write_json(out_dir / "val_select_metrics.json", val_select_metrics)
    write_json(out_dir / "test_metrics.json", test_metrics)
    write_json(out_dir / "mcu_rollup.json", mcu_rollup)

    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_package": {
            "model_family": package.get("model_family"),
            "variant": package.get("variant"),
            "model_config": package.get("model_config", {}),
        },
        "val_rank_metrics": val_rank_metrics,
        "val_select_metrics": val_select_metrics,
        "test_metrics": test_metrics,
        "benchmark_latency_10s": benchmark_seconds,
        "mcu_rollup": mcu_rollup,
    }
    write_json(out_dir / "summary.json", summary)
    _write_canonical_eval_csv(out_dir, summary)
    return summary


def command_report(
    config: dict[str, Any],
    *,
    evaluation_dir: str | None = None,
    report_dir: str | None = None,
) -> dict[str, Any]:
    reference_export_path = Path(config["paths"]["reference_export_json"])
    if not reference_export_path.exists():
        raise FileNotFoundError(
            f"Reference export missing: {reference_export_path}. "
            "Ruleaza mai intai scripts/export_reference_runs.py."
        )
    reference_export = read_json(reference_export_path)
    if evaluation_dir is None:
        default_eval = Path(config["paths"]["output_root"]) / "evaluations" / Path(config["paths"]["reference_final_checkpoint"]).stem
        if default_eval.exists():
            evaluation_dir = default_eval.as_posix()
        else:
            evaluation_dir = (Path(config["paths"]["output_root"]) / "evaluations" / "metricgan_plus_native8k_causal_s_qat").as_posix()
    target_report_dir = Path(report_dir) if report_dir else Path(config["paths"]["output_root"]) / "reports" / "reference_qat"
    return generate_report(
        report_dir=target_report_dir,
        config=config,
        reference_export=reference_export,
        evaluation_dir=evaluation_dir,
    )


def command_run_all(config: dict[str, Any], *, device: str) -> dict[str, Any]:
    prepare = command_prepare_data(config)
    teacher_cache = command_build_teacher_cache(config)
    stage1 = command_train_stage1(config, device=device)
    qat = command_train_qat(config, device=device)
    evaluation = command_evaluate(config, device=device)
    report = command_report(config)
    return {
        "prepare_data": prepare,
        "teacher_cache": teacher_cache,
        "stage1": stage1,
        "qat": qat,
        "evaluation": evaluation,
        "report": report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone repro pipeline for metricgan_plus_native8k_causal_s.")
    parser.add_argument("--config", default=(PROJECT_ROOT / "configs" / "default.yaml").as_posix())
    parser.add_argument("--device", default="auto")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare_data")
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--device", dest="subcommand_device", default=None)

    prepare_dataset = subparsers.add_parser("prepare_dataset")
    prepare_dataset.add_argument("--force", action="store_true")
    prepare_dataset.add_argument("--device", dest="subcommand_device", default=None)

    teacher_cache = subparsers.add_parser("build_teacher_cache")
    teacher_cache.add_argument("--force", action="store_true")
    teacher_cache.add_argument("--device", dest="subcommand_device", default=None)

    train_teacher = subparsers.add_parser("train_teacher")
    train_teacher.add_argument("--device", dest="subcommand_device", default=None)

    train_stage1 = subparsers.add_parser("train_stage1")
    train_stage1.add_argument("--device", dest="subcommand_device", default=None)

    train_qat = subparsers.add_parser("train_qat")
    train_qat.add_argument("--device", dest="subcommand_device", default=None)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", default=None)
    evaluate.add_argument("--label", default=None)
    evaluate.add_argument("--device", dest="subcommand_device", default=None)

    report = subparsers.add_parser("report")
    report.add_argument("--evaluation-dir", default=None)
    report.add_argument("--report-dir", default=None)
    report.add_argument("--device", dest="subcommand_device", default=None)

    run_all = subparsers.add_parser("run_all")
    run_all.add_argument("--device", dest="subcommand_device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("tracking", {})
    config["tracking"].setdefault("experiment_name", DEFAULT_EXPERIMENT_NAME)
    device = require_cuda_device(getattr(args, "subcommand_device", None) or args.device)

    if args.command in ["prepare_data", "prepare_dataset"]:
        payload = command_prepare_data(config, force=bool(args.force))
    elif args.command == "build_teacher_cache":
        payload = command_build_teacher_cache(config, force=bool(args.force))
    elif args.command == "train_teacher":
        payload = command_train_teacher(config, device=device)
    elif args.command == "train_stage1":
        payload = command_train_stage1(config, device=device)
    elif args.command == "train_qat":
        payload = command_train_qat(config, device=device)
    elif args.command == "evaluate":
        payload = command_evaluate(config, device=device, checkpoint=args.checkpoint, label=args.label)
    elif args.command == "report":
        payload = command_report(config, evaluation_dir=args.evaluation_dir, report_dir=args.report_dir)
    elif args.command == "run_all":
        payload = command_run_all(config, device=device)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
