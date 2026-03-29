from __future__ import annotations

import json
import os
import random
import shutil
import signal
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import mlflow
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from metrics.dnsmos import dnsmos_wav
from metrics.composite import composite_scores
from metrics.pesq import pesq_score
from metrics.sisdr import sisdr
from metrics.snr import delta_snr
from metrics.stoi import stoi_score
from sebench.audio import load_mono_audio, loop_to_length, manifest_hash, save_mono_audio, tensor_to_numpy_mono
from sebench.checkpoints import load_model_from_checkpoint, save_checkpoint_package
from sebench.data import ManifestRow, VoiceBankDemandDataset, read_pair_manifest
from sebench.losses import CompositeEnhancementLoss
from sebench.mlflow_utils import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    flatten_params,
    log_dict_artifact,
    terminate_matching_runs,
)
from sebench.models import build_enhancer
from sebench.runtime import require_cuda_device
from sebench.stm32sim import simulate_model_fit
from sebench.teacher_cache import TeacherCacheDataset


@dataclass
class ExperimentConfig:
    train_csv: str
    checkpoint_out: str
    model_family: str = "atennuate"
    variant: str = "base"
    loss_recipe: str = "R1"
    val_rank_csv: str | None = None
    val_select_csv: str | None = None
    test_csv: str | None = None
    run_name: str | None = None
    phase: str | None = None
    epochs: int = 30
    batch_size: int | None = None
    grad_accum: int | None = None
    lr: float = 1e-3
    segment_len: int = 32000
    num_workers: int | None = None
    lr_factor: float = 0.5
    lr_patience: int = 2
    min_lr: float = 1e-6
    early_stop_patience: int = 5
    min_epochs: int = 10
    eval_every: int = 2
    grad_clip: float = 5.0
    seed: int = 0
    amp: bool = True
    scheduler: str = "plateau"
    device: str = "cuda"
    mlflow_uri: str = DEFAULT_TRACKING_URI
    mlflow_artifact_root: str = DEFAULT_ARTIFACT_ROOT
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    parent_run_id: str | None = None
    selection_metric: str = "val_select_pesq_mean"
    eval_dnsmos: bool = True
    sample_count: int = 3
    benchmark_seconds: int = 10
    benchmark_repeats: int = 3
    max_eval_files: int | None = None
    eval_batch_size: int | None = None
    cache_eval_audio: bool = True
    rank_compute_composite: bool = True
    select_compute_composite: bool = True
    postfilter_mode: str = "none"
    postfilter_preset: str = "medium"
    train_postfilter: bool = False
    spectral_native_gate: bool = False
    teacher_source_run_id: str | None = None
    teacher_variant: str | None = None
    audit_only: bool = False
    teacher_cache_manifest: str | None = None
    guidance_classic: str = "none"
    erb_bands: int = 32
    context_frames: int = 5
    qat: bool = False
    mcu_profile: str | None = None
    init_checkpoint: str | None = None
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 320
    log_torch_model: bool = False
    log_system_metrics: bool = False
    quantize_dynamic: bool = False


def _normalize_manifest_path(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def _clean_key_from_path(value: str) -> str:
    path = _normalize_manifest_path(value)
    for marker in ("/clean_train/", "/clean_val/", "/clean_test/", "/clean_sources/"):
        if marker in path:
            return path.split(marker, 1)[1].lstrip("/")
    return path


def _manifest_keysets(csv_path: str) -> dict[str, Any]:
    rows = read_pair_manifest(csv_path)
    pair_set: set[str] = set()
    clean_set: set[str] = set()
    for row in rows:
        pair_set.add(f"{_normalize_manifest_path(row.noisy.as_posix())}|{_normalize_manifest_path(row.clean.as_posix())}")
        clean_set.add(_clean_key_from_path(row.clean.as_posix()))
    return {
        "path": Path(csv_path).resolve().as_posix(),
        "rows": len(rows),
        "pair_set": pair_set,
        "clean_set": clean_set,
        "duplicate_pairs": len(rows) - len(pair_set),
        "duplicate_clean_keys": len(rows) - len(clean_set),
    }


def _validate_manifest_integrity(config: ExperimentConfig) -> None:
    manifests: dict[str, str] = {"train": config.train_csv}
    if config.val_rank_csv:
        manifests["val_rank"] = config.val_rank_csv
    if config.val_select_csv:
        manifests["val_select"] = config.val_select_csv
    if config.test_csv:
        manifests["test"] = config.test_csv

    stats = {name: _manifest_keysets(path) for name, path in manifests.items()}
    for name, payload in stats.items():
        if payload["duplicate_pairs"] > 0 or payload["duplicate_clean_keys"] > 0:
            raise ValueError(
                f"Manifest `{name}` has duplicates (pairs={payload['duplicate_pairs']} clean_keys={payload['duplicate_clean_keys']}): "
                f"{payload['path']}"
            )

    train_pairs = stats["train"]["pair_set"]
    train_clean = stats["train"]["clean_set"]
    val_pairs: set[str] = set()
    val_clean: set[str] = set()
    for name in ("val_rank", "val_select"):
        if name in stats:
            val_pairs.update(stats[name]["pair_set"])
            val_clean.update(stats[name]["clean_set"])
    test_pairs = stats["test"]["pair_set"] if "test" in stats else set()
    test_clean = stats["test"]["clean_set"] if "test" in stats else set()

    overlap_train_val_pairs = len(train_pairs & val_pairs)
    overlap_train_val_clean = len(train_clean & val_clean)
    if overlap_train_val_pairs or overlap_train_val_clean:
        raise ValueError(
            f"Data leakage train<->val detected: pair_overlap={overlap_train_val_pairs} clean_overlap={overlap_train_val_clean}"
        )

    overlap_train_test_pairs = len(train_pairs & test_pairs)
    overlap_train_test_clean = len(train_clean & test_clean)
    if overlap_train_test_pairs or overlap_train_test_clean:
        raise ValueError(
            f"Data leakage train<->test detected: pair_overlap={overlap_train_test_pairs} clean_overlap={overlap_train_test_clean}"
        )

    overlap_val_test_pairs = len(val_pairs & test_pairs)
    overlap_val_test_clean = len(val_clean & test_clean)
    if overlap_val_test_pairs or overlap_val_test_clean:
        raise ValueError(
            f"Data leakage val<->test detected: pair_overlap={overlap_val_test_pairs} clean_overlap={overlap_val_test_clean}"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataloader_seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def suggest_num_workers(cpu_count: int | None = None) -> int:
    cpu_total = cpu_count or os.cpu_count() or 4
    if cpu_total <= 2:
        return 0
    return min(6, max(2, cpu_total - 2))


def suggest_runtime_profile(model_family: str, variant: str, segment_len: int) -> dict[str, int]:
    short_segment = segment_len <= 16000

    if model_family == "cmgan_small":
        if variant == "small":
            batch_size = 3 if short_segment else 2
        else:
            batch_size = 2 if short_segment else 1
    elif model_family == "metricgan_plus":
        batch_size = 12 if short_segment else 8
    elif model_family == "metricgan_plus_refiner":
        batch_size = 6 if short_segment else 4
    elif model_family == "metricgan_plus_native8k":
        batch_size = 12 if short_segment else 8
    elif model_family == "metricgan_plus_native8k_causal_s":
        batch_size = 16 if short_segment else 12
    elif model_family == "metricgan_plus_native8k_causal_xs":
        batch_size = 18 if short_segment else 14
    elif model_family == "metricgan_plus_native8k_causal_n6":
        batch_size = 12 if short_segment else 10
    elif model_family == "tiny_stm32_fc":
        batch_size = 16 if short_segment else 12
    elif model_family == "tiny_stm32_hybrid_sg":
        batch_size = 14 if short_segment else 10
    elif model_family == "tiny_stm32_tcn_hybrid":
        batch_size = 10 if short_segment else 8
    elif model_family == "atennuate":
        if variant == "small":
            batch_size = 8 if short_segment else 5
        else:
            batch_size = 6 if short_segment else 4
    elif model_family == "fullsubnet_plus":
        if variant == "small":
            batch_size = 10 if short_segment else 6
        else:
            batch_size = 8 if short_segment else 5
    elif model_family == "mp_senet":
        if variant == "small":
            batch_size = 12 if short_segment else 8
        else:
            batch_size = 10 if short_segment else 6
    elif variant == "small":
        batch_size = 8 if short_segment else 4
    else:
        batch_size = 6 if short_segment else 4

    target_effective_batch = 8
    grad_accum = max(1, (target_effective_batch + batch_size - 1) // batch_size)
    eval_batch_size = min(16, max(4, batch_size * (2 if model_family != "cmgan_small" else 1)))
    return {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "num_workers": suggest_num_workers(),
        "eval_batch_size": eval_batch_size,
    }


def apply_runtime_profile(config: ExperimentConfig) -> None:
    profile = suggest_runtime_profile(config.model_family, config.variant, config.segment_len)
    if config.batch_size is None or config.batch_size <= 0:
        config.batch_size = profile["batch_size"]
    if config.grad_accum is None or config.grad_accum <= 0:
        config.grad_accum = profile["grad_accum"]
    if config.num_workers is None or config.num_workers < 0:
        config.num_workers = profile["num_workers"]
    if config.eval_batch_size is None or config.eval_batch_size <= 0:
        config.eval_batch_size = profile["eval_batch_size"]


def install_termination_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def _raise_interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, _raise_interrupt)
    return previous


def restore_termination_handlers(previous: dict[int, Any]) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def build_dataloader(csv_path: str, config: ExperimentConfig, shuffle: bool) -> DataLoader:
    if config.teacher_cache_manifest and shuffle:
        dataset = TeacherCacheDataset(
            config.teacher_cache_manifest,
            segment_len=config.segment_len,
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
    else:
        dataset = VoiceBankDemandDataset(csv_path, segment_len=config.segment_len, sample_rate=config.sample_rate)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    num_workers = max(int(config.num_workers or 0), 0)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": shuffle,
        "pin_memory": config.device.startswith("cuda"),
        "worker_init_fn": dataloader_seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(
        **loader_kwargs,
    )


def _selection_score(metrics: dict[str, float]) -> float:
    if "pesq_mean" in metrics:
        return float(metrics["pesq_mean"])
    if "loss" in metrics:
        return float(-metrics["loss"])
    return float("-inf")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CompositeEnhancementLoss,
    config: ExperimentConfig,
    epoch: int,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    model.train()
    running_total = 0.0
    running_wave = 0.0
    running_spec = 0.0
    running_sisdr = 0.0
    running_noise_gate = 0.0
    running_speech_preserve = 0.0
    running_teacher_mask = 0.0
    running_teacher_wave = 0.0
    seen = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", unit="batch")
    autocast_enabled = config.amp and config.device.startswith("cuda")

    for step, batch in enumerate(progress, start=1):
        guidance = None
        teacher_wav = None
        teacher_mask_erb = None
        if isinstance(batch, dict):
            noisy = batch["noisy"]
            clean = batch["clean"]
            teacher_wav = batch.get("teacher_wav")
            teacher_mask_erb = batch.get("teacher_mask_erb")
            guidance = batch.get("guidance_sg")
        else:
            noisy, clean = batch
        noisy = noisy.to(config.device, non_blocking=config.device.startswith("cuda")).unsqueeze(1)
        clean = clean.to(config.device, non_blocking=config.device.startswith("cuda")).unsqueeze(1)
        if teacher_wav is not None:
            teacher_wav = teacher_wav.to(config.device, non_blocking=config.device.startswith("cuda")).unsqueeze(1)
        if teacher_mask_erb is not None:
            teacher_mask_erb = teacher_mask_erb.to(config.device, non_blocking=config.device.startswith("cuda"))
        if guidance is not None:
            guidance = guidance.to(config.device, non_blocking=config.device.startswith("cuda"))

        with torch.autocast(device_type="cuda" if config.device.startswith("cuda") else "cpu", enabled=autocast_enabled):
            try:
                enhanced = model(noisy, guidance=guidance) if guidance is not None else model(noisy)
            except TypeError:
                enhanced = model(noisy)
            breakdown = loss_fn(
                enhanced,
                clean,
                noisy,
                epoch=epoch,
                total_epochs=config.epochs,
                teacher_wav=teacher_wav,
                teacher_mask_erb=teacher_mask_erb,
            )
            scaled_loss = breakdown.total / float(config.grad_accum)

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = step % config.grad_accum == 0 or step == len(loader)
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = noisy.size(0)
        running_total += breakdown.total.item() * batch_size
        running_wave += breakdown.wave.item() * batch_size
        running_spec += breakdown.spectral.item() * batch_size
        running_sisdr += breakdown.sisdr.item() * batch_size
        running_noise_gate += breakdown.noise_gate.item() * batch_size
        running_speech_preserve += breakdown.speech_preserve.item() * batch_size
        running_teacher_mask += breakdown.teacher_mask.item() * batch_size
        running_teacher_wave += breakdown.teacher_wave.item() * batch_size
        seen += batch_size
        progress.set_postfix(
            loss=f"{running_total / seen:.4f}",
            wave=f"{running_wave / seen:.4f}",
            spectral=f"{running_spec / seen:.4f}",
            sisdr=f"{running_sisdr / seen:.4f}",
            gate=f"{running_noise_gate / seen:.4f}",
            teacher=f"{running_teacher_mask / seen:.4f}",
        )

    return {
        "loss": running_total / max(seen, 1),
        "wave_loss": running_wave / max(seen, 1),
        "spectral_loss": running_spec / max(seen, 1),
        "sisdr_loss": running_sisdr / max(seen, 1),
        "noise_gate_loss": running_noise_gate / max(seen, 1),
        "speech_preserve_loss": running_speech_preserve / max(seen, 1),
        "teacher_mask_loss": running_teacher_mask / max(seen, 1),
        "teacher_wave_loss": running_teacher_wave / max(seen, 1),
    }


@torch.inference_mode()
def benchmark_inference(
    model: nn.Module,
    sample_path: str | Path,
    device: str,
    sample_rate: int,
    duration_seconds: int = 10,
    repeats: int = 3,
) -> float:
    noisy, sr = load_mono_audio(sample_path, sample_rate)
    noisy = loop_to_length(noisy, duration_seconds * sr).unsqueeze(0).to(device)
    timings: list[float] = []
    autocast_enabled = device.startswith("cuda")

    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=autocast_enabled):
            _ = model.denoise_single(noisy)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    return float(mean(timings))


EvalAudioRow = tuple[ManifestRow, torch.Tensor, torch.Tensor, int]
_EVAL_AUDIO_CACHE: dict[tuple[str, int, int | None], list[EvalAudioRow]] = {}


def _load_eval_audio_rows(
    manifest_path: str,
    *,
    sample_rate: int,
    use_cache: bool,
    max_files: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[EvalAudioRow]:
    resolved = Path(manifest_path).resolve().as_posix()
    cache_key = (resolved, sample_rate, max_files)
    if use_cache and cache_key in _EVAL_AUDIO_CACHE:
        if progress_callback is not None:
            progress_callback(f"eval cache hit for {Path(manifest_path).name}: {len(_EVAL_AUDIO_CACHE[cache_key])} rows")
        return _EVAL_AUDIO_CACHE[cache_key]

    manifest_rows = read_pair_manifest(manifest_path)
    if max_files is not None:
        manifest_rows = manifest_rows[:max_files]
    if progress_callback is not None:
        progress_callback(f"loading eval audio from {Path(manifest_path).name}: {len(manifest_rows)} rows")

    loaded: list[EvalAudioRow] = []
    total_rows = len(manifest_rows)
    for row_index, row in enumerate(manifest_rows, start=1):
        noisy, sr = load_mono_audio(row.noisy, sample_rate)
        clean, _ = load_mono_audio(row.clean, sample_rate)
        loaded.append((row, noisy.contiguous(), clean.contiguous(), sr))
        if progress_callback is not None and (row_index == total_rows or row_index == 1 or row_index % 100 == 0):
            progress_callback(f"loaded eval audio {row_index}/{total_rows} from {Path(manifest_path).name}")

    if use_cache:
        _EVAL_AUDIO_CACHE[cache_key] = loaded
    return loaded


@torch.inference_mode()
def evaluate_manifest(
    model: nn.Module,
    manifest_path: str,
    device: str,
    *,
    sample_rate: int,
    compute_dnsmos: bool,
    compute_composite: bool = True,
    sample_dir: str | Path | None = None,
    sample_count: int = 0,
    max_files: int | None = None,
    batch_size: int = 1,
    cache_audio: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if compute_dnsmos and sample_rate != 16000:
        raise ValueError("DNSMOS evaluation requires 16000 Hz audio.")
    rows = _load_eval_audio_rows(
        manifest_path,
        sample_rate=sample_rate,
        use_cache=cache_audio,
        max_files=max_files,
        progress_callback=progress_callback,
    )

    pesq_values: list[float] = []
    stoi_values: list[float] = []
    sisdr_values: list[float] = []
    delta_snr_values: list[float] = []
    csig_values: list[float] = []
    cbak_values: list[float] = []
    covl_values: list[float] = []
    dnsmos_sig: list[float] = []
    dnsmos_bak: list[float] = []
    dnsmos_ovr: list[float] = []
    saved_samples: list[str] = []

    out_dir = Path(sample_dir) if sample_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    temp_dnsmos_dir = Path(tempfile.mkdtemp(prefix="sebench_dnsmos_")) if compute_dnsmos else None
    model_was_training = model.training
    model.eval()
    eval_batch_size = max(1, int(batch_size))
    autocast_enabled = device.startswith("cuda")

    try:
        row_index = 0
        total_rows = len(rows)
        while row_index < len(rows):
            batch_rows = rows[row_index:row_index + eval_batch_size]
            noisy_batch = pad_sequence([item[1] for item in batch_rows], batch_first=True)
            try:
                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=autocast_enabled):
                    enhanced_batch = model.denoise_single(
                        noisy_batch.to(device, non_blocking=device.startswith("cuda"))
                    ).cpu()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device.startswith("cuda") and eval_batch_size > 1:
                    torch.cuda.empty_cache()
                    eval_batch_size = max(1, eval_batch_size // 2)
                    continue
                raise
            if progress_callback is not None:
                completed = min(row_index + len(batch_rows), total_rows)
                if completed == total_rows or completed == len(batch_rows) or completed % 128 == 0:
                    progress_callback(
                        f"evaluated {completed}/{total_rows} files from {Path(manifest_path).name} "
                        f"(batch_size={eval_batch_size})"
                    )

            for batch_offset, (row, noisy_full, clean_full, sr) in enumerate(batch_rows):
                enhanced = enhanced_batch[batch_offset]
                aligned = min(noisy_full.numel(), clean_full.numel(), enhanced.numel())
                noisy = noisy_full[:aligned]
                clean = clean_full[:aligned]
                enhanced = enhanced[:aligned]

                clean_np = tensor_to_numpy_mono(clean)
                noisy_np = tensor_to_numpy_mono(noisy)
                enhanced_np = tensor_to_numpy_mono(enhanced)

                pesq = pesq_score(clean_np, enhanced_np, sr)
                pesq_values.append(pesq)
                stoi_values.append(stoi_score(clean_np, enhanced_np, sr, extended=False))
                sisdr_values.append(sisdr(clean_np, enhanced_np))
                delta_snr_values.append(delta_snr(clean_np, noisy_np, enhanced_np))
                if compute_composite:
                    composite = composite_scores(clean_np, enhanced_np, sr, pesq_value=pesq)
                    csig_values.append(composite["csig"])
                    cbak_values.append(composite["cbak"])
                    covl_values.append(composite["covl"])

                save_sample_triplet = out_dir is not None and row_index + batch_offset < sample_count
                enhanced_path: Path | None = None
                raw_enhanced_path: Path | None = None
                if save_sample_triplet:
                    enhanced_path = out_dir / f"{Path(row.noisy).stem}_enh.wav"
                    save_mono_audio(enhanced_path, enhanced, sr)
                    if getattr(model, "postfilter_active", False) and hasattr(model, "denoise_raw"):
                        raw_enhanced = model.denoise_raw(
                            noisy.unsqueeze(0).to(device, non_blocking=device.startswith("cuda"))
                        ).squeeze(0).cpu()
                        raw_enhanced = raw_enhanced[:aligned]
                        raw_enhanced_path = out_dir / f"{Path(row.noisy).stem}_raw_enh.wav"
                        save_mono_audio(raw_enhanced_path, raw_enhanced, sr)

                if compute_dnsmos:
                    dnsmos_path = enhanced_path
                    if dnsmos_path is None and temp_dnsmos_dir is not None:
                        dnsmos_path = temp_dnsmos_dir / f"{row_index + batch_offset:05d}_enh.wav"
                        save_mono_audio(dnsmos_path, enhanced, sr)
                    if dnsmos_path is not None:
                        dns = dnsmos_wav(dnsmos_path.as_posix())
                        dnsmos_sig.append(float(dns["mos_sig"]))
                        dnsmos_bak.append(float(dns["mos_bak"]))
                        dnsmos_ovr.append(float(dns["mos_ovr"]))
                        if temp_dnsmos_dir is not None and dnsmos_path.parent == temp_dnsmos_dir:
                            dnsmos_path.unlink(missing_ok=True)

                if save_sample_triplet and enhanced_path is not None:
                    noisy_path = out_dir / f"{Path(row.noisy).stem}_noisy.wav"
                    clean_path = out_dir / f"{Path(row.clean).stem}_clean.wav"
                    save_mono_audio(noisy_path, noisy, sr)
                    save_mono_audio(clean_path, clean, sr)
                    saved_samples.append(noisy_path.as_posix())
                    saved_samples.append(clean_path.as_posix())
                    if raw_enhanced_path is not None:
                        saved_samples.append(raw_enhanced_path.as_posix())
                    saved_samples.append(enhanced_path.as_posix())

            row_index += len(batch_rows)
    finally:
        if temp_dnsmos_dir is not None:
            shutil.rmtree(temp_dnsmos_dir, ignore_errors=True)
        if model_was_training:
            model.train()

    metrics: dict[str, Any] = {
        "count": len(rows),
        "pesq_mean": float(mean(pesq_values)),
        "stoi_mean": float(mean(stoi_values)),
        "sisdr_mean": float(mean(sisdr_values)),
        "delta_snr_mean": float(mean(delta_snr_values)),
        "sample_paths": saved_samples,
    }
    if csig_values:
        metrics["csig_mean"] = float(mean(csig_values))
        metrics["cbak_mean"] = float(mean(cbak_values))
        metrics["covl_mean"] = float(mean(covl_values))
    if dnsmos_sig:
        metrics["dnsmos_sig_mean"] = float(mean(dnsmos_sig))
        metrics["dnsmos_bak_mean"] = float(mean(dnsmos_bak))
        metrics["dnsmos_ovr_mean"] = float(mean(dnsmos_ovr))
    return metrics


def _start_run(config: ExperimentConfig) -> tuple[Any, str]:
    experiment_id = configure_mlflow(
        tracking_uri=config.mlflow_uri,
        experiment_name=config.experiment_name,
        artifact_root=config.mlflow_artifact_root,
    )
    nested = mlflow.active_run() is not None or bool(config.parent_run_id)
    terminate_matching_runs(
        tracking_uri=config.mlflow_uri,
        experiment_name=config.experiment_name,
        run_name=config.run_name,
        phase=config.phase,
    )
    tags = {
        "model_family": config.model_family,
        "variant": config.variant,
        "phase": config.phase or "",
        "loss_recipe": config.loss_recipe,
    }
    if config.parent_run_id and mlflow.active_run() is None:
        tags["mlflow.parentRunId"] = config.parent_run_id
    run = mlflow.start_run(
        run_name=config.run_name,
        experiment_id=experiment_id,
        nested=nested,
        tags=tags,
        log_system_metrics=config.log_system_metrics,
    )
    return run, experiment_id


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    config.device = require_cuda_device(config.device)
    set_seed(config.seed)
    apply_runtime_profile(config)
    _validate_manifest_integrity(config)
    if config.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    run, _ = _start_run(config)
    previous_handlers = install_termination_handlers()
    checkpoint_path = Path(config.checkpoint_out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(config.train_csv, config, shuffle=True)
    model = build_enhancer(
        config.model_family,
        config.variant,
        spectral_native_gate=config.spectral_native_gate,
        postfilter_mode=config.postfilter_mode,
        postfilter_preset=config.postfilter_preset,
        train_postfilter=config.train_postfilter,
        erb_bands=config.erb_bands,
        context_frames=config.context_frames,
        guidance_classic=config.guidance_classic,
        qat=config.qat,
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
    ).to(config.device)
    if config.init_checkpoint:
        init_model, _ = load_model_from_checkpoint(
            config.init_checkpoint,
            device=config.device,
            model_family=config.model_family,
            variant=config.variant,
        )
        source = init_model.base_model if hasattr(init_model, "base_model") else init_model
        target = model.base_model if hasattr(model, "base_model") else model
        target.load_state_dict(source.state_dict(), strict=False)
    loss_fn = CompositeEnhancementLoss(
        config.loss_recipe,
        sample_rate=config.sample_rate,
        erb_bands=config.erb_bands,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
    )
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError(f"Model family `{config.model_family}` has no trainable parameters.")
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.02)
    scheduler = None
    if config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=config.min_lr,
        )
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

    autocast_scaler = None
    if config.amp and config.device.startswith("cuda"):
        autocast_scaler = torch.amp.GradScaler("cuda", enabled=True)

    params = flatten_params(asdict(config))
    params["train_manifest_hash"] = manifest_hash(config.train_csv)
    if config.val_rank_csv:
        params["val_rank_manifest_hash"] = manifest_hash(config.val_rank_csv)
    if config.val_select_csv:
        params["val_select_manifest_hash"] = manifest_hash(config.val_select_csv)
    if config.teacher_cache_manifest:
        params["teacher_cache_manifest_hash"] = manifest_hash(config.teacher_cache_manifest)
    mlflow.log_params(params)
    stm32_summary: dict[str, Any] | None = None
    if config.mcu_profile:
        stm32_summary = simulate_model_fit(model, profile_name=config.mcu_profile)
        mlflow.log_metrics(
            {
                "stm32sim/flash_bytes": stm32_summary["flash_bytes"],
                "stm32sim/sram_peak_bytes": stm32_summary["sram_peak_bytes"],
                "stm32sim/macs_per_hop_total": stm32_summary["macs_per_hop_total"],
                "stm32sim/macs_fc": stm32_summary["macs_fc"],
                "stm32sim/macs_depthwise_conv1d": stm32_summary["macs_depthwise_conv1d"],
                "stm32sim/macs_pointwise_conv1d": stm32_summary["macs_pointwise_conv1d"],
                "stm32sim/macs_lstm": stm32_summary["macs_lstm"],
                "stm32sim/eltwise_ops": stm32_summary["eltwise_ops"],
                "stm32sim/lookup_ops": stm32_summary["lookup_ops"],
                "stm32sim/cycles_per_hop": stm32_summary["cycles_per_hop"],
                "stm32sim/ms_per_hop_80mhz": stm32_summary["ms_per_hop_80mhz"],
                "stm32sim/hop_ms": stm32_summary["hop_ms"],
                "stm32sim/lookahead_ms": stm32_summary["lookahead_ms"],
                "stm32sim/min_required_mhz": stm32_summary["min_required_mhz"],
                "stm32sim/recommended_rt_mhz": stm32_summary["recommended_rt_mhz"],
                "stm32sim/max_profile_mhz": stm32_summary["max_profile_mhz"],
                "stm32sim/cpu_load_pct": stm32_summary["cpu_load_pct"],
                "stm32sim/fit_ok": 1.0 if stm32_summary["fit_ok"] else 0.0,
                "stm32sim/frequency_ok": 1.0 if stm32_summary["frequency_ok"] else 0.0,
                "stm32sim/realtime_ok": 1.0 if stm32_summary["realtime_ok"] else 0.0,
                "stm32sim/latency_ok": 1.0 if stm32_summary["latency_ok"] else 0.0,
                "stm32sim/avg_power_mw": stm32_summary["avg_power_mw"],
                "stm32sim/avg_power_mw_at_recommended_mhz": stm32_summary["avg_power_mw_at_recommended_mhz"],
                "stm32sim/energy_uj_per_hop": stm32_summary["energy_uj_per_hop"],
                "stm32sim/energy_uj_per_hop_at_recommended_mhz": stm32_summary["energy_uj_per_hop_at_recommended_mhz"],
                "stm32sim/power_ok": 1.0 if stm32_summary["power_ok"] else 0.0,
                "stm32sim/deployment_ok": 1.0 if stm32_summary["deployment_ok"] else 0.0,
            }
        )
        log_dict_artifact(stm32_summary, "reports/stm32sim.json")

    best_score = float("-inf")
    best_epoch = 0
    best_select_metrics: dict[str, float] = {}
    best_rank_metrics: dict[str, float] = {}
    epochs_without_improve = 0
    sample_source = config.val_rank_csv or config.val_select_csv or config.train_csv
    sample_rows = read_pair_manifest(sample_source)
    sample_path = sample_rows[0].noisy

    summary: dict[str, Any] = {
        "run_id": run.info.run_id,
        "run_name": config.run_name,
        "model_family": config.model_family,
        "variant": config.variant,
        "loss_recipe": config.loss_recipe,
        "seed": config.seed,
        "postfilter_mode": config.postfilter_mode,
        "postfilter_preset": config.postfilter_preset,
        "train_postfilter": config.train_postfilter,
        "spectral_native_gate": config.spectral_native_gate,
        "teacher_source_run_id": config.teacher_source_run_id,
        "teacher_variant": config.teacher_variant,
        "audit_only": config.audit_only,
        "teacher_cache_manifest": config.teacher_cache_manifest,
        "guidance_classic": config.guidance_classic,
        "erb_bands": config.erb_bands,
        "context_frames": config.context_frames,
        "qat": config.qat,
        "mcu_profile": config.mcu_profile,
        "stm32sim": stm32_summary or {},
        "quantize_dynamic": config.quantize_dynamic,
    }

    run_status = "FINISHED"
    try:
        for epoch in range(1, config.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, loss_fn, config, epoch, autocast_scaler)
            mlflow.log_metrics({f"train/{key}": value for key, value in train_metrics.items()}, step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            if scheduler is not None and config.scheduler == "cosine":
                scheduler.step(epoch - 1)

            should_eval = config.val_rank_csv and epoch % config.eval_every == 0
            if not should_eval:
                continue

            rank_metrics = evaluate_manifest(
                model,
                config.val_rank_csv,
                config.device,
                sample_rate=config.sample_rate,
                compute_dnsmos=False,
                compute_composite=config.rank_compute_composite,
                max_files=config.max_eval_files,
                batch_size=config.eval_batch_size,
                cache_audio=config.cache_eval_audio,
            )
            rank_log_metrics = {
                "val_rank/pesq_mean": rank_metrics["pesq_mean"],
                "val_rank/stoi_mean": rank_metrics["stoi_mean"],
                "val_rank/sisdr_mean": rank_metrics["sisdr_mean"],
                "val_rank/delta_snr_mean": rank_metrics["delta_snr_mean"],
            }
            if "csig_mean" in rank_metrics:
                rank_log_metrics["val_rank/csig_mean"] = rank_metrics["csig_mean"]
                rank_log_metrics["val_rank/cbak_mean"] = rank_metrics["cbak_mean"]
                rank_log_metrics["val_rank/covl_mean"] = rank_metrics["covl_mean"]
            mlflow.log_metrics(rank_log_metrics, step=epoch)

            score = _selection_score(rank_metrics)
            if scheduler is not None and config.scheduler == "plateau":
                scheduler.step(score)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                epochs_without_improve = 0
                best_rank_metrics = rank_metrics
                mlflow.log_metric("best/val_rank_pesq_mean", score, step=epoch)

                save_checkpoint_package(
                    checkpoint_path,
                    model=model,
                    model_family=config.model_family,
                    variant=config.variant,
                    extra={
                        "epoch": epoch,
                        "loss_recipe": config.loss_recipe,
                        "seed": config.seed,
                        "best_val_rank_pesq": score,
                    },
                )
            else:
                epochs_without_improve += 1

            if epoch >= config.min_epochs and config.early_stop_patience > 0 and epochs_without_improve >= config.early_stop_patience:
                break

        if not checkpoint_path.exists():
            save_checkpoint_package(
                checkpoint_path,
                model=model,
                model_family=config.model_family,
                variant=config.variant,
                extra={"epoch": config.epochs, "loss_recipe": config.loss_recipe, "seed": config.seed},
            )

        mlflow.log_artifact(checkpoint_path.as_posix(), artifact_path="checkpoints")

        final_model, _ = load_model_from_checkpoint(
            checkpoint_path,
            device=config.device,
            model_family=config.model_family,
            variant=config.variant,
        )
        if config.val_select_csv:
            sample_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_samples"
            if sample_dir.exists():
                shutil.rmtree(sample_dir, ignore_errors=True)
            try:
                select_metrics = evaluate_manifest(
                    final_model,
                    config.val_select_csv,
                    config.device,
                    sample_rate=config.sample_rate,
                    compute_dnsmos=config.eval_dnsmos,
                    compute_composite=config.select_compute_composite,
                    sample_dir=sample_dir,
                    sample_count=config.sample_count,
                    max_files=config.max_eval_files,
                    batch_size=config.eval_batch_size,
                    cache_audio=config.cache_eval_audio,
                )
            except ValueError as exc:
                if "Manifest is empty" not in str(exc):
                    raise
                select_metrics = dict(best_rank_metrics)
            latency_seconds = benchmark_inference(
                final_model,
                sample_path=sample_path,
                device=config.device,
                sample_rate=config.sample_rate,
                duration_seconds=config.benchmark_seconds,
                repeats=config.benchmark_repeats,
            )
            select_metrics["benchmark_latency_10s"] = latency_seconds
            best_select_metrics = select_metrics
            select_log_metrics = {
                "best/val_select_pesq_mean": select_metrics["pesq_mean"],
                "best/val_select_stoi_mean": select_metrics["stoi_mean"],
                "best/val_select_sisdr_mean": select_metrics["sisdr_mean"],
                "best/val_select_delta_snr_mean": select_metrics["delta_snr_mean"],
                "best/inference_seconds_10s": latency_seconds,
            }
            if "csig_mean" in select_metrics:
                select_log_metrics["best/val_select_csig_mean"] = select_metrics["csig_mean"]
                select_log_metrics["best/val_select_cbak_mean"] = select_metrics["cbak_mean"]
                select_log_metrics["best/val_select_covl_mean"] = select_metrics["covl_mean"]
            mlflow.log_metrics(select_log_metrics)
            if "dnsmos_ovr_mean" in select_metrics:
                mlflow.log_metrics(
                    {
                        "best/val_select_dnsmos_sig_mean": select_metrics["dnsmos_sig_mean"],
                        "best/val_select_dnsmos_bak_mean": select_metrics["dnsmos_bak_mean"],
                        "best/val_select_dnsmos_ovr_mean": select_metrics["dnsmos_ovr_mean"],
                    }
                )
            if sample_dir.exists():
                mlflow.log_artifacts(sample_dir.as_posix(), artifact_path="samples")
                shutil.rmtree(sample_dir, ignore_errors=True)
            log_dict_artifact(select_metrics, "reports/best_val_select_metrics.json")

        if config.test_csv:
            test_sample_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_test_samples"
            if test_sample_dir.exists():
                shutil.rmtree(test_sample_dir, ignore_errors=True)
            test_metrics = evaluate_manifest(
                final_model,
                config.test_csv,
                config.device,
                sample_rate=config.sample_rate,
                compute_dnsmos=config.eval_dnsmos,
                compute_composite=config.select_compute_composite,
                sample_dir=test_sample_dir,
                sample_count=config.sample_count,
                max_files=config.max_eval_files,
                batch_size=config.eval_batch_size,
                cache_audio=config.cache_eval_audio,
            )
            test_log_metrics = {
                "test/pesq_mean": test_metrics["pesq_mean"],
                "test/stoi_mean": test_metrics["stoi_mean"],
                "test/sisdr_mean": test_metrics["sisdr_mean"],
                "test/delta_snr_mean": test_metrics["delta_snr_mean"],
            }
            if "csig_mean" in test_metrics:
                test_log_metrics["test/csig_mean"] = test_metrics["csig_mean"]
                test_log_metrics["test/cbak_mean"] = test_metrics["cbak_mean"]
                test_log_metrics["test/covl_mean"] = test_metrics["covl_mean"]
            mlflow.log_metrics(test_log_metrics)
            if "dnsmos_ovr_mean" in test_metrics:
                mlflow.log_metrics(
                    {
                        "test/dnsmos_sig_mean": test_metrics["dnsmos_sig_mean"],
                        "test/dnsmos_bak_mean": test_metrics["dnsmos_bak_mean"],
                        "test/dnsmos_ovr_mean": test_metrics["dnsmos_ovr_mean"],
                    }
                )
            if test_sample_dir.exists():
                mlflow.log_artifacts(test_sample_dir.as_posix(), artifact_path="test_samples")
                shutil.rmtree(test_sample_dir, ignore_errors=True)
            log_dict_artifact(test_metrics, "reports/test_metrics.json")
            summary["test_metrics"] = test_metrics

        summary.update(
            {
                "best_epoch": best_epoch,
                "best_val_rank_pesq": best_rank_metrics.get("pesq_mean"),
                "best_val_select_pesq": best_select_metrics.get("pesq_mean"),
                "best_val_select_dnsmos_ovr": best_select_metrics.get("dnsmos_ovr_mean"),
                "inference_seconds_10s": best_select_metrics.get("benchmark_latency_10s"),
                "checkpoint_out": checkpoint_path.as_posix(),
            }
        )
        if config.log_torch_model:
            mlflow.pytorch.log_model(final_model, artifact_path="model")
        log_dict_artifact(summary, "reports/run_summary.json")
        return summary
    except KeyboardInterrupt:
        run_status = "KILLED"
        raise
    except BaseException:
        run_status = "FAILED"
        raise
    finally:
        restore_termination_handlers(previous_handlers)
        mlflow.end_run(status=run_status)


def summary_from_existing(existing: dict[str, Any]) -> dict[str, Any]:
    metrics = existing.get("metrics", {})
    params = existing.get("params", {})
    return {
        "run_id": existing["run_id"],
        "run_name": existing["tags"].get("mlflow.runName"),
        "model_family": params.get("model_family"),
        "variant": params.get("variant"),
        "loss_recipe": params.get("loss_recipe"),
        "seed": int(params["seed"]) if "seed" in params else None,
        "postfilter_mode": params.get("postfilter_mode"),
        "postfilter_preset": params.get("postfilter_preset"),
        "train_postfilter": params.get("train_postfilter"),
        "spectral_native_gate": params.get("spectral_native_gate"),
        "teacher_source_run_id": params.get("teacher_source_run_id"),
        "teacher_variant": params.get("teacher_variant"),
        "audit_only": params.get("audit_only"),
        "teacher_cache_manifest": params.get("teacher_cache_manifest"),
        "guidance_classic": params.get("guidance_classic"),
        "erb_bands": int(params["erb_bands"]) if "erb_bands" in params and params["erb_bands"] not in {"null", ""} else None,
        "context_frames": int(params["context_frames"]) if "context_frames" in params and params["context_frames"] not in {"null", ""} else None,
        "qat": params.get("qat"),
        "mcu_profile": params.get("mcu_profile"),
        "quantize_dynamic": params.get("quantize_dynamic"),
        "stm32sim": {
            "flash_bytes": metrics.get("stm32sim/flash_bytes"),
            "sram_peak_bytes": metrics.get("stm32sim/sram_peak_bytes"),
            "macs_per_hop_total": metrics.get("stm32sim/macs_per_hop_total"),
            "macs_fc": metrics.get("stm32sim/macs_fc"),
            "macs_depthwise_conv1d": metrics.get("stm32sim/macs_depthwise_conv1d"),
            "macs_pointwise_conv1d": metrics.get("stm32sim/macs_pointwise_conv1d"),
            "macs_lstm": metrics.get("stm32sim/macs_lstm"),
            "eltwise_ops": metrics.get("stm32sim/eltwise_ops"),
            "lookup_ops": metrics.get("stm32sim/lookup_ops"),
            "cycles_per_hop": metrics.get("stm32sim/cycles_per_hop"),
            "ms_per_hop_80mhz": metrics.get("stm32sim/ms_per_hop_80mhz"),
            "hop_ms": metrics.get("stm32sim/hop_ms"),
            "lookahead_ms": metrics.get("stm32sim/lookahead_ms"),
            "min_required_mhz": metrics.get("stm32sim/min_required_mhz"),
            "recommended_rt_mhz": metrics.get("stm32sim/recommended_rt_mhz"),
            "max_profile_mhz": metrics.get("stm32sim/max_profile_mhz"),
            "cpu_load_pct": metrics.get("stm32sim/cpu_load_pct"),
            "fit_ok": metrics.get("stm32sim/fit_ok"),
            "frequency_ok": metrics.get("stm32sim/frequency_ok"),
            "realtime_ok": metrics.get("stm32sim/realtime_ok"),
            "latency_ok": metrics.get("stm32sim/latency_ok"),
            "avg_power_mw": metrics.get("stm32sim/avg_power_mw"),
            "avg_power_mw_at_recommended_mhz": metrics.get("stm32sim/avg_power_mw_at_recommended_mhz"),
            "energy_uj_per_hop": metrics.get("stm32sim/energy_uj_per_hop"),
            "energy_uj_per_hop_at_recommended_mhz": metrics.get("stm32sim/energy_uj_per_hop_at_recommended_mhz"),
            "power_ok": metrics.get("stm32sim/power_ok"),
            "deployment_ok": metrics.get("stm32sim/deployment_ok"),
        },
        "best_val_select_pesq": metrics.get("best/val_select_pesq_mean"),
        "best_val_select_stoi": metrics.get("best/val_select_stoi_mean"),
        "best_val_select_dnsmos_ovr": metrics.get("best/val_select_dnsmos_ovr_mean"),
        "best_val_rank_pesq": metrics.get("best/val_rank_pesq_mean") or metrics.get("val_rank/pesq_mean"),
        "inference_seconds_10s": metrics.get("best/inference_seconds_10s"),
        "teacher_accuracy_drop_pesq": metrics.get("teacher_accuracy_drop_pesq"),
        "teacher_accuracy_drop_stoi": metrics.get("teacher_accuracy_drop_stoi"),
        "teacher_accuracy_drop_sisdr": metrics.get("teacher_accuracy_drop_sisdr"),
        "test_metrics": {
            "pesq_mean": metrics.get("test/pesq_mean"),
            "stoi_mean": metrics.get("test/stoi_mean"),
            "sisdr_mean": metrics.get("test/sisdr_mean"),
            "delta_snr_mean": metrics.get("test/delta_snr_mean"),
            "csig_mean": metrics.get("test/csig_mean"),
            "cbak_mean": metrics.get("test/cbak_mean"),
            "covl_mean": metrics.get("test/covl_mean"),
            "dnsmos_sig_mean": metrics.get("test/dnsmos_sig_mean"),
            "dnsmos_bak_mean": metrics.get("test/dnsmos_bak_mean"),
            "dnsmos_ovr_mean": metrics.get("test/dnsmos_ovr_mean"),
        },
    }


def default_experiment_config(**overrides: Any) -> ExperimentConfig:
    family = overrides.get("model_family", "atennuate")
    variant = overrides.get("variant", "base")
    segment_len = int(overrides.get("segment_len", 32000))
    profile = suggest_runtime_profile(family, variant, segment_len)
    batch_size = overrides.get("batch_size", profile["batch_size"])
    grad_accum = overrides.get("grad_accum", profile["grad_accum"])
    num_workers = overrides.get("num_workers", profile["num_workers"])
    eval_batch_size = overrides.get("eval_batch_size", profile["eval_batch_size"])
    config = ExperimentConfig(
        batch_size=batch_size,
        grad_accum=grad_accum,
        num_workers=num_workers,
        eval_batch_size=eval_batch_size,
        **overrides,
    )
    return config
