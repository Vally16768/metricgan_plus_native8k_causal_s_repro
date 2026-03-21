from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_metric_history(reference_export: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, run in reference_export.get("runs", {}).items():
        for key, history in run.get("metric_history", {}).items():
            for point in history:
                rows.append(
                    {
                        "run_name": run_name,
                        "run_id": run["run_id"],
                        "metric": key,
                        "step": point.get("step", 0),
                        "timestamp": point.get("timestamp"),
                        "value": point.get("value"),
                    }
                )
    return rows


def _plot_metric_series(ax: Any, history: dict[str, list[dict[str, Any]]], keys: list[str], title: str) -> None:
    plotted = False
    for key in keys:
        series = history.get(key, [])
        if not series:
            continue
        x = [point.get("step", 0) for point in series]
        y = [point.get("value", 0.0) for point in series]
        ax.plot(x, y, marker="o", linewidth=1.5, markersize=3, label=key)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("Step/Epoch")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")


def plot_training_curves(final_run: dict[str, Any], out_path: str | Path) -> None:
    history = final_run.get("metric_history", {})
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    axes = axes.flatten()
    groups = [
        (["train/loss"], "Train Loss"),
        (["train/wave_loss", "train/spectral_loss"], "Wave + Spectral"),
        (["train/sisdr_loss"], "SI-SDR Loss"),
        (["train/teacher_mask_loss", "train/teacher_wave_loss"], "Teacher Alignment"),
        (["train/noise_gate_loss", "train/speech_preserve_loss"], "Auxiliary Losses"),
        (["lr"], "Learning Rate"),
        (["val_rank/pesq_mean", "val_rank/stoi_mean"], "Val Rank PESQ/STOI"),
        (["val_rank/sisdr_mean", "val_rank/delta_snr_mean"], "Val Rank SI-SDR/Delta SNR"),
    ]
    for ax, (keys, title) in zip(axes, groups, strict=True):
        _plot_metric_series(ax, history, keys, title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_stage1_comparison(stage1_runs: list[dict[str, Any]], out_path: str | Path) -> None:
    names = [run["run_name"] for run in stage1_runs]
    pesq = [run["latest_metrics"].get("best/val_select_pesq_mean", 0.0) for run in stage1_runs]
    stoi = [run["latest_metrics"].get("best/val_select_stoi_mean", 0.0) for run in stage1_runs]
    latency = [run["latest_metrics"].get("best/inference_seconds_10s", 0.0) for run in stage1_runs]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].bar(names, pesq, color="#0f766e")
    axes[0].set_title("Stage1 Best Val-Select PESQ")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(names, stoi, color="#1d4ed8")
    axes[1].set_title("Stage1 Best Val-Select STOI")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(names, latency, color="#b45309")
    axes[2].set_title("Stage1 Benchmark Latency For 10s")
    axes[2].tick_params(axis="x", labelrotation=25)
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_deployability_profiles(profile_rollup: dict[str, Any], out_path: str | Path) -> None:
    profiles = profile_rollup.get("profiles", {})
    names = list(profiles.keys())
    recommended_mhz = [profiles[name].get("recommended_rt_mhz", 0.0) for name in names]
    power_mw = [profiles[name].get("avg_power_mw_at_recommended_mhz", profiles[name].get("avg_power_mw", 0.0)) for name in names]
    deployable = [1.0 if profiles[name].get("deployment_ok") else 0.0 for name in names]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].bar(names, recommended_mhz, color="#475569")
    axes[0].set_title("Recommended RT MHz")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(names, power_mw, color="#7c3aed")
    axes[1].set_title("Estimated Power At Recommended RT MHz")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(names, deployable, color="#15803d")
    axes[2].set_title("Deployability Flag")
    axes[2].tick_params(axis="x", labelrotation=25)
    axes[2].set_yticks([0.0, 1.0], labels=["No", "Yes"])
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _find_first_sample_triplet(sample_root: Path) -> tuple[Path, Path, Path] | None:
    for enhanced in sorted(sample_root.rglob("*_enh.wav")):
        stem = enhanced.name[:-8]
        noisy = enhanced.with_name(f"{stem}_noisy.wav")
        clean = enhanced.with_name(f"{stem}_clean.wav")
        if noisy.exists() and clean.exists():
            return noisy, clean, enhanced
    return None


def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path.as_posix())
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0), sr


def _spec_db(wav: torch.Tensor, n_fft: int = 256, hop_length: int = 80, win_length: int = 160) -> np.ndarray:
    window = torch.hann_window(win_length)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = spec.abs().clamp_min(1e-6)
    return (20.0 * torch.log10(mag)).cpu().numpy()


def render_sample_figures(sample_root: str | Path, waveform_png: str | Path, spectrogram_png: str | Path) -> dict[str, str] | None:
    triplet = _find_first_sample_triplet(Path(sample_root))
    if triplet is None:
        return None
    noisy_path, clean_path, enhanced_path = triplet
    noisy, sr = _load_audio(noisy_path)
    clean, _ = _load_audio(clean_path)
    enhanced, _ = _load_audio(enhanced_path)
    length = min(noisy.numel(), clean.numel(), enhanced.numel())
    noisy = noisy[:length]
    clean = clean[:length]
    enhanced = enhanced[:length]
    time_axis = np.arange(length) / float(sr)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(time_axis, noisy.numpy(), color="#dc2626", linewidth=0.8)
    axes[0].set_title("Noisy")
    axes[1].plot(time_axis, clean.numpy(), color="#16a34a", linewidth=0.8)
    axes[1].set_title("Clean Reference")
    axes[2].plot(time_axis, enhanced.numpy(), color="#2563eb", linewidth=0.8)
    axes[2].set_title("Enhanced")
    axes[2].set_xlabel("Seconds")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    Path(waveform_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(waveform_png, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, wav, title in zip(axes, [noisy, clean, enhanced], ["Noisy", "Clean Reference", "Enhanced"], strict=True):
        ax.imshow(_spec_db(wav), origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_ylabel("Bin")
    axes[-1].set_xlabel("Frame")
    fig.tight_layout()
    Path(spectrogram_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(spectrogram_png, dpi=180)
    plt.close(fig)

    return {
        "noisy": noisy_path.as_posix(),
        "clean": clean_path.as_posix(),
        "enhanced": enhanced_path.as_posix(),
    }


def generate_report(
    *,
    report_dir: str | Path,
    config: dict[str, Any],
    reference_export: dict[str, Any],
    evaluation_dir: str | Path,
) -> dict[str, Any]:
    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)

    final_run_name = config["reference"]["final_qat_run_name"]
    stage1_names = list(config["reference"]["stage1_run_names"])
    final_run = reference_export["runs"][final_run_name]
    stage1_runs = [reference_export["runs"][name] for name in stage1_names]
    evaluation_root = Path(evaluation_dir)
    evaluation_summary = read_json(evaluation_root / "summary.json")
    metric_history_rows = flatten_metric_history(reference_export)

    write_csv(
        report_root / "metric_history.csv",
        metric_history_rows,
        ["run_name", "run_id", "metric", "step", "timestamp", "value"],
    )

    canonical_rows: list[dict[str, Any]] = []
    final_metrics = final_run.get("latest_metrics", {})
    for metric_key in [
        "train/loss",
        "train/wave_loss",
        "train/spectral_loss",
        "train/sisdr_loss",
        "train/teacher_mask_loss",
        "train/teacher_wave_loss",
        "lr",
        "val_rank/pesq_mean",
        "val_rank/stoi_mean",
        "val_rank/sisdr_mean",
        "val_rank/delta_snr_mean",
        "best/val_select_pesq_mean",
        "best/val_select_stoi_mean",
        "best/val_select_sisdr_mean",
        "best/val_select_delta_snr_mean",
        "best/inference_seconds_10s",
    ]:
        value = final_metrics.get(metric_key)
        if value is None:
            history = final_run.get("metric_history", {}).get(metric_key, [])
            value = history[-1]["value"] if history else None
        canonical_rows.append({"metric": metric_key, "value": value, "source": "reference_run"})

    for metric_key, value in evaluation_summary.get("test_metrics", {}).items():
        if metric_key == "sample_paths":
            continue
        canonical_rows.append({"metric": f"test/{metric_key}", "value": value, "source": "evaluate"})

    rollup = evaluation_summary.get("mcu_rollup", {})
    canonical_rows.extend(
        [
            {"metric": "deployability/best_profile_name", "value": rollup.get("best_profile_name"), "source": "evaluate"},
            {"metric": "deployability/best_power_profile_name", "value": rollup.get("best_power_profile_name"), "source": "evaluate"},
            {"metric": "deployability/best_power_profile_avg_power_mw", "value": rollup.get("best_power_profile_avg_power_mw"), "source": "evaluate"},
            {"metric": "deployability/supported_profile_count", "value": rollup.get("supported_profile_count"), "source": "evaluate"},
            {"metric": "deployability/power_supported_profile_count", "value": rollup.get("power_supported_profile_count"), "source": "evaluate"},
        ]
    )
    write_csv(report_root / "canonical_metrics.csv", canonical_rows, ["metric", "value", "source"])

    plot_training_curves(final_run, report_root / "training_curves.png")
    plot_stage1_comparison(stage1_runs, report_root / "stage1_comparison.png")
    plot_deployability_profiles(rollup, report_root / "deployability_profiles.png")

    sample_audio_dir = report_root / "audio_samples"
    sample_audio_dir.mkdir(parents=True, exist_ok=True)
    sample_triplet = render_sample_figures(
        evaluation_root / "samples",
        report_root / "sample_waveforms.png",
        report_root / "sample_spectrograms.png",
    )
    if sample_triplet is not None:
        for key, source in sample_triplet.items():
            shutil.copy2(source, sample_audio_dir / Path(source).name)

    summary = {
        "model": final_run_name,
        "reference_run_id": final_run["run_id"],
        "stage1_winner": reference_export.get("lineage", {}).get("stage1_winner"),
        "best_val_select_pesq": final_metrics.get("best/val_select_pesq_mean"),
        "test_metrics": evaluation_summary.get("test_metrics", {}),
        "benchmark_latency_10s": evaluation_summary.get("benchmark_latency_10s"),
        "mcu_rollup": rollup,
        "sample_triplet": sample_triplet,
    }
    write_json(report_root / "report_summary.json", summary)

    report_md = report_root / "report.md"
    report_md.write_text(
        "\n".join(
            [
                "# MetricGAN+ native8k causal_s repro report",
                "",
                f"- Model final: `{final_run_name}`",
                f"- Run de referinta: `{final_run['run_id']}`",
                f"- Castigator stage1: `{reference_export.get('lineage', {}).get('stage1_winner')}`",
                f"- Best val_select PESQ: `{final_metrics.get('best/val_select_pesq_mean')}`",
                f"- Test PESQ: `{evaluation_summary.get('test_metrics', {}).get('pesq_mean')}`",
                f"- Test STOI: `{evaluation_summary.get('test_metrics', {}).get('stoi_mean')}`",
                f"- Test SI-SDR: `{evaluation_summary.get('test_metrics', {}).get('sisdr_mean')}`",
                f"- Test Delta SNR: `{evaluation_summary.get('test_metrics', {}).get('delta_snr_mean')}`",
                f"- Benchmark latency 10s: `{evaluation_summary.get('benchmark_latency_10s')}`",
                f"- Deployable profiles: `{rollup.get('supported_profiles', [])}`",
                f"- Profiles <50mW: `{rollup.get('power_supported_profiles', [])}`",
                "",
                "## Artefacte",
                "",
                "- `canonical_metrics.csv`: sumar metrici canonice si de evaluare",
                "- `metric_history.csv`: istoric complet extras din run-urile originale",
                "- `training_curves.png`: curbele de antrenare/QAT",
                "- `stage1_comparison.png`: comparatie intre candidatii stage1",
                "- `deployability_profiles.png`: sumar simulare MCU",
                "- `sample_waveforms.png` si `sample_spectrograms.png`: audit pe sample audio",
                "",
                "## Observatie",
                "",
                "DNSMOS si metricele necanonice raman optionale si nu sunt generate implicit in acest proiect.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary
