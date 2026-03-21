#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
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
    rows = []
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
            rows.extend(list(reader))

    if headers is None:
        raise ValueError(f"No rows to concat for {output_csv}")

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    return output_path.as_posix()


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


def command_prepare_data(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    dataset_type = config["dataset"].get("dataset_type", "voicebank")
    if dataset_type == "voicebank":
        summary = _prepare_voicebank_dataset(config, force=force)
    elif dataset_type == "dns5":
        summary = _prepare_dns5_dataset(config, force=force)
    elif dataset_type == "combined":
        summary = _prepare_combined_dataset(config, force=force)
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

    teacher_model, _ = load_model_from_checkpoint(
        config["paths"]["teacher_source_checkpoint"],
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
        "teacher_source_checkpoint": config["paths"]["teacher_source_checkpoint"],
        "quantized_teacher": True,
    }
    write_json(Path(config["paths"]["output_root"]) / "teacher_cache" / "summary.json", summary)
    return summary


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
