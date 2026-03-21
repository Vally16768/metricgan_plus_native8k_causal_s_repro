#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _query_kv(cur: sqlite3.Cursor, table: str, run_uuid: str) -> dict[str, Any]:
    rows = cur.execute(f"select key, value from {table} where run_uuid=?", (run_uuid,)).fetchall()
    return {key: value for key, value in rows}


def _query_latest_metrics(cur: sqlite3.Cursor, run_uuid: str) -> dict[str, float]:
    rows = cur.execute("select key, value from latest_metrics where run_uuid=?", (run_uuid,)).fetchall()
    return {key: float(value) for key, value in rows}


def _query_metric_history(cur: sqlite3.Cursor, run_uuid: str) -> dict[str, list[dict[str, Any]]]:
    rows = cur.execute(
        "select key, value, step, timestamp from metrics where run_uuid=? order by key, step, timestamp",
        (run_uuid,),
    ).fetchall()
    history: dict[str, list[dict[str, Any]]] = {}
    for key, value, step, timestamp in rows:
        history.setdefault(key, []).append(
            {
                "step": int(step),
                "timestamp": int(timestamp),
                "value": float(value),
            }
        )
    return history


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _export_run(conn: sqlite3.Connection, run_name: str, target_root: Path) -> dict[str, Any]:
    cur = conn.cursor()
    row = cur.execute(
        "select run_uuid, name, status, start_time, end_time, artifact_uri from runs where name=? order by start_time desc limit 1",
        (run_name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Run not found: {run_name}")
    run_uuid, name, status, start_time, end_time, artifact_uri = row
    artifact_path = Path(str(artifact_uri).replace("file://", ""))
    local_artifacts = target_root / "runs" / run_uuid / "artifacts"
    if artifact_path.exists():
        _copy_tree(artifact_path, local_artifacts)

    reports_dir = local_artifacts / "reports"
    artifact_reports: dict[str, Any] = {}
    for report_name in [
        "run_summary.json",
        "best_val_select_metrics.json",
        "test_metrics.json",
        "stm32sim.json",
        "mcu_shortlist.json",
    ]:
        path = reports_dir / report_name
        if path.exists():
            artifact_reports[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    return {
        "run_id": run_uuid,
        "run_name": name,
        "status": status,
        "start_time": int(start_time),
        "end_time": int(end_time) if end_time is not None else None,
        "artifact_uri": artifact_uri,
        "local_artifact_dir": local_artifacts.as_posix() if local_artifacts.exists() else None,
        "params": _query_kv(cur, "params", run_uuid),
        "tags": _query_kv(cur, "tags", run_uuid),
        "latest_metrics": _query_latest_metrics(cur, run_uuid),
        "metric_history": _query_metric_history(cur, run_uuid),
        "artifact_reports": artifact_reports,
    }


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    reference_root = Path(config["paths"]["reference_root"])
    reference_root.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(config["paths"]["source_mlflow_db"])
    run_names = list(config["reference"]["stage1_run_names"]) + [config["reference"]["final_qat_run_name"]]
    runs: dict[str, Any] = {}
    for run_name in run_names:
        runs[run_name] = _export_run(conn, str(run_name), reference_root)
    conn.close()

    checkpoints_dir = reference_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "metricgan_plus_native8k_small.pt": config["paths"]["source_teacher_source_checkpoint"],
        "metricgan_plus_native8k_causal_s_seed0.pt": config["paths"]["source_stage1_s_seed0_checkpoint"],
        "metricgan_plus_native8k_causal_s_seed1.pt": config["paths"]["source_stage1_s_seed1_checkpoint"],
        "metricgan_plus_native8k_causal_xs_seed0.pt": config["paths"]["source_stage1_xs_seed0_checkpoint"],
        "metricgan_plus_native8k_causal_xs_seed1.pt": config["paths"]["source_stage1_xs_seed1_checkpoint"],
        "metricgan_plus_native8k_causal_s_qat.pt": config["paths"]["source_final_checkpoint"],
    }
    for out_name, src in copies.items():
        shutil.copy2(src, checkpoints_dir / out_name)

    split_summary_src = Path(config["dataset"]["campaign_dir_8k"]) / "split_summary.json"
    if split_summary_src.exists():
        dataset_ref = reference_root / "dataset"
        dataset_ref.mkdir(parents=True, exist_ok=True)
        shutil.copy2(split_summary_src, dataset_ref / "split_summary.json")

    winner = max(
        [runs[name] for name in config["reference"]["stage1_run_names"]],
        key=lambda item: float(item["latest_metrics"].get("best/val_select_pesq_mean", float("-inf"))),
    )
    payload = {
        "lineage": {
            "stage1_run_names": list(config["reference"]["stage1_run_names"]),
            "stage1_winner": winner["run_name"],
            "final_qat_run_name": config["reference"]["final_qat_run_name"],
        },
        "runs": runs,
    }
    out_path = Path(config["paths"]["reference_export_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"export": out_path.as_posix(), "winner": winner["run_name"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
