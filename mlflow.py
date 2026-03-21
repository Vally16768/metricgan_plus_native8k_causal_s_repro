from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


_tracking_uri: str | None = None
_experiment_name: str | None = None
_active_run: "Run | None" = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _tracking_root() -> Path:
    if _tracking_uri is None:
        return Path.cwd() / "tracking"
    return Path(_tracking_uri).expanduser().resolve()


def _experiments_path() -> Path:
    return _tracking_root() / "experiments.json"


def _load_experiments() -> dict[str, dict[str, Any]]:
    path = _experiments_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_experiments(payload: dict[str, dict[str, Any]]) -> None:
    path = _experiments_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class Experiment:
    experiment_id: str
    name: str
    artifact_location: str


@dataclass
class RunInfo:
    run_id: str
    experiment_id: str
    artifact_uri: str
    run_name: str | None
    status: str
    start_time: int
    end_time: int | None = None


@dataclass
class RunData:
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class Run:
    info: RunInfo
    data: RunData


def _run_root(run_id: str) -> Path:
    return _tracking_root() / "runs" / run_id


def _artifact_root(run_id: str) -> Path:
    return _run_root(run_id) / "artifacts"


def _run_meta_path(run_id: str) -> Path:
    return _run_root(run_id) / "meta.json"


def _run_params_path(run_id: str) -> Path:
    return _run_root(run_id) / "params.json"


def _run_metrics_path(run_id: str) -> Path:
    return _run_root(run_id) / "latest_metrics.json"


def _run_metrics_history_path(run_id: str) -> Path:
    return _run_root(run_id) / "metrics_history.jsonl"


def _run_tags_path(run_id: str) -> Path:
    return _run_root(run_id) / "tags.json"


def _write_run_state(run: Run) -> None:
    root = _run_root(run.info.run_id)
    root.mkdir(parents=True, exist_ok=True)
    _artifact_root(run.info.run_id).mkdir(parents=True, exist_ok=True)
    _run_meta_path(run.info.run_id).write_text(
        json.dumps(
            {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "artifact_uri": run.info.artifact_uri,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _run_params_path(run.info.run_id).write_text(json.dumps(run.data.params, indent=2, sort_keys=True), encoding="utf-8")
    _run_metrics_path(run.info.run_id).write_text(json.dumps(run.data.metrics, indent=2, sort_keys=True), encoding="utf-8")
    _run_tags_path(run.info.run_id).write_text(json.dumps(run.data.tags, indent=2, sort_keys=True), encoding="utf-8")


def _load_run(run_id: str) -> Run:
    meta = json.loads(_run_meta_path(run_id).read_text(encoding="utf-8"))
    params = json.loads(_run_params_path(run_id).read_text(encoding="utf-8")) if _run_params_path(run_id).exists() else {}
    metrics = json.loads(_run_metrics_path(run_id).read_text(encoding="utf-8")) if _run_metrics_path(run_id).exists() else {}
    tags = json.loads(_run_tags_path(run_id).read_text(encoding="utf-8")) if _run_tags_path(run_id).exists() else {}
    return Run(
        info=RunInfo(
            run_id=meta["run_id"],
            experiment_id=meta["experiment_id"],
            artifact_uri=meta["artifact_uri"],
            run_name=meta.get("run_name"),
            status=meta["status"],
            start_time=meta["start_time"],
            end_time=meta.get("end_time"),
        ),
        data=RunData(params=params, metrics=metrics, tags=tags),
    )


def set_tracking_uri(uri: str) -> None:
    global _tracking_uri
    _tracking_uri = uri
    _tracking_root().mkdir(parents=True, exist_ok=True)


def get_experiment_by_name(name: str) -> Experiment | None:
    experiments = _load_experiments()
    if name not in experiments:
        return None
    payload = experiments[name]
    return Experiment(
        experiment_id=payload["experiment_id"],
        name=payload["name"],
        artifact_location=payload["artifact_location"],
    )


def create_experiment(name: str, artifact_location: str | None = None) -> str:
    experiments = _load_experiments()
    if name in experiments:
        return experiments[name]["experiment_id"]
    experiment_id = str(len(experiments) + 1)
    experiments[name] = {
        "experiment_id": experiment_id,
        "name": name,
        "artifact_location": artifact_location or (_tracking_root() / "artifacts").as_posix(),
    }
    _save_experiments(experiments)
    return experiment_id


def set_experiment(name: str) -> None:
    global _experiment_name
    _experiment_name = name


def active_run() -> Run | None:
    return _active_run


def start_run(
    *,
    run_name: str | None = None,
    experiment_id: str | None = None,
    nested: bool = False,
    tags: dict[str, str] | None = None,
    log_system_metrics: bool = False,
) -> Run:
    del nested, log_system_metrics
    global _active_run
    if experiment_id is None:
        if _experiment_name is None:
            raise RuntimeError("No active experiment set.")
        experiment = get_experiment_by_name(_experiment_name)
        if experiment is None:
            experiment_id = create_experiment(_experiment_name)
        else:
            experiment_id = experiment.experiment_id
    run_id = uuid.uuid4().hex
    artifact_uri = _artifact_root(run_id).as_posix()
    run = Run(
        info=RunInfo(
            run_id=run_id,
            experiment_id=experiment_id,
            artifact_uri=artifact_uri,
            run_name=run_name,
            status="RUNNING",
            start_time=_now_ms(),
        ),
        data=RunData(
            params={},
            metrics={},
            tags={
                **(tags or {}),
                "mlflow.runName": run_name or run_id,
            },
        ),
    )
    _write_run_state(run)
    _active_run = run
    return run


def _require_active_run() -> Run:
    if _active_run is None:
        raise RuntimeError("No active run.")
    return _active_run


def log_params(params: dict[str, Any]) -> None:
    run = _require_active_run()
    run.data.params.update(params)
    _write_run_state(run)


def _append_metric(run_id: str, key: str, value: float, step: int) -> None:
    with _run_metrics_history_path(run_id).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "key": key,
                    "value": float(value),
                    "step": int(step),
                    "timestamp": _now_ms(),
                },
                sort_keys=True,
            )
        )
        handle.write("\n")


def log_metric(key: str, value: float, step: int = 0) -> None:
    run = _require_active_run()
    run.data.metrics[key] = float(value)
    _append_metric(run.info.run_id, key, float(value), step)
    _write_run_state(run)


def log_metrics(metrics: dict[str, float], step: int = 0) -> None:
    for key, value in metrics.items():
        log_metric(key, float(value), step=step)


def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    run = _require_active_run()
    source = Path(local_path)
    target_dir = _artifact_root(run.info.run_id)
    if artifact_path:
        target_dir = target_dir / artifact_path
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_dir / source.name)


def log_artifacts(local_dir: str, artifact_path: str | None = None) -> None:
    run = _require_active_run()
    source = Path(local_dir)
    target_dir = _artifact_root(run.info.run_id)
    if artifact_path:
        target_dir = target_dir / artifact_path
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        out_path = target_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_path)


class _PyTorchLogger:
    def log_model(self, model: torch.nn.Module, artifact_path: str = "model") -> None:
        run = _require_active_run()
        target_dir = _artifact_root(run.info.run_id) / artifact_path
        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), target_dir / "state_dict.pt")


pytorch = _PyTorchLogger()


def end_run(status: str = "FINISHED") -> None:
    global _active_run
    run = _require_active_run()
    run.info.status = status
    run.info.end_time = _now_ms()
    _write_run_state(run)
    _active_run = None


def register_model(model_uri: str, name: str) -> Any:
    del model_uri, name
    return SimpleNamespace(version="1")


class MlflowClient:
    def search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str = "",
        max_results: int = 100,
        order_by: list[str] | None = None,
    ) -> list[Run]:
        del order_by
        runs_root = _tracking_root() / "runs"
        if not runs_root.exists():
            return []
        runs = [_load_run(path.name) for path in runs_root.iterdir() if path.is_dir()]
        runs = [run for run in runs if run.info.experiment_id in experiment_ids]
        for clause in [part.strip() for part in filter_string.split(" and ") if part.strip()]:
            key, value = clause.split(" = ", 1)
            value = value.strip().strip("'")
            if key == "attributes.run_name":
                runs = [run for run in runs if (run.info.run_name or "") == value]
            elif key == "attributes.status":
                runs = [run for run in runs if run.info.status == value]
            elif key.startswith("tags."):
                tag_key = key[5:]
                runs = [run for run in runs if run.data.tags.get(tag_key) == value]
        runs.sort(key=lambda run: run.info.start_time, reverse=True)
        return runs[:max_results]

    def set_terminated(self, run_id: str, status: str = "FINISHED") -> None:
        run = _load_run(run_id)
        run.info.status = status
        run.info.end_time = _now_ms()
        _write_run_state(run)

    def create_registered_model(self, name: str) -> None:
        del name

    def set_registered_model_alias(self, model_name: str, alias: str, version: str) -> None:
        del model_name, alias, version

    def transition_model_version_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False,
    ) -> None:
        del model_name, version, stage, archive_existing_versions
