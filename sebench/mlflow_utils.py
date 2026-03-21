from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = (PROJECT_ROOT / "tracking").as_posix()
DEFAULT_ARTIFACT_ROOT = (PROJECT_ROOT / "tracking" / "artifacts").as_posix()
DEFAULT_EXPERIMENT_NAME = "metricgan_native8k_causal_s_repro"
DEFAULT_REGISTERED_MODEL = "metricgan-native8k-causal-s-repro"


def path_to_file_uri(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


def configure_mlflow(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_root)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def flatten_params(data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            flat[key] = "null"
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif isinstance(value, Path):
            flat[key] = value.as_posix()
        else:
            flat[key] = json.dumps(value, sort_keys=True)
    return flat


def log_dict_artifact(data: dict[str, Any], artifact_file: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / artifact_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        mlflow.log_artifact(out_path.as_posix(), artifact_path=str(Path(artifact_file).parent))


def find_finished_run(tracking_uri: str, experiment_name: str, run_name: str, phase: str | None = None) -> dict[str, Any] | None:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    client = MlflowClient()
    filter_parts = [f"attributes.run_name = '{run_name}'", "attributes.status = 'FINISHED'"]
    if phase:
        filter_parts.append(f"tags.phase = '{phase}'")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=" and ".join(filter_parts),
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    run = next((candidate for candidate in runs if candidate.data.tags.get("audit.invalidated") != "true"), None)
    if run is None:
        return None
    return {
        "run_id": run.info.run_id,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
    }


def count_runs_by_status(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    *,
    phase: str | None = None,
    statuses: tuple[str, ...] = ("FAILED",),
) -> int:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return 0

    client = MlflowClient()
    filter_parts = [f"attributes.run_name = '{run_name}'"]
    if phase:
        filter_parts.append(f"tags.phase = '{phase}'")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=" and ".join(filter_parts),
        max_results=200,
        order_by=["attributes.start_time DESC"],
    )
    return sum(1 for run in runs if run.info.status in statuses and run.data.tags.get("audit.invalidated") != "true")


def terminate_matching_runs(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    *,
    phase: str | None = None,
    statuses: tuple[str, ...] = ("RUNNING",),
    run_type: str | None = None,
) -> list[str]:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    client = MlflowClient()
    filter_parts = [f"attributes.run_name = '{run_name}'"]
    if phase:
        filter_parts.append(f"tags.phase = '{phase}'")
    if run_type:
        filter_parts.append(f"tags.run_type = '{run_type}'")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=" and ".join(filter_parts),
        max_results=50,
        order_by=["attributes.start_time DESC"],
    )

    terminated: list[str] = []
    for run in runs:
        if run.info.status in statuses:
            client.set_terminated(run.info.run_id, status="KILLED")
            terminated.append(run.info.run_id)
    return terminated


def register_run_model(
    tracking_uri: str,
    run_id: str,
    model_name: str = DEFAULT_REGISTERED_MODEL,
    alias: str = "best-pesq",
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass
    version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
    try:
        client.set_registered_model_alias(model_name, alias, version.version)
    except Exception:
        client.transition_model_version_stage(model_name, version.version, "Production", archive_existing_versions=False)
