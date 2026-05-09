from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from sat_app.registry import (
    ModelRecord,
    PredictionRunRecord,
    activate_model,
    get_active_model,
    get_model_record,
    model_dir,
    save_model_record,
    save_prediction_record,
)


def artifact_path_for(project_slug: str, checkpoint_slug: str, model_id: str) -> Path:
    return model_dir(project_slug, checkpoint_slug, model_id) / "model.joblib"


def save_artifact(project_slug: str, checkpoint_slug: str, model_id: str, artifact: dict[str, Any]) -> Path:
    path = artifact_path_for(project_slug, checkpoint_slug, model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(project_slug: str, checkpoint_slug: str, model_id: str) -> dict[str, Any]:
    return joblib.load(artifact_path_for(project_slug, checkpoint_slug, model_id))


def persist_model(
    project_slug: str,
    checkpoint_slug: str,
    record: ModelRecord,
    artifact: dict[str, Any],
    activate: bool = True,
) -> ModelRecord:
    artifact_path = save_artifact(project_slug, checkpoint_slug, record.model_id, artifact)
    record.artifact_path = str(artifact_path)
    save_model_record(project_slug, checkpoint_slug, record)
    if activate:
        return activate_model(project_slug, checkpoint_slug, record.model_id)
    return record


def load_active_artifact(project_slug: str, checkpoint_slug: str) -> tuple[ModelRecord, dict[str, Any]]:
    record = get_active_model(project_slug, checkpoint_slug)
    if record is None:
        raise FileNotFoundError("No active model found for checkpoint")
    return record, load_artifact(project_slug, checkpoint_slug, record.model_id)


def load_model_with_artifact(project_slug: str, checkpoint_slug: str, model_id: str) -> tuple[ModelRecord, dict[str, Any]]:
    record = get_model_record(project_slug, checkpoint_slug, model_id)
    return record, load_artifact(project_slug, checkpoint_slug, model_id)


def persist_prediction_record(project_slug: str, checkpoint_slug: str, record: PredictionRunRecord) -> None:
    save_prediction_record(project_slug, checkpoint_slug, record)
