from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sat_app.config import PROJECTS_DIR


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text or f"item-{uuid4().hex[:8]}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class ProjectRecord:
    project_id: str
    slug: str
    name: str
    description: str = ""
    legacy_sat: bool = False
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)


@dataclass
class CheckpointRecord:
    checkpoint_id: str
    project_id: str
    slug: str
    name: str
    description: str = ""
    dataset_ref: str = ""
    id_column: str = ""
    name_column: str = ""
    target_column: str = ""
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)


@dataclass
class ModelRecord:
    model_id: str
    project_id: str
    checkpoint_id: str
    version: int
    label: str
    algorithm: str
    objective_metric: str
    threshold: float
    metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    search_config: dict[str, Any]
    validation_config: dict[str, Any]
    dataset_ref: str
    artifact_path: str
    is_active: bool = False
    created_at: str = field(default_factory=now_iso)


@dataclass
class PredictionRunRecord:
    prediction_id: str
    project_id: str
    checkpoint_id: str
    model_id: str
    dataset_ref: str
    columns_validated: list[str]
    row_count: int
    export_refs: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)


def project_dir(slug: str) -> Path:
    return PROJECTS_DIR / slug


def project_metadata_path(slug: str) -> Path:
    return project_dir(slug) / "project.json"


def checkpoints_dir(slug: str) -> Path:
    return project_dir(slug) / "checkpoints"


def checkpoint_dir(project_slug: str, checkpoint_slug: str) -> Path:
    return checkpoints_dir(project_slug) / checkpoint_slug


def checkpoint_metadata_path(project_slug: str, checkpoint_slug: str) -> Path:
    return checkpoint_dir(project_slug, checkpoint_slug) / "checkpoint.json"


def models_dir(project_slug: str, checkpoint_slug: str) -> Path:
    return checkpoint_dir(project_slug, checkpoint_slug) / "models"


def model_dir(project_slug: str, checkpoint_slug: str, model_id: str) -> Path:
    return models_dir(project_slug, checkpoint_slug) / model_id


def predictions_dir(project_slug: str, checkpoint_slug: str) -> Path:
    return checkpoint_dir(project_slug, checkpoint_slug) / "predictions"


def create_project(name: str, description: str = "", legacy_sat: bool = False, slug: str | None = None) -> ProjectRecord:
    project_slug = slugify(slug or name)
    path = project_metadata_path(project_slug)
    if path.exists():
        data = _read_json(path)
        return ProjectRecord(**data)
    record = ProjectRecord(
        project_id=uuid4().hex,
        slug=project_slug,
        name=name.strip(),
        description=description.strip(),
        legacy_sat=legacy_sat,
    )
    save_project(record)
    return record


def save_project(record: ProjectRecord) -> None:
    record.updated_at = now_iso()
    _write_json(project_metadata_path(record.slug), asdict(record))


def list_projects() -> list[ProjectRecord]:
    items: list[ProjectRecord] = []
    for path in sorted(PROJECTS_DIR.glob("*/project.json")):
        items.append(ProjectRecord(**_read_json(path)))
    return items


def get_project(project_slug: str) -> ProjectRecord:
    return ProjectRecord(**_read_json(project_metadata_path(project_slug)))


def create_checkpoint(
    project_slug: str,
    name: str,
    description: str = "",
    dataset_ref: str = "",
    id_column: str = "",
    name_column: str = "",
    target_column: str = "",
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    slug: str | None = None,
) -> CheckpointRecord:
    project = get_project(project_slug)
    checkpoint_slug = slugify(slug or name)
    path = checkpoint_metadata_path(project_slug, checkpoint_slug)
    if path.exists():
        data = _read_json(path)
        return CheckpointRecord(**data)
    record = CheckpointRecord(
        checkpoint_id=uuid4().hex,
        project_id=project.project_id,
        slug=checkpoint_slug,
        name=name.strip(),
        description=description.strip(),
        dataset_ref=dataset_ref,
        id_column=id_column,
        name_column=name_column,
        target_column=target_column,
        numeric_features=list(numeric_features or []),
        categorical_features=list(categorical_features or []),
    )
    save_checkpoint(project_slug, record)
    return record


def save_checkpoint(project_slug: str, record: CheckpointRecord) -> None:
    record.updated_at = now_iso()
    _write_json(checkpoint_metadata_path(project_slug, record.slug), asdict(record))


def list_checkpoints(project_slug: str) -> list[CheckpointRecord]:
    items: list[CheckpointRecord] = []
    for path in sorted(checkpoints_dir(project_slug).glob("*/checkpoint.json")):
        items.append(CheckpointRecord(**_read_json(path)))
    return items


def get_checkpoint(project_slug: str, checkpoint_slug: str) -> CheckpointRecord:
    return CheckpointRecord(**_read_json(checkpoint_metadata_path(project_slug, checkpoint_slug)))


def next_model_version(project_slug: str, checkpoint_slug: str) -> int:
    versions = [item.version for item in list_models(project_slug, checkpoint_slug)]
    return max(versions, default=0) + 1


def save_model_record(project_slug: str, checkpoint_slug: str, record: ModelRecord) -> None:
    path = model_dir(project_slug, checkpoint_slug, record.model_id) / "metadata.json"
    _write_json(path, asdict(record))


def list_models(project_slug: str, checkpoint_slug: str) -> list[ModelRecord]:
    items: list[ModelRecord] = []
    for path in sorted(models_dir(project_slug, checkpoint_slug).glob("*/metadata.json")):
        items.append(ModelRecord(**_read_json(path)))
    return sorted(items, key=lambda item: (item.version, item.created_at))


def get_model_record(project_slug: str, checkpoint_slug: str, model_id: str) -> ModelRecord:
    path = model_dir(project_slug, checkpoint_slug, model_id) / "metadata.json"
    return ModelRecord(**_read_json(path))


def activate_model(project_slug: str, checkpoint_slug: str, model_id: str) -> ModelRecord:
    active: ModelRecord | None = None
    for item in list_models(project_slug, checkpoint_slug):
        item.is_active = item.model_id == model_id
        save_model_record(project_slug, checkpoint_slug, item)
        if item.is_active:
            active = item
    if active is None:
        raise FileNotFoundError(f"Model {model_id} not found")
    return active


def get_active_model(project_slug: str, checkpoint_slug: str) -> ModelRecord | None:
    for item in reversed(list_models(project_slug, checkpoint_slug)):
        if item.is_active:
            return item
    return None


def save_prediction_record(project_slug: str, checkpoint_slug: str, record: PredictionRunRecord) -> None:
    path = predictions_dir(project_slug, checkpoint_slug) / record.prediction_id / "prediction.json"
    _write_json(path, asdict(record))


def get_prediction_run(project_slug: str, checkpoint_slug: str, prediction_id: str) -> PredictionRunRecord:
    path = predictions_dir(project_slug, checkpoint_slug) / prediction_id / "prediction.json"
    return PredictionRunRecord(**_read_json(path))


def list_prediction_runs(project_slug: str, checkpoint_slug: str) -> list[PredictionRunRecord]:
    items: list[PredictionRunRecord] = []
    for path in sorted(predictions_dir(project_slug, checkpoint_slug).glob("*/prediction.json")):
        items.append(PredictionRunRecord(**_read_json(path)))
    return sorted(items, key=lambda item: item.created_at)
