from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sat_app.config import DERIVED_DEPENDENCIES, LEGACY_SAT_CHECKPOINTS, LEGACY_SAT_PROJECT_NAME, LEGACY_SAT_PROJECT_SLUG
from sat_app.model_store import persist_model
from sat_app.registry import ModelRecord, create_checkpoint, create_project, get_active_model, next_model_version, save_project


def expand_required_columns(features: list[str]) -> list[str]:
    required: list[str] = []
    for feature in features:
        deps = DERIVED_DEPENDENCIES.get(feature, [feature])
        for dep in deps:
            if dep not in required:
                required.append(dep)
    return required


def compute_legacy_sat_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "era" in work.columns:
        normalized_era = work["era"].astype(str).str.lower().str.strip()
        work["era_encoded"] = normalized_era.eq("formato_nuevo").astype(int)
    if "engagement_hasta_p1" in work.columns:
        work["log_eng_p1"] = np.log1p(pd.to_numeric(work["engagement_hasta_p1"], errors="coerce").fillna(0))
    if "engagement_hasta_p2" in work.columns:
        work["log_eng_p2"] = np.log1p(pd.to_numeric(work["engagement_hasta_p2"], errors="coerce").fillna(0))
    if "parcial_1" in work.columns:
        parcial_1 = pd.to_numeric(work["parcial_1"], errors="coerce")
        media = parcial_1.mean()
        std = parcial_1.std() + 1e-6
        work["p1_vs_media"] = (parcial_1 - media) / std
    if {"parcial_1", "parcial_2"}.issubset(work.columns):
        p1 = pd.to_numeric(work["parcial_1"], errors="coerce")
        p2 = pd.to_numeric(work["parcial_2"], errors="coerce")
        work["delta_p2_p1"] = p2 - p1
    if {"engagement_hasta_p1", "parcial_1_visitas"}.issubset(work.columns):
        eng = pd.to_numeric(work["engagement_hasta_p1"], errors="coerce").fillna(0)
        visitas = pd.to_numeric(work["parcial_1_visitas"], errors="coerce").fillna(0)
        work["intensidad_p1"] = np.where(visitas > 0, eng * 60 / visitas, 0.0)
    if {"parcial_1_visitas", "parcial_1_temas_unicos"}.issubset(work.columns):
        visitas = pd.to_numeric(work["parcial_1_visitas"], errors="coerce").fillna(0)
        temas = pd.to_numeric(work["parcial_1_temas_unicos"], errors="coerce").fillna(0)
        work["ratio_vt_p1"] = np.where(temas > 0, visitas / temas, 0.0)
    return work


def _legacy_artifact(bundle: dict[str, Any], metadata: dict[str, Any], project_id: str, checkpoint_id: str, model_id: str, version: int) -> dict[str, Any]:
    features = list(bundle["features"])
    return {
        "model": bundle["modelo"],
        "preprocessor": bundle["scaler"],
        "feature_schema": {
            "id_column": "id_estudiante",
            "name_column": "nombre",
            "target_column": "reprobo",
            "numeric_features": list(features),
            "categorical_features": [],
            "feature_order": list(features),
            "transformed_feature_names": list(features),
            "required_input_columns": expand_required_columns(features),
        },
        "training_summary": {
            "algorithm": bundle.get("familia", metadata.get("familia", "legacy")),
            "objective_metric": metadata.get("criterio", "f1").lower(),
            "threshold_strategy": "legacy_metadata",
            "selected_threshold": float(bundle.get("umbral", metadata.get("umbral", 0.5))),
            "validation_mode": "legacy_loso",
            "balancing": bundle.get("balanceo", metadata.get("balanceo", "none")),
            "hyperparameters": bundle.get("params", metadata.get("params", {})),
            "search_space": {},
            "hyperparameter_mode": "legacy_import",
            "search_iterations": 0,
            "group_column": "semestre",
            "holdout_size": 0.0,
            "random_state": 42,
        },
        "metrics": metadata.get("metricas_loso", bundle.get("metricas_loso", {})),
        "project_ref": {
            "project_id": project_id,
            "checkpoint_id": checkpoint_id,
            "model_id": model_id,
            "version": version,
        },
        "dataset_ref": str(metadata.get("cp", "")),
        "created_at": bundle.get("fecha_entrenamiento", metadata.get("fecha_entrenamiento", "")),
        "legacy": {
            "legacy_sat": True,
            "raw_features": list(features),
            "required_input_columns": expand_required_columns(features),
            "impute_zeros": list(bundle.get("impute_zeros", metadata.get("impute_zeros", []))),
            "comparison_image": metadata.get("comparison_image", ""),
            "threshold_image": metadata.get("threshold_image", ""),
        },
    }


def ensure_legacy_sat_project() -> None:
    project = create_project(LEGACY_SAT_PROJECT_NAME, "Proyecto migrado desde el SAT histórico.", legacy_sat=True, slug=LEGACY_SAT_PROJECT_SLUG)
    project.name = LEGACY_SAT_PROJECT_NAME
    project.description = "Proyecto base migrado desde los modelos historicos del curso Modelos Probabilisticos."
    project.legacy_sat = True
    save_project(project)
    for checkpoint_key, info in LEGACY_SAT_CHECKPOINTS.items():
        checkpoint = create_checkpoint(
            LEGACY_SAT_PROJECT_SLUG,
            name=info["label"],
            description=info.get("description", ""),
            dataset_ref=str(info["metadata_file"]),
            id_column="id_estudiante",
            name_column="nombre",
            target_column="reprobo",
            numeric_features=[],
            categorical_features=[],
            slug=checkpoint_key,
        )
        if get_active_model(LEGACY_SAT_PROJECT_SLUG, checkpoint.slug) is not None:
            continue
        metadata_all = json.loads(Path(info["metadata_file"]).read_text(encoding="utf-8"))
        for criterion, bundle_path in info["bundle_files"].items():
            bundle = joblib.load(bundle_path)
            metadata = dict(metadata_all[criterion])
            metadata["comparison_image"] = str(info["comparison_image"])
            metadata["threshold_image"] = str(info["threshold_image"])
            model_id = f"{checkpoint_key}-{criterion}"
            version = next_model_version(LEGACY_SAT_PROJECT_SLUG, checkpoint.slug)
            artifact = _legacy_artifact(bundle, metadata, project.project_id, checkpoint.checkpoint_id, model_id, version)
            record = ModelRecord(
                model_id=model_id,
                project_id=project.project_id,
                checkpoint_id=checkpoint.checkpoint_id,
                version=version,
                label=f"{info['label']} · {criterion.upper()}",
                algorithm=str(metadata.get("familia", bundle.get("familia", "legacy"))),
                objective_metric=str(metadata.get("criterio", criterion)).lower(),
                threshold=float(metadata.get("umbral", bundle.get("umbral", 0.5))),
                metrics=dict(metadata.get("metricas_loso", bundle.get("metricas_loso", {}))),
                hyperparameters=dict(bundle.get("params", metadata.get("params", {}))),
                search_config={"mode": "legacy_import", "space": {}, "iterations": 0},
                validation_config={"mode": "legacy_loso", "group_column": "semestre"},
                dataset_ref=str(info["metadata_file"]),
                artifact_path="",
                is_active=(criterion == "f1"),
            )
            persist_model(LEGACY_SAT_PROJECT_SLUG, checkpoint.slug, record, artifact, activate=(criterion == "f1"))
