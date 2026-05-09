from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from sat_app.training import TrainModelConfig, train_model


@dataclass
class RetrainConfig:
    checkpoint: str
    criterion: str
    model_family: str
    balancing: str
    validation: str
    test_size: float
    random_state: int
    target_column: str
    semester_column: str | None = None


def run_retraining(df: pd.DataFrame, features: list[str], config: RetrainConfig) -> dict[str, Any]:
    payload = TrainModelConfig(
        project_id="legacy",
        checkpoint_id=config.checkpoint,
        model_id="legacy-preview",
        version=1,
        dataset_ref="legacy-preview",
        id_column="id_estudiante",
        name_column="nombre",
        target_column=config.target_column,
        numeric_features=list(features),
        categorical_features=[],
        algorithm=config.model_family,
        validation_mode=config.validation,
        balancing=config.balancing,
        objective_metric=config.criterion,
        hyperparameter_mode="fixed",
        hyperparameter_value_or_space={},
        group_column=config.semester_column,
        random_state=config.random_state,
        holdout_size=config.test_size,
    )
    artifact = train_model(df, payload)
    return {
        "modelo": artifact["model"],
        "preprocessor": artifact["preprocessor"],
        "features": artifact["feature_schema"]["feature_order"],
        "checkpoint": config.checkpoint,
        "criterion": config.criterion,
        "metrics": artifact["metrics"],
        "trained_at": artifact["created_at"],
        "config": payload.__dict__,
        "artifact": artifact,
    }


def save_retrained_model(artifact: dict[str, Any]):
    raise NotImplementedError("El flujo legacy de guardado fue reemplazado por el registro de proyectos/modelos.")
