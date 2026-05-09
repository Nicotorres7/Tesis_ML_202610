from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sat_app.data import standardize_prediction_identity
from sat_app.legacy_sat_import import compute_legacy_sat_features


def required_prediction_columns(artifact: dict[str, Any]) -> list[str]:
    schema = artifact["feature_schema"]
    legacy_required = artifact.get("legacy", {}).get("required_input_columns")
    if legacy_required:
        return [schema["id_column"], schema["name_column"], *legacy_required]
    return [schema["id_column"], schema["name_column"], *schema["feature_order"]]


def missing_prediction_columns(df: pd.DataFrame, artifact: dict[str, Any]) -> list[str]:
    required = required_prediction_columns(artifact)
    return [column for column in required if column not in df.columns]


def validate_prediction_dataframe(df: pd.DataFrame, artifact: dict[str, Any]) -> list[str]:
    return missing_prediction_columns(df, artifact)


def _risk_levels(probabilities: np.ndarray) -> pd.Series:
    return pd.cut(
        probabilities,
        bins=[0.0, 0.35, 0.60, 1.0],
        labels=["BAJO", "MEDIO", "ALTO"],
        include_lowest=True,
    ).astype(str)


def predict_dataset(artifact: dict[str, Any], df_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = validate_prediction_dataframe(df_input, artifact)
    if missing:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(missing))

    schema = artifact["feature_schema"]
    threshold = float(artifact["training_summary"]["selected_threshold"])
    work = df_input.copy()
    if artifact.get("legacy", {}).get("legacy_sat"):
        work = compute_legacy_sat_features(work)
    work = standardize_prediction_identity(work, schema["id_column"], schema["name_column"])
    X = work[schema["feature_order"]].copy()
    if artifact.get("legacy", {}).get("legacy_sat"):
        X = X.apply(pd.to_numeric, errors="coerce")
        for column in artifact.get("legacy", {}).get("impute_zeros", []):
            if column in X.columns:
                X[column] = X[column].fillna(0)
        X = X.fillna(0)
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    X_t = preprocessor.transform(X)
    probabilities = model.predict_proba(X_t)[:, 1]
    alerts = (probabilities >= threshold).astype(int)
    results = pd.DataFrame(
        {
            "id": work["id"],
            "name": work["name"],
            "probabilidad": np.round(probabilities, 4),
            "alerta": alerts,
            "nivel_riesgo": _risk_levels(probabilities),
            "umbral_modelo": threshold,
            "student_key": work["student_key"],
            "student_label": work["student_label"],
        },
        index=work.index,
    )
    enriched = work.copy()
    for column in ["probabilidad", "alerta", "nivel_riesgo", "umbral_modelo", "student_key", "student_label"]:
        enriched[column] = results[column]
    return results, enriched


def summarize_predictions(df: pd.DataFrame) -> dict[str, Any]:
    total = int(len(df))
    counts = df["nivel_riesgo"].value_counts().to_dict()
    return {
        "total": total,
        "alto": int(counts.get("ALTO", 0)),
        "medio": int(counts.get("MEDIO", 0)),
        "bajo": int(counts.get("BAJO", 0)),
        "promedio": float(df["probabilidad"].mean()) if total else 0.0,
    }
