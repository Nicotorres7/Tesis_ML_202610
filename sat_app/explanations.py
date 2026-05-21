from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sat_app.config import DISPLAY_LABELS, RISK_COLORS


def feature_display_name(name: str) -> str:
    return DISPLAY_LABELS.get(name, name.replace("_", " ").title())


def _base_feature_name(transformed_name: str) -> str:
    if transformed_name.startswith("num__"):
        return transformed_name.split("num__", 1)[1]
    if transformed_name.startswith("cat__"):
        rest = transformed_name.split("cat__", 1)[1]
        parts = rest.rsplit("_", 1)
        return parts[0] if len(parts) > 1 else rest
    return transformed_name


def _aggregate_contributions(values: np.ndarray, transformed_names: list[str]) -> pd.Series:
    mapping: dict[str, float] = {}
    for name, value in zip(transformed_names, values):
        base = _base_feature_name(name)
        mapping[base] = mapping.get(base, 0.0) + float(value)
    return pd.Series(mapping)


def top_student_factors(artifact: dict[str, Any], df_features: pd.DataFrame, index: Any) -> pd.DataFrame:
    schema = artifact["feature_schema"]
    feature_order = schema["feature_order"]
    transformed_names = schema.get("transformed_feature_names") or feature_order
    X = df_features[feature_order].copy()
    X_t = artifact["preprocessor"].transform(X)
    if hasattr(X_t, 'toarray'):
        X_t = X_t.toarray()
    model = artifact["model"]
    row_position = df_features.index.get_loc(index)

    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        contributions = np.asarray(X_t[row_position]).ravel() * coef
    else:
        importances = getattr(model, "feature_importances_", np.ones(X_t.shape[1]))
        centered = np.asarray(X_t[row_position]).ravel() - np.asarray(X_t).mean(axis=0)
        contributions = centered * importances

    row = _aggregate_contributions(contributions, transformed_names)
    top = (
        row.sort_values(key=lambda s: np.abs(s), ascending=False)
        .head(6)
        .rename_axis("feature")
        .reset_index(name="impacto")
    )
    top["direccion"] = np.where(top["impacto"] >= 0, "incrementa", "reduce")
    top["color"] = np.where(top["impacto"] >= 0, RISK_COLORS["ALTO"], "#2563EB")
    top["feature_label"] = top["feature"].map(feature_display_name)
    top["magnitud"] = top["impacto"].abs()
    top["sentido"] = np.where(top["impacto"] >= 0, "Aumenta el riesgo estimado", "Reduce el riesgo estimado")
    return top


def cohort_feature_importance_data(artifact: dict[str, Any]) -> pd.DataFrame:
    schema = artifact["feature_schema"]
    transformed_names = schema.get("transformed_feature_names") or schema["feature_order"]
    model = artifact["model"]
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.ravel(model.coef_))
    else:
        importances = np.ones(len(transformed_names))
    row = _aggregate_contributions(importances, transformed_names)
    return (
        row.rename_axis("feature")
        .reset_index(name="importance")
        .assign(feature_label=lambda frame: frame["feature"].map(feature_display_name))
        .sort_values("importance", ascending=False)
    )
