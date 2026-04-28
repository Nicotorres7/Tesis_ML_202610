from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sat_app.config import (
    CHECKPOINTS,
    CRITERIA_DESCRIPTIONS,
    CRITERIA_LABELS,
    DERIVED_DEPENDENCIES,
    DISPLAY_LABELS,
    RISK_COLORS,
)


@lru_cache(maxsize=1)
def load_all_metadata() -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for checkpoint, info in CHECKPOINTS.items():
        data[checkpoint] = json.loads(info["metadata_file"].read_text(encoding="utf-8"))
    return data


@lru_cache(maxsize=None)
def load_bundle(checkpoint: str, criterion: str) -> dict[str, Any]:
    return joblib.load(CHECKPOINTS[checkpoint]["bundle_files"][criterion])


def available_model_cards() -> dict[str, list[dict[str, Any]]]:
    metadata = load_all_metadata()
    cards: dict[str, list[dict[str, Any]]] = {}
    for checkpoint, info in metadata.items():
        cards[checkpoint] = []
        for criterion in ["precision", "f1", "recall"]:
            item = info[criterion]
            family = item["familia"].lower()
            cards[checkpoint].append(
                {
                    "criterion": criterion,
                    "label": CRITERIA_LABELS[criterion],
                    "description": CRITERIA_DESCRIPTIONS[criterion],
                    "family": item["familia"],
                    "threshold": item["umbral"],
                    "features": item["features"],
                    "metrics": item["metricas_loso"],
                    "image_main": _pick_image(checkpoint, family, criterion),
                    "image_threshold": CHECKPOINTS[checkpoint]["threshold_image"],
                    "image_comparison": CHECKPOINTS[checkpoint]["comparison_image"],
                }
            )
    return cards


def _pick_image(checkpoint: str, family: str, criterion: str) -> Path:
    filename = f"fig_{family}_{checkpoint}_{criterion}.png"
    path = CHECKPOINTS[checkpoint]["comparison_image"].parent / filename
    return path if path.exists() else CHECKPOINTS[checkpoint]["comparison_image"]


def expand_required_columns(features: list[str]) -> list[str]:
    required: list[str] = []
    for feature in features:
        deps = DERIVED_DEPENDENCIES.get(feature, [feature])
        for dep in deps:
            if dep not in required:
                required.append(dep)
    if "id_estudiante" not in required:
        required.insert(0, "id_estudiante")
    if "nombre" not in required:
        required.insert(1, "nombre")
    return required


def feature_display_name(name: str) -> str:
    return DISPLAY_LABELS.get(name, name.replace("_", " ").title())


def compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "era" in df.columns:
        normalized_era = df["era"].astype(str).str.lower().str.strip()
        df["era_encoded"] = normalized_era.eq("formato_nuevo").astype(int)
    if "engagement_hasta_p1" in df.columns:
        df["log_eng_p1"] = np.log1p(pd.to_numeric(df["engagement_hasta_p1"], errors="coerce").fillna(0))
    if "engagement_hasta_p2" in df.columns:
        df["log_eng_p2"] = np.log1p(pd.to_numeric(df["engagement_hasta_p2"], errors="coerce").fillna(0))
    if "parcial_1" in df.columns:
        parcial_1 = pd.to_numeric(df["parcial_1"], errors="coerce")
        media = parcial_1.mean()
        std = parcial_1.std() + 1e-6
        df["p1_vs_media"] = (parcial_1 - media) / std
    if {"parcial_1", "parcial_2"}.issubset(df.columns):
        p1 = pd.to_numeric(df["parcial_1"], errors="coerce")
        p2 = pd.to_numeric(df["parcial_2"], errors="coerce")
        df["delta_p2_p1"] = p2 - p1
    if {"engagement_hasta_p1", "parcial_1_visitas"}.issubset(df.columns):
        eng = pd.to_numeric(df["engagement_hasta_p1"], errors="coerce").fillna(0)
        visitas = pd.to_numeric(df["parcial_1_visitas"], errors="coerce").fillna(0)
        df["intensidad_p1"] = np.where(visitas > 0, eng * 60 / visitas, 0.0)
    if {"parcial_1_visitas", "parcial_1_temas_unicos"}.issubset(df.columns):
        visitas = pd.to_numeric(df["parcial_1_visitas"], errors="coerce").fillna(0)
        temas = pd.to_numeric(df["parcial_1_temas_unicos"], errors="coerce").fillna(0)
        df["ratio_vt_p1"] = np.where(temas > 0, visitas / temas, 0.0)
    return df


def predict_risk(bundle: dict[str, Any], df_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = bundle["modelo"]
    scaler = bundle["scaler"]
    features = bundle["features"]
    threshold = bundle["umbral"]
    izeros = bundle.get("impute_zeros", [])

    df_ready = compute_derived_fields(df_input)
    X = df_ready.reindex(columns=features).copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    for col in izeros:
        if col in X.columns:
            X[col] = X[col].fillna(0)
    X = X.fillna(0)

    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    alerts = (probabilities >= threshold).astype(int)
    risk = pd.cut(
        probabilities,
        bins=[0.0, 0.35, 0.60, 1.0],
        labels=["BAJO", "MEDIO", "ALTO"],
        include_lowest=True,
    )
    results = pd.DataFrame(
        {
            "probabilidad": np.round(probabilities, 4),
            "alerta": alerts,
            "nivel_riesgo": risk.astype(str),
            "umbral_modelo": threshold,
        },
        index=df_ready.index,
    )
    enriched = df_ready.copy()
    enriched["probabilidad"] = results["probabilidad"]
    enriched["nivel_riesgo"] = results["nivel_riesgo"]
    enriched["alerta"] = results["alerta"]
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


def top_student_factors(bundle: dict[str, Any], df_features: pd.DataFrame, index: Any) -> pd.DataFrame:
    df_ready = compute_derived_fields(df_features)
    features = bundle["features"]
    X = df_ready.reindex(columns=features).copy().apply(pd.to_numeric, errors="coerce").fillna(0)
    scaler = bundle["scaler"]
    X_scaled = scaler.transform(X)
    model = bundle["modelo"]

    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        contributions = X_scaled * coef
        row = pd.Series(contributions[df_ready.index.get_loc(index)], index=features)
    else:
        importances = getattr(model, "feature_importances_", np.ones(len(features)))
        centered = X_scaled - np.nanmean(X_scaled, axis=0)
        row = pd.Series(centered[df_ready.index.get_loc(index)] * importances, index=features)

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
    top["sentido"] = np.where(
        top["impacto"] >= 0,
        "Lleva a un mayor riesgo",
        "Ayuda a contener el riesgo",
    )
    return top
