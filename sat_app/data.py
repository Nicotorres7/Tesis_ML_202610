from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

import pandas as pd

from sat_app.config import ALIASES


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower().split(".")[-1]
    content = uploaded_file.getvalue()
    if suffix == "csv":
        return pd.read_csv(BytesIO(content))
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(BytesIO(content))
    raise ValueError("Formato no soportado. Usa .csv o .xlsx.")


def normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def autodetect_mapping(columns: list[str], required: list[str]) -> dict[str, str]:
    normalized = {col: normalize_name(col) for col in columns}
    mapping: dict[str, str] = {}
    for canonical in required:
        aliases = [canonical, *ALIASES.get(canonical, [])]
        detected = None
        for alias in aliases:
            norm_alias = normalize_name(alias)
            for original, normalized_col in normalized.items():
                if normalized_col == norm_alias or norm_alias in normalized_col or normalized_col in norm_alias:
                    detected = original
                    break
            if detected:
                break
        mapping[canonical] = detected or "__MISSING__"
    return mapping


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    renamed = pd.DataFrame(index=df.index)
    for canonical, original in mapping.items():
        if original and original != "__MISSING__" and original in df.columns:
            renamed[canonical] = df[original]
        else:
            renamed[canonical] = pd.NA
    return renamed


def ensure_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id_estudiante" not in df.columns or df["id_estudiante"].isna().all():
        df["id_estudiante"] = [f"auto_{i+1:03d}" for i in range(len(df))]
    if "nombre" not in df.columns:
        df["nombre"] = pd.NA
    df["student_key"] = df["id_estudiante"].astype(str).apply(short_hash)
    df["student_label"] = df["nombre"].fillna("").astype(str).str.strip()
    df["student_label"] = df["student_label"].where(df["student_label"] != "", "[Anonimo]")
    return df


def short_hash(value: Any) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return f"0x{digest[:4]}"


def completeness_score(df: pd.DataFrame, required_cols: list[str]) -> float:
    if not required_cols or len(df) == 0:
        return 100.0
    subset = df.reindex(columns=required_cols)
    return round(float(subset.notna().mean().mean() * 100), 1)


def dataset_warnings(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    warnings: list[str] = []
    for col in required_cols:
        if col in df.columns:
            missing = int(df[col].isna().sum())
            if missing:
                warnings.append(f"{missing} registros sin {col.replace('_', ' ')}")
    return warnings[:5]
