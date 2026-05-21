from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

import pandas as pd

from sat_app.config import ALIASES


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower().split(".")[-1]
    content = uploaded_file.getvalue()
    try:
        if suffix == "csv":
            return pd.read_csv(BytesIO(content))
        if suffix in {"xlsx", "xls"}:
            return pd.read_excel(BytesIO(content))
        raise ValueError("Formato no soportado. Usa .csv o .xlsx.")
    except ValueError:
        raise
    except (UnicodeDecodeError, pd.errors.ParserError) as e:
        raise ValueError(f"Error al procesar el archivo: {e}") from e
    except Exception as e:
        raise ValueError(f"Error inesperado al cargar el archivo: {e}") from e


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


def standardize_prediction_identity(df: pd.DataFrame, id_column: str, name_column: str) -> pd.DataFrame:
    out = df.copy()
    out["id"] = out[id_column].astype(str)
    out["name"] = out[name_column].fillna("").astype(str).str.strip()
    out["name"] = out["name"].where(out["name"] != "", "[Anonimo]")
    out["student_key"] = out["id"].astype(str).apply(short_hash)
    out["student_label"] = out["name"]
    return out


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


def validate_binary_target(df: pd.DataFrame, target_column: str) -> tuple[bool, str]:
    if target_column not in df.columns:
        return False, f"No existe la columna objetivo '{target_column}'."
    values = pd.to_numeric(df[target_column], errors="coerce").dropna().astype(int)
    unique = sorted(values.unique().tolist())
    if not unique:
        return False, "La columna objetivo no contiene valores válidos."
    if any(value not in {0, 1} for value in unique):
        return False, f"El target debe ser binario 0/1. Valores detectados: {unique}."
    if len(unique) < 2:
        return False, "El target debe contener al menos las dos clases 0 y 1."
    return True, ""


def validate_role_columns(
    columns: list[str],
    id_column: str,
    name_column: str,
    target_column: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    errors: list[str] = []
    required = [("ID", id_column), ("Nombre", name_column), ("Target", target_column)]
    for label, column in required:
        if not column:
            errors.append(f"Debes seleccionar la columna de {label.lower()}.")
        elif column not in columns:
            errors.append(f"La columna de {label.lower()} '{column}' no existe en el dataset.")

    if id_column and name_column and id_column == name_column:
        errors.append("Las columnas de ID y nombre deben ser distintas.")
    if target_column and target_column in {id_column, name_column}:
        errors.append("El target no puede coincidir con ID ni con nombre.")

    if not numeric_features and not categorical_features:
        errors.append("Debes seleccionar al menos una feature numérica o categórica.")

    overlap = sorted(set(numeric_features) & set(categorical_features))
    if overlap:
        errors.append("Una columna no puede ser numérica y categórica al tiempo: " + ", ".join(overlap))

    forbidden = {id_column, name_column, target_column}
    bad_numeric = sorted(set(numeric_features) & forbidden)
    bad_categorical = sorted(set(categorical_features) & forbidden)
    if bad_numeric:
        errors.append("Las features numéricas no pueden incluir ID/nombre/target: " + ", ".join(bad_numeric))
    if bad_categorical:
        errors.append("Las features categóricas no pueden incluir ID/nombre/target: " + ", ".join(bad_categorical))

    for column in [*numeric_features, *categorical_features]:
        if column not in columns:
            errors.append(f"La feature '{column}' no existe en el dataset.")
    return errors


def feature_quality_warnings(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    warnings: list[str] = []
    for column in numeric_features:
        missing_ratio = float(df[column].isna().mean()) if column in df.columns else 0.0
        if missing_ratio > 0.30:
            warnings.append(f"La feature numérica '{column}' tiene {missing_ratio:.0%} de nulos.")
    for column in categorical_features:
        missing_ratio = float(df[column].isna().mean()) if column in df.columns else 0.0
        if missing_ratio > 0.30:
            warnings.append(f"La feature categórica '{column}' tiene {missing_ratio:.0%} de nulos.")
        if column in df.columns:
            nunique = int(df[column].nunique(dropna=True))
            if nunique > min(50, max(10, len(df) // 4)):
                warnings.append(f"La feature categórica '{column}' tiene alta cardinalidad ({nunique} valores).")
    return warnings[:8]
