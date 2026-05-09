from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


OBJECTIVE_LABELS = {
    "f1": "F1",
    "precision": "Prec",
    "recall": "Recall",
    "auc": "AUC",
}

SEARCH_SPACE_DEFAULTS: dict[str, dict[str, list[Any]]] = {
    "LR": {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [2000, 3000],
    },
    "DT": {
        "max_depth": [3, 5, 8, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
    },
    "RF": {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 8, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },
    "XGB": {
        "n_estimators": [100, 160, 240],
        "learning_rate": [0.03, 0.05, 0.08, 0.12],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    },
}


@dataclass
class TrainModelConfig:
    project_id: str
    checkpoint_id: str
    model_id: str
    version: int
    dataset_ref: str
    id_column: str
    name_column: str
    target_column: str
    numeric_features: list[str]
    categorical_features: list[str]
    algorithm: str
    validation_mode: str
    balancing: str
    objective_metric: str
    hyperparameter_mode: str
    hyperparameter_value_or_space: dict[str, Any] | None = None
    group_column: str | None = None
    random_state: int = 42
    holdout_size: float = 0.2
    search_iterations: int = 8


def _make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_onehot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_estimator(algorithm: str, params: dict[str, Any], random_state: int):
    if algorithm == "LR":
        payload = {"max_iter": 3000, "random_state": random_state, **params}
        return LogisticRegression(**payload)
    if algorithm == "DT":
        payload = {"random_state": random_state, "class_weight": "balanced", **params}
        return DecisionTreeClassifier(**payload)
    if algorithm == "RF":
        payload = {"n_estimators": 250, "random_state": random_state, "class_weight": "balanced", **params}
        return RandomForestClassifier(**payload)
    payload = {
        "n_estimators": 160,
        "learning_rate": 0.08,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": random_state,
        **params,
    }
    return XGBClassifier(**payload)


def _get_sampler(name: str, random_state: int):
    if name == "SMOTE":
        return SMOTE(random_state=random_state)
    if name == "SMOTEENN":
        return SMOTEENN(random_state=random_state)
    return None


def _balance_transformed(X, y, balancing: str, random_state: int):
    sampler = _get_sampler(balancing, random_state)
    if sampler is None:
        return X, y
    return sampler.fit_resample(X, y)


def _compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, scores)),
        "Recall": float(recall_score(y_true, preds, zero_division=0)),
        "Prec": float(precision_score(y_true, preds, zero_division=0)),
        "F1": float(f1_score(y_true, preds, zero_division=0)),
    }


def choose_best_threshold(y_true: np.ndarray, scores: np.ndarray, objective_metric: str) -> tuple[float, str]:
    metric_key = OBJECTIVE_LABELS.get(objective_metric.lower(), "F1")
    if metric_key == "AUC":
        metric_key = "F1"
    thresholds = np.unique(np.round(np.linspace(0.05, 0.95, 37), 4))
    best_threshold = 0.5
    best_value = -1.0
    for threshold in thresholds:
        value = _compute_metrics(y_true, scores, threshold)[metric_key]
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    strategy = objective_metric.lower() if objective_metric.lower() != "auc" else "auc_with_f1_threshold"
    return best_threshold, strategy


def _validation_splits(config: TrainModelConfig, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None):
    if config.validation_mode == "Holdout":
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.holdout_size,
            random_state=config.random_state,
            stratify=y,
        )
        return [(X_train, X_test, y_train, y_test)]
    if config.validation_mode == "LOSO":
        if groups is None:
            raise ValueError("LOSO requiere una columna de grupos.")
        splitter = LeaveOneGroupOut()
        return [(X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]) for train, test in splitter.split(X, y, groups)]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)
    return [(X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]) for train, test in splitter.split(X, y)]


def _evaluate_candidate(X: pd.DataFrame, y: pd.Series, groups: pd.Series | None, config: TrainModelConfig, params: dict[str, Any]) -> dict[str, Any]:
    y_true_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for X_train, X_test, y_train, y_test in _validation_splits(config, X, y, groups):
        preprocessor = build_preprocessor(config.numeric_features, config.categorical_features)
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        X_bal, y_bal = _balance_transformed(X_train_t, y_train, config.balancing, config.random_state)
        model = build_estimator(config.algorithm, params, config.random_state)
        model.fit(X_bal, y_bal)
        scores = model.predict_proba(X_test_t)[:, 1]
        y_true_parts.append(np.asarray(y_test))
        score_parts.append(np.asarray(scores))
    y_true_all = np.concatenate(y_true_parts)
    score_all = np.concatenate(score_parts)
    threshold, strategy = choose_best_threshold(y_true_all, score_all, config.objective_metric)
    metrics = _compute_metrics(y_true_all, score_all, threshold)
    metric_key = OBJECTIVE_LABELS.get(config.objective_metric.lower(), "F1")
    return {
        "params": params,
        "threshold": threshold,
        "threshold_strategy": strategy,
        "metrics": metrics,
        "selection_score": metrics.get(metric_key, metrics["F1"]),
        "y_true": y_true_all,
        "scores": score_all,
    }


def _sample_candidates(config: TrainModelConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = config.hyperparameter_value_or_space or {}
    if config.hyperparameter_mode == "fixed":
        return [payload], {"mode": "fixed", "space": payload, "iterations": 1}

    base_space = SEARCH_SPACE_DEFAULTS[config.algorithm].copy()
    for key, values in payload.items():
        if isinstance(values, list) and values:
            base_space[key] = values
        elif values not in (None, ""):
            base_space[key] = [values]

    keys = list(base_space.keys())
    combos = [dict(zip(keys, values)) for values in product(*(base_space[key] for key in keys))]
    rng = np.random.default_rng(config.random_state)
    if len(combos) > config.search_iterations:
        indices = rng.choice(len(combos), size=config.search_iterations, replace=False)
        combos = [combos[int(i)] for i in indices]
    return combos, {"mode": "auto", "space": base_space, "iterations": len(combos)}


def _fit_final_model(X: pd.DataFrame, y: pd.Series, config: TrainModelConfig, params: dict[str, Any]):
    preprocessor = build_preprocessor(config.numeric_features, config.categorical_features)
    X_t = preprocessor.fit_transform(X)
    X_bal, y_bal = _balance_transformed(X_t, y, config.balancing, config.random_state)
    model = build_estimator(config.algorithm, params, config.random_state)
    model.fit(X_bal, y_bal)
    return preprocessor, model


def train_model(training_df: pd.DataFrame, config: TrainModelConfig) -> dict[str, Any]:
    feature_order = [*config.numeric_features, *config.categorical_features]
    X = training_df[feature_order].copy()
    y = pd.to_numeric(training_df[config.target_column], errors="coerce").fillna(0).astype(int)
    groups = training_df[config.group_column] if config.validation_mode == "LOSO" and config.group_column else None

    candidates, search_config = _sample_candidates(config)
    evaluations = [_evaluate_candidate(X, y, groups, config, params) for params in candidates]
    best = max(evaluations, key=lambda item: item["selection_score"])
    preprocessor, model = _fit_final_model(X, y, config, best["params"])
    transformed_names = list(preprocessor.get_feature_names_out())

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_schema": {
            "id_column": config.id_column,
            "name_column": config.name_column,
            "target_column": config.target_column,
            "numeric_features": list(config.numeric_features),
            "categorical_features": list(config.categorical_features),
            "feature_order": feature_order,
            "transformed_feature_names": transformed_names,
        },
        "training_summary": {
            "algorithm": config.algorithm,
            "objective_metric": config.objective_metric.lower(),
            "threshold_strategy": best["threshold_strategy"],
            "selected_threshold": best["threshold"],
            "validation_mode": config.validation_mode,
            "balancing": config.balancing,
            "hyperparameters": best["params"],
            "search_space": search_config["space"],
            "hyperparameter_mode": search_config["mode"],
            "search_iterations": search_config["iterations"],
            "group_column": config.group_column,
            "holdout_size": config.holdout_size,
            "random_state": config.random_state,
        },
        "metrics": best["metrics"],
        "project_ref": {
            "project_id": config.project_id,
            "checkpoint_id": config.checkpoint_id,
            "model_id": config.model_id,
            "version": config.version,
        },
        "dataset_ref": config.dataset_ref,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
