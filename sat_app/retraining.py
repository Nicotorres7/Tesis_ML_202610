from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sat_app.config import RETRAINED_DIR


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


def _build_estimator(family: str, random_state: int):
    if family == "LR":
        return LogisticRegression(max_iter=3000, solver="saga", random_state=random_state)
    if family == "DT":
        return DecisionTreeClassifier(random_state=random_state, class_weight="balanced")
    if family == "RF":
        return RandomForestClassifier(n_estimators=250, random_state=random_state, class_weight="balanced")
    return XGBClassifier(
        n_estimators=160,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
    )


def _balance_data(X: pd.DataFrame, y: pd.Series, balancing: str, random_state: int):
    if balancing == "SMOTE":
        sampler = SMOTE(random_state=random_state)
    elif balancing == "SMOTEENN":
        sampler = SMOTEENN(random_state=random_state)
    else:
        return X, y
    X_res, y_res = sampler.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)


def _build_preprocessor(df: pd.DataFrame):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ]
    )


def _evaluate(y_true, scores, threshold=0.5):
    preds = (scores >= threshold).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, scores)),
        "Recall": float(recall_score(y_true, preds, zero_division=0)),
        "Prec": float(precision_score(y_true, preds, zero_division=0)),
        "F1": float(f1_score(y_true, preds, zero_division=0)),
    }


def run_retraining(df: pd.DataFrame, features: list[str], config: RetrainConfig) -> dict[str, Any]:
    work = df.copy()
    y = pd.to_numeric(work.pop(config.target_column), errors="coerce").fillna(0).astype(int)
    X = work[features].copy()

    preprocessor = _build_preprocessor(X)
    estimator = _build_estimator(config.model_family, config.random_state)

    if config.validation == "Holdout":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
        )
        X_bal, y_bal = _balance_data(X_train, y_train, config.balancing, config.random_state)
        Xt_bal = preprocessor.fit_transform(X_bal)
        estimator.fit(Xt_bal, y_bal)
        scores = estimator.predict_proba(preprocessor.transform(X_test))[:, 1]
        metrics = _evaluate(y_test, scores)
    else:
        if config.validation == "LOSO" and config.semester_column and config.semester_column in df.columns:
            splitter = LeaveOneGroupOut()
            groups = df[config.semester_column]
            splits = splitter.split(X, y, groups=groups)
        else:
            splitter = KFold(n_splits=5, shuffle=True, random_state=config.random_state)
            splits = splitter.split(X, y)
        fold_metrics = []
        last_model = None
        for train_idx, test_idx in splits:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_bal, y_bal = _balance_data(X_train, y_train, config.balancing, config.random_state)
            Xt_bal = preprocessor.fit_transform(X_bal)
            model = _build_estimator(config.model_family, config.random_state)
            model.fit(Xt_bal, y_bal)
            scores = model.predict_proba(preprocessor.transform(X_test))[:, 1]
            fold_metrics.append(_evaluate(y_test, scores))
            last_model = model
        metrics = {key: float(np.mean([fold[key] for fold in fold_metrics])) for key in fold_metrics[0]}
        estimator = last_model

    preprocessor = _build_preprocessor(X)
    X_bal, y_bal = _balance_data(X, y, config.balancing, config.random_state)
    Xt_bal = preprocessor.fit_transform(X_bal)
    estimator.fit(Xt_bal, y_bal)

    artifact = {
        "modelo": estimator,
        "preprocessor": preprocessor,
        "features": features,
        "checkpoint": config.checkpoint,
        "criterion": config.criterion,
        "metrics": metrics,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "config": config.__dict__,
    }
    return artifact


def save_retrained_model(artifact: dict[str, Any]) -> Path:
    filename = f"{artifact['checkpoint']}_{artifact['criterion']}_{artifact['config']['model_family'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    path = RETRAINED_DIR / filename
    joblib.dump(artifact, path)
    return path
