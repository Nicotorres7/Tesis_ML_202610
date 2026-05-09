from __future__ import annotations

from typing import Any

from sat_app.explanations import feature_display_name, top_student_factors
from sat_app.inference import predict_dataset, summarize_predictions
from sat_app.model_store import load_active_artifact, load_model_with_artifact


def load_bundle(project_slug: str, checkpoint_slug: str, model_id: str | None = None) -> dict[str, Any]:
    if model_id:
        _, artifact = load_model_with_artifact(project_slug, checkpoint_slug, model_id)
        return artifact
    _, artifact = load_active_artifact(project_slug, checkpoint_slug)
    return artifact


def predict_risk(bundle: dict[str, Any], df_input):
    return predict_dataset(bundle, df_input)


__all__ = [
    "feature_display_name",
    "load_bundle",
    "predict_risk",
    "summarize_predictions",
    "top_student_factors",
]
