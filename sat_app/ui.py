from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

from sat_app.charts import (
    cohort_feature_importance,
    cohort_heatmap,
    cohort_trend_chart,
    feature_importance_chart,
    metric_comparison_figure,
    percentile_table_data,
    probability_histogram,
    risk_boxplot,
    risk_distribution_chart,
    risk_radar_chart,
    scatter_p1_vs_prob,
    small_probability_gauge,
    student_percentile_chart,
)
from sat_app.config import CRITERIA_LABELS, DISPLAY_LABELS, LEGACY_SAT_PROJECT_SLUG, THEMES, build_base_style
from sat_app.data import (
    feature_quality_warnings,
    load_uploaded_dataset,
    validate_binary_target,
    validate_role_columns,
)
from sat_app.explanations import feature_display_name, top_student_factors
from sat_app.exporters import dataframe_to_excel_bytes, summary_pdf_bytes
from sat_app.inference import predict_dataset, summarize_predictions, validate_prediction_dataframe
from sat_app.legacy_sat_import import ensure_legacy_sat_project
from sat_app.model_store import load_active_artifact, load_model_with_artifact, persist_model
from sat_app.registry import (
    ModelRecord,
    PredictionRunRecord,
    activate_model,
    create_checkpoint,
    create_project,
    get_active_model,
    get_checkpoint,
    get_prediction_run,
    get_project,
    list_checkpoints,
    list_models,
    list_prediction_runs,
    list_projects,
    next_model_version,
)
from sat_app.training import SEARCH_SPACE_DEFAULTS, TrainModelConfig, train_model


def run_app():
    st.set_page_config(page_title="SAT Uniandes", page_icon=":bar_chart:", layout="wide")
    ensure_legacy_sat_project()
    _init_state()
    st.markdown(build_base_style(_theme_name()), unsafe_allow_html=True)

    with st.sidebar:
        _render_sidebar()

    page = st.session_state.current_page
    _render_top_context_bar()
    if page == "Proyectos":
        render_projects()
    elif page == "Entrenamiento":
        render_training()
    elif page == "Predicción":
        render_prediction()
    elif page == "Dashboard":
        render_dashboard()
    elif page == "Modelos":
        render_models()
    else:
        render_export()


NAV_PAGES = ["Proyectos", "Entrenamiento", "Predicción", "Dashboard", "Modelos", "Exportación"]
NAV_STEPS = {
    "Proyectos": "01",
    "Entrenamiento": "02",
    "Predicción": "03",
    "Dashboard": "04",
    "Modelos": "05",
    "Exportación": "06",
}


def _init_state() -> None:
    defaults = {
        "app_theme": "Claro",
        "current_page": "Proyectos",
        "sidebar_nav_page": "01 · Proyectos",
        "selected_project_slug": "",
        "selected_checkpoint_slug": "",
        "selected_model_id": "",
        "top_project_picker": "",
        "top_checkpoint_picker": "",
        "top_model_picker": "",
        "sidebar_project_picker": "",
        "sidebar_checkpoint_picker": "",
        "sidebar_model_picker": "",
        "latest_predictions": {},
        "selected_student_key": "",
        "train_dataset": None,
        "prediction_dataset": None,
        "training_step": 1,
        "training_success": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _theme_name() -> str:
    return "Claro"


def _chart_theme() -> str:
    return _theme_name()


def _set_page(page: str) -> None:
    st.session_state.current_page = page
    # sidebar_nav_page is a widget key — it is synced pre-render in _render_sidebar


def _set_project(slug: str) -> None:
    st.session_state.selected_project_slug = slug
    st.session_state.selected_checkpoint_slug = ""
    st.session_state.selected_model_id = ""
    st.session_state.training_step = 1
    st.session_state.training_success = None


def _set_checkpoint(slug: str) -> None:
    st.session_state.selected_checkpoint_slug = slug
    st.session_state.selected_model_id = ""
    st.session_state.training_step = 1
    st.session_state.training_success = None


def _set_model(model_id: str) -> None:
    st.session_state.selected_model_id = model_id


def _sync_top_project() -> None:
    _set_project(st.session_state.top_project_picker)


def _sync_top_checkpoint() -> None:
    _set_checkpoint(st.session_state.top_checkpoint_picker)


def _sync_top_model() -> None:
    _set_model(st.session_state.top_model_picker)


def _project_options() -> list[str]:
    return [project.slug for project in list_projects()]


def _ensure_valid_selection() -> None:
    """Synchronise canonical selection state before any widgets are rendered.

    Widget keys (top_*_picker, sidebar_*_picker) are synced here, before
    the widgets are instantiated, so they can never be mutated after render.
    """
    projects = _project_options()
    nav_options = [f"{NAV_STEPS[page]} · {page}" for page in NAV_PAGES]
    current_nav_label = f"{NAV_STEPS.get(st.session_state.current_page, '01')} · {st.session_state.current_page}"
    if st.session_state.sidebar_nav_page not in nav_options:
        st.session_state.sidebar_nav_page = current_nav_label
    if not projects:
        return

    # Canonical project
    if st.session_state.selected_project_slug not in projects:
        st.session_state.selected_project_slug = projects[0]
    proj = st.session_state.selected_project_slug
    # Sync pickers to canonical (only before widget render, safe here)
    st.session_state.top_project_picker = proj
    st.session_state.sidebar_project_picker = proj

    checkpoints = [cp.slug for cp in list_checkpoints(proj)]
    if not checkpoints:
        st.session_state.selected_checkpoint_slug = ""
        st.session_state.selected_model_id = ""
        st.session_state.top_checkpoint_picker = ""
        st.session_state.top_model_picker = ""
        st.session_state.sidebar_checkpoint_picker = ""
        st.session_state.sidebar_model_picker = ""
        return

    # Canonical checkpoint: always auto-select unless we are on the training page
    # at step 1 (where the user is intentionally choosing a checkpoint).
    on_training_step1 = (
        st.session_state.current_page == "Entrenamiento"
        and st.session_state.training_step == 1
    )
    if st.session_state.selected_checkpoint_slug not in checkpoints:
        if not on_training_step1:
            st.session_state.selected_checkpoint_slug = checkpoints[0]
        # else: leave it as "" so step 1 shows all checkpoints unselected

    cp = st.session_state.selected_checkpoint_slug or checkpoints[0]
    st.session_state.top_checkpoint_picker = cp
    st.session_state.sidebar_checkpoint_picker = cp

    if not st.session_state.selected_checkpoint_slug:
        # No canonical checkpoint chosen yet; don't sync model state
        st.session_state.selected_model_id = ""
        st.session_state.top_model_picker = ""
        st.session_state.sidebar_model_picker = ""
        return

    models = list_models(proj, cp)
    model_ids = [item.model_id for item in models]
    if not model_ids:
        st.session_state.selected_model_id = ""
        st.session_state.top_model_picker = ""
        st.session_state.sidebar_model_picker = ""
        return

    active = get_active_model(proj, cp)
    if st.session_state.selected_model_id not in model_ids:
        st.session_state.selected_model_id = active.model_id if active else model_ids[-1]
    mid = st.session_state.selected_model_id
    st.session_state.top_model_picker = mid
    st.session_state.sidebar_model_picker = mid


def _render_sidebar() -> None:
    _ensure_valid_selection()
    _render_sidebar_style()
    st.markdown("## SAT Uniandes")
    st.caption("Modelado académico y analítica de riesgo")
    nav_options = [f"{NAV_STEPS[page]} · {page}" for page in NAV_PAGES]
    current_nav_label = f"{NAV_STEPS[st.session_state.current_page]} · {st.session_state.current_page}"
    # Sync nav label before the widget is rendered (safe pre-render write)
    if st.session_state.sidebar_nav_page not in nav_options:
        st.session_state.sidebar_nav_page = current_nav_label
    # Keep sidebar_nav_page pointing at the current page (pre-render sync)
    st.session_state.sidebar_nav_page = current_nav_label
    st.radio(
        "Navegación",
        options=nav_options,
        key="sidebar_nav_page",
        label_visibility="collapsed",
        on_change=lambda: _set_page(st.session_state.sidebar_nav_page.split("·", 1)[1].strip()),
    )
    st.caption(_page_help_text(st.session_state.current_page))

    projects = list_projects()
    if not projects:
        st.info("No hay proyectos aún.")
        return

    st.markdown(
        """
        <div class="sat-sidebar-note">
            Barra fija de navegación.
            El contexto activo y el avance del flujo se muestran arriba del contenido principal.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_style() -> None:
    panel = "#F8FBFF"
    border = "#D5E4F4"
    title = "#102A43"
    muted = "#486581"
    accent = "#0F6CBD"
    st.markdown(
        f"""
        <style>
        .sat-sidebar-note {{
            padding:10px 12px;
            border-radius:12px;
            border:1px dashed {border};
            color:{muted};
            font-size:0.82rem;
            line-height:1.45;
            background:#F8FBFF;
            margin-top:10px;
        }}
        [data-testid="stSidebar"] h2 {{
            color:{title} !important;
            font-size:1.08rem !important;
            font-weight:800 !important;
            letter-spacing:-0.02em;
        }}
        [data-testid="stSidebar"] .stCaption {{
            color:{muted} !important;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg,#FFFFFF 0%, #F7FBFF 100%) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] > div {{
            gap: 8px;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label {{
            border:1px solid {border};
            border-radius:14px;
            padding:10px 12px;
            background:#F8FBFF;
            transition: all 0.18s ease;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
            background:#EAF4FF;
            border-color:#BFD8F2;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
            background:linear-gradient(135deg,#0F6CBD 0%, #2DA8FF 100%);
            border-color:#0F6CBD;
            box-shadow:0 10px 22px rgba(15,108,189,0.18);
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {{
            color:#FFFFFF !important;
            font-weight:700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_top_context_bar() -> None:
    project_slug = st.session_state.selected_project_slug
    checkpoint_slug = st.session_state.selected_checkpoint_slug
    if not project_slug:
        return
    project = get_project(project_slug)
    checkpoint = get_checkpoint(project_slug, checkpoint_slug) if checkpoint_slug else None
    models = list_models(project_slug, checkpoint_slug) if checkpoint_slug else []
    model_text = _model_label(project_slug, checkpoint_slug, st.session_state.selected_model_id) if checkpoint_slug and models and st.session_state.selected_model_id else "Sin modelo seleccionado"
    payload = _current_prediction_payload()
    data_ready = bool(checkpoint and checkpoint.dataset_ref)
    variables_ready = bool(
        checkpoint
        and checkpoint.id_column
        and checkpoint.name_column
        and checkpoint.target_column
        and (checkpoint.numeric_features or checkpoint.categorical_features)
    )
    predict_ready = bool(models)
    flow_rows = [
        ("Proyecto", "Listo"),
        ("Checkpoint", "Listo" if checkpoint else "Pendiente"),
        ("Datos", "Listo" if data_ready else "Pendiente"),
        ("Variables", "Listo" if variables_ready else "Pendiente"),
        ("Modelo", "Listo" if predict_ready else "Pendiente"),
        ("Predicción", "Lista" if payload else "Pendiente"),
    ]
    progress = int(sum(status in {"Listo", "Lista"} for _, status in flow_rows) / len(flow_rows) * 100)
    summary = summarize_predictions(payload["results"]) if payload else None
    flow_html = "".join(
        [
            f"<div class='sat-top-step {'is-done' if status in {'Listo', 'Lista'} else ''}'><span>{label}</span><strong>{status}</strong></div>"
            for label, status in flow_rows
        ]
    )
    stage = st.session_state.current_page
    hero_left, hero_right = st.columns([1.45, 0.95])
    with hero_left:
        st.markdown(
            f"""
            <div class="sat-top-context">
                <div class="sat-top-route">
                    <span class="sat-top-chip">Etapa {NAV_STEPS.get(stage, '--')}</span>
                    <span class="sat-top-stage">{stage}</span>
                </div>
                <div class="sat-top-title">{project.name}</div>
                <div class="sat-top-meta">
                    <span>Checkpoint: {checkpoint.name if checkpoint else 'Sin checkpoint'}</span>
                    <span>Modelo: {model_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_right:
        st.markdown(
            f"""
            <div class="sat-top-side">
                <div class="sat-top-progress-label">Avance del flujo</div>
                <div class="sat-top-progress-track"><div class="sat-top-progress-fill" style="width:{progress}%;"></div></div>
                <div class="sat-top-progress-value">{progress}% completo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if summary:
            s1, s2, s3 = st.columns(3)
            with s1:
                st.metric("Alto", summary["alto"])
            with s2:
                st.metric("Medio", summary["medio"])
            with s3:
                st.metric("Bajo", summary["bajo"])

    selector_left, selector_mid, selector_right = st.columns([1.15, 1.05, 1.3])
    projects = list_projects()
    project_options = [item.slug for item in projects]
    with selector_left:
        st.selectbox(
            "Proyecto activo",
            options=project_options,
            format_func=lambda slug: get_project(slug).name,
            key="top_project_picker",
            on_change=_sync_top_project,
        )
    checkpoints = list_checkpoints(st.session_state.selected_project_slug)
    checkpoint_options = [item.slug for item in checkpoints]
    with selector_mid:
        st.selectbox(
            "Checkpoint activo",
            options=checkpoint_options or [""],
            format_func=lambda slug: get_checkpoint(st.session_state.selected_project_slug, slug).name if slug else "Sin checkpoints",
            key="top_checkpoint_picker",
            on_change=_sync_top_checkpoint,
            disabled=not checkpoint_options,
        )
    current_checkpoint_slug = st.session_state.selected_checkpoint_slug
    current_models = list_models(project_slug, current_checkpoint_slug) if current_checkpoint_slug else []
    model_options = [item.model_id for item in current_models]
    with selector_right:
        st.selectbox(
            "Modelo en vista",
            options=model_options or [""],
            format_func=lambda model_id: _model_label(project_slug, current_checkpoint_slug, model_id) if model_id else "Sin modelos",
            key="top_model_picker",
            on_change=_sync_top_model,
            disabled=not model_options,
        )
    st.markdown(
        f"""
        <div class="sat-top-flow">{flow_html}</div>
        """,
        unsafe_allow_html=True,
    )
    _render_top_context_style()


def _render_top_context_style() -> None:
    dark = _theme_name() == "Oscuro"
    bg = "#101827" if dark else "#FFFFFF"
    bg_soft = "#172033" if dark else "#F8FAFC"
    border = "#334155" if dark else "#D9E2F1"
    text = "#F8FAFC" if dark else "#0F172A"
    muted = "#CBD5E1" if dark else "#475569"
    accent = "#2DA8FF" if dark else "#0F6CBD"
    accent_soft = "rgba(45,168,255,0.12)" if dark else "rgba(15,108,189,0.08)"
    st.markdown(
        f"""
        <style>
        .sat-top-context {{
            margin: 0 0 1rem 0;
            background:{bg};
            border:1px solid {border};
            border-radius:20px;
            padding:18px 20px 16px;
            box-shadow:0 18px 44px rgba(15,23,42,0.08);
        }}
        .sat-top-route {{
            display:flex;
            align-items:center;
            gap:10px;
            margin-bottom:8px;
            flex-wrap:wrap;
        }}
        .sat-top-chip {{
            display:inline-flex;
            align-items:center;
            padding:5px 10px;
            border-radius:999px;
            background:{accent_soft};
            color:{accent};
            font-size:0.75rem;
            font-weight:800;
            letter-spacing:0.04em;
            text-transform:uppercase;
        }}
        .sat-top-stage {{
            color:{muted};
            font-size:0.88rem;
            font-weight:600;
        }}
        .sat-top-title {{
            color:{text};
            font-size:1.45rem;
            font-weight:800;
            line-height:1.15;
            margin-bottom:6px;
        }}
        .sat-top-meta {{
            display:flex;
            flex-wrap:wrap;
            gap:10px 18px;
            color:{muted};
            font-size:0.9rem;
        }}
        .sat-top-side {{
            background:{bg_soft};
            border:1px solid {border};
            border-radius:16px;
            padding:14px 16px;
            margin: 0 0 1rem 0;
        }}
        .sat-top-progress-label {{
            color:{muted};
            font-size:0.8rem;
            font-weight:700;
            text-transform:uppercase;
            letter-spacing:0.04em;
            margin-bottom:8px;
        }}
        .sat-top-progress-track {{
            height:10px;
            background:{accent_soft};
            border-radius:999px;
            overflow:hidden;
        }}
        .sat-top-progress-fill {{
            height:100%;
            border-radius:999px;
            background:linear-gradient(90deg,{accent} 0%, #60A5FA 100%);
        }}
        .sat-top-progress-value {{
            color:{text};
            font-size:0.85rem;
            font-weight:700;
            margin-top:8px;
        }}
        .sat-top-flow {{
            display:grid;
            grid-template-columns:repeat(6,minmax(0,1fr));
            gap:10px;
            margin:0 0 1.1rem 0;
        }}
        .sat-top-step {{
            border:1px solid {border};
            background:{bg};
            border-radius:14px;
            padding:10px 12px;
            color:{text};
            display:flex;
            justify-content:space-between;
            gap:10px;
            font-size:0.83rem;
        }}
        .sat-top-step span {{
            color:{muted};
        }}
        .sat-top-step.is-done {{
            border-color:{accent};
            background:{accent_soft};
        }}
        @media (max-width: 980px) {{
            .sat-top-flow {{
                grid-template-columns:1fr 1fr 1fr;
            }}
        }}
        @media (max-width: 640px) {{
            .sat-top-flow {{
                grid-template-columns:1fr 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _page_help_text(page: str) -> str:
    descriptions = {
        "Proyectos": "Crea y revisa cursos/proyectos y sus checkpoints.",
        "Entrenamiento": "Carga el histórico, define variables y entrena modelos.",
        "Predicción": "Valida un dataset nuevo y genera alertas.",
        "Dashboard": "Analiza la cohorte y el caso individual seleccionado.",
        "Modelos": "Compara versiones y activa el mejor modelo.",
        "Exportación": "Descarga resultados en Excel o PDF.",
    }
    return descriptions.get(page, "")


def _render_model_summary(record: ModelRecord, artifact: dict[str, Any]) -> None:
    metrics = record.metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("AUC", f"{metrics.get('AUC', 0.0):.3f}")
    with c2:
        st.metric("Recall", f"{metrics.get('Recall', 0.0):.3f}")
    with c3:
        st.metric("Precisión", f"{metrics.get('Prec', 0.0):.3f}")
    with c4:
        st.metric("F1", f"{metrics.get('F1', 0.0):.3f}")

    info_col, config_col = st.columns([1.1, 1])
    with info_col:
        st.markdown(
            f"""
            <div class="sat-card">
                <div class="sat-badge">Modelo</div>
                <h3 style="margin:0 0 0.4rem;">{record.label}</h3>
                <p class="sat-muted" style="margin:0;">
                    Versión {record.version} · Algoritmo {record.algorithm} ·
                    Métrica objetivo {record.objective_metric.upper()} ·
                    Umbral {record.threshold:.3f}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with config_col:
        schema = artifact["feature_schema"]
        st.markdown(
            f"""
            <div class="sat-card">
                <div class="sat-badge">Esquema</div>
                <p style="margin:0 0 0.35rem;"><strong>ID:</strong> {schema['id_column']}</p>
                <p style="margin:0 0 0.35rem;"><strong>Nombre:</strong> {schema['name_column']}</p>
                <p style="margin:0 0 0.35rem;"><strong>Target:</strong> {schema['target_column']}</p>
                <p style="margin:0;"><strong>Variables:</strong> {len(schema['feature_order'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_key_value_table(title: str, payload: dict[str, Any]) -> None:
    rows = []
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False)
        elif value is None:
            rendered = "—"
        else:
            rendered = str(value)
        rows.append({"Campo": str(key), "Valor": rendered})
    st.markdown(f"#### {title}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _model_label(project_slug: str, checkpoint_slug: str, model_id: str) -> str:
    record = next((item for item in list_models(project_slug, checkpoint_slug) if item.model_id == model_id), None)
    if record is None:
        return f"Modelo {model_id[:8]}..."
    active = " · Activo" if record.is_active else ""
    return f"v{record.version} · {record.label}{active}"


def _payload_key(project_slug: str, checkpoint_slug: str, model_id: str) -> str:
    return f"{project_slug}:{checkpoint_slug}:{model_id}"


def _current_prediction_payload() -> dict[str, Any] | None:
    project_slug = st.session_state.selected_project_slug
    checkpoint_slug = st.session_state.selected_checkpoint_slug
    model_id = st.session_state.selected_model_id
    if not all([project_slug, checkpoint_slug, model_id]):
        return None
    return st.session_state.latest_predictions.get(_payload_key(project_slug, checkpoint_slug, model_id))


def _parse_json_payload(text: str, expect_lists: bool = False) -> dict[str, Any]:
    payload = json.loads(text) if text.strip() else {}
    if not isinstance(payload, dict):
        raise ValueError("El JSON debe representar un objeto.")
    if expect_lists:
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            normalized[key] = value if isinstance(value, list) else [value]
        return normalized
    return payload


def render_projects():
    st.markdown("## Proyectos")
    st.caption("Cada proyecto representa un curso o materia con sus propios checkpoints y modelos de predicción.")

    projects = list_projects()
    active_slug = st.session_state.selected_project_slug

    if not projects:
        st.markdown(
            "<div class='sat-note'><div class='sat-panel-title'>Sin proyectos todavía</div>"
            "Crea tu primer proyecto para empezar a entrenar modelos de predicción.</div>",
            unsafe_allow_html=True,
        )

    # Cards de proyectos
    for project in projects:
        checkpoints = list_checkpoints(project.slug)
        total_models = sum(len(list_models(project.slug, cp.slug)) for cp in checkpoints)
        is_active = project.slug == active_slug
        card_class = "sat-project-card is-active" if is_active else "sat-project-card"
        legacy_badge = " · <span style='color:#F59E0B;font-size:0.78rem;font-weight:700;'>LEGACY SAT</span>" if project.legacy_sat else ""
        desc = project.description or "Sin descripción"
        st.markdown(
            f"<div class='{card_class}'>"
            f"<div class='sat-project-title'>{project.name}{legacy_badge}</div>"
            f"<div style='color:#6B7280;font-size:0.85rem;margin:2px 0 6px;'>{desc}</div>"
            f"<div class='sat-project-meta'>"
            f"<span>📋 {len(checkpoints)} checkpoint{'s' if len(checkpoints) != 1 else ''}</span>"
            f"<span>🤖 {total_models} modelo{'s' if total_models != 1 else ''}</span>"
            f"<span>📅 {project.created_at[:10]}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            if st.button(
                f"{'✓ Seleccionado' if is_active else 'Seleccionar'} · {project.name}",
                key=f"sel_proj_{project.slug}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                _set_project(project.slug)
                st.rerun()
        with btn_col2:
            if is_active and st.button("→ Ir a Entrenamiento", key=f"go_train_{project.slug}", use_container_width=True):
                _set_page("Entrenamiento")
                st.rerun()

    st.markdown("---")
    with st.expander("➕ Crear nuevo proyecto", expanded=not projects):
        with st.form("create_project_form"):
            name = st.text_input("Nombre del proyecto", placeholder="ej: Cálculo I, Programación 101")
            description = st.text_area("Descripción", placeholder="Describe brevemente el curso o materia")
            submitted = st.form_submit_button("Crear proyecto", type="primary")
        if submitted:
            if not name.strip():
                st.error("Debes ingresar un nombre.")
            else:
                project = create_project(name=name, description=description)
                _set_project(project.slug)
                st.success(f"Proyecto '{project.name}' creado.")
                st.rerun()

    # Detalle del proyecto activo
    if active_slug:
        try:
            project = get_project(active_slug)
        except Exception:
            return
        st.markdown(f"### Detalle: {project.name}")
        checkpoints = list_checkpoints(project.slug)
        if checkpoints:
            for cp in checkpoints:
                models = list_models(project.slug, cp.slug)
                active_model = get_active_model(project.slug, cp.slug)
                active_label = f"v{active_model.version} · {active_model.label}" if active_model else "Sin modelo activo"
                target_display = cp.target_column if cp.target_column else "Sin configurar"
                st.markdown(
                    f"<div class='sat-card' style='margin-bottom:0.6rem;'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>"
                    f"<strong>{cp.name}</strong>"
                    f"<span class='sat-badge'>{cp.slug}</span></div>"
                    f"<div class='sat-project-meta'>"
                    f"<span>🎯 Target: {target_display}</span>"
                    f"<span>🔢 {len(cp.numeric_features)} numéricas · {len(cp.categorical_features)} categóricas</span>"
                    f"<span>🤖 {len(models)} modelo{'s' if len(models) != 1 else ''}</span>"
                    f"<span>✅ Activo: {active_label}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='sat-note'>Este proyecto no tiene checkpoints todavía. "
                "Ve a <strong>Entrenamiento</strong> para crear el primero.</div>",
                unsafe_allow_html=True,
            )

        if projects:
            st.markdown(
                "<div class='sat-next-step'>"
                "<div><div class='sat-next-step-text'>Siguiente: configurar y entrenar</div>"
                "<div class='sat-next-step-desc'>Crea un checkpoint, sube tus datos y entrena el primer modelo.</div></div>",
                unsafe_allow_html=True,
            )
            if st.button("→ Ir a Entrenamiento", type="primary"):
                _set_page("Entrenamiento")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def _render_training_stepper(step: int, checkpoint_ok: bool, dataset_ok: bool, variables_ok: bool) -> None:
    steps = [
        ("1", "Checkpoint"),
        ("2", "Dataset"),
        ("3", "Variables"),
        ("4", "Entrenar"),
    ]
    prereqs = [True, checkpoint_ok, dataset_ok, variables_ok]
    items = []
    for i, (num, label) in enumerate(steps):
        n = i + 1
        if n < step:
            state = "done"
            circle_content = "✓"
        elif n == step:
            state = "active"
            circle_content = num
        else:
            state = "locked"
            circle_content = num
        items.append(
            f"<div class='sat-wstep {state}'>"
            f"<div class='sat-wstep-circle'>{circle_content}</div>"
            f"<div class='sat-wstep-label'>{label}</div>"
            f"</div>"
        )
    st.markdown(
        f"<div class='sat-wizard-stepper'>{''.join(items)}</div>",
        unsafe_allow_html=True,
    )


def _build_column_preview(df: pd.DataFrame) -> str:
    rows_html = ""
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_pct = f"{df[col].isna().mean():.0%}"
        nunique = int(df[col].nunique(dropna=True))
        sample = str(df[col].dropna().iloc[0]) if df[col].notna().any() else "—"
        if len(sample) > 30:
            sample = sample[:27] + "..."
        rows_html += (
            f"<tr><td><strong>{col}</strong></td>"
            f"<td>{dtype}</td>"
            f"<td style='text-align:center'>{missing_pct}</td>"
            f"<td style='text-align:center'>{nunique}</td>"
            f"<td style='color:#6B7280'>{sample}</td></tr>"
        )
    return (
        "<table class='sat-col-preview-table'>"
        "<thead><tr><th>Columna</th><th>Tipo</th><th>% Nulos</th><th>Valores únicos</th><th>Ejemplo</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def render_training():
    st.markdown("## Entrenamiento")
    project_slug = st.session_state.selected_project_slug
    if not project_slug:
        st.info("Crea o selecciona un proyecto para empezar.")
        return
    project = get_project(project_slug)

    checkpoint_slug = st.session_state.selected_checkpoint_slug
    checkpoint = get_checkpoint(project_slug, checkpoint_slug) if checkpoint_slug else None
    has_dataset = st.session_state.train_dataset is not None
    has_variables = bool(
        checkpoint
        and checkpoint.id_column
        and checkpoint.target_column
        and (checkpoint.numeric_features or checkpoint.categorical_features)
    )
    # For legacy checkpoints the features live in the artifact, not the checkpoint record.
    # Consider variables done if there's an active model with a feature schema.
    if not has_variables and checkpoint_slug:
        active_model = get_active_model(project_slug, checkpoint_slug)
        if active_model is not None:
            has_variables = True

    step = st.session_state.training_step
    # Auto-advance only when variables are already fully configured (e.g. legacy checkpoints)
    if has_variables and step < 4:
        st.session_state.training_step = 4
        step = 4
    _render_training_stepper(step, bool(checkpoint_slug), has_dataset, has_variables)

    # ── PASO 1: Checkpoint ──────────────────────────────────────────────
    if step == 1:
        st.markdown(f"### Proyecto: **{project.name}**")
        checkpoints = list_checkpoints(project_slug)

        if checkpoints:
            st.markdown("**Checkpoints existentes** — selecciona uno o crea uno nuevo")
            for cp in checkpoints:
                models_count = len(list_models(project_slug, cp.slug))
                is_active = cp.slug == checkpoint_slug
                card_class = "sat-project-card is-active" if is_active else "sat-project-card"
                target_info = f"Target: {cp.target_column}" if cp.target_column else "Sin configurar"
                st.markdown(
                    f"<div class='{card_class}'>"
                    f"<div class='sat-project-title'>{cp.name}</div>"
                    f"<div class='sat-project-meta'>"
                    f"<span>📋 {target_info}</span>"
                    f"<span>🔢 {len(cp.numeric_features)} numéricas · {len(cp.categorical_features)} categóricas</span>"
                    f"<span>🤖 {models_count} modelo{'s' if models_count != 1 else ''}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                if st.button(f"Seleccionar '{cp.name}'", key=f"sel_cp_{cp.slug}", use_container_width=True):
                    _set_checkpoint(cp.slug)
                    st.session_state.training_step = 2
                    st.rerun()

        with st.expander("➕ Crear nuevo checkpoint", expanded=not checkpoints):
            with st.form("create_checkpoint_form"):
                name = st.text_input("Nombre del checkpoint", placeholder="ej: Semana 6, Parcial 1, Corte Medio")
                description = st.text_area("Descripción (opcional)", placeholder="Describe el momento del curso que representa este checkpoint")
                submitted = st.form_submit_button("Crear checkpoint", type="primary")
            if submitted:
                if not name.strip():
                    st.error("Debes ingresar un nombre para el checkpoint.")
                else:
                    cp = create_checkpoint(project_slug, name=name, description=description)
                    _set_checkpoint(cp.slug)
                    st.session_state.training_step = 2
                    st.success(f"Checkpoint '{cp.name}' creado.")
                    st.rerun()

        if checkpoint_slug:
            st.markdown("")
            if st.button("Continuar con checkpoint seleccionado →", type="primary", use_container_width=True):
                st.session_state.training_step = 2
                st.rerun()
        return

    # ── PASO 2: Dataset ──────────────────────────────────────────────────
    if step == 2:
        col_title, col_change = st.columns([3, 1])
        with col_title:
            st.markdown(f"### Checkpoint: **{checkpoint.name}**")
        with col_change:
            if st.button("↩ Cambiar checkpoint", use_container_width=True):
                st.session_state.training_step = 1
                st.rerun()

        if checkpoint.dataset_ref and not has_dataset:
            st.markdown(
                f"<div class='sat-note'><div class='sat-panel-title'>Dataset previo registrado</div>"
                f"Archivo: <code>{checkpoint.dataset_ref}</code><br>"
                f"Como recargaste la app, necesitas subir el archivo nuevamente para continuar al entrenamiento.</div>",
                unsafe_allow_html=True,
            )

        uploaded = st.file_uploader(
            "Sube el dataset histórico del checkpoint",
            type=["csv", "xlsx", "xls"],
            key="training_dataset_upload",
            help="Archivo con datos históricos de estudiantes incluyendo la variable objetivo (reprobó/pasó)",
        )
        if uploaded is not None:
            df_loaded = load_uploaded_dataset(uploaded)
            st.session_state.train_dataset = df_loaded
            st.session_state.train_dataset_name = uploaded.name
            has_dataset = True  # re-check after upload so button enables immediately

        if has_dataset:
            df = st.session_state.train_dataset
            st.markdown(
                f"<div class='sat-success-block'>"
                f"<div class='sat-success-title'>Dataset listo</div>"
                f"{len(df):,} filas · {len(df.columns)} columnas · Archivo: {getattr(st.session_state, 'train_dataset_name', 'cargado')}"
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Vista previa de columnas", expanded=True):
                st.markdown(_build_column_preview(df), unsafe_allow_html=True)

        col_back, col_next = st.columns([1, 2])
        with col_back:
            if st.button("← Atrás", use_container_width=True):
                st.session_state.training_step = 1
                st.rerun()
        with col_next:
            if st.button("Continuar a Variables →", type="primary", use_container_width=True, disabled=not has_dataset):
                st.session_state.training_step = 3
                st.rerun()
        return

    if not has_dataset and step < 4:
        st.warning("Primero carga un dataset en el paso 2.")
        if st.button("← Volver al paso 2"):
            st.session_state.training_step = 2
            st.rerun()
        return
    df = st.session_state.train_dataset  # may be None for legacy at step 4

    # ── PASO 3: Variables ───────────────────────────────────────────────
    if step == 3:
        st.markdown(f"### Asignar columnas para **{checkpoint.name}**")
        columns = df.columns.tolist()

        # Auto-detect button
        from sat_app.data import autodetect_mapping
        autodetect_keys = ["id_estudiante", "nombre", "reprobo"]
        if "autodetect_done" not in st.session_state:
            st.session_state.autodetect_done = False

        auto_col1, auto_col2 = st.columns([2, 1])
        with auto_col1:
            st.markdown(
                "<div class='sat-note'><div class='sat-panel-title'>Columnas de identidad y target</div>"
                "Selecciona qué columna es el ID único, el nombre del estudiante y la variable objetivo (0 = aprobó, 1 = reprobó).</div>",
                unsafe_allow_html=True,
            )
        with auto_col2:
            if st.button("Autodetectar columnas", use_container_width=True):
                mapping = autodetect_mapping(columns, autodetect_keys + ["semestre"])
                if mapping.get("id_estudiante") != "__MISSING__":
                    st.session_state["_auto_id"] = mapping["id_estudiante"]
                if mapping.get("nombre") != "__MISSING__":
                    st.session_state["_auto_name"] = mapping["nombre"]
                if mapping.get("reprobo") != "__MISSING__":
                    st.session_state["_auto_target"] = mapping["reprobo"]
                st.session_state.autodetect_done = True
                st.rerun()

        id_default = st.session_state.get("_auto_id", checkpoint.id_column) if checkpoint.id_column in columns or st.session_state.get("_auto_id") in columns else columns[0]
        name_default = st.session_state.get("_auto_name", checkpoint.name_column) if checkpoint.name_column in columns or st.session_state.get("_auto_name") in columns else columns[0]
        target_default = st.session_state.get("_auto_target", checkpoint.target_column) if checkpoint.target_column in columns or st.session_state.get("_auto_target") in columns else columns[0]
        numeric_defaults = [col for col in checkpoint.numeric_features if col in columns]
        categorical_defaults = [col for col in checkpoint.categorical_features if col in columns]

        def safe_idx(lst, val):
            return lst.index(val) if val in lst else 0

        with st.form("feature_assignment_form"):
            id_col, name_col = st.columns(2)
            with id_col:
                id_column = st.selectbox("Columna ID estudiante", columns, index=safe_idx(columns, id_default))
            with name_col:
                name_column = st.selectbox("Columna Nombre", columns, index=safe_idx(columns, name_default))
            target_column = st.selectbox(
                "Columna Target (0 = aprobó · 1 = reprobó)",
                columns,
                index=safe_idx(columns, target_default),
                help="Debe ser una columna binaria con valores 0 y 1 únicamente.",
            )

            st.markdown("---")
            st.markdown(
                "<div class='sat-note'><div class='sat-panel-title'>Variables del modelo</div>"
                "Selecciona las columnas que el modelo usará para predecir. No incluyas ID, nombre ni target.</div>",
                unsafe_allow_html=True,
            )
            feature_cols = st.columns(2)
            with feature_cols[0]:
                numeric_features = st.multiselect(
                    "Features numéricas",
                    columns,
                    default=numeric_defaults,
                    help="Variables continuas o discretas (notas, engagement, etc.)",
                )
            with feature_cols[1]:
                categorical_features = st.multiselect(
                    "Features categóricas",
                    columns,
                    default=categorical_defaults,
                    help="Variables con categorías o etiquetas (modalidad, grupo, etc.)",
                )
            save = st.form_submit_button("Guardar configuración", type="primary")

        # Validaciones en tiempo real
        errors = validate_role_columns(columns, id_column, name_column, target_column, numeric_features, categorical_features)
        target_ok, target_message = validate_binary_target(df, target_column)
        if not target_ok:
            errors.append(target_message)
        for warning in feature_quality_warnings(df, numeric_features, categorical_features):
            st.warning(warning)
        for error in errors:
            st.error(error)

        if save:
            if errors:
                st.error("Corrige los errores antes de guardar.")
            else:
                checkpoint.id_column = id_column
                checkpoint.name_column = name_column
                checkpoint.target_column = target_column
                checkpoint.numeric_features = numeric_features
                checkpoint.categorical_features = categorical_features
                checkpoint.dataset_ref = getattr(st.session_state, "train_dataset_name", checkpoint.dataset_ref)
                from sat_app.registry import save_checkpoint
                save_checkpoint(project_slug, checkpoint)
                st.success("Configuración guardada correctamente.")
                st.rerun()

        col_back, col_next = st.columns([1, 2])
        with col_back:
            if st.button("← Atrás", use_container_width=True):
                st.session_state.training_step = 2
                st.rerun()
        with col_next:
            checkpoint = get_checkpoint(project_slug, checkpoint_slug)
            vars_ready = bool(checkpoint.id_column and checkpoint.target_column and (checkpoint.numeric_features or checkpoint.categorical_features))
            if st.button("Continuar a Entrenamiento →", type="primary", use_container_width=True, disabled=not vars_ready):
                st.session_state.training_step = 4
                st.rerun()
        if not vars_ready:
            st.caption("Guarda la configuración de variables para poder continuar.")
        return

    if not checkpoint.id_column or not checkpoint.target_column:
        st.warning("Configura primero las variables del checkpoint en el paso 3.")
        if st.button("← Volver al paso 3"):
            st.session_state.training_step = 3
            st.rerun()
        return

    # ── PASO 4: Entrenamiento ───────────────────────────────────────────
    col_h, col_new_cp = st.columns([3, 1])
    with col_h:
        st.markdown(f"### Entrenamiento — **{checkpoint.name}**")
    with col_new_cp:
        if st.button("➕ Nuevo checkpoint", use_container_width=True):
            st.session_state.selected_checkpoint_slug = ""
            st.session_state.training_step = 1
            st.session_state.training_success = None
            st.rerun()

    active_record = get_active_model(project_slug, checkpoint_slug)
    if active_record is not None:
        st.markdown(
            f"<div class='sat-note'><div class='sat-panel-title'>Modelo activo actual</div>"
            f"v{active_record.version} · {active_record.label} · "
            f"AUC {active_record.metrics.get('AUC', 0):.3f} · "
            f"F1 {active_record.metrics.get('F1', 0):.3f}</div>",
            unsafe_allow_html=True,
        )

    if df is None:
        st.info(
            "Para entrenar un nuevo modelo para este checkpoint necesitas subir el dataset de entrenamiento. "
            "Usa el botón **← Atrás** para ir al paso 2 y cargar el archivo."
        )

    col1, col2 = st.columns(2)
    with col1:
        algorithm = st.selectbox("Algoritmo", ["LR", "DT", "RF", "XGB"],
            help="LR=Regresión Logística, DT=Árbol, RF=Random Forest, XGB=XGBoost")
        validation_mode = st.selectbox("Validación", ["Holdout", "K-Fold", "LOSO"],
            help="Holdout: división simple. K-Fold: validación cruzada. LOSO: por grupos (semestres).")
        balancing = st.selectbox("Balanceo de clases", ["none", "SMOTE", "SMOTEENN"],
            help="Útil cuando hay muchos más estudiantes que aprobaron vs reprobaron.")
        objective_metric = st.selectbox("Métrica objetivo", ["f1", "precision", "recall", "auc"],
            format_func=lambda key: {"f1": "F1 (recomendado)", "precision": "Precisión", "recall": "Recall", "auc": "AUC"}[key])
    with col2:
        hyperparameter_mode = st.selectbox("Hiperparámetros", ["auto", "fixed"],
            format_func=lambda x: "Búsqueda automática (recomendado)" if x == "auto" else "Valores fijos")
        random_state = st.number_input("Semilla aleatoria", min_value=1, value=42)
        holdout_size = st.slider("Proporción de prueba (Holdout)", min_value=0.1, max_value=0.4, value=0.2, step=0.05,
            format="%.0f%%", help="Porcentaje del dataset reservado para evaluar el modelo")
        search_iterations = st.slider("Combinaciones a explorar", min_value=4, max_value=20, value=8, step=2,
            help="Más iteraciones = mejor modelo potencial, pero más lento")

    if validation_mode == "LOSO":
        loso_cols = df.columns.tolist() if df is not None else []
        if loso_cols:
            group_column = st.selectbox("Columna de grupos (semestres) para LOSO", loso_cols)
        else:
            st.warning("Carga un dataset para poder usar LOSO.")
            group_column = None
    else:
        group_column = None

    default_space = SEARCH_SPACE_DEFAULTS[algorithm]
    with st.expander("Hiperparámetros avanzados (JSON)", expanded=False):
        st.caption("Modifica el espacio de búsqueda. Deja en blanco para usar valores por defecto.")
        if hyperparameter_mode == "fixed":
            raw_payload = st.text_area("Valores fijos (JSON)", value="{}", height=120)
        else:
            raw_payload = st.text_area(
                "Rangos de búsqueda (JSON)",
                value=json.dumps(default_space, ensure_ascii=False, indent=2),
                height=200,
            )

    col_back, col_train = st.columns([1, 2])
    with col_back:
        if st.button("← Atrás", use_container_width=True):
            # Go to step 3 only if dataset is loaded; otherwise skip to step 2
            st.session_state.training_step = 3 if has_dataset else 2
            st.rerun()
    with col_train:
        train_clicked = st.button("Entrenar modelo", type="primary", use_container_width=True)

    if train_clicked:
        if df is None:
            st.error("Debes cargar un dataset antes de entrenar. Ve al paso 2 para subir el archivo.")
            return
        try:
            payload = _parse_json_payload(raw_payload, expect_lists=(hyperparameter_mode == "auto"))
        except Exception as exc:
            st.error(f"JSON inválido: {exc}")
            return
        target_ok, target_message = validate_binary_target(df, checkpoint.target_column)
        if not target_ok:
            st.error(target_message)
            return
        errors = validate_role_columns(
            df.columns.tolist(),
            checkpoint.id_column,
            checkpoint.name_column,
            checkpoint.target_column,
            checkpoint.numeric_features,
            checkpoint.categorical_features,
        )
        if errors:
            for error in errors:
                st.error(error)
            return

        version = next_model_version(project_slug, checkpoint_slug)
        model_id = uuid4().hex[:12]
        config = TrainModelConfig(
            project_id=get_project(project_slug).project_id,
            checkpoint_id=checkpoint.checkpoint_id,
            model_id=model_id,
            version=version,
            dataset_ref=checkpoint.dataset_ref or getattr(st.session_state, "train_dataset_name", ""),
            id_column=checkpoint.id_column,
            name_column=checkpoint.name_column,
            target_column=checkpoint.target_column,
            numeric_features=list(checkpoint.numeric_features),
            categorical_features=list(checkpoint.categorical_features),
            algorithm=algorithm,
            validation_mode=validation_mode,
            balancing=balancing,
            objective_metric=objective_metric,
            hyperparameter_mode=hyperparameter_mode,
            hyperparameter_value_or_space=payload,
            group_column=group_column,
            random_state=int(random_state),
            holdout_size=float(holdout_size),
            search_iterations=int(search_iterations),
        )

        with st.spinner(f"Entrenando modelo {algorithm} con {search_iterations} combinaciones... esto puede tardar unos segundos."):
            artifact = train_model(df, config)

        record = ModelRecord(
            model_id=model_id,
            project_id=config.project_id,
            checkpoint_id=config.checkpoint_id,
            version=version,
            label=f"{algorithm} · {objective_metric.upper()}",
            algorithm=algorithm,
            objective_metric=objective_metric,
            threshold=float(artifact["training_summary"]["selected_threshold"]),
            metrics=dict(artifact["metrics"]),
            hyperparameters=dict(artifact["training_summary"]["hyperparameters"]),
            search_config={
                "mode": artifact["training_summary"]["hyperparameter_mode"],
                "space": artifact["training_summary"]["search_space"],
                "iterations": artifact["training_summary"]["search_iterations"],
            },
            validation_config={
                "mode": validation_mode,
                "group_column": group_column or "",
                "holdout_size": float(holdout_size),
            },
            dataset_ref=config.dataset_ref,
            artifact_path="",
            is_active=True,
        )
        saved_record = persist_model(project_slug, checkpoint_slug, record, artifact, activate=True)
        st.session_state.selected_model_id = model_id
        st.session_state.training_success = {
            "record": saved_record,
            "artifact": artifact,
            "prev_record": active_record,
            "artifact_path": saved_record.artifact_path,
        }
        st.rerun()

    # Mostrar resultado exitoso persistente
    success = st.session_state.get("training_success")
    if success and success["record"].model_id == st.session_state.selected_model_id:
        rec = success["record"]
        metrics = rec.metrics
        m = rec.metrics
        st.markdown(
            f"<div class='sat-success-block'>"
            f"<div class='sat-success-title'>Modelo entrenado y guardado correctamente</div>"
            f"v{rec.version} · {rec.label} · Umbral óptimo: {rec.threshold:.3f}<br>"
            f"<div class='sat-success-detail'>{success['artifact_path']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("AUC", f"{metrics.get('AUC', 0):.3f}")
        with m2:
            st.metric("Recall", f"{metrics.get('Recall', 0):.3f}")
        with m3:
            st.metric("Precisión", f"{metrics.get('Prec', 0):.3f}")
        with m4:
            st.metric("F1", f"{metrics.get('F1', 0):.3f}")

        if success["prev_record"] is not None:
            st.markdown("#### Comparación con modelo anterior")
            st.plotly_chart(
                metric_comparison_figure(success["prev_record"].metrics, rec.metrics, _chart_theme()),
                use_container_width=True,
            )

        st.markdown(
            "<div class='sat-next-step'>"
            "<div><div class='sat-next-step-text'>Modelo listo para predecir</div>"
            "<div class='sat-next-step-desc'>El modelo está activo y guardado. Sube un nuevo dataset para generar predicciones.</div></div>",
            unsafe_allow_html=True,
        )
        nav1, nav2 = st.columns(2)
        with nav1:
            if st.button("Ver en Modelos →", use_container_width=True):
                _set_page("Modelos")
                st.rerun()
        with nav2:
            if st.button("Ir a Predicción →", type="primary", use_container_width=True):
                _set_page("Predicción")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_prediction():
    st.markdown("## Predicción")
    project_slug = st.session_state.selected_project_slug
    checkpoint_slug = st.session_state.selected_checkpoint_slug
    model_id = st.session_state.selected_model_id
    if not all([project_slug, checkpoint_slug, model_id]):
        st.markdown(
            "<div class='sat-note'><div class='sat-panel-title'>Selecciona proyecto, checkpoint y modelo</div>"
            "Usa los selectores del encabezado para configurar el contexto activo antes de predecir.</div>",
            unsafe_allow_html=True,
        )
        return

    record, artifact = load_model_with_artifact(project_slug, checkpoint_slug, model_id)
    schema = artifact["feature_schema"]

    st.markdown(
        f"<div class='sat-note'><div class='sat-panel-title'>Modelo activo: v{record.version} · {record.label}</div>"
        f"Umbral: {record.threshold:.3f} · AUC {record.metrics.get('AUC',0):.3f} · F1 {record.metrics.get('F1',0):.3f}<br>"
        f"<span style='font-size:0.82rem;color:#6B7280'>Columnas requeridas: "
        f"{schema['id_column']}, {schema['name_column']}, "
        f"{', '.join(schema['feature_order'])}</span></div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Sube el dataset de estudiantes para predecir",
        type=["csv", "xlsx", "xls"],
        key="prediction_dataset_upload",
        help="El archivo debe contener las mismas columnas de features que el dataset de entrenamiento.",
    )
    if uploaded is not None:
        st.session_state.prediction_dataset = load_uploaded_dataset(uploaded)
        st.session_state.prediction_dataset_name = uploaded.name

    if st.session_state.prediction_dataset is None:
        st.caption("Carga un archivo CSV o Excel para continuar.")
        return

    df = st.session_state.prediction_dataset
    with st.expander("Vista previa del dataset", expanded=False):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        st.caption(f"{len(df):,} filas · {len(df.columns)} columnas")

    missing = validate_prediction_dataframe(df, artifact)
    if missing:
        st.error("Faltan columnas requeridas: " + ", ".join(missing))
        st.caption("El dataset debe contener las mismas variables que se usaron para entrenar.")
        return

    st.success(f"Dataset válido — {len(df):,} estudiantes listos para predecir.")

    if st.button("Ejecutar predicción", type="primary", use_container_width=True):
        with st.spinner("Generando predicciones..."):
            results, enriched = predict_dataset(artifact, df)
        prediction_id = uuid4().hex
        payload = {
            "record": record,
            "artifact": artifact,
            "results": results,
            "enriched": enriched,
            "prediction_id": prediction_id,
            "dataset_ref": getattr(st.session_state, "prediction_dataset_name", ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        st.session_state.latest_predictions[_payload_key(project_slug, checkpoint_slug, model_id)] = payload
        prediction_record = PredictionRunRecord(
            prediction_id=prediction_id,
            project_id=record.project_id,
            checkpoint_id=record.checkpoint_id,
            model_id=model_id,
            dataset_ref=payload["dataset_ref"],
            columns_validated=list(artifact["feature_schema"]["feature_order"]),
            row_count=len(results),
        )
        from sat_app.model_store import persist_prediction_record
        persist_prediction_record(project_slug, checkpoint_slug, prediction_record)
        st.rerun()

    # Mostrar resultados si hay predicción activa
    existing_payload = _current_prediction_payload()
    if existing_payload and existing_payload.get("prediction_id"):
        ep = existing_payload
        ep_results = ep["results"]
        summary = summarize_predictions(ep_results)
        st.markdown(
            f"<div class='sat-success-block'>"
            f"<div class='sat-success-title'>Predicción completada — {ep['created_at'][:16]}</div>"
            f"{summary['total']} estudiantes · "
            f"<span style='color:#EF4444;font-weight:700;'>{summary['alto']} alto</span> · "
            f"<span style='color:#F59E0B;font-weight:700;'>{summary['medio']} medio</span> · "
            f"<span style='color:#10B981;font-weight:700;'>{summary['bajo']} bajo</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(ep_results[["id", "name", "probabilidad", "nivel_riesgo", "alerta"]].head(20), use_container_width=True, hide_index=True)
        st.markdown(
            "<div class='sat-next-step'>"
            "<div><div class='sat-next-step-text'>Predicción lista para analizar</div>"
            "<div class='sat-next-step-desc'>Ve al dashboard para ver el análisis completo por estudiante.</div></div>",
            unsafe_allow_html=True,
        )
        if st.button("→ Ver Dashboard", type="primary"):
            _set_page("Dashboard")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def _render_risk_table_html(df: pd.DataFrame) -> str:
    risk_colors = {"ALTO": "#EF4444", "MEDIO": "#F59E0B", "BAJO": "#10B981"}
    rows_html = ""
    for _, row in df.sort_values("probabilidad", ascending=False).iterrows():
        nivel = str(row.get("nivel_riesgo", ""))
        color = risk_colors.get(nivel, "#6B7280")
        prob_pct = int(float(row.get("probabilidad", 0)) * 100)
        alerta = "⚠️" if row.get("alerta") else ""
        rows_html += (
            f"<tr>"
            f"<td><strong>{row.get('id', '—')}</strong></td>"
            f"<td>{row.get('name', '—')}</td>"
            f"<td>"
            f"<div class='sat-prob-bar-wrap'><div class='sat-prob-bar' style='width:{prob_pct}%;background:{color};'></div></div>"
            f"<small style='color:#6B7280'>{float(row.get('probabilidad',0)):.1%}</small>"
            f"</td>"
            f"<td><span style='background:{color};color:white;border-radius:999px;padding:2px 8px;font-size:0.73rem;font-weight:700;'>{nivel}</span> {alerta}</td>"
            f"</tr>"
        )
    return (
        "<table class='sat-risk-table'>"
        "<thead><tr><th>ID</th><th>Nombre</th><th>Probabilidad</th><th>Riesgo</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def render_dashboard():
    st.markdown("## Dashboard")
    payload = _current_prediction_payload()
    if payload is None:
        st.markdown(
            "<div class='sat-note'><div class='sat-panel-title'>Sin predicción activa</div>"
            "Ejecuta una predicción en la sección de <strong>Predicción</strong> para ver el dashboard.</div>",
            unsafe_allow_html=True,
        )
        if st.button("→ Ir a Predicción", type="primary"):
            _set_page("Predicción")
            st.rerun()
        return

    df = payload["enriched"]
    results = payload["results"]
    artifact = payload["artifact"]
    project_slug = st.session_state.selected_project_slug

    summary = summarize_predictions(results)

    # ── Métricas grandes ────────────────────────────────────────────────
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(
            f"<div class='sat-dash-stat sat-dash-stat-total'>"
            f"<div class='sat-kpi'>{summary['total']}</div>"
            f"<div class='sat-muted'>Total estudiantes</div></div>",
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            f"<div class='sat-dash-stat sat-dash-stat-alto'>"
            f"<div class='sat-kpi' style='color:#EF4444'>{summary['alto']}</div>"
            f"<div class='sat-muted'>Alto riesgo</div></div>",
            unsafe_allow_html=True,
        )
    with d3:
        st.markdown(
            f"<div class='sat-dash-stat sat-dash-stat-medio'>"
            f"<div class='sat-kpi' style='color:#F59E0B'>{summary['medio']}</div>"
            f"<div class='sat-muted'>Riesgo medio</div></div>",
            unsafe_allow_html=True,
        )
    with d4:
        st.markdown(
            f"<div class='sat-dash-stat sat-dash-stat-bajo'>"
            f"<div class='sat-kpi' style='color:#10B981'>{summary['bajo']}</div>"
            f"<div class='sat-muted'>Bajo riesgo</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # ── Tabs principales ────────────────────────────────────────────────
    tab_cohorte, tab_individual = st.tabs(["Vista Cohorte", "Vista Individual"])

    with tab_cohorte:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(risk_distribution_chart(results, _chart_theme()), use_container_width=True)
        with right:
            st.plotly_chart(probability_histogram(results, _chart_theme()), use_container_width=True)

        st.markdown("### Tabla de estudiantes")
        st.markdown(_render_risk_table_html(results), unsafe_allow_html=True)

        if project_slug == LEGACY_SAT_PROJECT_SLUG:
            st.markdown("### Análisis SAT")
            scatter_result = scatter_p1_vs_prob(df, _chart_theme())
            if scatter_result[0] is not None:
                st.plotly_chart(scatter_result[0], use_container_width=True)
            pct_df = percentile_table_data(df)
            if not pct_df.empty:
                st.dataframe(pct_df, use_container_width=True, hide_index=True)
            numeric_box = {
                feature_display_name(col): col
                for col in ["parcial_1", "parcial_2", "engagement_hasta_p1", "engagement_hasta_p2", "promedio_ams", "promedio_quices"]
                if col in df.columns
            }
            if numeric_box:
                chosen = st.selectbox("Variable a comparar por nivel de riesgo", list(numeric_box.keys()))
                st.plotly_chart(risk_boxplot(df, numeric_box[chosen], chosen, _chart_theme()), use_container_width=True)
            cp2_candidates = [v for k, v in st.session_state.latest_predictions.items() if k.startswith(f"{project_slug}:cp2:")]
            cp1_candidates = [v for k, v in st.session_state.latest_predictions.items() if k.startswith(f"{project_slug}:cp1:")]
            if cp1_candidates and cp2_candidates:
                fig, _ = cohort_trend_chart(cp1_candidates[-1]["enriched"], cp2_candidates[-1]["enriched"], _chart_theme())
                st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "<div class='sat-next-step'>"
            "<div><div class='sat-next-step-text'>¿Listo para exportar?</div>"
            "<div class='sat-next-step-desc'>Descarga los resultados en Excel o PDF para compartir con tu equipo.</div></div>",
            unsafe_allow_html=True,
        )
        if st.button("→ Ir a Exportación", type="primary"):
            _set_page("Exportación")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_individual:
        option_map = {f"{row['id']} — {row['name']}": row["student_key"] for _, row in df.iterrows()}
        labels = list(option_map.keys())
        if not labels:
            st.info("No hay estudiantes en esta predicción.")
        else:
            if st.session_state.selected_student_key not in option_map.values():
                st.session_state.selected_student_key = option_map[labels[0]]
            selected_label = st.selectbox(
                "Selecciona un estudiante",
                labels,
                index=list(option_map.values()).index(st.session_state.selected_student_key),
            )
            selected_key = option_map[selected_label]
            st.session_state.selected_student_key = selected_key
            student = df.loc[df["student_key"] == selected_key].iloc[0]
            factors = top_student_factors(artifact, df, student.name)

            prob = float(student["probabilidad"])
            nivel = str(student.get("nivel_riesgo", ""))
            risk_color = {"ALTO": "#EF4444", "MEDIO": "#F59E0B", "BAJO": "#10B981"}.get(nivel, "#6B7280")
            st.markdown(
                f"<div class='sat-card' style='margin-bottom:1rem;'>"
                f"<div style='display:flex;align-items:center;gap:12px;'>"
                f"<div style='font-size:2rem;font-weight:800;color:{risk_color}'>{prob:.1%}</div>"
                f"<div><div style='font-size:1.1rem;font-weight:700;'>{student.get('name','—')}</div>"
                f"<span style='background:{risk_color};color:white;border-radius:999px;padding:3px 10px;font-size:0.8rem;font-weight:700;'>{nivel}</span>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )

            detail_left, detail_right = st.columns([1, 1.2])
            with detail_left:
                st.plotly_chart(small_probability_gauge(prob, _chart_theme()), use_container_width=True)
                if not factors.empty:
                    st.dataframe(
                        factors[["feature_label", "sentido", "magnitud"]].rename(
                            columns={"feature_label": "Variable", "sentido": "Efecto", "magnitud": "Magnitud"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            with detail_right:
                st.plotly_chart(student_percentile_chart(df, selected_key, _chart_theme()), use_container_width=True)
                if not factors.empty:
                    st.plotly_chart(feature_importance_chart(factors, _chart_theme()), use_container_width=True)



def render_models():
    st.markdown("## Modelos")
    project_slug = st.session_state.selected_project_slug
    checkpoint_slug = st.session_state.selected_checkpoint_slug
    if not project_slug or not checkpoint_slug:
        st.info("Selecciona proyecto y checkpoint.")
        return

    models = list_models(project_slug, checkpoint_slug)
    if not models:
        st.markdown(
            "<div class='sat-note'><div class='sat-panel-title'>Sin modelos en este checkpoint</div>"
            "Ve a <strong>Entrenamiento</strong> para entrenar el primer modelo.</div>",
            unsafe_allow_html=True,
        )
        if st.button("→ Ir a Entrenamiento", type="primary"):
            _set_page("Entrenamiento")
            st.session_state.training_step = 4
            st.rerun()
        return

    # ── Timeline de versiones ────────────────────────────────────────────
    st.markdown("### Versiones entrenadas")
    timeline_html = "<div class='sat-model-timeline'>"
    for m in reversed(models):
        active_class = "is-active" if m.is_active else ""
        active_badge = "<span class='sat-active-badge'>● ACTIVO</span>" if m.is_active else "<span class='sat-inactive-badge'>INACTIVO</span>"
        timeline_html += (
            f"<div class='sat-model-timeline-item {active_class}'>"
            f"<span class='sat-model-version'>v{m.version}</span>"
            f"<span style='flex:1'>{m.label}</span>"
            f"<span style='color:#6B7280;font-size:0.8rem'>AUC {m.metrics.get('AUC',0):.3f} · F1 {m.metrics.get('F1',0):.3f}</span>"
            f"{active_badge}</div>"
        )
    timeline_html += "</div>"
    st.markdown(timeline_html, unsafe_allow_html=True)

    # ── Selector de modelo ────────────────────────────────────────────────
    model_ids = [m.model_id for m in models]
    default_idx = model_ids.index(st.session_state.selected_model_id) if st.session_state.selected_model_id in model_ids else 0

    selected_model_id = st.selectbox(
        "Modelo a inspeccionar",
        options=model_ids,
        format_func=lambda mid: _model_label(project_slug, checkpoint_slug, mid),
        key="models_page_selected_model_id",
        index=default_idx,
    )
    if st.session_state.selected_model_id != selected_model_id:
        st.session_state.selected_model_id = selected_model_id

    record, artifact = load_model_with_artifact(project_slug, checkpoint_slug, selected_model_id)

    # ── Header del modelo seleccionado ─────────────────────────────────
    active_badge_html = (
        "<span class='sat-active-badge'>● ACTIVO</span>"
        if record.is_active
        else "<span class='sat-inactive-badge'>INACTIVO</span>"
    )
    st.markdown(
        f"<div class='sat-card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>"
        f"<div style='font-size:1.15rem;font-weight:800;'>v{record.version} · {record.label}</div>"
        f"{active_badge_html}</div>"
        f"<div class='sat-project-meta'>"
        f"<span>🔧 {record.algorithm}</span>"
        f"<span>🎯 Objetivo: {record.objective_metric.upper()}</span>"
        f"<span>📐 Umbral: {record.threshold:.3f}</span>"
        f"<span>📅 {record.created_at[:16]}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # Métricas
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("AUC", f"{record.metrics.get('AUC', 0):.3f}")
    with m2:
        st.metric("Recall", f"{record.metrics.get('Recall', 0):.3f}")
    with m3:
        st.metric("Precisión", f"{record.metrics.get('Prec', 0):.3f}")
    with m4:
        st.metric("F1", f"{record.metrics.get('F1', 0):.3f}")

    overview_tab, config_tab, data_tab = st.tabs(["Resumen", "Configuración", "Predicciones"])

    with overview_tab:
        left, right = st.columns([1, 1])
        with left:
            _render_key_value_table("Métricas de validación", record.metrics)
        with right:
            _render_key_value_table(
                "Decisiones del modelo",
                {
                    "Umbral óptimo": round(float(record.threshold), 4),
                    "Métrica objetivo": record.objective_metric.upper(),
                    "Algoritmo": record.algorithm,
                    "Estado": "Activo" if record.is_active else "Inactivo",
                    "Entrenado el": record.created_at,
                },
            )

    with config_tab:
        left, right = st.columns([1, 1])
        with left:
            _render_key_value_table("Hiperparámetros finales", record.hyperparameters)
        with right:
            _render_key_value_table(
                "Búsqueda y validación",
                {
                    "Modo búsqueda": record.search_config.get("mode", "—"),
                    "Iteraciones": record.search_config.get("iterations", "—"),
                    "Validación": record.validation_config.get("mode", "—"),
                    "Dataset origen": record.dataset_ref or "—",
                },
            )
        schema = artifact["feature_schema"]
        st.dataframe(
            pd.DataFrame([
                {"Rol": "ID", "Columna": schema["id_column"]},
                {"Rol": "Nombre", "Columna": schema["name_column"]},
                {"Rol": "Target", "Columna": schema["target_column"]},
                {"Rol": "Numéricas", "Columna": ", ".join(schema["numeric_features"]) or "—"},
                {"Rol": "Categóricas", "Columna": ", ".join(schema["categorical_features"]) or "—"},
            ]),
            hide_index=True, use_container_width=True,
        )
        artifact_path_str = record.artifact_path or "No disponible"
        st.markdown(
            f"<div class='sat-success-block'>"
            f"<div class='sat-success-title'>Archivo del modelo guardado en:</div>"
            f"<div class='sat-success-detail'>{artifact_path_str}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with data_tab:
        runs = list_prediction_runs(project_slug, checkpoint_slug)
        if runs:
            st.dataframe(
                pd.DataFrame([
                    {
                        "ID predicción": run.prediction_id[:12] + "...",
                        "Modelo": run.model_id[:12] + "...",
                        "Dataset": run.dataset_ref or "—",
                        "Filas": run.row_count,
                        "Fecha": run.created_at[:16],
                    }
                    for run in runs
                ]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Este checkpoint no tiene corridas de predicción registradas todavía.")

    if not record.is_active:
        if st.button("Activar este modelo", type="primary", use_container_width=True):
            activate_model(project_slug, checkpoint_slug, selected_model_id)
            st.success("Modelo activado correctamente.")
            st.rerun()

    active = get_active_model(project_slug, checkpoint_slug)
    if active and active.model_id != record.model_id:
        st.markdown("#### Comparación con modelo activo")
        st.plotly_chart(metric_comparison_figure(active.metrics, record.metrics, _chart_theme()), use_container_width=True)


def render_export():
    st.markdown("## Exportación")
    payload = _current_prediction_payload()
    if payload is None:
        st.markdown(
            "<div class='sat-note'><div class='sat-panel-title'>Sin predicción para exportar</div>"
            "Ejecuta una predicción primero para poder descargar los resultados.</div>",
            unsafe_allow_html=True,
        )
        if st.button("→ Ir a Predicción", type="primary"):
            _set_page("Predicción")
            st.rerun()
        return

    project_slug = st.session_state.selected_project_slug
    checkpoint_slug = st.session_state.selected_checkpoint_slug
    model_id = st.session_state.selected_model_id
    project = get_project(project_slug)
    checkpoint = get_checkpoint(project_slug, checkpoint_slug)
    summary = summarize_predictions(payload["results"])

    st.markdown(
        f"<div class='sat-note'><div class='sat-panel-title'>{project.name} · {checkpoint.name}</div>"
        f"{summary['total']} estudiantes · "
        f"<span style='color:#EF4444;font-weight:700;'>{summary['alto']} alto riesgo</span> · "
        f"<span style='color:#F59E0B;font-weight:700;'>{summary['medio']} medio</span> · "
        f"<span style='color:#10B981;font-weight:700;'>{summary['bajo']} bajo</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    export_df = payload["results"][["id", "name", "probabilidad", "nivel_riesgo", "alerta", "umbral_modelo"]].copy()
    st.dataframe(export_df.head(20), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        excel_bytes = dataframe_to_excel_bytes(export_df)
        excel_clicked = st.download_button(
            "Descargar Excel",
            data=excel_bytes,
            file_name=f"{project_slug}_{checkpoint_slug}_{model_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col2:
        pdf_bytes = summary_pdf_bytes(
            export_df,
            title="Reporte SAT",
            subtitle=f"{project.name} · {checkpoint.name}",
        )
        pdf_clicked = st.download_button(
            "Generar PDF",
            data=pdf_bytes,
            file_name=f"{project_slug}_{checkpoint_slug}_{model_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    if excel_clicked or pdf_clicked:
        try:
            run = get_prediction_run(project_slug, checkpoint_slug, payload["prediction_id"])
            if excel_clicked:
                run.export_refs["excel"] = f"{project_slug}_{checkpoint_slug}_{model_id}.xlsx"
            if pdf_clicked:
                run.export_refs["pdf"] = f"{project_slug}_{checkpoint_slug}_{model_id}.pdf"
            from sat_app.model_store import persist_prediction_record
            persist_prediction_record(project_slug, checkpoint_slug, run)
        except Exception:
            pass
