from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from sat_app.charts import (
    cohort_heatmap,
    cp_scatter,
    feature_importance_chart,
    metric_comparison_figure,
    probability_histogram,
    risk_distribution_chart,
    small_probability_gauge,
)
from sat_app.config import CHECKPOINTS, CRITERIA_LABELS, RISK_COLORS, THEMES, build_base_style
from sat_app.data import (
    apply_mapping,
    autodetect_mapping,
    completeness_score,
    dataset_warnings,
    ensure_identity_columns,
    load_uploaded_dataset,
)
from sat_app.exporters import dataframe_to_excel_bytes, summary_pdf_bytes
from sat_app.models import (
    available_model_cards,
    compute_derived_fields,
    expand_required_columns,
    feature_display_name,
    load_all_metadata,
    load_bundle,
    predict_risk,
    summarize_predictions,
    top_student_factors,
)
from sat_app.retraining import RetrainConfig, run_retraining, save_retrained_model


def run_app():
    st.set_page_config(page_title="SAT Uniandes", page_icon=":bar_chart:", layout="wide")
    _init_state()
    st.markdown(build_base_style("Claro"), unsafe_allow_html=True)
    if st.session_state.pending_model_selection is not None:
        pending_checkpoint, pending_criterion = st.session_state.pending_model_selection
        st.session_state.sidebar_selected_checkpoint = pending_checkpoint
        st.session_state.sidebar_selected_criterion = pending_criterion
        st.session_state.selected_checkpoint = pending_checkpoint
        st.session_state.selected_criterion = pending_criterion
        st.session_state.pending_model_selection = None

    with st.sidebar:
        st.markdown("## SAT Uniandes")
        st.caption("Sistema de Alerta Temprana para docentes")
        page = st.radio(
            "Navegación",
            ["Inicio", "Modelos", "Carga y predicción", "Dashboard", "Estadísticas", "Exportación", "Gestión de modelos"],
        )
        checkpoint = st.selectbox(
            "Checkpoint activo",
            options=list(CHECKPOINTS.keys()),
            format_func=lambda key: f"{CHECKPOINTS[key]['label']} - {CHECKPOINTS[key]['title']}",
            key="sidebar_selected_checkpoint",
        )
        criterion = st.selectbox(
            "Criterio del modelo",
            options=["f1", "precision", "recall"],
            format_func=lambda key: CRITERIA_LABELS[key],
            key="sidebar_selected_criterion",
        )
        st.session_state.selected_checkpoint = checkpoint
        st.session_state.selected_criterion = criterion
        st.divider()
        st.caption("Estado")
        st.write(f"Archivo cargado: {'Sí' if st.session_state.raw_df is not None else 'No'}")
        st.write(f"Predicciones {checkpoint.upper()}: {'Sí' if checkpoint in st.session_state.predictions else 'No'}")

    if page == "Inicio":
        render_home()
    elif page == "Modelos":
        render_models()
    elif page == "Carga y predicción":
        render_upload_predict()
    elif page == "Dashboard":
        render_dashboard()
    elif page == "Estadísticas":
        render_statistics()
    elif page == "Exportación":
        render_export()
    else:
        render_model_management()


def _init_state():
    defaults = {
        "selected_checkpoint": "cp1",
        "selected_criterion": "f1",
        "sidebar_selected_checkpoint": "cp1",
        "sidebar_selected_criterion": "f1",
        "pending_model_selection": None,
        "raw_df": None,
        "mapped_df": None,
        "mapping": {},
        "predictions": {},
        "recent_runs": [],
        "dashboard_search": "",
        "dashboard_filters": ["ALTO", "MEDIO"],
        "selected_student": None,
        "retrained_artifact": None,
        "upload_step": 1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _chart_theme() -> str:
    return "Claro"


def _effective_theme_name() -> str:
    return "Claro"


def _criterion_card_style(criterion: str) -> tuple[str, str, str]:
    theme = THEMES["Claro"]
    palette = {
        "precision": ("#EFF6FF", "#0369A1", theme["text"]),
        "f1": ("#E0F2FE", "#0284C7", theme["text"]),
        "recall": ("#F0F9FF", "#06B6D4", theme["text"]),
    }
    return palette[criterion]


def _risk_tone(risk: str) -> tuple[str, str, str]:
    palette = {
        "ALTO": ("#EF4444", "#FEE2E2", "#7F1D1D"),
        "MEDIO": ("#F59E0B", "#FEF3C7", "#78350F"),
        "BAJO": ("#10B981", "#D1FAE5", "#064E3B"),
    }
    return palette.get(risk, ("#94A3B8", "#E5E7EB", "#334155"))


def _render_risk_badge(risk: str) -> str:
    border, bg, text = _risk_tone(risk)
    return (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{bg};color:{text};border:1px solid {border};font-weight:700;font-size:0.82rem;'>{risk.title()}</span>"
    )


def _format_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/D"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_parameter_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple, set)):
        text = ", ".join(_format_parameter_value(item) for item in value)
        return text[:120] + "..." if len(text) > 120 else text
    text = str(value)
    return text[:120] + "..." if len(text) > 120 else text


def _model_parameter_rows(bundle: dict[str, Any], card: dict[str, Any]) -> list[tuple[str, str]]:
    model = bundle["modelo"]
    params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
    preferred_keys = [
        "penalty",
        "C",
        "solver",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "n_estimators",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "max_features",
        "random_state",
    ]
    rows = [
        ("Familia", card["family"]),
        ("Criterio de selección", card["label"]),
        ("Umbral de alerta", f"{card['threshold']:.3f}"),
        ("Variables del modelo", str(len(card["features"]))),
    ]
    for key in preferred_keys:
        if key in params:
            rows.append((key, _format_parameter_value(params[key])))
    return rows


def _render_features_chips(features: list[str]) -> str:
    chips = "".join(f"<span class='sat-chip'>{feature_display_name(name)}</span>" for name in features)
    return f"<div>{chips}</div>"


def render_home():
    st.markdown(
        """
        <div class="sat-hero">
            <div class="sat-badge">MVP conectado a modelos reales</div>
            <h1 id="hero-title" style="margin:0;color:white !important;">Sistema de Alerta Temprana</h1>
            <p style="font-size:1.05rem;max-width:760px;color:white !important;margin-top:1rem;">
                Predice riesgo académico en semana 6 y semana 11, prioriza estudiantes y exporta reportes
                listos para seguimiento docente.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Use JavaScript to force the title color
    components.html("""
    <script>
    function forceWhiteTitle() {
        try {
            const title = document.getElementById('hero-title');
            if (title) {
                title.removeAttribute('style');
                title.setAttribute('style', 'color: white !important; text-shadow: 0 2px 4px rgba(0,0,0,0.1); font-weight: 700; font-size: 2.5rem; margin: 0;');
                // Also try adding a class
                title.classList.add('sat-hero-title-white');
            }
        } catch(e) {}
    }

    // Run immediately
    forceWhiteTitle();

    // Run repeatedly
    setInterval(forceWhiteTitle, 200);
    </script>
    <style>
    #hero-title {
        color: white !important;
    }
    .sat-hero-title-white {
        color: white !important;
    }
    </style>
    """, height=0)
    col1, col2, col3 = st.columns(3)
    steps = [
        ("1. Selecciona un modelo", "Compara precisión, balance y recall con métricas y gráficas exportadas."),
        ("2. Carga datos", "Sube Excel/CSV, mapea columnas y valida completitud antes de predecir."),
        ("3. Revisa resultados", "Tabla filtrable, analítica de cohorte y exportación a Excel o PDF."),
    ]
    for col, (title, description) in zip((col1, col2, col3), steps):
        with col:
            st.markdown(f"<div class='sat-card'><h3>{title}</h3><p>{description}</p></div>", unsafe_allow_html=True)

    st.markdown("### Análisis recientes")
    if st.session_state.recent_runs:
        st.dataframe(pd.DataFrame(st.session_state.recent_runs), width="stretch", hide_index=True)
    else:
        st.info("Aún no hay predicciones ejecutadas en esta sesión. Empieza en Carga y predicción.")


def render_models():
    st.markdown("## Selección de modelos")
    metadata_cards = available_model_cards()
    selected_checkpoint = st.session_state.selected_checkpoint
    selected_criterion = st.session_state.selected_criterion

    st.markdown(
        f"""
        <div class="sat-checkpoint-banner">
            <div class="sat-badge">Checkpoint activo</div>
            <h3 style="margin:0 0 0.35rem 0;">{CHECKPOINTS[selected_checkpoint]['label']} · {CHECKPOINTS[selected_checkpoint]['title']}</h3>
            <div class="sat-muted">El modelo seleccionado se resalta abajo y su contenido se muestra en una vista amplia para facilitar la lectura de las gráficas.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cp_cols = st.columns(len(CHECKPOINTS))
    for col, checkpoint in zip(cp_cols, CHECKPOINTS):
        active = checkpoint == selected_checkpoint
        with col:
            st.markdown(
                f"""
                <div class="sat-card sat-model-card {'is-selected' if active else 'is-dimmed'}">
                    <div class="sat-badge">{'Activo' if active else 'Disponible'}</div>
                    <h3 style="margin-top:0;">{CHECKPOINTS[checkpoint]['label']}</h3>
                    <p style="margin-bottom:0.35rem;"><strong>{CHECKPOINTS[checkpoint]['title']}</strong></p>
                    <p class="sat-muted">Selecciona este checkpoint para comparar sus tres versiones de modelo.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Ver este checkpoint" if not active else "Checkpoint actual",
                key=f"checkpoint_focus_{checkpoint}",
                disabled=active,
                use_container_width=True,
            ):
                st.session_state.pending_model_selection = (checkpoint, selected_criterion)
                st.rerun()

    st.markdown("### Versiones del modelo")
    cards = metadata_cards[selected_checkpoint]
    cols = st.columns(3)
    for col, card in zip(cols, cards):
        metrics = card["metrics"]
        is_selected = card["criterion"] == selected_criterion
        bg_color, border_color, text_color = _criterion_card_style(card["criterion"])
        badge_text = "Seleccionado" if is_selected else ("Recomendado" if card["criterion"] == "f1" else card["family"])
        with col:
            st.markdown(
                f"""
                <div class="sat-card sat-model-card {'is-selected' if is_selected else 'is-dimmed'}" style="background:{bg_color};border-color:{border_color};color:{text_color};">
                    <div class="sat-badge">{badge_text}</div>
                    <h3 style="margin-top:0;color:{text_color};">{card['label']}</h3>
                    <p style="color:{text_color};min-height:72px;">{card['description']}</p>
                    <div class="sat-model-highlight">
                        <div class="sat-metric-tile">
                            <div class="sat-metric-value">{metrics['AUC']:.3f}</div>
                            <div class="sat-metric-label">AUC</div>
                        </div>
                        <div class="sat-metric-tile">
                            <div class="sat-metric-value">{metrics['Recall']:.3f}</div>
                            <div class="sat-metric-label">Recall</div>
                        </div>
                        <div class="sat-metric-tile">
                            <div class="sat-metric-value">{metrics['F1']:.3f}</div>
                            <div class="sat-metric-label">F1</div>
                        </div>
                    </div>
                    <p class="sat-muted" style="color:{text_color};opacity:0.88;">Familia: {card['family']} · Umbral: {card['threshold']:.3f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Modelo actual" if is_selected else "Seleccionar modelo",
                key=f"use_{selected_checkpoint}_{card['criterion']}",
                disabled=is_selected,
                use_container_width=True,
            ):
                st.session_state.pending_model_selection = (selected_checkpoint, card["criterion"])
                st.rerun()

    selected_card = next(card for card in cards if card["criterion"] == selected_criterion)
    selected_bundle = load_bundle(selected_checkpoint, selected_criterion)

    st.markdown("### Vista ampliada del modelo seleccionado")
    st.markdown(
        f"""
        <div class="sat-card">
            <div class="sat-badge">Modelo activo</div>
            <h3 style="margin-top:0;">{selected_card['label']} · {CHECKPOINTS[selected_checkpoint]['label']} - {CHECKPOINTS[selected_checkpoint]['title']}</h3>
            <p>{selected_card['description']}</p>
            <div class="sat-model-highlight">
                <div class="sat-metric-tile"><div class="sat-metric-value">{selected_card['metrics']['AUC']:.3f}</div><div class="sat-metric-label">AUC</div></div>
                <div class="sat-metric-tile"><div class="sat-metric-value">{selected_card['metrics']['Prec']:.3f}</div><div class="sat-metric-label">Precisión</div></div>
                <div class="sat-metric-tile"><div class="sat-metric-value">{selected_card['metrics']['Recall']:.3f}</div><div class="sat-metric-label">Recall</div></div>
                <div class="sat-metric-tile"><div class="sat-metric-value">{selected_card['metrics']['F1']:.3f}</div><div class="sat-metric-label">F1</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    visual_tab, params_tab, features_tab = st.tabs(["Gráficas", "Parámetros", "Variables"])
    with visual_tab:
        if selected_card["image_main"].exists():
            st.image(str(selected_card["image_main"].resolve()), width="stretch", caption="Desempeño principal del modelo")
        comp_left, comp_right = st.columns(2)
        with comp_left:
            if selected_card["image_threshold"].exists():
                st.image(str(selected_card["image_threshold"].resolve()), width="stretch", caption="Comportamiento por umbral")
        with comp_right:
            if selected_card["image_comparison"].exists():
                st.image(str(selected_card["image_comparison"].resolve()), width="stretch", caption="Comparación general del checkpoint")
    with params_tab:
        rows = _model_parameter_rows(selected_bundle, selected_card)
        table_html = "".join(f"<tr><td><strong>{label}</strong></td><td>{value}</td></tr>" for label, value in rows)
        st.markdown(
            f"""
            <table class="sat-params-table">
                <thead><tr><th>Parámetro</th><th>Valor</th></tr></thead>
                <tbody>{table_html}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )
    with features_tab:
        st.markdown("#### Variables utilizadas")
        st.markdown(_render_features_chips(selected_card["features"]), unsafe_allow_html=True)
        metrics_df = pd.DataFrame(
            [
                {"Métrica": "AUC-ROC", "Valor": _format_metric(selected_card["metrics"]["AUC"])},
                {"Métrica": "Precisión", "Valor": _format_metric(selected_card["metrics"]["Prec"])},
                {"Métrica": "Recall", "Valor": _format_metric(selected_card["metrics"]["Recall"])},
                {"Métrica": "F1", "Valor": _format_metric(selected_card["metrics"]["F1"])},
            ]
        )
        st.dataframe(metrics_df, hide_index=True, width="stretch")


def render_upload_predict():
    st.markdown("## Carga de datos y predicción")
    step = st.radio(
        "",
        [1, 2, 3],
        horizontal=True,
        key="upload_step",
        format_func=lambda n: ["1. Carga tus datos", "2. Sigue al paso de mapeo", "3. Confirma tu base y sigue al dashboard"][n - 1],
        label_visibility="collapsed"
    )
    checkpoint = st.session_state.selected_checkpoint
    criterion = st.session_state.selected_criterion
    bundle = load_bundle(checkpoint, criterion)
    required = expand_required_columns(bundle["features"])

    if step == 1:
        uploaded = st.file_uploader("Arrastra o selecciona tu archivo .csv o .xlsx", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            df = load_uploaded_dataset(uploaded)
            st.session_state.raw_df = df
            st.session_state.mapping = autodetect_mapping(df.columns.tolist(), required)
            st.success(f"Archivo cargado: {uploaded.name} | filas: {len(df)} | columnas: {len(df.columns)}")
            st.dataframe(df.head(5), width="stretch")
        elif st.session_state.raw_df is not None:
            st.dataframe(st.session_state.raw_df.head(5), width="stretch")
    elif step == 2:
        if st.session_state.raw_df is None:
            st.warning("Primero carga un archivo en el paso 1.")
            return
        st.caption(
            f"{CHECKPOINTS[checkpoint]['label']} con modelo {CRITERIA_LABELS[criterion]} "
            f"requiere {len(required)} columnas base para calcular {len(bundle['features'])} variables."
        )
        if st.button("Auto-detectar columnas"):
            st.session_state.mapping = autodetect_mapping(st.session_state.raw_df.columns.tolist(), required)
        options = ["__MISSING__", *st.session_state.raw_df.columns.tolist()]
        with st.form("mapping_form"):
            updated = {}
            for canonical in required:
                updated[canonical] = st.selectbox(
                    f"{feature_display_name(canonical)}",
                    options=options,
                    index=options.index(st.session_state.mapping.get(canonical, "__MISSING__"))
                    if st.session_state.mapping.get(canonical, "__MISSING__") in options
                    else 0,
                    key=f"map_{canonical}",
                )
            submitted = st.form_submit_button("Guardar mapeo")
            if submitted:
                st.session_state.mapping = updated
                st.success("Mapeo actualizado.")
        mapped = apply_mapping(st.session_state.raw_df, st.session_state.mapping)
        completion = completeness_score(mapped, required)
        st.progress(min(int(completion), 100))
        st.write(f"Variables mapeadas o derivables: {(mapped.notna().any()).sum()}/{len(required)}")
        st.write(f"Completitud estimada: {completion}%")
    else:
        if st.session_state.raw_df is None:
            st.warning("Primero carga un archivo y define el mapeo.")
            return
        mapped = ensure_identity_columns(apply_mapping(st.session_state.raw_df, st.session_state.mapping))
        warnings = dataset_warnings(mapped, required)
        completion = completeness_score(mapped, required)
        st.markdown(
            f"""
            <div class="sat-card">
                <h3>Datos listos para predecir</h3>
                <p><strong>Checkpoint:</strong> {CHECKPOINTS[checkpoint]['label']}<br>
                <strong>Modelo:</strong> {CRITERIA_LABELS[criterion]}<br>
                <strong>Estudiantes:</strong> {len(mapped)}<br>
                <strong>Variables base:</strong> {len(required)}<br>
                <strong>Completitud:</strong> {completion}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if warnings:
            for warning in warnings:
                st.warning(warning)
        if st.button("Cargar y predecir", type="primary"):
            results, enriched = predict_risk(bundle, mapped)
            final_df = ensure_identity_columns(enriched)
            final_df["nombre_visible"] = final_df["student_label"]
            final_df["fecha_prediccion"] = datetime.now().strftime("%d %b %Y, %H:%M")
            st.session_state.predictions[checkpoint] = {
                "criterion": criterion,
                "bundle_name": CRITERIA_LABELS[criterion],
                "data": final_df,
                "summary": summarize_predictions(final_df),
            }
            st.session_state.recent_runs.insert(
                0,
                {
                    "fecha": final_df["fecha_prediccion"].iloc[0],
                    "checkpoint": CHECKPOINTS[checkpoint]["label"],
                    "modelo": CRITERIA_LABELS[criterion],
                    "estudiantes": len(final_df),
                    "alto_riesgo": int((final_df["nivel_riesgo"] == "ALTO").sum()),
                },
            )
            st.success("Predicción completada. Revisa el Dashboard y Estadísticas.")


def _get_active_predictions() -> dict[str, Any] | None:
    checkpoint = st.session_state.selected_checkpoint
    return st.session_state.predictions.get(checkpoint)


def _filter_dashboard_df(df: pd.DataFrame) -> pd.DataFrame:
    filters = st.session_state.dashboard_filters
    search = st.session_state.dashboard_search.strip().lower()
    filtered = df[df["nivel_riesgo"].isin(filters)] if filters else df.copy()
    if search:
        filtered = filtered[
            filtered["student_key"].str.lower().str.contains(search)
            | filtered["student_label"].str.lower().str.contains(search)
            | filtered["id_estudiante"].astype(str).str.lower().str.contains(search)
        ]
    return filtered.sort_values("probabilidad", ascending=False)


def render_dashboard():
    st.markdown("## Dashboard principal")
    payload = _get_active_predictions()
    if payload is None:
        st.info("Aún no hay predicciones para el checkpoint activo. Ve a Carga y predicción.")
        return
    df = payload["data"]
    summary = summarize_predictions(df)
    predominant_risk = max(
        [("ALTO", summary["alto"]), ("MEDIO", summary["medio"]), ("BAJO", summary["bajo"])],
        key=lambda item: item[1],
    )[0]
    st.caption(
        f"Predicciones de estudiantes - {CHECKPOINTS[st.session_state.selected_checkpoint]['label']} | "
        f"Modelo {payload['bundle_name']} | Fecha {datetime.now().strftime('%d %b %Y, %H:%M')}"
    )
    st.markdown(
        f"""
        <div class="sat-note">
            <div class="sat-panel-title">Lectura rápida de la cohorte</div>
            <div style="margin-bottom:8px;">{_render_risk_badge(predominant_risk)}</div>
            <div class="sat-muted">
                La cohorte actual tiene {summary['alto']} estudiantes en alto riesgo, {summary['medio']} en riesgo medio
                y {summary['bajo']} en bajo riesgo. Usa la tabla para priorizar casos y el panel derecho para entender
                por qué el modelo elevó o redujo el riesgo en cada estudiante.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    stats = [
        ("Total", summary["total"], "#94A3B8"),
        ("Alto riesgo", summary["alto"], RISK_COLORS["ALTO"]),
        ("Riesgo medio", summary["medio"], RISK_COLORS["MEDIO"]),
        ("Bajo riesgo", summary["bajo"], RISK_COLORS["BAJO"]),
    ]
    for col, (label, value, color) in zip(cols, stats):
        with col:
            st.markdown(
                f"<div class='sat-stat' style='--accent:{color};'><div class='sat-kpi'>{value}</div><div class='sat-muted'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Filtros y navegación")
    with st.form("dashboard_filters_form", clear_on_submit=False):
        filter_col, search_col, action_col = st.columns([1.4, 1.2, 0.6])
        with filter_col:
            selected = st.multiselect("Filtros de riesgo", ["ALTO", "MEDIO", "BAJO"], default=st.session_state.dashboard_filters)
        with search_col:
            search_value = st.text_input("Búsqueda por ID o nombre", value=st.session_state.dashboard_search)
        with action_col:
            submitted = st.form_submit_button("Aplicar", use_container_width=True)
        if submitted:
            st.session_state.dashboard_filters = selected
            st.session_state.dashboard_search = search_value

    quick1, quick2, quick3 = st.columns(3)
    with quick1:
        if st.button("Ver solo alto riesgo", width="stretch"):
            st.session_state.dashboard_filters = ["ALTO"]
            st.rerun()
    with quick2:
        if st.button("Ver alto + medio", width="stretch"):
            st.session_state.dashboard_filters = ["ALTO", "MEDIO"]
            st.rerun()
    with quick3:
        if st.button("Limpiar filtros", width="stretch"):
            st.session_state.dashboard_filters = ["ALTO", "MEDIO", "BAJO"]
            st.session_state.dashboard_search = ""
            st.rerun()

    filtered = _filter_dashboard_df(df)
    if filtered.empty:
        st.warning("No hay estudiantes con los filtros actuales.")
        return

    dashboard_tab, detail_tab = st.tabs(["Ranking", "Detalle del caso"])
    with dashboard_tab:
        st.markdown("### Ranking de estudiantes en riesgo")
        st.caption("Selecciona una fila para abrir el detalle y entender los factores del caso.")

        table_data = filtered.assign(Seleccionar=False)[["Seleccionar", "student_key", "student_label", "nivel_riesgo"]].copy()

        def get_bar_color(risk_level: str) -> str:
            colors = {"ALTO": "#EF4444", "MEDIO": "#FBBF24", "BAJO": "#10B981"}
            return colors.get(risk_level, "#94A3B8")

        # Create HTML bars for display
        def create_progress_bar_html(value: float, risk_level: str) -> str:
            color = get_bar_color(risk_level)
            percentage = int(value * 100)
            return f'<div style="width:100%;height:24px;background:#E5E7EB;border-radius:4px;overflow:hidden;"><div style="width:{percentage}%;height:100%;background:{color};"></div></div>'

        edited = st.data_editor(
            table_data,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            disabled=["student_key", "student_label", "nivel_riesgo"],
            column_config={
                "student_key": "ID",
                "student_label": "Nombre",
                "probabilidad": st.column_config.ProgressColumn("Probabilidad", min_value=0.0, max_value=1.0),
                "nivel_riesgo": "Categoría",
            },
            key="dashboard_table",
        )
        selected_rows = edited[edited["Seleccionar"]]

        # Display the colored bars
        st.markdown("### Probabilidades por estudiante")
        for idx, row in filtered.iterrows():
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                st.write(f"**{row['student_label']}**")
            with col2:
                st.markdown(create_progress_bar_html(row["probabilidad"], row["nivel_riesgo"]), unsafe_allow_html=True)
            with col3:
                st.write(f"{row['nivel_riesgo']}")

        selected_rows = edited[edited["Seleccionar"]]
    if not selected_rows.empty:
        st.session_state.selected_student = selected_rows.iloc[0]["student_key"]
    elif filtered.shape[0] and st.session_state.selected_student not in filtered["student_key"].tolist():
        st.session_state.selected_student = filtered.iloc[0]["student_key"]

    selected_key = st.session_state.selected_student or filtered.iloc[0]["student_key"]
    student = filtered.loc[filtered["student_key"] == selected_key].iloc[0]
    bundle = load_bundle(st.session_state.selected_checkpoint, st.session_state.selected_criterion)
    factors = top_student_factors(bundle, df, student.name)
    model_obj = bundle["modelo"]
    effect_explanation = (
        "En este modelo lineal, la contribución se obtiene al multiplicar cada variable estandarizada por su coeficiente."
        if hasattr(model_obj, "coef_")
        else "En este modelo no lineal, la contribución es una aproximación local basada en la desviación del estudiante respecto a la cohorte y la importancia global de cada variable."
    )

    with detail_tab:
        left, right = st.columns([1.05, 1.2])
        with left:
            st.markdown("### Resumen del caso")
            st.markdown(_render_risk_badge(student["nivel_riesgo"]), unsafe_allow_html=True)
            st.write(f"**ID anónimo:** {student['student_key']}")
            st.write(f"**Nombre visible:** {student['student_label']}")
            st.write(f"**Probabilidad estimada:** {float(student['probabilidad']):.2%}")
            st.plotly_chart(small_probability_gauge(float(student["probabilidad"]), _chart_theme()), width="stretch")
            st.dataframe(
                filtered[["student_key", "student_label", "probabilidad", "nivel_riesgo"]]
                .head(10)
                .rename(
                    columns={
                        "student_key": "ID",
                        "student_label": "Nombre",
                        "probabilidad": "Probabilidad",
                        "nivel_riesgo": "Riesgo",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        with right:
            st.markdown("### Qué explica este riesgo")
            st.markdown(
                """
                <div class="sat-note">
                    <div class="sat-panel-title">Cómo leer la gráfica de contribución</div>
                    <div class="sat-muted">
                        El cero queda en el centro. Las barras hacia la izquierda reducen el riesgo estimado y
                        las barras hacia la derecha lo aumentan. La longitud muestra el peso relativo de cada variable
                        en la predicción de este estudiante.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(effect_explanation)
            st.plotly_chart(feature_importance_chart(factors, _chart_theme()), width="stretch")
            st.dataframe(
                factors[["feature_label", "sentido", "magnitud"]].rename(
                    columns={
                        "feature_label": "Variable",
                        "sentido": "Lectura",
                        "magnitud": "Magnitud absoluta",
                    }
                ),
                hide_index=True,
                width="stretch",
            )


def render_statistics():
    st.markdown("## Estadísticas y analítica")
    payload = _get_active_predictions()
    if payload is None:
        st.info("Necesitas correr una predicción antes de ver la analítica.")
        return
    df = payload["data"]
    overview_tab, factors_tab, model_tab = st.tabs(["Cohorte", "Caso seleccionado", "Modelo"])
    bundle = load_bundle(st.session_state.selected_checkpoint, st.session_state.selected_criterion)
    with overview_tab:
        top_left, top_right = st.columns(2)
        with top_left:
            st.markdown("### Distribución de riesgo")
            st.plotly_chart(risk_distribution_chart(df, _chart_theme()), width="stretch")
        with top_right:
            st.markdown("### Histograma de probabilidades")
            st.plotly_chart(probability_histogram(df, _chart_theme()), width="stretch")

    with factors_tab:
        selected_key = st.session_state.selected_student or df.iloc[0]["student_key"]
        student = df.loc[df["student_key"] == selected_key].iloc[0]
        factors = top_student_factors(bundle, df, student.name)
        effect_explanation = (
            "Contribución exacta del puntaje lineal."
            if hasattr(bundle["modelo"], "coef_")
            else "Aproximación local basada en importancia global y desviación estandarizada."
        )
        fcol1, fcol2 = st.columns([1.25, 1])
        with fcol1:
            st.markdown(f"### Factores del caso {student['student_key']}")
            st.markdown(_render_risk_badge(student["nivel_riesgo"]), unsafe_allow_html=True)
            st.caption(effect_explanation)
            st.plotly_chart(feature_importance_chart(factors, _chart_theme()), width="stretch")
        with fcol2:
            st.markdown("### Explicación")
            st.markdown(
                """
                <div class="sat-note">
                    <div class="sat-panel-title">Interpretación docente</div>
                    <div class="sat-muted">
                        Este panel no muestra causalidad. Solo resume qué variables pesaron más en la predicción de este caso.
                        Si una barra sale a la derecha, esa variable aumentó la alerta. Si sale a la izquierda, la redujo.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                factors[["feature_label", "sentido", "magnitud"]].rename(
                    columns={"feature_label": "Variable", "sentido": "Lectura", "magnitud": "Magnitud absoluta"}
                ),
                hide_index=True,
                width="stretch",
            )

    with model_tab:
        st.markdown("### Métricas del modelo")
        meta = load_all_metadata()[st.session_state.selected_checkpoint][st.session_state.selected_criterion]["metricas_loso"]
        metrics_df = pd.DataFrame(
            {
                "Métrica": ["AUC-ROC", "Recall", "Precisión", "F1"],
                "Valor": [meta["AUC"], meta["Recall"], meta["Prec"], meta["F1"]],
            }
        )
        left, right = st.columns([0.8, 1.2])
        with left:
            st.dataframe(metrics_df, hide_index=True, width="stretch")
            st.caption(f"El modelo activo es {payload['bundle_name']} y usa {len(bundle['features'])} variables.")
        with right:
            if len(bundle["features"]) >= 2:
                st.markdown("#### Correlaciones de la cohorte")
                numeric_cols = [col for col in bundle["features"] if col in df.columns]
                if len(numeric_cols) >= 2:
                    st.plotly_chart(cohort_heatmap(df, numeric_cols, _chart_theme()), width="stretch")

    if "cp1" in st.session_state.predictions and "cp2" in st.session_state.predictions:
        st.markdown("### Comparación CP1 vs CP2")
        fig, merged = cp_scatter(
            st.session_state.predictions["cp1"]["data"],
            st.session_state.predictions["cp2"]["data"],
            _chart_theme(),
        )
        improved = int((merged["probabilidad_cp2"] < merged["probabilidad_cp1"]).sum())
        st.plotly_chart(fig, width="stretch")
        st.success(f"{improved} estudiantes mejoraron entre CP1 y CP2 en la sesión actual.")


def render_export():
    st.markdown("## Exportación")
    payload = _get_active_predictions()
    if payload is None:
        st.info("Primero genera predicciones para poder exportarlas.")
        return
    df = _filter_dashboard_df(payload["data"])
    st.caption("La exportación respeta los filtros y la búsqueda aplicados en el Dashboard.")
    options = st.multiselect(
        "Columnas a incluir",
        ["student_key", "student_label", "id_estudiante", "probabilidad", "nivel_riesgo", "alerta"],
        default=["student_key", "student_label", "probabilidad", "nivel_riesgo"],
    )
    export_df = df[options].copy()
    col1, col2 = st.columns(2)
    with col1:
        excel_bytes = dataframe_to_excel_bytes(export_df.rename(columns={"student_key": "id_anonimo"}))
        st.download_button(
            "Descargar Excel",
            data=excel_bytes,
            file_name=f"sat_{st.session_state.selected_checkpoint}_{st.session_state.selected_criterion}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    with col2:
        pdf_bytes = summary_pdf_bytes(
            export_df.assign(student_key=df["student_key"], probabilidad=df["probabilidad"], nivel_riesgo=df["nivel_riesgo"]),
            title="Reporte SAT",
            subtitle=f"{CHECKPOINTS[st.session_state.selected_checkpoint]['label']} - {payload['bundle_name']}",
        )
        st.download_button(
            "Generar PDF",
            data=pdf_bytes,
            file_name=f"sat_{st.session_state.selected_checkpoint}_{st.session_state.selected_criterion}.pdf",
            mime="application/pdf",
            width="stretch",
        )
    st.dataframe(export_df.head(15), width="stretch", hide_index=True)


def render_model_management():
    st.markdown("## Gestión de modelos")
    tab1, tab2 = st.tabs(["Estado de modelos", "Entrenar nuevo"])
    metadata = load_all_metadata()
    with tab1:
        rows = []
        for checkpoint, values in metadata.items():
            for criterion, item in values.items():
                rows.append(
                    {
                        "checkpoint": CHECKPOINTS[checkpoint]["label"],
                        "criterio": CRITERIA_LABELS[criterion],
                        "familia": item["familia"],
                        "ultimo_entrenamiento": item["fecha_entrenamiento"],
                        "n_train": item["n_train"],
                        "AUC": item["metricas_loso"]["AUC"],
                        "Recall": item["metricas_loso"]["Recall"],
                        "F1": item["metricas_loso"]["F1"],
                    }
                )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if st.session_state.retrained_artifact is not None:
            st.success("Hay un modelo reentrenado listo para activarse o guardarse.")
    with tab2:
        st.caption("Carga un dataset histórico. La app valida mínimo 100 filas y compara el nuevo modelo contra el actual.")
        uploaded = st.file_uploader("Dataset histórico", type=["csv", "xlsx", "xls"], key="retrain_file")
        if uploaded is not None:
            hist_df = load_uploaded_dataset(uploaded)
            st.dataframe(hist_df.head(5), width="stretch")
            if len(hist_df) < 100:
                st.error("El dataset histórico debe tener al menos 100 filas.")
                return

            checkpoint = st.selectbox("Checkpoint a reentrenar", list(CHECKPOINTS.keys()), format_func=lambda key: CHECKPOINTS[key]["label"], key="rt_cp")
            criterion = st.selectbox("Criterio objetivo", ["f1", "precision", "recall"], format_func=lambda key: CRITERIA_LABELS[key], key="rt_criterion")
            family = st.selectbox("Familia de modelo", ["LR", "DT", "RF", "XGB"])
            balancing = st.radio("Balanceo", ["none", "SMOTE", "SMOTEENN"], horizontal=True)
            validation = st.radio("Validación", ["Holdout", "K-Fold", "LOSO"], horizontal=True)
            target_column = st.selectbox("Columna objetivo (reprobo)", hist_df.columns.tolist())
            semester_column = st.selectbox("Columna de semestre para LOSO", ["", *hist_df.columns.tolist()])
            random_state = st.number_input("Random seed", min_value=1, value=42)
            test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

            bundle = load_bundle(checkpoint, criterion)
            hist_ready = compute_derived_fields(hist_df)
            feature_candidates = [col for col in bundle["features"] if col in hist_ready.columns]
            missing = [col for col in bundle["features"] if col not in hist_ready.columns]
            if missing:
                st.warning("Faltan columnas para el conjunto actual de variables: " + ", ".join(missing))
            if st.button("Reentrenar modelo", type="primary"):
                config = RetrainConfig(
                    checkpoint=checkpoint,
                    criterion=criterion,
                    model_family=family,
                    balancing=balancing,
                    validation=validation,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    target_column=target_column,
                    semester_column=semester_column or None,
                )
                artifact = run_retraining(hist_ready.dropna(subset=[target_column]), feature_candidates, config)
                st.session_state.retrained_artifact = artifact
                current_metrics = metadata[checkpoint][criterion]["metricas_loso"]
                st.plotly_chart(metric_comparison_figure(current_metrics, artifact["metrics"], _chart_theme()), width="stretch")
                verdict = (
                    "Mejor"
                    if artifact["metrics"]["F1"] > current_metrics["F1"]
                    else "Similar"
                    if abs(artifact["metrics"]["F1"] - current_metrics["F1"]) < 0.02
                    else "Peor"
                )
                st.write({"métricas_nuevas": artifact["metrics"], "veredicto": verdict})
                if st.button("Activar / guardar nuevo modelo"):
                    path = save_retrained_model(artifact)
                    st.success(f"Nuevo modelo guardado en {path}")
