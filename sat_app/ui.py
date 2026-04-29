from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

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
            "Navegacion",
            ["Inicio", "Modelos", "Carga y prediccion", "Dashboard", "Estadisticas", "Exportacion", "Gestion de modelos"],
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
        st.write(f"Archivo cargado: {'Si' if st.session_state.raw_df is not None else 'No'}")
        st.write(f"Predicciones {checkpoint.upper()}: {'Si' if checkpoint in st.session_state.predictions else 'No'}")

    if page == "Inicio":
        render_home()
    elif page == "Modelos":
        render_models()
    elif page == "Carga y prediccion":
        render_upload_predict()
    elif page == "Dashboard":
        render_dashboard()
    elif page == "Estadisticas":
        render_statistics()
    elif page == "Exportacion":
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


def _theme() -> str:
    return "Claro"


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


def render_home():
    st.markdown(
        """
        <style>
            .sat-hero-title {
                color: white !important;
                margin: 0;
            }
        </style>
        <div class="sat-hero">
            <div class="sat-badge">MVP conectado a modelos reales</div>
            <h1 class="sat-hero-title">Sistema de Alerta Temprana</h1>
            <p style="font-size:1.05rem;max-width:760px;">
                Predice riesgo academico en semana 6 y semana 11, prioriza estudiantes y exporta reportes
                listos para seguimiento docente.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    steps = [
        ("1. Selecciona un modelo", "Compara precision, balance y recall con metricas y graficas exportadas."),
        ("2. Carga datos", "Sube Excel/CSV, mapea columnas y valida completitud antes de predecir."),
        ("3. Revisa resultados", "Tabla filtrable, analytics de cohorte y exportacion a Excel o PDF."),
    ]
    for col, (title, description) in zip((col1, col2, col3), steps):
        with col:
            st.markdown(f"<div class='sat-card'><h3>{title}</h3><p>{description}</p></div>", unsafe_allow_html=True)

    st.markdown("### Analisis recientes")
    if st.session_state.recent_runs:
        st.dataframe(pd.DataFrame(st.session_state.recent_runs), width="stretch", hide_index=True)
    else:
        st.info("Aun no hay predicciones ejecutadas en esta sesion. Empieza en Carga y prediccion.")


def render_models():
    st.markdown("## Seleccion de modelos")
    metadata_cards = available_model_cards()
    tabs = st.tabs([f"{CHECKPOINTS[key]['label']} - {CHECKPOINTS[key]['title']}" for key in CHECKPOINTS])
    for tab, checkpoint in zip(tabs, CHECKPOINTS):
        with tab:
            st.markdown("### Modelos disponibles")
            cols = st.columns(3)
            for col, card in zip(cols, metadata_cards[checkpoint]):
                metrics = card["metrics"]
                with col:
                    bg_color, border_color, text_color = _criterion_card_style(card["criterion"])
                    st.markdown(
                        f"""
                        <div class="sat-card" style="background:{bg_color};border:1px solid {border_color};color:{text_color};">
                            <div class="sat-badge">{'Recomendado' if card['criterion'] == 'f1' else card['family']}</div>
                            <h3 style="margin-top:0;color:{text_color};">{card['label']}</h3>
                            <p style="color:{text_color};">{card['description']}</p>
                            <p style="color:{text_color};"><strong>AUC:</strong> {metrics['AUC']:.3f}<br>
                            <strong>Recall:</strong> {metrics['Recall']:.3f}<br>
                            <strong>F1:</strong> {metrics['F1']:.3f}</p>
                            <p class="sat-muted" style="color:{text_color};opacity:0.85;">Familia: {card['family']} | Umbral: {card['threshold']:.3f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("Ver detalles", key=f"details_{checkpoint}_{card['criterion']}", width="stretch"):
                            st.session_state[f"model_details_{checkpoint}"] = card["criterion"]
                            st.rerun()
                    with col_btn2:
                        is_selected = (st.session_state.selected_checkpoint == checkpoint and
                                      st.session_state.selected_criterion == card["criterion"])
                        btn_style = "background-color: #E5E7EB; color: #1F2937;" if is_selected else ""
                        btn_text = "✓ Modelo activo" if is_selected else "Usar modelo"
                        if st.button(btn_text, key=f"use_{checkpoint}_{card['criterion']}", width="stretch"):
                            st.session_state.pending_model_selection = (checkpoint, card["criterion"])
                            st.rerun()

            st.divider()

            details_key = f"model_details_{checkpoint}"
            if details_key in st.session_state and st.session_state[details_key]:
                selected_criterion = st.session_state[details_key]
                selected_card = next(card for card in metadata_cards[checkpoint] if card["criterion"] == selected_criterion)
                metrics = selected_card["metrics"]
                bg_color, border_color, text_color = _criterion_card_style(selected_card["criterion"])

                st.markdown(f"### Detalles - {selected_card['label']}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Métricas LOSO")
                    metric_cols = st.columns(4)
                    for idx, (metric_name, metric_value) in enumerate(metrics.items()):
                        with metric_cols[idx % 4]:
                            st.metric(metric_name, f"{metric_value:.3f}")

                with col2:
                    st.markdown("#### Información del modelo")
                    st.write(f"**Familia:** {selected_card['family']}")
                    st.write(f"**Umbral:** {selected_card['threshold']:.3f}")
                    st.write(f"**Features utilizados:** {len(selected_card['features'])}")

                st.markdown("#### Features")
                features_display = [feature_display_name(name) for name in selected_card["features"]]
                st.write(", ".join(features_display))

                st.markdown("#### Gráficas")
                if selected_card["image_main"].exists():
                    st.image(str(selected_card["image_main"].resolve()), width="stretch", caption="Comparativa del modelo")
                if selected_card["image_threshold"].exists():
                    st.image(str(selected_card["image_threshold"].resolve()), width="stretch", caption="Análisis de umbral")
                if selected_card["image_comparison"].exists():
                    st.image(str(selected_card["image_comparison"].resolve()), width="stretch", caption="Comparación general")

                st.divider()
                if st.button("Usar este modelo", key=f"use_details_{checkpoint}_{selected_criterion}", width="stretch"):
                    st.session_state.pending_model_selection = (checkpoint, selected_criterion)
                    st.rerun()


def render_upload_predict():
    st.markdown("## Carga de datos y prediccion")
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
            f"requiere {len(required)} columnas base para calcular {len(bundle['features'])} features."
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
            st.success("Prediccion completada. Revisa el Dashboard y Estadisticas.")


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
        st.info("Aun no hay predicciones para el checkpoint activo. Ve a Carga y prediccion.")
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
            <div class="sat-panel-title">Lectura rapida de la cohorte</div>
            <div style="margin-bottom:8px;">{_render_risk_badge(predominant_risk)}</div>
            <div class="sat-muted">
                La cohorte actual tiene {summary['alto']} estudiantes en alto riesgo, {summary['medio']} en riesgo medio
                y {summary['bajo']} en bajo riesgo. Usa la tabla para priorizar casos y el panel derecho para entender
                por que el modelo elevó o redujo el riesgo en cada estudiante.
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

    st.markdown("### Filtros y navegacion")
    with st.form("dashboard_filters_form", clear_on_submit=False):
        filter_col, search_col, action_col = st.columns([1.4, 1.2, 0.6])
        with filter_col:
            selected = st.multiselect("Filtros de riesgo", ["ALTO", "MEDIO", "BAJO"], default=st.session_state.dashboard_filters)
        with search_col:
            search_value = st.text_input("Busqueda por ID o nombre", value=st.session_state.dashboard_search)
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

    dashboard_tab, detail_tab = st.tabs(["Priorizacion", "Detalle del caso"])
    with dashboard_tab:
        st.markdown("### Estudiantes priorizados")
        st.caption("Selecciona una fila para abrir el detalle y entender los factores del caso.")

        table_data = filtered.assign(Seleccionar=False)[["Seleccionar", "student_key", "student_label", "probabilidad", "nivel_riesgo"]].copy()

        def get_bar_color(risk_level: str) -> str:
            colors = {"ALTO": "#EF4444", "MEDIO": "#FBBF24", "BAJO": "#10B981"}
            return colors.get(risk_level, "#94A3B8")

        bar_columns = {}
        for idx, row in table_data.iterrows():
            bar_color = get_bar_color(row["nivel_riesgo"])
            bar_columns[idx] = bar_color

        edited = st.data_editor(
            table_data,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            disabled=["student_key", "student_label", "probabilidad", "nivel_riesgo"],
            column_config={
                "student_key": "ID",
                "student_label": "Nombre",
                "probabilidad": st.column_config.ProgressColumn(
                    "Probabilidad",
                    min_value=0.0,
                    max_value=1.0,
                    format=""
                ),
                "nivel_riesgo": "Categoria",
            },
            key="dashboard_table",
        )

        st.markdown("""
        <script>
        setTimeout(() => {
            const riskColors = {
                'ALTO': '#EF4444',
                'MEDIO': '#FBBF24',
                'BAJO': '#10B981'
            };

            const table = document.querySelector('[data-testid="stDataFrame"] table');
            if (table) {
                const rows = table.querySelectorAll('tbody tr');
                rows.forEach(row => {
                    const cells = row.querySelectorAll('td');
                    if (cells.length > 0) {
                        const lastCell = cells[cells.length - 1];
                        const riskLevel = lastCell.textContent.trim();
                        const color = riskColors[riskLevel] || '#94A3B8';

                        const progressBar = row.querySelector('[role="progressbar"]');
                        if (progressBar) {
                            const barDiv = progressBar.querySelector('div');
                            if (barDiv) {
                                barDiv.style.backgroundColor = color;
                            }
                        }
                    }
                });
            }
        }, 100);
        </script>
        """, unsafe_allow_html=True)

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
        "En este modelo lineal, la contribucion sale de multiplicar cada feature estandarizada por su coeficiente."
        if hasattr(model_obj, "coef_")
        else "En este modelo no lineal, la contribucion es una aproximacion local basada en la desviacion del estudiante respecto a la cohorte y la importancia global de cada variable."
    )

    with detail_tab:
        left, right = st.columns([1.05, 1.2])
        with left:
            st.markdown("### Resumen del caso")
            st.markdown(_render_risk_badge(student["nivel_riesgo"]), unsafe_allow_html=True)
            st.write(f"**ID anonimo:** {student['student_key']}")
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
            st.markdown("### Que explica este riesgo")
            st.markdown(
                """
                <div class="sat-note">
                    <div class="sat-panel-title">Como leer la grafica de efecto</div>
                    <div class="sat-muted">
                        Esta visual muestra contribuciones locales del modelo para este estudiante.
                        Rojo: variables que suben la alerta. Verde: variables que la moderan.
                        La magnitud indica el peso relativo de ese factor en la puntuacion del caso.
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
                        "magnitud": "Magnitud",
                    }
                ),
                hide_index=True,
                width="stretch",
            )


def render_statistics():
    st.markdown("## Estadisticas y analytics")
    payload = _get_active_predictions()
    if payload is None:
        st.info("Necesitas correr una prediccion antes de ver analytics.")
        return
    df = payload["data"]
    overview_tab, factors_tab, model_tab = st.tabs(["Cohorte", "Caso seleccionado", "Modelo"])
    bundle = load_bundle(st.session_state.selected_checkpoint, st.session_state.selected_criterion)
    with overview_tab:
        top_left, top_right = st.columns(2)
        with top_left:
            st.markdown("### Distribucion de riesgo")
            st.plotly_chart(risk_distribution_chart(df, _chart_theme()), width="stretch")
        with top_right:
            st.markdown("### Histograma de probabilidades")
            st.plotly_chart(probability_histogram(df, _chart_theme()), width="stretch")

    with factors_tab:
        selected_key = st.session_state.selected_student or df.iloc[0]["student_key"]
        student = df.loc[df["student_key"] == selected_key].iloc[0]
        factors = top_student_factors(bundle, df, student.name)
        effect_explanation = (
            "Contribucion exacta del score lineal."
            if hasattr(bundle["modelo"], "coef_")
            else "Aproximacion local basada en importancia global y desviacion estandarizada."
        )
        fcol1, fcol2 = st.columns([1.25, 1])
        with fcol1:
            st.markdown(f"### Factores del caso {student['student_key']}")
            st.markdown(_render_risk_badge(student["nivel_riesgo"]), unsafe_allow_html=True)
            st.caption(effect_explanation)
            st.plotly_chart(feature_importance_chart(factors, _chart_theme()), width="stretch")
        with fcol2:
            st.markdown("### Explicacion")
            st.markdown(
                """
                <div class="sat-note">
                    <div class="sat-panel-title">Interpretacion docente</div>
                    <div class="sat-muted">
                        Este panel no dice causalidad. Solo muestra que variables pesaron mas en la prediccion de este caso.
                        Si una barra sale en rojo, esa variable elevó la alerta. Si sale en verde, amortiguó el riesgo.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                factors[["feature_label", "sentido", "magnitud"]].rename(
                    columns={"feature_label": "Variable", "sentido": "Lectura", "magnitud": "Magnitud"}
                ),
                hide_index=True,
                width="stretch",
            )

    with model_tab:
        st.markdown("### Metricas del modelo")
        meta = load_all_metadata()[st.session_state.selected_checkpoint][st.session_state.selected_criterion]["metricas_loso"]
        metrics_df = pd.DataFrame(
            {
                "Metrica": ["AUC-ROC", "Recall", "Precision", "F1"],
                "Valor": [meta["AUC"], meta["Recall"], meta["Prec"], meta["F1"]],
            }
        )
        left, right = st.columns([0.8, 1.2])
        with left:
            st.dataframe(metrics_df, hide_index=True, width="stretch")
            st.caption(f"El modelo activo es {payload['bundle_name']} y usa {len(bundle['features'])} features.")
        with right:
            if len(bundle["features"]) >= 2:
                st.markdown("#### Correlaciones de la cohorte")
                numeric_cols = [col for col in bundle["features"] if col in df.columns]
                if len(numeric_cols) >= 2:
                    st.plotly_chart(cohort_heatmap(df, numeric_cols, _chart_theme()), width="stretch")

    if "cp1" in st.session_state.predictions and "cp2" in st.session_state.predictions:
        st.markdown("### Comparacion CP1 vs CP2")
        fig, merged = cp_scatter(
            st.session_state.predictions["cp1"]["data"],
            st.session_state.predictions["cp2"]["data"],
            _chart_theme(),
        )
        improved = int((merged["probabilidad_cp2"] < merged["probabilidad_cp1"]).sum())
        st.plotly_chart(fig, width="stretch")
        st.success(f"{improved} estudiantes mejoraron entre CP1 y CP2 en la sesion actual.")


def render_export():
    st.markdown("## Exportacion")
    payload = _get_active_predictions()
    if payload is None:
        st.info("Primero genera predicciones para poder exportarlas.")
        return
    df = _filter_dashboard_df(payload["data"])
    st.caption("La exportacion respeta los filtros y la busqueda aplicados en el Dashboard.")
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
        pdf_bytes = summary_pdf_bytes(export_df.assign(student_key=df["student_key"], probabilidad=df["probabilidad"], nivel_riesgo=df["nivel_riesgo"]),
                                      title="Reporte SAT",
                                      subtitle=f"{CHECKPOINTS[st.session_state.selected_checkpoint]['label']} - {payload['bundle_name']}")
        st.download_button(
            "Generar PDF",
            data=pdf_bytes,
            file_name=f"sat_{st.session_state.selected_checkpoint}_{st.session_state.selected_criterion}.pdf",
            mime="application/pdf",
            width="stretch",
        )
    st.dataframe(export_df.head(15), width="stretch", hide_index=True)


def render_model_management():
    st.markdown("## Gestion de modelos")
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
        st.caption("Carga un dataset historico. La app valida minimo 100 filas y compara el nuevo modelo contra el actual.")
        uploaded = st.file_uploader("Dataset historico", type=["csv", "xlsx", "xls"], key="retrain_file")
        if uploaded is not None:
            hist_df = load_uploaded_dataset(uploaded)
            st.dataframe(hist_df.head(5), width="stretch")
            if len(hist_df) < 100:
                st.error("El dataset historico debe tener al menos 100 filas.")
                return

            checkpoint = st.selectbox("Checkpoint a reentrenar", list(CHECKPOINTS.keys()), format_func=lambda key: CHECKPOINTS[key]["label"], key="rt_cp")
            criterion = st.selectbox("Criterio objetivo", ["f1", "precision", "recall"], format_func=lambda key: CRITERIA_LABELS[key], key="rt_criterion")
            family = st.selectbox("Familia de modelo", ["LR", "DT", "RF", "XGB"])
            balancing = st.radio("Balanceo", ["none", "SMOTE", "SMOTEENN"], horizontal=True)
            validation = st.radio("Validacion", ["Holdout", "K-Fold", "LOSO"], horizontal=True)
            target_column = st.selectbox("Columna objetivo (reprobo)", hist_df.columns.tolist())
            semester_column = st.selectbox("Columna de semestre para LOSO", ["", *hist_df.columns.tolist()])
            random_state = st.number_input("Random seed", min_value=1, value=42)
            test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

            bundle = load_bundle(checkpoint, criterion)
            hist_ready = compute_derived_fields(hist_df)
            feature_candidates = [col for col in bundle["features"] if col in hist_ready.columns]
            missing = [col for col in bundle["features"] if col not in hist_ready.columns]
            if missing:
                st.warning("Faltan columnas para el feature set actual: " + ", ".join(missing))
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
                st.write({"metricas_nuevas": artifact["metrics"], "veredicto": verdict})
                if st.button("Activar / guardar nuevo modelo"):
                    path = save_retrained_model(artifact)
                    st.success(f"Nuevo modelo guardado en {path}")
