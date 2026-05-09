from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sat_app.config import RISK_COLORS


def _apply_theme(fig, theme_name: str):
    dark = theme_name == "Oscuro"
    paper = "rgba(0,0,0,0)"
    plot = "#121A2B" if dark else "#FFFFFF"
    font = "#E5EEF9" if dark else "#1F2937"
    grid = "#314158" if dark else "#E5E7EB"
    axis_line = "#41536F" if dark else "#CBD5E1"
    fig.update_layout(
        paper_bgcolor=paper,
        plot_bgcolor=plot,
        font=dict(color=font),
        margin=dict(l=10, r=10, t=48, b=10),
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=axis_line, linecolor=axis_line, tickfont=dict(size=12), title_font=dict(size=13))
    fig.update_yaxes(gridcolor=grid, zerolinecolor=axis_line, linecolor=axis_line, tickfont=dict(size=12), title_font=dict(size=13))
    return fig


def risk_distribution_chart(df: pd.DataFrame, theme_name: str):
    counts = (
        df["nivel_riesgo"]
        .value_counts()
        .rename_axis("nivel_riesgo")
        .reset_index(name="estudiantes")
        .sort_values("nivel_riesgo")
    )
    fig = px.pie(
        counts,
        names="nivel_riesgo",
        values="estudiantes",
        color="nivel_riesgo",
        color_discrete_map=RISK_COLORS,
        hole=0.45,
    )
    fig.update_traces(
        textinfo="percent+label",
        marker=dict(line=dict(color="#FFFFFF" if theme_name != "Oscuro" else "#0F172A", width=2)),
        sort=False,
    )
    fig.update_layout(title="Distribución de estudiantes por nivel de riesgo")
    return _apply_theme(fig, theme_name)


def probability_histogram(df: pd.DataFrame, theme_name: str):
    fig = px.histogram(
        df,
        x="probabilidad",
        color="nivel_riesgo",
        nbins=10,
        color_discrete_map=RISK_COLORS,
        opacity=0.92,
    )
    median = float(df["probabilidad"].median()) if len(df) else 0
    fig.add_vline(x=median, line_dash="dash", line_color="#F97316", annotation_text="Mediana")
    fig.update_layout(
        title="Distribución de probabilidades estimadas",
        bargap=0.06,
        xaxis_title="Probabilidad estimada de reprobar",
        yaxis_title="Número de estudiantes",
    )
    return _apply_theme(fig, theme_name)


def cp_scatter(cp1_df: pd.DataFrame, cp2_df: pd.DataFrame, theme_name: str):
    merged = cp1_df[["student_key", "probabilidad"]].merge(
        cp2_df[["student_key", "probabilidad", "nivel_riesgo"]],
        on="student_key",
        suffixes=("_cp1", "_cp2"),
    )
    fig = px.scatter(
        merged,
        x="probabilidad_cp1",
        y="probabilidad_cp2",
        color="nivel_riesgo",
        color_discrete_map=RISK_COLORS,
        hover_data=["student_key"],
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#9CA3AF", dash="dot"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Comparación de probabilidad entre Checkpoint 1 y Checkpoint 2",
        margin=dict(l=10, r=10, t=48, b=10),
        xaxis_title="Probabilidad en Checkpoint 1",
        yaxis_title="Probabilidad en Checkpoint 2",
    )
    return _apply_theme(fig, theme_name), merged


def feature_importance_chart(df_factors: pd.DataFrame, theme_name: str):
    fig = px.bar(
        df_factors.iloc[::-1],
        x="impacto",
        y="feature_label",
        orientation="h",
        color="direccion",
        color_discrete_map={"incrementa": "#EF4444", "reduce": "#10B981"},
        text="sentido",
    )
    max_abs = float(np.abs(df_factors["impacto"]).max()) if len(df_factors) else 1.0
    fig.update_layout(
        title="Contribución de variables a la predicción",
        xaxis_title="Contribución al riesgo estimado",
        yaxis_title="",
    )
    fig.update_xaxes(range=[-(max_abs * 1.2), max_abs * 1.2], zeroline=True, zerolinewidth=2)
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Contribución: %{x:.3f}<br>%{text}<extra></extra>",
    )
    return _apply_theme(fig, theme_name)


def cohort_heatmap(df: pd.DataFrame, numeric_cols: list[str], theme_name: str):
    corr = df[numeric_cols].apply(pd.to_numeric, errors="coerce").corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=["#0F766E", "#F8FAFC", "#B91C1C"],
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(title="Correlación entre variables del modelo", coloraxis_colorbar_title="Correlación")
    return _apply_theme(fig, theme_name)


def metric_comparison_figure(old_metrics: dict, new_metrics: dict, theme_name: str):
    metric_names = ["AUC", "Recall", "Prec", "F1"]
    fig = go.Figure()
    fig.add_bar(name="Actual", x=metric_names, y=[old_metrics.get(m, 0) for m in metric_names], marker_color="#CBD5E1")
    fig.add_bar(name="Nuevo", x=metric_names, y=[new_metrics.get(m, 0) for m in metric_names], marker_color="#2563EB")
    fig.update_layout(title="Comparación de métricas del modelo", barmode="group", yaxis_title="Valor")
    return _apply_theme(fig, theme_name)


def small_probability_gauge(value: float, theme_name: str):
    dark = theme_name == "Oscuro"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value) * 100,
            number={"suffix": "%", "font": {"color": "#E5EEF9" if dark else "#1F2937"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#F97316" if value >= 0.60 else "#FBBF24" if value >= 0.35 else "#10B981"},
                "steps": [
                    {"range": [0, 35], "color": "#D1FAE5"},
                    {"range": [35, 60], "color": "#FEF3C7"},
                    {"range": [60, 100], "color": "#FEE2E2"},
                ],
            },
        )
    )
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig
