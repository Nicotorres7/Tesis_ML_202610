from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sat_app.config import DISPLAY_LABELS, RISK_COLORS
from sat_app.explanations import cohort_feature_importance_data


def _apply_theme(fig, theme_name: str):
    dark = theme_name == "Oscuro"
    paper = "rgba(0,0,0,0)"
    plot = "#121A2B" if dark else "#FFFFFF"
    font = "#E5EEF9" if dark else "#1F2937"
    grid = "#314158" if dark else "#E5E7EB"
    axis_line = "#41536F" if dark else "#CBD5E1"
    hover_bg = "#1F2937" if dark else "#FFFFFF"
    hover_border = "#374151" if dark else "#D1D5DB"
    hover_font = "#F9FAFB" if dark else "#111827"
    fig.update_layout(
        paper_bgcolor=paper,
        plot_bgcolor=plot,
        font=dict(
            color=font,
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            size=12,
        ),
        margin=dict(l=10, r=10, t=52, b=10),
        legend_title_text="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor=hover_bg,
            bordercolor=hover_border,
            font=dict(color=hover_font, size=12, family="Inter, sans-serif"),
        ),
    )
    fig.update_xaxes(
        gridcolor=grid,
        zerolinecolor=axis_line,
        linecolor=axis_line,
        tickfont=dict(size=11),
        title_font=dict(size=12),
        showgrid=True,
        gridwidth=1,
    )
    fig.update_yaxes(
        gridcolor=grid,
        zerolinecolor=axis_line,
        linecolor=axis_line,
        tickfont=dict(size=11),
        title_font=dict(size=12),
        showgrid=True,
        gridwidth=1,
    )
    return fig


def risk_distribution_chart(df: pd.DataFrame, theme_name: str):
    dark = theme_name == "Oscuro"
    font_color = "#E5EEF9" if dark else "#1F2937"
    counts = (
        df["nivel_riesgo"]
        .value_counts()
        .rename_axis("nivel_riesgo")
        .reset_index(name="estudiantes")
        .sort_values("nivel_riesgo")
    )
    total = int(counts["estudiantes"].sum())
    fig = px.pie(
        counts,
        names="nivel_riesgo",
        values="estudiantes",
        color="nivel_riesgo",
        color_discrete_map=RISK_COLORS,
        hole=0.52,
    )
    fig.update_traces(
        textinfo="percent+label",
        textfont_size=13,
        marker=dict(
            line=dict(
                color="#FFFFFF" if not dark else "#0F172A",
                width=3,
            )
        ),
        sort=False,
        pull=[0.04, 0.04, 0.04],
        hovertemplate="<b>%{label}</b><br>%{value} estudiantes (%{percent})<extra></extra>",
    )
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:11px'>estudiantes</span>",
        x=0.5,
        y=0.5,
        font=dict(size=18, color=font_color),
        showarrow=False,
        xanchor="center",
        yanchor="middle",
    )
    fig.update_layout(title="Distribución de riesgo en la cohorte")
    return _apply_theme(fig, theme_name)


def probability_histogram(df: pd.DataFrame, theme_name: str):
    fig = px.histogram(
        df,
        x="probabilidad",
        color="nivel_riesgo",
        nbins=20,
        color_discrete_map=RISK_COLORS,
        opacity=0.88,
        barmode="overlay",
    )
    # Risk zone background bands
    fig.add_vrect(x0=0.0, x1=0.35, fillcolor="#10B981", opacity=0.07, layer="below", line_width=0)
    fig.add_vrect(x0=0.35, x1=0.60, fillcolor="#FBBF24", opacity=0.09, layer="below", line_width=0)
    fig.add_vrect(x0=0.60, x1=1.0, fillcolor="#EF4444", opacity=0.09, layer="below", line_width=0)
    # Threshold marker lines
    fig.add_vline(
        x=0.35, line_dash="dot", line_color="#FBBF24", line_width=1.5,
        annotation_text="Umbral medio", annotation_position="top right",
        annotation_font_size=10,
    )
    fig.add_vline(
        x=0.60, line_dash="dot", line_color="#EF4444", line_width=1.5,
        annotation_text="Umbral alto", annotation_position="top right",
        annotation_font_size=10,
    )
    median = float(df["probabilidad"].median()) if len(df) else 0
    fig.add_vline(
        x=median, line_dash="dash", line_color="#6366F1", line_width=2,
        annotation_text=f"Mediana: {median:.0%}",
        annotation_position="top left",
        annotation_font_size=10,
    )
    fig.update_layout(
        title="Distribución de probabilidades estimadas",
        bargap=0.04,
        xaxis_title="Probabilidad estimada de reprobar",
        yaxis_title="Número de estudiantes",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
    )
    return _apply_theme(fig, theme_name)


def cp_scatter(cp1_df: pd.DataFrame, cp2_df: pd.DataFrame, theme_name: str):
    """Kept for backward compatibility — delegates to cohort_trend_chart."""
    return cohort_trend_chart(cp1_df, cp2_df, theme_name)


def cohort_trend_chart(cp1_df: pd.DataFrame, cp2_df: pd.DataFrame, theme_name: str):
    """Enhanced CP1 vs CP2 scatter colored by student improvement direction."""
    dark = theme_name == "Oscuro"
    merged = cp1_df[["student_key", "probabilidad"]].merge(
        cp2_df[["student_key", "probabilidad", "nivel_riesgo"]],
        on="student_key",
        suffixes=("_cp1", "_cp2"),
    )
    merged["delta"] = merged["probabilidad_cp2"] - merged["probabilidad_cp1"]
    merged["direction"] = merged["delta"].apply(
        lambda d: "Mejoró" if d < -0.05 else ("Empeoró" if d > 0.05 else "Estable")
    )
    color_map = {"Mejoró": "#10B981", "Empeoró": "#EF4444", "Estable": "#94A3B8"}

    fig = px.scatter(
        merged,
        x="probabilidad_cp1",
        y="probabilidad_cp2",
        color="direction",
        color_discrete_map=color_map,
        hover_data={
            "student_key": True,
            "probabilidad_cp1": ":.1%",
            "probabilidad_cp2": ":.1%",
            "delta": ":.1%",
            "direction": False,
        },
        opacity=0.78,
        labels={"direction": "Tendencia"},
    )
    # Marker size and border
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color="rgba(255,255,255,0.5)")))

    # Identity line (no change reference)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#9CA3AF", dash="dot", width=1.5),
            showlegend=False,
            name="Sin cambio",
            hoverinfo="skip",
        )
    )
    # Quadrant shading
    fig.add_hrect(y0=0.60, y1=1.0, fillcolor="#EF4444", opacity=0.05, layer="below", line_width=0)
    fig.add_vrect(x0=0.60, x1=1.0, fillcolor="#EF4444", opacity=0.05, layer="below", line_width=0)
    # Zone labels
    fig.add_annotation(x=0.15, y=0.90, text="Empeoró", showarrow=False,
                       font=dict(size=10, color="#EF4444"), opacity=0.8)
    fig.add_annotation(x=0.88, y=0.10, text="Mejoró", showarrow=False,
                       font=dict(size=10, color="#10B981"), opacity=0.8)
    fig.update_layout(
        title="Evolución de probabilidad: CP1 → CP2",
        xaxis_title="Probabilidad en Checkpoint 1",
        yaxis_title="Probabilidad en Checkpoint 2",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(range=[0, 1], tickformat=".0%"),
    )
    return _apply_theme(fig, theme_name), merged


def feature_importance_chart(df_factors: pd.DataFrame, theme_name: str):
    dark = theme_name == "Oscuro"
    axis_line = "#41536F" if dark else "#9CA3AF"
    fig = px.bar(
        df_factors.iloc[::-1],
        x="impacto",
        y="feature_label",
        orientation="h",
        color="direccion",
        color_discrete_map={"incrementa": "#EF4444", "reduce": "#10B981"},
        text=df_factors.iloc[::-1]["impacto"].apply(lambda v: f"{v:+.3f}"),
    )
    max_abs = float(np.abs(df_factors["impacto"]).max()) if len(df_factors) else 1.0
    # Background zone shading
    fig.add_vrect(
        x0=0, x1=max_abs * 1.2,
        fillcolor="#EF4444", opacity=0.04, layer="below", line_width=0,
    )
    fig.add_vrect(
        x0=-max_abs * 1.2, x1=0,
        fillcolor="#10B981", opacity=0.04, layer="below", line_width=0,
    )
    fig.update_layout(
        title="Contribución de variables a la predicción",
        xaxis_title="Contribución al riesgo estimado",
        yaxis_title="",
        height=320,
    )
    fig.update_xaxes(
        range=[-(max_abs * 1.2), max_abs * 1.2],
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor=axis_line,
        tickformat=".3f",
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>Contribución: %{x:.3f}<extra></extra>",
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
    fig.update_layout(
        title="Correlación entre variables del modelo",
        coloraxis_colorbar_title="Corr.",
    )
    return _apply_theme(fig, theme_name)


def metric_comparison_figure(old_metrics: dict, new_metrics: dict, theme_name: str):
    metric_names = ["AUC", "Recall", "Prec", "F1"]
    dark = theme_name == "Oscuro"
    fig = go.Figure()
    fig.add_bar(
        name="Actual",
        x=metric_names,
        y=[old_metrics.get(m, 0) for m in metric_names],
        marker_color="#CBD5E1" if not dark else "#334155",
        marker_line_width=0,
    )
    fig.add_bar(
        name="Nuevo",
        x=metric_names,
        y=[new_metrics.get(m, 0) for m in metric_names],
        marker_color="#2563EB",
        marker_line_width=0,
    )
    fig.update_layout(
        title="Comparación de métricas del modelo",
        barmode="group",
        yaxis_title="Valor",
        yaxis=dict(range=[0, 1]),
        bargap=0.2,
        bargroupgap=0.08,
    )
    return _apply_theme(fig, theme_name)


def small_probability_gauge(value: float, theme_name: str):
    dark = theme_name == "Oscuro"
    pct = float(value) * 100
    bar_color = "#EF4444" if value >= 0.60 else "#FBBF24" if value >= 0.35 else "#10B981"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={
                "suffix": "%",
                "font": {"color": "#E5EEF9" if dark else "#1F2937", "size": 28},
            },
            delta={
                "reference": 50,
                "relative": False,
                "increasing": {"color": "#EF4444"},
                "decreasing": {"color": "#10B981"},
                "valueformat": ".1f",
            },
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#9CA3AF" if not dark else "#41536F",
                    "tickfont": {"size": 10},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "#D1FAE5" if not dark else "rgba(16,185,129,0.18)"},
                    {"range": [35, 60], "color": "#FEF3C7" if not dark else "rgba(251,191,36,0.18)"},
                    {"range": [60, 100], "color": "#FEE2E2" if not dark else "rgba(239,68,68,0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#6B7280", "width": 2},
                    "thickness": 0.8,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def student_percentile_chart(df: pd.DataFrame, student_key: str, theme_name: str):
    """Scatter showing selected student's position within the cohort probability distribution."""
    dark = theme_name == "Oscuro"
    font_color = "#E5EEF9" if dark else "#1F2937"

    sorted_df = df.sort_values("probabilidad").reset_index(drop=True)
    sorted_df["rank"] = range(1, len(sorted_df) + 1)
    student_row = sorted_df[sorted_df["student_key"] == student_key]

    fig = go.Figure()
    # Background risk bands
    fig.add_vrect(x0=0.35, x1=0.60, fillcolor="#FBBF24", opacity=0.07, layer="below", line_width=0)
    fig.add_vrect(x0=0.60, x1=1.0, fillcolor="#EF4444", opacity=0.07, layer="below", line_width=0)

    # All cohort students as small dots
    fig.add_trace(
        go.Scatter(
            x=sorted_df["probabilidad"],
            y=sorted_df["rank"],
            mode="markers",
            marker=dict(
                color=[RISK_COLORS.get(r, "#94A3B8") for r in sorted_df["nivel_riesgo"]],
                size=7,
                opacity=0.45,
                line=dict(width=0),
            ),
            hovertemplate="%{x:.0%} — rango %{y}<extra></extra>",
            name="Cohorte",
        )
    )
    # Highlighted selected student
    if not student_row.empty:
        s = student_row.iloc[0]
        percentile = int(s["rank"] / len(sorted_df) * 100)
        fig.add_trace(
            go.Scatter(
                x=[s["probabilidad"]],
                y=[s["rank"]],
                mode="markers+text",
                marker=dict(
                    color=RISK_COLORS.get(s["nivel_riesgo"], "#94A3B8"),
                    size=16,
                    symbol="diamond",
                    line=dict(color=font_color, width=2),
                ),
                text=[f"P{percentile}"],
                textposition="middle right",
                textfont=dict(color=font_color, size=12, family="Inter, sans-serif"),
                name=str(s["student_key"]),
                hovertemplate=(
                    f"<b>{s['student_key']}</b><br>"
                    f"Probabilidad: {s['probabilidad']:.0%}<br>"
                    f"Percentil: {percentile}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Posición del estudiante en la cohorte",
        xaxis_title="Probabilidad estimada",
        yaxis_title="Rango en cohorte",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        showlegend=False,
        height=280,
    )
    return _apply_theme(fig, theme_name)


def risk_boxplot(df: pd.DataFrame, feature_col: str, feature_label: str, theme_name: str):
    """Box plot comparing a numeric feature distribution across risk levels."""
    dark = theme_name == "Oscuro"
    plot_df = df[["nivel_riesgo", feature_col]].copy()
    plot_df[feature_col] = pd.to_numeric(plot_df[feature_col], errors="coerce")
    plot_df = plot_df.dropna()
    order = ["ALTO", "MEDIO", "BAJO"]
    plot_df["nivel_riesgo"] = pd.Categorical(plot_df["nivel_riesgo"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("nivel_riesgo")

    fig = px.box(
        plot_df,
        x="nivel_riesgo",
        y=feature_col,
        color="nivel_riesgo",
        color_discrete_map=RISK_COLORS,
        points="outliers",
        notched=False,
        labels={"nivel_riesgo": "Nivel de riesgo", feature_col: feature_label},
    )
    fig.update_traces(
        marker=dict(size=5, opacity=0.6),
        line_width=2,
        hovertemplate="<b>%{x}</b><br>" + feature_label + ": %{y:.2f}<extra></extra>",
    )
    fig.update_layout(
        title=f"Distribución de {feature_label} por nivel de riesgo",
        showlegend=False,
        xaxis_title="",
        yaxis_title=feature_label,
        height=320,
    )
    return _apply_theme(fig, theme_name)


def cohort_feature_importance(bundle: dict, df: pd.DataFrame, theme_name: str):
    """Global feature importance bar chart at cohort level (not per-student)."""
    dark = theme_name == "Oscuro"
    imp_df = cohort_feature_importance_data(bundle).rename(columns={"feature_label": "label"})
    imp_df = imp_df.sort_values("importance", ascending=True).tail(10)

    bar_color = "#60A5FA" if dark else "#2563EB"
    fig = go.Figure(go.Bar(
        x=imp_df["importance"],
        y=imp_df["label"],
        orientation="h",
        marker=dict(
            color=imp_df["importance"],
            colorscale=[[0, "#10B981"], [0.5, "#FBBF24"], [1, "#EF4444"]],
            showscale=False,
            line_width=0,
        ),
        hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Variables más influyentes en la cohorte",
        xaxis_title="Importancia global",
        yaxis_title="",
        height=360,
        margin=dict(l=10, r=20, t=52, b=10),
    )
    return _apply_theme(fig, theme_name)


def scatter_p1_vs_prob(df: pd.DataFrame, theme_name: str):
    """Scatter plot: Parcial 1 grade vs failure probability, colored by risk level."""
    p1_col = next((c for c in ["parcial_1", "p1_vs_media", "log_eng_p1"] if c in df.columns), None)
    if p1_col is None:
        return None, None

    plot_df = df[["student_key", "student_label", "nivel_riesgo", "probabilidad", p1_col]].copy()
    plot_df[p1_col] = pd.to_numeric(plot_df[p1_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[p1_col, "probabilidad"])

    label_map = {
        "parcial_1": "Nota Parcial 1",
        "p1_vs_media": "P1 vs media cohorte (z-score)",
        "log_eng_p1": "Log Engagement hasta P1",
    }
    x_label = label_map.get(p1_col, p1_col)

    fig = px.scatter(
        plot_df,
        x=p1_col,
        y="probabilidad",
        color="nivel_riesgo",
        color_discrete_map=RISK_COLORS,
        hover_data={"student_key": True, "student_label": True, p1_col: ":.2f", "probabilidad": ":.1%"},
        opacity=0.75,
        labels={"nivel_riesgo": "Nivel de riesgo", p1_col: x_label, "probabilidad": "Prob. reprobar"},
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color="rgba(255,255,255,0.4)")))

    # Risk threshold band
    fig.add_hrect(y0=0.60, y1=1.0, fillcolor="#EF4444", opacity=0.06, layer="below", line_width=0)
    fig.add_hrect(y0=0.35, y1=0.60, fillcolor="#FBBF24", opacity=0.06, layer="below", line_width=0)
    fig.add_hline(y=0.60, line_dash="dot", line_color="#EF4444", line_width=1.5,
                  annotation_text="Umbral alto", annotation_position="right",
                  annotation_font_size=10)
    fig.add_hline(y=0.35, line_dash="dot", line_color="#FBBF24", line_width=1.5,
                  annotation_text="Umbral medio", annotation_position="right",
                  annotation_font_size=10)

    fig.update_layout(
        title=f"{x_label} vs Probabilidad de reprobar",
        xaxis_title=x_label,
        yaxis_title="Probabilidad estimada de reprobar",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=380,
    )
    return _apply_theme(fig, theme_name), p1_col


def percentile_table_data(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a styled percentile summary table (P25, P50, P75, mean) by risk level."""
    numeric_cols = [c for c in ["parcial_1", "parcial_2", "engagement_hasta_p1",
                                 "engagement_hasta_p2", "promedio_ams", "promedio_quices"]
                    if c in df.columns]
    if not numeric_cols:
        return pd.DataFrame()

    rows = []
    for level in ["ALTO", "MEDIO", "BAJO"]:
        sub = df[df["nivel_riesgo"] == level]
        if sub.empty:
            continue
        for col in numeric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append({
                "Nivel": level,
                "Variable": DISPLAY_LABELS.get(col, col),
                "P25": round(float(vals.quantile(0.25)), 2),
                "P50 (mediana)": round(float(vals.median()), 2),
                "P75": round(float(vals.quantile(0.75)), 2),
                "Media": round(float(vals.mean()), 2),
                "n": int(len(vals)),
            })
    return pd.DataFrame(rows)


def risk_radar_chart(metrics: dict, theme_name: str, model_name: str = "Modelo actual"):
    """Radar/spider chart of model performance metrics."""
    dark = theme_name == "Oscuro"
    grid_color = "#2A3853" if dark else "#D1D5DB"
    fill_color = "rgba(96,165,250,0.22)" if dark else "rgba(37,99,235,0.15)"
    line_color = "#60A5FA" if dark else "#2563EB"

    categories = ["AUC-ROC", "Recall", "Precisión", "F1"]
    values = [
        metrics.get("AUC", 0),
        metrics.get("Recall", 0),
        metrics.get("Prec", 0),
        metrics.get("F1", 0),
    ]
    values_closed = values + [values[0]]
    cats_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=line_color, width=2.5),
            name=model_name,
            hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
            marker=dict(size=6, color=line_color),
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10),
                gridcolor=grid_color,
                linecolor=grid_color,
                tickformat=".2f",
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
                linecolor=grid_color,
                gridcolor=grid_color,
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        title=f"Perfil de calidad — {model_name}",
        height=320,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    # Apply font theme manually (polar charts need separate handling)
    font_color = "#E5EEF9" if dark else "#1F2937"
    fig.update_layout(font=dict(
        color=font_color,
        family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        size=12,
    ))
    return fig
