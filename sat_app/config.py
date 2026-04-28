from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"
MODELS_DIR = NOTEBOOKS_DIR / "modelos_finales"
DATA_DIR = ROOT / "data"
RUNTIME_DIR = ROOT / "sat_runtime"
RETRAINED_DIR = RUNTIME_DIR / "retrained_models"

RUNTIME_DIR.mkdir(exist_ok=True)
RETRAINED_DIR.mkdir(exist_ok=True)

CHECKPOINTS = {
    "cp1": {
        "label": "Checkpoint 1",
        "title": "Semana 6",
        "bundle_files": {
            "precision": MODELS_DIR / "cp1semana6_precision.joblib",
            "f1": MODELS_DIR / "cp1semana6_f1.joblib",
            "recall": MODELS_DIR / "cp1semana6_recall.joblib",
        },
        "metadata_file": MODELS_DIR / "cp1semana6_metadata.json",
        "comparison_image": NOTEBOOKS_DIR / "fig_comparacion_final_cp1.png",
        "threshold_image": NOTEBOOKS_DIR / "fig_umbral_cp1(semana6).png",
    },
    "cp2": {
        "label": "Checkpoint 2",
        "title": "Semana 11",
        "bundle_files": {
            "precision": MODELS_DIR / "cp2semana11_precision.joblib",
            "f1": MODELS_DIR / "cp2semana11_f1.joblib",
            "recall": MODELS_DIR / "cp2semana11_recall.joblib",
        },
        "metadata_file": MODELS_DIR / "cp2semana11_metadata.json",
        "comparison_image": NOTEBOOKS_DIR / "fig_comparacion_final_cp2.png",
        "threshold_image": NOTEBOOKS_DIR / "fig_umbral_cp2(semana11).png",
    },
}

CRITERIA_LABELS = {
    "precision": "Maxima precision",
    "f1": "Balance",
    "recall": "Maximo recall",
}

CRITERIA_DESCRIPTIONS = {
    "precision": "Reduce falsos positivos. Ideal cuando los recursos de intervencion son limitados.",
    "f1": "Equilibrio entre cobertura y precision. Es la opcion recomendada para uso general.",
    "recall": "Amplia la alerta para detectar mas estudiantes en riesgo, aceptando mas falsas alarmas.",
}

CRITERIA_COLORS = {
    "precision": "#FDECEC",
    "f1": "#E8F0FE",
    "recall": "#EAF7EF",
}

RISK_ORDER = ["ALTO", "MEDIO", "BAJO"]
RISK_COLORS = {"ALTO": "#EF4444", "MEDIO": "#FBBF24", "BAJO": "#10B981"}

DERIVED_DEPENDENCIES = {
    "era_encoded": ["era"],
    "log_eng_p1": ["engagement_hasta_p1"],
    "log_eng_p2": ["engagement_hasta_p2"],
    "p1_vs_media": ["parcial_1"],
    "delta_p2_p1": ["parcial_1", "parcial_2"],
    "intensidad_p1": ["engagement_hasta_p1", "parcial_1_visitas"],
    "ratio_vt_p1": ["parcial_1_visitas", "parcial_1_temas_unicos"],
}

DISPLAY_LABELS = {
    "id_estudiante": "ID estudiante",
    "nombre": "Nombre",
    "era": "ERA pedagogica",
    "parcial_1": "Parcial 1",
    "parcial_2": "Parcial 2",
    "engagement_hasta_p1": "Engagement hasta P1",
    "engagement_hasta_p2": "Engagement hasta P2",
    "parcial_1_modulos_unicos": "Modulos unicos P1",
    "parcial_1_temas_unicos": "Temas unicos P1",
    "parcial_1_visitas": "Visitas P1",
    "parcial_1_visitas_por_tema": "Visitas por tema P1",
    "parcial_1_tiempo_por_visita": "Tiempo por visita P1",
    "promedio_ams": "Promedio AMs",
    "promedio_quices": "Promedio quices",
    "era_encoded": "ERA codificada",
    "log_eng_p1": "Log engagement P1",
    "log_eng_p2": "Log engagement P2",
    "p1_vs_media": "P1 vs media cohorte",
    "delta_p2_p1": "Delta P2 - P1",
    "intensidad_p1": "Intensidad P1",
    "ratio_vt_p1": "Ratio visitas/temas P1",
}

ALIASES = {
    "id_estudiante": ["id_estudiante", "id", "codigo", "student_id", "hash", "anon_id"],
    "nombre": ["nombre", "name", "student_name", "estudiante"],
    "era": ["era", "formato", "modalidad", "cohorte"],
    "parcial_1": ["parcial_1", "parcial1", "nota_parcial_1", "nota parcial 1", "p1"],
    "parcial_2": ["parcial_2", "parcial2", "nota_parcial_2", "nota parcial 2", "p2"],
    "engagement_hasta_p1": ["engagement_hasta_p1", "engagement_p1", "engagement hasta p1", "eng p1"],
    "engagement_hasta_p2": ["engagement_hasta_p2", "engagement_p2", "engagement hasta p2", "eng p2"],
    "parcial_1_modulos_unicos": ["parcial_1_modulos_unicos", "modulos_p1", "modulos unicos p1"],
    "parcial_1_temas_unicos": ["parcial_1_temas_unicos", "temas_p1", "temas unicos p1"],
    "parcial_1_visitas": ["parcial_1_visitas", "visitas_p1", "visitas parcial 1"],
    "parcial_1_visitas_por_tema": ["parcial_1_visitas_por_tema", "visitas_por_tema_p1"],
    "parcial_1_tiempo_por_visita": ["parcial_1_tiempo_por_visita", "tiempo_por_visita_p1"],
    "promedio_ams": ["promedio_ams", "promedio ams", "ams"],
    "promedio_quices": ["promedio_quices", "promedio quizzes", "quices", "quizzes"],
    "reprobo": ["reprobo", "reprobado", "objetivo", "target", "label"],
    "semestre": ["semestre", "periodo", "cohorte_semestre"],
}

THEMES = {
    "Claro": {
        "bg": "#F5F7FB",
        "bg_accent": "rgba(224,231,255,0.9)",
        "panel": "#FFFFFF",
        "panel_alt": "#F8FAFC",
        "text": "#1F2937",
        "muted": "#6B7280",
        "border": "#D1D5DB",
        "shadow": "rgba(15, 23, 42, 0.08)",
        "badge_bg": "rgba(59,130,246,0.12)",
        "badge_text": "#003D7A",
        "hero_from": "#003D7A",
        "hero_mid": "#2359A5",
        "hero_to": "#80A8E8",
        "table_highlight": "#F9FAFB",
    },
    "Oscuro": {
        "bg": "#0B1220",
        "bg_accent": "rgba(29,78,216,0.22)",
        "panel": "#121A2B",
        "panel_alt": "#182033",
        "text": "#E5EEF9",
        "muted": "#9DB0C7",
        "border": "#2A3853",
        "shadow": "rgba(2, 8, 23, 0.45)",
        "badge_bg": "rgba(96,165,250,0.18)",
        "badge_text": "#BFDBFE",
        "hero_from": "#102A5B",
        "hero_mid": "#1D4ED8",
        "hero_to": "#0F766E",
        "table_highlight": "#1C2940",
    },
}


def build_base_style(theme_name: str) -> str:
    if theme_name == "Auto":
        light = THEMES["Claro"]
        dark = THEMES["Oscuro"]
        return f"""
<style>
    :root {{
        --sat-blue: #003D7A;
        --sat-red: #EF4444;
        --sat-yellow: #FBBF24;
        --sat-green: #10B981;
        --sat-text: {light["text"]};
        --sat-muted: {light["muted"]};
        --sat-border: {light["border"]};
        --sat-bg: {light["bg"]};
        --sat-panel: {light["panel"]};
        --sat-panel-alt: {light["panel_alt"]};
        --sat-badge-bg: {light["badge_bg"]};
        --sat-badge-text: {light["badge_text"]};
    }}
    @media (prefers-color-scheme: dark) {{
        :root {{
            --sat-text: {dark["text"]};
            --sat-muted: {dark["muted"]};
            --sat-border: {dark["border"]};
            --sat-bg: {dark["bg"]};
            --sat-panel: {dark["panel"]};
            --sat-panel-alt: {dark["panel_alt"]};
            --sat-badge-bg: {dark["badge_bg"]};
            --sat-badge-text: {dark["badge_text"]};
        }}
        .stApp {{
            background:
                radial-gradient(circle at top right, {dark["bg_accent"]}, transparent 35%),
                linear-gradient(180deg, {dark["bg"]} 0%, {dark["panel_alt"]} 100%) !important;
        }}
        [data-testid="stSidebar"] {{
            background: {dark["panel"]} !important;
            border-right: 1px solid {dark["border"]} !important;
        }}
        .sat-hero {{
            background: linear-gradient(135deg, {dark["hero_from"]} 0%, {dark["hero_mid"]} 60%, {dark["hero_to"]} 100%) !important;
            box-shadow: 0 18px 48px {dark["shadow"]} !important;
        }}
    }}
    .stApp {{
        color: var(--sat-text);
        background:
            radial-gradient(circle at top right, {light["bg_accent"]}, transparent 35%),
            linear-gradient(180deg, {light["bg"]} 0%, {light["panel_alt"]} 100%);
    }}
    .stApp, .stApp * {{
        color: inherit;
    }}
    .main .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }}
    [data-testid="stSidebar"] {{
        background: {light["panel"]};
        border-right: 1px solid {light["border"]};
    }}
    h1, h2, h3, h4, h5, h6, p, label, span, div, li, small, strong, em, code {{
        color: inherit;
    }}
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stCaption, .stCaption * {{
        color: var(--sat-text) !important;
    }}
    .stTextInput label, .stSelectbox label, .stMultiSelect label, .stRadio label,
    .stNumberInput label, .stSlider label, .stFileUploader label, .stCheckbox label {{
        color: var(--sat-text) !important;
    }}
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {{
        color: var(--sat-text) !important;
    }}
    [data-baseweb="select"] *, [data-baseweb="input"] *, .stTextInput input,
    .stTextArea textarea, .stNumberInput input {{
        color: var(--sat-text) !important;
    }}
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, .stTextInput input,
    .stTextArea textarea, .stNumberInput input {{
        background: var(--sat-panel) !important;
        border-color: var(--sat-border) !important;
    }}
    button, button p, button span {{
        color: var(--sat-text) !important;
    }}
    [data-testid="stSidebar"] button, [data-testid="stSidebar"] button p, [data-testid="stSidebar"] button span {{
        color: var(--sat-text) !important;
    }}
    [data-testid="stTabs"] button, [data-testid="stTabs"] button p, [data-testid="stTabs"] button span {{
        color: var(--sat-text) !important;
    }}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{
        color: var(--sat-text) !important;
    }}
    [data-testid="stDataFrame"] *, [data-testid="stTable"] * {{
        color: var(--sat-text) !important;
    }}
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
        color: var(--sat-text) !important;
    }}
    .stAlert, .stAlert * {{
        color: var(--sat-text) !important;
    }}
    .sat-hero {{
        background: linear-gradient(135deg, {light["hero_from"]} 0%, {light["hero_mid"]} 60%, {light["hero_to"]} 100%);
        border-radius: 22px;
        padding: 30px 32px;
        color: white;
        box-shadow: 0 18px 48px {light["shadow"]};
        margin-bottom: 1rem;
    }}
    .sat-card, .sat-stat, .sat-note {{
        background: var(--sat-panel);
        border-color: var(--sat-border) !important;
    }}
    .sat-hero h1, .sat-hero h2, .sat-hero h3, .sat-hero p, .sat-hero div {{
        color: #FFFFFF !important;
    }}
    .sat-card {{
        border: 1px solid var(--sat-border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px {light["shadow"]};
        height: 100%;
    }}
    .sat-card h4, .sat-card h3, .sat-card p, .sat-card strong {{
        color: var(--sat-text);
    }}
    .sat-stat {{
        border: 1px solid var(--sat-border);
        border-radius: 16px;
        padding: 18px;
        border-top: 4px solid var(--accent);
    }}
    .sat-badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        background: var(--sat-badge-bg);
        color: var(--sat-badge-text);
        margin-bottom: 10px;
    }}
    .sat-kpi {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--sat-text);
        line-height: 1.1;
    }}
    .sat-muted {{
        color: var(--sat-muted);
        font-size: 0.92rem;
    }}
    .sat-section-title {{
        color: var(--sat-text);
        font-size: 1.18rem;
        font-weight: 700;
        margin: 0.2rem 0 0.7rem 0;
    }}
    .sat-chip {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid var(--sat-border);
        background: var(--sat-panel-alt);
        color: var(--sat-text);
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }}
    .sat-panel-title {{
        font-size: 1rem;
        font-weight: 700;
        color: var(--sat-text);
        margin-bottom: 0.35rem;
    }}
    .sat-note {{
        border-left: 4px solid #3B82F6;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        border-top: 1px solid var(--sat-border);
        border-right: 1px solid var(--sat-border);
        border-bottom: 1px solid var(--sat-border);
        margin-bottom: 0.8rem;
    }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        border-radius: 14px;
        overflow: hidden;
    }}
</style>
"""
    theme = THEMES.get(theme_name, THEMES["Claro"])
    return f"""
<style>
    :root {{
        --sat-blue: #003D7A;
        --sat-red: #EF4444;
        --sat-yellow: #FBBF24;
        --sat-green: #10B981;
        --sat-text: {theme["text"]};
        --sat-muted: {theme["muted"]};
        --sat-border: {theme["border"]};
        --sat-bg: {theme["bg"]};
        --sat-panel: {theme["panel"]};
        --sat-panel-alt: {theme["panel_alt"]};
        --sat-badge-bg: {theme["badge_bg"]};
        --sat-badge-text: {theme["badge_text"]};
    }}
    .stApp {{
        color: var(--sat-text);
        background:
            radial-gradient(circle at top right, {theme["bg_accent"]}, transparent 35%),
            linear-gradient(180deg, {theme["bg"]} 0%, {theme["panel_alt"]} 100%);
    }}
    .main .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }}
    [data-testid="stSidebar"] {{
        background: {theme["panel"]};
        border-right: 1px solid {theme["border"]};
    }}
    .stApp, .stApp * {{
        color: inherit;
    }}
    h1, h2, h3, h4, h5, h6, p, label, span, div, li, small, strong, em, code {{
        color: inherit;
    }}
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stCaption, .stCaption * {{
        color: var(--sat-text) !important;
    }}
    .stTextInput label, .stSelectbox label, .stMultiSelect label, .stRadio label,
    .stNumberInput label, .stSlider label, .stFileUploader label, .stCheckbox label {{
        color: var(--sat-text) !important;
    }}
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {{
        color: var(--sat-text) !important;
    }}
    [data-baseweb="select"] *, [data-baseweb="input"] *, .stTextInput input,
    .stTextArea textarea, .stNumberInput input {{
        color: var(--sat-text) !important;
    }}
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, .stTextInput input,
    .stTextArea textarea, .stNumberInput input {{
        background: var(--sat-panel) !important;
        border-color: var(--sat-border) !important;
    }}
    button, button p, button span {{
        color: var(--sat-text) !important;
    }}
    [data-testid="stSidebar"] *,
    [data-testid="stSidebarNav"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {{
        color: var(--sat-text) !important;
    }}
    [data-testid="stTabs"] button, [data-testid="stTabs"] button p, [data-testid="stTabs"] button span {{
        color: var(--sat-text) !important;
    }}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{
        color: var(--sat-text) !important;
    }}
    [data-testid="stDataFrame"] *, [data-testid="stTable"] * {{
        color: var(--sat-text) !important;
    }}
    .stAlert, .stAlert * {{
        color: var(--sat-text) !important;
    }}
    .sat-hero {{
        background: linear-gradient(135deg, {theme["hero_from"]} 0%, {theme["hero_mid"]} 60%, {theme["hero_to"]} 100%);
        border-radius: 22px;
        padding: 30px 32px;
        color: white;
        box-shadow: 0 18px 48px {theme["shadow"]};
        margin-bottom: 1rem;
    }}
    .sat-hero h1, .sat-hero h2, .sat-hero h3, .sat-hero p, .sat-hero div {{
        color: #FFFFFF !important;
    }}
    .sat-card {{
        background: var(--sat-panel);
        border: 1px solid var(--sat-border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px {theme["shadow"]};
        height: 100%;
    }}
    .sat-card h4, .sat-card h3, .sat-card p, .sat-card strong {{
        color: var(--sat-text);
    }}
    .sat-stat {{
        background: var(--sat-panel);
        border: 1px solid var(--sat-border);
        border-radius: 16px;
        padding: 18px;
        border-top: 4px solid var(--accent);
        box-shadow: 0 8px 24px {theme["shadow"]};
    }}
    .sat-badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        background: var(--sat-badge-bg);
        color: var(--sat-badge-text);
        margin-bottom: 10px;
    }}
    .sat-kpi {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--sat-text);
        line-height: 1.1;
    }}
    .sat-muted {{
        color: var(--sat-muted);
        font-size: 0.92rem;
    }}
    .sat-section-title {{
        color: var(--sat-text);
        font-size: 1.18rem;
        font-weight: 700;
        margin: 0.2rem 0 0.7rem 0;
    }}
    .sat-chip {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid var(--sat-border);
        background: var(--sat-panel-alt);
        color: var(--sat-text);
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }}
    .sat-panel-title {{
        font-size: 1rem;
        font-weight: 700;
        color: var(--sat-text);
        margin-bottom: 0.35rem;
    }}
    .sat-note {{
        background: var(--sat-panel);
        border-left: 4px solid #3B82F6;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        border-top: 1px solid var(--sat-border);
        border-right: 1px solid var(--sat-border);
        border-bottom: 1px solid var(--sat-border);
        margin-bottom: 0.8rem;
    }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        border-radius: 14px;
        overflow: hidden;
    }}

    [data-testid="stDataFrame"] [data-baseweb="table"] tbody tr:has-text("ALTO") [role="progressbar"] > div {{
        background-color: #EF4444 !important;
    }}

    [data-testid="stDataFrame"] [data-baseweb="table"] tbody tr:has-text("MEDIO") [role="progressbar"] > div {{
        background-color: #FBBF24 !important;
    }}

    [data-testid="stDataFrame"] [data-baseweb="table"] tbody tr:has-text("BAJO") [role="progressbar"] > div {{
        background-color: #10B981 !important;
    }}
</style>
"""
