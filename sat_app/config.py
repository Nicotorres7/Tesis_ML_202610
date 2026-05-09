from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RUNTIME_DIR = ROOT / "sat_runtime"
PROJECTS_DIR = RUNTIME_DIR / "projects"
EXPORTS_DIR = RUNTIME_DIR / "exports"
RETRAINED_DIR = RUNTIME_DIR / "retrained_models"

RUNTIME_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)
RETRAINED_DIR.mkdir(exist_ok=True)

LEGACY_SAT_PROJECT_NAME = "Modelos Probabilisticos"
LEGACY_SAT_PROJECT_SLUG = "sat-uniandes"
LEGACY_SAT_CHECKPOINTS = {
    "cp1": {
        "label": "Checkpoint 1",
        "title": "Semana 6",
        "bundle_files": {
            "precision": MODELS_DIR / "cp1semana6_precision.joblib",
            "f1": MODELS_DIR / "cp1semana6_f1.joblib",
            "recall": MODELS_DIR / "cp1semana6_recall.joblib",
        },
        "metadata_file": MODELS_DIR / "cp1semana6_metadata.json",
        "comparison_image": MODELS_DIR / "fig_comparacion_final_cp1.png",
        "threshold_image": MODELS_DIR / "fig_umbral_cp1(semana6).png",
        "description": "Modelo historico del SAT para semana 6.",
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
        "comparison_image": MODELS_DIR / "fig_comparacion_final_cp2.png",
        "threshold_image": MODELS_DIR / "fig_umbral_cp2(semana11).png",
        "description": "Modelo historico del SAT para semana 11.",
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
        "bg": "#F0F4FB",
        "bg_accent": "rgba(219,234,254,0.7)",
        "panel": "#FFFFFF",
        "panel_alt": "#F8FAFC",
        "text": "#111827",
        "muted": "#6B7280",
        "border": "#D1D5DB",
        "shadow": "rgba(15,23,42,0.07)",
        "badge_bg": "rgba(59,130,246,0.10)",
        "badge_text": "#1E40AF",
        "hero_from": "#003D7A",
        "hero_mid": "#1D4ED8",
        "hero_to": "#60A5FA",
        "table_highlight": "#F9FAFB",
        "table_row_alt": "#F1F5F9",
        "table_header": "#E2E8F0",
        "table_header_text": "#1E293B",
        "input_bg": "#FFFFFF",
        "tab_active": "#2563EB",
        "tab_text": "#374151",
    },
    "Oscuro": {
        "bg": "#0D1117",
        "bg_accent": "rgba(29,78,216,0.18)",
        "panel": "#161B27",
        "panel_alt": "#1E2535",
        "text": "#F1F5F9",
        "muted": "#94A3B8",
        "border": "#2D3748",
        "shadow": "rgba(0,0,0,0.5)",
        "badge_bg": "rgba(96,165,250,0.15)",
        "badge_text": "#93C5FD",
        "hero_from": "#0F1F4A",
        "hero_mid": "#1D4ED8",
        "hero_to": "#0F766E",
        "table_highlight": "#1A2336",
        "table_row_alt": "#1A2234",
        "table_header": "#1E2A3D",
        "table_header_text": "#CBD5E1",
        "input_bg": "#1E2535",
        "tab_active": "#60A5FA",
        "tab_text": "#CBD5E1",
    },
}


def build_base_style(theme_name: str) -> str:
    t = THEMES.get(theme_name, THEMES["Claro"])
    dark = theme_name == "Oscuro"

    banner_bg = (
        "linear-gradient(135deg,rgba(15,31,74,0.9) 0%,rgba(29,78,216,0.7) 100%)"
        if dark
        else "linear-gradient(135deg,rgba(0,61,122,0.08) 0%,rgba(59,130,246,0.05) 100%)"
    )
    criterion_precision_bg = "#1A3050" if dark else "#EFF6FF"
    criterion_f1_bg = "#162A4A" if dark else "#EFF6FF"
    criterion_recall_bg = "#12312A" if dark else "#F0FDF4"

    return f"""
<style>
/* ═══════════════════════════════════════════════════════
   SAT UNIANDES — {theme_name.upper()} THEME
═══════════════════════════════════════════════════════ */

/* ── CSS Variables ── */
:root {{
    --sat-blue:        #1D4ED8;
    --sat-blue-light:  #60A5FA;
    --sat-red:         #EF4444;
    --sat-yellow:      #F59E0B;
    --sat-green:       #10B981;
    --sat-text:        {t["text"]};
    --sat-muted:       {t["muted"]};
    --sat-border:      {t["border"]};
    --sat-bg:          {t["bg"]};
    --sat-panel:       {t["panel"]};
    --sat-panel-alt:   {t["panel_alt"]};
    --sat-badge-bg:    {t["badge_bg"]};
    --sat-badge-text:  {t["badge_text"]};
    --sat-shadow:      {t["shadow"]};
    --sat-input-bg:    {t["input_bg"]};
}}

/* ══════════════════════════════════════════════════════════════
   DARK-MODE BROWSER OVERRIDE
   Declare color-scheme BEFORE anything else so the browser
   never applies its own dark palette to ANY element.
══════════════════════════════════════════════════════════════ */
:root {{
    color-scheme: light !important;
    forced-color-adjust: none;
}}
html, body {{
    color-scheme: light !important;
    forced-color-adjust: none;
    -webkit-color-scheme: light;
}}
/* Catch every Streamlit container */
.stApp, .main, .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="element-container"],
[data-testid="stSidebar"],
[data-testid="stHeader"],
[data-testid="column"],
[data-testid="stForm"],
[data-testid="stExpander"],
[data-testid="stTabs"],
[data-testid="stTabsTabList"],
[data-testid="stTabsTabPanel"] {{
    color-scheme: light !important;
    forced-color-adjust: none;
}}

/* ── App background ── */
.stApp {{
    background:
        radial-gradient(ellipse at top right, {t["bg_accent"]}, transparent 50%),
        {t["bg"]} !important;
    color: {t["text"]} !important;
}}
html, body, [data-testid="stAppViewContainer"] {{
    background: {t["bg"]} !important;
    color: {t["text"]} !important;
}}
html {{ background: {t["bg"]} !important; }}
[data-testid="stHeader"] {{
    background: rgba(240,244,251,0.92) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid {t["border"]} !important;
}}
[data-testid="stToolbar"], [data-testid="stDecoration"] {{
    background: transparent !important;
}}

/* ── Main container ── */
.main .block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1400px;
    animation: sat-fadein 0.25s ease;
}}
@keyframes sat-fadein {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {t["panel"]} !important;
    border-right: 1px solid {t["border"]} !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    padding: 1.2rem 1rem 1rem;
}}
[data-testid="stSidebar"] * {{
    color: {t["text"]} !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: {t["border"]};
    margin: 0.75rem 0;
}}
/* Sidebar brand heading */
[data-testid="stSidebar"] .stMarkdown h2 {{
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    border-bottom: 2px solid var(--sat-blue);
    padding-bottom: 0.3rem;
    margin-bottom: 0.6rem;
    color: {"#60A5FA" if dark else "#1D4ED8"} !important;
}}
/* Sidebar radio nav — uniform height, no red, clean selection */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{
    gap: 4px;
    display: flex;
    flex-direction: column;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    border-radius: 10px;
    padding: 9px 14px;
    min-height: 42px;
    display: flex;
    align-items: center;
    transition: background 0.15s ease;
    cursor: pointer;
    color: {t["text"]} !important;
    background: {t["panel_alt"]};
    border: 1px solid transparent;
    box-sizing: border-box;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background: #EAF2FF;
    border-color: #BFDBFE;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
    background: #EFF6FF !important;
    border-color: #1D4ED8 !important;
    border-left: 3px solid #1D4ED8 !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {{
    color: #1D4ED8 !important;
    font-weight: 700 !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {{
    margin: 0;
    font-size: 0.88rem;
    font-weight: 500;
    color: {t["text"]} !important;
    line-height: 1.3;
}}
/* Hide the radio dot */
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {{
    display: none;
}}

/* ── Typography (global) — explicit colors prevent browser dark mode override ── */
h1, h2, h3, h4, h5, h6 {{ color: {t["text"]} !important; }}
p {{ color: {t["text"]}; }}
label, li, small, strong, em {{ color: inherit; }}
code {{ color: {t["text"]} !important; background: {t["panel_alt"]} !important; }}
.stMarkdown p, .stMarkdown li, .stMarkdown span {{ color: {t["text"]} !important; }}
.stMarkdown {{ color: {t["text"]} !important; }}
.stCaption, .stCaption * {{ color: {t["muted"]} !important; }}

/* ── File uploader + dropzone ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div,
[data-testid="stFileUploader"] section > button {{
    background: {t["panel"]} !important;
    color: {t["text"]} !important;
    border-color: {t["border"]} !important;
}}
[data-testid="stFileUploader"] * {{
    color: {t["text"]} !important;
}}
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"] > div {{
    background: {t["panel_alt"]} !important;
    border: 2px dashed {t["border"]} !important;
    border-radius: 12px !important;
    color: {t["text"]} !important;
}}
[data-testid="stFileUploadDropzone"] * {{
    color: {t["muted"]} !important;
}}
[data-testid="stFileUploadDropzone"] small {{
    color: {t["muted"]} !important;
}}

/* ── DataFrames (st.dataframe) ──────────────────────────────────────────
   st.dataframe renders via glide-data-grid on a <canvas> — CSS cannot
   paint inside a canvas. The only safe lever is color-scheme on the
   host/wrapper so the browser-internal rendering uses a light palette.
   DO NOT apply color/background to * or inner elements: that turns the
   canvas text invisible (white-on-white).
─────────────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"],
[data-testid="stDataFrameResizable"],
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrameResizable"] > div {{
    color-scheme: light !important;
    background: {t["panel"]} !important;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {t["border"]};
}}
/* st.table (legacy HTML table — safe to style fully) */
[data-testid="stTable"] {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {t["border"]};
    background: {t["panel"]} !important;
}}
[data-testid="stTable"] th {{
    background: {t["table_header"]} !important;
    color: {t["table_header_text"]} !important;
    font-weight: 700;
}}
[data-testid="stTable"] td {{
    background: {t["panel"]} !important;
    color: {t["text"]} !important;
}}
[data-testid="stTable"] tr:nth-child(even) td {{
    background: {t["table_row_alt"]} !important;
}}

/* ── Tooltip ── */
[data-baseweb="tooltip"] > div {{
    background: #1E293B !important;
    color: #F8FAFC !important;
}}
/* ── Spinner ── */
[data-testid="stSpinner"] * {{ color: {t["text"]} !important; }}

/* ══════════════════════════════════════════════════════════════
   MEDIA QUERY: neutralise every dark-mode browser override.
   These rules run ONLY when the browser's own dark mode would
   otherwise kick in, and they restore our explicit palette.
══════════════════════════════════════════════════════════════ */
@media (prefers-color-scheme: dark) {{
    :root {{ color-scheme: light !important; }}
    html, body {{ background: {t["bg"]} !important; color: {t["text"]} !important; }}
    .stApp, .main, .block-container {{ background: {t["bg"]} !important; color: {t["text"]} !important; }}

    /* Dropzone */
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploadDropzone"] > div {{
        background: {t["panel_alt"]} !important;
        border-color: {t["border"]} !important;
        color: {t["text"]} !important;
    }}
    [data-testid="stFileUploadDropzone"] * {{ color: {t["muted"]} !important; }}

    /* DataFrames — only set color-scheme on host, never paint * */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrameResizable"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrameResizable"] > div {{
        color-scheme: light !important;
        background: {t["panel"]} !important;
    }}
    [data-testid="stTable"] th {{
        background: {t["table_header"]} !important;
        color: {t["table_header_text"]} !important;
    }}
    [data-testid="stTable"] td {{
        background: {t["panel"]} !important;
        color: {t["text"]} !important;
    }}

    /* Inputs */
    input, textarea, select,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {{
        background: {t["input_bg"]} !important;
        color: {t["text"]} !important;
        border-color: {t["border"]} !important;
    }}

    /* Expanders, cards, panels */
    [data-testid="stExpander"],
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] > div {{
        background: {t["panel"]} !important;
        color: {t["text"]} !important;
    }}

    /* Buttons */
    button {{ background: {t["panel"]} !important; color: {t["text"]} !important; }}
    [data-testid="stBaseButton-primary"],
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {{
        background: {t["panel"]} !important;
        color: {t["text"]} !important;
    }}
    [data-testid="stSidebar"] * {{ color: {t["text"]} !important; }}

    /* Tabs */
    [data-testid="stTabs"],
    [data-testid="stTabsTabList"],
    [data-testid="stTabsTabPanel"] {{
        background: {t["panel"]} !important;
        color: {t["text"]} !important;
    }}
    [data-testid="stTabs"] button {{ color: {t["tab_text"]} !important; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{ color: {t["tab_active"]} !important; }}

    /* Metrics */
    [data-testid="stMetricValue"] {{ color: {t["text"]} !important; }}
    [data-testid="stMetricLabel"] {{ color: {t["muted"]} !important; }}

    /* All text */
    p, h1, h2, h3, h4, h5, h6, label, span, div, li, small, strong, em {{
        color: {t["text"]};
    }}
}}

/* ── Form labels ── */
.stTextInput label, .stSelectbox label, .stMultiSelect label, .stRadio label,
.stNumberInput label, .stSlider label, .stFileUploader label, .stCheckbox label,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {{
    color: {t["text"]} !important;
}}

/* ── Form inputs ── */
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
.stTextInput input, .stTextArea textarea, .stNumberInput input {{
    background: {t["input_bg"]} !important;
    border-color: {t["border"]} !important;
    color: {t["text"]} !important;
}}
[role="listbox"], [data-baseweb="popover"] {{
    background: {t["panel"]} !important;
    color: {t["text"]} !important;
    border: 1px solid {t["border"]} !important;
}}
[role="option"] {{
    background: {t["panel"]} !important;
    color: {t["text"]} !important;
}}
[role="option"][aria-selected="true"] {{
    background: rgba(15,108,189,0.10) !important;
}}
[data-baseweb="select"] ul, [data-baseweb="menu"] {{
    background: {t["panel"]} !important;
    color: {t["text"]} !important;
}}
[data-baseweb="select"] *, [data-baseweb="input"] * {{
    color: {t["text"]} !important;
}}
/* Multiselect tags */
[data-baseweb="tag"] {{
    background: {t["badge_bg"]} !important;
    color: {t["badge_text"]} !important;
}}

/* ── Buttons — override Streamlit defaults & browser dark mode ── */
/* Secondary / default button */
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondaryFormSubmit"],
.stButton > button:not([kind="primary"]),
.stFormSubmitButton > button:not([kind="primary"]) {{
    background: {t["panel"]} !important;
    color: {t["text"]} !important;
    border: 1px solid {t["border"]} !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}
[data-testid="stBaseButton-secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {{
    background: {t["panel_alt"]} !important;
    border-color: #93C5FD !important;
}}
/* Primary button — blue, NOT red */
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"],
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(29,78,216,0.30) !important;
}}
[data-testid="stBaseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #1E40AF 0%, #1D4ED8 100%) !important;
    box-shadow: 0 6px 20px rgba(29,78,216,0.40) !important;
}}
/* Download button */
[data-testid="stDownloadButton"] > button {{
    background: {t["panel"]} !important;
    color: {t["text"]} !important;
    border: 1px solid {t["border"]} !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}
/* All button text */
button, button p, button span, button div {{
    color: inherit !important;
}}

/* ── Tabs ── */
[data-testid="stTabs"] button {{
    color: {t["tab_text"]} !important;
    font-weight: 600;
    border-radius: 8px 8px 0 0;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {t["tab_active"]} !important;
    border-bottom: 2px solid {t["tab_active"]} !important;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    border: 1px solid {t["border"]} !important;
    border-radius: 12px !important;
    overflow: hidden;
    background: {t["panel"]};
}}
[data-testid="stExpander"] summary {{
    background: {t["panel_alt"]} !important;
    border-radius: 12px;
    padding: 0.65rem 1rem;
    color: {t["text"]} !important;
}}
[data-testid="stExpander"] summary * {{ color: {t["text"]} !important; }}

/* ── DataFrames ── */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {t["border"]};
    background: {t["panel"]} !important;
    color-scheme: light !important;
}}
[data-testid="stDataFrame"] > div, [data-testid="stTable"] > div {{
    background: {t["panel"]} !important;
    color-scheme: light !important;
}}

/* ── Alerts & metrics ── */
.stAlert, .stAlert * {{ color: {t["text"]} !important; }}
[data-testid="stMetricValue"] {{ color: {t["text"]} !important; font-weight: 700; }}
[data-testid="stMetricLabel"] {{ color: {t["muted"]} !important; }}
[data-testid="stMetricDelta"] {{ font-weight: 600; }}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div {{
    background: {t["panel_alt"]};
    border-radius: 99px;
}}
[data-testid="stProgressBar"] > div > div {{
    background: var(--sat-blue);
    border-radius: 99px;
}}

/* ═══════════════════════════════════════════════════
   SAT CUSTOM COMPONENTS
═══════════════════════════════════════════════════ */

/* ── Hero ── */
.sat-hero {{
    background: linear-gradient(135deg,
        {t["hero_from"]} 0%,
        {t["hero_mid"]} 55%,
        {t["hero_to"]} 100%);
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 1.4rem;
    box-shadow: 0 20px 56px {t["shadow"]};
    position: relative;
    overflow: hidden;
    color: white !important;
}}
.sat-hero::before {{
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
    pointer-events: none;
}}
.sat-hero::after {{
    content: '';
    position: absolute;
    bottom: -70px; left: 28%;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.03);
    border-radius: 50%;
    pointer-events: none;
}}
.sat-hero h1, .sat-hero h2, .sat-hero h3,
.sat-hero p, .sat-hero div, .sat-hero span {{
    color: white !important;
}}
#hero-title {{ color: white !important; }}

/* ── Cards ── */
.sat-card {{
    background: {t["panel"]};
    border: 1px solid {t["border"]};
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 4px 16px {t["shadow"]};
    height: 100%;
    transition: transform 0.2s cubic-bezier(0.34,1.56,0.64,1),
                box-shadow 0.2s ease;
}}
.sat-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 16px 40px {t["shadow"]};
}}
.sat-card h3, .sat-card h4, .sat-card p, .sat-card strong {{
    color: {t["text"]};
}}

/* ── Stat tiles ── */
.sat-stat {{
    background: {t["panel"]};
    border: 1px solid {t["border"]};
    border-radius: 14px;
    padding: 16px 20px;
    border-top: 4px solid var(--accent, var(--sat-blue));
    box-shadow: 0 4px 14px {t["shadow"]};
    position: relative;
    overflow: hidden;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}}
.sat-stat::after {{
    content: '';
    position: absolute;
    top: -10px; right: -10px;
    width: 70px; height: 70px;
    background: var(--accent, var(--sat-blue));
    opacity: {"0.12" if dark else "0.06"};
    border-radius: 50%;
    pointer-events: none;
}}
.sat-stat:hover {{
    transform: translateY(-2px);
    box-shadow: 0 12px 30px {t["shadow"]};
}}
.sat-kpi {{
    font-size: 2.1rem;
    font-weight: 800;
    color: {t["text"]};
    line-height: 1;
    margin-bottom: 0.2rem;
}}
.sat-muted {{
    color: {t["muted"]};
    font-size: 0.88rem;
    line-height: 1.4;
}}

/* ── Badge ── */
.sat-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.02em;
    background: {t["badge_bg"]};
    color: {t["badge_text"]};
    margin-bottom: 10px;
}}

/* ── Chips ── */
.sat-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border-radius: 999px;
    border: 1px solid {t["border"]};
    background: {t["panel_alt"]};
    color: {t["text"]};
    font-size: 0.83rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
}}

/* ── Note panel ── */
.sat-note {{
    background: {t["panel"]};
    border-left: 4px solid {"#60A5FA" if dark else "#2563EB"};
    border-top: 1px solid {t["border"]};
    border-right: 1px solid {t["border"]};
    border-bottom: 1px solid {t["border"]};
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.8rem;
}}
.sat-panel-title {{
    font-size: 0.95rem;
    font-weight: 700;
    color: {t["text"]};
    margin-bottom: 0.3rem;
}}

/* ── Section title ── */
.sat-section-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {t["text"]};
    margin: 0.4rem 0 0.6rem 0;
}}

/* ── Checkpoint banner ── */
.sat-checkpoint-banner {{
    background: {banner_bg};
    border: 1px solid {t["border"]};
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.1rem;
}}
.sat-checkpoint-banner h3 {{ color: {t["text"]} !important; }}
.sat-checkpoint-banner .sat-muted {{ color: {t["muted"]} !important; }}

/* ── Model cards ── */
.sat-model-card {{
    min-height: 270px;
    transition: transform 0.22s cubic-bezier(0.34,1.56,0.64,1),
                box-shadow 0.22s ease,
                opacity 0.18s ease,
                filter 0.18s ease;
}}
.sat-model-card.is-selected {{
    border-width: 2px !important;
    transform: translateY(-4px);
    box-shadow: 0 20px 48px rgba(29,78,216,0.28) !important;
}}
.sat-model-card.is-dimmed {{
    opacity: 0.5;
    filter: saturate(0.6) brightness({"0.85" if dark else "0.98"});
}}

/* ── Metric grid inside cards ── */
.sat-model-highlight {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.75rem;
    margin: 0.75rem 0 0.9rem 0;
}}
.sat-metric-tile {{
    background: {t["panel_alt"]};
    border: 1px solid {t["border"]};
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
    transition: transform 0.15s ease;
}}
.sat-metric-tile:hover {{ transform: translateY(-1px); }}
.sat-metric-value {{
    font-size: 1.5rem;
    font-weight: 800;
    color: {t["text"]};
    line-height: 1.1;
}}
.sat-metric-label {{
    color: {t["muted"]};
    font-size: 0.82rem;
    margin-top: 0.2rem;
}}

/* ── Params table ── */
.sat-params-table {{
    width: 100%;
    border-collapse: collapse;
    border: 1px solid {t["border"]};
    border-radius: 12px;
    overflow: hidden;
    background: {t["panel"]};
}}
.sat-params-table th, .sat-params-table td {{
    text-align: left;
    padding: 0.75rem 0.9rem;
    border-bottom: 1px solid {t["border"]};
    color: {t["text"]};
    vertical-align: top;
}}
.sat-params-table th {{
    background: {t["table_header"]};
    color: {t["table_header_text"]};
    font-size: 0.83rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.sat-params-table tr:last-child td {{ border-bottom: none; }}
.sat-params-table tr:hover td {{ background: {t["panel_alt"]}; }}

/* ── Student risk table (HTML custom) ── */
.sat-risk-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid {t["border"]};
    background: {t["panel"]};
    font-size: 0.9rem;
}}
.sat-risk-table thead th {{
    background: {t["table_header"]};
    color: {t["table_header_text"]};
    font-weight: 700;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 10px 14px;
    border-bottom: 2px solid {t["border"]};
}}
.sat-risk-table tbody tr {{
    cursor: pointer;
    transition: background 0.12s ease;
}}
.sat-risk-table tbody tr:nth-child(even) {{
    background: {t["table_row_alt"]};
}}
.sat-risk-table tbody tr:hover {{
    background: {t["badge_bg"]} !important;
}}
.sat-risk-table tbody tr.sat-row-selected {{
    background: {"rgba(96,165,250,0.15)" if dark else "rgba(37,99,235,0.08)"} !important;
    outline: 2px solid {"#60A5FA" if dark else "#2563EB"};
    outline-offset: -2px;
}}
.sat-risk-table td {{
    padding: 9px 14px;
    border-bottom: 1px solid {t["border"]};
    color: {t["text"]};
    vertical-align: middle;
}}
.sat-risk-table tbody tr:last-child td {{ border-bottom: none; }}
/* Prob bar inside table */
.sat-prob-bar-wrap {{
    background: {t["panel_alt"]};
    border-radius: 6px;
    height: 18px;
    overflow: hidden;
    min-width: 80px;
}}
.sat-prob-bar {{
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}}
/* Risk badge in table */
.sat-risk-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 700;
    white-space: nowrap;
}}

/* ── Detail panel ── */
.sat-detail-panel {{
    background: {t["panel"]};
    border: 1px solid {t["border"]};
    border-radius: 16px;
    padding: 1.2rem 1.3rem;
    box-shadow: 0 4px 16px {t["shadow"]};
}}
.sat-detail-panel h3 {{ color: {t["text"]} !important; margin-top: 0; }}

/* ── Upload stepper ── */
.sat-stepper {{
    display: flex;
    align-items: center;
    margin-bottom: 1.4rem;
    gap: 0;
}}
.sat-step-node {{ flex: 0 0 auto; text-align: center; }}
.sat-step-circle {{
    width: 32px; height: 32px;
    border-radius: 50%;
    margin: 0 auto 4px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.87rem;
    color: white;
    transition: background 0.2s ease;
}}
.sat-step-label {{
    font-size: 0.76rem;
    font-weight: 600;
    white-space: nowrap;
}}
.sat-step-line {{
    flex: 1; height: 2px;
    margin-bottom: 22px;
    transition: background 0.2s ease;
}}

/* ── Feature chips ── */
.sat-feature-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
    border: 1px solid {t["border"]};
    background: {t["panel_alt"]};
    color: {t["text"]};
}}

/* ── Percentile table ── */
.sat-percentile-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {t["border"]};
    font-size: 0.88rem;
}}
.sat-percentile-table th {{
    background: {t["table_header"]};
    color: {t["table_header_text"]};
    font-weight: 700;
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 9px 12px;
    border-bottom: 2px solid {t["border"]};
}}
.sat-percentile-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid {t["border"]};
    color: {t["text"]};
}}
.sat-percentile-table tr:last-child td {{ border-bottom: none; }}
.sat-percentile-table tr:nth-child(even) td {{ background: {t["table_row_alt"]}; }}

/* ── Criterion card colors ── */
.sat-criterion-precision {{ background: {criterion_precision_bg}; }}
.sat-criterion-f1       {{ background: {criterion_f1_bg}; }}
.sat-criterion-recall   {{ background: {criterion_recall_bg}; }}

/* ── Training Stepper (wizard) ── */
.sat-wizard-stepper {{
    display: flex;
    align-items: flex-start;
    margin-bottom: 1.8rem;
    gap: 0;
    padding: 1.2rem 1.4rem;
    background: {t["panel"]};
    border: 1px solid {t["border"]};
    border-radius: 18px;
    box-shadow: 0 2px 10px {t["shadow"]};
}}
.sat-wstep {{
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    z-index: 1;
}}
.sat-wstep:not(:last-child)::after {{
    content: '';
    position: absolute;
    top: 18px;
    left: 55%;
    right: -55%;
    height: 2px;
    background: {t["border"]};
    z-index: 0;
}}
.sat-wstep.done::after {{ background: #10B981; }}
.sat-wstep.active::after {{ background: {t["border"]}; }}
.sat-wstep-circle {{
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.9rem;
    margin-bottom: 8px;
    position: relative;
    z-index: 2;
    transition: all 0.2s ease;
}}
.sat-wstep.done .sat-wstep-circle {{
    background: #10B981;
    color: white;
    box-shadow: 0 4px 12px rgba(16,185,129,0.35);
}}
.sat-wstep.active .sat-wstep-circle {{
    background: linear-gradient(135deg, #1D4ED8 0%, #60A5FA 100%);
    color: white;
    box-shadow: 0 4px 16px rgba(29,78,216,0.4);
    transform: scale(1.1);
}}
.sat-wstep.locked .sat-wstep-circle {{
    background: {t["panel_alt"]};
    color: {t["muted"]};
    border: 2px solid {t["border"]};
}}
.sat-wstep-label {{
    font-size: 0.78rem;
    font-weight: 700;
    text-align: center;
    white-space: nowrap;
}}
.sat-wstep.done .sat-wstep-label {{ color: #10B981; }}
.sat-wstep.active .sat-wstep-label {{ color: #1D4ED8; }}
.sat-wstep.locked .sat-wstep-label {{ color: {t["muted"]}; }}

/* ── Risk badges (solid) ── */
.sat-risk-alto   {{ background:#EF4444; color:white; border-radius:999px; padding:3px 10px; font-size:0.75rem; font-weight:700; display:inline-block; }}
.sat-risk-medio  {{ background:#F59E0B; color:white; border-radius:999px; padding:3px 10px; font-size:0.75rem; font-weight:700; display:inline-block; }}
.sat-risk-bajo   {{ background:#10B981; color:white; border-radius:999px; padding:3px 10px; font-size:0.75rem; font-weight:700; display:inline-block; }}

/* ── Project cards ── */
.sat-project-card {{
    background: {t["panel"]};
    border: 1px solid {t["border"]};
    border-left: 4px solid var(--sat-blue);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 10px {t["shadow"]};
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    cursor: pointer;
}}
.sat-project-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 28px {t["shadow"]};
    border-left-color: #2DA8FF;
}}
.sat-project-card.is-active {{
    border-left-color: #10B981;
    background: {"rgba(16,185,129,0.05)" if not dark else "rgba(16,185,129,0.08)"};
}}
.sat-project-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {t["text"]};
    margin-bottom: 0.2rem;
}}
.sat-project-meta {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    color: {t["muted"]};
    font-size: 0.83rem;
    margin-top: 0.4rem;
}}
.sat-project-meta span {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
}}

/* ── Next-step CTA ── */
.sat-next-step {{
    background: linear-gradient(135deg, {"rgba(29,78,216,0.08)" if not dark else "rgba(96,165,250,0.1)"} 0%, {"rgba(16,185,129,0.06)" if not dark else "rgba(16,185,129,0.1)"} 100%);
    border: 1px solid {t["border"]};
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    margin-top: 1.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}}
.sat-next-step-text {{
    font-size: 0.95rem;
    font-weight: 700;
    color: {t["text"]};
}}
.sat-next-step-desc {{
    font-size: 0.83rem;
    color: {t["muted"]};
    margin-top: 2px;
}}

/* ── Success block ── */
.sat-success-block {{
    background: {"rgba(16,185,129,0.08)" if not dark else "rgba(16,185,129,0.12)"};
    border: 1px solid {"rgba(16,185,129,0.3)" if not dark else "rgba(16,185,129,0.4)"};
    border-left: 4px solid #10B981;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}}
.sat-success-title {{
    font-size: 1rem;
    font-weight: 700;
    color: #059669;
    margin-bottom: 0.5rem;
}}
.sat-success-detail {{
    font-size: 0.85rem;
    color: {t["muted"]};
    font-family: monospace;
    background: {t["panel_alt"]};
    border-radius: 8px;
    padding: 6px 10px;
    margin-top: 6px;
    word-break: break-all;
}}

/* ── Dataset preview ── */
.sat-col-preview-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    border-radius: 10px;
    overflow: hidden;
}}
.sat-col-preview-table th {{
    background: {t["table_header"]};
    color: {t["table_header_text"]};
    font-weight: 700;
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 8px 12px;
    border-bottom: 2px solid {t["border"]};
    text-align: left;
}}
.sat-col-preview-table td {{
    padding: 7px 12px;
    border-bottom: 1px solid {t["border"]};
    color: {t["text"]};
    vertical-align: middle;
}}
.sat-col-preview-table tr:last-child td {{ border-bottom: none; }}
.sat-col-preview-table tr:nth-child(even) td {{ background: {t["table_row_alt"]}; }}

/* ── Active model badge ── */
.sat-active-badge {{
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(16,185,129,0.12);
    color: #059669;
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 700;
}}
.sat-inactive-badge {{
    display: inline-flex; align-items: center;
    background: {t["panel_alt"]};
    color: {t["muted"]};
    border: 1px solid {t["border"]};
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}}

/* ── Dashboard stat cards (large) ── */
.sat-dash-stat {{
    background: {t["panel"]};
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 14px {t["shadow"]};
    border: 1px solid {t["border"]};
    text-align: center;
}}
.sat-dash-stat .sat-kpi {{ font-size: 2.4rem; }}
.sat-dash-stat .sat-muted {{ font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; margin-top: 4px; }}
.sat-dash-stat-alto  {{ border-top: 4px solid #EF4444; }}
.sat-dash-stat-medio {{ border-top: 4px solid #F59E0B; }}
.sat-dash-stat-bajo  {{ border-top: 4px solid #10B981; }}
.sat-dash-stat-total {{ border-top: 4px solid #1D4ED8; }}

/* ── Sidebar nav status indicators ── */
.sat-nav-status {{
    font-size: 0.72rem;
    padding: 1px 6px;
    border-radius: 999px;
    font-weight: 700;
    margin-left: 4px;
}}
.sat-nav-ok {{ background:rgba(16,185,129,0.15); color:#059669; }}
.sat-nav-warn {{ background:rgba(245,158,11,0.15); color:#B45309; }}

/* ── Model version timeline ── */
.sat-model-timeline {{
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    margin: 0.8rem 0;
}}
.sat-model-timeline-item {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.7rem 0.9rem;
    background: {t["panel_alt"]};
    border: 1px solid {t["border"]};
    border-radius: 10px;
    font-size: 0.85rem;
    color: {t["text"]};
}}
.sat-model-timeline-item.is-active {{
    border-color: #10B981;
    background: rgba(16,185,129,0.06);
}}
.sat-model-version {{
    font-weight: 800;
    color: #1D4ED8;
    min-width: 28px;
}}

/* ── Responsive ── */
@media (max-width: 900px) {{
    .main .block-container {{
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }}
    .sat-hero {{ padding: 20px 18px; border-radius: 16px; }}
    .sat-card, .sat-stat {{ border-radius: 12px; }}
    .sat-model-highlight {{ grid-template-columns: repeat(2,1fr); }}
    .sat-metric-value {{ font-size: 1.3rem; }}
    .sat-kpi {{ font-size: 1.75rem; }}
    .sat-model-card {{ min-height: auto !important; }}
}}
@media (max-width: 600px) {{
    .sat-hero {{ padding: 14px 14px; border-radius: 12px; }}
    .sat-kpi {{ font-size: 1.5rem; }}
    .sat-section-title {{ font-size: 1rem; }}
}}
</style>
"""
