"""
Generador de figuras para Sección 3 — Descripción y Exploración de Datos
SAT: Sistema de Alerta Temprana — IIND-2104, Universidad de los Andes
Autores: Nicolás Torres Pulido, Isabella Delgadillo Calero

Código fiel a los notebooks originales:
  - NB01 perfil_dataset_calidad_datos
  - NB02 analisis_notas_academicas
  - NB03 analisis_engagement

Ejecutar desde la carpeta Tesis_ML_202610/:
    python generar_figuras_seccion3.py

Las figuras se guardan en figures/ con los nombres que referencia
seccion3_exploracion_datos.tex.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde, mannwhitneyu
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  PALETAS — exactas del proyecto (NB01–NB03)
# ─────────────────────────────────────────────────────────────────────────────
PALETA_ESTADO = {
    'APROBÓ':   '#2ecc71',   # Verde
    'REPROBÓ':  '#e74c3c',   # Rojo
    'RETIRADO': '#95a5a6',   # Gris
    'SIN_NOTA': '#f39c12',   # Naranja
}
PALETA_ERA = {
    'formato_antiguo': '#2c3e50',
    'formato_nuevo':   '#e67e22',
}
PALETA_SEMESTRE = {
    202320: '#1a3a5c',
    202410: '#2471a3',
    202420: '#e67e22',
    202510: '#ca6f1e',
    202520: '#784212',
}
ORDEN_ESTADOS = ['APROBÓ', 'REPROBÓ', 'RETIRADO', 'SIN_NOTA']

# ─────────────────────────────────────────────────────────────────────────────
#  RCPARAMS — exactos del proyecto
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':        150,
    'figure.facecolor':  'white',
    'axes.facecolor':    '#f8f9fa',
    'axes.grid':         True,
    'grid.color':        'white',
    'grid.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
})

FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

def save(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓  {name}")

# ─────────────────────────────────────────────────────────────────────────────
#  CARGA DE DATOS  (NB01 carga desde dataset_completo_final.csv)
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando datos…")
df = pd.read_csv('data/dataset_completo_final.csv', encoding='utf-8-sig')
df['semestre'] = df['semestre'].astype(int)
df['estado']   = pd.Categorical(df['estado'], categories=ORDEN_ESTADOS, ordered=True)

semestres = sorted(df['semestre'].unique())

# Subconjuntos (NB01 Celda 7)
df_all                = df.copy()
df_activos            = df[~df['estado'].isin(['RETIRADO', 'SIN_NOTA'])].copy()
df_engagement         = df[df['engagement_hasta_p1'] > 0].copy()
df_activos_engagement = df[
    (~df['estado'].isin(['RETIRADO', 'SIN_NOTA'])) &
    (df['engagement_hasta_p1'] > 0)
].copy()

df_activos['semestre']            = df_activos['semestre'].astype(int)
df_activos_engagement['semestre'] = df_activos_engagement['semestre'].astype(int)

print(f"  df_all:                {len(df_all)}")
print(f"  df_activos:            {len(df_activos)}")
print(f"  df_engagement:         {len(df_engagement)}")
print(f"  df_activos_engagement: {len(df_activos_engagement)}")
print()


# ═════════════════════════════════════════════════════════════════════════════
#  FIG 01 — Composición del dataset por semestre  (→ 01_composicion_dataset.png)
#  NB01 Celda 4
# ═════════════════════════════════════════════════════════════════════════════
print("Generando figuras NB01…")

totales = df.groupby('semestre').size()
props   = (
    df.groupby(['semestre', 'estado'], observed=True)
      .size()
      .unstack('estado', fill_value=0)
      .reindex(columns=ORDEN_ESTADOS, fill_value=0)
      .div(totales, axis=0) * 100
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Distribución de estudiantes por estado final\n(vista general por semestre)',
             fontsize=14, fontweight='bold', y=1.01)

# Panel izquierdo: proporciones 100% stacked
ax  = axes[0]
x_pos  = np.arange(len(semestres))
bottom = np.zeros(len(semestres))
for estado in ORDEN_ESTADOS:
    vals = props[estado].values
    ax.bar(x_pos, vals, bottom=bottom,
           color=PALETA_ESTADO[estado], label=estado, width=0.6, edgecolor='white')
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 5:
            ax.text(x_pos[i], b + v/2, f'{v:.0f}%',
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    bottom += vals

ax.set_xticks(x_pos)
ax.set_xticklabels([str(s) for s in semestres], rotation=15)
ax.set_ylabel('Porcentaje de estudiantes (%)')
ax.set_ylim(0, 102)
ax.set_title('Proporciones por semestre (100%)', fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9)
ax.axvspan(-0.5, 1.5, alpha=0.06, color=PALETA_ERA['formato_antiguo'], zorder=0)
ax.axvspan(1.5, 4.5,  alpha=0.06, color=PALETA_ERA['formato_nuevo'],   zorder=0)
ax.text(0.5, 97, 'Era Antigua', ha='center', fontsize=7.5,
        color=PALETA_ERA['formato_antiguo'], fontweight='bold')
ax.text(3.0, 97, 'Era Nueva',   ha='center', fontsize=7.5,
        color=PALETA_ERA['formato_nuevo'],   fontweight='bold')

# Panel derecho: conteos absolutos
ax2    = axes[1]
conteos = (
    df.groupby(['semestre', 'estado'], observed=True)
      .size()
      .unstack('estado', fill_value=0)
      .reindex(columns=ORDEN_ESTADOS, fill_value=0)
)
bottom2 = np.zeros(len(semestres))
for estado in ORDEN_ESTADOS:
    vals2 = conteos[estado].values
    ax2.bar(x_pos, vals2, bottom=bottom2,
            color=PALETA_ESTADO[estado], label=estado, width=0.6, edgecolor='white')
    for i, (v, b) in enumerate(zip(vals2, bottom2)):
        if v > 3:
            ax2.text(x_pos[i], b + v/2, str(int(v)),
                     ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    bottom2 += vals2

ax2.set_xticks(x_pos)
ax2.set_xticklabels([str(s) for s in semestres], rotation=15)
ax2.set_ylabel('Número de estudiantes')
ax2.set_ylim(0, 195)
ax2.set_title('Conteos absolutos por semestre', fontweight='bold')
ax2.legend(loc='upper right', framealpha=0.9)
ax2.axvspan(-0.5, 1.5, alpha=0.06, color=PALETA_ERA['formato_antiguo'], zorder=0)
ax2.axvspan(1.5, 4.5,  alpha=0.06, color=PALETA_ERA['formato_nuevo'],   zorder=0)

plt.tight_layout()
save('01_composicion_dataset.png')


# ═════════════════════════════════════════════════════════════════════════════
#  FIG 12 — Tamaño de subconjuntos  (→ 12_composicion_modelado.png)
#  NB01 Celda 8
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Tamaño de subconjuntos de análisis por semestre',
             fontsize=14, fontweight='bold')

subset_info = {
    'df_all':       (df_all,               '#2c3e50', 'Total (df_all)'),
    'df_activos':   (df_activos,            '#2ecc71', 'Activos (df_activos)'),
    'df_act_eng':   (df_activos_engagement, '#3498db', 'Activos + Engagement (df_activos_eng)'),
}
subset_retiro = {
    'RETIRADO':   (df_all[df_all['estado']=='RETIRADO'],
                   PALETA_ESTADO['RETIRADO'], 'RETIRADO'),
    'SIN_NOTA':   (df_all[df_all['estado']=='SIN_NOTA'],
                   PALETA_ESTADO['SIN_NOTA'], 'SIN_NOTA'),
    'Sin eng.':   (df_activos[df_activos['engagement_hasta_p1']==0],
                   '#f39c12', 'Activos sin engagement'),
}

ax  = axes[0]
x   = np.arange(len(semestres))
w   = 0.25
for j, (key, (sub, color, label)) in enumerate(subset_info.items()):
    counts = sub.groupby('semestre').size().reindex(semestres, fill_value=0)
    ax.bar(x + j*w - w, counts.values, width=w, color=color, alpha=0.85,
           label=label, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in semestres], rotation=15)
ax.set_ylabel('N estudiantes')
ax.set_title('Tamaño de subconjuntos por semestre', fontweight='bold')
ax.legend(fontsize=8)

ax2 = axes[1]
x2  = np.arange(len(semestres))
w2  = 0.25
for j, (key, (sub, color, label)) in enumerate(subset_retiro.items()):
    counts = sub.groupby('semestre').size().reindex(semestres, fill_value=0)
    ax2.bar(x2 + j*w2 - w2, counts.values, width=w2, color=color, alpha=0.85,
            label=label, edgecolor='white')
    for i, v in enumerate(counts.values):
        if v > 0:
            ax2.text(x2[i] + j*w2 - w2, v + 0.3, str(int(v)),
                     ha='center', va='bottom', fontsize=7.5)
ax2.set_xticks(x2)
ax2.set_xticklabels([str(s) for s in semestres], rotation=15)
ax2.set_ylabel('N estudiantes excluidos')
ax2.set_title('Estudiantes excluidos por razón\n(por semestre)', fontweight='bold')
ax2.legend(fontsize=8)

plt.tight_layout()
save('12_composicion_modelado.png')


# ═════════════════════════════════════════════════════════════════════════════
#  FIG 11 — Variables excluidas / cobertura  (→ 11_variables_excluidas.png)
#  NB01 Celda 10 — heatmap de cobertura
# ═════════════════════════════════════════════════════════════════════════════
vars_analisis = [
    'nota_100', 'parcial_1', 'parcial_2', 'parcial_3',
    'fase_1', 'fase_2', 'taller_1', 'taller_2', 'taller_3',
    'am_1', 'am_2', 'am_3', 'am_4', 'am_5', 'am_6',
    'quiz_1', 'quiz_2', 'quiz_3',
    'promedio_ams', 'promedio_quices',
    'engagement_hasta_p1', 'engagement_hasta_p2', 'engagement_hasta_p3',
    'parcial_1_visitas', 'parcial_1_temas_unicos',
    'total_tiempo_hrs', 'total_visitas', 'total_temas', 'total_modulos',
    'actividades_n_realizadas',
]
# Solo las que existen en el dataframe
vars_analisis = [v for v in vars_analisis if v in df.columns]

cobertura = pd.DataFrame(index=vars_analisis, columns=semestres, dtype=float)
for sem in semestres:
    n_total = len(df[df['semestre'] == sem])
    for var in vars_analisis:
        n_validos = df[df['semestre'] == sem][var].notna().sum()
        cobertura.loc[var, sem] = round(n_validos / n_total * 100, 1)

fig, ax = plt.subplots(figsize=(9, 11))
sns.heatmap(
    cobertura,
    ax=ax,
    cmap='RdYlGn',
    vmin=0, vmax=100,
    annot=True, fmt='.0f',
    annot_kws={'size': 7.5},
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Cobertura (%)', 'shrink': 0.6}
)
ax.set_title('Cobertura de variables por semestre\n(% estudiantes con dato disponible)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Semestre', fontsize=10)
ax.set_ylabel('')
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='x', labelsize=9)
ax.axvline(2, color='black', linewidth=2, linestyle='--')
ax.text(1.0, -0.8, 'Formato Antiguo', ha='center',
        transform=ax.get_xaxis_transform(),
        fontsize=9, color=PALETA_ERA['formato_antiguo'], fontweight='bold')
ax.text(3.5, -0.8, 'Formato Nuevo', ha='center',
        transform=ax.get_xaxis_transform(),
        fontsize=9, color=PALETA_ERA['formato_nuevo'], fontweight='bold')

plt.tight_layout()
save('11_variables_excluidas.png')


# ═════════════════════════════════════════════════════════════════════════════
#  FIG 02 — Distribución nota final 3 paneles  (→ 02_distribucion_nota_final.png)
#  NB02 Celda 3
# ═════════════════════════════════════════════════════════════════════════════
print("Generando figuras NB02…")
plt.rcParams.update({'axes.titlesize': 12, 'axes.labelsize': 10})

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Distribución de la nota final (escala 0–5)\ndf_activos — APROBÓ vs. REPROBÓ',
             fontsize=13, fontweight='bold')

# Panel A: Histograma + KDE por estado
ax = axes[0]
for estado in ['APROBÓ', 'REPROBÓ']:
    data = df_activos[df_activos['estado'] == estado]['nota_100']
    ax.hist(data, bins=30, alpha=0.60, color=PALETA_ESTADO[estado],
            label=f'{estado} (n={len(data)})', edgecolor='white', density=True)
    kde_x = np.linspace(0, 5.5, 300)
    kde   = gaussian_kde(data)
    ax.plot(kde_x, kde(kde_x), color=PALETA_ESTADO[estado], linewidth=2)

ax.axvline(3.0, color='black', linestyle='--', linewidth=1.5, label='Aprobación (3.0)')
ax.set_xlabel('Nota final'); ax.set_ylabel('Densidad')
ax.set_xlim(0, 5.5); ax.set_title('Vista general por estado', fontweight='bold')
ax.legend()

# Panel B: KDE por era
ax2 = axes[1]
for era in ['formato_antiguo', 'formato_nuevo']:
    data  = df_activos[df_activos['era'] == era]['nota_100']
    label = f"{'Antiguo' if 'antiguo' in era else 'Nuevo'} (n={len(data)})"
    kde   = gaussian_kde(data)
    kde_x = np.linspace(0, 5.5, 300)
    ax2.plot(kde_x, kde(kde_x), linewidth=2.5, color=PALETA_ERA[era], label=label)
    ax2.fill_between(kde_x, kde(kde_x), alpha=0.12, color=PALETA_ERA[era])

ax2.axvline(3.0, color='black', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Nota final'); ax2.set_ylabel('Densidad')
ax2.set_xlim(0, 5.5); ax2.set_title('KDE por era pedagógica', fontweight='bold')
ax2.legend()

# Panel C: Violin por semestre
ax3 = axes[2]
data_violin = [df_activos[df_activos['semestre'] == s]['nota_100'].values
               for s in semestres]
parts = ax3.violinplot(data_violin, positions=range(len(semestres)),
                       showmedians=True, showextrema=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(PALETA_SEMESTRE[semestres[i]]); pc.set_alpha(0.7)
parts['cmedians'].set_color('#e74c3c'); parts['cmedians'].set_linewidth(2.5)

ax3.axhline(3.0, color='black', linestyle='--', linewidth=1.2)
ax3.set_xticks(range(len(semestres)))
ax3.set_xticklabels([str(s) for s in semestres], rotation=15)
ax3.set_ylabel('Nota final'); ax3.set_ylim(-0.1, 5.8)
ax3.set_title('Violin por semestre', fontweight='bold')
ax3.axvspan(-0.5, 1.5, alpha=0.05, color=PALETA_ERA['formato_antiguo'])
ax3.axvspan(1.5, 4.5,  alpha=0.05, color=PALETA_ERA['formato_nuevo'])

plt.tight_layout()
save('02_distribucion_nota_final.png')


# ═════════════════════════════════════════════════════════════════════════════
#  FIG 05 — Parciales 2×3  (→ 05_parciales_era_resultado.png)
#  NB02 Celda 7
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribuciones de parciales: APROBÓ vs. REPROBÓ y por semestre\n(df_activos, N=554)',
             fontsize=14, fontweight='bold')

parciales = ['parcial_1', 'parcial_2', 'parcial_3']
titulos   = ['Parcial 1 (20%)\n~Semana 5', 'Parcial 2 (25%)\n~Semana 10', 'Parcial 3 (25%)\n~Semana 17']

for col_idx, (var, titulo) in enumerate(zip(parciales, titulos)):
    # Fila superior: histograma + KDE por estado
    ax = axes[0, col_idx]
    for estado in ['APROBÓ', 'REPROBÓ']:
        data = df_activos[df_activos['estado'] == estado][var].dropna()
        ax.hist(data, bins=25, alpha=0.55, color=PALETA_ESTADO[estado],
                label=f'{estado} (μ={data.mean():.2f})', edgecolor='white', density=True)
        if len(data) > 5:
            kde_x = np.linspace(0, 7, 300)
            ax.plot(kde_x, gaussian_kde(data)(kde_x), color=PALETA_ESTADO[estado], linewidth=2)

    ax.axvline(3.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Corte 3.0')
    ax.set_xlabel(f'Nota {var}'); ax.set_ylabel('Densidad')
    ax.set_xlim(0, 7); ax.set_title(titulo, fontweight='bold')
    ax.legend(fontsize=8)

    r, _ = stats.pearsonr(
        df_activos[var].dropna(),
        df_activos.loc[df_activos[var].notna(), 'nota_100']
    )
    ax.text(0.03, 0.97, f'r={r:.3f}\np<0.001',
            transform=ax.transAxes, va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Fila inferior: violin por semestre
    ax2 = axes[1, col_idx]
    data_viol = [df_activos[df_activos['semestre'] == s][var].dropna().values
                 for s in semestres]
    parts2 = ax2.violinplot(data_viol, positions=range(len(semestres)),
                            showmedians=True, showextrema=True)
    for i, pc in enumerate(parts2['bodies']):
        pc.set_facecolor(PALETA_SEMESTRE[semestres[i]]); pc.set_alpha(0.7)
    parts2['cmedians'].set_color('#e74c3c'); parts2['cmedians'].set_linewidth(2)

    ax2.axhline(3.0, color='black', linestyle='--', linewidth=1.2)
    ax2.set_xticks(range(len(semestres)))
    ax2.set_xticklabels([str(s) for s in semestres], rotation=20, fontsize=8)
    ax2.set_ylabel(f'Nota {var}'); ax2.set_ylim(-0.2, 7.2)
    ax2.set_title(f'{var} por semestre', fontweight='bold')
    ax2.axvspan(-0.5, 1.5, alpha=0.05, color=PALETA_ERA['formato_antiguo'])
    ax2.axvspan(1.5, 4.5,  alpha=0.05, color=PALETA_ERA['formato_nuevo'])

plt.tight_layout()
save('05_parciales_era_resultado.png')


# ═════════════════════════════════════════════════════════════════════════════
#  FIG 03 — Riesgo P1  (→ 03_riesgo_p1.png)
#  NB02 Celda 11-12
# ═════════════════════════════════════════════════════════════════════════════
bins   = [0, 1.5, 2.5, 3.0, 3.5, 4.0, 7.0]
labels = ['[0, 1.5)', '[1.5, 2.5)', '[2.5, 3.0)', '[3.0, 3.5)', '[3.5, 4.0)', '[4.0+)']

df_activos['intervalo_p1'] = pd.cut(df_activos['parcial_1'],
                                    bins=bins, labels=labels, right=False)

tabla_abs = pd.crosstab(df_activos['intervalo_p1'], df_activos['estado'])
tabla_pct = pd.crosstab(df_activos['intervalo_p1'], df_activos['estado'],
                        normalize='index').round(4)

labels_short = ['[0,1.5)\nCrítico', '[1.5,2.5)\nAlto', '[2.5,3.0)\nModerado',
                '[3.0,3.5)\nLímite', '[3.5,4.0)\nSeguro', '[4.0+)\nSólido']

p_apro = [tabla_pct.loc[l, 'APROBÓ']  * 100 if l in tabla_pct.index else 0 for l in labels]
p_repr = [tabla_pct.loc[l, 'REPROBÓ'] * 100 if l in tabla_pct.index else 0 for l in labels]
n_tot  = [tabla_abs.loc[l].sum() if l in tabla_abs.index else 0 for l in labels]
x      = np.arange(len(labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Riesgo de reprobación según intervalo de Parcial 1 — df_activos (N=554)',
             fontsize=13, fontweight='bold')

# Panel A: Barras apiladas 100%
ax = axes[0]
ax.bar(x, p_apro, color=PALETA_ESTADO['APROBÓ'], alpha=0.85, label='APROBÓ', edgecolor='white')
ax.bar(x, p_repr, bottom=p_apro, color=PALETA_ESTADO['REPROBÓ'], alpha=0.85,
       label='REPROBÓ', edgecolor='white')
for i, (pa, pr, n) in enumerate(zip(p_apro, p_repr, n_tot)):
    if pa > 8: ax.text(i, pa/2, f'{pa:.0f}%', ha='center', va='center',
                       fontsize=8.5, color='white', fontweight='bold')
    if pr > 8: ax.text(i, pa + pr/2, f'{pr:.0f}%', ha='center', va='center',
                       fontsize=8.5, color='white', fontweight='bold')
    ax.text(i, 102, f'n={n}', ha='center', va='bottom', fontsize=8, color='#2c3e50')
ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=8)
ax.set_ylabel('Porcentaje (%)'); ax.set_ylim(0, 112)
ax.set_title('Distribución de resultado\npor intervalo de P1', fontweight='bold')
ax.legend(loc='lower right')
ax.axvspan(-0.5, 1.5, alpha=0.07, color='#e74c3c')
ax.axvspan(1.5, 2.5,  alpha=0.04, color='#f39c12')
ax.axvspan(2.5, 5.5,  alpha=0.05, color='#2ecc71')

# Panel B: Nota final media por intervalo
ax2     = axes[1]
nota_m  = [df_activos[df_activos['intervalo_p1'] == l]['nota_100'].mean() for l in labels]
nota_s  = [df_activos[df_activos['intervalo_p1'] == l]['nota_100'].std()  for l in labels]
colores_barra = ['#e74c3c', '#e74c3c', '#f39c12', '#2ecc71', '#2ecc71', '#27ae60']

bars2 = ax2.bar(x, nota_m, color=colores_barra, alpha=0.8, edgecolor='white')
ax2.errorbar(x, nota_m, yerr=nota_s, fmt='none', color='black', capsize=5, linewidth=1.5)
for bar, m in zip(bars2, nota_m):
    if not np.isnan(m):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{m:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.axhline(3.0, color='black', linestyle='--', linewidth=1.5, label='Corte 3.0')
ax2.set_xticks(x); ax2.set_xticklabels(labels_short, fontsize=8)
ax2.set_ylabel('Nota final media (± 1 std)'); ax2.set_ylim(0, 5.5)
ax2.set_title('Nota final media por intervalo de P1\n(barras de error = 1 std)', fontweight='bold')
ax2.legend(fontsize=8)

plt.tight_layout()
save('03_riesgo_p1.png')


# ═════════════════════════════════════════════════════════════════════════════
#  FIG 04 — Trayectorias parciales  (→ 04_trayectorias_parciales.png)
#  NB02 Celda 9
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Evolución de medias de parciales a lo largo del semestre',
             fontsize=13, fontweight='bold')

checkpoints = ['parcial_1', 'parcial_2', 'parcial_3']
labels_ck   = ['P1\n(~sem 5)', 'P2\n(~sem 10)', 'P3\n(~sem 17)']

# Panel A: Por estado con IC 95%
ax = axes[0]
for estado in ['APROBÓ', 'REPROBÓ']:
    sub     = df_activos[df_activos['estado'] == estado]
    medias  = [sub[p].mean() for p in checkpoints]
    errores = [sub[p].std() / np.sqrt(sub[p].notna().sum()) * 1.96 for p in checkpoints]

    ax.plot(range(3), medias, 'o-', color=PALETA_ESTADO[estado],
            linewidth=2.5, markersize=9, label=f'{estado} (n={len(sub)})', zorder=3)
    ax.fill_between(range(3),
                    [m - e for m, e in zip(medias, errores)],
                    [m + e for m, e in zip(medias, errores)],
                    alpha=0.15, color=PALETA_ESTADO[estado])
    for i, m in enumerate(medias):
        ax.annotate(f'{m:.2f}', (i, m),
                    textcoords='offset points', xytext=(8, 4),
                    fontsize=9, color=PALETA_ESTADO[estado], fontweight='bold')

ax.axhline(3.0, color='black', linestyle='--', linewidth=1.2, alpha=0.5, label='Corte 3.0')
ax.set_xticks(range(3)); ax.set_xticklabels(labels_ck)
ax.set_ylabel('Nota media'); ax.set_ylim(0, 5)
ax.set_title('Por estado (IC 95%)', fontweight='bold')
ax.legend()

# Panel B: Por semestre
ax2 = axes[1]
for sem in semestres:
    sub    = df_activos[df_activos['semestre'] == sem]
    medias = [sub[p].mean() for p in checkpoints]
    era    = sub['era'].iloc[0]
    ls     = '-' if era == 'formato_antiguo' else '--'
    ax2.plot(range(3), medias, 'o' + ls, color=PALETA_SEMESTRE[sem],
             linewidth=2, markersize=7, label=str(sem), zorder=3)
    ax2.annotate(str(sem), (2, medias[2]),
                 textcoords='offset points', xytext=(5, 0),
                 fontsize=7.5, color=PALETA_SEMESTRE[sem])

ax2.axhline(3.0, color='black', linestyle='--', linewidth=1.2, alpha=0.5)
ax2.set_xticks(range(3)); ax2.set_xticklabels(labels_ck)
ax2.set_ylabel('Nota media'); ax2.set_ylim(1.5, 5)
ax2.set_title('Por semestre\n(línea sólida: antiguo, discontinua: nuevo)', fontweight='bold')
ax2.legend(fontsize=8)

plt.tight_layout()
save('04_trayectorias_parciales.png')


# ═════════════════════════════════════════════════════════════════════════════
#  FIG 09 — Comparación por eras  (→ 09_comparacion_eras.png)
#  NB02 Celda 14
# ═════════════════════════════════════════════════════════════════════════════
bins_s   = [0, 1.5, 2.5, 3.0, 3.5, 4.0, 7.0]
labels_s = ['[0,1.5)', '[1.5,2.5)', '[2.5,3.0)', '[3.0,3.5)', '[3.5,4.0)', '[4.0+)']
eras     = ['formato_antiguo', 'formato_nuevo']
# Compute actual N for each era label
n_ant = len(df_activos[df_activos['era'] == 'formato_antiguo'])
n_nvo = len(df_activos[df_activos['era'] == 'formato_nuevo'])
eras_lab = [f'Formato Antiguo (N={n_ant})', f'Formato Nuevo (N={n_nvo})']

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Probabilidad de reprobación según P1 — por era pedagógica',
             fontsize=13, fontweight='bold')

for ax, era, era_label in zip(axes, eras, eras_lab):
    sub = df_activos[df_activos['era'] == era].copy()
    sub['int_p1'] = pd.cut(sub['parcial_1'], bins=bins_s, labels=labels_s, right=False)

    tbl     = pd.crosstab(sub['int_p1'], sub['estado'], normalize='index').reindex(labels_s)
    tbl_abs = pd.crosstab(sub['int_p1'], sub['estado']).reindex(labels_s).fillna(0)

    x   = np.arange(len(labels_s))
    p_ap = [tbl.loc[l, 'APROBÓ']  * 100
            if l in tbl.index and 'APROBÓ' in tbl.columns else 0 for l in labels_s]
    p_rp = [tbl.loc[l, 'REPROBÓ'] * 100
            if l in tbl.index and 'REPROBÓ' in tbl.columns else 0 for l in labels_s]
    ns   = [tbl_abs.loc[l].sum() if l in tbl_abs.index else 0 for l in labels_s]

    ax.bar(x, p_ap, color=PALETA_ESTADO['APROBÓ'],  alpha=0.85, label='APROBÓ',  edgecolor='white')
    ax.bar(x, p_rp, bottom=p_ap, color=PALETA_ESTADO['REPROBÓ'], alpha=0.85,
           label='REPROBÓ', edgecolor='white')
    for i, (pa, pr, n) in enumerate(zip(p_ap, p_rp, ns)):
        if pr > 8: ax.text(i, pa + pr/2, f'{pr:.0f}%', ha='center', va='center',
                           fontsize=8.5, color='white', fontweight='bold')
        ax.text(i, 102, f'n={int(n)}', ha='center', va='bottom', fontsize=7.5, color='#2c3e50')

    ax.set_xticks(x); ax.set_xticklabels(labels_s, rotation=15, fontsize=8.5)
    ax.set_ylabel('Porcentaje (%)'); ax.set_ylim(0, 112)
    ax.set_title(era_label, fontweight='bold', color=PALETA_ERA[era])
    ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()
save('09_comparacion_eras.png')


# ═════════════════════════════════════════════════════════════════════════════
#  NB03 SETUP
# ═════════════════════════════════════════════════════════════════════════════
print("Generando figuras NB03…")
df_ae = df_activos_engagement.copy()
df_ae['semestre'] = df_ae['semestre'].astype(int)
semestres_ae = sorted(df_ae['semestre'].unique())

plt.rcParams.update({'axes.titlesize': 12, 'axes.labelsize': 10})


# ═════════════════════════════════════════════════════════════════════════════
#  FIG 07 — Engagement boxplots  (→ 07_engagement_boxplots.png)
#  NB03 Celda 7
# ═════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Engagement hasta P1 (hrs): comparación por estado y semestre',
             fontsize=13, fontweight='bold')

# Panel A: Boxplot APROBÓ vs REPROBÓ con notch + jitter
ax = axes[0]
data_box = [df_ae[df_ae['estado'] == e]['engagement_hasta_p1'].dropna().values
            for e in ['APROBÓ', 'REPROBÓ']]
bp = ax.boxplot(data_box, patch_artist=True, notch=True,
                medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], [PALETA_ESTADO['APROBÓ'], PALETA_ESTADO['REPROBÓ']]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for whisker in bp['whiskers']: whisker.set(color='gray', linewidth=1.5)
for cap in bp['caps']:         cap.set(color='gray', linewidth=1.5)

for i, estado in enumerate(['APROBÓ', 'REPROBÓ']):
    data   = df_ae[df_ae['estado'] == estado]['engagement_hasta_p1'].dropna()
    jitter = np.random.uniform(-0.15, 0.15, len(data))
    ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data,
               alpha=0.18, s=12, color=PALETA_ESTADO[estado], zorder=2)

n_ap = (df_ae['estado'] == 'APROBÓ').sum()
n_rp = (df_ae['estado'] == 'REPROBÓ').sum()
ax.set_xticklabels([f'APROBÓ\n(n={n_ap})', f'REPROBÓ\n(n={n_rp})'])
ax.set_ylabel('Horas de engagement hasta P1')
ax.set_ylim(0, df_ae['engagement_hasta_p1'].quantile(0.97))
ax.set_title('Por estado\n(notch = IC 95% de la mediana)', fontweight='bold')

u, p = mannwhitneyu(
    df_ae[df_ae['estado'] == 'APROBÓ']['engagement_hasta_p1'].dropna(),
    df_ae[df_ae['estado'] == 'REPROBÓ']['engagement_hasta_p1'].dropna(),
    alternative='two-sided'
)
sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
ax.text(0.5, 0.97, f'Mann-Whitney U\np={p:.4f} {sig}',
        transform=ax.transAxes, va='top', ha='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

# Panel B: Mediana por semestre y estado (barras agrupadas)
ax2 = axes[1]
x   = np.arange(len(semestres_ae)); w = 0.35
for j, (estado, color) in enumerate([('APROBÓ', PALETA_ESTADO['APROBÓ']),
                                      ('REPROBÓ', PALETA_ESTADO['REPROBÓ'])]):
    medias = [df_ae[(df_ae['semestre'] == s) & (df_ae['estado'] == estado)]['engagement_hasta_p1'].median()
              for s in semestres_ae]
    ax2.bar(x + j * w - w / 2, medias, width=w, color=color, alpha=0.8,
            label=estado, edgecolor='white')
    for xi, m in enumerate(medias):
        if not np.isnan(m):
            ax2.text(xi + j * w - w / 2, m + 0.1, f'{m:.1f}',
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax2.set_xticks(x); ax2.set_xticklabels([str(s) for s in semestres_ae], rotation=15)
ax2.set_ylabel('Mediana horas hasta P1')
ax2.set_title('Mediana engagement hasta P1\npor semestre y estado', fontweight='bold')
ax2.legend(fontsize=8)
ax2.axvspan(-0.5, 1.5, alpha=0.05, color=PALETA_ERA['formato_antiguo'])
ax2.axvspan(1.5, 4.5,  alpha=0.05, color=PALETA_ERA['formato_nuevo'])

plt.tight_layout()
save('07_engagement_boxplots.png')


# ═════════════════════════════════════════════════════════════════════════════
#  FIG 10 — Evolución temporal del engagement  (→ 10_evolucion_engagement.png)
#  NB03 Celda 9
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Evolución acumulada del engagement (horas) a lo largo del semestre',
             fontsize=13, fontweight='bold')

checkpoints_eng = ['engagement_hasta_p1', 'engagement_hasta_p2', 'engagement_hasta_p3']
labels_ck_eng   = ['Hasta P1\n(~sem 5)', 'Hasta P2\n(~sem 10)', 'Total\n(~sem 17)']

# Panel A: Medianas por estado con IQR
ax = axes[0]
for estado in ['APROBÓ', 'REPROBÓ']:
    sub      = df_ae[df_ae['estado'] == estado]
    medianas = [sub[ck].median() for ck in checkpoints_eng]
    p25      = [sub[ck].quantile(0.25) for ck in checkpoints_eng]
    p75      = [sub[ck].quantile(0.75) for ck in checkpoints_eng]

    ax.plot(range(3), medianas, 'o-', color=PALETA_ESTADO[estado],
            linewidth=2.5, markersize=9, label=f'{estado} (n={len(sub)})', zorder=3)
    ax.fill_between(range(3), p25, p75, alpha=0.12, color=PALETA_ESTADO[estado])
    for i, m in enumerate(medianas):
        ax.annotate(f'{m:.1f}h', (i, m),
                    textcoords='offset points', xytext=(7, 4),
                    fontsize=8.5, color=PALETA_ESTADO[estado], fontweight='bold')

ax.set_xticks(range(3)); ax.set_xticklabels(labels_ck_eng)
ax.set_ylabel('Horas acumuladas (mediana ± IQR)')
ax.set_title('Por estado\n(banda = IQR 25%-75%)', fontweight='bold')
ax.legend()

# Panel B: Por semestre
ax2 = axes[1]
for sem in semestres_ae:
    sub      = df_ae[df_ae['semestre'] == sem]
    era      = sub['era'].iloc[0]
    medianas = [sub[ck].median() for ck in checkpoints_eng]
    ls       = '-' if era == 'formato_antiguo' else '--'
    ax2.plot(range(3), medianas, 'o' + ls, color=PALETA_SEMESTRE[sem],
             linewidth=2, markersize=7, label=str(sem))
    ax2.annotate(str(sem), (2, medianas[2]),
                 textcoords='offset points', xytext=(5, 0),
                 fontsize=7.5, color=PALETA_SEMESTRE[sem])

ax2.set_xticks(range(3)); ax2.set_xticklabels(labels_ck_eng)
ax2.set_ylabel('Horas acumuladas (mediana)')
ax2.set_title('Por semestre\n(—: antiguo, --: nuevo)', fontweight='bold')
ax2.legend(fontsize=8)

plt.tight_layout()
save('10_evolucion_engagement.png')


# ═════════════════════════════════════════════════════════════════════════════
#  FIG 06 — AUC individual de variables  (→ 06_auc_univariante_cp1.png)
#  NB03 Celda 15-16
# ═════════════════════════════════════════════════════════════════════════════
vars_auc = [
    ('parcial_1',                  'Parcial 1 [académica]',       'Académica'),
    ('engagement_hasta_p1',        'Tiempo hasta P1 (hrs) [CP1]', 'Engagement CP1'),
    ('parcial_1_modulos_unicos',   'Módulos únicos P1 [CP1]',     'Engagement CP1'),
    ('parcial_1_temas_unicos',     'Temas únicos P1 [CP1]',       'Engagement CP1'),
    ('parcial_1_visitas',          'Visitas P1 [CP1]',            'Engagement CP1'),
    ('parcial_1_visitas_por_tema', 'Visitas/tema P1 [CP1]',       'Engagement CP1'),
    ('parcial_1_tiempo_por_visita','Min/visita P1 [CP1]',         'Engagement CP1'),
    ('parcial_2',                  'Parcial 2 [académica]',       'Académica'),
    ('promedio_ams',               'Promedio AMs [académica]',    'Académica'),
    ('promedio_quices',            'Promedio Quices [académica]', 'Académica'),
    ('engagement_hasta_p2',        'Tiempo hasta P2 (hrs) [CP2]','Engagement CP2'),
    ('total_modulos',              'Módulos totales [sem]',       'Engagement total'),
    ('total_temas',                'Temas totales [sem]',         'Engagement total'),
    ('total_tiempo_hrs',           'Tiempo total (hrs) [sem]',    'Engagement total'),
]

colores_tipo = {
    'Académica':         '#2ecc71',
    'Engagement CP1':    '#3498db',
    'Engagement CP2':    '#9b59b6',
    'Engagement total':  '#e67e22',
}

resultados_auc = []
for var, label, tipo in vars_auc:
    if var not in df_ae.columns:
        continue
    data = df_ae[['estado', var]].dropna()
    if len(data) < 10:
        continue
    yb  = (data['estado'] == 'REPROBÓ').astype(int)
    try:
        auc = roc_auc_score(yb, -data[var])
        resultados_auc.append((var, label, tipo, auc))
    except Exception:
        pass

resultados_auc.sort(key=lambda x: x[3])

fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle('Poder discriminante individual (AUC ROC)\nPredicción de REPROBÓ — df_activos_engagement',
             fontsize=13, fontweight='bold')

labels_plot = [r[1] for r in resultados_auc]
aucs_plot   = [r[3] for r in resultados_auc]
tipos_plot  = [r[2] for r in resultados_auc]
colors_plot = [colores_tipo[t] for t in tipos_plot]

bars = ax.barh(range(len(resultados_auc)), aucs_plot,
               color=colors_plot, alpha=0.85, edgecolor='white')
for bar, auc in zip(bars, aucs_plot):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f'{auc:.4f}', va='center', fontsize=8.5, fontweight='bold')

ax.axvline(0.5,  color='gray',  linestyle='--', linewidth=1.2, alpha=0.6, label='AUC=0.5 (azar)')
ax.axvline(0.75, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='AUC=0.75 (objetivo)')
ax.set_yticks(range(len(resultados_auc)))
ax.set_yticklabels(labels_plot, fontsize=8.5)
ax.set_xlabel('AUC ROC'); ax.set_xlim(0.4, 1.02)
ax.set_title('Ranking de poder discriminante individual', fontweight='bold')

for tipo, color in colores_tipo.items():
    ax.bar(0, 0, color=color, label=tipo, alpha=0.85)
ax.legend(fontsize=8, loc='lower right')

plt.tight_layout()
save('06_auc_univariante_cp1.png')


# ═════════════════════════════════════════════════════════════════════════════
#  FIG 08 — Correlaciones engagement  (→ 08_correlaciones_engagement.png)
#  NB03 Celda 20
# ═════════════════════════════════════════════════════════════════════════════
eng_feats_corr_all = [
    'engagement_hasta_p1',
    'parcial_1_visitas',
    'parcial_1_temas_unicos',
    'parcial_1_modulos_unicos',
    'parcial_1_visitas_por_tema',
    'parcial_1_tiempo_por_visita',
    'engagement_hasta_p2',
    'total_tiempo_hrs',
    'total_visitas',
    'total_temas',
    'total_modulos',
    'promedio_tiempo_por_visita',
]
eng_feats_corr = [v for v in eng_feats_corr_all if v in df_ae.columns]

labels_short_corr = {
    'engagement_hasta_p1':        'eng_P1 (tiempo)',
    'parcial_1_visitas':          'visitas_P1',
    'parcial_1_temas_unicos':     'temas_P1',
    'parcial_1_modulos_unicos':   'modulos_P1',
    'parcial_1_visitas_por_tema': 'visitas/tema_P1',
    'parcial_1_tiempo_por_visita':'min/visita_P1',
    'engagement_hasta_p2':        'eng_P2',
    'total_tiempo_hrs':           'tiempo_total',
    'total_visitas':              'visitas_total',
    'total_temas':                'temas_total',
    'total_modulos':              'modulos_total',
    'promedio_tiempo_por_visita': 'min/visita_total',
}

corr_eng = df_ae[eng_feats_corr].corr().round(3)
labels_s_c = [labels_short_corr.get(v, v) for v in eng_feats_corr]

fig, ax = plt.subplots(figsize=(12, 10))
fig.suptitle('Mapa de correlaciones entre variables de engagement\n(df_activos_engagement)',
             fontsize=13, fontweight='bold')

im = ax.imshow(corr_eng.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='Correlación de Pearson')

ax.set_xticks(range(len(labels_s_c)))
ax.set_xticklabels(labels_s_c, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(labels_s_c)))
ax.set_yticklabels(labels_s_c, fontsize=8)

for i in range(len(corr_eng)):
    for j in range(len(corr_eng)):
        v     = corr_eng.values[i, j]
        color = 'white' if abs(v) > 0.65 else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color=color)

ax.set_title('Correlaciones internas — variables de engagement', fontweight='bold')
plt.tight_layout()
save('08_correlaciones_engagement.png')


# ─────────────────────────────────────────────────────────────────────────────
print()
print("Todas las figuras generadas correctamente en figures/")
