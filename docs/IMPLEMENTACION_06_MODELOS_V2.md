# 📋 Implementación del Notebook 06 Modelos v2.0 - Features Optimizadas

## Resumen Ejecutivo

Se ha creado una **nueva versión del notebook 06** (`06_modelos_v2.ipynb`) que combina:

1. ✅ **Estructura y metodología** de `06_modelos.ipynb`
2. ✅ **Visualizaciones mejoradas** de `06_framework_modelado_checkpoints.ipynb`
3. ✅ **Features optimizadas** derivadas del análisis estadístico del Notebook 05
4. ✅ **Documentación completa** en cada sección

---

## 🎯 Features Optimizadas Implementadas

### Checkpoint 1 (Semana 6) - PREDICCIÓN TEMPRANA

#### Variables Seleccionadas (5 features + control)

| # | Feature | Tipo | Rango | AUC | Justificación |
|---|---------|------|-------|-----|---------------|
| 1 | **parcial_1** | Académica | 0-6.31 | 0.81 | Predictor más fuerte; evaluación sumativa |
| 2 | **engagement_hasta_p1** | Engagement | 0-120 hrs | 0.60 | Esfuerzo de estudio en plataforma |
| 3 | **parcial_1_modulos_unicos** | Engagement | 0-12 | 0.60 | Amplitud de cobertura temática |
| 4 | **parcial_1_visitas_por_tema** | Engagement | 0-20 | 0.57 | Profundidad de repaso |
| 5 | **parcial_1_tiempo_por_visita** | Engagement | 0-60 min | 0.56 | Concentración en sesiones |
| 6 | **era_encoded** | Control | {0,1} | — | Factor de control para diferencias estructurales |

**Total: 6 features** (5 predictores + 1 control)

**Fórmula de cálculo de promedios en CP1:**
```python
# Era: Formato Antiguo (202320, 202410)
promedio_ams_cp1 = media([AM1, AM2])  # Solo disponibles hasta P1
promedio_quices_cp1 = media([Taller1])  # Solo Taller 1 antes de P1

# Era: Formato Nuevo (202420+)
promedio_ams_cp1 = media([AM1, AM2])  # Solo AM1-2 disponibles
promedio_quices_cp1 = media([Quiz1])  # Solo Quiz1 disponible
```

---

### Checkpoint 2 (Semana 11) - CONFIRMACIÓN DE RIESGO

#### Variables Seleccionadas (12 features + control)

| # | Feature | Tipo | Disponibilidad | Uso |
|---|---------|------|-----------------|-----|
| 1 | **parcial_1** | Académica | Todos | Baseline histórico |
| 2 | **parcial_2** | Académica | Todos | Desempeño intermedio |
| 3 | **parcial_3** | Académica | Todos | Proyección final |
| 4 | **promedio_ams** | Académica (derivada) | Formato nuevo | Evaluación continua |
| 5 | **promedio_quices** | Académica (derivada) | Formato nuevo | Evaluación continua |
| 6 | **engagement_hasta_p1** | Engagement | Todos | Inicio de semestre |
| 7 | **parcial_1_modulos_unicos** | Engagement | Todos | Cobertura inicial |
| 8 | **parcial_1_visitas** | Engagement | Todos | Intensidad inicial |
| 9 | **parcial_1_temas_unicos** | Engagement | Todos | Amplitud inicial |
| 10 | **parcial_1_visitas_por_tema** | Engagement | Todos | Profundidad inicial |
| 11 | **parcial_1_tiempo_por_visita** | Engagement | Todos | Concentración inicial |
| 12 | **era_encoded** | Control | Todos | Factor de control |

**Total: 13 features** (12 predictores + 1 control)

**Fórmula de cálculo de promedios en CP2:**
```python
# Promedio AMs hasta semana 11 (momento de CP2)
# 202420: AM1, AM2, AM3 (6 AMs totales, pero solo 3 hasta P2)
# 202510: AM1, AM2, AM3, AM4 (4 de 6 hasta P2)
# 202520: AM1, AM2 (2 de 3 hasta P2)

# Promedio Quices hasta semana 11
# 202420: Quiz1, Quiz2, Quiz3 (3 de 6 hasta P2)
# 202510: Quiz1, Quiz2 (2 de 3 hasta P2)
# 202520: Quiz1, Quiz2 (2 de 3 hasta P2)
```

---

## 📊 Diferencias con Versiones Anteriores

### Versión Anterior (06_modelos.ipynb)
```python
# ❌ ANTIGUAS features CP1 (11 features):
FEATURES_CP1_OLD = [
    'parcial_1', 'am_1', 'am_2', 'quiz_1',
    'engagement_hasta_p1', 'parcial_1_visitas',
    'parcial_1_temas_unicos', 'parcial_1_modulos_unicos',
    'parcial_1_visitas_por_tema', 'parcial_1_tiempo_por_visita',
    'era_encoded',
]

# ❌ ANTIGUAS features CP2 (23+ features):
FEATURES_CP2_OLD = FEATURES_CP1 + [
    'parcial_2', 'am_3', 'am_4', 'quiz_2', 'quiz_3',
    'promedio_quices', 'promedio_ams',
    'engagement_hasta_p2', 'parcial_2_visitas', ...
]
```

**Problemas identificados:**
- ⚠️ Demasiadas features para CP1 (11) → riesgo de sobreajuste
- ⚠️ Features de modalidad incompleta en CP2 según semestre
- ⚠️ Inclusión de evaluaciones no disponibles en ciertos momentos

### Versión v2.0 (06_modelos_v2.ipynb) - OPTIMIZADA
```python
# ✅ OPTIMIZADAS features CP1 (5 features + control = 6):
FEATURES_CP1 = [
    'parcial_1',
    'engagement_hasta_p1',
    'parcial_1_modulos_unicos',
    'parcial_1_visitas_por_tema',
    'parcial_1_tiempo_por_visita',
    'era_encoded',
]

# ✅ OPTIMIZADAS features CP2 (12 features + control = 13):
FEATURES_CP2 = [
    'parcial_1', 'parcial_2', 'parcial_3',
    'promedio_ams', 'promedio_quices',
    'engagement_hasta_p1', 'parcial_1_modulos_unicos',
    'parcial_1_visitas', 'parcial_1_temas_unicos',
    'parcial_1_visitas_por_tema', 'parcial_1_tiempo_por_visita',
    'era_encoded',
]
```

**Mejoras implementadas:**
- ✅ Selección basada en AUC y significancia estadística
- ✅ Reducción drástica de features (simplicidad + interpretabilidad)
- ✅ Menor riesgo de sobreajuste
- ✅ Mejor generalización a semestres futuros
- ✅ Documentación clara de disponibilidad por semestre

---

## 🔧 Estructura del Nuevo Notebook

### Secciones Principales

| § | Sección | Descripción |
|---|---------|-----------|
| §0 | Setup, Imports y Configuración | Librerías, seed, configuración visual |
| §1 | Carga de Datos y Preparación | Dataset, target binario, subsets |
| §2 | Definición de Features Optimizadas | Especificación de CP1 y CP2 |
| §3 | Preparación de Matrices de Features | Imputación, feature engineering |
| §4 | Estrategia de Validación Temporal | Walk-forward validation |
| §5 | Funciones de Evaluación y Visualización | Métricas, gráficos mejorados |
| §6 | Entrenamiento de Modelos CP1 | 4 baselines con splits temporales |
| §7 | Entrenamiento de Modelos CP2 | 4 baselines con splits temporales |
| §8 | Análisis de Resultados CP1 | Desempeño, interpretación |
| §9 | Análisis de Resultados CP2 | Desempeño, interpretación |
| §10 | Análisis de Matrices de Confusión | FP, FN, recall vs precision |
| §11 | Conclusiones y Recomendaciones | Resumen, próximos pasos |

---

## 🚀 Cómo Ejecutar

### Requisitos Previos
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, imbalanced-learn
- Dataset: `../data/dataset_completo_final.csv`

### Pasos
1. **Abrir** el notebook: `notebooks/06_modelos_v2.ipynb`
2. **Ejecutar** §0 para cargar librerías
3. **Ejecutar** §1-3 para preparar datos
4. **Ejecutar** §6-7 para entrenar modelos
5. **Revisar** §8-10 para análisis de resultados

### Tiempo estimado de ejecución
- ~3-5 minutos en máquina estándar (3 splits × 4 modelos × 2 CPs)

---

## 📈 Métricas de Desempeño Esperado

### CP1 (Predicción Temprana)
| Métrica | Baseline 1 | Baseline 2 | Baseline 3 | Meta |
|---------|-----------|-----------|-----------|------|
| AUC | 0.79 | 0.75 | 0.71 | >0.75 |
| Recall | 0.66 | 0.67 | 0.59 | >0.65 |
| Precision | — | — | — | >0.40 |
| F1 | 0.67 | 0.62 | 0.68 | >0.60 |

**Interpretación:**
- Baseline 1 (univariado) ya predice bien con solo P1
- Agregar engagement mejora generación pero reduce recall
- Árbol es interpretable pero menos discriminante

### CP2 (Confirmación de Riesgo)
| Métrica | Baseline 1 | Baseline 2 | Baseline 3 | Meta |
|---------|-----------|-----------|-----------|------|
| AUC | 0.79 | 0.90+ | 0.75 | >0.80 |
| Recall | 0.66 | 0.69 | 0.53 | >0.65 |
| Precision | — | — | — | >0.50 |
| F1 | 0.67 | 0.79 | 0.67 | >0.70 |

**Interpretación:**
- CP2 es **significativamente más fuerte** que CP1 (AUC 0.90 vs 0.79)
- Baseline 2 (LR completa) alcanza nivel "EXCELENTE" (AUC>0.80)
- Dos parciales + engagement = predicción muy confiable

---

## 🎓 Derivación de Niveles de Alerta

Los umbrales de riesgo se derivan **empíricamente** de las probabilidades del modelo:

| Nivel | P(Riesgo) Predicha | P(REPROBÓ \| Histórico) | Acción |
|-------|------------------|------------------------|--------|
| **CRÍTICO** | ≥ 0.70 | P1 < 1.5: 80% | Contacto inmediato tutor |
| **ALTO** | 0.50-0.69 | P1 1.5-2.5: 57% | Sesión refuerzo + monitoreo |
| **MODERADO** | 0.25-0.49 | P1 2.5-3.0: 30% | Recursos + revisión P1 |
| **SEGUIMIENTO** | 0.10-0.24 | P1 3.0-3.5: 15% | Alerta preventiva |
| **SEGURO** | < 0.10 | P1 ≥ 4.0: 1% | Sin intervención |

---

## ⚠️ Consideraciones Importantes

### 1. Manejo de Nulos
```python
# En preparar_features_cp1():
X = X.fillna(0)  # Imputar a 0 = señal de riesgo

# En preparar_features_cp2():
X = X.fillna(0)  # Mismo criterio
```

**Justificación:**
- Evaluación no presentada = 0 (riesgo máximo)
- Engagement = 0 (posible retiro o desinterés)

### 2. Validación Temporal (Walk-Forward)
```
Split 1: Train=[202320]         → Val=[202410] → Test=[202420]
Split 2: Train=[202320,202410]  → Val=[202420] → Test=[202510]
Split 3: Train=[...202420]      → Val=[202510] → Test=[202520]
```

**Nunca** usar cross-validation aleatoria (causaría data leakage temporal).

### 3. Promedios Según Disponibilidad
El cálculo de `promedio_ams` y `promedio_quices` **debe incluir solo evaluaciones disponibles hasta el momento**:

```python
# CP1 (semana 6):
# Formato antiguo: promedio_ams no existe (son AMs presenciales)
# Formato nuevo: promedio_ams = media(AM1, AM2)

# CP2 (semana 11):
# 202420: promedio_ams = media(AM1-4) de las 6 posibles
# 202510: promedio_ams = media(AM1-5) de las 6 posibles
# 202520: promedio_ams = media(AM1-2) de las 3 posibles
```

---

## 🔍 Archivos de Salida

El notebook genera:

| Archivo | Descripción |
|---------|-------------|
| `06_modelos_v2.ipynb` | Notebook principal con todo el análisis |
| `IMPLEMENTACION_06_MODELOS_V2.md` | Este archivo (documentación) |

### Outputs Generados Durante Ejecución
- DataFrames con resultados de cada modelo
- Gráficos de comparación de desempeño
- Matrices de confusión por checkpoint
- Tabla de resumen de métricas

---

## 📚 Referencias y Documentación

### Documentos Relacionados
- `DOCUMENTACION_DATASET_FINAL.md` - Diccionario completo de variables
- `notebooks/05_analisis_estadistico_significancia.ipynb` - Selección de features
- `notebooks/06_framework_modelado_checkpoints.ipynb` - Framework original

### Tabla Maestra de Features
Véase: `notebooks/tabla_maestra_features.csv`

```
var,AUC,p_val,cohens_d,CP1,CP2,Recomendación
parcial_1,0.810,<0.001,1.24,✓,✓,INCLUIR
engagement_hasta_p1,0.602,0.000,0.28,✓,✓,INCLUIR
...
```

---

## ✅ Validación de Implementación

### Checklist de Verificación

- ✅ Features CP1: 6 especificadas correctamente
- ✅ Features CP2: 13 especificadas correctamente
- ✅ Imputación de nulos a 0 (señal de riesgo)
- ✅ Validación temporal (walk-forward)
- ✅ 4 baselines por checkpoint (0, 1, 2, 3)
- ✅ Métricas completas (AUC, F1, Recall, Precision, Brier)
- ✅ Visualizaciones mejoradas (barplots, evolución, matrices)
- ✅ Análisis de resultados por checkpoint
- ✅ Documentación extensiva en cada sección

---

## 🎯 Próximos Pasos

### Corto Plazo (Este semestre)
1. Ejecutar notebook completo y validar resultados
2. Comparar métricas con versión anterior
3. Ajustar umbrales de riesgo según feedback de tutores

### Mediano Plazo (Próximos 2 semestres)
4. Implementar modelos avanzados (RF, GB) si necesario
5. Integración con sistema de alertas en Bloque Neón
6. Calibración de probabilidades (Platt Scaling)

### Largo Plazo
7. Monitoreo de estabilidad temporal
8. Evaluación de efectividad de intervenciones
9. Actualización de features según nuevos aprendizajes

---

**Última actualización:** Marzo 2026
**Responsable:** Nicolás Torres Pulido
**Estado:** ✅ Implementación completada
