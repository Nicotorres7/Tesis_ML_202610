# Screenshots Requeridos — Capítulo 7: Herramienta Web SAT Uniandes

## Descripción General

Este documento lista todos los screenshots necesarios para documentar visualmente la plataforma SAT Uniandes. Cada screenshot corresponde a una vista específica de la interfaz web.

---

## 1. Vista Inicial y Navegación

### Screenshot 1.1: Landing Page / Módulo Proyectos
**Ubicación en la aplicación:** Home → Módulo "Proyectos"  
**Descripción:** Pantalla de inicio mostrando:
- Barra lateral con navegación (Proyectos, Entrenamiento, Predicción, Dashboard, Modelos, Exportación)
- Listado de proyectos existentes con cards mostrando: nombre, descripción, cantidad de checkpoints, cantidad de modelos, fecha creación
- Botón "➕ Crear nuevo proyecto" con formulario expandible
- Barra superior de contexto con selectores de proyecto/checkpoint/modelo activos
- Barra de progreso del flujo

**Tamaño sugerido:** 1920x1080  
**Archivo esperado:** `fig_sat_01_landing.png`

---

### Screenshot 1.2: Barra Lateral de Navegación (Detalle)
**Ubicación en la aplicación:** Sidebar persistente  
**Descripción:** Zoom sobre:
- Logo y título "SAT Uniandes"
- Botones de radio para navegación (6 opciones con iconos numéricos)
- Nota informativa sobre barra fija
- Estados visuales: normal, hover, selected

**Tamaño sugerido:** 300x800  
**Archivo esperado:** `fig_sat_02_sidebar_nav.png`

---

### Screenshot 1.3: Barra Superior de Contexto
**Ubicación en la aplicación:** Top bar / Context bar  
**Descripción:**
- Sección izquierda: Chip de etapa, nombre de proyecto, metadata (checkpoint, modelo)
- Sección derecha: Barra de progreso del flujo, métricas KPI (Alto/Medio/Bajo)
- Fila de selectores: Proyecto activo, Checkpoint activo, Modelo en vista
- Barra de flujo: 6 pasos del pipeline (Proyecto, Checkpoint, Datos, Variables, Modelo, Predicción)

**Tamaño sugerido:** 1920x300  
**Archivo esperado:** `fig_sat_03_context_bar.png`

---

## 2. Flujo de Entrenamiento (4 Pasos)

### Screenshot 2.1: Paso 1 - Crear/Seleccionar Checkpoint
**Ubicación en la aplicación:** Entrenamiento → Paso 1  
**Descripción:**
- Stepper visual mostrando Paso 1 como activo
- Título: "Proyecto: **[nombre proyecto]**"
- Lista de checkpoints existentes con cards mostrando:
  - Nombre del checkpoint
  - Target configurado
  - Cantidad de features (numéricas/categóricas)
  - Cantidad de modelos
- Botón "Seleccionar" para cada checkpoint
- Expander "➕ Crear nuevo checkpoint" con formulario
- Botones de navegación: "← Atrás", "Continuar →"

**Tamaño sugerido:** 1200x900  
**Archivo esperado:** `fig_sat_04_training_step1.png`

---

### Screenshot 2.2: Paso 2 - Cargar Dataset
**Ubicación en la aplicación:** Entrenamiento → Paso 2  
**Descripción:**
- Stepper mostrando Paso 2 como activo
- Título: "Checkpoint: **[nombre checkpoint]**"
- Área de carga de archivo (drag & drop o clic para seleccionar)
- Vista previa post-carga mostrando:
  - Bloque de éxito: "Dataset listo — [número] filas · [número] columnas"
  - Expander "Vista previa de columnas" con tabla: Columna | Tipo | % Nulos | Valores únicos | Ejemplo
- Botones de navegación: "← Atrás", "Continuar a Variables →"

**Tamaño sugerido:** 1200x1000  
**Archivo esperado:** `fig_sat_05_training_step2.png`

---

### Screenshot 2.3: Paso 3 - Asignar Variables (Parte 1)
**Ubicación en la aplicación:** Entrenamiento → Paso 3  
**Descripción:**
- Stepper mostrando Paso 3 como activo
- Título: "Asignar columnas para **[nombre checkpoint]**"
- Nota informativa: "Columnas de identidad y target"
- Botón "Autodetectar columnas"
- Dos selectores en fila:
  - "Columna ID estudiante" (dropdown)
  - "Columna Nombre" (dropdown)
- Selector: "Columna Target (0 = aprobó · 1 = reprobó)"
- Línea separadora

**Tamaño sugerido:** 1200x500  
**Archivo esperado:** `fig_sat_06_training_step3_part1.png`

---

### Screenshot 2.4: Paso 3 - Asignar Variables (Parte 2)
**Ubicación en la aplicación:** Entrenamiento → Paso 3 (continuación)  
**Descripción:**
- Nota informativa: "Variables del modelo"
- Dos selectores multiselect en columnas:
  - "Features numéricas" con lista de opciones seleccionadas
  - "Features categóricas" con lista de opciones seleccionadas
- Botón de envío: "Guardar configuración" (primario)
- Botones de navegación: "← Atrás", "Continuar a Entrenamiento →"

**Tamaño sugerido:** 1200x600  
**Archivo esperado:** `fig_sat_07_training_step3_part2.png`

---

### Screenshot 2.5: Paso 4 - Configurar Entrenamiento (Parte 1)
**Ubicación en la aplicación:** Entrenamiento → Paso 4  
**Descripción:**
- Stepper mostrando Paso 4 como activo
- Título: "Entrenamiento — **[nombre checkpoint]**"
- Nota informativa sobre modelo activo anterior (si existe)
- **Columna izquierda:**
  - Selectbox: "Algoritmo" (LR, DT, RF, XGB)
  - Selectbox: "Validación" (Holdout, K-Fold, LOSO)
  - Selectbox: "Balanceo de clases" (none, SMOTE, SMOTEENN)
  - Selectbox: "Métrica objetivo" (f1, precision, recall, auc)
- **Columna derecha:**
  - Selectbox: "Hiperparámetros" (auto, fixed)
  - Number input: "Semilla aleatoria"
  - Slider: "Proporción de prueba (Holdout)"
  - Slider: "Combinaciones a explorar"

**Tamaño sugerido:** 1200x600  
**Archivo esperado:** `fig_sat_08_training_step4_part1.png`

---

### Screenshot 2.6: Paso 4 - Hiperparámetros Avanzados
**Ubicación en la aplicación:** Entrenamiento → Paso 4 (expander expandido)  
**Descripción:**
- Expander "Hiperparámetros avanzados (JSON)" en estado expandido
- Área de texto mostrando JSON con estructura de búsqueda
- Texto de ayuda: "Modifica el espacio de búsqueda. Deja en blanco para usar valores por defecto."
- Código JSON formateado con indentación

**Tamaño sugerido:** 1200x400  
**Archivo esperado:** `fig_sat_09_training_step4_advanced_params.png`

---

### Screenshot 2.7: Resultado Exitoso de Entrenamiento
**Ubicación en la aplicación:** Entrenamiento → Paso 4 (post-entrenamiento)  
**Descripción:**
- Bloque de éxito: "Modelo entrenado y guardado correctamente"
  - Versión, etiqueta, umbral óptimo
  - Ruta de archivo
- Cuatro métricas en tarjetas: AUC | Recall | Precisión | F1 (cada una con su valor)
- Título: "Comparación con modelo anterior"
- Gráfica comparativa (Plotly) mostrando barras lado a lado
- Sección "Siguiente: Modelo listo para predecir" con descripción
- Botones: "Ver en Modelos →", "Ir a Predicción →" (primario)

**Tamaño sugerido:** 1200x800  
**Archivo esperado:** `fig_sat_10_training_success.png`

---

## 3. Módulo Predicción

### Screenshot 3.1: Carga de Dataset para Predicción
**Ubicación en la aplicación:** Predicción  
**Descripción:**
- Título: "## Predicción"
- Nota informativa: "Modelo activo: v[número] · [etiqueta]"
  - Umbral, AUC, F1
  - Columnas requeridas listadas
- Área de carga: "Sube el dataset de estudiantes para predecir"
- Vista previa post-carga:
  - DataFrame con primeras 10 filas
  - Caption: "[número] filas · [número] columnas"
- Botón "Ejecutar predicción" (primario, deshabilitado hasta cargar)

**Tamaño sugerido:** 1200x700  
**Archivo esperado:** `fig_sat_11_prediction_upload.png`

---

### Screenshot 3.2: Resultados de Predicción
**Ubicación en la aplicación:** Predicción (post-ejecución)  
**Descripción:**
- Bloque de éxito: "Predicción completada — [timestamp]"
  - Total estudiantes
  - Conteos: [número] alto (rojo), [número] medio (amarillo), [número] bajo (verde)
- Tabla de resultados (primeras 20 filas): ID | Nombre | Probabilidad | Nivel Riesgo | Alerta
- Sección "Siguiente: Predicción lista para analizar" con botón "→ Ver Dashboard" (primario)

**Tamaño sugerido:** 1200x600  
**Archivo esperado:** `fig_sat_12_prediction_results.png`

---

## 4. Dashboard (3 Vistas)

### Screenshot 4.1: Dashboard - KPIs y Métricas
**Ubicación en la aplicación:** Dashboard → Parte superior  
**Descripción:**
- Cuatro tarjetas KPI en fila:
  - Total estudiantes (azul oscuro)
  - Alto riesgo (rojo)
  - Riesgo medio (amarillo)
  - Bajo riesgo (verde)
- Cada tarjeta muestra número prominente y etiqueta

**Tamaño sugerido:** 1200x150  
**Archivo esperado:** `fig_sat_13_dashboard_kpis.png`

---

### Screenshot 4.2: Dashboard - Vista Cohorte (Gráficas)
**Ubicación en la aplicación:** Dashboard → Tab "Vista Cohorte"  
**Descripción:**
- Dos gráficas lado a lado:
  - Izquierda: Distribución de riesgo (barras por categoría ALTO/MEDIO/BAJO)
  - Derecha: Histograma de probabilidades (distribución continua)
- Ambas gráficas con tema consistente (colores de riesgo)

**Tamaño sugerido:** 1400x500  
**Archivo esperado:** `fig_sat_14_dashboard_cohort_charts.png`

---

### Screenshot 4.3: Dashboard - Tabla de Estudiantes
**Ubicación en la aplicación:** Dashboard → Tab "Vista Cohorte" → Tabla  
**Descripción:**
- Título: "### Tabla de estudiantes"
- Tabla interactiva mostrando: ID | Nombre | Barra de probabilidad (visual) | Nivel de Riesgo
- Filas ordenadas por probabilidad descendente
- Colores de riesgo (rojo/amarillo/verde) en los badges de nivel

**Tamaño sugerido:** 1200x400  
**Archivo esperado:** `fig_sat_15_dashboard_students_table.png`

---

### Screenshot 4.4: Dashboard - Vista Individual (Seleccionar Estudiante)
**Ubicación en la aplicación:** Dashboard → Tab "Vista Individual"  
**Descripción:**
- Título: "### Vista Individual"
- Selectbox: "Selecciona un estudiante" con opciones "ID — Nombre"
- Tarjeta prominente de estudiante:
  - Probabilidad grande (ej. "45.2%")
  - Nombre, Nivel de Riesgo (badge coloreado)

**Tamaño sugerido:** 1200x300  
**Archivo esperado:** `fig_sat_16_dashboard_individual_select.png`

---

### Screenshot 4.5: Dashboard - Vista Individual (Análisis Detallado)
**Ubicación en la aplicación:** Dashboard → Tab "Vista Individual" (abajo)  
**Descripción:**
- Dos columnas:
  - **Izquierda:**
    - Gauge visual de probabilidad
    - Tabla: Variable | Efecto | Magnitud
  - **Derecha:**
    - Gráfica de percentiles (estudiante vs cohorte)
    - Gráfica de importancia de features

**Tamaño sugerido:** 1200x600  
**Archivo esperado:** `fig_sat_17_dashboard_individual_detail.png`

---

### Screenshot 4.6: Dashboard - Vista Modelo
**Ubicación en la aplicación:** Dashboard → Tab "Perfil del Modelo"  
**Descripción:**
- Dos gráficas lado a lado:
  - Izquierda: Radar chart de métricas (AUC, Recall, Precision, F1)
  - Derecha: Gráfica de importancia de features (barras horizontales)
- Abajo: Nota informativa con metadatos del modelo

**Tamaño sugerido:** 1400x600  
**Archivo esperado:** `fig_sat_18_dashboard_model_profile.png`

---

## 5. Módulo Modelos

### Screenshot 5.1: Timeline de Versiones Entrenadas
**Ubicación en la aplicación:** Modelos → Sección "Versiones entrenadas"  
**Descripción:**
- Título: "### Versiones entrenadas"
- Lista vertical mostrando cada versión con:
  - Versión (v1, v2, etc.)
  - Etiqueta (algoritmo + métrica objetivo)
  - Métricas: AUC [número] · F1 [número]
  - Badge: ACTIVO (verde) o INACTIVO (gris)
- Versiones en orden inverso (más reciente arriba)

**Tamaño sugerido:** 1200x350  
**Archivo esperado:** `fig_sat_19_models_timeline.png`

---

### Screenshot 5.2: Selector y Métricas de Modelo Seleccionado
**Ubicación en la aplicación:** Modelos → Selector y métricas  
**Descripción:**
- Selectbox: "Modelo a inspeccionar" con versión y etiqueta formateada
- Tarjeta de encabezado del modelo mostrando:
  - Versión, etiqueta, badge ACTIVO/INACTIVO
  - Metadata: Algoritmo, Objetivo, Umbral, Fecha
- Cuatro tarjetas de métricas: AUC | Recall | Precisión | F1

**Tamaño sugerido:** 1200x350  
**Archivo esperado:** `fig_sat_20_models_header_metrics.png`

---

### Screenshot 5.3: Tabs de Inspección de Modelo
**Ubicación en la aplicación:** Modelos → Tabs  
**Descripción:**
- Tres tabs: "Resumen", "Configuración", "Predicciones"
- En tab "Resumen":
  - **Izquierda:** Tabla de métricas de validación (AUC, Recall, Precision, F1, etc.)
  - **Derecha:** Tabla de decisiones del modelo (Umbral óptimo, Métrica objetivo, Algoritmo, Estado, Entrenado el)
- Colores coherentes, tablas HTML formateadas

**Tamaño sugerido:** 1400x500  
**Archivo esperado:** `fig_sat_21_models_overview_tab.png`

---

### Screenshot 5.4: Tab Configuración - Esquema de Features
**Ubicación en la aplicación:** Modelos → Tab "Configuración"  
**Descripción:**
- Título: Tab "Configuración"
- Dos tablas lado a lado:
  - Izquierda: Hiperparámetros finales
  - Derecha: Búsqueda y validación (modo búsqueda, iteraciones, validación, dataset origen)
- Tabla de esquema mostrando roles de columnas:
  - Rol (ID, Nombre, Target, Numéricas, Categóricas) | Columna (nombres)
- Bloque de éxito: "Archivo del modelo guardado en: [ruta]"

**Tamaño sugerido:** 1400x600  
**Archivo esperado:** `fig_sat_22_models_config_tab.png`

---

### Screenshot 5.5: Botón de Activación y Comparación con Modelo Activo
**Ubicación en la aplicación:** Modelos → Abajo  
**Descripción:**
- Botón primario (si modelo está inactivo): "Activar este modelo"
- Título: "#### Comparación con modelo activo"
- Gráfica Plotly mostrando barras lado a lado comparando métricas del modelo actual vs activo

**Tamaño sugerido:** 1200x400  
**Archivo esperado:** `fig_sat_23_models_activation_comparison.png`

---

## 6. Módulo Exportación

### Screenshot 6.1: Panel de Exportación
**Ubicación en la aplicación:** Exportación  
**Descripción:**
- Título: "## Exportación"
- Nota informativa: "[Proyecto] · [Checkpoint]"
  - Total estudiantes, conteos por riesgo (ALTO/MEDIO/BAJO)
- Tabla de preview (primeras 20 filas): ID | Nombre | Probabilidad | Nivel Riesgo | Alerta
- Dos botones lado a lado:
  - "Descargar Excel" (download button)
  - "Generar PDF" (download button)

**Tamaño sugerido:** 1200x700  
**Archivo esperado:** `fig_sat_24_export_panel.png`

---

## 7. Flujo Completo (Opcional - Composición)

### Screenshot 7.1: Flujo Completo Sintetizado
**Descripción:** Composición mostrando las 6 páginas principales del sistema en grid pequeño:
- Proyectos | Entrenamiento | Predicción
- Dashboard | Modelos | Exportación

**Tamaño sugerido:** 1920x1080  
**Archivo esperado:** `fig_sat_25_complete_flow_overview.png`

---

## Resumen de Archivos

| Num. | Archivo | Descripción |
|------|---------|-------------|
| 1.1 | fig_sat_01_landing.png | Landing page - Módulo Proyectos |
| 1.2 | fig_sat_02_sidebar_nav.png | Barra lateral de navegación |
| 1.3 | fig_sat_03_context_bar.png | Barra superior de contexto |
| 2.1 | fig_sat_04_training_step1.png | Entrenamiento - Paso 1 (Checkpoint) |
| 2.2 | fig_sat_05_training_step2.png | Entrenamiento - Paso 2 (Dataset) |
| 2.3 | fig_sat_06_training_step3_part1.png | Entrenamiento - Paso 3 Parte 1 (Variables identidad) |
| 2.4 | fig_sat_07_training_step3_part2.png | Entrenamiento - Paso 3 Parte 2 (Features) |
| 2.5 | fig_sat_08_training_step4_part1.png | Entrenamiento - Paso 4 Parte 1 (Config básica) |
| 2.6 | fig_sat_09_training_step4_advanced_params.png | Entrenamiento - Paso 4 (Parámetros avanzados JSON) |
| 2.7 | fig_sat_10_training_success.png | Resultado exitoso de entrenamiento |
| 3.1 | fig_sat_11_prediction_upload.png | Predicción - Carga de dataset |
| 3.2 | fig_sat_12_prediction_results.png | Predicción - Resultados post-ejecución |
| 4.1 | fig_sat_13_dashboard_kpis.png | Dashboard - Tarjetas KPI |
| 4.2 | fig_sat_14_dashboard_cohort_charts.png | Dashboard - Gráficas de cohorte |
| 4.3 | fig_sat_15_dashboard_students_table.png | Dashboard - Tabla de estudiantes |
| 4.4 | fig_sat_16_dashboard_individual_select.png | Dashboard - Seleccionar estudiante |
| 4.5 | fig_sat_17_dashboard_individual_detail.png | Dashboard - Análisis detallado individual |
| 4.6 | fig_sat_18_dashboard_model_profile.png | Dashboard - Perfil del modelo |
| 5.1 | fig_sat_19_models_timeline.png | Modelos - Timeline de versiones |
| 5.2 | fig_sat_20_models_header_metrics.png | Modelos - Encabezado y métricas |
| 5.3 | fig_sat_21_models_overview_tab.png | Modelos - Tab Resumen |
| 5.4 | fig_sat_22_models_config_tab.png | Modelos - Tab Configuración |
| 5.5 | fig_sat_23_models_activation_comparison.png | Modelos - Activación y comparación |
| 6.1 | fig_sat_24_export_panel.png | Exportación - Panel de descarga |
| 7.1 | fig_sat_25_complete_flow_overview.png | Flujo completo (opcional) |

---

## Procedimiento para Capturar Screenshots

1. **Entorno de prueba:** Ejecutar SAT Uniandes en navegador con datos de muestra.
2. **Resolución:** 1920x1080 para vistas completas, zoom según sea necesario.
3. **Datos de ejemplo:** Usar proyecto "Ejemplo Tesis" con:
   - Proyecto: "Cálculo I - 2025-2"
   - Checkpoint 1: "Semana 6 - Post Parcial 1"
   - Checkpoint 2: "Semana 11 - Post Parcial 2"
   - Modelos entrenados con métricas variadas para comparación.
4. **Formato:** PNG 1200x900 o mayor, sin marcas de agua.
5. **Organización:** Guardar en `/thesis/latex_sections/figuras/sat_app_screenshots/` con nombres según lista.

---

## Notas

- Todos los screenshots deben mostrar datos coherentes (mismo proyecto/checkpoint/modelo en las transiciones).
- Incluir la barra de navegación lateral para contexto.
- Si algunos datos son sensibles, usar nombres genéricos ("Estudiante 001", "Parcial 1", etc.).
- Capturar tanto estados iniciales como estados post-interacción (ej. formularios llenados, resultados visibles).

