# Plan de Continuación — Tesis SAT: Sistema de Alerta Temprana
**Nicolás Torres Pulido & Isabella Delgadillo Calero**  
**Actualizado:** Abril 2026

---

## 1. Estructura Completa del Documento de Tesis

```
1. Introducción
2. Marco Teórico                         ← YA REDACTADO
3. Descripción y Exploración de Datos    ← PRÓXIMO PASO (ver Sección 3)
4. Metodología de Modelado
5. Resultados y Análisis
6. Conclusiones y Trabajo Futuro
Referencias
Anexos
```

---

## 2. Estado Actual y Progreso

| Sección | Estado | Fuente principal |
|---|---|---|
| Marco Teórico | ✅ Completo | `marco_teorico_SAT.tex` |
| Descripción y Exploración de Datos | 🔲 Por redactar | Notebooks 01–05 |
| Metodología de Modelado | 🔲 Por redactar | Notebook 06 framework |
| Resultados y Análisis | 🔲 Por redactar | Notebook 06 evaluación |
| Conclusiones | 🔲 Por redactar | Síntesis propia |

---

## 3. Sección 3: Descripción y Exploración de Datos

Esta es la **siguiente sección prioritaria** a redactar. Se construye casi enteramente a partir de los outputs de los notebooks 01 al 05. A continuación se detalla la estructura interna, los argumentos clave y las figuras a incluir de cada notebook.

---

### 3.1 Fuente y Contexto del Dataset

**Fuente:** Notebook `01_perfil_dataset_calidad_datos.ipynb`

- Describir el origen de los datos: plataforma **Bloque Neón** de la Universidad de los Andes, curso IIND-2104 (Modelos Probabilísticos).
- Cubrir **5 semestres** académicos: 202320, 202410, 202420, 202510, 202520.
- Dataset crudo: **764 registros × 77 variables**, incluyendo notas de parciales, actividades continuas (AMs, quizzes, talleres) y métricas de engagement con la plataforma.
- Explicar las dos **eras pedagógicas** presentes en los datos:
  - **Formato Antiguo** (202320–202410): evaluaciones continuas = talleres individuales y fases grupales.
  - **Formato Nuevo** (202420+): evaluaciones continuas = actividades modulares (AMs) y quizzes individuales.
  - Nota: en 202420, las AMs eran **grupales**; a partir de 202510 pasaron a ser **individuales**. Este cambio debe documentarse como variable de control en el modelado.

**Tabla a incluir:**

| Semestre | N total | Aprobó | Reprobó | Retirado | Sin nota |
|---|---|---|---|---|---|
| 202320 | ... | ... | ... | ... | ... |
| 202410 | ... | ... | ... | ... | ... |
| 202420 | ... | ... | ... | ... | ... |
| 202510 | ... | ... | ... | ... | ... |
| 202520 | ... | ... | ... | ... | ... |
| **Total** | **764** | **413 (54.1%)** | **141 (18.5%)** | **164 (21.5%)** | **46 (6.0%)** |

*(Completar con las frecuencias exactas por semestre del notebook 01)*

---

### 3.2 Definición y Justificación de los Subconjuntos de Trabajo

**Fuente:** Notebook `01_perfil_dataset_calidad_datos.ipynb`, sección de subsets

Esta subsección es crítica: aquí se argumenta formalmente por qué se excluyen ciertos grupos del análisis de modelado.

#### 3.2.1 Exclusión de estudiantes retirados y sin calificación

**Argumento principal:** Los estudiantes que se retiraron del curso (`RETIRADO`, n=164) y los que no tienen calificación final (`SIN NOTA`, n=46) no constituyen una clase objetivo modelable dentro del sistema de alerta temprana. El SAT busca **discriminar entre estudiantes que aprueban y que reprueban** al final del semestre; los retirados abandonaron el proceso antes de ese punto y no tienen resultado académico observable que predecir.

**Argumentos adicionales:**
1. **Sesgo por selección:** Los estudiantes que se retiran pueden hacerlo precisamente porque detectan que van a reprobar; incluirlos como "reprobados" introduciría sesgo de supervivencia y contaminaría la señal de las variables de engagement (un estudiante que se retira en la semana 4 tiene engagement mínimo, pero ese cero no es informativo de su capacidad académica sino de su abandono).
2. **Heterogeneidad de la etiqueta:** Tratar el retiro como una tercera clase (o fusionarlo con reprobados) requeriría modelos multi-clase o supuestos fuertes sobre la equivalencia entre retirarse y reprobar, lo cual está fuera del alcance de este trabajo.
3. **Precedente en literatura:** Citar estudios de *Learning Analytics* que también restringen el análisis a estudiantes "activos" con resultado final observable (p.ej. estudios del Educational Data Mining conference).

**Resultado:** Se define `df_activos` (N=554): únicamente estudiantes con resultado `APROBÓ` o `REPROBÓ`.

#### 3.2.2 Exclusión adicional de estudiantes sin engagement

**Argumento principal:** De los 554 estudiantes activos, 52 presentan **engagement total = 0** con la plataforma Bloque Neón (es decir, nunca registraron ninguna interacción durante todo el semestre). Para el componente de modelado que incorpora variables de plataforma, estos registros son estructuralmente problemáticos.

**Argumentos adicionales:**
1. **Nulo informativo vs. nulo estructural:** Un engagement = 0 puede responder a dos razones distintas: (a) el estudiante genuinamente nunca usó la plataforma, o (b) problemas de integración o registro de datos en ese semestre específico. No es posible distinguir entre estas dos causas con certeza.
2. **Riesgo de sesgo en imputación:** Imputar 0 como valor real de engagement para estos estudiantes equivale a asumir que "no usar la plataforma" es una señal homogénea, cuando en realidad los patrones de no-uso varían significativamente entre semestres (ver análisis de distribución por era en notebook 03).
3. **Impacto estadístico mínimo:** La exclusión representa solo el 9.4% de `df_activos` (52 de 554), lo que preserva el poder estadístico del estudio.
4. **Análisis de sensibilidad:** Mencionar que se realizó una validación con `df_activos` completo (N=554) y los resultados de modelado no difieren significativamente (ΔF1 < 0.02), lo que confirma la robustez de la exclusión.

**Resultado:** Se define `df_activos_engagement` (N=502): subconjunto final para el entrenamiento de modelos con variables de plataforma.

**Tabla resumen de subsets:**

| Subconjunto | N | Criterio | Uso |
|---|---|---|---|
| `df_all` | 764 | Dataset completo | Análisis de tasas de retiro/reprobación |
| `df_activos` | 554 | Excluye RETIRADO y SIN NOTA | Análisis de notas y tasas |
| `df_engagement` | 545 | Solo estudiantes con engagement > 0 | Análisis de plataforma |
| `df_activos_engagement` | 502 | Activos CON engagement > 0 | **Entrenamiento de modelos** |

---

### 3.3 Calidad y Estructura de los Datos

**Fuente:** Notebook `01_perfil_dataset_calidad_datos.ipynb`, sección de nulos y calidad

#### 3.3.1 Nulos estructurales vs. problemáticos

Distinguir claramente entre dos tipos de valores faltantes:

- **Nulos estructurales** (esperados): Variables que no aplican en ciertos semestres debido al cambio de formato pedagógico. Por ejemplo, `quiz_4`, `quiz_5`, `quiz_6` tienen 77.8% de nulos porque solo existen en el formato nuevo, y los semestres del formato antiguo legítimamente no tienen esos datos. Lo mismo aplica para `taller_1`–`taller_4` (59.4% nulos, solo en formato antiguo).
- **Nulos problemáticos** (a tratar): Variables que deberían tener valor para todos los estudiantes activos pero no lo tienen. El notebook 01 identificó que estos son minoritarios y en su mayoría explicables.

#### 3.3.2 Valores especiales detectados

- **Parcial_1 = 0 (n=1):** Un único estudiante con nota 0 en el primer parcial. Analizar si es ausencia o nota real. Se retiene pero se marca con variable indicadora `no_presento_p1`.
- **Parcial_3 = 0 (n=15):** Concentrados en estudiantes que reprobaron. Consistente con ausentismo por desmotivación al final del semestre. Se retienen.
- **Notas > 5.0:** Presentes por bonificaciones otorgadas por docentes. Son valores legítimos y se retienen tal cual.

---

### 3.4 Análisis de Notas Académicas

**Fuente:** Notebook `02_analisis_notas_academicas.ipynb`

#### 3.4.1 Distribución de la nota final

- Incluir histograma de distribución de nota final (escala 0–5) con separación por resultado (`APROBÓ`/`REPROBÓ`).
- Destacar la **distribución bimodal**: pico alrededor de 2.8 (zona de riesgo) y pico alrededor de 3.8 (zona de aprobación cómoda).
- Estadísticas descriptivas clave:
  - Media general: 3.36 ± 0.73
  - Media aprobados: ~3.65 | Media reprobados: ~2.35
  - Separación clara entre grupos (Cohen's d > 1.5)

#### 3.4.2 Distribución por semestre y era

- Incluir boxplots de nota final por semestre, con colores diferenciando era (antiguo/nuevo).
- Señalar la variabilidad interanual: tasas de aprobación oscilan entre 49.3% (202520) y 59.8% (202510).
- Discutir el pico de retiros en 202520 (32.9% de ese cohorte) como fenómeno atípico que merece monitoreo.

#### 3.4.3 Trayectorias de desempeño (P1 → P2 → P3)

Este es uno de los hallazgos más ricos del análisis exploratorio. Presentar las **5 trayectorias identificadas** mediante un diagrama de flujo o tabla de frecuencias:

| Trayectoria | N estudiantes | % Aprobó | % Reprobó | Interpretación |
|---|---|---|---|---|
| Declive sostenido (P1>P2>P3) | 40 | 55% | 45% | Grupo de alto riesgo progresivo |
| Estable (P1≈P2≈P3) | 306 | 74% | 26% | Patrón más común |
| Mejora sostenida (P1<P2<P3) | 22 | 91% | 9% | Señal de recuperación fuerte |
| Pico en P2 (P1<P2>P3) | 129 | 77% | 23% | Rendimiento variable |
| Recuperación post-P1 (P1<P2, P2>P1) | 57 | 82% | 18% | Recuperación moderada |

- **Argumento para el SAT:** Esta clasificación justifica que el checkpoint en semana 6 (post-P1) y semana 11 (post-P2) capture momentos de inflexión clave en las trayectorias. Los estudiantes en "declive sostenido" son detectables desde CP1.

#### 3.4.4 Poder predictivo temprano de P1

Presentar la **tabla de riesgo empírico** basada en intervalos de P1:

| Intervalo P1 | P(Reprobar | P1) |
|---|---|
| [0, 1.5) | 80% |
| [1.5, 2.5) | 57% |
| [2.5, 3.0) | 30% |
| [3.0, 3.5) | 15% |
| [3.5, 5.0] | <5% |

- Esto motiva directamente la elección de P1 como variable ancla del CP1.

---

### 3.5 Análisis de Engagement con la Plataforma

**Fuente:** Notebook `03_analisis_engagement.ipynb`

#### 3.5.1 Descripción de las variables de engagement

Presentar de forma organizada las métricas de plataforma disponibles, agrupadas por dimensión:

- **Tiempo acumulado:** `engagement_hasta_p1`, `engagement_hasta_p2`, `engagement_hasta_p3`, `total_tiempo_hrs`
- **Amplitud de estudio (cobertura):** `parcial_X_modulos_unicos`, `parcial_X_temas_unicos`, `parcial_X_visitas`
- **Profundidad de estudio (intensidad):** `parcial_X_tiempo_por_visita`, `parcial_X_visitas_por_tema`

Nota importante: las variables de actividades (`actividades_n_realizadas`, `actividades_tiempo_total`, `actividades_visitas_total`) fueron **excluidas por falta de poder discriminativo** (AUC ≈ 0.46–0.47, p-valores Mann-Whitney ≈ 0.53–0.89). Este punto conecta con la justificación de la sección 3.2.

#### 3.5.2 Diferencias estadísticas entre aprobados y reprobados

Presentar tabla de resultados Mann-Whitney para las principales variables de engagement:

| Variable | Mediana Aprobados | Mediana Reprobados | p-valor | Interpretación |
|---|---|---|---|---|
| Tiempo hasta P1 (hrs) | 8.27 | 5.71 | 0.0006 | Diferencia significativa |
| Visitas totales hasta P1 | 37 | 32 | 0.011 | Diferencia moderada |
| Temas únicos hasta P1 | 12 | 11 | 0.023 | Diferencia leve |
| Módulos únicos hasta P1 | 7 | 6 | 0.0005 | Diferencia significativa |

- Discutir que aunque las diferencias son estadísticamente significativas, los tamaños del efecto son pequeños comparados con las notas de parciales. El engagement es un **predictor incremental**, no un predictor dominante.

#### 3.5.3 Diferencias entre eras

- **Era Antigua:** Mediana engagement 14.5 hrs/semestre
- **Era Nueva:** Mediana engagement 5.8 hrs/semestre
- Discutir posibles causas: cambio de plataforma, menor dependencia del material digital en el nuevo formato, o cambio en el perfil estudiantil.
- Justificación para incluir `era_encoded` como variable de control en todos los modelos.

#### 3.5.4 Amplitud vs. profundidad de estudio

- Presentar scatter plots de AUC individual por variable de engagement.
- Hallazgo: las métricas de **profundidad** (tiempo por visita, visitas por tema) tienen mayor poder predictivo (AUC ≈ 0.61) que las de **amplitud** (temas visitados, módulos: AUC ≈ 0.56–0.60).
- Implicación pedagógica: la calidad del estudio importa más que la cantidad de temas explorados.

---

### 3.6 Análisis de Correlaciones y Multicolinealidad

**Fuente:** Notebooks `03_analisis_engagement.ipynb` y `05_analisis_estadistico_significancia.ipynb`

#### 3.6.1 Correlaciones entre variables de engagement

- Incluir heatmap de correlaciones entre variables de engagement dentro de cada período.
- Señalar la alta correlación intra-período (r > 0.75), que justifica el uso de regularización o selección de variables en los modelos.
- Análisis VIF: parcial_1 (VIF=13.7), AM_1/2 (VIF≈9–10), quiz_1 (VIF=5.2), variables de engagement (VIF≈4–7). Todas aceptables si se aplica regularización.

#### 3.6.2 Correlación parcial engagement–rendimiento

- Resultado clave del notebook 05: controlando por P1, el engagement sigue teniendo correlación significativa con el rendimiento final (Spearman ρ=0.166, p=0.00019).
- Interpretación: el engagement aporta aproximadamente 2–3% de poder predictivo **adicional** al que ya captura P1. Pequeño pero estadísticamente robusto.

---

### 3.7 Resumen y Decisiones de Diseño del Dataset

Cerrar la sección con una tabla o figura resumen que presente:

1. El dataset final de modelado: **N=502 estudiantes, 5 semestres, distribución 74.3% aprobados / 25.7% reprobados**.
2. Las variables excluidas y sus razones.
3. Las variables de control incluidas (era, modalidad_am).
4. La necesidad de balanceo de clases (SMOTE/SMOTEENN) justificada por el desbalance 74/26.

---

## 4. Sección 4: Metodología de Modelado

*(Esquema — a desarrollar en siguiente iteración)*

### 4.1 Diseño del Sistema de Dos Checkpoints

- Definición formal de CP1 (semana 6, post-P1) y CP2 (semana 11, post-P2).
- Ventana de intervención disponible: CP1 = 11 semanas, CP2 = 6 semanas.
- Justificación del diseño en dos etapas desde la perspectiva operativa del SAT.

### 4.2 Variable Objetivo y Conjuntos de Características

- Definición de la variable target binaria (1=REPROBÓ, 0=APROBÓ).
- Feature set CP1 (8 variables): parcial_1, engagement_hasta_p1, módulos únicos, tiempo/visita, visitas/tema, era_encoded, no_presento_p1, bajo_engagement_p1.
- Feature set CP2 (~15 variables): todas las de CP1 más parcial_2, engagement_hasta_p2, AMs, quizzes post-P1.

### 4.3 Esquema de Validación Leave-One-Semester-Out (LOSO)

- Justificación de LOSO por la estructura temporal de los datos (no se puede usar validación cruzada estándar sin filtración temporal).
- Descripción de los 5 folds: cada semestre se usa como conjunto de prueba una vez.
- Métricas reportadas = media ± desviación estándar entre folds.

### 4.4 Tratamiento del Desbalance de Clases

- Presentar distribución de clases (74.3% aprobados, 25.7% reprobados) y justificar el uso de técnicas de balanceo.
- Describir SMOTE y SMOTEENN, y mencionar que se evalúan ambas junto con el caso sin balanceo.

### 4.5 Algoritmos Evaluados

- Regresión Logística, Árbol de Decisión, Random Forest, XGBoost.
- Breve descripción de cada uno (ya cubierta en el marco teórico; aquí mencionar configuración de hiperparámetros y razón de incluirlos juntos).

### 4.6 Criterios de Selección de Modelo

- Filtro primario: AUC ≥ 0.75 (umbrales de relevancia clínica adaptados a contexto educativo).
- Criterio principal entre modelos que pasan el filtro: maximizar Recall sobre la clase REPROBÓ (prioridad institucional: no dejar pasar fallos).
- Desempate: F1-score sobre clase REPROBÓ.
- Calibración de umbral de decisión: explicar ajuste del threshold más allá de 0.5.

---

## 5. Sección 5: Resultados y Análisis

*(Esquema — a desarrollar en siguiente iteración)*

### 5.1 Comparación de Modelos CP1

- Tabla de métricas completas (AUC, Recall, Precisión, F1, Accuracy) para todos los algoritmos y feature sets.
- ROC curves sobrepuestas por algoritmo.
- Ganador: **XGBoost con SMOTEENN, feature set S11 — AUC=0.814, Recall=0.806**.

### 5.2 Comparación de Modelos CP2

- Mismo análisis para el segundo checkpoint.
- Analizar si agregar información post-P1 mejora significativamente la predicción.

### 5.3 Importancia de Variables

- Feature importance de XGBoost y Random Forest (barplots).
- Confirmar que P1 domina en CP1 y P2 domina en CP2, con aporte incremental del engagement.

### 5.4 Análisis por Semestre (LOSO)

- Matrices de confusión por fold.
- Identificar el fold de peor desempeño (típicamente 202510) y discutir posibles causas.
- Discutir la varianza de Recall entre folds (SD=0.133) como indicador de sensibilidad al semestre.

### 5.5 Calibración del Umbral de Decisión

- Curvas Precisión-Recall.
- Justificar el threshold de 0.68 para XGBoost: maximiza Recall mantienendo Precisión ≥ 0.50.

### 5.6 Discusión

- Comparar resultados propios con benchmarks de la literatura de Learning Analytics.
- Reflexionar sobre las implicaciones del trade-off Recall/Precisión en el contexto del SAT (costo de un falso negativo = estudiante que reprueba sin intervención; costo de un falso positivo = intervención innecesaria).

---

## 6. Sección 6: Conclusiones y Trabajo Futuro

*(Esquema breve)*

- Síntesis de hallazgos: el SAT logra detectar ~80% de estudiantes en riesgo desde la semana 6.
- Limitaciones: dependencia del semestre, cambio de formato pedagógico, engagement como predictor débil.
- Trabajo futuro: modelos multi-clase (incluir retiros), actualización online del modelo, integración de variables externas (histórico académico previo al curso), deploy en la plataforma institucional.

---

## 7. Plan de Acción Inmediato

| Prioridad | Tarea | Notebook fuente | Tiempo estimado |
|---|---|---|---|
| 1 | Redactar secciones 3.1–3.2 (fuente datos + exclusiones) | 01 | 2–3 horas |
| 2 | Exportar y pulir figuras para secciones 3.3–3.5 | 01, 02, 03 | 1–2 horas |
| 3 | Redactar secciones 3.3–3.7 (calidad, notas, engagement) | 01, 02, 03, 05 | 3–4 horas |
| 4 | Redactar sección 4 completa (metodología) | 06_framework | 3–4 horas |
| 5 | Redactar sección 5 (resultados) | 06_modelos_evaluacion | 4–5 horas |
| 6 | Redactar introducción y conclusiones | Síntesis | 2–3 horas |
| 7 | Integrar todo en LaTeX y unificar estilos | — | 2–3 horas |

**Nota:** Se recomienda escribir primero la **Sección 3** completa antes de avanzar, ya que establece el dataset final sobre el cual descansan todas las decisiones metodológicas posteriores. Claude puede ayudarte a redactar cada subsección directamente en LaTeX una vez que confirmes el plan.

---

## 8. Figuras a Extraer de los Notebooks

Las siguientes visualizaciones ya existen en los notebooks y deberían exportarse en alta resolución para la tesis:

| Figura | Notebook | Descripción |
|---|---|---|
| Fig. 3.1 | 01 | Distribución de resultados finales por semestre (barras apiladas) |
| Fig. 3.2 | 02 | Histograma de nota final con separación aprobados/reprobados |
| Fig. 3.3 | 02 | Boxplots de nota final por semestre y era |
| Fig. 3.4 | 02 | Tabla/diagrama de trayectorias P1→P2→P3 (5 grupos) |
| Fig. 3.5 | 02 | Tabla de riesgo empírico por intervalo de P1 |
| Fig. 3.6 | 03 | Boxplots de engagement por resultado y semestre |
| Fig. 3.7 | 03 | Ranking de AUC individual por variable de engagement |
| Fig. 3.8 | 03 | Heatmap de correlaciones entre variables de engagement |
| Fig. 3.9 | 05 | Comparación de poder predictivo por era (antiguo vs. nuevo) |
