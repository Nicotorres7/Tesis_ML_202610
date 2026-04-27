# DOCUMENTACIÓN COMPLETA - DATASET MODELOS PROBABILÍSTICOS
## Sistema de Alerta Temprana - Universidad de los Andes

---

## 📋 Información General

**Archivo**: `dataset_completo_final.csv`  
**Dimensiones**: 764 estudiantes × 77 columnas  
**Período**: 5 semestres (202320 a 202520)  
**Curso**: IIND-2104 Modelos Probabilísticos (ahora: Modelado de Sistemas Bajo Incertidumbre)  
**Institución**: Universidad de los Andes, Facultad de Ingeniería Industrial  
**Propósito**: Predicción de rendimiento académico y sistema de alerta temprana

---

## 🎯 Contexto del Proyecto

### Problema
El curso IIND-2104 Modelos Probabilísticos presenta históricamente altas tasas de reprobación y deserción. Las intervenciones docentes actuales son **reactivas**, limitando su efectividad.

### Solución
Sistema de alerta temprana basado en machine learning que:
- Predice riesgo de reprobación/deserción
- Identifica estudiantes en riesgo de manera temprana
- Permite intervenciones proactivas en 3 momentos clave del semestre

### Datos Integrados
1. **Notas académicas**: Parciales, actividades magistrales, quices, fases de proyecto
2. **Engagement en Bloque Neón**: Tiempo, visitas, temas visitados en la plataforma

---

## 📊 Distribución de Estudiantes

### Por Semestre y Estado

| Semestre | Total | APROBÓ | REPROBÓ | RETIRADO | SIN_NOTA | Era |
|----------|-------|--------|---------|----------|----------|-----|
| 202320 | 170 | 116 | 27 | 27 | 0 | Formato Antiguo |
| 202410 | 150 | 83 | 36 | 30 | 1 | Formato Antiguo |
| 202420 | 164 | 90 | 33 | 40 | 1 | Formato Nuevo |
| 202510 | 140 | 70 | 47 | 21 | 2 | Formato Nuevo |
| 202520 | 140 | 69 | 20 | 46 | 5 | Formato Nuevo |
| **TOTAL** | **764** | **428** | **163** | **164** | **9** | - |

### Tasas de Éxito

- **Tasa de aprobación general**: 56.0% (428/764)
- **Tasa de reprobación**: 21.3% (163/764)
- **Tasa de retiro**: 21.5% (164/764)
- **Sin nota**: 1.2% (9/764)

---

## 🔄 Dos Eras Académicas

### Formato Antiguo (202320, 202410)

**Estructura de evaluación**:
- 3 Parciales (20%, 25%, 25%) = 70%
- Proyecto en 2 Fases (10% + 10%) = 20%
- Talleres 1, 2, 3 (~5% total)
- Actividades Magistrales (~5%)

**Características**:
- Proyecto grupal con entregas parciales
- Talleres grupales de práctica
- Énfasis en trabajo colaborativo

### Formato Nuevo (202420, 202510, 202520)

**Estructura de evaluación**:
- 3 Parciales (20%, 25%, 25%) = 70%
- Actividades Magistrales (15%)
- Quices (15%)

**Cambios clave**:
- Eliminación del proyecto grupal
- Más evaluaciones continuas individuales
- Mayor peso a actividades magistrales
- 202420: 6 AMs + 6 Quices (más evaluaciones)
- 202510: 6 AMs + 3 Quices
- 202520: 3 AMs + 3 Quices (consolidado)

---

## ⏰ Sistema de Checkpoints

El semestre se divide en **3 checkpoints** que representan momentos clave para intervención:

### Checkpoint 1: Hasta Primer Parcial
- **Semanas**: 1-5 aproximadamente
- **Evaluaciones incluidas**: 
  - Parcial 1 (semana 4-5)
  - AM 1, AM 2, Quiz 1 (según semestre)
- **Momento de alerta**: Semana 6
- **Ventana de intervención**: 11 semanas restantes
- **Información disponible**: Primera evaluación formal + engagement inicial

### Checkpoint 2: Hasta Segundo Parcial
- **Semanas**: 6-10 aproximadamente
- **Evaluaciones incluidas**:
  - Parcial 1 + Parcial 2 (semana 9-10)
  - AMs y Quices del checkpoint 2
- **Momento de alerta**: Semana 11
- **Ventana de intervención**: 6-7 semanas restantes
- **Información disponible**: Tendencia establecida, más datos de engagement

### Checkpoint 3: Hasta Tercer Parcial (Final)
- **Semanas**: 11-17 (fin del semestre)
- **Evaluaciones incluidas**: Todas
- **Momento de alerta**: Post-mortem / análisis final
- **Uso**: Validación de modelos, menos útil para intervención

---

## 📖 DICCIONARIO DE COLUMNAS (77 columnas)

### 🔍 Grupo 1: IDENTIFICACIÓN (3 columnas)

| # | Columna | Tipo | Descripción | Valores |
|---|---------|------|-------------|---------|
| 1 | **semestre** | int | Código del semestre académico | 202320, 202410, 202420, 202510, 202520 |
| 2 | **era** | str | Formato de evaluación del curso | `formato_antiguo`, `formato_nuevo` |
| 3 | **id_estudiante** | str | ID anónimo del estudiante (SHA256) | 64 caracteres hexadecimales |

---

### 🎯 Grupo 2: VARIABLE OBJETIVO (2 columnas)

| # | Columna | Tipo | Descripción | Valores | Uso en Modelado |
|---|---------|------|-------------|---------|-----------------|
| 4 | **estado** | str | Resultado final del estudiante | `APROBÓ`, `REPROBÓ`, `RETIRADO`, `SIN_NOTA` | Target clasificación |
| 5 | **nota_100** | float | Nota final en escala 0-5 | 0.0 - 5.0 (puede ser >5.0 por bonos) | Target regresión |

**Interpretación**:
- `APROBÓ`: nota_100 ≥ 3.0
- `REPROBÓ`: nota_100 < 3.0
- `RETIRADO`: Retiro oficial del curso
- `SIN_NOTA`: Registrado pero sin calificación

---

### 📝 Grupo 3: PARCIALES (6 columnas = 3×2)

Cada parcial tiene **2 columnas**: nota + modalidad

| # | Columna | Tipo | Rango | Peso | Unidad | Modalidad |
|---|---------|------|-------|------|--------|-----------|
| 6 | **parcial_1** | float | 0.0 - 6.31 | 20% | Escala 0-5 | Individual |
| 7 | **parcial_1_modo** | str | - | - | - | `ind` |
| 8 | **parcial_2** | float | 0.0 - 6.00 | 25% | Escala 0-5 | Individual |
| 9 | **parcial_2_modo** | str | - | - | - | `ind` |
| 10 | **parcial_3** | float | 0.0 - 5.95 | 25% | Escala 0-5 | Individual |
| 11 | **parcial_3_modo** | str | - | - | - | `ind` |

**Características**:
- Evaluaciones individuales (siempre `ind`)
- Pueden superar 5.0 por bonos del profesor
- Son las evaluaciones de mayor peso
- Disponible en todos los semestres

**Contenido temático**:
- **Parcial 1**: Introducción, exponenciales, Poisson, cadenas de Markov básicas
- **Parcial 2**: Clasificación de estados, estado estable, tiempos, absorbentes
- **Parcial 3**: Teoría de colas, redes de Jackson, MDP, SDP

---

### 📚 Grupo 4: FORMATO ANTIGUO (10 columnas = 5×2)

Solo aplica para 202320 y 202410. **NaN** en formato nuevo.

#### Fases del Proyecto (4 columnas)

| # | Columna | Tipo | Rango | Peso | Checkpoint | Modalidad |
|---|---------|------|-------|------|------------|-----------|
| 12 | **fase_1** | float | 0.0 - 5.0 | 10% | 1 | Grupal |
| 13 | **fase_1_modo** | str | - | - | - | `grup` |
| 14 | **fase_2** | float | 0.0 - 5.0 | 10% | 3 | Grupal |
| 15 | **fase_2_modo** | str | - | - | - | `grup` |

**Descripción**: Proyecto de aplicación de modelos probabilísticos a un caso real. Trabajo en equipo de 3-4 estudiantes.

#### Talleres (6 columnas)

| # | Columna | Tipo | Rango | Peso | Checkpoint | Modalidad |
|---|---------|------|-------|------|------------|-----------|
| 16 | **taller_1** | float | 0.0 - 5.0 | ~1.67% | 1 | Grupal |
| 17 | **taller_1_modo** | str | - | - | - | `grup` |
| 18 | **taller_2** | float | 0.0 - 5.0 | ~1.67% | 2 | Grupal |
| 19 | **taller_2_modo** | str | - | - | - | `grup` |
| 20 | **taller_3** | float | 0.0 - 5.0 | ~1.67% | 3 | Grupal |
| 21 | **taller_3_modo** | str | - | - | - | `grup` |

**Descripción**: Talleres de práctica grupal equivalentes a los quices del formato nuevo.

---

### 🎓 Grupo 5: ACTIVIDADES MAGISTRALES (12 columnas = 6×2)

Solo aplica para formato nuevo. **NaN** en formato antiguo.

| # | Columna | Tipo | Rango | Checkpoint | Semestre(s) | Modalidad 202420 | Modalidad 202510/202520 |
|---|---------|------|-------|------------|-------------|------------------|-------------------------|
| 22 | **am_1** | float | 0.0 - 6.5 | 1 | 202420, 202510, 202520 | `grup` | `ind` |
| 23 | **am_1_modo** | str | - | - | - | - | - |
| 24 | **am_2** | float | 0.0 - 6.5 | 1 | 202420, 202510, 202520 | `grup` | `ind` |
| 25 | **am_2_modo** | str | - | - | - | - | - |
| 26 | **am_3** | float | 0.0 - 6.5 | 1-2 | 202420, 202510, 202520 | `grup` | `ind` |
| 27 | **am_3_modo** | str | - | - | - | - | - |
| 28 | **am_4** | float | 0.0 - 6.67 | 2 | 202420, 202510 | `grup` | `ind` |
| 29 | **am_4_modo** | str | - | - | - | - | - |
| 30 | **am_5** | float | 0.0 - 6.5 | 2-3 | 202420, 202510 | `grup` | `ind` |
| 31 | **am_5_modo** | str | - | - | - | - | - |
| 32 | **am_6** | float | 0.0 - 5.5 | 3 | 202420, 202510 | `grup` | `ind` |
| 33 | **am_6_modo** | str | - | - | - | - | - |

**Descripción**: Actividades Magistrales realizadas en clase. Aplicación práctica de conceptos.

**Peso total**: 15% del curso (2.5% cada una en 202420/202510, 5% cada una en 202520)

**Cambio importante de modalidad**:
- **202420**: AMs grupales (`grup`) - trabajo colaborativo en clase
- **202510/202520**: AMs individuales (`ind`) - trabajo individual

**Número de AMs por semestre**:
- 202420: 6 AMs
- 202510: 6 AMs  
- 202520: 3 AMs (consolidación)

**Valores >5.0**: Posibles por bonificaciones del profesor.

---

### 📊 Grupo 6: QUICES (12 columnas = 6×2)

Solo aplica para formato nuevo. **NaN** en formato antiguo.

| # | Columna | Tipo | Rango | Checkpoint | Semestre(s) | Modalidad |
|---|---------|------|-------|------------|-------------|-----------|
| 34 | **quiz_1** | float | 0.0 - 5.0 | 1 | 202420, 202510, 202520 | `ind` |
| 35 | **quiz_1_modo** | str | - | - | - | - |
| 36 | **quiz_2** | float | 0.0 - 5.5 | 1-2 | 202420, 202510, 202520 | `ind` |
| 37 | **quiz_2_modo** | str | - | - | - | - |
| 38 | **quiz_3** | float | 0.0 - 5.5 | 2-3 | 202420, 202510, 202520 | `ind` |
| 39 | **quiz_3_modo** | str | - | - | - | - |
| 40 | **quiz_4** | float | 0.0 - 5.0 | 2 | 202420 | `ind` |
| 41 | **quiz_4_modo** | str | - | - | - | - |
| 42 | **quiz_5** | float | 0.0 - 5.0 | 2 | 202420 | `ind` |
| 43 | **quiz_5_modo** | str | - | - | - | - |
| 44 | **quiz_6** | float | 0.0 - 5.0 | 3 | 202420 | `ind` |
| 45 | **quiz_6_modo** | str | - | - | - | - |

**Descripción**: Evaluaciones cortas individuales sobre conceptos específicos.

**Peso total**: 15% del curso (distribuido entre los quices disponibles)

**Modalidad**: Siempre individual (`ind`)

**Número de quices por semestre**:
- 202420: 6 quices
- 202510: 3 quices
- 202520: 3 quices

---

### 📈 Grupo 7: PROMEDIOS (2 columnas)

| # | Columna | Tipo | Descripción | Rango | Cálculo |
|---|---------|------|-------------|-------|---------|
| 46 | **promedio_ams** | float | Promedio de actividades magistrales | 0.0 - 6.40 | Media de am_1 a am_6 (las disponibles) |
| 47 | **promedio_quices** | float | Promedio de quices/talleres | 0.0 - 5.33 | Media de quiz_1 a quiz_6 o taller_1 a taller_3 |

**Uso**:
- Retrocompatibilidad con análisis anteriores
- Variable derivada de evaluaciones individuales
- Útil para análisis agregado

**Nota**: En formato antiguo, `promedio_quices` = promedio de talleres.

---

### 🎮 Grupo 8: ENGAGEMENT POR PARCIAL (27 columnas = 9×3)

Métricas de interacción en la plataforma Bloque Neón **por cada parcial**.

#### Features por Parcial (9 features × 3 parciales = 27 columnas)

**Para parcial_1, parcial_2, parcial_3**:

| Sufijo | Tipo | Descripción | Unidad | Rango Típico |
|--------|------|-------------|--------|--------------|
| **_visitas** | int | Número de visitas al contenido del parcial | visitas | 0 - 454 |
| **_tiempo_hrs** | float | Horas dedicadas al contenido del parcial | horas | 0 - 120 |
| **_temas_unicos** | int | Temas diferentes visitados | temas | 0 - 29 |
| **_modulos_unicos** | int | Módulos diferentes visitados | módulos | 0 - 12 |
| **_visitas_por_tema** | float | Promedio de visitas por tema | visitas/tema | 0 - 20 |
| **_tiempo_por_visita** | float | Minutos promedio por sesión | minutos | 0 - 60 |

**Columnas completas** (48-74):
```
48. parcial_1_visitas
49. parcial_1_tiempo_hrs
50. parcial_1_temas_unicos
51. parcial_1_modulos_unicos
52. parcial_1_visitas_por_tema
53. parcial_1_tiempo_por_visita

54. parcial_2_visitas
55. parcial_2_tiempo_hrs
56. parcial_2_temas_unicos
57. parcial_2_modulos_unicos
58. parcial_2_visitas_por_tema
59. parcial_2_tiempo_por_visita

60. parcial_3_visitas
61. parcial_3_tiempo_hrs
62. parcial_3_temas_unicos
63. parcial_3_modulos_unicos
64. parcial_3_visitas_por_tema
65. parcial_3_tiempo_por_visita
```

**Qué contenido incluye "parcial_X"**:
- Notas de aula relacionadas (ej: Aula 1-5 para P1)
- Bases de problemas del parcial X
- Presentaciones teóricas
- Material de estudio específico

**Interpretación**:
- `visitas_por_tema > 5`: Estudiante revisita contenido, buen signo de repaso
- `visitas_por_tema < 2`: Estudio superficial, solo un vistazo
- `tiempo_por_visita > 20 min`: Estudio profundo y concentrado
- `tiempo_por_visita < 5 min`: Vistazos rápidos, posible cramming

---

### 📚 Grupo 9: ENGAGEMENT EN ACTIVIDADES (4 columnas)

Métricas agregadas de interacción en actividades magistrales en Bloque Neón.

| # | Columna | Tipo | Descripción | Unidad | Interpretación |
|---|---------|------|-------------|--------|----------------|
| 66 | **actividades_n_realizadas** | int | Cantidad de actividades con interacción | actividades | 0-3 (cuántas hizo) |
| 67 | **actividades_tiempo_promedio** | float | Horas promedio por actividad participada | horas | Solo cuenta las que hizo |
| 68 | **actividades_tiempo_total** | float | Horas totales en todas las actividades | horas | Suma de act 1+2+3 |
| 69 | **actividades_visitas_total** | int | Visitas totales a todas las actividades | visitas | Suma de visitas |

**Nota importante**: Estas métricas son de **engagement en la plataforma** (preparación, consulta de enunciados), NO reflejan el trabajo en clase de las AMs/quices, ya que estas se hacen presencialmente.

**Por qué están agregadas**: Las actividades magistrales y quices se realizan en clase, no en Bloque Neón. El engagement captura solo la preparación previa o consulta de materiales.

---

### 📊 Grupo 10: ENGAGEMENT ACUMULADO (3 columnas)

Permite ver la **evolución temporal** del engagement durante el semestre.

| # | Columna | Tipo | Descripción | Unidad | Cálculo |
|---|---------|------|-------------|--------|---------|
| 70 | **engagement_hasta_p1** | float | Horas acumuladas hasta checkpoint 1 | horas | = parcial_1_tiempo_hrs |
| 71 | **engagement_hasta_p2** | float | Horas acumuladas hasta checkpoint 2 | horas | = parcial_1 + parcial_2 |
| 72 | **engagement_hasta_p3** | float | Horas acumuladas hasta checkpoint 3 | horas | = parcial_1 + parcial_2 + parcial_3 |

**Para qué sirven**:
- Identificar tendencias de engagement a lo largo del semestre
- Detectar abandono progresivo
- Comparar ritmo de estudio entre estudiantes

**Ejemplos de patrones**:

```
Estudiante Consistente:
  engagement_hasta_p1: 10 hrs
  engagement_hasta_p2: 20 hrs (+10)
  engagement_hasta_p3: 30 hrs (+10)
→ Mantiene ritmo constante ✅

Estudiante en Declive:
  engagement_hasta_p1: 12 hrs
  engagement_hasta_p2: 17 hrs (+5)
  engagement_hasta_p3: 18 hrs (+1)
→ Perdiendo interés progresivamente ⚠️

Estudiante Recuperándose:
  engagement_hasta_p1: 5 hrs
  engagement_hasta_p2: 12 hrs (+7)
  engagement_hasta_p3: 25 hrs (+13)
→ Incrementó esfuerzo después de mal inicio ✅
```

---

### 📊 Grupo 11: ENGAGEMENT TOTAL DEL SEMESTRE (5 columnas)

Métricas agregadas de toda la interacción en la plataforma durante el semestre.

| # | Columna | Tipo | Descripción | Unidad | Rango | Interpretación |
|---|---------|------|-------------|--------|-------|----------------|
| 73 | **total_tiempo_hrs** | float | Horas totales en toda la plataforma | horas | 0 - 748 | Engagement global |
| 74 | **total_visitas** | int | Visitas totales a toda la plataforma | visitas | 1 - 1412 | Frecuencia de uso |
| 75 | **total_temas** | int | Temas únicos visitados en el semestre | temas | 1 - 213 | Amplitud de estudio |
| 76 | **total_modulos** | int | Módulos únicos visitados en el semestre | módulos | 1 - 70 | Diversidad de contenido |
| 77 | **promedio_tiempo_por_visita** | float | Minutos promedio por visita (todo el semestre) | minutos | 0 - 88 | Profundidad de estudio |

**Relación con otras métricas**:
```
total_tiempo_hrs ≈ parcial_1_tiempo_hrs + parcial_2_tiempo_hrs + parcial_3_tiempo_hrs 
                   + actividades_tiempo_total 
                   + tiempo en contenido ADMIN/AMBIGUO/PROYECTO
```

**Interpretación**:
- `total_tiempo_hrs > 50`: Estudiante muy comprometido
- `total_tiempo_hrs < 10`: Bajo engagement, señal de riesgo
- `promedio_tiempo_por_visita > 30 min`: Sesiones de estudio profundo
- `promedio_tiempo_por_visita < 10 min`: Consultas rápidas

---

## 📊 Calendario de Evaluaciones por Semestre

### 202520 (Formato Nuevo - 3 AMs + 3 Quices)

| Semana | Checkpoint | Evaluación | Modalidad | Peso |
|--------|------------|------------|-----------|------|
| 3 | 1 | Actividad Magistral 1 | Individual | 5% |
| 4 | 1 | **Parcial 1** | Individual | 20% |
| 4 | 1 | Quiz 1 | Individual | 5% |
| 8 | 2 | Actividad Magistral 2 | Individual | 5% |
| 9 | 2 | **Parcial 2** | Individual | 25% |
| 9 | 2 | Quiz 2 | Individual | 5% |
| 15 | 3 | Actividad Magistral 3 | Individual | 5% |
| 17 | 3 | **Parcial 3** | Individual | 25% |
| 17 | 3 | Quiz 3 | Individual | 5% |

**Total**: 100%

### 202510 (Formato Nuevo - 6 AMs + 3 Quices)

| Semana | Checkpoint | Evaluación | Modalidad | Peso |
|--------|------------|------------|-----------|------|
| 3 | 1 | AM1 | Individual | 2.5% |
| 5 | 1 | **Parcial 1** | Individual | 20% |
| 5 | 1 | AM2 | Individual | 2.5% |
| 7 | 1 | Quiz 1 | Individual | 5% |
| 7 | 1 | AM3 | Individual | 2.5% |
| 10 | 2 | **Parcial 2** | Individual | 25% |
| 10 | 2 | AM4 | Individual | 2.5% |
| 11 | 2 | Quiz 2 | Individual | 5% |
| 13 | 2 | AM5 | Individual | 2.5% |
| 15 | 3 | Quiz 3 | Individual | 5% |
| 15 | 3 | AM6 | Individual | 2.5% |
| 17 | 3 | **Parcial 3** | Individual | 25% |

**Total**: 100%

### 202420 (Formato Nuevo - 6 AMs + 6 Quices)

| Semana | Checkpoint | Evaluación | Modalidad | Peso |
|--------|------------|------------|-----------|------|
| 3 | 1 | AM 1 | Grupal | 2.5% |
| 5 | 1 | **Parcial 1** | Individual | 20% |
| 5 | 1 | AM 2 | Grupal | 2.5% |
| 6 | 1 | Quiz 1 | Individual | 2.5% |
| 7 | 1 | AM 3 | Grupal | 2.5% |
| 7 | 1 | Quiz 2 | Individual | 2.5% |
| 8 | 2 | Quiz 3 | Individual | 2.5% |
| 9 | 2 | Quiz 4 | Individual | 2.5% |
| 10 | 2 | **Parcial 2** | Individual | 25% |
| 10 | 2 | AM 4 | Grupal | 2.5% |
| 11 | 2 | Quiz 5 | Individual | 2.5% |
| 13 | 3 | AM 5 | Grupal | 2.5% |
| 15 | 3 | AM 6 | Grupal | 2.5% |
| 15 | 3 | Quiz 6 | Individual | 2.5% |
| 17 | 3 | **Parcial 3** | Individual | 25% |

**Total**: 100%

### 202410 y 202320 (Formato Antiguo)

| Semana | Checkpoint | Evaluación | Modalidad | Peso |
|--------|------------|------------|-----------|------|
| 3 | 1 | Taller 1 | Grupal | 1.67% |
| 5 | 1 | **Parcial 1** | Individual | 20% |
| 7 | 1 | Fase 1 (Proyecto) | Grupal | 10% |
| 8 | 2 | Actividades Magistrales | Grupal | 5% |
| 10 | 2 | Taller 2 | Grupal | 1.67% |
| 12 | 2 | **Parcial 2** | Individual | 25% |
| 14 | 3 | Taller 3 | Grupal | 1.67% |
| 16 | 3 | Fase 2 (Proyecto) | Grupal | 10% |
| 17 | 3 | **Parcial 3** | Individual | 25% |

**Total**: 100%

---

## 🎯 Guía de Uso para Modelado

### Modelos de Predicción por Checkpoint

#### Modelo 1: Predicción Temprana (Post-Parcial 1)

**Momento**: Semana 6  
**Features disponibles**:
```python
features_checkpoint_1 = [
    'parcial_1',
    'am_1', 'am_2',  # Según semestre
    'quiz_1',        # Según semestre
    'parcial_1_tiempo_hrs',
    'parcial_1_visitas',
    'parcial_1_visitas_por_tema',
    'engagement_hasta_p1',
    'actividades_n_realizadas'
]
```

**Target**: `estado` (APROBÓ/REPROBÓ/RETIRADO)

**Ventaja**: 11 semanas para intervención

#### Modelo 2: Predicción Intermedia (Post-Parcial 2)

**Momento**: Semana 11  
**Features disponibles**: Checkpoint 1 + Checkpoint 2  
**Ventana de intervención**: 6-7 semanas

#### Modelo 3: Predicción Final (Validación)

**Momento**: Post-semestre  
**Uso**: Validación de modelos, análisis retrospectivo

### Estrategias de Feature Engineering

**1. Tendencias**:
```python
df['tendencia_engagement'] = df['parcial_2_tiempo_hrs'] - df['parcial_1_tiempo_hrs']
df['mejora_notas'] = df['parcial_2'] - df['parcial_1']
```

**2. Consistencia en AMs/Quices**:
```python
df['std_ams'] = df[['am_1', 'am_2', 'am_3']].std(axis=1)
df['consistencia_alta'] = df['std_ams'] < 0.5
```

**3. Señales tempranas**:
```python
df['bajo_engagement_p1'] = df['parcial_1_tiempo_hrs'] < 5
df['bajo_rendimiento_p1'] = df['parcial_1'] < 3.0
df['riesgo_temprano'] = df['bajo_engagement_p1'] & df['bajo_rendimiento_p1']
```

**4. Modalidad**:
```python
# Analizar si trabajo grupal vs individual afecta rendimiento
df_202420['rendimiento_grupal'] = df_202420[['am_1', 'am_2', 'am_3']].mean(axis=1)
df_202420['rendimiento_individual'] = df_202420[['quiz_1', 'quiz_2', 'quiz_3']].mean(axis=1)
df_202420['mejor_individual'] = df_202420['rendimiento_individual'] > df_202420['rendimiento_grupal']
```

### Manejo de Valores Nulos

**Nulls esperados** (no son errores):
```python
# am_4, am_5, am_6: Solo en 202420/202510
df['am_4'].fillna(0)  # Si el semestre no las tiene

# fase_1, fase_2: Solo en formato antiguo
df['fase_1'].fillna(0)  # Si es formato nuevo

# Estudiantes retirados: engagement = 0
df.loc[df['estado'] == 'RETIRADO', 'total_tiempo_hrs'].fillna(0)
```

**Nulls problemáticos** (revisar):
```python
# Estudiantes activos sin nota de parcial
df[(df['estado'] != 'RETIRADO') & (df['parcial_1'].isna())]
```

---

## ⚠️ Advertencias y Consideraciones

### 1. Cambios en Modalidad

**202420** tiene AMs **grupales**, mientras que **202510/202520** son **individuales**.

**Implicación**: No comparar directamente `am_1` entre 202420 y 202510.

**Solución**: 
- Modelar por separado 202420
- O crear feature `es_grupal` como variable de control

### 2. Número Variable de Evaluaciones

| Semestre | AMs | Quices | Estrategia |
|----------|-----|--------|------------|
| 202420 | 6 | 6 | Usar solo am_1-3 y quiz_1-3 para comparabilidad |
| 202510 | 6 | 3 | Usar solo am_1-3 para comparabilidad |
| 202520 | 3 | 3 | Usar am_1-3 completo |

### 3. Engagement vs Aprendizaje Real

El engagement mide **interacción con plataforma**, no aprendizaje real:
- Un estudiante puede pasar 10 hrs sin estudiar efectivamente
- Otro puede estudiar offline y tener pocas visitas

**Sin embargo**: En promedio, más engagement correlaciona con mejor rendimiento.

### 4. Retirados con Engagement = 0

143 estudiantes se retiraron sin generar engagement.

**Implicación**: El modelo debe manejar `engagement = 0` como señal de riesgo extremo, no como dato faltante.

### 5. Valores > 5.0

Notas pueden superar 5.0 por bonos del profesor (especialmente en AMs).

**No eliminar** como outliers.

---

## 📈 Estadísticas Descriptivas

### Notas Académicas

| Variable | Media | Mediana | Min | Max | Std |
|----------|-------|---------|-----|-----|-----|
| nota_100 | 2.87 | 3.01 | 0.00 | 5.26 | 1.47 |
| parcial_1 | 3.45 | 3.58 | 0.00 | 6.31 | 1.52 |
| parcial_2 | 3.45 | 3.60 | 0.00 | 6.00 | 1.81 |
| parcial_3 | 2.92 | 3.00 | 0.00 | 5.95 | 1.73 |
| promedio_ams | 3.82 | 4.10 | 0.00 | 6.40 | 1.55 |
| promedio_quices | 3.91 | 4.15 | 0.00 | 5.33 | 1.48 |

### Engagement por Parcial (tiempo en horas)

| Parcial | Media | Mediana | P25 | P75 | Max |
|---------|-------|---------|-----|-----|-----|
| P1 | 10.9 | 7.3 | 3.0 | 14.6 | 120 |
| P2 | 9.9 | 6.1 | 2.3 | 12.6 | 115 |
| P3 | 10.0 | 4.8 | 0.8 | 12.6 | 98 |

**Observación**: La mediana de P3 (4.8 hrs) es mucho menor que P1/P2, indicando abandono hacia el final.

### Actividades Magistrales por Semestre

**202420** (grupales):
- am_1: Media 3.8, Mediana 4.0
- am_2: Media 3.9, Mediana 4.1  
- am_3: Media 4.0, Mediana 4.3

**202510** (individuales):
- am_1: Media 4.2, Mediana 4.5
- am_2: Media 3.9, Mediana 4.2
- am_3: Media 3.8, Mediana 4.0

---

## 🔍 Ejemplo Completo de Estudiante

**Semestre**: 202420  
**ID**: 00b3a84bea6a2e92337a5355fe21f474...  
**Estado**: APROBÓ  
**Nota final**: 3.28

### Evaluaciones

| Evaluación | Nota | Modalidad | Checkpoint |
|------------|------|-----------|------------|
| Parcial 1 | 3.45 | ind | 1 |
| Parcial 2 | 2.13 | ind | 2 |
| Parcial 3 | 3.40 | ind | 3 |
| AM 1 | 3.30 | grup | 1 |
| AM 2 | 3.74 | grup | 1 |
| AM 3 | 4.70 | grup | 1 |
| AM 4 | 3.40 | grup | 2 |
| AM 5 | 5.20 | grup | 3 |
| AM 6 | 3.50 | grup | 3 |
| Quiz 1 | 3.00 | ind | 1 |
| Quiz 2 | 2.00 | ind | 1 |
| Quiz 3 | 5.00 | ind | 2 |
| Quiz 4 | 5.00 | ind | 2 |
| Quiz 5 | 4.00 | ind | 2 |
| Quiz 6 | 3.00 | ind | 3 |

### Promedios
- **AMs**: 3.97
- **Quices**: 4.11

### Engagement
- **Total tiempo**: 25.3 hrs
- **Engagement hasta P1**: 4.2 hrs
- **Engagement hasta P2**: 5.6 hrs  
- **Engagement hasta P3**: 7.7 hrs

### Interpretación
Estudiante con:
- ✅ Engagement moderado (25 hrs)
- ⚠️ Baja inicial en P2 (2.13) pero recuperó en P3 (3.40)
- ✅ Buen desempeño en trabajo grupal (AMs promedio 3.97)
- ✅ Excelente desempeño en quices (promedio 4.11)
- ✅ Resultado final: APROBÓ con 3.28

---

## 📚 Referencias

**Fuentes de datos**:
- Notas: Registros académicos Universidad de los Andes (anonimizados)
- Engagement: Plataforma Bloque Neón (Learning Management System)

**Normatividad**:
- Ley 1581 de 2012 (Habeas Data) - Anonimización de datos
- IEEE Standard 7000-2021 - Buenas prácticas en ciencia de datos

**Proyecto**:
- Curso: IIND-2104 Modelos Probabilísticos
- Estudiantes: Nicolás Torres Pulido, Isabella Delgadillo Calero
- Director: Juan Fernando Pérez Bernal
- Facultad de Ingeniería Industrial, Universidad de los Andes

---

## 📝 Historial de Versiones

**v3.0 (Febrero 2026)** - Dataset Final Expandido
- ✅ Notas individuales por AM y Quiz
- ✅ Modalidad (individual/grupal) para cada evaluación
- ✅ Eliminación de columnas de fechas innecesarias
- ✅ 764 estudiantes × 77 columnas
- ✅ Integración completa de 5 semestres

**v2.0 (Febrero 2026)** - Dataset Completo
- Integración de notas + engagement
- 743 estudiantes × 51 columnas
- Promedios de AMs y Quices

**v1.0 (Febrero 2026)** - Features de Engagement
- 549 estudiantes con engagement
- 43 features de interacción en Bloque Neón

---

**Última actualización**: Febrero 2026  
**Mantenedor**: Nicolás Torres Pulido  
**Contacto**: Proyecto de Grado - Universidad de los Andes
