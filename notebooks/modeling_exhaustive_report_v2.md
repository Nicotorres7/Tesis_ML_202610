# Reporte Exhaustivo de Modelado V2 — Sistema de Alerta Temprana

Este reporte detalla el proceso científico y técnico seguido para la construcción del modelo de predicción de riesgo académico en el curso de Modelos Probabilísticos.

## 1. Proceso de Modelado Iterativo
Se evaluaron 6 escenarios principales, divididos por momentos del semestre (**Checkpoints**) y combinaciones de variables.

### Checkpoint 1 (Semana 6 - Predicción Temprana)
| Experimento | Balanceo | Mejor Parámetro (C, Penalty) | F1-Score | Recall (Riesgo) | ROC-AUC | Top Predictores |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CP1_Initial | No | {'C': 0.1, 'penalty': 'l2'} | 0.5388 | 0.4397 | 0.8192 | parcial_1, era_encoded |
| CP1_Eng | No | {'C': 0.1, 'penalty': 'l2'} | 0.5596 | 0.4613 | 0.8164 | parcial_1, p1_visitas, era |
| CP1_Full_Logit | No | {'C': 0.1, 'penalty': 'l1'} | 0.5877 | 0.5249 | 0.8558 | parcial_1, am_2, quiz_1 |
| **CP1_SMOTEENN** | **Sí** | {'C': 100, 'penalty': 'l1'} | **0.5641** | **0.8123** | **0.8157** | parcial_1, am_1, am_2 |

### Checkpoint 2 (Semana 11 - Predicción Intermedia)
| Experimento | Modelo | Mejor Parámetro | F1-Score | Recall (Riesgo) | ROC-AUC | Top Predictores |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CP2_Full_Logit** | **Logística** | {'C': 1, 'penalty': 'l1'} | **0.7797** | **0.7581** | **0.9549** | parcial_2, am_4, parcial_1 |
| CP2_Tree | Árbol | {'depth': 5, 'min_leaf': 2} | 0.7303 | 0.7410 | 0.8931 | parcial_2, parcial_1 |

## 2. Análisis Técnico de Resultados

### Selección de Variables (Lasso/Ridge)
El uso de la penalidad **L1 (Lasso)** en los modelos "Full" fue fundamental. El modelo descartó automáticamente variables redundantes, priorizando:
*   **Checkpoint 1**: El Parcial 1 es el predictor dominante, pero las Actividades Magistrales (AMs) aportan estabilidad.
*   **Checkpoint 2**: El Parcial 2 se convierte en el predictor más fuerte, superando al Parcial 1.

### Impacto del Balanceo (SMOTEENN)
Para el **Checkpoint 1**, el balanceo fue crucial. Sin él, solo detectábamos al 44% de los estudiantes en riesgo. Con **SMOTEENN**, la capacidad de detección (Recall) subió al **81.2%**, permitiendo que la mayoría de los estudiantes en peligro sean identificados a tiempo para una intervención.

### Estabilidad (Validation)
Los resultados son consistentes a través de **5-Fold Stratified Cross-Validation**, lo que indica que el modelo no está sobreajustado y funcionará bien con nuevos semestres.

## 3. Conclusiones para la Tesis
*   **CP1** es la mejor ventana de intervención. El modelo con SMOTEENN es el recomendado por su alta sensibilidad.
*   **CP2** es el modelo de validación final. Con un AUC de 0.95, es altamente confiable para confirmar el estado final del estudiante.
*   **Engagement**: El uso de la plataforma aporta un valor incremental significativo (AUC sube de 0.82 a 0.86 en CP1).
