# Reporte de Modelado — Predicción de Riesgo Estudiantil (Notebook 06)

Este documento resume los hallazgos y el proceso de modelado realizado para predecir qué estudiantes podrían estar en riesgo de reprobar el curso. El objetivo fue crear un sistema de alerta temprana que sea útil para los profesores antes de que sea demasiado tarde.

## 1. ¿Cómo se hizo el estudio? (Metodología)

No se analizó el semestre como un bloque único, sino en dos momentos clave llamados **Checkpoints**:
*   **Checkpoint 1 (Semana 6):** Justo después del primer parcial. Es el momento ideal para intervenir.
*   **Checkpoint 2 (Semana 11):** Después del segundo parcial. Los modelos son más precisos aquí porque tienen más historia del estudiante.

En cada momento, probamos añadir información paso a paso (desde solo notas hasta comportamiento en la plataforma) para ver qué tanto mejoraba la predicción.

## 2. Hallazgos Principales

### ¿Qué tan bien funcionan los modelos?
*   **En el Checkpoint 1 (Temprano):** El modelo logra identificar correctamente a cerca del **78%** de los estudiantes que van a reprobar si usamos técnicas de balanceo (SMOTEENN). Es decir, detectamos a 8 de cada 10 estudiantes en riesgo muy pronto en el semestre.
*   **En el Checkpoint 2 (Intermedio):** La precisión sube muchísimo. Aquí detectamos al **85%** de los estudiantes en riesgo y con una seguridad (AUC) del **95%**, lo cual es excelente.

### ¿Qué variables son las más importantes?
1.  **Parcial 1 y 2:** Siguen siendo los predictores más fuertes. Una nota baja aquí es la señal de alerta número uno.
2.  **Engagement (Uso de la plataforma):** El tiempo dedicado y, sobre todo, la **diversidad de temas visitados** ayuda a diferenciar a los estudiantes que están "estudiando de verdad" de los que solo dan vistazos rápidos.
3.  **Actividades Continuas (Quices/AMs):** Añadir estas notas mejora la precisión del modelo, confirmando que la evaluación continua es clave.

### El Efecto de la "Era" (Cambio de Formato)
El modelo tiene en cuenta si el estudiante está en el formato antiguo (con proyecto) o nuevo (con más quices). Esto permite que las predicciones sean justas y precisas sin importar el semestre.

## 3. Glosario para no expertos
*   **Recall (Sensibilidad):** Es nuestra capacidad de no "dejar pasar" a ningún estudiante en riesgo. Un Recall alto significa que la alerta temprana está funcionando.
*   **SMOTEENN:** Una técnica matemática para "equilibrar" los datos. Como hay menos estudiantes que reprueban que los que aprueban, esta técnica ayuda al modelo a aprender mejor cómo son los estudiantes en riesgo.
*   **Lasso/Ridge:** Herramientas que ayudan al modelo a elegir automáticamente qué variables son basura y cuáles son importantes, evitando que el modelo se confunda con datos irrelevantes.

## 4. Conclusión
El sistema de alerta temprana propuesto en el **Checkpoint 1** es viable y altamente efectivo para detectar el riesgo a tiempo. Se recomienda a los docentes prestar especial atención a la combinación de la nota del **Parcial 1** con el **tiempo de estudio en la plataforma**, ya que son las señales que el modelo identificó como primordiales.
