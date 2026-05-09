# SAT Uniandes - Sistema de Alerta Temprana

Este repositorio contiene la aplicación **SAT Uniandes**, una herramienta diseñada para el modelado académico y la analítica de riesgo estudiantil. La aplicación permite gestionar proyectos, entrenar modelos de aprendizaje automático y generar predicciones de riesgo de reprobación.

## Estructura del Proyecto

- `app.py`: Punto de entrada de la aplicación Streamlit.
- `sat_app/`: Paquete principal que contiene la lógica de la aplicación.
  - `ui.py`: Definición de la interfaz de usuario y navegación.
  - `training.py`: Lógica de entrenamiento de modelos y optimización de hiperparámetros.
  - `inference.py`: Motores de predicción y validación de datos.
  - `charts.py`: Visualizaciones interactivas utilizando Plotly.
  - `data.py`: Procesamiento y normalización de datasets.
  - `registry.py`: Gestión de persistencia para proyectos, checkpoints y modelos.
  - `exporters.py`: Generación de reportes en formatos Excel y PDF.
  - `config.py`: Configuraciones globales, temas visuales y etiquetas.
- `models/`: Directorio para almacenar artefactos de modelos (joblib) y metadatos.
- `sat_runtime/`: Directorio de ejecución donde se almacenan los datos de proyectos y predicciones generadas.

## Requisitos Previos

Asegúrate de tener instalado Python 3.9 o superior. Se recomienda el uso de un entorno virtual.

## Instalación

1. **Clonar el repositorio** (o descargar los archivos).
2. **Crear un entorno virtual** (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. **Instalar las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

Para iniciar la aplicación, ejecuta el siguiente comando en la raíz del proyecto:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador predeterminado (usualmente en `http://localhost:8501`).

## Funcionalidades Principales

1. **Gestión de Proyectos**: Permite organizar diferentes cursos o periodos académicos.
2. **Entrenamiento**: Flujo guiado para cargar datos históricos, seleccionar variables y entrenar modelos (LR, DT, RF, XGBoost).
3. **Dashboard de Riesgo**: Visualización detallada de la probabilidad de riesgo a nivel de cohorte e individual.
4. **Comparación de Modelos**: Herramientas para evaluar métricas (AUC, F1, Recall) y activar la mejor versión.
5. **Exportación**: Descarga de resultados en formatos procesables (Excel) o listos para presentación (PDF).

---
*Desarrollado para la Tesis de ML - 2026*
