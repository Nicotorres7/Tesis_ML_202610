# Guia de Uso — Modelos SAT (Sistema de Alerta Temprana)
## IIND-2104 Modelos Probabilisticos — Universidad de los Andes
**Generado**: 2026-04-21 19:19
**Autores**: Nicolas Torres Pulido, Isabella Delgadillo Calero
**Director**: Juan Fernando Perez Bernal

---

## 1. Descripcion general

El SAT predice el riesgo de reprobacion en dos momentos del semestre:

| Checkpoint | Momento | Ventana de intervencion |
| :--- | :--- | :--- |
| **CP1** | Semana 6 (post Parcial 1) | 11 semanas restantes |
| **CP2** | Semana 11 (post Parcial 2) | 6-7 semanas restantes |

Por cada checkpoint existen **3 modelos** seleccionados por busqueda global
entre todas las familias (LR, DT, RF, XGB) con AUC_loso >= 0.75:

| Criterio | Archivo | Uso recomendado |
| :--- | :--- | :--- |
| `recall` | `cp*_recall.joblib` | Recursos abundantes — alerta amplia (min FN) |
| `precision` | `cp*_precision.joblib` | Recursos limitados — solo casos de alta confianza (min FP) |
| `f1` | `cp*_f1.joblib` | **Uso general recomendado** — balance recall/precision |

---

## 2. Instalacion de dependencias

```bash
pip install scikit-learn xgboost imbalanced-learn joblib pandas numpy
```

---

## 3. Carga de un modelo

```python
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo deseado (ejemplo: CP1 optimizado por F1)
bundle = joblib.load('modelos_finales/cp1semana6_f1.joblib')

modelo   = bundle['modelo']    # objeto sklearn con .predict_proba()
scaler   = bundle['scaler']    # StandardScaler ajustado al dataset completo
features = bundle['features']  # lista ordenada de columnas requeridas
umbral   = bundle['umbral']    # umbral optimo de clasificacion
```

---

## 4. Funcion de prediccion

```python
def predecir_riesgo(bundle, df_nuevos):
    """
    Predice riesgo de reprobacion para nuevos estudiantes.
    Parametros: bundle = dict de joblib.load()
                df_nuevos = DataFrame con features requeridas
    Retorna: DataFrame con prob_reprobo, alerta, nivel_riesgo
    """
    modelo   = bundle['modelo']
    scaler   = bundle['scaler']
    features = bundle['features']
    umbral   = bundle['umbral']
    izeros   = bundle.get('impute_zeros', [])

    X = df_nuevos[features].copy()
    for f in izeros:
        if f in X.columns:
            X[f] = X[f].fillna(0)
    X = X.fillna(0)

    Xs    = scaler.transform(X)
    prob  = modelo.predict_proba(Xs)[:, 1]
    alerta = (prob >= umbral).astype(int)
    nivel  = pd.cut(prob, bins=[0, 0.35, 0.60, 1.0],
                    labels=['BAJO', 'MEDIO', 'ALTO'], include_lowest=True)
    return pd.DataFrame({
        'prob_reprobo': prob.round(4),
        'alerta':       alerta,
        'nivel_riesgo': nivel,
    }, index=df_nuevos.index)
```

**Ejemplo:**

```python
df_activos = pd.read_csv('datos_semestre_actual.csv')
bundle     = joblib.load('modelos_finales/cp1semana6_f1.joblib')
resultado  = predecir_riesgo(bundle, df_activos)
en_riesgo  = df_activos[resultado['nivel_riesgo'] == 'ALTO']
print(f'Estudiantes en riesgo ALTO: {len(en_riesgo)}')
```

---

## 5. Variables derivadas que deben calcularse antes de predecir

```python
# Siempre requeridas
df['era_encoded']   = (df['era'] == 'formato_nuevo').astype(int)
df['log_eng_p1']    = np.log1p(df['engagement_hasta_p1'])
df['log_eng_p2']    = np.log1p(df['engagement_hasta_p2'])

# Posicion relativa (requiere datos del semestre actual)
media = df['parcial_1'].mean(); std = df['parcial_1'].std() + 1e-6
df['p1_vs_media']   = (df['parcial_1'] - media) / std

# Solo CP2
df['delta_p2_p1']   = df['parcial_2'] - df['parcial_1']

# Derivadas de engagement (requieren visitas > 0 y temas > 0)
df['intensidad_p1'] = np.where(df['parcial_1_visitas'] > 0,
    df['engagement_hasta_p1'] * 60 / df['parcial_1_visitas'], 0.0)
df['ratio_vt_p1']   = np.where(df['parcial_1_temas_unicos'] > 0,
    df['parcial_1_visitas'] / df['parcial_1_temas_unicos'], 0.0)
```

---

## 6. Features requeridas por modelo

## CP1 — Semana 6 (post Parcial 1)

### Modelo optimizado por RECALL

- **Familia**: `XGB`
- **Feature set**: `S11_smoteenn`
- **Balanceo**: `smoteenn`
- **Umbral de clasificacion**: `0.68`
- **Metricas LOSO**: AUC=0.814+-0.072 | Recall=0.806+-0.133 | Prec=0.519 | F1=0.613+-0.057
- **Features requeridas** (7):
  - `parcial_1`
  - `p1_vs_media`
  - `log_eng_p1`
  - `parcial_1_modulos_unicos`
  - `intensidad_p1`
  - `ratio_vt_p1`
  - `era_encoded`
- **Archivo**: `modelos_finales/cp1semana6_recall.joblib`

### Modelo optimizado por PRECISION

- **Familia**: `LR`
- **Feature set**: `S05_eng_log`
- **Balanceo**: `none`
- **Umbral de clasificacion**: `0.348`
- **Metricas LOSO**: AUC=0.844+-0.068 | Recall=0.732+-0.161 | Prec=0.637 | F1=0.666+-0.054
- **Features requeridas** (3):
  - `parcial_1`
  - `log_eng_p1`
  - `era_encoded`
- **Archivo**: `modelos_finales/cp1semana6_precision.joblib`

### Modelo optimizado por F1

- **Familia**: `LR`
- **Feature set**: `S05_eng_log`
- **Balanceo**: `none`
- **Umbral de clasificacion**: `0.348`
- **Metricas LOSO**: AUC=0.844+-0.068 | Recall=0.732+-0.161 | Prec=0.637 | F1=0.666+-0.054
- **Features requeridas** (3):
  - `parcial_1`
  - `log_eng_p1`
  - `era_encoded`
- **Archivo**: `modelos_finales/cp1semana6_f1.joblib`


## CP2 — Semana 11 (post Parcial 2)

### Modelo optimizado por RECALL

- **Familia**: `DT`
- **Feature set**: `S11_smoteenn`
- **Balanceo**: `smoteenn`
- **Umbral de clasificacion**: `0.34`
- **Metricas LOSO**: AUC=0.866+-0.030 | Recall=0.932+-0.083 | Prec=0.536 | F1=0.672+-0.083
- **Features requeridas** (8):
  - `parcial_2`
  - `delta_p2_p1`
  - `p1_vs_media`
  - `log_eng_p1`
  - `parcial_1_modulos_unicos`
  - `parcial_1_visitas_por_tema`
  - `parcial_1_tiempo_por_visita`
  - `era_encoded`
- **Archivo**: `modelos_finales/cp2semana11_recall.joblib`

### Modelo optimizado por PRECISION

- **Familia**: `XGB`
- **Feature set**: `S06_eng_p1`
- **Balanceo**: `none`
- **Umbral de clasificacion**: `0.716`
- **Metricas LOSO**: AUC=0.911+-0.051 | Recall=0.705+-0.110 | Prec=0.846 | F1=0.764+-0.098
- **Features requeridas** (5):
  - `parcial_1`
  - `parcial_2`
  - `engagement_hasta_p1`
  - `parcial_1_modulos_unicos`
  - `era_encoded`
- **Archivo**: `modelos_finales/cp2semana11_precision.joblib`

### Modelo optimizado por F1

- **Familia**: `LR`
- **Feature set**: `S05_smote`
- **Balanceo**: `smote`
- **Umbral de clasificacion**: `0.544`
- **Metricas LOSO**: AUC=0.968+-0.015 | Recall=0.886+-0.025 | Prec=0.818 | F1=0.848+-0.055
- **Features requeridas** (5):
  - `parcial_2`
  - `delta_p2_p1`
  - `promedio_ams`
  - `promedio_quices`
  - `era_encoded`
- **Archivo**: `modelos_finales/cp2semana11_f1.joblib`


---

## 7. Criterio de reentrenamiento

Re-ejecutar el notebook completo cuando:
- Se dispone de datos de un nuevo semestre finalizado.
- El AUC_loso del fold mas reciente cae mas de 0.05 puntos respecto al historico.

```python
import json
meta = json.load(open('modelos_finales/cp1semana6_metadata.json'))
print('F1 LOSO actual:', meta['f1']['metricas_loso'])
```

---

## 8. Estructura de archivos exportados

```
modelos_finales/
  cp1semana6_recall.joblib      # CP1 — maximo recall
  cp1semana6_precision.joblib   # CP1 — maxima precision
  cp1semana6_f1.joblib          # CP1 — maximo F1 (recomendado)
  cp1semana6_metadata.json      # CP1 — metricas y config legibles
  cp2semana11_recall.joblib     # CP2 — maximo recall
  cp2semana11_precision.joblib  # CP2 — maxima precision
  cp2semana11_f1.joblib         # CP2 — maximo F1 (recomendado)
  cp2semana11_metadata.json     # CP2 — metricas y config legibles
```

---

## 9. Consideraciones eticas y de privacidad

- Los datos de estudiantes deben estar **anonimizados** (SHA-256) antes
  de cualquier procesamiento, segun Ley 1581 de 2012 (Habeas Data).
- Las predicciones son **herramientas de apoyo**, no decisiones automaticas.
  El docente valida cada alerta antes de intervenir (CEI-0539-26).
- Revisar metricas por subgrupo (era pedagogica, modalidad) antes de desplegar
  el modelo en un formato de curso diferente al de entrenamiento.
- Los umbrales son ajustables: un umbral menor aumenta el recall pero
  genera mas falsas alarmas. Ver `fig_umbral_*.png` para el trade-off.