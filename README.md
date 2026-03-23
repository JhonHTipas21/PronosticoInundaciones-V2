<p align="center">
  <h1 align="center">🌊 Sistema de Pronóstico de Inundaciones — Canales Pluviales del Sur de Cali</h1>
  <p align="center">
    <em>Plataforma de alerta temprana con Machine Learning para la predicción de caudales en canales urbanos</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
    <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" />
    <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit" />
    <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn" />
    <img src="https://img.shields.io/badge/Plotly-Visualization-3F4F75?logo=plotly" />
    <img src="https://img.shields.io/badge/License-MIT-green" />
  </p>
</p>

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Guía de Instalación y Ejecución](#-guía-de-instalación-y-ejecución)
- [Uso del Dashboard](#-uso-del-dashboard)
- [API REST — Endpoints](#-api-rest--endpoints)
- [Modelo Predictivo](#-modelo-predictivo)
- [Ingesta de Datos — API SODA](#-ingesta-de-datos--api-soda)
- [Despliegue en Producción](#-despliegue-en-producción)
- [Autores](#-autores)

---

## 📖 Descripción del Proyecto

Este sistema es un **prototipo de alerta temprana** para la gestión preventiva de inundaciones en los canales pluviales del sur de Santiago de Cali, Colombia. Utiliza modelos de aprendizaje automático (Ridge, Lasso, ElasticNet) entrenados con datos históricos de 20 años de la **Corporación Autónoma Regional del Valle del Cauca (CVC)** para predecir el caudal de los canales urbanos en horizontes de **3 a 6 horas**.

### Canales Monitoreados (Estaciones Objetivo)

| # | Estación | Capacidad Máxima (m³/s) |
|---|------------------------|--------------------------|
| 1 | Canal Cañaveralejo     | 116.3                    |
| 2 | Canal Ciudad Jardín    | 45.0                     |
| 3 | Canal Interceptor Sur  | 254.4                    |
| 4 | Río Cañaveralejo       | 152.0                    |
| 5 | Río Meléndez           | 98.5                     |
| 6 | Río Lilí               | 63.2                     |

### Características Principales

- 🔮 **Predicción recursiva a 48h** en intervalos de 3 horas con envolvente de incertidumbre IC 95%
- 📡 **Ingesta continua** desde la API SODA de datos.gov.co (ETL incremental)
- 🕐 **Anclaje en tiempo real** — el dashboard siempre muestra la hora actual con `datetime.now()`
- 📊 **Validación científica** con Scatter Plots por estación (Matplotlib/Seaborn), R² Global y R² en Crecientes
- 🚨 **Límites POMCA** — capacidad máxima de cada canal según el Plan de Ordenación y Manejo de Cuencas
- 🔄 **Comparación de modelos** Ridge vs Lasso con GridSearchCV y validación cruzada temporal

---

## 🏗 Arquitectura del Sistema

```
┌───────────────────────────────────────────────────────────────┐
│                    FRONTEND (Streamlit)                        │
│  Dashboard · Predicción CSV · Validación · Comparación · SODA │
│                    Puerto: 8501                                │
└───────────────────────┬───────────────────────────────────────┘
                        │ HTTP (requests)
                        ▼
┌───────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                           │
│  /api/v1/predict · /forecast-48h · /retrain · /stations       │
│                    Puerto: 8000                                │
└───────────────────────┬───────────────────────────────────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
   ┌──────────────────┐  ┌─────────────────┐
   │  Modelo ML (.pkl) │  │  Dataset (.csv)  │
   │  Ridge/Lasso      │  │  20 años CVC     │
   │  + RobustScaler   │  │  + Ruido físico  │
   │  + PolyFeatures   │  │  calibrado       │
   └──────────────────┘  └─────────────────┘
                                  ▲
                                  │ ETL Incremental
                          ┌───────┴───────┐
                          │   API SODA     │
                          │  datos.gov.co  │
                          └───────────────┘
```

---

## 📁 Estructura del Proyecto

```
agente_caudales/
├── main.py                          # Punto de entrada del backend (uvicorn)
├── streamlit_app.py                 # Frontend interactivo completo
├── requirements.txt                 # Dependencias del proyecto
├── DEPLOYMENT.md                    # Guía de despliegue en producción
├── .env.example                     # Variables de entorno (plantilla)
├── .gitignore
│
├── app/
│   ├── main.py                      # Configuración FastAPI + CORS + rutas
│   ├── config.py                    # Settings centralizados (Pydantic)
│   │
│   ├── data/
│   │   ├── dataset_historico_calibrado.csv  # Dataset principal (20 años, ruido físico)
│   │   ├── dataset_6h.csv                   # Dataset con features a horizonte 6h
│   │   └── dataset_soda.csv                 # Datos descargados de API SODA
│   │
│   ├── model/
│   │   ├── modelo_regresion.pkl     # Modelo entrenado (serializado)
│   │   ├── meta.json                # Metadata: features, σ residuales, best params
│   │   ├── ridge/                   # Modelos Ridge por estación
│   │   ├── lasso/                   # Modelos Lasso por estación
│   │   └── per_estacion/            # Holdout data para Scatter Plots
│   │
│   ├── routes/
│   │   ├── predict_routes.py        # Endpoints: /predict, /forecast-48h, /stations
│   │   ├── train_routes.py          # Endpoint: /retrain
│   │   └── health_routes.py         # Endpoint: /health
│   │
│   ├── schemas/
│   │   └── predict_schema.py        # Modelos Pydantic (request/response)
│   │
│   └── services/
│       ├── feature_service.py       # Ingeniería de features (lags, CN, API, ventanas)
│       ├── predict_service.py       # Predicción con clipping dual + autoregresión
│       ├── train_service.py         # Pipeline ML: RobustScaler → Poly → GridSearchCV
│       ├── geo_service.py           # Datos geográficos y Q_max por estación
│       └── soda_api_service.py      # ETL incremental desde API SODA (datos.gov.co)
```

---

## 🛠 Tecnologías Utilizadas

| Componente | Tecnología |
|---|---|
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Machine Learning** | scikit-learn (Ridge, Lasso, ElasticNet) |
| **Pipeline ML** | RobustScaler → PolynomialFeatures → TransformedTargetRegressor (log1p/expm1) |
| **Optimización** | GridSearchCV + TimeSeriesSplit (α = 0.001 → 10.0) |
| **Visualización** | Plotly (hidrogramas) + Matplotlib/Seaborn (scatter plots científicos) |
| **Datos Abiertos** | API SODA — datos.gov.co (Socrata) |
| **Lenguaje** | Python 3.10+ |

---

## 🚀 Guía de Instalación y Ejecución

### Requisitos Previos

- **Python 3.10 o superior** instalado
- **pip** (gestor de paquetes de Python)
- **Git** instalado
- Conexión a internet (para la API SODA, opcional)

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/JhonHTipas21/PronosticoInundaciones-V2.git
cd PronosticoInundaciones-V2
```

### Paso 2: Crear y Activar el Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar (macOS/Linux)
source venv/bin/activate

# Activar (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activar (Windows CMD)
venv\Scripts\activate.bat
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Configurar Variables de Entorno

Crear un archivo `.env` en la raíz del proyecto:

```bash
cp .env.example .env
```

O crear manualmente con el siguiente contenido:

```env
APP_NAME=agente_caudales
DATA_PATH=app/data/dataset_6h.csv
MODEL_DIR=app/model
HORIZON=1
HORIZON_UNITS=6H
API_BASE_URL=http://localhost:8000
STREAMLIT_URL=http://localhost:8501
```

### Paso 5: Iniciar el Backend (FastAPI)

**Abrir una terminal** y ejecutar:

```bash
python main.py
```

O directamente con uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Verás en consola:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

> ✅ El backend estará disponible en `http://localhost:8000`
> 📄 Documentación interactiva en `http://localhost:8000/docs` (Swagger UI)

### Paso 6: Iniciar el Frontend (Streamlit)

**Abrir otra terminal** (mantener el backend corriendo) y ejecutar:

```bash
streamlit run streamlit_app.py
```

Verás en consola:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

> ✅ El dashboard estará disponible en `http://localhost:8501`

### Resumen de Ejecución

| Terminal | Comando | Puerto | Descripción |
|---|---|---|---|
| **Terminal 1** | `python main.py` | `:8000` | Backend FastAPI (API REST) |
| **Terminal 2** | `streamlit run streamlit_app.py` | `:8501` | Frontend Dashboard interactivo |

> ⚠️ **Importante**: El backend **DEBE** estar corriendo antes de abrir el frontend. El dashboard se conecta al backend en `http://localhost:8000/api/v1`.

---

## 📊 Uso del Dashboard

### Vistas Disponibles

| Vista | Descripción |
|---|---|
| **Dashboard** | Hidrograma principal anclado en tiempo real con predicción a 48h y curva de decaimiento |
| **Predicción CSV** | Subir un archivo CSV personalizado y obtener predicciones |
| **Validación Estadística** | Scatter Plot científico (Seaborn) con R² Global, R² Crecientes, RMSE, MAE por estación |
| **Comparación Modelos** | Benchmark Ridge vs Lasso con GridSearchCV y validación cruzada temporal |
| **Datos SODA API** | Descarga incremental de datos oficiales de la CVC desde datos.gov.co |

### Panel Lateral (Sidebar)

- **Estación**: Selecciona cualquiera de las 6 estaciones del sur de Cali
- **Horizonte**: Configura el horizonte de predicción (3h o 6h)
- **Estado del Backend**: Indicador visual de conexión al servidor

---

## 🔌 API REST — Endpoints

Base URL: `http://localhost:8000/api/v1`

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servidor |
| `GET` | `/stations` | Lista de estaciones con Q_max |
| `GET` | `/metrics` | Métricas del modelo actual |
| `POST` | `/predict` | Predicción desde CSV |
| `POST` | `/forecast-48h` | Pronóstico recursivo a 48h (3h/paso) |
| `POST` | `/retrain` | Reentrenamiento con nuevos datos |

### Ejemplo: Pronóstico a 48h

```bash
curl -X POST http://localhost:8000/api/v1/forecast-48h \
  -H "Content-Type: application/json" \
  -d '{"estacion": "Canal Cañaveralejo", "lluvia_mm": 5.0, "steps": 16}'
```

---

## 🧠 Modelo Predictivo

### Pipeline de Machine Learning

```
Datos Crudos (CSV / API SODA)
    ↓ limpiar_dataframe()          # Limpieza IQR + interpolación temporal
    ↓ build_features()             # Ingeniería: lags, CN, API, ventanas móviles
    ↓ ColumnTransformer
    │   ├── RobustScaler (numéricos lineales)
    │   ├── RobustScaler + PolynomialFeatures(degree=2) (CN, API, lluvia)
    │   └── OneHotEncoder (estaciones)
    ↓ TransformedTargetRegressor(log1p / expm1)
    ↓ GridSearchCV(TimeSeriesSplit, cv=3)
    │   └── Alpha: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    ↓ Ridge / Lasso / ElasticNet
    ↓ Clipping dual (estadístico q99 + físico Q_max canal)
    ↓ Predicción final (m³/s)
```

### Correcciones Físicas Implementadas

1. **No más ceros artificiales**: `ffill().bfill()` en lugar de `fillna(0)` — un canal nunca se seca instantáneamente
2. **Autoregresión con seed**: cada predicción se retroalimenta como nuevo registro base
3. **Curva de vaciado exponencial** (τ = 12h): decaimiento natural cuando no hay lluvia, en lugar de colapso a cero
4. **Punto de costura visual**: la línea de predicción nace exactamente del último valor real

---

## 📡 Ingesta de Datos — API SODA

El sistema se conecta a la API pública de la CVC a través del portal de datos abiertos de Colombia:

- **Fuente**: `https://www.datos.gov.co/resource/avya-p282.json`
- **Protocolo**: API SODA (Socrata Open Data API)
- **Modo**: ETL incremental — solo descarga datos nuevos posteriores a la última fecha en el dataset local
- **Paginación**: Bloques de 5000 registros con `$offset` y `$limit`

---

## 🚢 Despliegue en Producción

> ⚠️ **No recomendado**: Vercel (no soporta Streamlit ni modelos ML grandes)

### Plataformas Recomendadas

| Plataforma | Ventajas |
|---|---|
| **Render.com** | Soporta Python, workers largos, volúmenes persistentes |
| **Streamlit Community Cloud** | Despliegue directo desde GitHub, gratis para proyectos públicos |
| **Railway** | Docker nativo, escalado automático |

Consulte `DEPLOYMENT.md` para instrucciones detalladas.

---

## 👥 Autores

- **Jhon Harvey Tipas** — Ingeniería de Software, Universidad del Valle

---

## 📜 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.

---

<p align="center">
  <em>Desarrollado como trabajo de grado para la gestión preventiva de inundaciones urbanas en Santiago de Cali, Colombia 🇨🇴</em>
</p>
