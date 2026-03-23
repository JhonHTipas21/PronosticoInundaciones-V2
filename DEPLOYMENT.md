# Guía de Despliegue (Production)

Este proyecto consta de dos partes principales:
1. **Backend (FastAPI)**: Procesamiento de datos (`app/main.py`)
2. **Frontend (Streamlit)**: Dashboard visual interactivo (`streamlit_app.py`)

## Advertencia sobre Vercel
Vercel está diseñado principalmente para aplicaciones sin servidor (Serverless) como Next.js. El límite de tamaño del paquete (50MB) y el límite de ejecución (10s) en el plan gratuito hacen que sea **difícil o imposible** desplegar modelos de machine learning con dependencias pesadas como `scikit-learn`, `pandas` y `scipy`.

Además, **Streamlit no es compatible con el entorno Serverless de Vercel**, ya que requiere estar ejecutándose de fondo en un proceso continuo (Websockets).

---

## Recomendaciones de Despliegue Profesionales

### Opción A: Render.com (Recomendado - Todo en uno)
Render permite desplegar el proyecto como Web Services.
1. Crea un Web Service para el **Backend**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
2. Crea otro Web Service para el **Frontend** (Streamlit):
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
   - Configura la variable de entorno `API_URL` apuntando a la URL del backend.

### Opción B: Streamlit Community Cloud + Railway / Render
1. **Backend**: Despliégalo en Render o Railway usando el comando de `uvicorn`.
2. **Frontend**: Sincroniza tu repositorio con [share.streamlit.io](https://share.streamlit.io/). Es gratuito y está optimizado específicamente para Streamlit.

---

## Limpieza Realizada
El repositorio ha sido optimizado:
- Se eliminaron copias duplicadas de la carpeta del proyecto.
- Se eliminaron scripts de desarrollo (`figs_informe.py`, pruebas aisladas).
- Se borraron imágenes estáticas pesadas de diagramas antiguos en `app/data/processed/`.
- Dependencias estrictas necesarias para producción consolidadas en `requirements.txt`.
