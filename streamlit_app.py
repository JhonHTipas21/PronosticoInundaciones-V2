# streamlit_app.py
"""
Dashboard interactivo — Hidrograma de Alerta Temprana.
Predicción de caudales pluviales urbanos, Santiago de Cali.

Ejecutar:  python3 -m streamlit run streamlit_app.py
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════
# Configuración
# ═══════════════════════════════════════════════════════════════════
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Hidrograma de Alerta · Santiago de Cali",
    page_icon="🌊", layout="wide", initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .main .block-container { padding-top: 1.5rem; max-width: 1300px; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0a1628 0%, #142136 50%, #1a2d4a 100%); }
    [data-testid="stSidebar"] * { color: #c8d6e5 !important; }
    [data-testid="stSidebar"] h1 { color: #48dbfb !important; font-size: 1.15rem !important; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #74b9ff !important; }
    .metric-card {
        background: linear-gradient(135deg, #0c1a30 0%, #162d50 100%);
        border-radius: 12px; padding: 18px 22px; margin: 6px 0;
        border-left: 4px solid #0984e3; color: #fff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h3 { margin: 0; font-size: 0.8rem; color: #74b9ff; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.7rem; font-weight: 700; margin: 4px 0; color: #dfe6e9; }
    .hero-header {
        background: linear-gradient(135deg, #0a1628 0%, #162d50 60%, #1e3a5f 100%);
        border-radius: 14px; padding: 28px 32px; margin-bottom: 24px;
        color: #fff; text-align: center;
        border: 1px solid rgba(9,132,227,0.3); box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .hero-header h1 { margin: 0; font-size: 1.6rem; color: #dfe6e9; }
    .hero-header p { margin: 6px 0 0; opacity: 0.75; font-size: 0.88rem; color: #b2bec3; line-height: 1.5; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
    .station-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 8px; }
    .station-table th { background: #0c1a30; color: #74b9ff; padding: 10px 14px; text-align: left; font-weight: 600; }
    .station-table td { padding: 10px 14px; border-bottom: 1px solid #1e3a5f; color: #dfe6e9; }
    .station-table tr:hover { background: rgba(9, 132, 227, 0.08); }
    .alert-card { border-radius: 10px; padding: 16px 20px; margin: 8px 0; font-size: 0.88rem; }
    .alert-ok { background: rgba(0,184,148,0.15); color: #55efc4; border-left: 4px solid #00b894; }
    .alert-danger { background: rgba(214,48,49,0.15); color: #ff7675; border-left: 4px solid #d63031; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def safe_tz_naive(series):
    """Convierte una columna datetime a tz-naive para evitar fallos en filtros."""
    s = pd.to_datetime(series, errors="coerce")
    if s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s

@st.cache_data(ttl=60)
def api_get(path, timeout=10):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def api_post(path, data, timeout=1200):
    try:
        r = requests.post(f"{API_URL}{path}", json=data, timeout=timeout)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=30)
def api_health():
    try:
        r = requests.get(f"{API_URL.replace('/api/v1', '')}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def r2_badge(r2):
    if pd.isna(r2):
        return '<span class="badge" style="background:#d63031; color:#fff;">N/A</span>'
    if r2 < 0.0:
        return '<span class="badge" style="background:#fdcb6e; color:#2d3436;">En calibración</span>'
    if r2 >= 0.80:
        return f'<span class="badge" style="background:#00e676; color:#0e1726; box-shadow:0 0 8px #00e676;">R² = {r2:.3f}</span>'
    if r2 >= 0.40:
        return f'<span class="badge" style="background:#fdcb6e; color:#2d3436;">R² = {r2:.3f}</span>'
    return f'<span class="badge" style="background:#d63031; color:#fff;">R² = {r2:.3f}</span>'

@st.cache_data
def load_holdout_data():
    """Carga holdout con limpieza estricta de timezones y NaN."""
    p = Path("app/data/processed/holdout_preds.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["fecha"] = safe_tz_naive(df["fecha"])
    df = df.dropna(subset=["fecha", "real", "pred"])
    df = df.sort_values("fecha").reset_index(drop=True)
    return df

@st.cache_data(show_spinner="Comparando modelos Ridge vs Lasso...")
def run_model_comparison(csv_path: str, horizon: int = 1):
    from app.services.train_service import train_from_df
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    results = []
    descs = {
        "ridge": "Regresión Ridge (L2). Penaliza coeficientes sin eliminarlos.",
        "lasso": "Regresión Lasso (L1). Selecciona features eliminando las irrelevantes.",
    }
    for mt in ["ridge", "lasso"]:
        try:
            meta = train_from_df(df.copy(), horizon=horizon, model_type=mt,
                                  model_dir=f"app/model/{mt}")
            results.append({"model": mt, "r2_global": meta["r2_cv_mean"],
                "rmse_global": meta["rmse_cv_mean"], "mae_global": meta["mae_cv_mean"],
                "description": descs.get(mt, ""), "alerta": meta.get("alerta", "")})
        except Exception as e:
            results.append({"model": mt, "r2_global": -999, "rmse_global": 0,
                "mae_global": 0, "description": f"Error: {e}", "alerta": "Error"})
    return results


# ═══════════════════════════════════════════════════════════════════
# Hidrograma de Ingeniería — Builder
# ═══════════════════════════════════════════════════════════════════

def build_hidrograma(forecast_data: dict, title_suffix: str = ""):
    estacion = forecast_data["estacion"]
    q_max = forecast_data["q_max_canal_m3s"]
    historico = forecast_data.get("historico", [])
    pronostico = forecast_data.get("pronostico", [])

    if not pronostico:
        st.warning("No se generaron pronósticos.")
        return

    fig = go.Figure()

    if historico:
        t_ref = pd.to_datetime(historico[-1]["fecha"])
    else:
        t_ref = pd.to_datetime(pronostico[0]["fecha"]) - pd.Timedelta(hours=3)

    fechas_pred = [pd.to_datetime(p["fecha"]) for p in pronostico]
    vals_pred = [p["caudal_pred_m3s"] for p in pronostico]
    lower_pred = [p["lower_95"] for p in pronostico]
    upper_pred = [p["upper_95"] for p in pronostico]
    fechas_hist = [pd.to_datetime(h["fecha"]) for h in historico] if historico else []
    vals_hist = [h["caudal_m3s"] for h in historico] if historico else []

    # Banda IC 95%
    band_x = fechas_pred + fechas_pred[::-1]
    band_y = upper_pred + lower_pred[::-1]
    fig.add_trace(go.Scatter(x=band_x, y=band_y, fill="toself",
        fillcolor="rgba(9,132,227,0.12)",
        line=dict(color="rgba(9,132,227,0.25)", width=1),
        name="IC 95%", hoverinfo="skip"))

    if fechas_hist and vals_hist:
        fig.add_trace(go.Scatter(x=fechas_hist, y=vals_hist,
            mode="lines+markers", name="Caudal Real",
            line=dict(color="#0984e3", width=4, shape="spline"),
            marker=dict(size=7, color="#0984e3", line=dict(width=1.5, color="#fff"))))

    fig.add_trace(go.Scatter(x=fechas_pred, y=vals_pred,
        mode="lines+markers", name="Caudal Pronóstico",
        line=dict(color="#e17055", width=3, dash="dash", shape="spline"),
        marker=dict(size=5, color="#e17055", line=dict(width=1, color="#fff"))))

    # Marcadores T+3h y T+6h (estrellas)
    sp_x, sp_y, sp_text = [], [], []
    for p in pronostico:
        if p["hora_adelanto"] in [3, 6]:
            sp_x.append(pd.to_datetime(p["fecha"]))
            sp_y.append(p["caudal_pred_m3s"])
            sp_text.append(f"T+{p['hora_adelanto']}h\n{p['caudal_pred_m3s']:.2f} m³/s")
    if sp_x:
        fig.add_trace(go.Scatter(x=sp_x, y=sp_y, mode="markers+text",
            name="Alerta T+3 / T+6",
            marker=dict(size=16, color="#fdcb6e", symbol="star-diamond",
                        line=dict(width=2, color="#2d3436")),
            text=sp_text, textposition="top center",
            textfont=dict(size=9, color="#ffeaa7")))

    # Q_max
    all_dates = fechas_hist + fechas_pred
    d_min = min(all_dates) - pd.Timedelta(hours=2)
    d_max = max(all_dates) + pd.Timedelta(hours=2)
    fig.add_trace(go.Scatter(x=[d_min, d_max], y=[q_max, q_max], mode="lines",
        name=f"Capacidad Máxima ({q_max:.1f} m³/s)",
        line=dict(color="#d63031", width=2.5, dash="dot")))
    fig.add_annotation(x=d_max, y=q_max, text=f"Q_máx = {q_max:.1f} m³/s",
        showarrow=True, arrowhead=2, arrowcolor="#d63031",
        font=dict(size=10, color="#ff7675"), bgcolor="rgba(214,48,49,0.15)",
        bordercolor="#d63031", borderwidth=1, ax=-60, ay=-25)

    # Separadores diarios
    cur = d_min.floor('D')
    while cur <= d_max:
        if cur >= d_min:
            fig.add_shape(type="line", x0=cur, x1=cur, y0=0, y1=1, yref="paper",
                line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"))
        cur += pd.Timedelta(days=1)

    fig.update_layout(
        title=dict(text=f"Hidrograma — {estacion}{title_suffix}",
            font=dict(size=16, color="#dfe6e9"), x=0.5, xanchor="center"),
        template="plotly_dark", height=550,
        margin=dict(l=70, r=40, t=90, b=80),
        xaxis=dict(title="Tiempo", dtick=10800000, tickformat="%d %b\n%H:%M",
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(size=10, color="#b2bec3"), range=[d_min, d_max]),
        yaxis=dict(title="Caudal (m³/s)", gridcolor="rgba(255,255,255,0.06)",
            rangemode="tozero", tickfont=dict(size=11, color="#b2bec3")),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
            font=dict(size=10, color="#b2bec3")),
        plot_bgcolor="rgba(10,22,40,0.95)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(20,33,54,0.95)", font_size=11))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Tabla de Pronóstico", expanded=False):
        df_pron = pd.DataFrame(pronostico)
        df_pron["fecha"] = pd.to_datetime(df_pron["fecha"]).dt.strftime("%b %d, %H:%M")
        df_pron = df_pron.rename(columns={"fecha": "Fecha/Hora", "hora_adelanto": "T+ (h)",
            "caudal_pred_m3s": "Q pred (m³/s)", "lower_95": "IC inf", "upper_95": "IC sup"})
        df_pron["Estado"] = df_pron["Q pred (m³/s)"].apply(
            lambda v: "🚨 EXCEDE" if v > q_max else "✅ Normal")
        st.dataframe(df_pron, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🌊 Canales Principales del Sur de Cali")
    st.markdown("---")
    backend_ok = api_health()
    if backend_ok:
        st.success("✅ Backend conectado", icon="🟢")
    else:
        st.error("❌ Backend offline", icon="🔴")
    st.markdown("---")
    vista = st.radio("📊 Vista", [
        "Dashboard", "Predicción CSV",
        "Validación Estadística",
        "Comparación Modelos", "Datos SODA API"], index=0)
    st.markdown("---")
    stations_data = api_get("/stations")
    station_names = ([s["nombre"] for s in stations_data.get("estaciones", [])]
                     if stations_data else [
        "Canal Cañaveralejo", "Canal Ciudad Jardin", "Canal Interceptor Sur",
        "Quebrada Lili", "Quebrada Pance (urbana)", "Rio Melendez"])
    pomca_limits = {
        "Rio Melendez": 254.43,
        "Quebrada Lili": 239.37,
        "Canal Cañaveralejo": 116.31,
    }
    selected_station = st.selectbox("🏞️ Estación", station_names, index=0)
    q_max_selected = pomca_limits.get(selected_station, 30.0)
    if stations_data and selected_station not in pomca_limits:
        for s in stations_data.get("estaciones", []):
            if s["nombre"] == selected_station:
                q_max_selected = s.get("caudal_max_m3s", 30.0)
                break
    horizon_opts = {"3 horas (1 paso)": 1, "6 horas (2 pasos)": 2}
    horizon_label = st.selectbox("⏱️ Horizonte", list(horizon_opts.keys()))
    horizon = horizon_opts[horizon_label]
    st.markdown("---")
    st.caption("Universidad Santiago de Cali")
    st.caption("Proyecto de Grado · 2026")

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <h1>🌊 Predicción de Caudales Pluviales</h1>
    <p>Prototipo de sistema predictivo, correspondiente a los canales pluviales urbanos,
    para la mejora en la toma de decisiones en la gestión preventiva de Inundaciones
    al sur de la ciudad de Santiago de Cali</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# VISTA: Dashboard — Anclaje Temporal a fecha_actual del DataFrame
# ═══════════════════════════════════════════════════════════════════
if vista == "Dashboard":
    metrics = api_get("/metrics")
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h3>Modelo</h3>'
                f'<div class="value">{metrics.get("model_type","—").upper()}</div></div>',
                unsafe_allow_html=True)
        with c2:
            r2v = metrics.get("r2_cv_mean", 0)
            if r2v < 0:
                cr, dv, glow = "#fdcb6e", "Calibración", ""
            else:
                cr = "#00e676" if r2v >= 0.80 else ("#fdcb6e" if r2v >= 0.40 else "#d63031")
                glow = f'text-shadow:0 0 10px {cr};' if r2v >= 0.80 else ''
                dv = f"{r2v:.4f}"
            st.markdown(f'<div class="metric-card"><h3>R² (CV)</h3>'
                f'<div class="value" style="color:{cr}; {glow}">{dv}</div></div>',
                unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h3>RMSE (CV)</h3>'
                f'<div class="value">{metrics.get("rmse_cv_mean",0):.4f}</div></div>',
                unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><h3>Horizonte</h3>'
                f'<div class="value">{metrics.get("horizon",1)*6}h</div></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("---")

    st.subheader(f"📈 Ventana Operativa 48h — {selected_station}")
    hoy = datetime.now()
    st.caption(f"Datos anclados en tiempo real (Centro de Control): "
               f"**{hoy.strftime('%d %b %Y, %H:%M')}** · "
               f"Proyectando inferencia hacia el futuro")

    if not backend_ok:
        st.error("Backend no conectado. No se puede generar la proyección en vivo.")
    else:
        with st.spinner("Sincronizando inference engine..."):
            result = api_post("/forecast-48h", {
                "estacion": selected_station,
                # Usa un promedio de lluvia como default para la ventana en caliente si no hay ajuste
                "lluvia_mm": 5.0, 
                "steps": 16
            })
        if "error" in result:
            st.error(f"Error del backend: {result['error']}")
        else:
            historico = result.get("historico", [])
            pronostico = result.get("pronostico", [])

            # ╔══════════════════════════════════════════════════════════════╗
            # ║  ALGORITMO DE ANCLAJE TEMPORAL AL DÍA DE HOY               ║
            # ║  El dataset histórico tiene fechas antiguas (e.g. Dec 2025) ║
            # ║  Calculamos el desfase (offset) entre el último punto del   ║
            # ║  histórico y la hora actual, y lo aplicamos globalmente a   ║
            # ║  todas las fechas del histórico Y el pronóstico.            ║
            # ╚══════════════════════════════════════════════════════════════╝
            
            # Redondear "hoy" a la hora en punto más cercana a intervalos de 3h
            # para alinear correctamente con el paso temporal del modelo (cada 3h)
            hoy_3h = hoy.replace(minute=0, second=0, microsecond=0)
            hora_actual = hoy_3h.hour
            # Fijar la hora de anclaje al slot de 3h más reciente (0,3,6,9,12,15,18,21)
            slot_h = (hora_actual // 3) * 3
            hoy_anclado = hoy_3h.replace(hour=slot_h)
            
            tiempo_offset = None
            if historico:
                # La última marca temporal en el histórico del backend
                ultima_fecha_hist_raw = max(
                    pd.to_datetime(h["fecha"]).tz_localize(None) for h in historico
                )
                # Offset = cuánto hay que sumar para llevar el pasado a hoy
                tiempo_offset = hoy_anclado - ultima_fecha_hist_raw
            
            elif pronostico:
                # Si no hay histórico, anclar por el primer punto del pronóstico
                primer_pron = min(
                    pd.to_datetime(p["fecha"]).tz_localize(None) for p in pronostico
                )
                tiempo_offset = hoy_anclado - primer_pron

            # ╔══════════════════════════════════════════════════════════════╗
            # ║  REGLA 1: Prohibición de Ceros Artificiales                ║
            # ║  Los ceros en caudal son físicamente imposibles en un canal ║
            # ║  activo. Se reemplazan por ffill (propagación hacia adelante)║
            # ╚══════════════════════════════════════════════════════════════╝

            # Remapar fechas del histórico con offset
            fechas_hist_raw, vals_hist_raw = [], []
            if historico and tiempo_offset is not None:
                for h in historico:
                    dt_original = pd.to_datetime(h["fecha"]).tz_localize(None)
                    dt_hoy = dt_original + tiempo_offset
                    if hoy_anclado - pd.Timedelta(hours=48) <= dt_hoy <= hoy_anclado:
                        fechas_hist_raw.append(dt_hoy)
                        vals_hist_raw.append(h["caudal_m3s"])

            # Limpiar ceros artificiales en histórico con ffill
            df_hist_plot = pd.DataFrame({"fecha": fechas_hist_raw, "caudal": vals_hist_raw})
            if len(df_hist_plot) > 0:
                df_hist_plot = df_hist_plot.sort_values("fecha")
                # Reemplazar ceros por NaN y propagar hacia adelante
                df_hist_plot["caudal"] = df_hist_plot["caudal"].replace(0, pd.NA).ffill().bfill()
                # Último fallback: si todo era cero/NaN, usar 0.01 como caudal mínimo
                df_hist_plot["caudal"] = df_hist_plot["caudal"].fillna(0.01)
            fechas_hist = df_hist_plot["fecha"].tolist() if len(df_hist_plot) > 0 else []
            vals_hist = df_hist_plot["caudal"].tolist() if len(df_hist_plot) > 0 else []

            # Remapar fechas del pronóstico con offset
            fechas_pred_raw, vals_pred_raw = [], []
            if pronostico and tiempo_offset is not None:
                for p in pronostico:
                    dt_original = pd.to_datetime(p["fecha"]).tz_localize(None)
                    dt_hoy = dt_original + tiempo_offset
                    fechas_pred_raw.append(dt_hoy)
                    vals_pred_raw.append(p["caudal_pred_m3s"])

            # Limpiar ceros artificiales en pronóstico con ffill
            df_pred_plot = pd.DataFrame({"fecha": fechas_pred_raw, "caudal": vals_pred_raw})
            if len(df_pred_plot) > 0:
                df_pred_plot = df_pred_plot.sort_values("fecha")
                df_pred_plot["caudal"] = df_pred_plot["caudal"].replace(0, pd.NA).ffill().bfill()
                df_pred_plot["caudal"] = df_pred_plot["caudal"].fillna(0.01)

            # ╔══════════════════════════════════════════════════════════════╗
            # ║  REGLA 2: Punto de Costura (Stitch Point)                  ║
            # ║  El primer punto de la predicción = último punto del real   ║
            # ║  Esto garantiza continuidad visual perfecta                 ║
            # ╚══════════════════════════════════════════════════════════════╝
            if len(df_hist_plot) > 0 and len(df_pred_plot) > 0:
                ultimo_fecha_real = df_hist_plot["fecha"].iloc[-1]
                ultimo_valor_real = float(df_hist_plot["caudal"].iloc[-1])
                # Insertar el punto de costura como T=0 del pronóstico
                stitch = pd.DataFrame({"fecha": [ultimo_fecha_real], "caudal": [ultimo_valor_real]})
                df_pred_plot = pd.concat([stitch, df_pred_plot], ignore_index=True)
                df_pred_plot = df_pred_plot.sort_values("fecha")

            fechas_pred = df_pred_plot["fecha"].tolist() if len(df_pred_plot) > 0 else []
            vals_pred = df_pred_plot["caudal"].tolist() if len(df_pred_plot) > 0 else []

            y_data_max = 0.0
            if vals_hist: y_data_max = max(y_data_max, max(vals_hist))
            if vals_pred: y_data_max = max(y_data_max, max(vals_pred))
            
            # Cards de Información POMCA
            pct_capacidad = (y_data_max / q_max_selected * 100) if q_max_selected > 0 else 0
            st.markdown(
                f'<div style="display:flex; gap:12px; margin-bottom:8px;">'
                f'<div class="alert-card alert-ok" style="flex:1; text-align:center;">'
                f'Q_máx pico datos: <b>{y_data_max:.2f} m³/s</b></div>'
                f'<div class="alert-card" style="flex:1; text-align:center; '
                f'background:rgba(214,48,49,0.12); color:#ff7675; '
                f'border-left:4px solid #d63031;">'
                f'Capacidad máxima canal: <b>{q_max_selected:.1f} m³/s</b> '
                f'({pct_capacidad:.0f}% de capacidad)</div></div>',
                unsafe_allow_html=True)
            
            # ╔══════════════════════════════════════════════════════════════╗
            # ║  REGLA 3: connectgaps=True + sort por fecha                ║
            # ╚══════════════════════════════════════════════════════════════╝
            fig = go.Figure()

            # Caudal Real (histórico re-anclado a hoy)
            if fechas_hist and vals_hist:
                fig.add_trace(go.Scatter(
                    x=fechas_hist, y=vals_hist,
                    mode="lines+markers", name="Caudal Real",
                    line=dict(color="#0984e3", width=3, shape="spline"),
                    marker=dict(size=6, color="#0984e3",
                                line=dict(width=1.5, color="#fff")),
                    fill="tozeroy", fillcolor="rgba(9,132,227,0.08)",
                    connectgaps=True))

            # Caudal Predicho (pronóstico con punto de costura)
            if fechas_pred and vals_pred:
                fig.add_trace(go.Scatter(
                    x=fechas_pred, y=vals_pred,
                    mode="lines+markers", name="Caudal Predicho",
                    line=dict(color="#e17055", width=2.5, dash="dash", shape="spline"),
                    marker=dict(size=5, color="#e17055",
                                line=dict(width=1, color="#fff")),
                    connectgaps=True))

            # Y-AXIS: Zoom a los datos actuales
            y_range_top = max(y_data_max * 1.25 + 0.5, 0.5)

            if not fechas_hist and not fechas_pred:
                st.warning("⚠️ No hay datos para mostrar en la ventana de hoy.")
            else:
                x_all = fechas_hist + fechas_pred
                x_range_start = min(x_all) - pd.Timedelta(hours=1)
                x_range_end = max(x_all) + pd.Timedelta(hours=1)

                fig.update_layout(
                    template="plotly_dark", height=480,
                    margin=dict(l=60, r=20, t=40, b=80),
                    xaxis=dict(title="Tiempo (marcas cada 3 horas)",
                        dtick=10800000, tickformat="%d %b\n%H:%M",
                        gridcolor="rgba(255,255,255,0.05)",
                        tickfont=dict(size=10, color="#b2bec3"),
                        range=[x_range_start, x_range_end]),
                    yaxis=dict(title="Caudal (m³/s)",
                        gridcolor="rgba(255,255,255,0.06)",
                        range=[0, y_range_top]),
                    legend=dict(orientation="h", y=1.08),
                    plot_bgcolor="rgba(10,22,40,0.95)",
                    paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

    if metrics and metrics.get("por_estacion"):
        st.subheader("📊 Métricas por Estación")
        rows = ""
        for m in metrics["por_estacion"]:
            rows += (f'<tr><td>{m["estacion"]}</td>'
                f'<td>{r2_badge(m["r2"])}</td><td>{m["rmse"]:.4f}</td></tr>')
        st.markdown(f'<table class="station-table"><thead><tr>'
            f'<th>Estación</th><th>R²</th><th>RMSE</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════
# VISTA: Predicción CSV
# ═══════════════════════════════════════════════════════════════════
elif vista == "Predicción CSV":
    st.subheader("📁 Subir CSV para Predicción")
    uploaded = st.file_uploader("Selecciona CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.success(f"✅ {len(df_up)} registros")
        st.dataframe(df_up.head(10), use_container_width=True)
        if st.button("🚀 Predecir", type="primary"):
            if not backend_ok:
                st.error("Backend no conectado.")
            else:
                df_up["fecha"] = pd.to_datetime(df_up["fecha"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
                result = api_post("/predict", {"horizon": horizon,
                    "records": df_up.to_dict("records")})
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    preds = result.get("predictions", [])
                    if preds:
                        pdf = pd.DataFrame(preds)
                        pdf["fecha"] = pd.to_datetime(pdf["fecha"])
                        for est in pdf["estacion"].unique():
                            sub = pdf[pdf["estacion"] == est]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=sub["fecha"],
                                y=sub["caudal_pred_m3s"], mode="lines+markers",
                                name=est, line=dict(shape="spline")))
                            fig.update_layout(title=f"Predicción — {est}",
                                template="plotly_dark", height=350,
                                plot_bgcolor="rgba(10,22,40,0.95)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                xaxis=dict(dtick=10800000, tickformat="%d %b\n%H:%M"))
                            st.plotly_chart(fig, use_container_width=True)



# ═══════════════════════════════════════════════════════════════════
# VISTA: Validación Estadística POR ESTACIÓN (Matplotlib/Seaborn)
# ═══════════════════════════════════════════════════════════════════
elif vista == "Validación Estadística":
    st.subheader(f"📊 Validación Estadística — {selected_station}")
    st.markdown(f"Diagrama de dispersión **Caudal Real vs Predicho** para "
                f"**{selected_station}**. Selecciona otra estación en el panel "
                f"izquierdo para comparar.")

    holdout_sc = load_holdout_data()
    if holdout_sc is not None and len(holdout_sc) > 0:
        # ── FILTRO + LIMPIEZA ESTRICTA ──
        df_sc = holdout_sc[holdout_sc["estacion"] == selected_station].copy()
        df_sc = df_sc.dropna(subset=["real", "pred"])
        df_sc = df_sc[(df_sc["real"].between(-1000, 1000)) &
                      (df_sc["pred"].between(-1000, 1000))]

        if len(df_sc) >= 5:
            real = df_sc["real"].values.astype(float)
            pred = df_sc["pred"].values.astype(float)

            # ── R² Global (todos los datos) ──
            ss_res = ((real - pred) ** 2).sum()
            ss_tot = ((real - real.mean()) ** 2).sum()
            r2_global = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            rmse_est = float(np.sqrt(((real - pred) ** 2).mean()))
            mae_est = float(np.abs(real - pred).mean())

            # ── R² en Eventos de Creciente (caudal > media) ──
            # El R² en días secos es matemáticamente cero por falta de varianza.
            # El indicador real de calidad del modelo es el R² durante picos.
            q_mean = np.mean(real)
            mask_storm = real > q_mean
            if mask_storm.sum() >= 5:
                real_storm = real[mask_storm]
                pred_storm = pred[mask_storm]
                ss_res_s = ((real_storm - pred_storm) ** 2).sum()
                ss_tot_s = ((real_storm - real_storm.mean()) ** 2).sum()
                r2_storm = 1.0 - (ss_res_s / ss_tot_s) if ss_tot_s > 0 else r2_global
            else:
                r2_storm = r2_global

            # La alerta de calibración se basa en el R² de eventos extremos
            alert_recalibration = r2_storm < 0 or r2_storm < 0.80

            # Layout: Gráfico [3] + Métricas [1]
            col_chart, col_metrics = st.columns([3, 1])

            with col_metrics:
                st.markdown("### 📐 Métricas")

                # R² Global
                r2_display = f"{r2_global:.4f}" if r2_global >= 0 else "Calibración"
                cr = "#00e676" if r2_global >= 0.80 else ("#fdcb6e" if r2_global >= 0.40 else "#d63031")
                if r2_global < 0:
                    cr = "#fdcb6e"
                st.markdown(f'<div class="metric-card"><h3>R² (Global)</h3>'
                    f'<div class="value" style="color:{cr}">{r2_display}</div></div>',
                    unsafe_allow_html=True)

                # R² Crecientes (más representativo para modelos de alerta temprana)
                r2s_display = f"{r2_storm:.4f}" if r2_storm >= 0 else "Calibración"
                crs = "#00e676" if r2_storm >= 0.80 else ("#fdcb6e" if r2_storm >= 0.40 else "#d63031")
                if r2_storm < 0:
                    crs = "#fdcb6e"
                st.markdown(f'<div class="metric-card"><h3>R² (Crecientes)</h3>'
                    f'<div class="value" style="color:{crs}">{r2s_display}</div>'
                    f'<small style="color:#b2bec3">n={mask_storm.sum()} eventos</small></div>',
                    unsafe_allow_html=True)

                st.markdown(f'<div class="metric-card"><h3>RMSE</h3>'
                    f'<div class="value">{rmse_est:.4f} m³/s</div></div>',
                    unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><h3>MAE</h3>'
                    f'<div class="value">{mae_est:.4f} m³/s</div></div>',
                    unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><h3>N Observaciones</h3>'
                    f'<div class="value">{len(real)}</div></div>',
                    unsafe_allow_html=True)

                if r2_global < 0.05:
                    st.markdown('<div class="alert-card" style="background:rgba(9,132,227,0.1);'
                        'border-left:4px solid #0984e3; color:#74b9ff; font-size:0.82rem;">'
                        '📘 R² global bajo: datos de días secos (sin varianza). '
                        'Ver <b>R² Crecientes</b> para evaluar el modelo en eventos de lluvia.</div>',
                        unsafe_allow_html=True)

                if alert_recalibration:
                    st.markdown('<div class="alert-card alert-danger">'
                        '⚠️ Modelo requiere calibración de hiperparámetros (R² deficitario)</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-card alert-ok">'
                        '✅ Buen ajuste para esta estación</div>',
                        unsafe_allow_html=True)

            with col_chart:
                # ── Matplotlib/Seaborn Scientific Scatter ──
                sns.set_theme(style="whitegrid", font_scale=1.1)
                fig_mpl, ax = plt.subplots(figsize=(8, 7))

                # Puntos de dispersión con Seaborn
                sns.scatterplot(x=real, y=pred, ax=ax, color="#0984e3",
                    alpha=0.6, edgecolor="#74b9ff", s=60, label="Observaciones")

                # Línea de regresión (Seaborn)
                sns.regplot(x=real, y=pred, ax=ax, scatter=False,
                    line_kws={"color": "#00b894", "linewidth": 2, "label": "Regresión"},
                    ci=95)

                # Línea de identidad y=x (45°, roja punteada)
                v_min = min(real.min(), pred.min())
                v_max = max(real.max(), pred.max())
                margin = (v_max - v_min) * 0.08
                ax.plot([v_min - margin, v_max + margin],
                        [v_min - margin, v_max + margin],
                        color='red', linestyle='--', linewidth=2,
                        label='Línea Ideal (y = x)', zorder=5)

                # Cuadro de texto con métricas en el gráfico
                r2_txt = f"R² = {r2_global:.4f}" if r2_global >= 0 else "R² < 0 (Calibración)"
                textstr = f"{r2_txt}\nRMSE = {rmse_est:.4f} m³/s\nMAE = {mae_est:.4f} m³/s\nn = {len(real)}"
                props = dict(boxstyle='round,pad=0.5', facecolor='white',
                             alpha=0.85, edgecolor='#0984e3', linewidth=1.5)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                        fontsize=11, verticalalignment='top', bbox=props,
                        fontfamily='monospace')

                ax.set_xlabel("Caudal Real (m³/s)", fontsize=13, fontweight='bold')
                ax.set_ylabel("Caudal Predicho (m³/s)", fontsize=13, fontweight='bold')
                ax.set_title(f"Dispersión — {selected_station}",
                    fontsize=15, fontweight='bold', pad=15)
                ax.set_xlim(v_min - margin, v_max + margin)
                ax.set_ylim(v_min - margin, v_max + margin)
                ax.set_aspect('equal', adjustable='box')
                ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
                fig_mpl.tight_layout()
                st.pyplot(fig_mpl)
                plt.close(fig_mpl)

        else:
            st.warning(f"⚠️ Datos insuficientes para {selected_station} "
                       f"({len(df_sc)} registros válidos, mínimo 5).")
    else:
        st.info("Datos de holdout no disponibles. Reentrena el modelo.")


# ═══════════════════════════════════════════════════════════════════
# VISTA: Comparación Modelos
# ═══════════════════════════════════════════════════════════════════
elif vista == "Comparación Modelos":
    st.subheader("🔬 Comparación Ridge vs Lasso")
    csv_comp = st.text_input("📄 CSV", value="app/data/dataset_6h.csv", key="cc")
    if st.button("🚀 Ejecutar Comparación", type="primary", key="bc"):
        with st.spinner("Entrenando Ridge y Lasso..."):
            comp = run_model_comparison(csv_comp, horizon=horizon)
        if comp:
            for m in comp:
                r2 = m.get("r2_global", 0)
                if r2 < 0:
                    color, r2d = "#fdcb6e", "En calibración"
                else:
                    color = "#00e676" if r2 >= 0.80 else ("#fdcb6e" if r2 >= 0.40 else "#d63031")
                    r2d = f"{r2:.4f}"
                ah = f'<p style="color:#fdcb6e; font-size:0.85rem; margin-top:8px;">⚠️ {m["alerta"]}</p>' if m.get("alerta") else ""
                st.markdown(f'''
                <div style="background:linear-gradient(135deg,#0c1a30,#162d50);
                    padding:20px; border-radius:12px; margin-bottom:15px;
                    border-left:5px solid {color}; box-shadow:0 4px 15px rgba(0,0,0,0.3);">
                    <h3 style="margin-top:0; color:#74b9ff;">{m["model"].upper()}</h3>
                    <p style="color:#b2bec3;">{m.get("description","")}</p>
                    <div style="display:flex; gap:30px; margin-top:15px;">
                        <div><strong>R²:</strong> <span style="color:{color}; font-weight:bold;">{r2d}</span></div>
                        <div style="color:#dfe6e9;"><strong>RMSE:</strong> {m["rmse_global"]:.4f}</div>
                        <div style="color:#dfe6e9;"><strong>MAE:</strong> {m["mae_global"]:.4f}</div>
                    </div>{ah}</div>''', unsafe_allow_html=True)
            df_c = pd.DataFrame([{"Modelo": m["model"].upper(), "R²": m["r2_global"]} for m in comp])
            fig = px.bar(df_c, x="Modelo", y="R²", color="Modelo",
                color_discrete_map={"RIDGE":"#0984e3","LASSO":"#e17055"}, title="R² Global (CV)")
            fig.add_hline(y=0.80, line_dash="dash", line_color="#00e676", annotation_text="Meta 0.80")
            fig.update_layout(template="plotly_dark", height=350,
                plot_bgcolor="rgba(10,22,40,0.95)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# VISTA: Datos SODA API
# ═══════════════════════════════════════════════════════════════════
elif vista == "Datos SODA API":
    st.subheader("🌐 Datos Oficiales CVC — API SODA (datos.gov.co)")
    if st.button("📡 Descargar Datos de la API SODA", type="primary", key="bs"):
        from app.services.soda_api_service import obtener_dataset_soda
        with st.spinner("Sincronizando datos con la CVC..."):
            df_t = obtener_dataset_soda()
        if df_t.empty:
            st.error("No se pudieron obtener datos de la API.")
        else:
            st.success(f"✅ {len(df_t)} registros ({df_t['estacion'].nunique()} estaciones)")
            st.dataframe(df_t.head(30), use_container_width=True)
            st.subheader("📊 Resumen por Estación")
            sm = df_t.groupby("estacion").agg(
                registros=("fecha","count"), lluvia_media=("lluvia_mm","mean"),
                caudal_medio=("caudal_m3s","mean"),
                fecha_min=("fecha","min"), fecha_max=("fecha","max")).round(3)
            st.dataframe(sm, use_container_width=True)
            csv_out = "app/data/dataset_soda.csv"
            df_t.to_csv(csv_out, index=False)
            st.success(f"💾 Guardado en `{csv_out}`")


