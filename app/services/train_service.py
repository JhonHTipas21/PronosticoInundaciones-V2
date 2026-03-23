# app/services/train_service.py
"""
Servicio de entrenamiento de modelos de predicción de caudales.

Pipeline: StandardScaler → PolynomialFeatures(degree=2) → Regressor
con TransformedTargetRegressor(log1p/expm1), clipping de picos,
y cálculo de σ-residuales para intervalos de incertidumbre.

Modelos soportados: Ridge, Lasso, ElasticNet.
"""
from pathlib import Path
import json
import functools
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import joblib

from app.services.feature_service import build_features


# ─────────────────────────────────────────────────────────────────────
# Pipeline de Limpieza Estricta
# ─────────────────────────────────────────────────────────────────────

def limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de limpieza profunda para datos crudos de la API o CSV.

    1. Conversión de fechas: pd.to_datetime, elimina NaT
    2. Casteo numérico: convierte columnas clave a float
    3. Interpolación temporal: rellena huecos ≤2 registros
    4. Filtro IQR: elimina outliers estadísticos por estación
    5. Elimina caudales negativos (físicamente imposibles)
    """
    df = df.copy()

    # 1. Conversión de fechas
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        # Strip timezone para evitar filtros vacíos
        if df['fecha'].dt.tz is not None:
            df['fecha'] = df['fecha'].dt.tz_localize(None)
        df = df.dropna(subset=['fecha']).sort_values('fecha')

    # 2. Casteo numérico forzado
    cols_num = ['lluvia_mm', 'temperatura_C', 'impermeabilidad_pct', 'caudal_m3s']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Interpolación temporal para huecos pequeños (≤2 registros)
    if 'estacion' in df.columns and 'fecha' in df.columns:
        frames = []
        for est, grp in df.groupby('estacion'):
            grp = grp.set_index('fecha').sort_index()
            for col in cols_num:
                if col in grp.columns:
                    grp[col] = grp[col].interpolate(method='time', limit=2)
            grp = grp.reset_index()
            frames.append(grp)
        df = pd.concat(frames, ignore_index=True)
    else:
        for col in cols_num:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit=2)

    # 4. Eliminar NaN restantes en columnas críticas
    mandatory = [c for c in ['caudal_m3s', 'lluvia_mm'] if c in df.columns]
    if mandatory:
        df = df.dropna(subset=mandatory)

    # 5. Eliminar caudales negativos (físicamente imposibles)
    if 'caudal_m3s' in df.columns:
        df = df[df['caudal_m3s'] >= 0].copy()

    # 6. Filtro IQR por estación para eliminar outliers
    if 'caudal_m3s' in df.columns and 'estacion' in df.columns:
        frames_clean = []
        for est, grp in df.groupby('estacion'):
            Q1 = grp['caudal_m3s'].quantile(0.25)
            Q3 = grp['caudal_m3s'].quantile(0.75)
            IQR = Q3 - Q1
            lower = max(0, Q1 - 1.5 * IQR)
            upper = Q3 + 1.5 * IQR
            grp_clean = grp[(grp['caudal_m3s'] >= lower) & (grp['caudal_m3s'] <= upper)]
            frames_clean.append(grp_clean)
        df = pd.concat(frames_clean, ignore_index=True)

    return df.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────
# Configuración de modelos y grillas de hiperparámetros
# ─────────────────────────────────────────────────────────────────────

def _make_estimator(model_type: str):
    """Crea estimador base y grilla de hiperparámetros."""
    if model_type == "ridge":
        base = Ridge(random_state=0)
        grid = {
            # Incluye alphas muy pequeños para evitar sobre-regularización (aplastamiento)
            "reg__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        }
    elif model_type == "lasso":
        base = Lasso(max_iter=200, tol=0.01, random_state=0)
        grid = {
            "reg__alpha": [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        }
    elif model_type == "elasticnet":
        base = ElasticNet(max_iter=200, tol=0.01, random_state=0)
        grid = {
            "reg__alpha": [0.001, 0.01, 0.1],
            "reg__l1_ratio": [0.3, 0.5, 0.7],
        }
    else:
        raise ValueError("model_type debe ser 'ridge', 'lasso' o 'elasticnet'")
    return base, grid


def _build_pipeline(feats_num: list, feats_cat: list, model_type: str):
    """
    Construye el pipeline completo:
      ColumnTransformer(RobustScaler + OneHot)
      → PolynomialFeatures(degree=2, interaction_only=True)
      → Regressor
    envuelto con TransformedTargetRegressor(log1p/expm1).
    """
    feats_poly = ["cn_number", "api_4", "lluvia_mm"]
    feats_num_linear = [f for f in feats_num if f not in feats_poly]

    num_pipe_poly = Pipeline([
        ("scaler", RobustScaler()),
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num_poly", num_pipe_poly, feats_poly),
            ("num_lin", RobustScaler(), feats_num_linear),
            ("cat", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False), feats_cat),
        ],
        remainder="drop",
    )

    base_est, grid = _make_estimator(model_type)

    pipe_core = Pipeline([
        ("pre", pre),
        ("reg", base_est),
    ])

    model = TransformedTargetRegressor(
        regressor=pipe_core,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    param_grid = {f"regressor__{k}": v for k, v in grid.items()}
    return model, param_grid


def _clip_predictions(y_pred: np.ndarray, y_train: np.ndarray,
                      factor: float = 1.5) -> np.ndarray:
    """Clipping estadístico: q99 × factor."""
    upper = np.percentile(y_train, 99) * factor
    return np.clip(y_pred, 0.0, max(upper, 1.0))


# ─────────────────────────────────────────────────────────────────────
# Función principal de entrenamiento
# ─────────────────────────────────────────────────────────────────────

def train_from_df(df: pd.DataFrame, horizon: int,
                  model_type: str, model_dir: str) -> dict:
    """
    Entrena un modelo desde un DataFrame crudo.

    Además de las métricas CV, calcula la desviación estándar de
    residuales por estación para cuantificación de incertidumbre.
    """
    # Pipeline de limpieza estricta antes de entrenar
    df = limpiar_dataframe(df)

    dfm, feats_num, feats_cat = build_features(df, horizon=horizon)

    X = dfm[feats_num + feats_cat].copy()
    y = dfm["y_target"].values
    stations = dfm["estacion"].values

    # Validación
    X_num = X[feats_num].to_numpy(dtype=float)
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y_target contiene NaN/Inf.")
    if np.isnan(X_num).any() or np.isinf(X_num).any():
        raise ValueError("Features numéricas contienen NaN/Inf.")

    # Pipeline
    model, param_grid = _build_pipeline(feats_num, feats_cat, model_type)

    # Cross-validation temporal
    tscv = TimeSeriesSplit(n_splits=3)
    gscv = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
        refit="rmse",
        n_jobs=1,
        verbose=1,
        error_score="raise",
    )
    gscv.fit(X, y)

    best = gscv.best_estimator_
    best_params = gscv.best_params_

    # ── Métricas CV + acumulación de residuales ──
    r2s, rmses, maes = [], [], []
    all_residuals = []       # (estacion, residual)
    for tr, te in tscv.split(X):
        best.fit(X.iloc[tr], y[tr])
        yp = best.predict(X.iloc[te])
        yp = _clip_predictions(yp, y[tr])
        
        # Filtro de validación rigurosa antes de métricas (elimina NaNs y residuales atípicos)
        yt = y[te]
        valid = (yt >= 0) & (yp >= 0) & ~np.isnan(yt) & ~np.isnan(yp)
        yt_v = yt[valid]
        yp_v = yp[valid]
        
        if len(yt_v) > 1:
            r2 = float(r2_score(yt_v, yp_v))
            # Ignorar r2 extremadamente negativos que distorsionan si hay un solo pico mal predicho
            if r2 >= -1.0:
                r2s.append(r2)
            rmses.append(float(np.sqrt(mean_squared_error(yt_v, yp_v))))
            maes.append(float(mean_absolute_error(yt_v, yp_v)))
            
            # Residuales para incertidumbre
            valid_te = te[valid]
            for j, val in enumerate(yt_v):
                idx = valid_te[j]
                all_residuals.append((stations[idx], float(val - yp_v[j])))

    r2_cv = float(np.mean(r2s))
    rmse_cv = float(np.mean(rmses))
    mae_cv = float(np.mean(maes))

    # ── σ de residuales por estación ──
    res_df = pd.DataFrame(all_residuals, columns=["estacion", "residual"])
    std_by_station = {}
    for est, grp in res_df.groupby("estacion"):
        std_by_station[est] = round(float(grp["residual"].std()), 6)
    std_global = round(float(res_df["residual"].std()), 6)

    # ── Modelo final entrenado con todos los datos ──
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    best.fit(X, y)
    joblib.dump(best, out / "modelo_regresion.pkl")

    q99 = float(np.percentile(y, 99))

    meta = {
        "model_type": model_type,
        "horizon": int(horizon),
        "features_numeric": list(feats_num),
        "features_categorical": list(feats_cat),
        "best_params": best_params,
        "r2_cv_mean": r2_cv,
        "rmse_cv_mean": rmse_cv,
        "mae_cv_mean": mae_cv,
        "n_samples": int(len(dfm)),
        "y_q99": q99,
        "clip_factor": 1.5,
        "residual_std_by_station": std_by_station,
        "residual_std_global": std_global,
    }

    # Alerta de recalibración si R² < 0.80
    if r2_cv < 0.80:
        meta["alerta"] = "Modelo requiere recalibración de hiperparámetros"
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return meta


@functools.lru_cache(maxsize=1)
def load_model(model_dir: str):
    """Carga modelo y metadata desde disco."""
    p = Path(model_dir)
    pkl_files = list(p.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No se encontró ningún archivo .pkl en {model_dir}")
    # Priorizar ridge si existe
    ridge_files = [f for f in pkl_files if "ridge" in f.name.lower()]
    model_file = ridge_files[0] if ridge_files else pkl_files[0]
    
    model = joblib.load(model_file)
    meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
    return model, meta
