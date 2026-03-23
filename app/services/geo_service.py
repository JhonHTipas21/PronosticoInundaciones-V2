# app/services/geo_service.py
"""
Servicio geoespacial para el cálculo de coeficientes de escorrentía
dinámicos por estación, basados en el método SCS-CN (Curve Number)
del Servicio de Conservación de Suelos (USDA).

Referencia hidrológica:
  - Chow, V.T., Maidment, D.R., Mays, L.W. (1988). Applied Hydrology.
  - CVC / DAGMA: Datos de cobertura de suelo de Santiago de Cali.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    HAS_GEO = True
except ImportError:
    HAS_GEO = False


# ─────────────────────────────────────────────────────────────────────
# Metadatos de cuencas – valores de referencia para Santiago de Cali
# Fuentes: IGAC, CVC, DAGMA, literatura DOI:10.1016/j.jhydrol.2019.124070
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CuencaInfo:
    """Datos geomorfológicos e hidrológicos de una cuenca."""
    nombre: str
    cn: float                   # Curve Number (condición AMC-II)
    area_km2: float             # Área de la cuenca (km²)
    pendiente_media: float      # Pendiente media del cauce (%)
    pct_impermeable: float      # Porcentaje de superficie impermeable
    longitud_cauce_km: float    # Longitud del cauce principal (km)
    cobertura: str              # Tipo predominante de cobertura
    lat: float                  # Latitud del punto de monitoreo
    lon: float                  # Longitud del punto de monitoreo
    caudal_max_m3s: float = 30.0  # Capacidad máxima teórica del canal (m³/s)
    station_code: str = ""        # Código de la estación CVC/DAGMA

    @property
    def s_retention_mm(self) -> float:
        """Retención potencial máxima S (mm) según método SCS."""
        return 25400.0 / self.cn - 254.0

    @property
    def ia_mm(self) -> float:
        """Abstracción inicial Ia = 0.2 * S (mm)."""
        return 0.2 * self.s_retention_mm

    @property
    def tiempo_concentracion_h(self) -> float:
        """Tiempo de concentración (Kirpich, horas)."""
        if self.pendiente_media <= 0 or self.longitud_cauce_km <= 0:
            return 1.0
        l_m = self.longitud_cauce_km * 1000.0
        s_frac = self.pendiente_media / 100.0
        tc = 0.0195 * (l_m ** 0.77) * (s_frac ** (-0.385)) / 60.0
        return max(tc, 0.1)

    def runoff_mm(self, lluvia_mm: float) -> float:
        """Escorrentía directa Q (mm) por el método SCS-CN."""
        p = max(lluvia_mm, 0.0)
        ia = self.ia_mm
        s = self.s_retention_mm
        if p <= ia:
            return 0.0
        return (p - ia) ** 2 / (p - ia + s)

    def runoff_coefficient(self, lluvia_mm: float) -> float:
        """Coeficiente de escorrentía C = Q/P  ∈ [0, 1]."""
        p = max(lluvia_mm, 0.0)
        if p <= 0:
            return 0.0
        q = self.runoff_mm(p)
        return min(q / p, 1.0)


# ─────────────────────────────────────────────────────────────────────
# Catálogo de cuencas de Santiago de Cali
# ─────────────────────────────────────────────────────────────────────

CUENCAS_CALI: Dict[str, CuencaInfo] = {
    "Canal Cañaveralejo": CuencaInfo(
        nombre="Canal Cañaveralejo",
        cn=85, area_km2=4.2, pendiente_media=3.5,
        pct_impermeable=72.0, longitud_cauce_km=5.8,
        cobertura="urbana_alta_densidad",
        lat=3.397, lon=-76.553, caudal_max_m3s=116.31,
        station_code="CALI002",
    ),
    "Canal Ciudad Jardin": CuencaInfo(
        nombre="Canal Ciudad Jardín",
        cn=78, area_km2=3.1, pendiente_media=4.2,
        pct_impermeable=55.0, longitud_cauce_km=4.1,
        cobertura="urbana_residencial",
        lat=3.392, lon=-76.545, caudal_max_m3s=15.0,
        station_code="CALI003",
    ),
    "Canal Interceptor Sur": CuencaInfo(
        nombre="Canal Interceptor Sur",
        cn=82, area_km2=12.5, pendiente_media=2.8,
        pct_impermeable=65.0, longitud_cauce_km=9.3,
        cobertura="urbana_colector",
        lat=3.400, lon=-76.540, caudal_max_m3s=45.0,
        station_code="CALI004",
    ),
    "Quebrada Lili": CuencaInfo(
        nombre="Quebrada Lili",
        cn=70, area_km2=8.7, pendiente_media=8.5,
        pct_impermeable=35.0, longitud_cauce_km=7.2,
        cobertura="periurbana_vegetal",
        lat=3.385, lon=-76.550, caudal_max_m3s=239.37,
        station_code="CALI005",
    ),
    "Quebrada Pance (urbana)": CuencaInfo(
        nombre="Quebrada Pance (urbana)",
        cn=65, area_km2=22.3, pendiente_media=12.0,
        pct_impermeable=20.0, longitud_cauce_km=14.5,
        cobertura="periurbana_montañosa",
        lat=3.358, lon=-76.575, caudal_max_m3s=35.0,
        station_code="CALI006",
    ),
    "Rio Melendez": CuencaInfo(
        nombre="Río Meléndez",
        cn=72, area_km2=17.8, pendiente_media=10.2,
        pct_impermeable=40.0, longitud_cauce_km=12.1,
        cobertura="mixta_montaña_urbana",
        lat=3.380, lon=-76.560, caudal_max_m3s=254.43,
        station_code="CALI007",
    ),
}

# Alias para búsqueda insensible a acentos / mayúsculas
_ALIAS: Dict[str, str] = {}
for _k in CUENCAS_CALI:
    _low = _k.lower().strip()
    _ALIAS[_low] = _k
    # sin acentos
    import unicodedata as _ud
    _norm = "".join(
        c for c in _ud.normalize("NFD", _low) if _ud.category(c) != "Mn"
    )
    _ALIAS[_norm] = _k


def _resolve_name(estacion: str) -> str:
    """Busca el nombre canónico de la estación."""
    if estacion in CUENCAS_CALI:
        return estacion
    low = estacion.lower().strip()
    if low in _ALIAS:
        return _ALIAS[low]
    import unicodedata as ud
    norm = "".join(
        c for c in ud.normalize("NFD", low) if ud.category(c) != "Mn"
    )
    if norm in _ALIAS:
        return _ALIAS[norm]
    # búsqueda parcial
    for alias_key, canon in _ALIAS.items():
        if norm in alias_key or alias_key in norm:
            return canon
    return estacion  # fallback → devuelve tal cual


def get_cuenca(estacion: str) -> Optional[CuencaInfo]:
    """Retorna la información de cuenca de una estación, o None."""
    name = _resolve_name(estacion)
    return CUENCAS_CALI.get(name)


def get_station_geo_features(estacion: str) -> Dict[str, float]:
    """
    Retorna un diccionario con features geoespaciales estáticas
    para usar directamente como columnas del modelo.
    """
    cuenca = get_cuenca(estacion)
    if cuenca is None:
        return {
            "cn_number": 75.0,
            "area_km2": 5.0,
            "pendiente_media": 5.0,
            "pct_impermeable": 50.0,
            "tiempo_concentracion_h": 1.0,
            "longitud_cauce_km": 5.0,
        }
    return {
        "cn_number": cuenca.cn,
        "area_km2": cuenca.area_km2,
        "pendiente_media": cuenca.pendiente_media,
        "pct_impermeable": cuenca.pct_impermeable,
        "tiempo_concentracion_h": cuenca.tiempo_concentracion_h,
        "longitud_cauce_km": cuenca.longitud_cauce_km,
    }


def compute_runoff_coefficient(estacion: str, lluvia_mm: float) -> float:
    """Calcula C = Q/P dinámico para una estación e intensidad dada."""
    cuenca = get_cuenca(estacion)
    if cuenca is None:
        return 0.5  # valor conservador por defecto
    return cuenca.runoff_coefficient(lluvia_mm)


def compute_runoff_mm(estacion: str, lluvia_mm: float) -> float:
    """Calcula la escorrentía directa Q (mm) para una estación."""
    cuenca = get_cuenca(estacion)
    if cuenca is None:
        return max(lluvia_mm * 0.5, 0.0)
    return cuenca.runoff_mm(lluvia_mm)


# ─────────────────────────────────────────────────────────────────────
# Enriquecimiento de DataFrames
# ─────────────────────────────────────────────────────────────────────

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features geoespaciales al DataFrame de entrenamiento o
    predicción.  Requiere columnas 'estacion' y 'lluvia_mm'.

    Features agregadas:
      - cn_number, area_km2, pendiente_media, pct_impermeable,
        tiempo_concentracion_h, longitud_cauce_km  (estáticas)
      - coef_escorrentia  (dinámico, depende de lluvia_mm)
      - escorrentia_mm    (dinámico, Q en mm)
      - saturacion_suelo  (api_acum × CN / 100, si disponible)
    """
    df = df.copy()

    # Features estáticas por estación (vectorizado)
    geo_cols = [
        "cn_number", "area_km2", "pendiente_media",
        "pct_impermeable", "tiempo_concentracion_h", "longitud_cauce_km",
    ]
    geo_map = {}
    for est in df["estacion"].unique():
        geo_map[est] = get_station_geo_features(est)

    geo_df = pd.DataFrame.from_dict(geo_map, orient="index")
    geo_df.index.name = "estacion"

    for col in geo_cols:
        df[col] = df["estacion"].map(geo_df[col])

    # Features dinámicas (dependen de lluvia_mm)
    if "lluvia_mm" in df.columns:
        # Coeficiente de escorrentía C = f(estación, lluvia)
        df["coef_escorrentia"] = df.apply(
            lambda r: compute_runoff_coefficient(r["estacion"], r["lluvia_mm"]),
            axis=1,
        )
        # Escorrentía directa Q (mm)
        df["escorrentia_mm"] = df.apply(
            lambda r: compute_runoff_mm(r["estacion"], r["lluvia_mm"]),
            axis=1,
        )
    else:
        df["coef_escorrentia"] = 0.5
        df["escorrentia_mm"] = 0.0

    # Interacción saturación del suelo
    if "api_20" in df.columns:
        df["saturacion_suelo"] = df["api_20"] * df["cn_number"] / 100.0
    elif "api_8" in df.columns:
        df["saturacion_suelo"] = df["api_8"] * df["cn_number"] / 100.0

    return df


def get_caudal_max(estacion: str) -> float:
    """Retorna la capacidad máxima teórica del canal (m³/s)."""
    cuenca = get_cuenca(estacion)
    if cuenca is None:
        return 30.0  # valor conservador por defecto
    return cuenca.caudal_max_m3s


def get_stations_metadata() -> List[Dict]:
    """Retorna lista de metadata de todas las estaciones (para API)."""
    result = []
    for name, cuenca in CUENCAS_CALI.items():
        result.append({
            "nombre": name,
            "cobertura": cuenca.cobertura,
            "cn_number": cuenca.cn,
            "area_km2": cuenca.area_km2,
            "pendiente_media": cuenca.pendiente_media,
            "pct_impermeable": cuenca.pct_impermeable,
            "longitud_cauce_km": cuenca.longitud_cauce_km,
            "tiempo_concentracion_h": round(cuenca.tiempo_concentracion_h, 2),
            "caudal_max_m3s": cuenca.caudal_max_m3s,
            "lat": cuenca.lat,
            "lon": cuenca.lon,
            "station_code": cuenca.station_code,
        })
    return result


def build_cuencas_geodataframe() -> "gpd.GeoDataFrame":
    """
    Construye un GeoDataFrame con las geometrías de las estaciones.
    Útil para visualización y análisis espacial.
    """
    if not HAS_GEO:
        raise ImportError(
            "geopandas y shapely son necesarios: pip install geopandas shapely"
        )
    records = []
    for name, c in CUENCAS_CALI.items():
        records.append({
            "estacion": name,
            "cobertura": c.cobertura,
            "cn_number": c.cn,
            "area_km2": c.area_km2,
            "pendiente_media": c.pendiente_media,
            "pct_impermeable": c.pct_impermeable,
            "geometry": Point(c.lon, c.lat),
        })
    return gpd.GeoDataFrame(records, crs="EPSG:4326")
