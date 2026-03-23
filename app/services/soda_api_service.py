import pandas as pd
import requests
import os
from datetime import datetime
import streamlit as st

@st.cache_data(show_spinner="Sincronizando datos en tiempo real con la CVC...")
def obtener_dataset_soda():
    archivo_local = "app/data/dataset_historico_calibrado.csv"
    
    # 1. Cargar histórico calibrado
    if not os.path.exists(archivo_local):
        st.error(f"No se encontró {archivo_local}")
        return pd.DataFrame()
        
    df_local = pd.read_csv(archivo_local, parse_dates=['fecha'])
    ultima_fecha_local = df_local['fecha'].max()
    
    # 2. Descargar datos NUEVOS de la API SODA
    url = "https://www.datos.gov.co/resource/avya-p282.json?$limit=5000&$order=:id"
    # Añadir filtro SODA para traer solo fechas mayores a la última local
    fecha_iso = ultima_fecha_local.strftime('%Y-%m-%dT%H:%M:%S.000')
    url += f"&$where=fecha_valor > '{fecha_iso}'"
    
    try:
        response = requests.get(url, timeout=30)
        datos_nuevos = response.json()
        
        if datos_nuevos:
            df_api = pd.DataFrame(datos_nuevos)
            
            # 3. Mapeo dinámico y limpieza (Basado en la estructura real de la API)
            df_api['fecha'] = pd.to_datetime(df_api.get('fecha_valor', df_api.get('fecha')), errors='coerce')
            df_api['caudal_m3s'] = pd.to_numeric(df_api.get('caudal_m3s', df_api.get('valor_observado')), errors='coerce')
            df_api['lluvia_mm'] = pd.to_numeric(df_api.get('lluvia_mm', 0), errors='coerce')
            
            col_est = next((c for c in ['estacion', 'nombre_estacion'] if c in df_api.columns), 'estacion')
            df_api['estacion'] = df_api[col_est].astype(str).str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
            
            df_api = df_api.dropna(subset=['fecha', 'caudal_m3s'])
            
            # Estaciones objetivo
            estaciones = ["canal canaveralejo", "canal ciudad jardin", "canal interceptor sur", 
                          "quebrada lili", "quebrada pance", "rio melendez"]
            df_api = df_api[df_api['estacion'].isin(estaciones)]
            
            # 4. Concatenar y guardar si hay datos válidos nuevos
            if not df_api.empty:
                # Opcional: asegurarse de que contenga las mismas columnas requeridas por el modelo
                # como temperatura_C y impermeabilidad_pct si no vienen en la API,
                # pero el dataset histórico ya las tiene
                for col in ['temperatura_C', 'impermeabilidad_pct']:
                    if col not in df_api.columns:
                        if col in df_local.columns:
                            # Default fill
                            df_api[col] = df_local[col].median() if not df_local.empty else 0.0

                df_actualizado = pd.concat([df_local, df_api], ignore_index=True)
                df_actualizado = df_actualizado.drop_duplicates(subset=['fecha', 'estacion'], keep='last')
                # Ordenar para mantener consistencia temporal
                df_actualizado = df_actualizado.sort_values(['estacion', 'fecha']).reset_index(drop=True)
                df_actualizado.to_csv(archivo_local, index=False)
                return df_actualizado
                
    except Exception as e:
        st.warning(f"Operando en modo offline. Error API: {e}")
        
    return df_local # Retorna el local si la API falla o no hay datos nuevos
