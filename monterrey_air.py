#!/usr/bin/env python3
"""
Sistema de Análisis de la Calidad del Aire de Monterrey
Versión: Clasificación por concentraciones (reglas) + ventanas como filtro

Requisitos mínimos:
- Python 3.11+
- streamlit, pandas, numpy, plotly, requests

Ejecuta:
    streamlit run este_archivo.py
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("monterrey_air_rules")
def _tol(v, r=0.10):  # ±10% para convertir valores puntuales en rangos tolerantes
    return (v*(1-r), v*(1+r))

# ---------------------------------------------
# Utilidades de tolerancia y coherencia NOx
# ---------------------------------------------
def _clip_tuple(t):
    lo, hi = t
    return (min(lo, hi), max(lo, hi))

def _derive_nox_from_no_no2(bounds: dict, tol=0.10):
    """
    Impone NOX ≈ NO + NO2 con tolerancia ±tol.
    Si falta NO o NO2, no modifica NOX.
    """
    if "NO" in bounds and "NO2" in bounds:
        lo = bounds["NO"][0] + bounds["NO2"][0]
        hi = bounds["NO"][1] + bounds["NO2"][1]
        bounds["NOX"] = (lo * (1 - tol), hi * (1 + tol))
    return bounds

def enforce_nox_sum_on_templates(templates, tol=0.10):
    out = []
    for t in templates:
        b = dict(t["bounds"])
        t = dict(t)
        t["bounds"] = _derive_nox_from_no_no2(b, tol=tol)
        out.append(t)
    return out

# ---------------------------------------------
# Motor de reglas: k-of-n + par prioritario (NOX, O3)
# ---------------------------------------------
from typing import Dict as _Dict, Tuple as _Tuple, List as _List

def _in_range(val, rng: _Tuple[float, float], eps=1e-9):
    lo, hi = _clip_tuple(rng)
    return (val is not None) and (lo - eps <= float(val) <= hi + eps)

def score_cluster(
    sample: _Dict[str, float],
    bounds: _Dict[str, _Tuple[float, float]],
    k_required: int = 5,
    priority_pair: _Tuple[str, str] = ("NOX", "O3"),
):
    matched = []
    for k, rng in bounds.items():
        if k in sample and _in_range(sample[k], rng):
            matched.append(k)

    priority_ok = all(
        (p in bounds) and (p in sample) and _in_range(sample[p], bounds[p])
        for p in priority_pair
    )
    match = priority_ok and (len(matched) >= k_required)
    return match, len(matched), matched

def classify_sample(
    sample: _Dict[str, float],
    templates,
    k_required: int = 5,
    priority_pair: _Tuple[str, str] = ("NOX", "O3"),
):
    """
    Devuelve el mejor cluster por score, respetando el par prioritario.
    """
    best = None
    for t in templates:
        m, s, keys = score_cluster(sample, t["bounds"], k_required, priority_pair)
        rec = {"id": t["id"], "name": t["name"], "match": m, "score": s, "matched_keys": keys}
        if (best is None) or (s > best["score"]) or (s == best["score"] and m and not best["match"]):
            best = rec
    return best





# -----------------------------------------------------------------------------
# I/O DE DATOS
# -----------------------------------------------------------------------------
def render_executive_summary(df: pd.DataFrame) -> None:
    st.title("Resumen Ejecutivo: Panorama de la Calidad del Aire en Monterrey")
    st.markdown("""
    Bienvenida/o. Este resumen destaca tendencias clave de la calidad del aire en Monterrey. 
    Niveles altos de PM afectan vías respiratorias; el ozono (O3) suele elevarse al mediodía soleado.
    Usa los gráficos interactivos para explorar.
    """)

    # Métricas clave
    try:
        avg_pm25 = float(df['PM2.5'].mean())
    except Exception:
        avg_pm25 = float('nan')

    try:
        idx = df['O3'].fillna(-np.inf).idxmax()
        max_o3 = float(df.loc[idx, 'O3'])
        max_date = pd.to_datetime(df.loc[idx, 'date'])
        max_station = str(df.loc[idx, 'station']) if 'station' in df.columns else "N/D"
        max_label = f"{max_station}: {max_date}"
    except Exception:
        max_o3, max_label = float('nan'), "N/D"

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "PM2.5 promedio (partículas finas)",
        f"{avg_pm25:.1f} µg/m³" if np.isfinite(avg_pm25) else "N/D",
        "Moderado" if (np.isfinite(avg_pm25) and avg_pm25 < 25) else "No saludable"
    )
    col2.metric("Pico histórico de Ozono (O3)", f"{max_o3:.0f} (ppd)" if np.isfinite(max_o3) else "N/D", max_label)
    col3.metric("Peor ventana horaria (típica)", "Pico vespertino", "Asociado a NOX/CO por tráfico")

    # Serie temporal por contaminante/estación
    cols_exist = [c for c in ['date', 'station', 'O3', 'NOX', 'CO'] if c in df.columns]
    if set(['date','station']).issubset(cols_exist):
        plot_cols = [c for c in ['O3','NOX','CO'] if c in df.columns]
        if plot_cols:
            df_long = (df[['date','station', *plot_cols]]
                       .melt(id_vars=['date','station'], var_name='pollutant', value_name='value')
                       .sort_values(['pollutant','station','date']))
            fig_ts = px.line(
                df_long, x='date', y='value',
                color='pollutant',
                line_group='station',
                hover_data=['station'],
                title="Tendencias por estación y contaminante"
            )
            fig_ts.update_traces(connectgaps=False)
            fig_ts.update_layout(hovermode="x unified")
            st.plotly_chart(fig_ts, use_container_width=True)

    # Pastel (dummy)
    sources = {'Tráfico (NOX/CO)': 45, 'Polvo/Industria (PM)': 30, 'Formación de ozono': 15, 'Otras': 10}
    fig_pie = px.pie(names=list(sources.keys()), values=list(sources.values()),
                     title="Fuentes estimadas de contaminación (referencial)")
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("**Insight:** La calidad del aire empeora en horas pico por el tráfico. Desplázate para ver más.")

def load_excel_frames(path_pattern: str = "Bases_Datos/f24_clean.xlsx") -> pd.DataFrame:
    """
    Cargar archivo de Excel con múltiples hojas (estaciones) y combinarlos.
    - Filtra fines de semana.
    """
    try:
        excel_data = pd.read_excel(path_pattern, sheet_name=None)
        frames = []

        for station, df in excel_data.items():
            df = df.copy()
            df['station'] = station

            if 'date' in df.columns:
                if pd.api.types.is_numeric_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'], unit='s', errors='coerce')
                else:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')

            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
        combined = combined.dropna(subset=['date'])

        before = len(combined)
        combined = combined[combined['date'].dt.dayofweek < 5].copy()
        after = len(combined)

        logger.info(f"Fines de semana excluidos. Registros antes: {before}, después: {after}")
        logger.info(f"Se cargaron {len(excel_data)} estaciones con {after} registros en total")
        return combined

    except Exception as e:
        logger.error(f"Error al cargar Excel '{path_pattern}': {e}")
        return pd.DataFrame()

def make_temporal_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la columna 'time_window' por hora del día."""
    d = df.copy()
    if 'date' in d.columns:
        d['hour'] = pd.to_datetime(d['date']).dt.hour
        conditions = [
            (d['hour'] >= 6) & (d['hour'] < 10),   # morning_peak
            (d['hour'] >= 10) & (d['hour'] < 16),  # midday
            (d['hour'] >= 16) & (d['hour'] < 20),  # evening_peak
            (d['hour'] >= 20) | (d['hour'] < 6)    # night
        ]
        choices = ['morning_peak', 'midday', 'evening_peak', 'night']
        d['time_window'] = np.select(conditions, choices, default='night')
    return d

def add_peak_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Señales de pico (z-score dentro de estación+ventana).
    """
    d = df.copy()
    needed = ['station','time_window','NOX','O3','PM10','PM2.5']
    for c in needed:
        if c not in d.columns:
            d[c] = np.nan

    def zscore(g, col):
        mu = g[col].mean()
        sd = g[col].std(ddof=0) or 1.0
        return (g[col] - mu) / sd

    d['z_NOX'] = d.groupby(['station','time_window'], dropna=False).apply(lambda g: zscore(g, 'NOX')).reset_index(level=[0,1], drop=True)
    d['z_O3']  = d.groupby(['station','time_window'], dropna=False).apply(lambda g: zscore(g, 'O3')).reset_index(level=[0,1], drop=True)
    d['z_PM']  = d.groupby(['station','time_window'], dropna=False).apply(lambda g: zscore(g, 'PM2.5')).reset_index(level=[0,1], drop=True)

    TH = 1.0  # ~p84
    d['peak_morning_nox'] = ((d['time_window']=='morning_peak') & (d['z_NOX']>=TH)).astype(int)
    d['peak_evening_nox'] = ((d['time_window']=='evening_peak') & (d['z_NOX']>=TH)).astype(int)
    d['peak_midday_o3']   = ((d['time_window']=='midday')       & (d['z_O3'] >=TH)).astype(int)
    d['peak_pm']          = (d['z_PM']>=TH).astype(int)

    def _ptype(row):
        flags = []
        if row['peak_morning_nox']: flags.append('Tráfico AM')
        if row['peak_evening_nox']: flags.append('Tráfico PM')
        if row['peak_midday_o3']:   flags.append('Ozono mediodía')
        if row['peak_pm']:          flags.append('Partículas')
        if not flags: return 'Sin pico / Mixto'
        return " + ".join(flags[:2])
    d['peak_type'] = d.apply(_ptype, axis=1)
    return d

# -----------------------------------------------------------------------------
# CLASIFICACIÓN POR CONCENTRACIONES (REGLAS)
# -----------------------------------------------------------------------------
def _approx_eq(a: float, b: float, tol: float = 0.1) -> bool:
    """Comparación aproximada con tolerancia."""
    return abs(a - b) <= tol

def classify_concentration(row: Dict[str, float], window: str) -> Tuple[int, str]:
    """
    Reglas basadas en patrones típicos de contaminación urbana.
    Usamos rangos más flexibles y patrones característicos.
    """
    nox  = float(row.get('NOX',  0.0))
    no   = float(row.get('NO',   0.0))
    no2  = float(row.get('NO2',  0.0))
    pm10 = float(row.get('PM10', 0.0))
    pm25 = float(row.get('PM2.5',0.0))
    co   = float(row.get('CO',   0.0))
    o3   = float(row.get('O3',   0.0))
    so2  = float(row.get('SO2',  0.0))

    # 1) Tráfico intenso - Alto NOX, CO, PM
    if (nox > 0.15 and co > 1.0 and no > 0.05 and no2 > 0.06 and pm10 > 80):
        return 0, "Tráfico Intenso - Alto NOX, CO y PM"

    # 2) Formación de Ozono - Alto O3, bajo NOX
    elif (o3 > 0.07 and nox < 0.08 and pm10 < 60):
        return 1, "Formación Fotoquímica de Ozono"

    # 3) Industrial - Alto SO2, PM moderado
    elif (so2 > 0.01 and pm10 > 70 and nox > 0.1):
        return 2, "Fuentes Industriales - Alto SO2"

    # 4) Mixto Urbano - Valores moderados en todos los parámetros
    elif (0.08 <= nox <= 0.15 and 0.7 <= co <= 1.2 and 50 <= pm10 <= 80):
        return 3, "Mixto Urbano - Valores Moderados"

    # 5) Fondo Limpio - Bajos valores en todos los parámetros
    elif (nox < 0.05 and co < 0.5 and pm10 < 40 and so2 < 0.005):
        return 4, "Fondo Limpio - Bajas Concentraciones"

    # 6) Alta Contaminación Particulada - Muy alto PM
    elif (pm10 > 100 or pm25 > 50):
        return 5, "Alta Contaminación Particulada"

    return -1, "No clasificado - Patrón Atípico"

def simulate_scenario(window: str, values: Dict[str, float], features: list) -> Dict:
    """
    Simula el escenario y clasifica usando las reglas actualizadas.
    Incluye recomendaciones basadas en evidencia científica con referencias APA.
    """
    cluster, class_name = classify_concentration(values, window)
    
    result = {
        'cluster': cluster,
        'window': window,
        'class_name': class_name,
        'approximate': False,
        'approx_score': 0.0
    }
    
    # Interpretaciones basadas en evidencia científica con referencias APA
    interpretations = {
        0: (
            "**Recomendaciones basadas en evidencia:**\n\n"
            "• Implementar horarios escalonados de trabajo para reducir congestión vehicular en horas pico (Molina et al., 2007)\n"
            "• Promover transporte público eléctrico y sistemas de bicicletas compartidas (ITF, 2017)\n"
            "• Establecer zonas de bajas emisiones en áreas urbanas críticas (WHO, 2021)\n\n"
            "**Referencias:**\n"
            "Molina, L. T., Velasco, E., Retama, A., & Zavala, M. (2007). *JAWMA*, 57(12), 1465–1475. https://doi.org/10.3155/1047-3289.57.12.1465\n"
            "International Transport Forum. (2017). *Air pollution mitigation strategy for Mexico City*. OECD Publishing.\n"
            "World Health Organization. (2021). *WHO global air quality guidelines*. WHO Press."
        ),
        1: (
            "**Recomendaciones basadas en evidencia:**\n\n"
            "• Reducir emisiones de precursores de ozono (NOx y COV) en horas matutinas (EPA, 2023)\n\n"
            "• Limitar actividades al aire libre entre 12:00-16:00 horas durante episodios de ozono (WHO, 2021)\n\n"
            "• Implementar programas de mantenimiento vehicular para reducir emisiones de precursores (Molina et al., 2007)\n\n"
            "**Referencias:**\n"
            "U.S. Environmental Protection Agency. (2023). *Ground-level ozone basics*. EPA.gov\\n"
            "World Health Organization. (2021). *WHO global air quality guidelines*. WHO Press.\n\n"
            "Molina, L. T., et al. (2007). *Journal of the Air & Waste Management Association*, 57(12), 1465-1475."
        ),
        2: (
            "**Recomendaciones basadas en evidencia:**\n\n"
            "• Mejorar sistemas de filtración y control de emisiones industriales (EPA, 2023)\n\n"
            "• Implementar monitoreo continuo de SO2 en corredores industriales (WHO, 2021)\n\n"
            "• Promover transición a combustibles limpios en procesos industriales (ITF, 2017)\n\n"
            "**Referencias:**\n"
            "U.S. Environmental Protection Agency. (2023). *Sulfur dioxide (SO2) pollution*. EPA.gov\n\n"
            "World Health Organization. (2021). *WHO global air quality guidelines*. WHO Press.\n\n"
            "International Transport Forum. (2017). *Reducing sulphur emissions from ships*. OECD Publishing."
        ),
        3: (
            "**Recomendaciones basadas en evidencia:**\n\n"
            "• Desarrollar plan integral de movilidad urbana sostenible (ITF, 2017)\n\n"
            "• Incrementar áreas verdes urbanas para mejorar calidad del aire (WHO, 2021)\n\n"
            "• Implementar regulaciones integradas para múltiples fuentes de contaminación (EPA, 2023)\n\n"
            "**Referencias:**\n"
            "International Transport Forum. (2017). *Transition to Shared Mobility*. OECD Publishing.\n\n"
            "World Health Organization. (2021). *Urban green spaces and health*. WHO Regional Office for Europe.\n\n"
            "U.S. Environmental Protection Agency. (2023). *Integrated Science Assessment for Particulate Matter*. EPA/600/R-23/028."
        ),
        4: (
            "**Recomendaciones basadas en evidencia:**\n\n"
            "• Mantener políticas actuales de control de calidad del aire (WHO, 2021)\n\n"
            "• Continuar monitoreo y vigilancia epidemiológica (EPA, 2023)\n\n"
            "• Fortalecer programas de educación ambiental comunitaria (Molina et al., 2007)\n\n"
            "**Referencias:**\n"
            "World Health Organization. (2021). *WHO global air quality guidelines*. WHO Press.\\n"
            "U.S. Environmental Protection Agency. (2023). *Air Quality Monitoring and Assessment*. EPA.gov\n\n"
            "Molina, L. T., et al. (2007). *Journal of the Air & Waste Management Association*, 57(12), 1465-1475."
        ),
        5: (
            "**Recomendaciones basadas en evidencia:**\n\n"
            "• Activar protocolos de alerta por material particulado (WHO, 2021)\n\n"
            "• Recomendar uso de mascarillas N95 en exteriores (EPA, 2023)\n\n"
            "• Limitar actividades físicas al aire libre durante episodios críticos (Molina et al., 2007)\n\n"
            "**Referencias:**\n"
            "World Health Organization. (2021). *Health effects of particulate matter*. WHO Press.\n\n"
            "U.S. Environmental Protection Agency. (2023). *Particle Pollution and Health*. EPA.gov\n\n"
            "Molina, L. T., et al. (2007). *Journal of the Air & Waste Management Association*, 57(12), 1465-1475."
        ),
        -1: (
            "**Recomendaciones basadas en evidencia:**\n\n"
            "• Intensificar monitoreo y análisis de fuentes atípicas (EPA, 2023)\n\n"
            "• Implementar vigilancia especializada para patrones inusuales (WHO, 2021)\n\n"
            "• Desarrollar estudios específicos para identificar fuentes emergentes (Molina et al., 2007)\n\n"
            "**Referencias:**\n"
            "U.S. Environmental Protection Agency. (2023). *Air Quality Monitoring Innovations*. EPA.gov\n\n"
            "World Health Organization. (2021). *Emerging issues in air quality*. WHO Press.\n\n"
            "Molina, L. T., et al. (2007). *Journal of the Air & Waste Management Association*, 57(12), 1465-1475."
        )
    }
    
    result['interpretation'] = interpretations.get(cluster, "Sin recomendación específica basada en evidencia disponible.")
    return result
def fetch_live_data() -> Dict:
    """
    Simula la obtención de datos en tiempo real.
    """
    return {
        'CO': 0.6, 'NO': 0.03, 'NO2': 0.04, 'NOX': 0.07,
        'O3': 0.05, 'PM10': 45.0, 'PM2.5': 20.0, 'SO2': 0.005
    }

def render_simulator() -> None:
    """
    Simulador por reglas de concentración con sliders reactivos.
    """
    st.header("Simulador de Calidad del Aire")

    # --- Configuración base ---
    windows = ['mañana', 'mediodía', 'tarde', 'noche']
    features = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'SO2']
    
    # Rangos realistas en mg/m³ para gases y μg/m³ para partículas
    ranges = {
        'CO':   (0.0, 2.0, 0.01),
        'NO':   (0.0, 0.2, 0.001),
        'NO2':  (0.0, 0.2, 0.001),
        'NOX':  (0.0, 0.3, 0.001),
        'O3':   (0.0, 0.15, 0.001),
        'PM10': (0.0, 150.0, 1.0),
        'PM2.5':(0.0, 75.0, 0.5),
        'SO2':  (0.0, 0.05, 0.0001),
    }

    # --- Estado inicial ---
    if 'sim_defaults_by_window' not in st.session_state:
        st.session_state.sim_defaults_by_window = {}
        live = fetch_live_data()
        for w in windows:
            st.session_state.sim_defaults_by_window[w] = live.copy()

    selected_window = st.selectbox("Selecciona periodo del día:", windows)

    # --- Inicializar valores ---
    if 'sim_values_by_window' not in st.session_state:
        st.session_state.sim_values_by_window = {}
    if selected_window not in st.session_state.sim_values_by_window:
        st.session_state.sim_values_by_window[selected_window] = st.session_state.sim_defaults_by_window[selected_window].copy()

    # --- Sliders ---
    st.subheader("Ajusta niveles de contaminantes")
    col1, col2 = st.columns(2)
    
    current_values = {}
    for i, f in enumerate(features):
        mn, mx, step = ranges[f]
        key = f"slider_{selected_window}_{f}"
        
        with (col1 if i < 4 else col2):
            current_val = st.session_state.sim_values_by_window[selected_window].get(f, ranges[f][0])
            
            new_val = st.slider(
                label=f"{f} ({'mg/m³' if f in ['CO','NO','NO2','NOX','O3','SO2'] else 'μg/m³'})",
                min_value=float(mn),
                max_value=float(mx),
                step=float(step),
                value=float(current_val),
                key=key
            )
            
            current_values[f] = new_val
            st.session_state.sim_values_by_window[selected_window][f] = new_val

    # --- Botones ---
    col1, col2, col3 = st.columns(3)
    simular = col1.button("🚀 Simular", type="primary")
    aleatorio = col2.button("🎲 Aleatorio")
    reset = col3.button("↩ Restablecer")

    # --- Acciones de botones ---
    if aleatorio:
        new_vals = {}
        for f in features:
            mn, mx, step = ranges[f]
            new_vals[f] = round(np.random.uniform(mn, mx), 4)
        st.session_state.sim_values_by_window[selected_window] = new_vals
        st.rerun()

    if reset:
        st.session_state.sim_values_by_window[selected_window] = st.session_state.sim_defaults_by_window[selected_window].copy()
        st.rerun()

    # --- Simulación ---
    if simular:
        with st.spinner("Analizando patrón de contaminación..."):
            result = simulate_scenario(selected_window, current_values, features)
        
        st.success("¡Análisis completado!")
        
        # Mostrar resultados
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Clasificación", result['class_name'])
            st.metric("Ventana temporal", selected_window.capitalize())
        
        with col2:
            if result['cluster'] != -1:
                st.success("✅ Patrón reconocido")
            else:
                st.warning("⚠️ Patrón atípico")
        
        # Recomendación
        st.info(f"**Recomendación:**\n\n{result['interpretation']}")
        
        # Gráfico
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=features,
            y=[current_values[f] for f in features],
            name='Tus valores',
            marker_color='lightblue'
        ))
        
        # Añadir línea de referencia (valores típicos)
        typical = fetch_live_data()
        fig.add_trace(go.Scatter(
            x=features,
            y=[typical[f] for f in features],
            name='Valores típicos',
            mode='lines+markers',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Perfil de Contaminantes vs Valores Típicos",
            xaxis_title="Contaminante",
            yaxis_title="Concentración",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)



# -------------------------------------------------------------------
# PLANTILLAS DE CLÚSTERES (mismos rangos que tus reglas; los "==" se relajan a ±10%)
# -------------------------------------------------------------------
def _tol(v, r=0.10):  # ±10% para convertir valores puntuales en rangos tolerantes
    return (v*(1-r), v*(1+r))

# -------------------------------------------------------------------
# PLANTILLAS DE CLÚSTERES (NOX derivado de NO+NO2; ±10% de tolerancia)
# -------------------------------------------------------------------
CLUSTER_TEMPLATES = [
    {
        "id": 0,
        "name": "Clúster de Emisiones por Tráfico Intenso",
        "bounds": {
            "NO":   (13, 30),
            "NO2":  (26, 30),
            "PM10": (61, 95),
            "PM2.5": (18, 31),
            "CO":   (1.5, 2.0),
            "O3":   (6, 22),
            # "SO2" opcional según datos
        }
    },
    {
        "id": 1,
        "name": "Clúster de Formación Fotoquímica e Industrial",
        "bounds": {
            "NO":    _tol(4.0),
            "NO2":   _tol(7.0),
            "PM10":  _tol(41.0),
            "PM2.5": _tol(11.0),
            "CO":    _tol(1.3),
            "O3":    (19, 53),
            "SO2":   (4, 8),   # añade trazo industrial
        }
    },
    {
        "id": 2,
        "name": "Clúster de Fuentes Mixtas Urbanas Estándar",
        "bounds": {
            # NOX derivará ≈ 18–46 antes de tolerancia
            "NO":    (3, 23),
            "NO2":   (15, 23),
            "PM10":  (35, 71),
            "PM2.5": (8, 22),
            "CO":    (0.9, 1.6),
            "O3":    (9, 37),
            "SO2":   (3, 4),
        }
    },
    {
        "id": 3,
        "name": "Clúster de Fondo Urbano Bajo (CO y PM bajos)",
        "bounds": {
            # NOX derivará ≈ 11–22 antes de tolerancia
            "NO":    (4, 8),
            "NO2":   (7, 14),
            "PM10":  (34, 41),
            "PM2.5": (9, 10),
            "CO":    (0.6, 0.7),
            "O3":    (11, 36),
            "SO2":   _tol(2.6),
        }
    },
    {
        "id": 4,
        "name": "Clúster de Baja Contaminación (O3 medio, NOx bajo)",
        "bounds": {
            "NO":    (2, 3),
            "NO2":   (6, 7),
            "PM10":  (40, 58),
            "PM2.5": (10, 16),
            "CO":    (1.0, 1.5),
            "O3":    (23, 45),
            "SO2":   (3, 4),
        }
    }
]

# Enforce: recalcula NOX = NO + NO2 con ±10%
CLUSTER_TEMPLATES = enforce_nox_sum_on_templates(CLUSTER_TEMPLATES, tol=0.10)


# -------------------------------------------------------------------
# UTILIDADES PARA APROXIMACIÓN Y CENTROIDES
# -------------------------------------------------------------------
def _cluster_def_by_id(cid: int):
    for c in CLUSTER_TEMPLATES:
        if c["id"] == cid:
            return c
    return None

def _range_distance(x: float, lo: float, hi: float) -> float:
    """0 si está dentro; si está fuera, distancia normalizada al ancho del rango."""
    width = max(hi - lo, 1e-9)
    if x < lo:
        return (lo - x) / width
    if x > hi:
        return (x - hi) / width
    return 0.0

def _approximate_best_cluster(values: dict):
    """
    Devuelve (cid, cname, score, contribs) para el clúster más cercano.
    score = promedio de distancias normalizadas (0 perfecto; ↓ mejor).
    contribs = {polutante: distancia_normalizada}
    """
    best = (None, None, float('inf'), {})
    for c in CLUSTER_TEMPLATES:
        dists, contrib = [], {}
        for pol, (lo, hi) in c["bounds"].items():
            if pol not in values or not np.isfinite(values[pol]):
                continue
            di = _range_distance(float(values[pol]), float(lo), float(hi))
            dists.append(di)
            contrib[pol] = di
        if not dists:
            continue
        score = float(np.mean(dists))
        if score < best[2]:
            best = (c["id"], c["name"], score, contrib)
    return best

def _centroid_from_bounds(bounds: dict) -> dict:
    """Centroide como punto medio de cada rango disponible."""
    center = {}
    for k, (lo, hi) in bounds.items():
        lo = float(lo); hi = float(hi)
        center[k] = (lo + hi) / 2.0
    return center

# 🔧 FIX: reemplaza tu función `generate_interpretation(...)` por ESTA versión
# Evita el TypeError al intentar float(Timestamp) u otros campos no numéricos.
# Solo usa contaminantes numéricos (CO, NO, NO2, NOX, O3, PM10, PM2.5, SO2) para
# calcular desviaciones, drivers y recomendaciones.

from math import isfinite

POLLUTANTS = ['CO','NO','NO2','NOX','O3','PM10','PM2.5','SO2']

def _to_float_or_none(x):
    try:
        v = float(x)
        return v if isfinite(v) else None
    except Exception:
        return None

def generate_interpretation(window: str,
                            sample: dict,
                            centroid: dict,
                            class_name: str = None) -> str:
    """
    Narrativa con 3 partes:
    1) Etiqueta del patrón (y clasificación si existe)
    2) Por qué lo creemos (desviaciones vs. referencia/centroide)
    3) Qué hacer ahora (acciones accionables, sensibles a la ventana)
    (Seguro ante columnas no numéricas como 'date', 'station', etc.)
    """
    # Nombres "bonitos"
    nice = {
        'CO':   'monóxido de carbono (escape vehicular) (EPA, 2023)',
        'NO':   'óxido nítrico (emisión fresca de tráfico) (EPA, 2023)',
        'NO2':  'dióxido de nitrógeno (tráfico/combustión) (EPA, 2023)',
        'NOX':  'NOx (mezcla de tráfico) (EPA, 2023)',
        'O3':   'ozono (fotoquímica sol + precursores) (WHO, 2021)',
        'PM10': 'PM10 (polvo grueso) (WHO, 2021)',
        'PM2.5':'PM2.5 (partículas finas) (WHO, 2021)',
        'SO2':  'dióxido de azufre (industrial/combustibles) (EPA, 2023)'
    }

    # Filtrar a contaminantes y a valores realmente numéricos
    s_num = {k: _to_float_or_none(sample.get(k))   for k in POLLUTANTS}
    c_num = {k: _to_float_or_none(centroid.get(k)) for k in POLLUTANTS}
    keys  = [k for k in POLLUTANTS if (s_num.get(k) is not None and c_num.get(k) is not None)]
    if not keys:  # fallback: si no hay centroide válido, usa lo que haya en sample
        keys = [k for k in POLLUTANTS if s_num.get(k) is not None]

    # Desviaciones relativas
    deltas = {}
    for k in keys:
        c = c_num.get(k)
        s = s_num.get(k, c)
        if c is None or c == 0:
            deltas[k] = 0.0
        else:
            deltas[k] = (s - c) / abs(c)

    # Top drivers (2–3)
    drivers = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

    # Etiqueta por ventana + refuerzo por class_name
    label = "condiciones típicas"
    if window in ("morning_peak", "evening_peak"):
        if deltas.get('NOX', 0) > 0.20 or deltas.get('CO', 0) > 0.20:
            label = "pico impulsado por emisiones vehiculares"
    if window == "midday":
        if deltas.get('O3', 0) > 0.20:
            label = "acumulación fotoquímica de ozono"
        if deltas.get('PM10', 0) > 0.25 and deltas.get('O3', 0) <= 0.20:
            label = "día dominado por partículas suspendidas"
    if deltas.get('PM2.5', 0) > 0.25 and deltas.get('PM10', 0) > 0.15:
        label = "carga elevada de material particulado"
    if keys and all(abs(deltas[k]) < 0.10 for k in keys):
        label = "niveles cercanos a lo normal"

    if class_name:
        if "Tráfico" in class_name:
            label = "pico impulsado por emisiones vehiculares"
        elif "Fotoquímica" in class_name or "Ozono" in class_name:
            label = "acumulación fotoquímica de ozono"
        elif "Fondo Urbano" in class_name:
            label = "fondo urbano limpio/alta dispersión"
        elif "Baja Contaminación" in class_name:
            label = "niveles bajos con ozono medio (transición)"

    # Narrativa "por qué"
    def pct(x: float) -> str:
        return f"{x*100:.0f}%"
    if drivers:
        why_bits = []
        for k, v in drivers:
            direction = "más alto" if v > 0 else "más bajo"
            why_bits.append(f"{nice.get(k, k)} está {direction} que lo usual por ~{pct(abs(v))}")
        why_txt = "; ".join(why_bits) + "."
    else:
        why_txt = "Los indicadores disponibles no muestran desviaciones significativas."

    # Acciones
    actions = []
    if "emisiones vehiculares" in label:
        if window == "morning_peak":
            actions += [
                "Escalonar horarios de entrada 30–60 min. (Molina et al., 2007)",
                "Priorizar carriles bus y semaforización adaptativa. (ITF, 2017)",
                "Desincentivar viajes cortos en auto entre 7–9 h. (Molina et al., 2007)"
            ]
        else:
            actions += [
                "Mover reparto/logística fuera de 17–20 h. (ITF, 2017)",
                "Zonas de bajas emisiones en arterias congestionadas. (Molina et al., 2007)"
            ]
    if "ozono" in label:
        actions += [
            "Reducir uso de solventes/pinturas al mediodía; programar mañana/tarde. (WHO, 2021)",
            "Promover TP/teletrabajo en días soleados 12–16 h. (WHO, 2021)"
        ]
    if "partículas" in label or "material particulado" in label:
        actions += [
            "Refuerzo de barrido/aspersión y control de polvo en obras. (EPA, 2023)",
            "Mascarilla a grupos sensibles; limitar deporte al aire libre. (WHO, 2021)"
        ]
    if not actions:
        actions = ["Mantener controles actuales; los niveles siguen patrón habitual. (EPA, 2023)"]

    narrative = (
        (f"**Clasificación:** {class_name}\n\n" if class_name else "") +
        f"Patrón: **{label}** durante **{window.replace('_',' ')}**.\n\n"
    )

    references = (
        "\n\n**Referencias:**\n"
        "- U.S. Environmental Protection Agency. (2023). *Criteria air pollutants*. https://www.epa.gov/criteria-air-pollutants\n"
        "- World Health Organization. (2021). *WHO global air quality guidelines*. https://iris.who.int/bitstream/handle/10665/345329/9789240034228-eng.pdf\n"
        "- Molina, L. T., Velasco, E., Retama, A., & Zavala, M. (2007). *JAWMA*, 57(12), 1465–1475. https://doi.org/10.3155/1047-3289.57.12.1465\n"
        "- International Transport Forum. (2017). *Air pollution mitigation strategy for Mexico City*. https://www.itf-oecd.org/sites/default/files/docs/air-pollution-mitigation-strategy-mexico-city.pdf"
    )
    return narrative + references


# -------------------------------------------------------------------
# NUEVO simulate_scenario: usa reglas + aproximación + centroide por rangos
# -------------------------------------------------------------------

# -----------------------------------------------------------------------------
# UI: EDA
# -----------------------------------------------------------------------------
def render_eda(df: pd.DataFrame) -> None:
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("Explora distribuciones, tendencias y conexiones entre contaminantes.")

    # Selección de contaminante
    candidates = [c for c in df.columns if c not in ['date','station','time_window','hour','peak_type','z_NOX','z_O3','z_PM']]
    if not candidates:
        st.info("No hay columnas numéricas para EDA.")
        return
    pollutant = st.selectbox("Selecciona contaminante a explorar:", candidates)

    # Columna 'hour'
    if 'hour' not in df.columns:
        df = df.copy()
        df['hour'] = pd.to_datetime(df['date']).dt.hour

    hour_order = list(range(24))

    # Promedio por hora y ventana
    fig_hour_hist = px.histogram(
        df, x='hour', y=pollutant, color='time_window',
        histfunc='avg', barmode='group',
        category_orders={'hour': hour_order},
        title=f"{pollutant} por hora del día (promedio por ventana)",
        labels={'hour': 'Hora del día (0–23)', pollutant: f"Nivel medio de {pollutant}", 'time_window': 'Ventana'}
    )
    fig_hour_hist.update_traces(hovertemplate="Hora: %{x}:00<br>Nivel medio: %{y:.3f}")
    st.plotly_chart(fig_hour_hist, use_container_width=True)

    # Caja por hora
    fig_hour_box = px.box(
        df, x='hour', y=pollutant, color='time_window',
        category_orders={'hour': hour_order},
        points='outliers',
        title=f"Distribución de {pollutant} por hora (todas las observaciones)",
        labels={'hour': 'Hora (0–23)', pollutant: f"Nivel de {pollutant}"}
    )
    st.plotly_chart(fig_hour_box, use_container_width=True)

    # Correlaciones
    conts = [c for c in ['CO','NO','NO2','NOX','O3','PM10','PM2.5','SO2'] if c in df.columns]
    if len(conts) >= 2:
        corr = df[conts].corr()
        fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                             colorscale='RdYlGn', zmin=-1, zmax=1))
        fig_corr.update_layout(title="Conexiones entre contaminantes (Rojo/Verde = vínculo fuerte)")
        st.plotly_chart(fig_corr, use_container_width=True)

    # Serie temporal PM
    if set(['PM10','PM2.5','date']).issubset(df.columns):
        min_date = pd.to_datetime(df['date'].min()).to_pydatetime()
        max_date = pd.to_datetime(df['date'].max()).to_pydatetime()
        date_range = st.slider("Rango de fechas:", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        filtered_df = df[(df['date'] >= pd.Timestamp(date_range[0])) & (df['date'] <= pd.Timestamp(date_range[1]))]

        filtered_df_ts = (filtered_df.sort_values('date')
                          .groupby('date', as_index=False)[['PM10','PM2.5']].mean())
        fig_ts = px.line(filtered_df_ts, x='date', y=['PM10','PM2.5'],
                         title="Tendencias de PM (promedio sobre estaciones)")
        fig_ts.update_traces(connectgaps=False)
        fig_ts.update_layout(hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)

    # Comparación por estación
    if set(['station','NOX']).issubset(df.columns):
        fig_station = px.box(df, x='station', y='NOX', color='time_window',
                             title="NOX por estación (compara ubicaciones)")
        st.plotly_chart(fig_station, use_container_width=True)

# -----------------------------------------------------------------------------
# UI: ANÁLISIS TEMPORAL (por reglas)
# -----------------------------------------------------------------------------
def render_temporal_analysis(df: pd.DataFrame, windows: List[str]) -> None:
    st.title("Análisis temporal: Clasificación por concentraciones (reglas)")
    st.markdown("La ventana horaria actúa como **filtro**; la categoría se asigna por **reglas**.")

    selected_window = st.selectbox("Selecciona ventana horaria:", windows, index=0)
    wdf = df[df['time_window'] == selected_window].copy()

    if wdf.empty:
        st.warning("No hay datos para esta ventana.")
        return

    # Clasificar filas y recolectar ejemplos por clase
    by_class: Dict[int, Dict[str, Any]] = {}
    for _, row in wdf.iterrows():
        class_id, class_name = classify_concentration(row.to_dict(), selected_window)
        if class_id == -1:
            continue
        if class_id not in by_class:
            by_class[class_id] = {'name': class_name, 'rows': []}
        by_class[class_id]['rows'].append(row)

    if not by_class:
        st.info("No se detectaron patrones que cumplan reglas en esta ventana.")
        return

    # Frecuencias
    counts_df = pd.DataFrame([
        {'class_id': cid, 'class_name': info['name'], 'count': len(info['rows'])}
        for cid, info in by_class.items()
    ]).sort_values('class_id')

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Frecuencia por categoría")
        fig_bar = px.bar(counts_df, x='class_name', y='count', text='count',
                         labels={'class_name':'Categoría','count':'Conteo'},
                         title="Frecuencia de categorías")
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(xaxis_tickangle=-15)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        # Medianas por categoría
        pollutants = [c for c in ['CO','NO','NO2','NOX','O3','PM10','PM2.5','SO2'] if c in wdf.columns]
        med_list = []
        for cid, info in by_class.items():
            med_list.append(pd.Series(pd.DataFrame(info['rows'])[pollutants].median(), name=info['name']))
        med = pd.DataFrame(med_list)
        fig_heat = px.imshow(
            med.values, x=pollutants, y=med.index,
            color_continuous_scale='RdYlGn_r',
            labels={'x':'Contaminantes','y':'Categoría','color':'Mediana (unidades originales)'},
            title="Perfiles típicos por categoría (medianas)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("¿Qué significan estas clasificaciones?")
    for cid in sorted(by_class.keys()):
        cname = by_class[cid]['name']
        sample_row = by_class[cid]['rows'][0].to_dict()
        interp = generate_interpretation(selected_window, sample_row, sample_row, cname)
        st.markdown(f"**Clasificación {cid} — {cname}:**\n\n{interp}\n")

# -----------------------------------------------------------------------------
# === Sustituye COMPLETAMENTE tu función `render_simulator` por esta versión ===
# Fix: evita el error "widget fue creado con valor por defecto y también seteado vía Session State".
# Clave: si el key del slider YA está en st.session_state, NO pases `value=` al crear el widget.




# -----------------------------------------------------------------------------
# MAIN (Streamlit)
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Tablero de Calidad del Aire de Monterrey",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Sistema de Análisis de la Calidad del Aire de Monterrey")
    st.markdown("**Monitoreo en tiempo real, análisis y recomendaciones basadas en reglas por concentraciones.**")

    # Navegación
    st.sidebar.title("Navegación")
    section = st.sidebar.radio(
        "Selecciona sección:",
        ["Resumen Ejecutivo", "Análisis Exploratorio de los Datos", "Análisis Temporal", "Simulador"]
    )

    # Filtros globales
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros globales")

    @st.cache_data(show_spinner=False)
    def _load_data_cached() -> pd.DataFrame:
        return load_excel_frames()

    with st.spinner("Cargando datos..."):
        df = _load_data_cached()

    if df.empty:
        st.error("❌ No se cargaron datos. Revisa el archivo en Bases_Datos/f24_clean.xlsx")
        return

    # Preparación
    df = make_temporal_windows(df)
    df = add_peak_features(df)

    # Filtros
    stations_all = sorted(df['station'].dropna().unique()) if 'station' in df.columns else []
    stations = st.sidebar.multiselect(
        "Estaciones:", stations_all,
        default=stations_all[:3] if stations_all else []
    )

    min_dt = pd.to_datetime(df['date'].min())
    max_dt = pd.to_datetime(df['date'].max())
    date_range = st.sidebar.date_input(
        "Rango de fechas:",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date()
    )

    # Aplicar filtros
    fdf = df.copy()
    if stations:
        fdf = fdf[fdf['station'].isin(stations)]
    fdf = fdf[(fdf['date'] >= pd.Timestamp(date_range[0])) & (fdf['date'] <= pd.Timestamp(date_range[1]))]

    # Router
    windows = ['morning_peak', 'midday', 'evening_peak', 'night']
    if section == "Resumen Ejecutivo":
        render_executive_summary(fdf)
    elif section == "Análisis Exploratorio de los Datos":
        render_eda(fdf)
    elif section == "Análisis Temporal":
        render_temporal_analysis(fdf, windows)
    elif section == "Simulador":
        render_simulator()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Construido con Streamlit")
    st.sidebar.caption("© 2025 - Steffany Lara")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Modo CLI simple
        import argparse
        parser = argparse.ArgumentParser(description="Análisis de Calidad del Aire (reglas por concentraciones)")
        parser.add_argument("--window", type=str, help="Ventana horaria (morning_peak/midday/evening_peak/night)")
        parser.add_argument("--simulate", type=str, help="Valores separados por coma: CO,NO,NO2,NOX,O3,PM10,PM2.5,SO2")
        args = parser.parse_args()

        if args.window and args.simulate:
            features = ['CO','NO','NO2','NOX','O3','PM10','PM2.5','SO2']
            try:
                values = [float(v.strip()) for v in args.simulate.split(',')]
                if len(values) != len(features):
                    raise ValueError
            except ValueError:
                print(f"Error: Se esperaban {len(features)} valores float separados por comas, p.ej.: 0.6,0.03,0.04,0.07,0.05,45,20,0.005")
                sys.exit(1)

            result = simulate_scenario(args.window, values, features)
            print(f"Clasificación: {result['cluster']} — ventana: {result['window']}")
            print(f"Interpretación:\n{result['interpretation']}")
        else:
            print("Iniciando aplicación de Streamlit...")
            os.system(f"streamlit run {__file__}")
    else:
        main()




