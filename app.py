# =============================================================================
# SIAGRM - Dashboard Enterprise en Streamlit
# Basado en el Bloque 23 del notebook original
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import pickle
import joblib
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="SIAGRM — Dashboard Enterprise",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CSS PERSONALIZADO (estilo similar al Dash original)
# =============================================================================
st.markdown("""
    <style>
        .main { background-color: #F4F7FB; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #E8EEF5;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            font-weight: 600;
            color: #173F73;
        }
        .stTabs [aria-selected="true"] {
            background-color: #173F73 !important;
            color: white !important;
        }
        .metric-card {
            background-color: white;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 4px 14px rgba(0,0,0,.07);
            text-align: center;
        }
        .metric-title { font-size: 12px; color: #6B7280; }
        .metric-value { font-size: 25px; font-weight: 800; color: #173F73; }
        .metric-subtitle { font-size: 10px; color: #6B7280; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# COLORES OFICIALES
# =============================================================================
COLORS = {
    "Bajo": "#2E7D32",
    "Medio": "#F9A825",
    "Alto": "#EF6C00",
    "Critico": "#C62828",
    "Fondo": "#F4F7FB"
}

# =============================================================================
# CARGA DE ARTEFACTOS
# =============================================================================
@st.cache_data
def cargar_datos():
    """Carga los datos exportados desde el notebook."""
    data = {}
    archivos = {
        'segmentos': 'segmentos_23.csv',
        'territorial': 'territorial_23.csv',
        'comparison': 'comparison_23.csv',
        'importance': 'importance_23.csv',
        'qa_report': 'qa_report_23.csv',
    }
    for key, filename in archivos.items():
        path = Path(filename)
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            data[key] = None
    return data

@st.cache_resource
def cargar_modelo():
    """Carga el modelo, preprocesador y configuración."""
    config_path = Path('config_23.json')
    modelo_path = Path('modelo.pkl')
    preproc_path = Path('preprocesador.pkl')
    
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    modelo = None
    if modelo_path.exists():
        with open(modelo_path, 'rb') as f:
            modelo = pickle.load(f)
    
    preprocesador = None
    if preproc_path.exists():
        with open(preproc_path, 'rb') as f:
            preprocesador = pickle.load(f)
    
    return config, modelo, preprocesador

# =============================================================================
# INTENTAR CARGAR
# =============================================================================
try:
    datos = cargar_datos()
    config, modelo, preprocesador = cargar_modelo()
    ARTEFACTOS_OK = True
except Exception as e:
    st.error(f"Error cargando artefactos: {e}")
    datos = {}
    config, modelo, preprocesador = {}, None, None
    ARTEFACTOS_OK = False

# =============================================================================
# EXTRAER CONFIGURACIÓN CON VALORES POR DEFECTO SI NO EXISTEN
# =============================================================================
def get_config(key, default=None):
    return config.get(key, default)

TOTAL = get_config('TOTAL_23', 0)
ALERTAS = get_config('ALERTAS_23', 0)
PCT_ALERTAS = get_config('PCT_ALERTAS_23', 0.0)
CRITICOS = get_config('CRITICOS_23', 0)
ALTOS = get_config('ALTOS_23', 0)
MEDIOS = get_config('MEDIOS_23', 0)
BAJOS = get_config('BAJOS_23', 0)
ROC_AUC = get_config('ROC_AUC_23', None)
DQS = get_config('DQS_23', None)
THRESHOLD = get_config('THRESHOLD_23', 0.286)
UMBRAL_MEDIO = get_config('UMBRAL_MEDIO_23', 0.35)
UMBRAL_ALTO = get_config('UMBRAL_ALTO_23', 0.65)
UMBRAL_CRITICO = get_config('UMBRAL_CRITICO_23', 0.90)
MODELO_CANONICO = get_config('MODELO_CANONICO_23', 'No disponible')
MODELO_RUNTIME = get_config('MODELO_RUNTIME_23', 'No disponible')
METODO_CALIBRACION = get_config('METODO_CALIBRACION_23', 'No disponible')
FEATURE_INPUTS = get_config('FEATURE_INPUTS_23', [])
RAW_INPUTS = get_config('RAW_INPUTS_23', [])
SIM_BASE = get_config('SIM_BASE_23', {})
VARIABLES_SOCIODEMOGRAFICAS = get_config('VARIABLES_SOCIODEMOGRAFICAS_23', [])
VARIABLES_ACADEMICAS = get_config('VARIABLES_ACADEMICAS_23', [])
VARIABLES_OFICIALES = get_config('VARIABLES_OFICIALES_23', [])
INSIGHTS = get_config('INSIGHTS_23', [])
# Fallback: algunos exports usan 'HALLAZGOS_CLAVE_23' con estructura de dicts
if not INSIGHTS:
    hallazgos = get_config('HALLAZGOS_CLAVE_23', [])
    if isinstance(hallazgos, list) and hallazgos:
        # Extraer texto de 'hallazgo' si está disponible, o stringify como fallback
        INSIGHTS = [h.get('hallazgo') if isinstance(h, dict) and 'hallazgo' in h else str(h) for h in hallazgos]

# =============================================================================
# HEADER PRINCIPAL
# =============================================================================
st.markdown("""
    <div style="background: linear-gradient(120deg,#173F73,#2F80ED); 
                color: white; padding: 24px; border-radius: 18px; margin-bottom: 16px;">
        <h2 style="font-weight: 800; margin: 0;">SIAGRM</h2>
        <div style="font-size: 18px;">
            Sistema Inteligente de Analítica para la Gestión Preventiva del Riesgo de Mora
        </div>
        <small>
            Modelo canónico: {} | Runtime: {} | Calibración: {}
        </small>
    </div>
""".format(MODELO_CANONICO, MODELO_RUNTIME, METODO_CALIBRACION), unsafe_allow_html=True)

# =============================================================================
# VERIFICACIÓN DE ARTEFACTOS
# =============================================================================
if not ARTEFACTOS_OK or datos.get('segmentos') is None:
    st.error("""
        ⚠️ **No se encontraron los artefactos necesarios.**
        
        Para que este dashboard funcione, debes exportar los siguientes archivos 
        desde tu notebook de Colab y subirlos a este repositorio:
        
        1. `segmentos_23.csv` — Base de segmentos evaluados
        2. `territorial_23.csv` — Datos territoriales agregados
        3. `comparison_23.csv` — Comparativo de modelos
        4. `importance_23.csv` — Importancia de variables
        5. `qa_report_23.csv` — Reporte de QA
        6. `config_23.json` — Configuración, umbrales y métricas
        7. `modelo.pkl` — Modelo entrenado (PIPELINE_23 o MODEL_23)
        8. `preprocesador.pkl` — Preprocesador (solo si aplica)
        
        **Al final de este mensaje te doy el código exacto para exportarlos desde Colab.**
    """)
    
    st.code("""
# === CÓDIGO PARA EXPORTAR DESDE COLAB (ejecutar al final del Bloque 23) ===
import json, pickle, pandas as pd
from pathlib import Path

# 1. Exportar DataFrames
SEGMENTOS_23.to_csv('segmentos_23.csv', index=False, encoding='utf-8-sig')
TERRITORIAL_23.to_csv('territorial_23.csv', index=False, encoding='utf-8-sig')
COMPARISON_23.to_csv('comparison_23.csv', index=False, encoding='utf-8-sig')
IMPORTANCE_23.to_csv('importance_23.csv', index=False, encoding='utf-8-sig')
QA_REPORT_23.to_csv('qa_report_23.csv', index=False, encoding='utf-8-sig')

# 2. Exportar configuración
config_23 = {
    'TOTAL_23': int(len(SEGMENTOS_23)),
    'ALERTAS_23': int(ALERTAS_23),
    'PCT_ALERTAS_23': float(PCT_ALERTAS_23),
    'CRITICOS_23': int(CRITICOS_23),
    'ALTOS_23': int(ALTOS_23),
    'MEDIOS_23': int(MEDIOS_23),
    'BAJOS_23': int(BAJOS_23),
    'ROC_AUC_23': float(ROC_AUC_23) if ROC_AUC_23 is not None else None,
    'DQS_23': float(DQS_23) if DQS_23 is not None else None,
    'THRESHOLD_23': float(THRESHOLD_23),
    'UMBRAL_MEDIO_23': float(UMBRAL_MEDIO_23),
    'UMBRAL_ALTO_23': float(UMBRAL_ALTO_23),
    'UMBRAL_CRITICO_23': float(UMBRAL_CRITICO_23),
    'MODELO_CANONICO_23': str(MODELO_CANONICO_23),
    'MODELO_RUNTIME_23': str(MODELO_RUNTIME_23),
    'METODO_CALIBRACION_23': str(METODO_CALIBRACION_23),
    'FEATURE_INPUTS_23': list(FEATURE_INPUTS_23),
    'RAW_INPUTS_23': list(RAW_INPUTS_23),
    'SIM_BASE_23': {k: str(v) if pd.notna(v) else None for k, v in SIM_BASE_23.items()},
    'VARIABLES_SOCIODEMOGRAFICAS_23': list(VARIABLES_SOCIODEMOGRAFICAS_23),
    'VARIABLES_ACADEMICAS_23': list(VARIABLES_ACADEMICAS_23),
    'VARIABLES_OFICIALES_23': list(VARIABLES_OFICIALES_23),
    'INSIGHTS_23': list(INSIGHTS_23) if 'INSIGHTS_23' in globals() else [],
    'ALERTA_MIN_23': float(ALERTA_MIN_23) if 'ALERTA_MIN_23' in globals() else float(UMBRAL_ALTO_23),
}
with open('config_23.json', 'w', encoding='utf-8') as f:
    json.dump(config_23, f, ensure_ascii=False, indent=2)

# 3. Exportar modelo y preprocesador
with open('modelo.pkl', 'wb') as f:
    pickle.dump(PIPELINE_23 if PIPELINE_23 is not None else MODEL_23, f)

if PREPROCESSOR_23 is not None:
    with open('preprocesador.pkl', 'wb') as f:
        pickle.dump(PREPROCESSOR_23, f)

# 4. Descargar archivos
from google.colab import files
for fname in ['segmentos_23.csv', 'territorial_23.csv', 'comparison_23.csv',
              'importance_23.csv', 'qa_report_23.csv', 'config_23.json',
              'modelo.pkl', 'preprocesador.pkl']:
    if Path(fname).exists():
        files.download(fname)
    """, language='python')
    
    st.stop()

# =============================================================================
# SI LLEGAMOS AQUÍ, TODO ESTÁ CARGADO
# =============================================================================
SEGMENTOS = datos['segmentos']
TERRITORIAL = datos['territorial']
COMPARISON = datos['comparison']
IMPORTANCE = datos['importance']
QA_REPORT = datos['qa_report']

PREFLIGHT_SCORE = None
PRE_FLIGHT_ROW = None

# Asegurar tipos
if 'score_riesgo' in SEGMENTOS.columns:
    SEGMENTOS['score_riesgo'] = pd.to_numeric(SEGMENTOS['score_riesgo'], errors='coerce').clip(0, 1)
if 'nivel_riesgo' not in SEGMENTOS.columns and 'score_riesgo' in SEGMENTOS.columns:
    SEGMENTOS['nivel_riesgo'] = np.select(
        [SEGMENTOS['score_riesgo'] >= UMBRAL_CRITICO,
         SEGMENTOS['score_riesgo'] >= UMBRAL_ALTO,
         SEGMENTOS['score_riesgo'] >= UMBRAL_MEDIO],
        ['Critico', 'Alto', 'Medio'],
        default='Bajo'
    )
if 'alerta_activada' not in SEGMENTOS.columns:
    SEGMENTOS['alerta_activada'] = SEGMENTOS['nivel_riesgo'].isin(['Alto', 'Critico']).astype(int)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def kpi_card(title, value, subtitle="", color="#173F73"):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color: {color};">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)

def clasificar_riesgo(prob):
    if prob >= UMBRAL_CRITICO:
        return "CRÍTICO", "#C62828", "Priorizar gestión preventiva intensiva."
    elif prob >= UMBRAL_ALTO:
        return "ALTO", "#EF6C00", "Activar seguimiento preventivo focalizado."
    elif prob >= UMBRAL_MEDIO:
        return "MEDIO", "#F9A825", "Mantener monitoreo preventivo."
    else:
        return "BAJO", "#2E7D32", "Mantener seguimiento regular."

def predecir_simulacion(row_df):
    """
    Replica la lógica del Bloque 23 para inferencia.
    Soporta: Pipeline completo, Preprocesador+Modelo, o Modelo directo.
    """
    if not isinstance(row_df, pd.DataFrame):
        raise TypeError("Se requiere un DataFrame")
    if len(row_df) != 1:
        raise RuntimeError("Se requiere exactamente una fila")
    
    # Verificar columnas RAW
    faltantes = [c for c in RAW_INPUTS if c not in row_df.columns]
    if faltantes:
        raise RuntimeError(f"Faltan columnas RAW: {', '.join(faltantes)}")

    X = row_df[RAW_INPUTS].copy()
    
    # Ruta 1: Pipeline completo
    if modelo is not None and hasattr(modelo, 'predict_proba'):
        try:
            # Verificar si es pipeline completo o modelo directo
            n_features_modelo = getattr(modelo, 'n_features_in_', None)
            if n_features_modelo is None or n_features_modelo == len(RAW_INPUTS):
                proba = modelo.predict_proba(X)[:, 1]
                score = float(proba[0])
                if 0 <= score <= 1:
                    return score
        except Exception:
            pass
    
    # Ruta 2: Preprocesador + Modelo
    if preprocesador is not None and modelo is not None:
        try:
            X_proc = preprocesador.transform(X)
            proba = modelo.predict_proba(X_proc)[:, 1]
            score = float(proba[0])
            if 0 <= score <= 1:
                return score
        except Exception:
            pass
    
    # Ruta 3: Fallback - no hay modelo disponible
    raise RuntimeError("No se pudo realizar la inferencia. Verifica que el modelo y preprocesador estén correctamente exportados.")


def construir_fila_simulacion(data_dict):
    """Reproduce el simulador del notebook.

    La diferencia crítica es que la app debe usar un registro real del SSOT que
    coincida exactamente con las 5 variables editables del simulador, y no una
    fila cualquiera de la base. Eso preserva el resto del contexto RAW
    (categoria_municipio, modalidad_linea, sector_ies, rango_valor_total, etc.)
    y hace que el score coincida con el notebook.
    """
    if not SIM_BASE:
        raise RuntimeError("SIM_BASE_23 no está disponible en la configuración.")
    if SEGMENTOS is None or SEGMENTOS.empty:
        raise RuntimeError("La base SEGMENTOS no está disponible para la simulación.")

    data_dict = data_dict or {}

    # 1) Buscar la fila real del SSOT que coincide exactamente con el perfil elegido
    # en los campos canónicos editables. Priorizar coincidencia contra SIM_BASE
    # para mantener el mismo baseline que usa el notebook cuando exista.
    fila_base = SEGMENTOS.loc[:, RAW_INPUTS].dropna(how='all').head(1).copy()

    # Intentar localizar una fila que coincida con SIM_BASE (si está disponible)
    try:
        if SIM_BASE and FEATURE_INPUTS:
            mascara_sim = pd.Series(True, index=SEGMENTOS.index)
            any_sim = False
            for col in FEATURE_INPUTS:
                if col not in SEGMENTOS.columns or col not in SIM_BASE:
                    continue
                val_sim = SIM_BASE.get(col)
                if pd.isna(val_sim):
                    continue
                any_sim = True
                val_norm = str(val_sim).strip()
                mascara_sim &= SEGMENTOS[col].astype(str).str.strip().str.lower().eq(val_norm.lower())
            if any_sim and mascara_sim.any():
                fila_base = SEGMENTOS.loc[mascara_sim, RAW_INPUTS].head(1).copy()
    except Exception:
        # Si algo falla en la comparación por tipos, ignorar y seguir
        pass

    # Si el usuario pasó valores específicos, intentar encontrar la fila exacta
    if FEATURE_INPUTS:
        mascara = pd.Series(True, index=SEGMENTOS.index)
        for col in FEATURE_INPUTS:
            if col not in SEGMENTOS.columns or col not in data_dict:
                continue
            valor = data_dict[col]
            if pd.isna(valor):
                continue
            valor_norm = str(valor).strip()
            try:
                mascara &= SEGMENTOS[col].astype(str).str.strip().str.lower().eq(valor_norm.lower())
            except Exception:
                mascara &= SEGMENTOS[col].astype(str).str.strip().str.lower().eq(str(valor).lower())
        if mascara.any():
            fila_base = SEGMENTOS.loc[mascara, RAW_INPUTS].head(1).copy()

    if fila_base.empty:
        raise RuntimeError("No hay una fila base válida en SEGMENTOS para reconstruir el registro RAW.")

    fila = fila_base.iloc[0].copy()

    # 2) Aplicar valores editables sobre la fila real
    for col, valor in data_dict.items():
        if col in fila.index:
            fila[col] = valor

    # 3) Si no hubo valor entregado para una variable editable, usar SIM_BASE
    for col in FEATURE_INPUTS:
        if col in fila.index and pd.isna(fila[col]):
            if col in SIM_BASE:
                fila[col] = SIM_BASE.get(col)

    # 4) Asegurar que todas las columnas RAW necesarias existan y no queden vacías
    for col in RAW_INPUTS:
        if col not in fila.index or pd.isna(fila[col]):
            if col in SIM_BASE:
                fila[col] = SIM_BASE.get(col)
            elif col in SEGMENTOS.columns:
                modo = SEGMENTOS[col].dropna().mode()
                fila[col] = modo.iloc[0] if not modo.empty else np.nan
            else:
                fila[col] = np.nan

    # 5) Mantener exactamente el orden y el tipo del contrato RAW del notebook
    return fila[RAW_INPUTS].to_frame().T


def obtener_preflight():
    """Reproduce el preflight del notebook: valida que el registro base pueda inferir un score finito."""
    if not FEATURE_INPUTS:
        raise RuntimeError("FEATURE_INPUTS_23 no está configurado.")

    valores_preflight = {}
    for col in FEATURE_INPUTS:
        if col in SIM_BASE and SIM_BASE.get(col) is not None:
            valores_preflight[col] = SIM_BASE[col]
        elif col in SEGMENTOS.columns and not SEGMENTOS[col].dropna().empty:
            valores_preflight[col] = SEGMENTOS[col].dropna().iloc[0]
        else:
            raise RuntimeError(f"No se pudo construir el valor de preflight para la variable {col}.")

    fila = construir_fila_simulacion(valores_preflight)
    score = predecir_simulacion(fila)

    if not np.isfinite(score):
        raise RuntimeError("El preflight del simulador produjo un score no finito.")
    if not 0.0 <= score <= 1.0:
        raise RuntimeError(f"El score del preflight está fuera de [0,1]: {score}")

    return score, fila


def profile_data(profile_col):
    """Genera datos de perfil agregados."""
    if profile_col not in SEGMENTOS.columns:
        return pd.DataFrame()
    prof = SEGMENTOS.groupby(profile_col, dropna=False).agg(
        registros=('score_riesgo', 'size'),
        alertas=('alerta_activada', 'sum'),
        score_promedio=('score_riesgo', 'mean')
    ).reset_index()
    prof['tasa_alerta_pct'] = (prof['alertas'] / prof['registros'] * 100).fillna(0).round(2)
    return prof

# Ejecutar preflight ahora que la función está definida
try:
    PREFLIGHT_SCORE, PRE_FLIGHT_ROW = obtener_preflight()
except Exception:
    PREFLIGHT_SCORE = None
    PRE_FLIGHT_ROW = None

# =============================================================================
# TABS
# =============================================================================
tabs = st.tabs([
    "📊 Resumen ejecutivo",
    "🤖 Modelos comparados",
    "📈 Variables explicativas",
    "🗺️ Territorio",
    "👥 Perfiles",
    "✅ QA y gobernanza",
    "🗃️ Base analítica",
    "🎛️ Simulador predictivo"
])

# =============================================================================
# TAB 1: RESUMEN EJECUTIVO
# =============================================================================
with tabs[0]:
    # KPIs
    cols = st.columns(6)
    with cols[0]:
        kpi_card("Segmentos evaluados", f"{TOTAL:,}")
    with cols[1]:
        kpi_card("Alertas activas", f"{ALERTAS:,}", f"{PCT_ALERTAS:.2f}%")
    with cols[2]:
        kpi_card("Casos críticos", f"{CRITICOS:,}", color="#C62828")
    with cols[3]:
        kpi_card("Casos altos", f"{ALTOS:,}", color="#EF6C00")
    with cols[4]:
        kpi_card("ROC-AUC", f"{ROC_AUC:.4f}" if ROC_AUC is not None else "N/D")
    with cols[5]:
        kpi_card("DQS", f"{DQS:.1f}%" if DQS is not None else "N/D")
    
    st.markdown("---")
    
    # Gráficas
    c1, c2 = st.columns(2)
    with c1:
        dist = SEGMENTOS['nivel_riesgo'].value_counts().reindex(['Bajo', 'Medio', 'Alto', 'Critico'], fill_value=0).rename_axis('nivel').reset_index(name='casos')
        fig = px.pie(dist, names='nivel', values='casos', hole=.52,
                     title="Distribución del Nivel de Riesgo",
                     color='nivel', color_discrete_map=COLORS)
        fig.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        fig = px.histogram(SEGMENTOS, x='score_riesgo', color='nivel_riesgo', nbins=30,
                           color_discrete_map=COLORS,
                           title="Distribución del Score de Riesgo")
        fig.update_layout(template="plotly_white", height=450, xaxis_title="Score", yaxis_title="Segmentos")
        st.plotly_chart(fig, use_container_width=True)
    
    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(SEGMENTOS, x='nivel_riesgo', y='score_riesgo', color='nivel_riesgo',
                     color_discrete_map=COLORS,
                     category_orders={'nivel_riesgo': ['Bajo', 'Medio', 'Alto', 'Critico']},
                     title="Dispersión del Score por Nivel")
        fig.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with c4:
        fig = go.Figure()
        fig_data = SEGMENTOS['score_riesgo'].sort_values().to_numpy()
        if len(fig_data):
            y = np.linspace(0, 100, len(fig_data))
            fig.add_trace(go.Scatter(x=fig_data, y=y, mode='lines', name='Distribución acumulada (%)'))
        for valor, nombre in [(THRESHOLD, "Threshold modelo"), (UMBRAL_MEDIO, "Medio"),
                              (UMBRAL_ALTO, "Alto"), (UMBRAL_CRITICO, "Crítico")]:
            if valor is not None:
                fig.add_vline(x=valor, line_dash="dash", annotation_text=f"{nombre}: {valor:.3f}")
        fig.update_layout(template="plotly_white", title="Score acumulado y umbrales operativos",
                          xaxis_title="Score de riesgo", yaxis_title="% acumulado", height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    # Resumen e insights
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("#### Resumen del modelo")

        # Preparar representación segura de ROC_AUC
        ROC_AUC_STR = f"{ROC_AUC:.4f}" if ROC_AUC is not None else "N/D"

        st.markdown(f"""
        - **Modelo canónico:** {MODELO_CANONICO}
        - **Runtime:** {MODELO_RUNTIME}
        - **Calibración:** {METODO_CALIBRACION}
        - **ROC-AUC:** {ROC_AUC_STR}
        - **Threshold modelo:** {THRESHOLD:.6f}
        - **Umbral Alto:** {UMBRAL_ALTO:.4f}
        - **Umbral Crítico:** {UMBRAL_CRITICO:.4f}
        - **Alertas:** {ALERTAS:,} ({PCT_ALERTAS:.2f}%)
        - **Uso:** priorización preventiva y apoyo analítico.
        """)
    
    with c6:
        st.markdown("#### Hallazgos clave")
        if INSIGHTS:
            for item in INSIGHTS:
                st.markdown(f"- {item}")
        else:
            st.info("No hay hallazgos configurados en config_23.json")

# =============================================================================
# TAB 2: MODELOS COMPARADOS
# =============================================================================
with tabs[1]:
    if COMPARISON is not None and not COMPARISON.empty:
        # Comparativo de modelos
        fig = px.bar(COMPARISON.sort_values('ROC_AUC_CV', ascending=True) if 'ROC_AUC_CV' in COMPARISON.columns else COMPARISON,
                     x='ROC_AUC_CV' if 'ROC_AUC_CV' in COMPARISON.columns else COMPARISON.columns[1],
                     y='Modelo' if 'Modelo' in COMPARISON.columns else COMPARISON.columns[0],
                     orientation='h', text='ROC_AUC_CV' if 'ROC_AUC_CV' in COMPARISON.columns else None,
                     title="Comparación de modelos — Group-CV ROC-AUC")
        fig.update_layout(template="plotly_white", height=500, xaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de métricas
        st.markdown("#### Comparativo de métricas")
        st.dataframe(COMPARISON, use_container_width=True, hide_index=True)
    else:
        st.warning("No se encontró el comparativo de modelos.")

# =============================================================================
# TAB 3: VARIABLES EXPLICATIVAS
# =============================================================================
# TAB 3: VARIABLES EXPLICATIVAS (adaptado a columnas: Variable, Importancia, Participacion_pct, Tipo)
with tabs[2]:
    if IMPORTANCE is not None and not IMPORTANCE.empty:
        col_x = 'Participacion_pct' if 'Participacion_pct' in IMPORTANCE.columns else 'Importancia'
        col_y = 'variable_original' if 'variable_original' in IMPORTANCE.columns else (IMPORTANCE.columns[0] if len(IMPORTANCE.columns) > 0 else None)
        
        if col_y:
            # Preparar dataframe para la gráfica: identificar tipo (Sociodemográfica / Académica / Otro)
            df_plot = IMPORTANCE.sort_values(col_x, ascending=True).tail(20).copy()
            def tipo_variable(v):
                if v in VARIABLES_SOCIODEMOGRAFICAS:
                    return 'Sociodemográfica'
                if v in VARIABLES_ACADEMICAS:
                    return 'Académica'
                return 'Otro'
            df_plot['tipo'] = df_plot[col_y].apply(tipo_variable)

            # Ordenar para que los grupos queden visualmente agrupados
            df_plot = df_plot.sort_values(['tipo', col_x], ascending=[True, True])

            color_map = {
                'Sociodemográfica': '#2E7D32',
                'Académica': '#2F80ED',
                'Otro': '#6B7280'
            }

            fig = px.bar(df_plot, x=col_x, y=col_y, orientation='h',
                         title="Importancia de variables originales",
                         color='tipo', color_discrete_map=color_map,
                         category_orders={'tipo': ['Sociodemográfica', 'Académica', 'Otro']})
            fig.update_traces(marker_line_color='rgba(0,0,0,0.06)', marker_line_width=0.6)
            fig.update_layout(template='plotly_white', height=650, bargap=0.12, legend_title_text='Tipo')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Detalle de importancia")
        st.dataframe(IMPORTANCE, use_container_width=True, hide_index=True)
        st.caption("La importancia es relativa y no implica causalidad individual.")
    else:
        st.warning("No se encontró la importancia de variables.")

# =============================================================================
# TAB 4: TERRITORIO
# =============================================================================
with tabs[3]:
    if TERRITORIAL is not None and not TERRITORIAL.empty:
        terr_plot = TERRITORIAL.copy()
        if 'tasa_alerta_pct' not in terr_plot.columns and 'alertas' in terr_plot.columns and 'registros' in terr_plot.columns:
            terr_plot['tasa_alerta_pct'] = (terr_plot['alertas'] / terr_plot['registros'] * 100).fillna(0)
        
        # Tasa de alerta
        fig = px.bar(terr_plot.sort_values('tasa_alerta_pct').tail(20),
                     x='tasa_alerta_pct', y='departamento', orientation='h',
                     title="Tasa de Alerta por Departamento", text='tasa_alerta_pct')
        fig.update_layout(template="plotly_white", height=max(500, 35*len(terr_plot.tail(20))),
                          xaxis_title="Tasa de alerta (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Score promedio
        if 'score_promedio' in terr_plot.columns:
            fig = px.bar(terr_plot.sort_values('score_promedio'),
                         x='score_promedio', y='departamento', orientation='h',
                         title="Score Promedio por Departamento")
            fig.update_layout(template="plotly_white", height=700)
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        if 'nivel_riesgo' in SEGMENTOS.columns and 'departamento' in SEGMENTOS.columns:
            terr_matrix = SEGMENTOS.pivot_table(
                index='departamento', columns='nivel_riesgo',
                values='score_riesgo', aggfunc='size', fill_value=0
            ).reindex(columns=['Bajo', 'Medio', 'Alto', 'Critico'], fill_value=0).reset_index()
            
            if not terr_matrix.empty:
                z = terr_matrix[['Bajo', 'Medio', 'Alto', 'Critico']].to_numpy()
                fig = go.Figure(data=go.Heatmap(
                    z=z, x=['Bajo', 'Medio', 'Alto', 'Crítico'],
                    y=terr_matrix['departamento'].astype(str),
                    text=z, texttemplate="%{text}", colorscale="Reds"
                ))
                fig.update_layout(template="plotly_white", title="Matriz Departamento × Nivel de Riesgo",
                                  height=max(550, 28*len(terr_matrix)))
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Detalle territorial")
        st.dataframe(TERRITORIAL, use_container_width=True, hide_index=True)
    else:
        st.warning("No se encontraron datos territoriales.")

# =============================================================================
# TAB 5: PERFILES
# =============================================================================
with tabs[4]:
    profile_options = [c for c in VARIABLES_OFICIALES if c in SEGMENTOS.columns]
    if profile_options:
        selected_profile = st.selectbox(
            "Selecciona una variable de perfil:",
            options=profile_options,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        prof = profile_data(selected_profile)
        if not prof.empty:
            fig = px.bar(prof.sort_values('tasa_alerta_pct'),
                         x='tasa_alerta_pct', y=selected_profile, orientation='h',
                         text='tasa_alerta_pct',
                         title=f"Tasa de Alerta por {selected_profile.replace('_', ' ').title()} (%)")
            fig.update_layout(template="plotly_white", height=650)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Detalle del perfil")
            st.dataframe(prof, use_container_width=True, hide_index=True)
    else:
        st.warning("No hay variables de perfil disponibles.")

# =============================================================================
# TAB 6: QA Y GOBERNANZA
# =============================================================================
with tabs[5]:
    st.markdown("#### Validaciones del modelo y SSOT")
    if QA_REPORT is not None and not QA_REPORT.empty:
        st.dataframe(QA_REPORT, use_container_width=True, hide_index=True)
    else:
        st.info("No se encontró el reporte de QA.")
    
    st.markdown("---")
    st.markdown("#### Identidad de validación")
    identidad = get_config('IDENTIDAD_18_23', {})
    if identidad:
        st.json(identidad)
    else:
        st.info("Identidad no configurada en config_23.json")

# =============================================================================
# TAB 7: BASE ANALÍTICA
# =============================================================================
with tabs[6]:
    st.markdown(f"**Segmentos evaluados: {TOTAL:,}**")
    st.dataframe(SEGMENTOS, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 8: SIMULADOR PREDICTIVO
# =============================================================================
with tabs[7]:
    st.markdown("""
        <div class="alert alert-info" style="padding: 12px; border-radius: 8px; background-color: #D1ECF1; color: #0C5460;">
            ℹ️ La inferencia se realiza exclusivamente mediante los artefactos oficiales 
            del Runtime Contract. Las 5 variables editables se aplican sobre un registro 
            RAW real completo del SSOT.
        </div>
    """, unsafe_allow_html=True)
    
    if not FEATURE_INPUTS:
        st.error("FEATURE_INPUTS_23 no está configurado. Verifica config_23.json")
        st.stop()
    
    # Crear controles del simulador
    col_inputs, col_result = st.columns([3, 2])
    
    with col_inputs:
        st.markdown("#### Variables de entrada")
        sim_values = {}
        
        for col in FEATURE_INPUTS:
            tipo = "Sociodemográfica" if col in VARIABLES_SOCIODEMOGRAFICAS else "Académica"
            label = f"{col.replace('_', ' ').title()} — {tipo}"

            # Obtener datos y opciones únicas de la base
            serie = SEGMENTOS[col] if col in SEGMENTOS.columns else pd.Series([])
            opciones = serie.dropna().astype(str).unique().tolist()
            valor_default = SIM_BASE.get(col, opciones[0] if opciones else '')

            # Detectar columnas numéricas y ofrecer un number_input cuando aplique
            try:
                es_numerica = False
                if col in SEGMENTOS.columns:
                    es_numerica = pd.api.types.is_numeric_dtype(SEGMENTOS[col])
                # También aceptar como numérica si las opciones son números y no demasiadas categorías
                if not es_numerica and opciones:
                    candidatas_numericas = [s for s in opciones if str(s).replace('.', '', 1).replace('-', '', 1).isdigit()]
                    if len(candidatas_numericas) == len(opciones) and len(opciones) > 1:
                        es_numerica = True

                if es_numerica:
                    vals = pd.to_numeric(serie.dropna(), errors='coerce')
                    minv = float(vals.min()) if not vals.empty else 0.0
                    maxv = float(vals.max()) if not vals.empty else 1.0
                    # Normalizar defaults
                    try:
                        default_num = float(valor_default)
                    except Exception:
                        default_num = (minv + maxv) / 2 if minv != maxv else minv
                    step = (maxv - minv) / 100 if maxv > minv else 1.0
                    sim_values[col] = st.number_input(label, min_value=minv, max_value=maxv, value=default_num, step=step, format="%g")
                else:
                    # Comportamiento por defecto: selectbox (con manejo especial para 'estrato')
                    if col == 'estrato':
                        try:
                            opciones_num = sorted([float(x) for x in opciones if pd.notna(x) and str(x).replace('.','').isdigit()])
                            if opciones_num:
                                default_num = float(valor_default) if str(valor_default).replace('.','').isdigit() else opciones_num[0]
                                sim_values[col] = st.selectbox(label, options=opciones_num, 
                                                               index=opciones_num.index(default_num) if default_num in opciones_num else 0,
                                                               format_func=lambda x: str(int(x)) if x == int(x) else str(x))
                            else:
                                sim_values[col] = st.selectbox(label, options=opciones, index=opciones.index(str(valor_default)) if str(valor_default) in opciones else 0)
                        except Exception:
                            sim_values[col] = st.selectbox(label, options=opciones, index=opciones.index(str(valor_default)) if str(valor_default) in opciones else 0)
                    else:
                        sim_values[col] = st.selectbox(label, options=opciones, index=opciones.index(str(valor_default)) if str(valor_default) in opciones else 0)
            except Exception:
                # Fallback seguro a selectbox
                sim_values[col] = st.selectbox(label, options=opciones, index=opciones.index(str(valor_default)) if str(valor_default) in opciones else 0)
        
        calcular = st.button("🔮 Calcular riesgo", type="primary", use_container_width=True)
    
    with col_result:
        st.markdown("#### Resultado de la predicción")

        pass

        if calcular:
            try:
                with st.spinner("Calculando score de riesgo..."):
                    # 1. Construir fila RAW completa
                    row = construir_fila_simulacion(sim_values)

                    # 2. Inferencia
                    probability = predecir_simulacion(row)

                    # 3. Clasificación
                    level, color, action = clasificar_riesgo(probability)
                    alerta = probability >= UMBRAL_ALTO

                    # 4. Mostrar resultado
                    st.markdown(f"""
                        <div style="background-color: white; border-radius: 16px; padding: 24px; 
                                    box-shadow: 0 4px 14px rgba(0,0,0,.07); text-align: center;">
                            <div style="font-size: 14px; color: #6B7280;">Probabilidad estimada</div>
                            <div style="font-size: 42px; font-weight: 800; color: {color};">
                                {probability:.2%}
                            </div>
                            <div style="font-size: 20px; font-weight: 600; color: {color}; margin-top: 8px;">
                                Nivel de riesgo: {level}
                            </div>
                            <div style="font-size: 14px; color: #6B7280; margin-top: 8px;">
                                Score: {probability:.6f}
                            </div>
                            <div style="font-size: 14px; margin-top: 8px;">
                                Alerta activada: <b>{'Sí' if alerta else 'No'}</b>
                            </div>
                            <hr style="margin: 16px 0;">
                            <div style="font-size: 12px; color: #6B7280;">
                                Threshold modelo: {THRESHOLD:.4f} | 
                                Medio: {UMBRAL_MEDIO:.4f} | 
                                Alto: {UMBRAL_ALTO:.4f} | 
                                Crítico: {UMBRAL_CRITICO:.4f}
                            </div>
                            <div style="margin-top: 16px; padding: 12px; background-color: {color}15; 
                                        border-radius: 8px; color: {color}; font-weight: 600;">
                                {action}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Score Predictivo"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, UMBRAL_MEDIO*100], 'color': "#E8F5E9"},
                                {'range': [UMBRAL_MEDIO*100, UMBRAL_ALTO*100], 'color': "#FFF8E1"},
                                {'range': [UMBRAL_ALTO*100, UMBRAL_CRITICO*100], 'color': "#FFE0B2"},
                                {'range': [UMBRAL_CRITICO*100, 100], 'color': "#FFCDD2"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': UMBRAL_CRITICO*100
                            }
                        }
                    ))
                    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                    # Mostrar fila construida para transparencia
                    with st.expander("Ver registro RAW construido"):
                        st.dataframe(row, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error en la simulación: {str(e)}")
                st.info("Verifica que el modelo y preprocesador estén correctamente exportados.")
        else:
            st.info("Ajusta los parámetros y presiona **Calcular riesgo** para obtener una predicción.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("""
    © 2026 | SIAGRM v5.0 | María Alejandra Ruíz Rubio - Elias Alberto Cardona Rodriguez | 
    ICETEX | Universidad de Ibagué | Maestría en Analítica de Datos para la Toma de Decisiones
""")
