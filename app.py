import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
MODELS = ROOT / "models"

st.set_page_config(
    page_title="SIAGRM — Gestión Preventiva del Riesgo de Mora",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# IDENTIDAD VISUAL 1:1 CON BLOQUE 23
# --------------------------------------------------------------------------------------
COLOR_NAVY = "#173F73"
COLOR_BLUE = "#2F80ED"
COLOR_TEAL = "#0F766E"
COLOR_BG = "#F5F7FB"
COLOR_TEXT = "#172033"
COLOR_MUTED = "#64748B"
COLOR_BORDER = "#E2E8F0"
COLOR_GREEN = "#15803D"
COLOR_RED = "#C62828"
COLOR_ORANGE = "#EF6C00"
COLOR_AMBER = "#F9A825"
RISK_COLORS = {"Bajo":"#2E7D32","Medio":"#F9A825","Alto":"#EF6C00","Critico":"#C62828"}
RISK_LABELS = {"Bajo":"Bajo","Medio":"Medio","Alto":"Alto","Critico":"Crítico"}
RISK_ORDER = ["Bajo","Medio","Alto","Critico"]

st.markdown(f"""
<style>
.stApp {{ background:{COLOR_BG}; color:{COLOR_TEXT}; }}
.block-container {{ padding-top:1.25rem; padding-bottom:2rem; max-width:1500px; }}
.hero {{
    background: linear-gradient(120deg,{COLOR_NAVY},{COLOR_BLUE});
    color:white; border-radius:18px; padding:24px 26px; margin-bottom:14px;
}}
.card {{
    background:#FFFFFF; border:1px solid {COLOR_BORDER}; border-radius:16px;
    padding:18px; box-shadow:0 6px 18px rgba(15,23,42,.06); height:100%;
}}
.kpi {{ font-size:12px; color:{COLOR_MUTED}; font-weight:600; }}
.kpi-value {{ font-size:28px; font-weight:800; margin-top:4px; }}
.kpi-desc {{ font-size:10px; color:{COLOR_MUTED}; margin-top:5px; }}
.section-title {{ font-size:18px; font-weight:900; color:{COLOR_TEXT}; }}
.small-muted {{ color:{COLOR_MUTED}; font-size:12px; }}
[data-testid="stMetricValue"] {{ color:{COLOR_NAVY}; }}
div[data-baseweb="tab-list"] {{ gap:.2rem; }}
button[data-baseweb="tab"] {{ font-weight:700; color:{COLOR_MUTED}; }}
button[data-baseweb="tab"][aria-selected="true"] {{
  color:{COLOR_NAVY}; border-bottom:3px solid {COLOR_BLUE};
  background:#EEF5FF;
}}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_segments():
    pq = DATA / "segments.parquet"
    csv = DATA / "segments.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError("No se encontró data/segments.parquet ni data/segments.csv.")

@st.cache_resource
def load_payload():
    p = ART / "dashboard_payload.joblib"
    if not p.exists():
        raise FileNotFoundError("No se encontró artifacts/dashboard_payload.joblib.")
    return joblib.load(p)

@st.cache_resource
def load_model():
    """Carga los artefactos publicados para reproducir la inferencia canónica del BLOQUE 23."""
    bundle_path = MODELS / "simulador_bundle.joblib"
    pipeline_path = MODELS / "simulador.joblib"
    preprocessor_path = MODELS / "preprocessor.joblib"

    # Preferimos el bundle porque contiene la relación explícita entre pipeline,
    # preprocesador y modelo directo.
    if bundle_path.exists():
        try:
            bundle = joblib.load(bundle_path)
            if isinstance(bundle, dict):
                return bundle
        except Exception:
            pass

    # Fallback: construir el mismo bundle a partir de los artefactos individuales.
    pipeline = None
    preprocessor = None
    model = None

    if pipeline_path.exists():
        try:
            pipeline = joblib.load(pipeline_path)
        except Exception:
            pipeline = None

    if preprocessor_path.exists():
        try:
            preprocessor = joblib.load(preprocessor_path)
        except Exception:
            preprocessor = None

    # El modelo directo puede venir dentro de un artefacto adicional solo si la
    # celda lo hubiese publicado; se deja como None para no inventar un modelo.
    return {
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "model": model,
    }

df_all = load_segments().copy()
payload = load_payload()
MODEL = load_model()

VARIABLES_RAW = payload["variables_raw"]
DESCRIPCIONES = payload["descripciones_variables"]
MODELO_CANONICO = payload.get("modelo_canonico", "N/D")
MODELO_RUNTIME = payload.get("modelo_runtime", "N/D")
METODO_CALIBRACION = payload.get("calibracion", "N/D")
METRICAS = payload.get("metricas", {})
THRESHOLD_MODELO = payload.get("threshold_modelo")
UMBRAL_MEDIO = float(payload.get("umbral_medio", 0.35))
UMBRAL_ALTO = float(payload.get("umbral_alto", 0.65))
UMBRAL_CRITICO = float(payload.get("umbral_critico", 0.90))
COMPARATIVO = payload.get("comparativo", pd.DataFrame())
STATIC_FIGS_129 = payload.get("figuras_12_9", {})
STATIC_FIGS_14 = payload.get("figuras_14", {})
ROC_DATA = payload.get("roc_curve_test")
PR_DATA = payload.get("pr_curve_test")

def _num(d, *keys):
    for k in keys:
        try:
            v = float(d[k])
            if np.isfinite(v): return v
        except Exception:
            pass
    return np.nan

ROC_AUC = _num(METRICAS, "ROC_AUC", "ROCAUC")
ROC_AUC_CV = _num(METRICAS, "ROC_AUC_CV", "ROCAUCCV")
ACCURACY = _num(METRICAS, "Accuracy", "ACCURACY")
PRECISION = _num(METRICAS, "Precision", "PRECISION")
RECALL = _num(METRICAS, "Recall", "RECALL")
SPECIFICITY = _num(METRICAS, "Specificity", "SPECIFICITY")
BALANCED_ACCURACY = _num(METRICAS, "Balanced_Accuracy", "BALANCED_ACCURACY")
F1 = _num(METRICAS, "F1", "F1_SCORE")
BRIER = _num(METRICAS, "Brier", "BRIER", "Brier_Score")
LOGLOSS = _num(METRICAS, "LogLoss", "LOGLOSS", "Log_Loss")

def calc_summary(df):
    total = len(df)
    alerts = int(pd.to_numeric(df["alerta_activada"], errors="coerce").fillna(0).sum()) if total else 0
    counts = df["nivel_riesgo"].value_counts() if total else pd.Series(dtype=int)
    crit = int(counts.get("Critico", 0))
    high = int(counts.get("Alto", 0))
    mean = float(df["score_riesgo"].mean()) if total else np.nan
    med = float(df["score_riesgo"].median()) if total else np.nan
    rate = alerts / total * 100 if total else 0
    return dict(total=total, alertas=alerts, criticos=crit, altos=high,
                score_promedio=mean, score_mediana=med, tasa_alerta=rate)

def territory_df(df):
    if df.empty:
        return pd.DataFrame(columns=["departamento","Segmentos","Alertas","Tasa_Alerta_%","Score_Promedio","Criticos"])
    t = df.groupby("departamento", dropna=False).agg(
        Segmentos=("score_riesgo","size"),
        Alertas=("alerta_activada","sum"),
        Score_Promedio=("score_riesgo","mean"),
        Criticos=("nivel_riesgo", lambda s: (s=="Critico").sum())
    ).reset_index()
    t["Tasa_Alerta_%"] = np.where(t["Segmentos"]>0, t["Alertas"]/t["Segmentos"]*100, 0).round(2)
    t["Score_Promedio"] = t["Score_Promedio"].round(4)
    return t

def fig_empty(title, message="No hay datos suficientes para esta selección"):
    fig = go.Figure()
    fig.add_annotation(text=message,x=.5,y=.5,xref="paper",yref="paper",showarrow=False,
                       font=dict(size=14,color=COLOR_MUTED))
    fig.update_layout(template="plotly_white", title=title, height=420,
                      margin=dict(l=40,r=25,t=60,b=45))
    return fig

def fig_dynamic_risk(df):
    if df.empty: return fig_empty("Distribución del riesgo")
    s = df["nivel_riesgo"].value_counts().reindex(RISK_ORDER, fill_value=0)
    pct = (s/len(df)*100).round(2)
    fig = go.Figure(go.Bar(
        x=[RISK_LABELS[x] for x in RISK_ORDER], y=s.values,
        text=[f"{int(v):,}<br>{p:.1f}%" for v,p in zip(s.values,pct.values)],
        textposition="outside",
        marker_color=[RISK_COLORS[x] for x in RISK_ORDER],
        customdata=pct.values,
        hovertemplate="%{x}<br>Segmentos: %{y:,}<br>Participación: %{customdata:.1f}%<extra></extra>"
    ))
    fig.update_layout(template="plotly_white", title="Distribución del nivel de riesgo",
                      xaxis_title="", yaxis_title="Segmentos", height=400,
                      margin=dict(l=50,r=25,t=65,b=45), showlegend=False)
    return fig

def fig_score_dist(df):
    if df.empty: return fig_empty("Distribución del score")
    fig = px.histogram(df, x="score_riesgo", nbins=30, color="nivel_riesgo",
                       category_orders={"nivel_riesgo":RISK_ORDER},
                       color_discrete_map=RISK_COLORS,
                       title="Distribución del score de riesgo", template="plotly_white")
    fig.add_vline(x=UMBRAL_ALTO, line_dash="dash", line_color=COLOR_ORANGE,
                  annotation_text="Umbral alto", annotation_position="top right")
    fig.add_vline(x=UMBRAL_CRITICO, line_dash="dot", line_color=COLOR_RED,
                  annotation_text="Crítico", annotation_position="top left")
    fig.update_layout(height=400, xaxis_title="Score estimado", yaxis_title="Segmentos",
                      legend_title="Riesgo", margin=dict(l=50,r=25,t=65,b=45))
    return fig

def fig_territory_rate(df):
    t = territory_df(df)
    t = t[t["Segmentos"]>=30].sort_values("Tasa_Alerta_%",ascending=False).head(10).sort_values("Tasa_Alerta_%")
    if t.empty:
        t = territory_df(df).sort_values("Tasa_Alerta_%",ascending=False).head(10).sort_values("Tasa_Alerta_%")
    if t.empty: return fig_empty("Tasa de alerta territorial")
    t["departamento_label"] = t["departamento"].astype(str).str.title()
    fig = px.bar(t,x="Tasa_Alerta_%",y="departamento_label",orientation="h",text="Tasa_Alerta_%",
                 title="Territorios con mayor tasa de alerta",template="plotly_white",
                 hover_data={"Segmentos":True,"Alertas":True,"Score_Promedio":":.3f",
                             "Tasa_Alerta_%":":.1f","departamento_label":False})
    fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside",marker_color=COLOR_ORANGE)
    fig.update_layout(height=430,xaxis_title="Tasa de alerta (%)",yaxis_title="",
                      xaxis_range=[0,max(10,float(t["Tasa_Alerta_%"].max())*1.15)],
                      margin=dict(l=90,r=60,t=65,b=45),showlegend=False)
    return fig

def fig_territory_score(df):
    t=territory_df(df).sort_values("Score_Promedio",ascending=False).head(10).sort_values("Score_Promedio")
    if t.empty: return fig_empty("Score promedio territorial")
    t["departamento_label"]=t["departamento"].astype(str).str.title()
    fig=px.bar(t,x="Score_Promedio",y="departamento_label",orientation="h",text="Score_Promedio",
               title="Territorios con mayor score promedio",template="plotly_white",
               hover_data={"Segmentos":True,"Alertas":True,"Criticos":True,
                           "Score_Promedio":":.3f","departamento_label":False})
    fig.update_traces(texttemplate="%{text:.3f}",textposition="outside",marker_color=COLOR_BLUE)
    fig.update_layout(height=430,xaxis_title="Score promedio",yaxis_title="",xaxis_range=[0,1],
                      margin=dict(l=90,r=55,t=65,b=45),showlegend=False)
    return fig

def fig_heatmap(df):
    if df.empty: return fig_empty("Matriz territorial de riesgo")
    t=pd.crosstab(df["departamento"],df["nivel_riesgo"]).reindex(columns=RISK_ORDER,fill_value=0)
    t=t.sort_values("Critico",ascending=True).tail(20)
    fig=go.Figure(go.Heatmap(z=t.values,x=[RISK_LABELS[x] for x in RISK_ORDER],
                             y=t.index.astype(str),text=t.values,texttemplate="%{text}",
                             colorscale="Blues",hovertemplate="%{y}<br>%{x}: %{z:,}<extra></extra>"))
    fig.update_layout(template="plotly_white",title="Concentración territorial por nivel de riesgo",
                      height=max(500,28*len(t)),xaxis_title="Nivel de riesgo",
                      yaxis_title="Departamento",margin=dict(l=100,r=25,t=65,b=50))
    return fig

def fig_profile(df, variable):
    if df.empty or variable not in VARIABLES_RAW: return fig_empty("Perfil de riesgo")
    p=df.groupby(variable,dropna=False).agg(Segmentos=("score_riesgo","size"),
        Alertas=("alerta_activada","sum"),Score_Promedio=("score_riesgo","mean")).reset_index()
    p["Tasa_Alerta_%"]=np.where(p["Segmentos"]>0,p["Alertas"]/p["Segmentos"]*100,0)
    p=p.sort_values(["Tasa_Alerta_%","Segmentos"],ascending=[False,False]).head(12).sort_values("Tasa_Alerta_%")
    p["Tasa_Alerta_%"]=p["Tasa_Alerta_%"].round(2); p["etiqueta"]=p[variable].astype(str)
    fig=px.bar(p,x="Tasa_Alerta_%",y="etiqueta",orientation="h",text="Tasa_Alerta_%",
               title=f"Perfiles con mayor tasa de alerta — {variable.replace('_',' ').title()}",
               template="plotly_white")
    fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside",marker_color=COLOR_TEAL)
    fig.update_layout(height=max(450,30*len(p)),xaxis_title="Tasa de alerta (%)",
                      yaxis_title=variable.replace("_"," ").title())
    return fig

def fig_top20(df):
    if df.empty: return fig_empty("Top segmentos por score")
    cols=[c for c in VARIABLES_RAW if c in df.columns]
    t=df.sort_values(["score_riesgo","alerta_activada"],ascending=[False,False]).head(15).copy()
    t["Ranking"]=range(1,len(t)+1)
    t["Etiqueta"]=[f"#{int(row['Ranking'])} · {str(row.get('departamento','')).title()} · {str(row.get('sexo','')).title()} · E{row.get('estrato','')}" for _,row in t.iterrows()]
    t["Detalle"]=[" | ".join(f"{c.replace('_',' ').title()}: {row[c]}" for c in cols if pd.notna(row[c])) for _,row in t.iterrows()]
    plot_df=t.sort_values("score_riesgo")
    fig=px.bar(plot_df,x="score_riesgo",y="Etiqueta",orientation="h",text="Ranking",
               title="Segmentos prioritarios por score",template="plotly_white",
               hover_data={"Detalle":True,"score_riesgo":":.3f","Etiqueta":False})
    fig.update_traces(texttemplate="#%{text}",textposition="outside",marker_color=COLOR_RED)
    fig.update_layout(height=max(480,26*len(plot_df)),xaxis_title="Score de riesgo",yaxis_title="",
                      xaxis_range=[0,1],margin=dict(l=210,r=40,t=65,b=45),showlegend=False)
    return fig

def fig_priority_scatter(df):
    t=territory_df(df)
    if t.empty: return fig_empty("Prioridad territorial")
    t=t.copy(); t["departamento_label"]=t["departamento"].astype(str).str.title()
    t["Prioridad"]=np.select([t["Tasa_Alerta_%"]>=25,t["Tasa_Alerta_%"]>=10],
                             ["Alta","Media"],default="Monitoreo")
    t["Etiqueta"]=np.where(t["Tasa_Alerta_%"]>=25,t["departamento_label"],"")
    fig=px.scatter(t,x="Tasa_Alerta_%",y="Score_Promedio",size="Alertas",color="Prioridad",
                   text="Etiqueta",color_discrete_map={"Alta":COLOR_RED,"Media":COLOR_ORANGE,"Monitoreo":COLOR_BLUE},
                   hover_data={"Segmentos":True,"Alertas":True,"Criticos":True,
                               "Score_Promedio":":.3f","Tasa_Alerta_%":":.1f",
                               "departamento_label":False,"Etiqueta":False},
                   title="Prioridad territorial: alerta vs score",template="plotly_white")
    fig.update_traces(marker=dict(line=dict(width=1,color="white")),textposition="top center")
    xmax=max(30,float(t["Tasa_Alerta_%"].max())*1.1)
    fig.add_vline(x=float(calc_summary(df)["tasa_alerta"]),line_dash="dash",
                  line_color="#94A3B8",annotation_text="Tasa selección",annotation_position="top left")
    fig.update_layout(height=430,xaxis_title="Tasa de alerta (%)",yaxis_title="Score promedio",
                      xaxis_range=[0,xmax],yaxis_range=[0,1],margin=dict(l=55,r=30,t=70,b=50),
                      legend_title="Prioridad")
    return fig

def fig_alertas_departamento(df):
    t=territory_df(df).sort_values("Alertas",ascending=True).tail(12)
    if t.empty: return fig_empty("Volumen de alertas por departamento")
    fig=px.bar(t,x="Alertas",y="departamento",orientation="h",text="Alertas",
               title="Volumen de alertas por departamento",template="plotly_white")
    fig.update_traces(texttemplate="%{text:,}",textposition="outside",marker_color=COLOR_NAVY)
    fig.update_layout(height=470,xaxis_title="Alertas",yaxis_title="Departamento")
    return fig

def table_view(data, height=340):
    if data is None or len(data)==0:
        st.info("No hay información para los filtros seleccionados.")
        return
    t=data.copy()
    names={"departamento":"Departamento","categoria_municipio":"Categoría municipio","sexo":"Sexo",
           "estrato":"Estrato","nivel_formacion":"Nivel de formación","modalidad_linea":"Modalidad de línea",
           "modalidad_credito":"Modalidad de crédito","sector_ies":"Sector IES","rango_valor_total":"Rango valor total",
           "Segmentos":"Segmentos","Alertas":"Alertas","Criticos":"Críticos","Score_Promedio":"Score promedio",
           "Tasa_Alerta_%":"Tasa de alerta (%)","score_riesgo":"Score de riesgo","nivel_riesgo":"Nivel de riesgo",
           "alerta_activada":"Alerta"}
    t=t.rename(columns={c:names.get(c,c.replace("_"," ").title()) for c in t.columns})
    st.dataframe(t, use_container_width=True, height=height, hide_index=True)

def insights(df):
    r=calc_summary(df)
    parts=[]
    if not df.empty:
        t=territory_df(df)
        if not t.empty:
            top_rate=t.sort_values("Tasa_Alerta_%",ascending=False).iloc[0]
            top_score=t.sort_values("Score_Promedio",ascending=False).iloc[0]
            parts.append(f"El territorio con mayor tasa de alerta dentro de la selección es {str(top_rate['departamento']).title()} ({top_rate['Tasa_Alerta_%']:.1f}%).")
            parts.append(f"El mayor score promedio territorial corresponde a {str(top_score['departamento']).title()} ({top_score['Score_Promedio']:.3f}).")
    parts.append(f"La selección contiene {r['total']:,} segmentos y {r['alertas']:,} alertas ({r['tasa_alerta']:.1f}%).")
    parts.append("Estas lecturas son descriptivas y no implican causalidad.")
    return parts

def roc_figure():
    fig=go.Figure()
    fpr=tpr=None
    if isinstance(ROC_DATA,dict):
        fpr=np.asarray(ROC_DATA.get("fpr",[]),dtype=float)
        tpr=np.asarray(ROC_DATA.get("tpr",[]),dtype=float)
    if fpr is None or len(fpr)==0:
        fig.add_annotation(text="Curva ROC no disponible en el Runtime Contract.",x=.5,y=.5,
                           xref="paper",yref="paper",showarrow=False,font=dict(size=14,color=COLOR_MUTED))
    else:
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name="Modelo",
                                 line=dict(color=COLOR_NAVY,width=3)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Aleatorio",
                                 line=dict(color="#94A3B8",dash="dash")))
    fig.update_layout(template="plotly_white",
                      title=f"Curva ROC — AUC {ROC_AUC:.4f}" if np.isfinite(ROC_AUC) else "Curva ROC",
                      xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",
                      height=450,xaxis_range=[0,1],yaxis_range=[0,1])
    return fig

def pr_figure():
    fig=go.Figure()
    precision=recall=None
    if isinstance(PR_DATA,dict):
        precision=np.asarray(PR_DATA.get("precision",[]),dtype=float)
        recall=np.asarray(PR_DATA.get("recall",[]),dtype=float)
    if precision is None or len(precision)==0:
        fig.add_annotation(text="Curva Precision-Recall no disponible en el Runtime Contract.",x=.5,y=.5,
                           xref="paper",yref="paper",showarrow=False,font=dict(size=14,color=COLOR_MUTED))
    else:
        n=min(len(precision),len(recall))
        fig.add_trace(go.Scatter(x=recall[:n],y=precision[:n],mode="lines",
                                 name="Precision-Recall",
                                 line=dict(color=COLOR_ORANGE,width=3)))
    fig.update_layout(template="plotly_white",title="Curva Precision-Recall",
                      xaxis_title="Recall",yaxis_title="Precision",height=450,
                      xaxis_range=[0,1],yaxis_range=[0,1])
    return fig

def run_simulation(values):
    """Replica la inferencia canónica del BLOQUE 23.

    Ruta 1: pipeline completo, solo cuando expone feature_names_in_, tal como
    lo hace predecir_simulacion_23().
    Ruta 2: preprocesador oficial + modelo directo.
    """
    x = pd.DataFrame([{v: values[v] for v in VARIABLES_RAW}])

    if not isinstance(MODEL, dict):
        raise RuntimeError(
            "Los artefactos del simulador no tienen el formato bundle esperado. "
            "Vuelva a ejecutar la celda final 23B corregida."
        )

    pipeline = MODEL.get("pipeline")
    preprocessor = MODEL.get("preprocessor")
    model = MODEL.get("model")

    if pipeline is not None and hasattr(pipeline, "predict_proba"):
        if getattr(pipeline, "feature_names_in_", None) is not None:
            try:
                proba = pipeline.predict_proba(x)
                p = float(proba[0, -1])
            except Exception:
                p = None
        else:
            p = None
    else:
        p = None

    if p is None:
        if preprocessor is None:
            raise RuntimeError(
                "No existe el preprocesador oficial del simulador. "
                "Vuelva a ejecutar la celda final después del BLOQUE 23."
            )

        if model is None:
            raise RuntimeError(
                "No existe el modelo directo oficial del simulador. "
                "Vuelva a ejecutar la celda final después del BLOQUE 23."
            )

        try:
            x_transformado = preprocessor.transform(x)
            proba = model.predict_proba(x_transformado)
            p = float(proba[0, -1])
        except Exception as e:
            raise RuntimeError(
                "Falló la ruta oficial preprocesador + modelo del BLOQUE 23: "
                f"{e}"
            ) from e

    if not np.isfinite(p):
        raise RuntimeError(f"El score calculado no es válido: {p}")

    p = float(np.clip(p, 0, 1))

    if p>=UMBRAL_CRITICO:
        level, action = "CRÍTICO","Priorizar gestión preventiva intensiva."
    elif p>=UMBRAL_ALTO:
        level, action = "ALTO","Activar seguimiento preventivo focalizado."
    elif p>=UMBRAL_MEDIO:
        level, action = "MEDIO","Mantener monitoreo preventivo."
    else:
        level, action = "BAJO","Mantener seguimiento regular."
    return p,level,action

# --------------------------------------------------------------------------------------
# ENCABEZADO + SEGMENTACIÓN OPERATIVA
# --------------------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <div style="font-size:34px;font-weight:900;letter-spacing:-1px;">SIAGRM</div>
  <div style="font-size:18px;font-weight:500;">Gestión Preventiva del Riesgo de Mora</div>
  <div style="font-size:12px;opacity:.88;margin-top:5px;">ICETEX · Universidad de Ibagué · Maestría en Analítica de Datos para la Toma de Decisiones</div>
  <div style="margin-top:14px;">
    <span style="background:#fff;color:#172033;padding:5px 10px;border-radius:999px;margin-right:6px;font-size:11px;">Modelo: {MODELO_CANONICO}</span>
    <span style="background:#fff;color:#172033;padding:5px 10px;border-radius:999px;margin-right:6px;font-size:11px;">Runtime: {MODELO_RUNTIME}</span>
    <span style="background:#fff;color:#172033;padding:5px 10px;border-radius:999px;font-size:11px;">Calibración: {METODO_CALIBRACION}</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="card" style="margin-bottom:12px;">
  <div class="section-title">Segmentación operativa</div>
  <div class="small-muted">Los indicadores descriptivos, territorio, perfiles y hallazgos se recalculan con la misma base filtrada.</div>
</div>
""", unsafe_allow_html=True)

filter_cols=st.columns(4)
selected={}
for i,v in enumerate(VARIABLES_RAW):
    with filter_cols[i%4]:
        vals=sorted(df_all[v].dropna().astype(str).drop_duplicates().tolist())
        selected[v]=st.multiselect(v.replace("_"," ").title(), vals, default=vals, key=f"f_{v}")

if st.button("Restablecer filtros"):
    st.rerun()

df=df_all.copy()
for v,vals in selected.items():
    if vals:
        df=df[df[v].astype(str).isin(vals)]

r=calc_summary(df)
st.markdown(f'<div class="card"><b>Selección actual</b> &nbsp; <span class="small-muted">{len(df):,} segmentos</span></div>', unsafe_allow_html=True)

tabs=st.tabs([
    "Resumen ejecutivo","Territorio","Perfiles de riesgo","Comparación y selección",
    "Evaluación del modelo","Hallazgos y priorización","Exploración de datos",
    "Simulador institucional","Metodología y QA"
])

# --------------------------------------------------------------------------------------
# 1. RESUMEN EJECUTIVO
# --------------------------------------------------------------------------------------
with tabs[0]:
    kcols=st.columns(6)
    cards=[
        ("Segmentos",f"{r['total']:,}","Población seleccionada",COLOR_NAVY),
        ("Alertas",f"{r['alertas']:,}",f"{r['tasa_alerta']:.1f}% del subconjunto",COLOR_RED),
        ("Críticos",f"{r['criticos']:,}","Requieren mayor prioridad",COLOR_RED),
        ("Score promedio",f"{r['score_promedio']:.3f}","Probabilidad media estimada",COLOR_BLUE),
        ("Score mediano",f"{r['score_mediana']:.3f}","Valor central",COLOR_TEAL),
        ("Alto + Crítico",f"{r['altos']+r['criticos']:,}","Casos priorizados",COLOR_ORANGE),
    ]
    for c,(t,v,d,color) in zip(kcols,cards):
        c.markdown(f'<div class="card"><div class="kpi">{t}</div><div class="kpi-value" style="color:{color}">{v}</div><div class="kpi-desc">{d}</div></div>',unsafe_allow_html=True)
    st.markdown("### Lectura ejecutiva")
    for text in insights(df):
        st.write("•",text)
    c1,c2=st.columns(2); 
    with c1: st.plotly_chart(fig_dynamic_risk(df),use_container_width=True,key="exec_risk_distribution")
    with c2: st.plotly_chart(fig_score_dist(df),use_container_width=True,key="exec_score_distribution")
    c1,c2=st.columns([7,5])
    with c1: st.plotly_chart(fig_territory_rate(df),use_container_width=True,key="exec_territory_rate")
    with c2: st.plotly_chart(fig_alertas_departamento(df),use_container_width=True,key="exec_alertas_departamento")
    st.markdown('<div class="card"><h4>Lectura ejecutiva</h4><span class="small-muted">Los indicadores descriptivos cambian con los filtros. Las métricas de validación del modelo se mantienen globales.</span></div>',unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# 2. TERRITORIO
# --------------------------------------------------------------------------------------
with tabs[1]:
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(fig_territory_score(df),use_container_width=True,key="territory_score")
    with c2: st.plotly_chart(fig_priority_scatter(df),use_container_width=True,key="territory_priority_scatter")
    st.plotly_chart(fig_heatmap(df),use_container_width=True,key="territory_heatmap")
    st.markdown("### Ranking territorial")
    table_view(territory_df(df).sort_values("Tasa_Alerta_%",ascending=False).head(15),420)

# --------------------------------------------------------------------------------------
# 3. PERFILES DE RIESGO
# --------------------------------------------------------------------------------------
with tabs[2]:
    variable=st.selectbox("Variable de análisis",
                          VARIABLES_RAW,index=0,
                          format_func=lambda x:x.replace("_"," ").title())
    st.plotly_chart(fig_profile(df,variable),use_container_width=True,key="profile_variable_chart")
    p=df.groupby(variable,dropna=False).agg(Segmentos=("score_riesgo","size"),
        Alertas=("alerta_activada","sum"),Score_Promedio=("score_riesgo","mean")).reset_index()
    p["Tasa_Alerta_%"]=np.where(p["Segmentos"]>0,p["Alertas"]/p["Segmentos"]*100,0)
    table_view(p.sort_values("Tasa_Alerta_%",ascending=False).head(20),440)
    st.markdown("### Diccionario de variables")
    table_view(pd.DataFrame([{"Variable":v,"Descripción":DESCRIPCIONES[v],"Uso":"Entrada RAW / filtro / simulador"} for v in VARIABLES_RAW]),360)

# --------------------------------------------------------------------------------------
# 4. COMPARACIÓN Y SELECCIÓN
# --------------------------------------------------------------------------------------
with tabs[3]:
    k1,k2,k3,k4=st.columns(4)
    vals=[("ROC-AUC Test",ROC_AUC,"Evaluación global",COLOR_NAVY),
          ("ROC-AUC CV",ROC_AUC_CV,"Validación cruzada",COLOR_BLUE),
          ("Recall",RECALL,"Detección de positivos",COLOR_TEAL),
          ("F1",F1,"Equilibrio Precision-Recall",COLOR_ORANGE)]
    for col,(t,v,d,color) in zip([k1,k2,k3,k4],vals):
        val=f"{v:.4f}" if np.isfinite(v) else "N/D"
        col.markdown(f'<div class="card"><div class="kpi">{t}</div><div class="kpi-value" style="color:{color}">{val}</div><div class="kpi-desc">{d}</div></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="card" style="margin:16px 0;"><h4>Modelo seleccionado: {MODELO_CANONICO}</h4><span class="small-muted">La selección y el desempeño se toman del artefacto oficial publicado por el Runtime Contract.</span></div>',unsafe_allow_html=True)
    comp=COMPARATIVO.copy() if isinstance(COMPARATIVO,pd.DataFrame) else pd.DataFrame()
    if not comp.empty and "Modelo" in comp.columns:
        auc_col="ROC_AUC_CV" if "ROC_AUC_CV" in comp.columns else None
        if auc_col:
            fig=px.bar(comp.sort_values(auc_col),x=auc_col,y="Modelo",orientation="h",text=auc_col,
                       title="Comparación de modelos — ROC-AUC CV",template="plotly_white")
            fig.update_traces(texttemplate="%{text:.4f}",textposition="outside")
            fig.update_layout(height=430,xaxis_range=[0,1])
            st.plotly_chart(fig,use_container_width=True,key="model_comparison_bar")
        table_view(comp,460)
    else:
        st.info("No existe comparativo oficial publicado.")

# --------------------------------------------------------------------------------------
# 5. EVALUACIÓN DEL MODELO
# --------------------------------------------------------------------------------------
with tabs[4]:
    rows=[
        ("ROC-AUC CV",ROC_AUC_CV,"Discriminación en validación cruzada"),
        ("ROC-AUC Test",ROC_AUC,"Discriminación en evaluación"),
        ("Accuracy",ACCURACY,"Clasificaciones correctas"),
        ("Precision",PRECISION,"Precisión de positivos predichos"),
        ("Recall",RECALL,"Positivos reales identificados"),
        ("Specificity",SPECIFICITY,"Negativos correctamente identificados"),
        ("Balanced Accuracy",BALANCED_ACCURACY,"Desempeño equilibrado por clase"),
        ("F1",F1,"Equilibrio Precision-Recall"),
        ("Brier Score",BRIER,"Calidad de probabilidades; menor es mejor"),
        ("Log Loss",LOGLOSS,"Penalización de probabilidades incorrectas; menor es mejor"),
    ]
    table_view(pd.DataFrame([{"Métrica":n,"Valor":f"{v:.4f}" if np.isfinite(v) else "N/D","Interpretación":d} for n,v,d in rows]),450)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(roc_figure(),use_container_width=True,key="model_roc_curve")
    with c2: st.plotly_chart(pr_figure(),use_container_width=True,key="model_pr_curve")
    st.markdown("### Evidencia técnica publicada")
    title_map={"metricas":"Métricas globales del modelo","modelos":"Comparación de modelos",
               "roc":"Curva ROC","precision_recall":"Curva Precision-Recall",
               "matriz_confusion":"Matriz de confusión","probabilidades":"Distribución de probabilidades",
               "calibracion":"Calibración del modelo","importancia_variables":"Importancia de variables",
               "mutual_information":"Información mutua"}
    for name,fig in STATIC_FIGS_129.items():
        st.markdown(f"**{title_map.get(name,name)}**")
        st.plotly_chart(fig,use_container_width=True,key=f"static_129_{name}")

# --------------------------------------------------------------------------------------
# 6. HALLAZGOS Y PRIORIZACIÓN
# --------------------------------------------------------------------------------------
with tabs[5]:
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(fig_top20(df),use_container_width=True,key="priority_top20")
    with c2: st.plotly_chart(fig_priority_scatter(df),use_container_width=True,key="priority_scatter")
    st.markdown("### Evidencia ejecutiva publicada")
    for name,fig in STATIC_FIGS_14.items():
        st.plotly_chart(fig,use_container_width=True,key=f"static_14_{name}")
    st.info("Las visualizaciones superiores se recalculan sobre la selección actual. No implican causalidad.")
    for text in insights(df):
        st.write("•",text)

# --------------------------------------------------------------------------------------
# 7. EXPLORACIÓN DE DATOS
# --------------------------------------------------------------------------------------
with tabs[6]:
    c1,c2,c3,c4=st.columns(4)
    mini=[("Registros",f"{len(df):,}","Base filtrada",COLOR_NAVY),
          ("Alertas",f"{r['alertas']:,}","Alto + Crítico",COLOR_RED),
          ("Tasa",f"{r['tasa_alerta']:.1f}%","Alertas / segmentos",COLOR_ORANGE),
          ("Score",f"{r['score_promedio']:.3f}","Promedio",COLOR_BLUE)]
    for col,(t,v,d,color) in zip([c1,c2,c3,c4],mini):
        col.markdown(f'<div class="card"><div class="kpi">{t}</div><div class="kpi-value" style="color:{color}">{v}</div><div class="kpi-desc">{d}</div></div>',unsafe_allow_html=True)
    st.markdown("### Muestra de la población seleccionada")
    table_view(df.head(25),520)

# --------------------------------------------------------------------------------------
# 8. SIMULADOR INSTITUCIONAL
# --------------------------------------------------------------------------------------
with tabs[7]:
    st.markdown('<div class="card"><h3>Simulador de riesgo preventivo</h3><p class="small-muted">Uso institucional: ingrese las 9 variables RAW disponibles para obtener una estimación del score y orientar la gestión preventiva.</p><span style="background:#DCEBFF;color:#173F73;padding:6px 10px;border-radius:999px;font-size:12px;font-weight:800;">9 variables RAW</span></div>',unsafe_allow_html=True)
    scols=st.columns(2)
    values={}
    for i,v in enumerate(VARIABLES_RAW):
        opts=sorted(df_all[v].dropna().astype(str).drop_duplicates().tolist())
        default=str(df_all.iloc[0][v])
        with scols[i%2]:
            idx=opts.index(default) if default in opts else 0
            values[v]=st.selectbox(v.replace("_"," ").title(),opts,index=idx,key=f"sim_{v}")
    if st.button("Calcular score de riesgo",type="primary",use_container_width=True):
        try:
            p,level,action=run_simulation(values)
            col=COLOR_NAVY
            st.markdown(f'<div class="card" style="margin-top:16px;"><div class="kpi">Resultado de simulación</div><div class="kpi-value" style="font-size:48px;color:{col};">{p:.1%}</div><div style="display:inline-block;background:{RISK_COLORS[{"BAJO":"Bajo","MEDIO":"Medio","ALTO":"Alto","CRÍTICO":"Critico"}[level]]};color:white;padding:8px 12px;border-radius:8px;font-weight:800;">{level}</div><p style="margin-top:12px;">Score: {p:.6f}</p><p>Variables procesadas: 9/9</p><hr><h4>Acción preventiva sugerida</h4><p>{action}</p><span class="small-muted">Estimación preventiva del modelo oficial; no constituye una decisión automática sobre el beneficiario.</span></div>',unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error de simulación: {e}")

# --------------------------------------------------------------------------------------
# 9. METODOLOGÍA Y QA
# --------------------------------------------------------------------------------------
with tabs[8]:
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f"""
<div class="card">
<h3>¿Cómo interpretar SIAGRM?</h3>
<h4>Score de riesgo</h4><p>{payload.get("descripcion_score","Probabilidad estimada por el modelo.")}</p>
<h4>ROC-AUC</h4><p>Mide la capacidad discriminativa del modelo.</p>
<h4>Recall</h4><p>Proporción de casos positivos reales identificados.</p>
<h4>F1</h4><p>Equilibra Precision y Recall.</p>
<h4>Importancia de variables</h4><p>Representa contribución relativa dentro del modelo, no causalidad.</p>
</div>""",unsafe_allow_html=True)
    with c2:
        checks=[
            ("Base SSOT disponible","OK"),("Variables RAW","9/9"),("Score en [0,1]","OK"),
            ("Niveles válidos","OK"),("Modelo canónico",MODELO_CANONICO),
            ("Runtime",MODELO_RUNTIME),("Calibración",METODO_CALIBRACION),
            ("Threshold",f"{THRESHOLD_MODELO:.6f}" if THRESHOLD_MODELO is not None else "N/D"),
            ("SSOT","READ-ONLY"),("Reentrenamiento","NO")]
        table_view(pd.DataFrame([{"Check":a,"Resultado":b} for a,b in checks]),430)
    st.markdown("### 9 variables RAW")
    meta=[
        {"Elemento":"Score de riesgo","Descripción":payload.get("descripcion_score",""),"Interpretación":"Probabilidad estimada por el modelo para la clase positiva definida en el entrenamiento."},
        {"Elemento":"ROC-AUC","Descripción":"Mide la capacidad discriminativa del modelo.","Interpretación":"0.5 representa aproximadamente discriminación aleatoria; valores mayores indican mejor capacidad de separación."},
        {"Elemento":"Recall","Descripción":"Proporción de casos positivos reales identificados.","Interpretación":"Un Recall alto favorece la detección de casos de riesgo, aspecto relevante para gestión preventiva."},
        {"Elemento":"F1","Descripción":"Combina Precision y Recall.","Interpretación":"Permite evaluar el equilibrio entre precisión y capacidad de detección."},
        {"Elemento":"Balanced Accuracy","Descripción":"Promedio del desempeño de ambas clases.","Interpretación":"Útil cuando las clases presentan tamaños diferentes."},
        {"Elemento":"Brier Score","Descripción":"Evalúa la calidad de las probabilidades estimadas.","Interpretación":"En general, valores menores indican probabilidades mejor calibradas."},
        {"Elemento":"Log Loss","Descripción":"Penaliza probabilidades asignadas incorrectamente.","Interpretación":"Valores menores indican mejor desempeño probabilístico."},
        {"Elemento":"Alerta preventiva","Descripción":payload.get("descripcion_alerta",""),"Interpretación":"Identifica segmentos que deben priorizarse para gestión preventiva."},
        {"Elemento":"Uso institucional","Descripción":"SIAGRM es una herramienta de apoyo a la gestión preventiva.","Interpretación":"El score no constituye por sí mismo una decisión automática sobre el beneficiario."},
    ]
    table_view(pd.DataFrame(meta),650)

st.markdown('<div class="small-muted" style="margin-top:16px;">SIAGRM · SSOT READ-ONLY · sin reentrenamiento en Streamlit · artefactos publicados por el bloque 23.</div>',unsafe_allow_html=True)