# SIAGRM — Streamlit

Este paquete es un espejo operacional del dashboard del BLOQUE 23.

## Importante
- `app.py` se entrega por separado y NO se incluye dentro de este ZIP, según la solicitud.
- La aplicación funciona en modo `SSOT_READ_ONLY`.
- No reentrena modelos.
- No genera datos sintéticos.
- La base y el pipeline del simulador son artefactos publicados desde el BLOQUE 23.

## Estructura
- `data/segments.parquet` — base canónica `SEGMENTOS_23`.
- `data/segments.csv` — respaldo legible.
- `models/simulador.joblib` — pipeline oficial del simulador.
- `models/preprocessor.joblib` — preprocesador oficial usado por la inferencia canónica.
- `models/simulador_bundle.joblib` — bundle con pipeline + preprocesador + modelo directo.
- `artifacts/dashboard_payload.joblib` — métricas, umbrales, curvas, textos y figuras publicadas.
- `artifacts/manifest_siagrm_streamlit.json` — trazabilidad.
- `requirements.txt`
- `.streamlit/config.toml`

## Ejecución
Copiar `app.py` en la raíz de este proyecto y ejecutar:

```bash
pip install -r requirements.txt
streamlit run app.py
```
