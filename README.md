# Visión y copiloto de línea con COIL-100

Prueba técnica AI Engineer. Sustituye la tarea manual de identificar cada producto y girarlo a su orientación de referencia: una cámara captura el objeto, un modelo de ML clásico dice qué producto es, en qué ángulo está y si es seguro que el robot actúe, y un asistente agéntico permite a supervisión, calidad, mantenimiento e ingeniería de ML operar la línea, resolver excepciones y gobernar los cambios de riesgo.

Desarrollado por César Eraso con desarrollo asistido por IA (Claude Code).

## Entregables

| # | Entregable | Dónde |
|---|---|---|
| 2 | Notebook de experimentación reproducible | [`notebooks/01_experimentacion.ipynb`](notebooks/01_experimentacion.ipynb) (generado por `notebooks/build_notebook.py`, ejecutado de punta a punta) |
| 3 | Reporte por protocolo, análisis de errores y model card | [`docs/reporte_vision.pdf`](docs/reporte_vision.pdf) y [`models/vision/v1.0.0/MODEL_CARD.md`](models/vision/v1.0.0/MODEL_CARD.md) |
| 5 | Repositorio con historial de commits | este repositorio |
| 7 | Catálogo de tools y contratos | [`docs/catalogo_tools.md`](docs/catalogo_tools.md), [`docs/tools_schema.json`](docs/tools_schema.json), [`docs/openapi_vision.json`](docs/openapi_vision.json) |
| 8 | Threat model y controles | [`docs/threat_model.md`](docs/threat_model.md) y [`SEGURIDAD-2026-10-28.md`](SEGURIDAD-2026-10-28.md) |
| 10 | Servicio desplegado | URL `https://ml-indra.isagis.com` (credenciales en el correo de entrega) y despliegue local de un comando (abajo) |
| 12 | Arquitectura Azure y modelo de costos | [`docs/azure/arquitectura_azure.md`](docs/azure/arquitectura_azure.md) y [`docs/azure/costos.xlsx`](docs/azure/costos.xlsx) |

Además: arquitectura y diagramas de secuencia en [`docs/arquitectura.md`](docs/arquitectura.md), decisiones en [`docs/adr/decisiones.md`](docs/adr/decisiones.md) y evaluación del agente en [`docs/evaluacion_agente.md`](docs/evaluacion_agente.md).

## Ejecutar en local con un comando

Requisitos: Docker con Compose v2. Opcional: una clave de OpenAI para el asistente (sin ella, el servicio de visión funciona y el chat responde que el asistente no está disponible).

```bash
cp .env.example .env          # y poner OPENAI_API_KEY si se quiere el asistente
docker compose up --build
```

El primer arranque descarga COIL-100 (unos 125 MB, verificado por SHA-256) y simula 24 horas de línea. Después:

- Chat del asistente: http://localhost:8000 (elegir usuario y rol; en la demo el rol se elige en pantalla).
- API de visión: http://localhost:8001/docs

## Estructura

```
src/vision/    datos, protocolos, características, modelo, política, calidad, métricas, experimentos, API de visión
src/agent/     grafo LangGraph, guardia, presupuesto, aprobaciones, 13 tools, API y chat
src/sim/       simulador de 24 h de línea (incidente de cámara a las 3:00 y de ventosa a las 17:00)
src/common/    configuración por entorno y esquema SQLite
models/vision/v1.0.0/   artefacto versionado (modelo, umbrales, manifiesto con SHA-256, métricas, model card)
eval/          casos y arnés de evaluación del agente, resultados del barrido de visión
mlruns/        experimentos registrados en MLflow (SQLite)
docs/          arquitectura, ADR, catálogo de tools, threat model, Azure, reporte
sops/          procedimientos sintéticos que consulta el asistente
tests/         pruebas automáticas
```

## Reproducir los experimentos y la evaluación

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt   # en Linux: .venv/bin/pip
python data/download.py
python -m src.vision.sweep --stage grid      # unos 60 min en CPU; registra en mlruns/
python -m src.vision.sweep --stage angle
python -m src.vision.sweep --stage export    # models/vision/v1.0.0
jupyter nbconvert --to notebook --execute --inplace notebooks/01_experimentacion.ipynb
python eval/make_assets.py && python eval/run_eval.py --repeats 1
pytest -q
```

## Resultados

Se completan en `docs/reporte_vision.md` y `docs/evaluacion_agente.md`.

## Privacidad y seguridad

Los datos de línea, operarios y procedimientos son sintéticos. El asistente nunca envía imágenes al LLM: trabaja con identificadores (SHA-256) y con la salida estructurada del modelo. Los controles y riesgos aceptados están en el threat model.
