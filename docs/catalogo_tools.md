# Catálogo de tools y contratos

Trece tools reales, cada una sobre una operación distinta del backend. Los esquemas JSON completos de entrada están en `docs/tools_schema.json` (generados desde los modelos Pydantic) y el contrato HTTP del servicio de visión en `docs/openapi_vision.json`.

## Reglas comunes

- Punto único de entrada: `run_tool` (`src/agent/tools/__init__.py`). Valida el esquema, comprueba el rol, ejecuta y audita. Antes, la guardia del grafo (`src/agent/guard.py`) corta bucles, el máximo de 8 llamadas por turno y el tiempo de 60 s.
- Errores estructurados hacia el LLM: `{"error": código, "detail": texto}`. Códigos: `unknown_tool`, `invalid_args`, `forbidden`, `not_found`, `conflict`, `invalid_request`, `image_not_found`, `insufficient_views`, `vision_timeout`, `vision_server`, `vision_bad_request`, `vision_circuit_open`. Los fallos de visión llevan la pista "no inventes el resultado".
- Contexto del servidor: `plant_id`, `line_id`, `user_id` y rol vienen de la sesión; ninguna tool los acepta como argumento.
- Lo que nunca recibe el LLM: imágenes, base64, rutas de archivo, nombres de operarios ni secretos. Las salidas largas se recortan a 4.000 caracteres.
- Auditoría: cada llamada (incluidas las bloqueadas por la guardia) deja fila en `audit` con fecha, sesión, usuario, rol, planta, línea, tool, SHA-256 de los argumentos, estado y detalle.

## Catálogo

| Tool | Tipo | Roles | Entrada (resumen) | Salida (resumen) | Fallos | Efecto repetido |
|---|---|---|---|---|---|---|
| `check_image_quality` | Lectura | todos | `image_id` (SHA-256) | ok, motivos, desenfoque, brillo | `image_not_found`, `vision_*` | Sin efecto |
| `analyze_image` | Lectura | todos | `image_id`, `top_k` 1-5 | producto, prob. calibrada, ángulo, incertidumbre, modo de ángulo, simetría, puntuación de agarre, umbral de línea, PICK/DIVERT con motivos, top-k (producto, ángulo, distancia) | `image_not_found`, `vision_*` | Sin efecto |
| `list_review_queue` | Lectura | todos | `limit` 1-20, filtro de motivo | total pendiente y casos con evidencia | ninguno | Sin efecto |
| `submit_review_label` | Escritura | supervisor, calidad | `case_id`, `obj_id` 1-100, `angle` 0-359 | creado sí/no | `not_found`, `conflict` | Idempotente por `case_id`: misma etiqueta → `created=false`; etiqueta distinta → `conflict` |
| `query_line_metrics` | Lectura | todos | turno 1-3 u horas, agrupación hora/turno/versión | filas con total, PICK, DIVERT, tasa, confianza y desenfoque medios | `invalid_args` (turno y horas a la vez) | Sin efecto |
| `get_camera_health` | Lectura | todos | horas | desenfoque, brillo, fotogramas perdidos y temperatura por hora, mediana de desenfoque | ninguno | Sin efecto |
| `get_grasp_events` | Lectura | todos | código de falla, horas | fallos por código y hora, 5 ejemplos con la decisión de visión | ninguno | Sin efecto |
| `search_procedures` | Lectura | todos | `query` ≤ 200, `top_k` 1-5 | fragmentos ≤ 400 caracteres con cita `doc#sección` y puntuación | lista vacía si nada supera 0,10 | Sin efecto |
| `create_ticket` | Escritura | mantenimiento, supervisor, calidad, gerente | título, prioridad, cuerpo, inspecciones relacionadas | `ticket_id`, creado sí/no, avisado a n8n | webhook caído no impide guardar | Idempotente: clave = SHA-256 de planta, línea, título y lista ordenada de inspecciones |
| `register_sku_candidate` | Escritura | ingeniero de ML, calidad | nombre, 6-72 `image_ids`, ángulos | `candidate_id`, estado registrado o insuficiente, hueco máximo | `invalid_args`, `image_not_found` | Cada llamada crea un candidato nuevo en área aislada (sin efecto en producción) |
| `evaluate_sku_candidate` | Lectura (cálculo) | ingeniero de ML, calidad | `candidate_id` | acierto de ángulo dejando una vista fuera, simetría, producto del catálogo más cercano, cumple sí/no con motivos | `not_found`, `insufficient_views`, `vision_*` | Recalcula y sobrescribe las métricas del candidato |
| `request_risk_operation` | Escritura (petición) | según operación | `op`, `params`, justificación | `approval_id`, `pending_approval`, roles aprobadores | `forbidden`, `invalid_request` (parámetros fuera de rango) | Cada llamada crea una petición; nunca ejecuta |
| `generate_shift_report` | Lectura | todos | turno 1-3 | total, tasa de desvío, incidentes detectados, fallos de agarre, tickets abiertos, casos pendientes, aprobaciones pendientes, estado de la línea | ninguno | Sin efecto |

## Operaciones de riesgo

| Operación | Parámetros | Puede pedirla | Puede aprobarla | Efecto al aprobarse |
|---|---|---|---|---|
| `set_threshold` | `min_pick_score` 0,5-0,999 | calidad, ingeniero de ML, supervisor | supervisor, gerente | cambia el umbral de la línea |
| `pause_line` / `resume_line` | ninguno | supervisor, mantenimiento, gerente | supervisor, gerente | pausa o reanuda la línea |
| `send_manual_angle` | `inspection_id`, `obj_id`, `angle` | supervisor, calidad | supervisor, gerente | comando PICK manual con el ángulo que escribe quien aprueba |
| `promote_model` | `version` registrada | ingeniero de ML | gerente | cambia la versión de esta planta y línea |
| `promote_sku` | `candidate_id` evaluado con éxito | ingeniero de ML, calidad | gerente | el producto pasa a producción |

Regla de cuatro ojos: el aprobador debe ser una persona distinta del solicitante. La decisión es una transición atómica (`UPDATE ... WHERE status='pending'`), así que dos aprobaciones simultáneas no ejecutan dos veces.

## Contrato del servicio de visión

| Endpoint | Entrada | Salida | Errores |
|---|---|---|---|
| `GET /health` | ninguna | estado y versión | ninguno |
| `GET /v1/models` | ninguna | manifiesto del modelo | ninguno |
| `POST /v1/quality` | archivo PNG/JPEG ≤ 5 MB | `image_id` y calidad | 400 `invalid_image` |
| `POST /v1/predict` | archivo, `top_k` 1-10 | calidad, predicción, decisión, versión, latencia | 400, 422, 500 genérico con `error_id` |
| `POST /v1/sku/evaluate` | 6-72 archivos y ángulos | métricas del candidato | 400 |

En modo demo (`DEMO_FAULTS=true`), la cabecera `X-Demo-Fault: 500` devuelve 503 y `timeout` tarda 15 s. Sirve para los escenarios adversos. En la ruta pública la cabecera se elimina.
