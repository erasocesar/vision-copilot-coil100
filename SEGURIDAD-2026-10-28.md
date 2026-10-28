# Seguridad: revisión antes de publicar

Fases aplicadas: planificación (lista de control al diseñar), controles al programar (revisión automática en cada commit con el plugin `security-guidance`), análisis de dependencias y escaneo de Claude Security antes del despliegue. El detalle de amenazas y controles está en [`docs/threat_model.md`](docs/threat_model.md).

## Lista de control

| Punto | Aplica | Cómo se resolvió |
|---|---|---|
| Acceso a recursos de otro usuario o planta (IDOR) | Sí | `plant_id` y `line_id` salen de la sesión; cada consulta filtra por ellos; un `case_id` ajeno devuelve "no encontrado" |
| Inyección SQL | Sí | Solo consultas parametrizadas de `sqlite3`; test con `'; DROP TABLE` |
| Claves en el repositorio | Sí | `.env` en `.gitignore`, `.env.example` vacío, búsqueda de patrones `sk-` antes de cada push |
| Contraseñas propias | Sí | La app no guarda contraseñas; la demo usa basic auth de Traefik con hash bcrypt |
| Webhooks | Parcial | No hay webhooks entrantes; el aviso saliente a n8n lleva cabecera secreta |
| Puertas de servicio sin autenticación | Sí | Sin endpoints de administración; inyección de fallos solo con bandera explícita y cabecera anulada en la ruta pública |
| Errores con trazas | Sí | Manejador global con mensaje genérico y `error_id` |
| Supabase / RLS | No aplica | No se usa Supabase |
| Gasto sin techo en el LLM | Sí | La clave no tiene límite propio: tope en código de 1,50 USD/día y 3,50 USD total, comprobado antes de cada llamada |
| Límite de peticiones | Sí | Traefik por IP (antes de la autenticación), 30 mensajes/h por sesión, 150/h global, una sesión por usuario |
| Bucles del agente | Sí | Máximo 8 tools por turno, corte a la tercera llamada idéntica, 60 s por turno; reintentos de visión acotados y corte de circuito |
| Contenedores sin límites | Sí | `mem_limit` 512 MB y 0,75 CPU por servicio |
| Datos personales | Casi no | Datos sintéticos; al LLM no van imágenes ni nombres |
| Inyección de prompts | Sí | Texto de documentos y usuarios tratado como dato; las operaciones de riesgo exigen aprobación humana en código (casos A08 y A09) |
| Archivos subidos | Sí | Solo PNG y JPEG, 5 MB, 4.096 px por lado; `image_id` validado y ruta confinada |
| Deserialización del modelo (pickle) | Sí | SHA-256 del artefacto comprobado contra el manifiesto antes de cargarlo |

## Hallazgos de la revisión automática en commits (corregidos)

| Hallazgo | Corrección |
|---|---|
| Decodificador de imágenes sin restringir (Pillow con EPS invoca programas externos) | Solo PNG y JPEG; captura de bomba de descompresión |
| Promoción de modelo con alcance global | Limitada a la planta y línea que la pide |
| Ruta de imagen construida con identificadores sin validar | Patrón SHA-256 por elemento y ruta resuelta dentro del directorio |
| Tope de gasto comprobado solo al empezar el turno | Comprobado antes de cada llamada al LLM |
| Rol elegido por el cliente | Detrás de `DEMO_ROLE_SELECTION`, apagada por defecto y obligatoria en el VPS; riesgo aceptado y documentado |
| Límite por sesión esquivable y expulsión de sesiones vivas | Límite global, una sesión por usuario, caducidad por inactividad |
| Orden de middlewares en Traefik | Límite de peticiones antes de la autenticación |

## Dependencias

`pip-audit -r requirements.txt` (27 oct 2026): sin vulnerabilidades conocidas.

## Claude Security

Escaneo del repositorio completo en la revisión `7b8eb75`, modo codebase, esfuerzo medio, el 28 oct 2026, con el árbol de trabajo limpio. Se leyeron 316 archivos con 28 investigadores y un panel de tres verificadores por candidato (16 candidatos, 13 tras deduplicar). Sobrevivieron 2 hallazgos, los dos de severidad baja; no hubo críticos, altos ni medios. Todo se derivó de leer el código: no se ejecutó ningún exploit.

| Hallazgo | Severidad | Estado |
|---|---|---|
| F1: `evaluate_sku_candidate` podía pisar el estado `production`, reservado a la aprobación del `plant_manager` (CWE-841) | Baja | Corregido |
| F2: `submit_review_label` guardaba como `human_review` una etiqueta elegida por el LLM, alcanzable por inyección vía título de ticket (CWE-441) | Baja | Mitigado, queda un paso |

**F1.** La herramienta solo evalúa desde `registered`, `evaluated_ok` y `evaluated_fail`; con `production` devuelve `invalid_state`. El `UPDATE` es condicional al estado y comprueba `rowcount`, de modo que una promoción concurrente durante la evaluación tampoco se pisa. Tests: `test_sku_evaluate_cannot_overwrite_production` y su variante con carrera.

**F2.** Se aplicó la mitigación rápida: el origen en `retrain_dataset` pasa a `agent_proposed`, y `generate_shift_report` marca cada título de ticket como texto de usuario que no son instrucciones. Tests: `test_label_source_is_not_human_review` y `test_shift_report_marks_ticket_titles_as_untrusted`. El éxito del ataque dependía de que el modelo obedeciera, y eso sigue sin garantía en código. Lo que falta es una confirmación humana determinista para `submit_review_label` (aprobación pendiente ligada a `case_id`, `obj_id` y `angle`), como ya existe para las operaciones de riesgo. No se hizo por plazo: añade un flujo nuevo y riesgo de romper lo que funciona. Mientras tanto, ningún reentrenamiento llega a producción sin evaluación y aprobación (`promote_model`).

El informe completo del escáner está en `CLAUDE-SECURITY-20261028-054913/CLAUDE-SECURITY-RESULTS.md` (carpeta local, fuera del repositorio).

## Servidor (VPS)

Observación sin cambios: Traefik tiene `--api.insecure=true`; el panel no está publicado (solo 80 y 443), pero conviene desactivarlo.
