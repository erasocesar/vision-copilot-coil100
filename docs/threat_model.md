# Threat model y controles

Método: STRIDE sobre los flujos de `docs/arquitectura.md` y el OWASP Top 10 para aplicaciones con LLM (edición 2025). Cada amenaza tiene su control y la prueba que lo cubre (test automático o caso de `eval/cases.yaml`).

## Activos

| Activo | Por qué importa |
|---|---|
| Comandos al robot (PICK, ángulo, DIVERT) | Un comando erróneo daña producto o para la línea |
| Umbral de la línea y versión de modelo | Cambian el riesgo de agarre erróneo de todas las piezas |
| Imágenes de planta | Pueden mostrar procesos o personas |
| Registro de auditoría y aprobaciones | Prueba de quién hizo qué |
| Clave del LLM y presupuesto | Coste directo |

## Límites de confianza

1. Navegador → API del agente (Internet; en la demo, detrás de Traefik con HTTPS, contraseña y límite por IP).
2. API del agente → LLM externo (OpenAI). Solo viaja texto: nunca imágenes ni base64.
3. Agente → servicio de visión (red interna de Docker).
4. Texto de documentos (SOP) y resultados de tools → contexto del LLM (dato no confiable).

## STRIDE

| Amenaza | Ejemplo | Control | Prueba |
|---|---|---|---|
| Suplantación (S) | Alguien abre sesión como gerente de planta | Demo: contraseña de Traefik y bandera `DEMO_ROLE_SELECTION` (apagada por defecto). Producción: rol desde Entra ID | `test_role_selection_disabled_by_default` |
| Manipulación (T) | Cambiar el umbral o el ángulo enviado al robot | Solo `approvals.execute` modifica la línea, tras aprobación de otra persona con rol autorizado; UPDATE atómico sobre `status='pending'` | `test_self_approval_rejected`, `test_double_approval_no_double_effect`, casos C23-C27 |
| Manipulación (T) | Artefacto de modelo alterado (joblib ejecuta pickle) | SHA-256 del artefacto en `manifest.json`, comprobado antes de cargar | `VisionModel.load` |
| Repudio (R) | "Yo no pedí pausar la línea" | Auditoría por llamada (usuario, rol, planta, tool, hash de argumentos, resultado) y aprobaciones con solicitante y aprobador | `test_audit_hashes_args`, `test_every_call_audited` |
| Divulgación (I) | Consultar datos de otra planta | `plant_id` y `line_id` salen de la sesión del servidor y filtran cada consulta | `test_metrics_isolated_by_plant`, `test_review_queue_other_plant_case_not_found`, A11 |
| Divulgación (I) | Trazas internas en errores | Manejador global: mensaje genérico con `error_id`; detalle solo en el log | `test_500_hides_traceback`, `test_errors_hide_traceback` |
| Denegación (D) | Bucle del agente o ráfaga de mensajes que agota el presupuesto | Máximo 8 tools por turno, corte a la tercera llamada idéntica, 60 s por turno, tope de 1,50 USD/día y 3,50 USD total comprobado antes de cada llamada, 30 mensajes/h por sesión, 150/h global, una sesión por usuario, límite por IP en Traefik | `test_loop_cut_on_third_identical_call`, `test_budget_checked_before_each_llm_call`, `test_rate_limit`, A07 |
| Denegación (D) | Imagen gigante o bomba de descompresión | 5 MB, 4.096 px por lado, solo PNG y JPEG | `test_decode_rejects_too_big`, `test_decode_rejects_other_formats` |
| Denegación (D) | Servicio de visión caído | Reintentos acotados (0,5 y 1 s), corte de circuito 30 s, error estructurado sin inventar resultado | `test_circuit_breaker`, A05, A06 |
| Elevación (E) | Mantenimiento intenta promover un modelo | Roles por tool y por operación (`REQUESTERS`, `APPROVERS`) | `test_requester_role_checked`, `test_forbidden_role_audited`, A10 |
| Elevación (E) | Recorrido de rutas con un `image_id` | `image_id` validado como SHA-256 hexadecimal y ruta resuelta dentro del directorio de imágenes | `test_sku_rejects_path_like_ids` |

## OWASP Top 10 para LLM (2025)

| Riesgo | Aplica | Control |
|---|---|---|
| LLM01 Inyección de prompts | Sí | El prompt de sistema declara que documentos y resultados son datos. Lo importante no depende del prompt: las operaciones de riesgo exigen aprobación humana en código, así que una inyección como mucho crea una petición pendiente. Casos A08 (inyección dentro de un SOP) y A09 (orden directa) |
| LLM02 Divulgación de información sensible | Sí | Al LLM no van imágenes ni nombres reales; operarios con seudónimo; auditoría guarda hash de argumentos |
| LLM03 Cadena de suministro | Sí | Dependencias con versión fija, `pip-audit` antes de publicar, pesos de ResNet-50 solo en el notebook |
| LLM04 Envenenamiento de datos | Parcial | Las etiquetas humanas van a `retrain_dataset`; ningún reentrenamiento pasa a producción sin evaluación y aprobación (`promote_model`) |
| LLM05 Manejo inseguro de la salida | Sí | La interfaz pinta todo con `textContent` (sin HTML del modelo); los argumentos de tools se validan con Pydantic antes de ejecutar |
| LLM06 Agencia excesiva | Sí | 13 tools con permisos mínimos; las de riesgo solo piden; límites de pasos y tiempo |
| LLM07 Fuga del prompt de sistema | Bajo | El prompt no contiene secretos |
| LLM08 Debilidades de embeddings y vectores | No aplica | La búsqueda en SOP es TF-IDF local |
| LLM09 Desinformación | Sí | Respuestas con cifras de tools y citas `[doc#sección]`; búsqueda sin coincidencias devuelve lista vacía con la instrucción de decirlo (caso C18) |
| LLM10 Consumo sin límite | Sí | Topes de gasto, mensajes y pasos (ver Denegación) |

## Riesgos aceptados en la demo

- Cualquiera con la contraseña de la demo puede elegir rol y, con dos usuarios distintos, aprobar su propia petición. Es una limitación de la demo; en producción el rol y la identidad vendrían de Entra ID y la regla "aprobador distinto del solicitante" se apoyaría en identidades reales.
- Las sesiones viven en memoria del proceso: se pierden al reiniciar.
- La inyección de fallos del servicio de visión (`DEMO_FAULTS=true`) existe para mostrar los escenarios adversos. En la ruta pública Traefik elimina la cabecera `X-Demo-Fault`.
