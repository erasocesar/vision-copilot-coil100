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

Pendiente: escaneo de nivel medio sobre el repositorio limpio, lanzado por César antes del despliegue.

## Servidor (VPS)

Observación sin cambios: Traefik tiene `--api.insecure=true`; el panel no está publicado (solo 80 y 443), pero conviene desactivarlo.
