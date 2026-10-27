# Registro de decisiones (ADR)

Cada decisión con su contexto, las opciones que se descartaron y lo que cuesta si sale mal.

## ADR-01 El lazo imagen → robot es determinista y no pasa por el LLM

Contexto: un error de clase empaca el producto equivocado y un error de ángulo produce un agarre fallido o una parada. El robot necesita respuesta en milisegundos.

Decisión: visión clásica más una política con umbrales fijos decide PICK o DIVERT. El LLM no participa.

Descartado: que el agente decida con la salida del modelo (latencia de 1 a 5 s por llamada, coste por pieza, respuestas distintas ante la misma entrada, sin garantía de repetibilidad para auditoría).

Consecuencia: el asistente puede caerse sin parar la línea. El coste del LLM depende de las consultas de las personas, no de las piezas.

## ADR-02 Espacios de características distintos para clase y ángulo

Contexto: en COIL-100 el objeto gira sobre una plataforma; es una rotación fuera del plano. Para identificar conviene algo que no cambie con el giro; para el ángulo, algo que sí cambie.

Decisión: clase con histograma HSV más HOG (y LBP en la variante elegida); ángulo con HOG más píxeles, buscando solo entre las vistas del objeto predicho.

Descartado: un único espacio para todo (el histograma de color casi no distingue ángulos) y regresión directa del ángulo sin condicionar a la clase.

## ADR-03 Simetría explícita

Contexto: latas, platillos o macetas no tienen ángulo definido; cajas y paquetes se repiten cada 180°. Para la pinza, ese giro no importa.

Decisión: cada objeto se clasifica como asimétrico, simétrico cada 180° o simétrico total, comparando cuánto cambia al girar 90° y 180° frente al objeto mediano del catálogo. El robot recibe el modo `exact`, `mod180` o `any`, y el error se mide teniendo en cuenta la simetría.

Coste si sale mal: un objeto asimétrico tomado por simétrico recibiría un ángulo arbitrario. Por eso los umbrales se fijaron mirando el catálogo (`docs/img/simetria_candidatos.png`) y las dos métricas (cruda y con simetría) se reportan juntas.

## ADR-04 Calibración de la etiqueta ganadora y puntuación de agarre

Contexto: la calibración sigmoide de scikit-learn uno contra todos con 100 clases bajaba el acierto del 99,9 % al 95 %, y con folds sin barajar cada submodelo perdía un sector de ángulos.

Decisión: la clase la decide el clasificador sin tocar; su confianza se recalibra con regresión isotónica sobre predicciones fuera de fold barajadas. Después, una regresión logística combina confianza, distancia al catálogo, incertidumbre del ángulo y simetría en una sola "probabilidad de agarre correcto". El umbral se elige en validación para un riesgo de agarre erróneo ≤ 1 %.

Descartado: tres umbrales sucesivos (ningún protocolo encontraba un punto que cumpliera y la cobertura salía cero).

## ADR-05 Un solo agente de tool-calling con nodos deterministas

Contexto: siete casos de uso, 13 tools, un dominio (una línea).

Decisión: un agente LangGraph que elige tools, rodeado de nodos de código: presupuesto, guardia (esquema, rol, bucle, máximo de 8 llamadas, 60 s) y mensajes fijos de corte.

Descartado:

| Alternativa | Coste | Latencia | Riesgo | Mantenibilidad |
|---|---|---|---|---|
| Supervisor con un agente por rol | Una llamada extra por salto entre agentes | Más saltos | Contexto que se pierde entre agentes | Cinco prompts que mantener |
| Enrutador fijo por intención | Menor | Menor | Bajo | Una rama nueva por cada pregunta nueva; no hace diagnósticos de varios pasos |
| Agente de n8n | Similar | Similar | Sin arnés de evaluación por tool | Flujos poco versionables en Git |

No hay evidencia de que separar agentes mejore el acierto con 13 tools; sin ella, la separación es coste.

## ADR-06 Aprobaciones durables en base de datos

Contexto: quien aprueba es otra persona, en otra sesión, quizá horas después.

Decisión: `request_risk_operation` crea una petición `pending`. La aprobación es una llamada de API de otra sesión que comprueba en código que el aprobador es distinto del solicitante, que su rol puede aprobar y que el estado sigue `pending` (UPDATE atómico). Solo entonces se ejecuta.

Descartado: `interrupt()` de LangGraph con el hilo del solicitante en pausa (frágil si la espera dura horas o si el solicitante sigue conversando).

## ADR-07 Demo en VPS propio; Azure solo en diseño

Contexto: 48 horas para la entrega; el enunciado pide la arquitectura de Azure "independientemente de dónde corra la demo".

Decisión: Docker Compose en local (un comando) y en un VPS con Traefik, HTTPS y contraseña. Azure se diseña con servicios concretos y costes en `docs/azure/`.

Coste si sale mal: ninguno para la evaluación; desplegar en Azure exigiría tiempo de configuración sin aportar a lo que se evalúa.

## ADR-08 Tope de gasto en código

Contexto: la clave de OpenAI de la demo no tiene límite de gasto propio y la URL es pública.

Decisión: el agente suma el coste de cada llamada desde los tokens reales y se detiene al pasar 1,50 USD al día o 3,50 USD en total, con un mensaje fijo. El servicio de visión sigue funcionando sin LLM. Además: contraseña en Traefik, límite por IP, 30 mensajes por hora y sesión y 150 por hora en total.

## ADR-09 n8n como integración opcional

Decisión: `create_ticket` guarda siempre el ticket en la base y, si existe `N8N_TICKET_WEBHOOK_URL`, avisa a un flujo de n8n (correo o mensajería) con una cabecera secreta. Un fallo del webhook no impide guardar el ticket. Así el repositorio sigue siendo reproducible sin n8n.
