# Arquitectura

El sistema tiene dos mundos separados a propósito. El lazo cámara → visión → robot es código determinista y vive en la planta: decide en milisegundos, siempre igual ante la misma imagen, y sigue funcionando si se cae internet o el LLM. El asistente agéntico vive fuera de ese lazo. Consulta, explica y propone, pero toda acción que cambia la línea pasa por permisos en código y por la aprobación de una persona distinta de quien la pidió.

## Componentes

| Componente | Código | Qué hace | Determinista |
|---|---|---|---|
| Servicio de visión | `src/vision/api.py` | API propia: calidad, clase, ángulo, vistas similares, decisión PICK/DIVERT, evaluación aislada de productos nuevos | Sí |
| Modelo clásico | `src/vision/model.py` | Características HSV, HOG, LBP y píxeles; SVM o regresión logística calibrada; ángulo por vecinos del objeto predicho | Sí |
| Política de decisión | `src/vision/policy.py` | Puerta de calidad, puerta de desconocidos y puntuación de agarre con umbral fijo | Sí |
| Simulador de línea | `src/sim/line_sim.py` | 24 h de inspecciones con el modelo real, incidente de cámara a las 3:00 y fallo de ventosa a las 17:00 | Sí (semilla) |
| Agente | `src/agent/graph.py` | LangGraph: un agente de tool-calling con nodos deterministas de guardia, presupuesto y corte | No (solo el nodo `agent`) |
| Tools | `src/agent/tools/` | 13 operaciones con contrato Pydantic, roles, idempotencia y auditoría | Sí |
| Aprobaciones | `src/agent/approvals.py` | Operaciones de riesgo: petición, decisión de otra persona y ejecución | Sí |
| API del agente y chat | `src/agent/api.py` | Sesiones, subida de imágenes, chat, panel de aprobaciones | Sí |
| Base operativa | `src/common/schema.sql` | SQLite en la demo (PostgreSQL en Azure); todas las tablas de línea con `plant_id` y `line_id` | Sí |

## Diagrama de arquitectura

```mermaid
flowchart LR
  subgraph Planta["Planta (edge, determinista)"]
    CAM[Cámara] --> VIS[Servicio de visión<br/>FastAPI + modelo clásico v1.0.0]
    VIS --> POL[Política de decisión<br/>calidad, desconocido, puntuación de agarre]
    POL -->|PICK producto, ángulo, modo| ROB[Robot pick and pack]
    POL -->|DIVERT| REV[(Cola de revisión)]
    VIS --> DB[(Inspecciones, salud de cámara,<br/>agarres)]
  end
  subgraph Asistente["Asistente (fuera del lazo)"]
    UI[Chat web] --> API[API del agente<br/>sesión, rol, límites]
    API --> AG[Grafo LangGraph]
    AG --> GU{Guardia<br/>esquema, rol, bucle, límites}
    GU --> TOOLS[13 tools<br/>auditoría e idempotencia]
    TOOLS -->|HTTP| VIS
    TOOLS --> DB
    TOOLS --> REV
    TOOLS --> SOP[(Procedimientos SOP)]
    TOOLS --> APR[(Aprobaciones)]
    APR -->|aprobada por otra persona| EXE[Ejecución en código]
    EXE --> ROB
    EXE --> POL
    TOOLS -.opcional.-> N8N[Webhook n8n]
  end
  AG <--> LLM[OpenAI gpt-4.1-mini]
```

## Flujo 1: imagen → robot

```mermaid
sequenceDiagram
  autonumber
  participant C as Cámara
  participant V as Servicio de visión
  participant Q as Control de calidad
  participant M as Modelo clásico
  participant P as Política
  participant R as Robot
  participant D as Base operativa
  C->>V: POST /v1/predict (imagen PNG/JPEG ≤ 5 MB)
  V->>V: decode_image (solo PNG/JPEG, 128×128 RGB)
  V->>Q: check_quality (desenfoque, exposición, objeto aislado)
  V->>M: predict (clase + prob. calibrada, distancia al catálogo, ángulo + incertidumbre, simetría, top-k)
  M-->>V: Prediction
  V->>P: decide(prediction, quality, umbrales v1.0.0)
  alt calidad, catálogo y puntuación de agarre correctos
    P-->>V: PICK (producto, ángulo, modo exact / mod180 / any)
    V-->>R: PICK
    R-->>D: agarre (éxito o código de falla)
  else cualquier condición falla
    P-->>V: DIVERT + motivos
    V-->>R: DIVERT (bandeja de revisión)
    V-->>D: caso en la cola de revisión
  end
```

El robot nunca recibe un ángulo "aproximado" sin respaldo: o recibe PICK con una puntuación de agarre que supera el umbral elegido en validación (riesgo de agarre erróneo ≤ 1 %), o recibe DIVERT. Para objetos simétricos recibe el modo `any` (cualquier giro vale) o `mod180`.

## Flujo 2: agente con aprobación humana

Mantenimiento detecta agarres fallidos, el agente diagnostica y pide pausar la línea; el supervisor aprueba desde su sesión.

```mermaid
sequenceDiagram
  autonumber
  actor MT as Mantenimiento (luis)
  participant A as API del agente
  participant G as Grafo LangGraph
  participant L as LLM
  participant T as Tools + guardia
  participant DB as Base operativa
  actor SV as Supervisor (ana)
  MT->>A: "Hubo agarres fallidos por la tarde, diagnostica y pausa la línea"
  A->>G: run_turn (rol y planta desde la sesión)
  G->>G: start: recorta historial, comprueba tope de gasto
  G->>L: mensajes + 13 esquemas de tools
  L-->>G: get_grasp_events(start_hour=14, end_hour=22)
  G->>T: guardia (esquema, rol, bucle, máx. 8) y ejecución
  T->>DB: consulta filtrada por P1/L1, auditoría
  T-->>G: VACUUM_LOSS concentrados a las 17 h, visión correcta
  G->>L: resultado
  L-->>G: search_procedures("pérdida de vacío ventosa")
  T-->>G: [SOP-ROB-02#Ventosa y circuito neumático]
  L-->>G: request_risk_operation(op=pause_line, justificación)
  T->>DB: aprobación PENDIENTE (no ejecuta nada)
  L-->>G: respuesta con diagnóstico, cita y "pausa pendiente de aprobación"
  G-->>A: respuesta + approval_id
  A-->>MT: diagnóstico y estado pendiente
  SV->>A: GET /api/approvals → POST decision (approve)
  A->>A: approvals.decide: aprobador ≠ solicitante, rol autorizado, estado pending (UPDATE atómico)
  A->>DB: line_state.paused = 1, resultado y auditoría
  A-->>SV: aprobada y ejecutada
  MT->>A: siguiente mensaje
  A->>G: aviso del sistema: "pause_line aprobada por ana"
```

## Dónde vive cada decisión

| Decisión | Quién la toma | Por qué |
|---|---|---|
| Agarrar o desviar | Política determinista | Latencia de milisegundos, mismo resultado ante la misma imagen, auditable |
| Valor de los umbrales | Validación (código) y aprobación humana para cambiarlos | Un LLM no debe mover el punto de riesgo de la línea |
| Rol y planta del usuario | Sesión del servidor | El texto del mensaje puede mentir |
| Si una operación de riesgo se ejecuta | Otra persona con rol autorizado | Cuatro ojos; el LLM solo puede pedir |
| Qué tools llamar y cómo explicar | LLM | Es donde aporta: preguntas abiertas y diagnósticos de varios pasos |
| Cuándo parar el agente | Guardia y límites en código | Bucles, coste y tiempo acotados |
