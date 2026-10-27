# Model card: identificación de producto y ángulo, v1.0.0

## Uso previsto

Identificar qué producto llega a la estación de alineación, estimar su ángulo de giro sobre el eje vertical (0-360°) con una medida de incertidumbre y decidir si el robot de pick & pack puede actuar (PICK con producto, ángulo y modo de ángulo) o si la pieza va a revisión manual (DIVERT con motivo). Usuarios: el lazo automático de la línea y, a través del asistente, supervisión, calidad, mantenimiento e ingeniería de ML.

## Modelo

| Parte | Elección |
|---|---|
| Clase | Regresión logística sobre histograma HSV (128) + LBP en rejilla 4×4 (160) + píxeles en gris 32×32 (1.024), estandarizados y reducidos con PCA a 128 |
| Confianza de clase | Probabilidad de la clase ganadora recalibrada con regresión isotónica sobre predicciones fuera de fold (3 folds barajados) |
| Ángulo | Kernel ridge (RBF) por producto sobre HOG 64×64 + píxeles, que predice seno y coseno del ángulo; la norma del vector predicho da la incertidumbre |
| Simetría | Detectada por producto con las vistas de entrenamiento: 67 asimétricos, 16 simétricos cada 180° y 7 simétricos totales |
| Evidencia | 5 vistas de referencia más cercanas (producto, ángulo, distancia) |
| Política | Control de calidad → puerta de desconocidos (distancia al catálogo ≤ 11,0) → puntuación de agarre logística ≥ umbral |

Selección: entre 4 clasificadores × 3 combinaciones de características y 4 métodos de ángulo, por métricas de validación de los 4 protocolos. Artefacto: `model.joblib` (6,9 MB, SHA-256 en `manifest.json`, comprobado al cargar).

## Datos

COIL-100 (Columbia CAVE), ZIP SHA-256 `fcb4b8b0…4afe7`. Entrenamiento: partición D, 90 productos con vistas cada 10° (3.240 imágenes). Productos excluidos, desconocidos para el modelo: 9, 10, 20, 42, 61, 69, 72, 83, 95 y 97.

## Métricas

Protocolo D con umbrales elegidos en validación (sin tocar la prueba): acierto de clase 99,9 %, ángulo dentro de ±10° con simetría 98,9 % (crudo 82,7 %), error angular mediano 1,0°, ECE 0,0006, AUROC de la distancia al catálogo 0,955, desconocidos aceptados 0 %, cobertura 9,5 % con 0,27 % de agarres erróneos.

Resultados de los protocolos A, B y C en `docs/reporte_vision.pdf`.

Punto de trabajo desplegado, recalibrado con las vistas a 5° de los 90 conocidos y los 10 desconocidos, suponiendo que un 2 % de las piezas sean productos no dados de alta: el robot actúa en el 90 % de las piezas de productos conocidos con un 0,93 % de agarres erróneos; acepta el 11 % de las vistas de desconocidos. Son cifras dentro de muestra y optimistas.

Latencia: unos 12 ms por imagen en CPU, incluida la lectura y la extracción de características.

## Limitaciones conocidas

- Fondo negro, luz controlada, un objeto centrado y una sola elevación de cámara. Con fondo o luz distintos el control de calidad desviará casi todo: no está preparado para segmentar.
- El error residual en productos conocidos (alrededor del 1 %) son ángulos equivocados con mucha confianza, sobre todo en objetos casi simétricos que no llegan al umbral de simetría. La puntuación de agarre no los separa.
- Los desconocidos parecidos a un producto del catálogo pueden pasar la puerta de distancia (11 % de las vistas en calibración).
- En un arco de 60° sin referencias el ángulo falla en más de la mitad de los casos; con referencias cada 30°, en uno de cada cuatro.
- Simetría decidida con dos umbrales fijados mirando el catálogo.

## Cuándo no usarlo

- Productos con menos de una vista de referencia cada 15° o con huecos de más de 15° (usar el alta de producto con evaluación previa).
- Cámaras, fondos, iluminación o distancias distintas de las de entrenamiento sin recalibrar el control de calidad y los umbrales.
- Objetos que también giran en el plano de la imagen o llegan volcados.
- Como única barrera ante productos no dados de alta: el alta de productos y el control de entrada de referencias siguen siendo necesarios.
- Retirar la tarea manual sin un periodo en modo sombra que confirme cobertura y error en la línea real.
