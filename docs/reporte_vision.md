---
title: "Reconocimiento de objeto y ángulo con ML clásico (COIL-100)"
subtitle: "Reporte de resultados por protocolo, análisis de errores y model card"
date: "28 de octubre de 2026"
---

# Resumen

El modelo productivo es una regresión logística sobre histograma HSV, LBP y píxeles reducidos para la clase, más búsqueda de vecinos entre las vistas del objeto predicho para el ángulo. La decisión para el robot la toma una política determinista con una puntuación de agarre calibrada.

Con vistas de referencia cada 10° (protocolo A) acierta el producto en el 99,9 % de las vistas nuevas y deja el 99,0 % de los ángulos dentro de la tolerancia del robot (±10°, teniendo en cuenta la simetría). Con el umbral elegido solo en validación, el robot actúa solo en el 50,7 % de las piezas con un 0,11 % de agarres erróneos; el resto va a revisión. Los 10 objetos desconocidos del protocolo D se rechazan siempre (0 % aceptados, AUROC de la distancia al catálogo 0,955).

Los límites aparecen donde deben: en el arco ciego de 60° (protocolo B) el ángulo solo acierta el 37,8 % y en la escasez de vistas cada 30° (protocolo C) el 76,3 %. En B el sistema rechaza casi todo (cobertura 9,8 %); en C la confianza aprendida con validación no transfiere bien (9,4 % de agarres erróneos entre los aceptados) y la conclusión operativa es no admitir productos con menos de una vista cada 15°.

# 1. Dataset y qué condiciona el diseño

COIL-100: 7.200 imágenes RGB de 128×128, 100 objetos, 72 vistas por objeto cada 5°. ZIP oficial de Columbia CAVE, SHA-256 `fcb4b8b0…4afe7`, con manifiesto por imagen en `data/manifest.csv`.

| Propiedad | Consecuencia |
|---|---|
| El objeto gira sobre una plataforma: rotación fuera del plano | El ángulo no se obtiene girando la imagen; se aprende por objeto (vecinos entre sus vistas de referencia) |
| El ángulo es circular | Error `min(|a−b|, 360−|a−b|)`; media y dispersión circulares |
| Hay objetos simétricos | en el modelo desplegado, 7 simétricos totales y 16 cada 180° (detectados con las vistas de entrenamiento); el robot recibe `any` o `mod180` y el error se mide con simetría |
| Fondo negro, luz fija, una elevación | Control de calidad previo (desenfoque, exposición, fracción de objeto); en planta haría falta segmentación |
| Vistas cada 5° | Con referencias cada 10°, el vecino correcto queda a ≤ 5°: la tolerancia de ±10° es alcanzable; con cada 30°, no |

![Candidatos a simetría: rosca, envase, platillo y maceta (totales); paquetes y caja (180°); frasco](img/simetria_candidatos.png)

# 2. Metodología

**Protocolos.** A (entrenar en 0°, 10°…; probar en 5°, 15°…), B (excluir un arco contiguo de 60° por objeto), C (una vista cada 30°), D (90 objetos; probar en vistas intermedias de los 90 y en los 10 excluidos completos).

**Validación sin tocar la prueba.** Cada protocolo tiene una validación interna que imita su salto usando solo vistas de entrenamiento: A y D entrenan en múltiplos de 20° y validan en los intermedios (D además aparta 10 conocidos como desconocidos simulados); B aparta otro arco de 60° dentro de lo visto; C deja una vista fuera en 12 folds. Hiperparámetros, configuración y umbrales se eligen solo con esas métricas. La prueba se evalúa una vez.

**Características.** Clase: combinaciones de histograma HSV (casi invariante al giro), HOG, LBP y píxeles 32×32; estandarización y PCA a 128 componentes. Ángulo: HOG más píxeles (cambian con el giro), solo entre las vistas del objeto predicho.

**Clasificadores comparados.** k vecinos, SVM RBF, Random Forest y regresión logística, con 3 combinaciones de características cada uno (12 configuraciones), más un baseline (1 vecino sobre píxeles) y una referencia moderna (ResNet-50 preentrenada en ImageNet como extractor, con cabezal kNN o logístico).

**Confianza y rechazo.** La clase la decide el clasificador; su confianza se recalibra con regresión isotónica sobre predicciones fuera de fold (folds barajados). La política combina en una regresión logística la confianza de clase, la distancia al catálogo, la incertidumbre del ángulo y la simetría para estimar la probabilidad de agarre correcto, y elige el umbral con la curva riesgo-cobertura en validación (riesgo ≤ 1 %). Antes hay una puerta de desconocidos (distancia por encima del percentil 99 de los conocidos).

**Criterio de aceptación.** Un agarre es correcto si la clase es correcta y el error angular con simetría es ≤ 10°. Cifras de negocio: cobertura (piezas que el robot toma sin persona) y tasa de agarre erróneo entre las aceptadas.

**Trazabilidad.** 60 ejecuciones del barrido y 12 de ángulo registradas en MLflow (`mlruns/mlflow.db`) con parámetros, SHA-256 de datos, métricas, umbrales y predicciones de prueba; resumen en `eval/vision_summary.csv`.

# 3. Selección del modelo

Criterio en validación: media en A-D de (cobertura − 10 × agarre erróneo). Gana la regresión logística con HSV + LBP + píxeles (0,061), seguida de la logística con HSV + HOG (0,028) y Random Forest con HSV + HOG (0,012). Las SVM aciertan la clase igual de bien en prueba, pero en validación aceptan más agarres erróneos y quedan por detrás.

# 4. Resultados por protocolo (mejor clásico)

| Protocolo | Acierto de clase | Ángulo ±10° (crudo) | Ángulo ±10° (simetría) | Error mediano | ECE | Cobertura | Agarre erróneo |
|---|---|---|---|---|---|---|---|
| A interpolación | 99,9 % | 83,4 % | 99,0 % | 0,3° | 0,0008 | 50,7 % | 0,11 % |
| B sector ciego | 96,0 % | 21,8 % | 37,8 % | 17,2° | 0,0315 | 9,8 % | 1,71 % |
| C escasez | 98,8 % | 61,6 % | 76,3 % | 5,0° | 0,0093 | 20,2 % | 9,41 % |
| D desconocidos | 99,9 % (conocidos) | 82,7 % | 98,9 % | 0,3° | 0,0006 | 16,2 % | 0,00 % |

En D, con el umbral de validación, se aceptan el 0 % de las vistas de objetos desconocidos; la AUROC de la distancia al catálogo para separar conocidos de desconocidos es 0,955.

**Por qué la cobertura es baja con el umbral de validación.** La validación es más pesimista que la prueba: sus vistas quedan a 10° de la referencia más cercana, mientras que en prueba quedan a 5°. Exigir riesgo ≤ 1 % en validación da un umbral conservador. Con el umbral elegido para riesgo ≤ 2 % la cobertura en A sube al 96,3 % con 0,43 % de agarres erróneos, pero en D llega al 82,0 % con 5,2 % de erróneos porque empiezan a colarse objetos desconocidos. Por eso el punto de trabajo del protocolo se mantiene en 1 % y el artefacto desplegado se recalibra (sección 7).

![Fiabilidad de la confianza, riesgo-cobertura y distancia al catálogo](img/confianza.png)

# 5. Comparación obligatoria: baseline, mejor clásico y referencia moderna

| Protocolo | Modelo | Acierto de clase | Ángulo ±10° (simetría) | Cobertura | Agarre erróneo |
|---|---|---|---|---|---|
| A | Baseline 1 vecino, píxeles | 96,6 % | 97,6 % | 9,9 % | 0,00 % |
| A | Mejor clásico | 99,9 % | 99,0 % | 50,7 % | 0,11 % |
| A | ResNet-50 + logística | 100,0 % | 80,9 % | 6,4 % | 2,16 % |
| B | Baseline | 78,4 % | 47,2 % | 9,5 % | 0,00 % |
| B | Mejor clásico | 96,0 % | 37,8 % | 9,8 % | 1,71 % |
| B | ResNet-50 + logística | 98,8 % | 13,4 % | 3,0 % | 0,00 % |
| C | Baseline | 83,3 % | 72,8 % | 6,6 % | 22,42 % |
| C | Mejor clásico | 98,8 % | 76,3 % | 20,2 % | 9,41 % |
| C | ResNet-50 + logística | 99,9 % | 41,2 % | 4,0 % | 0,00 % |
| D | Baseline | 96,7 % | 97,6 % | 1,7 % | 0,00 % |
| D | Mejor clásico | 99,9 % | 98,9 % | 16,2 % | 0,00 % |
| D | ResNet-50 + logística | 100,0 % | 79,9 % | 1,8 % | 0,00 % |

Latencia en CPU por imagen: el clásico tarda unos 8 ms (6,4 ms de características y 1,4 ms de predicción); ResNet-50 tarda 73,5 ms en lotes de 64 y más de 0,5 s imagen a imagen en este equipo. El modelo clásico pesa 6,9 MB; ResNet-50, unos 100 MB de pesos.

**¿Justifica la brecha su coste?** No. La referencia moderna gana algo de acierto de clase donde faltan vistas (B: 98,8 % frente a 96,0 %; C: 99,9 % frente a 98,8 %), pero sus embeddings son casi invariantes a la pose, así que el ángulo empeora mucho (81 % frente a 99 % en A) y la cobertura útil se hunde. Para este problema, donde el ángulo es la mitad de la tarea, cuesta 10 veces más latencia y una GPU o un PC más potente en planta sin mejorar el resultado que importa. Una vía razonable sería un híbrido: embeddings modernos solo para la clase en productos con pocas vistas y el ángulo con características clásicas; queda como trabajo futuro.

# 6. Ángulo: vecinos frente a kernel ridge

| Método | Validación (puntuación media A-D) | Ángulo ±10° A | B | C | D |
|---|---|---|---|---|---|
| kernel_ridge-k3 | 0.074 | 99,1 % | 47,9 % | 83,3 % | 98,9 % |
| knn-k1 | 0.059 | 99,4 % | 46,5 % | 73,4 % | 99,4 % |
| knn-k5 | 0.023 | 90,0 % | 28,6 % | 40,7 % | 89,9 % |
| knn-k2 (barrido) | 0.061 | 99,0 % | 37,8 % | 76,3 % | 98,9 % |

Gana kernel ridge en validación (0,074 frente a 0,061 de vecinos con k=2): interpola mejor entre vistas de referencia y su incertidumbre (norma del vector seno-coseno predicho) separa algo mejor los ángulos dudosos. Es el método del artefacto desplegado.

# 7. Calibración del artefacto desplegado

El artefacto v1.0.0 se entrena con la partición D (90 productos, vistas cada 10°): los 10 excluidos siguen siendo desconocidos de verdad para la demo. Su punto de trabajo se recalibra con las vistas intermedias a 5° y los 10 desconocidos, que se parecen más a la operación que la validación interna (vistas a 10°). En ese conjunto de calibración: el robot actúa en el 90,0 % de las piezas de productos conocidos con un 0,9 % de agarres erróneos y acepta el 11,1 % de las vistas de desconocidos (puerta de distancia en el percentil 90, meta de riesgo 1,5 %, 2 % de piezas desconocidas supuesto). Cómo se eligió ese punto (conjunto de calibración, 2 % de piezas desconocidas supuesto):

| Puerta de distancia (percentil de conocidos) | Meta de riesgo | Cobertura de conocidos | Agarre erróneo en conocidos | Vistas de desconocidos aceptadas |
|---|---|---|---|---|
| 99 | 1,0 % | 29,5 % | 1,05 % | 0,0 % |
| 99 | 1,5 % | 98,7 % | 1,06 % | 21,4 % |
| 95 | 1,5 % | 95,0 % | 0,94 % | 22,1 % |
| 90 | 1,5 % | 90,0 % | 0,93 % | 11,1 % |
| 85 | 1,5 % | 85,0 % | 0,91 % | 5,7 % |

El error en conocidos se queda en torno al 1 % con cualquier umbral: son ángulos equivocados con mucha confianza que la puntuación no separa (por eso el umbral de la puntuación desplegado queda en 0,009 y quien filtra es la puerta de distancia). Lo que sí se controla es cuántos desconocidos se cuelan. Un producto equivocado empacado es peor que un agarre fallido, así que se elige el percentil 90: 90 % de cobertura y, con un 2 % de piezas desconocidas, un 0,25 % de piezas con producto equivocado. Es una decisión de negocio que la hoja de costos permite revisar.

Estas cifras son dentro de muestra y optimistas; el despliegue real empezaría en modo sombra (el sistema decide, la persona sigue alineando) para medir cobertura y error antes de retirar la tarea manual.

# 8. Análisis de errores

**Clase.** En el protocolo A solo hay 3 errores de clase en 3.600 vistas: el objeto 39 tomado por el 21 y el 44 tomado por el 21 y por el 57. La clase es el problema menor.

**Ángulo.** Con simetría, el error angular mediano por objeto en A no pasa de 5° (el peor, el objeto 66). El error residual que queda en torno al 1 % son casos sueltos con mucha confianza y no un objeto sistemáticamente mal: por eso la puntuación de agarre no los separa (sección 7). La diferencia entre la métrica cruda (82,7 % en D) y la que tiene en cuenta la simetría (98,9 %) es casi toda de los 24 objetos simétricos, para los que el robot recibe `any` o `mod180`.

**Arco ciego (B).** El error crece hacia el centro del arco excluido: en los extremos la referencia más cercana está a 5-10°, en el centro a 30°. La política rechaza casi todo (cobertura 9,8 %). Es el comportamiento buscado.

**Escasez (C).** Con referencias cada 30° el vecino correcto puede quedar a 15°. La validación dejando una vista fuera (vecinos a 30°) no predice bien la prueba y el umbral deja pasar un 9,4 % de agarres erróneos entre los aceptados. Conclusión operativa: el alta de productos exige vistas cada 15° como máximo (`register_sku_candidate` lo comprueba).

**Robustez ante condiciones fuera de dominio** (150 vistas de prueba de conocidos, modelo desplegado):

| Condición | Acierto de clase | Confianza media | PICK del robot | PICK con clase errónea | Qué lo detiene |
|---|---|---|---|---|---|
| Original | 100 % | 0,999 | 86 % | 0 | puerta de desconocidos (14), calidad (7) |
| Brillo −40 % | 19,8 % | 0,870 | 0 % | 0 | "no se ve un objeto aislado" |
| Ruido gaussiano σ=15 | 1,0 % | 1,000 | 0 % | 0 | puerta de distancia ("objeto desconocido") |
| Desenfoque σ=2 | 79,5 % | 0,891 | 0 % | 0 | "imagen desenfocada" |

Con ruido el clasificador se equivoca casi siempre y con confianza 1,00: la probabilidad calibrada solo vale dentro del dominio en que se calibró. Lo que protege al robot es la combinación de control de calidad y distancia al catálogo, y por eso la política no se apoya solo en la confianza de clase. En planta, un cambio de cámara o de iluminación haría que el sistema desviase todo a revisión en lugar de agarrar mal: es el fallo seguro que se busca, y la señal para recalibrar.

# 9. Model card

Copia de `models/vision/v1.0.0/MODEL_CARD.md`.


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

