# SOP-ROB-02 Diagnóstico de agarres fallidos

Documento sintético para la demo. Robot de pick & pack con pinza de vacío.

## Códigos de falla
WRONG_SKU: el robot recibió un producto distinto del real. Causa en visión: revisar confianza de clase y casos aceptados con baja puntuación.
MISALIGNED: el producto es correcto pero el ángulo enviado excede la tolerancia de ±10°. Causa en visión: incertidumbre del ángulo o producto simétrico mal tratado.
VACUUM_LOSS: pérdida de vacío en la ventosa. Causa mecánica, no de visión.

## Información a recopilar
Hora y turno del primer fallo, código de falla, producto, ángulo enviado, si la visión aceptó el caso (PICK) y la salud de la cámara en ese intervalo. Si los fallos son VACUUM_LOSS con visión correcta, el problema es neumático.

## Ventosa y circuito neumático
Revisar la presión del circuito (valor nominal 0,6 bar de vacío), el estado de la ventosa (grietas, desgaste) y el filtro. Cambiar la ventosa si tiene más de 200.000 ciclos.

## Escalamiento
Más de 5 agarres fallidos en una hora: ticket de prioridad alta a mantenimiento mecánico y aviso al supervisor del turno.
