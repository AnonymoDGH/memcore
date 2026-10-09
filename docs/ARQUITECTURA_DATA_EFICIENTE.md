# Arquitectura: entrenar modelos potentes con pocos datos

## Realidad primero

- Ninguna arquitectura hace que un modelo sea "super inteligente" con pocos datos
  por sí sola. Lo que sí funciona es combinar **eficiencia de datos** (aprender más
  de cada token), **cómputo en inferencia** (pensar más al responder) y
  **verificación** (aprender de señales correctas, no solo de texto).
- Con datos limitados el cuello de botella es la **información**, no los parámetros.
  Un modelo chico que repite datos generaliza peor que uno mediano con datos
  diversos, pero repetir datos hasta ~4 épocas tiene costo casi nulo
  (Muennighoff et al., 2023, *Scaling Data-Constrained Language Models*).
- MemCore ya tiene la pieza más útil para esto: **memoria que se escribe en
  runtime** y **replay anti-olvido**. Lo que falta es la capa que genera datos y
  verifica respuestas.

## Arquitectura propuesta: 5 capas

```
 ┌──────────────────────────────────────────────────────────────┐
 │ 5. Razonamiento en inferencia   (cómputo extra al responder)   │
 ├──────────────────────────────────────────────────────────────┤
 │ 4. Verificador        (señal exacta: tests, reglas, aritmética)│
 ├──────────────────────────────────────────────────────────────┤
 │ 3. Generador de datos (el modelo crea y filtra ejemplos)       │
 ├──────────────────────────────────────────────────────────────┤
 │ 2. Memoria + replay   (ya existe en memcore)                   │
 ├──────────────────────────────────────────────────────────────┤
 │ 1. Núcleo eficiente   (pesos compartidos, recurrencia)         │
 └──────────────────────────────────────────────────────────────┘
```

### 1. Núcleo eficiente
- **Transformer con pesos compartidos en bucle** (estilo *looped transformer* /
  Universal Transformer): N bloques reutilizados K veces. Más profundidad efectiva
  con menos parámetros, útil cuando hay poca memoria (8 GB).
- Mantener RoPE, RMSNorm, SwiGLU y head atado (ya implementados en `c0b472d`).
- **Tokenizador**: byte-level hoy (vocab 256/1024). Pasar a BPE entrenado sobre el
  propio corpus reduce secuencia ~3-4× → más contexto por byte de RAM.

### 2. Memoria + replay (ya existe)
- `neural_mem.c`: escritura por sorpresa. Se mantiene: es la forma barata de
  "aprender un hecho nuevo" sin reentrenar el núcleo.
- Extensión: **memoria indexada por contexto** (clave/valor con recuperación
  top-k) en lugar de solo MLP. Permite citar lo aprendido y borrarlo sin tocar pesos.
- `replay.c`: priorizar ejemplos con **alta pérdida o baja confianza**, no
  muestreo uniforme.

### 3. Generador de datos (el multiplicador de datos)
- Ciclo: el modelo propone ejemplos/preguntas → se filtran → se entrenan.
  Es la idea de STaR (Zelikman et al., 2022) y de *Textbooks Are All You Need*
  (Gunasekar et al., 2023): calidad y síntesis > volumen bruto.
- Filtro obligatorio: sin verificador no hay señal; el modelo solo refuerza sus
  propios errores.
- Destilación desde un modelo mayor (si hay acceso) aplicada solo a
  respuestas verificadas.

### 4. Verificador (la pieza que más sube "inteligencia" por dato)
- Tareas con **respuesta comprobable**: aritmética, código con tests, lógica,
  esquemas JSON, gramática. Recompensa binaria exacta.
- Entrenar con **RL con recompensas verificables** (RLVR, como en DeepSeek-R1,
  2025) o con rechazo-muestreo (quedarse solo con respuestas correctas y
  re-entrenar). Aprende de cada intento, no de cada frase.
- Empezar por **un dominio** (aritmética o código pequeño). Generalizar después.

### 5. Razonamiento en inferencia
- Cadena de pensamiento interna + **más pasos de cómputo** (bucles extra en el
  modelo recurrente, o muestreo con votación) cuando la confianza es baja.
- Inteligencia escalable en tiempo de respuesta, no en tamaño de datos.

## Programa de entrenamiento (`memcore train`, fases)

| Fase | Datos | Objetivo | Criterio de paso |
|------|-------|----------|------------------|
| 0. Base | corpus pequeño propio (`data/`) | modelar lenguaje | loss estable (ya hecho) |
| 1. Ejemplos sintéticos | generados y verificados | razonamiento en un dominio | >X% de precisión en held-out |
| 2. RL/rechazo | tareas verificables | mejorar precisión | sube sin degradar fase 0 |
| 3. Continuo | texto nuevo del usuario | memoria + replay | sin olvido en set de control |

Regla: siempre un **set de control** congelado (lenguaje base + tareas). Si baja,
se revierte. Sin control no sabes si el modelo mejora o solo memoriza.

## Qué implementar primero (orden por impacto/costo)

1. **Set de evaluación held-out y métricas de precisión por tarea** (no solo loss).
   Sin esto todo lo demás es a ciegas.
2. **Verificador de aritmética** en C (genera problemas, comprueba respuesta).
3. **Rechazo-muestreo**: generar → verificar → entrenar solo lo correcto.
   Es la forma más simple de "aprender de pocos datos" con señal exacta.
4. **Replay priorizado** por pérdida en `replay.c`.
5. **Looped transformer** (compartir bloques) como opción de tamaño.
6. **BPE propio** para reducir secuencia.

## Límites honestos

- Un modelo de 500k–10M parámetros en C no alcanzará capacidades de un LLM
  grande en conocimiento general. Sí puede ser muy bueno en **tareas verificables
  y dominio acotado**.
- Escalar la inteligencia aquí es: más verificación + más cómputo al responder +
  datos sintéticos de calidad. No "más parámetros con los mismos datos".
- Cualquier mejora debe medirse en el set de control; si no sube, no se integra.

## Siguiente paso propuesto

Implementar el punto 1 (evaluación held-out) y el 2 (verificador de aritmética)
como módulos nuevos `src/eval.c` y `src/verify.c`, y después el bucle de
rechazo-muestreo en `src/train.c`. Confirmar antes de empezar si el dominio
inicial debe ser aritmética o código.
