# Autoanálisis: cómo soy y qué cambiaría en la siguiente generación

Límite honesto: no tengo acceso a mis pesos ni a detalles internos de mi
entrenamiento. Lo que sigue se basa en lo que sé públicamente sobre modelos
como yo (transformers decodificadores entrenados con enormes corpus y luego
ajustados con retroalimentación) y en mis fallos observables.

## Cómo funciono

| Rasgo | Consecuencia |
|-------|--------------|
| Transformer decodificador, predice el siguiente token | Todo "razonamiento" es generación de texto |
| Cómputo fijo por token (misma profundidad para "2+2" y para una prueba) | Gasto de más en lo fácil y me quedo corto en lo difícil |
| Pesos congelados después del entrenamiento | No aprendo de esta conversación; olvido todo al cerrar la sesión |
| Entrenado con cantidades enormes de texto | Muy ineficiente en datos: un humano aprende sumas con decenas de ejemplos |
| Tokenizador por subpalabras | Los números se parten en trozos arbitrarios, por eso fallo en aritmética exacta |
| Sin verificador interno | Puedo afirmar algo falso con la misma seguridad que algo verdadero |
| Mejoro cuando "pienso en voz alta" o comparo varias respuestas | Más cómputo en inferencia sube la precisión |

## Siguiente generación: cada debilidad → un mecanismo

| Debilidad | Mecanismo en MemCore v3 | Archivo |
|-----------|-------------------------|---------|
| Cómputo fijo | **Profundidad adaptativa**: bloques compartidos aplicados en bucle; se entrena con nº de bucles aleatorio y en inferencia se piensa más si la entropía es alta | `src/model.c` |
| Hambre de datos | **Datos sintéticos verificables + currículo adaptativo**: el generador produce infinitos ejemplos exactos; la dificultad sube solo al dominar el nivel | `src/tasks.c`, `src/train.c` |
| No aprende de sus errores | **Minado de errores**: cada fallo en evaluación vuelve al entrenamiento | `src/train.c` |
| Sin auto-mejora | **STaR en la frontera**: el modelo intenta problemas más difíciles que su nivel; solo lo verificado como correcto se usa para entrenar | `src/train.c` |
| Alucina | **Verificador + votación**: varias respuestas, voto mayoritario y "no sé" si no hay consenso | `src/main.c` |
| Pesos congelados | **Memoria episódica kNN**: aprende texto nuevo al instante sin reentrenar | `src/memory.c` |
| Tokenización rota para números | **Bytes + dígitos invertidos** (Lee et al., 2023): la suma se calcula en el orden del acarreo | `src/tasks.c` |

## Qué no puedo prometer

Un modelo de ~1M de parámetros en CPU no tendrá conocimiento general ni
lenguaje rico. Lo que sí se puede medir aquí es **inteligencia por dato y por
FLOP** en tareas verificables, y la generalización a problemas más largos que
los vistos. Ese es el criterio de éxito de esta generación: números de
evaluación, no impresiones.
