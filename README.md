# MemCore

Dos motores con el mismo diseño:

| Versión | Dónde | Tamaño | Velocidad |
|---------|-------|--------|-----------|
| **v4** (`py/memcore4.py`, PyTorch) | GPU (Colab T4) | 7.2M parámetros, 4 bloques × 3 bucles (profundidad 12) | ~65k tokens/s |
| **v3** (`src/`, C puro) | CPU | 673k parámetros, 3 bloques × 2 bucles | ~10.8k tokens/s |

v4 añade tareas más difíciles (`mul2`: n×n dígitos, `chain`: a+b−c),
embedding por bucle, decodificación por lotes y entrenamiento fp16.

```sh
python py/memcore4.py train --steps 12000      # GPU
python py/memcore4.py eval --votes 5
python py/memcore4.py solve "4821+977"
```

## MemCore v3 (C)

Modelos pequeños que aprenden mucho con pocos datos. 100% C (OpenMP + AVX),
sin dependencias, entrena en CPU.

Diseño derivado de un autoanálisis de las debilidades de los LLM actuales
(`docs/AUTOANALISIS.md`): cada debilidad tiene un mecanismo concreto aquí.

```
                 ┌─────────────── datos sintéticos verificables ───────────────┐
                 │  generador → currículo adaptativo → minado de errores → STaR │
                 └──────────────────────────────┬──────────────────────────────┘
                                                ▼
 bytes ─► [ bloque 1 ─ bloque 2 ─ bloque 3 ] ×  1..N bucles  ─► logits
             pesos compartidos: profundidad variable (cómputo adaptativo)
                                                │
               inferencia: superficial → si duda, más profunda → votación
                                                │
                       memoria episódica kNN (aprende texto al instante)
```

## Mecanismos

| Mecanismo | Qué resuelve | Dónde |
|-----------|--------------|-------|
| Transformer en bucle (bloques compartidos, RoPE, RMSNorm, SwiGLU, embeddings atados) | Más profundidad por parámetro | `src/model.c` |
| Entrenamiento con nº de bucles aleatorio | El mismo modelo funciona superficial o profundo | `src/train.c` |
| Profundidad adaptativa en inferencia | Piensa más solo cuando duda (confianza < 0.9) | `src/infer.c` |
| Tareas verificables + solucionador de referencia | Datos infinitos y exactos; verificador para todo | `src/tasks.c` |
| Dígitos en orden de acarreo (LSB primero) | La suma se vuelve local y aprendible | `src/tasks.c` |
| Currículo adaptativo (sube nivel al 90%) | No gasta datos en lo dominado | `src/train.c` |
| Minado de errores | Cada fallo en evaluación vuelve al entrenamiento | `src/train.c` |
| STaR en la frontera | Intenta el nivel siguiente; entrena solo lo verificado | `src/train.c` |
| Votación + "no estoy seguro" | Más cómputo en inferencia, calibración | `src/infer.c` |
| Memoria episódica kNN | Aprende hechos nuevos sin gradientes | `src/memory.c` |
| Pérdida solo en respuestas, ejemplos empaquetados | Cada token de cómputo enseña algo | `src/train.c` |

Tareas: `add`, `sub`, `mul` (×1 dígito), `cmp`, `rev`, `sort`, `count`.

## Uso

```sh
make && make test                 # compila + gradcheck + kv-cache check
./memcore train -s 6000           # entrena (reanuda si existe model.mc)
./memcore eval -n 100             # precisión por tarea y nivel (+2 niveles nunca vistos)
./memcore solve "4821+977"        # resuelve con profundidad adaptativa
./memcore solve "4821+977" -k 9   # con votación de 9 muestras
./memcore chat                    # resuelve tareas; memoriza cualquier otra línea
./memcore train --text corpus.txt --mix 0.3   # mezcla modelado de texto
./memcore learn notas.txt         # memoria episódica desde archivo
./memcore gen "prompt" -m memory.knn
```

Configuración por defecto: d=128, 4 cabezas, 3 bloques × 2 bucles
(profundidad 6), ff=384, 673k parámetros.

## Verificación

- `make test`: gradiente numérico vs analítico en todos los tensores
  (incluido el reuso de bloques), y decodificación con KV-cache contra el
  forward en batch.
- `memcore eval` mide sobre problemas nuevos aleatorios; las columnas con `*`
  son longitudes mayores que cualquiera vista en entrenamiento.

## Resultados

Ver la sección de resultados en `docs/RESULTADOS.md`.
