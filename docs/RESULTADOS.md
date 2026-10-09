# Resultados

Todas las cifras son precisión exacta sobre problemas nuevos aleatorios,
verificados por el solucionador de referencia. `*` = longitud mayor que
cualquiera vista en entrenamiento (generalización).

## v4 — 510M parámetros (Colab, Tesla T4)

18 bloques, d=1536, 12 cabezas, ff=4096. ~3000 pasos × 8.2k tokens
(~24.6M tokens, ~2 h). Fp16, gradient checkpointing, acumulación ×4.
Evaluación: 200 problemas por celda.

| tarea | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 |
|-------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|
| add   | 100 | 100 | 100 | 98 | 66 | 0 | 0* | 0* | | | | | | |
| sub   | 100 | 100 | 100 | 96 | 77 | 0 | 0* | 0* | | | | | | |
| mul   | 100 | 100 | 99 | 68 | 12* | 6* | | | | | | | | |
| mul2  | 100 | 69 | 29 | 21* | 4* | | | | | | | | | |
| chain | 100 | 97 | 90 | 26 | 0* | 0* | | | | | | | | |
| cmp   | 100 | 100 | 92 | 94 | 91 | 80 | 68* | 16* | | | | | | |
| rev   | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 8* | 0* | | |
| sort  | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 98 | 98 | 97 | 44* | 0* | | |
| count | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100* | 100* |

**Media dentro del techo de entrenamiento: 91.2%.**

## v4 — 7.2M parámetros (checkpoint del paso 2000)

4 bloques × 3 bucles. La corrida se interrumpió en el paso 2100; el
currículo antiguo (un nivel cada 500 pasos) solo la llevó al nivel 2.

| | Media dentro del techo |
|-|------------------------|
| greedy | 31.6% |
| votación de 5 | 31.9% |

Domina L1–L2 de todo (88–100%) y cae a ~0% desde L3: nunca entrenó ahí.
**No es una comparación limpia con el de 510M** (distinto número de pasos y
currículo); sirve como referencia, no como medida de escala.

## v3 — 673k parámetros (C, CPU)

6000 pasos (~29 min, 4 núcleos). Última evaluación del currículo:
add/sub 5 dígitos 80–84%, mul 3 dígitos 83%, cmp/rev/sort/count en su techo
al 98–100%.

## Lectura honesta

- **Lo que funcionó**: tareas algorítmicas de cadenas (rev, sort, count)
  quedan resueltas; `count` incluso generaliza a longitudes nunca vistas
  (L13–L14 al 100%) y `cmp` parcialmente (L7: 68%).
- **Lo que no**: la aritmética no generaliza en longitud (0% en cuanto pasa
  del nivel entrenado). Es la limitación conocida de las posiciones
  absolutas/RoPE para sumar; la siguiente mejora es *Abacus embeddings*
  (posición del dígito dentro del número; McLeish et al., 2024).
- `chain` (dos operaciones) es lo más difícil: 26% en L4.
- La votación casi no ayuda cuando el modelo falla de forma sistemática
  (+0.3 pts): sirve contra errores aleatorios, no contra no saber.
- Incidente: al reanudar el de 510M sin estado del optimizador la pérdida
  saltó de 0.03 a 1.55; se recuperó, pero costó pasos. Los checkpoints ya
  guardan el optimizador.

## Siguiente paso recomendado

1. Abacus embeddings para aritmética (generalización de longitud).
2. Repetir 7M vs 510M con el mismo currículo y mismos tokens para medir
   eficiencia por dato de forma limpia.
