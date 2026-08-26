# MemCore

Micro-modelo de generación de texto que **aprende continuamente** con pocos datos,
diseñado para hardware mínimo (8 GB RAM, poco disco) e implementado **100% en C**.

## Idea central

El conocimiento nuevo **no vive en los pesos del núcleo**, sino en una
**memoria neuronal a largo plazo** que se actualiza en runtime mediante una
métrica de *sorpresa* (inspirado en *Titans: Learning to Memorize at Test Time*,
Google Research 2025 — arxiv.org/abs/2501.00663).

```
token ──> [Atención local pequeña] ──┬──> logits ──> predicción
              │                      ▲
              ▼                      │ suma proyectada
        [Memoria Neuronal MLP] ──────┘
              ▲
        actualización ONLINE por gradiente de sorpresa:
        M_t = M_(t-1)·(1 − λ·sorpresa) + lr · ∇(pérdida_local)
```

- **Sorpresa = |gradiente| de la pérdida del token actual respecto a la memoria.**
  Token esperado → gradiente ≈ 0 → no se gasta cómputo.
  Token nuevo/raro → se memoriza al vuelo mientras se lee.
- **Anti-olvido catastrófico**: decay λ + replay buffer circular re-muestreado
  cada N pasos (validado por arxiv.org/abs/2504.17780 y arXiv:2402.18865).
- **Sin Python**: patrón de entrenamiento manual inspirado en llm.c
  (github.com/karpathy/llm.c): forward + backward + Adam escritos a mano.

## Presupuesto de recursos

| Componente            | Costo                          |
|-----------------------|--------------------------------|
| Núcleo atención       | ~10M params int8 (~10 MB mmap) |
| Memoria neuronal MLP  | ~2M params fp32 (~8 MB RAM)    |
| Replay buffer         | ~5 MB                          |
| Total en entrenamiento| < 300 MB RAM                   |

## Módulos

| Archivo             | Rol                                                        |
|---------------------|------------------------------------------------------------|
| `src/tokenizer.c`   | Tokenizador byte-level (256 vocab, sin BPE todavía)        |
| `src/tensor.c`      | Utilidades base: alloc, matmul, softmax, layernorm         |
| `src/attention.c`   | Núcleo transformer pequeño (forward + backward)            |
| `src/neural_mem.c`  | Memoria neuronal con métrica de sorpresa (aprendizaje online) |
| `src/replay.c`      | Replay buffer circular anti-olvido                         |
| `src/train.c`       | Bucle de entrenamiento continuo                            |
| `src/main.c`        | Demo end-to-end                                            |

## Compilar

```sh
gcc -O2 -Wall -Wextra -std=c11 -Isrc -o memcore.exe \
    src/tensor.c src/tokenizer.c src/attention.c src/neural_mem.c \
    src/replay.c src/train.c src/main.c -lm
```

## Uso

```sh
./memcore info                          # arquitectura y nº de parámetros
./memcore train data.txt -n 3000        # aprendizaje continuo (resume si hay checkpoint)
./memcore gen "el sol" 140              # generar (usa model.core/model.mem)
    -t 0.5   temperatura                -m 0/1  memoria neuronal on/off
./memcore chat                          # REPL: aprende tu línea ANTES de responder
    :save :exit :temp 0.7
```

El entrenamiento es **incremental**: cada pasada por `train` reanuda desde el
checkpoint (`model.core` + `model.mem`), así que puedes alimentarlo con archivos
nuevos sin reentrenar desde cero.

## Verificación

`config/test.c` hace **gradient check numérico** del backward completo
(diferencias centrales vs gradiente analítico en 8 tensores):

```sh
gcc -g -O0 -Wall -std=c11 -Isrc -o gradcheck.exe config/test.c \
    src/tensor.c src/tokenizer.c src/attention.c src/neural_mem.c \
    src/replay.c src/train.c -lm && ./gradcheck.exe
```

## Resultados actuales

- Corpus demo (10 líneas en español, ~18 KB): loss 4.3 → **0.44** en 3 pasadas.
- Generación coherente con el corpus: `"el sol sale por la"` → `" manana…"`.
- La memoria neuronal muestra sorpresa adaptativa (se calma con texto esperable,
  se activa con contenido nuevo).

## Lecciones de implementación (hard-won)

1. Los buffers de gradiente que acumulan con `+=` deben inicializarse a cero
   (memoria no inicializada → NaNs silenciosos → divergencia).
2. Sin clipping de norma global, Adam a 3e-4 diverge en este micro-modelo;
   con clip a 1.0, converge estable.
3. En generación, la predicción debe leerse en la **posición del último token
   válido**: la atención causal hace que el padding posterior sea inerte.
4. Contextos sintéticos (padding masivo, prompts repetidos) descarrilan al
   modelo; sembrar desde el replay buffer mantiene la distribución real.

## Estado

- [x] Núcleo transformer con forward/backward/Adam en C puro
- [x] Memoria neuronal con escritura por sorpresa + persistencia
- [x] Replay buffer anti-olvido
- [x] Gradient check numérico verde
- [x] Entrenamiento continuo incremental entre sesiones
- [ ] Cuantización int8 de pesos (hoy fp32)
- [ ] KV-cache para generación incremental
