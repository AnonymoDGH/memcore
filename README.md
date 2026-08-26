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
make          # binario memcore
make test     # tests rápidos
./memcore train data.txt   # aprendizaje continuo sobre un archivo
./memcore gen "hola"       # generar texto
```

## Estado

- [ ] Esqueleto del proyecto
- [x] README
