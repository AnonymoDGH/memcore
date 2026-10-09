# Resultados

Configuración: d=128, 4 cabezas, 3 bloques × 2 bucles, 673k parámetros,
CPU de 4 núcleos, ~10.8k tokens/s (el motor anterior: ~140 tokens/s).

## Entrenamiento 1: 6000 pasos (~29 min)

Última evaluación del currículo (64 problemas nuevos por tarea, nivel actual):

| Tarea | Nivel alcanzado / techo | Precisión |
|-------|-------------------------|-----------|
| add   | 5 / 5 dígitos | 80% |
| sub   | 5 / 5 dígitos | 84% |
| mul   | 3 / 4 dígitos (×1 dígito) | 83% |
| cmp   | 5 / 5 dígitos | 100% |
| rev   | 8 / 8 letras | 100% |
| sort  | 8 / 8 letras | 98% |
| count | 10 / 10 letras | 98% |

Pendiente: la evaluación completa (`memcore eval`) con niveles más largos que
los de entrenamiento, para medir la generalización de longitud.
