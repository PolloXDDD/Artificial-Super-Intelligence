# Validación ejecutada — NOEMA 0.2.0

Fecha de ejecución: 2026-09-25.

- Python: 3.12.14.
- PyTorch: 2.8.0+cpu; NumPy: 2.3.5.
- Dispositivo: CPU; un hilo de PyTorch.
- Pruebas: **21 aprobadas** (`python -m pytest -q`).
- Entrenamiento incluido: 4 generaciones, 3 candidatos por generación, 150 pasos por candidato, batch 64, semilla 42.
- Promociones: **3**; replay final: **2592 transiciones**.
- Conjunto de prueba: 60 episodios independientes, 24 pasos por episodio, **1440 transiciones**.
- Tiempo del experimento de referencia en esta máquina: **13.52 segundos**. No es una estimación para otros equipos.

## Predicción en datos de prueba separados

| Escenario | MSE inicial | MSE final | Reducción |
|---|---:|---:|---:|
| free_fall | 0.238174 | 0.039518 | 83.41 % |
| impacts | 0.532107 | 0.297201 | 44.15 % |
| control | 0.478547 | 0.197951 | 58.64 % |
| **Total** | **0.416276** | **0.178223** | **57.19 %** |

El MSE usa escalas de cambio ajustadas únicamente con entrenamiento y congeladas. La referencia inicial es el predictor de persistencia. Los resultados prueban una mejora de predicción dentro de este simulador; no se midió inteligencia general ni superioridad frente a entrenamiento convencional.

## Historial de selección

| Generación | MSE de validación del campeón | Decisión |
|---|---:|---|
| 1 | 0.208756 | Promover candidato 2 (faster) |
| 2 | 0.208756 | Conservar campeón anterior |
| 3 | 0.189302 | Promover candidato 0 (continue) |
| 4 | 0.184579 | Promover candidato 1 (slower) |

Los candidatos de cada generación heredan el mismo padre. Los cambios aceptados pasan a ser el padre de la siguiente generación. El archivo `examples/trained_run/history.json` contiene todos los candidatos, incluidos los rechazados, sus recetas y el motivo de cada decisión.

## Reproducir

```bash
python run_rsi.py --generations 4 --candidates 3 --train-steps 150 --eval-episodes 20 --output runs/reproduccion
```

Las pruebas incluyen que reanudar y entrenar sin interrupción producen los mismos pesos en el entorno CPU validado. Otras versiones de PyTorch o dispositivos pueden producir diferencias numéricas. CUDA no se ejecutó en esta validación.

## Estado anterior

Las 9 pruebas unitarias originales podían pasar aunque el agente completo produjera pérdidas de nivel `NaN` con una sola observación y las sustituyera por `0.1`. Esa ejecución se reprodujo antes de corregirla. Las nuevas pruebas ejercitan esos fallos y exigen cambios medidos de pesos y predicción.
