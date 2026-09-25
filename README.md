# NOEMA — Recursive Self-Improvement Lab

**Versión 0.2.0 · proyecto de Kaoru Aguilera Katayama**

Esta versión añade auto-mejora recursiva **de pesos y de la estrategia de entrenamiento**. El mejor agente genera experiencia nueva, se clona en candidatos, entrena con recetas distintas, evalúa sus resultados y transmite los pesos y el optimizador del ganador a la siguiente generación. El controlador aprende qué modificaciones de la receta han funcionado.

Incluye una ejecución real con pesos entrenados en `examples/trained_run/`. El alcance medido es predicción del siguiente estado en el simulador `PhysicsPlayground`. Es un prototipo de investigación; estas mediciones no establecen AGI ni auto-mejora ilimitada.

## Ejecutarlo en tu PC

Necesitas Python 3.10 o posterior. Desde la carpeta descomprimida:

```bash
python -m pip install -r requirements.txt
python run_rsi.py --resume examples/trained_run/checkpoint.pt --generations 3 --output runs/continuacion
```

Ese comando **continúa los pesos ya entrenados**, recupera la memoria de transiciones, el optimizador Adam y el controlador de recetas. `--generations 3` significa tres generaciones adicionales. La consola imprime las promociones y la evaluación final. Puede decidir conservar el modelo anterior: una generación no garantiza progreso.

Para entrenar desde cero:

```bash
python run_rsi.py --generations 4 --candidates 3 --train-steps 150 --eval-episodes 20 --output runs/desde_cero
```

Para reanudar tu propia ejecución:

```bash
python run_rsi.py --resume runs/desde_cero/checkpoint.pt --generations 3
```

No sobrescribe una carpeta de resultados existente sin `--resume`. También puedes instalar el paquete con `python -m pip install -e ".[test]"` y usar `noema-rsi`.

## Google Colab

Abre `NOEMA_RSI_Colab.ipynb` en Colab y ejecuta su única celda de código. Te pedirá este ZIP, instalará las dependencias, continuará el checkpoint incluido y descargará un ZIP de resultados. La celda usa CPU por defecto: las redes son pequeñas. Puedes cambiar `DEVICE` a `"cuda"` si tu sesión dispone de GPU. CUDA está implementado, pero la validación incluida se ejecutó en CPU.

## Qué se añadió

| Componente | Función real |
|---|---|
| Ensemble de dinámica | Predice observaciones futuras desde la observación y la acción, con tres modelos y un codificador compartido. |
| Aprendizaje de transiciones | Optimiza sobre `(observación, acción, siguiente observación)` medidos en el entorno. |
| Replay acotado | Conserva experiencia por escenario, impide introducir validación/prueba en entrenamiento y evita que un escenario desplace a todos los demás. |
| Generación de experiencia | Combina exploración aleatoria con acciones elegidas por desacuerdo del ensemble del campeón. |
| Currículum | Puede dedicar más muestras a los escenarios con mayor error de validación, manteniendo una cuota de los otros. |
| Mutaciones de receta | Continúa, sube/baja la tasa de aprendizaje, cambia el énfasis del currículum o el peso auxiliar JEPA. |
| Meta-controlador | Usa el historial de ganancias de validación y un bono de exploración para elegir qué recetas probar. |
| Selección y retroceso | Cada candidato parte del mismo campeón. Solo un ganador aprobado sustituye al campeón; los rechazados no modifican sus pesos ni Adam. |
| Persistencia | Guarda pesos, optimizador, replay, linaje, generación, receta y estado del meta-controlador. |
| Consolidación | `agent.consolidate()` realiza gradientes sobre transiciones reales guardadas. |
| Conocimiento | `agent.build_knowledge()` conecta S2 → S4 → S5 con identificadores únicos y memoria de esquemas acotada. |

Se entrenan el codificador de observaciones y el ensemble. Periódicamente se entrenan S2/JEPA y el modelo de transición EFE con objetivos auxiliares. Los módulos de analogía, relaciones y coordinación se conservan, pero su calidad no se evalúa ni se demuestra mediante la métrica física de este paquete.

Cada generación vuelve a empezar desde el campeón elegido por la anterior.

El sistema modifica parámetros y recetas dentro del espacio implementado. No genera ni reescribe código fuente arbitrario.

## Cómo se decide una mejora

Los escenarios son **caída libre**, **impactos** y **control por fuerzas**; comparten el simulador físico original. Cada episodio comienza con un reinicio independiente. Los tres conjuntos usan espacios de semillas separados:

- **Entrenamiento:** sirve para calibrar la normalización una vez y actualizar los pesos; recibe experiencia nueva en cada generación.
- **Validación:** compara todos los candidatos sobre exactamente las mismas transiciones. También informa al currículum y al meta-controlador.
- **Prueba final:** se genera y evalúa después de cerrar la selección. Sus resultados solo se reportan.

Para promoverse, un candidato debe mejorar el MSE de validación al menos un **0,5 %**, no empeorar ningún escenario más de un **2 %**, tener estado finito y superar un límite inferior aproximado de ganancia pareada por episodio. El intervalo normal es una heurística; reutilizar validación de forma adaptativa no da una garantía estadística del 95 % sobre futuras mejoras.

El MSE principal divide cada error de coordenada entre la desviación de sus cambios observada en entrenamiento, con suelo 0,02, antes de elevar al cuadrado y promediar. La escala permanece fija entre generaciones. También se registra el MSE en unidades originales y el predictor de persistencia `siguiente observación = observación actual`. El modelo inicial equivale a esa referencia, porque sus cabezas de salida comienzan en cero. Los objetivos son estados observados, por lo que colapsar una representación latente no puede reducir artificialmente la métrica principal.

La prueba final compara el modelo entrenado con el inicial; no es una comparación contra otros algoritmos ni una ablación de la contribución del meta-controlador. La exploración y el planificador de un paso están implementados, pero no se reporta una mejora de recompensa o éxito en tareas de control.

## Archivos de salida

| Archivo | Contenido |
|---|---|
| `best.pt` | Configuración y pesos del campeón para inferencia. |
| `checkpoint.pt` | Estado completo del experimento para continuar. |
| `baseline.pt` | Pesos iniciales, con la misma normalización de entrenamiento. |
| `report.json` | Métricas iniciales/finales de validación y prueba, resultados por escenario y todos los candidatos. |
| `history.json` | Recetas, tiempos, motivos de rechazo y promociones de cada candidato. |
| `config.json` | Parámetros de ejecución. |

`checkpoint.pt` es la fuente de reanudación. Se escribe mediante sustitución atómica. Si se interrumpe una generación, se repite desde la última generación guardada. La memoria persistida corresponde al replay de este experimento; los esquemas semánticos creados manualmente durante una sesión y el estado recurrente de interacción no son una sesión persistente completa.

Reanudar mantiene la distribución, las semillas y la configuración de entrenamiento. Puedes cambiar el número adicional de generaciones, el dispositivo, los hilos y la paciencia. Un nuevo experimento con otros parámetros debe usar otra carpeta. Tras continuar se vuelve a reportar el conjunto de prueba; el código nunca utiliza sus valores para decidir candidatos.

## Usar el agente entrenado en Python

```python
import torch
from noema.self_improvement import load_agent
from noema.environments.playground import PhysicsPlayground

agent = load_agent("examples/trained_run/best.pt")
env = PhysicsPlayground(seed=123)
state = env.reset(seed=123)

# La inferencia no realiza pasos del optimizador.
result = agent(state.observation, state.proprio, state.extero)
action = result["action"][0]
next_state = env.step(action)
prediction = agent.predict_next(state.observation, action)

# Para aprendizaje explícito durante interacción:
agent.train()
agent.observe_transition(state.observation, action, next_state.observation)
agent.consolidate(n_steps=4, batch_size=32)

# Ruta simbólica/analógica experimental conservada:
knowledge = agent.build_knowledge(next_state.observation, domain="physics")
```

`predict_next()` admite lotes; `forward()`, `plan_action()` y `build_knowledge()` reciben una sola observación. `plan_action(obs, preferred_obs=objetivo)` puntúa acciones por distancia predicha al objetivo y desacuerdo del ensemble. Para otro entorno puedes alimentar transiciones de las dimensiones configuradas a `learn_transitions()`; el ejecutor de línea de comandos incluido está dedicado a `PhysicsPlayground` (32 observaciones, 4 acciones).

## Verificación

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Las pruebas comprueban aprendizaje con objetivos reales, separación de conjuntos, inmutabilidad de evaluación, finitud con lotes de uno, retroceso de candidatos fallidos, consolidación con actualización de pesos y equivalencia de ejecución continua/reanudada en CPU. Véase `VALIDACION.md` para el resultado ejecutado.

## Correcciones del proyecto original

- Se creó el paquete importable `noema/` y la configuración de instalación.
- Se retiró el entrenamiento interno de `forward()` con una copia ruidosa de la observación; ahora se aprende explícitamente de transiciones reales.
- Se corrigió la varianza indefinida con lotes de uno y se eliminó la sustitución silenciosa de pérdidas NaN por una constante.
- Se trasladó la actualización EMA a después del paso del optimizador y se congelaron estadísticas durante evaluación.
- Los niveles superiores de S2 reciben la representación del siguiente estado real.
- Las observaciones del simulador ya no son vistas que cambian al ejecutar `step()`.
- Se corrigió la reutilización del mismo ID para esquemas de conocimiento.
- Los experimentos antiguos con éxitos preasignados se conservaron en `legacy/`; no se cuentan como pruebas de esta versión. El benchmark activo escribe mediciones y no emite un veredicto de AGI.

## Referencias de implementación

- [PyTorch: guardar y cargar modelos](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html): checkpoints de `state_dict` y carga con `weights_only=True`.
- [PyTorch: modos de evaluación y gradientes](https://docs.pytorch.org/docs/stable/notes/autograd.html): evaluación sin actualización de parámetros.

Se conserva la licencia Apache 2.0 de `LICENSE`.
