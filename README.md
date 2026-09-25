# NOEMA — Recursive Self-Improvement Lab

**Version 0.2.0 · project by Kaoru Aguilera Katayama**

This version adds recursive self-improvement **of weights and the training strategy**. The best agent generates new experience, clones itself into candidates, trains with different recipes, evaluates their results, and passes the weights and optimizer of the winner to the next generation. The controller learns which recipe modifications have worked.

It includes a real run with trained weights in `examples/trained_run/`. The measured scope is next-state prediction in the `PhysicsPlayground` simulator. It is a research prototype; these measurements do not establish AGI or unlimited self-improvement.

## Run it on your PC

You need Python 3.10 or later. From the unzipped folder:

```bash
python -m pip install -r requirements.txt
python run_rsi.py --resume examples/trained_run/checkpoint.pt --generations 3 --output runs/continuation
```

That command **continues the already-trained weights**, restores the transition memory, the Adam optimizer, and the recipe controller. `--generations 3` means three additional generations. The console prints promotions and the final evaluation. It may decide to keep the previous model: a generation does not guarantee progress.

To train from scratch:

```bash
python run_rsi.py --generations 4 --candidates 3 --train-steps 150 --eval-episodes 20 --output runs/from_scratch
```

To resume your own run:

```bash
python run_rsi.py --resume runs/from_scratch/checkpoint.pt --generations 3
```

It does not overwrite an existing results folder without `--resume`. You can also install the package with `python -m pip install -e ".[test]"` and use `noema-rsi`.

## Google Colab

Open `NOEMA_RSI_Colab.ipynb` in Colab and run its single code cell. It will ask you for this ZIP, install the dependencies, continue the included checkpoint, and download a ZIP of results. The cell uses CPU by default: the networks are small. You can change `DEVICE` to `"cuda"` if your session has a GPU available. CUDA is implemented, but the included validation was run on CPU.

## What Was Added

| Component | Actual Function |
|---|---|
| Dynamics ensemble | Predicts future observations from the observation and action, with three models and a shared encoder. |
| Transition learning | Optimizes over `(observation, action, next observation)` measured in the environment. |
| Bounded replay | Preserves experience by scenario, prevents validation/test data from entering training, and prevents one scenario from displacing all the others. |
| Experience generation | Combines random exploration with actions chosen by disagreement among the champion's ensemble. |
| Curriculum | Can devote more samples to scenarios with higher validation error, while maintaining a quota from the others. |
| Recipe mutations | Continues, raises/lowers the learning rate, changes curriculum emphasis, or changes the auxiliary JEPA weight. |
| Meta-controller | Uses the history of validation gains and an exploration bonus to choose which recipes to try. |
| Selection and rollback | Each candidate starts from the same champion. Only an approved winner replaces the champion; rejected candidates do not modify its weights or Adam state. |
| Persistence | Saves weights, optimizer, replay, lineage, generation, recipe, and meta-controller state. |
| Consolidation | `agent.consolidate()` performs gradient updates on stored real transitions. |
| Knowledge | `agent.build_knowledge()` connects S2 → S4 → S5 with unique identifiers and bounded schema memory. |

The observation encoder and ensemble are trained. S2/JEPA and the EFE transition model are periodically trained with auxiliary objectives. The analogy, relation, and coordination modules are preserved, but their quality is neither evaluated nor demonstrated by the physical metric of this package.

Each generation starts again from the champion selected by the previous one.

The system modifies parameters and recipes within the implemented space. It does not generate or arbitrarily rewrite source code.

## How an Improvement Is Decided

The scenarios are **free fall**, **impacts**, and **force control**; they share the original physics simulator. Each episode begins with an independent reset. The three sets use separate seed spaces:

- **Training:** used to calibrate normalization once and update the weights; it receives new experience in every generation.
- **Validation:** compares all candidates on exactly the same transitions. It also informs the curriculum and meta-controller.
- **Final test:** generated and evaluated after selection is closed. Its results are only reported.

To be promoted, a candidate must improve validation MSE by at least **0.5%**, not worsen any scenario by more than **2%**, have a finite state, and exceed an approximate lower bound on paired gain per episode. The normal interval is a heuristic; adaptively reusing validation does not provide a 95% statistical guarantee of future improvements.

The main MSE divides each coordinate error by the standard deviation of its changes observed during training, with a floor of 0.02, before squaring and averaging. The scale remains fixed across generations. MSE in original units and the persistence predictor `next observation = current observation` are also recorded. The initial model is equivalent to that reference because its output heads start at zero. The targets are observed states, so collapsing a latent representation cannot artificially reduce the main metric.

The final test compares the trained model with the initial model; it is not a comparison against other algorithms or an ablation of the meta-controller's contribution. Exploration and the one-step planner are implemented, but no improvement in reward or success on control tasks is reported.

## Output Files

| File | Contents |
|---|---|
| `best.pt` | Champion configuration and weights for inference. |
| `checkpoint.pt` | Complete experiment state for continuation. |
| `baseline.pt` | Initial weights, with the same training normalization. |
| `report.json` | Initial/final validation and test metrics, per-scenario results, and all candidates. |
| `history.json` | Recipes, times, rejection reasons, and promotions for each candidate. |
| `config.json` | Execution parameters. |

`checkpoint.pt` is the source for resuming. It is written using atomic replacement. If a generation is interrupted, it repeats from the last saved generation. The persisted memory corresponds to this experiment's replay; semantic schemas created manually during a session and the recurrent interaction state do not constitute a complete persistent session.

Resuming preserves the distribution, seeds, and training configuration. You can change the number of additional generations, device, threads, and patience. A new experiment with other parameters must use another folder. After continuation, the test set is reported again; the code never uses its values to decide candidates.

## Using the Trained Agent in Python

```python
import torch
from noema.self_improvement import load_agent
from noema.environments.playground import PhysicsPlayground

agent = load_agent("examples/trained_run/best.pt")
env = PhysicsPlayground(seed=123)
state = env.reset(seed=123)

# Inference does not perform optimizer steps.
result = agent(state.observation, state.proprio, state.extero)
action = result["action"][0]
next_state = env.step(action)
prediction = agent.predict_next(state.observation, action)

# For explicit learning during interaction:
agent.train()
agent.observe_transition(state.observation, action, next_state.observation)
agent.consolidate(n_steps=4, batch_size=32)

# Preserved experimental symbolic/analogical path:
knowledge = agent.build_knowledge(next_state.observation, domain="physics")
```

`predict_next()` supports batches; `forward()`, `plan_action()`, and `build_knowledge()` receive a single observation. `plan_action(obs, preferred_obs=target)` scores actions by predicted distance to the target and ensemble disagreement. For another environment, you can feed transitions of the configured dimensions to `learn_transitions()`; the included command-line runner is dedicated to `PhysicsPlayground` (32 observations, 4 actions).

## Verification

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The tests verify learning with real targets, set separation, evaluation immutability, finiteness with batches of one, rollback of failed candidates, consolidation with weight updates, and equivalence between continuous/resumed execution on CPU. See `VALIDACION.md` for the executed result.

## Corrections to the Original Project

- The importable `noema/` package and installation configuration were created.
- Internal training in `forward()` using a noisy copy of the observation was removed; learning now occurs explicitly from real transitions.
- Undefined variance with batches of one was fixed, and silent replacement of NaN losses with a constant was removed.
- The EMA update was moved to after the optimizer step, and statistics are frozen during evaluation.
- Higher S2 levels receive the representation of the real next state.
- Simulator observations are no longer views that change when `step()` is executed.
- Reuse of the same ID for knowledge schemas was fixed.
- Old experiments with preassigned successes were preserved in `legacy/`; they are not counted as evidence for this version. The active benchmark writes measurements and does not issue an AGI verdict.

## Implementation References

- [PyTorch: saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html): `state_dict` checkpoints and loading with `weights_only=True`.
- [PyTorch: evaluation modes and gradients](https://docs.pytorch.org/docs/stable/notes/autograd.html): evaluation without parameter updates.

The Apache 2.0 license from `LICENSE` is preserved.
