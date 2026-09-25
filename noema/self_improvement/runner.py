"""Recursive weight/recipe improvement with rollback, audit logs and resumption."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os
import tempfile
import time
import torch
from ..agent import NOEMAAgent
from .data import collect, ReplayBuffer, Transitions
from .evaluation import evaluate, promotion_gate
from .strategy import MetaController, Recipe, curriculum


@dataclass
class RunConfig:
    seed: int = 42
    generations: int = 3
    candidates: int = 3
    train_steps: int = 100
    batch_size: int = 64
    initial_episodes: int = 12
    fresh_episodes: int = 6
    eval_episodes: int = 12
    horizon: int = 24
    replay_capacity: int = 12000
    auxiliary_every: int = 8
    min_improvement: float = 0.005
    max_task_regression: float = 0.02
    patience: int = 3
    threads: int = 1
    device: str = "cpu"

    def validate(self):
        for key in ("generations", "candidates", "train_steps", "batch_size", "initial_episodes",
                    "fresh_episodes", "eval_episodes", "horizon", "auxiliary_every", "patience", "threads"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.seed < 0 or self.batch_size < 2 or self.eval_episodes < 2:
            raise ValueError("Nonnegative seed and batch/evaluation sizes >= 2 required")
        if self.candidates > len(MetaController.OPERATORS):
            raise ValueError("At most six distinct candidates per generation")
        if self.replay_capacity < 3 or not 0 <= self.min_improvement < 1 or not 0 <= self.max_task_regression < 1:
            raise ValueError("Invalid replay capacity or promotion tolerances")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("Supported devices: cpu, cuda, cuda:N")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu")


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def atomic_json(payload, path):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def model_bundle(agent):
    return {"format_version": 1, "agent_config": agent.config,
            "model_state": {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}}


def restore_model(bundle, device="cpu"):
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    agent = NOEMAAgent(**bundle["agent_config"], device=device)
    agent.load_state_dict(bundle["model_state"], strict=True)
    return agent


def load_agent(path, device="cpu"):
    """Load an inference checkpoint using PyTorch's restricted weights-only loader."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    agent = restore_model(bundle, device)
    agent.eval()
    return agent


def _checkpoint(agent, initial, config, generation, replay, recipe, controller,
                history, stale, initial_validation, validation):
    return {"format_version": 1, "agent": model_bundle(agent), "initial": initial,
            "optimizer": agent.optimizer.state_dict(), "config": asdict(config),
            "generation": generation, "replay": replay.data.state_dict(),
            "recipe": recipe.as_dict(), "controller": controller.state_dict(),
            "history": history, "stale_generations": stale,
            "initial_validation": initial_validation, "validation": validation.as_dict()}


def run(config, output_dir, resume=None):
    config.validate()
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Output is not empty. Use a new --output or --resume CHECKPOINT.")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    controller, recipe = MetaController(), Recipe()
    replay = ReplayBuffer(config.replay_capacity)
    if resume is None:
        champion = NOEMAAgent(latent_dim=32, n_world_model_levels=2,
                              num_slots=4, slot_dim=16, device=config.device)
        replay.add(collect(seed=config.seed, split="train", episodes_per_task=config.initial_episodes,
                           horizon=config.horizon))
        champion.dynamics.calibrate(replay.data.obs.to(champion.device),
                                    replay.data.next_obs.to(champion.device))
        initial = model_bundle(champion)
        generation, stale, history = 0, 0, []
    else:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        mutable = {"generations", "device", "threads", "patience"}
        mismatches = [key for key, value in state["config"].items()
                      if key not in mutable and value != getattr(config, key)]
        if mismatches:
            raise ValueError(f"Resume configuration differs: {mismatches}")
        champion = restore_model(state["agent"], config.device)
        champion.optimizer.load_state_dict(state["optimizer"])
        replay.add(Transitions(**state["replay"]))
        recipe = Recipe(**state["recipe"])
        controller.load_state_dict(state["controller"])
        generation, history = state["generation"], state["history"]
        stale, initial = state["stale_generations"], state["initial"]
    validation_data = collect(seed=config.seed, split="validation", episodes_per_task=config.eval_episodes,
                              horizon=config.horizon)
    initial_agent = restore_model(initial, config.device)
    initial_validation = evaluate(initial_agent, validation_data).as_dict()
    validation = evaluate(champion, validation_data)
    atomic_save(initial, output / "baseline.pt")
    atomic_json(asdict(config), output / "config.json")

    def persist():
        atomic_save(_checkpoint(champion, initial, config, generation, replay, recipe,
                                controller, history, stale, initial_validation, validation),
                    output / "checkpoint.pt")
        atomic_save(model_bundle(champion), output / "best.pt")
        atomic_json(history, output / "history.json")
    persist()
    last_requested = generation + config.generations
    for current in range(generation + 1, last_requested + 1):
        replay.add(collect(seed=config.seed, split="train", episodes_per_task=config.fresh_episodes,
                           horizon=config.horizon, round_index=current, agent=champion))
        parent_score = validation
        records, best, best_recipe, best_score, best_index = [], None, None, None, None
        proposals = controller.propose(recipe, config.candidates, current - 1)
        for index, (operator, proposal) in enumerate(proposals):
            candidate = champion.clone_for_training()
            candidate.train()
            for group in candidate.optimizer.param_groups:
                group["lr"] = proposal.learning_rate
            # Same random stream for every candidate within a generation.
            trial_seed = (config.seed + current * 100003) % (2**63 - 1)
            torch.manual_seed(trial_seed)
            sampler = torch.Generator().manual_seed(trial_seed)
            probabilities = curriculum(parent_score, proposal.focus)
            record = {"generation": current, "candidate": index, "operator": operator,
                      "recipe": proposal.as_dict(), "curriculum": probabilities,
                      "promoted": False, "parent_mse": parent_score.mse}
            trial_start = time.perf_counter()
            losses = []
            try:
                for step in range(config.train_steps):
                    batch = replay.sample(config.batch_size, probabilities, sampler)
                    result = candidate.learn_transitions(*batch, jepa_weight=proposal.jepa_weight,
                                                          auxiliary=step % config.auxiliary_every == 0)
                    losses.append(result["prediction_loss"])
                if not all(bool(torch.isfinite(t).all()) for t in candidate.state_dict().values()):
                    raise FloatingPointError("Candidate has non-finite state")
                score = evaluate(candidate, validation_data)
                accepted, gate = promotion_gate(parent_score, score, config.min_improvement,
                                                 config.max_task_regression)
                record.update(validation=score.as_dict(), gate=gate, gate_passed=accepted,
                              mean_training_loss=sum(losses) / len(losses))
                controller.update(operator, gate["relative_gain"] if accepted else min(0.0, gate["relative_gain"]))
                if accepted and (best_score is None or score.mse < best_score.mse):
                    best, best_recipe, best_score, best_index = candidate, proposal, score, index
            except (FloatingPointError, RuntimeError) as exc:
                # Preserve the incumbent. Numerical failures cannot promote themselves.
                record.update(gate_passed=False, gate={"reason": "training_error"}, error=str(exc))
                controller.update(operator, -1.0)
            record["seconds"] = time.perf_counter() - trial_start
            records.append(record)
        if best is not None:
            champion, recipe, validation = best, best_recipe, best_score
            records[best_index]["promoted"] = True
            stale = 0
        else:
            stale += 1
        history.extend(records)
        generation = current
        persist()
        print(f"Generation {generation}: validation MSE {parent_score.mse:.6f} -> "
              f"{validation.mse:.6f}; {'promoted ' + str(best_index) if best is not None else 'incumbent retained'}",
              flush=True)
        if stale >= config.patience:
            break
    # Final test is generated and opened only after all selection has finished.
    test_data = collect(seed=config.seed, split="test", episodes_per_task=config.eval_episodes,
                        horizon=config.horizon)
    test_initial, test_final = evaluate(initial_agent, test_data), evaluate(champion, test_data)
    gain = (test_initial.mse - test_final.mse) / max(test_initial.mse, 1e-12)
    report = {"format_version": 1, "scope": "Grounded next-state prediction in PhysicsPlayground",
              "generation": generation, "promotions": sum(r["promoted"] for r in history),
              "config": asdict(config), "initial_validation": initial_validation,
              "final_validation": validation.as_dict(), "initial_test": test_initial.as_dict(),
              "final_test": test_final.as_dict(), "test_relative_improvement": gain,
              "winning_recipe": recipe.as_dict(), "meta_controller": controller.state_dict(),
              "history": history, "replay_transitions": len(replay.data),
              "invocation_seconds": time.perf_counter() - started,
              "torch_version": str(torch.__version__),
              "status": "measured_prediction_improvement" if gain > 0 else "no_test_improvement",
              "limitations": ["Only simulated physics prediction was measured; not general intelligence.",
                              "Selection reuses validation; its confidence bound is a heuristic.",
                              "Final test is report-only and never enters training or proposal selection.",
                              "No arbitrary source-code rewriting; weights and training recipes evolve."]}
    atomic_json(report, output / "report.json")
    print(f"Final test MSE: {test_initial.mse:.6f} -> {test_final.mse:.6f} "
          f"({100 * gain:.2f}% reduction). Saved in {output}", flush=True)
    return report
