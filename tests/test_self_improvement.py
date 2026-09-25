import copy
from dataclasses import replace
import json
import torch
import pytest
from noema.agent import NOEMAAgent
from noema.core.jepa import JEPAModule
from noema.subsystems.s2_world_model import HierarchicalWorldModel
from noema.environments.playground import PhysicsPlayground
from noema.self_improvement.data import collect, ReplayBuffer, Transitions
from noema.self_improvement.evaluation import Evaluation, evaluate, promotion_gate
from noema.self_improvement.runner import RunConfig, run, load_agent
from noema.self_improvement.strategy import MetaController, Recipe, curriculum


def small_agent():
    return NOEMAAgent(latent_dim=16, n_world_model_levels=1, num_slots=2, slot_dim=8)


def test_real_observations_do_not_alias_environment_state():
    env = PhysicsPlayground(seed=3)
    state = env.reset(seed=3)
    snapshot = state.observation.clone()
    env.step(torch.ones(4))
    assert torch.equal(state.observation, snapshot)


def test_single_observation_jepa_has_finite_loss_and_gradients():
    model = JEPAModule(8, 16, 4)
    result = model(torch.rand(1, 8), torch.rand(1, 8), torch.rand(1, 4))
    result['total_loss'].backward()
    assert torch.isfinite(result['total_loss'])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(model.running_var).all()


def test_single_level_and_eval_do_not_update_ema_or_statistics():
    model = HierarchicalWorldModel(obs_dim=8, n_levels=1, latent_dim=16, num_slots=2, slot_dim=8)
    model.eval()
    before = copy.deepcopy(model.state_dict())
    obs, nxt, action = torch.rand(3, 8), torch.rand(3, 8), torch.rand(3, 4)
    one = model([obs, nxt], [action])
    two = model([obs, nxt], [action])
    assert torch.equal(one['z_t'], two['z_t'])
    assert one['cross_level_loss'] == 0
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_split_isolation_reproducibility_and_replay_bound():
    args = dict(seed=7, episodes_per_task=3, horizon=4)
    training = collect(split='train', **args)
    same = collect(split='train', **args)
    validation = collect(split='validation', **args)
    testing = collect(split='test', **args)
    assert torch.equal(training.obs, same.obs)
    assert not set(training.episodes.tolist()) & set(validation.episodes.tolist())
    assert not set(validation.episodes.tolist()) & set(testing.episodes.tolist())
    assert not torch.equal(training.obs, validation.obs)
    replay = ReplayBuffer(capacity=18)
    replay.add(training)
    replay.add(collect(split='train', round_index=1, **args))
    assert len(replay.data) <= 18
    assert set(replay.data.tasks.tolist()) == {0, 1, 2}
    with pytest.raises(ValueError, match='Only training'):
        replay.add(validation)
    with pytest.raises(ValueError, match='provenance'):
        Transitions(**{**validation.state_dict(), 'split': 'train'})


def test_grounded_learning_improves_unseen_transitions_and_encoder():
    agent = small_agent()
    train = collect(seed=12, split='train', episodes_per_task=4, horizon=12)
    validation = collect(seed=12, split='validation', episodes_per_task=4, horizon=12)
    agent.dynamics.calibrate(train.obs, train.next_obs)
    before = evaluate(agent, validation)
    encoder_before = agent.obs_encoder[0].weight.detach().clone()
    for _ in range(45):
        indices = torch.randint(len(train), (64,))
        agent.learn_transitions(train.obs[indices], train.actions[indices], train.next_obs[indices], auxiliary=False)
    after = evaluate(agent, validation)
    assert after.mse < before.mse * 0.9
    assert not torch.equal(encoder_before, agent.obs_encoder[0].weight)
    assert after.mse < after.persistence_mse


def test_evaluation_is_read_only_and_restores_submodule_modes():
    agent = small_agent()
    agent.s2.eval()
    data = collect(seed=10, split='validation', episodes_per_task=2, horizon=3)
    before = copy.deepcopy(agent.state_dict())
    modes = [m.training for m in agent.modules()]
    opt_before = copy.deepcopy(agent.optimizer.state_dict())
    a, b = evaluate(agent, data), evaluate(agent, data)
    assert a == b
    assert all(torch.equal(v, agent.state_dict()[k]) for k, v in before.items())
    assert modes == [m.training for m in agent.modules()]
    assert opt_before == agent.optimizer.state_dict()
    assert len(agent.transition_memory) == 0


def test_inference_never_trains_and_sleep_really_updates_weights():
    agent = small_agent()
    env = PhysicsPlayground(seed=4)
    state = env.reset()
    params = {k: p.detach().clone() for k, p in agent.named_parameters()}
    result = agent(state.observation, state.proprio, state.extero)
    assert result['action'].shape == (1, 4)
    assert int(agent.updates) == 0
    assert all(torch.equal(p, dict(agent.named_parameters())[k]) for k, p in params.items())
    nxt = env.step(result['action'][0])
    agent.observe_transition(state.observation, result['action'], nxt.observation, learn=False)
    assert int(agent.updates) == 0
    result = agent.consolidate(n_steps=2, batch_size=4)
    assert result['n_replayed'] == 8 and int(agent.updates) == 2
    assert any(not torch.equal(p, dict(agent.named_parameters())[k]) for k, p in params.items())
    agent.eval()
    with pytest.raises(RuntimeError, match='train'):
        agent.learn_transitions(state.observation, torch.zeros(4), nxt.observation)


def metric(value, tasks=None, errors=None):
    return Evaluation(value, value, 1.0, tasks or {'a': value, 'b': value},
                      errors or [value] * 6, list(range(6)), 60)


def test_promotion_rejects_nonfinite_regression_and_uncertain_gains():
    parent = metric(1.0)
    assert promotion_gate(parent, metric(0.8))[0]
    assert not promotion_gate(parent, metric(1.1))[0]
    assert not promotion_gate(parent, metric(0.8, {'a': 0.5, 'b': 1.1}))[0]
    assert not promotion_gate(parent, metric(float('nan')))[0]
    assert not promotion_gate(parent, metric(0.9, errors=[0.1, 1.7, 0.1, 1.7, 0.1, 1.7]))[0]
    assert not promotion_gate(parent, parent)[0]


def test_meta_controller_changes_proposals_and_balances_weaknesses():
    controller = MetaController()
    controller.update('focus_weakness', 0.8)
    proposals = controller.propose(Recipe(), 6, 0)
    assert proposals[1][0] == 'focus_weakness'
    probs = curriculum(Evaluation(1, 1, 1, {'free_fall': 1, 'impacts': 4, 'control': 1}, [], [], 1), 0.6)
    assert probs[1] > probs[0] > 0 and sum(probs) == pytest.approx(1)
    for _, recipe in proposals:
        assert 0.00005 <= recipe.learning_rate <= 0.005
        assert 0 <= recipe.focus <= 0.75


def test_resume_matches_uninterrupted_training_and_restricted_load(tmp_path):
    config = RunConfig(generations=2, candidates=1, train_steps=5, batch_size=8,
                       initial_episodes=2, fresh_episodes=2, eval_episodes=2, horizon=4,
                       replay_capacity=120, patience=4)
    whole = run(config, tmp_path / 'whole')
    run(replace(config, generations=1), tmp_path / 'split')
    resumed = run(replace(config, generations=1), tmp_path / 'split', tmp_path / 'split/checkpoint.pt')
    a, b = load_agent(tmp_path / 'whole/best.pt'), load_agent(tmp_path / 'split/best.pt')
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert whole['final_test'] == resumed['final_test']
    assert whole['meta_controller'] == resumed['meta_controller']
    assert resumed['generation'] == 2 and len(resumed['history']) == 2
    assert json.loads((tmp_path / 'split/report.json').read_text())['generation'] == 2
    with pytest.raises(FileExistsError):
        run(config, tmp_path / 'whole')
    with pytest.raises(ValueError, match='differs'):
        run(replace(config, seed=4), tmp_path / 'bad', tmp_path / 'split/checkpoint.pt')


def test_failed_candidate_leaves_incumbent_and_all_failures_are_logged(tmp_path, monkeypatch):
    def fail(candidate, *args, **kwargs):
        with torch.no_grad():
            next(candidate.parameters()).add_(123.0)
        raise FloatingPointError('injected numerical failure after candidate mutation')
    monkeypatch.setattr(NOEMAAgent, 'learn_transitions', fail)
    config = RunConfig(generations=1, candidates=2, train_steps=2, initial_episodes=2,
                       fresh_episodes=2, eval_episodes=2, horizon=3)
    report = run(config, tmp_path / 'failed')
    base, best = load_agent(tmp_path / 'failed/baseline.pt'), load_agent(tmp_path / 'failed/best.pt')
    assert all(torch.equal(v, best.state_dict()[k]) for k, v in base.state_dict().items())
    assert report['promotions'] == 0
    assert all(item['gate']['reason'] == 'training_error' for item in report['history'])
    assert report['test_relative_improvement'] == 0.0


def test_knowledge_schema_ids_and_indexes_are_bounded():
    agent = small_agent()
    for i in range(5):
        result = agent.build_knowledge(torch.rand(32), domain=f'domain_{i % 2}', capacity=3)
        assert result['schema_id'] == f'schema_{i}'
    assert len(agent.s4.schemas) == 3
    for index in (agent.s4.domain_index, agent.s4.relation_type_index):
        assert all(schema_id in agent.s4.schemas for items in index.values() for schema_id in items)
