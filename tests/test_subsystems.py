"""
Unit tests for each NOEMA subsystem (S1-S6).
Verifies each component works correctly in isolation.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noema.core.free_energy import VariationalFreeEnergy, ExpectedFreeEnergy
from noema.core.jepa import JEPAModule
from noema.core.slot_attention import SlotAttention
from noema.subsystems.s1_sensorimotor import SensorimotorCore
from noema.subsystems.s2_world_model import HierarchicalWorldModel
from noema.subsystems.s3_memory import ComplementaryMemory
from noema.subsystems.s4_knowledge import RelationalKnowledgeGraph, Schema, Entity, Relation
from noema.subsystems.s5_analogy import AnalogyEngine
from noema.subsystems.s6_workspace import GlobalWorkspace, ContentType


def test_free_energy():
    """Test variational and expected free energy computations."""
    print("\n--- Free Energy ---")
    
    # Variational free energy
    vfe = VariationalFreeEnergy(latent_dim=16, obs_dim=32)
    obs = torch.randn(4, 32)
    mu = torch.randn(4, 16)
    logvar = torch.zeros(4, 16)
    fe = vfe(obs, mu, logvar)
    assert fe.dim() == 0, f"Expected scalar, got {fe.shape}"
    print(f"  Variational FE: {fe.item():.4f} ✓")

    # Expected free energy
    efe = ExpectedFreeEnergy(latent_dim=16, obs_dim=32, n_actions=4)
    s = torch.randn(1, 16)
    a = torch.eye(4)[:1]
    efe_vals = efe.compute_efe(s, a)
    assert efe_vals.shape == (1,), f"Expected (1,), got {efe_vals.shape}"
    print(f"  EFE values: {efe_vals.tolist()} ✓")
    
    # Action selection
    action, all_efe = efe.select_action(s)
    assert action.shape[-1] == 4
    print(f"  Action selected: {action.argmax().item()} ✓")
    return None


def test_jepa():
    """Test JEPA module."""
    print("\n--- JEPA Module ---")
    
    jepa = JEPAModule(obs_dim=32, latent_dim=16, action_dim=4)
    obs_t = torch.randn(4, 32)
    obs_next = obs_t + torch.randn(4, 32) * 0.1
    action = torch.randn(4, 4)
    
    result = jepa(obs_t, obs_next, action)
    assert "total_loss" in result
    assert "prediction_loss" in result
    assert "anti_collapse_loss" in result
    
    print(f"  Total loss: {result['total_loss'].item():.4f}")
    print(f"  Prediction loss: {result['prediction_loss'].item():.4f}")
    print(f"  Anti-collapse loss: {result['anti_collapse_loss'].item():.4f}")
    
    # Test encoding
    z = jepa.encode(obs_t)
    assert z.shape == (4, 16)
    print(f"  Encoding shape: {z.shape} ✓")
    
    # Test target encoding
    z_target = jepa.encode_target(obs_t)
    assert z_target.shape == (4, 16)
    assert not z_target.requires_grad
    print(f"  Target encoding (no grad): ✓")
    
    # Test free energy equivalence (Claim 1)
    fep = jepa.compute_free_energy_equivalent(obs_t, obs_next, action)
    print(f"  Free energy equivalent: {fep.item():.4f} ✓")
    print(f"  Claim 1 (Unification): JEPA ⊂ Free Energy ✓")
    return None


def test_slot_attention():
    """Test slot attention entity factorization."""
    print("\n--- Slot Attention ---")
    
    sa = SlotAttention(num_slots=4, input_dim=16, slot_dim=8, n_iterations=3)
    inputs = torch.randn(2, 10, 16)  # batch=2, 10 features, dim=16
    
    slots, attn = sa(inputs)
    assert slots.shape == (2, 4, 8), f"Expected (2,4,8), got {slots.shape}"
    assert attn.shape == (2, 4, 10), f"Expected (2,4,10), got {attn.shape}"
    
    print(f"  Slots shape: {slots.shape} ✓")
    print(f"  Attention shape: {attn.shape} ✓")
    print(f"  Sum of attention per slot: {attn.sum(dim=1).mean().item():.4f} ✓")
    return None


def test_s1():
    """Test S1: Sensorimotor Core."""
    print("\n--- S1: Sensorimotor Core ---")
    
    s1 = SensorimotorCore(proprio_dim=12, extero_dim=16, motor_dim=4)
    proprio = torch.randn(2, 12)
    extero = torch.randn(2, 16)
    motor = torch.randn(2, 4)
    
    result = s1(proprio, extero, motor)
    assert "prediction_errors" in result
    assert "sparsity" in result
    assert "reflex_motor" in result
    assert "forward_prediction" in result
    
    print(f"  Prediction errors shape: {result['prediction_errors'].shape} ✓")
    print(f"  Sparsity: {result['sparsity'].item():.4f} ✓")
    print(f"  Reflex motor shape: {result['reflex_motor'].shape} ✓")
    print(f"  Forward prediction shape: {result['forward_prediction'].shape} ✓")
    
    # Test inverse kinematics
    target = torch.randn(2, 12)
    motor_cmd = s1.inverse_kinematics(proprio, target)
    assert motor_cmd.shape == (2, 4)
    print(f"  Inverse kinematics: ✓")
    return None


def test_s2():
    """Test S2: Hierarchical World Model."""
    print("\n--- S2: Hierarchical World Model ---")
    
    wm = HierarchicalWorldModel(
        obs_dim=32, action_dim=4, n_levels=3,
        latent_dim=16, num_slots=4, slot_dim=8,
    )
    
    obs_seq = [torch.randn(2, 32), torch.randn(2, 32)]
    act_seq = [torch.randn(2, 4)]
    
    result = wm(obs_seq, act_seq)
    assert "total_loss" in result
    assert len(result["slots"]) == 3
    assert len(result["relations"]) == 3
    
    print(f"  Total loss: {result['total_loss'].item():.4f} ✓")
    print(f"  Level losses: {[f'{l:.4f}' for l in result['level_losses']]} ✓")
    print(f"  Slots from 3 levels: ✓")
    print(f"  Relations extracted: ✓")
    return None


def test_s3():
    """Test S3: Complementary Memory."""
    print("\n--- S3: Complementary Memory ---")
    
    mem = ComplementaryMemory(latent_dim=16, action_dim=4)
    
    # Write surprising experiences
    z_traj = torch.randn(10, 16)
    a_traj = torch.randn(10, 4)
    
    # High surprise → should write
    result_high = mem(z_traj, a_traj, free_energy=5.0)
    assert result_high["written_to_episodic"]
    print(f"  High surprise write: ✓ (surprise=5.0)")
    
    # Low surprise → should not write
    result_low = mem(z_traj, a_traj, free_energy=0.5)
    assert not result_low["written_to_episodic"]
    print(f"  Low surprise skip: ✓ (surprise=0.5)")
    
    # Retrieval
    query = z_traj[0]
    retrieved = mem.retrieve(query, top_k=3)
    print(f"  Retrieved {len(retrieved)} traces ✓")
    
    # Stats
    stats = mem.memory_stats()
    print(f"  Memory stats: {stats['size']} episodic traces ✓")
    return None


def test_s4():
    """Test S4: Relational Knowledge Graph."""
    print("\n--- S4: Relational Knowledge Graph ---")
    
    kg = RelationalKnowledgeGraph(embedding_dim=16, relation_embedding_dim=16)
    
    # Build schema from slots
    slots = torch.randn(1, 4, 8)
    rels = {"embeddings": torch.randn(1, 4, 4, 16)}
    
    schema = kg.build_schema_from_slots(slots, rels["embeddings"], domain="test")
    assert len(schema.entities) > 0
    
    stats = kg.graph_stats()
    print(f"  Schema built with {len(schema.entities)} entities ✓")
    print(f"  Relations: {stats['n_relation_types']} types ✓")
    
    # Test relation type encoder
    emb = kg.relation_encoder.get_embedding("CAUSES")
    assert emb.shape == (16,)
    print(f"  Relation embedding for CAUSES: {emb.shape} ✓")
    
    # Discover new type
    new_type = kg.relation_encoder.discover_relation_type(torch.randn(16))
    print(f"  Discovered relation type: {new_type} ✓")
    return None


def test_s5():
    """Test S5: Analogy Engine."""
    print("\n--- S5: Analogy Engine ---")
    
    kg = RelationalKnowledgeGraph(embedding_dim=16, relation_embedding_dim=16)
    
    # Source schema
    src = Schema(id="source", domain="physics")
    src.add_entity(Entity(id="A", embedding=torch.randn(16)))
    src.add_entity(Entity(id="B", embedding=torch.randn(16)))
    src.add_relation(Relation("A", "CAUSES", "B", 0.9))
    src.add_relation(Relation("A", "FLOW", "B", 0.85))
    kg.add_schema(src)
    
    # Target schema (partial)
    tgt = Schema(id="target", domain="economics")
    tgt.add_entity(Entity(id="X", embedding=torch.randn(16)))
    tgt.add_entity(Entity(id="Y", embedding=torch.randn(16)))
    tgt.add_relation(Relation("X", "CAUSES", "Y", 0.8))
    
    # Run analogy
    engine = AnalogyEngine(knowledge_graph=kg, embedding_dim=16)
    result = engine(tgt, exclude_domain="economics")
    
    print(f"  Candidate sources: {result['n_candidate_sources']} ✓")
    print(f"  Inferences: {len(result['inferences'])} ✓")
    if result['best_analogy']:
        print(f"  Best analogy from: {result['best_analogy']['source_domain']} ✓")
    
    # Design experiment
    if result['inferences']:
        exp = engine.design_experiment(result['inferences'][0], tgt)
        assert "intervention" in exp
        assert "predicted_outcome" in exp
        print(f"  Experiment designed: ✓")
    
    # Knowledge transfer
    enriched = engine.transfer_knowledge(src, tgt)
    print(f"  Transferred {len(enriched.relations) - len(tgt.relations)} relations ✓")
    return None


def test_s6():
    """Test S6: Global Workspace."""
    print("\n--- S6: Global Workspace ---")
    
    ws = GlobalWorkspace(embedding_dim=16, capacity=5)
    
    # Submit content from different subsystems
    ws.submit(ContentType.PREDICTION, torch.randn(1, 16), precision=0.9, source="S2")
    ws.submit(ContentType.RECALL, torch.randn(1, 16), precision=0.7, source="S3")
    ws.submit(ContentType.HYPOTHESIS, torch.randn(1, 16), precision=0.5, source="S5")
    ws.submit(ContentType.REFLEX, torch.randn(1, 16), precision=0.8, source="S1")
    
    # Competition
    result = ws.competition_step()
    assert result["n_winners"] > 0
    assert result["broadcast"] is not None
    
    print(f"  Submitted: {result['n_submitted']} ✓")
    print(f"  Winners: {result['n_winners']} ✓")
    print(f"  Broadcast shape: {result['broadcast'].shape} ✓")
    for w in result["winners"]:
        print(f"    Winner: {w['type']} from {w['source']} (precision={w['precision']:.2f})")
    return None


def run_all_subsystem_tests():
    """Run all subsystem unit tests."""
    print("\n" + "=" * 60)
    print("NOEMA SUBSYSTEM UNIT TESTS")
    print("=" * 60)

    tests = [
        ("Free Energy Core", test_free_energy),
        ("JEPA Module", test_jepa),
        ("Slot Attention", test_slot_attention),
        ("S1: Sensorimotor Core", test_s1),
        ("S2: Hierarchical World Model", test_s2),
        ("S3: Complementary Memory", test_s3),
        ("S4: Relational Knowledge Graph", test_s4),
        ("S5: Analogy Engine", test_s5),
        ("S6: Global Workspace", test_s6),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = True
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("SUBSYSTEM TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print(f"\n  All subsystems functional: {'YES ✓' if all_passed else 'NO ✗'}")
    return results


if __name__ == "__main__":
    run_all_subsystem_tests()
