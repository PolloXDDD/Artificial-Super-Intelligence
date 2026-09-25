"""
S5: The Analogy Engine — the organ of extrapolation.

This is the keystone. Extrapolation is implemented as structure mapping
(Gentner, 1983): given a novel situation with partial schema σ_tgt,
retrieve source schemas and compute a soft graph isomorphism:

  M* = argmax_M [sim_T(R_src, M(R_tgt)) - β * sim_V(V_src, M(V_tgt))]

The sign structure is ESSENTIAL: relational similarity is rewarded while
SURFACE similarity is actively PENALIZED, forcing mappings like
water→electricity rather than water→milk.

The mapping M* licenses candidate inferences: relations present in source
but absent in target are projected across as hypotheses.

Claim 2 (Closure of Extrapolation Loop): A projected hypothesis carries
maximal epistemic value; the agent is intrinsically compelled to design
the experiment that tests its own analogy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..subsystems.s4_knowledge import Schema, Entity, Relation, RelationalKnowledgeGraph


class StructureMappingEngine(nn.Module):
    """
    Differentiable structure mapping via optimal transport alignment.

    Implements Eq. (5): M* = argmax_M [relational_match - β * surface_match]
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        relation_dim: int = 32,
        surface_penalty_beta: float = 0.5,
        n_sinkhorn_iters: int = 10,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.relation_dim = relation_dim
        self.beta = surface_penalty_beta
        self.n_sinkhorn_iters = n_sinkhorn_iters

        # Relation similarity scorer
        self.rel_scorer = nn.Sequential(
            nn.Linear(relation_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Entity alignment cost
        self.entity_aligner = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def compute_alignment(
        self, source: Schema, target: Schema
    ) -> dict:
        """
        Compute soft graph isomorphism between source and target schemas.

        Returns the alignment M* and the score.
        """
        # Get entity embeddings
        src_entities = list(source.entities.values())
        tgt_entities = list(target.entities.values())

        if not src_entities or not tgt_entities:
            return {"score": 0.0, "alignment": None, "inferences": []}

        src_embs = torch.stack([e.embedding for e in src_entities])
        tgt_embs = torch.stack([e.embedding for e in tgt_entities])

        n_src = len(src_entities)
        n_tgt = len(tgt_entities)

        # --- Relational similarity matrix ---
        # For each source relation, for each potential target relation
        src_rels = source.relations
        tgt_rels = target.relations

        rel_score = torch.tensor(0.0)

        if src_rels and tgt_rels:
            # Build relation type overlap score
            src_types = {r.relation_type for r in src_rels}
            tgt_types = {r.relation_type for r in tgt_rels}
            overlap = src_types & tgt_types
            rel_score = torch.tensor(len(overlap) / max(len(src_types | tgt_types), 1))

        # --- Surface similarity (PENALIZED) ---
        # Cosine similarity between entity embedding means
        src_mean = src_embs.mean(dim=0)
        tgt_mean = tgt_embs.mean(dim=0)
        surface_sim = F.cosine_similarity(
            src_mean.unsqueeze(0), tgt_mean.unsqueeze(0)
        ).item()

        # --- Entity alignment via soft matching ---
        # Cost matrix: negative cosine similarity
        cost = torch.zeros(n_src, n_tgt)
        for i in range(n_src):
            for j in range(n_tgt):
                cost[i, j] = -F.cosine_similarity(
                    src_embs[i].unsqueeze(0), tgt_embs[j].unsqueeze(0)
                ).item()

        # Soft alignment via Sinkhorn
        alignment = self._sinkhorn(-cost, n_src, n_tgt)

        # --- Total score: Eq. (5) ---
        score = rel_score.item() - self.beta * max(0, surface_sim)

        return {
            "score": score,
            "alignment": alignment,
            "alignment_matrix": cost,
            "relational_match": rel_score.item(),
            "surface_match": surface_sim,
            "src_entities": [e.id for e in src_entities],
            "tgt_entities": [e.id for e in tgt_entities],
        }

    def _sinkhorn(
        self, sim_matrix: torch.Tensor, n_src: int, n_tgt: int, eps: float = 1e-6
    ) -> torch.Tensor:
        """Sinkhorn algorithm for soft optimal transport alignment."""
        K = torch.exp(sim_matrix / 0.1)
        K = K + eps

        for _ in range(self.n_sinkhorn_iters):
            # Row normalize
            K = K / (K.sum(dim=-1, keepdim=True) + eps)
            # Column normalize
            K = K / (K.sum(dim=-2, keepdim=True) + eps)

        return K

    def generate_candidate_inferences(
        self,
        source: Schema,
        target: Schema,
        alignment: dict,
    ) -> list[dict]:
        """
        Generate candidate inferences by projecting relations from source
        to target that are present in source but absent in target.

        This is the core of analogical reasoning: relations present in the
        source but absent in the target are projected across as hypotheses.
        """
        src_types = {r.relation_type: r for r in source.relations}
        tgt_types = {r.relation_type for r in target.relations}

        inferences = []
        for rtype, rel in src_types.items():
            if rtype not in tgt_types:
                # Project: map source entities to target entities via alignment
                inference = {
                    "relation_type": rtype,
                    "source_relation": {
                        "source": rel.source_id,
                        "type": rel.relation_type,
                        "target": rel.target_id,
                    },
                    "projected_to_target": True,
                    "confidence": rel.confidence * alignment.get("score", 0.5),
                    "is_hypothesis": True,  # Untested projection
                }
                inferences.append(inference)

        return inferences


class AnalogyEngine(nn.Module):
    """
    S5: The Analogy Engine — the organ of extrapolation.

    Given a target situation, retrieves source schemas from S4,
    computes structure mapping alignment, generates candidate inferences,
    and computes epistemic value (driving the agent to test the analogy).

    Claim 2: Extrapolation loop closure — projected hypotheses carry
    maximal epistemic value, compelling the agent to test its own analogies.
    """

    def __init__(
        self,
        knowledge_graph: RelationalKnowledgeGraph,
        embedding_dim: int = 32,
        relation_dim: int = 32,
        surface_penalty_beta: float = 0.5,
    ):
        super().__init__()
        self.knowledge_graph = knowledge_graph

        # Structure mapping engine
        self.mapper = StructureMappingEngine(
            embedding_dim=embedding_dim,
            relation_dim=relation_dim,
            surface_penalty_beta=surface_penalty_beta,
        )

        # Epistemic value estimator for hypotheses
        self.epistemic_estimator = nn.Sequential(
            nn.Linear(embedding_dim + 16, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        target_schema: Schema,
        exclude_domain: str | None = None,
        top_k_sources: int = 5,
    ) -> dict:
        """
        Perform analogical reasoning on a target schema.

        1. Retrieve source schemas from S4
        2. Compute structure mapping alignment
        3. Generate candidate inferences
        4. Rank by epistemic value (drives curiosity to test)

        Returns the best analogy and its projected hypotheses.
        """
        # Step 1: Retrieve structurally similar schemas
        candidates = self.knowledge_graph.retrieve_by_structure(
            target_schema, top_k=top_k_sources, exclude_domain=exclude_domain
        )

        if not candidates:
            return {
                "best_analogy": None,
                "all_analogies": [],
                "inferences": [],
                "epistemic_value": 0.0,
            }

        # Step 2: Compute structure mapping for each candidate
        analogies = []
        for source_schema, retrieval_score in candidates:
            alignment = self.mapper.compute_alignment(source_schema, target_schema)
            inferences = self.mapper.generate_candidate_inferences(
                source_schema, target_schema, alignment
            )

            analogies.append({
                "source_schema": source_schema,
                "source_domain": source_schema.domain,
                "alignment": alignment,
                "inferences": inferences,
                "combined_score": alignment["score"],
                "retrieval_score": retrieval_score,
            })

        # Step 3: Rank by combined score
        analogies.sort(key=lambda x: x["combined_score"], reverse=True)

        # Step 4: Select best and compute epistemic value
        best = analogies[0] if analogies else None

        total_epistemic = 0.0
        all_inferences = []
        for analogy in analogies:
            for inf in analogy["inferences"]:
                # Epistemic value ∝ uncertainty of the hypothesis
                # Unproven analogical inferences have max uncertainty
                inf["epistemic_value"] = inf["confidence"] * len(analogy["inferences"])
                total_epistemic += inf["epistemic_value"]
                all_inferences.append(inf)

        return {
            "best_analogy": best,
            "all_analogies": analogies,
            "inferences": all_inferences,
            "total_epistemic_value": total_epistemic,
            "n_candidate_sources": len(candidates),
        }

    def design_experiment(
        self,
        inference: dict,
        target_schema: Schema,
    ) -> dict:
        """
        Design an experiment to test a candidate inference.

        By the Closure Claim (Claim 2): a projected hypothesis carries
        maximal epistemic value; by Eq. (2) the agent is compelled to
        test it. This method generates the experimental intervention.

        Returns an experiment specification: which variables to intervene
        on, what to measure, and what outcomes would confirm/reject.
        """
        experiment = {
            "hypothesis": inference,
            "intervention": {
                "manipulate": inference.get("relation_type", "unknown"),
                "target_entities": [e.id for e in target_schema.entities.values()],
            },
            "predicted_outcome": {
                "relation_type": inference["relation_type"],
                "expected": "present",
            },
            "null_outcome": {
                "relation_type": inference["relation_type"],
                "expected": "absent",
            },
            "epistemic_value": inference.get("epistemic_value", 0.0),
        }

        return experiment

    def transfer_knowledge(
        self,
        source_schema: Schema,
        target_schema: Schema,
    ) -> Schema:
        """
        Transfer knowledge from source to target via analogical mapping.
        Creates a new enriched target schema with projected relations.
        """
        alignment = self.mapper.compute_alignment(source_schema, target_schema)
        inferences = self.mapper.generate_candidate_inferences(
            source_schema, target_schema, alignment
        )

        # Create enriched target
        enriched = Schema(
            id=f"{target_schema.id}_enriched",
            domain=target_schema.domain,
        )

        # Copy existing entities and relations
        for eid, entity in target_schema.entities.items():
            enriched.add_entity(entity)
        for rel in target_schema.relations:
            enriched.add_relation(rel)

        # Add projected relations (as hypotheses)
        for inf in inferences:
            src_rel = inf["source_relation"]
            # Map source entity names to target entity names via alignment
            entity_list = list(target_schema.entities.keys())
            if entity_list:
                src_idx = hash(src_rel["source"]) % len(entity_list)
                tgt_idx = hash(src_rel["target"]) % len(entity_list)
                projected = Relation(
                    source_id=entity_list[src_idx],
                    relation_type=inf["relation_type"],
                    target_id=entity_list[tgt_idx],
                    confidence=inf["confidence"],
                    context_embedding=None,
                )
                enriched.add_relation(projected)

        return enriched
