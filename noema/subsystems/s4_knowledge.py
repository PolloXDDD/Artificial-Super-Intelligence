"""
S4: Relational Knowledge Graph — the "semantic system".

Consolidation stores experience not as vectors but as typed relational schemas:
  σ = ⟨V, R⟩,  R ⊆ V × T × V

where V are entity slots with grounded embedding anchors into S2,
and T is a learned, open vocabulary of relation types
(SUPPORTS, CONTAINS, CAUSES, PRECEDES, ...).

Because relations are stored DETACHED from their fillers, knowledge
becomes portable: the schema "pressure-differential causes flow",
learned from water, is carried intact to heat, to crowds, to electric
charge. Invariant I3 is satisfied here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class Entity:
    """An entity slot with grounded embedding anchor."""
    id: str
    embedding: torch.Tensor  # Grounded anchor into S2 embeddings
    properties: dict = field(default_factory=dict)
    domain: str = "generic"


@dataclass
class Relation:
    """A typed relation between entities."""
    source_id: str
    relation_type: str  # From learned vocabulary T
    target_id: str
    confidence: float = 1.0
    context_embedding: Optional[torch.Tensor] = None


@dataclass
class Schema:
    """
    A relational schema σ = ⟨V, R⟩.
    
    Relations are stored DETACHED from their fillers — this is the
    representational precondition of extrapolation (Invariant I3).
    """
    id: str
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    domain: str = "generic"

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation):
        self.relations.append(relation)

    def get_relation_types(self) -> set[str]:
        return {r.relation_type for r in self.relations}

    def get_structure_embedding(self, relation_encoder: nn.Module) -> torch.Tensor:
        """Encode the relational structure of this schema."""
        if not self.relations:
            return torch.zeros(relation_encoder[-1].out_features
                               if hasattr(relation_encoder, 'out_features')
                               else 32)

        embeddings = []
        for rel in self.relations:
            src = self.entities.get(rel.source_id)
            tgt = self.entities.get(rel.target_id)
            if src is not None and tgt is not None:
                # Encode: source embedding + relation type + target embedding
                pair = torch.cat([src.embedding, tgt.embedding], dim=-1)
                embeddings.append(pair)

        if embeddings:
            return torch.stack(embeddings).mean(dim=0)
        return torch.zeros(32)


class RelationTypeEncoder(nn.Module):
    """Learned, open vocabulary of relation types."""

    def __init__(self, embedding_dim: int = 32, max_relation_types: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Learnable embeddings for relation types
        # Start with known types, expandable
        initial_types = [
            "SUPPORTS", "CONTAINS", "CAUSES", "PRECEDES",
            "EXTENDS", "PUSHES", "ABOVE", "BELOW",
            "ATTACHES", "FLOW", "PRESSURE", "RESISTANCE",
            "SOURCE", "SINK", "GRADIENT", "TRANSFER",
            "CONTAINS_PART", "ENABLES", "PREVENTS", "SEQUENCE",
        ]

        self.type_to_idx = {t: i for i, t in enumerate(initial_types)}
        self.idx_to_type = {i: t for t, i in self.type_to_idx.items()}
        self.next_idx = len(initial_types)

        self.embeddings = nn.Parameter(
            torch.randn(max_relation_types, embedding_dim) * 0.1
        )

    def get_embedding(self, relation_type: str) -> torch.Tensor:
        """Get embedding for a relation type, creating new if needed."""
        if relation_type not in self.type_to_idx:
            if self.next_idx < self.embeddings.shape[0]:
                self.type_to_idx[relation_type] = self.next_idx
                self.idx_to_type[self.next_idx] = relation_type
                self.next_idx += 1
            else:
                # Fallback: hash to existing
                idx = hash(relation_type) % self.embeddings.shape[0]
                self.type_to_idx[relation_type] = idx
        return self.embeddings[self.type_to_idx[relation_type]]

    def discover_relation_type(self, embedding: torch.Tensor, threshold: float = 0.5) -> str:
        """
        Discover a relation type from a latent embedding.
        If close to existing type, return it; otherwise create new.
        """
        similarities = F.cosine_similarity(
            embedding.unsqueeze(0), self.embeddings[:self.next_idx], dim=-1
        )
        best_idx = similarities.argmax().item()
        best_sim = similarities[best_idx].item()

        if best_sim > threshold:
            return self.idx_to_type.get(best_idx, f"REL_{best_idx}")
        else:
            new_type = f"DISCOVERED_{self.next_idx}"
            self.get_embedding(new_type)
            return new_type


class RelationalKnowledgeGraph(nn.Module):
    """
    S4: Relational Knowledge Graph.

    Stores knowledge as typed relational schemas with DETACHED relations.
    This detachment is the precondition for analogical transfer (S5).
    Invariant I3 is satisfied here.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        relation_embedding_dim: int = 32,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Relation type encoder (learned, open vocabulary)
        self.relation_encoder = RelationTypeEncoder(
            embedding_dim=relation_embedding_dim
        )

        # Schema storage
        self.schemas: dict[str, Schema] = {}
        self.schema_counter = 0

        # Domain index for retrieval
        self.domain_index: dict[str, list[str]] = defaultdict(list)

        # Relation type index
        self.relation_type_index: dict[str, list[str]] = defaultdict(list)

    def add_schema(self, schema: Schema) -> str:
        """Add a new schema to the knowledge graph."""
        if schema.id is None or schema.id == "":
            schema.id = f"schema_{self.schema_counter}"
            self.schema_counter += 1

        if schema.id in self.schemas:
            self.remove_schema(schema.id)
        self.schemas[schema.id] = schema
        self.domain_index[schema.domain].append(schema.id)

        # Index by relation types
        for rel in schema.relations:
            self.relation_type_index[rel.relation_type].append(schema.id)

        return schema.id

    def remove_schema(self, schema_id: str):
        """Remove a schema and its retrieval-index entries together."""
        old = self.schemas.pop(schema_id, None)
        if old is None:
            return
        for index in (self.domain_index, self.relation_type_index):
            for key in list(index):
                index[key] = [item for item in index[key] if item != schema_id]
                if not index[key]:
                    del index[key]

    def build_schema_from_slots(
        self,
        slots: torch.Tensor,
        relation_embeddings: torch.Tensor,
        domain: str = "generic",
        entity_names: Optional[list[str]] = None,
        relation_threshold: float = 0.3,
    ) -> Schema:
        """
        Build a schema from slot attention output and relation embeddings.

        Args:
            slots: entity slot representations (batch=1, n_slots, slot_dim)
            relation_embeddings: pairwise relation embeddings (batch=1, n_slots, n_slots, rel_dim)
            domain: domain label for this schema
            entity_names: optional names for entities
            relation_threshold: threshold for including relations

        Returns:
            Schema with entities and discovered relations
        """
        if slots.dim() == 3:
            slots = slots[0]  # Remove batch dim
        if relation_embeddings.dim() == 4:
            relation_embeddings = relation_embeddings[0]

        n_slots = slots.shape[0]
        schema = Schema(id=f"schema_{self.schema_counter}", domain=domain)
        self.schema_counter += 1

        # Add entities
        for i in range(n_slots):
            name = entity_names[i] if entity_names else f"entity_{i}"
            entity = Entity(
                id=name,
                embedding=slots[i].detach(),
                domain=domain,
            )
            schema.add_entity(entity)

        # Discover and add relations
        for i in range(n_slots):
            for j in range(n_slots):
                if i == j:
                    continue
                rel_emb = relation_embeddings[i, j]
                rel_type = self.relation_encoder.discover_relation_type(rel_emb)

                # Only add if embedding has sufficient magnitude
                if rel_emb.norm().item() > relation_threshold:
                    src_name = entity_names[i] if entity_names else f"entity_{i}"
                    tgt_name = entity_names[j] if entity_names else f"entity_{j}"
                    relation = Relation(
                        source_id=src_name,
                        relation_type=rel_type,
                        target_id=tgt_name,
                        confidence=rel_emb.norm().item(),
                        context_embedding=rel_emb.detach(),
                    )
                    schema.add_relation(relation)

        self.add_schema(schema)
        return schema

    def retrieve_by_structure(
        self, query_schema: Schema, top_k: int = 5, exclude_domain: str | None = None
    ) -> list[tuple[Schema, float]]:
        """
        Retrieve schemas with similar relational structure.
        This is the retrieval step for S5 (Analogy Engine).

        Relational match is prioritized over surface match.
        """
        if not self.schemas:
            return []

        query_types = query_schema.get_relation_types()
        if not query_types:
            return []

        scores = []
        for schema_id, schema in self.schemas.items():
            if exclude_domain and schema.domain == exclude_domain:
                continue
            if schema.id == query_schema.id:
                continue

            schema_types = schema.get_relation_types()

            # Relational overlap (Jaccard similarity on relation types)
            overlap = len(query_types & schema_types)
            union = len(query_types | schema_types)
            relational_sim = overlap / max(union, 1)

            # Surface similarity (penalized in S5)
            surface_sim = self._surface_similarity(query_schema, schema)

            # Score: relational reward, surface penalty (Eq. 5 in paper)
            score = relational_sim - 0.1 * surface_sim

            if relational_sim > 0:
                scores.append((schema, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _surface_similarity(self, s1: Schema, s2: Schema) -> float:
        """Surface similarity between schemas (based on entity embeddings)."""
        if not s1.entities or not s2.entities:
            return 0.0

        e1 = torch.stack([e.embedding for e in s1.entities.values()])
        e2 = torch.stack([e.embedding for e in s2.entities.values()])

        # Average pairwise cosine similarity
        sim = F.cosine_similarity(
            e1.mean(0).unsqueeze(0), e2.mean(0).unsqueeze(0)
        ).item()
        return max(0, sim)

    def get_schemas_by_domain(self, domain: str) -> list[Schema]:
        """Retrieve all schemas in a domain."""
        return [self.schemas[sid] for sid in self.domain_index.get(domain, [])
                if sid in self.schemas]

    def get_all_relation_types(self) -> set[str]:
        """Get all discovered relation types."""
        types = set()
        for schema in self.schemas.values():
            types |= schema.get_relation_types()
        return types

    def graph_stats(self) -> dict:
        """Return statistics about the knowledge graph."""
        all_types = self.get_all_relation_types()
        return {
            "n_schemas": len(self.schemas),
            "n_domains": len(self.domain_index),
            "n_relation_types": len(all_types),
            "relation_types": list(all_types),
            "total_relations": sum(len(s.relations) for s in self.schemas.values()),
        }
