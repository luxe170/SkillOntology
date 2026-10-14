from __future__ import annotations
from collections import defaultdict
from typing import Any, Iterable
from skillontology.schema import MetaOp, MetaOpEdge, clean_list, overlap_score
EXECUTABLE_EDGE_TYPES = {'dependency'}

def _next_edge_id(index: int) -> str:
    return f'E{index:06d}'

def _dedupe_edges(edges: Iterable[MetaOpEdge]) -> list[MetaOpEdge]:
    by_key: dict[tuple[str, str, str], MetaOpEdge] = {}
    for edge in edges:
        if edge.source_metaop_id == edge.target_metaop_id:
            continue
        if edge.edge_type not in EXECUTABLE_EDGE_TYPES:
            continue
        key = (edge.source_metaop_id, edge.target_metaop_id, edge.edge_type)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = edge
            continue
        existing.score = max(float(existing.score), float(edge.score))
        existing.support_count += max(1, int(edge.support_count))
        existing.source_skill_id = ', '.join(clean_list([existing.source_skill_id, edge.source_skill_id]))
        existing.evidence = ' | '.join(clean_list([existing.evidence, edge.evidence])[:5])
        existing.reason = existing.reason or edge.reason
    deduped = list(by_key.values())
    for index, edge in enumerate(deduped, start=1):
        edge.edge_id = _next_edge_id(index)
    return deduped

def build_workflow_edges(skill_metaop_sequences: dict[str, list[str]]) -> list[MetaOpEdge]:
    return []

def _sequence_support(skill_metaop_sequences: dict[str, list[str]]) -> dict[tuple[str, str], tuple[int, list[str]]]:
    support: dict[tuple[str, str], tuple[int, list[str]]] = {}
    for skill_id, sequence in sorted(skill_metaop_sequences.items()):
        canonical_sequence = clean_list(sequence)
        for source_index, source in enumerate(canonical_sequence):
            for target in canonical_sequence[source_index + 1:]:
                if source == target:
                    continue
                count, skills = support.get((source, target), (0, []))
                support[source, target] = (count + 1, clean_list([*skills, skill_id]))
    return support

def build_legacy_workflow_edges(skill_metaop_sequences: dict[str, list[str]]) -> list[MetaOpEdge]:
    edges: list[MetaOpEdge] = []
    for skill_id, sequence in sorted(skill_metaop_sequences.items()):
        canonical_sequence = clean_list(sequence)
        for source, target in zip(canonical_sequence, canonical_sequence[1:]):
            if source == target:
                continue
            edges.append(MetaOpEdge(edge_id='', source_metaop_id=source, target_metaop_id=target, edge_type='workflow', score=1.0, reason='same skill local step order', evidence=f'{source} precedes {target} in {skill_id}', source_skill_id=skill_id))
    return edges

def _precondition_score(source: MetaOp, target: MetaOp) -> tuple[float, str]:
    source_effects = source.effects or [*source.state_transitions, *source.postconditions, *source.outputs]
    post_pre = overlap_score(source_effects, target.preconditions)
    output_input = overlap_score(source.outputs, target.inputs)
    object_overlap = overlap_score(source.object, target.object)
    score = 0.75 * max(post_pre, output_input) + (0.25 * object_overlap if max(post_pre, output_input) else 0.0)
    evidence = []
    if post_pre:
        evidence.append(f'effects/preconditions overlap={post_pre:.2f}')
    if output_input:
        evidence.append(f'outputs/inputs overlap={output_input:.2f}')
    if object_overlap:
        evidence.append(f'object overlap={object_overlap:.2f}')
    return (score, '; '.join(evidence))

def build_field_edges(metaops: dict[str, MetaOp], *, threshold: float=0.2, max_outgoing_per_type: int=8, embedding_similarities: dict[tuple[str, str], float] | None=None, skill_metaop_sequences: dict[str, list[str]] | None=None) -> list[MetaOpEdge]:
    _ = embedding_similarities
    support = _sequence_support(skill_metaop_sequences or {})
    by_type: dict[tuple[str, str], list[MetaOpEdge]] = defaultdict(list)
    items = sorted(metaops.items())
    for source_id, source in items:
        for target_id, target in items:
            if source_id == target_id:
                continue
            precondition_score, precondition_evidence = _precondition_score(source, target)
            if precondition_score >= threshold:
                support_count, support_skills = support.get((source_id, target_id), (0, []))
                by_type[source_id, 'precondition'].append(MetaOpEdge(edge_id='', source_metaop_id=source_id, target_metaop_id=target_id, edge_type='dependency', score=precondition_score, reason='source effects satisfy target preconditions', evidence=precondition_evidence, source_skill_id=', '.join(support_skills), support_count=max(1, support_count)))
    edges: list[MetaOpEdge] = []
    for candidates in by_type.values():
        candidates.sort(key=lambda edge: edge.score, reverse=True)
        edges.extend(candidates[:max_outgoing_per_type])
    return edges

def build_policy_workflow_edges(metaops: dict[str, MetaOp], extraction_policy: Any | None, skill_metaop_sequences: dict[str, list[str]] | None=None) -> list[MetaOpEdge]:
    if extraction_policy is None:
        return []
    by_action: dict[str, list[str]] = defaultdict(list)
    for metaop_id, metaop in sorted(metaops.items()):
        if 'ontology_domain_scoped' not in set(metaop.quality_flags):
            continue
        by_action[metaop.action].append(metaop_id)
    sequence_support = _sequence_support(skill_metaop_sequences or {})
    edges: list[MetaOpEdge] = []
    workflows = getattr(extraction_policy, 'workflow_templates', {}) or {}
    for workflow_name, raw_actions in sorted(workflows.items()):
        actions = clean_list(raw_actions)
        rule_id = extraction_policy.rule_id('workflow_template', workflow_name)
        for source_action, target_action in zip(actions, actions[1:]):
            if source_action == target_action:
                continue
            for source_id in by_action.get(source_action, []):
                for target_id in by_action.get(target_action, []):
                    if source_id == target_id:
                        continue
                    support_count, support_skills = sequence_support.get((source_id, target_id), (0, []))
                    object_score = overlap_score(metaops[source_id].object, metaops[target_id].object)
                    if support_count == 0 and object_score < 0.2:
                        continue
                    edges.append(MetaOpEdge(edge_id='', source_metaop_id=source_id, target_metaop_id=target_id, edge_type='dependency', score=min(0.99, 0.8 + (0.1 if support_count else 0.0) + 0.09 * object_score), reason=f'workflow constraint {rule_id or workflow_name}', evidence=f'{source_action} precedes {target_action} in {workflow_name}; object overlap={object_score:.2f}; local sequence support={support_count}', source_skill_id=', '.join(support_skills) or 'domain_ontology', support_count=max(1, support_count)))
    return edges

def _acyclic_edges(edges: list[MetaOpEdge]) -> list[MetaOpEdge]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    kept: list[MetaOpEdge] = []

    def reaches(start: str, target: str) -> bool:
        stack = [start]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(sorted(adjacency.get(node, ())))
        return False
    for edge in sorted(edges, key=lambda item: (-float(item.score), item.source_metaop_id, item.target_metaop_id, item.reason)):
        if edge.source_metaop_id == edge.target_metaop_id:
            continue
        if reaches(edge.target_metaop_id, edge.source_metaop_id):
            continue
        adjacency[edge.source_metaop_id].add(edge.target_metaop_id)
        kept.append(edge)
    return kept

def build_metaop_edges(metaops: dict[str, MetaOp], skill_metaop_sequences: dict[str, list[str]], *, threshold: float=0.2, max_outgoing_per_type: int=8, embedding_similarities: dict[tuple[str, str], float] | None=None, extraction_policy: Any | None=None) -> list[MetaOpEdge]:
    edge_metaops = metaops
    if extraction_policy is not None:
        edge_metaops = {metaop_id: metaop for metaop_id, metaop in metaops.items() if 'ontology_domain_scoped' in set(metaop.quality_flags)}
    field_edges = build_field_edges(edge_metaops, threshold=threshold, max_outgoing_per_type=max_outgoing_per_type, embedding_similarities=embedding_similarities, skill_metaop_sequences=skill_metaop_sequences)
    policy_edges = build_policy_workflow_edges(metaops, extraction_policy, skill_metaop_sequences)
    combined = _dedupe_edges([*field_edges, *policy_edges])
    if extraction_policy is not None:
        combined = _acyclic_edges(combined)
        for index, edge in enumerate(combined, start=1):
            edge.edge_id = _next_edge_id(index)
    return combined

async def build_metaop_edges_with_embeddings(metaops: dict[str, MetaOp], skill_metaop_sequences: dict[str, list[str]], embedding_service: Any | None, *, threshold: float=0.2, max_outgoing_per_type: int=8, extraction_policy: Any | None=None) -> list[MetaOpEdge]:
    similarities: dict[tuple[str, str], float] = {}
    return build_metaop_edges(metaops, skill_metaop_sequences, threshold=threshold, max_outgoing_per_type=max_outgoing_per_type, embedding_similarities=similarities, extraction_policy=extraction_policy)
