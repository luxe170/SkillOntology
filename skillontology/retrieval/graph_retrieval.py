from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from skillontology.retrieval.index import MetaOpVectorIndex
try:
    from skillontology.retrieval.openai_embedding import build_embedding_service
except ImportError:

    def build_embedding_service() -> Any:
        raise ImportError('MetaOp graph retrieval requires an embedding service.')
from skillontology.runtime.query import parse_query as parse_generic_query
from skillontology.rendering import render_metaop_context
from skillontology.schema import ExecutableMetaOp, MetaOp, MetaOpDAG, MetaOpEdge, QueryDecomposition, QueryOperationNeed, clean_list, overlap_score
from skillontology.construction.step6_ontology_memory import OntologyMemory
DEFAULT_SEED_TOP_K = 5
DEFAULT_TOP_N = 8
DEFAULT_PPR_DAMPING = 0.2
DEFAULT_DEPENDENCY_REVERSE_WEIGHT = 1.0
DEFAULT_WORKFLOW_REVERSE_WEIGHT = 0.5
DAG_EDGE_TYPES = {'dependency'}
DEPENDENCY_EDGE_TYPES = {'dependency', 'dataflow', 'precondition', 'tool_setup'}

def query_to_retrieval_text(query: QueryDecomposition) -> str:
    return '\n'.join((part for part in [f'Goal: {query.normalized_goal or query.raw_query}', f'Task type: {query.task_type}' if query.task_type else '', f"Objects: {', '.join(query.domain_objects)}" if query.domain_objects else '', 'Operations: ' + ' | '.join((item.to_text() for item in query.required_operations)) if query.required_operations else '', f"Constraints: {', '.join(query.constraints)}" if query.constraints else ''] if part))

def build_rank_distribution(seed_hits: list[tuple[MetaOp, float, int]]) -> dict[str, float]:
    if not seed_hits:
        return {}
    scores = [max(0.0, float(score)) for _metaop, score, _rank in seed_hits]
    if max(scores, default=0.0) > 0.0:
        total = sum(scores) or 1.0
        return {metaop.metaop_id: max(0.0, float(score)) / total for metaop, score, _rank in seed_hits}
    reciprocal = {metaop.metaop_id: 1.0 / max(1, int(rank)) for metaop, _score, rank in seed_hits}
    total = sum(reciprocal.values()) or 1.0
    return {metaop_id: weight / total for metaop_id, weight in reciprocal.items()}

def build_personalization(metaop_ids: list[str], seed_weights: dict[str, float]) -> np.ndarray:
    vector = np.zeros(len(metaop_ids), dtype=np.float64)
    positions = {metaop_id: index for index, metaop_id in enumerate(metaop_ids)}
    for metaop_id, weight in seed_weights.items():
        if metaop_id in positions:
            vector[positions[metaop_id]] = max(0.0, float(weight))
    total = float(vector.sum())
    if total <= 0.0 and len(vector):
        vector[:] = 1.0 / len(vector)
    elif total > 0.0:
        vector /= total
    return vector

def _reverse_weight(edge_type: str, dependency_reverse_weight: float, workflow_reverse_weight: float) -> float:
    edge_type = edge_type.lower()
    if edge_type in DEPENDENCY_EDGE_TYPES:
        return dependency_reverse_weight
    if edge_type == 'workflow':
        return workflow_reverse_weight
    return 0.0

def build_transition_matrix(metaop_ids: list[str], edges: list[MetaOpEdge], *, dependency_reverse_weight: float=DEFAULT_DEPENDENCY_REVERSE_WEIGHT, workflow_reverse_weight: float=DEFAULT_WORKFLOW_REVERSE_WEIGHT) -> np.ndarray:
    n = len(metaop_ids)
    matrix = np.zeros((n, n), dtype=np.float64)
    positions = {metaop_id: index for index, metaop_id in enumerate(metaop_ids)}
    for edge in edges:
        source = positions.get(edge.source_metaop_id)
        target = positions.get(edge.target_metaop_id)
        if source is None or target is None or source == target:
            continue
        weight = max(float(edge.score), 0.01)
        matrix[source, target] += weight
        reverse = _reverse_weight(edge.edge_type, dependency_reverse_weight, workflow_reverse_weight)
        if reverse > 0.0:
            matrix[target, source] += weight * reverse
    for index in range(n):
        row_sum = float(matrix[index].sum())
        if row_sum > 0.0:
            matrix[index] /= row_sum
        else:
            matrix[index, index] = 1.0
    return matrix

def personalized_pagerank(transition_matrix: np.ndarray, personalization: np.ndarray, *, damping: float=DEFAULT_PPR_DAMPING, max_iter: int=100, tolerance: float=1e-09) -> np.ndarray:
    if len(personalization) == 0:
        return personalization
    restart = max(0.0, min(1.0, float(damping)))
    rank = personalization.astype(np.float64, copy=True)
    for _ in range(max_iter):
        next_rank = restart * personalization + (1.0 - restart) * (transition_matrix.T @ rank)
        if np.abs(next_rank - rank).sum() < tolerance:
            return next_rank
        rank = next_rank
    return rank

class MetaOpGraphRetriever:

    def __init__(self, meta_dir: str | Path, *, embedding_service: Any | None=None, **_ignored: Any) -> None:
        self.meta_dir = Path(meta_dir)
        self.memory = OntologyMemory.load(self.meta_dir)
        self.metaops = dict(self.memory.metaops)
        self.edges = list(self.memory.metaop_edges)
        self.index = MetaOpVectorIndex.load(self.meta_dir)
        self.embedding_service = embedding_service or build_embedding_service()

    async def retrieve(self, query: QueryDecomposition, *, seed_top_k: int=DEFAULT_SEED_TOP_K, top_n: int=DEFAULT_TOP_N, ppr_damping: float=DEFAULT_PPR_DAMPING, dependency_reverse_weight: float=DEFAULT_DEPENDENCY_REVERSE_WEIGHT, workflow_reverse_weight: float=DEFAULT_WORKFLOW_REVERSE_WEIGHT, max_chars: int | None=6000) -> MetaOpDAG:
        _ = (ppr_damping, dependency_reverse_weight, workflow_reverse_weight)
        max_nodes = max(1, int(top_n or DEFAULT_TOP_N))
        selected_ids, selected_scores, missing_operations = await self._select_by_required_effect_coverage(query, seed_top_k=max(1, int(seed_top_k or DEFAULT_SEED_TOP_K)), max_nodes=max_nodes)
        selected_ids, selected_scores = self._add_dependency_closure(selected_ids, selected_scores, max_nodes=max_nodes)
        selected_edges = self._execution_edges_for_selected(selected_ids)
        acyclic_edges = self._drop_cycle_edges(selected_ids, selected_edges)
        execution_order = self._stable_topological_order(selected_ids, acyclic_edges, selected_scores)
        selected_metaops = self._executable_metaops(execution_order, acyclic_edges)
        dag = MetaOpDAG(query=query.raw_query, query_decomposition=query, selected_metaops=selected_metaops, edges=acyclic_edges, execution_order=execution_order, missing_operations=missing_operations, rendered_context='')
        dag.rendered_context = render_metaop_context(dag, max_chars=max_chars)
        return dag

    async def async_retrieve_dag(self, raw_query: str, *, max_metaops: int=DEFAULT_TOP_N, seed_top_k: int=DEFAULT_SEED_TOP_K, ppr_damping: float=DEFAULT_PPR_DAMPING, **_ignored: Any) -> MetaOpDAG:
        query = parse_generic_query(raw_query)
        return await self.retrieve(query, seed_top_k=seed_top_k, top_n=max_metaops, ppr_damping=ppr_damping)

    @staticmethod
    def _top_ids(metaop_ids: list[str], ppr: np.ndarray, *, top_n: int) -> list[str]:
        order = sorted(range(len(metaop_ids)), key=lambda index: (-float(ppr[index]), metaop_ids[index]))
        return [metaop_ids[index] for index in order[:min(max(1, int(top_n)), len(order))]]

    async def _select_by_required_effect_coverage(self, query: QueryDecomposition, *, seed_top_k: int, max_nodes: int) -> tuple[list[str], dict[str, float], list[str]]:
        needs = sorted(query.required_operations, key=lambda item: item.priority)
        if not needs:
            hits = await self.index.search(query_to_retrieval_text(query), self.embedding_service, top_k=max_nodes)
            selected = [metaop.metaop_id for metaop, _score, _rank in hits[:max_nodes]]
            scores = {metaop.metaop_id: float(score) for metaop, score, _rank in hits}
            return (selected, scores, [])
        selected: list[str] = []
        selected_scores: dict[str, float] = {}
        missing: list[str] = []
        for need in needs:
            if len(selected) >= max_nodes:
                missing.append(f'not selected due to plan budget: {need.to_text()}')
                continue
            best = await self._best_metaop_for_need(query, need, seed_top_k=seed_top_k)
            if best is None:
                missing.append(need.to_text())
                continue
            metaop, score = best
            if metaop.metaop_id not in selected:
                selected.append(metaop.metaop_id)
            selected_scores[metaop.metaop_id] = max(selected_scores.get(metaop.metaop_id, 0.0), score)
        return (selected, selected_scores, missing)

    async def _best_metaop_for_need(self, query: QueryDecomposition, need: QueryOperationNeed, *, seed_top_k: int, min_score: float=0.16, min_lexical_score: float=0.08) -> tuple[MetaOp, float] | None:
        seed_hits = await self.index.search(self._need_query_text(query, need), self.embedding_service, top_k=seed_top_k)
        dense_scores = {metaop.metaop_id: max(0.0, float(score)) for metaop, score, _rank in seed_hits}
        ranks = {metaop.metaop_id: rank for metaop, _score, rank in seed_hits}
        candidates: list[tuple[MetaOp, float]] = []
        for metaop in self.index.metaops:
            lexical_score = self._lexical_score_metaop_for_need(metaop, need)
            if lexical_score < min_lexical_score:
                continue
            score = self._score_metaop_for_need(metaop, need, lexical_score=lexical_score, dense_score=dense_scores.get(metaop.metaop_id, 0.0), rank=ranks.get(metaop.metaop_id))
            if score >= min_score:
                candidates.append((metaop, score))
        candidates.sort(key=lambda row: (-row[1], row[0].metaop_id))
        return candidates[0] if candidates else None

    @staticmethod
    def _need_query_text(query: QueryDecomposition, need: QueryOperationNeed) -> str:
        return '\n'.join((part for part in [f'Goal: {query.normalized_goal or query.raw_query}', f'Task type: {query.task_type}' if query.task_type else '', f"Required effect: {('; '.join(need.outputs) if need.outputs else need.to_text())}", f'Required operation: {need.to_text()}', f"Objects: {', '.join(clean_list([need.object, *need.inputs, *query.domain_objects]))}", f"Tools: {', '.join(need.tools)}" if need.tools else '', f"Constraints: {', '.join(clean_list([*need.constraints, *query.constraints]))}"] if part))

    @staticmethod
    def _metaop_effects(metaop: MetaOp) -> list[str]:
        return clean_list([*metaop.effects, *metaop.state_transitions, *metaop.postconditions, *metaop.outputs])

    @classmethod
    def _lexical_score_metaop_for_need(cls, metaop: MetaOp, need: QueryOperationNeed) -> float:
        required_effects = clean_list(need.outputs) or [need.to_text()]
        metaop_effects = cls._metaop_effects(metaop)
        effect_score = overlap_score(metaop_effects, required_effects)
        action_score = overlap_score(metaop.action, need.action)
        object_score = overlap_score(clean_list([metaop.object, *metaop.inputs, *metaop.outputs]), clean_list([need.object, *need.inputs, *need.outputs]))
        tool_score = overlap_score(metaop.tools, need.tools)
        procedure_score = overlap_score(clean_list([*metaop.procedure, *metaop.action_templates, *metaop.termination_conditions]), need.to_text())
        return float(0.4 * effect_score + 0.25 * action_score + 0.2 * object_score + 0.1 * tool_score + 0.05 * procedure_score)

    @classmethod
    def _score_metaop_for_need(cls, metaop: MetaOp, need: QueryOperationNeed, *, lexical_score: float | None=None, dense_score: float=0.0, rank: int | None=None) -> float:
        lexical = cls._lexical_score_metaop_for_need(metaop, need) if lexical_score is None else lexical_score
        dense_bonus = 0.15 * max(0.0, min(1.0, float(dense_score)))
        rank_bonus = 0.03 / max(1, int(rank)) if rank else 0.0
        return float(lexical + dense_bonus + rank_bonus)

    def _add_dependency_closure(self, selected_ids: list[str], selected_scores: dict[str, float], *, max_nodes: int) -> tuple[list[str], dict[str, float]]:
        selected = list(selected_ids)
        selected_set = set(selected)
        incoming: dict[str, list[MetaOpEdge]] = defaultdict(list)
        for edge in self.edges:
            if edge.edge_type in DAG_EDGE_TYPES and edge.source_metaop_id in self.metaops and (edge.target_metaop_id in self.metaops):
                incoming[edge.target_metaop_id].append(edge)
        for edges in incoming.values():
            edges.sort(key=lambda edge: (-float(edge.score), -int(edge.support_count), edge.source_metaop_id))
        changed = True
        while changed and len(selected) < max_nodes:
            changed = False
            for target_id in list(selected):
                for edge in incoming.get(target_id, []):
                    source_id = edge.source_metaop_id
                    if source_id in selected_set:
                        continue
                    selected.append(source_id)
                    selected_set.add(source_id)
                    selected_scores[source_id] = max(selected_scores.get(source_id, 0.0), selected_scores.get(target_id, 0.0) * 0.8, float(edge.score))
                    changed = True
                    break
                if len(selected) >= max_nodes:
                    break
        return (selected, selected_scores)

    def _execution_edges_for_selected(self, selected_ids: list[str]) -> list[MetaOpEdge]:
        selected = set(selected_ids)
        edges = [edge for edge in self.edges if edge.source_metaop_id in selected and edge.target_metaop_id in selected and (edge.source_metaop_id != edge.target_metaop_id) and (edge.edge_type in DAG_EDGE_TYPES)]
        edges.sort(key=lambda edge: (-float(edge.score), edge.source_metaop_id, edge.target_metaop_id, edge.edge_type))
        for index, edge in enumerate(edges, start=1):
            edge.edge_id = edge.edge_id or f'DAG{index:04d}'
        return edges

    @staticmethod
    def _has_cycle(nodes: list[str], edges: list[MetaOpEdge]) -> bool:
        incoming: dict[str, int] = {node: 0 for node in nodes}
        outgoing: dict[str, list[str]] = {node: [] for node in nodes}
        for edge in edges:
            outgoing.setdefault(edge.source_metaop_id, []).append(edge.target_metaop_id)
            incoming[edge.target_metaop_id] = incoming.get(edge.target_metaop_id, 0) + 1
        ready = [node for node in nodes if incoming.get(node, 0) == 0]
        seen = 0
        while ready:
            node = ready.pop()
            seen += 1
            for target in outgoing.get(node, []):
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
        return seen != len(nodes)

    @classmethod
    def _drop_cycle_edges(cls, selected_ids: list[str], edges: list[MetaOpEdge]) -> list[MetaOpEdge]:
        kept = list(edges)
        while cls._has_cycle(selected_ids, kept):
            weakest_index = min(range(len(kept)), key=lambda index: (float(kept[index].score), kept[index].edge_type, kept[index].source_metaop_id, kept[index].target_metaop_id))
            kept.pop(weakest_index)
        return kept

    @staticmethod
    def _stable_topological_order(selected_ids: list[str], edges: list[MetaOpEdge], score: dict[str, float]) -> list[str]:
        requested_position = {node: index for index, node in enumerate(selected_ids)}

        def priority(node: str) -> tuple[int, float, str]:
            return (requested_position.get(node, len(requested_position)), -score.get(node, 0.0), node)
        incoming: dict[str, set[str]] = {node: set() for node in selected_ids}
        outgoing: dict[str, set[str]] = {node: set() for node in selected_ids}
        for edge in edges:
            incoming.setdefault(edge.target_metaop_id, set()).add(edge.source_metaop_id)
            outgoing.setdefault(edge.source_metaop_id, set()).add(edge.target_metaop_id)
        ready = sorted([node for node in selected_ids if not incoming.get(node)], key=priority)
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for target in sorted(outgoing.get(node, set())):
                incoming[target].discard(node)
                if not incoming[target] and target not in order and (target not in ready):
                    ready.append(target)
            ready.sort(key=priority)
        remaining = [node for node in selected_ids if node not in order]
        remaining.sort(key=priority)
        return [*order, *remaining]

    def _executable_metaops(self, selected_ids: list[str], edges: list[MetaOpEdge], **_ignored: Any) -> list[ExecutableMetaOp]:
        depends_on: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            depends_on[edge.target_metaop_id].append(edge.source_metaop_id)
        executable: list[ExecutableMetaOp] = []
        for metaop_id in selected_ids:
            metaop = self.metaops[metaop_id]
            executable.append(ExecutableMetaOp(metaop_id=metaop.metaop_id, name=metaop.name, definition=metaop.definition, action=metaop.action, object=metaop.object, inputs=metaop.inputs, outputs=metaop.outputs, tools=metaop.tools, preconditions=metaop.preconditions, procedure=metaop.procedure, effects=metaop.effects, failure_recovery=metaop.failure_recovery, postconditions=metaop.postconditions, evidence=metaop.evidence, depends_on=clean_list(depends_on.get(metaop_id, [])), argument_roles=metaop.argument_roles, state_transitions=metaop.state_transitions, action_templates=metaop.action_templates, success_checks=metaop.success_checks, failure_avoidance=metaop.failure_avoidance, ordering_constraints=metaop.ordering_constraints, termination_conditions=metaop.termination_conditions, excluded_scope=metaop.excluded_scope, evidence_spans=metaop.evidence_spans, residual_constraints=[], quality_flags=metaop.quality_flags))
        return executable
MetaOpDAGRetriever = MetaOpGraphRetriever
