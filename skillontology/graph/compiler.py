from __future__ import annotations

from dataclasses import replace
import re

from skillontology.ontology.domain_ontology import DomainOntologyMemory
from skillontology.ontology.grounding import ground_query
from skillontology.rendering import render_metaop_context
from skillontology.retrieval.graph_retrieval import MetaOpGraphRetriever
from skillontology.schema import MetaOpDAG, MetaOpEdge


class TaskCompiler:
    def __init__(self, asset_dir, embedding_service, config):
        self.ontology = DomainOntologyMemory.load(asset_dir / 'domain_ontology.json')
        self.retriever = MetaOpGraphRetriever(asset_dir, embedding_service=embedding_service)
        self.config = config

    async def compile(self, task, observation, adapter):
        if adapter.name != self.ontology.domain:
            raise ValueError('Benchmark and ontology domain disagree')
        query = ground_query(adapter.parse_task(task, observation, self.ontology), self.ontology)
        selected = await self.retriever.retrieve(query, seed_top_k=int(self.config.get('seed_top_k', 8)),
                                                 top_n=int(self.config.get('top_k', 8)), max_chars=None)
        roles = adapter.binding_roles(task, query)
        count = 1
        for value in query.constraints:
            match = re.fullmatch(r'count\s*=\s*(\d+)', value)
            if match:
                count = max(1, min(int(match.group(1)), 8))
        nodes = []
        missing = list(selected.missing_operations)
        used = set()
        ordered_nodes = []
        for index, need in enumerate(query.required_operations, 1):
            if need.action == 'repeat':
                continue
            ranked = sorted(selected.selected_metaops, key=lambda node: (-self.retriever._lexical_score_metaop_for_need(node, need), node.metaop_id))
            if not ranked or self.retriever._lexical_score_metaop_for_need(ranked[0], need) < 0.08:
                missing.append(need.name)
                continue
            node = ranked[0]
            ordered_nodes.append((index, node))
            used.add(node.metaop_id)
        for node in selected.selected_metaops:
            if node.metaop_id not in used:
                ordered_nodes.insert(0, (0, node))
        for ordinal in range(1, count + 1):
            for index, node in ordered_nodes:
                instance = f'I{ordinal:02d}_{index:02d}_{node.metaop_id}'
                placeholders = set(re.findall(r'\{([A-Za-z_][A-Za-z0-9_]*)\}', ' '.join([node.object, *node.procedure, *node.action_templates])))
                bindings = {key: value for key, value in roles.items() if key in placeholders}
                if count > 1 and roles.get('object'):
                    bindings['object_slot'] = f"{roles['object']}#{ordinal}"
                values = {}
                for field in ('object','inputs','outputs','tools','preconditions','postconditions','procedure','effects','action_templates'):
                    raw = getattr(node, field)
                    bind = lambda text: re.sub(r'\{([A-Za-z_][A-Za-z0-9_]*)\}', lambda match: roles.get(match.group(1), match.group(0)), text)
                    values[field] = [bind(text) for text in raw] if isinstance(raw, list) else bind(raw)
                nodes.append(replace(node, instance_id=instance, instance_ordinal=ordinal, instance_count=count,
                                     bindings=bindings, unresolved_roles=sorted(placeholders - set(bindings)), depends_on=[], **values))
        edges = []
        for edge in selected.edges:
            sources = [node for node in nodes if node.metaop_id == edge.source_metaop_id]
            targets = [node for node in nodes if node.metaop_id == edge.target_metaop_id]
            for source in sources:
                for target in targets:
                    if source.instance_id != target.instance_id and source.instance_ordinal == target.instance_ordinal:
                        edges.append(replace(edge, edge_id=f'{edge.edge_id}_{source.instance_id}_{target.instance_id}',
                                             source_metaop_id=source.instance_id, target_metaop_id=target.instance_id))
        for source, target in zip(nodes, nodes[1:]):
            if source.instance_id == target.instance_id:
                continue
            edges.append(MetaOpEdge(edge_id=f'Q_{source.instance_id}_{target.instance_id}',
                                   source_metaop_id=source.instance_id, target_metaop_id=target.instance_id,
                                   edge_type='workflow', score=1.0, reason='indexed public operation requirement order',
                                   evidence=query.normalized_goal))
        identifiers = [node.instance_id for node in nodes]
        edges = self.retriever._drop_cycle_edges(identifiers, edges)
        priorities = {node.instance_id: float(len(nodes)-index) for index,node in enumerate(nodes)}
        order = self.retriever._stable_topological_order(identifiers, edges, priorities)
        by_id = {node.instance_id: node for node in nodes}
        for edge in edges:
            by_id[edge.target_metaop_id].depends_on.append(edge.source_metaop_id)
        dag = MetaOpDAG(task.instruction, query, [by_id[key] for key in order], edges, order, sorted(set(missing)), '')
        limit = int(self.config.get('max_context_chars', 12000))
        dag.rendered_context = render_metaop_context(dag, max_chars=limit)[:limit]
        return dag
