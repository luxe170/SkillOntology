from __future__ import annotations
import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import re
from typing import Any
from tqdm import tqdm
from skillontology.runtime.agent_cache import AgentCallCache
from skillontology.ontology.domain_ontology import DomainOntologyMemory, ExtractionPolicy, compile_extraction_policy, sha256_file
from skillontology.graph.metaop_edges import build_metaop_edges, build_metaop_edges_with_embeddings
from skillontology.construction.metaop_quality import filter_candidate_metaops
from skillontology.schema import CandidateMetaOp, MetaOp, MetaOpEdge, ResidualConstraint, SkillRecord, clean_list, clean_text
from skillontology.construction.step1_skill_records import async_load_skill_records, save_skill_records, load_skill_records_jsonl
from skillontology.construction.step2_extract_metaops import MetaOpExtractionResult, extract_candidate_metaops_with_residuals
from skillontology.construction.merge_gate import evaluate_merge_gate
from skillontology.construction.step3_candidate_retrieval import EmbeddingCache, retrieve_candidate_metaops, llm_filter_candidate_metaops
from skillontology.construction.step4_conflict_detection import compare_candidate_to_metaop, detect_conflicts
from skillontology.construction.step5_conflict_resolution import resolve_conflicts
from skillontology.construction.adjudication_gate import apply_guarded_resolution_actions, validate_source_fields

class OntologyMemory:

    def __init__(self) -> None:
        self.skill_records: dict[str, SkillRecord] = {}
        self.candidate_metaops: list[CandidateMetaOp] = []
        self.metaops: dict[str, MetaOp] = {}
        self.metaop_to_skills: dict[str, list[str]] = {}
        self.skill_to_metaops: dict[str, list[str]] = {}
        self.candidate_to_metaop: dict[str, str] = {}
        self.residual_constraints: dict[str, ResidualConstraint] = {}
        self.skill_residual_constraints: dict[str, list[str]] = {}
        self.coverage_notes: dict[str, Any] = {}
        self.skill_metaop_sequences: dict[str, list[str]] = {}
        self.metaop_edges: list[MetaOpEdge] = []
        self.audit: dict[str, Any] = {'conflict_rounds': [], 'canonicalization': [], 'deferred_conflicts': []}

    def next_metaop_id(self) -> str:
        max_seen = 0
        for metaop_id in self.metaops:
            match = re.fullmatch('M(\\d+)', metaop_id)
            if match:
                max_seen = max(max_seen, int(match.group(1)))
        return f'M{max_seen + 1:04d}'

    def add_new_metaop(self, candidate: CandidateMetaOp) -> MetaOp:
        metaop = MetaOp.from_candidate(self.next_metaop_id(), candidate)
        self.metaops[metaop.metaop_id] = metaop
        self.attach(candidate.source_skill_id, metaop.metaop_id)
        return metaop

    def attach(self, skill_id: str, metaop_id: str) -> None:
        if not skill_id or not metaop_id:
            return
        self.skill_to_metaops[skill_id] = clean_list([*self.skill_to_metaops.get(skill_id, []), metaop_id])
        self.metaop_to_skills[metaop_id] = clean_list([*self.metaop_to_skills.get(metaop_id, []), skill_id])
        if metaop_id in self.metaops:
            self.metaops[metaop_id].attach_skill(skill_id)

    def record_candidate_mapping(self, candidate: CandidateMetaOp, metaop_id: str) -> None:
        local_id = candidate.local_id or f'{candidate.source_skill_id}::C{int(candidate.step_order):03d}'
        if local_id and metaop_id:
            self.candidate_to_metaop[local_id] = metaop_id

    def add_residual_constraints(self, skill_id: str, constraints: list[ResidualConstraint]) -> None:
        for constraint in constraints:
            if not constraint.constraint_id:
                continue
            self.residual_constraints[constraint.constraint_id] = constraint
            source_skill_id = constraint.source_skill_id or skill_id
            self.skill_residual_constraints[source_skill_id] = clean_list([*self.skill_residual_constraints.get(source_skill_id, []), constraint.constraint_id])

    def add_coverage_notes(self, skill_id: str, notes: list[dict[str, Any]]) -> None:
        if notes:
            self.coverage_notes[skill_id] = list(notes)

    def detach_skill(self, skill_id: str) -> None:
        old_metaop_ids = clean_list(self.skill_to_metaops.pop(skill_id, []))
        self.skill_metaop_sequences.pop(skill_id, None)
        self.candidate_to_metaop = {key: value for key, value in self.candidate_to_metaop.items() if not key.startswith(f'{skill_id}::')}
        for metaop_id in old_metaop_ids:
            if metaop_id in self.metaop_to_skills:
                self.metaop_to_skills[metaop_id] = [item for item in self.metaop_to_skills[metaop_id] if item != skill_id]
                if not self.metaop_to_skills[metaop_id]:
                    self.metaop_to_skills.pop(metaop_id, None)
            if metaop_id in self.metaops:
                self.metaops[metaop_id].skill_ids = [item for item in self.metaops[metaop_id].skill_ids if item != skill_id]
                if not self.metaops[metaop_id].skill_ids:
                    self.metaops.pop(metaop_id, None)

    def rebuild_mappings(self) -> None:
        metaop_to_skills: dict[str, list[str]] = {}
        skill_to_metaops: dict[str, list[str]] = {}
        for metaop_id, metaop in self.metaops.items():
            metaop.skill_ids = clean_list(metaop.skill_ids)
            for skill_id in metaop.skill_ids:
                metaop_to_skills[metaop_id] = clean_list([*metaop_to_skills.get(metaop_id, []), skill_id])
                skill_to_metaops[skill_id] = clean_list([*skill_to_metaops.get(skill_id, []), metaop_id])
        self.metaop_to_skills = metaop_to_skills
        self.skill_to_metaops = skill_to_metaops

    def remap_applied_resolution_actions(self, decisions: list[dict[str, Any]]) -> None:
        for decision in decisions:
            if decision.get('status') != 'applied':
                continue
            proposal = decision['proposal']
            ids = clean_list(proposal.get('source_metaop_ids', []))
            if proposal.get('action', '').lower() == 'merge':
                self.candidate_to_metaop = {key: ids[0] if value in ids else value for key, value in self.candidate_to_metaop.items()}
            elif proposal.get('action', '').lower() == 'discard':
                self.candidate_to_metaop = {key: value for key, value in self.candidate_to_metaop.items() if value != ids[0]}

    def rebuild_skill_metaop_sequences(self) -> None:
        grouped: dict[str, list[CandidateMetaOp]] = {}
        for candidate in self.candidate_metaops:
            if candidate.source_skill_id:
                grouped.setdefault(candidate.source_skill_id, []).append(candidate)
        sequences: dict[str, list[str]] = {}
        for skill_id, candidates in grouped.items():
            ordered = sorted(candidates, key=lambda item: (int(item.step_order or 0), item.local_id, item.name))
            sequence: list[str] = []
            for candidate in ordered:
                metaop_id = self.candidate_to_metaop.get(candidate.local_id)
                if metaop_id and metaop_id in self.metaops:
                    sequence.append(metaop_id)
            if sequence:
                sequences[skill_id] = sequence
        self.skill_metaop_sequences = sequences

    def rebuild_metaop_edges(self) -> None:
        self.rebuild_skill_metaop_sequences()
        self.metaop_edges = build_metaop_edges(self.metaops, self.skill_metaop_sequences)

    async def rebuild_metaop_edges_async(self, embedding_service: Any | None=None) -> None:
        self.rebuild_skill_metaop_sequences()
        self.metaop_edges = await build_metaop_edges_with_embeddings(self.metaops, self.skill_metaop_sequences, embedding_service)

    async def rebuild_guided_metaop_edges_async(self, embedding_service: Any | None, extraction_policy: ExtractionPolicy) -> None:
        self.rebuild_skill_metaop_sequences()
        self.metaop_edges = await build_metaop_edges_with_embeddings(self.metaops, self.skill_metaop_sequences, embedding_service, extraction_policy=extraction_policy)

    def to_dict(self) -> dict[str, Any]:
        return {'skill_records': [asdict(item) for item in self.skill_records.values()], 'candidate_metaops': [item.to_json_dict() for item in self.candidate_metaops], 'metaops': [item.to_json_dict() for item in self.metaops.values()], 'metaop_to_skills': self.metaop_to_skills, 'skill_to_metaops': self.skill_to_metaops, 'candidate_to_metaop': self.candidate_to_metaop, 'residual_constraints': [item.to_json_dict() for item in self.residual_constraints.values()], 'skill_residual_constraints': self.skill_residual_constraints, 'coverage_notes': self.coverage_notes, 'skill_metaop_sequences': self.skill_metaop_sequences, 'metaop_edges': [item.to_json_dict() for item in self.metaop_edges], 'audit': self.audit}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'OntologyMemory':
        memory = cls()
        memory.skill_records = {item['skill_id']: SkillRecord(**item) for item in data.get('skill_records', [])}
        memory.candidate_metaops = [CandidateMetaOp(**item) for item in data.get('candidate_metaops', [])]
        memory.metaops = {item['metaop_id']: MetaOp(**item) for item in data.get('metaops', [])}
        memory.metaop_to_skills = {str(k): clean_list(v) for k, v in data.get('metaop_to_skills', {}).items()}
        memory.skill_to_metaops = {str(k): clean_list(v) for k, v in data.get('skill_to_metaops', {}).items()}
        memory.candidate_to_metaop = {str(k): str(v) for k, v in data.get('candidate_to_metaop', {}).items()}
        raw_constraints = data.get('residual_constraints', [])
        if isinstance(raw_constraints, dict):
            raw_constraints = list(raw_constraints.values())
        memory.residual_constraints = {item['constraint_id']: ResidualConstraint(**item) for item in raw_constraints if isinstance(item, dict) and item.get('constraint_id')}
        memory.skill_residual_constraints = {str(k): clean_list(v) for k, v in data.get('skill_residual_constraints', {}).items()}
        memory.coverage_notes = data.get('coverage_notes', {}) if isinstance(data.get('coverage_notes', {}), dict) else {}
        memory.skill_metaop_sequences = {str(k): clean_list(v) for k, v in data.get('skill_metaop_sequences', {}).items()}
        memory.metaop_edges = [MetaOpEdge(**item) for item in data.get('metaop_edges', [])]
        memory.audit = data.get('audit', {})
        memory.audit.setdefault('conflict_rounds', [])
        memory.audit.setdefault('canonicalization', [])
        memory.audit.setdefault('deferred_conflicts', [])
        return memory

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'ontology_memory.json').write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        with (path / 'metaops.jsonl').open('w', encoding='utf-8') as handle:
            for metaop in self.metaops.values():
                handle.write(json.dumps(metaop.to_json_dict(), ensure_ascii=False) + '\n')
        with (path / 'candidate_metaops.jsonl').open('w', encoding='utf-8') as handle:
            for metaop in self.candidate_metaops:
                handle.write(json.dumps(metaop.to_json_dict(), ensure_ascii=False) + '\n')
        with (path / 'residual_constraints.jsonl').open('w', encoding='utf-8') as handle:
            for constraint in self.residual_constraints.values():
                handle.write(json.dumps(constraint.to_json_dict(), ensure_ascii=False) + '\n')
        (path / 'coverage_notes.json').write_text(json.dumps(self.coverage_notes, ensure_ascii=False, indent=2), encoding='utf-8')
        with (path / 'metaop_edges.jsonl').open('w', encoding='utf-8') as handle:
            for edge in self.metaop_edges:
                handle.write(json.dumps(edge.to_json_dict(), ensure_ascii=False) + '\n')
        save_skill_records(list(self.skill_records.values()), path / 'skill_records.jsonl')
        (path / 'mappings.json').write_text(json.dumps({'metaop_to_skills': self.metaop_to_skills, 'skill_to_metaops': self.skill_to_metaops}, ensure_ascii=False, indent=2), encoding='utf-8')
        (path / 'candidate_to_metaop.json').write_text(json.dumps(self.candidate_to_metaop, ensure_ascii=False, indent=2), encoding='utf-8')
        (path / 'skill_metaop_sequences.json').write_text(json.dumps(self.skill_metaop_sequences, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, output_dir: str | Path) -> 'OntologyMemory':
        path = Path(output_dir)
        monolith = path / 'ontology_memory.json'
        if monolith.exists():
            data = json.loads(monolith.read_text(encoding='utf-8'))
            return cls.from_dict(data)
        memory = cls()
        skill_records = path / 'skill_records.jsonl'
        if skill_records.exists():
            memory.skill_records = {item.skill_id: item for item in load_skill_records_jsonl(skill_records)}
        metaops = path / 'metaops.jsonl'
        if not metaops.exists():
            raise FileNotFoundError(f'Neither {monolith} nor {metaops} exists')
        memory.metaops = {item['metaop_id']: MetaOp(**item) for line in metaops.read_text(encoding='utf-8').splitlines() if line.strip() for item in [json.loads(line)]}
        candidates = path / 'candidate_metaops.jsonl'
        if candidates.exists():
            memory.candidate_metaops = [CandidateMetaOp(**json.loads(line)) for line in candidates.read_text(encoding='utf-8').splitlines() if line.strip()]
        mappings = path / 'mappings.json'
        if mappings.exists():
            data = json.loads(mappings.read_text(encoding='utf-8'))
            memory.metaop_to_skills = {str(key): clean_list(value) for key, value in data.get('metaop_to_skills', {}).items()}
            memory.skill_to_metaops = {str(key): clean_list(value) for key, value in data.get('skill_to_metaops', {}).items()}
        else:
            memory.rebuild_mappings()
        candidate_mapping = path / 'candidate_to_metaop.json'
        if candidate_mapping.exists():
            memory.candidate_to_metaop = {str(key): str(value) for key, value in json.loads(candidate_mapping.read_text(encoding='utf-8')).items()}
        sequences = path / 'skill_metaop_sequences.json'
        if sequences.exists():
            memory.skill_metaop_sequences = {str(key): clean_list(value) for key, value in json.loads(sequences.read_text(encoding='utf-8')).items()}
        coverage = path / 'coverage_notes.json'
        if coverage.exists():
            data = json.loads(coverage.read_text(encoding='utf-8'))
            memory.coverage_notes = data if isinstance(data, dict) else {}
        constraints = path / 'residual_constraints.jsonl'
        if constraints.exists():
            for line in constraints.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    item = ResidualConstraint(**json.loads(line))
                    memory.residual_constraints[item.constraint_id] = item
        edges = path / 'metaop_edges.jsonl'
        if edges.exists():
            memory.metaop_edges = [MetaOpEdge(**json.loads(line)) for line in edges.read_text(encoding='utf-8').splitlines() if line.strip()]
        memory.audit['loaded_from_component_artifacts'] = True
        return memory

async def _attach_or_create(memory: OntologyMemory, candidate: CandidateMetaOp, llm_service: Any | None, embedding_service: Any | None, embedding_cache: EmbeddingCache | None, *, use_llm: bool, candidate_filter_llm: bool, canonical_merge_llm: bool=True, candidate_top_k: int=8, agent_cache: AgentCallCache | None=None, extraction_policy: ExtractionPolicy | None=None) -> str:
    if not memory.metaops:
        metaop = memory.add_new_metaop(candidate)
        memory.record_candidate_mapping(candidate, metaop.metaop_id)
        memory.audit.setdefault('canonicalization', []).append({'candidate_local_id': candidate.local_id, 'source_skill_id': candidate.source_skill_id, 'candidate': candidate.to_json_dict(), 'retrieved_metaops': [], 'comparisons': [], 'approved_metaop_ids': [], 'final_action': 'CREATE_NEW_METAOP', 'final_metaop_id': metaop.metaop_id})
        return metaop.metaop_id
    ranked = await retrieve_candidate_metaops(candidate, list(memory.metaops.values()), embedding_service=embedding_service, embedding_cache=embedding_cache, candidate_top_k=candidate_top_k)
    selected_ids = await llm_filter_candidate_metaops(candidate, ranked, llm_service, use_llm=use_llm and candidate_filter_llm, agent_cache=agent_cache)
    selected = [memory.metaops[item] for item in selected_ids if item in memory.metaops]
    if not selected:
        metaop = memory.add_new_metaop(candidate)
        memory.record_candidate_mapping(candidate, metaop.metaop_id)
        memory.audit.setdefault('canonicalization', []).append({'candidate_local_id': candidate.local_id, 'source_skill_id': candidate.source_skill_id, 'candidate': candidate.to_json_dict(), 'retrieved_metaops': [{'metaop_id': item.metaop_id, 'metaop_name': item.name, 'score': score, 'selected_for_comparison': False} for item, score in ranked], 'comparisons': [], 'approved_metaop_ids': [], 'final_action': 'CREATE_NEW_METAOP', 'final_metaop_id': metaop.metaop_id})
        return metaop.metaop_id
    score_by_id = {item.metaop_id: score for item, score in ranked}
    comparison_items: list[dict[str, Any]] = []
    approved: list[MetaOp] = []
    for existing in selected:
        comparison = await compare_candidate_to_metaop(candidate, existing, llm_service, use_llm=use_llm and canonical_merge_llm, agent_cache=agent_cache, extraction_policy=extraction_policy)
        decision = evaluate_merge_gate(comparison)
        comparison_items.append({'metaop_id': existing.metaop_id, 'metaop_name': existing.name, 'candidate_score': score_by_id.get(existing.metaop_id, 0.0), 'comparison': comparison.to_json_dict(), 'merge_gate': decision.to_json_dict()})
        if decision.allowed:
            approved.append(existing)
    final_action = 'CREATE_NEW_METAOP'
    final_metaop_id = ''
    if len(approved) == 1:
        target = approved[0]
        target.merge_candidate_evidence(candidate)
        memory.attach(candidate.source_skill_id, target.metaop_id)
        memory.record_candidate_mapping(candidate, target.metaop_id)
        final_action = 'MERGE_TO_EXISTING'
        final_metaop_id = target.metaop_id
    else:
        metaop = memory.add_new_metaop(candidate)
        memory.record_candidate_mapping(candidate, metaop.metaop_id)
        final_metaop_id = metaop.metaop_id
        if len(approved) > 1:
            memory.audit.setdefault('deferred_conflicts', []).append({'reason': 'multiple_gate_approved_matches', 'candidate_local_id': candidate.local_id, 'source_skill_id': candidate.source_skill_id, 'approved_metaop_ids': [item.metaop_id for item in approved], 'created_metaop_id': metaop.metaop_id})
    memory.audit.setdefault('canonicalization', []).append({'candidate_local_id': candidate.local_id, 'source_skill_id': candidate.source_skill_id, 'candidate': candidate.to_json_dict(), 'retrieved_metaops': [{'metaop_id': item.metaop_id, 'metaop_name': item.name, 'score': score, 'selected_for_comparison': item.metaop_id in {metaop.metaop_id for metaop in selected}} for item, score in ranked], 'comparisons': comparison_items, 'approved_metaop_ids': [item.metaop_id for item in approved], 'final_action': final_action, 'final_metaop_id': final_metaop_id})
    return final_metaop_id

def _candidate_from_replacement_payload(skill_id: str, row: dict[str, Any], index: int) -> CandidateMetaOp | None:
    if not isinstance(row, dict):
        return None
    allowed = set(CandidateMetaOp.__dataclass_fields__)
    payload = {key: value for key, value in row.items() if key in allowed}
    payload['source_skill_id'] = skill_id
    payload.setdefault('step_order', index)
    try:
        candidate = CandidateMetaOp(**payload)
    except TypeError:
        return None
    if not (candidate.action and candidate.object and candidate.evidence):
        return None
    return candidate

async def _apply_replacement_candidate_metaops(memory: OntologyMemory, resolution: dict[str, Any], llm_service: Any | None, embedding_service: Any | None, embedding_cache: EmbeddingCache | None, *, use_llm: bool, candidate_filter_llm: bool, canonical_merge_llm: bool, candidate_top_k: int, agent_cache: AgentCallCache | None, extraction_policy: ExtractionPolicy | None, allowed_skill_ids: set[str] | None=None) -> dict[str, Any]:
    replacements = resolution.get('replacement_candidate_metaops', {})
    if not isinstance(replacements, dict) or not replacements:
        return {'skills_rewritten': 0, 'candidate_metaops_added': 0}
    rewritten_skills = 0
    added_candidates = 0
    replacement_audit = []
    for skill_id, rows in replacements.items():
        skill_id = str(skill_id)
        entry = {'skill_id': skill_id, 'status': 'rejected', 'reasons': []}
        replacement_audit.append(entry)
        if skill_id not in memory.skill_records or not isinstance(rows, list) or (allowed_skill_ids is not None and skill_id not in allowed_skill_ids):
            entry['reasons'] = ['unknown_skill_or_malformed_replacement']
            continue
        new_candidates = [candidate for index, row in enumerate(rows, start=1) if (candidate := _candidate_from_replacement_payload(skill_id, row, index)) is not None]
        if not new_candidates or len(new_candidates) != len(rows):
            entry['reasons'] = ['empty_or_partially_invalid_replacement']
            continue
        source = memory.skill_records[skill_id].raw_content
        for candidate in new_candidates:
            if extraction_policy is not None:
                extraction_policy.normalize_candidate(candidate)
            entry['reasons'].extend(validate_source_fields(candidate, [source]))
        filtered = filter_candidate_metaops(new_candidates)
        if len(filtered) != len(new_candidates):
            entry['reasons'].append('replacement_quality_filter_rejected_node')
        if entry['reasons']:
            continue
        draft = deepcopy(memory)
        draft.detach_skill(skill_id)
        draft.candidate_metaops = [candidate for candidate in draft.candidate_metaops if candidate.source_skill_id != skill_id]
        draft.candidate_metaops.extend(new_candidates)
        for candidate in new_candidates:
            await _attach_or_create(draft, candidate, llm_service, embedding_service, embedding_cache, use_llm=use_llm, candidate_filter_llm=candidate_filter_llm, canonical_merge_llm=canonical_merge_llm, candidate_top_k=candidate_top_k, agent_cache=agent_cache, extraction_policy=extraction_policy)
        memory.__dict__.update(draft.__dict__)
        rewritten_skills += 1
        added_candidates += len(new_candidates)
        entry['status'] = 'applied'
        entry['accepted_candidate_count'] = len(new_candidates)
    memory.rebuild_mappings()
    return {'skills_rewritten': rewritten_skills, 'candidate_metaops_added': added_candidates, 'decisions': replacement_audit}

async def _run_conflict_rounds(memory: OntologyMemory, llm_service: Any | None, embedding_service: Any | None, embedding_cache: EmbeddingCache | None, *, agent2_use_llm: bool, agent3_use_llm: bool, candidate_filter_llm: bool, canonical_merge_llm: bool, candidate_top_k: int, max_conflict_rounds: int, agent_cache: AgentCallCache | None, extraction_policy: ExtractionPolicy | None) -> None:
    for round_index in range(1, max_conflict_rounds + 1):
        conflicts = await detect_conflicts(list(memory.metaops.values()), {'metaop_to_skills': memory.metaop_to_skills, 'skill_to_metaops': memory.skill_to_metaops}, llm_service, use_llm=agent2_use_llm, agent_cache=agent_cache, extraction_policy=extraction_policy)
        round_record: dict[str, Any] = {'round': round_index, 'conflicts': conflicts}
        if not conflicts:
            round_record['status'] = 'no_conflicts'
            memory.audit.setdefault('conflict_rounds', []).append(round_record)
            break
        involved_ids = sorted({item for row in conflicts for item in row.get('metaop_ids', [])})
        involved = [memory.metaops[item] for item in involved_ids if item in memory.metaops]
        involved_skill_ids = sorted({skill_id for metaop_id in involved_ids for skill_id in memory.metaop_to_skills.get(metaop_id, [])})
        by_skill: dict[str, list[CandidateMetaOp]] = {}
        for candidate in memory.candidate_metaops:
            if candidate.source_skill_id in involved_skill_ids:
                by_skill.setdefault(candidate.source_skill_id, []).append(candidate)
        merge_gate_approved_sets: set[frozenset[str]] = set()
        conflicts_for_agent: list[dict[str, Any]] = []
        for conflict in conflicts:
            ids = [item for item in clean_list(conflict.get('metaop_ids', [])) if item in memory.metaops]
            pair_decisions = []
            all_allowed = len(ids) >= 2
            for left_id, right_id in combinations(ids, 2):
                left = memory.metaops[left_id]
                right = memory.metaops[right_id]
                candidate = CandidateMetaOp(name=left.name, definition=left.definition, action=left.action, object=left.object, inputs=left.inputs, outputs=left.outputs, tools=left.tools, preconditions=left.preconditions, procedure=left.procedure, effects=left.effects, failure_recovery=left.failure_recovery, argument_roles=left.argument_roles, state_transitions=left.state_transitions, postconditions=left.postconditions, action_templates=left.action_templates, success_checks=left.success_checks, failure_avoidance=left.failure_avoidance, ordering_constraints=left.ordering_constraints, termination_conditions=left.termination_conditions, excluded_scope=left.excluded_scope, evidence_spans=left.evidence_spans, evidence=left.evidence, source_skill_id=left.skill_ids[0] if left.skill_ids else '')
                comparison = await compare_candidate_to_metaop(candidate, right, None, use_llm=False)
                decision = evaluate_merge_gate(comparison)
                pair_decisions.append({'metaop_ids': [left_id, right_id], 'merge_gate': decision.to_json_dict()})
                all_allowed = all_allowed and decision.allowed
            if all_allowed:
                merge_gate_approved_sets.add(frozenset(ids))
            conflicts_for_agent.append({**conflict, 'deterministic_merge_gate_allowed': all_allowed, 'pairwise_merge_gates': pair_decisions})
        resolution = await resolve_conflicts(conflicts_for_agent, involved, by_skill, {'metaop_to_skills': memory.metaop_to_skills, 'skill_to_metaops': memory.skill_to_metaops}, llm_service, use_llm=agent3_use_llm, agent_cache=agent_cache, extraction_policy=extraction_policy, source_skill_texts={skill_id: memory.skill_records[skill_id].raw_content for skill_id in involved_skill_ids if skill_id in memory.skill_records}, previous_decisions=[{'actions': record.get('action_decisions', []), 'replacements': record.get('replacement_report', {}).get('decisions', [])} for record in memory.audit.get('conflict_rounds', [])[-1:]])
        action_report = apply_guarded_resolution_actions(memory.metaops, memory.skill_to_metaops, memory.metaop_to_skills, resolution, merge_gate_approved_sets=merge_gate_approved_sets, skill_records=memory.skill_records, extraction_policy=extraction_policy, allowed_metaop_ids=set(involved_ids))
        memory.remap_applied_resolution_actions(action_report)
        memory.rebuild_mappings()
        replacement_report = await _apply_replacement_candidate_metaops(memory, resolution, llm_service, embedding_service, embedding_cache, use_llm=agent2_use_llm, candidate_filter_llm=candidate_filter_llm, canonical_merge_llm=canonical_merge_llm, candidate_top_k=candidate_top_k, agent_cache=agent_cache, extraction_policy=extraction_policy, allowed_skill_ids=set(involved_skill_ids))
        round_record['resolution'] = resolution
        round_record['action_decisions'] = action_report
        round_record['replacement_report'] = replacement_report
        round_record['status'] = 'adjudicated_automatic_gate_completed'
        memory.audit.setdefault('conflict_rounds', []).append(round_record)
    if memory.audit.get('conflict_rounds') and memory.audit['conflict_rounds'][-1].get('status') != 'no_conflicts':
        remaining = await detect_conflicts(list(memory.metaops.values()), {'metaop_to_skills': memory.metaop_to_skills, 'skill_to_metaops': memory.skill_to_metaops}, llm_service, use_llm=agent2_use_llm, agent_cache=agent_cache, extraction_policy=extraction_policy)
        memory.audit['final_conflict_check'] = {'status': 'remaining_conflicts_retained' if remaining else 'no_conflicts', 'conflicts': remaining, 'automatic_round_budget': max_conflict_rounds}

async def _extract_all_candidate_metaops(records: list[SkillRecord], llm_service: Any | None, *, use_llm: bool, agent_workers: int, extraction_policy: ExtractionPolicy | None, agent_mode: str, agent_cache: AgentCallCache | None) -> list[MetaOpExtractionResult]:
    workers = max(1, int(agent_workers or 1))
    raw_timeout = os.environ.get('META_GOS_AGENT1_TIMEOUT_SECS') or os.environ.get('GOS_AGENT1_TIMEOUT_SECS')
    agent1_timeout = float(raw_timeout) if raw_timeout else 240.0
    if agent1_timeout <= 0:
        agent1_timeout = 0.0

    async def extract_one(skill: SkillRecord) -> MetaOpExtractionResult:
        coro = extract_candidate_metaops_with_residuals(skill, llm_service, use_llm=use_llm, extraction_policy=extraction_policy, agent_mode=agent_mode, agent_cache=agent_cache)
        try:
            if agent1_timeout:
                result = await asyncio.wait_for(coro, timeout=agent1_timeout)
            else:
                result = await coro
        except asyncio.TimeoutError:
            tqdm.write(f'[warn] Agent1 extraction timed out after {agent1_timeout:.0f}s; no fallback MetaOp generated for {skill.skill_id} ({skill.name}).')
            return MetaOpExtractionResult(audit_records=[{'candidate_local_id': '', 'source_skill_id': skill.skill_id, 'agent_mode': agent_mode, 'agent_status': 'timeout', 'ontology_guided': extraction_policy is not None, 'policy_sha256': extraction_policy.policy_sha256 if extraction_policy is not None else '', 'policy_rule_ids': [], 'skill_evidence': {}, 'response_cache_path': '', 'error_type': 'TimeoutError', 'error': f'Agent 1 exceeded {agent1_timeout:.0f}s'}])
        result.metaops = filter_candidate_metaops(result.metaops)
        return result
    if workers == 1 or len(records) <= 1:
        results: list[MetaOpExtractionResult] = []
        for skill in tqdm(records, desc='Step 2 Agent1 extract MetaOps'):
            results.append(await extract_one(skill))
        return results
    semaphore = asyncio.Semaphore(workers)

    async def run_one(index: int, skill: SkillRecord) -> tuple[int, MetaOpExtractionResult]:
        async with semaphore:
            return (index, await extract_one(skill))
    tasks = [asyncio.create_task(run_one(index, skill)) for index, skill in enumerate(records)]
    results: list[MetaOpExtractionResult] = [MetaOpExtractionResult() for _ in records]
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f'Step 2 Agent1 extract MetaOps x{workers}'):
        index, result = await future
        results[index] = result
    return results

async def _precache_candidate_and_metaop_embeddings(extraction_results: list[MetaOpExtractionResult], embedding_cache: EmbeddingCache | None) -> None:
    if embedding_cache is None:
        return
    candidates = [candidate for result in extraction_results for candidate in result.metaops]
    if not candidates:
        return
    items: list[tuple[str, str, str]] = []
    for candidate in candidates:
        items.append(('candidate', candidate.local_id or candidate.name, candidate.to_embedding_text()))
        metaop_text = MetaOp.from_candidate('M0000', candidate).to_embedding_text()
        items.append(('metaop', candidate.local_id or candidate.name, metaop_text))
    await embedding_cache.encode_items(items)

def _resolve_agent_configuration(agent_mode: str | None, *, use_llm: bool, llm_service: Any | None, candidate_filter_llm: bool, canonical_merge_llm: bool, max_conflict_rounds: int) -> dict[str, Any]:
    if agent_mode is None:
        return {'mode': 'legacy', 'explicit': False, 'agent1': bool(use_llm and llm_service is not None), 'agent2_filter': bool(use_llm and candidate_filter_llm and (llm_service is not None)), 'agent2_compare': bool(use_llm and canonical_merge_llm and (llm_service is not None)), 'agent2_conflicts': bool(use_llm and max_conflict_rounds > 0 and (llm_service is not None)), 'agent3': bool(use_llm and max_conflict_rounds > 0 and (llm_service is not None)), 'conflict_rounds': max(0, int(max_conflict_rounds))}
    mode = clean_text(agent_mode).lower()
    if mode not in {'deterministic', 'extract_only', 'full'}:
        raise ValueError('agent_mode must be one of: deterministic, extract_only, full')
    if mode != 'deterministic' and (not use_llm or llm_service is None):
        raise ValueError(f'agent_mode={mode!r} requires an LLM service; use deterministic for an offline build')
    if mode == 'deterministic':
        return {'mode': mode, 'explicit': True, 'agent1': False, 'agent2_filter': False, 'agent2_compare': False, 'agent2_conflicts': False, 'agent3': False, 'conflict_rounds': 0}
    if mode == 'extract_only':
        return {'mode': mode, 'explicit': True, 'agent1': True, 'agent2_filter': False, 'agent2_compare': False, 'agent2_conflicts': False, 'agent3': False, 'conflict_rounds': 0}
    return {'mode': mode, 'explicit': True, 'agent1': True, 'agent2_filter': True, 'agent2_compare': True, 'agent2_conflicts': True, 'agent3': True, 'conflict_rounds': max(1, int(max_conflict_rounds or 0))}

def _write_extraction_audit(output_dir: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(output_dir) / 'extraction_audit.jsonl'
    with target.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')

def _skill_input_manifest(skills_dir: str | Path) -> list[dict[str, str]]:
    root = Path(skills_dir).expanduser().resolve()
    paths = [root] if root.is_file() and root.name == 'SKILL.md' else list(root.rglob('SKILL.md'))
    return [{'path': path.name if root.is_file() else str(path.relative_to(root)).replace('\\', '/'), 'sha256': sha256_file(path)} for path in sorted(paths, key=lambda item: str(item).lower())]

def _write_build_manifest(output_dir: str | Path, *, skills_dir: str | Path, memory: OntologyMemory, policy: ExtractionPolicy | None, agent_config: dict[str, Any], build_config: dict[str, Any]) -> None:
    output = Path(output_dir).expanduser().resolve()
    failed = [row for row in memory.audit.get('extraction', []) if row.get('agent_status') in {'failed', 'timeout'}]
    artifacts = {str(path.relative_to(output)).replace('\\', '/'): sha256_file(path) for path in sorted(output.rglob('*'), key=lambda item: str(item).lower()) if path.is_file() and 'agent_cache' not in path.parts and (path.name != 'build_manifest.json')}
    source_root = Path(__file__).resolve().parent
    implementation_files = sorted(source_root.parent.rglob('*.py'))
    implementation_digest = hashlib.sha256()
    for path in implementation_files:
        implementation_digest.update(path.name.encode('utf-8'))
        implementation_digest.update(path.read_bytes())
    payload = {'schema_version': 1, 'status': 'completed_with_agent_failures' if failed else 'ok', 'source_skills_dir': str(Path(skills_dir).expanduser().resolve()), 'input_skills': _skill_input_manifest(skills_dir), 'agent_configuration': agent_config, 'build_config': build_config, 'agent_cache': {'key_scheme': 'request_sha256_v1'}, 'implementation_sha256': implementation_digest.hexdigest(), 'domain_ontology': {'enabled': policy is not None, 'domain': policy.domain if policy is not None else '', 'source_path': policy.source_domain_ontology_path if policy is not None else '', 'source_sha256': policy.source_domain_ontology_sha256 if policy is not None else '', 'policy_sha256': policy.policy_sha256 if policy is not None else '', 'compiler_version': policy.compiler_version if policy is not None else ''}, 'counts': {'skills': len(memory.skill_records), 'candidate_metaops': len(memory.candidate_metaops), 'metaops': len(memory.metaops), 'edges': len(memory.metaop_edges), 'agent_failures': len(failed)}, 'artifacts': artifacts}
    (output / 'build_manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')

async def build_ontology_memory(skills_dir: str | Path, output_dir: str | Path, *, llm_service: Any | None, embedding_service: Any | None, use_llm: bool=True, candidate_top_k: int=8, max_conflict_rounds: int=0, snippet_chars: int=900, agent_workers: int=1, candidate_filter_llm: bool=True, canonical_merge_llm: bool=True, use_full_markdown: bool=False, domain_ontology: str | Path | DomainOntologyMemory | None=None, ontology_guided_extraction: bool | None=None, extraction_policy_output: str | Path | None=None, replay_agent_cache: str | Path | None=None, agent_mode: str | None=None) -> OntologyMemory:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    guided = bool(domain_ontology is not None) if ontology_guided_extraction is None else bool(ontology_guided_extraction)
    if guided and domain_ontology is None:
        raise ValueError('ontology_guided_extraction requires domain_ontology')
    domain_memory: DomainOntologyMemory | None = None
    if guided:
        domain_memory = domain_ontology if isinstance(domain_ontology, DomainOntologyMemory) else DomainOntologyMemory.load(domain_ontology)
    extraction_policy = compile_extraction_policy(domain_memory) if domain_memory is not None else None
    if extraction_policy is not None:
        extraction_policy.save(extraction_policy_output or output_path / 'extraction_policy.json')
    agent_config = _resolve_agent_configuration(agent_mode, use_llm=use_llm, llm_service=llm_service, candidate_filter_llm=candidate_filter_llm, canonical_merge_llm=canonical_merge_llm, max_conflict_rounds=max_conflict_rounds)
    feature_enabled = bool(replay_agent_cache or extraction_policy is not None or agent_config['explicit'])
    needs_cache = feature_enabled
    agent_cache = AgentCallCache(output_path / 'agent_cache', replay_dir=replay_agent_cache, strict_failures=bool(agent_config['explicit'] and agent_config['mode'] != 'deterministic')) if needs_cache else None
    memory = OntologyMemory()
    if feature_enabled:
        memory.audit['agent_configuration'] = agent_config
        memory.audit['extraction_policy'] = extraction_policy.to_dict() if extraction_policy is not None else None
        memory.audit['extraction'] = []
    memory.audit['step1'] = {'use_full_markdown': bool(use_full_markdown and agent_config['agent1'] and (llm_service is not None)), 'snippet_chars': snippet_chars, 'workers': max(1, int(agent_workers or 1))}
    embedding_cache = EmbeddingCache(embedding_service) if embedding_service is not None else None
    records = await async_load_skill_records(skills_dir, llm_service, snippet_chars=snippet_chars, use_full_markdown=use_full_markdown and agent_config['agent1'] and (llm_service is not None), workers=agent_workers)
    memory.skill_records = {item.skill_id: item for item in records}
    extraction_results = await _extract_all_candidate_metaops(records, llm_service, use_llm=agent_config['agent1'], agent_workers=agent_workers, extraction_policy=extraction_policy, agent_mode=agent_config['mode'], agent_cache=agent_cache)
    await _precache_candidate_and_metaop_embeddings(extraction_results, embedding_cache)
    for skill, extraction in tqdm(list(zip(records, extraction_results)), desc='Steps 3-6 build ontology memory'):
        candidates = extraction.metaops
        memory.candidate_metaops.extend(candidates)
        memory.add_residual_constraints(skill.skill_id, extraction.residual_constraints)
        memory.add_coverage_notes(skill.skill_id, extraction.coverage_notes)
        if feature_enabled:
            memory.audit['extraction'].extend(extraction.audit_records)
        for candidate in candidates:
            await _attach_or_create(memory, candidate, llm_service, embedding_service, embedding_cache, use_llm=agent_config['agent2_filter'] or agent_config['agent2_compare'], candidate_filter_llm=agent_config['agent2_filter'], canonical_merge_llm=agent_config['agent2_compare'], candidate_top_k=candidate_top_k, agent_cache=agent_cache, extraction_policy=extraction_policy)
    if agent_config['conflict_rounds'] > 0:
        await _run_conflict_rounds(memory, llm_service, embedding_service, embedding_cache, agent2_use_llm=agent_config['agent2_conflicts'], agent3_use_llm=agent_config['agent3'], candidate_filter_llm=agent_config['agent2_filter'], canonical_merge_llm=agent_config['agent2_compare'], candidate_top_k=candidate_top_k, max_conflict_rounds=agent_config['conflict_rounds'], agent_cache=agent_cache, extraction_policy=extraction_policy)
    memory.rebuild_mappings()
    if extraction_policy is not None:
        await memory.rebuild_guided_metaop_edges_async(embedding_service, extraction_policy)
    else:
        await memory.rebuild_metaop_edges_async(embedding_service)
    memory.save(output_dir)
    if feature_enabled:
        _write_extraction_audit(output_dir, memory.audit['extraction'])
        _write_build_manifest(output_dir, skills_dir=skills_dir, memory=memory, policy=extraction_policy, agent_config=agent_config, build_config={'candidate_top_k': int(candidate_top_k), 'max_conflict_rounds_requested': int(max_conflict_rounds), 'snippet_chars': int(snippet_chars), 'agent_workers': max(1, int(agent_workers or 1)), 'use_full_markdown': bool(use_full_markdown), 'llm_model': clean_text(getattr(llm_service, 'model', '') or getattr(llm_service, 'model_name', '')), 'llm_temperature': getattr(llm_service, 'temperature', None), 'llm_seed': getattr(llm_service, 'seed', None), 'embedding_model': clean_text(getattr(embedding_service, 'model', '') or getattr(embedding_service, 'model_name', ''))})
    return memory
