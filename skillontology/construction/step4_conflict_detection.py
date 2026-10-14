from __future__ import annotations
import json
from typing import Any
from skillontology.runtime.agent_cache import AgentCallCache
from skillontology.runtime.llm_json import call_json
from skillontology.construction.prompts import AGENT2_PAIRWISE_SYSTEM, AGENT2_PAIRWISE_USER, AGENT2_SYSTEM, AGENT2_USER
from skillontology.schema import CandidateMetaOp, Counterexample, FieldComparison, MetaOp, MetaOpComparison, clean_list, overlap_score
COMPARISON_FIELDS = ['action', 'object', 'argument_roles', 'inputs', 'outputs', 'tools', 'preconditions', 'procedure', 'effects', 'failure_recovery', 'state_transitions', 'postconditions', 'action_templates', 'success_checks', 'failure_avoidance', 'ordering_constraints', 'termination_conditions', 'excluded_scope']
STRICT_COMPARISON_FIELDS = {'action', 'tools', 'procedure', 'effects', 'failure_recovery', 'state_transitions', 'action_templates', 'postconditions', 'success_checks', 'preconditions', 'ordering_constraints', 'termination_conditions'}
SAME_ONLY_COMPARISON_FIELDS = {'tools', 'procedure', 'effects', 'state_transitions', 'action_templates'}

def _heuristic_conflicts(metaops: list[MetaOp]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for left_index, left in enumerate(metaops):
        for right in metaops[left_index + 1:]:
            same_operation = overlap_score(left.action, right.action) >= 0.8 and overlap_score(left.object, right.object) >= 0.8
            if not same_operation:
                continue
            output_overlap = overlap_score(left.outputs, right.outputs)
            input_overlap = overlap_score(left.inputs, right.inputs)
            if left.outputs and right.outputs and (output_overlap == 0.0) and (input_overlap >= 0.5):
                conflicts.append({'metaop_ids': [left.metaop_id, right.metaop_id], 'reason': 'same action/object and similar inputs but incompatible outputs'})
    return conflicts

async def detect_conflicts(metaops: list[MetaOp], mappings: dict[str, Any], llm_service: Any | None, *, use_llm: bool=True, agent_cache: AgentCallCache | None=None, extraction_policy: Any | None=None) -> list[dict[str, Any]]:
    if len(metaops) < 2:
        return []
    if not use_llm or llm_service is None:
        return _heuristic_conflicts(metaops)
    try:
        prompt = AGENT2_USER.format(metaops_json=json.dumps([item.to_json_dict() for item in metaops], ensure_ascii=False, indent=2), mappings_json=json.dumps(mappings, ensure_ascii=False, indent=2))
        if extraction_policy is not None:
            prompt += '\nFrozen Extraction Policy for typed/workflow conflict checks:\n' + json.dumps(extraction_policy.prompt_dict(), ensure_ascii=False, indent=2)
        if agent_cache is not None:
            data, _record = await agent_cache.call_json('agent2_global_conflict_scan', llm_service, system_prompt=AGENT2_SYSTEM, prompt=prompt, max_tokens=1200)
        else:
            data = await call_json(llm_service, system_prompt=AGENT2_SYSTEM, prompt=prompt, max_tokens=1200)
        rows = data.get('conflict_sets', [])
        conflicts: list[dict[str, Any]] = []
        valid_ids = {item.metaop_id for item in metaops}
        for row in rows:
            if not isinstance(row, dict):
                continue
            ids = [str(item) for item in row.get('metaop_ids', []) if str(item) in valid_ids]
            if len(ids) >= 2:
                conflicts.append({'metaop_ids': ids, 'reason': str(row.get('reason', '')).strip()})
        return conflicts
    except Exception:
        if agent_cache is not None and agent_cache.strict_failures:
            raise
        return _heuristic_conflicts(metaops)

def _value_as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return clean_list(value)
    if isinstance(value, tuple):
        return clean_list(list(value))
    text = str(value or '').strip()
    return [text] if text else []

def _field_values(item: CandidateMetaOp | MetaOp, field_name: str) -> list[str]:
    return _value_as_list(getattr(item, field_name, ''))

def _evidence_for(item: CandidateMetaOp | MetaOp, field_name: str) -> list[str]:
    spans = getattr(item, 'evidence_spans', {}) or {}
    evidence = spans.get(field_name) or []
    if evidence:
        return clean_list(evidence)
    return clean_list([getattr(item, 'evidence', '')])

def _format_values(values: list[str]) -> str:
    return '; '.join(clean_list(values))

def _compare_field(candidate: CandidateMetaOp, existing: MetaOp, field_name: str) -> FieldComparison:
    candidate_values = _field_values(candidate, field_name)
    existing_values = _field_values(existing, field_name)
    candidate_text = _format_values(candidate_values)
    existing_text = _format_values(existing_values)
    if not candidate_values and (not existing_values):
        status = 'SAME'
        explanation = 'both values are absent'
    elif not candidate_values or not existing_values:
        status = 'UNKNOWN' if field_name in STRICT_COMPARISON_FIELDS else 'COMPATIBLE'
        explanation = 'one side lacks evidence for this field'
    else:
        score = overlap_score(candidate_values, existing_values)
        same_set = {item.lower() for item in candidate_values} == {item.lower() for item in existing_values}
        if same_set or score >= 0.8:
            status = 'SAME'
            explanation = 'field values are equivalent by lexical overlap'
        elif field_name in SAME_ONLY_COMPARISON_FIELDS:
            status = 'DIFFERENT'
            explanation = 'this field must match exactly for automatic merge'
        elif score >= 0.25:
            status = 'COMPATIBLE'
            explanation = 'field values overlap without changing executable identity'
        else:
            status = 'DIFFERENT'
            explanation = 'field values differ'
    return FieldComparison(status=status, candidate_value=candidate_text, existing_value=existing_text, explanation=explanation, candidate_evidence=_evidence_for(candidate, field_name), existing_evidence=_evidence_for(existing, field_name))

def _heuristic_pairwise_comparison(candidate: CandidateMetaOp, existing: MetaOp) -> MetaOpComparison:
    field_comparisons = {field_name: _compare_field(candidate, existing, field_name) for field_name in COMPARISON_FIELDS}
    blocking = [field_name for field_name, comparison in field_comparisons.items() if field_name in STRICT_COMPARISON_FIELDS and comparison.status in {'DIFFERENT', 'UNKNOWN'}]
    counterexample_found = any((field_comparisons[field_name].status == 'DIFFERENT' for field_name in ['tools', 'procedure', 'effects', 'state_transitions', 'action_templates', 'postconditions', 'success_checks']))
    strict_ok = not blocking
    action_ok = field_comparisons['action'].status == 'SAME'
    object_ok = field_comparisons['object'].status in {'SAME', 'COMPATIBLE'}
    if strict_ok and action_ok and object_ok:
        relation = 'EQUIVALENT'
        merge_recommendation = True
        confidence = 0.9
        summary = 'Heuristic comparison found no blocking executable differences.'
    elif blocking:
        relation = 'RELATED'
        merge_recommendation = False
        confidence = 0.75
        summary = 'Heuristic comparison found blocking executable differences.'
    else:
        relation = 'UNCERTAIN'
        merge_recommendation = False
        confidence = 0.6
        summary = 'Heuristic comparison lacks enough evidence for equivalence.'
    return MetaOpComparison(candidate_local_id=candidate.local_id, existing_metaop_id=existing.metaop_id, semantic_relation=relation, field_comparisons=field_comparisons, blocking_differences=blocking, counterexample=Counterexample(found=counterexample_found, explanation='A strict executable field differs.' if counterexample_found else ''), merge_recommendation=merge_recommendation, confidence=confidence, reasoning_summary=summary)

def _parse_pairwise_comparison(data: dict[str, Any], candidate: CandidateMetaOp, existing: MetaOp) -> MetaOpComparison:
    field_rows = data.get('field_comparisons', {})
    if not isinstance(field_rows, dict):
        field_rows = {}
    field_comparisons: dict[str, FieldComparison] = {}
    for field_name in COMPARISON_FIELDS:
        row = field_rows.get(field_name)
        if isinstance(row, dict):
            payload = {k: v for k, v in row.items() if k in FieldComparison.__dataclass_fields__}
            payload.setdefault('status', 'UNKNOWN')
            field_comparisons[field_name] = FieldComparison(**payload)
        else:
            field_comparisons[field_name] = FieldComparison(status='UNKNOWN', explanation='Agent 2 omitted this field; automatic merge is blocked.')
    counterexample = data.get('counterexample', {})
    if not isinstance(counterexample, dict):
        counterexample = {}
    counterexample = {k: v for k, v in counterexample.items() if k in Counterexample.__dataclass_fields__}
    return MetaOpComparison(candidate_local_id=candidate.local_id, existing_metaop_id=existing.metaop_id, semantic_relation=str(data.get('semantic_relation', 'UNCERTAIN')), field_comparisons=field_comparisons, blocking_differences=data.get('blocking_differences', []), counterexample=Counterexample(**counterexample), merge_recommendation=bool(data.get('merge_recommendation', False)), confidence=data.get('confidence', 0.0), reasoning_summary=str(data.get('reasoning_summary', '')))

async def compare_candidate_to_metaop(candidate: CandidateMetaOp, existing: MetaOp, llm_service: Any | None, *, use_llm: bool=True, agent_cache: AgentCallCache | None=None, extraction_policy: Any | None=None) -> MetaOpComparison:
    if not use_llm or llm_service is None:
        return _heuristic_pairwise_comparison(candidate, existing)
    try:
        prompt = AGENT2_PAIRWISE_USER.format(candidate_json=json.dumps(candidate.to_json_dict(), ensure_ascii=False, indent=2), existing_json=json.dumps(existing.to_json_dict(), ensure_ascii=False, indent=2))
        if extraction_policy is not None:
            prompt += '\nFrozen Extraction Policy for canonical action/type comparison:\n' + json.dumps(extraction_policy.prompt_dict(), ensure_ascii=False, indent=2)
        if agent_cache is not None:
            data, _record = await agent_cache.call_json('agent2_pairwise_compare', llm_service, system_prompt=AGENT2_PAIRWISE_SYSTEM, prompt=prompt, max_tokens=3500)
        else:
            data = await call_json(llm_service, system_prompt=AGENT2_PAIRWISE_SYSTEM, prompt=prompt, max_tokens=3500)
        if not isinstance(data, dict):
            data = {}
        fields = data.get('field_comparisons', {})
        fields = fields if isinstance(fields, dict) else {}
        incomplete = [name for name in COMPARISON_FIELDS if not isinstance(fields.get(name), dict) or not fields[name].get('status')]
        if incomplete:
            repair_prompt = prompt + '\nThe previous response omitted field comparison statuses for ' + ', '.join(incomplete) + '. Return the complete required schema with a status for EVERY field. Be concise; do not omit fields. Unknown is safer than guessing.'
            if agent_cache is not None:
                data, _record = await agent_cache.call_json('agent2_pairwise_compare_repair', llm_service, system_prompt=AGENT2_PAIRWISE_SYSTEM, prompt=repair_prompt, max_tokens=3500)
            else:
                data = await call_json(llm_service, system_prompt=AGENT2_PAIRWISE_SYSTEM, prompt=repair_prompt, max_tokens=3500)
        return _parse_pairwise_comparison(data, candidate, existing)
    except Exception:
        if agent_cache is not None and agent_cache.strict_failures:
            raise
        return _heuristic_pairwise_comparison(candidate, existing)
