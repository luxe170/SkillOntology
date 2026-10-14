from __future__ import annotations
from copy import deepcopy
import re
from typing import Any
from skillontology.schema import CandidateMetaOp, MetaOp, clean_list
from skillontology.construction.step5_conflict_resolution import apply_resolution_actions
FACT_FIELDS = ('action', 'object', 'inputs', 'outputs', 'tools', 'preconditions', 'procedure', 'effects', 'failure_recovery', 'argument_roles', 'state_transitions', 'postconditions', 'action_templates', 'success_checks', 'failure_avoidance', 'ordering_constraints', 'termination_conditions', 'excluded_scope')

def _normal(value: Any) -> str:
    return re.sub('\\s+', ' ', str(value)).strip().casefold()

def validate_source_fields(item: CandidateMetaOp | MetaOp, source_texts: list[str]) -> list[str]:
    sources = [_normal(text) for text in source_texts]
    issues = []
    for field in FACT_FIELDS:
        if not getattr(item, field):
            continue
        spans = item.evidence_spans.get(field, [])
        if not spans or any((len(_normal(span)) < 4 or not any((_normal(span) in text for text in sources)) for span in spans)):
            issues.append(f'unsupported_source_quote:{field}')
    for field in ('name', 'definition'):
        spans = item.evidence_spans.get(field, [])
        if any((len(_normal(span)) < 4 or not any((_normal(span) in text for text in sources)) for span in spans)):
            issues.append(f'unsupported_source_quote:{field}')
    if not item.action or not item.object:
        issues.append('missing_action_or_object')
    if not any((getattr(item, field) for field in ('outputs', 'effects', 'state_transitions', 'postconditions', 'success_checks', 'termination_conditions'))):
        issues.append('missing_source_supported_result')
    return issues

def normalize_metaop(metaop: MetaOp, policy: Any | None) -> None:
    if policy is None:
        return
    payload = {key: deepcopy(getattr(metaop, key)) for key in CandidateMetaOp.__dataclass_fields__ if hasattr(metaop, key)}
    payload['source_skill_id'] = metaop.skill_ids[0] if metaop.skill_ids else ''
    candidate = CandidateMetaOp(**payload)
    policy.normalize_candidate(candidate)
    for field in (*FACT_FIELDS, 'unknown_fields', 'quality_flags'):
        setattr(metaop, field, deepcopy(getattr(candidate, field)))

def apply_guarded_resolution_actions(metaops, skill_to_metaops, metaop_to_skills, resolution, *, skill_records, extraction_policy=None, merge_gate_approved_sets=None, allowed_metaop_ids=None) -> list[dict[str, Any]]:
    report = []
    for index, action in enumerate(resolution.get('actions', []), 1):
        entry = {'action_index': index, 'proposal': deepcopy(action), 'status': 'rejected', 'reasons': []}
        report.append(entry)
        if not isinstance(action, dict):
            entry['reasons'] = ['malformed_action']
            continue
        ids = clean_list(action.get('source_metaop_ids', []))
        kind = str(action.get('action', '')).lower()
        if not ids or any((i not in metaops or (allowed_metaop_ids is not None and i not in allowed_metaop_ids) for i in ids)):
            entry['reasons'] = ['unknown_or_out_of_conflict_source']
            continue
        if kind == 'keep_separate':
            entry['status'] = 'kept_separate'
            continue
        if kind not in {'modify', 'merge', 'discard'}:
            entry['reasons'] = ['unsupported_action_kind']
            continue
        if kind == 'merge' and frozenset(ids) not in (merge_gate_approved_sets or set()):
            entry['reasons'] = ['merge_gate_not_approved']
            continue
        original_sources = [deepcopy(metaops[i]) for i in ids]
        draft, forward, reverse = (deepcopy(metaops), deepcopy(skill_to_metaops), deepcopy(metaop_to_skills))
        try:
            apply_resolution_actions(draft, forward, reverse, {'actions': [action]}, merge_gate_approved_sets=merge_gate_approved_sets)
        except (TypeError, ValueError, KeyError) as exc:
            entry['reasons'] = [f'malformed_patch:{type(exc).__name__}']
            continue
        if draft == metaops and forward == skill_to_metaops and (reverse == metaop_to_skills):
            entry['reasons'] = ['low_level_gate_rejected_or_no_change']
            continue
        if kind != 'discard':
            target = draft[ids[0]]
            source_ids = set((s for m in original_sources for s in m.skill_ids))
            if set(target.skill_ids) != source_ids:
                entry['reasons'] = ['source_skill_mapping_changed']
                continue
            missing = source_ids - set(skill_records)
            if missing:
                entry['reasons'] = ['missing_source_skill_record']
                continue
            normalize_metaop(target, extraction_policy)
            entry['reasons'] = validate_source_fields(target, [skill_records[s].raw_content for s in sorted(source_ids)])
            if entry['reasons']:
                continue
        metaops.clear()
        metaops.update(draft)
        skill_to_metaops.clear()
        skill_to_metaops.update(forward)
        metaop_to_skills.clear()
        metaop_to_skills.update(reverse)
        entry['status'] = 'applied'
        entry['before_source_ids'] = ids
        entry['after_metaop_ids'] = [i for i in ids if i in metaops]
    return report
