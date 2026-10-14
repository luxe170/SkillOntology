from __future__ import annotations
from typing import Any
from skillontology.schema import CandidateMetaOp, token_set
GENERIC_ACTIONS = {'be', 'check', 'ensure', 'follow', 'inspect', 'review', 'run', 'think', 'use'}
LOW_SIGNAL_PHRASES = {'be careful', 'think carefully', 'inspect tests', 'always search', 'use this skill', 'read the instructions', 'follow the skill', 'this skill should be used', 'use when', 'allowed tools', 'available tools', 'return json', 'do not include'}
DROP_FLAGS = {'literal_heading', 'meta_instruction', 'non_operation_text', 'missing_executable_result'}

def executable_completeness_score(metaop: Any) -> float:
    score = 0.0
    if getattr(metaop, 'action', ''):
        score += 0.12
    if getattr(metaop, 'object', ''):
        score += 0.12
    if getattr(metaop, 'argument_roles', []):
        score += 0.1
    if getattr(metaop, 'inputs', []):
        score += 0.08
    if getattr(metaop, 'outputs', []):
        score += 0.08
    if getattr(metaop, 'tools', []):
        score += 0.06
    if getattr(metaop, 'preconditions', []):
        score += 0.08
    if getattr(metaop, 'procedure', []):
        score += 0.14
    if getattr(metaop, 'action_templates', []):
        score += 0.16
    if getattr(metaop, 'effects', []):
        score += 0.12
    if getattr(metaop, 'state_transitions', []):
        score += 0.1
    if getattr(metaop, 'postconditions', []):
        score += 0.08
    if getattr(metaop, 'success_checks', []):
        score += 0.1
    if getattr(metaop, 'termination_conditions', []):
        score += 0.07
    if getattr(metaop, 'failure_avoidance', []):
        score += 0.03
    if getattr(metaop, 'evidence', '') or getattr(metaop, 'evidence_spans', {}):
        score += 0.06
    return max(0.0, min(1.0, score))

def metaop_quality_score(candidate: CandidateMetaOp) -> float:
    score = 1.0
    if not candidate.action:
        score -= 0.35
    if not candidate.object:
        score -= 0.35
    if not candidate.evidence:
        score -= 0.2
    text = ' '.join([candidate.name, candidate.definition, candidate.action, candidate.object, candidate.evidence, ' '.join(candidate.action_templates), ' '.join(candidate.procedure), ' '.join(candidate.effects), ' '.join(candidate.failure_recovery), ' '.join(candidate.success_checks)]).lower()
    if any((phrase in text for phrase in LOW_SIGNAL_PHRASES)):
        score -= 0.5
    if candidate.action in GENERIC_ACTIONS:
        object_tokens = token_set(candidate.object)
        definition_tokens = token_set(candidate.definition)
        if len(object_tokens | definition_tokens) < 5:
            score -= 0.35
    if not (candidate.outputs or candidate.effects or candidate.state_transitions or candidate.postconditions or candidate.success_checks or candidate.termination_conditions or (candidate.inputs and candidate.outputs)):
        score -= 0.45
    if not candidate.action_templates:
        score -= 0.1
    if not (candidate.success_checks or candidate.termination_conditions or candidate.postconditions):
        score -= 0.1
    if not candidate.argument_roles and (not (candidate.inputs or candidate.outputs)):
        score -= 0.06
    if executable_completeness_score(candidate) < 0.45:
        score -= 0.18
    if len(token_set([candidate.name, candidate.definition, candidate.object])) < 3:
        score -= 0.2
    return max(0.0, min(1.0, score))

def should_keep_candidate_metaop(candidate: CandidateMetaOp, *, threshold: float=0.0) -> bool:
    flags = set(candidate.quality_flags or [])
    if flags & DROP_FLAGS:
        return False
    if 'generic_action' in flags and (not (candidate.inputs and candidate.outputs)):
        return False
    candidate.quality_score = metaop_quality_score(candidate)
    return candidate.quality_score >= threshold

def filter_candidate_metaops(candidates: list[CandidateMetaOp], *, threshold: float=0.25) -> list[CandidateMetaOp]:
    kept: list[CandidateMetaOp] = []
    for candidate in candidates:
        if should_keep_candidate_metaop(candidate, threshold=threshold):
            kept.append(candidate)
    return kept
