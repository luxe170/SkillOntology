from __future__ import annotations
from skillontology.schema import MergeDecision, MetaOpComparison
MERGEABLE_RELATIONS = {'EQUIVALENT', 'ALIAS'}
STRICT_FIELDS = {'action', 'tools', 'procedure', 'effects', 'failure_recovery', 'state_transitions', 'action_templates', 'postconditions', 'success_checks', 'preconditions', 'ordering_constraints', 'termination_conditions'}
SAME_ONLY_FIELDS = {'tools', 'procedure', 'effects', 'state_transitions', 'action_templates'}

def evaluate_merge_gate(comparison: MetaOpComparison, *, confidence_threshold: float=0.85) -> MergeDecision:
    reasons: list[str] = []
    if comparison.semantic_relation not in MERGEABLE_RELATIONS:
        reasons.append(f'non_mergeable_relation:{comparison.semantic_relation}')
    if comparison.counterexample.found:
        reasons.append('counterexample_found')
    for field_name in sorted(STRICT_FIELDS):
        field = comparison.field_comparisons.get(field_name)
        if field is None:
            reasons.append(f'missing_field_comparison:{field_name}')
            continue
        if field_name in SAME_ONLY_FIELDS:
            if field.status != 'SAME':
                reasons.append(f'strict_field_must_be_same:{field_name}:{field.status}')
            continue
        if field.status not in {'SAME', 'COMPATIBLE'}:
            reasons.append(f'strict_field_not_mergeable:{field_name}:{field.status}')
    for item in comparison.blocking_differences:
        reasons.append(f'blocking_difference:{item}')
    if comparison.confidence < confidence_threshold:
        reasons.append('confidence_below_threshold')
    return MergeDecision(allowed=not reasons, reasons=reasons, confidence=comparison.confidence)
