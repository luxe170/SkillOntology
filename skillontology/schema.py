from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Literal

@dataclass
class SkillNode:
    name: str
    description: str = ''
    one_line_capability: str = ''
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    domain_tags_list: list[str] = field(default_factory=list)
    tooling_list: list[str] = field(default_factory=list)
    example_tasks_list: list[str] = field(default_factory=list)
    script_entrypoints_list: list[str] = field(default_factory=list)
    compatibility_list: list[str] = field(default_factory=list)
    allowed_tools_list: list[str] = field(default_factory=list)
    source_path: str = ''
    rendered_snippet: str = ''
    raw_content: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)
    skill_id: str = ''

    @classmethod
    def from_lists(cls, **kwargs: Any) -> 'SkillNode':
        aliases = {'inputs': 'input_types', 'outputs': 'output_types', 'domain_tags': 'domain_tags_list', 'tooling': 'tooling_list', 'example_tasks': 'example_tasks_list', 'script_entrypoints': 'script_entrypoints_list', 'compatibility': 'compatibility_list', 'allowed_tools': 'allowed_tools_list'}
        for source, target in aliases.items():
            if source in kwargs and target not in kwargs:
                kwargs[target] = kwargs.pop(source)
        return cls(**kwargs)

    def to_str(self) -> str:
        parts = [f'Name: {self.name}', f'Description: {self.description}' if self.description else '', f'Capability: {self.one_line_capability}' if self.one_line_capability else '', f"Inputs: {', '.join(self.input_types)}" if self.input_types else '', f"Outputs: {', '.join(self.output_types)}" if self.output_types else '', f"Tools: {', '.join(self.tooling_list)}" if self.tooling_list else '', f"Allowed tools: {', '.join(self.allowed_tools_list)}" if self.allowed_tools_list else '', self.rendered_snippet]
        return '\n'.join((part for part in parts if part))

def clean_text(value: Any) -> str:
    return re.sub('\\s+', ' ', str(value or '').strip())

def clean_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_items = re.split('[\\n;,]+', values)
    elif isinstance(values, (list, tuple, set)):
        raw_items = list(values)
    else:
        raw_items = [values]
    seen: set[str] = set()
    result: list[str] = []
    for item in raw_items:
        text = clean_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result

def clean_dict_lists(values: Any) -> dict[str, list[str]]:
    if not isinstance(values, dict):
        return {}
    result: dict[str, list[str]] = {}
    for key, value in values.items():
        key_text = clean_text(key)
        cleaned_values = clean_list(value)
        if key_text and cleaned_values:
            result[key_text] = cleaned_values
    return result

def evidence_summary_from_spans(values: Any, *, max_parts: int=5, max_chars: int=500) -> str:
    spans = clean_dict_lists(values)
    if not spans:
        return ''
    preferred = ['action', 'object', 'inputs', 'outputs', 'tools', 'preconditions', 'state_transitions', 'postconditions', 'action_templates', 'success_checks', 'failure_avoidance', 'termination_conditions', 'ordering_constraints', 'operation']
    ordered_keys = [key for key in preferred if key in spans]
    ordered_keys.extend((key for key in spans if key not in set(ordered_keys)))
    parts: list[str] = []
    seen: set[str] = set()
    for key in ordered_keys:
        for span in spans.get(key, []):
            text = clean_text(span)
            if not text:
                continue
            part = f'{key}: {text}'
            marker = part.lower()
            if marker in seen:
                continue
            seen.add(marker)
            parts.append(part)
            if len(parts) >= max_parts:
                return clean_text(' | '.join(parts))[:max_chars]
    return clean_text(' | '.join(parts))[:max_chars]

def token_set(values: list[str] | tuple[str, ...] | str) -> set[str]:
    if isinstance(values, str):
        text = values
    else:
        text = '\n'.join((str(v) for v in values))
    return {token.rstrip('s') for token in re.findall('[a-z0-9]+', text.lower()) if len(token) >= 3}

def overlap_score(left: list[str] | str, right: list[str] | str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
RESIDUAL_CONSTRAINT_KINDS = {'tool_policy', 'warning', 'success_condition', 'precondition', 'environment_assumption', 'usage_note', 'example', 'other'}

@dataclass
class ResidualConstraint:
    constraint_id: str
    source_skill_id: str
    kind: str
    text: str
    applies_to: list[str] = field(default_factory=list)
    evidence: str = ''

    def __post_init__(self) -> None:
        self.constraint_id = clean_text(self.constraint_id)
        self.source_skill_id = clean_text(self.source_skill_id)
        self.kind = clean_text(self.kind).lower() or 'other'
        if self.kind not in RESIDUAL_CONSTRAINT_KINDS:
            self.kind = 'other'
        self.text = clean_text(self.text)
        self.applies_to = clean_list(self.applies_to)
        self.evidence = clean_text(self.evidence)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class SkillRecord:
    skill_id: str
    name: str
    description: str
    one_line_capability: str = ''
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)
    tooling: list[str] = field(default_factory=list)
    example_tasks: list[str] = field(default_factory=list)
    script_entrypoints: list[str] = field(default_factory=list)
    compatibility: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    source_path: str = ''
    rendered_snippet: str = ''
    raw_content: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_skill_node(cls, node: SkillNode) -> 'SkillRecord':
        return cls(skill_id=node.skill_id or node.source_path or node.name, name=node.name, description=node.description, one_line_capability=node.one_line_capability, inputs=node.input_types, outputs=node.output_types, domain_tags=node.domain_tags_list, tooling=node.tooling_list, example_tasks=node.example_tasks_list, script_entrypoints=node.script_entrypoints_list, compatibility=node.compatibility_list, allowed_tools=node.allowed_tools_list, source_path=node.source_path, rendered_snippet=node.rendered_snippet, raw_content=node.raw_content, metadata=node.metadata)

    def to_skill_node(self) -> SkillNode:
        return SkillNode.from_lists(name=self.name, description=self.description, one_line_capability=self.one_line_capability, inputs=self.inputs, outputs=self.outputs, domain_tags=self.domain_tags, tooling=self.tooling, example_tasks=self.example_tasks, script_entrypoints=self.script_entrypoints, compatibility=self.compatibility, allowed_tools=self.allowed_tools, source_path=self.source_path, rendered_snippet=self.rendered_snippet, raw_content=self.raw_content, metadata=self.metadata, skill_id=self.skill_id)

    def to_embedding_text(self) -> str:
        return self.to_skill_node().to_str()

    def compact_summary(self) -> dict[str, Any]:
        return {'skill_id': self.skill_id, 'name': self.name, 'description': self.description, 'one_line_capability': self.one_line_capability, 'inputs': self.inputs, 'outputs': self.outputs, 'tooling': self.tooling}

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

@dataclass
class CandidateMetaOp:
    name: str
    definition: str
    action: str
    object: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    procedure: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    failure_recovery: list[str] = field(default_factory=list)
    argument_roles: list[str] = field(default_factory=list)
    state_transitions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    action_templates: list[str] = field(default_factory=list)
    success_checks: list[str] = field(default_factory=list)
    failure_avoidance: list[str] = field(default_factory=list)
    ordering_constraints: list[str] = field(default_factory=list)
    termination_conditions: list[str] = field(default_factory=list)
    excluded_scope: list[str] = field(default_factory=list)
    evidence_spans: dict[str, list[str]] = field(default_factory=dict)
    unknown_fields: list[str] = field(default_factory=list)
    residual_constraints: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    step_order: int = 1
    evidence: str = ''
    source_skill_id: str = ''
    local_id: str = ''
    quality_score: float = 1.0

    def __post_init__(self) -> None:
        self.name = clean_text(self.name)
        self.definition = clean_text(self.definition)
        self.action = clean_text(self.action).lower()
        self.object = clean_text(self.object)
        self.inputs = clean_list(self.inputs)
        self.outputs = clean_list(self.outputs)
        self.tools = clean_list(self.tools)
        self.preconditions = clean_list(self.preconditions)
        self.procedure = clean_list(self.procedure)
        self.effects = clean_list(self.effects)
        self.failure_recovery = clean_list(self.failure_recovery)
        self.argument_roles = clean_list(self.argument_roles)
        self.state_transitions = clean_list(self.state_transitions)
        self.postconditions = clean_list(self.postconditions)
        self.action_templates = clean_list(self.action_templates)
        self.success_checks = clean_list(self.success_checks)
        self.failure_avoidance = clean_list(self.failure_avoidance)
        self.ordering_constraints = clean_list(self.ordering_constraints)
        self.termination_conditions = clean_list(self.termination_conditions)
        self.excluded_scope = clean_list(self.excluded_scope)
        self.evidence_spans = clean_dict_lists(self.evidence_spans)
        self.unknown_fields = clean_list(self.unknown_fields)
        self.residual_constraints = clean_list(self.residual_constraints)
        self.quality_flags = clean_list(self.quality_flags)
        self.evidence = clean_text(self.evidence)
        if not self.evidence:
            self.evidence = evidence_summary_from_spans(self.evidence_spans)
        if not self.procedure:
            self.procedure = clean_list(self.action_templates)
        if not self.effects:
            self.effects = clean_list([*self.state_transitions, *self.postconditions, *self.outputs])
        if not self.failure_recovery:
            self.failure_recovery = clean_list(self.failure_avoidance)
        if not self.failure_avoidance:
            self.failure_avoidance = clean_list(self.failure_recovery)
        self.source_skill_id = clean_text(self.source_skill_id)
        try:
            self.quality_score = float(self.quality_score)
        except (TypeError, ValueError):
            self.quality_score = 1.0
        if not self.local_id and self.source_skill_id:
            self.local_id = f'{self.source_skill_id}::C{int(self.step_order):03d}'

    def to_embedding_text(self) -> str:
        return '\n'.join((part for part in [f'[CandidateMetaOp] {self.name}', f'[Definition] {self.definition}', f'[Action] {self.action}', f'[Object] {self.object}', f"[Inputs] {', '.join(self.inputs)}" if self.inputs else '', f"[Outputs] {', '.join(self.outputs)}" if self.outputs else '', f"[Tools] {', '.join(self.tools)}" if self.tools else '', f"[Preconditions] {', '.join(self.preconditions)}" if self.preconditions else '', f"[Procedure] {', '.join(self.procedure)}" if self.procedure else '', f"[Effects] {', '.join(self.effects)}" if self.effects else '', f"[FailureRecovery] {', '.join(self.failure_recovery)}" if self.failure_recovery else '', f"[ArgumentRoles] {', '.join(self.argument_roles)}" if self.argument_roles else '', f"[StateTransitions] {', '.join(self.state_transitions)}" if self.state_transitions else '', f"[Postconditions] {', '.join(self.postconditions)}" if self.postconditions else '', f"[ActionTemplates] {', '.join(self.action_templates)}" if self.action_templates else '', f"[SuccessChecks] {', '.join(self.success_checks)}" if self.success_checks else '', f"[FailureAvoidance] {', '.join(self.failure_avoidance)}" if self.failure_avoidance else '', f"[OrderingConstraints] {', '.join(self.ordering_constraints)}" if self.ordering_constraints else '', f"[TerminationConditions] {', '.join(self.termination_conditions)}" if self.termination_conditions else '', f"[ExcludedScope] {', '.join(self.excluded_scope)}" if self.excluded_scope else ''] if part))

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class MetaOp:
    metaop_id: str
    name: str
    definition: str
    action: str
    object: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    procedure: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    failure_recovery: list[str] = field(default_factory=list)
    argument_roles: list[str] = field(default_factory=list)
    state_transitions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    action_templates: list[str] = field(default_factory=list)
    success_checks: list[str] = field(default_factory=list)
    failure_avoidance: list[str] = field(default_factory=list)
    ordering_constraints: list[str] = field(default_factory=list)
    termination_conditions: list[str] = field(default_factory=list)
    excluded_scope: list[str] = field(default_factory=list)
    evidence_spans: dict[str, list[str]] = field(default_factory=dict)
    unknown_fields: list[str] = field(default_factory=list)
    residual_constraints: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    skill_ids: list[str] = field(default_factory=list)
    evidence: str = ''

    def __post_init__(self) -> None:
        self.name = clean_text(self.name)
        self.definition = clean_text(self.definition)
        self.action = clean_text(self.action).lower()
        self.object = clean_text(self.object)
        self.inputs = clean_list(self.inputs)
        self.outputs = clean_list(self.outputs)
        self.tools = clean_list(self.tools)
        self.preconditions = clean_list(self.preconditions)
        self.procedure = clean_list(self.procedure)
        self.effects = clean_list(self.effects)
        self.failure_recovery = clean_list(self.failure_recovery)
        self.argument_roles = clean_list(self.argument_roles)
        self.state_transitions = clean_list(self.state_transitions)
        self.postconditions = clean_list(self.postconditions)
        self.action_templates = clean_list(self.action_templates)
        self.success_checks = clean_list(self.success_checks)
        self.failure_avoidance = clean_list(self.failure_avoidance)
        self.ordering_constraints = clean_list(self.ordering_constraints)
        self.termination_conditions = clean_list(self.termination_conditions)
        self.excluded_scope = clean_list(self.excluded_scope)
        self.evidence_spans = clean_dict_lists(self.evidence_spans)
        self.unknown_fields = clean_list(self.unknown_fields)
        self.residual_constraints = clean_list(self.residual_constraints)
        self.quality_flags = clean_list(self.quality_flags)
        self.aliases = clean_list(self.aliases)
        self.skill_ids = clean_list(self.skill_ids)
        self.evidence = clean_text(self.evidence)
        if not self.evidence:
            self.evidence = evidence_summary_from_spans(self.evidence_spans)
        if not self.procedure:
            self.procedure = clean_list(self.action_templates)
        if not self.effects:
            self.effects = clean_list([*self.state_transitions, *self.postconditions, *self.outputs])
        if not self.failure_recovery:
            self.failure_recovery = clean_list(self.failure_avoidance)
        if not self.failure_avoidance:
            self.failure_avoidance = clean_list(self.failure_recovery)

    @classmethod
    def from_candidate(cls, metaop_id: str, candidate: CandidateMetaOp) -> 'MetaOp':
        return cls(metaop_id=metaop_id, name=candidate.name, definition=candidate.definition, action=candidate.action, object=candidate.object, inputs=candidate.inputs, outputs=candidate.outputs, tools=candidate.tools, preconditions=candidate.preconditions, procedure=candidate.procedure, effects=candidate.effects, failure_recovery=candidate.failure_recovery, argument_roles=candidate.argument_roles, state_transitions=candidate.state_transitions, postconditions=candidate.postconditions, action_templates=candidate.action_templates, success_checks=candidate.success_checks, failure_avoidance=candidate.failure_avoidance, ordering_constraints=candidate.ordering_constraints, termination_conditions=candidate.termination_conditions, excluded_scope=candidate.excluded_scope, evidence_spans=candidate.evidence_spans, unknown_fields=candidate.unknown_fields, residual_constraints=candidate.residual_constraints, quality_flags=candidate.quality_flags, aliases=[candidate.name], skill_ids=[candidate.source_skill_id] if candidate.source_skill_id else [], evidence=candidate.evidence)

    def attach_skill(self, skill_id: str) -> None:
        skill_id = clean_text(skill_id)
        if skill_id and skill_id not in self.skill_ids:
            self.skill_ids.append(skill_id)

    def merge_candidate_evidence(self, candidate: CandidateMetaOp) -> None:
        for attr in ['inputs', 'outputs', 'tools', 'preconditions', 'procedure', 'effects', 'failure_recovery', 'argument_roles', 'state_transitions', 'postconditions', 'action_templates', 'success_checks', 'failure_avoidance', 'ordering_constraints', 'termination_conditions', 'excluded_scope', 'unknown_fields', 'residual_constraints', 'quality_flags']:
            setattr(self, attr, clean_list([*getattr(self, attr), *getattr(candidate, attr)]))
        merged_evidence = dict(self.evidence_spans)
        for field_name, spans in candidate.evidence_spans.items():
            merged_evidence[field_name] = clean_list([*merged_evidence.get(field_name, []), *spans])
        self.evidence_spans = merged_evidence
        self.aliases = clean_list([*self.aliases, candidate.name])
        evidence_parts = clean_list([self.evidence, candidate.evidence])
        self.evidence = ' | '.join(evidence_parts[:5])
        if not self.evidence:
            self.evidence = evidence_summary_from_spans(self.evidence_spans)
        self.attach_skill(candidate.source_skill_id)

    def to_embedding_text(self) -> str:
        return '\n'.join((part for part in [f'[MetaOp] {self.name}', f'[Definition] {self.definition}', f'[Action] {self.action}', f'[Object] {self.object}', f"[Inputs] {', '.join(self.inputs)}" if self.inputs else '', f"[Outputs] {', '.join(self.outputs)}" if self.outputs else '', f"[Tools] {', '.join(self.tools)}" if self.tools else '', f"[Preconditions] {', '.join(self.preconditions)}" if self.preconditions else '', f"[Procedure] {', '.join(self.procedure)}" if self.procedure else '', f"[Effects] {', '.join(self.effects)}" if self.effects else '', f"[FailureRecovery] {', '.join(self.failure_recovery)}" if self.failure_recovery else '', f"[ArgumentRoles] {', '.join(self.argument_roles)}" if self.argument_roles else '', f"[StateTransitions] {', '.join(self.state_transitions)}" if self.state_transitions else '', f"[Postconditions] {', '.join(self.postconditions)}" if self.postconditions else '', f"[ActionTemplates] {', '.join(self.action_templates)}" if self.action_templates else '', f"[SuccessChecks] {', '.join(self.success_checks)}" if self.success_checks else '', f"[FailureAvoidance] {', '.join(self.failure_avoidance)}" if self.failure_avoidance else '', f"[OrderingConstraints] {', '.join(self.ordering_constraints)}" if self.ordering_constraints else '', f"[TerminationConditions] {', '.join(self.termination_conditions)}" if self.termination_conditions else '', f"[ExcludedScope] {', '.join(self.excluded_scope)}" if self.excluded_scope else '', f"[Aliases] {', '.join(self.aliases)}" if self.aliases else '', f'[Evidence] {self.evidence}' if self.evidence else ''] if part))

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class MetaOpHit:
    metaop_id: str
    metaop_score: float
    rank: int
ComparisonStatus = Literal['SAME', 'COMPATIBLE', 'DIFFERENT', 'UNKNOWN']
SemanticRelation = Literal['EQUIVALENT', 'ALIAS', 'SPECIALIZATION', 'GENERALIZATION', 'RELATED', 'CONTRADICTORY', 'DISTINCT', 'UNCERTAIN']

@dataclass
class FieldComparison:
    status: str
    candidate_value: str = ''
    existing_value: str = ''
    explanation: str = ''
    candidate_evidence: list[str] = field(default_factory=list)
    existing_evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.status = clean_text(self.status).upper() or 'UNKNOWN'
        if self.status not in {'SAME', 'COMPATIBLE', 'DIFFERENT', 'UNKNOWN'}:
            self.status = 'UNKNOWN'
        self.candidate_value = clean_text(self.candidate_value)
        self.existing_value = clean_text(self.existing_value)
        self.explanation = clean_text(self.explanation)
        self.candidate_evidence = clean_list(self.candidate_evidence)
        self.existing_evidence = clean_list(self.existing_evidence)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class Counterexample:
    found: bool = False
    environment_state: list[str] = field(default_factory=list)
    candidate_result: str = ''
    existing_result: str = ''
    explanation: str = ''

    def __post_init__(self) -> None:
        self.found = bool(self.found)
        self.environment_state = clean_list(self.environment_state)
        self.candidate_result = clean_text(self.candidate_result)
        self.existing_result = clean_text(self.existing_result)
        self.explanation = clean_text(self.explanation)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class MetaOpComparison:
    candidate_local_id: str
    existing_metaop_id: str
    semantic_relation: str
    field_comparisons: dict[str, FieldComparison] = field(default_factory=dict)
    blocking_differences: list[str] = field(default_factory=list)
    counterexample: Counterexample = field(default_factory=Counterexample)
    merge_recommendation: bool = False
    confidence: float = 0.0
    reasoning_summary: str = ''

    def __post_init__(self) -> None:
        self.candidate_local_id = clean_text(self.candidate_local_id)
        self.existing_metaop_id = clean_text(self.existing_metaop_id)
        self.semantic_relation = clean_text(self.semantic_relation).upper() or 'UNCERTAIN'
        if self.semantic_relation not in {'EQUIVALENT', 'ALIAS', 'SPECIALIZATION', 'GENERALIZATION', 'RELATED', 'CONTRADICTORY', 'DISTINCT', 'UNCERTAIN'}:
            self.semantic_relation = 'UNCERTAIN'
        normalized_fields: dict[str, FieldComparison] = {}
        for key, value in (self.field_comparisons or {}).items():
            field_name = clean_text(key)
            if not field_name:
                continue
            if isinstance(value, FieldComparison):
                normalized_fields[field_name] = value
            elif isinstance(value, dict):
                normalized_fields[field_name] = FieldComparison(**value)
        self.field_comparisons = normalized_fields
        self.blocking_differences = clean_list(self.blocking_differences)
        if isinstance(self.counterexample, dict):
            self.counterexample = Counterexample(**self.counterexample)
        elif not isinstance(self.counterexample, Counterexample):
            self.counterexample = Counterexample()
        self.merge_recommendation = bool(self.merge_recommendation)
        try:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        except (TypeError, ValueError):
            self.confidence = 0.0
        self.reasoning_summary = clean_text(self.reasoning_summary)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class MergeDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        self.allowed = bool(self.allowed)
        self.reasons = clean_list(self.reasons)
        try:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        except (TypeError, ValueError):
            self.confidence = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class QueryOperationNeed:
    name: str
    action: str
    object: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    priority: int = 1

    def __post_init__(self) -> None:
        self.name = clean_text(self.name)
        self.action = clean_text(self.action).lower()
        self.object = clean_text(self.object)
        self.inputs = clean_list(self.inputs)
        self.outputs = clean_list(self.outputs)
        self.tools = clean_list(self.tools)
        self.constraints = clean_list(self.constraints)
        try:
            self.priority = int(self.priority)
        except (TypeError, ValueError):
            self.priority = 1

    def to_text(self) -> str:
        return ' '.join((part for part in [self.name, self.action, self.object, ' '.join(self.inputs), ' '.join(self.outputs), ' '.join(self.tools), ' '.join(self.constraints)] if part))

@dataclass
class QueryDecomposition:
    raw_query: str
    normalized_goal: str
    task_type: str = ''
    domain_objects: list[str] = field(default_factory=list)
    input_artifacts: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    required_operations: list[QueryOperationNeed] = field(default_factory=list)
    tool_hints: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    validation_signals: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.raw_query = clean_text(self.raw_query)
        self.normalized_goal = clean_text(self.normalized_goal) or self.raw_query
        self.task_type = clean_text(self.task_type)
        self.domain_objects = clean_list(self.domain_objects)
        self.input_artifacts = clean_list(self.input_artifacts)
        self.expected_outputs = clean_list(self.expected_outputs)
        self.tool_hints = clean_list(self.tool_hints)
        self.constraints = clean_list(self.constraints)
        self.validation_signals = clean_list(self.validation_signals)
        self.missing_information = clean_list(self.missing_information)
        operations: list[QueryOperationNeed] = []
        for item in self.required_operations:
            if isinstance(item, QueryOperationNeed):
                operations.append(item)
            elif isinstance(item, dict):
                operations.append(QueryOperationNeed(**item))
        self.required_operations = operations

    def to_dense_query_text(self) -> str:
        return '\n'.join((part for part in [f'Goal: {self.normalized_goal}', f'Task type: {self.task_type}' if self.task_type else '', f"Objects: {', '.join(self.domain_objects)}" if self.domain_objects else '', f"Inputs: {', '.join(self.input_artifacts)}" if self.input_artifacts else '', f"Outputs: {', '.join(self.expected_outputs)}" if self.expected_outputs else '', f"Tools: {', '.join(self.tool_hints)}" if self.tool_hints else '', f"Operations: {' | '.join((item.to_text() for item in self.required_operations))}" if self.required_operations else '', f"Constraints: {', '.join(self.constraints)}" if self.constraints else '', f"Validation: {', '.join(self.validation_signals)}" if self.validation_signals else ''] if part))

    def to_lexical_query_text(self) -> str:
        return ' '.join((part for part in [self.raw_query, self.normalized_goal, self.task_type, ' '.join(self.domain_objects), ' '.join(self.input_artifacts), ' '.join(self.expected_outputs), ' '.join(self.tool_hints), ' '.join(self.constraints), ' '.join((item.to_text() for item in self.required_operations))] if part))

    def operation_texts(self) -> list[str]:
        return [item.to_text() for item in sorted(self.required_operations, key=lambda row: row.priority)]

    @property
    def goal(self) -> str:
        return self.normalized_goal

    @property
    def objects(self) -> list[str]:
        return self.domain_objects

    @property
    def operations(self) -> list[str]:
        return [item.action for item in sorted(self.required_operations, key=lambda row: row.priority)]

    @property
    def success_signals(self) -> list[str]:
        return self.validation_signals

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class MetaOpEdge:
    edge_id: str
    source_metaop_id: str
    target_metaop_id: str
    edge_type: str
    score: float
    reason: str
    evidence: str = ''
    source_skill_id: str = ''
    support_count: int = 1
    source_instance_id: str = ''
    target_instance_id: str = ''

    def __post_init__(self) -> None:
        self.edge_id = clean_text(self.edge_id)
        self.source_metaop_id = clean_text(self.source_metaop_id)
        self.target_metaop_id = clean_text(self.target_metaop_id)
        self.edge_type = clean_text(self.edge_type).lower()
        try:
            self.score = float(self.score)
        except (TypeError, ValueError):
            self.score = 0.0
        self.reason = clean_text(self.reason)
        self.evidence = clean_text(self.evidence)
        self.source_skill_id = clean_text(self.source_skill_id)
        self.source_instance_id = clean_text(self.source_instance_id)
        self.target_instance_id = clean_text(self.target_instance_id)
        try:
            self.support_count = max(1, int(self.support_count))
        except (TypeError, ValueError):
            self.support_count = 1

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ExecutableMetaOp:
    metaop_id: str
    name: str
    definition: str
    action: str
    object: str
    inputs: list[str]
    outputs: list[str]
    tools: list[str]
    preconditions: list[str]
    postconditions: list[str]
    evidence: str
    depends_on: list[str]
    procedure: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    failure_recovery: list[str] = field(default_factory=list)
    argument_roles: list[str] = field(default_factory=list)
    state_transitions: list[str] = field(default_factory=list)
    action_templates: list[str] = field(default_factory=list)
    success_checks: list[str] = field(default_factory=list)
    failure_avoidance: list[str] = field(default_factory=list)
    ordering_constraints: list[str] = field(default_factory=list)
    termination_conditions: list[str] = field(default_factory=list)
    excluded_scope: list[str] = field(default_factory=list)
    evidence_spans: dict[str, list[str]] = field(default_factory=dict)
    residual_constraints: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    status: str = 'pending'
    instance_id: str = ''
    bindings: dict[str, str] = field(default_factory=dict)
    unresolved_roles: list[str] = field(default_factory=list)
    instance_ordinal: int = 1
    instance_count: int = 1

    def __post_init__(self) -> None:
        self.metaop_id = clean_text(self.metaop_id)
        self.name = clean_text(self.name)
        self.definition = clean_text(self.definition)
        self.action = clean_text(self.action).lower()
        self.object = clean_text(self.object)
        self.inputs = clean_list(self.inputs)
        self.outputs = clean_list(self.outputs)
        self.tools = clean_list(self.tools)
        self.preconditions = clean_list(self.preconditions)
        self.postconditions = clean_list(self.postconditions)
        self.evidence = clean_text(self.evidence)
        self.depends_on = clean_list(self.depends_on)
        self.procedure = clean_list(self.procedure)
        self.effects = clean_list(self.effects)
        self.failure_recovery = clean_list(self.failure_recovery)
        self.argument_roles = clean_list(self.argument_roles)
        self.state_transitions = clean_list(self.state_transitions)
        self.action_templates = clean_list(self.action_templates)
        self.success_checks = clean_list(self.success_checks)
        self.failure_avoidance = clean_list(self.failure_avoidance)
        self.ordering_constraints = clean_list(self.ordering_constraints)
        self.termination_conditions = clean_list(self.termination_conditions)
        self.excluded_scope = clean_list(self.excluded_scope)
        self.evidence_spans = clean_dict_lists(self.evidence_spans)
        self.residual_constraints = clean_list(self.residual_constraints)
        self.quality_flags = clean_list(self.quality_flags)
        self.status = clean_text(self.status) or 'pending'
        self.instance_id = clean_text(self.instance_id)
        self.bindings = {clean_text(key): clean_text(value) for key, value in self.bindings.items() if clean_text(key) and clean_text(value)}
        self.unresolved_roles = clean_list(self.unresolved_roles)
        self.instance_ordinal = max(1, int(self.instance_ordinal))
        self.instance_count = max(1, int(self.instance_count))
        if not self.procedure:
            self.procedure = clean_list(self.action_templates)
        if not self.effects:
            self.effects = clean_list([*self.state_transitions, *self.postconditions, *self.outputs])
        if not self.failure_recovery:
            self.failure_recovery = clean_list(self.failure_avoidance)
        if not self.failure_avoidance:
            self.failure_avoidance = clean_list(self.failure_recovery)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class MetaOpDAG:
    query: str
    query_decomposition: QueryDecomposition
    selected_metaops: list[ExecutableMetaOp]
    edges: list[MetaOpEdge]
    execution_order: list[str]
    missing_operations: list[str]
    rendered_context: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
