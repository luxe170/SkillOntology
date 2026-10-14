from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any
from skillontology.runtime.agent_cache import AgentCallCache
from skillontology.ontology.domain_ontology import ExtractionPolicy
from skillontology.runtime.llm_json import call_json, extract_json_object
from skillontology.construction.prompts import AGENT1_ONTOLOGY_GUIDANCE, AGENT1_SYSTEM, AGENT1_USER
from skillontology.schema import CandidateMetaOp, ResidualConstraint, SkillRecord, clean_list, clean_text, evidence_summary_from_spans
AGENT1_RESPONSE_PARSER = extract_json_object
_ACTION_VERBS = ['add', 'aggregate', 'align', 'analyze', 'apply', 'associate', 'build', 'calculate', 'check', 'classify', 'close', 'combine', 'compile', 'compress', 'compute', 'configure', 'construct', 'convert', 'create', 'decompress', 'define', 'detect', 'determine', 'discover', 'evaluate', 'execute', 'examine', 'export', 'extract', 'filter', 'find', 'generate', 'go', 'grab', 'heat', 'impute', 'initialize', 'install', 'instantiate', 'load', 'locate', 'look', 'merge', 'migrate', 'modify', 'move', 'navigate', 'normalize', 'open', 'parse', 'pick', 'pickup', 'place', 'precompute', 'propose', 'put', 'rank', 'read', 'remove', 'render', 'replace', 'repeat', 'resample', 'retrieve', 'retry', 'run', 'scale', 'scan', 'search', 'segment', 'select', 'split', 'standardize', 'store', 'summarize', 'take', 'taper', 'toggle', 'train', 'transcribe', 'transform', 'turn', 'update', 'validate', 'verify', 'warp', 'write', 'clean', 'cool']
_ACTION_VERB_SET = set(_ACTION_VERBS)
_ACTION_ALIASES = {'setup': 'set up', 'remove_outliers': 'remove'}
_OVER_GENERIC_ACTIONS = {'use', 'handle', 'process', 'perform'}
_OVER_GENERIC_NAMES = {'HandleObject', 'ProcessObject', 'UseTool', 'ChangeState', 'InteractWithObject', 'PerformOperation'}
_LITERAL_HEADING_PREFIXES = ('allowed tools', 'use when', 'what to do', 'overview', 'prerequisites', 'requirements', 'background', 'instructions', 'capability', 'capabilities')
_META_INSTRUCTION_PATTERNS = ('allowed tools', 'available tools', 'tool permission', 'tool permissions', 'use when', 'this skill', 'skill assumes', 'capability', 'overview', 'prerequisites', 'background', 'instructions', 'should be used when', 'when you need', 'user wants to', 'do not include', 'do not use', 'always search', 'always create', 'find it and run', 'simply run', 'ready-to-run scripts')
_GENERIC_FLAG_ACTIONS = {'use', 'read', 'run', 'validate', 'check', 'ensure', 'follow', 'inspect'}
_SECONDARY_VERBS = {'analyze', 'build', 'classify', 'compute', 'convert', 'create', 'detect', 'edit', 'evaluate', 'execute', 'examine', 'export', 'extract', 'generate', 'go', 'heat', 'inspect', 'load', 'locate', 'look', 'measure', 'move', 'navigate', 'parse', 'place', 'put', 'rank', 'read', 'repeat', 'retry', 'render', 'resolve', 'run', 'scan', 'search', 'solve', 'take', 'train', 'validate', 'write', 'clean', 'cool'}
_NON_OPERATION_PREFIXES = ('always ', 'allowed tools', 'available tools', 'background', 'capability', 'capabilities', 'do not ', 'find it and run', 'for each ', 'if the ', 'includes ready-to-run', 'instructions', 'key point', 'overview', 'prerequisites', 'requirements', 'return json', 'simply run', 'this skill', 'use when', 'when to use', 'when you need')
_EXPLANATORY_STARTS = ('a .docx file is', "maven's dependency management is", 'reduces cognitive load', 'the following')
_EXECUTABLE_RESULT_FIELDS = ('outputs', 'state_transitions', 'postconditions', 'success_checks', 'termination_conditions', 'effects')
_EVIDENCE_MAX_CHARS = 500
_ALFWORLD_STATE_ACTIONS = {'heat', 'cool', 'clean'}
_ALFWORLD_ACTIONS = {'navigate', 'go', 'take', 'move', 'put', 'place', 'open', 'close', 'heat', 'cool', 'clean', 'look', 'locate', 'examine', 'repeat', 'retry', 'scan', 'turn', 'toggle', 'use'}
_ALFWORLD_TOOLS_BY_ACTION = {'heat': ['ALFWorld environment', 'microwave', 'stoveburner'], 'cool': ['ALFWorld environment', 'fridge'], 'clean': ['ALFWorld environment', 'sinkbasin'], 'navigate': ['ALFWorld environment'], 'go': ['ALFWorld environment'], 'take': ['ALFWorld environment'], 'move': ['ALFWorld environment'], 'put': ['ALFWorld environment'], 'place': ['ALFWorld environment'], 'open': ['ALFWorld environment'], 'close': ['ALFWorld environment'], 'look': ['ALFWorld environment', 'desklamp', 'floorlamp'], 'examine': ['ALFWorld environment', 'desklamp', 'floorlamp'], 'turn': ['ALFWorld environment', 'desklamp', 'floorlamp'], 'toggle': ['ALFWorld environment', 'desklamp', 'floorlamp']}

@dataclass
class MetaOpExtractionResult:
    metaops: list[CandidateMetaOp] = field(default_factory=list)
    residual_constraints: list[ResidualConstraint] = field(default_factory=list)
    coverage_notes: list[dict[str, Any]] = field(default_factory=list)
    audit_records: list[dict[str, Any]] = field(default_factory=list)

def _apply_extraction_policy(result: MetaOpExtractionResult, policy: ExtractionPolicy | None, *, agent_mode: str, cache_path: str='', agent_status: str='not_used') -> MetaOpExtractionResult:
    for candidate in result.metaops:
        rule_ids = policy.normalize_candidate(candidate) if policy is not None else []
        if policy is not None:
            candidate.quality_flags = clean_list([*candidate.quality_flags, 'ontology_domain_scoped'])
        validation = policy.validate_candidate(candidate, rule_ids) if policy is not None else {'valid': bool(candidate.action and candidate.object and (candidate.evidence or candidate.evidence_spans)), 'errors': []}
        result.audit_records.append({'candidate_local_id': candidate.local_id, 'source_skill_id': candidate.source_skill_id, 'agent_mode': agent_mode, 'agent_status': agent_status, 'ontology_guided': policy is not None, 'policy_sha256': policy.policy_sha256 if policy is not None else '', 'policy_rule_ids': rule_ids, 'validation': validation, 'skill_evidence': {'summary': candidate.evidence, 'spans': candidate.evidence_spans}, 'response_cache_path': cache_path})
    return result

def validate_candidate_extraction(candidate: CandidateMetaOp) -> list[str]:
    errors: list[str] = []
    if not candidate.action:
        errors.append('missing_action')
    if not candidate.object:
        errors.append('missing_object')
    if candidate.name in _OVER_GENERIC_NAMES:
        errors.append('over_generic_name')
    if _looks_like_non_operation(candidate):
        errors.append('non_operation_text')
    if _object_repeats_action(candidate):
        errors.append('object_repeats_action')
    rich_fields = [*candidate.inputs, *candidate.outputs, *candidate.procedure, *candidate.effects, *candidate.failure_recovery, *candidate.state_transitions, *candidate.postconditions, *candidate.success_checks, *candidate.termination_conditions]
    if candidate.action in _OVER_GENERIC_ACTIONS and len(candidate.object.split()) <= 1 and (not rich_fields):
        errors.append('over_generic_action')
    if _missing_executable_result(candidate):
        errors.append('missing_executable_result')
    return errors

def _has_literal_heading(text: str) -> bool:
    lowered = clean_text(text).lower().strip(' .:;-')
    if not lowered:
        return False
    if any((lowered.startswith(prefix) for prefix in _LITERAL_HEADING_PREFIXES)):
        return True
    return any((re.match(f'^(?:use|read|run|search|validate|execute)\\s+{re.escape(prefix)}\\b', lowered) for prefix in _LITERAL_HEADING_PREFIXES))

def _strip_markdown_prefix(text: str) -> str:
    value = clean_text(text)
    value = re.sub('^(?:#+\\s*)+', '', value).strip()
    value = re.sub('^(?:[-*]\\s*)+', '', value).strip()
    value = re.sub('^\\d+[.)]\\s*', '', value).strip()
    value = re.sub('^\\*\\*(.+?)\\*\\*', '\\1', value).strip()
    value = re.sub('\\s+\\(from section [\'\\"]?#+\\s*[^)]*\\)\\s*$', '', value, flags=re.IGNORECASE).strip()
    return value.strip(' :;.-')

def _normalize_action_text(action: str) -> str:
    value = _strip_markdown_prefix(action).lower()
    value = _ACTION_ALIASES.get(value, value)
    value = re.sub('[^a-z0-9_+ -]', ' ', value)
    value = re.sub('\\s+', ' ', value).strip()
    if not value:
        return ''
    if value in _ACTION_ALIASES:
        return _ACTION_ALIASES[value]
    tokens = value.replace('_', ' ').split()
    if not tokens:
        return ''
    if len(tokens) == 1:
        return _ACTION_ALIASES.get(tokens[0], tokens[0])
    if tokens[0] in _ACTION_VERB_SET:
        return _ACTION_ALIASES.get(tokens[0], tokens[0])
    for token in tokens:
        if token in _ACTION_VERB_SET:
            return _ACTION_ALIASES.get(token, token)
    return tokens[0]

def _action_object_remainder(raw_action: str, normalized_action: str) -> str:
    value = _strip_markdown_prefix(raw_action).lower()
    value = re.sub('[^a-z0-9_+ -]', ' ', value)
    tokens = value.replace('_', ' ').split()
    normalized_tokens = normalized_action.split()
    if not tokens or not normalized_tokens:
        return ''
    if tokens[:len(normalized_tokens)] == normalized_tokens:
        return ' '.join(tokens[len(normalized_tokens):])
    if tokens[0] in _ACTION_ALIASES and _ACTION_ALIASES[tokens[0]].split() == normalized_tokens:
        return ' '.join(tokens[1:])
    for index, token in enumerate(tokens):
        if token == normalized_tokens[0]:
            return ' '.join(tokens[:index] + tokens[index + len(normalized_tokens):])
    return ''

def _prepend_action_remainder_to_object(action: str, remainder: str, obj: str) -> str:
    remainder = clean_text(remainder)
    obj = clean_text(obj)
    if not remainder:
        return obj
    if not obj:
        return remainder
    if obj.lower().startswith(remainder.lower()):
        return obj
    if action == 'extract':
        return f'{remainder} from {obj}'
    if action == 'convert':
        return f'{remainder} to {obj}'
    if action == 'evaluate' and remainder == 'grid search':
        return f'{obj} by grid search'
    return f'{remainder} {obj}'

def _normalize_action_and_object(raw_action: str, raw_object: str) -> tuple[str, str]:
    action = _normalize_action_text(raw_action)
    obj = _strip_leading_action_from_object(action, raw_object)
    remainder = _action_object_remainder(raw_action, action)
    obj = _prepend_action_remainder_to_object(action, remainder, obj)
    obj = _strip_leading_action_from_object(action, obj)
    return (action, obj)

def _strip_leading_action_from_object(action: str, obj: str) -> str:
    value = _strip_markdown_prefix(obj)
    if not action or not value:
        return value
    escaped = re.escape(action)
    value = re.sub(f'(?i)^{escaped}\\b\\s*', '', value).strip()
    value = re.sub(f"(?i)^and\\s+(?:{'|'.join(sorted(_SECONDARY_VERBS))})\\b\\s*", '', value).strip()
    value = re.sub('(?i)^to\\s+', '', value).strip()
    value = re.sub('(?i)^(?:a|an|the)\\s+', '', value).strip()
    return value.strip(' :;.-')

def _object_repeats_action(candidate: CandidateMetaOp) -> bool:
    action = clean_text(candidate.action).lower()
    obj = clean_text(candidate.object).lower()
    return bool(action and obj.startswith(f'{action} '))

def _combined_text(candidate: CandidateMetaOp) -> str:
    return ' '.join([candidate.name, candidate.definition, candidate.action, candidate.object, candidate.evidence]).lower()

def _is_non_operation_phrase(text: str) -> bool:
    lowered = _strip_markdown_prefix(text).lower().strip(' .:;-')
    if not lowered:
        return False
    if any((lowered.startswith(prefix) for prefix in _NON_OPERATION_PREFIXES)):
        return True
    if any((lowered.startswith(prefix) for prefix in _EXPLANATORY_STARTS)):
        return True
    return False

def _looks_like_non_operation(candidate: CandidateMetaOp) -> bool:
    if _is_non_operation_phrase(candidate.object):
        return True
    if _is_non_operation_phrase(candidate.name) and (not _has_execution_result(candidate)):
        return True
    combined = _combined_text(candidate)
    if any((pattern in combined for pattern in _META_INSTRUCTION_PATTERNS)):
        if not _has_execution_result(candidate):
            return True
    return False

def _has_execution_result(candidate: CandidateMetaOp) -> bool:
    return any((getattr(candidate, field_name) for field_name in _EXECUTABLE_RESULT_FIELDS))

def _missing_executable_result(candidate: CandidateMetaOp) -> bool:
    if _has_execution_result(candidate):
        return False
    if candidate.inputs and candidate.outputs:
        return False
    return True

def _source_evidence_units(skill: SkillRecord) -> list[str]:
    text = '\n'.join((part for part in [skill.raw_content, skill.rendered_snippet, skill.description, skill.one_line_capability] if part))
    units: list[str] = []
    for line in text.splitlines():
        stripped = _strip_markdown_prefix(line)
        if not stripped or _is_non_operation_phrase(stripped):
            continue
        if _looks_like_source_heading_unit(stripped):
            continue
        if len(stripped) <= _EVIDENCE_MAX_CHARS:
            units.append(stripped)
            continue
        units.extend((_strip_markdown_prefix(part) for part in re.split('(?<=[.!?。！？])\\s+', stripped) if _strip_markdown_prefix(part)))
    seen: set[str] = set()
    deduped: list[str] = []
    for unit in units:
        key = unit.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(unit[:_EVIDENCE_MAX_CHARS])
    return deduped

def _has_evidence_execution_cue(text: str) -> bool:
    value = clean_text(text)
    lowered = value.lower()
    if re.search('(?:[=()[\\]{}]|```|`|>>>|--)', value):
        return True
    if re.search('\\b(?:pip|python|python3|pytest|make|gh|git|sqlite3|ffmpeg|RUBE_[A-Z_]+)\\b', value):
        return True
    if re.search('\\b(?:run|call|use|write|read|load|save|create|compute|execute|parse|extract|validate)\\b', lowered):
        return True
    if re.search('\\b(?:go to|take|move|open|close|heat|cool|clean|look|examine|turn on|locate|search|scan|repeat|retry)\\b', lowered):
        return True
    return False

def _looks_like_source_heading_unit(text: str) -> bool:
    value = clean_text(text).strip(' |')
    if not value:
        return True
    if _has_evidence_execution_cue(value):
        return False
    tokens = re.findall('[A-Za-z0-9_+-]+', value)
    if len(tokens) <= 4 and (not re.search('[.:;,]', value)):
        return True
    return False

def _evidence_tokens(candidate: CandidateMetaOp) -> set[str]:
    return {token for token in re.findall('[a-z0-9_+-]+', ' '.join([candidate.name, candidate.definition, candidate.action, candidate.object, ' '.join(candidate.inputs), ' '.join(candidate.outputs), ' '.join(candidate.tools), ' '.join(candidate.procedure), ' '.join(candidate.effects), ' '.join(candidate.failure_recovery), ' '.join(candidate.action_templates), ' '.join(candidate.postconditions), ' '.join(candidate.success_checks), ' '.join(candidate.termination_conditions)]).lower()) if len(token) >= 3}

def _score_evidence_unit(candidate: CandidateMetaOp, unit: str) -> tuple[int, int, int, int]:
    lowered = unit.lower()
    tokens = set(re.findall('[a-z0-9_+-]+', lowered))
    overlap = len(tokens & _evidence_tokens(candidate))
    action_hit = int(bool(candidate.action and re.search(f'\\b{re.escape(candidate.action)}\\b', lowered)))
    object_tokens = {token for token in re.findall('[a-z0-9_+-]+', candidate.object.lower()) if len(token) >= 3}
    object_hits = len(tokens & object_tokens)
    execution_cue = int(_has_evidence_execution_cue(unit))
    return (overlap + action_hit + object_hits + execution_cue, object_hits, execution_cue, -len(unit))

def _looks_like_span_summary_evidence(candidate: CandidateMetaOp) -> bool:
    generated = evidence_summary_from_spans(candidate.evidence_spans)
    return bool(candidate.evidence and generated and (candidate.evidence == generated))

def _looks_like_weak_evidence_text(text: str) -> bool:
    value = clean_text(text)
    lowered = value.lower()
    if not value:
        return True
    if _has_evidence_execution_cue(value) and (not value.endswith('=')):
        return False
    if lowered in {'resources', 'inheritance', 'solve the problem'}:
        return True
    if value.endswith('='):
        return True
    tokens = re.findall('[A-Za-z0-9_+-]+', value)
    if len(tokens) <= 4 and (not re.search('[.:;,]', value)):
        return True
    if re.fullmatch('(?:action|object|inputs|outputs|tools|preconditions|postconditions|operation):\\s*[A-Za-z]', value):
        return True
    if re.fullmatch('(?:outputs|tools|operation):\\s*[A-Za-z0-9_ -]{1,12}', value) and len(value.split()) <= 3:
        return True
    return False

def _fill_missing_evidence(candidate: CandidateMetaOp, skill: SkillRecord) -> None:
    units = _source_evidence_units(skill)
    if not units:
        return
    best = max(units, key=lambda unit: _score_evidence_unit(candidate, unit))
    best_score = _score_evidence_unit(candidate, best)[0]
    if best_score <= 0:
        return
    if candidate.evidence and (not (_looks_like_span_summary_evidence(candidate) or _looks_like_weak_evidence_text(candidate.evidence))):
        return
    candidate.evidence = best[:_EVIDENCE_MAX_CHARS]
    spans = dict(candidate.evidence_spans)
    spans['operation'] = clean_list([*spans.get('operation', []), candidate.evidence])
    candidate.evidence_spans = spans

def _normalize_candidate_operation(candidate: CandidateMetaOp, skill: SkillRecord) -> CandidateMetaOp:
    candidate.action, candidate.object = _normalize_action_and_object(candidate.action, candidate.object)
    if candidate.evidence:
        candidate.evidence = _strip_markdown_prefix(candidate.evidence)
    if candidate.definition:
        candidate.definition = _strip_markdown_prefix(candidate.definition)
    _fill_missing_evidence(candidate, skill)
    _backfill_execution_fields(candidate)
    if candidate.action and candidate.object:
        candidate.name = f'{candidate.action} {candidate.object}'[:120]
    elif candidate.action:
        candidate.name = candidate.action
    if not candidate.definition and candidate.action and candidate.object:
        candidate.definition = f'{candidate.action} {candidate.object}'
    candidate.source_skill_id = skill.skill_id
    return candidate

def _backfill_execution_fields(candidate: CandidateMetaOp) -> None:
    if not candidate.procedure:
        if candidate.action_templates:
            candidate.procedure = candidate.action_templates[:]
        else:
            template = clean_text(candidate.evidence or candidate.definition)
            if template and _has_evidence_execution_cue(template):
                candidate.procedure = [template[:_EVIDENCE_MAX_CHARS]]
            elif candidate.action and candidate.object:
                candidate.procedure = [f'{candidate.action} {candidate.object}']
    if not candidate.effects:
        candidate.effects = clean_list([*candidate.state_transitions, *candidate.postconditions, *candidate.outputs])
    if not candidate.failure_recovery:
        candidate.failure_recovery = clean_list(candidate.failure_avoidance)
    if not candidate.failure_avoidance:
        candidate.failure_avoidance = clean_list(candidate.failure_recovery)
    if not candidate.action_templates:
        template = clean_text(candidate.evidence or candidate.definition)
        if template and _has_evidence_execution_cue(template):
            candidate.action_templates = [template[:_EVIDENCE_MAX_CHARS]]
        elif candidate.action and candidate.object:
            candidate.action_templates = [f'{candidate.action} {candidate.object}']
    if not candidate.success_checks:
        if candidate.termination_conditions:
            candidate.success_checks = candidate.termination_conditions[:2]
        elif candidate.postconditions:
            candidate.success_checks = candidate.postconditions[:2]
        elif candidate.outputs:
            candidate.success_checks = [f'expected output is available: {item}' for item in candidate.outputs[:2]]

def infer_quality_flags(candidate: CandidateMetaOp) -> list[str]:
    flags: list[str] = []
    combined = _combined_text(candidate)
    if _has_literal_heading(candidate.name) or _has_literal_heading(candidate.object):
        flags.append('literal_heading')
    if any((pattern in combined for pattern in _META_INSTRUCTION_PATTERNS)) and (not _has_execution_result(candidate)):
        flags.append('meta_instruction')
    if _looks_like_non_operation(candidate):
        flags.append('non_operation_text')
    if _object_repeats_action(candidate):
        flags.append('object_repeats_action')
    if candidate.action in _GENERIC_FLAG_ACTIONS and (not candidate.inputs) and (not candidate.outputs) and (not candidate.effects) and (not candidate.state_transitions) and (not candidate.postconditions) and (not candidate.success_checks) and (not candidate.termination_conditions):
        flags.append('generic_action')
    if not candidate.outputs and (not candidate.effects) and (not candidate.state_transitions) and (not candidate.postconditions) and (not candidate.success_checks) and (not candidate.termination_conditions):
        flags.append('missing_state_or_output')
    if _missing_executable_result(candidate):
        flags.append('missing_executable_result')
    if not candidate.action_templates:
        flags.append('missing_action_template')
    if not (candidate.success_checks or candidate.termination_conditions or candidate.postconditions):
        flags.append('missing_success_check')
    if not candidate.evidence and (not candidate.evidence_spans):
        flags.append('weak_evidence')
    return clean_list(flags)

def _should_move_to_residual(candidate: CandidateMetaOp) -> bool:
    return False

def _should_drop_non_operation(candidate: CandidateMetaOp) -> bool:
    flags = set(candidate.quality_flags)
    if 'non_operation_text' in flags:
        return True
    if {'literal_heading', 'meta_instruction'}.intersection(flags):
        return True
    if 'missing_executable_result' in flags:
        return True
    if 'generic_action' in flags and (not (candidate.inputs and candidate.outputs)):
        return True
    return False

def _candidate_residual_kind(candidate: CandidateMetaOp) -> str:
    text = ' '.join([candidate.name, candidate.definition, candidate.object, candidate.evidence]).lower()
    if 'allowed tools' in text or 'available tools' in text or 'tool permission' in text:
        return 'tool_policy'
    if 'prerequisite' in text or 'requirement' in text:
        return 'precondition'
    return 'usage_note'

def _candidate_residual_text(candidate: CandidateMetaOp) -> str:
    for value in (candidate.evidence, candidate.definition, candidate.name, candidate.object):
        text = clean_text(value)
        if text:
            return text
    return 'Non-executable skill note preserved from Agent 1 output.'

def _allowed_tools_residual_row(skill: SkillRecord) -> dict[str, Any] | None:
    if skill.allowed_tools:
        tools = ', '.join(skill.allowed_tools)
        return {'kind': 'tool_policy', 'text': f'This skill assumes these tools are available: {tools}.', 'applies_to': [], 'evidence': f'Allowed tools: {tools}'}
    text = '\n'.join([skill.rendered_snippet, skill.raw_content[:3000]])
    match = re.search('(?im)^\\s*(?:#+\\s*)?Allowed tools\\s*:?\\s*(.+)$', text)
    if not match:
        return None
    evidence = clean_text(match.group(0))
    return {'kind': 'tool_policy', 'text': f'This skill includes a tool availability policy: {clean_text(match.group(1))}.', 'applies_to': [], 'evidence': evidence}

def _normalize_coverage_notes(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    notes: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            note = {clean_text(key): value for key, value in row.items() if clean_text(key)}
        else:
            note = {'source_span': clean_text(row), 'status': 'covered'}
        if note:
            notes.append(note)
    return notes

def _normalize_residual_constraints(skill_id: str, rows: Any) -> list[ResidualConstraint]:
    if not isinstance(rows, list):
        return []
    constraints: list[ResidualConstraint] = []
    seen_keys: set[tuple[str, str, str, tuple[str, ...]]] = set()
    seen_ids: set[str] = set()
    next_index = 1
    for row in rows:
        if isinstance(row, ResidualConstraint):
            payload = row.to_json_dict()
        elif isinstance(row, dict):
            payload = dict(row)
        else:
            payload = {'kind': 'other', 'text': clean_text(row)}
        text = clean_text(payload.get('text') or payload.get('source_span') or payload.get('evidence'))
        evidence = clean_text(payload.get('evidence') or payload.get('source_span'))
        kind = clean_text(payload.get('kind') or payload.get('target') or 'other').lower()
        applies_to = clean_list(payload.get('applies_to', []))
        if not text:
            continue
        dedupe_key = (kind, text.lower(), evidence.lower(), tuple(applies_to))
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        raw_id = clean_text(payload.get('constraint_id'))
        constraint_id = raw_id
        while not constraint_id or constraint_id in seen_ids:
            constraint_id = f'{skill_id}::R{next_index:03d}'
            next_index += 1
        seen_ids.add(constraint_id)
        constraints.append(ResidualConstraint(constraint_id=constraint_id, source_skill_id=payload.get('source_skill_id') or skill_id, kind=kind, text=text, applies_to=applies_to, evidence=evidence))
    return constraints

def _is_alfworld_skill(skill: SkillRecord) -> bool:
    text = ' '.join([skill.skill_id, skill.name, skill.source_path, skill.description, skill.one_line_capability, ' '.join(skill.domain_tags)]).lower()
    return 'alfworld' in text

def _alfworld_source_text(skill: SkillRecord) -> str:
    return '\n'.join((part for part in [skill.raw_content, skill.rendered_snippet, skill.description, skill.one_line_capability, '\n'.join(skill.example_tasks)] if part))

def _alfworld_find_evidence(skill: SkillRecord, action: str, default: str) -> str:
    pattern = re.compile(f'\\b(?:go to|turn on|{re.escape(action)})\\b', re.IGNORECASE)
    for line in _alfworld_source_text(skill).splitlines():
        stripped = _strip_markdown_prefix(line)
        if not stripped:
            continue
        if pattern.search(stripped):
            return stripped[:_EVIDENCE_MAX_CHARS]
    return default[:_EVIDENCE_MAX_CHARS]

def _alfworld_has_template(text: str, action: str) -> bool:
    lowered = text.lower()
    if action in _ALFWORLD_STATE_ACTIONS:
        return bool(re.search(f'`?\\s*{re.escape(action)}\\s+\\{{?(?:object|obj|held_object|target_object)\\b', lowered) or re.search('`?\\s*\\{action\\}\\s+\\{?(?:object|obj)\\b', lowered))
    if action == 'navigate':
        return bool(re.search('`?\\s*go to\\s+\\{?(?:recep|receptacle|appliance|target)', lowered))
    if action == 'take':
        return bool(re.search('`?\\s*take\\s+\\{?(?:object|obj|target_object)', lowered))
    if action in {'move', 'put', 'place'}:
        return bool(re.search('`?\\s*move\\s+\\{?(?:object|obj|held_object|target_object)\\b.*\\bto\\s+\\{?', lowered) or re.search('`?\\s*put\\s+\\{?(?:object|obj|held_object|target_object)\\b', lowered))
    if action in {'open', 'close'}:
        return bool(re.search(f'`?\\s*{re.escape(action)}\\s+\\{{?(?:recep|receptacle|appliance|target)', lowered))
    if action in {'look', 'examine'}:
        return bool(re.search('`?\\s*(?:look at|examine)\\s+\\{?(?:object|obj|target_object)', lowered))
    return False

def _alfworld_state_action_supported(skill: SkillRecord, action: str, text: str) -> bool:
    name = f' {skill.name.lower()} {skill.skill_id.lower()} '
    if action == 'heat' and any((token in name for token in ['object-heater', 'state-modifier'])):
        return True
    if action == 'cool' and any((token in name for token in ['object-cooler', 'state-modifier'])):
        return True
    if action == 'clean' and any((token in name for token in ['clean-object', 'object-cleaner', 'state-modifier'])):
        return True
    return _alfworld_has_template(text, action)

def _alfworld_candidate(*, skill: SkillRecord, action: str, obj: str, definition: str, inputs: list[str], outputs: list[str], tools: list[str], preconditions: list[str], state_transitions: list[str], postconditions: list[str], action_templates: list[str], success_checks: list[str], failure_avoidance: list[str], ordering_constraints: list[str], evidence_default: str, step_order: int) -> CandidateMetaOp:
    evidence = _alfworld_find_evidence(skill, action, evidence_default)
    return CandidateMetaOp(name=f'{action} {obj}', definition=definition, action=action, object=obj, inputs=inputs, outputs=outputs, tools=tools, preconditions=preconditions, argument_roles=[], state_transitions=state_transitions, postconditions=postconditions, action_templates=action_templates, success_checks=success_checks, failure_avoidance=failure_avoidance, ordering_constraints=ordering_constraints, termination_conditions=success_checks[:1], evidence_spans={'action_templates': action_templates, 'postconditions': postconditions, 'success_checks': success_checks, 'operation': [evidence]}, unknown_fields=[], step_order=step_order, evidence=evidence, source_skill_id=skill.skill_id)

def _alfworld_deterministic_candidate_metaops(skill: SkillRecord) -> list[CandidateMetaOp]:
    if not _is_alfworld_skill(skill):
        return []
    text = _alfworld_source_text(skill)
    lowered = text.lower()
    candidates: list[CandidateMetaOp] = []

    def add(candidate: CandidateMetaOp) -> None:
        key = (candidate.action, candidate.object.lower(), tuple(candidate.action_templates))
        existing_keys = {(item.action, item.object.lower(), tuple(item.action_templates)) for item in candidates}
        if key not in existing_keys:
            candidates.append(candidate)
    if _alfworld_has_template(lowered, 'navigate') or 'navigate' in lowered or 'go to' in lowered:
        add(_alfworld_candidate(skill=skill, action='navigate', obj='target receptacle or appliance', definition='Move the ALFWorld agent to the target receptacle or appliance.', inputs=['target_receptacle_or_appliance'], outputs=['agent at target location'], tools=_ALFWORLD_TOOLS_BY_ACTION['navigate'], preconditions=['target receptacle or appliance is known or visible'], state_transitions=['agent_location -> target location'], postconditions=['agent is near the target receptacle or appliance'], action_templates=['go to {receptacle_or_appliance}'], success_checks=['observation confirms arrival at the target location'], failure_avoidance=['use the exact visible receptacle or appliance identifier'], ordering_constraints=['navigate before taking, placing, heating, cooling, or cleaning at that location'], evidence_default='Use `go to {recep}` to navigate to the target receptacle or appliance.', step_order=len(candidates) + 1))
    if _alfworld_has_template(lowered, 'open') or 'open {' in lowered or ' open ' in lowered:
        add(_alfworld_candidate(skill=skill, action='open', obj='closed receptacle or appliance', definition='Open a closed ALFWorld receptacle or appliance before interacting with its contents.', inputs=['closed_receptacle_or_appliance'], outputs=['opened receptacle or appliance'], tools=_ALFWORLD_TOOLS_BY_ACTION['open'], preconditions=['target receptacle or appliance is closed and reachable'], state_transitions=['receptacle_state -> open'], postconditions=['target receptacle or appliance is open'], action_templates=['open {receptacle_or_appliance}'], success_checks=['observation confirms the receptacle or appliance is open'], failure_avoidance=['do not open an already open receptacle unless feedback requires retry'], ordering_constraints=['open closed containers before take, move, heat, cool, or clean when needed'], evidence_default='If the receptacle or appliance is closed, execute `open {recep}`.', step_order=len(candidates) + 1))
    if _alfworld_has_template(lowered, 'take') or re.search('\\btake\\s+\\{?(?:object|obj|target)', lowered):
        add(_alfworld_candidate(skill=skill, action='take', obj='target object from source receptacle', definition='Take the target object from its current receptacle in ALFWorld.', inputs=['target_object', 'source_receptacle'], outputs=['held target object'], tools=_ALFWORLD_TOOLS_BY_ACTION['take'], preconditions=['agent is at the source receptacle', 'target object is visible or accessible'], state_transitions=['target_object_location -> inventory'], postconditions=['agent is holding the target object'], action_templates=['take {object} from {source_receptacle}'], success_checks=['observation confirms the object was taken'], failure_avoidance=['navigate to the source receptacle and open it if needed before taking'], ordering_constraints=['take the target object before state change or final placement'], evidence_default='Use `take {obj} from {recep}` after navigating to the source receptacle.', step_order=len(candidates) + 1))
    for action, devices, target_state in [('heat', ['microwave', 'stoveburner'], 'heated'), ('cool', ['fridge'], 'cooled'), ('clean', ['sinkbasin'], 'cleaned')]:
        if not _alfworld_state_action_supported(skill, action, lowered):
            continue
        add(_alfworld_candidate(skill=skill, action=action, obj='held object', definition=f'Execute the ALFWorld {action} action on a held object using the correct appliance.', inputs=['held_object', f'{action}_appliance'], outputs=[f'{target_state}_object'], tools=['ALFWorld environment', *devices], preconditions=['agent is holding the target object', f'agent is at a valid {action} appliance'], state_transitions=[f'held object state -> {target_state}'], postconditions=[f'object is {target_state} before subsequent placement'], action_templates=[f'{action} {{object}} with {{appliance}}'], success_checks=[f'observation confirms the object was {action}ed or changed to {target_state}'], failure_avoidance=['if feedback says Nothing happened, verify inventory, location, appliance state, and exact object id'], ordering_constraints=['perform the state change after taking the object and before final placement'], evidence_default=f'Use `{action} {{object}} with {{appliance}}` after holding the object at the valid appliance.', step_order=len(candidates) + 1))
    if _alfworld_has_template(lowered, 'move') or _alfworld_has_template(lowered, 'put') or re.search('\\b(?:move|put)\\s+\\{?(?:object|obj|held_object)', lowered):
        add(_alfworld_candidate(skill=skill, action='move', obj='held object to target receptacle', definition='Move the held object to the target receptacle to satisfy the ALFWorld placement goal.', inputs=['held_object', 'target_receptacle'], outputs=['object at target receptacle'], tools=_ALFWORLD_TOOLS_BY_ACTION['move'], preconditions=['agent is holding the object', 'agent is at the target receptacle'], state_transitions=['object_location -> target receptacle'], postconditions=['object is placed at the target receptacle'], action_templates=['move {object} to {target_receptacle}'], success_checks=['observation confirms the object was moved to the target receptacle'], failure_avoidance=['navigate to the target receptacle before moving the held object'], ordering_constraints=['final placement happens after required state changes'], evidence_default='Use `move {obj} to {recep}` for final placement in ALFWorld.', step_order=len(candidates) + 1))
    if _alfworld_has_template(lowered, 'look') or 'desklamp' in lowered or 'floorlamp' in lowered:
        add(_alfworld_candidate(skill=skill, action='look', obj='target object in light', definition='Inspect the target object under a usable light source for ALFWorld examine tasks.', inputs=['target_object', 'light_source'], outputs=['examined target object'], tools=_ALFWORLD_TOOLS_BY_ACTION['look'], preconditions=['agent can access the target object and a light source'], state_transitions=['target object observed in light'], postconditions=['target object has been examined'], action_templates=['look at {object}', 'turn on {light_source}'], success_checks=['observation confirms the object was examined'], failure_avoidance=['use an available desklamp or floorlamp when the task requires light'], ordering_constraints=['use the light source before or during the examine step'], evidence_default='For examine tasks, look at the target object with a light source if required.', step_order=len(candidates) + 1))
    return candidates

def postprocess_extraction_result(skill: SkillRecord, candidates: list[CandidateMetaOp], residual_rows: Any | None=None, coverage_rows: Any | None=None) -> MetaOpExtractionResult:
    kept: list[CandidateMetaOp] = []
    all_candidates = list(candidates)
    for index, candidate in enumerate(all_candidates, start=1):
        candidate = _normalize_candidate_operation(candidate, skill)
        candidate.source_skill_id = skill.skill_id
        candidate.step_order = index
        candidate.local_id = f'{candidate.source_skill_id}::C{index:03d}'
        candidate.quality_flags = infer_quality_flags(candidate)
        candidate.residual_constraints = []
        if _should_drop_non_operation(candidate):
            continue
        if _should_move_to_residual(candidate):
            continue
        if validate_candidate_extraction(candidate):
            continue
        kept.append(candidate)
    coverage_notes = [row for row in coverage_rows or [] if isinstance(row, dict) and any((clean_text(value) for value in row.values()))]
    return MetaOpExtractionResult(metaops=kept, residual_constraints=[], coverage_notes=coverage_notes)

def _fallback_candidate_metaops(skill: SkillRecord) -> list[CandidateMetaOp]:
    alfworld_candidates = _alfworld_deterministic_candidate_metaops(skill)
    if alfworld_candidates:
        return alfworld_candidates
    text = '\n'.join([skill.one_line_capability, skill.description, '\n'.join(skill.example_tasks), skill.rendered_snippet[:1200]])
    sentences = re.split('(?<=[.!?])\\s+|\\n+', text)
    candidates: list[CandidateMetaOp] = []
    for sentence in sentences:
        lowered = sentence.lower()
        verb = next((item for item in _ACTION_VERBS if re.search(f'\\b{item}\\b', lowered)), '')
        if not verb:
            continue
        words = re.findall('[A-Za-z0-9_.+-]+', sentence)
        obj = ' '.join(words[:6]) if words else skill.name
        candidates.append(CandidateMetaOp(name=f'{verb} {obj}'[:80], definition=clean_text(sentence) or skill.one_line_capability or skill.description, action=verb, object=obj or skill.name, inputs=skill.inputs, outputs=skill.outputs, tools=skill.tooling, preconditions=skill.compatibility, action_templates=[clean_text(sentence)[:_EVIDENCE_MAX_CHARS]] if clean_text(sentence) else [], success_checks=[f'expected output is available: {item}' for item in skill.outputs[:2]], postconditions=[], step_order=len(candidates) + 1, evidence=clean_text(sentence)[:500], source_skill_id=skill.skill_id))
        if len(candidates) >= 6:
            break
    if candidates:
        return candidates
    return [CandidateMetaOp(name=skill.one_line_capability or skill.name, definition=skill.description or skill.one_line_capability or skill.name, action='use', object=skill.name, inputs=skill.inputs, outputs=skill.outputs, tools=skill.tooling, preconditions=skill.compatibility, action_templates=[skill.one_line_capability or skill.description or skill.name], success_checks=[f'expected output is available: {item}' for item in skill.outputs[:2]], postconditions=[], step_order=1, evidence=(skill.rendered_snippet or skill.description)[:500], source_skill_id=skill.skill_id)]

def _fallback_extraction_result(skill: SkillRecord) -> MetaOpExtractionResult:
    return postprocess_extraction_result(skill, _fallback_candidate_metaops(skill), [], [])

async def extract_candidate_metaops_with_residuals(skill: SkillRecord, llm_service: Any | None, *, use_llm: bool=True, extraction_policy: ExtractionPolicy | None=None, agent_mode: str='legacy', agent_cache: AgentCallCache | None=None) -> MetaOpExtractionResult:
    if not use_llm or llm_service is None:
        return _apply_extraction_policy(_fallback_extraction_result(skill), extraction_policy, agent_mode=agent_mode, agent_status='deterministic')
    payload = skill.compact_summary()
    payload['rendered_snippet'] = skill.rendered_snippet
    payload['raw_content_excerpt'] = skill.raw_content[:6000]
    user_prompt = AGENT1_USER.format(skill_json=json.dumps(payload, ensure_ascii=False, indent=2), skill_id=skill.skill_id)
    if extraction_policy is not None:
        user_prompt += AGENT1_ONTOLOGY_GUIDANCE.format(policy_json=json.dumps(extraction_policy.prompt_dict(), ensure_ascii=False, indent=2))
    cache_path = ''
    agent_status = 'live'
    try:
        if agent_cache is not None:
            data, cache_record = await agent_cache.call_json('agent1_extract', llm_service, system_prompt=AGENT1_SYSTEM, prompt=user_prompt, response_parser=AGENT1_RESPONSE_PARSER, max_tokens=2600)
            cache_path = cache_record.cache_path
            agent_status = cache_record.status
        else:
            data = await call_json(llm_service, system_prompt=AGENT1_SYSTEM, prompt=user_prompt, max_tokens=2600)
        if isinstance(data, list):
            rows = data
            residual_rows = []
            coverage_rows = []
        else:
            rows = data.get('metaops', []) if isinstance(data, dict) else []
            residual_rows = data.get('residual_constraints', []) if isinstance(data, dict) else []
            coverage_rows = data.get('coverage_notes', []) if isinstance(data, dict) else []
        candidates: list[CandidateMetaOp] = []
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            row.setdefault('source_skill_id', skill.skill_id)
            row.setdefault('step_order', index)
            item = CandidateMetaOp(**row)
            candidates.append(item)
        return _apply_extraction_policy(postprocess_extraction_result(skill, candidates, residual_rows, coverage_rows), extraction_policy, agent_mode=agent_mode, cache_path=cache_path, agent_status=agent_status)
    except Exception as exc:
        if agent_cache is not None and agent_cache.strict_failures:
            return MetaOpExtractionResult(audit_records=[{'candidate_local_id': '', 'source_skill_id': skill.skill_id, 'agent_mode': agent_mode, 'agent_status': 'failed', 'ontology_guided': extraction_policy is not None, 'policy_sha256': extraction_policy.policy_sha256 if extraction_policy is not None else '', 'policy_rule_ids': [], 'skill_evidence': {}, 'response_cache_path': cache_path, 'error_type': type(exc).__name__, 'error': str(exc)}])
        return _apply_extraction_policy(_fallback_extraction_result(skill), extraction_policy, agent_mode=agent_mode, agent_status='fallback_after_failure')

async def extract_candidate_metaops(skill: SkillRecord, llm_service: Any | None, *, use_llm: bool=True, extraction_policy: ExtractionPolicy | None=None, agent_mode: str='legacy', agent_cache: AgentCallCache | None=None) -> list[CandidateMetaOp]:
    return (await extract_candidate_metaops_with_residuals(skill, llm_service, use_llm=use_llm, extraction_policy=extraction_policy, agent_mode=agent_mode, agent_cache=agent_cache)).metaops

def candidate_metaops_to_jsonl(metaops: list[CandidateMetaOp]) -> str:
    return '\n'.join((json.dumps(asdict(item), ensure_ascii=False) for item in metaops))
