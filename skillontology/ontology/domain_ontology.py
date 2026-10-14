from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any
from skillontology.schema import CandidateMetaOp, clean_list, clean_text
EXTRACTION_POLICY_COMPILER_VERSION = 'domain-ontology-extraction-policy-v1'

def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def _rule_slug(value: str) -> str:
    cleaned = ''.join((ch.lower() if ch.isalnum() else '_' for ch in clean_text(value)))
    return '_'.join((part for part in cleaned.split('_') if part)) or 'unnamed'

@dataclass
class ExtractionPolicy:
    schema_version: int
    domain: str
    source_domain_ontology_path: str
    source_domain_ontology_sha256: str
    compiler_version: str
    canonical_actions_and_aliases: dict[str, Any]
    object_roles: dict[str, list[str]]
    state_predicates: dict[str, list[str]]
    transition_schemas: dict[str, Any]
    conditional_prerequisites: list[dict[str, Any]]
    workflow_templates: dict[str, list[str]]
    recovery_policies: list[dict[str, Any]]
    evidence_policies: list[str]
    rule_catalog: dict[str, dict[str, str]] = field(default_factory=dict)
    policy_sha256: str = ''

    def content_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop('policy_sha256', None)
        return payload

    def hash_dict(self) -> dict[str, Any]:
        payload = self.content_dict()
        payload.pop('source_domain_ontology_path', None)
        return payload

    def prompt_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop('source_domain_ontology_path', None)
        return payload

    def to_dict(self) -> dict[str, Any]:
        payload = self.content_dict()
        payload['policy_sha256'] = self.policy_sha256 or hashlib.sha256(_canonical_json(self.hash_dict()).encode('utf-8')).hexdigest()
        return payload

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        return target

    def rule_id(self, section: str, key: str) -> str:
        candidate = f'{section}:{_rule_slug(key)}'
        return candidate if candidate in self.rule_catalog else ''

    def canonicalize_action(self, action: str) -> tuple[str, list[str]]:
        normalized = clean_text(action).lower().replace('-', '_').replace(' ', '_')
        aliases = self.canonical_actions_and_aliases.get('aliases', {})
        canonical = clean_text(aliases.get(normalized, aliases.get(clean_text(action).lower(), normalized))).lower()
        rule_ids: list[str] = []
        alias_rule = self.rule_id('action_alias', normalized)
        if alias_rule:
            rule_ids.append(alias_rule)
        elif canonical in set(self.canonical_actions_and_aliases.get('canonical_actions', [])):
            canonical_rule = self.rule_id('canonical_action', canonical)
            if canonical_rule:
                rule_ids.append(canonical_rule)
        return (canonical or clean_text(action).lower(), rule_ids)

    def normalize_candidate(self, candidate: CandidateMetaOp) -> list[str]:
        canonical_action, used = self.canonicalize_action(candidate.action)
        candidate.action = canonical_action
        canonical_actions = set(self.canonical_actions_and_aliases.get('canonical_actions', []))
        if canonical_action and canonical_action not in canonical_actions:
            candidate.unknown_fields = clean_list([*candidate.unknown_fields, f'novel_action:{canonical_action}'])
        searchable = ' '.join([candidate.object, *candidate.argument_roles, *candidate.inputs, *candidate.outputs, *candidate.preconditions, *candidate.effects, *candidate.state_transitions, *candidate.postconditions, candidate.evidence]).lower()
        for role in self.object_roles:
            role_tokens = clean_text(role).lower().replace('_', ' ')
            if role_tokens and role_tokens in searchable:
                rule_id = self.rule_id('object_role', role)
                if rule_id:
                    used.append(rule_id)
        for predicate in self.state_predicates:
            predicate_tokens = clean_text(predicate).lower().replace('_', ' ')
            if predicate_tokens and predicate_tokens in searchable:
                rule_id = self.rule_id('state_predicate', predicate)
                if rule_id:
                    used.append(rule_id)
        for name, schema in self.transition_schemas.items():
            transition_text = f'{name} {_canonical_json(schema)}'.lower().replace('_', ' ')
            if canonical_action and canonical_action.replace('_', ' ') in transition_text:
                rule_id = self.rule_id('transition_schema', name)
                if rule_id:
                    used.append(rule_id)
        for name, actions in self.workflow_templates.items():
            if canonical_action in {clean_text(item).lower() for item in actions}:
                rule_id = self.rule_id('workflow_template', name)
                if rule_id:
                    used.append(rule_id)
        if not candidate.evidence and (not candidate.evidence_spans):
            candidate.quality_flags = clean_list([*candidate.quality_flags, 'ontology_evidence_missing'])
        return clean_list(used)

    def validate_candidate(self, candidate: CandidateMetaOp, rule_ids: list[str] | None=None) -> dict[str, Any]:
        canonical_actions = set(self.canonical_actions_and_aliases.get('canonical_actions', []))
        used = clean_list(rule_ids or [])
        matched_roles = [item for item in used if item.startswith('object_role:')]
        matched_predicates = [item for item in used if item.startswith('state_predicate:')]
        novel_action = bool(candidate.action and candidate.action not in canonical_actions)
        evidence_grounded = bool(candidate.evidence or candidate.evidence_spans)
        errors: list[str] = []
        if not candidate.action:
            errors.append('missing_action')
        if not candidate.object:
            errors.append('missing_object')
        if not evidence_grounded:
            errors.append('missing_skill_evidence')
        return {'valid': not errors, 'errors': errors, 'canonical_action': not novel_action, 'novel_action': novel_action, 'matched_object_role_rule_ids': matched_roles, 'matched_state_predicate_rule_ids': matched_predicates, 'evidence_grounded': evidence_grounded}

@dataclass
class DomainOntologyMemory:
    domain: str
    schema_version: int = 1
    object_roles: dict[str, list[str]] = field(default_factory=dict)
    state_predicates: dict[str, list[str]] = field(default_factory=dict)
    operation_schemas: dict[str, Any] = field(default_factory=dict)
    transition_schemas: dict[str, Any] = field(default_factory=dict)
    conditional_prerequisites: list[dict[str, Any]] = field(default_factory=list)
    workflow_templates: dict[str, list[str]] = field(default_factory=dict)
    recovery_policies: list[dict[str, Any]] = field(default_factory=list)
    evidence_policies: list[str] = field(default_factory=list)
    rendering_contracts: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    source_path: str = ''
    sha256: str = ''

    @classmethod
    def load(cls, path: str | Path) -> 'DomainOntologyMemory':
        source = Path(path).expanduser().resolve()
        data = json.loads(source.read_text(encoding='utf-8'))
        memory = cls(domain=clean_text(data.get('domain')), schema_version=int(data.get('schema_version', 1)), object_roles=_clean_dict_of_lists(data.get('object_roles')), state_predicates=_clean_dict_of_lists(data.get('state_predicates')), operation_schemas=data.get('operation_schemas', {}) if isinstance(data.get('operation_schemas'), dict) else {}, transition_schemas=data.get('transition_schemas', {}) if isinstance(data.get('transition_schemas'), dict) else {}, conditional_prerequisites=_clean_list_of_dicts(data.get('conditional_prerequisites')), workflow_templates=_clean_dict_of_lists(data.get('workflow_templates')), recovery_policies=_clean_list_of_dicts(data.get('recovery_policies')), evidence_policies=clean_list(data.get('evidence_policies')), rendering_contracts=data.get('rendering_contracts', {}) if isinstance(data.get('rendering_contracts'), dict) else {}, provenance=data.get('provenance', {}) if isinstance(data.get('provenance'), dict) else {}, raw=data, source_path=str(source), sha256=sha256_file(source))
        if not memory.domain:
            raise ValueError(f'Domain ontology memory is missing domain: {source}')
        return memory

    def action_aliases(self) -> dict[str, str]:
        aliases = self.operation_schemas.get('action_aliases', {})
        if not isinstance(aliases, dict):
            return {}
        return {clean_text(key).lower(): clean_text(value).lower() for key, value in aliases.items() if clean_text(key) and clean_text(value)}

    def conditional_insertions(self) -> list[dict[str, Any]]:
        return [row for row in self.conditional_prerequisites if clean_text(row.get('type')).lower() == 'conditional_insertion']

    def rendering_profile(self, name: str) -> dict[str, Any]:
        profiles = self.rendering_contracts.get('profiles', {})
        if not isinstance(profiles, dict):
            return {}
        profile = profiles.get(str(name or ''))
        return profile if isinstance(profile, dict) else {}

def compile_extraction_policy(memory: DomainOntologyMemory) -> ExtractionPolicy:
    aliases = memory.action_aliases()
    canonical_actions = set(aliases.values())
    state_order = memory.operation_schemas.get('state_order', {})
    if isinstance(state_order, dict):
        canonical_actions.update((clean_text(item).lower() for item in state_order if clean_text(item)))
    normalized_workflows = {name: [aliases.get(clean_text(item).lower().replace('-', '_').replace(' ', '_'), clean_text(item).lower()) for item in actions if clean_text(item)] for name, actions in memory.workflow_templates.items()}
    for actions in normalized_workflows.values():
        canonical_actions.update((clean_text(item).lower() for item in actions if clean_text(item)))
    rule_catalog: dict[str, dict[str, str]] = {}

    def add_rules(section: str, keys: list[str]) -> None:
        for key in sorted(clean_list(keys), key=str.lower):
            rule_id = f'{section}:{_rule_slug(key)}'
            rule_catalog[rule_id] = {'section': section, 'key': key}
    add_rules('canonical_action', sorted(canonical_actions))
    add_rules('action_alias', list(aliases))
    add_rules('object_role', list(memory.object_roles))
    add_rules('state_predicate', list(memory.state_predicates))
    add_rules('transition_schema', list(memory.transition_schemas))
    add_rules('workflow_template', list(memory.workflow_templates))
    add_rules('conditional_prerequisite', [str(row.get('name') or row.get('insert') or index) for index, row in enumerate(memory.conditional_prerequisites, start=1)])
    add_rules('recovery_policy', [str(row.get('name') or row.get('failure') or index) for index, row in enumerate(memory.recovery_policies, start=1)])
    add_rules('evidence_policy', [str(index) for index, _ in enumerate(memory.evidence_policies, start=1)])
    policy = ExtractionPolicy(schema_version=1, domain=memory.domain, source_domain_ontology_path=memory.source_path, source_domain_ontology_sha256=memory.sha256, compiler_version=EXTRACTION_POLICY_COMPILER_VERSION, canonical_actions_and_aliases={'canonical_actions': sorted(canonical_actions), 'aliases': dict(sorted(aliases.items()))}, object_roles=memory.object_roles, state_predicates=memory.state_predicates, transition_schemas=memory.transition_schemas, conditional_prerequisites=memory.conditional_prerequisites, workflow_templates=normalized_workflows, recovery_policies=memory.recovery_policies, evidence_policies=memory.evidence_policies, rule_catalog=rule_catalog)
    policy.policy_sha256 = hashlib.sha256(_canonical_json(policy.hash_dict()).encode('utf-8')).hexdigest()
    return policy

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _clean_dict_of_lists(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    return {clean_text(key): clean_list(items) for key, items in value.items() if clean_text(key) and clean_list(items)}

def _clean_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
