from __future__ import annotations
import re
from typing import Any
from skillontology.schema import QueryDecomposition, QueryOperationNeed, clean_list, clean_text
TASK_ACTIONS: dict[str, list[str]] = {'boil': ['locate', 'fetch', 'heat', 'wait', 'verify_state'], 'melt': ['locate', 'fetch', 'heat', 'wait', 'verify_state'], 'freeze': ['locate', 'fetch', 'cool', 'wait', 'verify_state'], 'change-the-state-of-matter-of': ['locate', 'fetch', 'heat_or_cool', 'wait', 'verify_state'], 'use-thermometer': ['locate', 'fetch', 'measure_temperature', 'read_measurement'], 'measure-melting-point-known-substance': ['locate', 'fetch', 'heat', 'measure_temperature', 'record_threshold'], 'measure-melting-point-unknown-substance': ['locate', 'fetch', 'heat', 'measure_temperature', 'identify_threshold'], 'power-component': ['locate', 'fetch', 'connect_circuit', 'activate_device', 'verify_power'], 'power-component-renewable-vs-nonrenewable-energy': ['locate', 'fetch', 'connect_circuit', 'classify_energy_source', 'verify_power'], 'test-conductivity': ['locate', 'fetch', 'connect_circuit', 'observe_conductivity', 'record_result'], 'test-conductivity-of-unknown-substances': ['locate', 'fetch', 'connect_circuit', 'observe_conductivity', 'classify_material'], 'find-living-thing': ['scan_environment', 'identify_living_entity', 'select_target'], 'find-non-living-thing': ['scan_environment', 'identify_non_living_entity', 'select_target'], 'find-plant': ['scan_environment', 'identify_plant', 'select_target'], 'find-animal': ['scan_environment', 'identify_animal', 'select_target'], 'grow-plant': ['locate', 'prepare_container', 'plant_seed', 'water', 'wait', 'verify_growth'], 'grow-fruit': ['locate', 'prepare_container', 'plant_seed', 'water', 'wait', 'verify_fruit'], 'chemistry-mix': ['locate', 'fetch', 'pour', 'mix', 'observe_result'], 'chemistry-mix-paint-secondary-color': ['locate', 'fetch', 'pour', 'mix', 'verify_color'], 'chemistry-mix-paint-tertiary-color': ['locate', 'fetch', 'pour', 'mix', 'verify_color'], 'lifespan-longest-lived': ['read_reference', 'compare_lifespan', 'select_longest_lived'], 'lifespan-shortest-lived': ['read_reference', 'compare_lifespan', 'select_shortest_lived'], 'lifespan-longest-lived-then-shortest-lived': ['read_reference', 'compare_lifespan', 'select_longest_lived', 'select_shortest_lived'], 'identify-life-stages-1': ['read_reference', 'identify_life_stage', 'select_answer'], 'identify-life-stages-2': ['read_reference', 'identify_life_stage', 'select_answer'], 'inclined-plane-determine-angle': ['locate', 'use_inclined_plane', 'adjust_angle', 'record_result'], 'inclined-plane-friction-named-surfaces': ['locate', 'use_inclined_plane', 'compare_friction', 'record_result'], 'inclined-plane-friction-unnamed-surfaces': ['locate', 'use_inclined_plane', 'compare_friction', 'record_result'], 'mendelian-genetics-known-plant': ['read_reference', 'observe_traits', 'apply_genetics_rule', 'select_answer'], 'mendelian-genetics-unknown-plant': ['read_reference', 'observe_traits', 'infer_genotype', 'select_answer']}
TOOL_HINTS_BY_ACTION: dict[str, list[str]] = {'locate': ['target object', 'room'], 'fetch': ['target object', 'inventory'], 'wait': ['time', 'environment state'], 'verify_state': ['target object', 'state'], 'heat': ['stove', 'burner', 'hot plate', 'beaker'], 'heat_or_cool': ['stove', 'freezer', 'thermometer'], 'cool': ['freezer', 'cold room'], 'measure_temperature': ['thermometer'], 'read_measurement': ['thermometer'], 'record_threshold': ['thermometer'], 'identify_threshold': ['thermometer'], 'connect_circuit': ['wire', 'battery', 'switch', 'light bulb'], 'activate_device': ['switch', 'power source'], 'verify_power': ['light bulb', 'device'], 'observe_conductivity': ['conductivity tester', 'wire', 'battery', 'light bulb'], 'record_result': ['task description', 'observation'], 'classify_material': ['conductivity tester'], 'classify_energy_source': ['power source', 'generator'], 'scan_environment': ['room', 'visible objects'], 'identify_living_entity': ['living thing'], 'identify_non_living_entity': ['non-living thing'], 'identify_plant': ['plant'], 'identify_animal': ['animal'], 'select_target': ['target object'], 'prepare_container': ['pot', 'soil', 'container'], 'plant_seed': ['seed', 'pot', 'soil'], 'water': ['water', 'sink', 'container'], 'verify_growth': ['plant'], 'verify_fruit': ['fruit'], 'pour': ['container', 'beaker', 'substance'], 'mix': ['container', 'substance'], 'observe_result': ['mixture', 'observation'], 'verify_color': ['paint', 'color'], 'use_inclined_plane': ['inclined plane', 'ramp', 'block'], 'adjust_angle': ['inclined plane'], 'compare_friction': ['inclined plane', 'surface'], 'compare_lifespan': ['reference', 'lifespan'], 'select_longest_lived': ['reference', 'organism'], 'select_shortest_lived': ['reference', 'organism'], 'identify_life_stage': ['reference', 'life stage'], 'select_answer': ['answer', 'task description'], 'observe_traits': ['plant', 'traits'], 'apply_genetics_rule': ['genetics', 'traits'], 'infer_genotype': ['genetics', 'traits'], 'read_reference': ['book', 'reference', 'task description']}
GENERIC_GOAL_OBJECT_STOPWORDS = {'a', 'an', 'and', 'answer', 'boil', 'by', 'change', 'classify', 'color', 'component', 'conductivity', 'cool', 'determine', 'find', 'freeze', 'get', 'identify', 'in', 'into', 'is', 'it', 'living', 'matter', 'melt', 'mix', 'non', 'of', 'on', 'paint', 'power', 'stage', 'state', 'substance', 'task', 'test', 'the', 'thing', 'to', 'turn', 'use', 'using', 'your'}

def parse_query(task: Any, domain_schema: Any | None=None) -> QueryDecomposition:
    raw_input = str(task or '')
    raw_query = clean_text(raw_input)
    fields = _parse_fields(raw_input)
    task_name = fields.get('task_name') or fields.get('task') or _extract_task_name(raw_query)
    goal = fields.get('goal') or fields.get('task_description') or _extract_goal(raw_input)
    observation = fields.get('initial_observation') or _extract_initial_observation(raw_input)
    task_type = _normalize_task_name(task_name)
    schema_row = _match_domain_schema(task_type, goal, domain_schema)
    actions = (clean_list(schema_row.get('canonical_operations')) if schema_row else []) or TASK_ACTIONS.get(task_type) or _fallback_actions(goal)
    domain_objects = _domain_objects(goal, observation, task_type, actions)
    operations = [_operation_need(action, goal=goal, task_type=task_type, domain_objects=domain_objects, priority=index) for index, action in enumerate(actions[:8], start=1)]
    constraints = clean_list(['use exact ScienceWorld valid actions', 'prefer actions available in the current valid action list', 'track inventory, containers, rooms, and measured state', 'stop when ScienceWorld reports task completion or reward'])
    return QueryDecomposition(raw_query=raw_query, normalized_goal=goal or raw_query, task_type=f'scienceworld-{task_type}' if task_type else 'scienceworld', domain_objects=clean_list([*domain_objects, 'scienceworld', 'text environment']), input_artifacts=clean_list([goal, observation]), expected_outputs=_expected_outputs(task_type, goal), required_operations=operations, tool_hints=clean_list([*([f"domain_schema:{schema_row.get('name')}", f"workflow:{schema_row.get('workflow_template')}"] if schema_row else []), *[tool for action in actions for tool in TOOL_HINTS_BY_ACTION.get(action, [])]]), constraints=constraints, validation_signals=['ScienceWorld observation confirms the state changed', 'ScienceWorld score or reward increases', 'ScienceWorld done flag is true'])

def _match_domain_schema(task_type: str, goal: str, domain_schema: Any | None) -> dict[str, Any] | None:
    rows = _task_schema_rows(domain_schema)
    task_key = _normalize_task_name(task_type).removeprefix('scienceworld-')
    goal_lower = (goal or '').lower()
    fallback = None
    for row in rows:
        canonical = _normalize_task_name(str(row.get('canonical_task_type') or row.get('name') or ''))
        parser_type = _normalize_task_name(str(row.get('task_type') or '')).removeprefix('scienceworld-')
        if task_key and task_key in {canonical, parser_type}:
            return row
        trigger_all = [item.lower() for item in clean_list(row.get('trigger_all'))]
        trigger_any = [item.lower() for item in clean_list(row.get('trigger_any'))]
        if trigger_all and (not all((item in goal_lower for item in trigger_all))):
            continue
        if trigger_any and any((item in goal_lower for item in trigger_any)):
            fallback = row
    return _schema_alias_row(task_key, domain_schema) or fallback

def _schema_alias_row(task_key: str, domain_schema: Any | None) -> dict[str, Any] | None:
    operation_schemas = getattr(domain_schema, 'operation_schemas', None)
    workflow_templates = getattr(domain_schema, 'workflow_templates', None)
    if isinstance(domain_schema, dict):
        operation_schemas = domain_schema.get('operation_schemas', operation_schemas)
        workflow_templates = domain_schema.get('workflow_templates', workflow_templates)
    if not isinstance(operation_schemas, dict) or not isinstance(workflow_templates, dict):
        return None
    aliases = operation_schemas.get('task_type_workflow_aliases')
    if not isinstance(aliases, dict):
        return None
    normalized_key = _normalize_task_name(task_key).removeprefix('scienceworld-')
    alias_candidates = [f'scienceworld-{normalized_key}', normalized_key, normalized_key.replace('_', '-')]
    workflow_name = next((str(aliases[key]) for key in alias_candidates if key in aliases and aliases[key]), '')
    workflow = workflow_templates.get(workflow_name)
    if not workflow_name or not isinstance(workflow, list):
        return None
    return {'name': normalized_key, 'canonical_task_type': normalized_key, 'workflow_template': workflow_name, 'canonical_operations': [str(item) for item in workflow if str(item).strip()]}

def _task_schema_rows(domain_schema: Any | None) -> list[dict[str, Any]]:
    if domain_schema is None:
        return []
    if hasattr(domain_schema, 'task_schema_rows'):
        rows = domain_schema.task_schema_rows()
    elif isinstance(domain_schema, dict):
        rows = domain_schema.get('task_schemas', [])
    else:
        rows = getattr(domain_schema, 'task_schemas', [])
    return [row for row in rows if isinstance(row, dict)]

def _parse_fields(raw_query: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in re.split('[;\\n]+', raw_query):
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        fields[clean_text(key).lower().replace(' ', '_')] = clean_text(value)
    return fields

def _extract_task_name(raw_query: str) -> str:
    patterns = ['\\bscienceworld\\s+task\\s*=\\s*([a-z0-9_.-]+)', '\\btask_name\\s*=\\s*([a-z0-9_.-]+)', '\\btask\\s*=\\s*([a-z0-9_.-]+)']
    for pattern in patterns:
        match = re.search(pattern, raw_query, re.IGNORECASE)
        if match:
            return match.group(1)
    for name in sorted(TASK_ACTIONS, key=len, reverse=True):
        if name in raw_query.lower():
            return name
    return ''

def _normalize_task_name(task_name: str) -> str:
    return clean_text(task_name).lower().replace('_', '-')

def _extract_goal(raw_query: str) -> str:
    for pattern in ['\\bgoal\\s*=\\s*([^;\\n]+)', '\\btask_description\\s*=\\s*([^;\\n]+)', 'Your task is to:\\s*(.+)']:
        match = re.search(pattern, raw_query, re.IGNORECASE)
        if match:
            return clean_text(match.group(1))
    lines = [line.strip() for line in raw_query.splitlines() if line.strip()]
    for line in lines:
        if not line.lower().startswith(('environment=', 'benchmark=', 'task_name=', 'variation_idx=')):
            return clean_text(line)
    return raw_query

def _extract_initial_observation(raw_query: str) -> str:
    match = re.search('Initial observation:\\s*(.+)', raw_query, re.IGNORECASE | re.DOTALL)
    return clean_text(match.group(1)) if match else ''

def _fallback_actions(goal: str) -> list[str]:
    text = goal.lower()
    actions = []
    for action, markers in [('locate', ['find', 'locate', 'get', 'obtain']), ('fetch', ['get', 'take', 'obtain', 'bring']), ('heat', ['heat', 'boil', 'melt', 'warm']), ('cool', ['cool', 'freeze']), ('measure_temperature', ['thermometer', 'temperature', 'melting point']), ('connect_circuit', ['circuit', 'conductivity', 'power', 'battery']), ('pour', ['pour', 'mix', 'chemistry']), ('mix', ['mix', 'paint', 'chemistry']), ('select_answer', ['which', 'identify', 'choose', 'answer'])]:
        if any((marker in text for marker in markers)):
            actions.append(action)
    return clean_list(actions) or ['locate', 'fetch', 'operate', 'verify_result']

def _domain_objects(goal: str, observation: str, task_type: str, actions: list[str] | None=None) -> list[str]:
    goal_text = (goal or '').lower()
    observation_text = (observation or '').lower()
    objects = [task_type.replace('-', ' ')]
    goal_objects = _goal_objects(goal, task_type)
    objects.extend(goal_objects)
    vocabulary = ['water', 'ice', 'lead', 'tin', 'iron', 'copper', 'aluminum', 'aluminium', 'salt', 'sugar', 'thermometer', 'beaker', 'substance', 'unknown substance', 'electric motor', 'electric buzzer', 'generator', 'wind generator', 'solar panel', 'nuclear generator', 'wire', 'battery', 'switch', 'light bulb', 'conductivity tester', 'plant', 'animal', 'seed', 'soil', 'pot', 'fruit', 'paint', 'color', 'lifespan', 'life stage', 'inclined plane', 'ramp', 'surface', 'friction', 'genetics', 'trait']
    allowed_by_action = {tool for action in actions or [] for tool in TOOL_HINTS_BY_ACTION.get(action, [])}
    for item in vocabulary:
        dashed = item.replace(' ', '-')
        if item in goal_text or dashed in task_type:
            objects.append(item)
            continue
        if item in observation_text and item in allowed_by_action:
            objects.append(item)
    return clean_list(objects)

def _goal_objects(goal: str, task_type: str) -> list[str]:
    text = clean_text(goal).lower().rstrip('.')
    if not text:
        return []
    patterns = ['\\b(?:boil|melt|freeze)\\s+(?:the\\s+|a\\s+|an\\s+|some\\s+)?([a-z][a-z0-9 -]+?)(?:\\.|$)', '\\bchange\\s+the\\s+state\\s+of\\s+matter\\s+of\\s+(?:the\\s+|a\\s+|an\\s+|some\\s+)?([a-z][a-z0-9 -]+?)(?:\\.|$)', '\\bturn\\s+on\\s+(?:the\\s+|a\\s+|an\\s+)?([a-z][a-z0-9 -]+?)(?:\\.|$)', '\\bpower\\s+(?:the\\s+|a\\s+|an\\s+)?([a-z][a-z0-9 -]+?)(?:\\.|$)', '\\btest\\s+the\\s+conductivity\\s+of\\s+(?:the\\s+|a\\s+|an\\s+|some\\s+)?([a-z][a-z0-9 -]+?)(?:\\.|$)', '\\bmeasure\\s+the\\s+temperature\\s+of\\s+(?:the\\s+|a\\s+|an\\s+|some\\s+)?([a-z][a-z0-9 -]+?)(?:\\.|$)', '\\buse\\s+(?:the\\s+|a\\s+|an\\s+)?([a-z][a-z0-9 -]+?)\\s+to\\b', '\\bgrow\\s+(?:the\\s+|a\\s+|an\\s+|some\\s+)?([a-z][a-z0-9 -]+?)(?:\\.|$)', '\\bmake\\s+(?:the\\s+|a\\s+|an\\s+|some\\s+)?([a-z][a-z0-9 -]+?)(?:\\.|$)']
    objects: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            candidate = _clean_goal_object(match.group(1))
            if candidate:
                objects.append(candidate)
    if 'thermometer' in text:
        objects.append('thermometer')
    if 'unknown substance' in text:
        objects.append('unknown substance')
    if 'renewable' in task_type and 'generator' in text:
        objects.append('generator')
    return clean_list(objects)

def _clean_goal_object(candidate: str) -> str:
    text = re.sub('\\b(?:in|with|using|from|until|and then|then)\\b.*$', '', candidate.lower()).strip(' .')
    text = re.sub('\\s+', ' ', text)
    if not text:
        return ''
    tokens = [token for token in re.findall('[a-z0-9]+', text) if token not in GENERIC_GOAL_OBJECT_STOPWORDS]
    if not tokens:
        return ''
    cleaned = ' '.join(tokens)
    return cleaned if len(cleaned) > 1 else ''

def _operation_need(action: str, *, goal: str, task_type: str, domain_objects: list[str], priority: int) -> QueryOperationNeed:
    action_text = clean_text(action).lower()
    obj = _operation_object(action_text, domain_objects, task_type)
    return QueryOperationNeed(name=f'{action_text} {obj}'.strip(), action=action_text, object=obj, inputs=clean_list(domain_objects[:6]), outputs=_operation_outputs(action_text, obj, task_type, goal), tools=TOOL_HINTS_BY_ACTION.get(action_text, []), constraints=['ScienceWorld command must be chosen from valid actions', 'use environment feedback before continuing'], priority=priority)

def _operation_object(action: str, domain_objects: list[str], task_type: str) -> str:
    joined = ' '.join(domain_objects)
    if action in {'scan_environment', 'locate'}:
        return joined or 'target object'
    if action in {'fetch', 'pour', 'mix', 'heat', 'cool', 'heat_or_cool'}:
        return joined or 'substance'
    if 'conductivity' in action or action in {'connect_circuit', 'activate_device', 'verify_power'}:
        return 'circuit material'
    if action in {'measure_temperature', 'read_measurement', 'record_threshold', 'identify_threshold'}:
        return 'temperature threshold'
    if action in {'prepare_container', 'plant_seed', 'water', 'verify_growth', 'verify_fruit'}:
        return 'plant growth setup'
    if action in {'read_reference', 'compare_lifespan', 'identify_life_stage', 'apply_genetics_rule', 'infer_genotype'}:
        return task_type.replace('-', ' ')
    if action in {'use_inclined_plane', 'adjust_angle', 'compare_friction'}:
        return 'inclined plane experiment'
    if action.startswith('identify_') or action.startswith('select_'):
        return joined or task_type.replace('-', ' ')
    return joined or task_type.replace('-', ' ')

def _operation_outputs(action: str, obj: str, task_type: str, goal: str) -> list[str]:
    if action in {'locate', 'scan_environment'}:
        return [f'{obj} located', 'relevant room or object known']
    if action == 'fetch':
        return [f'{obj} available in inventory or local context']
    if action in {'heat', 'cool', 'heat_or_cool'}:
        return [f'{obj} state changed', 'temperature condition satisfied']
    if action in {'wait'}:
        return ['process advanced', 'state transition completed']
    if action in {'verify_state', 'verify_growth', 'verify_fruit', 'verify_color', 'verify_power'}:
        return ['target condition verified', 'task reward achieved']
    if action in {'measure_temperature', 'read_measurement', 'record_threshold', 'identify_threshold'}:
        return ['measurement recorded', 'temperature threshold known']
    if action in {'connect_circuit', 'observe_conductivity'}:
        return ['circuit observation available', 'conductivity result known']
    if action in {'classify_energy_source', 'classify_material'}:
        return ['classification selected', 'answer justified by observation']
    if action in {'prepare_container', 'plant_seed', 'water'}:
        return ['plant growth precondition satisfied']
    if action in {'pour', 'mix'}:
        return ['mixture created', 'resulting substance observed']
    if action in {'read_reference', 'compare_lifespan', 'identify_life_stage', 'apply_genetics_rule', 'infer_genotype'}:
        return ['answer inferred from reference information']
    if action in {'use_inclined_plane', 'adjust_angle', 'compare_friction'}:
        return ['experiment result observed']
    if action.startswith('identify_') or action.startswith('select_'):
        return clean_list(['target answer selected', goal])
    return ['task subgoal completed']

def _expected_outputs(task_type: str, goal: str) -> list[str]:
    outputs = ['task reward achieved', 'ScienceWorld done flag true']
    if any((marker in task_type for marker in ['boil', 'melt', 'freeze', 'state-of-matter'])):
        outputs.append('target matter state reached')
    if 'thermometer' in task_type or 'melting-point' in task_type:
        outputs.append('temperature measurement completed')
    if 'conductivity' in task_type or 'power-component' in task_type:
        outputs.append('circuit or conductivity conclusion reached')
    if 'grow' in task_type:
        outputs.append('plant growth result reached')
    if goal:
        outputs.append(goal)
    return clean_list(outputs)
from skillontology.benchmark.base import BenchmarkAdapter

class Adapter(BenchmarkAdapter):
    name = 'scienceworld'

    def parse_task(self, task, observation, ontology):
        text = f"task_name={task.identity['task_name']}\ngoal={task.instruction}\ninitial_observation={observation.text}"
        return parse_query(text, domain_schema=ontology)

    def binding_roles(self, task, query):
        return {'goal': task.instruction}
