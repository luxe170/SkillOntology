from __future__ import annotations
import re
from typing import Any
from skillontology.schema import QueryDecomposition, QueryOperationNeed, clean_list, clean_text
ACTION_SEQUENCES = {'put': ['locate', 'take', 'move'], 'put_two': ['locate', 'take', 'move', 'repeat'], 'clean_and_place': ['locate', 'take', 'clean', 'move'], 'cool_and_place': ['locate', 'take', 'cool', 'move'], 'heat_and_place': ['locate', 'take', 'heat', 'move'], 'examine': ['locate', 'take', 'turn_on', 'look']}
DEVICES = {'desklamp', 'microwave', 'fridge', 'sinkbasin', 'coffeemachine', 'stoveburner', 'cabinet', 'drawer', 'dresser', 'garbagecan', 'diningtable', 'countertop', 'desk', 'sidetable', 'table', 'toilet'}

def parse_query(task: Any) -> QueryDecomposition:
    raw_query = str(task or '')
    structured = _parse_structured_query(raw_query)
    if structured is not None:
        return structured
    goal = _extract_goal(raw_query)
    task_type = _infer_task_type(goal)
    obj = _extract_primary_object(goal)
    target_receptacle = _extract_target_receptacle(goal)
    required_state = _extract_required_state(goal)
    count = _extract_count(goal)
    device = _extract_device(goal)
    actions = ACTION_SEQUENCES.get(task_type, ['navigate', 'take', 'move', 'open', 'close', 'heat', 'cool', 'clean', 'look', 'put'])
    return _build_decomposition(raw_query=raw_query, goal=goal, task_type=task_type, obj=obj, target_receptacle=target_receptacle, required_state=required_state, count=count, device=device, actions=actions)

def _parse_structured_query(raw_query: str) -> QueryDecomposition | None:
    fields: dict[str, str] = {}
    for part in re.split('[;\\n]+', raw_query):
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        fields[clean_text(key).lower().replace(' ', '_')] = clean_text(value)
    if fields.get('environment', '').lower() != 'alfworld':
        return None
    return _build_decomposition(raw_query=raw_query, goal=fields.get('goal') or raw_query, task_type=fields.get('task_type') or 'other', obj=fields.get('object') or 'unknown', target_receptacle=fields.get('target_receptacle') or 'unknown', required_state=fields.get('required_state') or 'none', count=fields.get('count') or 'unspecified', device=fields.get('device') or 'none', actions=clean_list(re.split('[,|]', fields.get('actions', ''))) or ACTION_SEQUENCES['put'])

def _build_decomposition(*, raw_query: str, goal: str, task_type: str, obj: str, target_receptacle: str, required_state: str, count: str, device: str, actions: list[str]) -> QueryDecomposition:
    constraints = clean_list([f'required_state={required_state}', f'count={count}', 'use exact ALFWorld actions', 'stop when environment reward indicates completion'])
    operations: list[QueryOperationNeed] = []
    for priority, action in enumerate(actions[:8], start=1):
        action = clean_text(action).lower()
        op_object = _operation_object(action, obj, target_receptacle, device)
        operations.append(QueryOperationNeed(name=f'{action} {op_object}'.strip(), action=action, object=op_object, inputs=clean_list([obj]), outputs=_operation_outputs(action, obj, target_receptacle, count), tools=_operation_tools(action, target_receptacle, device), constraints=constraints, priority=priority))
    return QueryDecomposition(raw_query=raw_query, normalized_goal=goal, task_type=f'alfworld-{task_type}' if task_type else 'alfworld', domain_objects=clean_list([obj, target_receptacle, device if device != 'none' else '', 'alfworld', 'household manipulation']), input_artifacts=clean_list([obj]), expected_outputs=clean_list([f'{required_state} {obj}' if required_state != 'none' and obj != 'unknown' else '', f'{obj} at {target_receptacle}' if target_receptacle != 'unknown' and obj != 'unknown' else '', 'task reward achieved']), required_operations=operations, tool_hints=clean_list(['ALFWorld action grammar', target_receptacle, device if device != 'none' else '']), constraints=constraints, validation_signals=['ALFWorld feedback confirms action success', 'environment reward indicates completion'])

def _extract_goal(task: str) -> str:
    match = re.search('your task is to:\\s*(.+)', task, re.IGNORECASE)
    if match:
        return match.group(1).splitlines()[0].strip().rstrip('.')
    return clean_text(task)

def _infer_task_type(goal: str) -> str:
    text = goal.lower()
    if 'look at' in text or 'examine' in text:
        return 'examine'
    if 'find two' in text or 'put two' in text:
        return 'put_two'
    if 'clean' in text:
        return 'clean_and_place'
    if 'cool' in text:
        return 'cool_and_place'
    if 'heat' in text or 'hot ' in text:
        return 'heat_and_place'
    if 'put' in text:
        return 'put'
    return 'other'

def _extract_required_state(goal: str) -> str:
    text = goal.lower()
    if 'clean' in text:
        return 'clean'
    if 'cool' in text:
        return 'cool'
    if 'heat' in text or 'hot ' in text:
        return 'hot'
    return 'none'

def _extract_count(goal: str) -> str:
    text = goal.lower()
    if 'find two' in text or 'put two' in text:
        return '2'
    if re.search('\\bsome\\b', text):
        return 'some'
    if re.search('\\ban?\\b', text):
        return '1'
    return 'unspecified'

def _extract_target_receptacle(goal: str) -> str:
    patterns = ['\\b(?:in|into|inside|on|onto|under)\\s+([a-z0-9]+)', '\\bto\\s+([a-z0-9]+)$']
    for pattern in patterns:
        matches = re.findall(pattern, goal, re.IGNORECASE)
        if matches:
            return matches[-1].lower()
    return 'unknown'

def _extract_device(goal: str) -> str:
    text = goal.lower()
    for device in sorted(DEVICES, key=len, reverse=True):
        if device in text:
            return device
    return 'none'

def _extract_primary_object(goal: str) -> str:
    text = goal.lower().rstrip('.')
    patterns = ['(?:look at|examine)\\s+the\\s+([a-z0-9]+)', '(?:look at|examine)\\s+([a-z0-9]+)', '(?:put|find|clean|cool|heat)\\s+(?:a|an|some|two)?\\s*(?:clean|cool|hot|heated)?\\s*([a-z0-9]+)']
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group(1) not in {'clean', 'cool', 'hot', 'heated'}:
            return match.group(1)
    stop = {'put', 'find', 'clean', 'cool', 'heat', 'hot', 'heated', 'look', 'at', 'examine', 'the', 'a', 'an', 'some', 'two', 'in', 'into', 'inside', 'on', 'onto', 'under', 'with', 'and', 'it', 'them'}
    for token in re.findall('[a-z0-9]+', text):
        if token not in stop:
            return token
    return 'unknown'

def _operation_object(action: str, obj: str, target_receptacle: str, device: str) -> str:
    if action in {'locate', 'search', 'find'}:
        return clean_text(obj or 'target object')
    if action in {'repeat'}:
        return clean_text(' '.join(clean_list([obj, target_receptacle, 'until required count is placed'])))
    if action in {'navigate', 'go'}:
        return clean_text(target_receptacle or device or obj or 'target location')
    if action in {'heat', 'cool', 'clean'}:
        return clean_text(obj or 'target object')
    if action in {'look', 'turn_on'}:
        return clean_text(obj or device or 'target object')
    if action in {'move', 'put', 'open', 'close'}:
        return clean_text(' '.join(clean_list([obj, target_receptacle])))
    return clean_text(obj or target_receptacle or 'target object')

def _operation_outputs(action: str, obj: str, target_receptacle: str, count: str='unspecified') -> list[str]:
    if action in {'locate', 'search', 'find'}:
        return clean_list([f'{obj} location known', 'source receptacle known'])
    if action in {'take', 'pick', 'pickup'}:
        return clean_list([f'held {obj}'])
    if action in {'move', 'put'}:
        return clean_list([f'{obj} at {target_receptacle}'])
    if action == 'clean':
        return clean_list([f'clean {obj}'])
    if action == 'cool':
        return clean_list([f'cool {obj}'])
    if action == 'heat':
        return clean_list([f'hot {obj}'])
    if action == 'look':
        return clean_list([f'examined {obj}'])
    if action == 'turn_on':
        return ['device on']
    if action == 'repeat':
        return clean_list([f'count {count} satisfied', f'{obj} instances placed at {target_receptacle}'])
    return []

def _operation_tools(action: str, target_receptacle: str, device: str) -> list[str]:
    tools = {'clean': ['sinkbasin'], 'cool': ['fridge'], 'heat': ['microwave'], 'look': [device if device != 'none' else 'desklamp'], 'turn_on': [device if device != 'none' else 'desklamp'], 'open': [target_receptacle, device], 'close': [target_receptacle, device], 'move': [target_receptacle], 'put': [target_receptacle], 'locate': ['ALFWorld observation'], 'search': ['ALFWorld observation'], 'find': ['ALFWorld observation'], 'repeat': [target_receptacle]}
    return clean_list(tools.get(action, []))
from skillontology.benchmark.base import BenchmarkAdapter

class Adapter(BenchmarkAdapter):
    name = 'alfworld'

    def parse_task(self, task, observation, ontology):
        return parse_query(task.instruction)

    def binding_roles(self, task, query):
        objects = [item for item in query.domain_objects if item not in {'alfworld', 'household manipulation', 'unknown', 'none'}]
        roles = {}
        if objects:
            for name in ('object', 'obj', 'target_object', 'held_object'):
                roles[name] = objects[0]
        if len(objects) > 1:
            for name in ('target', 'destination', 'target_receptacle'):
                roles[name] = objects[1]
        if len(objects) > 2:
            for name in ('device', 'appliance', 'light_source'):
                roles[name] = objects[2]
        return roles
