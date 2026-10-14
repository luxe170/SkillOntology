from __future__ import annotations
from skillontology.schema import MetaOpDAG, QueryDecomposition, clean_list, clean_text

def _clip(text: str, max_chars: int=700) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rstrip() + '...'

def _list(values: list[str], *, max_items: int=5, max_chars: int=500) -> str:
    values = [clean_text(item) for item in values if clean_text(item)]
    if not values:
        return ''
    text = ', '.join(values[:max_items])
    if len(values) > max_items:
        text += f', ... (+{len(values) - max_items} more)'
    return _clip(text, max_chars)

def _append_list(lines: list[str], label: str, values: list[str], *, prefix: str='') -> None:
    rendered = _list(values, max_items=6, max_chars=700)
    if rendered:
        lines.append(f'{prefix}{label}: {rendered}')

def _query_parameters(query: QueryDecomposition) -> list[str]:
    values: list[str] = []
    values.extend(query.domain_objects)
    values.extend(query.input_artifacts)
    for operation in query.required_operations:
        values.extend([operation.object, *operation.inputs, *operation.tools])
    values.extend(query.constraints)
    return clean_list(values)

def _required_effects(query: QueryDecomposition) -> list[str]:
    effects: list[str] = []
    for operation in sorted(query.required_operations, key=lambda item: item.priority):
        effects.extend(operation.outputs or [operation.to_text()])
    return clean_list(effects)

def render_metaop_context(dag: MetaOpDAG, *, max_chars: int | None=6000) -> str:
    _ = max_chars
    by_id = {item.instance_id or item.metaop_id: item for item in dag.selected_metaops}
    goal = dag.query_decomposition.normalized_goal or dag.query
    lines = ['# MetaOp Execution Plan', f'Goal: {goal}']
    parameters = _query_parameters(dag.query_decomposition)
    if parameters:
        lines.append(f'Task parameters: {_list(parameters, max_items=10, max_chars=900)}')
    effects = _required_effects(dag.query_decomposition)
    if effects:
        lines.append(f'Required effects: {_list(effects, max_items=10, max_chars=900)}')
    if dag.missing_operations:
        lines.append(f'Missing MetaOps: {_list(dag.missing_operations, max_items=10, max_chars=900)}')
    lines.extend(['', 'Plan:'])
    for index, metaop_id in enumerate(dag.execution_order, start=1):
        metaop = by_id.get(metaop_id)
        if metaop is None:
            continue
        lines.extend(['', f'{index}. {metaop.name}'])
        if metaop.depends_on:
            lines.append('   Depends on: ' + ', '.join(metaop.depends_on))
        if metaop.action:
            lines.append(f'   Action: {_clip(metaop.action, 140)}')
        if metaop.object:
            lines.append(f'   Object: {_clip(metaop.object, 220)}')
        _append_list(lines, 'Preconditions', metaop.preconditions, prefix='   ')
        _append_list(lines, 'Procedure', metaop.procedure or metaop.action_templates, prefix='   ')
        _append_list(lines, 'Effects', metaop.effects or metaop.postconditions or metaop.outputs, prefix='   ')
        _append_list(lines, 'Success Checks', metaop.success_checks, prefix='   ')
        _append_list(lines, 'Failure Recovery', metaop.failure_recovery or metaop.failure_avoidance, prefix='   ')
    return '\n'.join(lines)
