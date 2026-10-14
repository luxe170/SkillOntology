from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import re


@dataclass(frozen=True)
class ExecutionProtocol:
    max_proposals: int
    max_environment_calls: int
    max_environment_ticks: int | None
    max_episode_tokens: int
    max_output_tokens: int
    history_turns: int | None
    max_protocol_errors: int = 3

    @classmethod
    def from_config(cls, config):
        values = config['runtime']
        protocol = cls(**{key: values.get(key) for key in cls.__dataclass_fields__ if key != 'max_protocol_errors'},
                       max_protocol_errors=int(values.get('max_protocol_errors', 3)))
        if min(protocol.max_proposals, protocol.max_environment_calls, protocol.max_output_tokens, protocol.max_protocol_errors) < 1:
            raise ValueError('Execution limits must be positive')
        if protocol.max_episode_tokens < 0 or (protocol.history_turns is not None and protocol.history_turns < 1):
            raise ValueError('Invalid token or history limit')
        return protocol

    def for_task(self, task, config):
        values = config['runtime'].get('task_limits', {}).get(task.identity.get('task_name', ''), {})
        if not values:
            return self
        protocol = replace(self, **values)
        if min(protocol.max_proposals, protocol.max_environment_calls) < 1:
            raise ValueError('Task execution limits must be positive')
        return protocol


def extract_action(response, pattern):
    text = str(response or '').strip()
    labelled = re.findall(r'(?im)^\s*Action\s*:\s*(.+?)\s*$', text)
    if len(labelled) > 1:
        raise ValueError('Return exactly one action')
    candidate = labelled[0] if labelled else text
    candidate = candidate.strip('` \t\r\n')
    if '\n' in candidate or len(candidate) > 500 or not re.fullmatch(pattern, candidate, re.IGNORECASE):
        raise ValueError('Return one valid action using the configured syntax')
    return candidate


def validate_public_action(action, observation, config):
    if config.get('validate_clicks', False) and action.lower().startswith('click['):
        argument = action[action.index('[') + 1:-1]
        controls = {value.casefold(): value for value in observation.controls}
        if argument.casefold() not in controls:
            raise ValueError('Click target is absent from the current public controls')
        action = 'click[' + controls[argument.casefold()] + ']'
    return action
