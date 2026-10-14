from __future__ import annotations

from dataclasses import dataclass, field

from .memory import InteractionMemory
from .protocol import ExecutionProtocol, extract_action, validate_public_action


@dataclass
class EpisodeResult:
    identity: dict
    success: bool
    score: float
    stop_reason: str
    proposals: int
    environment_calls: int
    usage: dict = field(default_factory=dict)


class Runner:
    def __init__(self, adapter, environment, compiler, model, config):
        self.adapter = adapter
        self.environment = environment
        self.compiler = compiler
        self.model = model
        self.config = config
        self.protocol = ExecutionProtocol.from_config(config)

    async def run(self, task):
        protocol = self.protocol.for_task(task, self.config)
        usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        proposals = calls = errors = 0
        try:
            observation = self.environment.reset_task(task)
            if observation.done or observation.success:
                return EpisodeResult(task.identity, observation.success, observation.score, 'environment_terminal', proposals, calls, usage)
            dag = await self.compiler.compile(task, observation, self.adapter)
            system = 'Choose actions using only the task, procedural guidance, and public observations. Return exactly one line: Action: <action>. '
            system += 'Action syntax: ' + self.config['action']['syntax']
            memory = InteractionMemory(system, task.instruction, dag.rendered_context, observation)
            stop_reason = 'proposal_limit'
            for proposals in range(1, protocol.max_proposals + 1):
                messages = memory.messages(protocol.history_turns)
                estimated = sum((len(message['content']) + 2) // 3 for message in messages) + protocol.max_output_tokens
                if protocol.max_episode_tokens and usage['total_tokens'] + estimated > protocol.max_episode_tokens:
                    proposals -= 1
                    stop_reason = 'token_budget'
                    break
                response = await self.model.complete(messages)
                for key in usage:
                    if key not in response.usage:
                        raise ValueError('Model response lacks token usage')
                    usage[key] += int(response.usage[key])
                if protocol.max_episode_tokens and usage['total_tokens'] > protocol.max_episode_tokens:
                    stop_reason = 'token_budget'
                    break
                try:
                    action = extract_action(response.text, self.config['action']['pattern'])
                    action = validate_public_action(action, observation, self.config['action'])
                except ValueError as error:
                    errors += 1
                    memory.record(response.text, observation, str(error))
                    if errors >= protocol.max_protocol_errors:
                        stop_reason = 'protocol_error'
                        break
                    continue
                errors = 0
                observation = self.environment.step_action(action)
                calls += 1
                memory.record(response.text, observation)
                if observation.done or observation.success:
                    stop_reason = 'environment_terminal'
                    break
                if calls >= protocol.max_environment_calls:
                    stop_reason = 'environment_call_limit'
                    break
                if protocol.max_environment_ticks is not None and observation.ticks is not None and observation.ticks >= protocol.max_environment_ticks:
                    stop_reason = 'environment_tick_limit'
                    break
            return EpisodeResult(task.identity, observation.success, observation.score, stop_reason, proposals, calls, usage)
        finally:
            self.environment.close()
