from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from skillontology.runtime.llm_json import extract_json_object

def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

@dataclass(frozen=True)
class AgentCallRecord:
    agent: str
    request_sha256: str
    cache_path: str
    status: str

class AgentCallCache:

    def __init__(self, output_dir: str | Path, *, replay_dir: str | Path | None=None, strict_failures: bool=False) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.replay_dir = Path(replay_dir).expanduser().resolve() if replay_dir else None
        self.strict_failures = bool(strict_failures)
        self.records: list[AgentCallRecord] = []

    @staticmethod
    def _request_payload(agent: str, system_prompt: str, prompt: str, kwargs: dict[str, Any], response_parser: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        return {'agent': str(agent), 'system_prompt': str(system_prompt), 'prompt': str(prompt), 'generation_args': kwargs, 'response_parser': f"{getattr(response_parser, '__module__', '')}.{getattr(response_parser, '__qualname__', getattr(response_parser, '__name__', 'callable'))}".strip('.')}

    @staticmethod
    def _relative_path(agent: str, request_sha256: str) -> Path:
        safe_agent = ''.join((ch if ch.isalnum() or ch in '-_' else '_' for ch in agent.lower()))
        return Path(safe_agent) / f'{request_sha256}.json'

    def _replay_path(self, relative: Path) -> Path | None:
        if self.replay_dir is None:
            return None
        direct = self.replay_dir / relative
        nested = self.replay_dir / 'agent_cache' / relative
        if direct.exists():
            return direct
        if nested.exists():
            return nested
        return direct

    async def call_json(self, agent: str, llm_service: Any, *, system_prompt: str, prompt: str, response_parser: Callable[[str], dict[str, Any]]=extract_json_object, **kwargs: Any) -> tuple[dict[str, Any], AgentCallRecord]:
        request = self._request_payload(agent, system_prompt, prompt, kwargs, response_parser)
        request_sha256 = hashlib.sha256(_canonical_json(request).encode('utf-8')).hexdigest()
        relative = self._relative_path(agent, request_sha256)
        output_path = self.output_dir / relative
        replay_path = self._replay_path(relative)
        if replay_path is not None:
            if not replay_path.exists():
                raise FileNotFoundError(f'Missing replay response for {agent}: {replay_path}')
            cached = json.loads(replay_path.read_text(encoding='utf-8'))
            if cached.get('request_sha256') != request_sha256:
                raise ValueError(f'Replay request hash mismatch: {replay_path}')
            if cached.get('status') != 'ok':
                raise RuntimeError(f'Replay cache contains failed {agent} call: {replay_path}')
            data = cached.get('parsed_response')
            if not isinstance(data, dict):
                raise ValueError(f'Replay cache response is not a JSON object: {replay_path}')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding='utf-8')
            stable_path = str(Path('agent_cache') / relative).replace('\\', '/')
            record = AgentCallRecord(agent, request_sha256, stable_path, 'ok')
            self.records.append(record)
            return (data, record)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response, _history = await llm_service.send_message(system_prompt=system_prompt, prompt=prompt, response_model=None, **kwargs)
            raw_response = str(response)
            data = response_parser(raw_response)
            payload = {'schema_version': 1, 'agent': agent, 'request_sha256': request_sha256, 'status': 'ok', 'request': request, 'raw_response': raw_response, 'parsed_response': data}
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            stable_path = str(Path('agent_cache') / relative).replace('\\', '/')
            record = AgentCallRecord(agent, request_sha256, stable_path, 'ok')
            self.records.append(record)
            return (data, record)
        except Exception as exc:
            payload = {'schema_version': 1, 'agent': agent, 'request_sha256': request_sha256, 'status': 'failed', 'request': request, 'error_type': type(exc).__name__, 'error': str(exc)}
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            stable_path = str(Path('agent_cache') / relative).replace('\\', '/')
            record = AgentCallRecord(agent, request_sha256, stable_path, 'failed')
            self.records.append(record)
            raise
