from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any
MODEL_ENV = 'SKILLONTOLOGY_CHAT_MODEL'
BASE_URL_ENV = 'SKILLONTOLOGY_CHAT_BASE_URL'
API_KEY_ENV = 'SKILLONTOLOGY_CHAT_API_KEY'
API_KEY_FILE_ENV = 'SKILLONTOLOGY_CHAT_API_KEY_FILE'
REASONING_EFFORT_ENV = 'SKILLONTOLOGY_CHAT_REASONING_EFFORT'
REQUEST_PROFILE_ENV = 'SKILLONTOLOGY_CHAT_REQUEST_PROFILE'

def _env(name: str) -> str:
    return os.environ.get(name, '').strip()

def apply_chat_overrides(config: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(config)
    model = resolved.setdefault('model', {})
    if (value := _env(MODEL_ENV)):
        model['name'] = value
    if (value := _env(BASE_URL_ENV)):
        model['base_url'] = value.rstrip('/')
    if (value := _env(REASONING_EFFORT_ENV)):
        model['reasoning_effort'] = value
    if (value := _env(REQUEST_PROFILE_ENV)):
        model['request_profile'] = value
    return resolved

def resolve_chat_base_url(config: dict[str, Any]) -> str:
    model = config.get('model', {})
    if (value := _env(BASE_URL_ENV)):
        return value.rstrip('/')
    if (value := str(model.get('base_url') or '').strip()):
        return value.rstrip('/')
    env_name = str(model.get('base_url_env') or 'BASE_URL').strip()
    if (value := os.environ.get(env_name, '').strip()):
        return value.rstrip('/')
    raise RuntimeError(f'Chat API base URL is missing; set {BASE_URL_ENV} or {env_name}')

def _split_secret_values(value: str) -> list[str]:
    return [item.strip().strip('"\'') for item in re.split('[,;\\r\\n]+', value) if item.strip()]

def read_chat_api_keys(fallback_path: str | Path | None=None, *, model_name: str | None=None) -> list[str]:
    direct = _env(API_KEY_ENV)
    if direct:
        return list(dict.fromkeys(_split_secret_values(direct)))
    configured_path = _env(API_KEY_FILE_ENV)
    source = Path(configured_path or str(fallback_path or '')).expanduser()
    if not str(source) or not source.is_file():
        raise RuntimeError(f'Chat API key is missing; set {API_KEY_ENV} or {API_KEY_FILE_ENV}')
    metaop_values: list[str] = []
    openai_values: list[str] = []
    deepseek_values: list[str] = []
    generic_values: list[str] = []
    unlabelled: list[str] = []
    for raw_line in source.read_text(encoding='utf-8-sig', errors='ignore').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            label, candidate = line.split('=', 1)
            normalized_label = label.strip().upper()
            if 'API_KEY' in normalized_label or normalized_label.endswith('TOKEN'):
                values = _split_secret_values(candidate)
                if normalized_label.startswith('SKILLONTOLOGY_CHAT_'):
                    metaop_values.extend(values)
                elif normalized_label.startswith('OPENAI_'):
                    openai_values.extend(values)
                elif normalized_label.startswith('DEEPSEEK_'):
                    deepseek_values.extend(values)
                else:
                    generic_values.extend(values)
            continue
        unlabelled.extend(_split_secret_values(line))
    configured_model = str(model_name or _env(MODEL_ENV)).lower().rsplit('/', 1)[-1]
    if metaop_values:
        selected = metaop_values
    elif configured_model.startswith(('gpt-', 'o1', 'o3', 'o4')) and openai_values:
        selected = openai_values
    elif configured_model.startswith('deepseek') and deepseek_values:
        selected = deepseek_values
    else:
        selected = generic_values or deepseek_values or openai_values or unlabelled
    values = list(dict.fromkeys(selected))
    if not values:
        raise RuntimeError(f'No API keys found in configured file: {source}')
    return values

def request_profile(model_config: dict[str, Any]) -> str:
    explicit = str(model_config.get('request_profile') or '').strip().lower()
    if explicit:
        return explicit
    model = str(model_config.get('name') or '').strip().lower().rsplit('/', 1)[-1]
    if model.startswith(('gpt-5', 'gpt-6', 'o1', 'o3', 'o4')):
        return 'openai_reasoning'
    return 'openai_compatible'

def chat_completion_kwargs(model_config: dict[str, Any], messages: list[dict[str, str]], *, seed: int | None=None) -> dict[str, Any]:
    profile = request_profile(model_config)
    kwargs: dict[str, Any] = {'model': str(model_config['name']), 'messages': messages}
    max_tokens = int(model_config.get('max_completion_tokens', 0) or 0)
    if profile == 'openai_reasoning':
        if max_tokens > 0:
            kwargs['max_completion_tokens'] = max_tokens
        effort = str(model_config.get('reasoning_effort') or 'none').strip().lower()
        if effort:
            kwargs['reasoning_effort'] = effort
        return kwargs
    kwargs['temperature'] = float(model_config.get('temperature', 0.0))
    if max_tokens > 0:
        kwargs['max_tokens'] = max_tokens
    if seed is not None:
        kwargs['seed'] = int(seed)
    effort = str(model_config.get('reasoning_effort') or '').strip().lower()
    if effort:
        kwargs['reasoning_effort'] = effort
    thinking = str(model_config.get('thinking', 'provider_default')).strip().lower()
    if thinking in {'enabled', 'disabled'}:
        kwargs['extra_body'] = {'thinking': {'type': thinking}}
    return kwargs

def validate_backbone_config(model_config: dict[str, Any]) -> None:
    if not str(model_config.get('name') or '').strip():
        raise ValueError('Chat model name must be configured')
    effort = str(model_config.get('reasoning_effort') or '').strip().lower()
    if effort and effort not in {'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'}:
        raise ValueError(f'Unsupported reasoning_effort: {effort}')
    profile = request_profile(model_config)
    if profile not in {'openai_compatible', 'openai_reasoning'}:
        raise ValueError(f'Unsupported chat request_profile: {profile}')
