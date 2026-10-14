from __future__ import annotations

from dataclasses import dataclass
import os

from .model_api import chat_completion_kwargs, resolve_chat_base_url, validate_backbone_config


@dataclass(frozen=True)
class ModelResponse:
    text: str
    usage: dict[str, int]


class ChatModel:
    def __init__(self, config):
        self.config = config
        self.model = config['model']['name']
        validate_backbone_config(config['model'])

    async def complete(self, messages, max_tokens=None):
        from openai import AsyncOpenAI
        model = dict(self.config['model'])
        if max_tokens is not None:
            model['max_completion_tokens'] = int(max_tokens)
        kwargs = chat_completion_kwargs(model, messages, seed=self.config.get('seed'))
        if model.get('sampling_policy') == 'provider_defaults' or model.get('omit_temperature'):
            kwargs.pop('temperature', None)
        if model.get('omit_seed'):
            kwargs.pop('seed', None)
        if 'enable_thinking' in model:
            kwargs.setdefault('extra_body', {})['enable_thinking'] = bool(model['enable_thinking'])
        key = os.environ.get(model.get('api_key_env', 'SKILLONTOLOGY_CHAT_API_KEY'), '')
        if not key:
            raise ValueError('Missing chat API key environment variable')
        async with AsyncOpenAI(api_key=key, base_url=resolve_chat_base_url(self.config),
                               timeout=float(model.get('timeout_seconds', 120)), max_retries=int(model.get('max_retries', 3))) as client:
            response = await client.chat.completions.create(**kwargs)
        if response.usage is None:
            raise ValueError('Chat API omitted token usage')
        usage = {key: int(getattr(response.usage, key)) for key in ('prompt_tokens','completion_tokens','total_tokens')}
        if usage['prompt_tokens'] + usage['completion_tokens'] != usage['total_tokens']:
            raise ValueError('Chat token usage is inconsistent')
        return ModelResponse(response.choices[0].message.content or '', usage)

    async def send_message(self, *, system_prompt, prompt, response_model=None, max_tokens=2600, **kwargs):
        if response_model is not None:
            raise ValueError('Structured response models are not supported by this transport')
        response = await self.complete([{'role':'system','content':system_prompt},{'role':'user','content':prompt}], max_tokens)
        return response.text, []
