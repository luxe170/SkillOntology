from __future__ import annotations
import json
import re
from typing import Any
try:
    from json_repair import repair_json
except Exception:
    repair_json = None
_JSON_BLOCK = re.compile('```(?:json)?\\s*(.*?)```', re.IGNORECASE | re.DOTALL)

def extract_json_object(text: str) -> dict[str, Any]:
    text = str(text or '').strip()
    block = _JSON_BLOCK.search(text)
    if block:
        text = block.group(1).strip()
    if repair_json is not None:
        text = repair_json(text)
    else:
        match = re.search('\\{.*\\}', text, re.DOTALL)
        if match:
            text = match.group(0)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('LLM JSON response must be an object')
    return value

async def call_json(llm_service: Any, *, system_prompt: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
    response, _history = await llm_service.send_message(system_prompt=system_prompt, prompt=prompt, response_model=None, **kwargs)
    return extract_json_object(str(response))
