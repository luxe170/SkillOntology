from __future__ import annotations
import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any
import httpx
import numpy as np
from openai import OpenAI

def _openai_model_name(model: str) -> str:
    model = (model or 'text-embedding-3-large').strip()
    return model.split('/', 1)[1] if model.startswith('openai/') else model

def _normalize_proxy_url(proxy_url: str) -> str:
    proxy_url = (proxy_url or '').strip()
    if proxy_url.lower() in {'none', 'off', 'disabled', 'direct'}:
        return ''
    return proxy_url

@dataclass
class OpenAIEmbeddingService:
    model: str = 'text-embedding-3-large'
    embedding_dim: int = 3072
    batch_size: int = 100
    timeout: float = 30.0
    api_key: str = ''
    base_url: str = ''
    proxy_url: str = ''
    max_retries: int = 4
    retry_base_delay: float = 1.0

    def __post_init__(self) -> None:
        self.model = _openai_model_name(self.model)

    def _client(self) -> OpenAI:
        kwargs: dict[str, Any] = {'timeout': self.timeout, 'max_retries': 0}
        if self.api_key:
            kwargs['api_key'] = self.api_key
        if self.base_url:
            kwargs['base_url'] = self.base_url
        if self.proxy_url:
            kwargs['http_client'] = httpx.Client(proxy=self.proxy_url, timeout=self.timeout)
        return OpenAI(**kwargs)

    def _encode_batch(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(self.max_retries + 1):
            client = self._client()
            try:
                response = client.embeddings.create(model=self.model, input=batch)
                return [item.embedding for item in response.data]
            except Exception:
                if attempt >= self.max_retries:
                    raise
                delay = min(self.retry_base_delay * 2 ** attempt, 20.0)
                time.sleep(delay)
            finally:
                client.close()
        raise RuntimeError('OpenAI embeddings request failed without an exception')

    def _encode_sync(self, texts: list[str]) -> np.ndarray[Any, np.dtype[np.float32]]:
        vectors: list[list[float]] = []
        batch_size = max(1, int(self.batch_size or 1))
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            vectors.extend(self._encode_batch(batch))
        return np.asarray(vectors, dtype=np.float32)

    async def encode(self, texts: list[str], model: str | None=None) -> np.ndarray[Any, np.dtype[np.float32]]:
        if model is not None:
            original_model = self.model
            self.model = _openai_model_name(model)
            try:
                return await asyncio.to_thread(self._encode_sync, texts)
            finally:
                self.model = original_model
        return await asyncio.to_thread(self._encode_sync, texts)

def build_embedding_service(config=None):
    config = config or {}
    model = config.get('model', os.environ.get('SKILLONTOLOGY_EMBEDDING_MODEL', 'text-embedding-3-large'))
    return OpenAIEmbeddingService(model=model, embedding_dim=int(config.get('dimension', 3072)), api_key=os.environ.get(config.get('api_key_env', 'OPENAI_API_KEY'), ''), base_url=config.get('base_url', os.environ.get('OPENAI_EMBEDDING_BASE_URL', 'https://api.openai.com/v1')), proxy_url=config.get('proxy_url', os.environ.get('SKILLONTOLOGY_EMBEDDING_PROXY', '')))
