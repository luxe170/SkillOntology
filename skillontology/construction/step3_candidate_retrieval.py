from __future__ import annotations
import hashlib
import json
from typing import Any
import numpy as np
from skillontology.runtime.agent_cache import AgentCallCache
from skillontology.runtime.llm_json import call_json
from skillontology.construction.prompts import CANDIDATE_FILTER_SYSTEM, CANDIDATE_FILTER_USER
from skillontology.schema import CandidateMetaOp, MetaOp, overlap_score

class EmbeddingCache:

    def __init__(self, embedding_service: Any | None) -> None:
        self.embedding_service = embedding_service
        self._vectors: dict[str, np.ndarray] = {}

    @staticmethod
    def _key(prefix: str, item_id: str, text: str) -> str:
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return f'{prefix}:{digest}'

    async def encode_items(self, items: list[tuple[str, str, str]]) -> list[np.ndarray]:
        if self.embedding_service is None or not items:
            return []
        keys = [self._key(prefix, item_id, text) for prefix, item_id, text in items]
        missing_positions = [index for index, key in enumerate(keys) if key not in self._vectors]
        if missing_positions:
            missing_texts = [items[index][2] for index in missing_positions]
            vectors = np.asarray(await self.embedding_service.encode(missing_texts), dtype=np.float32)
            for index, vector in zip(missing_positions, vectors):
                self._vectors[keys[index]] = np.asarray(vector, dtype=np.float32)
        return [self._vectors[key] for key in keys]

def rule_candidate_score(new_metaop: CandidateMetaOp, existing: MetaOp) -> float:
    action_object, io, tool = rule_overlap_components(new_metaop, existing)
    return 0.55 * action_object + 0.3 * io + 0.15 * tool

def rule_overlap_components(new_metaop: CandidateMetaOp, existing: MetaOp) -> tuple[float, float, float]:
    action_object = 0.5 * overlap_score(new_metaop.action, existing.action) + 0.5 * overlap_score(new_metaop.object, existing.object)
    io = 0.5 * overlap_score(new_metaop.inputs, existing.inputs) + 0.5 * overlap_score(new_metaop.outputs, existing.outputs)
    tool = overlap_score(new_metaop.tools, existing.tools)
    return (action_object, io, tool)

async def embedding_similarity(embedding_service: Any | None, new_metaop: CandidateMetaOp, candidates: list[MetaOp], *, embedding_cache: EmbeddingCache | None=None) -> dict[str, float]:
    if embedding_service is None or not candidates:
        return {}
    try:
        if embedding_cache is not None:
            items = [('candidate', new_metaop.local_id or new_metaop.name, new_metaop.to_embedding_text()), *[('metaop', item.metaop_id, item.to_embedding_text()) for item in candidates]]
            vectors = np.asarray(await embedding_cache.encode_items(items), dtype=np.float32)
        else:
            texts = [new_metaop.to_embedding_text(), *(item.to_embedding_text() for item in candidates)]
            vectors = np.asarray(await embedding_service.encode(texts), dtype=np.float32)
    except Exception:
        return {}
    query = vectors[0]
    matrix = vectors[1:]
    denom = np.linalg.norm(matrix, axis=1) * max(float(np.linalg.norm(query)), 1e-12)
    sims = np.dot(matrix, query) / np.maximum(denom, 1e-12)
    return {candidate.metaop_id: float(score) for candidate, score in zip(candidates, sims)}

async def retrieve_candidate_metaops(new_metaop: CandidateMetaOp, ontology: list[MetaOp], *, embedding_service: Any | None=None, embedding_cache: EmbeddingCache | None=None, candidate_top_k: int=8) -> list[tuple[MetaOp, float]]:
    if not ontology:
        return []
    sim = await embedding_similarity(embedding_service, new_metaop, ontology, embedding_cache=embedding_cache)
    scored: list[tuple[MetaOp, float]] = []
    for item in ontology:
        emb_score = sim.get(item.metaop_id, 0.0)
        action_object, io, tool = rule_overlap_components(new_metaop, item)
        score = 0.55 * emb_score + 0.2 * action_object + 0.15 * io + 0.1 * tool if sim else rule_candidate_score(new_metaop, item)
        scored.append((item, float(score)))
    scored.sort(key=lambda row: row[1], reverse=True)
    return scored[:max(candidate_top_k, 1)]

async def llm_filter_candidate_metaops(new_metaop: CandidateMetaOp, candidates: list[tuple[MetaOp, float]], llm_service: Any | None, *, use_llm: bool=True, agent_cache: AgentCallCache | None=None) -> list[str]:
    if not candidates:
        return []
    if not use_llm or llm_service is None:
        return [item.metaop_id for item, _score in candidates]
    candidate_payload = [{**item.to_json_dict(), 'candidate_score': score} for item, score in candidates]
    try:
        prompt = CANDIDATE_FILTER_USER.format(new_metaop_json=json.dumps(new_metaop.to_json_dict(), ensure_ascii=False, indent=2), candidate_json=json.dumps(candidate_payload, ensure_ascii=False, indent=2))
        if agent_cache is not None:
            data, _record = await agent_cache.call_json('agent2_candidate_filter', llm_service, system_prompt=CANDIDATE_FILTER_SYSTEM, prompt=prompt, max_tokens=600)
        else:
            data = await call_json(llm_service, system_prompt=CANDIDATE_FILTER_SYSTEM, prompt=prompt, max_tokens=600)
        selected = data.get('selected_metaop_ids', [])
        selected_ids = [str(item) for item in selected if str(item)]
        allowed = {item.metaop_id for item, _score in candidates}
        return [item for item in selected_ids if item in allowed] or [candidates[0][0].metaop_id]
    except Exception:
        if agent_cache is not None and agent_cache.strict_failures:
            raise
        return [item.metaop_id for item, _score in candidates]
