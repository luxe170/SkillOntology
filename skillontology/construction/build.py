from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from skillontology.retrieval.index import INDEX_META_FILE, build_metaop_vector_index
from skillontology.construction.step6_ontology_memory import OntologyMemory, build_ontology_memory

async def build_metaop_workspace(skills_dir: str | Path, output_dir: str | Path, *, llm_service: Any | None, embedding_service: Any | None, use_llm: bool=True, candidate_top_k: int=8, max_conflict_rounds: int=0, snippet_chars: int=900, agent_workers: int=1, candidate_filter_llm: bool=True, canonical_merge_llm: bool=True, use_full_markdown: bool=False, domain_ontology: str | Path | None=None, ontology_guided_extraction: bool | None=None, extraction_policy_output: str | Path | None=None, replay_agent_cache: str | Path | None=None, agent_mode: str | None=None) -> OntologyMemory:
    memory = await build_ontology_memory(skills_dir, output_dir, llm_service=llm_service, embedding_service=embedding_service, use_llm=use_llm, candidate_top_k=candidate_top_k, max_conflict_rounds=max_conflict_rounds, snippet_chars=snippet_chars, agent_workers=agent_workers, candidate_filter_llm=candidate_filter_llm, canonical_merge_llm=canonical_merge_llm, use_full_markdown=use_full_markdown, domain_ontology=domain_ontology, ontology_guided_extraction=ontology_guided_extraction, extraction_policy_output=extraction_policy_output, replay_agent_cache=replay_agent_cache, agent_mode=agent_mode)
    if embedding_service is not None:
        await build_metaop_vector_index(output_dir, embedding_service, source_skills_dir=skills_dir)
    _write_workspace_meta(skills_dir, output_dir, memory, embedding_service)
    return memory

async def build_index_for_existing_workspace(meta_dir: str | Path, embedding_service: Any | None, *, source_skills_dir: str | Path | None=None) -> None:
    await build_metaop_vector_index(meta_dir, embedding_service, source_skills_dir=source_skills_dir)
    memory = OntologyMemory.load(meta_dir)
    _write_workspace_meta(source_skills_dir or '', meta_dir, memory, embedding_service)

def _write_workspace_meta(skills_dir: str | Path, output_dir: str | Path, memory: OntologyMemory, embedding_service: Any) -> None:
    path = Path(output_dir)
    meta_path = path / INDEX_META_FILE
    existing: dict[str, Any] = {}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            existing = {}
    payload = {**existing, 'source_skills_dir': str(skills_dir), 'skill_count': len(memory.skill_records), 'metaop_count': len(memory.metaops), 'edge_count': len(memory.metaop_edges), 'embedding_model': getattr(embedding_service, 'model', '') or getattr(embedding_service, 'model_name', '') or ''}
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
