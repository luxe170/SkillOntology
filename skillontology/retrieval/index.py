from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from skillontology.schema import MetaOp
from skillontology.construction.step6_ontology_memory import OntologyMemory
METAOPS_FILE = 'metaops.jsonl'
EDGES_FILE = 'metaop_edges.jsonl'
VECTORS_FILE = 'metaop_vectors.npy'
INDEX_META_FILE = 'index_meta.json'

def load_metaops(meta_dir: str | Path) -> list[MetaOp]:
    path = Path(meta_dir)
    metaops_path = path / METAOPS_FILE
    if metaops_path.exists():
        metaops: list[MetaOp] = []
        for line in metaops_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                metaops.append(MetaOp(**json.loads(line)))
        return metaops
    memory = OntologyMemory.load(path)
    return list(memory.metaops.values())

class MetaOpVectorIndex:

    def __init__(self, metaops: list[MetaOp], vectors: np.ndarray) -> None:
        if len(metaops) != len(vectors):
            raise ValueError('MetaOp count must match vector count')
        self.metaops = list(metaops)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.metaops_by_id = {item.metaop_id: item for item in self.metaops}
        self._normalized_vectors = self._normalize(self.vectors)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    @classmethod
    async def build(cls, meta_dir: str | Path, embedding_service: Any, *, source_skills_dir: str | Path | None=None) -> 'MetaOpVectorIndex':
        metaops = load_metaops(meta_dir)
        if not metaops:
            raise ValueError('Cannot build MetaOp index with zero MetaOps')
        vectors = np.asarray(await embedding_service.encode([item.to_embedding_text() for item in metaops]), dtype=np.float32)
        index = cls(metaops, vectors)
        index.save(meta_dir, source_skills_dir=source_skills_dir)
        return index

    @classmethod
    def load(cls, meta_dir: str | Path) -> 'MetaOpVectorIndex':
        path = Path(meta_dir)
        vectors_path = path / VECTORS_FILE
        if not vectors_path.exists():
            raise FileNotFoundError(f'{vectors_path} is missing. Build the vector index with skillontology before retrieval.')
        return cls(load_metaops(path), np.load(vectors_path))

    def save(self, meta_dir: str | Path, *, source_skills_dir: str | Path | None=None) -> None:
        path = Path(meta_dir)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / VECTORS_FILE, self.vectors)
        existing_meta: dict[str, Any] = {}
        meta_path = path / INDEX_META_FILE
        if meta_path.exists():
            try:
                existing_meta = json.loads(meta_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                existing_meta = {}
        payload = {**existing_meta, 'source_skills_dir': str(source_skills_dir or existing_meta.get('source_skills_dir', '')), 'metaop_count': len(self.metaops), 'embedding_dim': int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0, 'metaop_ids': [item.metaop_id for item in self.metaops], 'vector_file': VECTORS_FILE}
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    async def search(self, query_text: str, embedding_service: Any, *, top_k: int=8) -> list[tuple[MetaOp, float, int]]:
        if not self.metaops:
            return []
        query_vector = np.asarray(await embedding_service.encode([query_text]), dtype=np.float32)
        query = self._normalize(query_vector)[0]
        scores = self._normalized_vectors @ query
        order = np.argsort(-scores)[:min(max(1, int(top_k)), len(self.metaops))]
        return [(self.metaops[int(index)], float(scores[int(index)]), rank) for rank, index in enumerate(order, start=1)]

async def build_metaop_vector_index(meta_dir: str | Path, embedding_service: Any, *, source_skills_dir: str | Path | None=None) -> MetaOpVectorIndex:
    return await MetaOpVectorIndex.build(meta_dir, embedding_service, source_skills_dir=source_skills_dir)
