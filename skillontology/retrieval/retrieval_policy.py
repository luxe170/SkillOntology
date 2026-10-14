from __future__ import annotations
from pathlib import Path
from typing import Any
from skillontology.retrieval.graph_retrieval import MetaOpGraphRetriever
from skillontology.schema import MetaOp, QueryDecomposition, QueryOperationNeed
class ActionCompatibilityPolicy:

    def should_filter_need(self, need: QueryOperationNeed) -> bool:
        return False

    def compatible_with_need(self, metaop: MetaOp, need: QueryOperationNeed) -> bool:
        return True

class PolicyMetaOpGraphRetriever(MetaOpGraphRetriever):

    def __init__(self, meta_dir: str | Path, *, compatibility_policy: ActionCompatibilityPolicy | None=None, embedding_service: Any | None=None, **kwargs: Any) -> None:
        super().__init__(meta_dir, embedding_service=embedding_service, **kwargs)
        self.compatibility_policy = compatibility_policy or ActionCompatibilityPolicy()

    async def _best_metaop_for_need(self, query: QueryDecomposition, need: QueryOperationNeed, *, seed_top_k: int, min_score: float=0.16, min_lexical_score: float=0.08) -> tuple[MetaOp, float] | None:
        if not self.compatibility_policy.should_filter_need(need):
            return await super()._best_metaop_for_need(query, need, seed_top_k=seed_top_k, min_score=min_score, min_lexical_score=min_lexical_score)
        seed_hits = await self.index.search(self._need_query_text(query, need), self.embedding_service, top_k=seed_top_k)
        dense_scores = {metaop.metaop_id: max(0.0, float(score)) for metaop, score, _rank in seed_hits}
        ranks = {metaop.metaop_id: rank for metaop, _score, rank in seed_hits}
        candidates: list[tuple[MetaOp, float]] = []
        for metaop in self.index.metaops:
            if not self.compatibility_policy.compatible_with_need(metaop, need):
                continue
            lexical_score = self._lexical_score_metaop_for_need(metaop, need)
            if lexical_score < min_lexical_score:
                continue
            score = self._score_metaop_for_need(metaop, need, lexical_score=lexical_score, dense_score=dense_scores.get(metaop.metaop_id, 0.0), rank=ranks.get(metaop.metaop_id))
            if score >= min_score:
                candidates.append((metaop, score))
        candidates.sort(key=lambda row: (-row[1], row[0].metaop_id))
        return candidates[0] if candidates else None
