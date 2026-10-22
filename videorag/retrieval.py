"""Hybrid retrieval: BM25 keyword search + dense vector search, fused with RRF, then cross-encoder reranking."""

from __future__ import annotations

import re
import threading
from typing import Protocol

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .store import TranscriptStore

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i if in into is it its of on or so that the their them then "
    "there these they this to was we were what when where which who why will with you your do does did how can".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens without common stopwords, used by BM25."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class Reranker(Protocol):
    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]: ...


class CrossEncoderReranker:
    """Scores (question, chunk) pairs with a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        if not text_pairs:
            return []
        return [float(s) for s in self.model.predict(text_pairs, show_progress_bar=False)]


class KeywordIndex:
    """BM25 (Okapi) over a fixed set of chunks."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d.page_content) or ["_"] for d in docs])

    def search(self, terms: list[str], k: int) -> list[Document]:
        scores = self.bm25.get_scores(terms)
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in best if scores[i] > 0]


def reciprocal_rank_fusion(ranked_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Merge several rankings into one using Reciprocal Rank Fusion.

    Each document scores sum(1 / (k + rank)) over the lists it appears in, so a chunk ranked
    well by both keyword and semantic search beats one ranked first by only one of them. RRF
    needs no score calibration, which matters because BM25 and cosine scores are not comparable.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in ranked_lists:
        for rank, doc in enumerate(ranking, start=1):
            key = doc.metadata["chunk_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs.setdefault(key, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [_with_meta(docs[key], rrf_score=round(scores[key], 6)) for key in ordered]


def _with_meta(doc: Document, **extra) -> Document:
    return Document(page_content=doc.page_content, metadata={**doc.metadata, **extra})


class HybridRetriever:
    """Retrieves the best transcript chunks for a question.

    1. Dense: top-N chunks by embedding similarity (Chroma).
    2. Sparse: top-N chunks by BM25 over the same chunks (exact names, numbers, jargon).
    3. Fuse both rankings with RRF.
    4. Rerank the fused candidates with a cross-encoder that reads question and chunk together.
    """

    def __init__(
        self,
        store: TranscriptStore,
        reranker: Reranker | None = None,
        candidates_per_retriever: int = 20,
        rerank_candidates: int = 20,
        rrf_k: int = 60,
    ):
        self.store = store
        self.reranker = reranker
        self.candidates_per_retriever = candidates_per_retriever
        self.rerank_candidates = rerank_candidates
        self.rrf_k = rrf_k
        self._bm25_cache: dict[tuple, KeywordIndex] = {}
        self._lock = threading.Lock()

    def _bm25(self, video_ids: list[str] | None) -> KeywordIndex | None:
        """Keyword index for the given videos, rebuilt only when the store has changed."""
        key = (self.store.version, tuple(sorted(video_ids)) if video_ids else None)
        with self._lock:
            if key not in self._bm25_cache:
                docs = self.store.documents(video_ids)
                if not docs:
                    return None
                # Indexes from older store versions are stale; drop them to keep the cache small.
                self._bm25_cache = {k: v for k, v in self._bm25_cache.items() if k[0] == self.store.version}
                self._bm25_cache[key] = KeywordIndex(docs)
            return self._bm25_cache[key]

    def dense(self, query: str, video_ids: list[str] | None = None) -> list[Document]:
        return self.store.similarity_search(query, k=self.candidates_per_retriever, video_ids=video_ids)

    def sparse(self, query: str, video_ids: list[str] | None = None) -> list[Document]:
        """BM25 ranking, keeping only chunks that share at least one term with the query."""
        index = self._bm25(video_ids)
        terms = tokenize(query)
        if index is None or not terms:
            return []
        return index.search(terms, self.candidates_per_retriever)

    def retrieve(
        self,
        query: str,
        video_ids: list[str] | None = None,
        top_k: int = 5,
        mode: str = "hybrid_rerank",
    ) -> list[Document]:
        """Return the ``top_k`` best chunks. ``mode`` is one of "dense", "hybrid", "hybrid_rerank"."""
        dense = [_with_meta(d, dense_rank=i) for i, d in enumerate(self.dense(query, video_ids), start=1)]
        if mode == "dense":
            return dense[:top_k]

        sparse = [_with_meta(d, bm25_rank=i) for i, d in enumerate(self.sparse(query, video_ids), start=1)]
        ranks = {d.metadata["chunk_id"]: {"dense_rank": d.metadata["dense_rank"]} for d in dense}
        for d in sparse:
            ranks.setdefault(d.metadata["chunk_id"], {})["bm25_rank"] = d.metadata["bm25_rank"]

        fused = reciprocal_rank_fusion([dense, sparse], k=self.rrf_k)
        fused = [
            Document(
                page_content=d.page_content,
                metadata={**{k: v for k, v in d.metadata.items() if k not in ("dense_rank", "bm25_rank")},
                          **ranks[d.metadata["chunk_id"]]},
            )
            for d in fused
        ]
        if mode == "hybrid" or self.reranker is None:
            return fused[:top_k]

        candidates = fused[: self.rerank_candidates]
        scores = self.reranker.score([(query, d.page_content) for d in candidates])
        reranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [_with_meta(d, rerank_score=round(float(s), 4)) for d, s in reranked[:top_k]]
