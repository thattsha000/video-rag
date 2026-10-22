"""The VideoRAG pipeline: ingest videos, answer questions with citations, summarise videos."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from .chunking import chunk_segments
from .config import Settings
from .generation import NOT_FOUND, Citation, answer_chain, cited_indices, format_sources, summarize_documents
from .retrieval import CrossEncoderReranker, HybridRetriever, Reranker
from .store import TranscriptStore
from .transcripts import Segment, extract_video_id, fetch_title, fetch_transcript

log = logging.getLogger(__name__)


class VideoNotFound(LookupError):
    """The video has not been indexed."""


class LLMNotConfigured(RuntimeError):
    """No LLM is available (usually GROQ_API_KEY is missing)."""


@dataclass
class IngestResult:
    video_id: str
    title: str
    chunks: int
    already_indexed: bool


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation]  # sources the answer cites, in citation order
    sources: list[Citation] = field(default_factory=list)  # every chunk the LLM was shown


class VideoRAG:
    def __init__(
        self,
        settings: Settings,
        store: TranscriptStore,
        retriever: HybridRetriever,
        llm: BaseChatModel | None,
        transcript_fetcher: Callable[[str, list[str]], list[Segment]] = fetch_transcript,
        title_fetcher: Callable[[str], str] = fetch_title,
    ):
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.llm = llm
        self._fetch_transcript = transcript_fetcher
        self._fetch_title = title_fetcher

    # -- indexing ---------------------------------------------------------

    def ingest(self, url_or_id: str) -> IngestResult:
        video_id = extract_video_id(url_or_id)
        if self.store.has_video(video_id):
            existing = next(v for v in self.store.list_videos() if v["video_id"] == video_id)
            return IngestResult(video_id, existing["title"], existing["chunks"], already_indexed=True)

        segments = self._fetch_transcript(video_id, self.settings.transcript_languages)
        title = self._fetch_title(video_id)
        docs = chunk_segments(
            segments,
            video_id=video_id,
            title=title,
            target_chars=self.settings.chunk_target_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        self.store.add(docs)
        log.info("Indexed %s (%s): %d chunks", video_id, title, len(docs))
        return IngestResult(video_id, title, len(docs), already_indexed=False)

    def delete(self, video_id: str) -> None:
        if not self.store.delete_video(video_id):
            raise VideoNotFound(video_id)

    def list_videos(self) -> list[dict]:
        return self.store.list_videos()

    # -- question answering ----------------------------------------------

    def _require_llm(self) -> BaseChatModel:
        if self.llm is None:
            raise LLMNotConfigured("Set GROQ_API_KEY to enable answers and summaries.")
        return self.llm

    def _check_videos(self, video_ids: list[str] | None) -> list[str] | None:
        if not video_ids:
            return None
        known = {v["video_id"] for v in self.store.list_videos()}
        missing = [v for v in video_ids if v not in known]
        if missing:
            raise VideoNotFound(", ".join(missing))
        return video_ids

    def ask(self, question: str, video_ids: list[str] | None = None, top_k: int | None = None) -> Answer:
        llm = self._require_llm()
        video_ids = self._check_videos(video_ids)
        docs = self.retriever.retrieve(question, video_ids=video_ids, top_k=top_k or self.settings.top_k)
        if not docs:
            return Answer(question=question, answer=NOT_FOUND, citations=[], sources=[])

        text = answer_chain(llm).invoke({"question": question, "sources": format_sources(docs)}).strip()
        sources = [Citation.from_document(i, d) for i, d in enumerate(docs, start=1)]
        cited = [sources[i - 1] for i in cited_indices(text, len(sources))]
        return Answer(question=question, answer=text, citations=cited, sources=sources)

    def summarize(self, video_id: str) -> str:
        llm = self._require_llm()
        docs = self.store.documents([video_id])
        if not docs:
            raise VideoNotFound(video_id)
        title = docs[0].metadata.get("title", video_id)
        return summarize_documents(llm, docs, title, self.settings.summary_window_chars)


def build_pipeline(settings: Settings) -> VideoRAG:
    """Create the production pipeline: local embedding + reranker models, persistent Chroma, Groq LLM."""
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    store = TranscriptStore(embeddings, settings.chroma_dir, settings.collection_name)

    reranker: Reranker | None = CrossEncoderReranker(settings.reranker_model) if settings.use_reranker else None

    retriever = HybridRetriever(
        store,
        reranker=reranker,
        candidates_per_retriever=settings.candidates_per_retriever,
        rerank_candidates=settings.rerank_candidates,
        rrf_k=settings.rrf_k,
    )

    llm = None
    if settings.groq_api_key:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            max_retries=2,
        )
    else:
        log.warning("GROQ_API_KEY is not set: indexing works, but /ask and summaries will return 503.")

    return VideoRAG(settings, store, retriever, llm)
