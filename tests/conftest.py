"""Shared fixtures: a real pipeline wired to fakes, so tests need no network, API key or model downloads."""

from __future__ import annotations

import uuid

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from videorag.config import Settings
from videorag.pipeline import VideoRAG
from videorag.retrieval import HybridRetriever
from videorag.store import TranscriptStore
from videorag.transcripts import Segment

VIDEO_A = "aaaaaaaaaaa"
VIDEO_B = "bbbbbbbbbbb"

TRANSCRIPTS = {
    VIDEO_A: [
        "welcome to this lecture on neural networks",
        "a neuron holds a number called its activation",
        "the network has layers of neurons connected by weights",
        "gradient descent adjusts the weights to reduce the cost function",
        "backpropagation computes the gradient efficiently",
        "thanks for watching and see you next time",
    ],
    VIDEO_B: [
        "today we cook a classic margherita pizza",
        "the dough needs flour water yeast and salt",
        "let the dough rise for about two hours",
        "bake at the highest temperature your oven allows",
    ],
}
TITLES = {VIDEO_A: "Neural Networks 101", VIDEO_B: "Pizza Night"}


def fake_transcript(video_id: str, languages: list[str]) -> list[Segment]:
    from videorag.transcripts import TranscriptUnavailable

    if video_id not in TRANSCRIPTS:
        raise TranscriptUnavailable(f"No transcript available for {video_id}")
    return [Segment(text=t, start=i * 10.0, duration=9.0) for i, t in enumerate(TRANSCRIPTS[video_id])]


class KeywordReranker:
    """Stand-in cross-encoder: scores a chunk by how many question words it contains."""

    def score(self, text_pairs):
        return [float(sum(w in text for w in q.lower().split())) for q, text in text_pairs]


@pytest.fixture
def settings() -> Settings:
    # Small chunks so each fake transcript becomes several chunks.
    return Settings(groq_api_key=None, chunk_target_chars=60, chunk_overlap_chars=0, top_k=3, _env_file=None)


@pytest.fixture
def store() -> TranscriptStore:
    return TranscriptStore(DeterministicFakeEmbedding(size=32), None, f"test-{uuid.uuid4().hex}")


def make_pipeline(settings: Settings, store: TranscriptStore, responses: list[str] | None = None) -> VideoRAG:
    retriever = HybridRetriever(store, reranker=KeywordReranker(), candidates_per_retriever=10, rerank_candidates=10)
    llm = FakeListChatModel(responses=responses) if responses is not None else None
    return VideoRAG(settings, store, retriever, llm, transcript_fetcher=fake_transcript, title_fetcher=TITLES.get)


@pytest.fixture
def pipeline(settings, store) -> VideoRAG:
    return make_pipeline(settings, store, responses=["Gradient descent adjusts the weights [1]. See also [9]."])
