"""Runtime configuration, read from environment variables or a .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (Groq)
    groq_api_key: str | None = None
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Embeddings and reranking (run locally with sentence-transformers)
    embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

    # Vector store
    chroma_dir: str = "chroma_db"
    collection_name: str = "video_transcripts"

    # Chunking: transcript segments are grouped into windows of about this many characters
    chunk_target_chars: int = 1000
    chunk_overlap_chars: int = 200

    # Retrieval
    candidates_per_retriever: int = 20  # taken from BM25 and from the vector index
    rerank_candidates: int = 20  # fused candidates passed to the cross-encoder
    top_k: int = 5  # chunks given to the LLM as sources
    rrf_k: int = 60  # reciprocal rank fusion constant

    # Summaries: transcript is summarised in windows of this many characters, then combined
    summary_window_chars: int = 12000

    transcript_languages: list[str] = ["en"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
