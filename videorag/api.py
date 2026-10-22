"""FastAPI service exposing ingestion, cited question answering and summaries."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .pipeline import LLMNotConfigured, VideoNotFound, VideoRAG, build_pipeline
from .transcripts import InvalidVideoURL, TranscriptUnavailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("videorag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests inject a pipeline built from fakes; otherwise load the real models once at startup.
    if getattr(app.state, "pipeline", None) is None:
        app.state.pipeline = build_pipeline(get_settings())
    yield


app = FastAPI(
    title="VideoRAG API",
    version=__version__,
    description="Ask questions about YouTube videos and get answers cited to the exact timestamp.",
    lifespan=lifespan,
)


def get_pipeline(request: Request) -> VideoRAG:
    return request.app.state.pipeline


# -- schemas ---------------------------------------------------------------


class IngestRequest(BaseModel):
    url: str = Field(..., description="YouTube URL (watch, youtu.be, shorts, embed) or an 11-character video ID",
                     examples=["https://www.youtube.com/watch?v=aircAruvnKk"])


class IngestResponse(BaseModel):
    video_id: str
    title: str
    chunks: int
    already_indexed: bool


class VideoInfo(BaseModel):
    video_id: str
    title: str
    chunks: int
    duration: float = Field(description="Approximate length in seconds (end of the last caption)")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    video_ids: list[str] | None = Field(None, description="Restrict the search to these videos; omit to search all")
    top_k: int | None = Field(None, ge=1, le=10, description="Number of transcript chunks to give the LLM")


class CitationOut(BaseModel):
    index: int = Field(description="The [n] number used in the answer")
    video_id: str
    title: str
    start: float
    end: float
    timestamp: str
    url: str = Field(description="YouTube link that starts playback at this chunk")
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut] = Field(description="Sources the answer cites")
    sources: list[CitationOut] = Field(description="All chunks retrieved for the question")


class SummaryResponse(BaseModel):
    video_id: str
    summary: str


class HealthResponse(BaseModel):
    status: str
    version: str
    videos: int
    chunks: int
    llm_configured: bool


# -- error mapping ---------------------------------------------------------


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, InvalidVideoURL):
        return HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    if isinstance(exc, TranscriptUnavailable):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc))
    if isinstance(exc, VideoNotFound):
        return HTTPException(status.HTTP_404_NOT_FOUND, f"Video not indexed: {exc}")
    if isinstance(exc, LLMNotConfigured):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    log.exception("Unexpected error")
    return HTTPException(status.HTTP_502_BAD_GATEWAY, "The language model request failed. Try again shortly.")


# -- routes ----------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(pipeline: VideoRAG = Depends(get_pipeline)):
    return HealthResponse(
        status="ok",
        version=__version__,
        videos=len(pipeline.list_videos()),
        chunks=pipeline.store.count(),
        llm_configured=pipeline.llm is not None,
    )


@app.post("/videos", response_model=IngestResponse, tags=["videos"],
          responses={201: {"description": "Indexed"}, 200: {"description": "Already indexed"}})
def ingest_video(body: IngestRequest, response: Response, pipeline: VideoRAG = Depends(get_pipeline)):
    """Fetch a video's transcript, chunk it with timestamps, and index it for hybrid search."""
    try:
        result = pipeline.ingest(body.url)
    except (InvalidVideoURL, TranscriptUnavailable) as exc:
        raise _http_error(exc) from exc
    response.status_code = status.HTTP_200_OK if result.already_indexed else status.HTTP_201_CREATED
    return IngestResponse(**result.__dict__)


@app.get("/videos", response_model=list[VideoInfo], tags=["videos"])
def list_videos(pipeline: VideoRAG = Depends(get_pipeline)):
    return pipeline.list_videos()


@app.delete("/videos/{video_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["videos"])
def delete_video(video_id: str, pipeline: VideoRAG = Depends(get_pipeline)):
    try:
        pipeline.delete(video_id)
    except VideoNotFound as exc:
        raise _http_error(exc) from exc


@app.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(body: AskRequest, pipeline: VideoRAG = Depends(get_pipeline)):
    """Hybrid BM25 + vector retrieval, cross-encoder reranking, then an answer with [n] citations."""
    try:
        result = pipeline.ask(body.question, video_ids=body.video_ids, top_k=body.top_k)
    except Exception as exc:
        raise _http_error(exc) from exc
    return AskResponse(
        question=result.question,
        answer=result.answer,
        citations=[c.__dict__ for c in result.citations],
        sources=[c.__dict__ for c in result.sources],
    )


@app.post("/videos/{video_id}/summary", response_model=SummaryResponse, tags=["qa"])
def summarize(video_id: str, pipeline: VideoRAG = Depends(get_pipeline)):
    """Map-reduce summary with timestamps for each main point."""
    try:
        summary = pipeline.summarize(video_id)
    except Exception as exc:
        raise _http_error(exc) from exc
    return SummaryResponse(video_id=video_id, summary=summary)
