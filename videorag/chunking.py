"""Turning timed transcript segments into overlapping, timestamped LangChain documents."""

from __future__ import annotations

import re

from langchain_core.documents import Document

from .transcripts import Segment

# Caption noise that carries no meaning for retrieval, e.g. "[Music]", "[Applause]", "(laughs)".
_NOISE = re.compile(r"\[(?:music|applause|laughter|laughs|inaudible|silence)\]|\((?:laughs|music|applause)\)", re.I)
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _SPACES.sub(" ", _NOISE.sub(" ", text)).strip()


def chunk_segments(
    segments: list[Segment],
    video_id: str,
    title: str,
    target_chars: int = 1000,
    overlap_chars: int = 200,
) -> list[Document]:
    """Group caption segments into windows of roughly ``target_chars`` characters.

    Chunks never split a caption line, so every chunk keeps an exact start time. Consecutive
    chunks share about ``overlap_chars`` of trailing text so an answer that straddles a boundary
    can still be retrieved from one chunk.
    """
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if not 0 <= overlap_chars < target_chars:
        raise ValueError("overlap_chars must be between 0 and target_chars")

    lines = [(clean_text(s.text), s) for s in segments]
    lines = [(text, seg) for text, seg in lines if text]

    docs: list[Document] = []
    start = 0
    while start < len(lines):
        end = start
        size = 0
        while end < len(lines) and (size < target_chars or end == start):
            size += len(lines[end][0]) + 1
            end += 1

        window = lines[start:end]
        first, last = window[0][1], window[-1][1]
        index = len(docs)
        docs.append(
            Document(
                page_content=" ".join(text for text, _ in window),
                metadata={
                    "chunk_id": f"{video_id}:{index}",
                    "chunk_index": index,
                    "video_id": video_id,
                    "title": title,
                    "start": round(first.start, 2),
                    "end": round(last.start + last.duration, 2),
                },
            )
        )

        if end >= len(lines):
            break
        # Step back far enough to carry ~overlap_chars into the next chunk, but always move forward.
        next_start = end
        carried = 0
        while next_start - 1 > start and carried + len(lines[next_start - 1][0]) + 1 <= overlap_chars:
            next_start -= 1
            carried += len(lines[next_start][0]) + 1
        start = next_start

    return docs
