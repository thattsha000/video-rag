"""Prompting the LLM: cited answers and timestamped summaries (LangChain LCEL chains)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .transcripts import format_timestamp, video_url

NOT_FOUND = "I couldn't find that in the indexed videos."

ANSWER_SYSTEM = f"""You answer questions about YouTube videos using only the numbered transcript excerpts provided.

Rules:
- Use only facts stated in the excerpts. Do not add outside knowledge.
- After every sentence that uses an excerpt, cite it with its number in square brackets, e.g. [2] or [1][3].
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Be concise: a direct answer first, then supporting detail if useful.
- Transcripts are auto-generated and may contain spelling errors; interpret them sensibly."""

ANSWER_HUMAN = "Transcript excerpts:\n\n{sources}\n\nQuestion: {question}"

MAP_SYSTEM = """You summarise one part of a YouTube video transcript.
Write 3-6 concise bullet points covering the main ideas in this part.
Start each bullet with the timestamp in square brackets where that idea begins, e.g. [12:30].
Use only the transcript. Timestamps appear in the text as [mm:ss] markers."""

REDUCE_SYSTEM = """You combine partial summaries of a single YouTube video into one summary.
Write a one-sentence overview, then 5-10 bullet points in chronological order.
Keep the [mm:ss] timestamp at the start of each bullet. Merge duplicates. Use only the partial summaries."""

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Citation:
    index: int
    video_id: str
    title: str
    start: float
    end: float
    timestamp: str
    url: str
    text: str

    @classmethod
    def from_document(cls, index: int, doc: Document) -> "Citation":
        meta = doc.metadata
        return cls(
            index=index,
            video_id=meta["video_id"],
            title=meta.get("title", meta["video_id"]),
            start=float(meta["start"]),
            end=float(meta["end"]),
            timestamp=format_timestamp(meta["start"]),
            url=video_url(meta["video_id"], meta["start"]),
            text=doc.page_content,
        )


def format_sources(docs: list[Document]) -> str:
    """Numbered excerpts the LLM cites by number."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = f'[{i}] "{meta.get("title", meta["video_id"])}" at {format_timestamp(meta["start"])}'
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def cited_indices(answer: str, n_sources: int) -> list[int]:
    """Source numbers the answer actually cites, in order of first use; out-of-range numbers are dropped."""
    seen: list[int] = []
    for group in _CITATION.findall(answer):
        for part in group.split(","):
            num = int(part)
            if 1 <= num <= n_sources and num not in seen:
                seen.append(num)
    return seen


def answer_chain(llm: BaseChatModel) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([("system", ANSWER_SYSTEM), ("human", ANSWER_HUMAN)])
    return prompt | llm | StrOutputParser()


def map_chain(llm: BaseChatModel) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([("system", MAP_SYSTEM), ("human", "Video: {title}\n\n{text}")])
    return prompt | llm | StrOutputParser()


def reduce_chain(llm: BaseChatModel) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([("system", REDUCE_SYSTEM), ("human", "Video: {title}\n\n{summaries}")])
    return prompt | llm | StrOutputParser()


def summary_windows(docs: list[Document], window_chars: int) -> list[str]:
    """Split a transcript (chunks in order) into windows of text with [mm:ss] markers, skipping overlap."""
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for doc in docs:
        piece = f"[{format_timestamp(doc.metadata['start'])}] {doc.page_content}"
        if current and size + len(piece) > window_chars:
            windows.append("\n".join(current))
            current, size = [], 0
        current.append(piece)
        size += len(piece)
    if current:
        windows.append("\n".join(current))
    return windows


def summarize_documents(llm: BaseChatModel, docs: list[Document], title: str, window_chars: int) -> str:
    """Map-reduce summary: summarise each window, then merge (one call if the video is short)."""
    windows = summary_windows(docs, window_chars)
    if not windows:
        return "This video has no transcript text to summarise."
    partials = map_chain(llm).batch([{"title": title, "text": w} for w in windows], config={"max_concurrency": 2})
    if len(partials) == 1:
        return partials[0]
    return reduce_chain(llm).invoke({"title": title, "summaries": "\n\n".join(partials)})
