"""Persistent transcript store backed by Chroma (via LangChain)."""

from __future__ import annotations

import threading
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def video_filter(video_ids: list[str] | None) -> dict[str, Any] | None:
    """Chroma ``where`` clause restricting results to the given videos (None means all videos)."""
    if not video_ids:
        return None
    if len(video_ids) == 1:
        return {"video_id": video_ids[0]}
    return {"video_id": {"$in": list(video_ids)}}


class TranscriptStore:
    """Thin wrapper over a Chroma collection that knows about videos and chunks."""

    def __init__(self, embeddings: Embeddings, persist_dir: str | None, collection_name: str):
        self._vs = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        self._lock = threading.Lock()
        self._version = 0  # bumped on every write so keyword indexes know when to rebuild

    @property
    def version(self) -> int:
        return self._version

    def has_video(self, video_id: str) -> bool:
        return bool(self._vs.get(where={"video_id": video_id}, limit=1)["ids"])

    def add(self, docs: list[Document]) -> None:
        if not docs:
            return
        with self._lock:
            self._vs.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
            self._version += 1

    def delete_video(self, video_id: str) -> int:
        with self._lock:
            ids = self._vs.get(where={"video_id": video_id})["ids"]
            if ids:
                self._vs.delete(ids=ids)
                self._version += 1
            return len(ids)

    def documents(self, video_ids: list[str] | None = None) -> list[Document]:
        """All chunks (optionally for some videos), in transcript order."""
        data = self._vs.get(where=video_filter(video_ids), include=["documents", "metadatas"])
        docs = [Document(page_content=text, metadata=meta) for text, meta in zip(data["documents"], data["metadatas"])]
        docs.sort(key=lambda d: (d.metadata["video_id"], d.metadata["chunk_index"]))
        return docs

    def similarity_search(self, query: str, k: int, video_ids: list[str] | None = None) -> list[Document]:
        return self._vs.similarity_search(query, k=k, filter=video_filter(video_ids))

    def list_videos(self) -> list[dict[str, Any]]:
        videos: dict[str, dict[str, Any]] = {}
        for meta in self._vs.get(include=["metadatas"])["metadatas"]:
            entry = videos.setdefault(
                meta["video_id"],
                {"video_id": meta["video_id"], "title": meta.get("title", meta["video_id"]), "chunks": 0, "duration": 0.0},
            )
            entry["chunks"] += 1
            entry["duration"] = max(entry["duration"], float(meta.get("end", 0.0)))
        return sorted(videos.values(), key=lambda v: v["title"].lower())

    def count(self) -> int:
        return len(self._vs.get(include=[])["ids"])
