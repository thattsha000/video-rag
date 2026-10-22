import pytest

from videorag.chunking import chunk_segments, clean_text
from videorag.transcripts import Segment


def segments(n: int, words: int = 5) -> list[Segment]:
    return [Segment(text=" ".join(f"w{i}x{j}" for j in range(words)), start=i * 3.0, duration=2.5) for i in range(n)]


def test_clean_text_removes_caption_noise():
    assert clean_text("[Music]  hello\n world [Applause]") == "hello world"


def test_chunks_cover_every_segment_in_order():
    segs = segments(40)
    docs = chunk_segments(segs, "vid", "Title", target_chars=100, overlap_chars=0)
    joined = " ".join(d.page_content for d in docs)
    assert joined == " ".join(s.text for s in segs)
    assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))


def test_chunk_metadata_has_exact_timestamps():
    segs = segments(10)
    docs = chunk_segments(segs, "vid", "Title", target_chars=60, overlap_chars=0)
    first = docs[0].metadata
    assert first["start"] == 0.0
    assert first["video_id"] == "vid" and first["title"] == "Title"
    assert first["chunk_id"] == "vid:0"
    for doc in docs:
        assert doc.metadata["end"] > doc.metadata["start"]


def test_overlap_repeats_text_but_always_advances():
    segs = segments(30)
    docs = chunk_segments(segs, "vid", "Title", target_chars=100, overlap_chars=40)
    assert len(docs) > 1
    for prev, nxt in zip(docs, docs[1:]):
        assert nxt.metadata["start"] > prev.metadata["start"]
        tail = prev.page_content.split()[-1]
        assert tail in nxt.page_content  # some text carried over


def test_long_single_segment_becomes_its_own_chunk():
    segs = [Segment(text="x" * 500, start=0, duration=5), Segment(text="short", start=5, duration=1)]
    docs = chunk_segments(segs, "vid", "Title", target_chars=100, overlap_chars=0)
    assert [d.page_content for d in docs] == ["x" * 500, "short"]


def test_empty_and_invalid_input():
    assert chunk_segments([], "vid", "T") == []
    with pytest.raises(ValueError):
        chunk_segments(segments(3), "vid", "T", target_chars=100, overlap_chars=100)
