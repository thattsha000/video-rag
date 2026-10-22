from langchain_core.documents import Document

from tests.conftest import VIDEO_A, VIDEO_B
from videorag.generation import cited_indices, format_sources, summary_windows
from videorag.retrieval import reciprocal_rank_fusion, tokenize


def doc(cid: str) -> Document:
    return Document(page_content=cid, metadata={"chunk_id": cid})


def test_rrf_prefers_documents_ranked_by_both_lists():
    dense = [doc("a"), doc("b"), doc("c")]
    sparse = [doc("c"), doc("d"), doc("b")]
    fused = [d.metadata["chunk_id"] for d in reciprocal_rank_fusion([dense, sparse], k=60)]
    # b and c appear in both lists, so they beat a and d which appear once each
    assert set(fused[:2]) == {"b", "c"}
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
    assert [d.metadata["chunk_id"] for d in reciprocal_rank_fusion([[doc("x")], []])] == ["x"]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the GRADIENT, of a cost-function?") == ["gradient", "cost", "function"]


def test_hybrid_finds_keyword_match_and_respects_video_filter(pipeline):
    pipeline.ingest(VIDEO_A)
    pipeline.ingest(VIDEO_B)

    docs = pipeline.retriever.retrieve("backpropagation gradient", top_k=3)
    assert "backpropagation" in docs[0].page_content
    assert "rerank_score" in docs[0].metadata

    only_b = pipeline.retriever.retrieve("backpropagation gradient", video_ids=[VIDEO_B], top_k=3)
    assert only_b and all(d.metadata["video_id"] == VIDEO_B for d in only_b)


def test_modes_return_expected_metadata(pipeline):
    pipeline.ingest(VIDEO_A)
    dense = pipeline.retriever.retrieve("weights", mode="dense", top_k=2)
    hybrid = pipeline.retriever.retrieve("weights", mode="hybrid", top_k=2)
    assert all("dense_rank" in d.metadata for d in dense)
    assert all("rrf_score" in d.metadata for d in hybrid)
    assert all("rerank_score" not in d.metadata for d in hybrid)


def test_bm25_index_rebuilds_after_new_video(pipeline):
    pipeline.ingest(VIDEO_A)
    assert not pipeline.retriever.sparse("pizza dough")
    pipeline.ingest(VIDEO_B)
    assert any("dough" in d.page_content for d in pipeline.retriever.sparse("pizza dough"))


def test_cited_indices_keeps_order_and_drops_out_of_range():
    answer = "First [2]. Then [1][3]. Also [2, 5] and [12:30] and [99]."
    assert cited_indices(answer, n_sources=5) == [2, 1, 3, 5]


def test_format_sources_numbers_and_timestamps():
    docs = [Document(page_content="hello", metadata={"video_id": "v", "title": "T", "start": 75.0})]
    assert format_sources(docs) == '[1] "T" at 1:15\nhello'


def test_summary_windows_split_on_size():
    docs = [Document(page_content="x" * 50, metadata={"start": i * 60.0}) for i in range(6)]
    windows = summary_windows(docs, window_chars=130)
    assert len(windows) == 3
    assert windows[0].startswith("[0:00]") and "[1:00]" in windows[0]
