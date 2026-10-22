import pytest
from fastapi.testclient import TestClient

from tests.conftest import VIDEO_A, VIDEO_B, make_pipeline
from videorag.api import app


@pytest.fixture
def client(pipeline):
    app.state.pipeline = pipeline
    with TestClient(app) as c:
        yield c
    app.state.pipeline = None


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["videos"] == 0 and body["llm_configured"] is True


def test_ingest_list_and_reingest(client):
    r = client.post("/videos", json={"url": f"https://www.youtube.com/watch?v={VIDEO_A}"})
    assert r.status_code == 201
    assert r.json()["title"] == "Neural Networks 101" and r.json()["chunks"] > 1

    again = client.post("/videos", json={"url": f"https://youtu.be/{VIDEO_A}"})
    assert again.status_code == 200 and again.json()["already_indexed"] is True

    videos = client.get("/videos").json()
    assert [v["video_id"] for v in videos] == [VIDEO_A]


def test_ingest_errors(client):
    assert client.post("/videos", json={"url": "https://vimeo.com/1"}).status_code == 400
    assert client.post("/videos", json={"url": "ccccccccccc"}).status_code == 422  # no transcript


def test_ask_returns_answer_with_resolved_citations(client):
    client.post("/videos", json={"url": VIDEO_A})
    r = client.post("/ask", json={"question": "what does gradient descent adjust?", "video_ids": [VIDEO_A]})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Gradient descent")
    # [1] resolves to a real chunk; [9] is out of range and dropped
    assert [c["index"] for c in body["citations"]] == [1]
    cite = body["citations"][0]
    assert cite["video_id"] == VIDEO_A
    assert cite["url"] == f"https://www.youtube.com/watch?v={VIDEO_A}&t={int(cite['start'])}s"
    assert len(body["sources"]) == 3


def test_ask_validation_and_unknown_video(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "top_k": 50}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "video_ids": [VIDEO_B]}).status_code == 404


def test_ask_without_llm_returns_503(settings, store):
    app.state.pipeline = make_pipeline(settings, store, responses=None)
    with TestClient(app) as c:
        c.post("/videos", json={"url": VIDEO_A})
        assert c.post("/ask", json={"question": "anything"}).status_code == 503
    app.state.pipeline = None


def test_summary_and_delete(settings, store):
    app.state.pipeline = make_pipeline(settings, store, responses=["- [0:00] Neural networks intro"])
    with TestClient(app) as c:
        c.post("/videos", json={"url": VIDEO_A})
        r = c.post(f"/videos/{VIDEO_A}/summary")
        assert r.status_code == 200 and "Neural networks" in r.json()["summary"]

        assert c.delete(f"/videos/{VIDEO_A}").status_code == 204
        assert c.delete(f"/videos/{VIDEO_A}").status_code == 404
        assert c.post(f"/videos/{VIDEO_A}/summary").status_code == 404
    app.state.pipeline = None
