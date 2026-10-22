"""Streamlit front end for the VideoRAG API.

Run the API first (uvicorn videorag.api:app), then:  streamlit run ui/app.py
Set API_URL if the API is not on http://localhost:8000.
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = httpx.Timeout(120.0, connect=5.0)

st.set_page_config(page_title="VideoRAG", page_icon="V", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>", unsafe_allow_html=True)


# --- API helpers -------------------------------------------------------------


def api(method: str, path: str, **kwargs):
    """Call the API and return (data, error_message)."""
    try:
        resp = httpx.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except httpx.HTTPError:
        return None, f"Can't reach the VideoRAG API at {API_URL}. Is it running?"
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text
        return None, detail if isinstance(detail, str) else f"Request failed ({resp.status_code})"
    return (resp.json() if resp.content else None), None


@st.cache_data(ttl=5)
def fetch_videos():
    data, err = api("GET", "/videos")
    return data or [], err


def render_answer(text: str) -> str:
    """Escape model output, keep line breaks, and highlight [n] citation markers."""
    safe = html.escape(text).replace("\n", "<br>")
    return re.sub(r"\[(\d+(?:,\s*\d+)*)\]", r'<span style="color:#00b4ff">[\1]</span>', safe)


def render_citations(citations: list[dict]) -> str:
    items = []
    for c in citations:
        snippet = c["text"][:180] + ("…" if len(c["text"]) > 180 else "")
        items.append(
            f'<a class="citation" href="{html.escape(c["url"])}" target="_blank" rel="noopener">'
            f'<div class="citation-head">[{c["index"]}] {html.escape(c["title"])} · {c["timestamp"]}</div>'
            f'<div class="citation-text">{html.escape(snippet)}</div></a>'
        )
    return f'<div class="citation-list">{"".join(items)}</div>'


# --- Session state -----------------------------------------------------------

st.session_state.setdefault("video_id", None)  # None means "search all videos"
st.session_state.setdefault("chats", {})  # chat history per scope


def scope_key() -> str:
    return st.session_state.video_id or "__all__"


videos, videos_error = fetch_videos()
titles = {v["video_id"]: v["title"] for v in videos}

# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.markdown('<p class="section-label">VideoRAG</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="section-label">Load Video</p>', unsafe_allow_html=True)

    url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
    if st.button("LOAD VIDEO", use_container_width=True, type="primary") and url:
        with st.spinner("Fetching transcript and indexing..."):
            result, err = api("POST", "/videos", json={"url": url})
        if err:
            st.error(err)
        else:
            fetch_videos.clear()
            st.session_state.video_id = result["video_id"]
            note = "Already indexed" if result["already_indexed"] else "Indexed"
            st.success(f"{note}: {result['chunks']} chunks")
            st.rerun()

    st.markdown("---")
    st.markdown('<p class="section-label">Search Scope</p>', unsafe_allow_html=True)
    if st.button(f"ALL VIDEOS ({len(videos)})", use_container_width=True, key="scope_all"):
        st.session_state.video_id = None
        st.rerun()
    for video in videos:
        title = video["title"]
        label = title[:28] + "…" if len(title) > 28 else title
        if st.session_state.video_id == video["video_id"]:
            label = f"▸ {label}"
        if st.button(label, key=f"switch_{video['video_id']}", use_container_width=True, help=title):
            st.session_state.video_id = video["video_id"]
            st.rerun()

    if videos_error:
        st.warning(videos_error)

# --- Main --------------------------------------------------------------------

scope_title = titles.get(st.session_state.video_id, "All indexed videos")
st.markdown(f'<p class="section-label">Query Interface · {html.escape(scope_title)}</p>', unsafe_allow_html=True)

if not videos:
    st.markdown(
        '<div class="empty-state"><p style="font-family: JetBrains Mono; font-size: 12px; letter-spacing: 4px">'
        "PASTE A YOUTUBE URL TO BEGIN</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

messages = st.session_state.chats.setdefault(scope_key(), [])

if st.session_state.video_id:
    col1, _ = st.columns([1, 5])
    with col1:
        if st.button("SUMMARIZE", use_container_width=True):
            with st.spinner("Summarizing..."):
                data, err = api("POST", f"/videos/{st.session_state.video_id}/summary")
            messages.append({"role": "assistant", "content": err or data["summary"], "citations": []})
            st.rerun()

for msg in messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg"><span>{html.escape(msg["content"])}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="answer-box">{render_answer(msg["content"])}</div>', unsafe_allow_html=True)
        if msg.get("citations"):
            st.markdown(render_citations(msg["citations"]), unsafe_allow_html=True)

question = st.chat_input("Ask anything about the video..." if st.session_state.video_id else "Ask across all videos...")
if question:
    messages.append({"role": "user", "content": question})
    payload = {"question": question}
    if st.session_state.video_id:
        payload["video_ids"] = [st.session_state.video_id]
    with st.spinner("Searching transcripts..."):
        data, err = api("POST", "/ask", json=payload)
    messages.append(
        {"role": "assistant", "content": err or data["answer"], "citations": [] if err else data["citations"]}
    )
    st.rerun()
