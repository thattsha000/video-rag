import streamlit as st
from transcriber import get_transcript, get_video_id, get_video_title
from rag import chunk_transcript, embed_chunks, store_chunks, query, summarize

st.set_page_config(
    page_title="VideoRAG",
    page_icon="V",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

* { font-family: 'IBM Plex Sans', sans-serif; }

.stApp {
    background-color: #050d1a;
    color: #a8c4e0;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 180, 255, 0.015) 2px,
        rgba(0, 180, 255, 0.015) 4px
    );
    pointer-events: none;
    z-index: 999;
}

[data-testid="stSidebar"] {
    background-color: #060e1d !important;
    border-right: 1px solid #0d2540 !important;
}

[data-testid="stSidebar"] * {
    color: #a8c4e0 !important;
}

.stTextInput input {
    background-color: #08152b !important;
    border: 1px solid #0d2540 !important;
    color: #00b4ff !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
.stTextInput input:focus {
    border-color: #00b4ff !important;
    box-shadow: 0 0 8px rgba(0, 180, 255, 0.2) !important;
}
.stTextInput input::placeholder {
    color: #1e3a5a !important;
}

/* Main area buttons */
[data-testid="stMain"] .stButton button {
    background-color: transparent !important;
    border: 1px solid #00b4ff !important;
    color: #00b4ff !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    transition: all 0.15s !important;
}
[data-testid="stMain"] .stButton button:hover {
    background-color: #00b4ff !important;
    color: #050d1a !important;
    box-shadow: 0 0 12px rgba(0, 180, 255, 0.4) !important;
}

/* Sidebar buttons styled as clickable boxes */
[data-testid="stSidebar"] .stButton button {
    background-color: #08152b !important;
    border: 1px solid #0d2540 !important;
    color: #3a6a9a !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0px !important;
    text-align: left !important;
    padding: 10px 12px !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    border-color: #00b4ff !important;
    color: #00b4ff !important;
    background-color: #08152b !important;
    box-shadow: none !important;
}

/* Load video button override */
[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background-color: transparent !important;
    border: 1px solid #00b4ff !important;
    color: #00b4ff !important;
    letter-spacing: 2px !important;
}
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
    background-color: #00b4ff !important;
    color: #050d1a !important;
}

[data-testid="stChatInputContainer"] {
    background-color: #060e1d !important;
    border-top: 1px solid #0d2540 !important;
    padding: 12px 16px !important;
}
[data-testid="stChatInput"] {
    background-color: #08152b !important;
    border: 1px solid #0d2540 !important;
    border-radius: 2px !important;
    color: #a8c4e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #00b4ff !important;
    box-shadow: 0 0 8px rgba(0, 180, 255, 0.15) !important;
}

.answer-box {
    background-color: #08152b;
    border: 1px solid #0d2540;
    border-left: 2px solid #00b4ff;
    border-radius: 2px;
    padding: 18px 20px;
    margin: 12px 0;
    font-size: 14px;
    line-height: 1.8;
    color: #a8c4e0;
}

.user-msg {
    text-align: right;
    margin: 12px 0;
}
.user-msg span {
    background: #08152b;
    border: 1px solid #0d2540;
    border-radius: 2px;
    padding: 8px 14px;
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    color: #00b4ff;
}

.timestamp-chip {
    display: inline-block;
    background-color: #08152b;
    border: 1px solid #0d2540;
    color: #00b4ff;
    padding: 3px 10px;
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    margin: 3px;
    text-decoration: none;
    transition: all 0.15s;
}
.timestamp-chip:hover {
    border-color: #00b4ff;
    box-shadow: 0 0 8px rgba(0, 180, 255, 0.3);
    color: #00b4ff;
}

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    color: #1e3a5a;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.empty-state {
    text-align: center;
    padding: 100px 0;
    color: #0d2540;
}

hr { border-color: #0d2540 !important; }
h1, h2, h3 { color: #a8c4e0 !important; }
</style>
""", unsafe_allow_html=True)

# --- Session state ---
if "collection" not in st.session_state:
    st.session_state.collection = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "video_history" not in st.session_state:
    st.session_state.video_history = []
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.markdown('<p class="section-label">VideoRAG</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="section-label">Load Video</p>', unsafe_allow_html=True)

    url = st.text_input("", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
    load_button = st.button("LOAD VIDEO", use_container_width=True)

    if load_button and url:
        with st.spinner("Indexing..."):
            try:
                transcript = get_transcript(url)
                chunks = chunk_transcript(transcript, 10)
                embeddings = embed_chunks(chunks)
                video_id = get_video_id(url)
                st.session_state.collection = store_chunks(chunks, embeddings, video_id)
                st.session_state.video_id = video_id

                if video_id not in st.session_state.chats:
                    st.session_state.chats[video_id] = []

                st.session_state.messages = st.session_state.chats[video_id]

                if video_id not in [v["id"] for v in st.session_state.video_history]:
                    title = get_video_title(video_id)
                    st.session_state.video_history.append({
                        "id": video_id,
                        "url": url,
                        "chunks": len(chunks),
                        "title": title
                    })
                st.success(f"Indexed {len(chunks)} chunks")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.video_history:
        st.markdown("---")
        st.markdown('<p class="section-label">History</p>', unsafe_allow_html=True)
        for video in st.session_state.video_history:
            is_active = video["id"] == st.session_state.video_id
            title = video.get("title", video["id"])
            short_title = title[:28] + "..." if len(title) > 28 else title
            button_style = "border-color: #00b4ff !important; color: #00b4ff !important;" if is_active else ""
            st.markdown(f"<style>div[data-testid='stSidebar'] div[data-testid='stButton']:has(button[title='{title}']) button {{ {button_style} }}</style>", unsafe_allow_html=True)
            if st.button(
                short_title,
                key=f"switch_{video['id']}",
                use_container_width=True,
                help=title
            ):
                st.session_state.video_id = video["id"]
                st.session_state.messages = st.session_state.chats.get(video["id"], [])
                st.rerun()

# --- Main ---
st.markdown('<p class="section-label">Query Interface</p>', unsafe_allow_html=True)

if not st.session_state.collection:
    st.markdown("""
        <div class="empty-state">
            <p style="font-family: JetBrains Mono; font-size: 12px; letter-spacing: 4px">
                PASTE A YOUTUBE URL TO BEGIN
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("SUMMARIZE", use_container_width=True):
            with st.spinner("Summarizing..."):
                summary = summarize(st.session_state.collection, st.session_state.video_id)
                st.session_state.chats[st.session_state.video_id].append({"role": "summary", "content": summary})
                st.session_state.messages = st.session_state.chats[st.session_state.video_id]
            st.rerun()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
                <div class="user-msg">
                    <span>{msg['content']}</span>
                </div>
            """, unsafe_allow_html=True)
        elif msg["role"] in ("assistant", "summary"):
            st.markdown(f'<div class="answer-box">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("timestamps"):
                links = "".join([
                    f'<a href="https://www.youtube.com/watch?v={st.session_state.video_id}&t={int(t)}s" target="_blank" class="timestamp-chip">{int(t)}s</a>'
                    for t in msg["timestamps"]
                ])
                st.markdown(f'<div style="margin-top:8px">{links}</div>', unsafe_allow_html=True)

    question = st.chat_input("ask anything about the video...")
    if question:
        st.session_state.chats[st.session_state.video_id].append({"role": "user", "content": question})
        with st.spinner("thinking..."):
            answer, timestamps = query(st.session_state.collection, question, st.session_state.video_id)
        st.session_state.chats[st.session_state.video_id].append({
            "role": "assistant",
            "content": answer,
            "timestamps": timestamps
        })
        st.session_state.messages = st.session_state.chats[st.session_state.video_id]
        st.rerun()