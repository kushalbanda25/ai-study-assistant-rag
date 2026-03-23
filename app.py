import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import re
import pickle
import os
import requests

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES  (ChatGPT-like)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0f0f0f;
    color: #ececec;
    font-family: 'Segoe UI', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a1a1a;
    border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] * { color: #d1d1d1 !important; }

/* ── Main title area ── */
h1 { color: #ffffff !important; font-weight: 700; letter-spacing: -0.5px; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 6px;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
div[data-testid="stChatMessage"]:has(img[alt="user"]) {
    background: #1e1e2e;
}

/* Assistant bubble */
div[data-testid="stChatMessage"]:has(img[alt="assistant"]) {
    background: #161b22;
}

/* ── Chat input ── */
[data-testid="stChatInputTextArea"] {
    background: #1e1e1e !important;
    color: #ececec !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 12px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #2d2d2d;
    color: #ececec;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    font-size: 0.85rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: #3d3d3d; }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 10px; font-size: 0.9rem; }

/* ── Spinner text ── */
.stSpinner > div { color: #a0a0a0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL CACHE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
STORAGE_DIR        = "storage"
FAISS_PATH         = os.path.join(STORAGE_DIR, "index.faiss")
CHUNKS_PATH        = os.path.join(STORAGE_DIR, "chunks.pkl")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL          = "deepseek/deepseek-chat-v3-0324:free"
LLM_FALLBACK       = "meta-llama/llama-3.3-70b-instruct:free"
MAX_CONTEXT_WORDS  = 600
TOP_K_CHUNKS       = 3
MAX_HISTORY_TURNS  = 6   # keep last N user+assistant pairs

# ─────────────────────────────────────────────
#  PDF UTILITIES
# ─────────────────────────────────────────────
def read_pdf(file) -> str:
    text = ""
    try:
        pdf = PdfReader(file)
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + " "
    except Exception as e:
        st.sidebar.error(f"Could not read PDF: {e}")
    return re.sub(r'\s+', ' ', text).strip()


def chunk_text(text: str, chunk_size: int = 120) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# ─────────────────────────────────────────────
#  RAG: RETRIEVE CONTEXT
# ─────────────────────────────────────────────
def retrieve_context(query: str) -> str:
    """Return the top-K most relevant chunks, capped at MAX_CONTEXT_WORDS."""
    index  = st.session_state.index
    chunks = st.session_state.chunks

    query_vec = model.encode([query])
    k         = min(TOP_K_CHUNKS, len(chunks))
    _, I      = index.search(np.array(query_vec), k=k)

    selected = [chunks[i] for i in I[0] if i < len(chunks)]

    # Limit total context length
    context_words = []
    for chunk in selected:
        words = chunk.split()
        if len(context_words) + len(words) > MAX_CONTEXT_WORDS:
            remaining = MAX_CONTEXT_WORDS - len(context_words)
            if remaining > 0:
                context_words.extend(words[:remaining])
            break
        context_words.extend(words)

    return " ".join(context_words)


# ─────────────────────────────────────────────
#  LLM CALL
# ─────────────────────────────────────────────
def build_messages(context: str, question: str, history: list[dict]) -> list[dict]:
    """Construct the full message list: system → trimmed history → new user turn."""
    system_prompt = (
        "You are an expert AI study assistant. "
        "Your job is to help students understand their study material precisely and clearly.\n\n"
        "## Guidelines\n"
        "- Answer **only** from the provided context when a factual question is asked.\n"
        "- If the context does not contain the answer, say: "
        "\"I couldn't find that in your uploaded notes. Could you check the document or rephrase?\"\n"
        "- Structure long answers with **Markdown headings**, bullet points, or numbered lists.\n"
        "- Keep answers concise yet complete. Avoid padding or repetition.\n"
        "- When greeted or asked a casual question, respond naturally.\n"
        "- Never fabricate facts or invent information not present in the context.\n"
        "- If the question is ambiguous, ask a clarifying follow-up.\n\n"
        f"## Study Context\n{context}"
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Inject trimmed conversation history
    trim = history[-(MAX_HISTORY_TURNS * 2):]
    messages.extend(trim)

    # Append current user question
    messages.append({"role": "user", "content": question})
    return messages


def _call_openrouter(model: str, messages: list, api_key: str) -> requests.Response:
    return requests.post(
        url=OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://ai-study-assistant",
            "X-Title":       "AI Study Assistant",
        },
        json={
            "model":       model,
            "messages":    messages,
            "temperature": 0.3,
            "max_tokens":  1024,
        },
        timeout=30,
    )


def ask_llm(context: str, question: str, history: list[dict]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return (
            "⚠️ **API key not set.** "
            "Please set the `OPENROUTER_API_KEY` environment variable and restart the app."
        )

    messages = build_messages(context, question, history)

    # Try primary model, fall back on 404
    for model_id in [LLM_MODEL, LLM_FALLBACK]:
        try:
            response = _call_openrouter(model_id, messages, api_key)

            if response.status_code == 404:
                continue   # try next model

            response.raise_for_status()
            result = response.json()

            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"].strip()

            error_msg = result.get("error", {}).get("message", str(result))
            return f"⚠️ **API responded unexpectedly:** {error_msg}"

        except requests.exceptions.Timeout:
            return "⏳ **Request timed out.** The AI service is taking too long. Please try again."
        except requests.exceptions.ConnectionError:
            return "🌐 **Network error.** Check your internet connection and try again."
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 401:
                return "🔑 **Invalid API key.** Please check your `OPENROUTER_API_KEY`."
            if status == 429:
                return "🚦 **Rate limit reached.** Please wait a moment and try again."
            return f"⚠️ **HTTP error {status}.** {str(e)}"
        except Exception as e:
            return f"⚠️ **Unexpected error:** {str(e)}"

    return "⚠️ **All models are currently unavailable.** Please try again in a moment."


# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
if "index"    not in st.session_state: st.session_state.index    = None
if "chunks"   not in st.session_state: st.session_state.chunks   = []
if "messages" not in st.session_state: st.session_state.messages = []
if "loaded"   not in st.session_state: st.session_state.loaded   = False

# ─────────────────────────────────────────────
#  AUTO-LOAD SAVED FAISS INDEX
# ─────────────────────────────────────────────
if not st.session_state.loaded and os.path.exists(FAISS_PATH):
    try:
        st.session_state.index  = faiss.read_index(FAISS_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            st.session_state.chunks = pickle.load(f)
        st.session_state.loaded = True
    except Exception as e:
        st.sidebar.warning(f"Could not restore saved index: {e}")

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Upload Notes")
    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("⚙️ Process", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    # Clear saved data
    if clear_btn:
        for path in [FAISS_PATH, CHUNKS_PATH]:
            if os.path.exists(path):
                os.remove(path)
        st.session_state.index    = None
        st.session_state.chunks   = []
        st.session_state.messages = []
        st.session_state.loaded   = False
        st.success("✅ Data cleared.")

    # Process uploaded files
    if process_btn and uploaded_files:
        all_text = ""
        with st.spinner("Reading PDFs…"):
            for file in uploaded_files:
                all_text += read_pdf(file) + " "

        if all_text.strip():
            chunks = chunk_text(all_text)
            with st.spinner(f"Indexing {len(chunks)} chunks…"):
                embeddings = model.encode(chunks, show_progress_bar=False)
                dimension  = embeddings.shape[1]
                index      = faiss.IndexFlatL2(dimension)
                index.add(np.array(embeddings))

            st.session_state.index  = index
            st.session_state.chunks = chunks
            st.session_state.loaded = True

            os.makedirs(STORAGE_DIR, exist_ok=True)
            faiss.write_index(index, FAISS_PATH)
            with open(CHUNKS_PATH, "wb") as f:
                pickle.dump(chunks, f)

            st.success(f"✅ {len(uploaded_files)} file(s) indexed — {len(chunks)} chunks ready.")
        else:
            st.warning("No readable text found in the uploaded PDFs.")
    elif process_btn:
        st.info("Please upload at least one PDF first.")

    # Status indicator
    st.markdown("---")
    if st.session_state.index is not None:
        st.markdown(
            f"<small>📚 {len(st.session_state.chunks)} chunks loaded</small>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<small>No document loaded yet.</small>", unsafe_allow_html=True)

    # Clear chat history only
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown("# 📘 AI Study Assistant")
st.markdown(
    "<p style='color:#888;margin-top:-12px;font-size:0.9rem;'>"
    "Upload your notes → Ask anything → Get smart answers</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Empty state ──
if st.session_state.index is None:
    st.info("⬅️ Upload your PDF notes and click **⚙️ Process** to get started.")

# ── Render chat history ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
#  CHAT INPUT
# ─────────────────────────────────────────────
query = st.chat_input("Ask something from your notes…")

# ── Greeting detection (no API call needed) ──
GREETING_TRIGGERS = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "ty", "thx",
    "bye", "goodbye", "see you",
    "who are you", "what are you", "what can you do",
}

def is_greeting(text: str) -> str | None:
    """Return a canned reply if the message is a casual greeting, else None."""
    t = text.strip().lower().rstrip("!.,?")
    if t in GREETING_TRIGGERS:
        if t in {"hi", "hello", "hey", "hiya", "howdy", "greetings"}:
            return "👋 Hello! I'm your AI Study Assistant. Upload your PDF notes and ask me anything about them!"
        if t in {"good morning", "good afternoon", "good evening"}:
            return "😊 Good day! Ready to help you study. Upload your notes and fire away!"
        if t in {"thanks", "thank you", "ty", "thx"}:
            return "You're welcome! 😊 Feel free to ask more questions."
        if t in {"bye", "goodbye", "see you"}:
            return "Goodbye! Good luck with your studies! 📚"
        if t in {"who are you", "what are you", "what can you do"}:
            return (
                "I'm your **AI Study Assistant** 📘\n\n"
                "Here's what I can do:\n"
                "- 📄 Read your uploaded PDF notes\n"
                "- 🔍 Find the most relevant passages for your question\n"
                "- 💬 Give clear, structured answers based on your material\n"
                "- 🧠 Remember our conversation history for follow-up questions"
            )
    return None


if query:
    # Check for greeting first — no API call needed
    canned = is_greeting(query)
    if canned:
        st.session_state.messages.append({"role": "user",      "content": query})
        st.session_state.messages.append({"role": "assistant", "content": canned})
        with st.chat_message("user"):      st.markdown(query)
        with st.chat_message("assistant"): st.markdown(canned)
    elif st.session_state.index is None:
        st.warning("⬅️ Please upload and process your PDFs first.")
    else:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Retrieve + generate
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                context = retrieve_context(query)
                history = st.session_state.messages[:-1]
                answer  = ask_llm(context, query, history)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})